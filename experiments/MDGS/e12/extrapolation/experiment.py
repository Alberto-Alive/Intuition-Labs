"""Training, warmup, evaluation, and regression for the E12 Regime 2 run."""

from __future__ import annotations

import json
import random
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW

from .analysis import RegressionResult, fit_primary_pass_regression, fit_support_regression
from .config import E12Config
from .data import Regime2Data, build_regime2_data, make_loader
from .models.regime2_transformer import SupportAwareTransformerClassifier, VARIANT_SPECS


@dataclass(frozen=True)
class WarmupStatus:
    local_metric: float
    joint_auc_mean: float
    joint_auc_per_layer: list[float]
    support_active: bool
    activation_step: int | None


@dataclass(frozen=True)
class VariantRunResult:
    variant: str
    support_active: bool
    activation_step: int | None
    bucket_authority: dict[str, float]
    records: list[dict[str, float | int | str]]


def set_global_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _local_stabilization_metric(history: deque[torch.Tensor]) -> float:
    if len(history) < 2:
        return float("inf")
    diffs = []
    snapshots = list(history)
    for previous, current in zip(snapshots[:-1], snapshots[1:], strict=True):
        diffs.append(torch.mean(torch.abs(current - previous)).item())
    return float(np.mean(diffs))


def _joint_probe_auc(
    model: SupportAwareTransformerClassifier,
    tokens_seen: torch.Tensor,
    tokens_perturbed: torch.Tensor,
    variant_name: str,
    device: torch.device,
) -> tuple[float, list[float]]:
    model.eval()
    with torch.no_grad():
        seen_out = model(
            tokens=tokens_seen.to(device),
            variant_name=variant_name,
            support_active=False,
            update_support_states=False,
        )
        perturbed_out = model(
            tokens=tokens_perturbed.to(device),
            variant_name=variant_name,
            support_active=False,
            update_support_states=False,
        )

    labels = np.concatenate(
        [
            np.zeros(tokens_seen.size(0), dtype=np.int64),
            np.ones(tokens_perturbed.size(0), dtype=np.int64),
        ]
    )
    aucs = []
    for layer_idx in range(model.config.num_layers):
        scores = np.concatenate(
            [
                seen_out.layer_metrics[layer_idx].raw_joint_distance.cpu().numpy(),
                perturbed_out.layer_metrics[layer_idx].raw_joint_distance.cpu().numpy(),
            ]
        )
        aucs.append(float(roc_auc_score(labels, scores)))
    return float(np.mean(aucs)), aucs


def _bucket_name(correct: torch.Tensor, support: torch.Tensor) -> list[str]:
    names = []
    for is_correct, support_value in zip(correct.tolist(), support.tolist(), strict=True):
        support_label = "high" if support_value >= 0.5 else "low"
        correctness_label = "right" if is_correct else "wrong"
        names.append(f"{support_label}_support_{correctness_label}")
    return names


def _difficulty_support(per_example_loss: torch.Tensor) -> torch.Tensor:
    normalized = per_example_loss.detach()
    scale = normalized.max().clamp_min(1e-6)
    return (normalized / scale).clamp(0.0, 1.0)


def verify_neutral_certainty(config: E12Config) -> dict[str, float]:
    set_global_determinism(config.seed)
    baseline = SupportAwareTransformerClassifier(config)
    full = SupportAwareTransformerClassifier(config)
    full.load_state_dict(baseline.state_dict())

    device = torch.device(config.device)
    baseline.to(device)
    full.to(device)

    tokens = torch.tensor(
        [
            [0, 1, 9],
            [0, 2, 10],
            [0, 3, 11],
            [0, 4, 12],
        ],
        dtype=torch.long,
        device=device,
    )
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)

    baseline_out = baseline(tokens, "baseline", support_active=False, update_support_states=False, labels=labels)
    full_out = full(tokens, "full", support_active=False, update_support_states=False, labels=labels)

    logits_max_abs_diff = float(torch.max(torch.abs(baseline_out.logits - full_out.logits)).item())

    baseline.zero_grad(set_to_none=True)
    full.zero_grad(set_to_none=True)
    baseline_out.loss_per_example.mean().backward()
    full_out.loss_per_example.mean().backward()

    grad_diffs = []
    for (baseline_name, baseline_param), (full_name, full_param) in zip(
        baseline.named_parameters(),
        full.named_parameters(),
        strict=True,
    ):
        if baseline_param.grad is None and full_param.grad is None:
            continue
        if baseline_name != full_name:
            raise RuntimeError("Parameter ordering mismatch in neutral-certainty verification")
        grad_diffs.append(torch.max(torch.abs(baseline_param.grad - full_param.grad)).item())
    grad_max_abs_diff = float(max(grad_diffs, default=0.0))

    baseline_optimizer = AdamW(baseline.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    full_optimizer = AdamW(full.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    baseline_optimizer.step()
    full_optimizer.step()

    parameter_diffs = []
    for (baseline_name, baseline_param), (full_name, full_param) in zip(
        baseline.named_parameters(),
        full.named_parameters(),
        strict=True,
    ):
        if baseline_name != full_name:
            raise RuntimeError("Parameter ordering mismatch after optimizer step")
        parameter_diffs.append(torch.max(torch.abs(baseline_param - full_param)).item())

    return {
        "logits_max_abs_diff": logits_max_abs_diff,
        "grad_max_abs_diff": grad_max_abs_diff,
        "parameter_step_max_abs_diff": float(max(parameter_diffs, default=0.0)),
    }


def _evaluate_variant(
    model: SupportAwareTransformerClassifier,
    dataset: Regime2Data,
    config: E12Config,
    variant_name: str,
    support_active: bool,
) -> list[dict[str, float | int | str]]:
    device = torch.device(config.device)
    loader = make_loader(dataset.test, config, shuffle=False, seed=config.seed)
    model.eval()
    records: list[dict[str, float | int | str]] = []

    with torch.no_grad():
        for tokens, labels, seen_unseen, color_ids, shape_ids in loader:
            tokens = tokens.to(device)
            labels = labels.to(device)
            output = model(
                tokens=tokens,
                variant_name=variant_name,
                support_active=support_active,
                update_support_states=False,
                labels=labels,
            )
            predictions = output.logits.argmax(dim=-1)
            correctness = (predictions == labels).to(torch.int64)

            for batch_idx in range(tokens.size(0)):
                color_id = int(color_ids[batch_idx].item())
                shape_id = int(shape_ids[batch_idx].item())
                color_freq = float(dataset.color_frequencies[color_id].item())
                shape_freq = float(dataset.shape_frequencies[shape_id].item())
                records.append(
                    {
                        "variant": variant_name,
                        "correctness": int(correctness[batch_idx].item()),
                        "confidence": float(output.confidence[batch_idx].item()),
                        "local_support": float(output.local_support[batch_idx].item()),
                        "joint_support": float(output.joint_support[batch_idx].item()),
                        "propagated_support": float(output.propagated_support[batch_idx].item()),
                        "final_support": float(output.final_support[batch_idx].item()),
                        "seen_unseen_label": int(seen_unseen[batch_idx].item()),
                        "color_frequency": color_freq,
                        "shape_frequency": shape_freq,
                        "marginal_frequency": 0.5 * (color_freq + shape_freq),
                        "loss": float(output.loss_per_example[batch_idx].item()),
                    }
                )
    return records


def run_variant(
    variant_name: str,
    config: E12Config,
    data: Regime2Data,
) -> VariantRunResult:
    set_global_determinism(config.seed)
    device = torch.device(config.device)
    model = SupportAwareTransformerClassifier(config).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_loader = make_loader(data.train, config, shuffle=True, seed=config.seed)

    support_active = False
    activation_step: int | None = None
    step = 0
    support_history: deque[torch.Tensor] = deque(maxlen=config.local_stabilization_window)
    bucket_authority_sum = Counter()
    bucket_authority_count = Counter()

    for _ in range(config.num_epochs):
        model.train()
        for tokens, labels, _, _, _ in train_loader:
            tokens = tokens.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            output = model(
                tokens=tokens,
                variant_name=variant_name,
                support_active=support_active,
                update_support_states=True,
                labels=labels,
            )
            per_example_loss = output.loss_per_example
            variant = VARIANT_SPECS[variant_name]
            if support_active and variant.backward_mode == "support":
                authority_support = output.final_support.detach()
                loss = model.backward_modulator.modulate_error(per_example_loss, authority_support).mean()
            elif support_active and variant.backward_mode == "difficulty":
                difficulty_support = _difficulty_support(per_example_loss)
                loss = model.backward_modulator.modulate_error(per_example_loss, difficulty_support).mean()
            else:
                authority_support = torch.ones_like(per_example_loss)
                loss = per_example_loss.mean()

            loss.backward()
            optimizer.step()

            predictions = output.logits.argmax(dim=-1)
            correct = predictions.eq(labels)
            if support_active:
                bucket_names = _bucket_name(correct=correct.cpu(), support=output.final_support.detach().cpu())
                if variant.backward_mode == "support":
                    authority_values = model.backward_modulator.authority(output.final_support.detach()).cpu()
                elif variant.backward_mode == "difficulty":
                    authority_values = model.backward_modulator.authority(_difficulty_support(per_example_loss)).cpu()
                else:
                    authority_values = torch.ones_like(output.final_support.detach()).cpu()
                for bucket_name, authority_value in zip(bucket_names, authority_values.tolist(), strict=True):
                    bucket_authority_sum[bucket_name] += authority_value
                    bucket_authority_count[bucket_name] += 1

            step += 1
            support_history.append(model.support_snapshot())

        if not support_active and len(support_history) == config.local_stabilization_window:
            local_metric = _local_stabilization_metric(support_history)
            joint_auc_mean, joint_auc_per_layer = _joint_probe_auc(
                model=model,
                tokens_seen=data.probe_seen_tokens,
                tokens_perturbed=data.probe_perturbed_tokens,
                variant_name=variant_name,
                device=device,
            )
            if (
                local_metric < config.local_stabilization_mae_threshold
                and joint_auc_mean > config.joint_stabilization_auc_threshold
            ):
                support_active = True
                activation_step = step

    records = _evaluate_variant(
        model=model,
        dataset=data,
        config=config,
        variant_name=variant_name,
        support_active=support_active,
    )
    bucket_authority = {}
    for bucket_name in [
        "high_support_wrong",
        "low_support_wrong",
        "low_support_right",
        "high_support_right",
    ]:
        if bucket_authority_count[bucket_name]:
            bucket_authority[bucket_name] = bucket_authority_sum[bucket_name] / bucket_authority_count[bucket_name]
        else:
            bucket_authority[bucket_name] = 0.0

    return VariantRunResult(
        variant=variant_name,
        support_active=support_active,
        activation_step=activation_step,
        bucket_authority=bucket_authority,
        records=records,
    )


def run_regime2_experiment(output_dir: Path, config: E12Config | None = None) -> dict[str, object]:
    config = config or E12Config()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_determinism(config.seed)
    data = build_regime2_data(config)

    neutral_verification = verify_neutral_certainty(config)

    variant_results = {}
    for variant_name in ["baseline", "full", "a1", "a2", "a3", "a4", "a5", "a6", "a7"]:
        variant_results[variant_name] = run_variant(variant_name=variant_name, config=config, data=data)

    full_regression = fit_support_regression(variant_results["full"].records)
    primary_regression = fit_primary_pass_regression(variant_results["full"].records)

    serializable = {
        "config": asdict(config),
        "neutral_verification": neutral_verification,
        "variants": {
            name: {
                "variant": result.variant,
                "support_active": result.support_active,
                "activation_step": result.activation_step,
                "bucket_authority": result.bucket_authority,
                "records": result.records,
            }
            for name, result in variant_results.items()
        },
        "regression": {
            "feature_columns": primary_regression.feature_columns,
            "coefficients": primary_regression.coefficients,
            "p_values": primary_regression.p_values,
            "standardized_coefficients": primary_regression.standardized_coefficients,
            "r2_full": primary_regression.r2_full,
            "r2_reduced": primary_regression.r2_reduced,
            "partial_r2_seen_unseen": primary_regression.partial_r2_seen_unseen,
            "pass_condition": primary_regression.pass_condition,
        },
        "regression_full": {
            "feature_columns": full_regression.feature_columns,
            "coefficients": full_regression.coefficients,
            "p_values": full_regression.p_values,
            "standardized_coefficients": full_regression.standardized_coefficients,
            "r2_full": full_regression.r2_full,
            "r2_reduced": full_regression.r2_reduced,
            "partial_r2_seen_unseen": full_regression.partial_r2_seen_unseen,
            "pass_condition": full_regression.pass_condition,
        },
    }
    (output_dir / "regime2_results.json").write_text(json.dumps(serializable, indent=2))
    return serializable
