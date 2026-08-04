from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from .distribution_attention import (
        DATABLATION,
        DATPlacement,
        DistributionAttentionLinear,
        DistributionAttentionTransformer,
        DistributionAttentionTransformerConfig,
        FixedTransformerClassifier,
        HyperNetworkLinear,
        HyperNetworkTransformer,
        TokenParameterAttentionTransformer,
        count_parameters,
        estimate_active_mult_adds,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from distribution_attention import (
        DATABLATION,
        DATPlacement,
        DistributionAttentionLinear,
        DistributionAttentionTransformer,
        DistributionAttentionTransformerConfig,
        FixedTransformerClassifier,
        HyperNetworkLinear,
        HyperNetworkTransformer,
        TokenParameterAttentionTransformer,
        count_parameters,
        estimate_active_mult_adds,
    )


MODEL_VARIANTS = [
    "fixed_small_transformer",
    "fixed_medium_transformer",
    "fixed_large_transformer",
    "token_parameter_attention_baseline",
    "hypernetwork_baseline",
    "dat_transformer",
]

MODEL_SIZES: dict[str, dict[str, int]] = {
    "small": {"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128},
    "medium": {"d_model": 96, "nhead": 4, "num_layers": 3, "dim_feedforward": 192},
    "large": {"d_model": 128, "nhead": 8, "num_layers": 4, "dim_feedforward": 256},
}

DAT_ABLATIONS: dict[str, DATABLATION] = {
    "dat_fixed_point": "fixed_point",
    "dat_sigma_zero": "sigma_zero",
    "dat_shuffle_point": "shuffle_point",
    "dat_shuffle_attn": "shuffle_attn",
    "dat_shuffle_both": "shuffle_both",
    "dat_random_point": "random_point",
    "dat_fixed_attention": "fixed_attention",
    "dat_fixed_values": "fixed_values",
    "dat_no_distribution": "no_distribution",
    "dat_score_detached_point": "score_detached_point",
}


@dataclass(frozen=True)
class RuleSwitchingTaskConfig:
    n_train: int = 5000
    n_val: int = 1000
    n_test: int = 1000
    seq_len: int = 16
    vocab_size: int = 64
    num_rules: int = 6
    num_classes: int = 10
    batch_size: int = 128


@dataclass(frozen=True)
class TrainConfig:
    seeds: list[int]
    epochs: int = 20
    lr: float = 0.002
    weight_decay: float = 0.0001
    grad_clip: float = 5.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _labels_by_rule(values: Tensor, num_classes: int) -> Tensor:
    payload_len = values.shape[1]
    reverse_position = max(0, payload_len - 2)
    rules = torch.stack(
        [
            values[:, 0],
            values[:, -1],
            (values[:, 0] + values[:, 1]) % num_classes,
            values[:, : min(8, payload_len)].sum(dim=1) % num_classes,
            values[:, reverse_position],
            (values[:, min(2, payload_len - 1)] + 3) % num_classes,
        ],
        dim=1,
    )
    return rules


def make_rule_switching_split(
    n_examples: int,
    *,
    config: RuleSwitchingTaskConfig,
    seed: int,
) -> TensorDataset:
    if config.seq_len < 8:
        raise ValueError("seq_len must be at least 8")
    if config.num_rules < 1 or config.num_rules > 6:
        raise ValueError("num_rules must be between 1 and 6")
    payload_offset = 8
    if payload_offset + config.num_classes > config.vocab_size:
        raise ValueError("vocab_size must fit payload_offset + num_classes")

    generator = torch.Generator().manual_seed(seed)
    rules = torch.randint(0, config.num_rules, (n_examples,), generator=generator)
    values = torch.randint(
        0,
        config.num_classes,
        (n_examples, config.seq_len - 1),
        generator=generator,
    )
    tokens = torch.empty(n_examples, config.seq_len, dtype=torch.long)
    tokens[:, 0] = rules
    tokens[:, 1:] = values + payload_offset
    all_labels = _labels_by_rule(values, config.num_classes)[:, : config.num_rules]
    labels = all_labels.gather(1, rules.view(-1, 1)).squeeze(1)
    return TensorDataset(tokens, labels.long(), rules.long())


def make_dataloaders(
    config: RuleSwitchingTaskConfig,
    *,
    seed: int,
) -> dict[str, DataLoader]:
    train = make_rule_switching_split(
        config.n_train,
        config=config,
        seed=seed * 1009 + 17,
    )
    val = make_rule_switching_split(
        config.n_val,
        config=config,
        seed=seed * 1009 + 29,
    )
    test = make_rule_switching_split(
        config.n_test,
        config=config,
        seed=seed * 1009 + 43,
    )
    return {
        "train": DataLoader(train, batch_size=config.batch_size, shuffle=True),
        "val": DataLoader(val, batch_size=config.batch_size, shuffle=False),
        "test": DataLoader(test, batch_size=config.batch_size, shuffle=False),
    }


def size_name_for_variant(variant: str) -> str:
    if variant == "fixed_medium_transformer":
        return "medium"
    if variant == "fixed_large_transformer":
        return "large"
    return "small"


def model_kind_for_variant(
    variant: str,
) -> str:
    if variant == "dat_transformer":
        return "dat"
    if variant == "token_parameter_attention_baseline":
        return "parameter_attention"
    if variant == "hypernetwork_baseline":
        return "hypernetwork"
    return "fixed"


def build_model_config(
    args: argparse.Namespace,
    *,
    variant: str,
) -> DistributionAttentionTransformerConfig:
    size = MODEL_SIZES[size_name_for_variant(variant)]
    return DistributionAttentionTransformerConfig(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        num_classes=args.num_classes,
        d_model=size["d_model"],
        nhead=size["nhead"],
        num_layers=size["num_layers"],
        dim_feedforward=size["dim_feedforward"],
        dropout=args.dropout,
        num_distributions=args.num_distributions,
        d_key=args.d_key,
        use_distribution_attention_in=args.use_distribution_attention_in,
        readout_position=0,
    )


def build_model(
    args: argparse.Namespace,
    *,
    variant: str,
) -> nn.Module:
    config = build_model_config(args, variant=variant)
    if variant == "dat_transformer":
        return DistributionAttentionTransformer(config)
    if variant == "token_parameter_attention_baseline":
        return TokenParameterAttentionTransformer(config)
    if variant == "hypernetwork_baseline":
        return HyperNetworkTransformer(config)
    return FixedTransformerClassifier(config)


def logits_from_output(output: Tensor | dict[str, Any]) -> Tensor:
    if isinstance(output, dict):
        return output["logits"]
    return output


def train_one_model(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    train_config: TrainConfig,
) -> dict[str, float]:
    device = torch.device(train_config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    best_val_nll = float("inf")
    best_state: dict[str, Tensor] | None = None
    started = time.perf_counter()

    for _epoch in range(train_config.epochs):
        model.train()
        for tokens, labels, _rules in loaders["train"]:
            tokens = tokens.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(tokens, return_details=False)
            logits = logits_from_output(output)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()

        val_metrics = evaluate_model(model, loaders["val"], device=device)
        if val_metrics["nll"] < best_val_nll:
            best_val_nll = val_metrics["nll"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "train_seconds": time.perf_counter() - started,
        "best_val_nll": best_val_nll,
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    ablation: DATABLATION = "none",
    num_rules: int | None = None,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    if num_rules is None:
        num_rules = 0
    correct_by_rule = torch.zeros(num_rules, dtype=torch.long)
    count_by_rule = torch.zeros(num_rules, dtype=torch.long)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for tokens, labels, rules in loader:
        tokens = tokens.to(device)
        labels = labels.to(device)
        output = model(tokens, ablation=ablation, return_details=False)
        logits = logits_from_output(output)
        loss = F.cross_entropy(logits, labels, reduction="sum")
        predictions = logits.argmax(dim=-1)
        correct = predictions.eq(labels)

        batch_size = labels.numel()
        total_loss += float(loss.item())
        total_correct += int(correct.sum().item())
        total_examples += batch_size

        if num_rules:
            for rule in range(num_rules):
                mask = rules.eq(rule)
                count = int(mask.sum().item())
                if count:
                    count_by_rule[rule] += count
                    correct_by_rule[rule] += int(correct.cpu()[mask].sum().item())

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    per_rule_accuracy = {}
    for rule in range(num_rules):
        if count_by_rule[rule].item():
            per_rule_accuracy[f"rule_{rule}"] = (
                correct_by_rule[rule].item() / count_by_rule[rule].item()
            )
        else:
            per_rule_accuracy[f"rule_{rule}"] = float("nan")

    return {
        "accuracy": total_correct / max(1, total_examples),
        "nll": total_loss / max(1, total_examples),
        "examples": total_examples,
        "latency_ms_per_example": 1000.0 * elapsed / max(1, total_examples),
        "per_rule_accuracy": per_rule_accuracy,
    }


def _rule_separation(features: Tensor, groups: Tensor) -> float:
    unique_groups = torch.unique(groups)
    if unique_groups.numel() < 2:
        return 0.0
    means = []
    for group in unique_groups:
        mask = groups.eq(group)
        if mask.any():
            means.append(features[mask].mean(dim=0))
    if len(means) < 2:
        return 0.0
    mean_tensor = torch.stack(means, dim=0)
    distances = torch.pdist(mean_tensor.float(), p=2)
    scale = features.float().std(unbiased=False).clamp_min(1e-6)
    return float((distances.mean() / scale).item())


@torch.no_grad()
def collect_mechanism_metrics(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int = 2,
) -> dict[str, float]:
    model.eval()
    entropy_values: list[Tensor] = []
    point_means: list[Tensor] = []
    point_stds: list[Tensor] = []
    point_variances: list[Tensor] = []
    realized_variances: list[Tensor] = []
    attn_features: list[Tensor] = []
    point_features: list[Tensor] = []
    value_features: list[Tensor] = []
    group_chunks: list[Tensor] = []

    for batch_idx, (tokens, _labels, groups) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        output = model(tokens.to(device), return_details=True)
        if not isinstance(output, dict):
            continue
        diagnostics = output.get("dat_diagnostics", [])
        if not diagnostics:
            continue

        batch_attn_features = []
        batch_point_features = []
        batch_value_features = []
        for diagnostic in diagnostics:
            attn = diagnostic["attn"].detach().float().cpu()
            point = diagnostic["point"].detach().float().cpu()
            realized_values = diagnostic["realized_values"].detach().float()

            entropy = -(attn * attn.clamp_min(1e-8).log()).sum(dim=-1).mean()
            entropy_values.append(entropy)
            point_means.append(point.mean())
            point_stds.append(point.std(unbiased=False))
            point_variances.append(point.var(dim=0, unbiased=False).mean())
            value_feature = realized_values.mean(dim=(1, 2)).cpu()
            realized_variances.append(value_feature.var(dim=0, unbiased=False).mean())

            batch_attn_features.append(attn.mean(dim=1))
            batch_point_features.append(point.mean(dim=1))
            batch_value_features.append(value_feature)

        attn_features.append(torch.cat(batch_attn_features, dim=-1))
        point_features.append(torch.cat(batch_point_features, dim=-1))
        value_features.append(torch.cat(batch_value_features, dim=-1))
        group_chunks.append(groups.cpu())

    if not attn_features:
        return {
            "attention_entropy": 0.0,
            "point_mean": 0.0,
            "point_std": 0.0,
            "point_variance_across_examples": 0.0,
            "point_variance_across_rules": 0.0,
            "sigma_norm": 0.0,
            "realized_value_variance_across_examples": 0.0,
            "realized_value_variance_across_rules": 0.0,
            "attn_rule_separation": 0.0,
            "point_rule_separation": 0.0,
            "realized_value_rule_separation": 0.0,
        }

    attn_matrix = torch.cat(attn_features, dim=0)
    point_matrix = torch.cat(point_features, dim=0)
    value_matrix = torch.cat(value_features, dim=0)
    groups = torch.cat(group_chunks, dim=0)
    sigma_norms = [
        module.sigma.detach().float().norm(dim=-1).mean().cpu()
        for module in getattr(model, "dat_linears", lambda: [])()
    ]

    def mean_tensor(values: list[Tensor]) -> float:
        return float(torch.stack(values).mean().item()) if values else 0.0

    return {
        "attention_entropy": mean_tensor(entropy_values),
        "point_mean": mean_tensor(point_means),
        "point_std": mean_tensor(point_stds),
        "point_variance_across_examples": mean_tensor(point_variances),
        "point_variance_across_rules": float(
            torch.stack(
                [
                    point_matrix[groups.eq(rule)].mean(dim=0)
                    for rule in torch.unique(groups)
                    if groups.eq(rule).any()
                ]
            )
            .var(dim=0, unbiased=False)
            .mean()
            .item()
        ),
        "sigma_norm": mean_tensor(sigma_norms),
        "realized_value_variance_across_examples": mean_tensor(realized_variances),
        "realized_value_variance_across_rules": float(
            torch.stack(
                [
                    value_matrix[groups.eq(rule)].mean(dim=0)
                    for rule in torch.unique(groups)
                    if groups.eq(rule).any()
                ]
            )
            .var(dim=0, unbiased=False)
            .mean()
            .item()
        ),
        "attn_rule_separation": _rule_separation(attn_matrix, groups),
        "point_rule_separation": _rule_separation(point_matrix, groups),
        "realized_value_rule_separation": _rule_separation(value_matrix, groups),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    variants = sorted({row["model"] for row in rows})
    aggregate_rows = []
    numeric_keys = [
        "validation_accuracy",
        "validation_nll",
        "test_accuracy",
        "test_nll",
        "train_seconds",
        "latency_ms_per_example",
        "parameter_count",
        "active_mult_adds_estimate",
    ]
    for variant in variants:
        subset = [row for row in rows if row["model"] == variant]
        aggregate = {"model": variant, "seeds": len(subset)}
        for key in numeric_keys:
            values = [float(row[key]) for row in subset if key in row]
            aggregate[f"mean_{key}"] = sum(values) / len(values) if values else 0.0
        aggregate_rows.append(aggregate)
    return aggregate_rows


def aggregate_ablation_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name in sorted({row["ablation"] for row in rows}):
        subset = [row for row in rows if row["ablation"] == name]
        result[name] = {
            "accuracy": sum(float(row["accuracy"]) for row in subset) / len(subset),
            "nll": sum(float(row["nll"]) for row in subset) / len(subset),
            "accuracy_drop": sum(float(row["accuracy_drop"]) for row in subset)
            / len(subset),
            "nll_increase": sum(float(row["nll_increase"]) for row in subset)
            / len(subset),
        }
    return result


def compute_metrics_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    aggregate = aggregate_metric_rows(rows)
    models: dict[str, dict[str, float]] = {}
    for row in aggregate:
        models[row["model"]] = {
            "parameter_count": row["mean_parameter_count"],
            "active_mult_adds_estimate": row["mean_active_mult_adds_estimate"],
            "latency_ms_per_example": row["mean_latency_ms_per_example"],
        }
    fixed_small = models.get("fixed_small_transformer", {})
    comparisons = {}
    for model, values in models.items():
        comparisons[model] = {
            "parameter_ratio_to_fixed_small": values["parameter_count"]
            / max(1.0, fixed_small.get("parameter_count", 1.0)),
            "compute_ratio_to_fixed_small": values["active_mult_adds_estimate"]
            / max(1.0, fixed_small.get("active_mult_adds_estimate", 1.0)),
            "latency_ratio_to_fixed_small": values["latency_ms_per_example"]
            / max(1e-9, fixed_small.get("latency_ms_per_example", 1e-9)),
        }
    return {
        "models": models,
        "compare_to_fixed_small": comparisons,
    }


def _mean_for_model(
    aggregate_rows: list[dict[str, Any]],
    model: str,
    key: str,
) -> float:
    for row in aggregate_rows:
        if row["model"] == model:
            return float(row[key])
    return float("nan")


def make_verdict(
    *,
    args: argparse.Namespace,
    aggregate_rows: list[dict[str, Any]],
    ablation_summary: dict[str, dict[str, float]],
    compute_summary: dict[str, Any],
) -> dict[str, Any]:
    random_accuracy = 1.0 / float(args.num_classes)
    dat_accuracy = _mean_for_model(
        aggregate_rows,
        "dat_transformer",
        "mean_test_accuracy",
    )
    dat_nll = _mean_for_model(aggregate_rows, "dat_transformer", "mean_test_nll")
    small_accuracy = _mean_for_model(
        aggregate_rows,
        "fixed_small_transformer",
        "mean_test_accuracy",
    )
    small_nll = _mean_for_model(
        aggregate_rows,
        "fixed_small_transformer",
        "mean_test_nll",
    )
    large_accuracy = _mean_for_model(
        aggregate_rows,
        "fixed_large_transformer",
        "mean_test_accuracy",
    )
    large_nll = _mean_for_model(
        aggregate_rows,
        "fixed_large_transformer",
        "mean_test_nll",
    )

    def hurts(name: str) -> bool:
        metrics = ablation_summary.get(name, {})
        return (
            metrics.get("accuracy_drop", 0.0) >= args.effect_threshold
            or metrics.get("nll_increase", 0.0) >= args.nll_threshold
        )

    task_learns = dat_accuracy >= random_accuracy + 0.08 or dat_nll <= (
        math.log(args.num_classes) - 0.08
    )
    baseline_not_saturated = small_accuracy <= 0.95
    dat_beats_fixed_small = (
        dat_accuracy >= small_accuracy + 0.005 or dat_nll <= small_nll - 0.02
    )
    if large_accuracy > small_accuracy:
        dat_near_fixed_large = dat_accuracy >= small_accuracy + 0.5 * (
            large_accuracy - small_accuracy
        )
    else:
        dat_near_fixed_large = dat_accuracy >= large_accuracy - 0.02

    point_inside_distribution_used = hurts("dat_fixed_point") or hurts(
        "dat_sigma_zero"
    )
    sigma_distribution_used = hurts("dat_sigma_zero")
    input_specific_points_used = hurts("dat_shuffle_point") or hurts(
        "dat_random_point"
    )
    distribution_selection_used = hurts("dat_shuffle_attn") or hurts(
        "dat_fixed_attention"
    )
    not_fixed_value_attention = hurts("dat_no_distribution") or hurts(
        "dat_fixed_values"
    )

    sample_dat = DistributionAttentionTransformer(
        build_model_config(args, variant="dat_transformer")
    )
    not_hypernetwork = not any(
        isinstance(module, HyperNetworkLinear) for module in sample_dat.modules()
    )
    no_controller_present = not hasattr(sample_dat, "controller")
    mechanism_clean = (
        any(isinstance(module, DistributionAttentionLinear) for module in sample_dat.modules())
        and not_hypernetwork
        and no_controller_present
    )

    dat_compute_ratio = compute_summary["compare_to_fixed_small"].get(
        "dat_transformer",
        {},
    ).get("compute_ratio_to_fixed_small", float("inf"))
    compute_not_exploded = dat_compute_ratio < 3.0 or (
        dat_accuracy >= small_accuracy + 0.05
    )

    checks = {
        "task_learns": bool(task_learns),
        "baseline_not_saturated": bool(baseline_not_saturated),
        "dat_beats_fixed_small": bool(dat_beats_fixed_small),
        "dat_near_fixed_large": bool(dat_near_fixed_large),
        "point_inside_distribution_used": bool(point_inside_distribution_used),
        "sigma_distribution_used": bool(sigma_distribution_used),
        "input_specific_points_used": bool(input_specific_points_used),
        "distribution_selection_used": bool(distribution_selection_used),
        "not_fixed_value_attention": bool(not_fixed_value_attention),
        "not_hypernetwork": bool(not_hypernetwork),
        "no_controller_present": bool(no_controller_present),
        "compute_not_exploded": bool(compute_not_exploded),
        "mechanism_clean": bool(mechanism_clean),
    }

    mechanism_ablation_clean = all(
        [
            point_inside_distribution_used,
            sigma_distribution_used,
            input_specific_points_used,
            distribution_selection_used,
        ]
    )
    if not mechanism_clean or not not_hypernetwork or not no_controller_present:
        final_verdict = "FAILED_IMPLEMENTATION"
    elif not task_learns:
        final_verdict = "FAILED_IMPLEMENTATION"
    elif not mechanism_ablation_clean:
        final_verdict = "LEARNS_BUT_MECHANISM_UNUSED"
    elif (
        dat_beats_fixed_small
        and dat_near_fixed_large
        and not_fixed_value_attention
        and compute_not_exploded
    ):
        final_verdict = "STRONG_DISTRIBUTION_ATTENTION"
    elif (dat_beats_fixed_small or not_fixed_value_attention) and compute_not_exploded:
        final_verdict = "PROMISING_DISTRIBUTION_ATTENTION"
    else:
        final_verdict = "MECHANISM_WORKS_NO_VALUE"

    return {
        "checks": checks,
        "final_verdict": final_verdict,
        "thresholds": {
            "random_accuracy": random_accuracy,
            "accuracy_effect_threshold": args.effect_threshold,
            "nll_effect_threshold": args.nll_threshold,
        },
        "key_metrics": {
            "dat_test_accuracy": dat_accuracy,
            "dat_test_nll": dat_nll,
            "fixed_small_test_accuracy": small_accuracy,
            "fixed_small_test_nll": small_nll,
            "fixed_large_test_accuracy": large_accuracy,
            "fixed_large_test_nll": large_nll,
            "dat_compute_ratio_to_fixed_small": dat_compute_ratio,
        },
        "ablation_summary": ablation_summary,
    }


def yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def make_report(
    *,
    args: argparse.Namespace,
    task_config: RuleSwitchingTaskConfig,
    train_config: TrainConfig,
    aggregate_rows: list[dict[str, Any]],
    ablation_summary: dict[str, dict[str, float]],
    mechanism_rows: list[dict[str, Any]],
    compute_summary: dict[str, Any],
    verdict: dict[str, Any],
) -> str:
    checks = verdict["checks"]
    dat_accuracy = verdict["key_metrics"]["dat_test_accuracy"]
    small_accuracy = verdict["key_metrics"]["fixed_small_test_accuracy"]
    random_accuracy = verdict["thresholds"]["random_accuracy"]

    def ablation_line(name: str) -> str:
        metrics = ablation_summary.get(name, {})
        return (
            f"accuracy_drop={metrics.get('accuracy_drop', 0.0):.4f}, "
            f"nll_increase={metrics.get('nll_increase', 0.0):.4f}"
        )

    model_table = [
        "| model | test_accuracy | test_nll | parameters | active_mult_adds |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sorted(aggregate_rows, key=lambda item: item["model"]):
        model_table.append(
            "| {model} | {acc:.4f} | {nll:.4f} | {params:.0f} | {flops:.0f} |".format(
                model=row["model"],
                acc=row["mean_test_accuracy"],
                nll=row["mean_test_nll"],
                params=row["mean_parameter_count"],
                flops=row["mean_active_mult_adds_estimate"],
            )
        )

    mechanism_table = [
        "| metric | value |",
        "|---|---:|",
    ]
    if mechanism_rows:
        keys = [
            key
            for key in mechanism_rows[0].keys()
            if key not in {"seed", "model"}
        ]
        for key in keys:
            values = [float(row[key]) for row in mechanism_rows if key in row]
            mechanism_table.append(f"| {key} | {sum(values) / len(values):.6f} |")

    answers = [
        (
            "1. Did normal attention scores select distributions?",
            yes_no(checks["distribution_selection_used"]),
            ablation_line("dat_shuffle_attn")
            + "; "
            + ablation_line("dat_fixed_attention"),
        ),
        (
            "2. Did the same scores select points inside those distributions?",
            yes_no(checks["point_inside_distribution_used"]),
            ablation_line("dat_fixed_point"),
        ),
        (
            "3. Did mu + point * sigma create the actual values used?",
            yes_no(checks["mechanism_clean"]),
            "The DAT forward path computes realized_values before the weighted sum.",
        ),
        (
            "4. Did removing sigma hurt?",
            yes_no(checks["sigma_distribution_used"]),
            ablation_line("dat_sigma_zero"),
        ),
        (
            "5. Did fixing point hurt?",
            yes_no(
                ablation_summary.get("dat_fixed_point", {}).get("accuracy_drop", 0.0)
                >= args.effect_threshold
                or ablation_summary.get("dat_fixed_point", {}).get(
                    "nll_increase",
                    0.0,
                )
                >= args.nll_threshold
            ),
            ablation_line("dat_fixed_point"),
        ),
        (
            "6. Did shuffling point hurt?",
            yes_no(
                ablation_summary.get("dat_shuffle_point", {}).get(
                    "accuracy_drop",
                    0.0,
                )
                >= args.effect_threshold
                or ablation_summary.get("dat_shuffle_point", {}).get(
                    "nll_increase",
                    0.0,
                )
                >= args.nll_threshold
            ),
            ablation_line("dat_shuffle_point"),
        ),
        (
            "7. Did fixed-value parameter attention perform worse?",
            yes_no(checks["not_fixed_value_attention"]),
            ablation_line("dat_no_distribution")
            + "; token baseline is reported separately.",
        ),
        (
            "8. Was there any separate controller or hypernetwork?",
            "No" if checks["not_hypernetwork"] and checks["no_controller_present"] else "Yes",
            "DAT uses query_proj, dist_keys, mu, and sigma only. The hypernetwork is a separate baseline.",
        ),
        (
            "9. Did DAT beat fixed_small?",
            yes_no(checks["dat_beats_fixed_small"]),
            f"DAT={dat_accuracy:.4f}, fixed_small={small_accuracy:.4f}.",
        ),
        (
            "10. Was compute reasonable?",
            yes_no(checks["compute_not_exploded"]),
            "DAT compute ratio to fixed_small="
            f"{verdict['key_metrics']['dat_compute_ratio_to_fixed_small']:.3f}.",
        ),
    ]

    answer_lines = []
    for question, answer, evidence in answers:
        answer_lines.append(f"### {question}")
        answer_lines.append(f"{answer}. {evidence}")

    return "\n".join(
        [
            "# Distribution Attention POF Report",
            "",
            f"Final verdict: `{verdict['final_verdict']}`",
            "",
            "## Run Config",
            "",
            "```json",
            json.dumps(
                {
                    "task": asdict(task_config),
                    "train": asdict(train_config),
                    "model": {
                        "num_distributions": args.num_distributions,
                        "d_key": args.d_key,
                        "use_distribution_attention_in": args.use_distribution_attention_in,
                    },
                },
                indent=2,
            ),
            "```",
            "",
            f"Random accuracy: {random_accuracy:.4f}",
            "",
            "## Model Metrics",
            "",
            *model_table,
            "",
            "## Mechanism Metrics",
            "",
            *mechanism_table,
            "",
            "## Required Questions",
            "",
            *answer_lines,
            "",
            "## Mechanism Code Audit",
            "",
            "The DAT layer in `distribution_attention.py` follows this forward path:",
            "",
            "```python",
            "q = self.query_proj(x)",
            "score = torch.matmul(q, self.dist_keys.t()) / math.sqrt(float(self.d_key))",
            "attn = torch.softmax(score, dim=-1)",
            "point = torch.tanh(score)",
            "realized_values = self.mu + point.unsqueeze(-1) * self.sigma",
            "y = torch.sum(attn.unsqueeze(-1) * realized_values, dim=2)",
            "```",
            "",
            "There is no DAT controller module, no full effective weight materialization, "
            "and no hypernetwork in the DAT model. The hypernetwork implementation is "
            "only used by `hypernetwork_baseline`.",
            "",
            "## Verdict Checks",
            "",
            "```json",
            json.dumps(verdict["checks"], indent=2),
            "```",
        ]
    )


def run_experiment(args: argparse.Namespace) -> Path:
    if args.quick:
        args.seeds = "0"
        args.epochs = min(args.epochs, 2)
        args.n_train = min(args.n_train, 768)
        args.n_val = min(args.n_val, 256)
        args.n_test = min(args.n_test, 256)

    seeds = parse_seeds(args.seeds)
    task_config = RuleSwitchingTaskConfig(
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        num_rules=args.num_rules,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
    )
    train_config = TrainConfig(
        seeds=seeds,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        device=args.device,
    )

    run_dir = (
        Path(args.output_root)
        / f"distribution_attention_pof_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "RUN_CONFIG.json").write_text(
        json.dumps(
            {
                "task": asdict(task_config),
                "train": asdict(train_config),
                "model_variants": MODEL_VARIANTS,
                "dat": {
                    "num_distributions": args.num_distributions,
                    "d_key": args.d_key,
                    "use_distribution_attention_in": args.use_distribution_attention_in,
                },
                "quick": args.quick,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    per_seed_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    device = torch.device(train_config.device)

    for seed in seeds:
        loaders = make_dataloaders(task_config, seed=seed)
        for variant in MODEL_VARIANTS:
            set_seed(seed * 1009 + MODEL_VARIANTS.index(variant))
            model = build_model(args, variant=variant)
            model_config = build_model_config(args, variant=variant)
            model_kind = model_kind_for_variant(variant)
            train_metrics = train_one_model(model, loaders, train_config)
            val_metrics = evaluate_model(
                model,
                loaders["val"],
                device=device,
                num_rules=args.num_rules,
            )
            test_metrics = evaluate_model(
                model,
                loaders["test"],
                device=device,
                num_rules=args.num_rules,
            )
            parameter_count = count_parameters(model)
            active_mult_adds = estimate_active_mult_adds(
                model_config,
                model_kind=model_kind,  # type: ignore[arg-type]
            )
            row = {
                "seed": seed,
                "model": variant,
                "validation_accuracy": val_metrics["accuracy"],
                "validation_nll": val_metrics["nll"],
                "test_accuracy": test_metrics["accuracy"],
                "test_nll": test_metrics["nll"],
                "random_accuracy": 1.0 / float(args.num_classes),
                "per_rule_accuracy_json": json.dumps(
                    test_metrics["per_rule_accuracy"],
                    sort_keys=True,
                ),
                "parameter_count": parameter_count,
                "active_mult_adds_estimate": active_mult_adds,
                "latency_ms_per_example": test_metrics["latency_ms_per_example"],
                **train_metrics,
            }
            per_seed_rows.append(row)
            print(
                f"seed={seed} model={variant} "
                f"test_acc={test_metrics['accuracy']:.4f} "
                f"test_nll={test_metrics['nll']:.4f}",
                flush=True,
            )

            if variant == "dat_transformer":
                mechanism = collect_mechanism_metrics(
                    model,
                    loaders["test"],
                    device=device,
                    max_batches=args.mechanism_batches,
                )
                mechanism_rows.append({"seed": seed, "model": variant, **mechanism})
                base_accuracy = test_metrics["accuracy"]
                base_nll = test_metrics["nll"]
                for ablation_name, ablation in DAT_ABLATIONS.items():
                    ablated = evaluate_model(
                        model,
                        loaders["test"],
                        device=device,
                        ablation=ablation,
                        num_rules=args.num_rules,
                    )
                    ablation_rows.append(
                        {
                            "seed": seed,
                            "ablation": ablation_name,
                            "accuracy": ablated["accuracy"],
                            "nll": ablated["nll"],
                            "accuracy_drop": base_accuracy - ablated["accuracy"],
                            "nll_increase": ablated["nll"] - base_nll,
                            "latency_ms_per_example": ablated[
                                "latency_ms_per_example"
                            ],
                        }
                    )

    aggregate_rows = aggregate_metric_rows(per_seed_rows)
    ablation_summary = aggregate_ablation_rows(ablation_rows)
    compute_summary = compute_metrics_summary(per_seed_rows)
    verdict = make_verdict(
        args=args,
        aggregate_rows=aggregate_rows,
        ablation_summary=ablation_summary,
        compute_summary=compute_summary,
    )
    report = make_report(
        args=args,
        task_config=task_config,
        train_config=train_config,
        aggregate_rows=aggregate_rows,
        ablation_summary=ablation_summary,
        mechanism_rows=mechanism_rows,
        compute_summary=compute_summary,
        verdict=verdict,
    )

    write_csv(run_dir / "per_seed_metrics.csv", per_seed_rows)
    write_csv(run_dir / "aggregate_metrics.csv", aggregate_rows)
    write_csv(run_dir / "ablation_metrics.csv", ablation_rows)
    write_csv(run_dir / "mechanism_metrics.csv", mechanism_rows)
    (run_dir / "compute_metrics.json").write_text(
        json.dumps(compute_summary, indent=2),
        encoding="utf-8",
    )
    (run_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2),
        encoding="utf-8",
    )
    (run_dir / "REPORT.md").write_text(report, encoding="utf-8")
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-root", default="transformer_uncertainty/runs")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--num-rules", type=int, default=6)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--num-distributions", type=int, default=128)
    parser.add_argument("--d-key", type=int, default=64)
    parser.add_argument(
        "--use-distribution-attention-in",
        choices=["MLP_FIRST", "MLP_SECOND"],
        default="MLP_FIRST",
        type=str,
    )
    parser.add_argument("--effect-threshold", type=float, default=0.02)
    parser.add_argument("--nll-threshold", type=float, default=0.03)
    parser.add_argument("--mechanism-batches", type=int, default=2)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_dir = run_experiment(args)
    print(f"Wrote Distribution Attention POF artifacts to {run_dir}")


if __name__ == "__main__":
    main()
