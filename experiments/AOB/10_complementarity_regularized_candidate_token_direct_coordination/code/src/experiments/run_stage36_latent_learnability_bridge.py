from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.coordinators.latent_coordination import LatentCoordinatorConfig, make_candidate_query_coordinator
from src.datasets.multiview_code_patch_selection import (
    ATTRIBUTE_VALUE_PAIRS,
    BALANCED_34B_CANDIDATE_REPRESENTATION,
    BALANCED_34B_DATASET_SOURCE,
    MultiViewTaskExample,
    apply_example_control,
    build_multiview_code_patch_splits,
    example_oracle_metadata,
    format_clone_prompt,
    format_full_context_prompt,
)
from src.experiments.architecture_search import (
    StageConfig,
    _agent_config,
    _candidate_config,
    _candidate_specs,
    _coordinator_config,
    _fit_baselines,
    _training_config,
)
from src.experiments.real_shared_weight_latent_coordination import (
    MessageChannelConfig,
    RealSharedWeightTrainingConfig,
    fit_context_baseline,
    fit_latent_system,
    predict_context_baseline,
    predict_latent_system,
    _text_tokens,
)
from src.experiments.run_stage34_candidate_sanitization_diagnostics import (
    _accuracy,
    _structured_oracle_predictions,
    run_diagnostics,
    stage34_dataset_config,
)
from src.experiments.run_stage34_candidate_sanitization_smoke import _run_seed as _run_stage34_smoke_seed
from src.experiments.run_stage34_candidate_sanitization_smoke import _summary as _stage34_smoke_summary
from src.experiments.run_stage35_model_facing_learnability import (
    _all_views_text,
    _model_facing_candidate_bits,
    _model_facing_view_bits,
    _run_tiny_smoke,
    _stage34b_config,
)
from src.experiments.run_stage3_gpu_hard_validation import ARCHITECTURE, _clear_cuda, _configure_cuda, _stage_from_config


DEFAULT_CONFIG = "configs/stage34_candidate_sanitization_cuda_smoke.json"
DEFAULT_OUTPUT = "results/stage36_latent_learnability_bridge.json"
DEFAULT_REPORT = "reports/STAGE36_LATENT_LEARNABILITY_BRIDGE.md"
OVERFIT_TARGET = 0.95


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 3.6 latent learnability bridge diagnostics.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = run_stage36(config, config_path=config_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(result), encoding="utf-8")


def run_stage36(config: Dict[str, object], config_path: Path | None = None) -> Dict[str, object]:
    overfit_config = _stage36_overfit_config(config)
    device, hardware = _configure_cuda(overfit_config)
    stage = _stage_from_config(overfit_config)
    splits = build_multiview_code_patch_splits(stage34_dataset_config(overfit_config), seed=0, repo_root=Path("."))
    diagnostics = _load_stage34b_diagnostics(config, config_path)
    candidate_pair_audit = _candidate_pair_compatibility_audit(splits, diagnostics)

    candidate_pair = _run_candidate_pair_compatibility_mlp(splits, device=device, epochs=200, seed=31)
    trainable_locked = _run_locked_latent_overfit(
        splits,
        stage=stage,
        device=device,
        seed=41,
        trainable_agent=True,
        method="trainable_locked_latent_topk_attention_no_head",
        message_config=_locked_candidate().message_config,
    )
    frozen_locked = _run_locked_latent_overfit(
        splits,
        stage=stage,
        device=device,
        seed=42,
        trainable_agent=False,
        method="frozen_locked_latent_topk_attention_no_head",
        message_config=_locked_candidate().message_config,
    )
    direct_query = _run_direct_candidate_query_bridge(splits, stage=stage, device=device, seed=43, epochs=200)
    topk_no_clone = _run_tiny_cross_encoder_overfit(
        splits,
        device=device,
        seed=44,
        epochs=200,
        method="active_topk_readout_no_cloned_agent_split",
        text_builder=lambda example, index: _all_views_text(example) + "\n" + example.candidates[index].text,
    )
    full_context = _run_full_context_tiny_transformer_overfit(splits, stage=stage, device=device, seed=45)

    overfit_tests = {
        "candidate_pair_compatibility_mlp": candidate_pair,
        "trainable_locked_latent": trainable_locked,
        "frozen_locked_latent": frozen_locked,
        "direct_candidate_query_structured_embeddings": direct_query,
        "active_topk_readout_without_cloned_agent_split": topk_no_clone,
        "single_agent_full_context_tiny_transformer": full_context,
    }

    locked_can_overfit = float(trainable_locked["accuracy"]["train"]) >= OVERFIT_TARGET
    message_probes = _message_probe_suite(trainable_locked["_fit_result"], splits, device=device) if trainable_locked.get("_fit_result") is not None else {}
    readout_diagnostics = (
        _active_readout_diagnostics(trainable_locked["_fit_result"], splits["train"][:32])
        if trainable_locked.get("_fit_result") is not None
        else {}
    )
    topk_tokens = (
        _topk_selected_token_samples(trainable_locked["_fit_result"], splits["train"][:3])
        if trainable_locked.get("_fit_result") is not None
        else []
    )

    bottleneck_ablations: Dict[str, object] = {}
    if not locked_can_overfit:
        bottleneck_ablations = _run_bottleneck_ablations(splits, stage=stage, device=device, locked_result=trainable_locked["_fit_result"])

    inferred_bottleneck = _infer_bottleneck(
        locked_can_overfit=locked_can_overfit,
        overfit_tests=overfit_tests,
        ablations=bottleneck_ablations,
        message_probes=message_probes,
        readout_diagnostics=readout_diagnostics,
    )

    tiny_smoke = None
    if locked_can_overfit:
        tiny_smoke = _run_tiny_smoke(_stage36_smoke_config(config), build_multiview_code_patch_splits(stage34_dataset_config(_stage36_smoke_config(config)), seed=0, repo_root=Path(".")))

    serializable_overfit = {
        name: _drop_nonserializable_fit_result(row)
        for name, row in overfit_tests.items()
    }
    summary = {
        "candidate_pair_compatibility_mlp_model_facing": bool(candidate_pair_audit["uses_only_model_facing_inputs"]),
        "candidate_pair_compatibility_mlp_learnability_valid": bool(candidate_pair_audit["uses_only_model_facing_inputs"]),
        "locked_latent_overfits_64": locked_can_overfit,
        "inferred_bottleneck": inferred_bottleneck,
        "best_non_oracle_overfit_train_accuracy": max(float(row["accuracy"]["train"]) for row in serializable_overfit.values()),
        "best_non_oracle_overfit_test_accuracy": max(float(row["accuracy"]["test"]) for row in serializable_overfit.values()),
        "bottleneck_ablations_ran": bool(bottleneck_ablations),
        "ablation_fixes_overfitting": bool(_best_ablation_train_accuracy(bottleneck_ablations) >= OVERFIT_TARGET),
        "tiny_smoke_ran": tiny_smoke is not None,
        "full_validation_justified": bool(tiny_smoke and tiny_smoke.get("summary", {}).get("smoke_passed", False)),
    }
    # This stage is diagnostic only. Full validation also requires a successful locked-latent bridge, so
    # keep the justification false unless the gated one-seed smoke actually ran and passed.
    if not locked_can_overfit:
        summary["full_validation_justified"] = False

    return {
        "metadata": {
            "stage": "stage3.6_latent_learnability_bridge",
            "created_at_utc": _now(),
            "config_path": str(config_path) if config_path else None,
            "dataset_source": BALANCED_34B_DATASET_SOURCE,
            "candidate_representation": BALANCED_34B_CANDIDATE_REPRESENTATION,
            "architecture": ARCHITECTURE,
            "architecture_changes": "none",
            "benchmark_changes": "none",
            "scope": "64-example overfit diagnostics plus gated one-seed smoke only; no full 10-seed validation",
            "device": device,
            "hardware": hardware,
            "stage_config": asdict(stage),
        },
        "dataset_diagnostics": diagnostics,
        "candidate_pair_compatibility_audit": candidate_pair_audit,
        "overfit_tests": serializable_overfit,
        "message_probes": message_probes,
        "active_readout_diagnostics": readout_diagnostics,
        "topk_selected_token_samples": topk_tokens,
        "bottleneck_ablations": bottleneck_ablations,
        "tiny_smoke": tiny_smoke,
        "summary": summary,
    }


def _stage36_overfit_config(config: Dict[str, object]) -> Dict[str, object]:
    out = _stage34b_config(config)
    stage = dict(out["stage"])
    stage.update(
        {
            "name": "stage36_overfit64",
            "n_train": 64,
            "n_dev": 64,
            "n_test": 64,
            "seeds": [0],
            "epochs": 200,
            "patience": 201,
            "batch_size": 16,
            "lr": 0.002,
            "weight_decay": 0.0001,
            "mixed_precision": stage.get("mixed_precision", "bf16"),
        }
    )
    out["stage"] = stage
    out["dataset_config"] = {
        **dict(out.get("dataset_config", {})),
        "dataset_source": BALANCED_34B_DATASET_SOURCE,
        "generator_version": "real_import_restore_candidate_balanced_34b_stage36_overfit",
        "candidate_representation": BALANCED_34B_CANDIDATE_REPRESENTATION,
    }
    return out


def _stage36_smoke_config(config: Dict[str, object]) -> Dict[str, object]:
    out = _stage34b_config(config)
    out["stage"] = {
        **dict(out["stage"]),
        "name": "stage36_smoke",
        "n_train": 256,
        "n_dev": 128,
        "n_test": 256,
        "seeds": [0],
    }
    out["dataset_config"] = {
        **dict(out.get("dataset_config", {})),
        "dataset_source": BALANCED_34B_DATASET_SOURCE,
        "generator_version": "real_import_restore_candidate_balanced_34b_stage36_smoke",
        "candidate_representation": BALANCED_34B_CANDIDATE_REPRESENTATION,
    }
    return out


def _load_stage34b_diagnostics(config: Dict[str, object], config_path: Path | None) -> Dict[str, object]:
    stage35_path = Path("results/stage35_model_facing_learnability.json")
    if stage35_path.exists():
        data = json.loads(stage35_path.read_text(encoding="utf-8"))
        variants = data.get("stage34b_variants", [])
        if variants:
            diagnostics = variants[0].get("dataset_diagnostics")
            if isinstance(diagnostics, dict):
                return diagnostics
    return run_diagnostics(_stage34b_config(config), config_path=config_path)


def _candidate_pair_compatibility_audit(
    splits: Dict[str, Sequence[MultiViewTaskExample]],
    diagnostics: Dict[str, object],
) -> Dict[str, object]:
    summary = diagnostics.get("summary", {}) if isinstance(diagnostics, dict) else {}
    candidate_only_acc = float(summary.get("mean_candidate_only_accuracy", 1.0))
    metadata_only_acc = float(summary.get("mean_candidate_metadata_only_accuracy", 1.0))
    view_masked_acc = float(summary.get("mean_view_masked_candidates_visible_accuracy", 1.0))
    candidate_only_predictive = candidate_only_acc > 0.18 or metadata_only_acc > 0.18 or view_masked_acc > 0.18
    table = [
        {
            "field_name": "view_bits[4]",
            "source": "model-facing private view text parsed for balanced category phrases",
            "model_facing": True,
            "candidate_only_predictive": False,
            "allowed": True,
        },
        {
            "field_name": "candidate_bits[4]",
            "source": "model-facing candidate.attributes balanced_categories_v3",
            "model_facing": True,
            "candidate_only_predictive": candidate_only_predictive,
            "allowed": not candidate_only_predictive,
        },
        {
            "field_name": "equality_bits[4]",
            "source": "derived comparison of model-facing view_bits and candidate_bits",
            "model_facing": True,
            "candidate_only_predictive": False,
            "allowed": True,
        },
        {
            "field_name": "mean_equality",
            "source": "derived aggregate of equality_bits",
            "model_facing": True,
            "candidate_only_predictive": False,
            "allowed": True,
        },
        {
            "field_name": "valid_parse_flag",
            "source": "derived from whether model-facing category phrases are present",
            "model_facing": True,
            "candidate_only_predictive": False,
            "allowed": True,
        },
    ]
    forbidden = {
        "oracle_metadata",
        "gold_tuple",
        "candidate_bit_tuples",
        "candidate_tuples",
        "candidate_order",
        "candidate_id",
        "patch_hash",
        "source_path",
        "candidate_index",
        "role_pair_id",
        "problem_family",
    }
    sample = splits["train"][0]
    sample_features = _compatibility_features([sample])[0].tolist()
    feature_shape = list(_compatibility_features(splits["train"][:2]).shape)
    received_field_names = [row["field_name"] for row in table]
    uses_forbidden = bool(set(received_field_names) & forbidden)
    return {
        "method": "candidate_pair_compatibility_mlp",
        "implementation": "src.experiments.run_stage35_model_facing_learnability.CompatibilityFeatureScorer",
        "uses_only_model_facing_inputs": bool(not uses_forbidden and all(bool(row["allowed"]) and bool(row["model_facing"]) for row in table)),
        "uses_forbidden_or_oracle_like_fields": uses_forbidden,
        "forbidden_fields_checked": sorted(forbidden),
        "field_table": table,
        "diagnostic_candidate_only_accuracy": candidate_only_acc,
        "diagnostic_candidate_metadata_only_accuracy": metadata_only_acc,
        "diagnostic_view_masked_candidates_visible_accuracy": view_masked_acc,
        "feature_shape": feature_shape,
        "sample_feature_matrix_first_example": sample_features,
        "explicitly_not_used": sorted(forbidden),
    }


def _compatibility_features(examples: Sequence[MultiViewTaskExample]) -> np.ndarray:
    rows = []
    for example in examples:
        view_bits = np.asarray(_model_facing_view_bits(example), dtype=np.float32)
        per_candidate = []
        for candidate in example.candidates:
            candidate_bits = np.asarray(_model_facing_candidate_bits(candidate.attributes), dtype=np.float32)
            equality = (candidate_bits == view_bits).astype(np.float32)
            valid = np.asarray([float(np.all(candidate_bits >= 0.0) and np.all(view_bits >= 0.0))], dtype=np.float32)
            if not bool(valid[0]):
                candidate_bits = np.zeros(4, dtype=np.float32)
                view_for_features = np.zeros(4, dtype=np.float32)
                equality = np.zeros(4, dtype=np.float32)
            else:
                view_for_features = view_bits
            per_candidate.append(
                np.concatenate(
                    [view_for_features, candidate_bits, equality, np.asarray([equality.mean()], dtype=np.float32), valid]
                )
            )
        rows.append(per_candidate)
    return np.asarray(rows, dtype=np.float32)


def _candidate_bits_features(examples: Sequence[MultiViewTaskExample]) -> np.ndarray:
    rows = []
    for example in examples:
        rows.append([_model_facing_candidate_bits(candidate.attributes) for candidate in example.candidates])
    return np.asarray(rows, dtype=np.float32)


def _view_bits(examples: Sequence[MultiViewTaskExample]) -> np.ndarray:
    return np.asarray([_model_facing_view_bits(example) for example in examples], dtype=np.int64)


def _candidate_bits(examples: Sequence[MultiViewTaskExample]) -> np.ndarray:
    return np.asarray([[_model_facing_candidate_bits(candidate.attributes) for candidate in example.candidates] for example in examples], dtype=np.float32)


def _run_candidate_pair_compatibility_mlp(
    splits: Dict[str, Sequence[MultiViewTaskExample]],
    device: str,
    epochs: int,
    seed: int,
) -> Dict[str, object]:
    return _fit_candidatewise_mlp(
        method="candidate_pair_compatibility_mlp",
        x_train=_compatibility_features(splits["train"]),
        y_train=_labels(splits["train"]),
        x_dev=_compatibility_features(splits["dev"]),
        y_dev=_labels(splits["dev"]),
        x_test=_compatibility_features(splits["test"]),
        y_test=_labels(splits["test"]),
        device=device,
        seed=seed,
        epochs=epochs,
        lr=0.003,
        weight_decay=0.0,
        hidden_dims=(64, 64),
    )


class _CandidatewiseMLP(nn.Module):
    def __init__(self, feature_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        previous = int(feature_dim)
        for hidden in hidden_dims:
            layers.extend([nn.Linear(previous, int(hidden)), nn.ReLU()])
            previous = int(hidden)
        layers.append(nn.Linear(previous, 1))
        self.scorer = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, candidates, dim = x.shape
        return self.scorer(x.reshape(batch * candidates, dim)).reshape(batch, candidates)


def _fit_candidatewise_mlp(
    method: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    device: str,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dims: Sequence[int],
) -> Dict[str, object]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch_device = torch.device(device)
    model = _CandidatewiseMLP(feature_dim=x_train.shape[-1], hidden_dims=hidden_dims).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    tx = torch.as_tensor(x_train, dtype=torch.float32, device=torch_device)
    ty = torch.as_tensor(y_train, dtype=torch.long, device=torch_device)
    dx = torch.as_tensor(x_dev, dtype=torch.float32, device=torch_device)
    dy = torch.as_tensor(y_dev, dtype=torch.long, device=torch_device)
    vx = torch.as_tensor(x_test, dtype=torch.float32, device=torch_device)
    vy = torch.as_tensor(y_test, dtype=torch.long, device=torch_device)
    initial = _flat_parameters(model)
    history: List[Dict[str, float]] = []
    grad_norms: List[float] = []
    batch_size = min(32, len(x_train))
    for epoch in range(int(epochs)):
        model.train()
        order = rng.permutation(len(x_train))
        loss_sum = 0.0
        count = 0
        for start in range(0, len(order), batch_size):
            idx_np = order[start : start + batch_size]
            idx = torch.as_tensor(idx_np, dtype=torch.long, device=torch_device)
            logits = model(tx.index_select(0, idx))
            loss = F.cross_entropy(logits, ty.index_select(0, idx))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norms.append(_grad_norm(model.parameters()))
            optimizer.step()
            loss_sum += float(loss.detach().cpu().item()) * len(idx_np)
            count += len(idx_np)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(loss_sum / max(1, count)),
                "train_acc": _torch_acc(model, tx, ty),
                "dev_acc": _torch_acc(model, dx, dy),
                "test_acc": _torch_acc(model, vx, vy),
            }
        )
    return _overfit_row(
        method=method,
        history=history,
        train_acc=history[-1]["train_acc"],
        dev_acc=history[-1]["dev_acc"],
        test_acc=history[-1]["test_acc"],
        grad_norm_mean=float(mean(grad_norms)) if grad_norms else 0.0,
        param_delta=float(torch.linalg.vector_norm(_flat_parameters(model) - initial).detach().cpu().item()),
        extra={"feature_dim": int(x_train.shape[-1]), "hidden_dims": list(hidden_dims)},
    )


class _DirectCandidateQueryBridge(nn.Module):
    def __init__(self, message_dim: int, coordinator_config: LatentCoordinatorConfig) -> None:
        super().__init__()
        self.value_embedding = nn.Embedding(3, message_dim)
        self.coordinator = make_candidate_query_coordinator(
            n_roles=4,
            candidate_feature_dim=4,
            config=replace(coordinator_config, input_dim=message_dim, family="cross_attention"),
        )

    def forward(self, view_bits: torch.Tensor, candidate_bits: torch.Tensor) -> torch.Tensor:
        role_ids = torch.arange(4, dtype=torch.long, device=view_bits.device).view(1, 4).expand(view_bits.shape[0], 4)
        value_ids = view_bits.clamp(min=-1, max=1).to(dtype=torch.long) + 1
        messages = self.value_embedding(value_ids)
        return self.coordinator(messages, role_ids, candidate_bits.float())


def _run_direct_candidate_query_bridge(
    splits: Dict[str, Sequence[MultiViewTaskExample]],
    stage: StageConfig,
    device: str,
    seed: int,
    epochs: int,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch_device = torch.device(device)
    model = _DirectCandidateQueryBridge(message_dim=16, coordinator_config=_coordinator_config(stage, family="cross_attention", num_layers=1, dropout=0.0)).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    train = _structured_tensors(splits["train"], torch_device)
    dev = _structured_tensors(splits["dev"], torch_device)
    test = _structured_tensors(splits["test"], torch_device)
    initial = _flat_parameters(model)
    history: List[Dict[str, float]] = []
    grad_norms: List[float] = []
    batch_size = min(32, len(splits["train"]))
    for epoch in range(int(epochs)):
        model.train()
        order = rng.permutation(len(splits["train"]))
        loss_sum = 0.0
        count = 0
        for start in range(0, len(order), batch_size):
            idx_np = order[start : start + batch_size]
            idx = torch.as_tensor(idx_np, dtype=torch.long, device=torch_device)
            logits = model(train["views"].index_select(0, idx), train["candidates"].index_select(0, idx))
            loss = F.cross_entropy(logits, train["labels"].index_select(0, idx))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norms.append(_grad_norm(model.parameters()))
            optimizer.step()
            loss_sum += float(loss.detach().cpu().item()) * len(idx_np)
            count += len(idx_np)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(loss_sum / max(1, count)),
                "train_acc": _structured_acc(model, train),
                "dev_acc": _structured_acc(model, dev),
                "test_acc": _structured_acc(model, test),
            }
        )
    return _overfit_row(
        method="direct_candidate_query_structured_embeddings",
        history=history,
        train_acc=history[-1]["train_acc"],
        dev_acc=history[-1]["dev_acc"],
        test_acc=history[-1]["test_acc"],
        grad_norm_mean=float(mean(grad_norms)) if grad_norms else 0.0,
        param_delta=float(torch.linalg.vector_norm(_flat_parameters(model) - initial).detach().cpu().item()),
        extra={"message_dim": 16, "coordinator_family": "candidate_query_cross_attention"},
    )


def _structured_tensors(examples: Sequence[MultiViewTaskExample], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "views": torch.as_tensor(_view_bits(examples), dtype=torch.long, device=device),
        "candidates": torch.as_tensor(_candidate_bits(examples), dtype=torch.float32, device=device),
        "labels": torch.as_tensor(_labels(examples), dtype=torch.long, device=device),
    }


def _run_locked_latent_overfit(
    splits: Dict[str, Sequence[MultiViewTaskExample]],
    stage: StageConfig,
    device: str,
    seed: int,
    trainable_agent: bool,
    method: str,
    message_config: MessageChannelConfig,
) -> Dict[str, object]:
    result = fit_latent_system(
        train_examples=splits["train"],
        dev_examples=splits["train"],
        agent_config=_agent_config(stage),
        coordinator_config=_locked_candidate().coordinator_config,
        training_config=_training_config(stage),
        num_classes=8,
        seed=seed,
        device=device,
        trainable_agent=trainable_agent,
        method=method,
        message_config=message_config,
    )
    row = _latent_overfit_row(method, result, splits, seed)
    row["_fit_result"] = result
    return row


class _TinyPairEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, max_len: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.position = nn.Embedding(max_len, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=2,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch, candidates, length = token_ids.shape
        flat = token_ids.reshape(batch * candidates, length)
        positions = torch.arange(length, dtype=torch.long, device=flat.device)
        encoded = self.encoder(self.embedding(flat) + self.position(positions).unsqueeze(0), src_key_padding_mask=flat.eq(0))
        mask = flat.ne(0).unsqueeze(-1).to(dtype=encoded.dtype)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.head(pooled).reshape(batch, candidates)


def _run_tiny_cross_encoder_overfit(
    splits: Dict[str, Sequence[MultiViewTaskExample]],
    device: str,
    seed: int,
    epochs: int,
    method: str,
    text_builder: Callable[[MultiViewTaskExample, int], str],
) -> Dict[str, object]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch_device = torch.device(device)
    vocab_size = 2048
    max_len = 128
    model = _TinyPairEncoder(vocab_size=vocab_size, hidden_dim=48, max_len=max_len).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    train = _encoded_candidate_texts(splits["train"], text_builder, vocab_size, max_len, torch_device)
    dev = _encoded_candidate_texts(splits["dev"], text_builder, vocab_size, max_len, torch_device)
    test = _encoded_candidate_texts(splits["test"], text_builder, vocab_size, max_len, torch_device)
    y_train = torch.as_tensor(_labels(splits["train"]), dtype=torch.long, device=torch_device)
    y_dev = torch.as_tensor(_labels(splits["dev"]), dtype=torch.long, device=torch_device)
    y_test = torch.as_tensor(_labels(splits["test"]), dtype=torch.long, device=torch_device)
    initial = _flat_parameters(model)
    history: List[Dict[str, float]] = []
    grad_norms: List[float] = []
    batch_size = min(16, len(splits["train"]))
    for epoch in range(int(epochs)):
        model.train()
        order = rng.permutation(len(splits["train"]))
        loss_sum = 0.0
        count = 0
        for start in range(0, len(order), batch_size):
            idx_np = order[start : start + batch_size]
            idx = torch.as_tensor(idx_np, dtype=torch.long, device=torch_device)
            logits = model(train.index_select(0, idx))
            loss = F.cross_entropy(logits, y_train.index_select(0, idx))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norms.append(_grad_norm(model.parameters()))
            optimizer.step()
            loss_sum += float(loss.detach().cpu().item()) * len(idx_np)
            count += len(idx_np)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(loss_sum / max(1, count)),
                "train_acc": _torch_acc(model, train, y_train),
                "dev_acc": _torch_acc(model, dev, y_dev),
                "test_acc": _torch_acc(model, test, y_test),
            }
        )
    return _overfit_row(
        method=method,
        history=history,
        train_acc=history[-1]["train_acc"],
        dev_acc=history[-1]["dev_acc"],
        test_acc=history[-1]["test_acc"],
        grad_norm_mean=float(mean(grad_norms)) if grad_norms else 0.0,
        param_delta=float(torch.linalg.vector_norm(_flat_parameters(model) - initial).detach().cpu().item()),
        extra={"vocab_size": vocab_size, "max_len": max_len, "hidden_dim": 48},
    )


def _run_full_context_tiny_transformer_overfit(
    splits: Dict[str, Sequence[MultiViewTaskExample]],
    stage: StageConfig,
    device: str,
    seed: int,
) -> Dict[str, object]:
    result = fit_context_baseline(
        train_examples=splits["train"],
        dev_examples=splits["train"],
        agent_config=_agent_config(stage),
        training_config=_training_config(stage),
        num_classes=8,
        seed=seed,
        device=device,
        method="single_agent_full_context_tiny_transformer",
        prompt_mode="full",
    )
    train_acc = _accuracy(predict_context_baseline(result, splits["train"]), _labels(splits["train"]))
    dev_acc = _accuracy(predict_context_baseline(result, splits["dev"]), _labels(splits["dev"]))
    test_acc = _accuracy(predict_context_baseline(result, splits["test"]), _labels(splits["test"]))
    history = [
        {
            "epoch": float(row.get("epoch", 0.0)),
            "train_acc": float(row.get("train_acc", 0.0)),
            "dev_acc": float(row.get("dev_acc", 0.0)),
            "test_acc": float("nan"),
        }
        for row in result.history
    ]
    if history:
        history[-1]["train_acc"] = train_acc
        history[-1]["dev_acc"] = dev_acc
        history[-1]["test_acc"] = test_acc
    return _overfit_row(
        method="single_agent_full_context_tiny_transformer",
        history=history,
        train_acc=train_acc,
        dev_acc=dev_acc,
        test_acc=test_acc,
        grad_norm_mean=None,
        param_delta=None,
        extra={"param_count": int(result.param_count), "prompt_mode": result.prompt_mode, "loss_curve_available": False},
    )


def _run_bottleneck_ablations(
    splits: Dict[str, Sequence[MultiViewTaskExample]],
    stage: StageConfig,
    device: str,
    locked_result,
) -> Dict[str, object]:
    readout_variants: Dict[str, object] = {}
    locked = _locked_candidate()
    for name, message_config in {
        "topk_attention_no_head_locked": locked.message_config,
        "mean_pooled_messages": replace(locked.message_config, readout_source="pooled", use_message_head=False),
        "attention_pool_no_head": replace(locked.message_config, active_message_readout_type="attention_pool", use_message_head=False),
        "active_query_no_head": replace(locked.message_config, active_message_readout_type="active_query", use_message_head=False),
    }.items():
        ablation_stage = replace(stage, epochs=50, patience=51)
        readout_variants[name] = _drop_nonserializable_fit_result(
            _run_locked_latent_overfit(
                splits,
                stage=ablation_stage,
                device=device,
                seed=1000 + len(readout_variants),
                trainable_agent=True,
                method=f"readout_ablation_{name}",
                message_config=message_config,
            )
        )
        _clear_cuda()

    candidate_query_mlp = _fit_candidatewise_mlp(
        method="candidate_query_bottleneck_concat_mlp_clean_fields",
        x_train=_compatibility_features(splits["train"]),
        y_train=_labels(splits["train"]),
        x_dev=_compatibility_features(splits["dev"]),
        y_dev=_labels(splits["dev"]),
        x_test=_compatibility_features(splits["test"]),
        y_test=_labels(splits["test"]),
        device=device,
        seed=1100,
        epochs=100,
        lr=0.003,
        weight_decay=0.0,
        hidden_dims=(64, 64),
    )

    lr_sweep: List[Dict[str, object]] = []
    for lr in (1e-4, 3e-4, 1e-3, 3e-3):
        for epochs in (50, 100, 200):
            sweep_stage = replace(stage, epochs=int(epochs), patience=int(epochs) + 1, lr=float(lr), weight_decay=0.0)
            row = _drop_nonserializable_fit_result(
                _run_locked_latent_overfit(
                    splits,
                    stage=sweep_stage,
                    device=device,
                    seed=1200 + len(lr_sweep),
                    trainable_agent=True,
                    method=f"training_sweep_lr_{lr:g}_epochs_{epochs}",
                    message_config=locked.message_config,
                )
            )
            row["lr"] = float(lr)
            row["epochs"] = int(epochs)
            row["weight_decay"] = 0.0
            row["gradient_clip_norm"] = 0.0
            lr_sweep.append(row)
            _clear_cuda()

    clip_stage = replace(stage, epochs=200, patience=201, weight_decay=0.0)
    clip_training = replace(_training_config(clip_stage), gradient_clip_norm=1.0)
    clipped = fit_latent_system(
        train_examples=splits["train"],
        dev_examples=splits["train"],
        agent_config=_agent_config(clip_stage),
        coordinator_config=locked.coordinator_config,
        training_config=clip_training,
        num_classes=8,
        seed=1300,
        device=device,
        trainable_agent=True,
        method="training_sweep_gradient_clip_1_no_weight_decay",
        message_config=locked.message_config,
    )
    clipping_row = _drop_nonserializable_fit_result(_latent_overfit_row("training_sweep_gradient_clip_1_no_weight_decay", clipped, splits, 1300))

    representation = {
        "natural_language_phrases_locked_topk": readout_variants["topk_attention_no_head_locked"],
        "learned_category_embeddings_direct_candidate_query": _run_direct_candidate_query_bridge(splits, stage=stage, device=device, seed=1400, epochs=100),
        "compatibility_feature_mlp": candidate_query_mlp,
    }
    return {
        "readout_bottleneck": readout_variants,
        "candidate_query_bottleneck": {"concat_mlp_over_clean_fields": candidate_query_mlp},
        "training_bottleneck": {"lr_epoch_sweep": lr_sweep, "gradient_clipping_on": clipping_row, "dropout": 0.0, "weight_decay": 0.0},
        "representation_bottleneck": representation,
    }


def _message_probe_suite(fit_result, splits: Dict[str, Sequence[MultiViewTaskExample]], device: str) -> Dict[str, object]:
    train_messages = _collect_messages(fit_result, splits["train"])
    dev_messages = _collect_messages(fit_result, splits["dev"])
    test_messages = _collect_messages(fit_result, splits["test"])
    train_labels = _view_bits(splits["train"])
    dev_labels = _view_bits(splits["dev"])
    test_labels = _view_bits(splits["test"])
    rows = []
    for role in range(train_messages.shape[1]):
        row = _fit_message_probe(
            role,
            train_messages[:, role, :],
            train_labels[:, role],
            dev_messages[:, role, :],
            dev_labels[:, role],
            test_messages[:, role, :],
            test_labels[:, role],
            device=device,
            seed=2000 + role,
        )
        rows.append(row)
    return {
        "rows": rows,
        "mean_train_accuracy": float(mean(float(row["train_acc"]) for row in rows)) if rows else 0.0,
        "mean_dev_accuracy": float(mean(float(row["dev_acc"]) for row in rows)) if rows else 0.0,
        "mean_test_accuracy": float(mean(float(row["test_acc"]) for row in rows)) if rows else 0.0,
        "interpretation": "message probes use oracle labels only for diagnostics; probe inputs are learned clone messages",
    }


def _fit_message_probe(
    role: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    device: str,
    seed: int,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    torch_device = torch.device(device)
    model = nn.Sequential(nn.LayerNorm(x_train.shape[-1]), nn.Linear(x_train.shape[-1], 2)).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    tx = torch.as_tensor(x_train, dtype=torch.float32, device=torch_device)
    ty = torch.as_tensor(y_train, dtype=torch.long, device=torch_device).clamp(min=0, max=1)
    dx = torch.as_tensor(x_dev, dtype=torch.float32, device=torch_device)
    dy = torch.as_tensor(y_dev, dtype=torch.long, device=torch_device).clamp(min=0, max=1)
    vx = torch.as_tensor(x_test, dtype=torch.float32, device=torch_device)
    vy = torch.as_tensor(y_test, dtype=torch.long, device=torch_device).clamp(min=0, max=1)
    for _epoch in range(100):
        model.train()
        logits = model(tx)
        loss = F.cross_entropy(logits, ty)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return {
        "role": int(role),
        "role_name": f"{ATTRIBUTE_VALUE_PAIRS[role][0]} / {ATTRIBUTE_VALUE_PAIRS[role][1]}" if role < len(ATTRIBUTE_VALUE_PAIRS) else str(role),
        "train_acc": _torch_class_acc(model, tx, ty),
        "dev_acc": _torch_class_acc(model, dx, dy),
        "test_acc": _torch_class_acc(model, vx, vy),
    }


def _collect_messages(fit_result, examples: Sequence[MultiViewTaskExample]) -> np.ndarray:
    fit_result.system.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(examples), 32):
            batch = list(examples[start : start + 32])
            readouts = fit_result.system.collect_clone_representations(batch)
            rows.append(readouts["message"].detach().float().cpu().numpy())
    return np.concatenate(rows, axis=0)


def _active_readout_diagnostics(fit_result, examples: Sequence[MultiViewTaskExample]) -> Dict[str, object]:
    messages = _collect_messages(fit_result, examples)
    flat = torch.as_tensor(messages.reshape(-1, messages.shape[-1]), dtype=torch.float32)
    variance = float(flat.var(dim=0, unbiased=False).mean().item()) if flat.numel() else 0.0
    cosine = 0.0
    if flat.shape[0] > 1:
        normalized = F.normalize(flat, dim=-1)
        sims = normalized @ normalized.T
        mask = ~torch.eye(sims.shape[0], dtype=torch.bool)
        cosine = float(sims[mask].mean().item())
    return {
        "message_variance_mean": variance,
        "mean_pairwise_cosine": cosine,
        "messages_shape": list(messages.shape),
    }


def _topk_selected_token_samples(fit_result, examples: Sequence[MultiViewTaskExample]) -> List[Dict[str, object]]:
    system = fit_result.system
    active = system.active_message_readout
    if active is None or getattr(active, "readout_type", "") != "topk_attention":
        return []
    out = []
    system.eval()
    with torch.no_grad():
        for example in examples:
            role_rows = []
            for role_index in range(len(example.views)):
                text = format_clone_prompt(example, role_index)
                readouts = system.shared_agent.forward_texts_with_readouts(
                    [text],
                    use_msg_token=system.message_config.use_msg_token,
                    msg_position=system.message_config.msg_position,
                    selected_layer_ids=system.message_config.active_message_layers,
                )
                token_states = readouts["token_states"][0]
                token_mask = readouts["token_mask"][0].bool()
                role_id = torch.as_tensor([role_index], dtype=torch.long, device=token_states.device)
                role_embedding = active.role_embedding(role_id)[0]
                scores = active.pool_scorer(token_states + role_embedding.unsqueeze(0)).squeeze(-1)
                scores = scores.masked_fill(~token_mask, -1e9)
                k = min(8, int(token_mask.sum().item()))
                values, indices = torch.topk(scores, k=k)
                raw_tokens = _text_tokens(text)[: system.shared_agent.max_length]
                token_rows = []
                for value, index in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()):
                    base_index = int(index) % max(1, len(raw_tokens))
                    layer_index = int(index) // max(1, len(raw_tokens))
                    token = raw_tokens[base_index] if raw_tokens else ""
                    token_rows.append({"position": int(index), "token": token, "layer_repeat": layer_index, "score": float(value)})
                role_rows.append({"role": int(role_index), "top_tokens": token_rows})
            out.append({"example_id": example.id, "gold_index": int(example.label), "roles": role_rows})
    return out


def _latent_overfit_row(method: str, result, splits: Dict[str, Sequence[MultiViewTaskExample]], seed: int) -> Dict[str, object]:
    train_acc = _accuracy(predict_latent_system(result, splits["train"], "none", seed), _labels(splits["train"]))
    dev_acc = _accuracy(predict_latent_system(result, splits["dev"], "none", seed), _labels(splits["dev"]))
    test_acc = _accuracy(predict_latent_system(result, splits["test"], "none", seed), _labels(splits["test"]))
    history = [
        {
            "epoch": float(row.get("epoch", 0.0)),
            "train_loss": float(row.get("train_loss", float("nan"))),
            "train_acc": float(row.get("train_acc", 0.0)),
            "dev_acc": float(row.get("dev_acc", 0.0)),
            "test_acc": float("nan"),
        }
        for row in result.history
    ]
    if history:
        history[-1]["train_acc"] = train_acc
        history[-1]["dev_acc"] = dev_acc
        history[-1]["test_acc"] = test_acc
    return _overfit_row(
        method=method,
        history=history,
        train_acc=train_acc,
        dev_acc=dev_acc,
        test_acc=test_acc,
        grad_norm_mean=float(result.audit.get("agent_grad_norm_mean", 0.0)) + float(result.audit.get("coordinator_grad_norm_mean", 0.0)),
        param_delta=float(result.audit.get("agent_parameter_delta", 0.0)) + float(result.audit.get("coordinator_parameter_delta", 0.0)) + float(result.audit.get("active_message_readout_parameter_delta", 0.0)),
        extra={
            "param_count": int(result.param_count),
            "audit": _audit_subset_for_stage36(result.audit),
            "epochs_run": int(result.audit.get("epochs_run", len(result.history))),
        },
    )


def _overfit_row(
    method: str,
    history: Sequence[Dict[str, float]],
    train_acc: float,
    dev_acc: float,
    test_acc: float,
    grad_norm_mean: float | None,
    param_delta: float | None,
    extra: Dict[str, object] | None = None,
) -> Dict[str, object]:
    loss_curve = [float(row["train_loss"]) for row in history if "train_loss" in row and not math.isnan(float(row["train_loss"]))]
    return {
        "method": method,
        "accuracy": {"train": float(train_acc), "dev": float(dev_acc), "test": float(test_acc)},
        "max_train_accuracy": float(max([row.get("train_acc", 0.0) for row in history] or [train_acc])),
        "epochs_to_train_accuracy": _epochs_to_thresholds(history),
        "loss_curve": loss_curve,
        "history": list(history),
        "gradient_norm_mean": None if grad_norm_mean is None else float(grad_norm_mean),
        "parameter_delta": None if param_delta is None else float(param_delta),
        "overfit_passed": bool(float(train_acc) >= OVERFIT_TARGET),
        **(extra or {}),
    }


def _epochs_to_thresholds(history: Sequence[Dict[str, float]]) -> Dict[str, int | None]:
    out: Dict[str, int | None] = {}
    for threshold in (0.5, 0.8, 0.95):
        key = f"{threshold:.2f}"
        out[key] = None
        for row in history:
            if float(row.get("train_acc", 0.0)) >= threshold:
                out[key] = int(row.get("epoch", 0))
                break
    return out


def _drop_nonserializable_fit_result(row: Dict[str, object]) -> Dict[str, object]:
    return {key: value for key, value in row.items() if key != "_fit_result"}


def _infer_bottleneck(
    locked_can_overfit: bool,
    overfit_tests: Dict[str, Dict[str, object]],
    ablations: Dict[str, object],
    message_probes: Dict[str, object],
    readout_diagnostics: Dict[str, object],
) -> str:
    if locked_can_overfit:
        return "none"
    direct_train = float(overfit_tests["direct_candidate_query_structured_embeddings"]["accuracy"]["train"])
    candidate_mlp_train = float(overfit_tests["candidate_pair_compatibility_mlp"]["accuracy"]["train"])
    readout_best = _best_ablation_train_accuracy(ablations.get("readout_bottleneck", {}) if isinstance(ablations, dict) else {})
    training_best = _best_ablation_train_accuracy(ablations.get("training_bottleneck", {}) if isinstance(ablations, dict) else {})
    probe_mean = float(message_probes.get("mean_train_accuracy", 0.0)) if isinstance(message_probes, dict) else 0.0
    variance = float(readout_diagnostics.get("message_variance_mean", 0.0)) if isinstance(readout_diagnostics, dict) else 0.0
    if readout_best >= OVERFIT_TARGET:
        return "readout"
    if direct_train >= OVERFIT_TARGET and candidate_mlp_train >= OVERFIT_TARGET and (probe_mean < 0.80 or variance < 1e-4):
        return "message"
    if direct_train >= OVERFIT_TARGET and candidate_mlp_train >= OVERFIT_TARGET and training_best < OVERFIT_TARGET:
        return "representation"
    if training_best >= OVERFIT_TARGET:
        return "training"
    return "coordinator"


def _best_ablation_train_accuracy(value: object) -> float:
    best = 0.0
    if isinstance(value, dict):
        for child in value.values():
            best = max(best, _best_ablation_train_accuracy(child))
    elif isinstance(value, list):
        for child in value:
            best = max(best, _best_ablation_train_accuracy(child))
    elif isinstance(value, tuple):
        for child in value:
            best = max(best, _best_ablation_train_accuracy(child))
    elif isinstance(value, (int, float)):
        best = max(best, 0.0)
    if isinstance(value, dict) and "accuracy" in value and isinstance(value["accuracy"], dict):
        best = max(best, float(value["accuracy"].get("train", 0.0)))
    return float(best)


def _audit_subset_for_stage36(audit: Dict[str, object]) -> Dict[str, object]:
    keys = (
        "agent_grad_norm_mean",
        "coordinator_grad_norm_mean",
        "active_message_readout_grad_norm_mean",
        "message_head_grad_norm_mean",
        "agent_parameter_delta",
        "coordinator_parameter_delta",
        "active_message_readout_parameter_delta",
        "message_head_parameter_delta",
        "loss_backward_reaches_shared_agent",
        "loss_backward_reaches_active_message_readout",
        "per_clone_gradient_contribution",
        "per_clone_activation_grad_norms",
        "message_config",
        "gradient_clip_norm",
        "epochs_run",
    )
    return {key: audit.get(key) for key in keys}


def _encoded_candidate_texts(
    examples: Sequence[MultiViewTaskExample],
    text_builder: Callable[[MultiViewTaskExample, int], str],
    vocab_size: int,
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    rows = []
    for example in examples:
        per_candidate = []
        for index in range(len(example.candidates)):
            ids = [_token_id(token, vocab_size) for token in _text_tokens(text_builder(example, index))[:max_len]]
            ids.extend([0] * max(0, max_len - len(ids)))
            per_candidate.append(ids[:max_len])
        rows.append(per_candidate)
    return torch.as_tensor(rows, dtype=torch.long, device=device)


def _token_id(token: str, vocab_size: int) -> int:
    raw = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return 1 + (int(raw[:8], 16) % max(1, vocab_size - 1))


def _labels(examples: Sequence[MultiViewTaskExample]) -> np.ndarray:
    return np.asarray([example.label for example in examples], dtype=np.int64)


def _torch_acc(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(x), dim=1)
        return float((preds == y).float().mean().detach().cpu().item())


def _structured_acc(model: _DirectCandidateQueryBridge, tensors: Dict[str, torch.Tensor]) -> float:
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(tensors["views"], tensors["candidates"]), dim=1)
        return float((preds == tensors["labels"]).float().mean().detach().cpu().item())


def _torch_class_acc(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(x), dim=1)
        return float((preds == y).float().mean().detach().cpu().item())


def _flat_parameters(module: nn.Module) -> torch.Tensor:
    values = [parameter.detach().float().reshape(-1).cpu() for parameter in module.parameters()]
    return torch.cat(values) if values else torch.zeros(0)


def _grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().float().pow(2).sum().cpu().item())
    return float(total**0.5)


def _locked_candidate():
    return next(candidate for candidate in _candidate_specs() if candidate.name == ARCHITECTURE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _render_report(result: Dict[str, object]) -> str:
    summary = result.get("summary", {})
    diagnostics = result.get("dataset_diagnostics", {}).get("summary", {})
    audit = result.get("candidate_pair_compatibility_audit", {})
    overfit = result.get("overfit_tests", {})
    ablations = result.get("bottleneck_ablations", {})
    probes = result.get("message_probes", {})
    smoke = result.get("tiny_smoke")
    lines = [
        "# Stage 3.6 Latent Learnability Bridge",
        "",
        "## Scope",
        "",
        f"- Dataset mode: `{BALANCED_34B_DATASET_SOURCE}` / `{BALANCED_34B_CANDIDATE_REPRESENTATION}`.",
        f"- Locked architecture: `{ARCHITECTURE}`.",
        "- Benchmark changes: `none`.",
        "- Full 10-seed validation: not run.",
        "",
        "## Candidate Pair Compatibility MLP Audit",
        "",
        f"- Uses only model-facing inputs: `{bool(audit.get('uses_only_model_facing_inputs', False))}`",
        f"- Candidate-only diagnostic accuracy: `{float(audit.get('diagnostic_candidate_only_accuracy', 0.0)):.4f}`",
        f"- Candidate-metadata-only diagnostic accuracy: `{float(audit.get('diagnostic_candidate_metadata_only_accuracy', 0.0)):.4f}`",
        f"- View-masked+candidates diagnostic accuracy: `{float(audit.get('diagnostic_view_masked_candidates_visible_accuracy', 0.0)):.4f}`",
        "",
        "| field name | source | model-facing? | candidate-only predictive? | allowed? |",
        "|---|---|---:|---:|---:|",
    ]
    for row in audit.get("field_table", []):
        lines.append(
            "| {field} | {source} | `{mf}` | `{pred}` | `{allowed}` |".format(
                field=row.get("field_name"),
                source=row.get("source"),
                mf=bool(row.get("model_facing")),
                pred=bool(row.get("candidate_only_predictive")),
                allowed=bool(row.get("allowed")),
            )
        )
    lines.extend(
        [
            "",
            "## Leakage Gates",
            "",
            f"- Candidate-only: `{float(diagnostics.get('mean_candidate_only_accuracy', 0.0)):.4f}`",
            f"- Candidate-metadata-only: `{float(diagnostics.get('mean_candidate_metadata_only_accuracy', 0.0)):.4f}`",
            f"- View-masked + candidates-visible: `{float(diagnostics.get('mean_view_masked_candidates_visible_accuracy', 0.0)):.4f}`",
            f"- Lexical overlap: `{float(diagnostics.get('mean_lexical_overlap_accuracy', 0.0)):.4f}`",
            f"- Static frequency: `{float(diagnostics.get('mean_static_frequency_accuracy', 0.0)):.4f}`",
            f"- All-role oracle: `{float(diagnostics.get('mean_all_role_structured_oracle_accuracy', 0.0)):.4f}`",
            "",
            "## 64-Example Overfit Tests",
            "",
            "| method | train | dev | test | max train | epoch >=0.50 | epoch >=0.80 | epoch >=0.95 | grad norm | param delta |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in overfit.items():
        acc = row.get("accuracy", {})
        thresholds = row.get("epochs_to_train_accuracy", {})
        lines.append(
            "| {name} | {train:.4f} | {dev:.4f} | {test:.4f} | {max_train:.4f} | {e50} | {e80} | {e95} | {grad} | {delta} |".format(
                name=name,
                train=float(acc.get("train", 0.0)),
                dev=float(acc.get("dev", 0.0)),
                test=float(acc.get("test", 0.0)),
                max_train=float(row.get("max_train_accuracy", 0.0)),
                e50=_display_epoch(thresholds.get("0.50")),
                e80=_display_epoch(thresholds.get("0.80")),
                e95=_display_epoch(thresholds.get("0.95")),
                grad=_display_float(row.get("gradient_norm_mean")),
                delta=_display_float(row.get("parameter_delta")),
            )
        )
    lines.extend(
        [
            "",
            "## Message And Readout Diagnostics",
            "",
            f"- Probe mean train accuracy: `{float(probes.get('mean_train_accuracy', 0.0)):.4f}`",
            f"- Probe mean dev accuracy: `{float(probes.get('mean_dev_accuracy', 0.0)):.4f}`",
            f"- Probe mean test accuracy: `{float(probes.get('mean_test_accuracy', 0.0)):.4f}`",
            f"- Active/readout message variance: `{float(result.get('active_readout_diagnostics', {}).get('message_variance_mean', 0.0)):.6f}`",
            f"- Active/readout mean pairwise cosine: `{float(result.get('active_readout_diagnostics', {}).get('mean_pairwise_cosine', 0.0)):.4f}`",
            "",
            "## Bottleneck Ablations",
            "",
        ]
    )
    if ablations:
        lines.extend(
            [
                f"- Best ablation train accuracy: `{_best_ablation_train_accuracy(ablations):.4f}`",
                f"- Inferred bottleneck: `{summary.get('inferred_bottleneck')}`",
            ]
        )
    else:
        lines.extend(
            [
                "- Bottleneck ablations were not run because the trainable locked latent model reached the 64-example overfit target.",
                f"- Inferred bottleneck: `{summary.get('inferred_bottleneck')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Tiny Smoke",
            "",
        ]
    )
    if isinstance(smoke, dict):
        smoke_summary = smoke.get("summary", {})
        smoke_acc = smoke_summary.get("mean_test_accuracy", {}) if isinstance(smoke_summary, dict) else {}
        lines.extend(
            [
                f"- Smoke passed: `{bool(smoke_summary.get('smoke_passed', False))}`",
                f"- Trainable: `{float(smoke_acc.get('trainable', 0.0)):.4f}`",
                f"- Frozen: `{float(smoke_acc.get('frozen', 0.0)):.4f}`",
                f"- Text-only: `{float(smoke_acc.get('text_only', 0.0)):.4f}`",
                f"- Raw latent: `{float(smoke_acc.get('raw_latent', 0.0)):.4f}`",
                f"- Candidate-only: `{float(smoke_acc.get('candidate_only', 0.0)):.4f}`",
                f"- View-masked: `{float(smoke_acc.get('view_masked', 0.0)):.4f}`",
                f"- Role-label-shuffled: `{float(smoke_acc.get('role_labels_shuffled', 0.0)):.4f}`",
                f"- Oracle: `{float(smoke_acc.get('oracle', 0.0)):.4f}`",
            ]
        )
    else:
        lines.append("- Tiny smoke was not run because the locked latent model did not meet the overfit gate.")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Candidate pair compatibility MLP truly model-facing: `{bool(summary.get('candidate_pair_compatibility_mlp_model_facing', False))}`",
            f"- Locked latent can overfit 64 examples: `{bool(summary.get('locked_latent_overfits_64', False))}`",
            f"- Bottleneck ablations ran: `{bool(summary.get('bottleneck_ablations_ran', False))}`",
            f"- Any tiny ablation fixes overfitting: `{bool(summary.get('ablation_fixes_overfitting', False))}`",
            f"- Tiny smoke ran: `{bool(summary.get('tiny_smoke_ran', False))}`",
            f"- Full validation justified: `{bool(summary.get('full_validation_justified', False))}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _display_epoch(value: object) -> str:
    return "" if value is None else str(int(value))


def _display_float(value: object) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
