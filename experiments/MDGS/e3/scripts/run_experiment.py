"""Trace-driven staged training and evaluation for DIGIT Extrapolation."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torch.optim import AdamW
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
VARIANT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(VARIANT_DIR))

from eval_common import (
    apply_trace_perturbation,
    audit_monotonicity,
    build_prediction_records,
    compute_label_audit,
    compute_success_safety_metrics,
    compute_threshold_occupancy,
    compute_threshold_sensitivity,
    evaluate_correctness_baselines,
    summarize_scalar_series,
    write_json,
    write_jsonl,
)
from extrapolation.config import Config
from extrapolation.data.ground_truth import (
    ATTENTION_PATTERN_LABELS,
    CONFIDENCE_LABELS,
    OUTCOME_LABELS,
    TRAJECTORY_SHAPE_LABELS,
)
from extrapolation.data.trace_dataset import TraceEntropyDataset, load_trace_records, trace_collate_fn
from extrapolation.data.vocabulary import Vocabulary
from extrapolation.losses import DIGITLoss
from extrapolation.models.digit import DIGITModel
from extrapolation.trace_pipeline import (
    TRACE_LABEL_VERSION,
    ensure_trace_corpus,
    evaluate_trace_baselines,
    set_global_determinism,
)


def _move_batch(batch, device: torch.device):
    queries, trace_inputs, evidence_targets, prim_targets, target_ids = batch
    return (
        queries.to(device),
        trace_inputs.to(device),
        evidence_targets.to(device),
        prim_targets.to(device),
        target_ids.to(device),
    )


def _one_hot_primitives(prim_targets: torch.Tensor, config: Config) -> tuple[torch.Tensor, ...]:
    return (
        F.one_hot(prim_targets[:, 0], config.num_trajectory_classes).float(),
        F.one_hot(prim_targets[:, 1], config.num_pattern_classes).float(),
        F.one_hot(prim_targets[:, 2], config.num_confidence_classes).float(),
        F.one_hot(prim_targets[:, 3], config.num_outcome_classes).float(),
    )


def _decoder_logits_from_ground_truth(
    model: DIGITModel,
    queries: torch.Tensor,
    prim_targets: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    trajectory_disc, pattern_disc, confidence_disc, outcome_disc = _one_hot_primitives(
        prim_targets,
        model.config,
    )
    z_q = model.encoder(queries)
    return model.decoder(
        z_q=z_q,
        trajectory_shape_disc=trajectory_disc.to(queries.device),
        attention_pattern_disc=pattern_disc.to(queries.device),
        confidence_disc=confidence_disc.to(queries.device),
        outcome_disc=outcome_disc.to(queries.device),
        target_ids=target_ids[:, :-1],
    )


def _generation_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    pad_idx: int,
) -> torch.Tensor:
    targets = target_ids[:, 1:]
    batch_size, target_len, vocab_size = logits.shape
    return F.cross_entropy(
        logits.reshape(batch_size * target_len, vocab_size),
        targets.reshape(batch_size * target_len),
        ignore_index=pad_idx,
        reduction="mean",
    )


def _compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float | list[list[int]]]:
    metrics: dict[str, float | list[list[int]]] = {}
    head_names = ["trajectory", "pattern", "confidence", "outcome"]
    head_label_sizes = [
        len(TRAJECTORY_SHAPE_LABELS),
        len(ATTENTION_PATTERN_LABELS),
        len(CONFIDENCE_LABELS),
        len(OUTCOME_LABELS),
    ]
    for idx, head_name in enumerate(head_names):
        head_pred = predictions[:, idx]
        head_target = targets[:, idx]
        metrics[f"{head_name}_acc"] = (head_pred == head_target).float().mean().item()
        metrics[f"{head_name}_macro_f1"] = float(
            f1_score(head_target.numpy(), head_pred.numpy(), average="macro", zero_division=0)
        )
        metrics[f"{head_name}_confusion"] = confusion_matrix(
            head_target.numpy(),
            head_pred.numpy(),
            labels=list(range(head_label_sizes[idx])),
        ).tolist()

    metrics["joint_acc"] = (predictions == targets).all(dim=1).float().mean().item()
    outcome_precision, outcome_recall, outcome_f1, _ = precision_recall_fscore_support(
        targets[:, 3].numpy(),
        predictions[:, 3].numpy(),
        labels=list(range(len(OUTCOME_LABELS))),
        zero_division=0,
    )
    for idx, label in enumerate(OUTCOME_LABELS):
        label_key = label.lower()
        metrics[f"{label_key}_precision"] = float(outcome_precision[idx])
        metrics[f"{label_key}_recall"] = float(outcome_recall[idx])
        metrics[f"{label_key}_f1"] = float(outcome_f1[idx])

    metrics["failure_recall"] = float(
        outcome_recall[OUTCOME_LABELS.index("FAILURE_LIKELY")]
    )

    pattern_precision, pattern_recall, pattern_f1, _ = precision_recall_fscore_support(
        targets[:, 1].numpy(),
        predictions[:, 1].numpy(),
        labels=list(range(len(ATTENTION_PATTERN_LABELS))),
        zero_division=0,
    )
    for idx, label in enumerate(ATTENTION_PATTERN_LABELS):
        label_key = label.lower()
        metrics[f"pattern_{label_key}_precision"] = float(pattern_precision[idx])
        metrics[f"pattern_{label_key}_recall"] = float(pattern_recall[idx])
        metrics[f"pattern_{label_key}_f1"] = float(pattern_f1[idx])
    return metrics


def _evaluate_stage1(
    model: DIGITModel,
    criterion: DIGITLoss,
    loader: DataLoader,
    device: torch.device,
    records: list[dict] | None = None,
    collect_bundle: bool = False,
) -> dict[str, float | list[list[int]]] | tuple[dict[str, float | list[list[int]]], dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    all_predictions = []
    all_targets = []
    all_logits: dict[str, list[torch.Tensor]] = {
        "trajectory_shape": [],
        "attention_pattern": [],
        "confidence": [],
        "outcome": [],
    }
    all_probs: dict[str, list[torch.Tensor]] = {
        "trajectory_shape": [],
        "attention_pattern": [],
        "confidence": [],
        "outcome": [],
    }
    all_evidence: dict[str, list[torch.Tensor]] = {
        "stability": [],
        "support": [],
        "success_guard": [],
        "success_core": [],
        "success_residual_weight": [],
        "success_residual": [],
        "success_base": [],
        "success_trace_strength": [],
        "success_bonus_weight": [],
        "success_bonus_weight_zero_raw": [],
        "success_bonus": [],
        "success_bonus_zero_raw": [],
        "success_proposal": [],
        "success": [],
        "failure": [],
        "ambiguity": [],
    }
    all_evidence_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            queries, trace_inputs, evidence_targets, prim_targets, target_ids = _move_batch(batch, device)
            out = model(
                queries,
                trace_inputs,
                target_ids,
                bottleneck_mode="hard",
                tau=1.0,
                skip_decoder=True,
            )
            losses = criterion(
                out["decoder_logits"],
                out["primitives"],
                target_ids,
                prim_targets,
                out["executor_features"],
                evidence_targets=evidence_targets,
            )

            batch_size = queries.size(0)
            total_loss += losses["total"].item() * batch_size
            total_items += batch_size

            predictions = torch.stack(
                [
                    out["primitives"].trajectory_shape_logits.argmax(dim=-1),
                    out["primitives"].attention_pattern_logits.argmax(dim=-1),
                    out["primitives"].confidence_logits.argmax(dim=-1),
                    out["primitives"].outcome_logits.argmax(dim=-1),
                ],
                dim=1,
            )
            all_predictions.append(predictions.cpu())
            all_targets.append(prim_targets.cpu())
            all_evidence_targets.append(evidence_targets.cpu())
            all_logits["trajectory_shape"].append(out["primitives"].trajectory_shape_logits.cpu())
            all_logits["attention_pattern"].append(out["primitives"].attention_pattern_logits.cpu())
            all_logits["confidence"].append(out["primitives"].confidence_logits.cpu())
            all_logits["outcome"].append(out["primitives"].outcome_logits.cpu())
            all_probs["trajectory_shape"].append(F.softmax(out["primitives"].trajectory_shape_logits, dim=-1).cpu())
            all_probs["attention_pattern"].append(F.softmax(out["primitives"].attention_pattern_logits, dim=-1).cpu())
            all_probs["confidence"].append(F.softmax(out["primitives"].confidence_logits, dim=-1).cpu())
            all_probs["outcome"].append(F.softmax(out["primitives"].outcome_logits, dim=-1).cpu())
            if hasattr(out["primitives"], "stability_cert"):
                all_evidence["stability"].append(out["primitives"].stability_cert.cpu())
                all_evidence["support"].append(out["primitives"].support_cert.cpu())
                all_evidence["success_guard"].append(out["primitives"].success_guard.cpu())
                all_evidence["success_core"].append(out["primitives"].success_core.cpu())
                all_evidence["success_residual_weight"].append(out["primitives"].success_residual_weight.cpu())
                all_evidence["success_residual"].append(out["primitives"].success_residual.cpu())
                all_evidence["success_base"].append(out["primitives"].success_base.cpu())
                all_evidence["success_trace_strength"].append(out["primitives"].success_trace_strength.cpu())
                all_evidence["success_bonus_weight"].append(out["primitives"].success_bonus_weight.cpu())
                all_evidence["success_bonus_weight_zero_raw"].append(
                    out["primitives"].success_bonus_weight_zero_raw.cpu()
                )
                all_evidence["success_bonus"].append(out["primitives"].success_bonus.cpu())
                all_evidence["success_bonus_zero_raw"].append(out["primitives"].success_bonus_zero_raw.cpu())
                all_evidence["success_proposal"].append(out["primitives"].success_proposal.cpu())
                all_evidence["success"].append(out["primitives"].success_cert.cpu())
                all_evidence["failure"].append(out["primitives"].failure_cert.cpu())
                all_evidence["ambiguity"].append(out["primitives"].ambiguity_cert.cpu())

    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = _compute_metrics(predictions, targets)
    if records is not None:
        if len(records) != predictions.size(0):
            raise ValueError(f"Expected {predictions.size(0)} records, got {len(records)}")
        outcome_probs = torch.cat(all_probs["outcome"], dim=0)
        is_correct = [int(bool(record.get("is_correct", False))) for record in records]
        metrics.update(
            compute_success_safety_metrics(
                outcome_probabilities=outcome_probs.numpy(),
                outcome_predictions=predictions[:, 3].numpy(),
                outcome_targets=targets[:, 3].numpy(),
                is_correct=is_correct,
            )
        )
    metrics["loss"] = total_loss / max(total_items, 1)
    if not collect_bundle:
        return metrics
    return metrics, {
        "predictions": predictions,
        "targets": targets,
        "logits": {key: torch.cat(values, dim=0) for key, values in all_logits.items()},
        "probs": {key: torch.cat(values, dim=0) for key, values in all_probs.items()},
        "evidence_targets": torch.cat(all_evidence_targets, dim=0),
        "evidence": {
            key: torch.cat(values, dim=0)
            for key, values in all_evidence.items()
            if values
        },
    }


def _evaluate_stage2(
    model: DIGITModel,
    loader: DataLoader,
    device: torch.device,
    pad_idx: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for batch in loader:
            queries, _, _, prim_targets, target_ids = _move_batch(batch, device)
            logits = _decoder_logits_from_ground_truth(model, queries, prim_targets, target_ids)
            loss = _generation_loss(logits, target_ids, pad_idx=pad_idx)
            batch_size = queries.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size

    return total_loss / max(total_items, 1)


def _metric_value(metrics: dict[str, float | list[list[int]]], key: str) -> float:
    value = metrics.get(key)
    if isinstance(value, list):
        raise TypeError(f"Metric '{key}' is not scalar")
    if value is None:
        raise KeyError(f"Metric '{key}' not found")
    return float(value)


PATH_SCALE_CONFIG_FIELDS = {
    "shared_trace": "trace_path_scale_shared_trace",
    "success_trace": "trace_path_scale_success_trace",
    "success_core": "trace_path_scale_success_core",
    "success_residual": "trace_path_scale_success_residual",
    "success_base": "trace_path_scale_success_base",
    "success_bonus": "trace_path_scale_success_bonus",
    "failure_trace": "trace_path_scale_failure_trace",
    "ambiguity_trace": "trace_path_scale_ambiguity_trace",
    "cert_context": "trace_path_scale_cert_context",
    "outcome_context": "trace_path_scale_outcome_context",
    "outcome_trajectory_context": "trace_path_scale_outcome_trajectory_context",
    "outcome_pattern_context": "trace_path_scale_outcome_pattern_context",
    "success_raw": "trace_path_scale_success_raw",
    "failure_raw": "trace_path_scale_failure_raw",
    "ambiguity_raw": "trace_path_scale_ambiguity_raw",
}


CONFIG_OVERRIDE_FIELDS = {
    "outcome_label_smoothing": "primitive_outcome_label_smoothing",
    "success_prob_guard_penalty_multiplier": "trace_success_prob_guard_penalty_multiplier",
    "success_prob_guard_margin": "trace_success_prob_guard_margin",
}


def _parse_path_scale_overrides(values: list[str] | None) -> dict[str, float]:
    overrides: dict[str, float] = {}
    if not values:
        return overrides
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --path-scale value {item!r}; expected NAME=VALUE")
        name, raw_value = item.split("=", 1)
        name = name.strip()
        if name not in PATH_SCALE_CONFIG_FIELDS:
            raise ValueError(
                f"Unknown path scale {name!r}; expected one of {sorted(PATH_SCALE_CONFIG_FIELDS)}"
            )
        overrides[name] = float(raw_value)
    return overrides


def _apply_path_scale_overrides(config: Config, overrides: dict[str, float]) -> None:
    for key, value in overrides.items():
        setattr(config, PATH_SCALE_CONFIG_FIELDS[key], float(value))
    if "outcome_context" in overrides:
        outcome_value = float(overrides["outcome_context"])
        config.trace_path_scale_outcome_trajectory_context = outcome_value
        config.trace_path_scale_outcome_pattern_context = outcome_value


def _parse_config_overrides(values: list[str] | None) -> dict[str, float]:
    overrides: dict[str, float] = {}
    if not values:
        return overrides
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --config-override value {item!r}; expected NAME=VALUE")
        name, raw_value = item.split("=", 1)
        name = name.strip()
        if name not in CONFIG_OVERRIDE_FIELDS:
            raise ValueError(
                f"Unknown config override {name!r}; expected one of {sorted(CONFIG_OVERRIDE_FIELDS)}"
            )
        overrides[name] = float(raw_value)
    return overrides


def _apply_config_overrides(config: Config, overrides: dict[str, float]) -> None:
    for key, value in overrides.items():
        setattr(config, CONFIG_OVERRIDE_FIELDS[key], float(value))


def _prediction_audit_from_bundle(records: list[dict[str, Any]], bundle: dict[str, Any]) -> dict[str, Any]:
    predictions = bundle["predictions"].cpu()
    outcome_predictions = predictions[:, 3].tolist()
    confidence_predictions = predictions[:, 2].tolist()
    if len(records) != len(outcome_predictions):
        raise ValueError(f"Expected {len(records)} predictions, got {len(outcome_predictions)}")
    predicted_records = [
        {**record, "outcome": int(predicted_outcome)}
        for record, predicted_outcome in zip(records, outcome_predictions)
    ]
    audit = compute_label_audit(predicted_records)
    total = max(len(records), 1)
    confidence_distribution = {
        label: int(sum(int(pred == idx) for pred in confidence_predictions))
        for idx, label in enumerate(CONFIDENCE_LABELS)
    }
    audit["confidence_distribution"] = confidence_distribution
    audit["confidence_share"] = {
        label: count / total
        for label, count in confidence_distribution.items()
    }
    evidence = bundle.get("evidence", {})
    if "success" in evidence and "failure" in evidence:
        margins = (evidence["success"] - evidence["failure"]).cpu().tolist()
        audit["certificate_margin"] = summarize_scalar_series(margins)
    probs = bundle.get("probs", {})
    if "outcome" in probs and "success_guard" in evidence:
        outcome_probs = probs["outcome"].cpu()
        success_guard = evidence["success_guard"].cpu()
        success_prob = outcome_probs[:, 0]
        success_excess = torch.relu(success_prob - success_guard - 0.05)
        over_guard = success_excess > 0.0
        correctness = torch.tensor(
            [bool(record.get("is_correct", False)) for record in records],
            dtype=torch.bool,
        )
        by_label: dict[str, float] = {}
        excess_by_label: dict[str, float] = {}
        mean_prob_by_label: dict[str, float] = {}
        mean_guard_by_label: dict[str, float] = {}
        for idx, label in enumerate(OUTCOME_LABELS):
            mask = predictions[:, 3].cpu() == idx
            if int(mask.sum().item()) == 0:
                by_label[label] = 0.0
                excess_by_label[label] = 0.0
                mean_prob_by_label[label] = 0.0
                mean_guard_by_label[label] = 0.0
            else:
                by_label[label] = float(over_guard[mask].float().mean().item())
                excess_by_label[label] = float(success_excess[mask].mean().item())
                mean_prob_by_label[label] = float(success_prob[mask].mean().item())
                mean_guard_by_label[label] = float(success_guard[mask].mean().item())
        audit["success_prob_over_guard_rate_by_predicted_outcome"] = by_label
        audit["success_prob_over_guard_excess_by_predicted_outcome"] = excess_by_label
        audit["success_prob_mean_by_predicted_outcome"] = mean_prob_by_label
        audit["success_guard_mean_by_predicted_outcome"] = mean_guard_by_label
        predicted_success = predictions[:, 3].cpu() == 0
        correctness_groups = {
            "correct": predicted_success & correctness,
            "incorrect": predicted_success & ~correctness,
        }
        audit["predicted_success_over_guard_rate_by_correctness"] = {
            name: (
                float(over_guard[mask].float().mean().item())
                if int(mask.sum().item()) > 0
                else 0.0
            )
            for name, mask in correctness_groups.items()
        }
        audit["predicted_success_over_guard_excess_by_correctness"] = {
            name: (
                float(success_excess[mask].mean().item())
                if int(mask.sum().item()) > 0
                else 0.0
            )
            for name, mask in correctness_groups.items()
        }
        audit["predicted_success_count_by_correctness"] = {
            name: int(mask.sum().item()) for name, mask in correctness_groups.items()
        }
    return audit


def _collapse_audit_from_bundle(bundle: dict[str, Any], collapse_share_threshold: float) -> dict[str, Any]:
    predictions = bundle["predictions"].cpu()
    outcome_counts = Counter(int(value) for value in predictions[:, 3].tolist())
    confidence_counts = Counter(int(value) for value in predictions[:, 2].tolist())
    total = max(int(predictions.size(0)), 1)

    outcome_share = {
        label: outcome_counts.get(idx, 0) / total
        for idx, label in enumerate(OUTCOME_LABELS)
    }
    confidence_share = {
        label: confidence_counts.get(idx, 0) / total
        for idx, label in enumerate(CONFIDENCE_LABELS)
    }
    outcome_predicted_classes = sum(int(outcome_counts.get(idx, 0) > 0) for idx in range(len(OUTCOME_LABELS)))
    confidence_predicted_classes = sum(int(confidence_counts.get(idx, 0) > 0) for idx in range(len(CONFIDENCE_LABELS)))
    outcome_max_share = max(outcome_share.values(), default=0.0)
    confidence_max_share = max(confidence_share.values(), default=0.0)

    return {
        "outcome_share": outcome_share,
        "confidence_share": confidence_share,
        "outcome_predicted_classes": int(outcome_predicted_classes),
        "confidence_predicted_classes": int(confidence_predicted_classes),
        "outcome_max_share": float(outcome_max_share),
        "confidence_max_share": float(confidence_max_share),
        "all_success_mode": bool(outcome_share["SUCCESS_LIKELY"] >= collapse_share_threshold),
        "all_uncertain_mode": bool(outcome_share["UNCERTAIN"] >= collapse_share_threshold),
        "all_failure_mode": bool(outcome_share["FAILURE_LIKELY"] >= collapse_share_threshold),
        "single_confidence_mode": bool(confidence_max_share >= collapse_share_threshold or confidence_predicted_classes < 2),
        "collapsed_model": bool(
            outcome_max_share >= collapse_share_threshold
            or confidence_max_share >= collapse_share_threshold
            or outcome_predicted_classes < 2
            or confidence_predicted_classes < 2
        ),
    }


def _safe_corrcoef(lhs: list[float], rhs: list[float]) -> float:
    if len(lhs) != len(rhs) or len(lhs) < 2:
        return float("nan")
    lhs_arr = torch.tensor(lhs, dtype=torch.float32)
    rhs_arr = torch.tensor(rhs, dtype=torch.float32)
    lhs_std = float(lhs_arr.std(unbiased=False).item())
    rhs_std = float(rhs_arr.std(unbiased=False).item())
    if lhs_std < 1e-8 or rhs_std < 1e-8:
        return float("nan")
    return float(torch.corrcoef(torch.stack([lhs_arr, rhs_arr], dim=0))[0, 1].item())


def _success_guard_target_from_evidence_targets(evidence_targets: torch.Tensor) -> torch.Tensor:
    stability_target = evidence_targets[:, 0]
    support_target = evidence_targets[:, 1]
    success_target = evidence_targets[:, 2]
    guard_anchor = torch.clamp(0.65 * stability_target + 0.35 * support_target, min=0.0, max=1.0)
    return torch.minimum(success_target, guard_anchor)


def _evidence_audit_from_bundle(records: list[dict[str, Any]], bundle: dict[str, Any]) -> dict[str, Any]:
    evidence = bundle.get("evidence", {})
    if not evidence:
        return {}

    payload: dict[str, Any] = {
        name: summarize_scalar_series(values.cpu().tolist())
        for name, values in evidence.items()
    }
    confidence_probs = bundle["probs"]["confidence"][:, 2].cpu().tolist()
    agreements = [float(record.get("agreement", 0.0)) for record in records]
    support_ratios = [float(record.get("stats", {}).get("support_ratio", 0.0)) for record in records]
    payload["correlations"] = {
        "success_vs_failure": _safe_corrcoef(
            evidence["success"].cpu().tolist(),
            evidence["failure"].cpu().tolist(),
        ),
        "ambiguity_vs_high_confidence_prob": _safe_corrcoef(
            evidence["ambiguity"].cpu().tolist(),
            confidence_probs,
        ),
        "stability_vs_agreement": _safe_corrcoef(
            evidence["stability"].cpu().tolist(),
            agreements,
        ),
        "support_vs_support_ratio": _safe_corrcoef(
            evidence["support"].cpu().tolist(),
            support_ratios,
        ),
    }
    if "success_guard" in evidence:
        success_values = evidence["success"].cpu()
        success_guard_values = evidence["success_guard"].cpu()
        success_probs = bundle["probs"]["outcome"][:, 0].cpu()
        success_core_values = evidence.get("success_core", evidence.get("success_base", evidence["success"])).cpu()
        success_residual_weight_values = evidence.get(
            "success_residual_weight",
            torch.zeros_like(success_values),
        ).cpu()
        success_residual_values = evidence.get("success_residual", torch.zeros_like(success_values)).cpu()
        success_base_values = evidence.get("success_base", evidence["success"]).cpu()
        success_trace_strength_values = evidence.get(
            "success_trace_strength",
            torch.zeros_like(success_values),
        ).cpu()
        success_bonus_weight_values = evidence.get("success_bonus_weight", torch.zeros_like(success_values)).cpu()
        success_bonus_weight_zero_raw_values = evidence.get(
            "success_bonus_weight_zero_raw",
            torch.zeros_like(success_values),
        ).cpu()
        success_bonus_values = evidence.get("success_bonus", torch.zeros_like(success_values)).cpu()
        success_bonus_zero_raw_values = evidence.get(
            "success_bonus_zero_raw",
            torch.zeros_like(success_values),
        ).cpu()
        success_proposal_values = evidence.get("success_proposal", evidence["success"]).cpu()
        stability_values = evidence["stability"].cpu()
        support_values = evidence["support"].cpu()
        trajectory_probs = bundle["probs"]["trajectory_shape"].cpu()
        pattern_probs = bundle["probs"]["attention_pattern"].cpu()
        trajectory_good_values = torch.maximum(trajectory_probs[:, 0], trajectory_probs[:, 1])
        pattern_focused_values = pattern_probs[:, 0]
        proposal_gain_values = success_proposal_values - success_base_values
        base_gain_values = success_base_values - success_core_values
        bonus_weight_delta_values = success_bonus_weight_values - success_bonus_weight_zero_raw_values
        bonus_delta_values = success_bonus_values - success_bonus_zero_raw_values
        payload["correlations"]["success_guard_vs_success"] = _safe_corrcoef(
            success_guard_values.tolist(),
            success_values.tolist(),
        )
        payload["correlations"]["success_guard_vs_success_base"] = _safe_corrcoef(
            success_guard_values.tolist(),
            success_base_values.tolist(),
        )
        payload["correlations"]["success_guard_vs_success_proposal"] = _safe_corrcoef(
            success_guard_values.tolist(),
            success_proposal_values.tolist(),
        )
        payload["correlations"]["success_base_vs_success"] = _safe_corrcoef(
            success_base_values.tolist(),
            success_values.tolist(),
        )
        payload["correlations"]["success_core_vs_success"] = _safe_corrcoef(
            success_core_values.tolist(),
            success_values.tolist(),
        )
        payload["correlations"]["success_core_vs_success_guard"] = _safe_corrcoef(
            success_core_values.tolist(),
            success_guard_values.tolist(),
        )
        payload["correlations"]["success_base_vs_stability"] = _safe_corrcoef(
            success_base_values.tolist(),
            stability_values.tolist(),
        )
        payload["correlations"]["success_base_vs_support"] = _safe_corrcoef(
            success_base_values.tolist(),
            support_values.tolist(),
        )
        payload["correlations"]["success_base_vs_trajectory_good"] = _safe_corrcoef(
            success_base_values.tolist(),
            trajectory_good_values.tolist(),
        )
        payload["correlations"]["success_base_vs_pattern_focused"] = _safe_corrcoef(
            success_base_values.tolist(),
            pattern_focused_values.tolist(),
        )
        payload["correlations"]["success_base_vs_success_trace_strength"] = _safe_corrcoef(
            success_base_values.tolist(),
            success_trace_strength_values.tolist(),
        )
        payload["correlations"]["success_core_vs_stability"] = _safe_corrcoef(
            success_core_values.tolist(),
            stability_values.tolist(),
        )
        payload["correlations"]["success_core_vs_trajectory_good"] = _safe_corrcoef(
            success_core_values.tolist(),
            trajectory_good_values.tolist(),
        )
        payload["correlations"]["success_core_vs_pattern_focused"] = _safe_corrcoef(
            success_core_values.tolist(),
            pattern_focused_values.tolist(),
        )
        payload["correlations"]["success_residual_vs_trajectory_good"] = _safe_corrcoef(
            success_residual_values.tolist(),
            trajectory_good_values.tolist(),
        )
        payload["correlations"]["success_residual_vs_pattern_focused"] = _safe_corrcoef(
            success_residual_values.tolist(),
            pattern_focused_values.tolist(),
        )
        payload["correlations"]["success_residual_vs_success_trace_strength"] = _safe_corrcoef(
            success_residual_values.tolist(),
            success_trace_strength_values.tolist(),
        )
        payload["correlations"]["success_bonus_weight_vs_success"] = _safe_corrcoef(
            success_bonus_weight_values.tolist(),
            success_values.tolist(),
        )
        payload["correlations"]["success_bonus_weight_vs_zero_raw"] = _safe_corrcoef(
            success_bonus_weight_values.tolist(),
            success_bonus_weight_zero_raw_values.tolist(),
        )
        payload["correlations"]["success_bonus_weight_vs_success_bonus"] = _safe_corrcoef(
            success_bonus_weight_values.tolist(),
            success_bonus_values.tolist(),
        )
        payload["correlations"]["success_bonus_vs_zero_raw_bonus"] = _safe_corrcoef(
            success_bonus_values.tolist(),
            success_bonus_zero_raw_values.tolist(),
        )
        payload["correlations"]["success_proposal_vs_success"] = _safe_corrcoef(
            success_proposal_values.tolist(),
            success_values.tolist(),
        )
        payload["success_guard_alignment"] = {
            "guard_utilization_mean": float(
                (success_values / success_guard_values.clamp_min(1e-6)).mean().item()
            ),
            "proposal_utilization_mean": float(
                (success_proposal_values / success_guard_values.clamp_min(1e-6)).mean().item()
            ),
            "core_utilization_mean": float(
                (success_core_values / success_guard_values.clamp_min(1e-6)).mean().item()
            ),
            "base_utilization_mean": float(
                (success_base_values / success_guard_values.clamp_min(1e-6)).mean().item()
            ),
            "success_core_mean": float(success_core_values.mean().item()),
            "success_residual_mean": float(success_residual_values.mean().item()),
            "success_residual_weight_mean": float(success_residual_weight_values.mean().item()),
            "success_residual_weight_std": float(success_residual_weight_values.std(unbiased=False).item()),
            "bonus_weight_mean": float(success_bonus_weight_values.mean().item()),
            "bonus_weight_std": float(success_bonus_weight_values.std(unbiased=False).item()),
            "bonus_weight_zero_raw_mean": float(success_bonus_weight_zero_raw_values.mean().item()),
            "bonus_weight_zero_raw_std": float(success_bonus_weight_zero_raw_values.std(unbiased=False).item()),
            "bonus_weight_evidence_delta_mean": float(bonus_weight_delta_values.mean().item()),
            "bonus_weight_zero_raw_fraction_mean": float(
                (success_bonus_weight_zero_raw_values / success_bonus_weight_values.clamp_min(1e-6)).mean().item()
            ),
            "bonus_mean": float(success_bonus_values.mean().item()),
            "bonus_zero_raw_mean": float(success_bonus_zero_raw_values.mean().item()),
            "bonus_zero_raw_fraction_of_bonus_mean": float(
                (success_bonus_zero_raw_values / success_bonus_values.clamp_min(1e-6)).mean().item()
            ),
            "bonus_evidence_fraction_of_bonus_mean": float(
                (bonus_delta_values / success_bonus_values.clamp_min(1e-6)).mean().item()
            ),
            "bonus_fraction_of_success_mean": float(
                (success_bonus_values / success_values.clamp_min(1e-6)).mean().item()
            ),
            "success_trace_strength_mean": float(success_trace_strength_values.mean().item()),
            "success_trace_strength_std": float(success_trace_strength_values.std(unbiased=False).item()),
            "success_core_fraction_of_success_mean": float(
                (success_core_values / success_values.clamp_min(1e-6)).mean().item()
            ),
            "success_residual_fraction_of_success_mean": float(
                (base_gain_values / success_values.clamp_min(1e-6)).mean().item()
            ),
            "proposal_gain_mean": float(proposal_gain_values.mean().item()),
            "proposal_gain_fraction_of_success_mean": float(
                (proposal_gain_values / success_values.clamp_min(1e-6)).mean().item()
            ),
            "success_over_guard_mean": float(
                torch.relu(success_values - success_guard_values).mean().item()
            ),
            "success_over_guard_rate": float(
                (success_values > (success_guard_values + 0.02)).float().mean().item()
            ),
            "base_over_guard_mean": float(
                torch.relu(success_base_values - success_guard_values).mean().item()
            ),
            "base_over_guard_rate": float(
                (success_base_values > (success_guard_values + 0.02)).float().mean().item()
            ),
            "proposal_over_guard_mean": float(
                torch.relu(success_proposal_values - success_guard_values).mean().item()
            ),
            "proposal_over_guard_rate": float(
                (success_proposal_values > (success_guard_values + 0.02)).float().mean().item()
            ),
            "success_prob_over_guard_mean": float(
                torch.relu(success_probs - success_guard_values).mean().item()
            ),
            "success_prob_over_guard_rate": float(
                (success_probs > (success_guard_values + 0.05)).float().mean().item()
            ),
            "guard_low_saturation_rate": float(
                (success_guard_values < 0.05).float().mean().item()
            ),
            "guard_high_saturation_rate": float(
                (success_guard_values > 0.95).float().mean().item()
            ),
        }
    target_tensor = bundle.get("evidence_targets")
    if isinstance(target_tensor, torch.Tensor) and target_tensor.numel() > 0:
        target_fit: dict[str, Any] = {}
        target_names = ["stability", "support", "success", "failure", "ambiguity"]
        target_matrix = target_tensor.cpu()
        for idx, name in enumerate(target_names):
            pred_values = evidence[name].cpu().tolist()
            target_values = target_matrix[:, idx].tolist()
            pred_summary = summarize_scalar_series(pred_values)
            target_summary = summarize_scalar_series(target_values)
            target_fit[name] = {
                "corr": _safe_corrcoef(pred_values, target_values),
                "mae": float(sum(abs(a - b) for a, b in zip(pred_values, target_values)) / max(len(pred_values), 1)),
                "pred_std": pred_summary["std"],
                "target_std": target_summary["std"],
                "spread_ratio": float(
                    pred_summary["std"] / max(float(target_summary["std"]), 1e-8)
                ),
            }
        payload["target_fit"] = target_fit
        if "success_guard" in evidence:
            success_target_values = target_matrix[:, 2]
            success_guard_target_values = _success_guard_target_from_evidence_targets(target_matrix)
            success_core_values = evidence.get("success_core", evidence.get("success_base", evidence["success"])).cpu()
            success_base_values = evidence.get("success_base", evidence["success"]).cpu()
            payload["correlations"]["success_guard_vs_success_target"] = _safe_corrcoef(
                evidence["success_guard"].cpu().tolist(),
                success_target_values.tolist(),
            )
            payload["correlations"]["success_guard_vs_guard_target"] = _safe_corrcoef(
                evidence["success_guard"].cpu().tolist(),
                success_guard_target_values.tolist(),
            )
            payload["correlations"]["success_base_vs_success_target"] = _safe_corrcoef(
                success_base_values.tolist(),
                success_target_values.tolist(),
            )
            payload["correlations"]["success_core_vs_success_target"] = _safe_corrcoef(
                success_core_values.tolist(),
                success_target_values.tolist(),
            )
            payload["correlations"]["success_base_vs_guard_target"] = _safe_corrcoef(
                success_base_values.tolist(),
                success_guard_target_values.tolist(),
            )
            payload["correlations"]["success_core_vs_guard_target"] = _safe_corrcoef(
                success_core_values.tolist(),
                success_guard_target_values.tolist(),
            )
            payload["success_guard_alignment"].update(
                {
                    "guard_target_mae": float(
                        (evidence["success_guard"].cpu() - success_guard_target_values).abs().mean().item()
                    ),
                    "guard_target_bias_mean": float(
                        (evidence["success_guard"].cpu() - success_guard_target_values).mean().item()
                    ),
                    "guard_target_over_rate": float(
                        (
                            evidence["success_guard"].cpu()
                            > (success_guard_target_values + 0.02)
                        ).float().mean().item()
                    ),
                    "base_target_excess_mean": float(
                        torch.relu(success_base_values - success_guard_target_values).mean().item()
                    ),
                    "base_target_over_rate": float(
                        (
                            success_base_values
                            > (success_guard_target_values + 0.02)
                        ).float().mean().item()
                    ),
                }
            )
    return payload


def _counterfactual_path_audit(
    model: DIGITModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    scenarios = {
        "zero_shared_trace": {"shared_trace": 0.0},
        "zero_success_core": {"success_core": 0.0},
        "zero_success_residual": {"success_residual": 0.0},
        "zero_success_base": {"success_core": 0.0, "success_residual": 0.0},
        "zero_success_bonus": {"success_bonus": 0.0},
        "zero_success_trace": {"success_trace": 0.0},
        "zero_failure_trace": {"failure_trace": 0.0},
        "zero_ambiguity_trace": {"ambiguity_trace": 0.0},
        "zero_success_raw": {"success_raw": 0.0},
        "zero_failure_raw": {"failure_raw": 0.0},
        "zero_ambiguity_raw": {"ambiguity_raw": 0.0},
        "zero_all_raw_cert_features": {"success_raw": 0.0, "failure_raw": 0.0, "ambiguity_raw": 0.0},
        "zero_cert_context": {"cert_context": 0.0},
        "zero_outcome_context": {"outcome_context": 0.0},
        "zero_outcome_trajectory_context": {"outcome_trajectory_context": 0.0},
        "zero_outcome_pattern_context": {"outcome_pattern_context": 0.0},
        "certs_only_outcome": {
            "shared_trace": 0.0,
            "success_raw": 0.0,
            "failure_raw": 0.0,
            "ambiguity_raw": 0.0,
            "outcome_context": 0.0,
        },
    }
    accum: dict[str, dict[str, float]] = {
        name: {
            "count": 0.0,
            "outcome_label_flip_rate": 0.0,
            "confidence_label_flip_rate": 0.0,
            "success_prob_mae": 0.0,
            "failure_prob_mae": 0.0,
            "success_cert_mae": 0.0,
            "failure_cert_mae": 0.0,
            "ambiguity_cert_mae": 0.0,
            "outcome_flip_counts": [0.0 for _ in OUTCOME_LABELS],
            "outcome_base_counts": [0.0 for _ in OUTCOME_LABELS],
        }
        for name in scenarios
    }

    model.eval()
    with torch.no_grad():
        for batch in loader:
            queries, trace_inputs, _, _, target_ids = _move_batch(batch, device)
            base_out = model(
                queries,
                trace_inputs,
                target_ids,
                bottleneck_mode="hard",
                tau=1.0,
                skip_decoder=True,
            )
            base_outcome_probs = F.softmax(base_out["primitives"].outcome_logits, dim=-1)
            base_confidence = base_out["primitives"].confidence_logits.argmax(dim=-1)
            base_outcome = base_out["primitives"].outcome_logits.argmax(dim=-1)
            batch_size = float(queries.size(0))

            for scenario_name, path_scales in scenarios.items():
                cf_out = model(
                    queries,
                    trace_inputs,
                    target_ids,
                    bottleneck_mode="hard",
                    tau=1.0,
                    skip_decoder=True,
                    path_scales=path_scales,
                )
                cf_outcome_probs = F.softmax(cf_out["primitives"].outcome_logits, dim=-1)
                cf_confidence = cf_out["primitives"].confidence_logits.argmax(dim=-1)
                cf_outcome = cf_out["primitives"].outcome_logits.argmax(dim=-1)

                bucket = accum[scenario_name]
                bucket["count"] += batch_size
                bucket["outcome_label_flip_rate"] += float((cf_outcome != base_outcome).float().sum().item())
                bucket["confidence_label_flip_rate"] += float((cf_confidence != base_confidence).float().sum().item())
                bucket["success_prob_mae"] += float((cf_outcome_probs[:, 0] - base_outcome_probs[:, 0]).abs().sum().item())
                bucket["failure_prob_mae"] += float((cf_outcome_probs[:, 2] - base_outcome_probs[:, 2]).abs().sum().item())
                bucket["success_cert_mae"] += float((cf_out["primitives"].success_cert - base_out["primitives"].success_cert).abs().sum().item())
                bucket["failure_cert_mae"] += float((cf_out["primitives"].failure_cert - base_out["primitives"].failure_cert).abs().sum().item())
                bucket["ambiguity_cert_mae"] += float((cf_out["primitives"].ambiguity_cert - base_out["primitives"].ambiguity_cert).abs().sum().item())
                for idx in range(len(OUTCOME_LABELS)):
                    mask = base_outcome == idx
                    bucket["outcome_base_counts"][idx] += float(mask.float().sum().item())
                    bucket["outcome_flip_counts"][idx] += float(((cf_outcome != base_outcome) & mask).float().sum().item())

    output: dict[str, Any] = {}
    for scenario_name, values in accum.items():
        count = max(values["count"], 1.0)
        base_counts = values["outcome_base_counts"]
        flip_counts = values["outcome_flip_counts"]
        output[scenario_name] = {
            key: float(val / count) if key != "count" else int(val)
            for key, val in values.items()
            if key not in {"count", "outcome_flip_counts", "outcome_base_counts"}
        }
        output[scenario_name]["outcome_flip_rate_by_base_label"] = {
            label: (
                float(flip_counts[idx] / base_counts[idx])
                if base_counts[idx] > 0.0
                else 0.0
            )
            for idx, label in enumerate(OUTCOME_LABELS)
        }
    return output


def _path_audit_summary(counterfactual_audit: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not counterfactual_audit:
        return {}
    single_path_items = {
        name: values
        for name, values in counterfactual_audit.items()
        if name not in {"certs_only_outcome", "zero_all_raw_cert_features", "zero_shared_trace", "zero_success_base"}
    }
    cert_driver = max(
        counterfactual_audit.items(),
        key=lambda item: float(item[1].get("success_cert_mae", 0.0)) + float(item[1].get("failure_cert_mae", 0.0)),
    )
    outcome_driver = max(
        counterfactual_audit.items(),
        key=lambda item: float(item[1].get("outcome_label_flip_rate", 0.0)),
    )
    confidence_driver = max(
        counterfactual_audit.items(),
        key=lambda item: float(item[1].get("confidence_label_flip_rate", 0.0)),
    )
    single_path_cert_driver = max(
        single_path_items.items(),
        key=lambda item: float(item[1].get("success_cert_mae", 0.0)) + float(item[1].get("failure_cert_mae", 0.0)),
    )
    single_path_outcome_driver = max(
        single_path_items.items(),
        key=lambda item: float(item[1].get("outcome_label_flip_rate", 0.0)),
    )
    return {
        "largest_cert_shift": {
            "scenario": cert_driver[0],
            "value": float(cert_driver[1].get("success_cert_mae", 0.0)) + float(cert_driver[1].get("failure_cert_mae", 0.0)),
        },
        "largest_outcome_flip": {
            "scenario": outcome_driver[0],
            "value": float(outcome_driver[1].get("outcome_label_flip_rate", 0.0)),
        },
        "largest_confidence_flip": {
            "scenario": confidence_driver[0],
            "value": float(confidence_driver[1].get("confidence_label_flip_rate", 0.0)),
        },
        "largest_single_path_cert_shift": {
            "scenario": single_path_cert_driver[0],
            "value": float(single_path_cert_driver[1].get("success_cert_mae", 0.0))
            + float(single_path_cert_driver[1].get("failure_cert_mae", 0.0)),
        },
        "largest_single_path_outcome_flip": {
            "scenario": single_path_outcome_driver[0],
            "value": float(single_path_outcome_driver[1].get("outcome_label_flip_rate", 0.0)),
        },
    }


def _strictly_beats(lhs: float, rhs: float, tol: float = 1e-6) -> bool:
    return lhs > (rhs + tol)


def _is_better_stage1(
    metrics: dict[str, float | list[list[int]]],
    best_metrics: dict[str, float | list[list[int]]] | None,
    candidate_collapsed: bool,
    best_collapsed: bool,
    tolerance: float = 0.01,
) -> bool:
    if best_metrics is None:
        return True
    if candidate_collapsed != best_collapsed:
        return not candidate_collapsed

    candidate_outcome = _metric_value(metrics, "outcome_macro_f1")
    best_outcome = _metric_value(best_metrics, "outcome_macro_f1")
    if candidate_outcome > best_outcome + tolerance:
        return True
    if abs(candidate_outcome - best_outcome) <= tolerance:
        candidate_failure = _metric_value(metrics, "failure_recall")
        best_failure = _metric_value(best_metrics, "failure_recall")
        if candidate_failure > best_failure + tolerance:
            return True
        if abs(candidate_failure - best_failure) <= tolerance:
            candidate_confidence = _metric_value(metrics, "confidence_macro_f1")
            best_confidence = _metric_value(best_metrics, "confidence_macro_f1")
            if candidate_confidence > best_confidence + tolerance:
                return True
            if abs(candidate_confidence - best_confidence) <= tolerance:
                candidate_trajectory = _metric_value(metrics, "trajectory_macro_f1")
                best_trajectory = _metric_value(best_metrics, "trajectory_macro_f1")
                if candidate_trajectory > best_trajectory + tolerance:
                    return True
                if abs(candidate_trajectory - best_trajectory) <= tolerance:
                    return _metric_value(metrics, "joint_acc") > _metric_value(best_metrics, "joint_acc") + 1e-8
    return False


def _format_confusion(matrix: list[list[int]], labels: list[str]) -> str:
    rows = []
    for label, row in zip(labels, matrix):
        rows.append(f"{label}: {row}")
    return " | ".join(rows)


def _collect_primitive_predictions(
    model: DIGITModel,
    loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in loader:
            queries, trace_inputs, _, _, target_ids = _move_batch(batch, device)
            out = model(
                queries,
                trace_inputs,
                target_ids,
                bottleneck_mode="hard",
                tau=1.0,
                skip_decoder=True,
            )
            outputs.append(
                torch.stack(
                    [
                        out["primitives"].trajectory_shape_logits.argmax(dim=-1),
                        out["primitives"].attention_pattern_logits.argmax(dim=-1),
                        out["primitives"].confidence_logits.argmax(dim=-1),
                        out["primitives"].outcome_logits.argmax(dim=-1),
                    ],
                    dim=1,
                ).cpu()
            )
    return torch.cat(outputs, dim=0)


def _majority_baseline(train_records: list[dict], test_records: list[dict]) -> dict[str, float | int]:
    majority_class = Counter(record["outcome"] for record in train_records).most_common(1)[0][0]
    targets = [record["outcome"] for record in test_records]
    predictions = [majority_class] * len(test_records)
    accuracy = sum(int(pred == target) for pred, target in zip(predictions, targets)) / max(len(targets), 1)
    macro_f1 = float(f1_score(targets, predictions, average="macro", zero_division=0))
    return {
        "class_idx": int(majority_class),
        "accuracy": float(accuracy),
        "macro_f1": macro_f1,
    }


def _freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _unfreeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True


def _reference_metrics_path(trace_dir: Path) -> Path:
    return trace_dir / f"reference_metrics.{VARIANT_DIR.name}.json"


def _default_output_dir(variant_name: str, seed: int) -> Path:
    return ROOT_DIR / "results" / variant_name / f"seed_{seed}"


def _success_mode_summary() -> dict[str, str]:
    return {
        "success_core_mode": "guard_identity",
        "success_residual_mode": "disabled",
    }


def _default_path_scales(config: Config) -> dict[str, float]:
    return {
        "shared_trace": float(config.trace_path_scale_shared_trace),
        "success_trace": float(config.trace_path_scale_success_trace),
        "success_core": float(config.trace_path_scale_success_core),
        "success_residual": float(config.trace_path_scale_success_residual),
        "success_bonus": float(config.trace_path_scale_success_bonus),
        "failure_trace": float(config.trace_path_scale_failure_trace),
        "ambiguity_trace": float(config.trace_path_scale_ambiguity_trace),
        "cert_context": float(config.trace_path_scale_cert_context),
        "outcome_context": float(config.trace_path_scale_outcome_context),
        "outcome_trajectory_context": float(config.trace_path_scale_outcome_trajectory_context),
        "outcome_pattern_context": float(config.trace_path_scale_outcome_pattern_context),
        "success_raw": float(config.trace_path_scale_success_raw),
        "failure_raw": float(config.trace_path_scale_failure_raw),
        "ambiguity_raw": float(config.trace_path_scale_ambiguity_raw),
    }


def _load_reference_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _reference_metrics_need_refresh(reference_metrics: dict | None) -> bool:
    if reference_metrics is None:
        return True
    test_metrics = reference_metrics.get("test_metrics", {})
    required_keys = [
        "pattern_focused_precision",
        "pattern_focused_recall",
        "pattern_focused_f1",
        "pattern_mixed_precision",
        "pattern_mixed_recall",
        "pattern_mixed_f1",
        "pattern_diffuse_precision",
        "pattern_diffuse_recall",
        "pattern_diffuse_f1",
    ]
    return any(key not in test_metrics for key in required_keys)


def _maybe_write_reference_metrics(
    path: Path,
    trace_info: dict,
    test_metrics: dict,
    baseline_results: dict,
) -> bool:
    if trace_info.get("label_version") != TRACE_LABEL_VERSION:
        return False
    if trace_info.get("pattern_threshold_mode") != "auto":
        return False
    existing_reference = _load_reference_metrics(path)
    if existing_reference is not None and not _reference_metrics_need_refresh(existing_reference):
        return False
    payload = {
        "label_version": trace_info.get("label_version"),
        "pattern_z_threshold": trace_info.get("pattern_z_threshold"),
        "pattern_threshold_mode": trace_info.get("pattern_threshold_mode"),
        "corpus_fingerprint": trace_info.get("corpus_fingerprint"),
        "test_metrics": test_metrics,
        "baselines": baseline_results,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-traces", action="store_true")
    parser.add_argument("--pattern-threshold-override", type=float, default=None)
    parser.add_argument("--train-loop-seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-reference-write", action="store_true")
    parser.add_argument("--experiment-label", type=str, default=None)
    parser.add_argument("--path-scale", action="append", default=None, metavar="NAME=VALUE")
    parser.add_argument("--config-override", action="append", default=None, metavar="NAME=VALUE")
    args = parser.parse_args()

    config = Config()
    if args.train_loop_seed is not None:
        config.train_loop_seed = int(args.train_loop_seed)
    config.trace_pattern_threshold_override = args.pattern_threshold_override
    path_scale_overrides = _parse_path_scale_overrides(args.path_scale)
    config_overrides = _parse_config_overrides(args.config_override)
    _apply_path_scale_overrides(config, path_scale_overrides)
    _apply_config_overrides(config, config_overrides)
    variant_name = VARIANT_DIR.name
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(variant_name, config.train_loop_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab = Vocabulary()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    set_global_determinism(config.train_loop_seed)

    trace_dir = Path(__file__).parent.parent / config.trace_dir
    trace_info = ensure_trace_corpus(
        config=config,
        root_dir=trace_dir,
        device=device,
        rebuild=args.rebuild_traces,
        strict_existing=False,
    )

    with open(trace_dir / "metadata.json") as f:
        trace_metadata = json.load(f)

    train_dataset = TraceEntropyDataset.from_jsonl(trace_dir / "train.jsonl", vocab=vocab, config=config)
    val_dataset = TraceEntropyDataset.from_jsonl(trace_dir / "val.jsonl", vocab=vocab, config=config)
    test_dataset = TraceEntropyDataset.from_jsonl(trace_dir / "test.jsonl", vocab=vocab, config=config)

    if config.trace_use_weighted_sampler:
        raise ValueError("trace_use_weighted_sampler=True is not supported in the recovery plan")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=trace_collate_fn,
        generator=torch.Generator().manual_seed(config.train_loop_seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=trace_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=trace_collate_fn,
    )

    model = DIGITModel(config, vocab).to(device)
    model.set_trace_metadata(trace_metadata)
    class_weights = {
        key: value.to(device)
        for key, value in train_dataset.get_class_weights(
            power=config.trace_class_weight_power,
            clip=config.trace_class_weight_clip,
        ).items()
    }
    class_weights["outcome"] = torch.tensor(
        [1.0, 1.0, config.trace_outcome_failure_weight],
        dtype=torch.float32,
        device=device,
    )

    stage1_config = copy.deepcopy(config)
    stage1_config.lambda_primitive = 1.0
    stage1_config.lambda_generation = 0.0
    stage1_config.lambda_policy = 0.0
    stage1_config.lambda_leakage = 0.0
    stage1_config.lambda_abstention = 0.0
    stage1_config.primitive_pattern_loss_weight = 1.5
    stage1_config.primitive_confidence_loss_weight = 1.0
    stage1_criterion = DIGITLoss(
        stage1_config,
        pad_idx=vocab.pad_idx,
        class_weights=class_weights,
    ).to(device)

    warmup_config = copy.deepcopy(stage1_config)
    warmup_config.lambda_evidence_targets = 0.0
    warmup_config.lambda_consistency = 0.0
    warmup_config.lambda_evidence_mono = 0.0
    warmup_config.lambda_usage = 0.0
    warmup_config.primitive_confidence_loss_weight = 0.0
    warmup_config.primitive_outcome_loss_weight = 0.0
    warmup_criterion = DIGITLoss(
        warmup_config,
        pad_idx=vocab.pad_idx,
        class_weights=class_weights,
    ).to(device)
    stage1_optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_stage1_state = copy.deepcopy(model.state_dict())
    best_stage1_metrics: dict[str, float | list[list[int]]] | None = None
    best_stage1_collapse_audit = {"collapsed_model": True}
    patience_counter = 0

    print(f"DIGIT Extrapolation {variant_name.upper()} partial-monotone run")
    print("Success shape: guard_only")
    if args.experiment_label:
        print(f"Experiment label: {args.experiment_label}")
    print(f"Device: {device}")
    print(f"Results dir: {output_dir}")
    print(f"Trace dir: {trace_info['trace_dir']}")
    print(f"Trace label version: {trace_info.get('label_version', 'unknown')}")
    print(f"Evidence target version: {trace_info.get('evidence_target_version', 'unknown')}")
    print(f"Corpus fingerprint: {trace_info.get('corpus_fingerprint')}")
    print(f"Pattern z-threshold: {trace_info.get('pattern_z_threshold')}")
    print(f"Pattern threshold mode: {trace_info.get('pattern_threshold_mode')}")
    print(
        f"Trace records train/val/test: "
        f"{len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}"
    )
    print(
        "Failure shares train/val/test: "
        f"{trace_info.get('train_failure_share', 0.0):.3f}/"
        f"{trace_info.get('val_failure_share', 0.0):.3f}/"
        f"{trace_info.get('test_failure_share', 0.0):.3f}"
    )
    print(f"Train hard-mined cases: {trace_info.get('train_hard_case_count', 0)}")
    print(f"Train label distribution: {train_dataset.get_label_distribution()}")
    print(
        f"Train attention-pattern distribution: "
        f"{trace_metadata.get('attention_pattern_distribution', {})}"
    )
    print(f"Success modes: {_success_mode_summary()}")
    if path_scale_overrides:
        print(f"Path scale overrides: {path_scale_overrides}")
    if config_overrides:
        print(f"Config overrides: {config_overrides}")
    print(f"Default path scales: {_default_path_scales(config)}")
    print(
        "Stage 1: joint primitive training "
        "(selection: non-collapsed -> outcome_macro_f1 -> failure_recall -> confidence_macro_f1 -> trajectory_macro_f1 -> joint_acc)"
    )

    warmup_epochs = min(config.trace_stage1_head_warmup_epochs, config.trace_stage1_epochs)
    perturbation_kinds = [
        "lower_agreement",
        "lower_margin",
        "higher_entropy",
        "lower_attention_concentration",
        "higher_variation_ratio",
    ]
    for epoch in range(config.trace_stage1_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_items = 0
        criterion = warmup_criterion if epoch < warmup_epochs else stage1_criterion

        for batch_idx, batch in enumerate(train_loader):
            queries, trace_inputs, evidence_targets, prim_targets, target_ids = _move_batch(batch, device)
            out = model(
                queries,
                trace_inputs,
                target_ids,
                bottleneck_mode="gumbel",
                tau=1.0,
                skip_decoder=True,
            )
            perturbed_primitives = None
            if epoch >= warmup_epochs and criterion.config.lambda_evidence_mono != 0.0:
                perturbation = perturbation_kinds[(epoch + batch_idx) % len(perturbation_kinds)]
                perturbed_out = model(
                    queries,
                    apply_trace_perturbation(trace_inputs, perturbation),
                    target_ids,
                    bottleneck_mode="gumbel",
                    tau=1.0,
                    skip_decoder=True,
                )
                perturbed_primitives = perturbed_out["primitives"]

            losses = criterion(
                out["decoder_logits"],
                out["primitives"],
                target_ids,
                prim_targets,
                out["executor_features"],
                evidence_targets=evidence_targets,
                perturbed_primitives=perturbed_primitives,
            )

            stage1_optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            stage1_optimizer.step()

            batch_size = queries.size(0)
            epoch_loss += losses["total"].item() * batch_size
            epoch_items += batch_size

        train_loss = epoch_loss / max(epoch_items, 1)
        val_metrics, val_bundle = _evaluate_stage1(
            model,
            stage1_criterion,
            val_loader,
            device,
            records=val_dataset.records,
            collect_bundle=True,
        )
        val_collapse_audit = _collapse_audit_from_bundle(
            val_bundle,
            collapse_share_threshold=config.trace_selection_collapse_share_threshold,
        )
        val_prediction_audit = _prediction_audit_from_bundle(val_dataset.records, val_bundle)
        val_evidence_audit = _evidence_audit_from_bundle(val_dataset.records, val_bundle)
        print(
            f"Stage1 Epoch {epoch + 1:02d}/{config.trace_stage1_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={_metric_value(val_metrics, 'loss'):.4f} | "
            f"val_outcome_acc={_metric_value(val_metrics, 'outcome_acc'):.3f} | "
            f"val_outcome_macro_f1={_metric_value(val_metrics, 'outcome_macro_f1'):.3f} | "
            f"val_failure_recall={_metric_value(val_metrics, 'failure_recall'):.3f} | "
            f"val_confidence_macro_f1={_metric_value(val_metrics, 'confidence_macro_f1'):.3f} | "
            f"val_trajectory_macro_f1={_metric_value(val_metrics, 'trajectory_macro_f1'):.3f} | "
            f"val_pattern_macro_f1={_metric_value(val_metrics, 'pattern_macro_f1'):.3f} | "
            f"val_joint_acc={_metric_value(val_metrics, 'joint_acc'):.3f} | "
            f"val_collapsed={val_collapse_audit['collapsed_model']}"
        )
        should_log_val_audit = (
            epoch == 0
            or bool(val_collapse_audit["collapsed_model"])
            or (epoch + 1) % max(int(config.trace_epoch_audit_interval), 1) == 0
            or epoch + 1 == config.trace_stage1_epochs
        )
        if should_log_val_audit:
            val_corr = val_evidence_audit.get("correlations", {})
            print(
                "  Val audit: "
                f"out_share={val_prediction_audit['outcome_share']['SUCCESS_LIKELY']:.3f}/"
                f"{val_prediction_audit['outcome_share']['UNCERTAIN']:.3f}/"
                f"{val_prediction_audit['outcome_share']['FAILURE_LIKELY']:.3f} | "
                f"conf_share={val_prediction_audit['confidence_share']['LOW']:.3f}/"
                f"{val_prediction_audit['confidence_share']['MEDIUM']:.3f}/"
                f"{val_prediction_audit['confidence_share']['HIGH']:.3f} | "
                f"cert_std(st/su/fa)={val_evidence_audit['stability']['std']:.4f}/"
                f"{val_evidence_audit['success']['std']:.4f}/"
                f"{val_evidence_audit['failure']['std']:.4f} | "
                f"succ_vs_fail_corr={val_corr.get('success_vs_failure', float('nan')):.3f}"
            )
            val_guard_alignment = val_evidence_audit.get("success_guard_alignment", {})
            if val_guard_alignment:
                print(
                    "  Val success guard: "
                    f"base_over_rate={val_guard_alignment['base_over_guard_rate']:.3f} | "
                    f"base_target_over={val_guard_alignment.get('base_target_over_rate', float('nan')):.3f} | "
                    f"proposal_over_rate={val_guard_alignment['proposal_over_guard_rate']:.3f} | "
                    f"success_over_rate={val_guard_alignment['success_over_guard_rate']:.3f} | "
                    f"success_prob_over_rate={val_guard_alignment['success_prob_over_guard_rate']:.3f} | "
                    f"bonus_weight={val_guard_alignment['bonus_weight_mean']:.3f} | "
                    f"bonus_frac={val_guard_alignment['bonus_fraction_of_success_mean']:.3f} | "
                    f"proposal_gain_frac={val_guard_alignment['proposal_gain_fraction_of_success_mean']:.3f} | "
                    f"bonus_zero_raw_frac={val_guard_alignment.get('bonus_zero_raw_fraction_of_bonus_mean', float('nan')):.3f} | "
                    f"guard_target_over={val_guard_alignment.get('guard_target_over_rate', float('nan')):.3f}"
                )
                print(
                    "  Val success parts: "
                    f"core_frac={val_guard_alignment.get('success_core_fraction_of_success_mean', float('nan')):.3f} | "
                    f"resid_frac={val_guard_alignment.get('success_residual_fraction_of_success_mean', float('nan')):.3f} | "
                    f"base_vs_stability={val_corr.get('success_base_vs_stability', float('nan')):.3f} | "
                    f"core_vs_stability={val_corr.get('success_core_vs_stability', float('nan')):.3f} | "
                    f"core_vs_good={val_corr.get('success_core_vs_trajectory_good', float('nan')):.3f} | "
                    f"base_vs_support={val_corr.get('success_base_vs_support', float('nan')):.3f} | "
                    f"resid_vs_good={val_corr.get('success_residual_vs_trajectory_good', float('nan')):.3f} | "
                    f"resid_vs_pattern={val_corr.get('success_residual_vs_pattern_focused', float('nan')):.3f} | "
                    f"resid_vs_trace={val_corr.get('success_residual_vs_success_trace_strength', float('nan')):.3f} | "
                    f"bonus_weight_vs_zero_raw={val_corr.get('success_bonus_weight_vs_zero_raw', float('nan')):.3f}"
                )
            val_target_fit = val_evidence_audit.get("target_fit", {})
            if val_target_fit:
                print(
                    "  Val target fit: "
                    f"stability={val_target_fit['stability']['corr']:.3f}/{val_target_fit['stability']['spread_ratio']:.3f} | "
                    f"support={val_target_fit['support']['corr']:.3f}/{val_target_fit['support']['spread_ratio']:.3f} | "
                    f"success={val_target_fit['success']['corr']:.3f}/{val_target_fit['success']['spread_ratio']:.3f} | "
                    f"failure={val_target_fit['failure']['corr']:.3f}/{val_target_fit['failure']['spread_ratio']:.3f}"
                )

        if _is_better_stage1(
            val_metrics,
            best_stage1_metrics,
            candidate_collapsed=bool(val_collapse_audit["collapsed_model"]),
            best_collapsed=bool(best_stage1_collapse_audit["collapsed_model"]),
            tolerance=config.trace_stage1_selection_tolerance,
        ):
            best_stage1_state = copy.deepcopy(model.state_dict())
            best_stage1_metrics = val_metrics
            best_stage1_collapse_audit = val_collapse_audit
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.trace_stage1_patience:
                print(f"Early stopping stage 1 at epoch {epoch + 1}")
                break

    model.load_state_dict(best_stage1_state)
    if best_stage1_collapse_audit.get("collapsed_model", False):
        print("Warning: every stage-1 checkpoint was collapsed under the validation audit.")
    pre_stage2_predictions = _collect_primitive_predictions(model, test_loader, device)

    print("Stage 2: generation-only training")
    _freeze_module(model.bottleneck)
    _freeze_module(model.encoder)
    _unfreeze_module(model.decoder)
    stage2_optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_stage2_state = copy.deepcopy(model.state_dict())
    best_stage2_loss = float("inf")
    for epoch in range(config.trace_stage2_epochs):
        model.train()
        model.encoder.eval()
        model.bottleneck.eval()
        epoch_loss = 0.0
        epoch_items = 0

        for batch in train_loader:
            queries, _, _, prim_targets, target_ids = _move_batch(batch, device)
            logits = _decoder_logits_from_ground_truth(model, queries, prim_targets, target_ids)
            loss = _generation_loss(logits, target_ids, pad_idx=vocab.pad_idx)

            stage2_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            stage2_optimizer.step()

            batch_size = queries.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_items += batch_size

        train_gen_loss = epoch_loss / max(epoch_items, 1)
        val_gen_loss = _evaluate_stage2(model, val_loader, device, pad_idx=vocab.pad_idx)
        print(
            f"Stage2 Epoch {epoch + 1:02d}/{config.trace_stage2_epochs} | "
            f"train_gen_loss={train_gen_loss:.4f} | "
            f"val_gen_loss={val_gen_loss:.4f}"
        )
        if val_gen_loss < best_stage2_loss:
            best_stage2_loss = val_gen_loss
            best_stage2_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_stage2_state)
    post_stage2_predictions = _collect_primitive_predictions(model, test_loader, device)
    primitive_predictions_preserved = torch.equal(pre_stage2_predictions, post_stage2_predictions)

    test_metrics, test_bundle = _evaluate_stage1(
        model,
        stage1_criterion,
        test_loader,
        device,
        records=test_dataset.records,
        collect_bundle=True,
    )
    print(
        "Test metrics: "
        f"trajectory_acc={_metric_value(test_metrics, 'trajectory_acc'):.3f}, "
        f"trajectory_macro_f1={_metric_value(test_metrics, 'trajectory_macro_f1'):.3f}, "
        f"pattern_acc={_metric_value(test_metrics, 'pattern_acc'):.3f}, "
        f"pattern_macro_f1={_metric_value(test_metrics, 'pattern_macro_f1'):.3f}, "
        f"confidence_acc={_metric_value(test_metrics, 'confidence_acc'):.3f}, "
        f"confidence_macro_f1={_metric_value(test_metrics, 'confidence_macro_f1'):.3f}, "
        f"outcome_acc={_metric_value(test_metrics, 'outcome_acc'):.3f}, "
        f"outcome_macro_f1={_metric_value(test_metrics, 'outcome_macro_f1'):.3f}, "
        f"failure_recall={_metric_value(test_metrics, 'failure_recall'):.3f}, "
        f"joint_acc={_metric_value(test_metrics, 'joint_acc'):.3f}"
    )
    print(
        f"Trajectory confusion: "
        f"{_format_confusion(test_metrics['trajectory_confusion'], TRAJECTORY_SHAPE_LABELS)}"
    )
    print(
        f"Pattern confusion: "
        f"{_format_confusion(test_metrics['pattern_confusion'], ATTENTION_PATTERN_LABELS)}"
    )
    print(
        "Pattern per-class: "
        f"FOCUSED(p/r/f1)={_metric_value(test_metrics, 'pattern_focused_precision'):.3f}/"
        f"{_metric_value(test_metrics, 'pattern_focused_recall'):.3f}/"
        f"{_metric_value(test_metrics, 'pattern_focused_f1'):.3f} | "
        f"MIXED(p/r/f1)={_metric_value(test_metrics, 'pattern_mixed_precision'):.3f}/"
        f"{_metric_value(test_metrics, 'pattern_mixed_recall'):.3f}/"
        f"{_metric_value(test_metrics, 'pattern_mixed_f1'):.3f} | "
        f"DIFFUSE(p/r/f1)={_metric_value(test_metrics, 'pattern_diffuse_precision'):.3f}/"
        f"{_metric_value(test_metrics, 'pattern_diffuse_recall'):.3f}/"
        f"{_metric_value(test_metrics, 'pattern_diffuse_f1'):.3f}"
    )
    print(f"Outcome confusion: {_format_confusion(test_metrics['outcome_confusion'], OUTCOME_LABELS)}")
    print(
        "Outcome per-class: "
        f"SUCCESS(p/r/f1)={_metric_value(test_metrics, 'success_likely_precision'):.3f}/"
        f"{_metric_value(test_metrics, 'success_likely_recall'):.3f}/"
        f"{_metric_value(test_metrics, 'success_likely_f1'):.3f} | "
        f"UNCERTAIN(p/r/f1)={_metric_value(test_metrics, 'uncertain_precision'):.3f}/"
        f"{_metric_value(test_metrics, 'uncertain_recall'):.3f}/"
        f"{_metric_value(test_metrics, 'uncertain_f1'):.3f} | "
        f"FAILURE(p/r/f1)={_metric_value(test_metrics, 'failure_likely_precision'):.3f}/"
        f"{_metric_value(test_metrics, 'failure_likely_recall'):.3f}/"
        f"{_metric_value(test_metrics, 'failure_likely_f1'):.3f}"
    )
    print(f"Primitive predictions preserved through stage 2: {primitive_predictions_preserved}")

    train_records = load_trace_records(trace_dir / "train.jsonl")
    val_records = load_trace_records(trace_dir / "val.jsonl")
    test_records = load_trace_records(trace_dir / "test.jsonl")
    corpus_records = train_records + val_records + test_records
    majority = _majority_baseline(train_records, test_records)
    baseline_results = evaluate_trace_baselines(train_records, test_records, config=config)
    correctness_baselines = evaluate_correctness_baselines(
        train_records,
        test_records,
        mlp_max_iter=config.baseline_mlp_max_iter,
        mlp_early_stopping=config.baseline_mlp_early_stopping,
    )
    label_audit = compute_label_audit(corpus_records)
    threshold_occupancy = compute_threshold_occupancy(corpus_records, trace_metadata)
    threshold_sensitivity = compute_threshold_sensitivity(corpus_records, trace_metadata)
    monotonicity = audit_monotonicity(model, test_loader, device)
    prediction_audit = _prediction_audit_from_bundle(test_dataset.records, test_bundle)
    collapse_audit = _collapse_audit_from_bundle(
        test_bundle,
        collapse_share_threshold=config.trace_selection_collapse_share_threshold,
    )
    evidence_audit = _evidence_audit_from_bundle(test_dataset.records, test_bundle)
    counterfactual_audit = _counterfactual_path_audit(model, test_loader, device)
    path_audit_summary = _path_audit_summary(counterfactual_audit)
    metadata_path = trace_dir / "metadata.json"
    if metadata_path.exists() and not args.skip_reference_write:
        with open(metadata_path) as f:
            metadata = json.load(f)
        metadata["mlp_converged"] = baseline_results["mlp"]["outcome"]["converged"]
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    reference_path = _reference_metrics_path(trace_dir)
    created_reference = False
    if not args.skip_reference_write:
        created_reference = _maybe_write_reference_metrics(
            reference_path,
            trace_info=trace_info,
            test_metrics=test_metrics,
            baseline_results=baseline_results,
        )
    reference_metrics = _load_reference_metrics(reference_path)
    print(
        "Baselines: "
        f"majority_outcome={OUTCOME_LABELS[int(majority['class_idx'])]}, "
        f"majority_acc={majority['accuracy']:.3f}, "
        f"majority_macro_f1={majority['macro_f1']:.3f}, "
        f"entropy_threshold_acc={baseline_results['entropy_threshold']['outcome_accuracy']:.3f}, "
        f"entropy_threshold_macro_f1={baseline_results['entropy_threshold']['outcome_macro_f1']:.3f}, "
        f"logreg_outcome_acc={baseline_results['logistic_regression']['outcome']['accuracy']:.3f}, "
        f"logreg_outcome_macro_f1={baseline_results['logistic_regression']['outcome']['macro_f1']:.3f}, "
        f"mlp_outcome_acc={baseline_results['mlp']['outcome']['accuracy']:.3f}, "
        f"mlp_outcome_macro_f1={baseline_results['mlp']['outcome']['macro_f1']:.3f}, "
        f"mlp_converged={baseline_results['mlp']['outcome']['converged']}, "
        f"mlp_n_iter={baseline_results['mlp']['outcome']['n_iter']}"
    )
    print(
        "Correctness baselines: "
        f"logreg_acc={correctness_baselines['logistic_regression']['accuracy']:.3f}, "
        f"logreg_auroc={correctness_baselines['logistic_regression']['auroc']:.3f}, "
        f"mlp_acc={correctness_baselines['mlp']['accuracy']:.3f}, "
        f"mlp_auroc={correctness_baselines['mlp']['auroc']:.3f}, "
        f"mlp_converged={correctness_baselines['mlp']['converged']}"
    )
    print(
        "Safety metrics: "
        f"unsafe_success_rate_label={_metric_value(test_metrics, 'unsafe_success_rate_label'):.3f}, "
        f"unsafe_success_rate_correctness={_metric_value(test_metrics, 'unsafe_success_rate_correctness'):.3f}, "
        f"success_precision_vs_is_correct={_metric_value(test_metrics, 'success_precision_vs_is_correct'):.3f}, "
        f"success_auroc={_metric_value(test_metrics, 'success_auroc'):.3f}, "
        f"success_auprc={_metric_value(test_metrics, 'success_auprc'):.3f}, "
        f"success_brier={_metric_value(test_metrics, 'success_brier'):.3f}, "
        f"success_ece_10bin={_metric_value(test_metrics, 'success_ece_10bin'):.3f}"
    )
    print(
        "Corpus label audit: "
        f"P(correct|SUCCESS)={label_audit['p_is_correct_given_outcome']['SUCCESS_LIKELY']:.3f}, "
        f"P(correct|UNCERTAIN)={label_audit['p_is_correct_given_outcome']['UNCERTAIN']:.3f}, "
        f"P(correct|FAILURE)={label_audit['p_is_correct_given_outcome']['FAILURE_LIKELY']:.3f}, "
        f"correctness_separation={label_audit['correctness_separation_success_minus_uncertain']:.3f}, "
        f"max_outcome_share_swing={threshold_sensitivity['max_outcome_share_swing']:.3f}, "
        f"label_revision_trigger={threshold_sensitivity['label_revision_trigger']}"
    )
    print(
        "Prediction audit: "
        f"pred_share_SUCCESS={prediction_audit['outcome_share']['SUCCESS_LIKELY']:.3f}, "
        f"pred_share_UNCERTAIN={prediction_audit['outcome_share']['UNCERTAIN']:.3f}, "
        f"pred_share_FAILURE={prediction_audit['outcome_share']['FAILURE_LIKELY']:.3f}, "
        f"pred_conf_LOW={prediction_audit['confidence_share']['LOW']:.3f}, "
        f"pred_conf_MEDIUM={prediction_audit['confidence_share']['MEDIUM']:.3f}, "
        f"pred_conf_HIGH={prediction_audit['confidence_share']['HIGH']:.3f}, "
        f"P(correct|pred_SUCCESS)={prediction_audit['p_is_correct_given_outcome']['SUCCESS_LIKELY']:.3f}, "
        f"P(correct|pred_UNCERTAIN)={prediction_audit['p_is_correct_given_outcome']['UNCERTAIN']:.3f}, "
        f"P(correct|pred_FAILURE)={prediction_audit['p_is_correct_given_outcome']['FAILURE_LIKELY']:.3f}"
    )
    success_over_by_pred = prediction_audit.get("success_prob_over_guard_rate_by_predicted_outcome")
    success_over_excess_by_pred = prediction_audit.get("success_prob_over_guard_excess_by_predicted_outcome")
    success_prob_mean_by_pred = prediction_audit.get("success_prob_mean_by_predicted_outcome")
    success_guard_mean_by_pred = prediction_audit.get("success_guard_mean_by_predicted_outcome")
    if success_over_by_pred:
        print(
            "Prediction guard audit: "
            f"pred_SUCCESS_over_guard={success_over_by_pred['SUCCESS_LIKELY']:.3f}, "
            f"pred_UNCERTAIN_over_guard={success_over_by_pred['UNCERTAIN']:.3f}, "
            f"pred_FAILURE_over_guard={success_over_by_pred['FAILURE_LIKELY']:.3f}"
        )
    if success_over_excess_by_pred and success_prob_mean_by_pred and success_guard_mean_by_pred:
        print(
            "Prediction guard gap: "
            f"pred_SUCCESS(prob/guard/excess)={success_prob_mean_by_pred['SUCCESS_LIKELY']:.3f}/"
            f"{success_guard_mean_by_pred['SUCCESS_LIKELY']:.3f}/"
            f"{success_over_excess_by_pred['SUCCESS_LIKELY']:.3f}, "
            f"pred_UNCERTAIN(prob/guard/excess)={success_prob_mean_by_pred['UNCERTAIN']:.3f}/"
            f"{success_guard_mean_by_pred['UNCERTAIN']:.3f}/"
            f"{success_over_excess_by_pred['UNCERTAIN']:.3f}, "
            f"pred_FAILURE(prob/guard/excess)={success_prob_mean_by_pred['FAILURE_LIKELY']:.3f}/"
            f"{success_guard_mean_by_pred['FAILURE_LIKELY']:.3f}/"
            f"{success_over_excess_by_pred['FAILURE_LIKELY']:.3f}"
        )
    success_over_by_correctness = prediction_audit.get("predicted_success_over_guard_rate_by_correctness")
    success_excess_by_correctness = prediction_audit.get("predicted_success_over_guard_excess_by_correctness")
    success_count_by_correctness = prediction_audit.get("predicted_success_count_by_correctness")
    if success_over_by_correctness and success_excess_by_correctness and success_count_by_correctness:
        print(
            "Prediction success calibration: "
            f"correct(count/rate/excess)={success_count_by_correctness['correct']}/"
            f"{success_over_by_correctness['correct']:.3f}/"
            f"{success_excess_by_correctness['correct']:.3f}, "
            f"incorrect(count/rate/excess)={success_count_by_correctness['incorrect']}/"
            f"{success_over_by_correctness['incorrect']:.3f}/"
            f"{success_excess_by_correctness['incorrect']:.3f}"
        )
    print(
        "Collapse audit: "
        f"collapsed_model={collapse_audit['collapsed_model']}, "
        f"all_success_mode={collapse_audit['all_success_mode']}, "
        f"all_uncertain_mode={collapse_audit['all_uncertain_mode']}, "
        f"all_failure_mode={collapse_audit['all_failure_mode']}, "
        f"single_confidence_mode={collapse_audit['single_confidence_mode']}"
    )
    if evidence_audit:
        corr = evidence_audit.get("correlations", {})
        print(
            "Evidence geometry: "
            f"stability_std={evidence_audit['stability']['std']:.4f}, "
            f"support_std={evidence_audit['support']['std']:.4f}, "
            f"success_guard_std={evidence_audit['success_guard']['std']:.4f}, "
            f"success_core_std={evidence_audit['success_core']['std']:.4f}, "
            f"success_residual_std={evidence_audit['success_residual']['std']:.4f}, "
            f"success_base_std={evidence_audit['success_base']['std']:.4f}, "
            f"success_trace_strength_std={evidence_audit['success_trace_strength']['std']:.4f}, "
            f"success_bonus_weight_std={evidence_audit['success_bonus_weight']['std']:.4f}, "
            f"success_bonus_zero_raw_std={evidence_audit['success_bonus_zero_raw']['std']:.4f}, "
            f"success_bonus_std={evidence_audit['success_bonus']['std']:.4f}, "
            f"success_proposal_std={evidence_audit['success_proposal']['std']:.4f}, "
            f"success_std={evidence_audit['success']['std']:.4f}, "
            f"failure_std={evidence_audit['failure']['std']:.4f}, "
            f"ambiguity_std={evidence_audit['ambiguity']['std']:.4f}, "
            f"success_vs_failure_corr={corr.get('success_vs_failure', float('nan')):.3f}, "
            f"success_guard_vs_success_corr={corr.get('success_guard_vs_success', float('nan')):.3f}, "
            f"success_guard_vs_base_corr={corr.get('success_guard_vs_success_base', float('nan')):.3f}, "
            f"success_guard_vs_proposal_corr={corr.get('success_guard_vs_success_proposal', float('nan')):.3f}, "
            f"success_guard_vs_target_corr={corr.get('success_guard_vs_success_target', float('nan')):.3f}, "
            f"success_guard_vs_clipped_target_corr={corr.get('success_guard_vs_guard_target', float('nan')):.3f}, "
            f"success_core_vs_clipped_target_corr={corr.get('success_core_vs_guard_target', float('nan')):.3f}, "
            f"success_base_vs_clipped_target_corr={corr.get('success_base_vs_guard_target', float('nan')):.3f}, "
            f"success_base_vs_stability_corr={corr.get('success_base_vs_stability', float('nan')):.3f}, "
            f"success_core_vs_stability_corr={corr.get('success_core_vs_stability', float('nan')):.3f}, "
            f"success_core_vs_good_corr={corr.get('success_core_vs_trajectory_good', float('nan')):.3f}, "
            f"success_core_vs_pattern_corr={corr.get('success_core_vs_pattern_focused', float('nan')):.3f}, "
            f"success_residual_vs_good_corr={corr.get('success_residual_vs_trajectory_good', float('nan')):.3f}, "
            f"success_residual_vs_pattern_corr={corr.get('success_residual_vs_pattern_focused', float('nan')):.3f}, "
            f"success_residual_vs_trace_corr={corr.get('success_residual_vs_success_trace_strength', float('nan')):.3f}, "
            f"bonus_weight_vs_success_corr={corr.get('success_bonus_weight_vs_success', float('nan')):.3f}, "
            f"bonus_weight_vs_zero_raw_corr={corr.get('success_bonus_weight_vs_zero_raw', float('nan')):.3f}, "
            f"proposal_vs_success_corr={corr.get('success_proposal_vs_success', float('nan')):.3f}, "
            f"ambiguity_vs_high_conf_corr={corr.get('ambiguity_vs_high_confidence_prob', float('nan')):.3f}, "
            f"stability_vs_agreement_corr={corr.get('stability_vs_agreement', float('nan')):.3f}, "
            f"support_vs_support_ratio_corr={corr.get('support_vs_support_ratio', float('nan')):.3f}"
        )
        guard_alignment = evidence_audit.get("success_guard_alignment", {})
        if guard_alignment:
            print(
                "Success guard: "
                f"utilization_mean={guard_alignment['guard_utilization_mean']:.3f}, "
                f"core_utilization_mean={guard_alignment.get('core_utilization_mean', float('nan')):.3f}, "
                f"base_utilization_mean={guard_alignment['base_utilization_mean']:.3f}, "
                f"proposal_utilization_mean={guard_alignment['proposal_utilization_mean']:.3f}, "
                f"guard_target_mae={guard_alignment.get('guard_target_mae', float('nan')):.3f}, "
                f"guard_target_bias_mean={guard_alignment.get('guard_target_bias_mean', float('nan')):.3f}, "
                f"guard_target_over_rate={guard_alignment.get('guard_target_over_rate', float('nan')):.3f}, "
                f"base_target_excess_mean={guard_alignment.get('base_target_excess_mean', float('nan')):.3f}, "
                f"base_target_over_rate={guard_alignment.get('base_target_over_rate', float('nan')):.3f}, "
                f"guard_low_sat_rate={guard_alignment['guard_low_saturation_rate']:.3f}, "
                f"guard_high_sat_rate={guard_alignment['guard_high_saturation_rate']:.3f}, "
                f"base_over_guard_mean={guard_alignment['base_over_guard_mean']:.3f}, "
                f"base_over_guard_rate={guard_alignment['base_over_guard_rate']:.3f}, "
                f"bonus_weight_mean={guard_alignment['bonus_weight_mean']:.3f}, "
                f"bonus_weight_std={guard_alignment['bonus_weight_std']:.3f}, "
                f"bonus_weight_zero_raw_mean={guard_alignment.get('bonus_weight_zero_raw_mean', float('nan')):.3f}, "
                f"bonus_weight_zero_raw_fraction_mean={guard_alignment.get('bonus_weight_zero_raw_fraction_mean', float('nan')):.3f}, "
                f"bonus_mean={guard_alignment['bonus_mean']:.3f}, "
                f"bonus_zero_raw_mean={guard_alignment.get('bonus_zero_raw_mean', float('nan')):.3f}, "
                f"bonus_zero_raw_fraction_of_bonus_mean={guard_alignment.get('bonus_zero_raw_fraction_of_bonus_mean', float('nan')):.3f}, "
                f"bonus_evidence_fraction_of_bonus_mean={guard_alignment.get('bonus_evidence_fraction_of_bonus_mean', float('nan')):.3f}, "
                f"success_core_fraction_of_success_mean={guard_alignment.get('success_core_fraction_of_success_mean', float('nan')):.3f}, "
                f"success_residual_fraction_of_success_mean={guard_alignment.get('success_residual_fraction_of_success_mean', float('nan')):.3f}, "
                f"bonus_fraction_of_success_mean={guard_alignment['bonus_fraction_of_success_mean']:.3f}, "
                f"success_trace_strength_mean={guard_alignment.get('success_trace_strength_mean', float('nan')):.3f}, "
                f"success_trace_strength_std={guard_alignment.get('success_trace_strength_std', float('nan')):.3f}, "
                f"success_residual_weight_mean={guard_alignment.get('success_residual_weight_mean', float('nan')):.3f}, "
                f"success_residual_weight_std={guard_alignment.get('success_residual_weight_std', float('nan')):.3f}, "
                f"proposal_gain_mean={guard_alignment['proposal_gain_mean']:.3f}, "
                f"proposal_gain_fraction_of_success_mean={guard_alignment['proposal_gain_fraction_of_success_mean']:.3f}, "
                f"success_over_guard_mean={guard_alignment['success_over_guard_mean']:.3f}, "
                f"success_over_guard_rate={guard_alignment['success_over_guard_rate']:.3f}, "
                f"proposal_over_guard_mean={guard_alignment['proposal_over_guard_mean']:.3f}, "
                f"proposal_over_guard_rate={guard_alignment['proposal_over_guard_rate']:.3f}, "
                f"success_prob_over_guard_mean={guard_alignment['success_prob_over_guard_mean']:.3f}, "
                f"success_prob_over_guard_rate={guard_alignment['success_prob_over_guard_rate']:.3f}"
            )
    target_fit = evidence_audit.get("target_fit", {})
    if target_fit:
        print(
            "Evidence target fit: "
            f"stability(corr/mae/spread)={target_fit['stability']['corr']:.3f}/{target_fit['stability']['mae']:.3f}/{target_fit['stability']['spread_ratio']:.3f} | "
            f"support(corr/mae/spread)={target_fit['support']['corr']:.3f}/{target_fit['support']['mae']:.3f}/{target_fit['support']['spread_ratio']:.3f} | "
            f"success(corr/mae/spread)={target_fit['success']['corr']:.3f}/{target_fit['success']['mae']:.3f}/{target_fit['success']['spread_ratio']:.3f} | "
            f"failure(corr/mae/spread)={target_fit['failure']['corr']:.3f}/{target_fit['failure']['mae']:.3f}/{target_fit['failure']['spread_ratio']:.3f} | "
            f"ambiguity(corr/mae/spread)={target_fit['ambiguity']['corr']:.3f}/{target_fit['ambiguity']['mae']:.3f}/{target_fit['ambiguity']['spread_ratio']:.3f}"
        )
    print(
        "Monotonicity: "
        f"violation_rate={monotonicity['monotonicity_violation_rate']:.3f}, "
        f"all_supported_violation_rate={monotonicity['all_supported_monotonicity_violation_rate']:.3f}, "
        f"stability_cert_violation_rate={monotonicity.get('stability_cert_violation_rate', float('nan')):.3f}, "
        f"support_cert_violation_rate={monotonicity.get('support_cert_violation_rate', float('nan')):.3f}, "
        f"success_guard_violation_rate={monotonicity.get('success_guard_violation_rate', float('nan')):.3f}, "
        f"trajectory_good_violation_rate={monotonicity.get('trajectory_good_violation_rate', float('nan')):.3f}, "
        f"pattern_focused_violation_rate={monotonicity.get('pattern_focused_violation_rate', float('nan')):.3f}, "
        f"success_core_violation_rate={monotonicity.get('success_core_violation_rate', float('nan')):.3f}, "
        f"success_base_violation_rate={monotonicity.get('success_base_violation_rate', float('nan')):.3f}, "
        f"success_base_guard_gap_violation_rate={monotonicity.get('success_base_guard_gap_violation_rate', float('nan')):.3f}, "
        f"success_cert_violation_rate={monotonicity.get('success_cert_violation_rate', float('nan')):.3f}, "
        f"failure_cert_violation_rate={monotonicity.get('failure_cert_violation_rate', float('nan')):.3f}"
    )
    print(
        "Path audit: "
        f"zero_success_core(cert_shift/outcome_flip)={counterfactual_audit['zero_success_core']['success_cert_mae'] + counterfactual_audit['zero_success_core']['failure_cert_mae']:.3f}/{counterfactual_audit['zero_success_core']['outcome_label_flip_rate']:.3f}, "
        f"zero_success_residual(cert_shift/outcome_flip)={counterfactual_audit['zero_success_residual']['success_cert_mae'] + counterfactual_audit['zero_success_residual']['failure_cert_mae']:.3f}/{counterfactual_audit['zero_success_residual']['outcome_label_flip_rate']:.3f}, "
        f"zero_success_base(cert_shift/outcome_flip)={counterfactual_audit['zero_success_base']['success_cert_mae'] + counterfactual_audit['zero_success_base']['failure_cert_mae']:.3f}/{counterfactual_audit['zero_success_base']['outcome_label_flip_rate']:.3f}, "
        f"zero_success_bonus(cert_shift/outcome_flip)={counterfactual_audit['zero_success_bonus']['success_cert_mae'] + counterfactual_audit['zero_success_bonus']['failure_cert_mae']:.3f}/{counterfactual_audit['zero_success_bonus']['outcome_label_flip_rate']:.3f}, "
        f"zero_success_trace(cert_shift/outcome_flip)={counterfactual_audit['zero_success_trace']['success_cert_mae'] + counterfactual_audit['zero_success_trace']['failure_cert_mae']:.3f}/{counterfactual_audit['zero_success_trace']['outcome_label_flip_rate']:.3f}, "
        f"zero_success_raw(cert_shift/outcome_flip)={counterfactual_audit['zero_success_raw']['success_cert_mae'] + counterfactual_audit['zero_success_raw']['failure_cert_mae']:.3f}/{counterfactual_audit['zero_success_raw']['outcome_label_flip_rate']:.3f}, "
        f"zero_failure_trace(cert_shift/outcome_flip)={counterfactual_audit['zero_failure_trace']['success_cert_mae'] + counterfactual_audit['zero_failure_trace']['failure_cert_mae']:.3f}/{counterfactual_audit['zero_failure_trace']['outcome_label_flip_rate']:.3f}, "
        f"zero_failure_raw(cert_shift/outcome_flip)={counterfactual_audit['zero_failure_raw']['success_cert_mae'] + counterfactual_audit['zero_failure_raw']['failure_cert_mae']:.3f}/{counterfactual_audit['zero_failure_raw']['outcome_label_flip_rate']:.3f}, "
        f"zero_ambiguity_trace(cert_shift/outcome_flip)={counterfactual_audit['zero_ambiguity_trace']['success_cert_mae'] + counterfactual_audit['zero_ambiguity_trace']['failure_cert_mae']:.3f}/{counterfactual_audit['zero_ambiguity_trace']['outcome_label_flip_rate']:.3f}, "
        f"zero_ambiguity_raw(cert_shift/outcome_flip)={counterfactual_audit['zero_ambiguity_raw']['success_cert_mae'] + counterfactual_audit['zero_ambiguity_raw']['failure_cert_mae']:.3f}/{counterfactual_audit['zero_ambiguity_raw']['outcome_label_flip_rate']:.3f}, "
        f"zero_shared_trace(cert_shift/outcome_flip)={counterfactual_audit['zero_shared_trace']['success_cert_mae'] + counterfactual_audit['zero_shared_trace']['failure_cert_mae']:.3f}/{counterfactual_audit['zero_shared_trace']['outcome_label_flip_rate']:.3f}, "
        f"zero_cert_context(cert_shift/outcome_flip)={counterfactual_audit['zero_cert_context']['success_cert_mae'] + counterfactual_audit['zero_cert_context']['failure_cert_mae']:.3f}/{counterfactual_audit['zero_cert_context']['outcome_label_flip_rate']:.3f}, "
        f"zero_outcome_trajectory_context(cert_shift/outcome_flip)={counterfactual_audit['zero_outcome_trajectory_context']['success_cert_mae'] + counterfactual_audit['zero_outcome_trajectory_context']['failure_cert_mae']:.3f}/{counterfactual_audit['zero_outcome_trajectory_context']['outcome_label_flip_rate']:.3f}, "
        f"zero_outcome_pattern_context(cert_shift/outcome_flip)={counterfactual_audit['zero_outcome_pattern_context']['success_cert_mae'] + counterfactual_audit['zero_outcome_pattern_context']['failure_cert_mae']:.3f}/{counterfactual_audit['zero_outcome_pattern_context']['outcome_label_flip_rate']:.3f}, "
        f"zero_outcome_context(cert_shift/outcome_flip)={counterfactual_audit['zero_outcome_context']['success_cert_mae'] + counterfactual_audit['zero_outcome_context']['failure_cert_mae']:.3f}/{counterfactual_audit['zero_outcome_context']['outcome_label_flip_rate']:.3f}"
    )
    outcome_context_breakdown = counterfactual_audit["zero_outcome_context"].get("outcome_flip_rate_by_base_label", {})
    if outcome_context_breakdown:
        print(
            "Outcome-context flips by base label: "
            f"SUCCESS={outcome_context_breakdown['SUCCESS_LIKELY']:.3f}, "
            f"UNCERTAIN={outcome_context_breakdown['UNCERTAIN']:.3f}, "
            f"FAILURE={outcome_context_breakdown['FAILURE_LIKELY']:.3f}"
        )
    print(
        "Path dominance: "
        f"cert_driver={path_audit_summary['largest_cert_shift']['scenario']}:{path_audit_summary['largest_cert_shift']['value']:.3f}, "
        f"outcome_driver={path_audit_summary['largest_outcome_flip']['scenario']}:{path_audit_summary['largest_outcome_flip']['value']:.3f}, "
        f"confidence_driver={path_audit_summary['largest_confidence_flip']['scenario']}:{path_audit_summary['largest_confidence_flip']['value']:.3f}, "
        f"single_path_cert_driver={path_audit_summary['largest_single_path_cert_shift']['scenario']}:{path_audit_summary['largest_single_path_cert_shift']['value']:.3f}, "
        f"single_path_outcome_driver={path_audit_summary['largest_single_path_outcome_flip']['scenario']}:{path_audit_summary['largest_single_path_outcome_flip']['value']:.3f}"
    )
    if created_reference:
        print(f"Reference metrics saved: {reference_path.name}")
    elif reference_metrics is not None:
        if reference_metrics.get("corpus_fingerprint") == trace_info.get("corpus_fingerprint"):
            ref_test = reference_metrics["test_metrics"]
            ref_mlp_macro_f1 = reference_metrics["baselines"]["mlp"]["outcome"]["macro_f1"]
            current_gap = baseline_results["mlp"]["outcome"]["macro_f1"] - _metric_value(test_metrics, "outcome_macro_f1")
            ref_gap = ref_mlp_macro_f1 - float(ref_test["outcome_macro_f1"])
            gap_shrink = 0.0 if ref_gap <= 1e-8 else max(0.0, (ref_gap - current_gap) / ref_gap)
            print(
                "Reference deltas: "
                f"outcome_macro_f1_delta={_metric_value(test_metrics, 'outcome_macro_f1') - float(ref_test['outcome_macro_f1']):+.3f}, "
                f"failure_recall_delta={_metric_value(test_metrics, 'failure_recall') - float(ref_test['failure_recall']):+.3f}, "
                f"trajectory_macro_f1_delta={_metric_value(test_metrics, 'trajectory_macro_f1') - float(ref_test['trajectory_macro_f1']):+.3f}, "
                f"pattern_macro_f1_delta={_metric_value(test_metrics, 'pattern_macro_f1') - float(ref_test['pattern_macro_f1']):+.3f}, "
                f"pattern_focused_recall_delta={_metric_value(test_metrics, 'pattern_focused_recall') - float(ref_test['pattern_focused_recall']):+.3f}, "
                f"pattern_diffuse_recall_delta={_metric_value(test_metrics, 'pattern_diffuse_recall') - float(ref_test['pattern_diffuse_recall']):+.3f}, "
                f"confidence_macro_f1_delta={_metric_value(test_metrics, 'confidence_macro_f1') - float(ref_test['confidence_macro_f1']):+.3f}, "
                f"mlp_gap_shrink={gap_shrink:.3f}"
            )
            outcome_gain_ok = (
                _metric_value(test_metrics, "outcome_macro_f1") >= float(ref_test["outcome_macro_f1"]) + 0.02
                or (
                    _metric_value(test_metrics, "outcome_macro_f1") >= float(ref_test["outcome_macro_f1"]) - 0.01
                    and _metric_value(test_metrics, "failure_recall") >= float(ref_test["failure_recall"]) + 0.05
                )
            )
            print(
                "Reference acceptance: "
                f"outcome_gain_ok={outcome_gain_ok}, "
                f"trajectory_gain_ok={_metric_value(test_metrics, 'trajectory_macro_f1') >= float(ref_test['trajectory_macro_f1']) + 0.08}, "
                f"pattern_gain_ok={_metric_value(test_metrics, 'pattern_macro_f1') >= float(ref_test['pattern_macro_f1']) + 0.05}, "
                f"focused_recall_gain_ok={_metric_value(test_metrics, 'pattern_focused_recall') >= float(ref_test['pattern_focused_recall']) + 0.10}, "
                f"diffuse_recall_gain_ok={_metric_value(test_metrics, 'pattern_diffuse_recall') >= float(ref_test['pattern_diffuse_recall']) + 0.10}, "
                f"pattern_not_collapsed_ok={_metric_value(test_metrics, 'pattern_focused_recall') >= 0.10 and _metric_value(test_metrics, 'pattern_diffuse_recall') >= 0.10}, "
                f"confidence_drop_ok={_metric_value(test_metrics, 'confidence_macro_f1') >= float(ref_test['confidence_macro_f1']) - 0.02}, "
                f"mlp_gap_shrink_ok={gap_shrink >= 0.30}"
            )
        else:
            print(
                "Reference metrics fingerprint mismatch: "
                f"{reference_metrics.get('corpus_fingerprint')} != {trace_info.get('corpus_fingerprint')}"
            )
    outcome_acc = _metric_value(test_metrics, "outcome_acc")
    beats_majority = _strictly_beats(outcome_acc, float(majority["accuracy"]))
    beats_entropy_threshold = _strictly_beats(
        outcome_acc,
        float(baseline_results["entropy_threshold"]["outcome_accuracy"]),
    )
    print(
        "Acceptance: "
        f"failure_recall_ok={_metric_value(test_metrics, 'failure_recall') >= 0.30}, "
        f"outcome_macro_f1_ok={_metric_value(test_metrics, 'outcome_macro_f1') >= 0.62}, "
        f"outcome_acc_ok={outcome_acc >= 0.82}, "
        f"beats_majority={beats_majority}, "
        f"beats_entropy_threshold={beats_entropy_threshold}, "
        f"trajectory_ok={_metric_value(test_metrics, 'trajectory_acc') >= 0.75}, "
        f"not_collapsed={not collapse_audit['collapsed_model']}, "
        f"monotonicity_ok={max(monotonicity.get('stability_cert_violation_rate', 0.0), monotonicity.get('support_cert_violation_rate', 0.0), monotonicity.get('success_guard_violation_rate', 0.0), monotonicity.get('success_cert_violation_rate', 0.0), monotonicity.get('failure_cert_violation_rate', 0.0)) < 0.05}"
    )

    query, trace_inputs, _, prim_targets, target_ids = test_dataset[0]
    batch_query = query.unsqueeze(0).to(device)
    batch_trace = trace_inputs.unsqueeze(0).to(device)
    batch_targets = target_ids.unsqueeze(0).to(device)

    out = model(batch_query, batch_trace, batch_targets, bottleneck_mode="hard", tau=1.0)
    pred = model.bottleneck.get_primitive_indices(out["primitives"])
    text = vocab.decode(model.generate(batch_query, batch_trace)["token_ids"][0].cpu().tolist())
    ground_truth_text = vocab.decode(target_ids.tolist())

    print(
        "Ground truth: "
        f"trajectory={TRAJECTORY_SHAPE_LABELS[prim_targets[0].item()]}, "
        f"pattern={ATTENTION_PATTERN_LABELS[prim_targets[1].item()]}, "
        f"confidence={CONFIDENCE_LABELS[prim_targets[2].item()]}, "
        f"outcome={OUTCOME_LABELS[prim_targets[3].item()]}"
    )
    print(
        "Prediction: "
        f"trajectory={TRAJECTORY_SHAPE_LABELS[pred['trajectory_shape'][0].item()]}, "
        f"pattern={ATTENTION_PATTERN_LABELS[pred['attention_pattern'][0].item()]}, "
        f"confidence={CONFIDENCE_LABELS[pred['confidence'][0].item()]}, "
        f"outcome={OUTCOME_LABELS[pred['outcome'][0].item()]}"
    )
    print(f"Decoder logits shape: {tuple(out['decoder_logits'].shape)}")
    print(f"Executor features shape: {tuple(out['executor_features'].shape)}")
    print(f"Reference text: {ground_truth_text}")
    print(f"Generated text: {text}")

    label_names = {
        "trajectory_shape": TRAJECTORY_SHAPE_LABELS,
        "attention_pattern": ATTENTION_PATTERN_LABELS,
        "confidence": CONFIDENCE_LABELS,
        "outcome": OUTCOME_LABELS,
    }
    prediction_rows = build_prediction_records(
        test_dataset.records,
        test_bundle,
        run_seed=config.train_loop_seed,
        label_names=label_names,
        collapse_audit=collapse_audit,
    )
    evidence_monotonicity_ok = bool(
        max(
            monotonicity.get("stability_cert_violation_rate", 0.0),
            monotonicity.get("support_cert_violation_rate", 0.0),
            monotonicity.get("success_guard_violation_rate", 0.0),
            monotonicity.get("success_cert_violation_rate", 0.0),
            monotonicity.get("failure_cert_violation_rate", 0.0),
        )
        < 0.05
    )
    acceptance = {
        "failure_recall_ok": bool(_metric_value(test_metrics, "failure_recall") > 0.0),
        "outcome_macro_f1_ok": bool(_metric_value(test_metrics, "outcome_macro_f1") > float(majority["macro_f1"])),
        "outcome_acc_ok": bool(outcome_acc > float(baseline_results["entropy_threshold"]["outcome_accuracy"])),
        "trajectory_ok": bool(_metric_value(test_metrics, "trajectory_acc") >= 0.75),
        "not_collapsed": bool(not collapse_audit["collapsed_model"]),
        "monotonicity_ok": evidence_monotonicity_ok,
        "label_revision_trigger": bool(threshold_sensitivity["label_revision_trigger"]),
    }
    sample_prediction = {
        "ground_truth": {
            "trajectory": TRAJECTORY_SHAPE_LABELS[prim_targets[0].item()],
            "pattern": ATTENTION_PATTERN_LABELS[prim_targets[1].item()],
            "confidence": CONFIDENCE_LABELS[prim_targets[2].item()],
            "outcome": OUTCOME_LABELS[prim_targets[3].item()],
        },
        "prediction": {
            "trajectory": TRAJECTORY_SHAPE_LABELS[pred["trajectory_shape"][0].item()],
            "pattern": ATTENTION_PATTERN_LABELS[pred["attention_pattern"][0].item()],
            "confidence": CONFIDENCE_LABELS[pred["confidence"][0].item()],
            "outcome": OUTCOME_LABELS[pred["outcome"][0].item()],
        },
        "reference_text": ground_truth_text,
        "generated_text": text,
    }
    metrics_payload = {
        "variant": variant_name,
        "experiment_label": args.experiment_label,
        "train_loop_seed": int(config.train_loop_seed),
        "corpus_fingerprint": trace_info.get("corpus_fingerprint"),
        "trace_label_version": trace_info.get("label_version"),
        "result_dir": str(output_dir),
        "trace_dir": str(trace_dir),
        "skip_reference_write": bool(args.skip_reference_write),
        "best_val_metrics": best_stage1_metrics,
        "best_val_collapse_audit": best_stage1_collapse_audit,
        "stage2": {
            "best_val_generation_loss": float(best_stage2_loss),
            "primitive_predictions_preserved": bool(primitive_predictions_preserved),
        },
        "test_metrics": test_metrics,
        "label_audit": label_audit,
        "prediction_audit": prediction_audit,
        "collapse_audit": collapse_audit,
        "evidence_audit": evidence_audit,
        "success_modes": _success_mode_summary(),
        "path_scales": _default_path_scales(config),
        "config_overrides": config_overrides,
        "counterfactual_audit": counterfactual_audit,
        "path_audit_summary": path_audit_summary,
        "threshold_occupancy": threshold_occupancy,
        "threshold_sensitivity": threshold_sensitivity,
        "baselines": {
            "majority": majority,
            "label_prediction": baseline_results,
            "correctness": correctness_baselines,
        },
        "monotonicity": monotonicity,
        "reference": {
            "path": str(reference_path),
            "created_reference": bool(created_reference),
            "available": bool(reference_metrics is not None),
            "fingerprint_match": bool(
                reference_metrics is not None
                and reference_metrics.get("corpus_fingerprint") == trace_info.get("corpus_fingerprint")
            ),
        },
        "acceptance": acceptance,
        "sample_prediction": sample_prediction,
    }
    summary_payload = {
        "variant": variant_name,
        "experiment_label": args.experiment_label,
        "train_loop_seed": int(config.train_loop_seed),
        "corpus_fingerprint": trace_info.get("corpus_fingerprint"),
        "result_dir": str(output_dir),
        "test_metrics": test_metrics,
        "label_audit": {
            "p_is_correct_given_outcome": label_audit["p_is_correct_given_outcome"],
            "correctness_separation_success_minus_uncertain": label_audit[
                "correctness_separation_success_minus_uncertain"
            ],
            "max_outcome_share_swing": threshold_sensitivity["max_outcome_share_swing"],
            "label_revision_trigger": threshold_sensitivity["label_revision_trigger"],
        },
        "prediction_audit": {
            "outcome_share": prediction_audit["outcome_share"],
            "confidence_share": prediction_audit["confidence_share"],
            "p_is_correct_given_outcome": prediction_audit["p_is_correct_given_outcome"],
            "correctness_separation_success_minus_uncertain": prediction_audit[
                "correctness_separation_success_minus_uncertain"
            ],
            "certificate_margin": prediction_audit.get("certificate_margin", {}),
            "success_prob_over_guard_rate_by_predicted_outcome": prediction_audit.get(
                "success_prob_over_guard_rate_by_predicted_outcome",
                {},
            ),
            "success_prob_over_guard_excess_by_predicted_outcome": prediction_audit.get(
                "success_prob_over_guard_excess_by_predicted_outcome",
                {},
            ),
            "success_prob_mean_by_predicted_outcome": prediction_audit.get(
                "success_prob_mean_by_predicted_outcome",
                {},
            ),
            "success_guard_mean_by_predicted_outcome": prediction_audit.get(
                "success_guard_mean_by_predicted_outcome",
                {},
            ),
            "predicted_success_over_guard_rate_by_correctness": prediction_audit.get(
                "predicted_success_over_guard_rate_by_correctness",
                {},
            ),
            "predicted_success_over_guard_excess_by_correctness": prediction_audit.get(
                "predicted_success_over_guard_excess_by_correctness",
                {},
            ),
            "predicted_success_count_by_correctness": prediction_audit.get(
                "predicted_success_count_by_correctness",
                {},
            ),
        },
        "collapse_audit": collapse_audit,
        "evidence_audit": evidence_audit,
        "success_modes": _success_mode_summary(),
        "path_scales": _default_path_scales(config),
        "config_overrides": config_overrides,
        "counterfactual_audit": counterfactual_audit,
        "path_audit_summary": path_audit_summary,
        "correctness_baselines": correctness_baselines,
        "monotonicity": monotonicity,
        "acceptance": acceptance,
        "artifacts": {
            "metrics": str(output_dir / "metrics.json"),
            "predictions": str(output_dir / "predictions.jsonl"),
            "summary": str(output_dir / "summary.json"),
        },
    }
    write_jsonl(output_dir / "predictions.jsonl", prediction_rows)
    write_json(output_dir / "metrics.json", metrics_payload)
    write_json(output_dir / "summary.json", summary_payload)


if __name__ == "__main__":
    main()
