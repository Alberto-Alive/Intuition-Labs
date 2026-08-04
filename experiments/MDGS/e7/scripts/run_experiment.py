"""Trace-driven staged training and evaluation for DIGIT Extrapolation E7."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import sys
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torch.amp import GradScaler, autocast
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
    path_scales: dict[str, float] | None = None,
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
        "success_affinity": [],
        "success_trace_strength": [],
        "success_bonus_weight": [],
        "success_bonus_weight_zero_raw": [],
        "success_bonus": [],
        "success_bonus_zero_raw": [],
        "success_proposal": [],
        "success": [],
        "failure": [],
        "ambiguity": [],
        "cert_risk": [],
        "certainty_score": [],
        "fragility_risk": [],
        "fragility_score": [],
        "decisiveness_score": [],
        "prototype_top_support": [],
        "prototype_family_margin": [],
        "prototype_anchor_switch_rate": [],
        "trust_support_deficit": [],
        "trust_approach_deficit": [],
        "trust_leap_cost": [],
        "trust_conflict": [],
        "trust_agreement_deficit": [],
        "trust_margin_deficit": [],
        "trust_concentration_deficit": [],
        "trust_variation_pressure": [],
        "worsening_radius": [],
        "worsening_ambiguity": [],
        "worsening_failure": [],
        "anchor_state": [],
        "anchor_prev_state": [],
        "anchor_transition": [],
        "success_family_support": [],
        "failure_family_support": [],
        "boundary_family_support": [],
        "success_family_approach": [],
        "failure_family_approach": [],
        "boundary_family_approach": [],
        "success_family_leap_penalty": [],
        "failure_family_leap_penalty": [],
        "boundary_family_leap_penalty": [],
        "anchor_conflict": [],
        "commitment_depth": [],
        "top_family_id": [],
        "top_anchor_ids": [],
        "anchor_assignment_probs": [],
        "anchor_support_scores": [],
        "anchor_approach_scores": [],
        "anchor_leap_scores": [],
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
                path_scales=path_scales,
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
                all_evidence["success_affinity"].append(out["primitives"].success_affinity.cpu())
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
                all_evidence["cert_risk"].append(out["primitives"].cert_risk.cpu())
                all_evidence["certainty_score"].append(out["primitives"].certainty_score.cpu())
                all_evidence["fragility_risk"].append(out["primitives"].fragility_risk.cpu())
                all_evidence["fragility_score"].append(out["primitives"].fragility_score.cpu())
                all_evidence["decisiveness_score"].append(out["primitives"].decisiveness_score.cpu())
                all_evidence["prototype_top_support"].append(out["primitives"].prototype_top_support.cpu())
                all_evidence["prototype_family_margin"].append(out["primitives"].prototype_family_margin.cpu())
                all_evidence["prototype_anchor_switch_rate"].append(
                    out["primitives"].prototype_anchor_switch_rate.cpu()
                )
                all_evidence["trust_support_deficit"].append(out["primitives"].trust_support_deficit.cpu())
                all_evidence["trust_approach_deficit"].append(out["primitives"].trust_approach_deficit.cpu())
                all_evidence["trust_leap_cost"].append(out["primitives"].trust_leap_cost.cpu())
                all_evidence["trust_conflict"].append(out["primitives"].trust_conflict.cpu())
                all_evidence["trust_agreement_deficit"].append(out["primitives"].trust_agreement_deficit.cpu())
                all_evidence["trust_margin_deficit"].append(out["primitives"].trust_margin_deficit.cpu())
                all_evidence["trust_concentration_deficit"].append(
                    out["primitives"].trust_concentration_deficit.cpu()
                )
                all_evidence["trust_variation_pressure"].append(out["primitives"].trust_variation_pressure.cpu())
                if hasattr(out["primitives"], "worsening_radius"):
                    all_evidence["worsening_radius"].append(out["primitives"].worsening_radius.cpu())
                if hasattr(out["primitives"], "worsening_ambiguity"):
                    all_evidence["worsening_ambiguity"].append(out["primitives"].worsening_ambiguity.cpu())
                if hasattr(out["primitives"], "worsening_failure"):
                    all_evidence["worsening_failure"].append(out["primitives"].worsening_failure.cpu())
                all_evidence["anchor_state"].append(out["primitives"].anchor_state.cpu())
                all_evidence["anchor_prev_state"].append(out["primitives"].anchor_prev_state.cpu())
                all_evidence["anchor_transition"].append(out["primitives"].anchor_transition.cpu())
                all_evidence["success_family_support"].append(out["primitives"].success_family_support.cpu())
                all_evidence["failure_family_support"].append(out["primitives"].failure_family_support.cpu())
                all_evidence["boundary_family_support"].append(out["primitives"].boundary_family_support.cpu())
                all_evidence["success_family_approach"].append(out["primitives"].success_family_approach.cpu())
                all_evidence["failure_family_approach"].append(out["primitives"].failure_family_approach.cpu())
                all_evidence["boundary_family_approach"].append(out["primitives"].boundary_family_approach.cpu())
                all_evidence["success_family_leap_penalty"].append(out["primitives"].success_family_leap_penalty.cpu())
                all_evidence["failure_family_leap_penalty"].append(out["primitives"].failure_family_leap_penalty.cpu())
                all_evidence["boundary_family_leap_penalty"].append(out["primitives"].boundary_family_leap_penalty.cpu())
                all_evidence["anchor_conflict"].append(out["primitives"].anchor_conflict.cpu())
                all_evidence["commitment_depth"].append(out["primitives"].commitment_depth.cpu())
                all_evidence["top_family_id"].append(out["primitives"].top_family_id.cpu())
                all_evidence["top_anchor_ids"].append(out["primitives"].top_anchor_ids.cpu())
                all_evidence["anchor_assignment_probs"].append(out["primitives"].anchor_assignment_probs.cpu())
                all_evidence["anchor_support_scores"].append(out["primitives"].anchor_support_scores.cpu())
                all_evidence["anchor_approach_scores"].append(out["primitives"].anchor_approach_scores.cpu())
                all_evidence["anchor_leap_scores"].append(out["primitives"].anchor_leap_scores.cpu())

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


def _metric_or_default(
    metrics: dict[str, float | list[list[int]]],
    key: str,
    default: float,
) -> float:
    try:
        value = _metric_value(metrics, key)
    except (KeyError, TypeError):
        return float(default)
    if math.isnan(value):
        return float(default)
    return float(value)


PATH_SCALE_CONFIG_FIELDS = {
    "shared_trace": "trace_path_scale_shared_trace",
    "success_family": "trace_path_scale_success_trace",
    "failure_family": "trace_path_scale_failure_trace",
    "boundary_family": "trace_path_scale_ambiguity_trace",
    "outcome_context": "trace_path_scale_outcome_context",
    "outcome_trajectory_context": "trace_path_scale_outcome_trajectory_context",
    "outcome_pattern_context": "trace_path_scale_outcome_pattern_context",
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


def _parse_config_value(raw_value: str) -> Any:
    lowered = raw_value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        return raw_value


def _parse_config_overrides(values: list[str] | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if not values:
        return overrides
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --config-override value {item!r}; expected NAME=VALUE")
        name, raw_value = item.split("=", 1)
        name = name.strip()
        resolved = CONFIG_OVERRIDE_FIELDS.get(name, name)
        if resolved not in Config.__annotations__:
            raise ValueError(
                f"Unknown config override {name!r}; expected a Config field or alias"
            )
        overrides[resolved] = _parse_config_value(raw_value)
    return overrides


def _apply_config_overrides(config: Config, overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        setattr(config, key, value)


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


def _collapse_audit_from_bundle(
    bundle: dict[str, Any],
    collapse_share_threshold: float,
    min_active_families: int = 0,
    min_family_share: float = 0.0,
    require_success_and_failure: bool = False,
    min_commitment_std: float = 0.0,
) -> dict[str, Any]:
    predictions = bundle["predictions"].cpu()
    evidence = bundle.get("evidence", {})
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
    family_usage_shares = {}
    if "top_family_id" in evidence:
        top_family_ids = evidence["top_family_id"].long().cpu()
        family_usage_shares = {
            family_name: float((top_family_ids == idx).float().mean().item())
            for idx, family_name in enumerate(["success", "failure", "boundary"])
        }
    families_with_nontrivial_usage = int(sum(share >= min_family_share for share in family_usage_shares.values()))
    success_failure_active = (
        family_usage_shares.get("success", 0.0) >= min_family_share
        and family_usage_shares.get("failure", 0.0) >= min_family_share
    )
    commitment_std = (
        float(evidence["commitment_depth"].float().std(unbiased=False).item())
        if "commitment_depth" in evidence and evidence["commitment_depth"].numel() > 0
        else float("nan")
    )
    prediction_collapsed = bool(
        outcome_max_share >= collapse_share_threshold
        or confidence_max_share >= collapse_share_threshold
        or outcome_predicted_classes < 2
        or confidence_predicted_classes < 2
    )
    structure_collapsed = bool(
        (family_usage_shares and families_with_nontrivial_usage < int(min_active_families))
        or (
            family_usage_shares
            and require_success_and_failure
            and not success_failure_active
        )
        or (
            not math.isnan(commitment_std)
            and commitment_std < float(min_commitment_std)
        )
    )

    return {
        "outcome_share": outcome_share,
        "confidence_share": confidence_share,
        "outcome_predicted_classes": int(outcome_predicted_classes),
        "confidence_predicted_classes": int(confidence_predicted_classes),
        "outcome_max_share": float(outcome_max_share),
        "confidence_max_share": float(confidence_max_share),
        "family_usage_shares": family_usage_shares,
        "families_with_nontrivial_usage": int(families_with_nontrivial_usage),
        "success_failure_active": bool(success_failure_active),
        "commitment_std": float(commitment_std),
        "all_success_mode": bool(outcome_share["SUCCESS_LIKELY"] >= collapse_share_threshold),
        "all_uncertain_mode": bool(outcome_share["UNCERTAIN"] >= collapse_share_threshold),
        "all_failure_mode": bool(outcome_share["FAILURE_LIKELY"] >= collapse_share_threshold),
        "single_confidence_mode": bool(confidence_max_share >= collapse_share_threshold or confidence_predicted_classes < 2),
        "prediction_collapsed": bool(prediction_collapsed),
        "structure_collapsed": bool(structure_collapsed),
        "collapsed_model": bool(prediction_collapsed or structure_collapsed),
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
        success_affinity_values = evidence.get("success_affinity", success_base_values).cpu()
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
        payload["correlations"]["success_affinity_vs_success_base"] = _safe_corrcoef(
            success_affinity_values.tolist(),
            success_base_values.tolist(),
        )
        payload["correlations"]["success_affinity_vs_success"] = _safe_corrcoef(
            success_affinity_values.tolist(),
            success_values.tolist(),
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
            "affinity_to_base_mean": float(
                (success_base_values / success_affinity_values.clamp_min(1e-6)).mean().item()
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


def _tensor_summary_with_quantiles(values: torch.Tensor) -> dict[str, Any]:
    series = values.float().reshape(-1).cpu()
    payload = summarize_scalar_series(series.tolist())
    if series.numel() == 0:
        payload.update({"q10": float("nan"), "q50": float("nan"), "q90": float("nan")})
        return payload
    payload.update(
        {
            "q10": float(torch.quantile(series, 0.10).item()),
            "q50": float(torch.quantile(series, 0.50).item()),
            "q90": float(torch.quantile(series, 0.90).item()),
        }
    )
    return payload


def _tensor_summary_by_predicted_outcome(values: torch.Tensor, predicted_outcome: torch.Tensor) -> dict[str, Any]:
    values = values.float().reshape(-1).cpu()
    predicted_outcome = predicted_outcome.long().reshape(-1).cpu()
    return {
        label: _tensor_summary_with_quantiles(values[predicted_outcome == idx])
        if int((predicted_outcome == idx).sum().item()) > 0
        else {}
        for idx, label in enumerate(OUTCOME_LABELS)
    }


def _decision_decomposition_from_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    evidence = bundle.get("evidence", {})
    probs = bundle.get("probs", {})
    predictions = bundle.get("predictions")
    required_keys = [
        "success_base",
        "success_guard",
        "success",
        "failure",
        "ambiguity",
        "commitment_depth",
        "success_family_support",
        "failure_family_support",
        "boundary_family_support",
    ]
    if predictions is None or "outcome" not in probs or any(key not in evidence for key in required_keys):
        return {}

    predicted_outcome = predictions[:, 3].cpu()
    outcome_probs = probs["outcome"].cpu()
    success_base = evidence["success_base"].cpu()
    success_affinity = evidence.get("success_affinity", success_base).cpu()
    success_guard = evidence["success_guard"].cpu()
    success_cert = evidence["success"].cpu()
    failure_cert = evidence["failure"].cpu()
    ambiguity_cert = evidence["ambiguity"].cpu()
    commitment_depth = evidence["commitment_depth"].cpu()
    success_gap = torch.relu(success_base - success_guard)
    success_contract_ratio = torch.where(
        success_base > 1e-6,
        success_guard / success_base.clamp_min(1e-6),
        torch.zeros_like(success_base),
    ).clamp(min=0.0, max=1.5)
    family_supports = torch.stack(
        [
            evidence["success_family_support"].cpu(),
            evidence["failure_family_support"].cpu(),
            evidence["boundary_family_support"].cpu(),
        ],
        dim=1,
    )
    top2_support = torch.topk(family_supports, k=2, dim=1).values
    family_margin = (
        (top2_support[:, 0] - top2_support[:, 1]).clamp_min(0.0)
        / (top2_support[:, 0] + top2_support[:, 1]).clamp_min(1e-6)
    ).clamp(min=0.0, max=1.0)
    runner_up_pressure = (top2_support[:, 1] * (1.0 - family_margin)).clamp(min=0.0, max=1.0)
    failure_success_margin = failure_cert - success_cert
    ambiguity_minus_boundary = ambiguity_cert - evidence["boundary_family_support"].cpu()

    scalar_fields: dict[str, torch.Tensor] = {
        "success_prob": outcome_probs[:, 0],
        "uncertain_prob": outcome_probs[:, 1],
        "failure_prob": outcome_probs[:, 2],
        "success_affinity": success_affinity,
        "success_base": success_base,
        "success_guard": success_guard,
        "affinity_to_base_gap": torch.relu(success_affinity - success_base),
        "success_gap": success_gap,
        "success_contract_ratio": success_contract_ratio,
        "success_cert": success_cert,
        "failure_cert": failure_cert,
        "ambiguity_cert": ambiguity_cert,
        "commitment_depth": commitment_depth,
        "family_margin": family_margin,
        "runner_up_pressure": runner_up_pressure,
        "failure_success_margin": failure_success_margin,
        "ambiguity_minus_boundary": ambiguity_minus_boundary,
        "success_family_support": evidence["success_family_support"].cpu(),
        "failure_family_support": evidence["failure_family_support"].cpu(),
        "boundary_family_support": evidence["boundary_family_support"].cpu(),
    }
    for optional_key in [
        "cert_risk",
        "certainty_score",
        "fragility_risk",
        "fragility_score",
        "decisiveness_score",
        "prototype_top_support",
        "prototype_family_margin",
        "prototype_anchor_switch_rate",
        "worsening_radius",
        "worsening_ambiguity",
        "worsening_failure",
        "trust_support_deficit",
        "trust_approach_deficit",
        "trust_leap_cost",
        "trust_conflict",
        "trust_agreement_deficit",
        "trust_margin_deficit",
        "trust_concentration_deficit",
        "trust_variation_pressure",
    ]:
        if optional_key in evidence:
            scalar_fields[optional_key] = evidence[optional_key].cpu()

    overall = {
        name: _tensor_summary_with_quantiles(values)
        for name, values in scalar_fields.items()
    }
    by_predicted_outcome = {
        label: {
            name: _tensor_summary_with_quantiles(values[predicted_outcome == idx])
            if int((predicted_outcome == idx).sum().item()) > 0
            else {}
            for name, values in scalar_fields.items()
        }
        for idx, label in enumerate(OUTCOME_LABELS)
    }
    return {
        "overall": overall,
        "by_predicted_outcome": by_predicted_outcome,
    }


def _diagnostic_examples_from_bundle(
    records: list[dict[str, Any]],
    bundle: dict[str, Any],
    limit_per_bucket: int = 12,
) -> list[dict[str, Any]]:
    evidence = bundle.get("evidence", {})
    probs = bundle.get("probs", {})
    predictions = bundle.get("predictions")
    targets = bundle.get("targets")
    required_keys = [
        "success_base",
        "success_guard",
        "success",
        "failure",
        "ambiguity",
        "commitment_depth",
        "anchor_conflict",
        "top_family_id",
        "success_family_support",
        "failure_family_support",
        "boundary_family_support",
    ]
    if (
        predictions is None
        or targets is None
        or "outcome" not in probs
        or any(key not in evidence for key in required_keys)
    ):
        return []

    family_names = ["success", "failure", "boundary"]
    outcome_probs = probs["outcome"].cpu()
    predicted_outcome = predictions[:, 3].cpu()
    target_outcome = targets[:, 3].cpu()
    success_base = evidence["success_base"].cpu()
    success_affinity = evidence.get("success_affinity", success_base).cpu()
    success_guard = evidence["success_guard"].cpu()
    success_gap = torch.relu(success_base - success_guard)
    success_cert = evidence["success"].cpu()
    failure_cert = evidence["failure"].cpu()
    ambiguity_cert = evidence["ambiguity"].cpu()
    commitment_depth = evidence["commitment_depth"].cpu()
    anchor_conflict = evidence["anchor_conflict"].cpu()
    top_family_id = evidence["top_family_id"].cpu()
    worsening_radius = evidence.get("worsening_radius", torch.zeros_like(success_cert)).cpu()
    worsening_ambiguity = evidence.get("worsening_ambiguity", torch.zeros_like(success_cert)).cpu()
    worsening_failure = evidence.get("worsening_failure", torch.zeros_like(success_cert)).cpu()
    success_support = evidence["success_family_support"].cpu()
    failure_support = evidence["failure_family_support"].cpu()
    boundary_support = evidence["boundary_family_support"].cpu()

    def build_row(idx: int, category: str, score: float, reason: str) -> dict[str, Any]:
        record = records[idx]
        return {
            "bucket": category,
            "score": float(score),
            "reason": reason,
            "example_id": record.get("example_id", idx),
            "is_correct": bool(record.get("is_correct", False)),
            "predicted_outcome": OUTCOME_LABELS[int(predicted_outcome[idx].item())],
            "target_outcome": OUTCOME_LABELS[int(target_outcome[idx].item())],
            "top_family": family_names[int(top_family_id[idx].item())],
            "outcome_probs": {
                "success": float(outcome_probs[idx, 0].item()),
                "uncertain": float(outcome_probs[idx, 1].item()),
                "failure": float(outcome_probs[idx, 2].item()),
            },
            "success_affinity": float(success_affinity[idx].item()),
            "success_base": float(success_base[idx].item()),
            "success_guard": float(success_guard[idx].item()),
            "success_gap": float(success_gap[idx].item()),
            "success_cert": float(success_cert[idx].item()),
            "failure_cert": float(failure_cert[idx].item()),
            "ambiguity_cert": float(ambiguity_cert[idx].item()),
            "commitment_depth": float(commitment_depth[idx].item()),
            "anchor_conflict": float(anchor_conflict[idx].item()),
            "family_supports": {
                "success": float(success_support[idx].item()),
                "failure": float(failure_support[idx].item()),
                "boundary": float(boundary_support[idx].item()),
            },
            "worsening_radius": float(worsening_radius[idx].item()),
            "worsening_ambiguity": float(worsening_ambiguity[idx].item()),
            "worsening_failure": float(worsening_failure[idx].item()),
        }

    buckets: list[tuple[str, list[int], Any, str]] = [
        (
            "incorrect_predicted_success",
            [
                idx
                for idx in range(len(records))
                if int(predicted_outcome[idx].item()) == 0
                and (not bool(records[idx].get("is_correct", False)) or int(target_outcome[idx].item()) != 0)
            ],
            lambda idx: float(
                outcome_probs[idx, 0].item()
                + success_gap[idx].item()
                + 0.5 * success_cert[idx].item()
            ),
            "Predicted SUCCESS despite incorrect outcome or mismatch.",
        ),
        (
            "washed_out_failure",
            [idx for idx in range(len(records)) if int(predicted_outcome[idx].item()) != 2],
            lambda idx: float(failure_cert[idx].item() - outcome_probs[idx, 2].item()),
            "Failure cert is stronger than the final failure probability.",
        ),
        (
            "washed_out_ambiguity",
            [idx for idx in range(len(records)) if int(predicted_outcome[idx].item()) != 1],
            lambda idx: float(ambiguity_cert[idx].item() - outcome_probs[idx, 1].item()),
            "Ambiguity cert is stronger than the final uncertainty probability.",
        ),
        (
            "large_success_guard_gap",
            list(range(len(records))),
            lambda idx: float(success_gap[idx].item()),
            "Large contraction between success base and success guard.",
        ),
        (
            "unsafe_success_high_conflict",
            [
                idx
                for idx in range(len(records))
                if int(predicted_outcome[idx].item()) == 0 and float(anchor_conflict[idx].item()) >= 0.60
            ],
            lambda idx: float(outcome_probs[idx, 0].item() + anchor_conflict[idx].item()),
            "Predicted SUCCESS while conflict stays high.",
        ),
    ]

    rows: list[dict[str, Any]] = []
    for bucket_name, indices, score_fn, reason in buckets:
        ranked = sorted(indices, key=score_fn, reverse=True)[:limit_per_bucket]
        for idx in ranked:
            score = score_fn(idx)
            if score <= 0.0 and bucket_name not in {"large_success_guard_gap", "unsafe_success_high_conflict"}:
                continue
            rows.append(build_row(idx, bucket_name, score, reason))
    return rows


def _anchor_audit_from_bundle(bundle: dict[str, Any], anchor_domination_threshold: float = 0.95) -> dict[str, Any]:
    evidence = bundle.get("evidence", {})
    if "top_family_id" not in evidence:
        return {}

    top_family_ids = evidence["top_family_id"].long().cpu()
    top_anchor_ids = evidence["top_anchor_ids"].long().cpu()
    anchors_per_family = int(evidence.get("anchor_assignment_probs", torch.empty(0, 0, 0)).shape[-1] or 0)
    num_examples = max(int(top_family_ids.numel()), 1)
    family_names = ["success", "failure", "boundary"]

    family_usage_shares = {
        family_name: float((top_family_ids == idx).float().mean().item())
        for idx, family_name in enumerate(family_names)
    }
    top_anchor_usage_shares = {
        family_name: {
            f"anchor_{anchor_idx}": float((top_anchor_ids[:, family_idx] == anchor_idx).float().mean().item())
            for anchor_idx in range(anchors_per_family)
        }
        for family_idx, family_name in enumerate(family_names)
    }

    family_support = {
        "success": _tensor_summary_with_quantiles(evidence["success_family_support"]),
        "failure": _tensor_summary_with_quantiles(evidence["failure_family_support"]),
        "boundary": _tensor_summary_with_quantiles(evidence["boundary_family_support"]),
    }
    family_approach = {
        "success": _tensor_summary_with_quantiles(evidence["success_family_approach"]),
        "failure": _tensor_summary_with_quantiles(evidence["failure_family_approach"]),
        "boundary": _tensor_summary_with_quantiles(evidence["boundary_family_approach"]),
    }
    family_leap = {
        "success": _tensor_summary_with_quantiles(evidence["success_family_leap_penalty"]),
        "failure": _tensor_summary_with_quantiles(evidence["failure_family_leap_penalty"]),
        "boundary": _tensor_summary_with_quantiles(evidence["boundary_family_leap_penalty"]),
    }
    family_domination = max(family_usage_shares.values()) if family_usage_shares else 0.0
    anchor_domination = 0.0
    for family_usage in top_anchor_usage_shares.values():
        if family_usage:
            anchor_domination = max(anchor_domination, max(family_usage.values()))

    return {
        "num_examples": num_examples,
        "family_usage_shares": family_usage_shares,
        "top_anchor_usage_shares": top_anchor_usage_shares,
        "family_support": family_support,
        "family_approach": family_approach,
        "family_leap_penalty": family_leap,
        "conflict": _tensor_summary_with_quantiles(evidence["anchor_conflict"]),
        "commitment_depth": _tensor_summary_with_quantiles(evidence["commitment_depth"]),
        "collapse_flags": {
            "family_domination": bool(family_domination >= anchor_domination_threshold),
            "anchor_domination": bool(anchor_domination >= anchor_domination_threshold),
            "max_family_share": float(family_domination),
            "max_anchor_share": float(anchor_domination),
            "families_with_nontrivial_usage": int(sum(share >= 0.05 for share in family_usage_shares.values())),
        },
    }


def _anchor_geometry_from_model(model: DIGITModel) -> dict[str, Any]:
    del model
    return {"variant": "diffusion_prototype_e7"}


def _commitment_audit_from_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    evidence = bundle.get("evidence", {})
    probs = bundle.get("probs", {})
    predictions = bundle.get("predictions")
    if "commitment_depth" not in evidence or "outcome" not in probs or predictions is None:
        return {}

    conflict = evidence["anchor_conflict"].cpu()
    commitment = evidence["commitment_depth"].cpu()
    outcome_probs = probs["outcome"].cpu()
    predicted_outcome = predictions[:, 3].cpu()
    predicted_success = predicted_outcome == 0
    high_conflict = conflict >= 0.60

    success_prob = outcome_probs[:, 0]
    unsafe_success = predicted_success & high_conflict
    payload = {
        "predicted_success_high_conflict_rate": float(unsafe_success.float().mean().item()),
        "predicted_success_high_conflict_count": int(unsafe_success.long().sum().item()),
        "high_conflict_rate": float(high_conflict.float().mean().item()),
        "commitment_depth": _tensor_summary_with_quantiles(commitment),
        "success_prob": _tensor_summary_with_quantiles(success_prob),
        "success_prob_when_high_conflict": _tensor_summary_with_quantiles(success_prob[high_conflict])
        if high_conflict.any()
        else {},
        "commitment_depth_by_predicted_outcome": {
            label: _tensor_summary_with_quantiles(commitment[predicted_outcome == idx])
            if (predicted_outcome == idx).any()
            else {}
            for idx, label in enumerate(OUTCOME_LABELS)
        },
    }
    if "success_base" in evidence and "success_guard" in evidence:
        success_gap = torch.relu(evidence["success_base"].cpu() - evidence["success_guard"].cpu())
        payload["success_gap"] = _tensor_summary_with_quantiles(success_gap)
        payload["success_gap_by_predicted_outcome"] = _tensor_summary_by_predicted_outcome(
            success_gap,
            predicted_outcome,
        )
    if "failure" in evidence and "ambiguity" in evidence:
        failure_margin = (evidence["failure"].cpu() - evidence["success"].cpu())
        payload["failure_margin"] = _tensor_summary_with_quantiles(failure_margin)
        payload["failure_margin_by_predicted_outcome"] = _tensor_summary_by_predicted_outcome(
            failure_margin,
            predicted_outcome,
        )
        payload["ambiguity_cert_by_predicted_outcome"] = _tensor_summary_by_predicted_outcome(
            evidence["ambiguity"].cpu(),
            predicted_outcome,
        )
    return payload


def _anchor_examples_from_bundle(records: list[dict[str, Any]], bundle: dict[str, Any], limit: int = 256) -> list[dict[str, Any]]:
    evidence = bundle.get("evidence", {})
    predictions = bundle.get("predictions")
    targets = bundle.get("targets")
    if predictions is None or targets is None or "top_family_id" not in evidence:
        return []
    limit = min(int(limit), len(records), int(predictions.size(0)))
    rows = []
    family_names = ["success", "failure", "boundary"]
    family_supports = torch.stack(
        [
            evidence["success_family_support"].cpu(),
            evidence["failure_family_support"].cpu(),
            evidence["boundary_family_support"].cpu(),
        ],
        dim=1,
    )
    top2_support = torch.topk(family_supports, k=2, dim=1).values
    family_margin = (
        (top2_support[:, 0] - top2_support[:, 1]).clamp_min(0.0)
        / (top2_support[:, 0] + top2_support[:, 1]).clamp_min(1e-6)
    ).clamp(min=0.0, max=1.0)
    runner_up_pressure = (top2_support[:, 1] * (1.0 - family_margin)).clamp(min=0.0, max=1.0)
    outcome_probs = bundle.get("probs", {}).get("outcome")
    if outcome_probs is not None:
        outcome_probs = outcome_probs.cpu()
    for idx in range(limit):
        success_base = float(evidence["success_base"][idx].item()) if "success_base" in evidence else float("nan")
        success_guard = float(evidence["success_guard"][idx].item()) if "success_guard" in evidence else float("nan")
        row = {
            "example_id": records[idx].get("example_id", idx),
            "predicted_outcome": OUTCOME_LABELS[int(predictions[idx, 3].item())],
            "target_outcome": OUTCOME_LABELS[int(targets[idx, 3].item())],
            "predicted_confidence": CONFIDENCE_LABELS[int(predictions[idx, 2].item())],
            "target_confidence": CONFIDENCE_LABELS[int(targets[idx, 2].item())],
            "top_family": family_names[int(evidence["top_family_id"][idx].item())],
            "top_anchor_ids": {
                family_name: int(evidence["top_anchor_ids"][idx, family_idx].item())
                for family_idx, family_name in enumerate(family_names)
            },
            "family_supports": {
                "success": float(evidence["success_family_support"][idx].item()),
                "failure": float(evidence["failure_family_support"][idx].item()),
                "boundary": float(evidence["boundary_family_support"][idx].item()),
            },
            "family_approach": {
                "success": float(evidence["success_family_approach"][idx].item()),
                "failure": float(evidence["failure_family_approach"][idx].item()),
                "boundary": float(evidence["boundary_family_approach"][idx].item()),
            },
            "family_leap_penalty": {
                "success": float(evidence["success_family_leap_penalty"][idx].item()),
                "failure": float(evidence["failure_family_leap_penalty"][idx].item()),
                "boundary": float(evidence["boundary_family_leap_penalty"][idx].item()),
            },
            "anchor_conflict": float(evidence["anchor_conflict"][idx].item()),
            "commitment_depth": float(evidence["commitment_depth"][idx].item()),
            "family_margin": float(family_margin[idx].item()),
            "runner_up_pressure": float(runner_up_pressure[idx].item()),
            "success_base": success_base,
            "success_guard": success_guard,
            "success_gap": float(max(success_base - success_guard, 0.0)),
            "success_cert": float(evidence["success"][idx].item()) if "success" in evidence else float("nan"),
            "failure_cert": float(evidence["failure"][idx].item()) if "failure" in evidence else float("nan"),
            "ambiguity_cert": float(evidence["ambiguity"][idx].item()) if "ambiguity" in evidence else float("nan"),
        }
        if outcome_probs is not None:
            row["outcome_probs"] = {
                "success": float(outcome_probs[idx, 0].item()),
                "uncertain": float(outcome_probs[idx, 1].item()),
                "failure": float(outcome_probs[idx, 2].item()),
            }
        if "worsening_radius" in evidence:
            row["worsening_radius"] = float(evidence["worsening_radius"][idx].item())
            row["worsening_ambiguity"] = float(evidence["worsening_ambiguity"][idx].item())
            row["worsening_failure"] = float(evidence["worsening_failure"][idx].item())
        rows.append(row)
    return rows


def _bundle_with_scalar_evidence(bundle: dict[str, Any]) -> dict[str, Any]:
    evidence = bundle.get("evidence", {})
    scalar_evidence = {
        name: values
        for name, values in evidence.items()
        if torch.is_tensor(values) and values.ndim == 1
    }
    output = dict(bundle)
    output["evidence"] = scalar_evidence
    return output


def _commitment_monotonicity_audit(
    model: DIGITModel,
    loader: DataLoader,
    device: torch.device,
    path_scales: dict[str, float] | None = None,
) -> dict[str, Any]:
    perturbation_kinds = [
        "lower_agreement",
        "lower_margin",
        "higher_entropy",
        "lower_attention_concentration",
        "higher_variation_ratio",
    ]
    total_checks = 0
    commitment_violations = 0
    success_prob_violations = 0
    per_kind: dict[str, dict[str, float]] = {
        kind: {
            "count": 0.0,
            "commitment": 0.0,
            "success_prob": 0.0,
            "commitment_mean_delta_sum": 0.0,
            "success_prob_mean_delta_sum": 0.0,
        }
        for kind in perturbation_kinds
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
                path_scales=path_scales,
            )
            base_commitment = base_out["primitives"].commitment_depth
            base_success_prob = F.softmax(base_out["primitives"].outcome_logits, dim=-1)[:, 0]
            for kind in perturbation_kinds:
                perturbed_out = model(
                    queries,
                    apply_trace_perturbation(trace_inputs, kind),
                    target_ids,
                    bottleneck_mode="hard",
                    tau=1.0,
                    skip_decoder=True,
                    path_scales=path_scales,
                )
                perturbed_commitment = perturbed_out["primitives"].commitment_depth
                perturbed_success_prob = F.softmax(perturbed_out["primitives"].outcome_logits, dim=-1)[:, 0]
                commitment_bad = perturbed_commitment > (base_commitment + 1e-6)
                success_prob_bad = perturbed_success_prob > (base_success_prob + 1e-6)
                count = float(commitment_bad.numel())
                total_checks += int(commitment_bad.numel())
                commitment_violations += int(commitment_bad.long().sum().item())
                success_prob_violations += int(success_prob_bad.long().sum().item())
                per_kind[kind]["count"] += count
                per_kind[kind]["commitment"] += float(commitment_bad.float().sum().item())
                per_kind[kind]["success_prob"] += float(success_prob_bad.float().sum().item())
                per_kind[kind]["commitment_mean_delta_sum"] += float(
                    (perturbed_commitment - base_commitment).sum().item()
                )
                per_kind[kind]["success_prob_mean_delta_sum"] += float(
                    (perturbed_success_prob - base_success_prob).sum().item()
                )
    payload = {
        "num_checks": int(total_checks),
        "commitment_monotonicity_violation_rate": float(commitment_violations / max(total_checks, 1)),
        "success_prob_monotonicity_violation_rate": float(success_prob_violations / max(total_checks, 1)),
        "by_perturbation": {},
    }
    for kind, stats in per_kind.items():
        denom = max(stats["count"], 1.0)
        payload["by_perturbation"][kind] = {
            "commitment_violation_rate": float(stats["commitment"] / denom),
            "success_prob_violation_rate": float(stats["success_prob"] / denom),
            "commitment_mean_delta": float(stats["commitment_mean_delta_sum"] / denom),
            "success_prob_mean_delta": float(stats["success_prob_mean_delta_sum"] / denom),
        }
    if payload["by_perturbation"]:
        worst_commitment = max(
            payload["by_perturbation"].items(),
            key=lambda item: item[1]["commitment_violation_rate"],
        )
        worst_success = max(
            payload["by_perturbation"].items(),
            key=lambda item: item[1]["success_prob_violation_rate"],
        )
        payload["worst_perturbation"] = {
            "commitment": {
                "perturbation": worst_commitment[0],
                "value": float(worst_commitment[1]["commitment_violation_rate"]),
            },
            "success_prob": {
                "perturbation": worst_success[0],
                "value": float(worst_success[1]["success_prob_violation_rate"]),
            },
        }
    return payload


def _monotonicity_offender_examples(
    model: DIGITModel,
    loader: DataLoader,
    device: torch.device,
    records: list[dict[str, Any]],
    path_scales: dict[str, float] | None = None,
    limit: int = 64,
) -> dict[str, Any]:
    perturbation_kinds = [
        "lower_agreement",
        "lower_margin",
        "higher_entropy",
        "lower_attention_concentration",
        "higher_variation_ratio",
    ]
    rows: list[dict[str, Any]] = []
    record_offset = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            queries, trace_inputs, _, prim_targets, target_ids = _move_batch(batch, device)
            batch_size = int(queries.size(0))
            batch_records = records[record_offset : record_offset + batch_size]
            base_out = model(
                queries,
                trace_inputs,
                target_ids,
                bottleneck_mode="hard",
                tau=1.0,
                skip_decoder=True,
                path_scales=path_scales,
            )
            base_probs = F.softmax(base_out["primitives"].outcome_logits, dim=-1).cpu()
            base_success_prob = base_probs[:, 0]
            base_uncertain_prob = base_probs[:, 1]
            base_failure_prob = base_probs[:, 2]
            base_nonfailure_prob = 1.0 - base_failure_prob
            base_commitment = base_out["primitives"].commitment_depth.cpu()
            base_cert_risk = base_out["primitives"].cert_risk.cpu()
            base_certainty_score = base_out["primitives"].certainty_score.cpu()
            base_fragility_risk = base_out["primitives"].fragility_risk.cpu()
            base_fragility_score = base_out["primitives"].fragility_score.cpu()
            base_decisiveness = base_out["primitives"].decisiveness_score.cpu()
            base_prototype_top_support = base_out["primitives"].prototype_top_support.cpu()
            base_prototype_family_margin = base_out["primitives"].prototype_family_margin.cpu()
            base_prototype_switch_rate = base_out["primitives"].prototype_anchor_switch_rate.cpu()
            base_success_guard = base_out["primitives"].success_guard.cpu()
            base_success_base = base_out["primitives"].success_base.cpu()
            base_success_gap = torch.relu(base_success_base - base_success_guard)
            base_success_cert = base_out["primitives"].success_cert.cpu()
            base_failure_cert = base_out["primitives"].failure_cert.cpu()
            base_ambiguity_cert = base_out["primitives"].ambiguity_cert.cpu()
            base_top_family = base_out["primitives"].top_family_id.cpu()

            for kind in perturbation_kinds:
                perturbed_out = model(
                    queries,
                    apply_trace_perturbation(trace_inputs, kind),
                    target_ids,
                    bottleneck_mode="hard",
                    tau=1.0,
                    skip_decoder=True,
                    path_scales=path_scales,
                )
                perturbed_probs = F.softmax(perturbed_out["primitives"].outcome_logits, dim=-1).cpu()
                perturbed_success_prob = perturbed_probs[:, 0]
                perturbed_uncertain_prob = perturbed_probs[:, 1]
                perturbed_failure_prob = perturbed_probs[:, 2]
                perturbed_nonfailure_prob = 1.0 - perturbed_failure_prob
                perturbed_commitment = perturbed_out["primitives"].commitment_depth.cpu()
                perturbed_cert_risk = perturbed_out["primitives"].cert_risk.cpu()
                perturbed_certainty_score = perturbed_out["primitives"].certainty_score.cpu()
                perturbed_fragility_risk = perturbed_out["primitives"].fragility_risk.cpu()
                perturbed_fragility_score = perturbed_out["primitives"].fragility_score.cpu()
                perturbed_decisiveness = perturbed_out["primitives"].decisiveness_score.cpu()
                perturbed_prototype_top_support = perturbed_out["primitives"].prototype_top_support.cpu()
                perturbed_prototype_family_margin = perturbed_out["primitives"].prototype_family_margin.cpu()
                perturbed_prototype_switch_rate = perturbed_out["primitives"].prototype_anchor_switch_rate.cpu()
                perturbed_success_guard = perturbed_out["primitives"].success_guard.cpu()
                perturbed_success_base = perturbed_out["primitives"].success_base.cpu()
                perturbed_success_gap = torch.relu(perturbed_success_base - perturbed_success_guard)
                perturbed_success_cert = perturbed_out["primitives"].success_cert.cpu()
                perturbed_failure_cert = perturbed_out["primitives"].failure_cert.cpu()
                perturbed_ambiguity_cert = perturbed_out["primitives"].ambiguity_cert.cpu()

                score = (
                    torch.relu(perturbed_success_prob - base_success_prob)
                    + torch.relu(perturbed_nonfailure_prob - base_nonfailure_prob)
                    + 0.75 * torch.relu(perturbed_commitment - base_commitment)
                    + 0.50 * torch.relu(perturbed_success_guard - base_success_guard)
                    + 0.50 * torch.relu(perturbed_success_cert - base_success_cert)
                    + 0.75 * torch.relu(base_failure_cert - perturbed_failure_cert)
                    + 0.35 * torch.relu(perturbed_success_gap - base_success_gap)
                    + 0.25 * torch.relu(base_ambiguity_cert - perturbed_ambiguity_cert)
                )
                base_predicted = base_probs.argmax(dim=-1)
                perturbed_predicted = perturbed_probs.argmax(dim=-1)

                for idx in range(batch_size):
                    score_value = float(score[idx].item())
                    if score_value <= 0.0:
                        continue
                    record = batch_records[idx] if idx < len(batch_records) else {}
                    rows.append(
                        {
                            "example_id": record.get("example_id", record_offset + idx),
                            "perturbation": kind,
                            "score": score_value,
                            "is_correct": bool(record.get("is_correct", False)),
                            "target_outcome": OUTCOME_LABELS[int(prim_targets[idx, 3].item())],
                            "base_predicted_outcome": OUTCOME_LABELS[int(base_predicted[idx].item())],
                            "perturbed_predicted_outcome": OUTCOME_LABELS[int(perturbed_predicted[idx].item())],
                            "base_top_family": ["success", "failure", "boundary"][int(base_top_family[idx].item())],
                            "base_success_prob": float(base_success_prob[idx].item()),
                            "perturbed_success_prob": float(perturbed_success_prob[idx].item()),
                            "base_uncertain_prob": float(base_uncertain_prob[idx].item()),
                            "perturbed_uncertain_prob": float(perturbed_uncertain_prob[idx].item()),
                            "base_failure_prob": float(base_failure_prob[idx].item()),
                            "perturbed_failure_prob": float(perturbed_failure_prob[idx].item()),
                            "base_commitment_depth": float(base_commitment[idx].item()),
                            "perturbed_commitment_depth": float(perturbed_commitment[idx].item()),
                            "base_cert_risk": float(base_cert_risk[idx].item()),
                            "perturbed_cert_risk": float(perturbed_cert_risk[idx].item()),
                            "base_certainty_score": float(base_certainty_score[idx].item()),
                            "perturbed_certainty_score": float(perturbed_certainty_score[idx].item()),
                            "base_fragility_risk": float(base_fragility_risk[idx].item()),
                            "perturbed_fragility_risk": float(perturbed_fragility_risk[idx].item()),
                            "base_fragility_score": float(base_fragility_score[idx].item()),
                            "perturbed_fragility_score": float(perturbed_fragility_score[idx].item()),
                            "base_decisiveness_score": float(base_decisiveness[idx].item()),
                            "perturbed_decisiveness_score": float(perturbed_decisiveness[idx].item()),
                            "base_prototype_top_support": float(base_prototype_top_support[idx].item()),
                            "perturbed_prototype_top_support": float(perturbed_prototype_top_support[idx].item()),
                            "base_prototype_family_margin": float(base_prototype_family_margin[idx].item()),
                            "perturbed_prototype_family_margin": float(perturbed_prototype_family_margin[idx].item()),
                            "base_prototype_anchor_switch_rate": float(base_prototype_switch_rate[idx].item()),
                            "perturbed_prototype_anchor_switch_rate": float(perturbed_prototype_switch_rate[idx].item()),
                            "base_success_guard": float(base_success_guard[idx].item()),
                            "perturbed_success_guard": float(perturbed_success_guard[idx].item()),
                            "base_success_base": float(base_success_base[idx].item()),
                            "perturbed_success_base": float(perturbed_success_base[idx].item()),
                            "base_success_gap": float(base_success_gap[idx].item()),
                            "perturbed_success_gap": float(perturbed_success_gap[idx].item()),
                            "base_success_cert": float(base_success_cert[idx].item()),
                            "perturbed_success_cert": float(perturbed_success_cert[idx].item()),
                            "base_failure_cert": float(base_failure_cert[idx].item()),
                            "perturbed_failure_cert": float(perturbed_failure_cert[idx].item()),
                            "base_ambiguity_cert": float(base_ambiguity_cert[idx].item()),
                            "perturbed_ambiguity_cert": float(perturbed_ambiguity_cert[idx].item()),
                            "score_components": {
                                "success_prob_up": float(
                                    torch.relu(perturbed_success_prob[idx] - base_success_prob[idx]).item()
                                ),
                                "nonfailure_up": float(
                                    torch.relu(perturbed_nonfailure_prob[idx] - base_nonfailure_prob[idx]).item()
                                ),
                                "commitment_up": float(
                                    torch.relu(perturbed_commitment[idx] - base_commitment[idx]).item()
                                ),
                                "success_guard_up": float(
                                    torch.relu(perturbed_success_guard[idx] - base_success_guard[idx]).item()
                                ),
                                "success_cert_up": float(
                                    torch.relu(perturbed_success_cert[idx] - base_success_cert[idx]).item()
                                ),
                                "failure_cert_down": float(
                                    torch.relu(base_failure_cert[idx] - perturbed_failure_cert[idx]).item()
                                ),
                                "success_gap_up": float(
                                    torch.relu(perturbed_success_gap[idx] - base_success_gap[idx]).item()
                                ),
                                "ambiguity_cert_down": float(
                                    torch.relu(base_ambiguity_cert[idx] - perturbed_ambiguity_cert[idx]).item()
                                ),
                            },
                        }
                    )
            record_offset += batch_size

    rows.sort(key=lambda row: row["score"], reverse=True)
    top_by_perturbation = {
        kind: [row for row in rows if row["perturbation"] == kind][: min(12, limit)]
        for kind in perturbation_kinds
    }
    return {
        "top_global": rows[:limit],
        "top_by_perturbation": top_by_perturbation,
    }


def _counterfactual_path_audit(
    model: DIGITModel,
    loader: DataLoader,
    device: torch.device,
    path_scales: dict[str, float] | None = None,
) -> dict[str, Any]:
    scenarios = {
        "zero_success_family": {"success_family": 0.0},
        "zero_failure_family": {"failure_family": 0.0},
        "zero_boundary_family": {"boundary_family": 0.0},
        "zero_outcome_context": {"outcome_context": 0.0},
        "zero_outcome_trajectory_context": {"outcome_trajectory_context": 0.0},
        "zero_outcome_pattern_context": {"outcome_pattern_context": 0.0},
        "zero_approach_signal": {"approach": 0.0},
        "zero_leap_signal": {"leap": 0.0},
        "zero_commitment_barrier": {"barrier": 0.0},
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
            "commitment_depth_mae": 0.0,
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
                path_scales=path_scales,
            )
            base_outcome_probs = F.softmax(base_out["primitives"].outcome_logits, dim=-1)
            base_confidence = base_out["primitives"].confidence_logits.argmax(dim=-1)
            base_outcome = base_out["primitives"].outcome_logits.argmax(dim=-1)
            batch_size = float(queries.size(0))

            for scenario_name, scenario_path_scales in scenarios.items():
                cf_out = model(
                    queries,
                    trace_inputs,
                    target_ids,
                    bottleneck_mode="hard",
                    tau=1.0,
                    skip_decoder=True,
                    path_scales=_merge_path_scale_dicts(path_scales, scenario_path_scales),
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
                bucket["commitment_depth_mae"] += float(
                    (cf_out["primitives"].commitment_depth - base_out["primitives"].commitment_depth).abs().sum().item()
                )
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
    single_path_items = dict(counterfactual_audit)
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
    # Heavily penalize collapse in selection
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


def _stage1_candidate_retention_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    metrics = candidate["metrics"]
    collapsed = bool(candidate["collapse_audit"].get("collapsed_model", True))
    return (
        float(int(not collapsed)),
        _metric_or_default(metrics, "outcome_macro_f1", -1e9),
        _metric_or_default(metrics, "failure_recall", -1e9),
        _metric_or_default(metrics, "trajectory_macro_f1", -1e9),
        _metric_or_default(metrics, "confidence_macro_f1", -1e9),
        _metric_or_default(metrics, "joint_acc", -1e9),
        -_metric_or_default(metrics, "success_aurc", 1e9),
    )


def _update_stage1_candidate_bank(
    candidates: list[dict[str, Any]],
    candidate: dict[str, Any],
    *,
    max_candidates: int,
) -> None:
    candidates.append(candidate)
    candidates.sort(key=_stage1_candidate_retention_key, reverse=True)
    del candidates[max_candidates:]


def _select_stage1_candidate(
    candidates: list[dict[str, Any]],
    config: Config,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("No stage-1 candidates available for selection")

    non_collapsed = [
        candidate
        for candidate in candidates
        if not bool(candidate["collapse_audit"].get("collapsed_model", True))
    ]
    pool = non_collapsed or candidates
    if non_collapsed:
        best_outcome = max(_metric_or_default(candidate["metrics"], "outcome_macro_f1", -1e9) for candidate in pool)
        best_failure = max(_metric_or_default(candidate["metrics"], "failure_recall", -1e9) for candidate in pool)
        outcome_floor = best_outcome - float(config.trace_stage1_frontier_outcome_slack)
        failure_floor = best_failure - float(config.trace_stage1_frontier_failure_slack)
        frontier = [
            candidate
            for candidate in pool
            if _metric_or_default(candidate["metrics"], "outcome_macro_f1", -1e9) >= outcome_floor
            and _metric_or_default(candidate["metrics"], "failure_recall", -1e9) >= failure_floor
        ]
        pool = frontier or pool
        return max(
            pool,
            key=lambda candidate: (
                _metric_or_default(candidate["metrics"], "trajectory_macro_f1", -1e9),
                _metric_or_default(candidate["metrics"], "confidence_macro_f1", -1e9),
                -_metric_or_default(candidate["metrics"], "success_aurc", 1e9),
                _metric_or_default(candidate["metrics"], "joint_acc", -1e9),
                _metric_or_default(candidate["metrics"], "outcome_macro_f1", -1e9),
                _metric_or_default(candidate["metrics"], "failure_recall", -1e9),
            ),
        )
    return max(pool, key=_stage1_candidate_retention_key)


def _format_confusion(matrix: list[list[int]], labels: list[str]) -> str:
    rows = []
    for label, row in zip(labels, matrix):
        rows.append(f"{label}: {row}")
    return " | ".join(rows)


def _collect_primitive_predictions(
    model: DIGITModel,
    loader: DataLoader,
    device: torch.device,
    path_scales: dict[str, float] | None = None,
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
                path_scales=path_scales,
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


def _merge_path_scale_dicts(
    base: dict[str, float] | None,
    overrides: dict[str, float] | None,
) -> dict[str, float] | None:
    if base is None and overrides is None:
        return None
    merged: dict[str, float] = {}
    if base:
        merged.update({key: float(value) for key, value in base.items()})
    if overrides:
        merged.update({key: float(value) for key, value in overrides.items()})
    return merged


def _freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _unfreeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True


def _reference_metrics_path(trace_dir: Path) -> Path:
    del trace_dir
    return VARIANT_DIR / "reference_metrics.json"


def _default_output_dir(variant_name: str, seed: int) -> Path:
    return ROOT_DIR / "results" / variant_name / f"seed_{seed}"


def _success_mode_summary() -> dict[str, str]:
    return {
        "epistemic_mode": "anchor_barrier",
        "family_mode": "success_failure_boundary",
        "assignment_mode": "soft_or_hard_configured",
    }


def _default_path_scales(config: Config) -> dict[str, float]:
    return {
        "shared_trace": float(config.trace_path_scale_shared_trace),
        "success_family": float(config.trace_path_scale_success_trace),
        "failure_family": float(config.trace_path_scale_failure_trace),
        "boundary_family": float(config.trace_path_scale_ambiguity_trace),
        "outcome_context": float(config.trace_path_scale_outcome_context),
        "outcome_trajectory_context": float(config.trace_path_scale_outcome_trajectory_context),
        "outcome_pattern_context": float(config.trace_path_scale_outcome_pattern_context),
        "approach": float(1.0 if config.anchor_use_approach else 0.0),
        "leap": float(1.0 if config.anchor_use_leap else 0.0),
        "barrier": float(1.0 if config.anchor_use_barrier else 0.0),
    }


def _runtime_path_scales(config: Config, barrier_active: bool) -> dict[str, float]:
    path_scales = _default_path_scales(config)
    if config.anchor_use_barrier and not barrier_active:
        path_scales["barrier"] = 0.0
    return path_scales


def _barrier_ready_from_anchor_audit(anchor_audit: dict[str, Any], config: Config) -> bool:
    if not anchor_audit:
        return False
    family_shares = anchor_audit.get("family_usage_shares", {})
    if not family_shares:
        return False
    min_share = float(config.trace_barrier_activation_min_family_share)
    min_families = int(config.trace_barrier_activation_min_families)
    max_family_share = max(float(value) for value in family_shares.values())
    families_with_signal = sum(float(value) >= min_share for value in family_shares.values())
    require_success_and_failure = bool(config.trace_barrier_activation_require_success_and_failure)
    success_failure_ready = (
        family_shares.get("success", 0.0) >= min_share
        and family_shares.get("failure", 0.0) >= min_share
    )
    commitment_std = float(anchor_audit.get("commitment_depth", {}).get("std", 0.0))
    return (
        families_with_signal >= min_families
        and max_family_share <= float(config.trace_barrier_activation_max_family_share)
        and commitment_std >= float(config.trace_barrier_activation_min_commitment_std)
        and (not require_success_and_failure or success_failure_ready)
    )


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
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _default_output_dir(variant_name, config.train_loop_seed).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab = Vocabulary()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    use_amp = bool(config.use_amp and device.type == "cuda")
    effective_batch_size = int(config.trace_batch_size_override or config.batch_size)
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
        batch_size=effective_batch_size,
        shuffle=True,
        collate_fn=trace_collate_fn,
        generator=torch.Generator().manual_seed(config.train_loop_seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        collate_fn=trace_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=effective_batch_size,
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
    # Freeze decoder during Stage 1 to prevent weight decay from destroying
    # its parameters while the generation loss is inactive.
    _freeze_module(model.decoder)

    stage1_optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = GradScaler("cuda", enabled=use_amp)

    best_stage1_state = copy.deepcopy(model.state_dict())
    best_stage1_metrics: dict[str, float | list[list[int]]] | None = None
    best_stage1_collapse_audit = {"collapsed_model": True}
    best_stage1_path_scales = _runtime_path_scales(config, barrier_active=False)
    best_stage1_epoch: int | None = None
    stage1_candidate_bank: list[dict[str, Any]] = []
    patience_counter = 0
    epoch_history: list[dict[str, Any]] = []
    barrier_active = False
    barrier_activation_epoch: int | None = None

    print(f"DIGIT Extrapolation {variant_name.upper()} diffusion-prototype run")
    print("Success shape: anchor_barrier")
    if args.experiment_label:
        print(f"Experiment label: {args.experiment_label}")
    print(f"Device: {device}")
    print(f"AMP enabled: {use_amp}")
    print(f"Batch size: {effective_batch_size}")
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
    if config.anchor_use_barrier:
        print(
            "Barrier schedule: "
            f"warmup_epochs={config.trace_barrier_warmup_epochs}, "
            f"activation_min_families={config.trace_barrier_activation_min_families}, "
            f"activation_min_share={config.trace_barrier_activation_min_family_share:.2f}, "
            f"activation_max_share={config.trace_barrier_activation_max_family_share:.2f}, "
            f"activation_min_commitment_std={config.trace_barrier_activation_min_commitment_std:.2f}"
        )
    print(
        "Stage 1: joint primitive training "
        "(retention: non-collapsed -> outcome_macro_f1 -> failure_recall -> trajectory_macro_f1 -> confidence_macro_f1 -> joint_acc; "
        "final selection: frontier(outcome/failure) -> trajectory_macro_f1 -> confidence_macro_f1 -> lower success_aurc -> joint_acc)"
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
        current_stage1_path_scales = _runtime_path_scales(config, barrier_active=barrier_active)

        for batch_idx, batch in enumerate(train_loader):
            queries, trace_inputs, evidence_targets, prim_targets, target_ids = _move_batch(batch, device)

            # Calculate annealed Gumbel-Softmax temperature
            global_step = epoch * len(train_loader) + batch_idx
            tau = max(
                config.gumbel_tau_end,
                config.gumbel_tau_start - (config.gumbel_tau_start - config.gumbel_tau_end) * (global_step / config.gumbel_anneal_steps)
            )

            amp_context = autocast("cuda", enabled=use_amp) if use_amp else nullcontext()
            with amp_context:
                out = model(
                    queries,
                    trace_inputs,
                    target_ids,
                    bottleneck_mode="gumbel",
                    tau=tau,
                    skip_decoder=True,
                    path_scales=current_stage1_path_scales,
                )
                perturbed_primitives = None
                perturbation = None
                aux_losses = None
                if epoch >= warmup_epochs and criterion.config.lambda_evidence_mono != 0.0:
                    perturbation = perturbation_kinds[(epoch + batch_idx) % len(perturbation_kinds)]
                    perturbed_out = model(
                        queries,
                        apply_trace_perturbation(trace_inputs, perturbation),
                        target_ids,
                        bottleneck_mode="gumbel",
                        tau=tau,
                        skip_decoder=True,
                        path_scales=current_stage1_path_scales,
                    )
                    perturbed_primitives = perturbed_out["primitives"]
                    aux_losses = model.bottleneck.compute_perturbation_aux_losses(
                        z_q=out["z_q"],
                        executor_features=out["executor_features"],
                        perturbed_executor_features=perturbed_out["executor_features"],
                        primitives=out["primitives"],
                        perturbed_primitives=perturbed_primitives,
                    )

                losses = criterion(
                    out["decoder_logits"],
                    out["primitives"],
                    target_ids,
                    prim_targets,
                    out["executor_features"],
                    evidence_targets=evidence_targets,
                    perturbed_primitives=perturbed_primitives,
                    perturbation_kind=perturbation,
                    aux_losses=aux_losses,
                )

            if not torch.isfinite(losses["total"]):
                print(f"Warning: non-finite loss at Stage 1 Epoch {epoch + 1}, Batch {batch_idx}; skipping backward")
                continue

            stage1_optimizer.zero_grad()
            if use_amp:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(stage1_optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(stage1_optimizer)
                scaler.update()
            else:
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
            path_scales=current_stage1_path_scales,
        )
        val_collapse_audit = _collapse_audit_from_bundle(
            val_bundle,
            collapse_share_threshold=config.trace_selection_collapse_share_threshold,
            min_active_families=config.trace_selection_min_active_families,
            min_family_share=config.trace_selection_min_family_share,
            require_success_and_failure=config.trace_selection_require_success_and_failure,
            min_commitment_std=config.trace_selection_min_commitment_std,
        )
        val_prediction_audit = _prediction_audit_from_bundle(val_dataset.records, val_bundle)
        val_evidence_audit = _evidence_audit_from_bundle(val_dataset.records, val_bundle)
        val_anchor_audit = _anchor_audit_from_bundle(val_bundle)
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
            f"val_collapsed={val_collapse_audit['collapsed_model']} | "
            f"barrier_active={current_stage1_path_scales['barrier'] > 0.0}"
        )
        barrier_ready = (
            config.anchor_use_barrier
            and not barrier_active
            and (epoch + 1) >= int(config.trace_barrier_warmup_epochs)
            and _barrier_ready_from_anchor_audit(val_anchor_audit, config)
        )
        should_log_val_audit = (
            epoch == 0
            or bool(val_collapse_audit["collapsed_model"])
            or barrier_ready
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
            if val_anchor_audit:
                print(
                    "  Val anchors: "
                    f"usage={val_anchor_audit['family_usage_shares']['success']:.3f}/"
                    f"{val_anchor_audit['family_usage_shares']['failure']:.3f}/"
                    f"{val_anchor_audit['family_usage_shares']['boundary']:.3f} | "
                    f"conflict={val_anchor_audit['conflict']['mean']:.3f} | "
                    f"commitment={val_anchor_audit['commitment_depth']['mean']:.3f} | "
                    f"family_domination={val_anchor_audit['collapse_flags']['family_domination']}"
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

        _update_stage1_candidate_bank(
            stage1_candidate_bank,
            {
                "epoch": int(epoch + 1),
                "metrics": dict(val_metrics),
                "collapse_audit": dict(val_collapse_audit),
                "path_scales": dict(current_stage1_path_scales),
                "state": copy.deepcopy(model.state_dict()),
            },
            max_candidates=int(config.trace_stage1_frontier_size),
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
            best_stage1_path_scales = dict(current_stage1_path_scales)
            best_stage1_epoch = int(epoch + 1)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.trace_stage1_patience:
                print(f"Early stopping stage 1 at epoch {epoch + 1}")
                break

        if barrier_ready:
            barrier_active = True
            barrier_activation_epoch = int(epoch + 2)
            print(
                "Barrier activation scheduled: "
                f"validation family usage opened up after epoch {epoch + 1}; "
                f"barrier turns on at epoch {barrier_activation_epoch}."
            )

        epoch_history.append(
            {
                "stage": "stage1",
                "epoch": int(epoch + 1),
                "train_loss": float(train_loss),
                "val_loss": float(_metric_value(val_metrics, "loss")),
                "val_outcome_acc": float(_metric_value(val_metrics, "outcome_acc")),
                "val_outcome_macro_f1": float(_metric_value(val_metrics, "outcome_macro_f1")),
                "val_failure_recall": float(_metric_value(val_metrics, "failure_recall")),
                "val_confidence_macro_f1": float(_metric_value(val_metrics, "confidence_macro_f1")),
                "val_trajectory_acc": float(_metric_value(val_metrics, "trajectory_acc")),
                "val_trajectory_macro_f1": float(_metric_value(val_metrics, "trajectory_macro_f1")),
                "val_pattern_macro_f1": float(_metric_value(val_metrics, "pattern_macro_f1")),
                "val_joint_acc": float(_metric_value(val_metrics, "joint_acc")),
                "val_success_aurc": float(_metric_or_default(val_metrics, "success_aurc", float("nan"))),
                "val_base_over_guard_rate": float(
                    val_evidence_audit.get("success_guard_alignment", {}).get("base_over_guard_rate", float("nan"))
                ),
                "val_not_collapsed": bool(not val_collapse_audit["collapsed_model"]),
                "barrier_active": bool(current_stage1_path_scales["barrier"] > 0.0),
            }
        )

    if stage1_candidate_bank:
        selected_stage1 = _select_stage1_candidate(stage1_candidate_bank, config)
        best_stage1_state = selected_stage1["state"]
        best_stage1_metrics = selected_stage1["metrics"]
        best_stage1_collapse_audit = selected_stage1["collapse_audit"]
        best_stage1_path_scales = selected_stage1["path_scales"]
        best_stage1_epoch = int(selected_stage1["epoch"])

    model.load_state_dict(best_stage1_state)
    if best_stage1_collapse_audit.get("collapsed_model", False):
        print("Warning: every stage-1 checkpoint was collapsed under the validation audit.")
    if best_stage1_epoch is not None:
        print(f"Selected stage-1 epoch: {best_stage1_epoch}")
    print(f"Best stage-1 path scales: {best_stage1_path_scales}")
    pre_stage2_predictions = _collect_primitive_predictions(
        model,
        test_loader,
        device,
        path_scales=best_stage1_path_scales,
    )
    best_stage2_loss = float("nan")
    primitive_predictions_preserved = True
    if not config.trace_force_stage1_only and config.trace_stage2_epochs > 0:
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
                amp_context = autocast("cuda", enabled=use_amp) if use_amp else nullcontext()

                loss = torch.tensor(float('nan'), device=device)
                with amp_context:
                    logits = _decoder_logits_from_ground_truth(model, queries, prim_targets, target_ids)
                    loss = _generation_loss(logits, target_ids, pad_idx=vocab.pad_idx)

                if not torch.isfinite(loss):
                    print(f"Warning: non-finite generation loss at Stage 2 Epoch {epoch + 1}; skipping batch")
                    continue

                stage2_optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(stage2_optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    scaler.step(stage2_optimizer)
                    scaler.update()
                else:
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
            epoch_history.append(
                {
                    "stage": "stage2",
                    "epoch": int(epoch + 1),
                    "train_gen_loss": float(train_gen_loss),
                    "val_gen_loss": float(val_gen_loss),
                }
            )
            if val_gen_loss < best_stage2_loss:
                best_stage2_loss = val_gen_loss
                best_stage2_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_stage2_state)
        post_stage2_predictions = _collect_primitive_predictions(
            model,
            test_loader,
            device,
            path_scales=best_stage1_path_scales,
        )
        primitive_predictions_preserved = torch.equal(pre_stage2_predictions, post_stage2_predictions)
    else:
        print("Stage 2 skipped")

    test_metrics, test_bundle = _evaluate_stage1(
        model,
        stage1_criterion,
        test_loader,
        device,
        records=test_dataset.records,
        collect_bundle=True,
        path_scales=best_stage1_path_scales,
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
    monotonicity = audit_monotonicity(model, test_loader, device, path_scales=best_stage1_path_scales)
    commitment_monotonicity = _commitment_monotonicity_audit(
        model,
        test_loader,
        device,
        path_scales=best_stage1_path_scales,
    )
    prediction_audit = _prediction_audit_from_bundle(test_dataset.records, test_bundle)
    collapse_audit = _collapse_audit_from_bundle(
        test_bundle,
        collapse_share_threshold=config.trace_selection_collapse_share_threshold,
        min_active_families=config.trace_selection_min_active_families,
        min_family_share=config.trace_selection_min_family_share,
        require_success_and_failure=config.trace_selection_require_success_and_failure,
        min_commitment_std=config.trace_selection_min_commitment_std,
    )
    evidence_audit = _evidence_audit_from_bundle(test_dataset.records, test_bundle)
    decision_decomposition = _decision_decomposition_from_bundle(test_bundle)
    anchor_audit = _anchor_audit_from_bundle(test_bundle)
    anchor_geometry = _anchor_geometry_from_model(model)
    commitment_audit = _commitment_audit_from_bundle(test_bundle)
    diagnostic_examples = _diagnostic_examples_from_bundle(test_dataset.records, test_bundle)
    monotonicity_offenders = _monotonicity_offender_examples(
        model,
        test_loader,
        device,
        test_dataset.records,
        path_scales=best_stage1_path_scales,
    )
    counterfactual_audit = _counterfactual_path_audit(
        model,
        test_loader,
        device,
        path_scales=best_stage1_path_scales,
    )
    path_audit_summary = _path_audit_summary(counterfactual_audit)
    test_metrics["commitment_monotonicity_violation_rate"] = float(
        commitment_monotonicity["commitment_monotonicity_violation_rate"]
    )
    test_metrics["predicted_success_high_conflict_rate"] = float(
        commitment_audit.get("predicted_success_high_conflict_rate", 0.0)
    )
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
        f"success_aurc={_metric_value(test_metrics, 'success_aurc'):.3f}, "
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
    if decision_decomposition:
        overall_decomp = decision_decomposition["overall"]
        by_pred_decomp = decision_decomposition["by_predicted_outcome"]
        def _decomp_mean(label: str, metric: str) -> float:
            return float(by_pred_decomp.get(label, {}).get(metric, {}).get("mean", float("nan")))
        print(
            "Decision decomposition: "
            f"contract_ratio_mean={overall_decomp['success_contract_ratio']['mean']:.3f}, "
            f"success_gap_mean={overall_decomp['success_gap']['mean']:.3f}, "
            f"failure_margin_mean={overall_decomp['failure_success_margin']['mean']:.3f}, "
            f"runner_up_pressure_mean={overall_decomp['runner_up_pressure']['mean']:.3f}, "
            f"ambiguity_minus_boundary_mean={overall_decomp['ambiguity_minus_boundary']['mean']:.3f}, "
            f"worsening_radius_mean={overall_decomp.get('worsening_radius', {}).get('mean', float('nan')):.3f}"
        )
        print(
            "Decision by outcome: "
            f"SUCCESS(contract/gap/fail/amb)="
            f"{_decomp_mean('SUCCESS_LIKELY', 'success_contract_ratio'):.3f}/"
            f"{_decomp_mean('SUCCESS_LIKELY', 'success_gap'):.3f}/"
            f"{_decomp_mean('SUCCESS_LIKELY', 'failure_cert'):.3f}/"
            f"{_decomp_mean('SUCCESS_LIKELY', 'ambiguity_cert'):.3f}, "
            f"UNCERTAIN(contract/gap/fail/amb)="
            f"{_decomp_mean('UNCERTAIN', 'success_contract_ratio'):.3f}/"
            f"{_decomp_mean('UNCERTAIN', 'success_gap'):.3f}/"
            f"{_decomp_mean('UNCERTAIN', 'failure_cert'):.3f}/"
            f"{_decomp_mean('UNCERTAIN', 'ambiguity_cert'):.3f}, "
            f"FAILURE(contract/gap/fail/amb)="
            f"{_decomp_mean('FAILURE_LIKELY', 'success_contract_ratio'):.3f}/"
            f"{_decomp_mean('FAILURE_LIKELY', 'success_gap'):.3f}/"
            f"{_decomp_mean('FAILURE_LIKELY', 'failure_cert'):.3f}/"
            f"{_decomp_mean('FAILURE_LIKELY', 'ambiguity_cert'):.3f}"
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
    if anchor_audit:
        print(
            "Anchor audit: "
            f"usage(success/failure/boundary)={anchor_audit['family_usage_shares']['success']:.3f}/"
            f"{anchor_audit['family_usage_shares']['failure']:.3f}/"
            f"{anchor_audit['family_usage_shares']['boundary']:.3f}, "
            f"conflict_mean={anchor_audit['conflict']['mean']:.3f}, "
            f"commitment_mean={anchor_audit['commitment_depth']['mean']:.3f}, "
            f"anchor_domination={anchor_audit['collapse_flags']['anchor_domination']}, "
            f"family_domination={anchor_audit['collapse_flags']['family_domination']}"
        )
    if commitment_audit:
        print(
            "Commitment audit: "
            f"pred_success_high_conflict_rate={commitment_audit['predicted_success_high_conflict_rate']:.3f}, "
            f"high_conflict_rate={commitment_audit['high_conflict_rate']:.3f}, "
            f"commitment_q50={commitment_audit['commitment_depth']['q50']:.3f}"
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
        "Commitment monotonicity: "
        f"commitment_violation_rate={commitment_monotonicity['commitment_monotonicity_violation_rate']:.3f}, "
        f"success_prob_violation_rate={commitment_monotonicity['success_prob_monotonicity_violation_rate']:.3f}"
    )
    monotonicity_worst = monotonicity.get("worst_perturbation_by_metric", {})
    commitment_worst = commitment_monotonicity.get("worst_perturbation", {})
    if monotonicity_worst or commitment_worst:
        print(
            "Monotonicity drivers: "
            f"success={monotonicity_worst.get('success_violation_rate', {}).get('perturbation', 'n/a')}:"
            f"{monotonicity_worst.get('success_violation_rate', {}).get('value', float('nan')):.3f}, "
            f"nonfailure={monotonicity_worst.get('nonfailure_violation_rate', {}).get('perturbation', 'n/a')}:"
            f"{monotonicity_worst.get('nonfailure_violation_rate', {}).get('value', float('nan')):.3f}, "
            f"guard_gap={monotonicity_worst.get('success_base_guard_gap_violation_rate', {}).get('perturbation', 'n/a')}:"
            f"{monotonicity_worst.get('success_base_guard_gap_violation_rate', {}).get('value', float('nan')):.3f}, "
            f"failure={monotonicity_worst.get('failure_cert_violation_rate', {}).get('perturbation', 'n/a')}:"
            f"{monotonicity_worst.get('failure_cert_violation_rate', {}).get('value', float('nan')):.3f}, "
            f"commitment={commitment_worst.get('commitment', {}).get('perturbation', 'n/a')}:"
            f"{commitment_worst.get('commitment', {}).get('value', float('nan')):.3f}"
        )
    print(
        "Path audit: "
        f"zero_success_family(cert_shift/outcome_flip)={counterfactual_audit['zero_success_family']['success_cert_mae'] + counterfactual_audit['zero_success_family']['failure_cert_mae']:.3f}/{counterfactual_audit['zero_success_family']['outcome_label_flip_rate']:.3f}, "
        f"zero_failure_family(cert_shift/outcome_flip)={counterfactual_audit['zero_failure_family']['success_cert_mae'] + counterfactual_audit['zero_failure_family']['failure_cert_mae']:.3f}/{counterfactual_audit['zero_failure_family']['outcome_label_flip_rate']:.3f}, "
        f"zero_boundary_family(cert_shift/outcome_flip)={counterfactual_audit['zero_boundary_family']['success_cert_mae'] + counterfactual_audit['zero_boundary_family']['failure_cert_mae']:.3f}/{counterfactual_audit['zero_boundary_family']['outcome_label_flip_rate']:.3f}, "
        f"zero_approach_signal(cert_shift/outcome_flip)={counterfactual_audit['zero_approach_signal']['success_cert_mae'] + counterfactual_audit['zero_approach_signal']['failure_cert_mae']:.3f}/{counterfactual_audit['zero_approach_signal']['outcome_label_flip_rate']:.3f}, "
        f"zero_leap_signal(cert_shift/outcome_flip)={counterfactual_audit['zero_leap_signal']['success_cert_mae'] + counterfactual_audit['zero_leap_signal']['failure_cert_mae']:.3f}/{counterfactual_audit['zero_leap_signal']['outcome_label_flip_rate']:.3f}, "
        f"zero_commitment_barrier(cert_shift/outcome_flip)={counterfactual_audit['zero_commitment_barrier']['success_cert_mae'] + counterfactual_audit['zero_commitment_barrier']['failure_cert_mae']:.3f}/{counterfactual_audit['zero_commitment_barrier']['outcome_label_flip_rate']:.3f}"
    )
    outcome_context_breakdown = counterfactual_audit["zero_commitment_barrier"].get("outcome_flip_rate_by_base_label", {})
    if outcome_context_breakdown:
        print(
            "Commitment-barrier flips by base label: "
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
    print(
        "Diagnostic artifacts: "
        f"decision_examples={len(diagnostic_examples)}, "
        f"monotonicity_top_global={len(monotonicity_offenders.get('top_global', []))}"
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
        f"monotonicity_ok={max(monotonicity.get('stability_cert_violation_rate', 0.0), monotonicity.get('support_cert_violation_rate', 0.0), monotonicity.get('success_guard_violation_rate', 0.0), monotonicity.get('success_cert_violation_rate', 0.0), monotonicity.get('failure_cert_violation_rate', 0.0), commitment_monotonicity.get('commitment_monotonicity_violation_rate', 0.0)) < 0.05}"
    )

    query, trace_inputs, _, prim_targets, target_ids = test_dataset[0]
    batch_query = query.unsqueeze(0).to(device)
    batch_trace = trace_inputs.unsqueeze(0).to(device)
    batch_targets = target_ids.unsqueeze(0).to(device)

    out = model(
        batch_query,
        batch_trace,
        batch_targets,
        bottleneck_mode="hard",
        tau=1.0,
        path_scales=best_stage1_path_scales,
    )
    pred = {
        "trajectory_shape": out["primitives"].trajectory_shape_logits.argmax(dim=-1),
        "attention_pattern": out["primitives"].attention_pattern_logits.argmax(dim=-1),
        "confidence": out["primitives"].confidence_logits.argmax(dim=-1),
        "outcome": out["primitives"].outcome_logits.argmax(dim=-1),
    }
    text = vocab.decode(
        model.generate(
            batch_query,
            batch_trace,
            path_scales=best_stage1_path_scales,
        )["token_ids"][0].cpu().tolist()
    )
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
    prediction_bundle = _bundle_with_scalar_evidence(test_bundle)
    prediction_rows = build_prediction_records(
        test_dataset.records,
        prediction_bundle,
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
            commitment_monotonicity.get("commitment_monotonicity_violation_rate", 0.0),
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
        "selected_stage1_epoch": best_stage1_epoch,
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
        "decision_decomposition": decision_decomposition,
        "anchor_audit": anchor_audit,
        "anchor_geometry": anchor_geometry,
        "commitment_audit": commitment_audit,
        "diagnostic_examples": diagnostic_examples,
        "success_modes": _success_mode_summary(),
        "path_scales": best_stage1_path_scales,
        "config_overrides": config_overrides,
        "barrier_activation_epoch": barrier_activation_epoch,
        "counterfactual_audit": counterfactual_audit,
        "anchor_counterfactual_audit": counterfactual_audit,
        "path_audit_summary": path_audit_summary,
        "monotonicity_offenders": monotonicity_offenders,
        "threshold_occupancy": threshold_occupancy,
        "threshold_sensitivity": threshold_sensitivity,
        "baselines": {
            "majority": majority,
            "label_prediction": baseline_results,
            "correctness": correctness_baselines,
        },
        "monotonicity": monotonicity,
        "commitment_monotonicity": commitment_monotonicity,
        "epoch_history": epoch_history,
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
        "selected_stage1_epoch": best_stage1_epoch,
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
        "decision_decomposition": decision_decomposition,
        "anchor_audit": anchor_audit,
        "anchor_geometry": anchor_geometry,
        "commitment_audit": commitment_audit,
        "success_modes": _success_mode_summary(),
        "path_scales": best_stage1_path_scales,
        "config_overrides": config_overrides,
        "barrier_activation_epoch": barrier_activation_epoch,
        "counterfactual_audit": counterfactual_audit,
        "anchor_counterfactual_audit": counterfactual_audit,
        "path_audit_summary": path_audit_summary,
        "correctness_baselines": correctness_baselines,
        "monotonicity": monotonicity,
        "commitment_monotonicity": commitment_monotonicity,
        "diagnostic_examples_preview": diagnostic_examples[: min(8, len(diagnostic_examples))],
        "monotonicity_offenders_preview": monotonicity_offenders.get("top_global", [])[:8],
        "epoch_history": {
            "num_epochs": len(epoch_history),
            "history": epoch_history,
        },
        "acceptance": acceptance,
        "artifacts": {
            "metrics": str(output_dir / "metrics.json"),
            "epoch_metrics": str(output_dir / "epoch_metrics.jsonl"),
            "predictions": str(output_dir / "predictions.jsonl"),
            "anchor_examples": str(output_dir / "anchor_examples.jsonl"),
            "decision_examples": str(output_dir / "decision_examples.jsonl"),
            "monotonicity_offenders": str(output_dir / "monotonicity_offenders.json"),
            "summary": str(output_dir / "summary.json"),
        },
    }
    write_jsonl(output_dir / "predictions.jsonl", prediction_rows)
    write_jsonl(output_dir / "epoch_metrics.jsonl", epoch_history)
    write_jsonl(output_dir / "anchor_examples.jsonl", _anchor_examples_from_bundle(test_dataset.records, test_bundle))
    write_jsonl(output_dir / "decision_examples.jsonl", diagnostic_examples)
    write_json(output_dir / "monotonicity_offenders.json", monotonicity_offenders)
    write_json(output_dir / "metrics.json", metrics_payload)
    write_json(output_dir / "summary.json", summary_payload)


if __name__ == "__main__":
    main()
