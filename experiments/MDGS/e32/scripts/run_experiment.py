"""Train and evaluate the E32 cooperative path/witness uncertainty experiment."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import asdict, fields
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[5]
ROOT_DIR = Path(__file__).resolve().parents[2]
VARIANT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(VARIANT_DIR))

from experiments.DIGIT.Extrapolation.e32.extrapolation.config import E31Config
from experiments.DIGIT.Extrapolation.e32.extrapolation.data.ground_truth import OUTCOME_LABELS
from experiments.DIGIT.Extrapolation.e32.extrapolation.data.trace_dataset import (
    TraceWitnessDataset,
    load_trace_records,
    trace_collate_fn,
)
from experiments.DIGIT.Extrapolation.e32.extrapolation.data.vocabulary import Vocabulary
from experiments.DIGIT.Extrapolation.e32.extrapolation.losses import E31Loss, build_risk_targets
from experiments.DIGIT.Extrapolation.e32.extrapolation.metrics import (
    build_stratified_masks,
    compute_binary_probability_metrics,
    compute_outcome_metrics,
    summarize_metric_bundle_with_strata,
    summarize_tensor,
)
from experiments.DIGIT.Extrapolation.e32.extrapolation.trace_pipeline import (
    ensure_trace_corpus,
    evaluate_trace_baselines,
    set_global_determinism,
)
from experiments.DIGIT.Extrapolation.e32.extrapolation.models.digit import E31Model
from experiments.DIGIT.Extrapolation.e32.extrapolation.models.readout import (
    bucketize_commitment_decisions,
    bucketize_outcome_logits,
    summarize_raw_witness_geometry,
)

OUTCOME_INDEX = {label: idx for idx, label in enumerate(OUTCOME_LABELS)}
SUCCESS_CLASS_IDX = OUTCOME_INDEX["SUCCESS_LIKELY"]
UNCERTAIN_CLASS_IDX = OUTCOME_INDEX["UNCERTAIN"]
FAILURE_CLASS_IDX = OUTCOME_INDEX["FAILURE_LIKELY"]


def _move_batch(batch, device: torch.device):
    queries, trace_inputs, evidence_targets, prim_targets, target_ids = batch
    del target_ids
    return (
        queries.to(device),
        trace_inputs.to(device),
        evidence_targets.to(device),
        prim_targets.to(device),
    )


def _limit_records(records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None or limit <= 0 or len(records) <= limit:
        return records
    return records[:limit]


def _lazy_module_status(module: torch.nn.Module) -> dict[str, list[str]]:
    status = {
        "uninitialized_parameters": [],
        "uninitialized_buffers": [],
    }
    for name, parameter in module.named_parameters(recurse=True):
        if parameter.__class__.__name__.startswith("Uninitialized"):
            status["uninitialized_parameters"].append(name)
    for name, buffer in module.named_buffers(recurse=True):
        if buffer.__class__.__name__.startswith("Uninitialized"):
            status["uninitialized_buffers"].append(name)
    return status


def _state_dict_lazy_status(state_dict: dict[str, Any]) -> dict[str, list[str]]:
    status = {
        "uninitialized_parameters": [],
        "uninitialized_buffers": [],
    }
    for name, value in state_dict.items():
        class_name = value.__class__.__name__
        if class_name == "UninitializedParameter":
            status["uninitialized_parameters"].append(name)
        elif class_name == "UninitializedBuffer":
            status["uninitialized_buffers"].append(name)
    return status


def _lazy_status_message(status: dict[str, list[str]]) -> str:
    return (
        f"uninitialized_parameters={status['uninitialized_parameters']} "
        f"uninitialized_buffers={status['uninitialized_buffers']}"
    )


def _config_field_names() -> set[str]:
    return {field.name for field in fields(E31Config)}


def _parse_value(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _apply_config_overrides(config: E31Config, overrides: list[str] | None) -> None:
    if not overrides:
        return
    valid_fields = _config_field_names()
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override {item!r}; expected NAME=VALUE")
        name, raw_value = item.split("=", 1)
        if name not in valid_fields:
            raise ValueError(f"Unknown config field {name!r}")
        setattr(config, name, _parse_value(raw_value))


def _resolve_trace_dir(config: E31Config) -> Path:
    trace_dir = Path(config.trace_dir)
    if trace_dir.is_absolute():
        return trace_dir
    return (VARIANT_DIR / trace_dir).resolve()


def _resolve_output_dir(config: E31Config, seed: int, explicit_output_dir: str | None) -> Path:
    if explicit_output_dir:
        output_dir = Path(explicit_output_dir)
        if output_dir.is_absolute():
            return output_dir
        if len(output_dir.parts) == 1:
            return (VARIANT_DIR / config.output_root / output_dir).resolve()
        repo_relative_prefix = ("experiments", "DIGIT", "Extrapolation", "e32")
        if output_dir.parts[: len(repo_relative_prefix)] == repo_relative_prefix:
            return (REPO_ROOT / output_dir).resolve()
        return (VARIANT_DIR / output_dir).resolve()

    output_root = Path(config.output_root)
    if output_root.is_absolute():
        return output_root / f"seed_{seed}"
    return (VARIANT_DIR / output_root / f"seed_{seed}").resolve()


def _rowwise_pearson_correlation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError("left and right must have matching shapes")
    left = left.float()
    right = right.float()
    left_centered = left - left.mean(dim=-1, keepdim=True)
    right_centered = right - right.mean(dim=-1, keepdim=True)
    numerator = (left_centered * right_centered).sum(dim=-1)
    denominator = torch.sqrt(
        left_centered.pow(2).sum(dim=-1) * right_centered.pow(2).sum(dim=-1)
    )
    return torch.where(denominator > 0, numerator / denominator, torch.zeros_like(numerator))


def _vector_pearson_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 or right.numel() == 0:
        return float("nan")
    left = left.detach().float().reshape(-1)
    right = right.detach().float().reshape(-1)
    if left.shape != right.shape:
        raise ValueError("left and right must have matching flattened shapes")
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = torch.sqrt(left_centered.pow(2).sum() * right_centered.pow(2).sum())
    if float(denominator.item()) <= 0.0:
        return float("nan")
    return float(((left_centered * right_centered).sum() / denominator).item())


def _masked_vector_pearson_correlation(
    left: torch.Tensor,
    right: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    mask = mask.detach().reshape(-1).to(device=left.device, dtype=torch.bool)
    if int(mask.sum().item()) < 2:
        return float("nan")
    return _vector_pearson_correlation(left.reshape(-1)[mask], right.reshape(-1)[mask])


def _risk_targets_from_labels(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    config: E31Config,
) -> torch.Tensor:
    mode = str(getattr(config, "risk_target_mode", "incorrect_only")).strip().lower()
    return build_risk_targets(labels, predictions, mode)


def _threshold_distance_to_bounds(value: float, lower: float, upper: float) -> float:
    if value < lower:
        return float(lower - value)
    if value > upper:
        return float(value - upper)
    return 0.0


def _disturbance_stage(epoch: int, total_epochs: int, config: E31Config) -> str:
    if not getattr(config, "enable_disturbance_profile", False):
        return "base"
    total_epochs = max(int(total_epochs), 1)
    base_epochs = max(1, total_epochs // 3)
    profile_epochs = max(1, total_epochs // 3)
    if epoch <= base_epochs:
        return "base"
    if epoch <= base_epochs + profile_epochs:
        return "profile"
    return "finetune"


def _prediction_metric_summary(
    logits: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    *,
    uncertainty: torch.Tensor,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = compute_outcome_metrics(
        logits,
        targets,
        predictions=predictions,
        uncertainty=uncertainty,
        records=records,
    )
    keep_keys = (
        "outcome_acc",
        "outcome_macro_f1",
        "outcome_confusion",
        "success_recall",
        "failure_recall",
        "class_shares",
        "uncertainty_error_corr",
        "unsafe_success_rate_label",
        "unsafe_success_rate_correctness",
        "success_precision_vs_is_correct",
        "success_label_auroc",
        "uncertain_recall",
        "failure_incorrect_auroc",
    )
    return {key: metrics.get(key) for key in keep_keys}


def _build_signal_forensics(
    signals: dict[str, torch.Tensor],
    *,
    targets: torch.Tensor,
    predictions: torch.Tensor,
) -> dict[str, Any]:
    incorrect = (predictions != targets).float()
    unsafe_success = (predictions == SUCCESS_CLASS_IDX) & (targets != SUCCESS_CLASS_IDX)
    masks = {
        "all": torch.ones_like(targets, dtype=torch.bool),
        "target_success_or_failure": targets != UNCERTAIN_CLASS_IDX,
        "target_uncertain": targets == UNCERTAIN_CLASS_IDX,
        "predicted_success": predictions == SUCCESS_CLASS_IDX,
        "predicted_uncertain": predictions == UNCERTAIN_CLASS_IDX,
        "unsafe_success_label": unsafe_success,
    }
    unsafe_indicator = unsafe_success.float()
    forensics: dict[str, Any] = {}
    for name, values in signals.items():
        flat = values.detach().float().reshape(-1)
        by_slice = {}
        for slice_name, mask in masks.items():
            mask = mask.reshape(-1).to(device=flat.device, dtype=torch.bool)
            by_slice[slice_name] = {
                "count": int(mask.sum().item()),
                "mean": float(flat[mask].mean().item()) if int(mask.sum().item()) else float("nan"),
                "error_corr": _masked_vector_pearson_correlation(flat, incorrect, mask),
            }
        by_slice["unsafe_success_indicator_corr"] = _vector_pearson_correlation(
            flat,
            unsafe_indicator,
        )
        forensics[name] = by_slice
    return forensics


def _primary_metric_value(metrics: dict[str, Any], metric_name: str) -> float:
    if metric_name == "macro_f1":
        return float(metrics["outcome_macro_f1"])
    if metric_name == "failure_recall":
        return float(metrics["failure_recall"])
    if metric_name == "balanced_margin_primary":
        outcome_share = metrics["outcome_share"]
        collapse = metrics["collapse_diagnostics"]
        margin_diagnostics = metrics.get("margin_channel_diagnostics", {})
        margin_positive_fraction = (
            margin_diagnostics.get("margin_positive_fraction", {})
            .get("overall", {})
            .get("mean", 0.0)
        )
        success_share = float(outcome_share["SUCCESS_LIKELY"])
        failure_share = float(outcome_share["FAILURE_LIKELY"])
        uncertain_share = float(outcome_share["UNCERTAIN"])
        dominant_share = float(collapse["dominant_class_share"])
        return float(
            metrics["outcome_macro_f1"]
            + 0.25 * metrics["failure_recall"]
            + 0.15 * metrics["success_recall"]
            - max(0.0, 0.08 - success_share)
            - max(0.0, 0.08 - failure_share)
            - max(0.0, uncertain_share - 0.70)
            - 0.5 * max(0.0, 0.15 - uncertain_share)
            - 0.5 * max(0.0, dominant_share - 0.75)
            - 0.5 * max(0.0, 0.10 - float(margin_positive_fraction))
        )
    if metric_name == "balanced_commitment_primary":
        outcome_share = metrics["outcome_share"]
        collapse = metrics["collapse_diagnostics"]
        failure_share = float(outcome_share["FAILURE_LIKELY"])
        uncertain_share = float(outcome_share["UNCERTAIN"])
        dominant_share = float(collapse["dominant_class_share"])
        return float(
            metrics["outcome_macro_f1"]
            + 0.35 * metrics["failure_recall"]
            - max(0.0, failure_share - 0.35)
            - max(0.0, uncertain_share - 0.65)
            - 0.5 * max(0.0, dominant_share - 0.75)
        )
    raise ValueError(f"Unknown primary metric {metric_name!r}")


def _build_report_highlights(test_metrics: dict[str, Any]) -> dict[str, Any]:
    def _summary_path(section: dict[str, Any] | None, *path: str) -> Any:
        current: Any = section
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    def _maybe_section(name: str) -> dict[str, Any] | None:
        section = test_metrics.get(name)
        return section if isinstance(section, dict) else None

    pre_reader = _maybe_section("pre_reader_witness_geometry")
    post_reader = _maybe_section("post_reader_witness_geometry")
    diversity_retention = _maybe_section("diversity_retention")
    decision_geometry = _maybe_section("decision_geometry")
    margin_channel_diagnostics = _maybe_section("margin_channel_diagnostics")
    failure_channel_diagnostics = _maybe_section("failure_channel_diagnostics")
    success_channel_diagnostics = _maybe_section("success_channel_diagnostics")
    weighting_diagnostics = _maybe_section("weighting_diagnostics")
    path_uncertainty_diagnostics = _maybe_section("path_uncertainty_diagnostics")
    path_signed_evidence_diagnostics = _maybe_section("path_signed_evidence_diagnostics")
    decision_delta_diagnostics = _maybe_section("decision_delta_diagnostics")
    decision_comparison = _maybe_section("decision_comparison")
    disturbance_profile = _maybe_section("disturbance_profile")
    risk_gate_threshold_sweep = _maybe_section("risk_gate_threshold_sweep")
    success_recovery_diagnostics = _build_success_recovery_diagnostics(test_metrics)

    return {
        "prediction_mode": test_metrics.get("prediction_mode"),
        "pre_reader_pairwise_agreement_mean": _summary_path(
            pre_reader, "pairwise_cosine_agreement", "overall", "mean"
        ),
        "post_reader_pairwise_agreement_mean": _summary_path(
            post_reader, "pairwise_cosine_agreement", "overall", "mean"
        ),
        "reader_agreement_gain_mean": _summary_path(
            diversity_retention, "reader_agreement_gain", "overall", "mean"
        ),
        "pre_reader_witness_latent_variance_mean": _summary_path(
            pre_reader, "witness_latent_variance", "overall", "mean"
        ),
        "post_reader_witness_margin_std_mean": _summary_path(
            post_reader, "witness_margin_std", "overall", "mean"
        ),
        "witness_weight_entropy_mean": _summary_path(
            post_reader, "witness_weight_entropy", "overall", "mean"
        ),
        "witness_weight_kl_uniform_mean": _summary_path(
            post_reader, "witness_weight_kl_uniform", "overall", "mean"
        ),
        "risk_auroc": _summary_path(disturbance_profile, "risk_auroc"),
        "abstention_rate": _summary_path(disturbance_profile, "abstention_rate"),
        "committed_accuracy": _summary_path(disturbance_profile, "committed_accuracy"),
        "sign_flip_rate_mean": _summary_path(disturbance_profile, "sign_flip_rate_mean"),
        "margin_variance_mean": _summary_path(disturbance_profile, "margin_variance_mean"),
        "boundary_flip_distance_mean": _summary_path(
            disturbance_profile, "boundary_flip_distance_mean"
        ),
        "profile_feature_means": _summary_path(disturbance_profile, "profile_feature_means"),
        "uncertainty_axis_score_means": _summary_path(
            disturbance_profile, "axis_score_means"
        ),
        "uncertainty_axis_error_corr": _summary_path(
            disturbance_profile, "axis_score_error_corr"
        ),
        "profile_support_memory_size": _summary_path(
            disturbance_profile, "profile_state", "support_memory_size"
        ),
        "profile_normalizer_count": _summary_path(
            disturbance_profile, "profile_state", "profile_normalizer_count"
        ),
        "profile_risk_input_mode": _summary_path(
            disturbance_profile, "profile_state", "profile_risk_input_mode"
        ),
        "class_shares": test_metrics["class_shares"],
        "failure_recall": test_metrics["failure_recall"],
        "success_recall": test_metrics.get("success_recall"),
        "uncertain_recall": test_metrics.get("uncertain_recall"),
        "uncertainty_error_corr": test_metrics.get("uncertainty_error_corr"),
        "path_disagreement_error_corr": test_metrics.get("path_disagreement_error_corr"),
        "endpoint_path_conflict_error_corr": test_metrics.get("endpoint_path_conflict_error_corr"),
        "path_signed_margin_error_corr": test_metrics.get("path_signed_margin_error_corr"),
        "path_failure_logit_true_failure_corr": test_metrics.get(
            "path_failure_logit_true_failure_corr"
        ),
        "path_failure_probability_true_failure_corr": test_metrics.get(
            "path_failure_probability_true_failure_corr"
        ),
        "base_hard_failure_recall": _summary_path(
            decision_comparison, "base_hard_gate", "failure_recall"
        ),
        "final_hard_failure_recall": _summary_path(
            decision_comparison, "final_hard_gate", "failure_recall"
        ),
        "base_hard_success_recall": _summary_path(
            decision_comparison, "base_hard_gate", "success_recall"
        ),
        "final_hard_success_recall": _summary_path(
            decision_comparison, "final_hard_gate", "success_recall"
        ),
        "base_soft_success_recall": _summary_path(
            decision_comparison, "base_soft_logits", "success_recall"
        ),
        "final_soft_success_recall": _summary_path(
            decision_comparison, "final_soft_logits", "success_recall"
        ),
        "path_only_failure_recall": _summary_path(
            decision_comparison, "path_only_logits", "failure_recall"
        ),
        "path_only_macro_f1": _summary_path(
            decision_comparison, "path_only_logits", "outcome_macro_f1"
        ),
        "path_disagreement_mean": _summary_path(
            path_uncertainty_diagnostics, "path_disagreement", "overall", "mean"
        ),
        "path_confidence_mean": _summary_path(
            path_uncertainty_diagnostics, "path_confidence", "overall", "mean"
        ),
        "method_conflict_mean": _summary_path(
            path_uncertainty_diagnostics, "method_conflict", "overall", "mean"
        ),
        "path_signed_margin_mean": _summary_path(
            path_signed_evidence_diagnostics, "path_signed_margin", "overall", "mean"
        ),
        "path_signed_margin_by_true_class": _summary_path(
            path_signed_evidence_diagnostics, "path_signed_margin", "by_true_class"
        ),
        "path_failure_gate_logit_true_failure_corr": test_metrics.get(
            "path_failure_gate_logit_true_failure_corr"
        ),
        "path_failure_gate_probability_true_failure_corr": test_metrics.get(
            "path_failure_gate_probability_true_failure_corr"
        ),
        "path_failure_gate_signal_true_failure_corr": test_metrics.get(
            "path_failure_gate_signal_true_failure_corr"
        ),
        "path_failure_gate_signal_mean": _summary_path(
            path_signed_evidence_diagnostics, "path_failure_gate_signal", "overall", "mean"
        ),
        "path_failure_gate_logit_by_true_class": _summary_path(
            path_signed_evidence_diagnostics, "path_failure_gate_logit", "by_true_class"
        ),
        "path_failure_gate_probability_by_true_class": _summary_path(
            path_signed_evidence_diagnostics, "path_failure_gate_probability", "by_true_class"
        ),
        "path_failure_gate_signal_by_true_class": _summary_path(
            path_signed_evidence_diagnostics, "path_failure_gate_signal", "by_true_class"
        ),
        "path_failure_probability_by_true_class": _summary_path(
            path_signed_evidence_diagnostics, "path_failure_probability", "by_true_class"
        ),
        "path_only_predicts_failure_mean": _summary_path(
            path_signed_evidence_diagnostics, "path_only_predicts_failure", "overall", "mean"
        ),
        "path_margin_predicts_failure_mean": _summary_path(
            path_signed_evidence_diagnostics, "path_margin_predicts_failure", "overall", "mean"
        ),
        "base_to_final_hard_flip_mean": _summary_path(
            decision_delta_diagnostics, "base_to_final_hard_flip", "overall", "mean"
        ),
        "base_to_final_margin_sign_flip_mean": _summary_path(
            decision_delta_diagnostics, "base_to_final_margin_sign_flip", "overall", "mean"
        ),
        "base_to_final_margin_delta_mean": _summary_path(
            decision_delta_diagnostics, "base_to_final_margin_delta", "overall", "mean"
        ),
        "commitment_score_overall_mean": _summary_path(
            decision_geometry, "commitment_score", "overall", "mean"
        ),
        "commitment_score_by_predicted_class": _summary_path(
            decision_geometry, "commitment_score", "by_predicted_class"
        ),
        "commitment_score_by_correctness": _summary_path(
            decision_geometry, "commitment_score", "by_correctness"
        ),
        "margin_positive_fraction_mean": _summary_path(
            margin_channel_diagnostics, "margin_positive_fraction", "overall", "mean"
        ),
        "margin_positive_fraction_by_true_class": _summary_path(
            margin_channel_diagnostics, "margin_positive_fraction", "by_true_class"
        ),
        "margin_negative_fraction_mean": _summary_path(
            margin_channel_diagnostics, "margin_negative_fraction", "overall", "mean"
        ),
        "margin_near_zero_fraction_mean": _summary_path(
            margin_channel_diagnostics, "margin_near_zero_fraction", "overall", "mean"
        ),
        "witness_positive_count_mean": _summary_path(
            margin_channel_diagnostics, "witness_positive_count", "overall", "mean"
        ),
        "witness_negative_count_mean": _summary_path(
            margin_channel_diagnostics, "witness_negative_count", "overall", "mean"
        ),
        "true_failure_margin_negative_fraction_mean": _summary_path(
            failure_channel_diagnostics, "true_failure_margin_negative_fraction", "overall", "mean"
        ),
        "true_failure_margin_positive_fraction_mean": _summary_path(
            failure_channel_diagnostics, "true_failure_margin_positive_fraction", "overall", "mean"
        ),
        "true_failure_margin_near_zero_fraction_mean": _summary_path(
            failure_channel_diagnostics, "true_failure_margin_near_zero_fraction", "overall", "mean"
        ),
        "true_failure_commitment_below_tau_fraction_mean": _summary_path(
            failure_channel_diagnostics,
            "true_failure_commitment_below_tau_fraction",
            "overall",
            "mean",
        ),
        "true_failure_predicted_success_fraction_mean": _summary_path(
            failure_channel_diagnostics, "true_failure_predicted_success_fraction", "overall", "mean"
        ),
        "true_failure_predicted_uncertain_fraction_mean": _summary_path(
            failure_channel_diagnostics, "true_failure_predicted_uncertain_fraction", "overall", "mean"
        ),
        "true_failure_predicted_failure_fraction_mean": _summary_path(
            failure_channel_diagnostics, "true_failure_predicted_failure_fraction", "overall", "mean"
        ),
        "true_success_margin_negative_fraction_mean": _summary_path(
            success_channel_diagnostics, "true_success_margin_negative_fraction", "overall", "mean"
        ),
        "true_success_margin_positive_fraction_mean": _summary_path(
            success_channel_diagnostics, "true_success_margin_positive_fraction", "overall", "mean"
        ),
        "true_success_margin_near_zero_fraction_mean": _summary_path(
            success_channel_diagnostics, "true_success_margin_near_zero_fraction", "overall", "mean"
        ),
        "true_success_commitment_below_tau_fraction_mean": _summary_path(
            success_channel_diagnostics,
            "true_success_commitment_below_tau_fraction",
            "overall",
            "mean",
        ),
        "true_success_predicted_success_fraction_mean": _summary_path(
            success_channel_diagnostics, "true_success_predicted_success_fraction", "overall", "mean"
        ),
        "true_success_predicted_uncertain_fraction_mean": _summary_path(
            success_channel_diagnostics, "true_success_predicted_uncertain_fraction", "overall", "mean"
        ),
        "true_success_predicted_failure_fraction_mean": _summary_path(
            success_channel_diagnostics, "true_success_predicted_failure_fraction", "overall", "mean"
        ),
        "success_recovery_diagnostics": success_recovery_diagnostics,
        "risk_gate_selected_threshold": _summary_path(
            risk_gate_threshold_sweep, "selected_threshold"
        ),
        "risk_gate_selection_reason": _summary_path(
            risk_gate_threshold_sweep, "selection_reason"
        ),
        "risk_gate_target_commit_rate_min": _summary_path(
            risk_gate_threshold_sweep, "target_commit_rate_min"
        ),
        "risk_gate_target_commit_rate_max": _summary_path(
            risk_gate_threshold_sweep, "target_commit_rate_max"
        ),
        "risk_gate_selected_commit_rate": _summary_path(
            risk_gate_threshold_sweep, "selected_operating_point", "commit_rate"
        ),
        "risk_gate_selected_macro_f1": _summary_path(
            risk_gate_threshold_sweep, "selected_operating_point", "outcome_macro_f1"
        ),
        "risk_gate_operating_threshold": _summary_path(
            risk_gate_threshold_sweep, "operating_threshold"
        ),
        "weighted_margin_sign_flip_rate": _summary_path(
            margin_channel_diagnostics, "weighted_margin_sign_flip_rate", "overall", "mean"
        ),
        "weighted_minus_unweighted_margin_mean": _summary_path(
            margin_channel_diagnostics, "weighted_minus_unweighted_margin", "overall", "mean"
        ),
        "coherence_margin_corr_mean": _summary_path(
            weighting_diagnostics, "coherence_margin_corr", "overall", "mean"
        ),
        "coherence_abs_margin_corr_mean": _summary_path(
            weighting_diagnostics, "coherence_abs_margin_corr", "overall", "mean"
        ),
    }


def _summary_diagnostic_sections(test_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "pre_reader_witness_geometry": test_metrics.get("pre_reader_witness_geometry"),
        "post_reader_witness_geometry": test_metrics.get("post_reader_witness_geometry"),
        "diversity_retention": test_metrics.get("diversity_retention"),
        "decision_geometry": test_metrics["decision_geometry"],
        "margin_channel_diagnostics": test_metrics.get("margin_channel_diagnostics"),
        "failure_channel_diagnostics": test_metrics.get("failure_channel_diagnostics"),
        "success_channel_diagnostics": test_metrics.get("success_channel_diagnostics"),
        "path_uncertainty_diagnostics": test_metrics.get("path_uncertainty_diagnostics"),
        "path_signed_evidence_diagnostics": test_metrics.get("path_signed_evidence_diagnostics"),
        "decision_comparison": test_metrics.get("decision_comparison"),
        "decision_delta_diagnostics": test_metrics.get("decision_delta_diagnostics"),
        "disturbance_profile": test_metrics.get("disturbance_profile"),
        "uncertainty_axis_diagnostics": test_metrics.get("uncertainty_axis_diagnostics"),
        "risk_gate_threshold_sweep": test_metrics.get("risk_gate_threshold_sweep"),
        "success_recovery_diagnostics": test_metrics.get("success_recovery_diagnostics"),
        "signal_forensics": test_metrics.get("signal_forensics"),
        "base_signal_forensics": test_metrics.get("base_signal_forensics"),
        "base_to_final_error_corr_delta": test_metrics.get("base_to_final_error_corr_delta"),
        "weighting_diagnostics": test_metrics.get("weighting_diagnostics"),
    }


def _build_success_recovery_diagnostics(test_metrics: dict[str, Any]) -> dict[str, Any]:
    def _summary_path(section: dict[str, Any] | None, *path: str) -> Any:
        current: Any = section
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    def _summary_mean(section: dict[str, Any] | None, *path: str) -> float | None:
        value = _summary_path(section, *path)
        return value if isinstance(value, (int, float)) else None

    def _maybe_section(name: str) -> dict[str, Any] | None:
        section = test_metrics.get(name)
        return section if isinstance(section, dict) else None

    margin_channel_diagnostics = _maybe_section("margin_channel_diagnostics")
    success_channel_diagnostics = _maybe_section("success_channel_diagnostics")
    decision_comparison = _maybe_section("decision_comparison")
    decision_geometry = _maybe_section("decision_geometry")

    margin_positive_fraction_mean = _summary_mean(
        margin_channel_diagnostics, "margin_positive_fraction", "overall", "mean"
    )
    true_success_margin_positive_fraction_mean = _summary_mean(
        success_channel_diagnostics, "true_success_margin_positive_fraction", "overall", "mean"
    )
    true_success_margin_negative_fraction_mean = _summary_mean(
        success_channel_diagnostics, "true_success_margin_negative_fraction", "overall", "mean"
    )
    true_success_margin_near_zero_fraction_mean = _summary_mean(
        success_channel_diagnostics, "true_success_margin_near_zero_fraction", "overall", "mean"
    )
    true_success_commitment_below_tau_fraction_mean = _summary_mean(
        success_channel_diagnostics,
        "true_success_commitment_below_tau_fraction",
        "overall",
        "mean",
    )
    true_success_predicted_success_fraction_mean = _summary_mean(
        success_channel_diagnostics, "true_success_predicted_success_fraction", "overall", "mean"
    )
    true_success_predicted_uncertain_fraction_mean = _summary_mean(
        success_channel_diagnostics, "true_success_predicted_uncertain_fraction", "overall", "mean"
    )
    true_success_predicted_failure_fraction_mean = _summary_mean(
        success_channel_diagnostics, "true_success_predicted_failure_fraction", "overall", "mean"
    )
    true_success_uncertainty_mean = _summary_mean(success_channel_diagnostics, "uncertainty_U", "overall", "mean")
    overall_uncertainty_mean = _summary_mean(decision_geometry, "uncertainty_U", "overall", "mean")
    success_commitment_score_mean = _summary_mean(success_channel_diagnostics, "commitment_score", "overall", "mean")
    success_tau_uncertain_mean = _summary_mean(success_channel_diagnostics, "tau_uncertain", "overall", "mean")
    final_commitment_score_mean = _summary_mean(decision_geometry, "commitment_score", "overall", "mean")
    final_commitment_threshold_mean = _summary_mean(
        decision_geometry, "final_commitment_threshold", "overall", "mean"
    )
    tau_uncertain_mean = _summary_mean(decision_geometry, "tau_uncertain", "overall", "mean")

    base_hard_success_recall = _summary_mean(
        decision_comparison, "base_hard_gate", "success_recall"
    )
    final_hard_success_recall = _summary_mean(
        decision_comparison, "final_hard_gate", "success_recall"
    )
    base_soft_success_recall = _summary_mean(
        decision_comparison, "base_soft_logits", "success_recall"
    )
    final_soft_success_recall = _summary_mean(
        decision_comparison, "final_soft_logits", "success_recall"
    )

    blockers = []
    if isinstance(margin_positive_fraction_mean, (int, float)) and margin_positive_fraction_mean < 0.05:
        blockers.append("margin_never_positive")
    if (
        isinstance(true_success_margin_positive_fraction_mean, (int, float))
        and true_success_margin_positive_fraction_mean < 0.05
    ):
        blockers.append("success_margin_never_positive")
    if (
        isinstance(true_success_commitment_below_tau_fraction_mean, (int, float))
        and true_success_commitment_below_tau_fraction_mean > 0.50
    ):
        blockers.append("commitment_too_strict")
    if (
        isinstance(final_commitment_score_mean, (int, float))
        and isinstance(final_commitment_threshold_mean, (int, float))
        and final_commitment_threshold_mean > final_commitment_score_mean
    ):
        blockers.append("tau_uncertain_too_high")
    if (
        isinstance(success_tau_uncertain_mean, (int, float))
        and isinstance(success_commitment_score_mean, (int, float))
        and success_tau_uncertain_mean > success_commitment_score_mean
    ):
        blockers.append("success_commitment_below_tau")
    if (
        isinstance(true_success_uncertainty_mean, (int, float))
        and isinstance(overall_uncertainty_mean, (int, float))
        and true_success_uncertainty_mean > overall_uncertainty_mean + 0.05
    ):
        blockers.append("uncertainty_too_high")
    if (
        isinstance(base_hard_success_recall, (int, float))
        and isinstance(final_hard_success_recall, (int, float))
        and final_hard_success_recall + 0.05 < base_hard_success_recall
    ):
        blockers.append("hard_gate_suppresses_success")
    if (
        isinstance(base_soft_success_recall, (int, float))
        and isinstance(final_soft_success_recall, (int, float))
        and final_soft_success_recall + 0.05 < base_soft_success_recall
    ):
        blockers.append("soft_logits_fail_to_recover_success")
    if (
        isinstance(final_soft_success_recall, (int, float))
        and final_soft_success_recall < 0.05
    ):
        blockers.append("soft_logits_reaches_no_success")
    if (
        isinstance(true_success_predicted_failure_fraction_mean, (int, float))
        and true_success_predicted_failure_fraction_mean > 0.50
    ):
        blockers.append("predicted_failure_bias")

    is_m_never_positive = bool(
        isinstance(margin_positive_fraction_mean, (int, float))
        and margin_positive_fraction_mean < 0.05
    )
    is_c_always_below_tau = bool(
        isinstance(true_success_commitment_below_tau_fraction_mean, (int, float))
        and true_success_commitment_below_tau_fraction_mean > 0.50
    )
    is_u_too_high = bool(
        isinstance(true_success_uncertainty_mean, (int, float))
        and isinstance(overall_uncertainty_mean, (int, float))
        and true_success_uncertainty_mean > overall_uncertainty_mean + 0.05
    )
    is_tau_uncertain_too_high = bool(
        isinstance(final_commitment_score_mean, (int, float))
        and isinstance(final_commitment_threshold_mean, (int, float))
        and final_commitment_threshold_mean > final_commitment_score_mean
    )
    is_hard_gate_suppressing_success = bool(
        isinstance(base_hard_success_recall, (int, float))
        and isinstance(final_hard_success_recall, (int, float))
        and final_hard_success_recall + 0.05 < base_hard_success_recall
    )

    dominant_reason = blockers[0] if blockers else "mixed_or_unclear"
    return {
        "margin_positive_fraction_mean": margin_positive_fraction_mean,
        "true_success_margin_positive_fraction_mean": true_success_margin_positive_fraction_mean,
        "true_success_margin_negative_fraction_mean": true_success_margin_negative_fraction_mean,
        "true_success_margin_near_zero_fraction_mean": true_success_margin_near_zero_fraction_mean,
        "true_success_commitment_below_tau_fraction_mean": true_success_commitment_below_tau_fraction_mean,
        "true_success_predicted_success_fraction_mean": true_success_predicted_success_fraction_mean,
        "true_success_predicted_uncertain_fraction_mean": true_success_predicted_uncertain_fraction_mean,
        "true_success_predicted_failure_fraction_mean": true_success_predicted_failure_fraction_mean,
        "true_success_uncertainty_mean": true_success_uncertainty_mean,
        "overall_uncertainty_mean": overall_uncertainty_mean,
        "success_commitment_score_mean": success_commitment_score_mean,
        "success_tau_uncertain_mean": success_tau_uncertain_mean,
        "final_commitment_score_mean": final_commitment_score_mean,
        "final_commitment_threshold_mean": final_commitment_threshold_mean,
        "tau_uncertain_mean": tau_uncertain_mean,
        "base_hard_success_recall": base_hard_success_recall,
        "final_hard_success_recall": final_hard_success_recall,
        "base_soft_success_recall": base_soft_success_recall,
        "final_soft_success_recall": final_soft_success_recall,
        "is_m_never_positive": is_m_never_positive,
        "is_c_always_below_tau": is_c_always_below_tau,
        "is_u_too_high": is_u_too_high,
        "is_tau_uncertain_too_high": is_tau_uncertain_too_high,
        "is_hard_gate_suppressing_success": is_hard_gate_suppressing_success,
        "hard_gate_suppression_delta": (
            None
            if not isinstance(base_hard_success_recall, (int, float))
            or not isinstance(final_hard_success_recall, (int, float))
            else float(final_hard_success_recall - base_hard_success_recall)
        ),
        "blockers": blockers,
        "dominant_reason": dominant_reason,
    }


def _build_true_channel_diagnostics(
    *,
    channel_name: str,
    mask: torch.Tensor,
    global_margin_t: torch.Tensor,
    commitment_score_t: torch.Tensor,
    uncertainties_t: torch.Tensor,
    witness_positive_count_t: torch.Tensor,
    witness_negative_count_t: torch.Tensor,
    witness_sign_disagreement_rate_t: torch.Tensor,
    mixed_sign_witness_fraction_t: torch.Tensor,
    tau_uncertain_t: torch.Tensor,
    predictions_t: torch.Tensor,
    margin_band: float,
) -> dict[str, Any]:
    channel_mask = mask.to(dtype=torch.bool)
    prefix = f"true_{channel_name}"
    subset_margin = global_margin_t[channel_mask]
    subset_commitment = commitment_score_t[channel_mask]
    subset_uncertainty = uncertainties_t[channel_mask]
    subset_positive = witness_positive_count_t[channel_mask]
    subset_negative = witness_negative_count_t[channel_mask]
    subset_disagreement = witness_sign_disagreement_rate_t[channel_mask]
    subset_mixed = mixed_sign_witness_fraction_t[channel_mask]
    subset_tau = tau_uncertain_t[channel_mask]
    subset_predictions = predictions_t[channel_mask]

    return {
        "margin_M": summarize_tensor(subset_margin),
        "commitment_score": summarize_tensor(subset_commitment),
        "uncertainty_U": summarize_tensor(subset_uncertainty),
        "witness_positive_count": summarize_tensor(subset_positive),
        "witness_negative_count": summarize_tensor(subset_negative),
        "witness_sign_disagreement_rate": summarize_tensor(subset_disagreement),
        "mixed_sign_witness_fraction": summarize_tensor(subset_mixed),
        f"{prefix}_margin_negative_fraction": summarize_tensor((subset_margin < 0).float()),
        f"{prefix}_margin_positive_fraction": summarize_tensor((subset_margin > 0).float()),
        f"{prefix}_margin_near_zero_fraction": summarize_tensor((subset_margin.abs() <= margin_band).float()),
        f"{prefix}_commitment_below_tau_fraction": summarize_tensor(
            (subset_commitment < subset_tau).float()
        ),
        f"{prefix}_predicted_success_fraction": summarize_tensor(
            (subset_predictions == SUCCESS_CLASS_IDX).float()
        ),
        f"{prefix}_predicted_uncertain_fraction": summarize_tensor(
            (subset_predictions == UNCERTAIN_CLASS_IDX).float()
        ),
        f"{prefix}_predicted_failure_fraction": summarize_tensor(
            (subset_predictions == FAILURE_CLASS_IDX).float()
        ),
    }


def _make_loader(records: list[dict[str, Any]], config: E31Config, shuffle: bool) -> DataLoader:
    dataset = TraceWitnessDataset(records, vocab=Vocabulary(), config=config)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=trace_collate_fn,
    )


def _ordered_loader_like(loader: DataLoader) -> DataLoader:
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        collate_fn=loader.collate_fn,
    )


@torch.no_grad()
def _fit_uncertainty_profile_state(
    model: E31Model,
    loader: DataLoader,
    device: torch.device,
    config: E31Config,
) -> dict[str, Any]:
    if model.disturbance_profile is None:
        return {}

    ordered_loader = _ordered_loader_like(loader)
    previous_training = model.training
    previous_collect = model._collect_disturbance_profile
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = None
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state_all()

    anchors: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    incorrect: list[torch.Tensor] = []
    raw_profiles: list[torch.Tensor] = []

    try:
        torch.manual_seed(int(config.train_loop_seed) + 7919)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(config.train_loop_seed) + 7919)

        model.eval()
        model._collect_disturbance_profile = False
        for batch in ordered_loader:
            queries, trace_inputs, evidence_targets, prim_targets = _move_batch(batch, device)
            del evidence_targets
            outputs = model(queries, trace_inputs=trace_inputs)
            target = prim_targets[:, 3]
            prediction = bucketize_outcome_logits(outputs.base_outcome_logits)
            anchors.append(outputs.anchor.detach())
            labels.append(target.detach())
            incorrect.append((prediction.detach() != target).to(dtype=outputs.anchor.dtype))

        if anchors:
            anchors_t = torch.cat(anchors, dim=0)
            labels_t = torch.cat(labels, dim=0)
            incorrect_t = torch.cat(incorrect, dim=0)
            model.disturbance_profile.fit_support_memory(anchors_t, labels_t, incorrect_t)

        model._collect_disturbance_profile = True
        for batch in ordered_loader:
            queries, trace_inputs, evidence_targets, prim_targets = _move_batch(batch, device)
            del evidence_targets, prim_targets
            outputs = model(queries, trace_inputs=trace_inputs)
            if outputs.disturbance_profile is not None:
                raw_profiles.append(outputs.disturbance_profile.profile.detach())

        if raw_profiles:
            model.disturbance_profile.fit_feature_normalizer(torch.cat(raw_profiles, dim=0))
    finally:
        model._collect_disturbance_profile = previous_collect
        model.train(previous_training)
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)

    summary = model.disturbance_profile.profile_state_summary()
    print(
        "uncertainty_profile_state_fit "
        f"support_memory_size={summary.get('support_memory_size')} "
        f"profile_normalizer_count={summary.get('profile_normalizer_count')} "
        f"profile_normalizer_dim={summary.get('profile_normalizer_dim')} "
        f"risk_input_mode={summary.get('profile_risk_input_mode')}"
    )
    return summary


def _evaluate(
    model: E31Model,
    criterion: E31Loss,
    loader: DataLoader,
    device: torch.device,
    records: list[dict[str, Any]],
    config: E31Config,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    logits = []
    base_logits = []
    path_outcome_logits = []
    targets = []
    margins = []
    uncertainties = []
    witness_margins = []
    coherence_scores = []
    witness_weights = []
    pairwise_agreement_mean = []
    pairwise_agreement_std = []
    witness_weight_entropy = []
    witness_weight_max = []
    witness_weight_kl_uniform = []
    witness_margin_mean = []
    witness_margin_std = []
    global_margin = []
    base_global_margin = []
    vote_disagreement = []
    geometric_disagreement = []
    commitment_score = []
    base_commitment_score = []
    commitment_logit = []
    base_commitment_logit = []
    commitment_scale = []
    lambda_commit = []
    lambda_profile_risk = []
    profile_risk_temperature = []
    tau_uncertain = []
    base_commitment_threshold = []
    final_commitment_threshold = []
    uncertain_logit = []
    margin_minus_lambda_u = []
    calibrated_risk = []
    risk_adjustment = []
    pre_pairwise_cosine_agreement = []
    pre_pairwise_cosine_agreement_std = []
    pre_mean_squared_distance_to_anchor = []
    pre_witness_latent_variance = []
    pre_witness_latent_norm_mean = []
    pre_witness_latent_norm_std = []
    pre_witness_latent_spread = []
    pre_principal_singular_value_ratio = []
    post_pairwise_cosine_agreement = []
    post_pairwise_cosine_agreement_std = []
    post_mean_squared_distance_to_anchor = []
    post_witness_latent_variance = []
    post_witness_latent_norm_mean = []
    post_witness_latent_norm_std = []
    post_witness_latent_spread = []
    post_principal_singular_value_ratio = []
    path_disagreement = []
    path_similarity = []
    path_confidence = []
    endpoint_confidence = []
    method_conflict = []
    path_delta = []
    method_rho = []
    lambda_path = []
    path_weight_eta = []
    per_witness_path_disagreement = []
    path_consistency_loss = []
    path_signed_margin = []
    path_signed_margin_abs = []
    path_signed_margin_delta = []
    path_signed_logit_scale = []
    path_signed_margin_scale = []
    path_failure_gate_logit = []
    path_failure_gate_probability = []
    path_failure_gate_signal = []
    path_failure_gate_threshold = []
    path_success_logit = []
    path_uncertain_logit = []
    path_failure_logit = []
    path_success_minus_uncertain = []
    path_failure_minus_uncertain = []
    probe_order = []
    probe_boundary = []
    probe_support = []
    risk_prob = []
    risk_logit = []
    commit = []
    disturbance_profiles = []
    disturbance_normalized_profiles = []
    disturbance_axis_scores = []
    disturbance_risk_profiles = []
    disturbance_profile_feature_names: list[str] | None = None
    disturbance_axis_names: list[str] | None = None
    disturbance_risk_profile_names: list[str] | None = None
    disturbance_profile_feature_specs: list[Any] | None = None

    with torch.no_grad():
        for batch in loader:
            queries, trace_inputs, evidence_targets, prim_targets = _move_batch(batch, device)
            outputs = model(queries, trace_inputs=trace_inputs)
            losses = criterion(outputs, prim_targets, evidence_targets)
            batch_size = queries.size(0)
            total_loss += float(losses["total"].item()) * batch_size
            total_items += batch_size

            logits.append(outputs.outcome_logits.cpu())
            base_logits.append(outputs.base_outcome_logits.cpu())
            targets.append(prim_targets[:, 3].cpu())
            margins.append(outputs.margin.cpu())
            uncertainties.append(outputs.uncertainty.cpu())

            stats = outputs.witness_statistics
            witness_margins.append(stats.witness_margins.cpu())
            coherence_scores.append(stats.coherence_scores.cpu())
            witness_weights.append(stats.witness_weights.cpu())
            pairwise_agreement_mean.append(stats.pairwise_agreement_mean.cpu())
            pairwise_agreement_std.append(stats.pairwise_agreement_std.cpu())
            witness_weight_entropy.append(stats.witness_weight_entropy.cpu())
            witness_weight_max.append(stats.witness_weight_max.cpu())
            witness_weight_kl_uniform.append(stats.witness_weight_kl_uniform.cpu())
            witness_margin_mean.append(stats.witness_margin_mean.cpu())
            witness_margin_std.append(stats.witness_margin_std.cpu())
            decision_inputs = outputs.decision_inputs
            global_margin.append(decision_inputs["global_margin"].cpu())
            base_global_margin.append(decision_inputs["base_global_margin"].cpu())
            vote_disagreement.append(stats.vote_disagreement.cpu())
            geometric_disagreement.append(stats.geometric_disagreement.cpu())

            commitment_score.append(decision_inputs["commitment_score"].cpu())
            base_commitment_score.append(decision_inputs["base_commitment_score"].cpu())
            commitment_logit.append(decision_inputs["commitment_logit"].cpu())
            base_commitment_logit.append(decision_inputs["base_commitment_logit"].cpu())
            commitment_scale.append(decision_inputs["commitment_scale"].cpu())
            lambda_commit.append(decision_inputs["lambda_commit"].cpu())
            if "lambda_profile_risk" in decision_inputs:
                lambda_profile_risk.append(decision_inputs["lambda_profile_risk"].cpu())
            if "profile_risk_temperature" in decision_inputs:
                profile_risk_temperature.append(decision_inputs["profile_risk_temperature"].cpu())
            tau_uncertain.append(decision_inputs["tau_uncertain"].cpu())
            if "base_commitment_threshold" in decision_inputs:
                base_commitment_threshold.append(decision_inputs["base_commitment_threshold"].cpu())
            if "final_commitment_threshold" in decision_inputs:
                final_commitment_threshold.append(decision_inputs["final_commitment_threshold"].cpu())
            uncertain_logit.append(decision_inputs["uncertain_logit"].cpu())
            margin_minus_lambda_u.append(decision_inputs["margin_minus_lambda_u"].cpu())
            if "calibrated_risk" in decision_inputs:
                calibrated_risk.append(decision_inputs["calibrated_risk"].cpu())
            if "risk_adjustment" in decision_inputs:
                risk_adjustment.append(decision_inputs["risk_adjustment"].cpu())
            if outputs.path_uncertainty is not None:
                path_disagreement.append(decision_inputs["path_disagreement"].cpu())
                path_similarity.append(outputs.path_uncertainty.path_similarity.cpu())
                path_confidence.append(decision_inputs["path_confidence"].cpu())
                endpoint_confidence.append(decision_inputs["endpoint_confidence"].cpu())
                method_conflict.append(decision_inputs["method_conflict"].cpu())
                path_delta.append(decision_inputs["path_delta"].cpu())
                method_rho.append(decision_inputs["method_rho"].cpu())
                lambda_path.append(decision_inputs["lambda_path"].cpu())
                path_weight_eta.append(decision_inputs["path_weight_eta"].cpu())
                per_witness_path_disagreement.append(
                    outputs.path_uncertainty.per_witness_path_disagreement.cpu()
                )
                path_consistency_loss.append(outputs.path_consistency_loss.detach().reshape(1).cpu())
            if outputs.path_outcome_logits is not None:
                path_outcome_logits.append(outputs.path_outcome_logits.cpu())
                path_signed_margin.append(decision_inputs["path_signed_margin"].cpu())
                path_signed_margin_abs.append(decision_inputs["path_signed_margin_abs"].cpu())
                path_signed_margin_delta.append(decision_inputs["path_signed_margin_delta"].cpu())
                path_signed_logit_scale.append(decision_inputs["path_signed_logit_scale"].cpu())
                path_signed_margin_scale.append(decision_inputs["path_signed_margin_scale"].cpu())
                path_failure_gate_logit.append(decision_inputs["path_failure_gate_logit"].cpu())
                path_failure_gate_probability.append(
                    decision_inputs["path_failure_gate_probability"].cpu()
                )
                path_failure_gate_signal.append(decision_inputs["path_failure_gate_signal"].cpu())
                path_failure_gate_threshold.append(
                    decision_inputs["path_failure_gate_threshold"].cpu()
                )
                path_success_logit.append(decision_inputs["path_success_logit"].cpu())
                path_uncertain_logit.append(decision_inputs["path_uncertain_logit"].cpu())
                path_failure_logit.append(decision_inputs["path_failure_logit"].cpu())
                path_success_minus_uncertain.append(
                    decision_inputs["path_success_minus_uncertain"].cpu()
                )
                path_failure_minus_uncertain.append(
                    decision_inputs["path_failure_minus_uncertain"].cpu()
                )

            if outputs.pre_reader_witness_geometry is not None:
                raw = outputs.pre_reader_witness_geometry
                pre_pairwise_cosine_agreement.append(raw.pairwise_cosine_agreement.cpu())
                pre_pairwise_cosine_agreement_std.append(raw.pairwise_cosine_agreement_std.cpu())
                pre_mean_squared_distance_to_anchor.append(raw.mean_squared_distance_to_anchor.cpu())
                pre_witness_latent_variance.append(raw.witness_latent_variance.cpu())
                pre_witness_latent_norm_mean.append(raw.witness_latent_norm_mean.cpu())
                pre_witness_latent_norm_std.append(raw.witness_latent_norm_std.cpu())
                pre_witness_latent_spread.append(raw.witness_latent_spread.cpu())
                pre_principal_singular_value_ratio.append(raw.principal_singular_value_ratio.cpu())

            point_reader_output = outputs.point_reader_output
            post_anchor = outputs.anchor
            if point_reader_output.shape[-1] != post_anchor.shape[-1]:
                post_anchor = post_anchor.new_zeros(post_anchor.shape[0], point_reader_output.shape[-1])
            post_reader_geometry = summarize_raw_witness_geometry(post_anchor, point_reader_output)
            post_pairwise_cosine_agreement.append(post_reader_geometry.pairwise_cosine_agreement.cpu())
            post_pairwise_cosine_agreement_std.append(post_reader_geometry.pairwise_cosine_agreement_std.cpu())
            post_mean_squared_distance_to_anchor.append(post_reader_geometry.mean_squared_distance_to_anchor.cpu())
            post_witness_latent_variance.append(post_reader_geometry.witness_latent_variance.cpu())
            post_witness_latent_norm_mean.append(post_reader_geometry.witness_latent_norm_mean.cpu())
            post_witness_latent_norm_std.append(post_reader_geometry.witness_latent_norm_std.cpu())
            post_witness_latent_spread.append(post_reader_geometry.witness_latent_spread.cpu())
            post_principal_singular_value_ratio.append(post_reader_geometry.principal_singular_value_ratio.cpu())

            if outputs.diagnostic_probes is not None:
                probe_order.append(outputs.diagnostic_probes.order.cpu())
                probe_boundary.append(outputs.diagnostic_probes.boundary.cpu())
                probe_support.append(outputs.diagnostic_probes.support.cpu())

            if outputs.risk_prob is not None:
                risk_prob.append(outputs.risk_prob.cpu())
            if outputs.risk_logit is not None:
                risk_logit.append(outputs.risk_logit.cpu())
            if outputs.commit is not None:
                commit.append(outputs.commit.cpu())
            if outputs.disturbance_profile is not None:
                disturbance_profiles.append(outputs.disturbance_profile.profile.cpu())
                disturbance_normalized_profiles.append(outputs.disturbance_profile.normalized_profile.cpu())
                disturbance_axis_scores.append(outputs.disturbance_profile.axis_scores.cpu())
                disturbance_risk_profiles.append(outputs.disturbance_profile.risk_profile.cpu())
                if disturbance_profile_feature_names is None:
                    disturbance_profile_feature_names = list(outputs.disturbance_profile.feature_names)
                    disturbance_axis_names = list(outputs.disturbance_profile.axis_names)
                    disturbance_risk_profile_names = list(outputs.disturbance_profile.risk_profile_names)
                    disturbance_profile_feature_specs = list(outputs.disturbance_profile.feature_specs)

    logits_t = torch.cat(logits, dim=0)
    base_logits_t = torch.cat(base_logits, dim=0)
    targets_t = torch.cat(targets, dim=0)
    margins_t = torch.cat(margins, dim=0)
    uncertainties_t = torch.cat(uncertainties, dim=0)
    witness_margins_t = torch.cat(witness_margins, dim=0)
    coherence_scores_t = torch.cat(coherence_scores, dim=0)
    witness_weights_t = torch.cat(witness_weights, dim=0)
    pairwise_agreement_mean_t = torch.cat(pairwise_agreement_mean, dim=0)
    pairwise_agreement_std_t = torch.cat(pairwise_agreement_std, dim=0)
    witness_weight_entropy_t = torch.cat(witness_weight_entropy, dim=0)
    witness_weight_max_t = torch.cat(witness_weight_max, dim=0)
    witness_weight_kl_uniform_t = torch.cat(witness_weight_kl_uniform, dim=0)
    witness_margin_mean_t = torch.cat(witness_margin_mean, dim=0)
    witness_margin_std_t = torch.cat(witness_margin_std, dim=0)
    global_margin_t = torch.cat(global_margin, dim=0)
    base_global_margin_t = torch.cat(base_global_margin, dim=0)
    vote_disagreement_t = torch.cat(vote_disagreement, dim=0)
    geometric_disagreement_t = torch.cat(geometric_disagreement, dim=0)
    commitment_score_t = torch.cat(commitment_score, dim=0)
    base_commitment_score_t = torch.cat(base_commitment_score, dim=0)
    commitment_logit_t = torch.cat(commitment_logit, dim=0)
    base_commitment_logit_t = torch.cat(base_commitment_logit, dim=0)
    commitment_scale_t = torch.cat(commitment_scale, dim=0)
    lambda_commit_t = torch.cat(lambda_commit, dim=0)
    lambda_profile_risk_t = torch.cat(lambda_profile_risk, dim=0) if lambda_profile_risk else None
    profile_risk_temperature_t = (
        torch.cat(profile_risk_temperature, dim=0) if profile_risk_temperature else None
    )
    tau_uncertain_t = torch.cat(tau_uncertain, dim=0)
    base_commitment_threshold_t = (
        torch.cat(base_commitment_threshold, dim=0) if base_commitment_threshold else None
    )
    final_commitment_threshold_t = (
        torch.cat(final_commitment_threshold, dim=0) if final_commitment_threshold else None
    )
    uncertain_logit_t = torch.cat(uncertain_logit, dim=0)
    margin_minus_lambda_u_t = torch.cat(margin_minus_lambda_u, dim=0)
    calibrated_risk_t = torch.cat(calibrated_risk, dim=0) if calibrated_risk else None
    risk_adjustment_t = torch.cat(risk_adjustment, dim=0) if risk_adjustment else None
    post_pairwise_cosine_agreement_t = torch.cat(post_pairwise_cosine_agreement, dim=0)
    post_pairwise_cosine_agreement_std_t = torch.cat(post_pairwise_cosine_agreement_std, dim=0)
    post_mean_squared_distance_to_anchor_t = torch.cat(post_mean_squared_distance_to_anchor, dim=0)
    post_witness_latent_variance_t = torch.cat(post_witness_latent_variance, dim=0)
    post_witness_latent_norm_mean_t = torch.cat(post_witness_latent_norm_mean, dim=0)
    post_witness_latent_norm_std_t = torch.cat(post_witness_latent_norm_std, dim=0)
    post_witness_latent_spread_t = torch.cat(post_witness_latent_spread, dim=0)
    post_principal_singular_value_ratio_t = torch.cat(post_principal_singular_value_ratio, dim=0)
    path_disagreement_t = torch.cat(path_disagreement, dim=0) if path_disagreement else None
    path_similarity_t = torch.cat(path_similarity, dim=0) if path_similarity else None
    path_confidence_t = torch.cat(path_confidence, dim=0) if path_confidence else None
    endpoint_confidence_t = torch.cat(endpoint_confidence, dim=0) if endpoint_confidence else None
    method_conflict_t = torch.cat(method_conflict, dim=0) if method_conflict else None
    path_delta_t = torch.cat(path_delta, dim=0) if path_delta else None
    method_rho_t = torch.cat(method_rho, dim=0) if method_rho else None
    lambda_path_t = torch.cat(lambda_path, dim=0) if lambda_path else None
    path_weight_eta_t = torch.cat(path_weight_eta, dim=0) if path_weight_eta else None
    per_witness_path_disagreement_t = (
        torch.cat(per_witness_path_disagreement, dim=0)
        if per_witness_path_disagreement
        else None
    )
    path_consistency_loss_t = (
        torch.cat(path_consistency_loss, dim=0)
        if path_consistency_loss
        else None
    )
    path_outcome_logits_t = torch.cat(path_outcome_logits, dim=0) if path_outcome_logits else None
    path_signed_margin_t = torch.cat(path_signed_margin, dim=0) if path_signed_margin else None
    path_signed_margin_abs_t = (
        torch.cat(path_signed_margin_abs, dim=0) if path_signed_margin_abs else None
    )
    path_signed_margin_delta_t = (
        torch.cat(path_signed_margin_delta, dim=0) if path_signed_margin_delta else None
    )
    path_signed_logit_scale_t = (
        torch.cat(path_signed_logit_scale, dim=0) if path_signed_logit_scale else None
    )
    path_signed_margin_scale_t = (
        torch.cat(path_signed_margin_scale, dim=0) if path_signed_margin_scale else None
    )
    path_failure_gate_logit_t = (
        torch.cat(path_failure_gate_logit, dim=0) if path_failure_gate_logit else None
    )
    path_failure_gate_probability_t = (
        torch.cat(path_failure_gate_probability, dim=0)
        if path_failure_gate_probability
        else None
    )
    path_failure_gate_signal_t = (
        torch.cat(path_failure_gate_signal, dim=0) if path_failure_gate_signal else None
    )
    path_failure_gate_threshold_t = (
        torch.cat(path_failure_gate_threshold, dim=0)
        if path_failure_gate_threshold
        else None
    )
    path_success_logit_t = torch.cat(path_success_logit, dim=0) if path_success_logit else None
    path_uncertain_logit_t = (
        torch.cat(path_uncertain_logit, dim=0) if path_uncertain_logit else None
    )
    path_failure_logit_t = torch.cat(path_failure_logit, dim=0) if path_failure_logit else None
    path_success_minus_uncertain_t = (
        torch.cat(path_success_minus_uncertain, dim=0)
        if path_success_minus_uncertain
        else None
    )
    path_failure_minus_uncertain_t = (
        torch.cat(path_failure_minus_uncertain, dim=0)
        if path_failure_minus_uncertain
        else None
    )
    risk_prob_t = torch.cat(risk_prob, dim=0) if risk_prob else None
    risk_logit_t = torch.cat(risk_logit, dim=0) if risk_logit else None
    commit_t = torch.cat(commit, dim=0) if commit else None
    disturbance_profile_t = (
        torch.cat(disturbance_profiles, dim=0) if disturbance_profiles else None
    )
    disturbance_normalized_profile_t = (
        torch.cat(disturbance_normalized_profiles, dim=0)
        if disturbance_normalized_profiles
        else None
    )
    disturbance_axis_scores_t = (
        torch.cat(disturbance_axis_scores, dim=0) if disturbance_axis_scores else None
    )
    disturbance_risk_profile_t = (
        torch.cat(disturbance_risk_profiles, dim=0) if disturbance_risk_profiles else None
    )

    if base_commitment_threshold_t is None:
        base_commitment_threshold_t = tau_uncertain_t
    if final_commitment_threshold_t is None:
        final_commitment_threshold_t = tau_uncertain_t
    final_hard_predictions_t = bucketize_commitment_decisions(
        global_margin_t,
        commitment_score_t,
        final_commitment_threshold_t,
    )
    base_hard_predictions_t = bucketize_commitment_decisions(
        base_global_margin_t,
        base_commitment_score_t,
        base_commitment_threshold_t,
    )
    if config.use_hard_eval_gate:
        predictions_t = final_hard_predictions_t
    else:
        predictions_t = bucketize_outcome_logits(logits_t)
    base_soft_predictions_t = bucketize_outcome_logits(base_logits_t)
    final_soft_predictions_t = bucketize_outcome_logits(logits_t)
    path_only_predictions_t = (
        bucketize_outcome_logits(path_outcome_logits_t)
        if path_outcome_logits_t is not None
        else None
    )

    bundle: dict[str, torch.Tensor] = {
        "outcome_logits": logits_t,
        "base_outcome_logits": base_logits_t,
        "outcome_targets": targets_t,
        "predictions": predictions_t,
        "base_hard_predictions": base_hard_predictions_t,
        "final_hard_predictions": final_hard_predictions_t,
        "base_soft_predictions": base_soft_predictions_t,
        "final_soft_predictions": final_soft_predictions_t,
        "margin": margins_t,
        "uncertainty": uncertainties_t,
        "witness_margins": witness_margins_t,
        "coherence_scores": coherence_scores_t,
        "witness_weights": witness_weights_t,
        "pairwise_agreement_mean": pairwise_agreement_mean_t,
        "pairwise_agreement_std": pairwise_agreement_std_t,
        "witness_weight_entropy": witness_weight_entropy_t,
        "witness_weight_max": witness_weight_max_t,
        "witness_weight_kl_uniform": witness_weight_kl_uniform_t,
        "witness_margin_mean": witness_margin_mean_t,
        "witness_margin_std": witness_margin_std_t,
        "global_margin": global_margin_t,
        "base_global_margin": base_global_margin_t,
        "vote_disagreement": vote_disagreement_t,
        "geometric_disagreement": geometric_disagreement_t,
        "commitment_score": commitment_score_t,
        "base_commitment_score": base_commitment_score_t,
        "commitment_logit": commitment_logit_t,
        "base_commitment_logit": base_commitment_logit_t,
        "commitment_scale": commitment_scale_t,
        "lambda_commit": lambda_commit_t,
        "lambda_profile_risk": lambda_profile_risk_t,
        "profile_risk_temperature": profile_risk_temperature_t,
        "tau_uncertain": tau_uncertain_t,
        "base_commitment_threshold": base_commitment_threshold_t,
        "final_commitment_threshold": final_commitment_threshold_t,
        "uncertain_logit": uncertain_logit_t,
        "margin_minus_lambda_u": margin_minus_lambda_u_t,
        "calibrated_risk": calibrated_risk_t,
        "risk_adjustment": risk_adjustment_t,
    }
    if risk_prob_t is not None:
        bundle.update(
            {
                "risk_prob": risk_prob_t,
                "risk_logit": risk_logit_t,
                "commit": commit_t,
            }
        )
    if disturbance_profile_t is not None:
        bundle["disturbance_profile"] = disturbance_profile_t
    if disturbance_normalized_profile_t is not None:
        bundle["disturbance_normalized_profile"] = disturbance_normalized_profile_t
    if disturbance_axis_scores_t is not None:
        bundle["disturbance_axis_scores"] = disturbance_axis_scores_t
    if disturbance_risk_profile_t is not None:
        bundle["disturbance_risk_profile"] = disturbance_risk_profile_t
    if path_disagreement_t is not None:
        bundle.update(
            {
                "path_disagreement": path_disagreement_t,
                "path_similarity": path_similarity_t,
                "path_confidence": path_confidence_t,
                "endpoint_confidence": endpoint_confidence_t,
                "method_conflict": method_conflict_t,
                "path_delta": path_delta_t,
                "method_rho": method_rho_t,
                "lambda_path": lambda_path_t,
                "path_weight_eta": path_weight_eta_t,
                "per_witness_path_disagreement": per_witness_path_disagreement_t,
                "path_consistency_loss": path_consistency_loss_t,
            }
        )
    if path_outcome_logits_t is not None:
        assert path_signed_margin_t is not None
        assert path_signed_margin_abs_t is not None
        assert path_signed_margin_delta_t is not None
        assert path_signed_logit_scale_t is not None
        assert path_signed_margin_scale_t is not None
        assert path_failure_gate_logit_t is not None
        assert path_failure_gate_probability_t is not None
        assert path_failure_gate_signal_t is not None
        assert path_failure_gate_threshold_t is not None
        assert path_success_logit_t is not None
        assert path_uncertain_logit_t is not None
        assert path_failure_logit_t is not None
        assert path_success_minus_uncertain_t is not None
        assert path_failure_minus_uncertain_t is not None
        assert path_only_predictions_t is not None
        bundle.update(
            {
                "path_outcome_logits": path_outcome_logits_t,
                "path_only_predictions": path_only_predictions_t,
                "path_signed_margin": path_signed_margin_t,
                "path_signed_margin_abs": path_signed_margin_abs_t,
                "path_signed_margin_delta": path_signed_margin_delta_t,
                "path_signed_logit_scale": path_signed_logit_scale_t,
                "path_signed_margin_scale": path_signed_margin_scale_t,
                "path_failure_gate_logit": path_failure_gate_logit_t,
                "path_failure_gate_probability": path_failure_gate_probability_t,
                "path_failure_gate_signal": path_failure_gate_signal_t,
                "path_success_logit": path_success_logit_t,
                "path_uncertain_logit": path_uncertain_logit_t,
                "path_failure_logit": path_failure_logit_t,
                "path_success_minus_uncertain": path_success_minus_uncertain_t,
                "path_failure_minus_uncertain": path_failure_minus_uncertain_t,
            }
        )

    metrics = compute_outcome_metrics(
        logits_t,
        targets_t,
        predictions=predictions_t,
        uncertainty=uncertainties_t,
        records=records,
    )
    metrics["prediction_mode"] = "hard_commitment_gate" if config.use_hard_eval_gate else "soft_logits"
    metrics["loss"] = total_loss / max(total_items, 1)

    if (
        (calibrated_risk_t is not None or risk_prob_t is not None)
        and risk_logit_t is not None
        and final_hard_predictions_t is not None
    ):
        gate_prob_t = calibrated_risk_t if calibrated_risk_t is not None else risk_prob_t
        assert gate_prob_t is not None
        risk_targets_t = _risk_targets_from_labels(targets_t, base_soft_predictions_t, config).float()
        risk_metrics = compute_binary_probability_metrics(
            gate_prob_t.detach().cpu().numpy(),
            risk_targets_t.detach().cpu().numpy(),
        )
        commit_mask_t = final_hard_predictions_t.detach().reshape(-1) != UNCERTAIN_CLASS_IDX
        committed_accuracy = (
            float(
                (
                    final_hard_predictions_t[commit_mask_t]
                    == targets_t[commit_mask_t]
                )
                .float()
                .mean()
                .item()
            )
            if bool(commit_mask_t.any())
            else float("nan")
        )
        profile_feature_means: dict[str, float] = {}
        sign_flip_feature_means = []
        margin_var_feature_means = []
        boundary_flip_distance_means = []
        axis_score_means: dict[str, float] = {}
        axis_score_error_corr: dict[str, float] = {}
        axis_score_risk_metrics: dict[str, dict[str, float]] = {}
        profile_feature_specs = (
            [asdict(spec) for spec in disturbance_profile_feature_specs]
            if disturbance_profile_feature_specs is not None
            else []
        )
        profile_state = (
            model.disturbance_profile.profile_state_summary()
            if model.disturbance_profile is not None
            else {}
        )
        if disturbance_profile_t is not None and disturbance_profile_feature_names is not None:
            profile_mean_t = disturbance_profile_t.mean(dim=0)
            profile_feature_means = {
                name: float(value)
                for name, value in zip(disturbance_profile_feature_names, profile_mean_t.tolist())
            }
            sign_flip_feature_means = [
                value for name, value in profile_feature_means.items() if name.endswith("/flip_rate")
            ]
            margin_var_feature_means = [
                value for name, value in profile_feature_means.items() if name.endswith("/margin_var")
            ]
            boundary_flip_distance_means = [
                value
                for name, value in profile_feature_means.items()
                if name.endswith("/boundary_flip_distance")
            ]

        if disturbance_axis_scores_t is not None and disturbance_axis_names is not None:
            for idx, axis_name in enumerate(disturbance_axis_names):
                values = disturbance_axis_scores_t[:, idx].detach()
                axis_score_means[axis_name] = float(values.mean().item())
                axis_score_error_corr[axis_name] = _vector_pearson_correlation(
                    values,
                    risk_targets_t,
                )
                try:
                    score_metrics = compute_binary_probability_metrics(
                        values.cpu().numpy(),
                        risk_targets_t.detach().cpu().numpy(),
                    )
                    axis_score_risk_metrics[axis_name] = {
                        "auroc": float(score_metrics["auroc"]),
                        "auprc": float(score_metrics["auprc"]),
                        "aurc": float(score_metrics["aurc"]),
                    }
                except ValueError:
                    axis_score_risk_metrics[axis_name] = {
                        "auroc": float("nan"),
                        "auprc": float("nan"),
                        "aurc": float("nan"),
                    }

        sweep_thresholds = [round(0.1 * idx, 1) for idx in range(1, 10)]
        sweep_rows: list[dict[str, Any]] = []
        for threshold in sweep_thresholds:
            threshold_tensor = commitment_score_t.new_full(
                commitment_score_t.shape,
                threshold,
            )
            sweep_predictions_t = bucketize_commitment_decisions(
                global_margin_t,
                commitment_score_t,
                threshold_tensor,
            )
            sweep_metrics = compute_outcome_metrics(
                logits_t,
                targets_t,
                predictions=sweep_predictions_t,
                uncertainty=uncertainties_t,
                records=records,
            )
            sweep_commit_mask_t = sweep_predictions_t != UNCERTAIN_CLASS_IDX
            sweep_committed_accuracy = (
                float((sweep_predictions_t[sweep_commit_mask_t] == targets_t[sweep_commit_mask_t]).float().mean().item())
                if bool(sweep_commit_mask_t.any())
                else float("nan")
            )
            sweep_rows.append(
                {
                    "threshold": threshold,
                    "coverage": float(sweep_commit_mask_t.float().mean().item()),
                    "commit_rate": float(sweep_commit_mask_t.float().mean().item()),
                    "abstention_rate": float(1.0 - sweep_commit_mask_t.float().mean().item()),
                    "committed_accuracy": sweep_committed_accuracy,
                    "success_recall": float(sweep_metrics["success_recall"]),
                    "failure_recall": float(sweep_metrics["failure_recall"]),
                    "uncertain_recall": float(sweep_metrics.get("uncertain_recall", float("nan"))),
                    "outcome_macro_f1": float(sweep_metrics["outcome_macro_f1"]),
                    "risk_auroc": float(risk_metrics["auroc"]),
                    "risk_aurc": float(risk_metrics["aurc"]),
                    "risk_auprc": float(risk_metrics["auprc"]),
                }
            )

        target_commit_rate_min = float(getattr(config, "target_commit_rate_min", 0.0))
        target_commit_rate_max = float(getattr(config, "target_commit_rate_max", 1.0))
        target_commit_rate_mid = 0.5 * (target_commit_rate_min + target_commit_rate_max)
        within_bounds = [
            row
            for row in sweep_rows
            if target_commit_rate_min <= row["commit_rate"] <= target_commit_rate_max
        ]
        if within_bounds:
            selected_row = max(
                within_bounds,
                key=lambda row: (
                    row["outcome_macro_f1"],
                    -abs(row["commit_rate"] - target_commit_rate_mid),
                ),
            )
            selection_reason = "best_macro_f1_within_target_commit_rate_bounds"
        else:
            selected_row = max(
                sweep_rows,
                key=lambda row: (
                    -_threshold_distance_to_bounds(
                        row["commit_rate"],
                        target_commit_rate_min,
                        target_commit_rate_max,
                    ),
                    row["outcome_macro_f1"],
                ),
            )
            selection_reason = "closest_commit_rate_to_target_bounds"

        metrics["disturbance_profile"] = {
            "risk_auroc": risk_metrics["auroc"],
            "risk_auprc": risk_metrics["auprc"],
            "risk_brier": risk_metrics["brier"],
            "risk_ece_10bin": risk_metrics["ece_10bin"],
            "risk_aurc": risk_metrics["aurc"],
            "raw_risk_prob_mean": float(risk_prob_t.mean().item()) if risk_prob_t is not None else float("nan"),
            "calibrated_risk_mean": float(gate_prob_t.mean().item()),
            "calibrated_risk_std": float(gate_prob_t.std(unbiased=False).item()),
            "risk_target_rate": float(risk_targets_t.float().mean().item()),
            "commit_rate": float(commit_mask_t.float().mean().item()),
            "abstention_rate": float(1.0 - commit_mask_t.float().mean().item()),
            "committed_accuracy": committed_accuracy,
            "base_commitment_score_mean": float(base_commitment_score_t.mean().item()),
            "final_commitment_score_mean": float(commitment_score_t.mean().item()),
            "base_commitment_threshold_mean": float(base_commitment_threshold_t.mean().item()),
            "final_commitment_threshold_mean": float(final_commitment_threshold_t.mean().item()),
            "lambda_profile_risk_mean": float(lambda_profile_risk_t.mean().item())
            if lambda_profile_risk_t is not None
            else float("nan"),
            "profile_risk_temperature_mean": float(profile_risk_temperature_t.mean().item())
            if profile_risk_temperature_t is not None
            else float("nan"),
            "risk_adjustment_mean": float(risk_adjustment_t.mean().item()) if risk_adjustment_t is not None else float("nan"),
            "profile_feature_means": profile_feature_means,
            "profile_feature_names": disturbance_profile_feature_names or [],
            "profile_feature_specs": profile_feature_specs,
            "normalized_profile_feature_names": disturbance_profile_feature_names or [],
            "risk_profile_names": disturbance_risk_profile_names or [],
            "axis_names": disturbance_axis_names or [],
            "axis_score_means": axis_score_means,
            "axis_score_error_corr": axis_score_error_corr,
            "axis_score_risk_metrics": axis_score_risk_metrics,
            "profile_state": profile_state,
            "sign_flip_rate_mean": float(sum(sign_flip_feature_means) / len(sign_flip_feature_means))
            if sign_flip_feature_means
            else float("nan"),
            "margin_variance_mean": float(sum(margin_var_feature_means) / len(margin_var_feature_means))
            if margin_var_feature_means
            else float("nan"),
            "boundary_flip_distance_mean": float(
                sum(boundary_flip_distance_means) / len(boundary_flip_distance_means)
            )
            if boundary_flip_distance_means
            else float("nan"),
            "threshold_sweep": {
                "target_commit_rate_min": target_commit_rate_min,
                "target_commit_rate_max": target_commit_rate_max,
                "target_commit_rate_mid": target_commit_rate_mid,
                "operating_threshold": float(final_commitment_threshold_t.flatten()[0].item()),
                "selection_reason": selection_reason,
                "selected_threshold": float(selected_row["threshold"]),
                "selected_operating_point": selected_row,
                "rows": sweep_rows,
            },
        }
        metrics["risk_gate_threshold_sweep"] = metrics["disturbance_profile"]["threshold_sweep"]

    strata = (
        build_stratified_masks(targets_t, final_hard_predictions_t)
        if config.log_grouped_geometry
        else None
    )
    incorrect_t = (final_hard_predictions_t != targets_t).float()
    base_hard_incorrect_t = (base_hard_predictions_t != targets_t).float()
    if disturbance_axis_scores_t is not None and disturbance_axis_names is not None:
        metrics["uncertainty_axis_diagnostics"] = summarize_metric_bundle_with_strata(
            {
                f"axis/{axis_name}": disturbance_axis_scores_t[:, idx]
                for idx, axis_name in enumerate(disturbance_axis_names)
            },
            strata,
        )
    metrics["decision_comparison"] = {
        "base_hard_gate": _prediction_metric_summary(
            base_logits_t,
            targets_t,
            base_hard_predictions_t,
            uncertainty=uncertainties_t,
            records=records,
        ),
        "final_hard_gate": _prediction_metric_summary(
            logits_t,
            targets_t,
            final_hard_predictions_t,
            uncertainty=uncertainties_t,
            records=records,
        ),
        "base_soft_logits": _prediction_metric_summary(
            base_logits_t,
            targets_t,
            base_soft_predictions_t,
            uncertainty=uncertainties_t,
            records=records,
        ),
        "final_soft_logits": _prediction_metric_summary(
            logits_t,
            targets_t,
            final_soft_predictions_t,
            uncertainty=uncertainties_t,
            records=records,
        ),
    }
    if path_outcome_logits_t is not None and path_only_predictions_t is not None:
        metrics["decision_comparison"]["path_only_logits"] = _prediction_metric_summary(
            path_outcome_logits_t,
            targets_t,
            path_only_predictions_t,
            uncertainty=uncertainties_t,
            records=records,
        )
    metrics["decision_delta_diagnostics"] = summarize_metric_bundle_with_strata(
        {
            "base_hard_correct": (base_hard_predictions_t == targets_t).float(),
            "final_hard_correct": (final_hard_predictions_t == targets_t).float(),
            "base_to_final_hard_flip": (base_hard_predictions_t != final_hard_predictions_t).float(),
            "base_to_final_margin_sign_flip": (
                (base_global_margin_t >= 0) != (global_margin_t >= 0)
            ).float(),
            "base_to_final_commitment_delta": commitment_score_t - base_commitment_score_t,
            "base_to_final_margin_delta": global_margin_t - base_global_margin_t,
        },
        strata,
    )

    if path_disagreement_t is not None:
        assert path_similarity_t is not None
        assert path_confidence_t is not None
        assert endpoint_confidence_t is not None
        assert method_conflict_t is not None
        assert path_delta_t is not None
        assert method_rho_t is not None
        assert lambda_path_t is not None
        assert path_weight_eta_t is not None
        assert per_witness_path_disagreement_t is not None

        metrics["path_disagreement_error_corr"] = _vector_pearson_correlation(
            path_disagreement_t,
            incorrect_t,
        )
        metrics["endpoint_path_conflict_error_corr"] = _vector_pearson_correlation(
            method_conflict_t,
            incorrect_t,
        )
        metrics["path_uncertainty_diagnostics"] = summarize_metric_bundle_with_strata(
            {
                "path_disagreement": path_disagreement_t,
                "path_similarity": path_similarity_t,
                "path_confidence": path_confidence_t,
                "endpoint_confidence": endpoint_confidence_t,
                "method_conflict": method_conflict_t,
                "per_witness_path_disagreement_mean": per_witness_path_disagreement_t.mean(dim=-1),
                "per_witness_path_disagreement_std": per_witness_path_disagreement_t.std(
                    dim=-1,
                    unbiased=False,
                ),
                "per_witness_path_disagreement_max": per_witness_path_disagreement_t.max(dim=-1).values,
                "path_delta": path_delta_t,
                "method_rho": method_rho_t,
                "lambda_path": lambda_path_t,
                "path_weight_eta": path_weight_eta_t,
                "path_commitment_penalty": lambda_path_t * path_disagreement_t,
            },
            strata,
        )
        if path_consistency_loss_t is not None:
            metrics["path_consistency_loss"] = summarize_tensor(path_consistency_loss_t)

    signal_forensics = {
        "uncertainty": uncertainties_t,
        "abs_final_margin": global_margin_t.abs(),
        "abs_base_margin": base_global_margin_t.abs(),
        "base_commitment_score": base_commitment_score_t,
        "final_commitment_score": commitment_score_t,
        "base_commitment_threshold": base_commitment_threshold_t,
        "final_commitment_threshold": final_commitment_threshold_t,
        "raw_risk_prob": risk_prob_t if risk_prob_t is not None else torch.zeros_like(commitment_score_t),
        "calibrated_risk": calibrated_risk_t
        if calibrated_risk_t is not None
        else torch.zeros_like(commitment_score_t),
        "risk_adjustment": risk_adjustment_t
        if risk_adjustment_t is not None
        else torch.zeros_like(commitment_score_t),
    }
    if path_disagreement_t is not None:
        signal_forensics.update(
            {
                "path_disagreement": path_disagreement_t,
                "method_conflict": method_conflict_t,
                "per_witness_path_disagreement_mean": per_witness_path_disagreement_t.mean(dim=-1),
            }
        )
    if path_outcome_logits_t is not None:
        assert path_signed_margin_t is not None
        assert path_signed_margin_abs_t is not None
        assert path_signed_margin_delta_t is not None
        assert path_signed_logit_scale_t is not None
        assert path_signed_margin_scale_t is not None
        assert path_failure_gate_logit_t is not None
        assert path_failure_gate_probability_t is not None
        assert path_failure_gate_signal_t is not None
        assert path_failure_gate_threshold_t is not None
        assert path_success_logit_t is not None
        assert path_uncertain_logit_t is not None
        assert path_failure_logit_t is not None
        assert path_success_minus_uncertain_t is not None
        assert path_failure_minus_uncertain_t is not None
        path_probs_t = torch.softmax(path_outcome_logits_t, dim=-1)
        signal_forensics.update(
            {
                "path_signed_margin": path_signed_margin_t,
                "path_signed_margin_abs": path_signed_margin_abs_t,
                "path_signed_margin_delta": path_signed_margin_delta_t,
                "path_failure_gate_logit": path_failure_gate_logit_t,
                "path_failure_gate_probability": path_failure_gate_probability_t,
                "path_failure_gate_signal": path_failure_gate_signal_t,
                "path_failure_gate_threshold": path_failure_gate_threshold_t,
                "path_success_logit": path_success_logit_t,
                "path_uncertain_logit": path_uncertain_logit_t,
                "path_failure_logit": path_failure_logit_t,
                "path_success_probability": path_probs_t[:, SUCCESS_CLASS_IDX],
                "path_uncertain_probability": path_probs_t[:, UNCERTAIN_CLASS_IDX],
                "path_failure_probability": path_probs_t[:, FAILURE_CLASS_IDX],
            }
        )
        metrics["path_signed_margin_error_corr"] = _vector_pearson_correlation(
            path_signed_margin_t,
            incorrect_t,
        )
        metrics["path_failure_gate_logit_true_failure_corr"] = _vector_pearson_correlation(
            path_failure_gate_logit_t,
            (targets_t == FAILURE_CLASS_IDX).float(),
        )
        metrics["path_failure_gate_probability_true_failure_corr"] = _vector_pearson_correlation(
            path_failure_gate_probability_t,
            (targets_t == FAILURE_CLASS_IDX).float(),
        )
        metrics["path_failure_gate_signal_true_failure_corr"] = _vector_pearson_correlation(
            path_failure_gate_signal_t,
            (targets_t == FAILURE_CLASS_IDX).float(),
        )
        metrics["path_failure_logit_true_failure_corr"] = _vector_pearson_correlation(
            path_failure_logit_t,
            (targets_t == FAILURE_CLASS_IDX).float(),
        )
        metrics["path_failure_probability_true_failure_corr"] = _vector_pearson_correlation(
            path_probs_t[:, FAILURE_CLASS_IDX],
            (targets_t == FAILURE_CLASS_IDX).float(),
        )
        metrics["path_signed_evidence_diagnostics"] = summarize_metric_bundle_with_strata(
            {
                "path_signed_margin": path_signed_margin_t,
                "path_signed_margin_abs": path_signed_margin_abs_t,
                "path_signed_margin_delta": path_signed_margin_delta_t,
                "path_signed_logit_scale": path_signed_logit_scale_t,
                "path_signed_margin_scale": path_signed_margin_scale_t,
                "path_failure_gate_logit": path_failure_gate_logit_t,
                "path_failure_gate_probability": path_failure_gate_probability_t,
                "path_failure_gate_signal": path_failure_gate_signal_t,
                "path_failure_gate_threshold": path_failure_gate_threshold_t,
                "path_success_logit": path_success_logit_t,
                "path_uncertain_logit": path_uncertain_logit_t,
                "path_failure_logit": path_failure_logit_t,
                "path_success_minus_uncertain": path_success_minus_uncertain_t,
                "path_failure_minus_uncertain": path_failure_minus_uncertain_t,
                "path_success_probability": path_probs_t[:, SUCCESS_CLASS_IDX],
                "path_uncertain_probability": path_probs_t[:, UNCERTAIN_CLASS_IDX],
                "path_failure_probability": path_probs_t[:, FAILURE_CLASS_IDX],
                "path_failure_gate_logit": path_failure_gate_logit_t,
                "path_failure_gate_probability": path_failure_gate_probability_t,
                "path_failure_gate_signal": path_failure_gate_signal_t,
                "path_only_correct": (path_only_predictions_t == targets_t).float(),
                "path_only_predicts_failure": (
                    path_only_predictions_t == FAILURE_CLASS_IDX
                ).float(),
                "path_margin_predicts_failure": (path_signed_margin_t > 0).float(),
                "path_margin_success_failure_support": (
                    targets_t != UNCERTAIN_CLASS_IDX
                ).float(),
                "path_margin_correct_sign_on_success_failure": (
                    (
                        ((path_signed_margin_t == 0) & (targets_t == SUCCESS_CLASS_IDX))
                        | ((path_signed_margin_t > 0) & (targets_t == FAILURE_CLASS_IDX))
                    )
                    & (targets_t != UNCERTAIN_CLASS_IDX)
                ).float(),
                "final_minus_base_margin": global_margin_t - base_global_margin_t,
                "final_minus_base_commitment": commitment_score_t - base_commitment_score_t,
            },
            strata,
        )
    metrics["signal_forensics"] = _build_signal_forensics(
        signal_forensics,
        targets=targets_t,
        predictions=final_hard_predictions_t,
    )
    metrics["base_signal_forensics"] = _build_signal_forensics(
        {
            "uncertainty": uncertainties_t,
            "abs_base_margin": base_global_margin_t.abs(),
            "base_commitment_score": base_commitment_score_t,
            "base_commitment_threshold": base_commitment_threshold_t,
        },
        targets=targets_t,
        predictions=base_hard_predictions_t,
    )
    metrics["base_to_final_error_corr_delta"] = {
        "uncertainty_error_corr_delta": float(
            metrics.get("uncertainty_error_corr", float("nan"))
            - _vector_pearson_correlation(uncertainties_t, base_hard_incorrect_t)
        ),
        "base_uncertainty_error_corr": _vector_pearson_correlation(
            uncertainties_t,
            base_hard_incorrect_t,
        ),
        "final_uncertainty_error_corr": metrics.get("uncertainty_error_corr"),
    }

    margin_band = 0.05
    post_reader_bundle = {
        "pairwise_cosine_agreement": post_pairwise_cosine_agreement_t,
        "pairwise_cosine_agreement_std": post_pairwise_cosine_agreement_std_t,
        "mean_squared_distance_to_anchor": post_mean_squared_distance_to_anchor_t,
        "witness_latent_variance": post_witness_latent_variance_t,
        "witness_latent_norm_mean": post_witness_latent_norm_mean_t,
        "witness_latent_norm_std": post_witness_latent_norm_std_t,
        "witness_latent_spread": post_witness_latent_spread_t,
        "principal_singular_value_ratio": post_principal_singular_value_ratio_t,
        "witness_margin_mean": witness_margin_mean_t,
        "witness_margin_std": witness_margin_std_t,
        "coherence_score_mean": coherence_scores_t.mean(dim=-1),
        "coherence_score_std": coherence_scores_t.std(dim=-1, unbiased=False),
        "witness_weight_entropy": witness_weight_entropy_t,
        "witness_weight_max": witness_weight_max_t,
        "witness_weight_kl_uniform": witness_weight_kl_uniform_t,
        "witness_margin_range": witness_margins_t.max(dim=-1).values - witness_margins_t.min(dim=-1).values,
        "weighted_minus_unweighted_margin": global_margin_t - witness_margin_mean_t,
        "base_weighted_minus_unweighted_margin": base_global_margin_t - witness_margin_mean_t,
        "path_signed_margin_delta": global_margin_t - base_global_margin_t,
    }
    metrics["post_reader_witness_geometry"] = summarize_metric_bundle_with_strata(
        post_reader_bundle,
        strata,
    )
    metrics["witness_geometry"] = metrics["post_reader_witness_geometry"]

    margin_positive_fraction_t = (global_margin_t > 0).float()
    margin_negative_fraction_t = (global_margin_t < 0).float()
    margin_near_zero_fraction_t = (global_margin_t.abs() <= margin_band).float()
    witness_positive_count_t = (witness_margins_t > 0).sum(dim=-1).float()
    witness_negative_count_t = (witness_margins_t < 0).sum(dim=-1).float()
    witness_zero_count_t = (witness_margins_t == 0).sum(dim=-1).float()
    witness_count_t = torch.full_like(witness_positive_count_t, float(witness_margins_t.shape[-1]))
    witness_nonzero_count_t = (witness_positive_count_t + witness_negative_count_t).clamp_min(1.0)
    witness_sign_majority_fraction_t = torch.maximum(
        witness_positive_count_t, witness_negative_count_t
    ) / witness_count_t.clamp_min(1.0)
    witness_nonzero_sign_majority_fraction_t = torch.maximum(
        witness_positive_count_t, witness_negative_count_t
    ) / witness_nonzero_count_t
    witness_sign_disagreement_rate_t = (
        2.0 * witness_positive_count_t * witness_negative_count_t
    ) / witness_nonzero_count_t.clamp_min(2.0).mul(witness_nonzero_count_t.clamp_min(2.0) - 1.0)
    witness_sign_disagreement_rate_t = torch.where(
        witness_nonzero_count_t > 1.0,
        witness_sign_disagreement_rate_t,
        torch.zeros_like(witness_sign_disagreement_rate_t),
    )
    mixed_sign_witness_fraction_t = (
        (witness_positive_count_t > 0.0) & (witness_negative_count_t > 0.0)
    ).float()
    weighted_margin_sign_flip_rate_t = (
        (global_margin_t > 0) != (witness_margin_mean_t > 0)
    ).float()
    base_weighted_margin_sign_flip_rate_t = (
        (base_global_margin_t > 0) != (witness_margin_mean_t > 0)
    ).float()

    decision_geometry_bundle = {
        "margin_M": global_margin_t,
        "base_margin_M": base_global_margin_t,
        "final_minus_base_margin": global_margin_t - base_global_margin_t,
        "vote_disagreement_D": vote_disagreement_t,
        "geometric_disagreement_G": geometric_disagreement_t,
        "uncertainty_U": uncertainties_t,
        "success_minus_uncertain": global_margin_t - 2.0 * uncertainties_t,
        "failure_minus_uncertain": -global_margin_t - 2.0 * uncertainties_t,
        "abs_margin": global_margin_t.abs(),
        "margin_positive_fraction": margin_positive_fraction_t,
        "margin_negative_fraction": margin_negative_fraction_t,
        "margin_near_zero_fraction": margin_near_zero_fraction_t,
        "commitment_score": commitment_score_t,
        "base_commitment_score": base_commitment_score_t,
        "final_minus_base_commitment": commitment_score_t - base_commitment_score_t,
        "commitment_logit": commitment_logit_t,
        "base_commitment_logit": base_commitment_logit_t,
        "commitment_scale": commitment_scale_t,
        "lambda_commit": lambda_commit_t,
        "lambda_profile_risk": lambda_profile_risk_t if lambda_profile_risk_t is not None else torch.zeros_like(commitment_score_t),
        "profile_risk_temperature": profile_risk_temperature_t
        if profile_risk_temperature_t is not None
        else torch.zeros_like(commitment_score_t),
        "tau_uncertain": tau_uncertain_t,
        "base_commitment_threshold": base_commitment_threshold_t,
        "final_commitment_threshold": final_commitment_threshold_t,
        "uncertain_logit": uncertain_logit_t,
        "calibrated_risk": calibrated_risk_t
        if calibrated_risk_t is not None
        else torch.zeros_like(commitment_score_t),
        "risk_adjustment": risk_adjustment_t
        if risk_adjustment_t is not None
        else torch.zeros_like(commitment_score_t),
        "witness_sign_disagreement_rate": witness_sign_disagreement_rate_t,
        "mixed_sign_witness_fraction": mixed_sign_witness_fraction_t,
    }
    if path_disagreement_t is not None:
        assert path_confidence_t is not None
        assert endpoint_confidence_t is not None
        assert method_conflict_t is not None
        assert lambda_path_t is not None
        decision_geometry_bundle.update(
            {
                "path_disagreement": path_disagreement_t,
                "path_confidence": path_confidence_t,
                "endpoint_confidence": endpoint_confidence_t,
                "method_conflict": method_conflict_t,
                "lambda_path": lambda_path_t,
                "path_commitment_penalty": lambda_path_t * path_disagreement_t,
            }
        )
    if path_signed_margin_t is not None:
        decision_geometry_bundle.update(
            {
                "path_signed_margin": path_signed_margin_t,
                "path_signed_margin_abs": path_signed_margin_abs_t,
                "path_signed_margin_delta": path_signed_margin_delta_t,
            }
        )
    metrics["decision_geometry"] = summarize_metric_bundle_with_strata(
        decision_geometry_bundle,
        strata,
    )

    metrics["margin_channel_diagnostics"] = summarize_metric_bundle_with_strata(
        {
            "margin_positive_fraction": margin_positive_fraction_t,
            "margin_negative_fraction": margin_negative_fraction_t,
            "margin_near_zero_fraction": margin_near_zero_fraction_t,
            "witness_positive_count": witness_positive_count_t,
            "witness_negative_count": witness_negative_count_t,
            "witness_zero_count": witness_zero_count_t,
            "witness_sign_majority_fraction": witness_sign_majority_fraction_t,
            "witness_nonzero_sign_majority_fraction": witness_nonzero_sign_majority_fraction_t,
            "witness_sign_disagreement_rate": witness_sign_disagreement_rate_t,
            "mixed_sign_witness_fraction": mixed_sign_witness_fraction_t,
            "weighted_margin_sign_flip_rate": weighted_margin_sign_flip_rate_t,
            "base_weighted_margin_sign_flip_rate": base_weighted_margin_sign_flip_rate_t,
            "weighted_minus_unweighted_margin": global_margin_t - witness_margin_mean_t,
            "base_weighted_minus_unweighted_margin": base_global_margin_t - witness_margin_mean_t,
            "final_minus_base_margin": global_margin_t - base_global_margin_t,
        },
        strata,
    )

    failure_mask = targets_t == FAILURE_CLASS_IDX
    success_mask = targets_t == SUCCESS_CLASS_IDX
    metrics["failure_channel_diagnostics"] = _build_true_channel_diagnostics(
        channel_name="failure",
        mask=failure_mask,
        global_margin_t=global_margin_t,
        commitment_score_t=commitment_score_t,
        uncertainties_t=uncertainties_t,
        witness_positive_count_t=witness_positive_count_t,
        witness_negative_count_t=witness_negative_count_t,
        witness_sign_disagreement_rate_t=witness_sign_disagreement_rate_t,
        mixed_sign_witness_fraction_t=mixed_sign_witness_fraction_t,
        tau_uncertain_t=tau_uncertain_t,
        predictions_t=predictions_t,
        margin_band=margin_band,
    )
    metrics["success_channel_diagnostics"] = _build_true_channel_diagnostics(
        channel_name="success",
        mask=success_mask,
        global_margin_t=global_margin_t,
        commitment_score_t=commitment_score_t,
        uncertainties_t=uncertainties_t,
        witness_positive_count_t=witness_positive_count_t,
        witness_negative_count_t=witness_negative_count_t,
        witness_sign_disagreement_rate_t=witness_sign_disagreement_rate_t,
        mixed_sign_witness_fraction_t=mixed_sign_witness_fraction_t,
        tau_uncertain_t=tau_uncertain_t,
        predictions_t=predictions_t,
        margin_band=margin_band,
    )
    metrics["success_recovery_diagnostics"] = _build_success_recovery_diagnostics(metrics)

    if config.log_pre_reader_geometry and pre_pairwise_cosine_agreement:
        pre_reader_bundle = {
            "pairwise_cosine_agreement": torch.cat(pre_pairwise_cosine_agreement, dim=0),
            "pairwise_cosine_agreement_std": torch.cat(pre_pairwise_cosine_agreement_std, dim=0),
            "mean_squared_distance_to_anchor": torch.cat(pre_mean_squared_distance_to_anchor, dim=0),
            "witness_latent_variance": torch.cat(pre_witness_latent_variance, dim=0),
            "witness_latent_norm_mean": torch.cat(pre_witness_latent_norm_mean, dim=0),
            "witness_latent_norm_std": torch.cat(pre_witness_latent_norm_std, dim=0),
            "witness_latent_spread": torch.cat(pre_witness_latent_spread, dim=0),
            "principal_singular_value_ratio": torch.cat(pre_principal_singular_value_ratio, dim=0),
        }
        metrics["pre_reader_witness_geometry"] = summarize_metric_bundle_with_strata(
            pre_reader_bundle,
            strata,
        )

        pre_pairwise_agreement_t = pre_reader_bundle["pairwise_cosine_agreement"]
        reader_agreement_gain_t = post_pairwise_cosine_agreement_t - pre_pairwise_agreement_t
        reader_margin_std_ratio_t = witness_margin_std_t / pre_reader_bundle[
            "witness_latent_variance"
        ].clamp_min(1e-8).sqrt()
        metrics["diversity_retention"] = summarize_metric_bundle_with_strata(
            {
                "pre_reader_pairwise_agreement": pre_pairwise_agreement_t,
                "post_reader_pairwise_agreement": post_pairwise_cosine_agreement_t,
                "reader_agreement_gain": reader_agreement_gain_t,
                "pre_reader_mean_squared_distance_to_anchor": pre_reader_bundle[
                    "mean_squared_distance_to_anchor"
                ],
                "pre_reader_witness_latent_variance": pre_reader_bundle["witness_latent_variance"],
                "pre_reader_witness_latent_spread": pre_reader_bundle["witness_latent_spread"],
                "post_reader_mean_squared_distance_to_anchor": post_mean_squared_distance_to_anchor_t,
                "post_reader_witness_latent_variance": post_witness_latent_variance_t,
                "post_reader_witness_latent_spread": post_witness_latent_spread_t,
                "post_reader_witness_margin_std": witness_margin_std_t,
                "reader_margin_std_ratio": reader_margin_std_ratio_t,
                "reader_weighting_gain": witness_weight_kl_uniform_t,
            },
            strata,
        )

    if config.log_weighting_diagnostics:
        metrics["weighting_diagnostics"] = summarize_metric_bundle_with_strata(
            {
                "coherence_score_spread_before_softmax": coherence_scores_t.std(
                    dim=-1, unbiased=False
                ),
                "witness_weight_spread_after_softmax": witness_weights_t.std(
                    dim=-1, unbiased=False
                ),
                "coherence_margin_corr": _rowwise_pearson_correlation(
                    coherence_scores_t, witness_margins_t
                ),
                "coherence_abs_margin_corr": _rowwise_pearson_correlation(
                    coherence_scores_t, witness_margins_t.abs()
                ),
                "coherence_margin_abs_corr": _rowwise_pearson_correlation(
                    coherence_scores_t, witness_margins_t.abs()
                ),
                "witness_weight_kl_uniform": witness_weight_kl_uniform_t,
                "weighted_minus_unweighted_margin": global_margin_t - witness_margin_mean_t,
                "weighted_margin_sign_flip_rate": weighted_margin_sign_flip_rate_t,
            },
            strata,
        )

    if probe_order:
        bundle["probe_order"] = torch.cat(probe_order, dim=0)
        bundle["probe_boundary"] = torch.cat(probe_boundary, dim=0)
        bundle["probe_support"] = torch.cat(probe_support, dim=0)
        metrics["probes"] = {
            "order": summarize_tensor(bundle["probe_order"]),
            "boundary": summarize_tensor(bundle["probe_boundary"]),
            "support": summarize_tensor(bundle["probe_support"]),
        }

    return metrics, bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--rebuild-traces", action="store_true")
    parser.add_argument("--strict-existing-traces", action="store_true")
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-val-records", type=int, default=None)
    parser.add_argument("--max-test-records", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--train-loop-epochs", type=int, default=None)
    parser.add_argument("--num-cloud-samples", type=int, default=None)
    parser.add_argument("--witness-weight-temperature", type=float, default=None)
    parser.add_argument(
        "--point-reader-mode",
        type=str,
        choices=("identity", "linear_scalar_margin", "linear_shared_proj"),
        default=None,
    )
    parser.add_argument(
        "--gate-mode",
        type=str,
        choices=("soft_commitment_gate", "legacy_logits"),
        default=None,
    )
    parser.add_argument("--use-hard-eval-gate", dest="use_hard_eval_gate", action="store_true")
    parser.add_argument(
        "--disable-hard-eval-gate",
        dest="use_hard_eval_gate",
        action="store_false",
    )
    parser.set_defaults(use_hard_eval_gate=None)
    parser.add_argument("--commitment-scale-init", type=float, default=None)
    parser.add_argument("--tau-uncertain-init", type=float, default=None)
    parser.add_argument("--enable-diagnostic-probes", action="store_true")
    parser.add_argument("--log-pre-reader-geometry", action="store_true")
    parser.add_argument("--log-grouped-geometry", action="store_true")
    parser.add_argument("--log-weighting-diagnostics", action="store_true")
    parser.add_argument("--disable-vote-disagreement", action="store_true")
    parser.add_argument("--disable-geometric-disagreement", action="store_true")
    parser.add_argument("--disable-margin-term", action="store_true")
    parser.add_argument("--disable-coherence-weighting", action="store_true")
    parser.add_argument("--disable-path-uncertainty", action="store_true")
    parser.add_argument("--path-uncertainty-embedding-dim", type=int, default=None)
    parser.add_argument("--path-uncertainty-hidden-dim", type=int, default=None)
    parser.add_argument("--path-consistency-loss-weight", type=float, default=None)
    parser.add_argument("--path-contrastive-temperature", type=float, default=None)
    parser.add_argument("--enable-cooperative-path-uncertainty", action="store_true")
    parser.add_argument("--enable-path-disagreement-in-uncertainty", action="store_true")
    parser.add_argument("--enable-method-conflict-in-uncertainty", action="store_true")
    parser.add_argument("--enable-path-commitment-penalty", action="store_true")
    parser.add_argument("--enable-path-aware-witness-weights", action="store_true")
    parser.add_argument("--enable-path-signed-evidence", action="store_true")
    parser.add_argument(
        "--disable-path-signed-margin-in-hard-gate",
        action="store_true",
    )
    parser.add_argument("--path-signed-evidence-hidden-dim", type=int, default=None)
    parser.add_argument("--path-signed-logit-scale-init", type=float, default=None)
    parser.add_argument("--path-signed-margin-scale-init", type=float, default=None)
    parser.add_argument("--path-failure-loss-weight", type=float, default=None)
    parser.add_argument("--path-failure-positive-weight-scale", type=float, default=None)
    parser.add_argument("--path-failure-focal-gamma", type=float, default=None)
    parser.add_argument("--path-failure-gate-threshold", type=float, default=None)
    parser.add_argument("--path-outcome-loss-weight", type=float, default=None)
    parser.add_argument(
        "--primary-metric",
        type=str,
        choices=("macro_f1", "failure_recall", "balanced_commitment_primary", "balanced_margin_primary"),
        default=None,
    )
    parser.add_argument("--failure-class-weight-scale", type=float, default=None)
    parser.add_argument("--enable-disturbance-profile", action="store_true")
    parser.add_argument("--disturbance-num-samples", type=int, default=None)
    parser.add_argument("--disturbance-noise-std", type=float, default=None)
    parser.add_argument("--enable-boundary-probe", action="store_true")
    parser.add_argument("--enable-feature-mask-probe", action="store_true")
    parser.add_argument("--enable-learned-probe", action="store_true")
    parser.add_argument(
        "--detach-profile-from-base",
        dest="detach_profile_from_base",
        action="store_true",
    )
    parser.add_argument(
        "--attach-profile-to-base",
        dest="detach_profile_from_base",
        action="store_false",
    )
    parser.set_defaults(detach_profile_from_base=None)
    parser.add_argument(
        "--use-disturbance-risk-in-commitment-gate",
        dest="use_disturbance_risk_in_commitment_gate",
        action="store_true",
    )
    parser.add_argument(
        "--disable-disturbance-risk-in-commitment-gate",
        dest="use_disturbance_risk_in_commitment_gate",
        action="store_false",
    )
    parser.set_defaults(use_disturbance_risk_in_commitment_gate=None)
    parser.add_argument("--lambda-profile-risk", type=float, default=None)
    parser.add_argument("--profile-risk-temperature", type=float, default=None)
    parser.add_argument(
        "--profile-risk-input-mode",
        type=str,
        choices=("axis_scores", "normalized_profile", "raw_profile", "normalized_profile_plus_axes"),
        default=None,
    )
    parser.add_argument("--disable-uncertainty-axis-profile", action="store_true")
    parser.add_argument("--disable-support-novelty", action="store_true")
    parser.add_argument("--profile-support-k", type=int, default=None)
    parser.add_argument("--profile-support-max-items", type=int, default=None)
    parser.add_argument("--disable-profile-normalization", action="store_true")
    parser.add_argument("--profile-normalization-clip", type=float, default=None)
    parser.add_argument("--disable-calibration-features", action="store_true")
    parser.add_argument("--target-commit-rate-min", type=float, default=None)
    parser.add_argument("--target-commit-rate-max", type=float, default=None)
    parser.add_argument("--profile-loss-weight", type=float, default=None)
    parser.add_argument("--risk-loss-weight", type=float, default=None)
    parser.add_argument(
        "--risk-target-mode",
        type=str,
        choices=("incorrect", "incorrect_only", "uncertain_label", "incorrect_or_uncertain"),
        default=None,
    )
    parser.add_argument("--commit-risk-threshold", type=float, default=None)
    parser.add_argument("--min-clean-margin", type=float, default=None)
    parser.add_argument("--disturbance-feature-mask-prob", type=float, default=None)
    parser.add_argument("--disturbance-boundary-epsilon", type=float, default=None)
    parser.add_argument("--disturbance-profile-hidden-dim", type=int, default=None)
    parser.add_argument("--disturbance-profile-dropout", type=float, default=None)
    parser.add_argument("--disturbance-learned-hidden-dim", type=int, default=None)
    parser.add_argument("--config-override", action="append", default=None)
    args = parser.parse_args()

    config = E31Config()
    _apply_config_overrides(config, args.config_override)
    config.train_loop_seed = int(args.seed)
    if args.device is not None:
        config.device = args.device
    if args.batch_size is not None:
        config.batch_size = int(args.batch_size)
    if args.train_loop_epochs is not None:
        config.train_loop_epochs = int(args.train_loop_epochs)
    if args.num_cloud_samples is not None:
        config.num_cloud_samples = int(args.num_cloud_samples)
    if args.witness_weight_temperature is not None:
        config.witness_weight_temperature = float(args.witness_weight_temperature)
    if args.point_reader_mode is not None:
        config.point_reader_mode = args.point_reader_mode
    if args.gate_mode is not None:
        config.gate_mode = args.gate_mode
    if args.use_hard_eval_gate is not None:
        config.use_hard_eval_gate = bool(args.use_hard_eval_gate)
    if args.commitment_scale_init is not None:
        config.commitment_scale_init = float(args.commitment_scale_init)
    if args.tau_uncertain_init is not None:
        config.tau_uncertain_init = float(args.tau_uncertain_init)
    if args.enable_diagnostic_probes:
        config.enable_diagnostic_probes = True
    if args.log_pre_reader_geometry:
        config.log_pre_reader_geometry = True
    if args.log_grouped_geometry:
        config.log_grouped_geometry = True
    if args.log_weighting_diagnostics:
        config.log_weighting_diagnostics = True
    if args.disable_vote_disagreement:
        config.use_vote_disagreement = False
    if args.disable_geometric_disagreement:
        config.use_geometric_disagreement = False
    if args.disable_margin_term:
        config.use_margin_term = False
    if args.disable_coherence_weighting:
        config.use_coherence_weighting = False
    if args.disable_path_uncertainty:
        config.enable_path_uncertainty = False
        config.path_consistency_loss_weight = 0.0
    if args.path_uncertainty_embedding_dim is not None:
        config.path_uncertainty_embedding_dim = int(args.path_uncertainty_embedding_dim)
    if args.path_uncertainty_hidden_dim is not None:
        config.path_uncertainty_hidden_dim = int(args.path_uncertainty_hidden_dim)
    if args.path_consistency_loss_weight is not None:
        config.path_consistency_loss_weight = float(args.path_consistency_loss_weight)
    if args.path_contrastive_temperature is not None:
        config.path_contrastive_temperature = float(args.path_contrastive_temperature)
    if args.enable_cooperative_path_uncertainty:
        config.use_path_disagreement_in_uncertainty = True
        config.use_method_conflict_in_uncertainty = True
        config.use_path_commitment_penalty = True
        config.use_path_aware_witness_weights = True
    if args.enable_path_disagreement_in_uncertainty:
        config.use_path_disagreement_in_uncertainty = True
    if args.enable_method_conflict_in_uncertainty:
        config.use_method_conflict_in_uncertainty = True
    if args.enable_path_commitment_penalty:
        config.use_path_commitment_penalty = True
    if args.enable_path_aware_witness_weights:
        config.use_path_aware_witness_weights = True
    if args.enable_path_signed_evidence:
        config.enable_path_signed_evidence = True
        config.use_path_disagreement_in_uncertainty = True
        config.use_method_conflict_in_uncertainty = True
        config.use_path_commitment_penalty = True
        config.use_path_aware_witness_weights = True
    if args.disable_path_signed_margin_in_hard_gate:
        config.use_path_signed_margin_in_hard_gate = False
    if args.path_signed_evidence_hidden_dim is not None:
        config.path_signed_evidence_hidden_dim = int(args.path_signed_evidence_hidden_dim)
    if args.path_signed_logit_scale_init is not None:
        config.path_signed_logit_scale_init = float(args.path_signed_logit_scale_init)
    if args.path_signed_margin_scale_init is not None:
        config.path_signed_margin_scale_init = float(args.path_signed_margin_scale_init)
    if args.path_failure_loss_weight is not None:
        config.path_failure_loss_weight = float(args.path_failure_loss_weight)
    if args.path_failure_positive_weight_scale is not None:
        config.path_failure_positive_weight_scale = float(args.path_failure_positive_weight_scale)
    if args.path_failure_focal_gamma is not None:
        config.path_failure_focal_gamma = float(args.path_failure_focal_gamma)
    if args.path_failure_gate_threshold is not None:
        config.path_failure_gate_threshold = float(args.path_failure_gate_threshold)
    if args.path_outcome_loss_weight is not None:
        config.path_outcome_loss_weight = float(args.path_outcome_loss_weight)
    if args.primary_metric is not None:
        config.primary_metric = args.primary_metric
    if args.failure_class_weight_scale is not None:
        config.failure_class_weight_scale = float(args.failure_class_weight_scale)
    if args.enable_disturbance_profile:
        config.enable_disturbance_profile = True
    if args.disturbance_num_samples is not None:
        config.disturbance_num_samples = int(args.disturbance_num_samples)
    if args.disturbance_noise_std is not None:
        config.disturbance_noise_std = float(args.disturbance_noise_std)
    if args.enable_boundary_probe:
        config.enable_boundary_probe = True
    if args.enable_feature_mask_probe:
        config.enable_feature_mask_probe = True
    if args.enable_learned_probe:
        config.enable_learned_probe = True
    if args.detach_profile_from_base is not None:
        config.detach_profile_from_base = bool(args.detach_profile_from_base)
    if args.use_disturbance_risk_in_commitment_gate is not None:
        config.use_disturbance_risk_in_commitment_gate = bool(
            args.use_disturbance_risk_in_commitment_gate
        )
    if args.lambda_profile_risk is not None:
        config.lambda_profile_risk = float(args.lambda_profile_risk)
    if args.profile_risk_temperature is not None:
        config.profile_risk_temperature = float(args.profile_risk_temperature)
    if args.profile_risk_input_mode is not None:
        config.profile_risk_input_mode = args.profile_risk_input_mode
    if args.disable_uncertainty_axis_profile:
        config.enable_uncertainty_axis_profile = False
    if args.disable_support_novelty:
        config.enable_support_novelty = False
    if args.profile_support_k is not None:
        config.profile_support_k = int(args.profile_support_k)
    if args.profile_support_max_items is not None:
        config.profile_support_max_items = int(args.profile_support_max_items)
    if args.disable_profile_normalization:
        config.enable_profile_normalization = False
    if args.profile_normalization_clip is not None:
        config.profile_normalization_clip = float(args.profile_normalization_clip)
    if args.disable_calibration_features:
        config.use_calibration_features = False
    if args.target_commit_rate_min is not None:
        config.target_commit_rate_min = float(args.target_commit_rate_min)
    if args.target_commit_rate_max is not None:
        config.target_commit_rate_max = float(args.target_commit_rate_max)
    if args.profile_loss_weight is not None:
        config.profile_loss_weight = float(args.profile_loss_weight)
    if args.risk_loss_weight is not None:
        config.risk_loss_weight = float(args.risk_loss_weight)
    if args.risk_target_mode is not None:
        config.risk_target_mode = args.risk_target_mode
    if args.commit_risk_threshold is not None:
        config.commit_risk_threshold = float(args.commit_risk_threshold)
    if args.min_clean_margin is not None:
        config.min_clean_margin = float(args.min_clean_margin)
    if args.disturbance_feature_mask_prob is not None:
        config.disturbance_feature_mask_prob = float(args.disturbance_feature_mask_prob)
    if args.disturbance_boundary_epsilon is not None:
        config.disturbance_boundary_epsilon = float(args.disturbance_boundary_epsilon)
    if args.disturbance_profile_hidden_dim is not None:
        config.disturbance_profile_hidden_dim = int(args.disturbance_profile_hidden_dim)
    if args.disturbance_profile_dropout is not None:
        config.disturbance_profile_dropout = float(args.disturbance_profile_dropout)
    if args.disturbance_learned_hidden_dim is not None:
        config.disturbance_learned_hidden_dim = int(args.disturbance_learned_hidden_dim)

    output_dir = _resolve_output_dir(config, args.seed, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.device if torch.cuda.is_available() and config.device.startswith("cuda") else "cpu")
    set_global_determinism(config.train_loop_seed)

    if args.rebuild_traces:
        raise NotImplementedError(
            "--rebuild-traces is not supported in standalone e32. Use the local or fallback trace cache."
        )

    trace_dir = _resolve_trace_dir(config)
    trace_info = ensure_trace_corpus(
        config,
        root_dir=trace_dir,
        device=device,
        rebuild=args.rebuild_traces,
        strict_existing=args.strict_existing_traces,
    )
    trace_dir = Path(trace_info["trace_dir"])

    train_records = _limit_records(load_trace_records(trace_dir / "train.jsonl"), args.max_train_records)
    val_records = _limit_records(load_trace_records(trace_dir / "val.jsonl"), args.max_val_records)
    test_records = _limit_records(load_trace_records(trace_dir / "test.jsonl"), args.max_test_records)

    train_loader = _make_loader(train_records, config, shuffle=True)
    val_loader = _make_loader(val_records, config, shuffle=False)
    test_loader = _make_loader(test_records, config, shuffle=False)

    train_dataset = TraceWitnessDataset(train_records, vocab=Vocabulary(), config=config)
    outcome_weights = train_dataset.get_class_weights(
        power=config.trace_class_weight_power,
        clip=config.trace_class_weight_clip,
    )["outcome"]
    train_outcome_counts = Counter(record["outcome"] for record in train_records)
    failure_count = int(train_outcome_counts.get(2, 0))
    non_failure_count = max(len(train_records) - failure_count, 1)
    failure_count = max(failure_count, 1)
    path_failure_pos_weight = (
        float(non_failure_count / failure_count) * float(config.path_failure_positive_weight_scale)
    )
    path_failure_pos_weight = max(1.0, min(path_failure_pos_weight, 20.0))

    model = E31Model(config).to(device)
    criterion = E31Loss(
        config,
        outcome_class_weights=outcome_weights,
        path_failure_pos_weight=path_failure_pos_weight,
    ).to(device)
    current_stage = _disturbance_stage(1, config.train_loop_epochs, config)
    model.set_training_stage(current_stage)
    lazy_status_before_warmup = _lazy_module_status(model)
    print(
        "lazy_module_status_before_warmup "
        f"stage={current_stage} "
        f"{_lazy_status_message(lazy_status_before_warmup)}"
    )
    lazy_status_after_warmup = model.materialize_lazy_modules(device=device)
    print(
        "lazy_module_status_after_warmup "
        f"stage={current_stage} "
        f"{_lazy_status_message(lazy_status_after_warmup)}"
    )
    if lazy_status_after_warmup["uninitialized_parameters"] or lazy_status_after_warmup[
        "uninitialized_buffers"
    ]:
        raise RuntimeError(
            "Lazy-module warm-up did not fully materialize the model: "
            f"{_lazy_status_message(lazy_status_after_warmup)}"
        )
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    use_early_stopping = not bool(config.enable_disturbance_profile)

    best_state = copy.deepcopy(model.state_dict())
    best_state_lazy_status = _state_dict_lazy_status(best_state)
    print(
        "best_state_snapshot "
        f"stage={current_stage} "
        f"{_lazy_status_message(best_state_lazy_status)}"
    )
    best_primary_metric = float("-inf")
    best_val_metrics: dict[str, Any] = {}
    patience = 0
    epoch_history = []

    for epoch in range(1, config.train_loop_epochs + 1):
        stage = _disturbance_stage(epoch, config.train_loop_epochs, config)
        if stage != current_stage:
            if stage == "profile":
                _fit_uncertainty_profile_state(model, train_loader, device, config)
            current_stage = stage
            model.set_training_stage(stage)
            if stage == "finetune":
                _fit_uncertainty_profile_state(model, train_loader, device, config)
            optimizer = AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        model.train()
        total_loss = 0.0
        total_items = 0

        for batch in train_loader:
            queries, trace_inputs, evidence_targets, prim_targets = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(queries, trace_inputs=trace_inputs)
            losses = criterion(outputs, prim_targets, evidence_targets)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip))
            optimizer.step()

            batch_size = queries.size(0)
            total_loss += float(losses["total"].item()) * batch_size
            total_items += batch_size

        train_metrics, _ = _evaluate(model, criterion, train_loader, device, train_records, config)
        val_metrics, _ = _evaluate(model, criterion, val_loader, device, val_records, config)
        train_metrics["loss"] = total_loss / max(total_items, 1)
        train_primary_metric = _primary_metric_value(train_metrics, config.primary_metric)
        val_primary_metric = _primary_metric_value(val_metrics, config.primary_metric)
        epoch_history.append(
            {
                "epoch": epoch,
                "stage": stage,
                "train_outcome_acc": train_metrics["outcome_acc"],
                "train_outcome_macro_f1": train_metrics["outcome_macro_f1"],
                "val_outcome_acc": val_metrics["outcome_acc"],
                "val_outcome_macro_f1": val_metrics["outcome_macro_f1"],
                "train_primary_metric": train_primary_metric,
                "val_primary_metric": val_primary_metric,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
            }
        )
        print(
            f"epoch={epoch:02d} stage={stage} "
            f"train_acc={train_metrics['outcome_acc']:.3f} "
            f"train_f1={train_metrics['outcome_macro_f1']:.3f} "
            f"val_acc={val_metrics['outcome_acc']:.3f} "
            f"val_f1={val_metrics['outcome_macro_f1']:.3f} "
            f"val_primary={val_primary_metric:.3f}"
        )

        selection_metric = float(val_primary_metric)
        if selection_metric > best_primary_metric:
            best_primary_metric = selection_metric
            best_val_metrics = val_metrics
            best_state = copy.deepcopy(model.state_dict())
            best_state_lazy_status = _state_dict_lazy_status(best_state)
            print(
                "best_state_updated "
                f"epoch={epoch:02d} stage={stage} "
                f"selection_metric={selection_metric:.3f} "
                f"{_lazy_status_message(best_state_lazy_status)}"
            )
            patience = 0
        elif use_early_stopping:
            patience += 1
            if patience >= config.train_loop_patience:
                break

    restore_lazy_status = _lazy_module_status(model)
    best_state_lazy_status = _state_dict_lazy_status(best_state)
    print(
        "checkpoint_restore_start "
        f"{_lazy_status_message(restore_lazy_status)} "
        f"best_state_{_lazy_status_message(best_state_lazy_status)}"
    )
    if restore_lazy_status["uninitialized_parameters"] or restore_lazy_status["uninitialized_buffers"]:
        restore_lazy_status = model.materialize_lazy_modules(device=device)
        print(
            "checkpoint_restore_warmup "
            f"{_lazy_status_message(restore_lazy_status)}"
        )
        if restore_lazy_status["uninitialized_parameters"] or restore_lazy_status["uninitialized_buffers"]:
            raise RuntimeError(
                "Checkpoint restore still has uninitialized lazy state after warm-up: "
                f"{_lazy_status_message(restore_lazy_status)}"
            )
    model.load_state_dict(best_state)
    final_profile_state = _fit_uncertainty_profile_state(model, train_loader, device, config)
    print(
        "checkpoint_restore_done "
        f"{_lazy_status_message(_lazy_module_status(model))}"
    )
    train_metrics, _ = _evaluate(model, criterion, train_loader, device, train_records, config)
    val_metrics, _ = _evaluate(model, criterion, val_loader, device, val_records, config)
    test_metrics, test_bundle = _evaluate(model, criterion, test_loader, device, test_records, config)
    best_val_metrics = val_metrics
    best_primary_metric = _primary_metric_value(val_metrics, config.primary_metric)
    baselines = evaluate_trace_baselines(train_records, test_records, config=config)
    report_highlights = _build_report_highlights(test_metrics)
    derived_training_stats = {
        "train_outcome_counts": dict(train_outcome_counts),
        "failure_count": failure_count,
        "non_failure_count": non_failure_count,
        "path_failure_pos_weight": path_failure_pos_weight,
        "lazy_module_warmup": {
            "before": lazy_status_before_warmup,
            "after": lazy_status_after_warmup,
        },
        "checkpoint_restore_status": {
            "model": restore_lazy_status,
            "best_state": best_state_lazy_status,
        },
        "disturbance_profile_enabled": bool(config.enable_disturbance_profile),
        "detach_profile_from_base": bool(getattr(config, "detach_profile_from_base", False)),
        "profile_risk_input_mode": config.profile_risk_input_mode,
        "enable_support_novelty": bool(config.enable_support_novelty),
        "profile_support_k": int(config.profile_support_k),
        "profile_support_max_items": int(config.profile_support_max_items),
        "enable_profile_normalization": bool(config.enable_profile_normalization),
        "profile_normalization_clip": float(config.profile_normalization_clip),
        "use_calibration_features": bool(config.use_calibration_features),
        "use_disturbance_risk_in_commitment_gate": bool(
            config.use_disturbance_risk_in_commitment_gate
        ),
        "lambda_profile_risk": config.lambda_profile_risk,
        "profile_risk_temperature": config.profile_risk_temperature,
        "target_commit_rate_min": config.target_commit_rate_min,
        "target_commit_rate_max": config.target_commit_rate_max,
        "risk_target_mode": config.risk_target_mode,
        "commit_risk_threshold": config.commit_risk_threshold,
        "uncertainty_profile_state": final_profile_state,
        "disturbance_stage_schedule": {
            "base_epochs": max(1, int(config.train_loop_epochs) // 3)
            if config.enable_disturbance_profile
            else int(config.train_loop_epochs),
            "profile_epochs": max(1, int(config.train_loop_epochs) // 3)
            if config.enable_disturbance_profile
            else 0,
            "finetune_epochs": max(
                0,
                int(config.train_loop_epochs)
                - 2 * max(1, int(config.train_loop_epochs) // 3),
            )
            if config.enable_disturbance_profile
            else 0,
        },
    }

    metrics = {
        "experiment": config.experiment_name,
        "trace_info": trace_info,
        "config": asdict(config),
        "primary_metric": config.primary_metric,
        "best_primary_metric": best_primary_metric,
        "best_val_metrics": best_val_metrics,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "report_highlights": report_highlights,
        "derived_training_stats": derived_training_stats,
        "baselines": baselines,
        "epoch_history": epoch_history,
    }
    summary = {
        "experiment": config.experiment_name,
        "derived_training_stats": derived_training_stats,
        "decision_rule": {
            "margin": "M = sum_k omega_k * m_k",
            "vote_disagreement": "D = sum_k omega_k * (m_k - M)^2",
            "geometric_disagreement": "G = 1 - mean pairwise cosine agreement",
            "path_disagreement": "P = 1 - cosine(path_embed_a, path_embed_b)",
            "method_conflict": "R = abs(normalized(abs(M)) - normalized(path_confidence))",
            "path_signed_evidence": (
                "H_path_diag = MLP(mean(path_embed_a, path_embed_b)); "
                "H_path_fail = MLP(mean(path_embed_a, path_embed_b)); "
                "p_fail = sigmoid(H_path_fail); "
                "tau_fail = fixed gate threshold; "
                "s_fail = relu(p_fail - tau_fail); "
                "M_final = M - s_margin * s_fail; "
                "logits_final = logits_readout with failure_logit += s_logit * s_fail"
            ),
            "uncertainty": (
                "U = softplus(alpha * D + beta * G + delta * P + rho * R "
                "- gamma * abs(M) + b)"
            ),
            "soft_commitment_gate": {
                "commitment_score": "C = abs(M) - lambda_commit * U - lambda_path * P",
                "commitment_logit": "K = commitment_scale * (C - tau_uncertain)",
                "uncertain": "softplus(-K)",
                "success": "K + M - softplus(-K)",
                "failure": "K - M - softplus(-K)",
            },
            "risk_adjusted_gate": (
                "C_final = C_base - lambda_profile_risk * calibrated_risk; "
                "if C_final < commit_risk_threshold then UNCERTAIN else sign(M)"
            ),
            "hard_eval_gate": "if C_final < commit_risk_threshold then UNCERTAIN else sign(M)",
            "legacy_logits": {
                "success": "M - U",
                "uncertain": "U",
                "failure": "-M - U",
            },
        },
        "ablation_toggles": {
            "point_reader_mode": config.point_reader_mode,
            "gate_mode": config.gate_mode,
            "use_hard_eval_gate": config.use_hard_eval_gate,
            "use_vote_disagreement": config.use_vote_disagreement,
            "use_geometric_disagreement": config.use_geometric_disagreement,
            "use_margin_term": config.use_margin_term,
            "use_coherence_weighting": config.use_coherence_weighting,
            "enable_path_uncertainty": config.enable_path_uncertainty,
            "path_uncertainty_embedding_dim": config.path_uncertainty_embedding_dim,
            "path_uncertainty_hidden_dim": config.path_uncertainty_hidden_dim,
            "path_contrastive_temperature": config.path_contrastive_temperature,
            "path_consistency_loss_weight": config.path_consistency_loss_weight,
            "use_path_disagreement_in_uncertainty": config.use_path_disagreement_in_uncertainty,
            "use_method_conflict_in_uncertainty": config.use_method_conflict_in_uncertainty,
            "use_path_commitment_penalty": config.use_path_commitment_penalty,
            "use_path_aware_witness_weights": config.use_path_aware_witness_weights,
            "enable_path_signed_evidence": config.enable_path_signed_evidence,
            "use_path_signed_margin_in_hard_gate": config.use_path_signed_margin_in_hard_gate,
            "path_signed_evidence_hidden_dim": config.path_signed_evidence_hidden_dim,
            "path_signed_logit_scale_init": config.path_signed_logit_scale_init,
            "path_signed_margin_scale_init": config.path_signed_margin_scale_init,
            "path_failure_loss_weight": config.path_failure_loss_weight,
            "path_failure_positive_weight_scale": config.path_failure_positive_weight_scale,
            "path_failure_focal_gamma": config.path_failure_focal_gamma,
            "path_failure_gate_threshold": config.path_failure_gate_threshold,
            "path_outcome_loss_weight": config.path_outcome_loss_weight,
            "num_cloud_samples": config.num_cloud_samples,
            "witness_weight_temperature": config.witness_weight_temperature,
            "commitment_scale_init": config.commitment_scale_init,
            "tau_uncertain_init": config.tau_uncertain_init,
            "failure_class_weight_scale": config.failure_class_weight_scale,
            "primary_metric": config.primary_metric,
            "enable_diagnostic_probes": config.enable_diagnostic_probes,
            "log_pre_reader_geometry": config.log_pre_reader_geometry,
            "log_grouped_geometry": config.log_grouped_geometry,
            "log_weighting_diagnostics": config.log_weighting_diagnostics,
            "enable_disturbance_profile": config.enable_disturbance_profile,
            "detach_profile_from_base": config.detach_profile_from_base,
            "enable_uncertainty_axis_profile": config.enable_uncertainty_axis_profile,
            "profile_risk_input_mode": config.profile_risk_input_mode,
            "enable_support_novelty": config.enable_support_novelty,
            "profile_support_k": config.profile_support_k,
            "profile_support_max_items": config.profile_support_max_items,
            "enable_profile_normalization": config.enable_profile_normalization,
            "profile_normalization_clip": config.profile_normalization_clip,
            "use_calibration_features": config.use_calibration_features,
            "use_disturbance_risk_in_commitment_gate": config.use_disturbance_risk_in_commitment_gate,
            "lambda_profile_risk": config.lambda_profile_risk,
            "profile_risk_temperature": config.profile_risk_temperature,
            "target_commit_rate_min": config.target_commit_rate_min,
            "target_commit_rate_max": config.target_commit_rate_max,
            "disturbance_num_samples": config.disturbance_num_samples,
            "disturbance_noise_std": config.disturbance_noise_std,
            "enable_boundary_probe": config.enable_boundary_probe,
            "enable_feature_mask_probe": config.enable_feature_mask_probe,
            "enable_learned_probe": config.enable_learned_probe,
            "disturbance_feature_mask_prob": config.disturbance_feature_mask_prob,
            "disturbance_boundary_epsilon": config.disturbance_boundary_epsilon,
            "disturbance_profile_hidden_dim": config.disturbance_profile_hidden_dim,
            "disturbance_profile_dropout": config.disturbance_profile_dropout,
            "disturbance_learned_hidden_dim": config.disturbance_learned_hidden_dim,
            "profile_loss_weight": config.profile_loss_weight,
            "risk_loss_weight": config.risk_loss_weight,
            "risk_target_mode": config.risk_target_mode,
            "commit_risk_threshold": config.commit_risk_threshold,
            "min_clean_margin": config.min_clean_margin,
        },
        "selection": {
            "primary_metric": config.primary_metric,
            "best_primary_metric": best_primary_metric,
            "balanced_margin_primary_formula": (
                "outcome_macro_f1 + 0.25 * failure_recall + 0.15 * success_recall "
                "- max(0.0, 0.08 - success_share) - max(0.0, 0.08 - failure_share) "
                "- max(0.0, uncertain_share - 0.70) - 0.5 * max(0.0, 0.15 - uncertain_share) "
                "- 0.5 * max(0.0, dominant_class_share - 0.75) - 0.5 * max(0.0, 0.10 - margin_positive_fraction)"
            ),
            "balanced_commitment_primary_formula": (
                "outcome_macro_f1 + 0.35 * failure_recall - max(0.0, failure_share - 0.35) "
                "- max(0.0, uncertain_share - 0.65) - 0.5 * max(0.0, dominant_class_share - 0.75)"
            ),
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "report_highlights": report_highlights,
        **_summary_diagnostic_sections(test_metrics),
        "baselines": baselines,
    }

    with open(output_dir / "metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)
    with open(output_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    torch.save({"model_state_dict": model.state_dict(), "config": asdict(config)}, output_dir / "model.pt")

    risk_sweep = test_metrics.get("risk_gate_threshold_sweep")
    if isinstance(risk_sweep, dict):
        selected = risk_sweep.get("selected_operating_point", {})
        success_diag = test_metrics.get("success_recovery_diagnostics", {})
        print(
            "risk_gate_summary "
            f"operating_threshold={risk_sweep.get('operating_threshold')} "
            f"selected_threshold={risk_sweep.get('selected_threshold')} "
            f"selected_commit_rate={selected.get('commit_rate')} "
            f"selected_macro_f1={selected.get('outcome_macro_f1')} "
            f"risk_auroc={report_highlights.get('risk_auroc')} "
            f"base_to_final_hard_flip_mean={report_highlights.get('base_to_final_hard_flip_mean')} "
            f"margin_positive_fraction_mean={report_highlights.get('margin_positive_fraction_mean')} "
            f"success_blocker={success_diag.get('dominant_reason')} "
            f"selection_reason={risk_sweep.get('selection_reason')} "
            f"blockers={success_diag.get('blockers')}"
        )
        for row in risk_sweep.get("rows", []):
            print(
                "risk_gate_sweep "
                f"threshold={row.get('threshold')} "
                f"commit_rate={row.get('commit_rate')} "
                f"coverage={row.get('coverage')} "
                f"macro_f1={row.get('outcome_macro_f1')} "
                f"success_recall={row.get('success_recall')} "
                f"failure_recall={row.get('failure_recall')} "
                f"uncertain_recall={row.get('uncertain_recall')}"
            )

    print(f"Wrote {output_dir / 'metrics.json'}")
    print(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
