"""Train and evaluate the E30 witness-agreement uncertainty experiment."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import asdict, fields
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

from experiments.DIGIT.Extrapolation.e30.extrapolation.config import E30Config
from experiments.DIGIT.Extrapolation.e30.extrapolation.data.ground_truth import OUTCOME_LABELS
from experiments.DIGIT.Extrapolation.e30.extrapolation.data.trace_dataset import (
    TraceWitnessDataset,
    load_trace_records,
    trace_collate_fn,
)
from experiments.DIGIT.Extrapolation.e30.extrapolation.data.vocabulary import Vocabulary
from experiments.DIGIT.Extrapolation.e30.extrapolation.losses import E30Loss
from experiments.DIGIT.Extrapolation.e30.extrapolation.metrics import (
    build_stratified_masks,
    compute_outcome_metrics,
    summarize_metric_bundle_with_strata,
    summarize_tensor,
)
from experiments.DIGIT.Extrapolation.e30.extrapolation.trace_pipeline import (
    ensure_trace_corpus,
    evaluate_trace_baselines,
    set_global_determinism,
)
from experiments.DIGIT.Extrapolation.e30.extrapolation.models.digit import E30Model
from experiments.DIGIT.Extrapolation.e30.extrapolation.models.readout import (
    bucketize_hard_commitment_decisions,
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


def _config_field_names() -> set[str]:
    return {field.name for field in fields(E30Config)}


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


def _apply_config_overrides(config: E30Config, overrides: list[str] | None) -> None:
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


def _resolve_trace_dir(config: E30Config) -> Path:
    trace_dir = Path(config.trace_dir)
    if trace_dir.is_absolute():
        return trace_dir
    return (VARIANT_DIR / trace_dir).resolve()


def _resolve_output_dir(config: E30Config, seed: int, explicit_output_dir: str | None) -> Path:
    if explicit_output_dir:
        output_dir = Path(explicit_output_dir)
        if output_dir.is_absolute():
            return output_dir
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
        "class_shares": test_metrics["class_shares"],
        "failure_recall": test_metrics["failure_recall"],
        "success_recall": test_metrics.get("success_recall"),
        "uncertainty_error_corr": test_metrics.get("uncertainty_error_corr"),
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
        "weighting_diagnostics": test_metrics.get("weighting_diagnostics"),
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


def _make_loader(records: list[dict[str, Any]], config: E30Config, shuffle: bool) -> DataLoader:
    dataset = TraceWitnessDataset(records, vocab=Vocabulary(), config=config)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=trace_collate_fn,
    )


def _evaluate(
    model: E30Model,
    criterion: E30Loss,
    loader: DataLoader,
    device: torch.device,
    records: list[dict[str, Any]],
    config: E30Config,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    logits = []
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
    vote_disagreement = []
    geometric_disagreement = []
    commitment_score = []
    commitment_logit = []
    commitment_scale = []
    lambda_commit = []
    tau_uncertain = []
    uncertain_logit = []
    margin_minus_lambda_u = []
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
    probe_order = []
    probe_boundary = []
    probe_support = []

    with torch.no_grad():
        for batch in loader:
            queries, trace_inputs, evidence_targets, prim_targets = _move_batch(batch, device)
            outputs = model(queries, trace_inputs=trace_inputs)
            losses = criterion(outputs, prim_targets, evidence_targets)
            batch_size = queries.size(0)
            total_loss += float(losses["total"].item()) * batch_size
            total_items += batch_size

            logits.append(outputs.outcome_logits.cpu())
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
            global_margin.append(stats.global_margin.cpu())
            vote_disagreement.append(stats.vote_disagreement.cpu())
            geometric_disagreement.append(stats.geometric_disagreement.cpu())

            decision_inputs = outputs.decision_inputs
            commitment_score.append(decision_inputs["commitment_score"].cpu())
            commitment_logit.append(decision_inputs["commitment_logit"].cpu())
            commitment_scale.append(decision_inputs["commitment_scale"].cpu())
            lambda_commit.append(decision_inputs["lambda_commit"].cpu())
            tau_uncertain.append(decision_inputs["tau_uncertain"].cpu())
            uncertain_logit.append(decision_inputs["uncertain_logit"].cpu())
            margin_minus_lambda_u.append(decision_inputs["margin_minus_lambda_u"].cpu())

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

    logits_t = torch.cat(logits, dim=0)
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
    vote_disagreement_t = torch.cat(vote_disagreement, dim=0)
    geometric_disagreement_t = torch.cat(geometric_disagreement, dim=0)
    commitment_score_t = torch.cat(commitment_score, dim=0)
    commitment_logit_t = torch.cat(commitment_logit, dim=0)
    commitment_scale_t = torch.cat(commitment_scale, dim=0)
    lambda_commit_t = torch.cat(lambda_commit, dim=0)
    tau_uncertain_t = torch.cat(tau_uncertain, dim=0)
    uncertain_logit_t = torch.cat(uncertain_logit, dim=0)
    margin_minus_lambda_u_t = torch.cat(margin_minus_lambda_u, dim=0)
    post_pairwise_cosine_agreement_t = torch.cat(post_pairwise_cosine_agreement, dim=0)
    post_pairwise_cosine_agreement_std_t = torch.cat(post_pairwise_cosine_agreement_std, dim=0)
    post_mean_squared_distance_to_anchor_t = torch.cat(post_mean_squared_distance_to_anchor, dim=0)
    post_witness_latent_variance_t = torch.cat(post_witness_latent_variance, dim=0)
    post_witness_latent_norm_mean_t = torch.cat(post_witness_latent_norm_mean, dim=0)
    post_witness_latent_norm_std_t = torch.cat(post_witness_latent_norm_std, dim=0)
    post_witness_latent_spread_t = torch.cat(post_witness_latent_spread, dim=0)
    post_principal_singular_value_ratio_t = torch.cat(post_principal_singular_value_ratio, dim=0)

    if config.use_hard_eval_gate:
        predictions_t = bucketize_hard_commitment_decisions(
            global_margin_t,
            commitment_score_t,
            tau_uncertain_t,
        )
    else:
        predictions_t = bucketize_outcome_logits(logits_t)

    bundle: dict[str, torch.Tensor] = {
        "outcome_logits": logits_t,
        "outcome_targets": targets_t,
        "predictions": predictions_t,
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
        "vote_disagreement": vote_disagreement_t,
        "geometric_disagreement": geometric_disagreement_t,
        "commitment_score": commitment_score_t,
        "commitment_logit": commitment_logit_t,
        "commitment_scale": commitment_scale_t,
        "lambda_commit": lambda_commit_t,
        "tau_uncertain": tau_uncertain_t,
        "uncertain_logit": uncertain_logit_t,
        "margin_minus_lambda_u": margin_minus_lambda_u_t,
    }

    metrics = compute_outcome_metrics(
        logits_t,
        targets_t,
        predictions=predictions_t,
        uncertainty=uncertainties_t,
        records=records,
    )
    metrics["prediction_mode"] = "hard_commitment_gate" if config.use_hard_eval_gate else "soft_logits"
    metrics["loss"] = total_loss / max(total_items, 1)

    strata = build_stratified_masks(targets_t, predictions_t) if config.log_grouped_geometry else None

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

    metrics["decision_geometry"] = summarize_metric_bundle_with_strata(
        {
            "margin_M": global_margin_t,
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
            "commitment_logit": commitment_logit_t,
            "commitment_scale": commitment_scale_t,
            "lambda_commit": lambda_commit_t,
            "tau_uncertain": tau_uncertain_t,
            "uncertain_logit": uncertain_logit_t,
            "witness_sign_disagreement_rate": witness_sign_disagreement_rate_t,
            "mixed_sign_witness_fraction": mixed_sign_witness_fraction_t,
        },
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
            "weighted_minus_unweighted_margin": global_margin_t - witness_margin_mean_t,
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
    parser.add_argument(
        "--primary-metric",
        type=str,
        choices=("macro_f1", "failure_recall", "balanced_commitment_primary", "balanced_margin_primary"),
        default=None,
    )
    parser.add_argument("--failure-class-weight-scale", type=float, default=None)
    parser.add_argument("--config-override", action="append", default=None)
    args = parser.parse_args()

    config = E30Config()
    _apply_config_overrides(config, args.config_override)
    config.train_loop_seed = int(args.seed)
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
    if args.primary_metric is not None:
        config.primary_metric = args.primary_metric
    if args.failure_class_weight_scale is not None:
        config.failure_class_weight_scale = float(args.failure_class_weight_scale)

    output_dir = _resolve_output_dir(config, args.seed, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.device if torch.cuda.is_available() and config.device.startswith("cuda") else "cpu")
    set_global_determinism(config.train_loop_seed)

    if args.rebuild_traces:
        raise NotImplementedError(
            "--rebuild-traces is not supported in standalone e30. Use the local or fallback trace cache."
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

    model = E30Model(config).to(device)
    criterion = E30Loss(config, outcome_class_weights=outcome_weights).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_primary_metric = float("-inf")
    best_val_metrics: dict[str, Any] = {}
    patience = 0
    epoch_history = []

    for epoch in range(1, config.train_loop_epochs + 1):
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
            f"epoch={epoch:02d} "
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
            patience = 0
        else:
            patience += 1
            if patience >= config.train_loop_patience:
                break

    model.load_state_dict(best_state)
    train_metrics, _ = _evaluate(model, criterion, train_loader, device, train_records, config)
    val_metrics, _ = _evaluate(model, criterion, val_loader, device, val_records, config)
    test_metrics, test_bundle = _evaluate(model, criterion, test_loader, device, test_records, config)
    best_val_metrics = val_metrics
    best_primary_metric = _primary_metric_value(val_metrics, config.primary_metric)
    baselines = evaluate_trace_baselines(train_records, test_records, config=config)
    report_highlights = _build_report_highlights(test_metrics)

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
        "baselines": baselines,
        "epoch_history": epoch_history,
    }
    summary = {
        "experiment": config.experiment_name,
        "decision_rule": {
            "margin": "M = sum_k omega_k * m_k",
            "vote_disagreement": "D = sum_k omega_k * (m_k - M)^2",
            "geometric_disagreement": "G = 1 - mean pairwise cosine agreement",
            "uncertainty": "U = softplus(alpha * D + beta * G - gamma * abs(M) + b)",
            "soft_commitment_gate": {
                "commitment_score": "C = abs(M) - lambda_commit * U",
                "commitment_logit": "K = commitment_scale * (C - tau_uncertain)",
                "uncertain": "softplus(-K)",
                "success": "K + M - softplus(-K)",
                "failure": "K - M - softplus(-K)",
            },
            "hard_eval_gate": "if C < tau_uncertain then UNCERTAIN else sign(M)",
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

    print(f"Wrote {output_dir / 'metrics.json'}")
    print(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
