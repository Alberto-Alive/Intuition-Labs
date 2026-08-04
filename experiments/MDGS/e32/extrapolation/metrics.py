"""Evaluation helpers for E31."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from .data.ground_truth import OUTCOME_LABELS
from .models.readout import bucketize_outcome_logits


def summarize_tensor(values: torch.Tensor) -> dict[str, float]:
    values = values.float().reshape(-1).cpu()
    if values.numel() == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "count": 0,
        }
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "count": int(values.numel()),
    }


def summarize_offdiag_pairwise_agreement(pairwise_agreement: torch.Tensor) -> dict[str, float]:
    if pairwise_agreement.ndim != 3:
        raise ValueError("pairwise_agreement must have shape (batch, num_samples, num_samples)")
    num_samples = pairwise_agreement.shape[-1]
    if num_samples <= 1:
        return {
            "mean": 1.0,
            "std": 0.0,
            "min": 1.0,
            "max": 1.0,
            "count": 0,
        }
    mask = ~torch.eye(num_samples, dtype=torch.bool, device=pairwise_agreement.device)
    values = pairwise_agreement[:, mask].reshape(-1)
    return summarize_tensor(values)


def pairwise_offdiag_mean(pairwise_agreement: torch.Tensor) -> torch.Tensor:
    """Return one mean off-diagonal agreement value per example."""

    if pairwise_agreement.ndim != 3:
        raise ValueError("pairwise_agreement must have shape (batch, num_samples, num_samples)")
    num_samples = pairwise_agreement.shape[-1]
    if num_samples <= 1:
        return torch.ones(
            pairwise_agreement.shape[0],
            device=pairwise_agreement.device,
            dtype=pairwise_agreement.dtype,
        )
    mask = ~torch.eye(num_samples, dtype=torch.bool, device=pairwise_agreement.device)
    values = pairwise_agreement[:, mask].reshape(pairwise_agreement.shape[0], -1)
    return values.mean(dim=-1)


def build_stratified_masks(
    targets: torch.Tensor,
    predictions: torch.Tensor,
) -> dict[str, dict[str, torch.Tensor]]:
    targets = targets.detach().reshape(-1).long().cpu()
    predictions = predictions.detach().reshape(-1).long().cpu()
    correctness = predictions == targets

    by_true_class = {label: targets == idx for idx, label in enumerate(OUTCOME_LABELS)}
    by_predicted_class = {label: predictions == idx for idx, label in enumerate(OUTCOME_LABELS)}
    by_correctness = {"correct": correctness, "incorrect": ~correctness}
    return {
        "by_true_class": by_true_class,
        "by_predicted_class": by_predicted_class,
        "by_correctness": by_correctness,
    }


def summarize_tensor_by_groups(
    values: torch.Tensor,
    groups: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    flattened = values.reshape(-1)
    summaries: dict[str, dict[str, float]] = {}
    for group_name, mask in groups.items():
        mask = mask.reshape(-1).to(device=flattened.device, dtype=torch.bool)
        summaries[group_name] = summarize_tensor(flattened[mask])
    return summaries


def summarize_tensor_with_strata(
    values: torch.Tensor,
    strata: dict[str, dict[str, torch.Tensor]] | None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"overall": summarize_tensor(values)}
    if strata is None:
        return summary
    for strata_name, groups in strata.items():
        summary[strata_name] = summarize_tensor_by_groups(values, groups)
    return summary


def summarize_metric_bundle_with_strata(
    metric_bundle: dict[str, torch.Tensor],
    strata: dict[str, dict[str, torch.Tensor]] | None,
    *,
    pairwise_keys: set[str] | None = None,
) -> dict[str, Any]:
    pairwise_keys = pairwise_keys or set()
    summary: dict[str, Any] = {}
    for name, values in metric_bundle.items():
        if name in pairwise_keys:
            summary[name] = summarize_pairwise_metric_by_strata(values, strata)
        else:
            summary[name] = summarize_metric_by_strata(values, strata)
    return summary


def summarize_metric_bundle_by_strata(
    metric_bundle: dict[str, torch.Tensor],
    targets: torch.Tensor,
    predictions: torch.Tensor,
    *,
    pairwise_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Backward-compatible alias that derives strata from labels and predictions."""

    strata = build_stratified_masks(targets, predictions)
    return summarize_metric_bundle_with_strata(
        metric_bundle,
        strata,
        pairwise_keys=pairwise_keys,
    )


def summarize_metric_by_strata(
    values: torch.Tensor,
    strata: dict[str, dict[str, torch.Tensor]] | None,
    *,
    summarizer: Callable[[torch.Tensor], dict[str, float]] = summarize_tensor,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"overall": summarizer(values)}
    if strata is None:
        return summary
    for strata_name, groups in strata.items():
        summary[strata_name] = {}
        for group_name, mask in groups.items():
            mask = mask.reshape(-1).to(device=values.device, dtype=torch.bool)
            summary[strata_name][group_name] = summarizer(values.reshape(-1)[mask])
    return summary


def summarize_pairwise_metric_by_strata(
    values: torch.Tensor,
    strata: dict[str, dict[str, torch.Tensor]] | None,
    *,
    summarizer: Callable[[torch.Tensor], dict[str, float]] = summarize_offdiag_pairwise_agreement,
) -> dict[str, Any]:
    """Summarize pairwise tensors without flattening away the batch axis."""

    summary: dict[str, Any] = {"overall": summarizer(values)}
    if strata is None:
        return summary
    for strata_name, groups in strata.items():
        summary[strata_name] = {}
        for group_name, mask in groups.items():
            mask = mask.reshape(-1).to(device=values.device, dtype=torch.bool)
            summary[strata_name][group_name] = summarizer(values[mask])
    return summary


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    if probabilities.size == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(probabilities.shape[0], 1)
    ece = 0.0
    for idx in range(bins):
        low = edges[idx]
        high = edges[idx + 1]
        if idx == bins - 1:
            mask = (probabilities >= low) & (probabilities <= high)
        else:
            mask = (probabilities >= low) & (probabilities < high)
        if not np.any(mask):
            continue
        confidence = float(probabilities[mask].mean())
        accuracy = float(labels[mask].mean())
        ece += (mask.sum() / total) * abs(confidence - accuracy)
    return float(ece)


def area_under_risk_coverage_curve(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.size == 0 or labels.size == 0:
        return float("nan")

    finite_mask = np.isfinite(probabilities)
    if not np.any(finite_mask):
        return float("nan")

    probabilities = probabilities[finite_mask]
    labels = labels[finite_mask]
    order = np.argsort(-probabilities, kind="stable")
    sorted_labels = labels[order].astype(np.float64)
    coverage = np.arange(1, sorted_labels.size + 1, dtype=np.float64)
    cumulative_correct = np.cumsum(sorted_labels)
    risk = 1.0 - (cumulative_correct / coverage)
    return float(risk.mean())


def compute_binary_probability_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.size == 0:
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "brier": float("nan"),
            "ece_10bin": float("nan"),
            "aurc": float("nan"),
        }

    unique = np.unique(labels)
    if unique.size < 2:
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = float(roc_auc_score(labels, probabilities))
        auprc = float(average_precision_score(labels, probabilities))

    brier = float(np.mean((probabilities - labels) ** 2))
    ece = expected_calibration_error(probabilities, labels, bins=10)
    aurc = area_under_risk_coverage_curve(probabilities, labels)
    return {
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "ece_10bin": ece,
        "aurc": aurc,
    }


def compute_success_safety_metrics(
    outcome_probabilities: np.ndarray,
    outcome_predictions: np.ndarray,
    outcome_targets: np.ndarray,
    is_correct: np.ndarray,
    success_class_idx: int = 0,
    failure_class_idx: int = 2,
) -> dict[str, float]:
    outcome_probabilities = np.asarray(outcome_probabilities, dtype=np.float64)
    outcome_predictions = np.asarray(outcome_predictions, dtype=np.int64)
    outcome_targets = np.asarray(outcome_targets, dtype=np.int64)
    is_correct = np.asarray(is_correct, dtype=np.int64)

    predicted_success = outcome_predictions == success_class_idx
    target_success = outcome_targets == success_class_idx
    predicted_success_count = int(predicted_success.sum())
    label_failures = int(np.logical_and(predicted_success, ~target_success).sum())
    correctness_failures = int(np.logical_and(predicted_success, is_correct == 0).sum())

    success_probabilities = outcome_probabilities[:, success_class_idx]
    failure_probabilities = outcome_probabilities[:, failure_class_idx]
    nonfailure_probabilities = 1.0 - failure_probabilities
    incorrect = 1 - is_correct

    success_correct_metrics = compute_binary_probability_metrics(success_probabilities, is_correct)
    success_label_metrics = compute_binary_probability_metrics(
        success_probabilities, target_success.astype(np.int64)
    )
    nonfailure_correct_metrics = compute_binary_probability_metrics(
        nonfailure_probabilities, is_correct
    )
    failure_incorrect_metrics = compute_binary_probability_metrics(failure_probabilities, incorrect)
    success_precision = float(is_correct[predicted_success].mean()) if predicted_success_count else 0.0

    return {
        "unsafe_success_rate_label": _safe_rate(label_failures, predicted_success_count),
        "unsafe_success_rate_correctness": _safe_rate(correctness_failures, predicted_success_count),
        "success_precision_vs_is_correct": success_precision,
        "success_auroc": success_correct_metrics["auroc"],
        "success_auprc": success_correct_metrics["auprc"],
        "success_brier": success_correct_metrics["brier"],
        "success_ece_10bin": success_correct_metrics["ece_10bin"],
        "success_aurc": success_correct_metrics["aurc"],
        "success_label_auroc": success_label_metrics["auroc"],
        "success_label_auprc": success_label_metrics["auprc"],
        "success_label_brier": success_label_metrics["brier"],
        "success_label_ece_10bin": success_label_metrics["ece_10bin"],
        "success_label_aurc": success_label_metrics["aurc"],
        "nonfailure_correct_auroc": nonfailure_correct_metrics["auroc"],
        "nonfailure_correct_auprc": nonfailure_correct_metrics["auprc"],
        "nonfailure_correct_brier": nonfailure_correct_metrics["brier"],
        "nonfailure_correct_ece_10bin": nonfailure_correct_metrics["ece_10bin"],
        "nonfailure_correct_aurc": nonfailure_correct_metrics["aurc"],
        "failure_incorrect_auroc": failure_incorrect_metrics["auroc"],
        "failure_incorrect_auprc": failure_incorrect_metrics["auprc"],
        "failure_incorrect_brier": failure_incorrect_metrics["brier"],
        "failure_incorrect_ece_10bin": failure_incorrect_metrics["ece_10bin"],
        "failure_incorrect_aurc": failure_incorrect_metrics["aurc"],
        "predicted_success_rate": _safe_rate(predicted_success_count, len(outcome_predictions)),
    }


def _uncertainty_error_correlation(uncertainty: torch.Tensor, incorrect: torch.Tensor) -> float:
    uncertainty = uncertainty.detach().float().cpu().numpy().reshape(-1)
    incorrect = incorrect.detach().float().cpu().numpy().reshape(-1)
    if uncertainty.size == 0 or incorrect.size == 0:
        return float("nan")
    if np.std(uncertainty) <= 0.0 or np.std(incorrect) <= 0.0:
        return float("nan")
    return float(np.corrcoef(uncertainty, incorrect)[0, 1])


def compute_collapse_diagnostics(predictions: torch.Tensor) -> dict[str, Any]:
    predictions = predictions.detach().cpu().numpy().astype(np.int64).reshape(-1)
    if predictions.size == 0:
        return {
            "all_success": False,
            "all_uncertain": False,
            "all_failure": False,
            "dominant_class": None,
            "dominant_class_share": float("nan"),
            "prediction_entropy": float("nan"),
        }
    counts = np.bincount(predictions, minlength=len(OUTCOME_LABELS)).astype(np.float64)
    shares = counts / max(float(predictions.size), 1.0)
    if np.any(shares > 0.0):
        positive = shares > 0.0
        safe_shares = np.clip(shares[positive], 1e-12, 1.0)
        entropy = -float(np.sum(safe_shares * np.log(safe_shares)))
        entropy = entropy / float(np.log(len(OUTCOME_LABELS)))
    else:
        entropy = float("nan")
    return {
        "all_success": bool(np.all(predictions == 0)),
        "all_uncertain": bool(np.all(predictions == 1)),
        "all_failure": bool(np.all(predictions == 2)),
        "dominant_class": int(np.argmax(counts)),
        "dominant_class_share": float(shares.max()),
        "prediction_entropy": float(entropy),
    }


def compute_outcome_metrics(
    outcome_logits: torch.Tensor,
    outcome_targets: torch.Tensor,
    *,
    predictions: torch.Tensor | None = None,
    uncertainty: torch.Tensor | None = None,
    witness_bundle: dict[str, torch.Tensor] | None = None,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    probabilities = F.softmax(outcome_logits, dim=-1)
    if predictions is None:
        predictions = bucketize_outcome_logits(outcome_logits)
    predictions = predictions.long().reshape(-1)
    targets = outcome_targets.long().reshape(-1)

    metrics: dict[str, Any] = {
        "outcome_acc": float((predictions == targets).float().mean().item()),
        "outcome_macro_f1": float(
            f1_score(targets.cpu().numpy(), predictions.cpu().numpy(), average="macro", zero_division=0)
        ),
        "outcome_confusion": confusion_matrix(
            targets.cpu().numpy(),
            predictions.cpu().numpy(),
            labels=list(range(len(OUTCOME_LABELS))),
        ).tolist(),
    }

    precision, recall, f1, _ = precision_recall_fscore_support(
        targets.cpu().numpy(),
        predictions.cpu().numpy(),
        labels=list(range(len(OUTCOME_LABELS))),
        zero_division=0,
    )
    class_shares = {}
    for idx, label in enumerate(OUTCOME_LABELS):
        key = label.lower()
        metrics[f"{key}_precision"] = float(precision[idx])
        metrics[f"{key}_recall"] = float(recall[idx])
        metrics[f"{key}_f1"] = float(f1[idx])
        class_shares[label] = float((predictions == idx).float().mean().item())
    metrics["success_recall"] = float(recall[OUTCOME_LABELS.index("SUCCESS_LIKELY")])
    metrics["uncertain_recall"] = float(recall[OUTCOME_LABELS.index("UNCERTAIN")])
    metrics["failure_recall"] = float(recall[OUTCOME_LABELS.index("FAILURE_LIKELY")])
    metrics["class_shares"] = class_shares
    metrics["outcome_share"] = class_shares
    metrics["collapse_diagnostics"] = compute_collapse_diagnostics(predictions)

    if uncertainty is not None:
        incorrect = (predictions != targets).to(dtype=torch.float32)
        metrics["uncertainty_error_corr"] = _uncertainty_error_correlation(uncertainty, incorrect)

    if witness_bundle is not None:
        geometry_metrics: dict[str, Any] = {}
        for key, value in witness_bundle.items():
            if key == "pairwise_agreement":
                geometry_metrics[key] = summarize_offdiag_pairwise_agreement(value)
            else:
                geometry_metrics[key] = summarize_tensor(value)
        metrics["witness_geometry"] = geometry_metrics

    if records is not None:
        is_correct = [int(bool(record.get("is_correct", False))) for record in records]
        metrics.update(
            compute_success_safety_metrics(
                outcome_probabilities=probabilities.cpu().numpy(),
                outcome_predictions=predictions.cpu().numpy(),
                outcome_targets=targets.cpu().numpy(),
                is_correct=is_correct,
            )
        )

    return metrics
