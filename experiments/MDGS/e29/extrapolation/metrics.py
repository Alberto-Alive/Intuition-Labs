"""Evaluation helpers for E29."""

from __future__ import annotations

from typing import Any

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
        }
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


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
    nonfailure_correct_metrics = compute_binary_probability_metrics(nonfailure_probabilities, is_correct)
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


def compute_outcome_metrics(
    outcome_logits: torch.Tensor,
    outcome_targets: torch.Tensor,
    *,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    probabilities = F.softmax(outcome_logits, dim=-1)
    predictions = bucketize_outcome_logits(outcome_logits)
    targets = outcome_targets.long()

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
        "outcome_share": {
            label: float((predictions == idx).float().mean().item())
            for idx, label in enumerate(OUTCOME_LABELS)
        },
    }

    precision, recall, f1, _ = precision_recall_fscore_support(
        targets.cpu().numpy(),
        predictions.cpu().numpy(),
        labels=list(range(len(OUTCOME_LABELS))),
        zero_division=0,
    )
    for idx, label in enumerate(OUTCOME_LABELS):
        key = label.lower()
        metrics[f"{key}_precision"] = float(precision[idx])
        metrics[f"{key}_recall"] = float(recall[idx])
        metrics[f"{key}_f1"] = float(f1[idx])
    metrics["failure_recall"] = float(recall[OUTCOME_LABELS.index("FAILURE_LIKELY")])

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
