"""Comprehensive evaluation metrics for DIGIT."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    classification_report,
)


def compute_metrics(
    predictions: List[int],
    targets: List[int],
    num_classes: int,
    class_names: List[str] = None,
) -> Dict[str, float]:
    """Compute classification metrics.

    Returns:
        accuracy, macro_f1, per_class_f1, confusion_matrix.
    """
    preds = np.array(predictions)
    tgts = np.array(targets)

    accuracy = float(np.mean(preds == tgts))
    macro_f1 = float(f1_score(tgts, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(tgts, preds, average="weighted", zero_division=0))
    macro_precision = float(precision_score(tgts, preds, average="macro", zero_division=0))
    macro_recall = float(recall_score(tgts, preds, average="macro", zero_division=0))

    cm = confusion_matrix(tgts, preds, labels=list(range(num_classes)))

    result = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "confusion_matrix": cm.tolist(),
    }

    # Per-class F1
    per_class = f1_score(tgts, preds, average=None, labels=list(range(num_classes)),
                          zero_division=0)
    for i, f in enumerate(per_class):
        name = class_names[i] if class_names else str(i)
        result[f"f1_{name}"] = float(f)

    return result


def aggregate_across_seeds(
    seed_results: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across seeds with mean and std."""
    all_keys = set()
    for r in seed_results:
        all_keys.update(k for k, v in r.items() if isinstance(v, (int, float)))

    aggregated = {}
    for key in sorted(all_keys):
        values = [r[key] for r in seed_results if key in r and isinstance(r[key], (int, float))]
        if values:
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "values": values,
            }

    return aggregated
