from __future__ import annotations

from typing import Dict

import numpy as np

from src.agents.types import AttemptBatch


def accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(predictions.astype(np.int64) == labels.astype(np.int64)))


def per_task_accuracy(predictions: np.ndarray, batch: AttemptBatch) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for task_id in sorted(set(int(x) for x in batch.task_ids.tolist())):
        mask = batch.task_ids == task_id
        out[str(task_id)] = accuracy(predictions[mask], batch.labels[mask])
    return out


def evaluate_predictions(
    method: str,
    condition: str,
    predictions: np.ndarray,
    batch: AttemptBatch,
    seed: int,
    param_count: int,
    benchmark: str = "stage0_original",
    metadata: Dict[str, object] | None = None,
) -> Dict[str, object]:
    return {
        "benchmark": benchmark,
        "seed": seed,
        "split": batch.split,
        "condition": condition,
        "method": method,
        "accuracy": accuracy(predictions, batch.labels),
        "per_task_accuracy": per_task_accuracy(predictions, batch),
        "n_examples": batch.n_examples,
        "param_count": param_count,
        "metadata": metadata or {},
    }


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    scores = scores.astype(np.float64)
    positives = labels == 1
    negatives = labels == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Average ranks for ties.
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = float(np.mean(np.arange(start + 1, end + 1)))
        start = end
    sum_positive_ranks = float(ranks[positives].sum())
    return (sum_positive_ranks - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
