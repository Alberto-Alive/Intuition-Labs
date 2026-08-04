"""Comparative linkage attack."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..comparative.querying import QueryAccounting, build_partial_queries, primitive_vector_from_system
from ..data.base import DatasetInfo, PrivateDataset, TabularPrivateDataset


def _overlapping_datasets(
    private_data: PrivateDataset,
    overlap_fraction: float,
    seed: int,
) -> Tuple[PrivateDataset, PrivateDataset, np.ndarray]:
    rng = np.random.RandomState(seed)
    n = private_data.num_records
    overlap_n = max(10, int(n * overlap_fraction))
    shared = rng.choice(n, size=overlap_n, replace=False)
    remaining = np.array([i for i in range(n) if i not in set(shared.tolist())], dtype=int)
    half = max(1, len(remaining) // 2)
    left_extra = remaining[:half]
    right_extra = remaining[half:]

    def _make(indices: np.ndarray, split: str) -> PrivateDataset:
        feat = private_data.features[torch.tensor(indices, dtype=torch.long)]
        lbl = private_data.labels[torch.tensor(indices, dtype=torch.long)]
        info = DatasetInfo(
            dataset_name=private_data.info.dataset_name,
            split_name=split,
            n_records=int(feat.shape[0]),
            n_features=private_data.info.n_features,
            label_name=private_data.info.label_name,
            positive_rate=float(lbl.float().mean()) if len(lbl) else 0.0,
            query_fields=private_data.info.query_fields,
            field_vocab_sizes=private_data.info.field_vocab_sizes,
        )
        return TabularPrivateDataset(feat, lbl, private_data.query_fields, info)

    left = _make(np.concatenate([shared, left_extra]), "linkage_left")
    right = _make(np.concatenate([shared, right_extra]), "linkage_right")
    return left, right, shared


def _record_signature(
    system: Any,
    system_data: PrivateDataset,
    target_features: torch.Tensor,
    seed: int,
    query_budget: int,
    accounting: QueryAccounting,
) -> np.ndarray:
    epsilon = float(getattr(system, "epsilon", 0.0)) if hasattr(system, "epsilon") else None
    queries = build_partial_queries(target_features, query_budget, seed)
    responses = [primitive_vector_from_system(system, q, system_data) for q in queries]
    accounting.record(len(queries), epsilon)
    prims = np.stack([r["primitive_vector"].astype(float) for r in responses], axis=0)
    return np.concatenate([prims.mean(axis=0), prims.std(axis=0)], axis=0)


def run_linkage_attack(
    left_system: Any,
    right_system: Any,
    left_data: PrivateDataset,
    right_data: PrivateDataset,
    shared_indices: np.ndarray,
    seed: int,
    query_budget: int = 20,
) -> Dict[str, float]:
    accounting = QueryAccounting()
    left_signatures: List[np.ndarray] = []
    right_signatures: List[np.ndarray] = []
    shared = shared_indices[: min(100, len(shared_indices))]

    for pos, idx in enumerate(shared):
        rec = left_data.features[pos] if pos < left_data.num_records else left_data.features[0]
        left_signatures.append(_record_signature(left_system, left_data, rec, seed * 1000 + pos, query_budget, accounting))
        right_signatures.append(_record_signature(right_system, right_data, rec, seed * 2000 + pos, query_budget, accounting))

    left_arr = np.asarray(left_signatures, dtype=float)
    right_arr = np.asarray(right_signatures, dtype=float)
    if left_arr.size == 0 or right_arr.size == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "top1_accuracy": 0.0,
            "query_count": 0,
            "cumulative_epsilon": 0.0,
        }

    correct = 0
    for i in range(len(left_arr)):
        dists = np.linalg.norm(right_arr - left_arr[i], axis=1)
        match = int(np.argmin(dists))
        correct += int(match == i)
    acc = float(correct / len(left_arr))
    return {
        "precision": acc,
        "recall": acc,
        "top1_accuracy": acc,
        "query_count": accounting.query_count,
        "cumulative_epsilon": accounting.cumulative_epsilon,
    }


__all__ = ["_overlapping_datasets", "run_linkage_attack"]
