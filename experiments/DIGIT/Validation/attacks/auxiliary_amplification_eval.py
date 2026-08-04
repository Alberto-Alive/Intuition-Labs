"""Auxiliary knowledge amplification attack."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..comparative.querying import QueryAccounting, build_partial_queries, primitive_vector_from_system
from ..data.base import PrivateDataset
from .privacy_common import run_binary_attack


def _aux_features(target_data: PrivateDataset, aux_data: PrivateDataset, idx: int) -> np.ndarray:
    record = target_data.features[idx].cpu().numpy().astype(float)
    aux_feats = aux_data.features.cpu().numpy().astype(float)
    dists = ((aux_feats - record) ** 2).sum(axis=1)
    nearest = aux_feats[int(np.argmin(dists))]
    prevalence = float(aux_data.labels.float().mean())
    return np.concatenate([nearest, np.array([prevalence], dtype=float)])


def run_auxiliary_amplification_attack(
    system: Any,
    private_data: PrivateDataset,
    target_data: PrivateDataset,
    aux_data: PrivateDataset,
    seed: int,
    num_targets: int = 300,
    queries_per_target: int = 10,
) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    n = min(num_targets, target_data.num_records)
    idxs = rng.choice(target_data.num_records, size=n, replace=False)
    accounting = QueryAccounting()
    epsilon = float(getattr(system, "epsilon", 0.0)) if hasattr(system, "epsilon") else None
    X_no_aux: List[np.ndarray] = []
    X_with_aux: List[np.ndarray] = []
    y: List[int] = []

    for pos, idx in enumerate(idxs):
        rec = target_data.features[idx]
        queries = build_partial_queries(rec, queries_per_target, seed * 1000 + pos)
        responses = [primitive_vector_from_system(system, q, private_data) for q in queries]
        accounting.record(len(queries), epsilon)
        prims = np.stack([r["primitive_vector"].astype(float) for r in responses], axis=0)
        q_feat = np.concatenate([prims.mean(axis=0), prims.std(axis=0)], axis=0)
        aux_feat = _aux_features(target_data, aux_data, int(idx))
        X_no_aux.append(np.concatenate([rec.cpu().numpy().astype(float), q_feat]))
        X_with_aux.append(np.concatenate([rec.cpu().numpy().astype(float), q_feat, aux_feat]))
        y.append(int(target_data.labels[idx].item()))

    no_aux_metrics, _, _, _ = run_binary_attack(np.asarray(X_no_aux, dtype=float), np.asarray(y, dtype=int), seed)
    aux_metrics, _, _, _ = run_binary_attack(np.asarray(X_with_aux, dtype=float), np.asarray(y, dtype=int), seed)
    return {
        "without_aux_auroc": float(no_aux_metrics["auroc"]),
        "with_aux_auroc": float(aux_metrics["auroc"]),
        "aux_amplification": float(aux_metrics["auroc"] - no_aux_metrics["auroc"]),
        "query_count": accounting.query_count,
        "cumulative_epsilon": accounting.cumulative_epsilon,
    }
