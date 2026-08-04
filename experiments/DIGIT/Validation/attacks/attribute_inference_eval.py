"""Comparative attribute inference attack."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..comparative.querying import (
    QueryAccounting,
    build_partial_queries,
    primitive_vector_from_system,
)
from ..data.base import PrivateDataset
from .privacy_common import run_binary_attack


def _build_features(
    system: Any,
    private_data: PrivateDataset,
    target_data: PrivateDataset,
    seed: int,
    num_targets: int,
    queries_per_target: int,
    accounting: Optional[QueryAccounting] = None,
) -> Dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    n = min(num_targets, target_data.num_records)
    idxs = rng.choice(target_data.num_records, size=n, replace=False)
    X_demo: List[np.ndarray] = []
    X_inf: List[np.ndarray] = []
    y: List[int] = []

    epsilon = float(getattr(system, "epsilon", 0.0)) if hasattr(system, "epsilon") else None

    for i, idx in enumerate(idxs):
        rec = target_data.features[idx]
        queries = build_partial_queries(rec, queries_per_target, seed * 1000 + i)
        responses = [primitive_vector_from_system(system, q, private_data) for q in queries]
        if accounting is not None:
            accounting.record(len(queries), epsilon)
        prims = np.stack([r["primitive_vector"].astype(float) for r in responses], axis=0)
        summary = np.concatenate([prims.mean(axis=0), prims.std(axis=0)], axis=0)
        demo = rec.cpu().numpy().astype(float)
        X_demo.append(demo)
        X_inf.append(np.concatenate([demo, summary]))
        y.append(int(target_data.labels[idx].item()))

    return {
        "X_demo": np.asarray(X_demo, dtype=float),
        "X_inf": np.asarray(X_inf, dtype=float),
        "y": np.asarray(y, dtype=int),
    }


def run_attribute_inference_attack(
    system: Any,
    private_data: PrivateDataset,
    target_data: PrivateDataset,
    seed: int,
    num_targets: int = 500,
    queries_per_target: int = 12,
) -> Dict[str, Any]:
    accounting = QueryAccounting()
    bundle = _build_features(
        system,
        private_data,
        target_data,
        seed=seed,
        num_targets=num_targets,
        queries_per_target=queries_per_target,
        accounting=accounting,
    )
    demo_metrics, demo_y, demo_probs, demo_name = run_binary_attack(bundle["X_demo"], bundle["y"], seed)
    inf_metrics, inf_y, inf_probs, inf_name = run_binary_attack(bundle["X_inf"], bundle["y"], seed)

    result = dict(inf_metrics)
    result.update(
        {
            "baseline_auroc": float(demo_metrics["auroc"]),
            "baseline_accuracy": float(demo_metrics["accuracy"]),
            "baseline_macro_f1": float(demo_metrics["macro_f1"]),
            "auc_advantage_over_baseline": float(inf_metrics["auroc"] - demo_metrics["auroc"]),
            "accuracy_advantage_over_baseline": float(inf_metrics["accuracy"] - demo_metrics["accuracy"]),
            "classifier": inf_name,
            "baseline_classifier": demo_name,
            "query_count": accounting.query_count,
            "cumulative_epsilon": accounting.cumulative_epsilon,
            "predictions": [
                {
                    "kind": "baseline",
                    "y_true": int(y),
                    "y_score": float(score),
                }
                for y, score in zip(demo_y.tolist(), demo_probs.tolist())
            ]
            + [
                {
                    "kind": "informed",
                    "y_true": int(y),
                    "y_score": float(score),
                }
                for y, score in zip(inf_y.tolist(), inf_probs.tolist())
            ],
        }
    )
    return result
