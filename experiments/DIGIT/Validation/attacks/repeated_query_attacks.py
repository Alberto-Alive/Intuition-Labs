"""Repeated-query comparative attacks."""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..comparative.querying import (
    QueryAccounting,
    batch_rate_estimates,
    build_exact_match_query,
    build_partial_queries,
    build_refinement_pairs,
    choose_canary_feature_values,
    dataset_with_appended_record,
    find_answerable_queries,
    primitive_rate_estimate,
    primitive_vector_from_system,
)
from ..data.base import PrivateDataset
from ..data.ground_truth import GTConfig, coerce_gt_config
from ..data.query_gen import QueryGenerator
from .privacy_common import run_binary_attack


def _epsilon_for_system(system: Any) -> Optional[float]:
    if hasattr(system, "epsilon"):
        return float(system.epsilon)
    return None


def _checkpoints(max_budget: int, step: int = 50) -> List[int]:
    checkpoints = list(range(step, max_budget + 1, step))
    if max_budget not in checkpoints:
        checkpoints.append(max_budget)
    return checkpoints


def _rate_query_candidates(
    private_data: PrivateDataset,
    seed: int,
    n_candidates: int,
    min_group_size: int,
) -> List[torch.Tensor]:
    gen = QueryGenerator(private_data, include_specificity_field=True)
    queries = gen.generate(n_candidates, seed=seed)
    filtered = find_answerable_queries(private_data, queries, min_group_size=min_group_size)
    return filtered[: max(10, min(len(filtered), 64))]


def run_query_averaging_attack(
    system: Any,
    private_data: PrivateDataset,
    seed: int,
    query_budget: int = 1000,
    tolerance: float = 0.01,
    gt_cfg: Optional[GTConfig] = None,
) -> Dict[str, Any]:
    cfg = coerce_gt_config(gt_cfg or getattr(system, "cfg", None))
    overall_rate = float(private_data.info.positive_rate)
    candidates = _rate_query_candidates(private_data, seed, n_candidates=256, min_group_size=cfg.min_group_size)
    if not candidates:
        return {
            "queries_to_recovery": float("nan"),
            "final_mae": float("nan"),
            "query_count": 0,
            "cumulative_epsilon": 0.0,
            "events": [],
            "predictions": [],
        }

    accounting = QueryAccounting()
    epsilon = _epsilon_for_system(system)
    checkpoints = _checkpoints(min(query_budget, 500), step=10)
    events: List[Dict[str, float]] = []
    per_query_recovery: List[float] = []

    for checkpoint in checkpoints:
        maes: List[float] = []
        for q in candidates:
            responses = [primitive_vector_from_system(system, q, private_data) for _ in range(checkpoint)]
            accounting.record(checkpoint, epsilon)
            est = float(batch_rate_estimates(responses, overall_rate, cfg).mean())
            true_rate = float(private_data.get_subgroup_stats(q)["rate"])
            maes.append(abs(est - true_rate))
        events.append(
            {
                "query_count": checkpoint * len(candidates),
                "repeats": checkpoint,
                "mae": float(np.mean(maes)),
                "mae_std": float(np.std(maes)),
            }
        )

    for q in candidates:
        recovered = False
        for checkpoint in checkpoints:
            responses = [primitive_vector_from_system(system, q, private_data) for _ in range(checkpoint)]
            est = float(batch_rate_estimates(responses, overall_rate, cfg).mean())
            true_rate = float(private_data.get_subgroup_stats(q)["rate"])
            if abs(est - true_rate) <= tolerance:
                per_query_recovery.append(float(checkpoint))
                recovered = True
                break
        if not recovered:
            per_query_recovery.append(float(query_budget + 1))

    return {
        "queries_to_recovery": float(np.mean(per_query_recovery)),
        "final_mae": float(events[-1]["mae"]) if events else float("nan"),
        "query_count": accounting.query_count,
        "cumulative_epsilon": accounting.cumulative_epsilon,
        "events": events,
        "predictions": [
            {
                "query_idx": i,
                "queries_to_recovery": float(v),
            }
            for i, v in enumerate(per_query_recovery)
        ],
    }


def _build_budgeted_mia_features(
    system: Any,
    member_data: PrivateDataset,
    nonmember_data: PrivateDataset,
    seed: int,
    num_targets: int,
    queries_per_target: int,
    accounting: Optional[QueryAccounting] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    n_each = min(num_targets // 2, member_data.num_records, nonmember_data.num_records)
    member_idx = rng.choice(member_data.num_records, size=n_each, replace=False)
    nonmember_idx = rng.choice(nonmember_data.num_records, size=n_each, replace=False)
    epsilon = _epsilon_for_system(system)

    X: List[np.ndarray] = []
    y: List[int] = []
    for dataset, indices, label in [
        (member_data, member_idx, 1),
        (nonmember_data, nonmember_idx, 0),
    ]:
        for pos, idx in enumerate(indices):
            rec = dataset.features[idx]
            queries = build_partial_queries(rec, queries_per_target, seed * 10000 + pos + (label * 1000))
            responses = [primitive_vector_from_system(system, q, member_data) for q in queries]
            if accounting is not None:
                accounting.record(len(queries), epsilon)
            prims = np.stack([r["primitive_vector"].astype(float) for r in responses], axis=0)
            feat = np.concatenate([rec.cpu().numpy().astype(float), prims.mean(axis=0), prims.std(axis=0)])
            X.append(feat)
            y.append(label)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def run_composition_stress_attack(
    system: Any,
    member_data: PrivateDataset,
    nonmember_data: PrivateDataset,
    seed: int,
    query_budget: int = 1000,
    num_targets: int = 200,
) -> Dict[str, Any]:
    accounting = QueryAccounting()
    checkpoints = _checkpoints(query_budget, step=50)
    events: List[Dict[str, float]] = []
    threshold_hits = {"0.55": None, "0.60": None, "0.70": None}

    for checkpoint in checkpoints:
        queries_per_target = max(1, checkpoint // max(1, num_targets))
        X, y = _build_budgeted_mia_features(
            system,
            member_data,
            nonmember_data,
            seed=seed,
            num_targets=num_targets,
            queries_per_target=queries_per_target,
            accounting=accounting,
        )
        metrics, _, _, _ = run_binary_attack(X, y, seed)
        auroc = float(metrics["auroc"])
        events.append(
            {
                "query_count": checkpoint,
                "auroc": auroc,
                "advantage": float(metrics["advantage"]),
            }
        )
        for key in threshold_hits:
            threshold = float(key)
            if threshold_hits[key] is None and auroc >= threshold:
                threshold_hits[key] = checkpoint

    return {
        "auroc": float(events[-1]["auroc"]) if events else 0.5,
        "advantage": float(events[-1]["advantage"]) if events else 0.0,
        "query_count": accounting.query_count,
        "cumulative_epsilon": accounting.cumulative_epsilon,
        "budget_threshold_0p55": threshold_hits["0.55"],
        "budget_threshold_0p60": threshold_hits["0.60"],
        "budget_threshold_0p70": threshold_hits["0.70"],
        "events": events,
        "predictions": [],
    }


def run_differencing_attack(
    system: Any,
    private_data: PrivateDataset,
    seed: int,
    query_budget: int = 1000,
    num_targets: int = 200,
) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    n = min(num_targets, private_data.num_records)
    idxs = rng.choice(private_data.num_records, size=n, replace=False)
    max_pairs = max(1, query_budget // max(1, 2 * n))
    checkpoints = _checkpoints(max_pairs, step=5 if max_pairs >= 5 else 1)
    events: List[Dict[str, float]] = []
    epsilon = _epsilon_for_system(system)
    accounting = QueryAccounting()

    for n_pairs in checkpoints:
        X: List[np.ndarray] = []
        y: List[int] = []
        for offset, idx in enumerate(idxs):
            rec = private_data.features[idx]
            pairs = build_refinement_pairs(rec, n_pairs, seed * 1000 + offset)
            pair_feats: List[np.ndarray] = []
            for q1, q2 in pairs:
                r1 = primitive_vector_from_system(system, q1, private_data)
                r2 = primitive_vector_from_system(system, q2, private_data)
                accounting.record(2, epsilon)
                pair_feats.append(np.concatenate([r1["primitive_vector"], r2["primitive_vector"], r1["primitive_vector"] - r2["primitive_vector"]]).astype(float))
            summary = np.concatenate([np.mean(pair_feats, axis=0), np.std(pair_feats, axis=0)], axis=0)
            X.append(np.concatenate([rec.cpu().numpy().astype(float), summary]))
            y.append(int(private_data.labels[idx].item()))
        metrics, _, _, _ = run_binary_attack(np.asarray(X, dtype=float), np.asarray(y, dtype=int), seed)
        events.append(
            {
                "differencing_pairs": n_pairs,
                "query_count": 2 * n_pairs * n,
                "auroc": float(metrics["auroc"]),
                "accuracy": float(metrics["accuracy"]),
            }
        )

    queries_to_80 = None
    for e in events:
        if queries_to_80 is None and float(e["accuracy"]) >= 0.80:
            queries_to_80 = int(e["query_count"])
    return {
        "auroc": float(events[-1]["auroc"]) if events else 0.5,
        "accuracy": float(events[-1]["accuracy"]) if events else 0.5,
        "query_count": accounting.query_count,
        "cumulative_epsilon": accounting.cumulative_epsilon,
        "query_count_to_80pct_accuracy": queries_to_80,
        "events": events,
        "predictions": [],
    }


def run_reconstruction_attack(
    system: Any,
    private_data: PrivateDataset,
    seed: int,
    query_budget: int = 1000,
    num_targets: int = 200,
) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    n = min(num_targets, private_data.num_records)
    idxs = rng.choice(private_data.num_records, size=n, replace=False)
    max_queries = max(1, query_budget // max(1, n))
    checkpoints = _checkpoints(max_queries, step=5 if max_queries >= 5 else 1)
    epsilon = _epsilon_for_system(system)
    accounting = QueryAccounting()
    events: List[Dict[str, float]] = []

    for n_queries in checkpoints:
        X: List[np.ndarray] = []
        y: List[int] = []
        for offset, idx in enumerate(idxs):
            rec = private_data.features[idx]
            queries = build_partial_queries(rec, n_queries, seed * 1000 + offset)
            responses = [primitive_vector_from_system(system, q, private_data) for q in queries]
            accounting.record(len(queries), epsilon)
            prims = np.stack([r["primitive_vector"].astype(float) for r in responses], axis=0)
            X.append(np.concatenate([prims.mean(axis=0), prims.std(axis=0)]))
            y.append(int(private_data.labels[idx].item()))
        metrics, y_true, probs, _ = run_binary_attack(np.asarray(X, dtype=float), np.asarray(y, dtype=int), seed)
        events.append(
            {
                "queries_per_target": n_queries,
                "query_count": n_queries * n,
                "accuracy": float(metrics["accuracy"]),
                "auroc": float(metrics["auroc"]),
            }
        )
    final = events[-1] if events else {"accuracy": 0.0, "auroc": 0.5}
    return {
        "exact_match_rate": float(final["accuracy"]),
        "auroc": float(final["auroc"]),
        "query_count": accounting.query_count,
        "cumulative_epsilon": accounting.cumulative_epsilon,
        "events": events,
        "predictions": [],
    }


def run_canary_detection_attack(
    system_builder,
    private_data: PrivateDataset,
    seed: int,
    query_budget: int = 1000,
    canary_feature_values: Optional[Sequence[int]] = None,
    canary_label: int = 1,
) -> Dict[str, Any]:
    cfg = coerce_gt_config(getattr(system_builder["system"], "cfg", None))
    feature_values = choose_canary_feature_values(private_data, canary_feature_values)
    canary_ds = dataset_with_appended_record(private_data, feature_values, canary_label)
    base_system = system_builder["system"]
    canary_system = system_builder["factory"](canary_ds) if system_builder["factory"] is not None else system_builder["rebuild"](canary_ds)
    q = build_exact_match_query(torch.tensor(feature_values, dtype=torch.long))
    q[-1] = len(feature_values)
    epsilon = _epsilon_for_system(base_system)
    checkpoints = _checkpoints(query_budget, step=10)
    accounting = QueryAccounting()
    baseline_stats = private_data.get_subgroup_stats(q)
    baseline_rate = float(baseline_stats["rate"])
    overall_rate = float(private_data.info.positive_rate)
    events: List[Dict[str, float]] = []

    for n in checkpoints:
        canary_responses = [primitive_vector_from_system(canary_system, q, canary_ds) for _ in range(n)]
        accounting.record(n, epsilon)
        estimates = batch_rate_estimates(canary_responses, overall_rate, cfg)
        mean_est = float(estimates.mean())
        std_est = float(estimates.std())
        if std_est == 0.0:
            confidence = 1.0 if abs(mean_est - baseline_rate) > 1e-9 else 0.0
        else:
            z = abs(mean_est - baseline_rate) / max(std_est / math.sqrt(max(n, 1)), 1e-6)
            confidence = float(2.0 * (0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))) - 1.0)
        events.append(
            {
                "repeats": n,
                "query_count": n,
                "estimate": mean_est,
                "confidence": confidence,
            }
        )

    queries_to_detect = None
    for e in events:
        if queries_to_detect is None and float(e["confidence"]) >= 0.95:
            queries_to_detect = int(e["query_count"])

    return {
        "queries_to_detection_95pct": queries_to_detect,
        "final_confidence": float(events[-1]["confidence"]) if events else 0.0,
        "query_count": accounting.query_count,
        "cumulative_epsilon": accounting.cumulative_epsilon,
        "events": events,
        "predictions": [],
    }
