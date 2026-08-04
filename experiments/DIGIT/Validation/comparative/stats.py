"""Statistical helpers for comparative runs."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


def _coerce_float(value: Any) -> float:
    """Treat missing or non-numeric values as NaN during aggregation."""
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def bootstrap_ci(
    values: Sequence[float],
    seed: int = 0,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    vals = np.asarray(list(values), dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    if vals.size == 1:
        return float(vals[0]), float(vals[0])

    rng = np.random.RandomState(seed)
    samples = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        draw = rng.choice(vals, size=vals.size, replace=True)
        samples[i] = draw.mean()
    lo = np.percentile(samples, 100.0 * (alpha / 2.0))
    hi = np.percentile(samples, 100.0 * (1.0 - alpha / 2.0))
    return float(lo), float(hi)


def summarize_scalar(values: Sequence[float], seed: int = 0) -> Dict[str, float]:
    vals = np.asarray(list(values), dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    ci_low, ci_high = bootstrap_ci(vals, seed=seed)
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def flatten_metric_summary(prefix: str, values: Sequence[float], seed: int = 0) -> Dict[str, float]:
    summary = summarize_scalar(values, seed=seed)
    return {
        f"{prefix}_mean": summary["mean"],
        f"{prefix}_std": summary["std"],
        f"{prefix}_ci_low": summary["ci_low"],
        f"{prefix}_ci_high": summary["ci_high"],
    }


def summarize_per_seed_metrics(
    per_seed: List[Dict[str, float]],
    metric_names: Iterable[str],
    seed: int = 0,
) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for name in metric_names:
        vals = [_coerce_float(r.get(name, float("nan"))) for r in per_seed]
        flat.update(flatten_metric_summary(name, vals, seed=seed))
    return flat


def summarize_curve_events(
    events_per_seed: List[List[Dict[str, float]]],
    x_key: str,
    y_key: str,
    seed: int = 0,
) -> List[Dict[str, float]]:
    if not events_per_seed:
        return []

    checkpoints = sorted({int(e[x_key]) for events in events_per_seed for e in events if x_key in e})
    summaries: List[Dict[str, float]] = []
    for checkpoint in checkpoints:
        vals = [
            _coerce_float(e[y_key])
            for events in events_per_seed
            for e in events
            if int(e.get(x_key, -1)) == checkpoint and y_key in e
        ]
        summary = summarize_scalar(vals, seed=seed)
        summaries.append(
            {
                x_key: checkpoint,
                f"{y_key}_mean": summary["mean"],
                f"{y_key}_std": summary["std"],
                f"{y_key}_ci_low": summary["ci_low"],
                f"{y_key}_ci_high": summary["ci_high"],
            }
        )
    return summaries
