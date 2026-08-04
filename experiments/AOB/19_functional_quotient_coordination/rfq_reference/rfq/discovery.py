from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class DiscoveryConfig:
    n_masks: int = 8000
    keep_min: float = 0.35
    keep_max: float = 0.95
    success_tolerance: float = 0.04
    max_successes_for_clustering: int = 4000
    n_clusters: int = 5
    random_state: int = 0
    max_minimization_steps: int | None = None


@dataclass
class StrategyCandidate:
    mask: np.ndarray
    score: float
    cluster: int


def sample_masks(
    n_modules: int,
    config: DiscoveryConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    keep = rng.uniform(config.keep_min, config.keep_max, size=config.n_masks)
    masks = (rng.random((config.n_masks, n_modules)) < keep[:, None]).astype(np.int8)

    zero_rows = np.where(masks.sum(axis=1) == 0)[0]
    for row in zero_rows:
        masks[row, int(rng.integers(0, n_modules))] = 1

    return masks


def greedy_minimize(
    start_mask: np.ndarray,
    threshold: float,
    evaluate_one: Callable[[np.ndarray], float],
    evaluate_many: Callable[[np.ndarray], np.ndarray] | None = None,
    max_steps: int | None = None,
) -> tuple[np.ndarray, float]:
    """
    Remove one module at a time whenever the resulting subnetwork stays above
    the required threshold. Among currently safe removals, choose the one with
    highest retained score.
    """
    mask = start_mask.astype(np.int8, copy=True)

    steps = 0
    while True:
        if max_steps is not None and steps >= max_steps:
            break
        active = np.flatnonzero(mask)
        trials = []
        modules = []

        for module in active:
            trial = mask.copy()
            trial[module] = 0
            if trial.sum() == 0:
                continue
            trials.append(trial)
            modules.append(int(module))

        if not trials:
            break

        trial_array = np.stack(trials)

        if evaluate_many is not None:
            scores = np.asarray(evaluate_many(trial_array), dtype=float)
        else:
            scores = np.asarray(
                [evaluate_one(trial) for trial in trial_array],
                dtype=float,
            )

        safe = np.flatnonzero(scores >= threshold)
        if len(safe) == 0:
            break

        best_local = int(safe[np.argmax(scores[safe])])
        mask[modules[best_local]] = 0
        steps += 1

    return mask, float(evaluate_one(mask))


def discover_strategies(
    *,
    n_modules: int,
    full_score: float,
    evaluate_many: Callable[[np.ndarray], np.ndarray],
    evaluate_one: Callable[[np.ndarray], float],
    config: DiscoveryConfig,
    rng: np.random.Generator,
) -> tuple[list[StrategyCandidate], dict]:
    masks = sample_masks(n_modules, config, rng)
    scores = np.asarray(evaluate_many(masks), dtype=float)

    threshold = full_score - config.success_tolerance
    success_idx = np.flatnonzero(scores >= threshold)

    if len(success_idx) == 0:
        return [], {
            "threshold": threshold,
            "n_masks": len(masks),
            "successful_masks": 0,
        }

    order = success_idx[np.argsort(scores[success_idx])[::-1]]
    order = order[: config.max_successes_for_clustering]
    success_masks = masks[order]
    success_scores = scores[order]

    n_clusters = min(config.n_clusters, len(success_masks))
    if n_clusters == 1:
        labels = np.zeros(len(success_masks), dtype=int)
    else:
        labels = KMeans(
            n_clusters=n_clusters,
            random_state=config.random_state,
            n_init=20,
        ).fit_predict(success_masks)

    candidates: list[StrategyCandidate] = []

    for cluster in range(n_clusters):
        idx = np.flatnonzero(labels == cluster)
        if len(idx) == 0:
            continue

        cluster_masks = success_masks[idx]
        cluster_scores = success_scores[idx]

        # Prefer accurate but already sparse seeds.
        seed_objective = cluster_scores - 0.0005 * cluster_masks.sum(axis=1)
        start = cluster_masks[int(np.argmax(seed_objective))]

        minimized, minimized_score = greedy_minimize(
            start,
            threshold,
            evaluate_one,
            evaluate_many,
            config.max_minimization_steps,
        )

        candidates.append(
            StrategyCandidate(
                mask=minimized,
                score=minimized_score,
                cluster=cluster,
            )
        )

    # Exact-mask deduplication.
    unique: dict[bytes, StrategyCandidate] = {}
    for candidate in candidates:
        key = candidate.mask.tobytes()
        prev = unique.get(key)
        if prev is None or candidate.score > prev.score:
            unique[key] = candidate

    final = sorted(
        unique.values(),
        key=lambda c: (int(c.mask.sum()), -c.score),
    )

    metadata = {
        "threshold": threshold,
        "n_masks": len(masks),
        "successful_masks": int(len(success_idx)),
        "successful_fraction": float(len(success_idx) / len(masks)),
        "clusters": n_clusters,
        "strategies_found": len(final),
    }

    return final, metadata
