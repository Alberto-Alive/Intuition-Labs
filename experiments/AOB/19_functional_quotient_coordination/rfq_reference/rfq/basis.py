from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass
class BasisConfig:
    target_tolerance: float = 0.01
    max_exhaustive_strategies: int = 12


def _accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.argmax(logits, axis=1) == labels))


def _subset_record(
    subset: tuple[int, ...],
    strategy_logits: list[dict[str, np.ndarray]],
    full_logits: dict[str, np.ndarray],
    labels: np.ndarray,
    costs: list[float],
) -> dict:
    conditions = sorted(full_logits)
    condition_acc = {}
    condition_fidelity = {}
    condition_oracle = {}

    for condition in conditions:
        stack = np.stack(
            [strategy_logits[idx][condition] for idx in subset],
            axis=0,
        )
        ensemble = stack.mean(axis=0)
        condition_acc[condition] = _accuracy(ensemble, labels)

        condition_fidelity[condition] = float(
            np.mean(
                np.argmax(ensemble, axis=1)
                == np.argmax(full_logits[condition], axis=1)
            )
        )

        preds = np.argmax(stack, axis=2)
        condition_oracle[condition] = float(
            np.mean(np.any(preds == labels[None, :], axis=0))
        )

    return {
        "strategies": list(subset),
        "n_strategies": len(subset),
        "cost_sum": float(sum(costs[idx] for idx in subset)),
        "mean_accuracy": float(np.mean(list(condition_acc.values()))),
        "mean_fidelity_to_full": float(np.mean(list(condition_fidelity.values()))),
        "mean_oracle_coverage": float(np.mean(list(condition_oracle.values()))),
        "per_condition_accuracy": condition_acc,
    }


def select_complementary_basis(
    *,
    strategy_logits: list[dict[str, np.ndarray]],
    full_logits: dict[str, np.ndarray],
    labels: np.ndarray,
    costs: list[float],
    config: BasisConfig,
) -> dict:
    n = len(strategy_logits)
    if n == 0:
        return {
            "all_subsets": [],
            "pareto_frontier": [],
            "smallest_near_full": None,
            "best_accuracy": None,
        }

    full_mean = float(
        np.mean([
            _accuracy(logits, labels)
            for logits in full_logits.values()
        ])
    )

    records = []

    if n <= config.max_exhaustive_strategies:
        for k in range(1, n + 1):
            for subset in combinations(range(n), k):
                records.append(
                    _subset_record(
                        subset,
                        strategy_logits,
                        full_logits,
                        labels,
                        costs,
                    )
                )
    else:
        # Greedy forward fallback for larger quotient bases.
        chosen: list[int] = []
        remaining = set(range(n))
        while remaining:
            candidate_records = []
            for idx in remaining:
                subset = tuple(sorted(chosen + [idx]))
                candidate_records.append(
                    _subset_record(
                        subset,
                        strategy_logits,
                        full_logits,
                        labels,
                        costs,
                    )
                )
            best = max(candidate_records, key=lambda r: r["mean_accuracy"])
            chosen = best["strategies"]
            remaining -= set(chosen)
            records.append(best)

    pareto = []
    for record in records:
        dominated = any(
            other["cost_sum"] <= record["cost_sum"]
            and other["mean_accuracy"] >= record["mean_accuracy"]
            and (
                other["cost_sum"] < record["cost_sum"]
                or other["mean_accuracy"] > record["mean_accuracy"]
            )
            for other in records
        )
        if not dominated:
            pareto.append(record)

    eligible = [
        r for r in records
        if r["mean_accuracy"] >= full_mean - config.target_tolerance
    ]

    smallest_near_full = (
        min(
            eligible,
            key=lambda r: (
                r["cost_sum"],
                r["n_strategies"],
                -r["mean_accuracy"],
            ),
        )
        if eligible
        else None
    )

    best_accuracy = max(records, key=lambda r: r["mean_accuracy"])

    return {
        "full_mean_accuracy": full_mean,
        "all_subsets": records,
        "pareto_frontier": sorted(
            pareto,
            key=lambda r: (r["cost_sum"], -r["mean_accuracy"]),
        ),
        "smallest_near_full": smallest_near_full,
        "best_accuracy": best_accuracy,
    }
