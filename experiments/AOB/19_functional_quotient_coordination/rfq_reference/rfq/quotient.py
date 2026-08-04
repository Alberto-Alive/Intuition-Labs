from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import js_divergence, prediction_agreement


@dataclass
class QuotientConfig:
    min_prediction_agreement: float = 0.97
    max_js_divergence: float = 0.03
    representative_cost_weight: float = 0.001


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def quotient_strategies(
    strategy_records: list[dict],
    config: QuotientConfig,
) -> dict:
    """
    Each strategy record must contain:
      - mask
      - score
      - cost
      - behaviors: mapping condition -> {pred, probs}
    """
    n = len(strategy_records)
    uf = UnionFind(n)
    pairwise = []

    for i in range(n):
        for j in range(i + 1, n):
            agreements = []
            divergences = []

            shared_conditions = sorted(
                set(strategy_records[i]["behaviors"])
                & set(strategy_records[j]["behaviors"])
            )

            for condition in shared_conditions:
                bi = strategy_records[i]["behaviors"][condition]
                bj = strategy_records[j]["behaviors"][condition]
                agreements.append(
                    prediction_agreement(
                        np.asarray(bi["pred"]),
                        np.asarray(bj["pred"]),
                    )
                )
                divergences.append(
                    js_divergence(
                        np.asarray(bi["probs"]),
                        np.asarray(bj["probs"]),
                    )
                )

            mean_agreement = float(np.mean(agreements))
            mean_js = float(np.mean(divergences))
            equivalent = (
                mean_agreement >= config.min_prediction_agreement
                and mean_js <= config.max_js_divergence
            )

            if equivalent:
                uf.union(i, j)

            pairwise.append({
                "a": i,
                "b": j,
                "mean_prediction_agreement": mean_agreement,
                "mean_js_divergence": mean_js,
                "equivalent": equivalent,
            })

    classes: dict[int, list[int]] = {}
    for i in range(n):
        classes.setdefault(uf.find(i), []).append(i)

    equivalence_classes = list(classes.values())
    representatives = []

    for members in equivalence_classes:
        representative = max(
            members,
            key=lambda idx: (
                strategy_records[idx]["score"]
                - config.representative_cost_weight
                * strategy_records[idx]["cost"]
            ),
        )
        representatives.append(representative)

    return {
        "equivalence_classes": equivalence_classes,
        "representatives": representatives,
        "pairwise": pairwise,
    }
