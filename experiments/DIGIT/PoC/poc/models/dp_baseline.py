"""Differential Privacy Baseline using Laplace mechanism.

For comparison: answers the same queries as DIGIT but adds calibrated
Laplace noise to the true income rate, then quantizes to the same
6-class answer categories.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from ..data.private_store import PrivateAdultDataset
from ..data.ground_truth import (
    ANSWER_LABELS, SUPPORT_LABELS, CONFIDENCE_LABELS, RISK_LABELS,
    TEMPLATE_RESPONSES,
)


class DPLaplaceBaseline:
    """Answers queries using Laplace mechanism differential privacy.

    For each query:
    1. Compute true income_rate for the subgroup
    2. Add Laplace noise calibrated to sensitivity/epsilon
    3. Quantize to the 6 answer categories
    4. Map to template text

    Sensitivity: for a mean query on binary data, sensitivity = 1/n.
    """

    def __init__(
        self,
        private_data: PrivateAdultDataset,
        epsilon: float = 1.0,
        min_group_size: int = 10,
        income_margin: float = 0.05,
        seed: int = 42,
    ):
        self.private_data = private_data
        self.epsilon = epsilon
        self.min_group_size = min_group_size
        self.income_margin = income_margin
        self.overall_rate = private_data.overall_income_rate
        self.rng = np.random.RandomState(seed)

    def answer_query(self, query_fields: torch.Tensor) -> Dict:
        """Answer a single query with DP noise."""
        stats = self.private_data.get_subgroup_stats(query_fields)
        n = stats["n"]

        # Cannot answer if group too small
        if n < self.min_group_size:
            return {
                "answer": ANSWER_LABELS.index("INSUFFICIENT_EVIDENCE"),
                "support": 0,
                "confidence": 0,
                "risk": 2,
                "noisy_rate": 0.0,
                "true_rate": stats["income_rate"],
                "text": "not enough records meet the privacy threshold to answer this query.",
            }

        true_rate = stats["income_rate"]

        # Laplace mechanism: sensitivity = 1/n for mean of bounded [0,1] data
        sensitivity = 1.0 / n
        noise = self.rng.laplace(0, sensitivity / self.epsilon)
        noisy_rate = np.clip(true_rate + noise, 0.0, 1.0)

        # Quantize to answer categories
        if noisy_rate > self.overall_rate + self.income_margin:
            answer = ANSWER_LABELS.index("YES")
        elif noisy_rate < self.overall_rate - self.income_margin:
            answer = ANSWER_LABELS.index("NO")
        else:
            answer = ANSWER_LABELS.index("MAYBE")

        # Support from group size
        support_ratio = n / self.private_data.num_records
        if support_ratio < 0.01:
            support = 0
        elif support_ratio < 0.05:
            support = 1
        elif support_ratio < 0.15:
            support = 2
        else:
            support = 3

        # Confidence from group size
        if n < 30:
            confidence = 0
        elif n < 200:
            confidence = 1
        else:
            confidence = 2

        # Risk
        variance = noisy_rate * (1 - noisy_rate)
        if n >= 100 and variance < 0.20:
            risk = 0
        elif n >= 30:
            risk = 1
        else:
            risk = 2

        # Template text
        answer_label = ANSWER_LABELS[answer]
        support_label = SUPPORT_LABELS[support]
        key = f"{answer_label}_{support_label}"
        templates = TEMPLATE_RESPONSES.get(key)
        if templates:
            text = templates[0]
        elif answer_label in TEMPLATE_RESPONSES:
            text = TEMPLATE_RESPONSES[answer_label][0]
        else:
            text = "result is inconclusive."

        return {
            "answer": answer,
            "support": support,
            "confidence": confidence,
            "risk": risk,
            "noisy_rate": float(noisy_rate),
            "true_rate": float(true_rate),
            "text": text,
        }

    def answer_batch(self, queries: List[torch.Tensor]) -> List[Dict]:
        """Answer a batch of queries."""
        return [self.answer_query(q) for q in queries]

    def evaluate(
        self,
        queries: List[torch.Tensor],
        ground_truths: List[Dict],
    ) -> Dict[str, float]:
        """Evaluate DP baseline against ground truth.

        Returns accuracy metrics.
        """
        results = self.answer_batch(queries)

        answer_correct = sum(
            1 for r, gt in zip(results, ground_truths)
            if r["answer"] == gt["answer"]
        )
        support_correct = sum(
            1 for r, gt in zip(results, ground_truths)
            if r["support"] == gt["support"]
        )

        n = len(queries)
        return {
            "answer_accuracy": answer_correct / max(n, 1),
            "support_accuracy": support_correct / max(n, 1),
            "epsilon": self.epsilon,
            "num_queries": n,
        }
