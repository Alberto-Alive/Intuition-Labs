"""Generic DP-Laplace baseline for any PrivateDataset.

Answers subgroup-rate queries by adding Laplace noise calibrated to
sensitivity/epsilon, then quantising to the same 6-class answer scheme
used by DIGIT.

Sensitivity for a mean query on bounded [0,1] data: 1/n.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch

from ..data.base import PrivateDataset
from ..data.ground_truth import (
    ANSWER_LABELS, SUPPORT_LABELS, CONFIDENCE_LABELS, RISK_LABELS,
    GTConfig, compute_primitives,
)

logger = logging.getLogger(__name__)


class GenericDPLaplaceBaseline:
    """Answers queries with Laplace-mechanism DP noise.

    Parameters
    ----------
    private_data : PrivateDataset (the protected dataset)
    epsilon      : privacy budget ε
    cfg          : GTConfig for primitive thresholds (default GTConfig())
    seed         : RNG seed for the Laplace noise
    """

    def __init__(
        self,
        private_data: PrivateDataset,
        epsilon: float = 1.0,
        min_group_size: int = 10,
        cfg: Optional[GTConfig] = None,
        seed: int = 42,
    ):
        self.private_data = private_data
        self.epsilon = epsilon
        self.min_group_size = int(min_group_size)
        self.cfg = cfg or GTConfig(min_group_size=self.min_group_size)
        self.overall_rate = private_data.info.positive_rate
        self.rng = np.random.RandomState(seed)

    def answer_query(self, query_vec: torch.Tensor) -> Dict:
        """Answer a single query with DP Laplace noise."""
        stats = self.private_data.get_subgroup_stats(query_vec)
        n = float(stats["n"])
        true_rate = float(stats["rate"])
        total_n = max(1.0, float(self.private_data.num_records))

        # Public primitives should not reveal exact subgroup size. We privatize
        # count-like signals as well and derive all exposed primitives from the
        # noisy statistics only.
        count_noise = self.rng.laplace(0.0, 1.0 / self.epsilon)
        noisy_n = float(np.clip(n + count_noise, 0.0, total_n))

        sensitivity = 1.0 / max(n, 1.0)
        noise = self.rng.laplace(0, sensitivity / self.epsilon)
        noisy_rate = float(np.clip(true_rate + noise, 0.0, 1.0))

        noisy_stats = {
            "n": noisy_n,
            "rate": noisy_rate,
            "support_ratio": float(np.clip(noisy_n / total_n, 0.0, 1.0)),
            "variance": noisy_rate * (1.0 - noisy_rate),
        }
        specificity = int(query_vec[-1].item())
        prims = compute_primitives(noisy_stats, self.overall_rate, specificity, self.cfg)
        prims["noisy_rate"] = noisy_rate
        prims["noisy_n"] = noisy_n
        return prims

    def answer_batch(self, queries: List[torch.Tensor]) -> List[Dict]:
        return [self.answer_query(q) for q in queries]

    def get_primitive_vectors(
        self, queries: List[torch.Tensor]
    ) -> np.ndarray:
        """Return (N, 4) int array of [answer, support, confidence, risk] per query."""
        results = self.answer_batch(queries)
        return np.array(
            [[r["answer"], r["support"], r["confidence"], r["risk"]] for r in results],
            dtype=np.int64,
        )
