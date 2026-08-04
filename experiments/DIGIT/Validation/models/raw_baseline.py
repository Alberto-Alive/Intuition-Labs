"""Raw (no-privacy) baseline for MIA upper-bound comparison.

Returns exact aggregate statistics without noise.  Used to establish the
upper bound on attack success — an adversary querying the raw system can
distinguish members from non-members most easily.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from ..data.base import PrivateDataset
from ..data.ground_truth import (
    ANSWER_LABELS, GTConfig, compute_primitives,
)


class RawBaseline:
    """Answers queries with exact statistics (no privacy protection)."""

    def __init__(
        self,
        private_data: PrivateDataset,
        min_group_size: int = 1,
        cfg: Optional[GTConfig] = None,
    ):
        self.private_data = private_data
        self.min_group_size = int(min_group_size)
        self.cfg = cfg or GTConfig(min_group_size=self.min_group_size)
        self.overall_rate = private_data.info.positive_rate

    def answer_query(self, query_vec: torch.Tensor) -> Dict:
        stats = self.private_data.get_subgroup_stats(query_vec)
        specificity = int(query_vec[-1].item())
        prims = compute_primitives(stats, self.overall_rate, specificity, self.cfg)
        prims["exact_rate"] = stats["rate"]
        prims["n"] = stats["n"]
        return prims

    def answer_batch(self, queries: List[torch.Tensor]) -> List[Dict]:
        return [self.answer_query(q) for q in queries]

    def get_primitive_vectors(
        self, queries: List[torch.Tensor]
    ) -> np.ndarray:
        """Return (N, 4) int array of [answer, support, confidence, risk]."""
        results = self.answer_batch(queries)
        return np.array(
            [[r["answer"], r["support"], r["confidence"], r["risk"]] for r in results],
            dtype=np.int64,
        )
