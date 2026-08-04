"""Generic Safe Executor — dataset-agnostic version of AdultSafeExecutor.

Computes privacy-safe aggregate statistics from any PrivateDataset.
Uses get_subgroup_stats() which returns {n, rate, support_ratio, variance}.

Output layout (output_dim = 1 + variance_buckets + confidence_buckets + 1 + 3):
    rate              : aggregate positive rate in subgroup           (1)
    variance_bucket   : one-hot over variance_buckets                 (4)
    confidence_bucket : one-hot over confidence_buckets               (3)
    min_group_pass    : 1 if n >= min_group_size                      (1)
    policy_flags      : [too_specific, too_general, normal]           (3)
Total default: 12 dims.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GenericSafeExecutor(nn.Module):
    """Deterministic (non-learned) executor for any tabular PrivateDataset.

    Parameters
    ----------
    min_group_size    : minimum n for a non-trivial response
    variance_buckets  : number of variance one-hot bins
    confidence_buckets: number of confidence one-hot bins
    """

    def __init__(
        self,
        min_group_size: int = 10,
        variance_buckets: int = 4,
        confidence_buckets: int = 3,
    ):
        super().__init__()
        self.min_group_size = min_group_size
        self.variance_buckets = variance_buckets
        self.confidence_buckets = confidence_buckets
        self.output_dim = 1 + variance_buckets + confidence_buckets + 1 + 3

    @torch.no_grad()
    def forward(
        self,
        query_fields: torch.Tensor,
        private_dataset,
    ) -> torch.Tensor:
        """
        Args:
            query_fields   : (B, num_query_fields) int64
            private_dataset: any PrivateDataset (implements get_subgroup_stats)

        Returns:
            (B, output_dim) float32 safe aggregate features
        """
        B = query_fields.size(0)
        device = query_fields.device
        out = torch.zeros(B, self.output_dim, device=device)

        for i in range(B):
            stats = private_dataset.get_subgroup_stats(query_fields[i])
            n              = stats["n"]
            rate           = stats["rate"]
            variance       = stats["variance"]
            support_ratio  = stats["support_ratio"]

            # specificity = last field if has_specificity_field (value ≥ 0)
            # Use support_ratio as a proxy for specificity if the last field
            # is the specificity slot; otherwise derive from support.
            last_val = query_fields[i, -1].item()
            specificity = last_val  # may be the specificity count or another field

            col = 0

            # ── 1. Rate ─────────────────────────────────────────────────────
            out[i, col] = rate
            col += 1

            # ── 2. Variance bucket (one-hot) ────────────────────────────────
            # Bernoulli variance max = 0.25 at p=0.5
            boundaries = [0.10, 0.18, 0.24]
            bucket = sum(1 for b in boundaries if variance >= b)
            out[i, col + bucket] = 1.0
            col += self.variance_buckets

            # ── 3. Confidence bucket (one-hot) ──────────────────────────────
            if n < 30:
                conf_bucket = 0
            elif n < 200:
                conf_bucket = 1
            else:
                conf_bucket = min(2, self.confidence_buckets - 1)
            out[i, col + conf_bucket] = 1.0
            col += self.confidence_buckets

            # ── 4. Min group size pass/fail ──────────────────────────────────
            out[i, col] = 1.0 if n >= self.min_group_size else 0.0
            col += 1

            # ── 5. Policy flags [too_specific, too_general, normal] ──────────
            if specificity >= 6 and n < self.min_group_size:
                out[i, col] = 1.0       # too_specific
            elif specificity <= 1 and support_ratio > 0.5:
                out[i, col + 1] = 1.0   # too_general
            else:
                out[i, col + 2] = 1.0   # normal

        return out
