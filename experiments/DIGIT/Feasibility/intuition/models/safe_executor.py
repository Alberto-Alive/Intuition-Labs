"""Component 2: Safe Private Executor.

Deterministic module (NOT a neural net).
Input: structured query + private dataset.
Output: approved aggregate features only.
No raw rows may leave this module.
"""

import torch
import torch.nn as nn


class SafePrivateExecutor(nn.Module):
    """Computes privacy-safe aggregate statistics from private data.

    All operations are deterministic and non-differentiable.
    This module is wrapped in torch.no_grad() during forward.

    Output features (all aggregated, no raw rows):
      - support_ratio: fraction of records matching the query group
      - variance_bucket: one-hot over variance buckets
      - confidence_bucket: one-hot over confidence buckets
      - min_group_size_pass: 1 if group size >= threshold, else 0
      - policy_flags: [too_specific, too_general, normal]
    """

    def __init__(
        self,
        min_group_size: int = 5,
        variance_buckets: int = 4,
        confidence_buckets: int = 3,
    ):
        super().__init__()
        self.min_group_size = min_group_size
        self.variance_buckets = variance_buckets
        self.confidence_buckets = confidence_buckets
        # Total output dim
        self.output_dim = 1 + variance_buckets + confidence_buckets + 1 + 3

    @torch.no_grad()
    def forward(
        self,
        query_fields: torch.Tensor,
        group_ids: torch.Tensor,
        private_dataset,
    ) -> torch.Tensor:
        """
        Args:
            query_fields: (batch, num_fields) — the structured query.
            group_ids: (batch,) — which group in the private data to query.
            private_dataset: PrivateDataset instance.

        Returns:
            agg_features: (batch, output_dim) — safe aggregate features.
        """
        B = query_fields.size(0)
        device = query_fields.device
        out = torch.zeros(B, self.output_dim, device=device)

        for i in range(B):
            gid = group_ids[i].item()
            group_data = private_dataset.get_group_data(gid)  # (n, F)
            n = group_data.size(0)

            col = 0

            # 1. Support ratio
            support_ratio = n / max(private_dataset.num_records, 1)
            out[i, col] = support_ratio
            col += 1

            # 2. Variance bucket (one-hot)
            if n > 1:
                variance = group_data.var(dim=0).mean().item()
            else:
                variance = 0.0
            # Bucket boundaries: [0, 0.5), [0.5, 1.0), [1.0, 2.0), [2.0+)
            boundaries = [0.5, 1.0, 2.0]
            bucket = 0
            for b in boundaries:
                if variance >= b:
                    bucket += 1
            out[i, col + bucket] = 1.0
            col += self.variance_buckets

            # 3. Confidence bucket (one-hot)
            # Based on group size: small -> low, medium -> medium, large -> high
            if n < self.min_group_size:
                conf_bucket = 0  # LOW
            elif n < self.min_group_size * 3:
                conf_bucket = 1  # MEDIUM
            else:
                conf_bucket = min(2, self.confidence_buckets - 1)  # HIGH
            out[i, col + conf_bucket] = 1.0
            col += self.confidence_buckets

            # 4. Min group size pass/fail
            out[i, col] = 1.0 if n >= self.min_group_size else 0.0
            col += 1

            # 5. Policy flags: [too_specific, too_general, normal]
            query_norm = query_fields[i].norm().item()
            if query_norm > 3.0:
                # Query is very specific
                out[i, col] = 1.0
            elif query_norm < 0.3:
                # Query is too general
                out[i, col + 1] = 1.0
            else:
                out[i, col + 2] = 1.0

        return out
