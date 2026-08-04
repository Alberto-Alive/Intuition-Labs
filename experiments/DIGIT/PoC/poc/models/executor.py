"""Safe Private Executor for Adult Income data.

Deterministic module (NOT learned). Computes privacy-safe aggregate
statistics from real Adult records. No raw rows leave this module.
"""

import torch
import torch.nn as nn


class AdultSafeExecutor(nn.Module):
    """Computes privacy-safe aggregate statistics from Adult data.

    Output features (12 dims):
        - income_rate: fraction with income >50K in subgroup (1)
        - variance_bucket: one-hot over 4 buckets (4)
        - confidence_bucket: one-hot over 3 buckets based on group size (3)
        - min_group_size_pass: binary (1)
        - policy_flags: [too_specific, too_general, normal] (3)

    Total output_dim = 12.
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
            query_fields: (B, 13) integer query tensor.
            private_dataset: PrivateAdultDataset instance.

        Returns:
            agg_features: (B, output_dim) safe aggregate features.
        """
        B = query_fields.size(0)
        device = query_fields.device
        out = torch.zeros(B, self.output_dim, device=device)

        for i in range(B):
            stats = private_dataset.get_subgroup_stats(query_fields[i])
            n = stats["n"]
            income_rate = stats["income_rate"]
            variance = stats["variance"]
            specificity = query_fields[i, 12].item()

            col = 0

            # 1. Income rate (the key aggregate)
            out[i, col] = income_rate
            col += 1

            # 2. Variance bucket (one-hot)
            # Bernoulli variance p*(1-p): max=0.25 at p=0.5
            # Buckets: [0, 0.10), [0.10, 0.18), [0.18, 0.24), [0.24+)
            boundaries = [0.10, 0.18, 0.24]
            bucket = 0
            for b in boundaries:
                if variance >= b:
                    bucket += 1
            out[i, col + bucket] = 1.0
            col += self.variance_buckets

            # 3. Confidence bucket (one-hot) based on group size
            if n < 30:
                conf_bucket = 0  # LOW
            elif n < 200:
                conf_bucket = 1  # MEDIUM
            else:
                conf_bucket = min(2, self.confidence_buckets - 1)  # HIGH
            out[i, col + conf_bucket] = 1.0
            col += self.confidence_buckets

            # 4. Min group size pass/fail
            out[i, col] = 1.0 if n >= self.min_group_size else 0.0
            col += 1

            # 5. Policy flags: [too_specific, too_general, normal]
            if specificity >= 6 and n < self.min_group_size:
                out[i, col] = 1.0      # too_specific
            elif specificity <= 1 and n > private_dataset.num_records * 0.5:
                out[i, col + 1] = 1.0  # too_general
            else:
                out[i, col + 2] = 1.0  # normal

        return out
