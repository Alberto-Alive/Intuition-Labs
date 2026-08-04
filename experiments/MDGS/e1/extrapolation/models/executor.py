"""Executor producing entropy-trajectory features from stats or collected traces."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.ground_truth import derive_labels_from_stats


class EntropyTrajectoryExecutor(nn.Module):
    """Computes entropy features from synthetic stats or collected traces."""

    def __init__(
        self,
        min_group_size: int = 10,
        income_margin: float = 0.10,
        low_agreement_threshold: float = 0.55,
        low_margin_threshold: float = 0.15,
    ):
        super().__init__()
        self.min_group_size = min_group_size
        self.income_margin = income_margin
        self.low_agreement_threshold = low_agreement_threshold
        self.low_margin_threshold = low_margin_threshold
        self.output_dim = 15

    @torch.no_grad()
    def forward(self, query_fields: torch.Tensor, executor_source) -> torch.Tensor:
        if torch.is_tensor(executor_source):
            return self._from_trace_inputs(executor_source)
        return self._from_private_dataset(query_fields, executor_source)

    def _from_trace_inputs(self, trace_inputs: torch.Tensor) -> torch.Tensor:
        """Summarize recorded trajectories into executor features."""
        trajectory = trace_inputs[:, :6]
        max_attention_mass = trace_inputs[:, 6]
        agreement = trace_inputs[:, 7]
        prob_margin = trace_inputs[:, 8]

        mean_entropy = trajectory.mean(dim=-1)
        slope = trajectory[:, -1] - trajectory[:, 0]
        std_entropy = trajectory.std(dim=-1, unbiased=False)
        entropy_range = trajectory.max(dim=-1).values - trajectory.min(dim=-1).values
        low_margin_flag = (prob_margin < self.low_margin_threshold).float()
        unstable_flag = (agreement < self.low_agreement_threshold).float()

        return torch.cat(
            [
                trajectory,
                mean_entropy.unsqueeze(-1),
                slope.unsqueeze(-1),
                std_entropy.unsqueeze(-1),
                entropy_range.unsqueeze(-1),
                max_attention_mass.unsqueeze(-1),
                agreement.unsqueeze(-1),
                prob_margin.unsqueeze(-1),
                low_margin_flag.unsqueeze(-1),
                unstable_flag.unsqueeze(-1),
            ],
            dim=-1,
        )

    def _from_private_dataset(self, query_fields: torch.Tensor, private_dataset) -> torch.Tensor:
        batch_size = query_fields.size(0)
        device = query_fields.device
        out = torch.zeros(batch_size, self.output_dim, device=device)

        for i in range(batch_size):
            stats = dict(private_dataset.get_subgroup_stats(query_fields[i]))
            stats["specificity"] = int(query_fields[i, 12].item())
            stats["abs_margin"] = abs(stats["income_rate"] - private_dataset.overall_income_rate)

            derived = derive_labels_from_stats(
                stats=stats,
                min_group_size=self.min_group_size,
                income_margin=self.income_margin,
            )
            trajectory = derived["entropy_trajectory"]
            summary = derived["trajectory_summary"]

            col = 0
            out[i, col : col + 6] = torch.tensor(trajectory, dtype=torch.float32, device=device)
            col += 6

            out[i, col] = summary["mean_entropy"]
            out[i, col + 1] = summary["slope"]
            out[i, col + 2] = summary["std_entropy"]
            out[i, col + 3] = summary["entropy_range"]
            col += 4

            out[i, col] = stats["support_ratio"]
            out[i, col + 1] = min(stats["n"] / 200.0, 1.0)
            out[i, col + 2] = stats["abs_margin"]
            col += 3

            out[i, col] = 1.0 if stats["n"] < self.min_group_size else 0.0
            out[i, col + 1] = 1.0 if stats["specificity"] >= 6 and stats["n"] < 30 else 0.0

        return out
