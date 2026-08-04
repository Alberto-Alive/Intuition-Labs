"""Executor producing entropy-trajectory features from stats or collected traces."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.ground_truth import derive_labels_from_stats
from ..executor_schema import (
    EXECUTOR_FEATURE_DIM,
    EXECUTOR_FEATURE_INDEX,
    TRACE_INPUT_DIM,
    TRACE_INPUT_INDEX,
    TRAJECTORY_LEN,
)


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
        self.output_dim = EXECUTOR_FEATURE_DIM

    @torch.no_grad()
    def forward(self, query_fields: torch.Tensor, executor_source) -> torch.Tensor:
        if torch.is_tensor(executor_source):
            return self._from_trace_inputs(executor_source)
        return self._from_private_dataset(query_fields, executor_source)

    def _from_trace_inputs(self, trace_inputs: torch.Tensor) -> torch.Tensor:
        """Summarize recorded trajectories into executor features."""
        if trace_inputs.size(-1) != TRACE_INPUT_DIM:
            raise ValueError(
                f"Expected trace_inputs to have width {TRACE_INPUT_DIM}, got {trace_inputs.size(-1)}"
            )
        trajectory = trace_inputs[:, :TRAJECTORY_LEN]
        max_attention_mass = trace_inputs[:, TRACE_INPUT_INDEX["max_attention_mass"]]
        agreement = trace_inputs[:, TRACE_INPUT_INDEX["agreement"]]
        prob_margin = trace_inputs[:, TRACE_INPUT_INDEX["prob_margin"]]
        max_softmax = trace_inputs[:, TRACE_INPUT_INDEX["max_softmax"]]
        predictive_entropy = trace_inputs[:, TRACE_INPUT_INDEX["predictive_entropy"]]
        variation_ratio = trace_inputs[:, TRACE_INPUT_INDEX["variation_ratio"]]
        support_ratio = trace_inputs[:, TRACE_INPUT_INDEX["support_ratio"]]
        n_norm = trace_inputs[:, TRACE_INPUT_INDEX["n_norm"]]
        abs_margin = trace_inputs[:, TRACE_INPUT_INDEX["abs_margin"]]
        attention_top2_mass = trace_inputs[:, TRACE_INPUT_INDEX["attention_top2_mass"]]
        attention_effective_support = trace_inputs[:, TRACE_INPUT_INDEX["attention_effective_support"]]
        attention_gini = trace_inputs[:, TRACE_INPUT_INDEX["attention_gini"]]
        attention_concentration_drift = trace_inputs[:, TRACE_INPUT_INDEX["attention_concentration_drift"]]

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
                max_softmax.unsqueeze(-1),
                predictive_entropy.unsqueeze(-1),
                variation_ratio.unsqueeze(-1),
                support_ratio.unsqueeze(-1),
                n_norm.unsqueeze(-1),
                abs_margin.unsqueeze(-1),
                low_margin_flag.unsqueeze(-1),
                unstable_flag.unsqueeze(-1),
                attention_top2_mass.unsqueeze(-1),
                attention_effective_support.unsqueeze(-1),
                attention_gini.unsqueeze(-1),
                attention_concentration_drift.unsqueeze(-1),
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

            out[i, :TRAJECTORY_LEN] = torch.tensor(trajectory, dtype=torch.float32, device=device)
            out[i, EXECUTOR_FEATURE_INDEX["mean_entropy"]] = summary["mean_entropy"]
            out[i, EXECUTOR_FEATURE_INDEX["slope"]] = summary["slope"]
            out[i, EXECUTOR_FEATURE_INDEX["std_entropy"]] = summary["std_entropy"]
            out[i, EXECUTOR_FEATURE_INDEX["entropy_range"]] = summary["entropy_range"]

            support_ratio = float(stats["support_ratio"])
            n_norm = min(float(stats["n"]) / 200.0, 1.0)
            abs_margin = float(stats["abs_margin"])
            max_attention_proxy = support_ratio
            stability_proxy = 1.0 - min(summary["std_entropy"] / 0.25, 1.0)
            agreement_proxy = float(max(min(0.5 * n_norm + 0.5 * stability_proxy, 1.0), 0.0))
            prob_margin_proxy = abs_margin
            max_softmax_proxy = float(max(min(0.5 + abs_margin, 1.0), 1.0 / 3.0))
            predictive_entropy_proxy = float(summary["mean_entropy"])
            variation_ratio_proxy = 1.0 - agreement_proxy
            attention_top2_proxy = float(
                max(
                    max_attention_proxy,
                    min(
                        max_attention_proxy + 0.35 * (1.0 - summary["mean_entropy"]) * (1.0 - max_attention_proxy),
                        1.0,
                    ),
                )
            )
            attention_effective_support_proxy = float(
                max(
                    min(
                        0.65 * summary["mean_entropy"]
                        + 0.20 * predictive_entropy_proxy
                        + 0.15 * (1.0 - max_attention_proxy),
                        1.0,
                    ),
                    0.0,
                )
            )
            attention_gini_proxy = float(
                max(
                    min(
                        0.60 * max_attention_proxy
                        + 0.30 * (1.0 - summary["mean_entropy"])
                        + 0.10 * prob_margin_proxy,
                        1.0,
                    ),
                    0.0,
                )
            )
            attention_concentration_drift_proxy = float(
                max(
                    min(0.70 * summary["entropy_range"] + 0.60 * summary["std_entropy"], 1.0),
                    0.0,
                )
            )
            low_margin_flag = 1.0 if abs_margin < self.low_margin_threshold else 0.0
            unstable_flag = 1.0 if agreement_proxy < self.low_agreement_threshold else 0.0

            out[i, EXECUTOR_FEATURE_INDEX["max_attention_mass"]] = max_attention_proxy
            out[i, EXECUTOR_FEATURE_INDEX["agreement"]] = agreement_proxy
            out[i, EXECUTOR_FEATURE_INDEX["prob_margin"]] = prob_margin_proxy
            out[i, EXECUTOR_FEATURE_INDEX["max_softmax"]] = max_softmax_proxy
            out[i, EXECUTOR_FEATURE_INDEX["predictive_entropy"]] = predictive_entropy_proxy
            out[i, EXECUTOR_FEATURE_INDEX["variation_ratio"]] = variation_ratio_proxy
            out[i, EXECUTOR_FEATURE_INDEX["support_ratio"]] = support_ratio
            out[i, EXECUTOR_FEATURE_INDEX["n_norm"]] = n_norm
            out[i, EXECUTOR_FEATURE_INDEX["abs_margin"]] = abs_margin
            out[i, EXECUTOR_FEATURE_INDEX["low_margin_flag"]] = low_margin_flag
            out[i, EXECUTOR_FEATURE_INDEX["unstable_flag"]] = unstable_flag
            out[i, EXECUTOR_FEATURE_INDEX["attention_top2_mass"]] = attention_top2_proxy
            out[i, EXECUTOR_FEATURE_INDEX["attention_effective_support"]] = attention_effective_support_proxy
            out[i, EXECUTOR_FEATURE_INDEX["attention_gini"]] = attention_gini_proxy
            out[i, EXECUTOR_FEATURE_INDEX["attention_concentration_drift"]] = attention_concentration_drift_proxy

        return out
