"""Entropy bottleneck head for the Extrapolation smoke test."""

from __future__ import annotations

from typing import Dict, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrimitiveOutput(NamedTuple):
    """Container for entropy-gating outputs."""

    trajectory_shape_logits: torch.Tensor
    attention_pattern_logits: torch.Tensor
    confidence_logits: torch.Tensor
    outcome_logits: torch.Tensor
    trajectory_shape_discrete: torch.Tensor
    attention_pattern_discrete: torch.Tensor
    confidence_discrete: torch.Tensor
    outcome_discrete: torch.Tensor


class EntropyBottleneckHead(nn.Module):
    """Produces discrete entropy-gating heads from query + executor features."""

    def __init__(
        self,
        d_model: int = 256,
        executor_dim: int = 15,
        num_trajectory: int = 4,
        num_pattern: int = 3,
        num_confidence: int = 3,
        num_outcome: int = 3,
        hidden_dim: int = 256,
        trace_hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        input_dim = d_model + executor_dim

        self.shared_joint = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.shared_trace = nn.Sequential(
            nn.LayerNorm(executor_dim),
            nn.Linear(executor_dim, trace_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(trace_hidden_dim, trace_hidden_dim),
            nn.GELU(),
        )

        def _make_trace_tower(input_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, trace_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(trace_hidden_dim),
            )

        self.trajectory_shape_head_joint = nn.Linear(hidden_dim, num_trajectory)
        self.attention_pattern_head_joint = nn.Linear(hidden_dim, num_pattern)
        self.confidence_head_joint = nn.Linear(hidden_dim, num_confidence)
        self.outcome_head_joint = nn.Linear(hidden_dim, num_outcome)

        self.trajectory_trace_tower = _make_trace_tower(trace_hidden_dim + 10)
        self.pattern_trace_tower = _make_trace_tower(trace_hidden_dim + 9)
        self.confidence_trace_tower = _make_trace_tower(trace_hidden_dim + 4)
        self.outcome_trace_tower = _make_trace_tower(
            trace_hidden_dim + num_trajectory + num_pattern + num_confidence + 3
        )

        self.trajectory_shape_head_trace = nn.Linear(trace_hidden_dim, num_trajectory)
        self.attention_pattern_head_trace = nn.Linear(trace_hidden_dim, num_pattern)
        self.confidence_head_trace = nn.Linear(trace_hidden_dim, num_confidence)
        self.outcome_head_trace = nn.Linear(trace_hidden_dim, num_outcome)

        self.register_buffer("pattern_mean_entropy_mean", torch.tensor(0.0), persistent=False)
        self.register_buffer("pattern_mean_entropy_std", torch.tensor(1.0), persistent=False)
        self.register_buffer("pattern_max_attention_mean", torch.tensor(0.0), persistent=False)
        self.register_buffer("pattern_max_attention_std", torch.tensor(1.0), persistent=False)

    def set_trace_metadata(self, metadata: Dict | None) -> None:
        if not metadata:
            return
        self.pattern_mean_entropy_mean.fill_(float(metadata.get("mean_entropy_mean", 0.0)))
        self.pattern_mean_entropy_std.fill_(max(float(metadata.get("mean_entropy_std", 1.0)), 1e-6))
        self.pattern_max_attention_mean.fill_(float(metadata.get("max_attention_mean", 0.0)))
        self.pattern_max_attention_std.fill_(max(float(metadata.get("max_attention_std", 1.0)), 1e-6))

    def forward(
        self,
        z_q: torch.Tensor,
        executor_features: torch.Tensor,
        mode: str = "gumbel",
        tau: float = 1.0,
        use_query_features: bool = True,
    ) -> PrimitiveOutput:
        if use_query_features:
            x = torch.cat([z_q, executor_features], dim=-1)
            h = self.shared_joint(x)
            trajectory_shape_logits = self.trajectory_shape_head_joint(h)
            attention_pattern_logits = self.attention_pattern_head_joint(h)
            confidence_logits = self.confidence_head_joint(h)
            outcome_logits = self.outcome_head_joint(h)
        else:
            shared_trace = self.shared_trace(executor_features)

            trajectory_input = torch.cat([shared_trace, executor_features[:, :10]], dim=-1)
            trajectory_hidden = self.trajectory_trace_tower(trajectory_input)
            trajectory_shape_logits = self.trajectory_shape_head_trace(trajectory_hidden)

            mean_entropy = executor_features[:, 6]
            std_entropy = executor_features[:, 8]
            entropy_range = executor_features[:, 9]
            max_attention = executor_features[:, 10]
            mean_z = (mean_entropy - self.pattern_mean_entropy_mean) / self.pattern_mean_entropy_std
            attention_z = (max_attention - self.pattern_max_attention_mean) / self.pattern_max_attention_std
            pattern_features = torch.stack(
                [
                    mean_entropy,
                    std_entropy,
                    entropy_range,
                    max_attention,
                    mean_z,
                    attention_z,
                    mean_z.abs(),
                    attention_z.abs(),
                    mean_z * attention_z,
                ],
                dim=-1,
            )
            pattern_input = torch.cat([shared_trace, pattern_features], dim=-1)
            pattern_hidden = self.pattern_trace_tower(pattern_input)
            attention_pattern_logits = self.attention_pattern_head_trace(pattern_hidden)

            confidence_features = executor_features[:, [11, 12, 13, 14]]
            confidence_input = torch.cat([shared_trace, confidence_features], dim=-1)
            confidence_hidden = self.confidence_trace_tower(confidence_input)
            confidence_logits = self.confidence_head_trace(confidence_hidden)

            detached_descriptors = torch.cat(
                [
                    F.softmax(trajectory_shape_logits, dim=-1).detach(),
                    F.softmax(attention_pattern_logits, dim=-1).detach(),
                    F.softmax(confidence_logits, dim=-1).detach(),
                ],
                dim=-1,
            )
            outcome_features = executor_features[:, [11, 12, 10]]
            outcome_input = torch.cat(
                [shared_trace, detached_descriptors, outcome_features],
                dim=-1,
            )
            outcome_hidden = self.outcome_trace_tower(outcome_input)
            outcome_logits = self.outcome_head_trace(outcome_hidden)

        trajectory_shape_discrete = self._discretize(trajectory_shape_logits, mode, tau)
        attention_pattern_discrete = self._discretize(attention_pattern_logits, mode, tau)
        confidence_discrete = self._discretize(confidence_logits, mode, tau)
        outcome_discrete = self._discretize(outcome_logits, mode, tau)

        return PrimitiveOutput(
            trajectory_shape_logits=trajectory_shape_logits,
            attention_pattern_logits=attention_pattern_logits,
            confidence_logits=confidence_logits,
            outcome_logits=outcome_logits,
            trajectory_shape_discrete=trajectory_shape_discrete,
            attention_pattern_discrete=attention_pattern_discrete,
            confidence_discrete=confidence_discrete,
            outcome_discrete=outcome_discrete,
        )

    def _discretize(self, logits: torch.Tensor, mode: str, tau: float) -> torch.Tensor:
        if mode == "hard":
            idx = logits.argmax(dim=-1)
            return F.one_hot(idx, logits.size(-1)).float()
        if mode == "gumbel":
            return F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        if mode == "straight_through":
            soft = F.softmax(logits / tau, dim=-1)
            idx = logits.argmax(dim=-1)
            hard = F.one_hot(idx, logits.size(-1)).float()
            return hard - soft.detach() + soft
        if mode == "soft":
            return F.softmax(logits / tau, dim=-1)
        raise ValueError(f"Unknown discretization mode: {mode}")

    def get_primitive_indices(self, prim_output: PrimitiveOutput) -> Dict[str, torch.Tensor]:
        return {
            "trajectory_shape": prim_output.trajectory_shape_discrete.argmax(dim=-1),
            "attention_pattern": prim_output.attention_pattern_discrete.argmax(dim=-1),
            "confidence": prim_output.confidence_discrete.argmax(dim=-1),
            "outcome": prim_output.outcome_discrete.argmax(dim=-1),
        }
