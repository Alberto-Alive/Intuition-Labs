"""Entropy bottleneck head for the Extrapolation smoke test."""

from __future__ import annotations

from typing import Dict, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..executor_schema import (
    EXECUTOR_FEATURE_DIM,
    TRAJECTORY_SUMMARY_SLICE,
    UNCERTAINTY_FEATURE_DIM,
    UNCERTAINTY_FEATURE_SLICE,
    executor_feature_column,
)


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
        executor_dim: int = EXECUTOR_FEATURE_DIM,
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
        self.trace_hidden_dim = trace_hidden_dim

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

        self.query_context = nn.Sequential(
            nn.Linear(d_model, trace_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(trace_hidden_dim),
        )

        def _make_trace_tower(input_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, trace_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(trace_hidden_dim),
            )

        self.trajectory_shape_head_joint = nn.Linear(hidden_dim, num_trajectory)
        self.attention_pattern_head_joint = nn.Linear(hidden_dim, num_pattern)

        self.trajectory_trace_tower = _make_trace_tower(trace_hidden_dim + 10)
        self.pattern_trace_tower = _make_trace_tower(trace_hidden_dim + 9)
        self.uncertainty_tower = _make_trace_tower((trace_hidden_dim * 2) + UNCERTAINTY_FEATURE_DIM)
        self.outcome_uncertainty_tower = _make_trace_tower(
            trace_hidden_dim + num_trajectory + num_pattern + num_confidence
        )

        self.trajectory_shape_head_trace = nn.Linear(trace_hidden_dim, num_trajectory)
        self.attention_pattern_head_trace = nn.Linear(trace_hidden_dim, num_pattern)
        self.confidence_head_uncertainty = nn.Linear(trace_hidden_dim, num_confidence)
        self.outcome_head_uncertainty = nn.Linear(trace_hidden_dim, num_outcome)

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
            query_context = self.query_context(z_q)
        else:
            # In trace-driven mode the uncertainty tower should depend only on trace uncertainty signals.
            query_context = torch.zeros(
                z_q.size(0),
                self.trace_hidden_dim,
                device=z_q.device,
                dtype=z_q.dtype,
            )
        shared_trace = self.shared_trace(executor_features)
        uncertainty_features = self._explicit_uncertainty_features(executor_features)
        uncertainty_input = torch.cat([query_context, shared_trace, uncertainty_features], dim=-1)
        uncertainty_hidden = self.uncertainty_tower(uncertainty_input)

        if use_query_features:
            x = torch.cat([z_q, executor_features], dim=-1)
            h = self.shared_joint(x)
            trajectory_shape_logits = self.trajectory_shape_head_joint(h)
            attention_pattern_logits = self.attention_pattern_head_joint(h)
        else:
            trajectory_input = torch.cat([shared_trace, executor_features[:, TRAJECTORY_SUMMARY_SLICE]], dim=-1)
            trajectory_hidden = self.trajectory_trace_tower(trajectory_input)
            trajectory_shape_logits = self.trajectory_shape_head_trace(trajectory_hidden)

            mean_entropy = executor_feature_column(executor_features, "mean_entropy")
            std_entropy = executor_feature_column(executor_features, "std_entropy")
            entropy_range = executor_feature_column(executor_features, "entropy_range")
            max_attention = executor_feature_column(executor_features, "max_attention_mass")
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

        confidence_logits = self.confidence_head_uncertainty(uncertainty_hidden)
        detached_descriptors = torch.cat(
            [
                F.softmax(trajectory_shape_logits, dim=-1).detach(),
                F.softmax(attention_pattern_logits, dim=-1).detach(),
                F.softmax(confidence_logits, dim=-1).detach(),
            ],
            dim=-1,
        )
        outcome_input = torch.cat([uncertainty_hidden, detached_descriptors], dim=-1)
        outcome_hidden = self.outcome_uncertainty_tower(outcome_input)
        outcome_logits = self.outcome_head_uncertainty(outcome_hidden)

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

    def _explicit_uncertainty_features(self, executor_features: torch.Tensor) -> torch.Tensor:
        return executor_features[:, UNCERTAINTY_FEATURE_SLICE]

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
