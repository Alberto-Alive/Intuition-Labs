"""Cooperative-evidence bottleneck for the Extrapolation E3 variant."""

from __future__ import annotations

import math
from typing import Dict, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..executor_schema import (
    EXECUTOR_FEATURE_DIM,
    TRAJECTORY_SUMMARY_SLICE,
    executor_feature_column,
)


class PrimitiveOutput(NamedTuple):
    """Container for public primitive outputs plus internal evidence certificates."""

    trajectory_shape_logits: torch.Tensor
    attention_pattern_logits: torch.Tensor
    confidence_logits: torch.Tensor
    outcome_logits: torch.Tensor
    trajectory_shape_discrete: torch.Tensor
    attention_pattern_discrete: torch.Tensor
    confidence_discrete: torch.Tensor
    outcome_discrete: torch.Tensor
    stability_cert: torch.Tensor
    support_cert: torch.Tensor
    success_guard: torch.Tensor
    success_core: torch.Tensor
    success_residual_weight: torch.Tensor
    success_residual: torch.Tensor
    success_base: torch.Tensor
    success_bonus_weight: torch.Tensor
    success_bonus_weight_zero_raw: torch.Tensor
    success_bonus: torch.Tensor
    success_bonus_zero_raw: torch.Tensor
    success_proposal: torch.Tensor
    success_cert: torch.Tensor
    failure_cert: torch.Tensor
    ambiguity_cert: torch.Tensor
    success_trace_strength: torch.Tensor


class PositiveLinear(nn.Module):
    """Linear layer with nonnegative weights."""

    def __init__(self, in_features: int, out_features: int, init_weight: float = 0.01):
        super().__init__()
        self.raw_weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.constant_(self.raw_weight, _inverse_softplus(init_weight))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.softplus(self.raw_weight)
        return F.linear(x, weight, self.bias)


class MonotoneCertHead(nn.Module):
    """Partial-monotone certificate head over explicitly ordered badness features."""

    def __init__(self, input_dim: int, hidden_dim: int, scale_init: float = 2.0):
        super().__init__()
        self.hidden = PositiveLinear(input_dim, hidden_dim)
        self.output = PositiveLinear(hidden_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        self.raw_scale = nn.Parameter(torch.tensor(_inverse_softplus(scale_init)))

    def forward(self, badness: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.hidden(badness))
        risk = self.output(hidden) + self.bias
        scale = F.softplus(self.raw_scale).clamp_min(1e-4)
        return torch.sigmoid(-scale * risk).squeeze(-1)


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-6)
    return math.log(math.expm1(value))


def _make_mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


class EntropyBottleneckHead(nn.Module):
    """Produces public primitives from cooperative internal evidence heads."""

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
        monotone_hidden_dim: int = 96,
        success_min_support_ratio: float = 0.01,
        success_min_group_size_norm: float = 0.15,
        confidence_threshold_init: float = 0.70,
        confidence_threshold_gap_init: float = 0.45,
        outcome_threshold_init: float = 0.90,
        outcome_threshold_gap_init: float = 0.60,
        monotone_scale_init: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        del confidence_threshold_init
        del confidence_threshold_gap_init
        del outcome_threshold_init
        del outcome_threshold_gap_init

        input_dim = d_model + executor_dim
        self.trace_hidden_dim = trace_hidden_dim
        self.success_min_support_ratio = float(success_min_support_ratio)
        self.success_min_group_size_norm = float(success_min_group_size_norm)

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

        def _make_trace_tower(input_width: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_width, trace_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(trace_hidden_dim),
            )

        self.trajectory_shape_head_joint = nn.Linear(hidden_dim, num_trajectory)
        self.attention_pattern_head_joint = nn.Linear(hidden_dim, num_pattern)

        self.trajectory_trace_tower = _make_trace_tower(trace_hidden_dim + 10)
        self.pattern_trace_tower = _make_trace_tower(trace_hidden_dim + 9)
        self.trajectory_shape_head_trace = nn.Linear(trace_hidden_dim, num_trajectory)
        self.attention_pattern_head_trace = nn.Linear(trace_hidden_dim, num_pattern)

        self.stability_cert_head = MonotoneCertHead(input_dim=8, hidden_dim=monotone_hidden_dim, scale_init=monotone_scale_init)
        self.support_cert_head = MonotoneCertHead(input_dim=4, hidden_dim=max(16, monotone_hidden_dim // 2), scale_init=monotone_scale_init)
        self.success_guard_head = MonotoneCertHead(
            input_dim=12,
            hidden_dim=monotone_hidden_dim,
            scale_init=monotone_scale_init,
        )

        success_raw_input_dim = 4
        failure_input_dim = trace_hidden_dim + 2 + num_trajectory + num_pattern + 5
        ambiguity_input_dim = trace_hidden_dim + 6 + 7
        outcome_input_dim = 5 + num_trajectory + num_pattern
        self.success_raw_bonus_mlp = _make_mlp(
            success_raw_input_dim,
            hidden_dim=max(16, monotone_hidden_dim // 2),
            output_dim=1,
            dropout=dropout,
        )
        self.failure_cert_mlp = _make_mlp(failure_input_dim, hidden_dim=monotone_hidden_dim, output_dim=1, dropout=dropout)
        self.ambiguity_cert_mlp = _make_mlp(ambiguity_input_dim, hidden_dim=monotone_hidden_dim, output_dim=1, dropout=dropout)
        self.confidence_head = _make_mlp(5, hidden_dim=max(16, monotone_hidden_dim // 2), output_dim=num_confidence, dropout=dropout)
        self.outcome_head = _make_mlp(outcome_input_dim, hidden_dim=monotone_hidden_dim, output_dim=num_outcome, dropout=dropout)
        self.success_residual_limit = 0.20
        self.success_bonus_limit = 0.30

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
        path_scales: Dict[str, float] | None = None,
    ) -> PrimitiveOutput:
        scales = {
            "shared_trace": 1.0,
            "success_trace": 1.0,
            "success_core": 1.0,
            "success_residual": 1.0,
            "success_bonus": 1.0,
            "failure_trace": 1.0,
            "ambiguity_trace": 1.0,
            "cert_context": 1.0,
            "outcome_context": 1.0,
            "outcome_trajectory_context": 1.0,
            "outcome_pattern_context": 1.0,
            "success_raw": 1.0,
            "failure_raw": 1.0,
            "ambiguity_raw": 1.0,
        }
        if path_scales:
            if "shared_trace" in path_scales:
                trace_scale = float(path_scales["shared_trace"])
                if "success_trace" not in path_scales:
                    scales["success_trace"] = trace_scale
                if "failure_trace" not in path_scales:
                    scales["failure_trace"] = trace_scale
                if "ambiguity_trace" not in path_scales:
                    scales["ambiguity_trace"] = trace_scale
            if "success_base" in path_scales:
                base_scale = float(path_scales["success_base"])
                if "success_core" not in path_scales:
                    scales["success_core"] = base_scale
                if "success_residual" not in path_scales:
                    scales["success_residual"] = base_scale
            if "context" in path_scales:
                context_scale = float(path_scales["context"])
                if "cert_context" not in path_scales:
                    scales["cert_context"] = context_scale
                if "outcome_context" not in path_scales:
                    scales["outcome_context"] = context_scale
                if "outcome_trajectory_context" not in path_scales:
                    scales["outcome_trajectory_context"] = context_scale
                if "outcome_pattern_context" not in path_scales:
                    scales["outcome_pattern_context"] = context_scale
            if "outcome_context" in path_scales:
                outcome_scale = float(path_scales["outcome_context"])
                if "outcome_trajectory_context" not in path_scales:
                    scales["outcome_trajectory_context"] = outcome_scale
                if "outcome_pattern_context" not in path_scales:
                    scales["outcome_pattern_context"] = outcome_scale
            scales.update({key: float(value) for key, value in path_scales.items()})

        shared_trace = self.shared_trace(executor_features)

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

        trajectory_probs = F.softmax(trajectory_shape_logits, dim=-1)
        attention_pattern_probs = F.softmax(attention_pattern_logits, dim=-1)
        cert_trajectory_context = trajectory_probs.detach() * scales["cert_context"]
        cert_attention_pattern_context = attention_pattern_probs.detach() * scales["cert_context"]
        outcome_trajectory_context = trajectory_probs.detach() * scales["outcome_trajectory_context"]
        outcome_attention_pattern_context = attention_pattern_probs.detach() * scales["outcome_pattern_context"]
        success_trace_context = shared_trace * scales["success_trace"]
        success_trace_strength = success_trace_context.norm(dim=-1) / math.sqrt(success_trace_context.size(-1))
        failure_trace_context = shared_trace * scales["failure_trace"]
        ambiguity_trace_context = shared_trace * scales["ambiguity_trace"]
        trajectory_good = torch.maximum(trajectory_probs[:, 0], trajectory_probs[:, 1])
        pattern_focused = attention_pattern_probs[:, 0]

        stability_cert = self.stability_cert_head(self._stability_badness_features(executor_features))
        support_cert = self.support_cert_head(self._support_badness_features(executor_features))
        success_guard = self.success_guard_head(self._success_guard_badness_features(executor_features))

        mean_entropy = executor_feature_column(executor_features, "mean_entropy")
        max_attention_mass = executor_feature_column(executor_features, "max_attention_mass")
        agreement = executor_feature_column(executor_features, "agreement")
        prob_margin = executor_feature_column(executor_features, "prob_margin")
        max_softmax = executor_feature_column(executor_features, "max_softmax")
        predictive_entropy = executor_feature_column(executor_features, "predictive_entropy")
        variation_ratio = executor_feature_column(executor_features, "variation_ratio")
        abs_margin = executor_feature_column(executor_features, "abs_margin")
        low_margin_flag = executor_feature_column(executor_features, "low_margin_flag")
        unstable_flag = executor_feature_column(executor_features, "unstable_flag")
        success_raw_features = torch.stack(
            [
                agreement,
                prob_margin,
                max_softmax,
                abs_margin,
            ],
            dim=-1,
        ) * scales["success_raw"]
        failure_raw_features = torch.stack(
            [
                mean_entropy,
                predictive_entropy,
                variation_ratio,
                low_margin_flag,
                unstable_flag,
            ],
            dim=-1,
        ) * scales["failure_raw"]
        ambiguity_raw_features = torch.stack(
            [
                1.0 - max_attention_mass,
                1.0 - agreement,
                mean_entropy,
                predictive_entropy,
                variation_ratio,
                low_margin_flag,
                unstable_flag,
            ],
            dim=-1,
        ) * scales["ambiguity_raw"]

        success_core = torch.clamp(success_guard * scales["success_core"], min=0.0, max=1.0)
        success_residual_weight = torch.zeros_like(success_core)
        success_residual = torch.zeros_like(success_core)
        success_base = success_core
        success_bonus_weight = torch.sigmoid(self.success_raw_bonus_mlp(success_raw_features)).squeeze(-1)
        if self.training:
            success_bonus_weight_zero_raw = torch.zeros_like(success_bonus_weight)
        else:
            success_bonus_weight_zero_raw = torch.sigmoid(
                self.success_raw_bonus_mlp(torch.zeros_like(success_raw_features))
            ).squeeze(-1)
        success_bonus = self.success_bonus_limit * success_guard * success_bonus_weight
        success_bonus_zero_raw = self.success_bonus_limit * success_guard * success_bonus_weight_zero_raw
        success_bonus = success_bonus * scales["success_bonus"]
        success_bonus_zero_raw = success_bonus_zero_raw * scales["success_bonus"]
        success_proposal = success_base + (1.0 - success_base) * success_bonus
        success_cert = success_proposal

        failure_input = torch.cat(
            [
                failure_trace_context,
                (1.0 - stability_cert).unsqueeze(-1),
                (1.0 - support_cert).unsqueeze(-1),
                cert_trajectory_context,
                cert_attention_pattern_context,
                failure_raw_features,
            ],
            dim=-1,
        )
        failure_cert = torch.sigmoid(self.failure_cert_mlp(failure_input)).squeeze(-1)

        certificate_gap = (success_cert - failure_cert).abs()
        certificate_weakness = 1.0 - torch.maximum(success_cert, failure_cert)
        ambiguity_input = torch.cat(
            [
                ambiguity_trace_context,
                success_cert.unsqueeze(-1),
                failure_cert.unsqueeze(-1),
                stability_cert.unsqueeze(-1),
                support_cert.unsqueeze(-1),
                certificate_gap.unsqueeze(-1),
                certificate_weakness.unsqueeze(-1),
                ambiguity_raw_features,
            ],
            dim=-1,
        )
        ambiguity_cert = torch.sigmoid(self.ambiguity_cert_mlp(ambiguity_input)).squeeze(-1)

        evidence_vector = torch.stack(
            [
                stability_cert,
                support_cert,
                ambiguity_cert,
                success_cert,
                failure_cert,
            ],
            dim=-1,
        )
        confidence_logits = self.confidence_head(evidence_vector)
        outcome_input = torch.cat(
            [evidence_vector, outcome_trajectory_context, outcome_attention_pattern_context],
            dim=-1,
        )
        outcome_logits = self.outcome_head(outcome_input)

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
            stability_cert=stability_cert,
            support_cert=support_cert,
            success_guard=success_guard,
            success_core=success_core,
            success_residual_weight=success_residual_weight,
            success_residual=success_residual,
            success_base=success_base,
            success_bonus_weight=success_bonus_weight,
            success_bonus_weight_zero_raw=success_bonus_weight_zero_raw,
            success_bonus=success_bonus,
            success_bonus_zero_raw=success_bonus_zero_raw,
            success_proposal=success_proposal,
            success_cert=success_cert,
            failure_cert=failure_cert,
            ambiguity_cert=ambiguity_cert,
            success_trace_strength=success_trace_strength,
        )

    def _stability_badness_features(self, executor_features: torch.Tensor) -> torch.Tensor:
        mean_entropy = executor_feature_column(executor_features, "mean_entropy")
        predictive_entropy = executor_feature_column(executor_features, "predictive_entropy")
        max_attention_mass = executor_feature_column(executor_features, "max_attention_mass")
        agreement = executor_feature_column(executor_features, "agreement")
        prob_margin = executor_feature_column(executor_features, "prob_margin")
        variation_ratio = executor_feature_column(executor_features, "variation_ratio")
        low_margin_flag = executor_feature_column(executor_features, "low_margin_flag")
        unstable_flag = executor_feature_column(executor_features, "unstable_flag")
        return torch.stack(
            [
                mean_entropy,
                predictive_entropy,
                1.0 - max_attention_mass,
                1.0 - agreement,
                1.0 - prob_margin,
                variation_ratio,
                low_margin_flag,
                unstable_flag,
            ],
            dim=-1,
        )

    def _support_badness_features(self, executor_features: torch.Tensor) -> torch.Tensor:
        support_ratio = executor_feature_column(executor_features, "support_ratio")
        n_norm = executor_feature_column(executor_features, "n_norm")
        support_ratio_scaled = torch.log1p(support_ratio * 100.0) / math.log1p(100.0)
        support_deficit = F.relu(self.success_min_support_ratio - support_ratio)
        group_deficit = F.relu(self.success_min_group_size_norm - n_norm)
        return torch.stack(
            [
                1.0 - support_ratio_scaled,
                1.0 - n_norm,
                support_deficit,
                group_deficit,
            ],
            dim=-1,
        )

    def _success_guard_badness_features(self, executor_features: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                self._stability_badness_features(executor_features),
                self._support_badness_features(executor_features),
            ],
            dim=-1,
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
