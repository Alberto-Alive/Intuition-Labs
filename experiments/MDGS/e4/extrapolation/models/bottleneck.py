"""Epistemic-anchor bottleneck for the Extrapolation E4 variant."""

from __future__ import annotations

import math
from typing import Dict, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..executor_schema import (
    EXECUTOR_FEATURE_DIM,
    TRAJECTORY_SUMMARY_SLICE,
    UNCERTAINTY_FEATURE_SLICE,
    executor_feature_column,
)

FAMILY_NAMES = ("success", "failure", "boundary")


class PrimitiveOutput(NamedTuple):
    """Container for public primitive outputs plus anchor-derived audit tensors."""

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
    anchor_state: torch.Tensor
    anchor_prev_state: torch.Tensor
    anchor_transition: torch.Tensor
    success_family_support: torch.Tensor
    failure_family_support: torch.Tensor
    boundary_family_support: torch.Tensor
    success_family_approach: torch.Tensor
    failure_family_approach: torch.Tensor
    boundary_family_approach: torch.Tensor
    success_family_leap_penalty: torch.Tensor
    failure_family_leap_penalty: torch.Tensor
    boundary_family_leap_penalty: torch.Tensor
    anchor_conflict: torch.Tensor
    commitment_depth: torch.Tensor
    top_family_id: torch.Tensor
    top_anchor_ids: torch.Tensor
    anchor_assignment_probs: torch.Tensor
    anchor_support_scores: torch.Tensor
    anchor_approach_scores: torch.Tensor
    anchor_leap_scores: torch.Tensor
    anchor_centers: torch.Tensor
    anchor_scales: torch.Tensor
    anchor_directions: torch.Tensor
    anchor_leap_scales: torch.Tensor


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
    """Partial-monotone certainty head over explicitly ordered badness features."""

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


class OrderedThresholdHead(nn.Module):
    """Three-class ordinal head driven by a single scalar risk."""

    def __init__(
        self,
        threshold_init: float,
        threshold_gap_init: float,
        scale_init: float = 4.0,
        reorder_indices: tuple[int, int, int] = (0, 1, 2),
    ):
        super().__init__()
        self.raw_threshold = nn.Parameter(torch.tensor(float(threshold_init)))
        self.raw_gap = nn.Parameter(torch.tensor(_inverse_softplus(float(threshold_gap_init))))
        self.raw_scale = nn.Parameter(torch.tensor(_inverse_softplus(float(scale_init))))
        self.reorder_indices = reorder_indices

    def thresholds(self) -> torch.Tensor:
        threshold_0 = self.raw_threshold
        threshold_1 = threshold_0 + F.softplus(self.raw_gap).clamp_min(1e-4)
        return torch.stack([threshold_0, threshold_1])

    def forward(self, risk: torch.Tensor) -> torch.Tensor:
        thresholds = self.thresholds().to(risk.device)
        scale = F.softplus(self.raw_scale).clamp_min(1e-4)
        cdf_0 = torch.sigmoid(scale * (thresholds[0] - risk))
        cdf_1 = torch.sigmoid(scale * (thresholds[1] - risk))
        ordinal_probs = torch.stack(
            [
                cdf_0,
                (cdf_1 - cdf_0).clamp_min(1e-6),
                (1.0 - cdf_1).clamp_min(1e-6),
            ],
            dim=-1,
        )
        probs = ordinal_probs[..., list(self.reorder_indices)]
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.log(probs.clamp_min(1e-6))


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
    """Produces public primitives from an anchor-structured epistemic state."""

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
        anchor_state_dim: int = 48,
        anchors_per_family: int = 4,
        anchor_support_temperature: float = 0.25,
        anchor_assignment_mode: str = "soft",
        anchor_use_approach: bool = True,
        anchor_use_leap: bool = True,
        anchor_use_boundary_family: bool = True,
        anchor_use_barrier: bool = True,
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
        del num_confidence
        del num_outcome

        input_dim = d_model + executor_dim
        self.trace_hidden_dim = trace_hidden_dim
        self.success_min_support_ratio = float(success_min_support_ratio)
        self.success_min_group_size_norm = float(success_min_group_size_norm)
        self.anchor_state_dim = int(anchor_state_dim)
        self.anchors_per_family = int(anchors_per_family)
        self.anchor_support_temperature = float(anchor_support_temperature)
        self.anchor_assignment_mode = str(anchor_assignment_mode)
        self.anchor_use_approach = bool(anchor_use_approach)
        self.anchor_use_leap = bool(anchor_use_leap)
        self.anchor_use_boundary_family = bool(anchor_use_boundary_family)
        self.anchor_use_barrier = bool(anchor_use_barrier)

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

        anchor_input_dim = trace_hidden_dim + num_trajectory + num_pattern + (executor_dim - 6)
        self.anchor_state_proj = nn.Sequential(
            nn.Linear(anchor_input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, anchor_state_dim),
        )
        self.anchor_prev_proj = nn.Sequential(
            nn.Linear(trace_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, anchor_state_dim),
        )

        self.anchor_centers_param = nn.Parameter(torch.randn(len(FAMILY_NAMES), anchors_per_family, anchor_state_dim) * 0.02)
        self.anchor_raw_scales = nn.Parameter(
            torch.full(
                (len(FAMILY_NAMES), anchors_per_family, anchor_state_dim),
                _inverse_softplus(1.0),
            )
        )
        self.anchor_directions_param = nn.Parameter(torch.randn(len(FAMILY_NAMES), anchors_per_family, anchor_state_dim))
        self.anchor_raw_leap_scales = nn.Parameter(
            torch.full(
                (len(FAMILY_NAMES), anchors_per_family),
                _inverse_softplus(1.0),
            )
        )

        self.commitment_head = MonotoneCertHead(
            input_dim=6,
            hidden_dim=monotone_hidden_dim,
            scale_init=monotone_scale_init,
        )
        self.outcome_safety_head = MonotoneCertHead(
            input_dim=7,
            hidden_dim=monotone_hidden_dim,
            scale_init=monotone_scale_init,
        )
        self.confidence_head = OrderedThresholdHead(
            threshold_init=confidence_threshold_init,
            threshold_gap_init=confidence_threshold_gap_init,
            scale_init=monotone_scale_init,
            reorder_indices=(2, 1, 0),
        )
        self.outcome_head = OrderedThresholdHead(
            threshold_init=outcome_threshold_init,
            threshold_gap_init=outcome_threshold_gap_init,
            scale_init=monotone_scale_init,
            reorder_indices=(0, 1, 2),
        )
        self.commitment_rule_explicit_weight = 0.80
        self.outcome_rule_raw_scale = nn.Parameter(torch.tensor(_inverse_softplus(1.0)))
        self.outcome_rule_bias = nn.Parameter(torch.zeros(3))

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
        scales = self._resolve_path_scales(path_scales)
        shared_trace = self.shared_trace(executor_features) * scales["shared_trace"]

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

        uncertainty_features = executor_features[:, UNCERTAINTY_FEATURE_SLICE]
        anchor_input = torch.cat(
            [
                shared_trace,
                trajectory_probs.detach(),
                attention_pattern_probs.detach(),
                uncertainty_features,
            ],
            dim=-1,
        )
        anchor_state = self.anchor_state_proj(anchor_input)
        anchor_prev_state = self.anchor_prev_proj(shared_trace)
        anchor_transition = anchor_state - anchor_prev_state
        anchor_transition_norm = anchor_transition.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        anchor_transition_dir = anchor_transition / anchor_transition_norm

        family_supports = []
        family_approaches = []
        family_leaps = []
        family_top_anchor_ids = []
        family_assignments = []
        family_anchor_supports = []
        family_anchor_approaches = []
        family_anchor_leaps = []

        anchor_centers = self.anchor_centers_param
        anchor_scales = F.softplus(self.anchor_raw_scales).clamp_min(1e-4)
        anchor_directions = F.normalize(self.anchor_directions_param, dim=-1, eps=1e-6)
        anchor_leap_scales = F.softplus(self.anchor_raw_leap_scales).clamp_min(1e-4)

        for family_idx, family_name in enumerate(FAMILY_NAMES):
            family_scale_key = f"{family_name}_family"
            family_scale = scales[family_scale_key]
            centers = anchor_centers[family_idx]
            scales_diag = anchor_scales[family_idx]
            directions = anchor_directions[family_idx]
            leap_scales = anchor_leap_scales[family_idx]

            delta = (anchor_state.unsqueeze(1) - centers.unsqueeze(0)) / scales_diag.unsqueeze(0)
            dist_sq = (delta ** 2).sum(dim=-1)
            anchor_support = torch.exp(-0.5 * dist_sq)
            if self.anchor_assignment_mode == "hard":
                hard_idx = dist_sq.argmin(dim=-1)
                assignment = F.one_hot(hard_idx, num_classes=self.anchors_per_family).float()
            else:
                assignment = F.softmax(
                    (-dist_sq) / max(self.anchor_support_temperature, 1e-4),
                    dim=-1,
                )

            approach = ((anchor_transition_dir.unsqueeze(1) * directions.unsqueeze(0)).sum(dim=-1) + 1.0) * 0.5
            if not self.anchor_use_approach or scales["approach"] == 0.0:
                approach = torch.ones_like(approach)
            leap_ratio = anchor_transition_norm / leap_scales.unsqueeze(0)
            leap_penalty = (leap_ratio / (1.0 + leap_ratio)).clamp(min=0.0, max=1.0)
            if not self.anchor_use_leap or scales["leap"] == 0.0:
                leap_penalty = torch.zeros_like(leap_penalty)

            family_support = (assignment * anchor_support).sum(dim=-1) * family_scale
            family_approach = (assignment * approach).sum(dim=-1)
            family_leap = (assignment * leap_penalty).sum(dim=-1)
            if family_scale == 0.0:
                family_approach = torch.zeros_like(family_approach)
                family_leap = torch.zeros_like(family_leap)

            family_supports.append(family_support)
            family_approaches.append(family_approach)
            family_leaps.append(family_leap)
            family_top_anchor_ids.append(assignment.argmax(dim=-1))
            family_assignments.append(assignment)
            family_anchor_supports.append(anchor_support * family_scale)
            family_anchor_approaches.append(approach * family_scale)
            family_anchor_leaps.append(leap_penalty * family_scale)

        family_supports_t = torch.stack(family_supports, dim=-1)
        family_approaches_t = torch.stack(family_approaches, dim=-1)
        family_leaps_t = torch.stack(family_leaps, dim=-1)
        top_anchor_ids = torch.stack(family_top_anchor_ids, dim=-1)
        anchor_assignment_probs = torch.stack(family_assignments, dim=1)
        anchor_support_scores = torch.stack(family_anchor_supports, dim=1)
        anchor_approach_scores = torch.stack(family_anchor_approaches, dim=1)
        anchor_leap_scores = torch.stack(family_anchor_leaps, dim=1)

        success_family_support = family_supports_t[:, 0]
        failure_family_support = family_supports_t[:, 1]
        boundary_family_support = family_supports_t[:, 2]
        success_family_approach = family_approaches_t[:, 0]
        failure_family_approach = family_approaches_t[:, 1]
        boundary_family_approach = family_approaches_t[:, 2]
        success_family_leap_penalty = family_leaps_t[:, 0]
        failure_family_leap_penalty = family_leaps_t[:, 1]
        boundary_family_leap_penalty = family_leaps_t[:, 2]

        family_support_distribution = family_supports_t.clamp_min(1e-6)
        family_support_distribution = family_support_distribution / family_support_distribution.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        anchor_conflict = -(
            family_support_distribution * family_support_distribution.log()
        ).sum(dim=-1) / math.log(len(FAMILY_NAMES))

        top_family_id = family_supports_t.argmax(dim=-1)
        top_family_approach = family_approaches_t.gather(1, top_family_id.unsqueeze(-1)).squeeze(-1)
        top_family_leap_penalty = family_leaps_t.gather(1, top_family_id.unsqueeze(-1)).squeeze(-1)
        topk_family_support = torch.topk(
            family_supports_t,
            k=min(2, family_supports_t.size(-1)),
            dim=-1,
        ).values
        top_family_support = topk_family_support[:, 0]
        runner_up_family_support = (
            topk_family_support[:, 1]
            if topk_family_support.size(-1) > 1
            else torch.zeros_like(top_family_support)
        )
        family_margin = (
            (top_family_support - runner_up_family_support).clamp_min(0.0)
            / (top_family_support + runner_up_family_support).clamp_min(1e-6)
        ).clamp(min=0.0, max=1.0)

        success_support = success_family_support.clamp(min=0.0, max=1.0)
        failure_support = failure_family_support.clamp(min=0.0, max=1.0)
        boundary_support = boundary_family_support.clamp(min=0.0, max=1.0)
        top_support = top_family_support.clamp(min=0.0, max=1.0)
        winning_family_cert = (
            top_support * top_family_approach * (1.0 - top_family_leap_penalty)
        ).clamp(min=0.0, max=1.0)
        success_core = (
            success_support * success_family_approach * (1.0 - success_family_leap_penalty)
        ).clamp(min=0.0, max=1.0)
        failure_core = (
            failure_support * failure_family_approach * (1.0 - failure_family_leap_penalty)
        ).clamp(min=0.0, max=1.0)
        boundary_core = (
            boundary_support
            * (0.5 + 0.5 * boundary_family_approach)
            * (1.0 - 0.5 * boundary_family_leap_penalty)
        ).clamp(min=0.0, max=1.0)
        explicit_commitment = self._explicit_commitment_depth(
            success_support=success_support,
            failure_support=failure_support,
            boundary_support=boundary_support,
            success_family_approach=success_family_approach,
            failure_family_approach=failure_family_approach,
            success_core=success_core,
            failure_core=failure_core,
            anchor_conflict=anchor_conflict,
            family_margin=family_margin,
        )

        badness = torch.stack(
            [
                1.0 - winning_family_cert,
                1.0 - family_margin,
                boundary_support,
                anchor_conflict,
                1.0 - top_family_approach,
                top_family_leap_penalty,
            ],
            dim=-1,
        )
        if self.anchor_use_barrier and scales["barrier"] != 0.0:
            learned_commitment = self.commitment_head(badness * scales["barrier"])
            commitment_depth = torch.lerp(
                learned_commitment,
                explicit_commitment,
                self.commitment_rule_explicit_weight,
            ).clamp(min=0.0, max=1.0)
        else:
            commitment_depth = explicit_commitment

        success_structural = (
            success_core + 0.25 * success_support * family_margin
        ).clamp(min=0.0, max=1.0)
        failure_structural = (
            failure_core + 0.25 * failure_support * family_margin
        ).clamp(min=0.0, max=1.0)
        ambiguity_structural = torch.maximum(
            boundary_core,
            anchor_conflict * (1.0 - 0.5 * family_margin),
        ).clamp(min=0.0, max=1.0)

        confidence_risk = 1.0 - commitment_depth
        support_proxy = top_support
        stability_proxy = 1.0 - top_family_leap_penalty
        success_guard = torch.minimum(commitment_depth, success_structural).clamp(min=0.0, max=1.0)
        success_base = success_family_support
        success_residual_weight = torch.zeros_like(success_core)
        success_residual = torch.zeros_like(success_core)
        success_bonus_weight = torch.zeros_like(success_core)
        success_bonus_weight_zero_raw = torch.zeros_like(success_core)
        success_bonus = torch.zeros_like(success_core)
        success_bonus_zero_raw = torch.zeros_like(success_core)
        success_proposal = torch.minimum(success_guard, success_support)
        success_cert = torch.minimum(
            success_guard,
            success_structural
            * (1.0 - 0.5 * failure_structural)
            * (1.0 - 0.35 * boundary_support)
            * (1.0 - 0.35 * anchor_conflict),
        ).clamp(min=0.0, max=1.0)
        failure_cert = torch.minimum(
            commitment_depth,
            failure_structural
            * (1.0 + 0.25 * family_margin)
            * (1.0 - 0.25 * success_structural)
        ).clamp(min=0.0, max=1.0)
        ambiguity_cert = ambiguity_structural

        confidence_logits = self.confidence_head(confidence_risk)
        outcome_logits = self._explicit_outcome_logits(
            success_cert=success_cert,
            failure_cert=failure_cert,
            ambiguity_cert=ambiguity_cert,
            success_support=success_support,
            failure_support=failure_support,
            boundary_support=boundary_support,
            commitment_depth=commitment_depth,
            top_family_id=top_family_id,
        )
        success_trace_strength = anchor_state.norm(dim=-1) / math.sqrt(anchor_state.size(-1))

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
            stability_cert=stability_proxy,
            support_cert=support_proxy,
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
            anchor_state=anchor_state,
            anchor_prev_state=anchor_prev_state,
            anchor_transition=anchor_transition,
            success_family_support=success_family_support,
            failure_family_support=failure_family_support,
            boundary_family_support=boundary_family_support,
            success_family_approach=success_family_approach,
            failure_family_approach=failure_family_approach,
            boundary_family_approach=boundary_family_approach,
            success_family_leap_penalty=success_family_leap_penalty,
            failure_family_leap_penalty=failure_family_leap_penalty,
            boundary_family_leap_penalty=boundary_family_leap_penalty,
            anchor_conflict=anchor_conflict,
            commitment_depth=commitment_depth,
            top_family_id=top_family_id,
            top_anchor_ids=top_anchor_ids,
            anchor_assignment_probs=anchor_assignment_probs,
            anchor_support_scores=anchor_support_scores,
            anchor_approach_scores=anchor_approach_scores,
            anchor_leap_scores=anchor_leap_scores,
            anchor_centers=anchor_centers,
            anchor_scales=anchor_scales,
            anchor_directions=anchor_directions,
            anchor_leap_scales=anchor_leap_scales,
        )

    def _resolve_path_scales(self, path_scales: Dict[str, float] | None) -> Dict[str, float]:
        scales = {
            "shared_trace": 1.0,
            "success_family": 1.0,
            "failure_family": 1.0,
            "boundary_family": 1.0 if self.anchor_use_boundary_family else 0.0,
            "approach": 1.0 if self.anchor_use_approach else 0.0,
            "leap": 1.0 if self.anchor_use_leap else 0.0,
            "barrier": 1.0 if self.anchor_use_barrier else 0.0,
        }
        if not path_scales:
            return scales
        aliases = {
            "success_trace": "success_family",
            "failure_trace": "failure_family",
            "ambiguity_trace": "boundary_family",
            "success_core": "success_family",
            "success_base": "success_family",
            "success_residual": "success_family",
            "success_bonus": "success_family",
        }
        for key, value in path_scales.items():
            resolved = aliases.get(key, key)
            if resolved in scales:
                scales[resolved] = float(value)
        return scales

    def _explicit_commitment_depth(
        self,
        *,
        success_support: torch.Tensor,
        failure_support: torch.Tensor,
        boundary_support: torch.Tensor,
        success_family_approach: torch.Tensor,
        failure_family_approach: torch.Tensor,
        success_core: torch.Tensor,
        failure_core: torch.Tensor,
        anchor_conflict: torch.Tensor,
        family_margin: torch.Tensor,
    ) -> torch.Tensor:
        decisive_core = torch.maximum(success_core, failure_core)
        decisive_support = torch.maximum(success_support, failure_support)
        decisive_approach = torch.maximum(success_family_approach, failure_family_approach)
        decisive_margin = family_margin.clamp(min=0.0, max=1.0)
        return (
            0.45 * decisive_core
            + 0.20 * decisive_support
            + 0.15 * decisive_approach
            + 0.15 * decisive_margin
            + 0.15 * failure_support
            - 0.16 * boundary_support
            - 0.14 * anchor_conflict
        ).clamp(min=0.0, max=1.0)

    def _explicit_outcome_logits(
        self,
        *,
        success_cert: torch.Tensor,
        failure_cert: torch.Tensor,
        ambiguity_cert: torch.Tensor,
        success_support: torch.Tensor,
        failure_support: torch.Tensor,
        boundary_support: torch.Tensor,
        commitment_depth: torch.Tensor,
        top_family_id: torch.Tensor,
    ) -> torch.Tensor:
        top_is_success = (top_family_id == FAMILY_NAMES.index("success")).float()
        top_is_failure = (top_family_id == FAMILY_NAMES.index("failure")).float()
        decisive_support = torch.maximum(success_support, failure_support)
        certificate_gap = (success_cert - failure_cert).abs().clamp(min=0.0, max=1.0)

        success_score = (
            success_cert
            + 0.15 * success_support
            + 0.20 * top_is_success
            - 0.75 * ambiguity_cert
            - 0.65 * failure_cert
        )
        failure_score = (
            failure_cert
            + 0.65 * failure_support
            + 0.35 * top_is_failure
            + 0.10 * commitment_depth
            - 0.15 * ambiguity_cert
            - 0.70 * success_cert
        )
        uncertain_score = (
            ambiguity_cert
            + 0.45 * (1.0 - commitment_depth)
            + 0.25 * (1.0 - certificate_gap)
            + 0.05 * boundary_support
            - 0.10 * decisive_support
        )
        raw_logits = torch.stack(
            [success_score, uncertain_score, failure_score],
            dim=-1,
        )
        raw_logits = raw_logits - raw_logits.mean(dim=-1, keepdim=True)
        scale = F.softplus(self.outcome_rule_raw_scale).clamp_min(1e-4)
        return raw_logits * scale + self.outcome_rule_bias

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
