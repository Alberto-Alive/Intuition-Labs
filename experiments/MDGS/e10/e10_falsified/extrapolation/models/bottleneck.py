"""Cooperative diffusion-evidence bottleneck for DIGIT Extrapolation E10.

Architecture:
  Each reconstructed diffusion path emits shared evidence:
    order / boundary / support.
  Public certainty, guard, confidence, outcome, and commitment all read from
  that same aggregated path evidence state.
  Prototype geometry regularises the diffusion evidence manifold but does not
  replace it.
"""

from __future__ import annotations

import math
from typing import Any, Dict, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..executor_schema import EXECUTOR_FEATURE_DIM, TRAJECTORY_SUMMARY_SLICE, executor_feature_column


FAMILY_NAMES = ("success", "failure", "boundary")


class PrimitiveOutput(NamedTuple):
    """Container for public primitive outputs plus diffusion-derived audit tensors."""

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
    success_affinity: torch.Tensor
    success_bonus_weight: torch.Tensor
    success_bonus_weight_zero_raw: torch.Tensor
    success_bonus: torch.Tensor
    success_bonus_zero_raw: torch.Tensor
    success_proposal: torch.Tensor
    success_cert: torch.Tensor
    failure_cert: torch.Tensor
    ambiguity_cert: torch.Tensor
    cert_risk: torch.Tensor
    certainty_score: torch.Tensor
    fragility_risk: torch.Tensor
    fragility_score: torch.Tensor
    effective_certainty: torch.Tensor
    decisiveness_score: torch.Tensor
    prototype_top_support: torch.Tensor
    prototype_total_support: torch.Tensor
    prototype_family_margin: torch.Tensor
    prototype_overlap: torch.Tensor
    prototype_family_switch_rate: torch.Tensor
    prototype_anchor_switch_rate: torch.Tensor
    prototype_nearest_anchor_distance: torch.Tensor
    prototype_pull_loss: torch.Tensor
    prototype_usage_loss: torch.Tensor
    prototype_repulsion_loss: torch.Tensor
    prototype_family_support_paths: torch.Tensor
    prototype_family_distribution_paths: torch.Tensor
    prototype_order_score: torch.Tensor
    prototype_order_score_paths: torch.Tensor
    prototype_order_score_spread: torch.Tensor
    path_order_score_paths: torch.Tensor
    path_boundary_paths: torch.Tensor
    path_support_paths: torch.Tensor
    path_outcome_prob_paths: torch.Tensor
    path_reconstruction_energy: torch.Tensor

    trust_support_deficit: torch.Tensor
    trust_approach_deficit: torch.Tensor
    trust_leap_cost: torch.Tensor
    trust_conflict: torch.Tensor
    trust_agreement_deficit: torch.Tensor
    trust_margin_deficit: torch.Tensor
    trust_concentration_deficit: torch.Tensor
    trust_variation_pressure: torch.Tensor

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

    diffusion_denoise_loss: torch.Tensor


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


def _discretise(logits: torch.Tensor, mode: str, tau: float) -> torch.Tensor:
    if mode == "hard":
        idx = logits.argmax(dim=-1)
        return F.one_hot(idx, num_classes=logits.size(-1)).float()
    if mode == "gumbel":
        return F.gumbel_softmax(logits, tau=tau, hard=False)
    if mode == "straight_through":
        return F.gumbel_softmax(logits, tau=tau, hard=True)
    return F.softmax(logits / max(tau, 1e-4), dim=-1)


def _ordered_score_from_family_distribution(family_distribution: torch.Tensor) -> torch.Tensor:
    """Return an ordered semantic score: success > boundary > failure."""
    return family_distribution[..., 0] - family_distribution[..., 1]


def _outcome_targets_to_family_targets(outcome_targets: torch.Tensor) -> torch.Tensor:
    """Map outcome ids to prototype-family ids: success / failure / boundary."""
    return torch.where(
        outcome_targets == 2,
        torch.ones_like(outcome_targets),
        torch.where(outcome_targets == 1, torch.full_like(outcome_targets, 2), outcome_targets),
    )


def _path_outcome_probs_from_order_raw(
    order_raw: torch.Tensor,
    boundary: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    """Convert per-path raw order / boundary / support into semantic outcome probabilities."""
    success_side = torch.sigmoid(order_raw)
    failure_side = torch.sigmoid(-order_raw)
    admissible = support * (1.0 - boundary)
    outcome_probs = torch.stack(
        [
            admissible * success_side,
            1.0 - admissible,
            admissible * failure_side,
        ],
        dim=-1,
    ).clamp_min(1e-6)
    return outcome_probs / outcome_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)


class OrderedThresholdHead(nn.Module):
    """Three-class ordinal head driven by a single scalar signal in [0,1]."""

    def __init__(
        self,
        threshold_init: float = 0.35,
        threshold_gap_init: float = 0.30,
        scale_init: float = 4.0,
        reorder_indices: tuple = (0, 1, 2),
    ):
        super().__init__()
        self.raw_threshold = nn.Parameter(torch.tensor(float(threshold_init)))
        self.raw_gap = nn.Parameter(torch.tensor(_inverse_softplus(float(threshold_gap_init))))
        self.raw_scale = nn.Parameter(torch.tensor(_inverse_softplus(float(scale_init))))
        self.reorder_indices = reorder_indices

    def thresholds(self) -> torch.Tensor:
        t0 = self.raw_threshold
        t1 = t0 + F.softplus(self.raw_gap).clamp_min(1e-4)
        return torch.stack([t0, t1])

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        thresholds = self.thresholds().to(signal.device)
        scale = F.softplus(self.raw_scale).clamp_min(1e-4)
        cdf_0 = torch.sigmoid(scale * (thresholds[0] - signal))
        cdf_1 = torch.sigmoid(scale * (thresholds[1] - signal))
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


class PositiveLinear(nn.Module):
    """Linear layer with nonnegative weights for monotone badness heads."""

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


class PathEvidenceHead(nn.Module):
    """Decode each reconstructed path into shared evidence coordinates."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = _make_mlp(input_dim, hidden_dim, 3, dropout)

    def forward(self, z_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.net(z_hat)
        order_raw = raw[..., 0]
        boundary = torch.sigmoid(raw[..., 1])
        support = torch.sigmoid(raw[..., 2])
        order_score = torch.tanh(order_raw)
        outcome_probs = _path_outcome_probs_from_order_raw(order_raw, boundary, support)
        return order_score, boundary, support, outcome_probs


class LatentDenoiser(nn.Module):
    """Predicts the diffusion noise epsilon from the noised latent and timestep."""

    def __init__(self, d_model: int, hidden_dim: int, num_steps: int, dropout: float = 0.1):
        super().__init__()
        t_emb_dim = 32
        self.t_embed = nn.Embedding(num_steps + 1, t_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(d_model + t_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_embed(t.clamp(0, self.t_embed.num_embeddings - 1))
        return self.net(torch.cat([z_t, t_emb], dim=-1))


class DiffusionTrustHead(nn.Module):
    """Maps diffusion certainty statistics to a certainty score in [0,1]."""

    STATS_DIM = 5

    def __init__(
        self,
        hidden_dim: int = 128,
        confidence_threshold_init: float = 0.35,
        confidence_threshold_gap_init: float = 0.30,
        scale_init: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.STATS_DIM, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.certainty_proj = nn.Linear(hidden_dim, 1)

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        h = self.net(stats)
        return torch.sigmoid(self.certainty_proj(h)).squeeze(-1)


class DiffusionFragilityHead(nn.Module):
    """Maps certainty + prototype geometry statistics to a fragility risk."""

    def __init__(self, input_dim: int, hidden_dim: int = 96, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(stats)).squeeze(-1)


class DiffusionPrototypeBank(nn.Module):
    """Prototype geometry evaluated on reconstructed diffusion-path states."""

    def __init__(
        self,
        state_dim: int,
        prototypes_per_family: int = 4,
        num_families: int = len(FAMILY_NAMES),
        support_temperature: float = 0.25,
        scale_init: float = 0.55,
        scale_max: float = 0.70,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.prototypes_per_family = prototypes_per_family
        self.num_families = num_families
        self.support_temperature = support_temperature
        self.scale_max = scale_max

        self.centers = nn.Parameter(torch.randn(num_families, prototypes_per_family, state_dim) * 0.02)
        self.raw_scales = nn.Parameter(
            torch.full(
                (num_families, prototypes_per_family, state_dim),
                _inverse_softplus(scale_init),
            )
        )

    def forward(self, path_states: torch.Tensor) -> dict[str, torch.Tensor]:
        num_paths, _bsz, _ = path_states.shape
        scales = F.softplus(self.raw_scales).clamp(min=1e-4, max=self.scale_max)

        family_support_paths = []
        family_assignment_paths = []
        family_anchor_support_paths = []
        family_dist_paths = []

        for family_idx in range(self.num_families):
            centers = self.centers[family_idx]
            family_scales = scales[family_idx]
            delta = (
                path_states.unsqueeze(2) - centers.unsqueeze(0).unsqueeze(0)
            ) / family_scales.unsqueeze(0).unsqueeze(0)
            dist_sq = (delta ** 2).sum(dim=-1)
            anchor_support = torch.exp(-dist_sq)
            assignment = F.softmax(
                (-dist_sq) / max(self.support_temperature, 1e-4),
                dim=-1,
            )
            family_support = (assignment * anchor_support).sum(dim=-1)

            family_support_paths.append(family_support)
            family_assignment_paths.append(assignment)
            family_anchor_support_paths.append(anchor_support)
            family_dist_paths.append(dist_sq)

        family_support_paths_t = torch.stack(family_support_paths, dim=-1)
        family_distribution_paths = family_support_paths_t.clamp_min(1e-6)
        family_distribution_paths = family_distribution_paths / family_distribution_paths.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)

        family_support = family_support_paths_t.mean(dim=0)
        family_distribution = family_support.clamp_min(1e-6)
        family_distribution = family_distribution / family_distribution.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        order_score_paths = _ordered_score_from_family_distribution(family_distribution_paths)
        order_score = order_score_paths.mean(dim=0)
        order_score_spread = order_score_paths.std(dim=0, unbiased=False)

        topk = torch.topk(family_support, k=min(2, self.num_families), dim=-1).values
        top_support = topk[:, 0].clamp(min=0.0, max=1.0)
        runner_up = topk[:, 1] if topk.size(-1) > 1 else torch.zeros_like(top_support)
        family_margin = (
            (top_support - runner_up).clamp_min(0.0)
            / (top_support + runner_up).clamp_min(1e-6)
        ).clamp(min=0.0, max=1.0)
        family_entropy = -(
            family_distribution * family_distribution.log()
        ).sum(dim=-1) / math.log(self.num_families)
        non_top_support = (
            (family_support.sum(dim=-1) - top_support) / max(self.num_families - 1, 1)
        ).clamp(min=0.0, max=1.0)
        overlap = (non_top_support / top_support.clamp_min(1e-6)).clamp(min=0.0, max=1.0)
        family_conflict = (0.55 * family_entropy + 0.45 * overlap).clamp(min=0.0, max=1.0)

        top_family_per_path = family_support_paths_t.argmax(dim=-1)
        modal_family = top_family_per_path.mode(dim=0).values
        family_switch_rate = (top_family_per_path != modal_family.unsqueeze(0)).float().mean(dim=0)

        anchor_assignment_paths = torch.stack(family_assignment_paths, dim=2)
        anchor_support_paths = torch.stack(family_anchor_support_paths, dim=2)
        anchor_dist_paths = torch.stack(family_dist_paths, dim=2)
        anchor_assignment = anchor_assignment_paths.mean(dim=0)
        anchor_support = anchor_support_paths.mean(dim=0)
        top_anchor_ids = anchor_assignment.argmax(dim=-1)
        top_anchor_ids_per_path = anchor_assignment_paths.argmax(dim=-1)
        modal_anchor_ids = top_anchor_ids_per_path.mode(dim=0).values
        anchor_switch_rate = (top_anchor_ids_per_path != modal_anchor_ids.unsqueeze(0)).float().mean(dim=(0, 2))

        family_support_std = family_support_paths_t.std(dim=0, unbiased=False)
        family_approach = (
            1.0 - family_support_std / family_support.clamp_min(1e-3)
        ).clamp(min=0.0, max=1.0)
        family_leap_penalty = family_support_std.clamp(min=0.0, max=1.0)
        total_support = (family_support.sum(dim=-1) / max(self.num_families, 1)).clamp(min=0.0, max=1.0)
        nearest_anchor_distance = torch.sqrt(anchor_dist_paths.amin(dim=(2, 3)).mean(dim=0)).clamp_min(0.0)
        nearest_anchor_distance = torch.tanh(nearest_anchor_distance / math.sqrt(max(self.state_dim, 1)))

        return {
            "family_support_paths": family_support_paths_t,
            "family_distribution_paths": family_distribution_paths,
            "order_score_paths": order_score_paths,
            "order_score": order_score,
            "order_score_spread": order_score_spread,
            "family_support": family_support,
            "family_distribution": family_distribution,
            "family_approach": family_approach,
            "family_leap_penalty": family_leap_penalty,
            "top_support": top_support,
            "total_support": total_support,
            "family_margin": family_margin,
            "overlap": overlap,
            "family_conflict": family_conflict,
            "family_switch_rate": family_switch_rate,
            "anchor_switch_rate": anchor_switch_rate,
            "nearest_anchor_distance": nearest_anchor_distance,
            "top_family_id": family_support.argmax(dim=-1),
            "top_anchor_ids": top_anchor_ids,
            "anchor_assignment_probs": anchor_assignment,
            "anchor_assignment_paths": anchor_assignment_paths,
            "anchor_support_scores": anchor_support,
            "anchor_support_paths": anchor_support_paths,
            "anchor_centers": self.centers,
            "anchor_scales": scales,
        }


class DiffusionBottleneckHead(nn.Module):
    """Cooperative diffusion-evidence bottleneck for E10."""

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
        anchor_state_dim: int = 48,
        anchors_per_family: int = 4,
        num_diffusion_steps: int = 8,
        num_random_paths: int = 6,
        beta_start: float = 0.02,
        beta_end: float = 0.20,
        denoiser_hidden: int = 256,
        trust_hidden: int = 128,
        fragility_hidden: int = 96,
        confidence_threshold_init: float = 0.35,
        confidence_threshold_gap_init: float = 0.30,
        monotone_scale_init: float = 4.0,
        prototype_support_temperature: float = 0.25,
        prototype_scale_init: float = 0.55,
        prototype_scale_max: float = 0.70,
        perturbation_rank_margin: float = 0.03,
        basin_order_margin: float = 0.02,
        ordered_score_margin: float = 0.05,
        ordered_boundary_margin_scale: float = 0.50,
        ordered_energy_margin: float = 0.02,
        ordered_success_support_margin: float = 0.03,
        prototype_repulsion_margin: float = 0.75,
        family_activity_weight: float = 0.35,
        family_activity_ratio: float = 0.75,
        anchor_domination_weight: float = 0.30,
        anchor_domination_max_share: float = 0.70,
        pattern_prior_weight: float = 1.50,
        outcome_evidence_scale: float = 1.0,
        outcome_trajectory_scale: float = 1.0,
        outcome_pattern_scale: float = 0.80,
        dropout: float = 0.1,
        head_mode: str = "mlp",
        **_kwargs,
    ):
        super().__init__()

        del fragility_hidden

        self.d_model = d_model
        self.executor_dim = executor_dim
        self.num_trajectory = num_trajectory
        self.num_pattern = num_pattern
        self.num_outcome = num_outcome
        self.trace_hidden_dim = trace_hidden_dim
        self.anchor_state_dim = anchor_state_dim
        self.anchors_per_family = anchors_per_family
        self.T = num_diffusion_steps
        self.num_random_paths = num_random_paths
        self.head_mode = str(head_mode).strip().lower()
        self.perturbation_rank_margin = perturbation_rank_margin
        self.basin_order_margin = basin_order_margin
        self.ordered_score_margin = ordered_score_margin
        self.ordered_boundary_margin_scale = ordered_boundary_margin_scale
        self.ordered_energy_margin = ordered_energy_margin
        self.ordered_success_support_margin = ordered_success_support_margin
        self.prototype_repulsion_margin = prototype_repulsion_margin
        self.family_activity_weight = family_activity_weight
        self.family_activity_ratio = family_activity_ratio
        self.anchor_domination_weight = anchor_domination_weight
        self.anchor_domination_max_share = anchor_domination_max_share
        self.pattern_prior_weight = pattern_prior_weight
        self.outcome_evidence_scale = outcome_evidence_scale
        self.outcome_trajectory_scale = outcome_trajectory_scale
        self.outcome_pattern_scale = outcome_pattern_scale
        self.prototype_family_margin_target = 0.10
        self.prototype_scale_target = min(prototype_scale_max, prototype_scale_init + 0.05)

        betas = torch.linspace(beta_start, beta_end, num_diffusion_steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("alpha_bar", alpha_bar)

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
            nn.LayerNorm(trace_hidden_dim),
        )
        self.joint_latent_proj = _make_mlp(input_dim, hidden_dim, d_model, dropout)

        self.trajectory_shape_head_joint = nn.Linear(hidden_dim, num_trajectory)
        self.attention_pattern_head_joint = nn.Linear(hidden_dim, num_pattern)

        traj_summary_dim = TRAJECTORY_SUMMARY_SLICE.stop - TRAJECTORY_SUMMARY_SLICE.start
        self.trajectory_trace_tower = _make_mlp(
            trace_hidden_dim + traj_summary_dim, trace_hidden_dim, num_trajectory, dropout
        )
        self.attention_trace_tower = _make_mlp(
            trace_hidden_dim + 2, trace_hidden_dim, num_pattern, dropout
        )

        self.denoiser = LatentDenoiser(d_model, denoiser_hidden, num_diffusion_steps, dropout)
        self.path_evidence_head = PathEvidenceHead(d_model, max(32, hidden_dim // 2), dropout)
        self.stability_cert_head = MonotoneCertHead(input_dim=8, hidden_dim=max(32, trust_hidden), scale_init=monotone_scale_init)
        self.support_cert_head = MonotoneCertHead(input_dim=10, hidden_dim=max(24, trust_hidden // 2), scale_init=monotone_scale_init)
        self.success_guard_head = MonotoneCertHead(input_dim=12, hidden_dim=max(32, trust_hidden), scale_init=monotone_scale_init)
        if self.head_mode == "mlp":
            self.confidence_head = _make_mlp(5, max(32, hidden_dim // 3), num_confidence, dropout)
            self.outcome_head = _make_mlp(5 + num_trajectory + num_pattern, max(64, hidden_dim // 2), num_outcome, dropout)
        elif self.head_mode == "linear_evidence_only":
            self.confidence_linear_head = nn.Linear(5, num_confidence)
            self.outcome_linear_head = nn.Linear(5, num_outcome)
        elif self.head_mode == "ordered_threshold":
            self.confidence_threshold_head = OrderedThresholdHead(
                threshold_init=confidence_threshold_init,
                threshold_gap_init=confidence_threshold_gap_init,
                scale_init=monotone_scale_init,
            )
            self.outcome_threshold_head = OrderedThresholdHead(
                threshold_init=confidence_threshold_init,
                threshold_gap_init=confidence_threshold_gap_init,
                scale_init=monotone_scale_init,
                reorder_indices=(2, 1, 0),
            )
        else:
            raise ValueError(
                "Unsupported head_mode {!r}; expected 'mlp', 'linear_evidence_only', or 'ordered_threshold'".format(
                    self.head_mode
                )
            )

        self.prototype_state_proj = _make_mlp(d_model, hidden_dim, anchor_state_dim, dropout)
        self.prototype_bank = DiffusionPrototypeBank(
            state_dim=anchor_state_dim,
            prototypes_per_family=anchors_per_family,
            support_temperature=prototype_support_temperature,
            scale_init=prototype_scale_init,
            scale_max=prototype_scale_max,
        )

        self._trace_metadata: dict | None = None
        self.last_diffusion_state: dict[str, torch.Tensor] | None = None

    def set_trace_metadata(self, metadata: dict | None) -> None:
        self._trace_metadata = metadata

    def _pattern_prior_logits(self, executor_features: torch.Tensor) -> torch.Tensor:
        if not self._trace_metadata:
            return executor_features.new_zeros(executor_features.size(0), self.num_pattern)

        mean_entropy = executor_feature_column(executor_features, "mean_entropy")
        max_attention = executor_feature_column(executor_features, "max_attention_mass")
        mean_mu = float(self._trace_metadata.get("mean_entropy_mean", 0.0))
        mean_std = max(abs(float(self._trace_metadata.get("mean_entropy_std", 1.0))), 1e-6)
        attention_mu = float(self._trace_metadata.get("max_attention_mean", 0.0))
        attention_std = max(abs(float(self._trace_metadata.get("max_attention_std", 1.0))), 1e-6)
        threshold = float(self._trace_metadata.get("pattern_z_threshold", 0.75))

        mean_z = (mean_entropy - mean_mu) / mean_std
        attention_z = (max_attention - attention_mu) / attention_std
        focused_score = torch.minimum(threshold - mean_z, attention_z - threshold)
        diffuse_score = torch.minimum(mean_z - threshold, (-attention_z) - threshold)
        mixed_score = -torch.maximum(focused_score, diffuse_score)
        return torch.tanh(torch.stack([focused_score, mixed_score, diffuse_score], dim=-1))

    def _noise_z(self, z0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_t = self.alpha_bar[t - 1].unsqueeze(-1)
        eps = torch.randn_like(z0)
        z_t = torch.sqrt(alpha_t) * z0 + torch.sqrt(1.0 - alpha_t) * eps
        return z_t, eps

    def _noise_z_with_eps(self, z0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        alpha_t = self.alpha_bar[t - 1].unsqueeze(-1)
        return torch.sqrt(alpha_t) * z0 + torch.sqrt(1.0 - alpha_t) * eps

    def _denoise_z(self, z_t: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_t = self.alpha_bar[t - 1].unsqueeze(-1)
        eps_pred = self.denoiser(z_t, t)
        z0_hat = (z_t - torch.sqrt(1.0 - alpha_t) * eps_pred) / torch.sqrt(alpha_t).clamp_min(1e-4)
        return z0_hat, eps_pred

    def _build_joint_latent(self, z_q: torch.Tensor, executor_features: torch.Tensor) -> torch.Tensor:
        return self.joint_latent_proj(torch.cat([z_q, executor_features], dim=-1))

    def _run_random_paths(
        self,
        z0: torch.Tensor,
        diffusion_override: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        bsz, device = z0.size(0), z0.device
        z_rec_list, mse_list = [], []
        t_paths: list[torch.Tensor] = []
        eps_paths: list[torch.Tensor] = []

        override_t = None
        override_eps = None
        if diffusion_override is not None:
            override_t = diffusion_override.get("t_paths")
            override_eps = diffusion_override.get("eps_paths")
            if override_t is not None:
                override_t = torch.as_tensor(override_t, device=device, dtype=torch.long)
            if override_eps is not None:
                override_eps = torch.as_tensor(override_eps, device=device, dtype=z0.dtype)

        for path_idx in range(self.num_random_paths):
            if override_t is not None and override_eps is not None and path_idx < int(override_t.size(0)):
                t = override_t[path_idx].to(device)
                eps_true = override_eps[path_idx].to(device)
                z_t = self._noise_z_with_eps(z0, t, eps_true)
            else:
                t = torch.randint(1, self.T + 1, (bsz,), device=device)
                z_t, eps_true = self._noise_z(z0, t)
            z0_hat, eps_pred = self._denoise_z(z_t, t)
            mse_list.append(F.mse_loss(eps_pred, eps_true, reduction="none").mean(dim=-1))
            z_rec_list.append(z0_hat)
            t_paths.append(t)
            eps_paths.append(eps_true)

        z_rec_stack = torch.stack(z_rec_list, dim=0)
        denoise_mse = torch.stack(mse_list, dim=0).mean()
        diffusion_state = {
            "t_paths": torch.stack(t_paths, dim=0),
            "eps_paths": torch.stack(eps_paths, dim=0),
        }
        return z_rec_stack, denoise_mse, diffusion_state

    def _decode_path_evidence(self, z_rec_stack: torch.Tensor) -> dict[str, torch.Tensor]:
        num_paths, bsz, _ = z_rec_stack.shape
        flat = z_rec_stack.reshape(num_paths * bsz, -1)
        order_score, boundary, support, outcome_probs = self.path_evidence_head(flat)
        return {
            "order_score_paths": order_score.reshape(num_paths, bsz),
            "boundary_paths": boundary.reshape(num_paths, bsz),
            "support_paths": support.reshape(num_paths, bsz),
            "outcome_prob_paths": outcome_probs.reshape(num_paths, bsz, self.num_outcome),
        }

    def _evidence_selection_mask(
        self,
        path_outcome_prob_paths: torch.Tensor,
        evidence_override: dict[str, Any],
    ) -> torch.Tensor:
        apply_to = str(evidence_override.get("apply_to", "all_paths")).strip().lower()
        num_paths, bsz = path_outcome_prob_paths.shape[:2]
        if apply_to == "all_paths":
            return torch.ones((num_paths, bsz), dtype=torch.bool, device=path_outcome_prob_paths.device)
        if apply_to == "topk_paths":
            k = int(evidence_override.get("topk_paths_k", evidence_override.get("topk", 1)))
            k = max(1, min(k, num_paths))
            ranking = path_outcome_prob_paths[..., 0]
            topk_idx = ranking.topk(k, dim=0).indices
            mask = torch.zeros((num_paths, bsz), dtype=torch.bool, device=path_outcome_prob_paths.device)
            batch_idx = torch.arange(bsz, device=path_outcome_prob_paths.device)
            for rank in range(k):
                mask[topk_idx[rank], batch_idx] = True
            return mask
        if apply_to == "random_path":
            mask = torch.zeros((num_paths, bsz), dtype=torch.bool, device=path_outcome_prob_paths.device)
            if "path_indices" in evidence_override:
                path_indices = torch.as_tensor(
                    evidence_override["path_indices"],
                    device=path_outcome_prob_paths.device,
                    dtype=torch.long,
                )
                if path_indices.ndim == 0:
                    path_indices = path_indices.expand(bsz)
                elif path_indices.ndim != 1 or path_indices.numel() != bsz:
                    raise ValueError(
                        "evidence_override['path_indices'] must be a scalar or length-batch tensor"
                    )
            elif "path_index" in evidence_override:
                path_indices = torch.full(
                    (bsz,),
                    int(evidence_override["path_index"]),
                    dtype=torch.long,
                    device=path_outcome_prob_paths.device,
                )
            else:
                path_indices = torch.randint(num_paths, (bsz,), device=path_outcome_prob_paths.device)
            path_indices = path_indices.clamp(0, num_paths - 1)
            batch_idx = torch.arange(bsz, device=path_outcome_prob_paths.device)
            mask[path_indices, batch_idx] = True
            return mask
        raise ValueError(
            "Unsupported evidence_override['apply_to'] {!r}; expected 'all_paths', 'topk_paths', or 'random_path'".format(
                apply_to
            )
        )

    def _apply_evidence_override(
        self,
        path_order_score_paths: torch.Tensor,
        path_boundary_paths: torch.Tensor,
        path_support_paths: torch.Tensor,
        path_outcome_prob_paths: torch.Tensor,
        evidence_override: dict[str, Any] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not evidence_override:
            return (
                path_order_score_paths,
                path_boundary_paths,
                path_support_paths,
                path_outcome_prob_paths,
            )

        delta_order = float(evidence_override.get("delta_order", 0.0))
        delta_boundary = float(evidence_override.get("delta_boundary", 0.0))
        delta_support = float(evidence_override.get("delta_support", 0.0))
        if delta_order == 0.0 and delta_boundary == 0.0 and delta_support == 0.0:
            return (
                path_order_score_paths,
                path_boundary_paths,
                path_support_paths,
                path_outcome_prob_paths,
            )

        mask = self._evidence_selection_mask(path_outcome_prob_paths, evidence_override)
        if not bool(mask.any()):
            return (
                path_order_score_paths,
                path_boundary_paths,
                path_support_paths,
                path_outcome_prob_paths,
            )

        order = path_order_score_paths.clone()
        boundary = path_boundary_paths.clone()
        support = path_support_paths.clone()

        if delta_order != 0.0:
            order[mask] = (order[mask] + delta_order).clamp(-0.999, 0.999)
        if delta_boundary != 0.0:
            boundary[mask] = (boundary[mask] + delta_boundary).clamp(0.0, 1.0)
        if delta_support != 0.0:
            support[mask] = (support[mask] + delta_support).clamp(0.0, 1.0)

        order_raw = torch.atanh(order.clamp(-0.999, 0.999))
        outcome_probs = _path_outcome_probs_from_order_raw(order_raw, boundary, support)
        return order, boundary, support, outcome_probs

    def _aggregate_path_evidence(
        self,
        z0: torch.Tensor,
        z_rec_stack: torch.Tensor,
        path_order_score_paths: torch.Tensor,
        path_boundary_paths: torch.Tensor,
        path_support_paths: torch.Tensor,
        path_outcome_prob_paths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        latent_rec_var = z_rec_stack.var(dim=0, unbiased=False).mean(dim=-1)
        path_reconstruction_energy = (z_rec_stack - z0.unsqueeze(0)).norm(dim=-1)
        energy_mean = path_reconstruction_energy.mean(dim=0)
        energy_std = path_reconstruction_energy.std(dim=0, unbiased=False)

        base_outcome = path_outcome_prob_paths.mean(dim=0).clamp_min(1e-6)
        base_outcome = base_outcome / base_outcome.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        top_path_cls = path_outcome_prob_paths.argmax(dim=-1)
        modal_cls = top_path_cls.mode(dim=0).values
        path_agreement = (top_path_cls == modal_cls.unsqueeze(0)).float().mean(dim=0)

        outcome_entropy = -(base_outcome * base_outcome.log()).sum(dim=-1)
        outcome_entropy_norm = (outcome_entropy / math.log(self.num_outcome)).clamp(0.0, 1.0)
        order_mean = path_order_score_paths.mean(dim=0)
        order_std = path_order_score_paths.std(dim=0, unbiased=False).clamp(0.0, 1.0)
        boundary_mean = path_boundary_paths.mean(dim=0)
        boundary_std = path_boundary_paths.std(dim=0, unbiased=False).clamp(0.0, 1.0)
        support_mean = path_support_paths.mean(dim=0)
        support_std = path_support_paths.std(dim=0, unbiased=False).clamp(0.0, 1.0)

        var_ref = latent_rec_var.detach().mean().clamp_min(1e-4)
        energy_ref = energy_mean.detach().mean().clamp_min(1e-4)
        energy_std_ref = energy_std.detach().mean().clamp_min(1e-4)
        latent_var_norm = (latent_rec_var / var_ref).clamp(0.0, 2.0) / 2.0
        energy_mean_norm = (energy_mean / energy_ref).clamp(0.0, 2.0) / 2.0
        energy_std_norm = (energy_std / energy_std_ref).clamp(0.0, 2.0) / 2.0
        path_disagreement = (1.0 - path_agreement).clamp(0.0, 1.0)

        certainty_badness = torch.stack(
            [
                latent_var_norm,
                path_disagreement,
                outcome_entropy_norm,
                energy_mean_norm,
                energy_std_norm,
                order_std,
                boundary_std,
                support_std,
            ],
            dim=-1,
        )

        return {
            "base_outcome": base_outcome,
            "path_reconstruction_energy": path_reconstruction_energy,
            "latent_var_norm": latent_var_norm,
            "energy_mean_norm": energy_mean_norm,
            "energy_std_norm": energy_std_norm,
            "path_agreement": path_agreement,
            "path_disagreement": path_disagreement,
            "outcome_entropy_norm": outcome_entropy_norm,
            "order_mean": order_mean,
            "order_std": order_std,
            "boundary_mean": boundary_mean,
            "boundary_std": boundary_std,
            "support_mean": support_mean,
            "support_std": support_std,
            "certainty_badness": certainty_badness,
        }

    def _prototype_features(
        self,
        z0: torch.Tensor,
        z_rec_stack: torch.Tensor,
        path_outcome_prob_paths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        num_paths, bsz, _ = z_rec_stack.shape
        path_states = self.prototype_state_proj(z_rec_stack.reshape(num_paths * bsz, -1)).reshape(
            num_paths,
            bsz,
            self.anchor_state_dim,
        )
        proto = self.prototype_bank(path_states)
        del z0
        path_top_support = proto["family_support_paths"].amax(dim=-1)
        base_outcome = path_outcome_prob_paths.mean(dim=0).clamp_min(1e-6)
        outcome_entropy = -(base_outcome * base_outcome.log()).sum(dim=-1) / math.log(self.num_outcome)

        family_target = torch.stack(
            [
                base_outcome[:, 0],
                base_outcome[:, 2],
                base_outcome[:, 1],
            ],
            dim=-1,
        ).detach().clamp_min(1e-6)
        family_target = family_target / family_target.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        decisive_mask = (1.0 - family_target[:, 2]).detach().clamp(0.0, 1.0)

        family_alignment_loss = F.kl_div(
            proto["family_distribution"].clamp_min(1e-6).log(),
            family_target,
            reduction="batchmean",
        )

        family_usage_paths = F.softmax(proto["family_support_paths"] / 0.05, dim=-1)
        family_usage = family_usage_paths.mean(dim=(0, 1)).clamp_min(1e-6)
        family_usage = family_usage / family_usage.sum().clamp_min(1e-6)
        target_family_usage = family_target.mean(dim=0).clamp_min(1e-6)
        target_family_usage = target_family_usage / target_family_usage.sum().clamp_min(1e-6)
        family_usage_loss = F.kl_div(family_usage.log(), target_family_usage, reduction="sum") / family_usage.numel()
        decisive_activity_target = self.family_activity_ratio * target_family_usage[:2]
        decisive_activity_loss = torch.relu(decisive_activity_target - family_usage[:2]).sum()

        weighted_anchor_usage = (
            proto["anchor_assignment_paths"] * family_target.unsqueeze(0).unsqueeze(-1)
        ).mean(dim=(0, 1)).clamp_min(1e-6)
        weighted_anchor_usage = weighted_anchor_usage / weighted_anchor_usage.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)
        uniform_anchor = torch.full_like(weighted_anchor_usage, 1.0 / weighted_anchor_usage.size(-1))
        anchor_usage_loss = F.kl_div(weighted_anchor_usage.log(), uniform_anchor, reduction="batchmean")
        anchor_domination_penalty = torch.relu(
            weighted_anchor_usage.max(dim=-1).values - self.anchor_domination_max_share
        ).mean()

        switch_target = (0.12 * outcome_entropy.detach()).clamp(0.0, 0.12)
        switch_loss = (
            torch.relu(switch_target - proto["anchor_switch_rate"]).mean()
            + 0.5 * torch.relu(0.5 * switch_target - proto["family_switch_rate"]).mean()
        )
        overlap_penalty = (decisive_mask * torch.relu(proto["overlap"] - 0.35)).mean()
        margin_floor_loss = (
            decisive_mask * torch.relu(self.prototype_family_margin_target - proto["family_margin"])
        ).mean()

        pull_loss = (
            0.50 * (1.0 - path_top_support.clamp(0.0, 1.0)).mean()
            + 0.35 * (1.0 - proto["total_support"].clamp(0.0, 1.0)).mean()
            + 0.20 * proto["nearest_anchor_distance"].mean()
            + 0.25 * proto["overlap"].mean()
        )
        usage_loss = (
            family_alignment_loss
            + family_usage_loss
            + self.family_activity_weight * decisive_activity_loss
            + 0.75 * anchor_usage_loss
            + self.anchor_domination_weight * anchor_domination_penalty
            + 0.50 * switch_loss
            + 0.75 * overlap_penalty
            + 0.75 * margin_floor_loss
        )

        centers = self.prototype_bank.centers.reshape(-1, self.anchor_state_dim)
        pairwise = torch.cdist(centers, centers)
        pairwise_mask = ~torch.eye(pairwise.size(0), dtype=torch.bool, device=pairwise.device)
        family_centers = self.prototype_bank.centers.mean(dim=1)
        family_pairwise = torch.cdist(family_centers, family_centers)
        family_mask = ~torch.eye(family_pairwise.size(0), dtype=torch.bool, device=family_pairwise.device)
        repulsion_loss = torch.relu(self.prototype_repulsion_margin - pairwise[pairwise_mask]).mean()
        repulsion_loss = repulsion_loss + 0.75 * torch.relu(
            1.15 * self.prototype_repulsion_margin - family_pairwise[family_mask]
        ).mean()
        repulsion_loss = repulsion_loss + 0.10 * torch.relu(
            proto["anchor_scales"].mean() - self.prototype_scale_target
        )

        proto["path_states"] = path_states
        proto["prototype_pull_loss"] = pull_loss
        proto["prototype_usage_loss"] = usage_loss
        proto["prototype_repulsion_loss"] = repulsion_loss
        return proto

    def compute_perturbation_aux_losses(
        self,
        z_q: torch.Tensor,
        executor_features: torch.Tensor,
        perturbed_executor_features: torch.Tensor,
        primitives: PrimitiveOutput,
        perturbed_primitives: PrimitiveOutput,
        perturbation_kind: str | None = None,
        outcome_targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        z0_base = self._build_joint_latent(z_q, executor_features)
        z0_perturbed = self._build_joint_latent(z_q, perturbed_executor_features)

        bsz = z0_base.size(0)
        device = z0_base.device
        t = torch.randint(1, self.T + 1, (bsz,), device=device)
        eps = torch.randn_like(z0_base)

        base_z_t = self._noise_z_with_eps(z0_base, t, eps)
        perturbed_z_t = self._noise_z_with_eps(z0_perturbed, t, eps)
        base_rec, eps_pred_base = self._denoise_z(base_z_t, t)
        perturbed_rec, eps_pred_perturbed = self._denoise_z(perturbed_z_t, t)

        base_mse = F.mse_loss(eps_pred_base, eps, reduction="none").mean(dim=-1)
        perturbed_mse = F.mse_loss(eps_pred_perturbed, eps, reduction="none").mean(dim=-1)

        base_pair_order, base_pair_boundary, base_pair_support, base_pair_outcome = self.path_evidence_head(base_rec)
        pert_pair_order, pert_pair_boundary, pert_pair_support, pert_pair_outcome = self.path_evidence_head(perturbed_rec)

        base_pair_state = self.prototype_state_proj(base_rec).unsqueeze(0)
        perturbed_pair_state = self.prototype_state_proj(perturbed_rec).unsqueeze(0)
        base_pair_proto = self.prototype_bank(base_pair_state)
        perturbed_pair_proto = self.prototype_bank(perturbed_pair_state)

        if outcome_targets is None:
            outcome_targets = primitives.outcome_logits.argmax(dim=-1)
        outcome_targets = outcome_targets.long()
        success_mask = (outcome_targets == 0).float()
        boundary_mask = (outcome_targets == 1).float()
        failure_mask = (outcome_targets == 2).float()
        nonfailure_context = 1.0 - failure_mask

        order_margin = self.ordered_score_margin * (
            success_mask + self.ordered_boundary_margin_scale * boundary_mask
        )
        support_margin = self.ordered_success_support_margin * (
            success_mask + self.ordered_boundary_margin_scale * boundary_mask
        )
        geometry_margin = 0.5 * self.basin_order_margin

        base_path_success_tail = primitives.path_outcome_prob_paths[..., 0].amax(dim=0)
        pert_path_success_tail = perturbed_primitives.path_outcome_prob_paths[..., 0].amax(dim=0)
        base_path_order_tail = primitives.path_order_score_paths.amax(dim=0)
        pert_path_order_tail = perturbed_primitives.path_order_score_paths.amax(dim=0)
        base_path_support_tail = primitives.path_support_paths.amax(dim=0)
        pert_path_support_tail = perturbed_primitives.path_support_paths.amax(dim=0)
        base_path_energy_floor = primitives.path_reconstruction_energy.amin(dim=0)
        pert_path_energy_floor = perturbed_primitives.path_reconstruction_energy.amin(dim=0)

        pair_order_up = torch.relu(pert_pair_order - base_pair_order + order_margin)
        pair_success_up = torch.relu(pert_pair_outcome[:, 0] - base_pair_outcome[:, 0] + support_margin)
        pair_support_up = torch.relu(pert_pair_support - base_pair_support + 0.5 * support_margin)
        pair_energy_violation = torch.relu(base_mse - perturbed_mse + self.ordered_energy_margin)
        pair_failure_optimism = torch.relu(pert_pair_order - base_pair_order)
        pair_boundary_hardening = torch.relu(base_pair_boundary - pert_pair_boundary + geometry_margin)

        path_order_up = torch.relu(pert_path_order_tail - base_path_order_tail + order_margin)
        path_success_up = torch.relu(pert_path_success_tail - base_path_success_tail + support_margin)
        path_support_up = torch.relu(pert_path_support_tail - base_path_support_tail + 0.5 * support_margin)
        path_energy_violation = torch.relu(base_path_energy_floor - pert_path_energy_floor + self.ordered_energy_margin)
        path_failure_optimism = torch.relu(pert_path_order_tail - base_path_order_tail)

        certainty_up = torch.relu(perturbed_primitives.certainty_score - primitives.certainty_score)
        support_cert_up = torch.relu(perturbed_primitives.support_cert - primitives.support_cert)
        commitment_up = torch.relu(perturbed_primitives.commitment_depth - primitives.commitment_depth)
        success_up = torch.relu(perturbed_primitives.success_cert - primitives.success_cert)
        success_guard_up = torch.relu(perturbed_primitives.success_guard - primitives.success_guard)
        guard_gap_up = torch.relu(
            torch.relu(perturbed_primitives.success_base - perturbed_primitives.success_guard)
            - torch.relu(primitives.success_base - primitives.success_guard)
        )
        overlap_up = torch.relu(perturbed_primitives.prototype_overlap - primitives.prototype_overlap)
        margin_improve = torch.relu(
            perturbed_primitives.prototype_family_margin - primitives.prototype_family_margin + geometry_margin
        )
        support_drop = torch.relu(primitives.prototype_total_support - perturbed_primitives.prototype_total_support)
        boundary_bias = torch.relu(primitives.boundary_family_support - primitives.success_family_support)
        perturbation_multiplier = 1.75 if perturbation_kind == "lower_attention_concentration" else 1.0

        violation_gate = (
            0.85 * commitment_up
            + 0.70 * success_up
            + 0.60 * success_guard_up
            + 0.55 * guard_gap_up
            + 0.50 * certainty_up
            + 0.45 * support_cert_up
            + 0.45 * path_order_up
            + 0.40 * path_success_up
            + 0.35 * path_support_up
            + 0.30 * overlap_up
            + 0.25 * margin_improve
            + 0.25 * support_drop
            + perturbation_multiplier * 0.35 * boundary_bias * success_up
        ).clamp(min=0.0, max=1.0)

        basin_order = (
            (nonfailure_context * pair_order_up).mean()
            + 0.80 * (nonfailure_context * pair_success_up).mean()
            + 0.65 * (nonfailure_context * pair_support_up).mean()
            + 0.90 * (nonfailure_context * pair_energy_violation).mean()
            + 0.50 * (boundary_mask * pair_boundary_hardening).mean()
            + 0.35 * (failure_mask * pair_failure_optimism).mean()
        )
        ordered_geometry = (
            (nonfailure_context * path_order_up).mean()
            + 0.85 * (nonfailure_context * path_success_up).mean()
            + 0.70 * (nonfailure_context * path_support_up).mean()
            + 0.80 * (nonfailure_context * path_energy_violation).mean()
            + 0.45 * overlap_up.mean()
            + 0.35 * (boundary_mask * margin_improve).mean()
            + 0.30 * (failure_mask * path_failure_optimism).mean()
        )
        rank_gate = torch.clamp(0.15 + violation_gate, min=0.0, max=1.0)
        perturbation_rank = (rank_gate * torch.relu(base_mse - perturbed_mse + self.perturbation_rank_margin)).mean()
        perturbation_proximity = F.smooth_l1_loss(z0_perturbed, z0_base.detach(), reduction="none").mean(dim=-1).mean()

        fragility_target = (
            violation_gate
            + 0.45 * path_order_up
            + 0.40 * path_success_up
            + 0.35 * path_support_up
            + 0.35 * overlap_up
            + 0.35 * support_drop
            + 0.35 * success_up
            + 0.30 * success_guard_up
        ).clamp(min=0.0, max=1.0).detach()
        fragility_target_loss = F.smooth_l1_loss(perturbed_primitives.fragility_risk, fragility_target)
        fragility_rank = (
            violation_gate
            * torch.relu(primitives.fragility_risk - perturbed_primitives.fragility_risk + 0.05)
        ).mean()

        return {
            "basin_order": basin_order,
            "ordered_geometry": ordered_geometry,
            "perturbation_rank": perturbation_rank,
            "perturbation_proximity": perturbation_proximity,
            "fragility_target": fragility_target.mean(),
            "fragility_target_loss": fragility_target_loss,
            "fragility_rank": fragility_rank,
            "perturbation_geometry_gate": violation_gate.mean(),
            "base_denoise_pair_mse": base_mse.mean(),
            "perturbed_denoise_pair_mse": perturbed_mse.mean(),
        }

    def forward(
        self,
        z_q: torch.Tensor,
        executor_features: torch.Tensor,
        mode: str = "gumbel",
        tau: float = 1.0,
        use_query_features: bool = True,
        path_scales: Dict[str, float] | None = None,
        evidence_override: dict[str, Any] | None = None,
        diffusion_override: dict[str, torch.Tensor] | None = None,
    ) -> PrimitiveOutput:
        del path_scales

        bsz = z_q.size(0)

        if use_query_features:
            h_joint = self.shared_joint(torch.cat([z_q, executor_features], dim=-1))
            traj_logits = self.trajectory_shape_head_joint(h_joint)
            pattern_logits = self.attention_pattern_head_joint(h_joint)
        else:
            h_trace = self.shared_trace(executor_features)
            traj_feats = executor_features[:, TRAJECTORY_SUMMARY_SLICE]
            traj_logits = self.trajectory_trace_tower(torch.cat([h_trace, traj_feats], dim=-1))
            mean_ent = executor_feature_column(executor_features, "mean_entropy").unsqueeze(-1)
            max_att = executor_feature_column(executor_features, "max_attention_mass").unsqueeze(-1)
            pattern_logits = self.attention_trace_tower(torch.cat([h_trace, mean_ent, max_att], dim=-1))

        if self.pattern_prior_weight != 0.0:
            pattern_logits = pattern_logits + self.pattern_prior_weight * self._pattern_prior_logits(executor_features)

        traj_probs = F.softmax(traj_logits, dim=-1)
        pattern_probs = F.softmax(pattern_logits, dim=-1)
        traj_disc = _discretise(traj_logits, mode, tau)
        pattern_disc = _discretise(pattern_logits, mode, tau)

        z0 = self._build_joint_latent(z_q, executor_features)
        z_rec_stack, denoise_mse, diffusion_state = self._run_random_paths(
            z0,
            diffusion_override=diffusion_override,
        )
        self.last_diffusion_state = diffusion_state
        path_evidence = self._decode_path_evidence(z_rec_stack)
        path_order_score_paths = path_evidence["order_score_paths"]
        path_boundary_paths = path_evidence["boundary_paths"]
        path_support_paths = path_evidence["support_paths"]
        path_outcome_prob_paths = path_evidence["outcome_prob_paths"]
        (
            path_order_score_paths,
            path_boundary_paths,
            path_support_paths,
            path_outcome_prob_paths,
        ) = self._apply_evidence_override(
            path_order_score_paths,
            path_boundary_paths,
            path_support_paths,
            path_outcome_prob_paths,
            evidence_override,
        )

        aggregated = self._aggregate_path_evidence(
            z0,
            z_rec_stack,
            path_order_score_paths,
            path_boundary_paths,
            path_support_paths,
            path_outcome_prob_paths,
        )
        prototype = self._prototype_features(z0, z_rec_stack, path_outcome_prob_paths)

        agreement = executor_feature_column(executor_features, "agreement")
        prob_margin = executor_feature_column(executor_features, "prob_margin")
        max_att_mass = executor_feature_column(executor_features, "max_attention_mass")
        attention_effective_support = executor_feature_column(executor_features, "attention_effective_support")
        variation_ratio = executor_feature_column(executor_features, "variation_ratio")
        low_margin_flag = executor_feature_column(executor_features, "low_margin_flag")
        unstable_flag = executor_feature_column(executor_features, "unstable_flag")

        trust_agreement_deficit = (1.0 - agreement).clamp(0.0, 1.0)
        trust_margin_deficit = torch.clamp(1.0 - prob_margin, min=low_margin_flag)
        trust_concentration_deficit = (1.0 - max_att_mass).clamp(0.0, 1.0)
        trust_variation_pressure = torch.clamp(variation_ratio, min=unstable_flag)
        trust_effective_support_deficit = (1.0 - attention_effective_support).clamp(0.0, 1.0)

        stability_cert_head = self.stability_cert_head(aggregated["certainty_badness"])
        stability_native = torch.clamp(
            0.24 * (1.0 - aggregated["latent_var_norm"])
            + 0.24 * (1.0 - aggregated["path_disagreement"])
            + 0.16 * (1.0 - aggregated["outcome_entropy_norm"])
            + 0.12 * (1.0 - aggregated["energy_mean_norm"])
            + 0.08 * (1.0 - aggregated["energy_std_norm"])
            + 0.06 * (1.0 - aggregated["order_std"])
            + 0.05 * (1.0 - aggregated["boundary_std"])
            + 0.05 * (1.0 - aggregated["support_std"]),
            min=0.0,
            max=1.0,
        )
        stability_cert = torch.clamp(
            0.55 * stability_cert_head + 0.45 * stability_native,
            min=0.0,
            max=1.0,
        )
        cert_risk = 1.0 - stability_cert
        certainty_score = stability_cert

        support_badness = torch.stack(
            [
                (1.0 - aggregated["support_mean"]).clamp(0.0, 1.0),
                aggregated["support_std"],
                prototype["overlap"].clamp(0.0, 1.0),
                prototype["nearest_anchor_distance"].clamp(0.0, 1.0),
                (1.0 - prototype["top_support"]).clamp(0.0, 1.0),
                (1.0 - prototype["total_support"]).clamp(0.0, 1.0),
                trust_agreement_deficit,
                trust_margin_deficit,
                trust_concentration_deficit,
                torch.maximum(trust_variation_pressure, trust_effective_support_deficit),
            ],
            dim=-1,
        )
        support_cert_head = self.support_cert_head(support_badness)
        support_native = torch.clamp(
            0.24 * aggregated["support_mean"]
            + 0.14 * agreement
            + 0.10 * prob_margin
            + 0.08 * max_att_mass
            + 0.12 * attention_effective_support
            + 0.10 * prototype["total_support"]
            + 0.08 * prototype["family_margin"]
            + 0.07 * (1.0 - prototype["overlap"])
            + 0.07 * (1.0 - prototype["nearest_anchor_distance"]),
            min=0.0,
            max=1.0,
        )
        support_cert = torch.clamp(
            0.55 * support_cert_head + 0.45 * support_native,
            min=0.0,
            max=1.0,
        )

        base_outcome = aggregated["base_outcome"]
        success_base = base_outcome[:, 0]
        ambiguity_base = base_outcome[:, 1]
        failure_base = base_outcome[:, 2]

        family_support = prototype["family_support"]
        success_family_support = family_support[:, 0]
        failure_family_support = family_support[:, 1]
        boundary_family_support = family_support[:, 2]
        success_family_approach = prototype["family_approach"][:, 0]
        failure_family_approach = prototype["family_approach"][:, 1]
        boundary_family_approach = prototype["family_approach"][:, 2]
        success_family_leap_penalty = prototype["family_leap_penalty"][:, 0]
        failure_family_leap_penalty = prototype["family_leap_penalty"][:, 1]
        boundary_family_leap_penalty = prototype["family_leap_penalty"][:, 2]
        prototype_top_support = prototype["top_support"]
        prototype_total_support = prototype["total_support"]
        prototype_family_margin = prototype["family_margin"]
        prototype_overlap = prototype["overlap"]
        prototype_family_switch_rate = prototype["family_switch_rate"]
        prototype_anchor_switch_rate = prototype["anchor_switch_rate"]
        prototype_nearest_anchor_distance = prototype["nearest_anchor_distance"]
        prototype_order_score = prototype["order_score"]
        prototype_order_score_paths = prototype["order_score_paths"]
        prototype_order_score_spread = prototype["order_score_spread"]
        anchor_conflict = prototype["family_conflict"]
        prototype_pull_loss = prototype["prototype_pull_loss"]
        prototype_usage_loss = prototype["prototype_usage_loss"]
        prototype_repulsion_loss = prototype["prototype_repulsion_loss"]

        trust_support_deficit = (1.0 - prototype_total_support).clamp(0.0, 1.0)
        trust_approach_deficit = aggregated["path_disagreement"]
        trust_leap_cost = aggregated["energy_mean_norm"]
        trust_conflict = (
            0.30 * aggregated["outcome_entropy_norm"]
            + 0.20 * aggregated["path_disagreement"]
            + 0.20 * anchor_conflict
            + 0.15 * prototype_overlap
            + 0.15 * aggregated["boundary_mean"]
        ).clamp(0.0, 1.0)

        guard_badness = torch.stack(
            [
                (1.0 - stability_cert).clamp(0.0, 1.0),
                (1.0 - support_cert).clamp(0.0, 1.0),
                ambiguity_base.clamp(0.0, 1.0),
                failure_base.clamp(0.0, 1.0),
                trust_conflict,
                aggregated["boundary_mean"].clamp(0.0, 1.0),
                prototype_overlap.clamp(0.0, 1.0),
                (1.0 - prototype_family_margin).clamp(0.0, 1.0),
                trust_agreement_deficit,
                trust_margin_deficit,
                trust_concentration_deficit,
                torch.maximum(trust_variation_pressure, trust_effective_support_deficit),
            ],
            dim=-1,
        )
        success_guard = self.success_guard_head(guard_badness)

        success_core = success_base.clamp(0.0, 1.0)
        success_proposal = success_core
        success_cert = (success_base * success_guard).clamp(0.0, 1.0)
        failure_cert = failure_base.clamp(0.0, 1.0)
        ambiguity_cert = (
            0.70 * ambiguity_base
            + 0.20 * aggregated["boundary_mean"]
            + 0.10 * (1.0 - support_cert)
        ).clamp(0.0, 1.0)
        decisiveness_score = (1.0 - ambiguity_cert).clamp(0.0, 1.0)

        evidence_vector = torch.stack(
            [
                stability_cert,
                support_cert,
                ambiguity_cert,
                success_base,
                failure_base,
            ],
            dim=-1,
        )
        if self.head_mode == "mlp":
            confidence_logits = self.confidence_head(evidence_vector)
            outcome_context = torch.cat(
                [
                    self.outcome_evidence_scale * evidence_vector,
                    self.outcome_trajectory_scale * traj_probs.detach(),
                    self.outcome_pattern_scale * pattern_probs.detach(),
                ],
                dim=-1,
            )
            outcome_logits = self.outcome_head(outcome_context)
        elif self.head_mode == "linear_evidence_only":
            confidence_logits = self.confidence_linear_head(evidence_vector)
            outcome_logits = self.outcome_linear_head(evidence_vector)
        elif self.head_mode == "ordered_threshold":
            confidence_logits = self.confidence_threshold_head(decisiveness_score)
            outcome_signal = ((success_base - failure_base) + 1.0) * 0.5
            outcome_logits = self.outcome_threshold_head(outcome_signal)
        else:
            raise ValueError(f"Unsupported head_mode {self.head_mode!r}")
        conf_disc = _discretise(confidence_logits, mode, tau)
        outcome_disc = _discretise(outcome_logits, mode, tau)

        fragility_risk = (
            0.60 * (1.0 - support_cert)
            + 0.20 * prototype_overlap
            + 0.10 * aggregated["boundary_mean"]
            + 0.10 * (1.0 - prototype_family_margin)
        ).clamp(0.0, 1.0)
        fragility_score = 1.0 - fragility_risk
        effective_certainty = (stability_cert * support_cert).clamp(0.0, 1.0)
        success_affinity = torch.relu(
            success_family_support - torch.maximum(boundary_family_support, failure_family_support)
        ).clamp(0.0, 1.0)
        commitment_depth = (success_guard * support_cert * decisiveness_score).clamp(0.0, 1.0)

        zeros_b = z_q.new_zeros(bsz)
        anchors = self.anchors_per_family
        base_state = self.prototype_state_proj(z0)
        anchor_state = prototype["path_states"].mean(dim=0)
        anchor_prev_state = base_state
        anchor_transition = anchor_state - anchor_prev_state
        top_family_id = prototype["top_family_id"]
        top_anchor_ids = prototype["top_anchor_ids"]
        anchor_assignment_probs = prototype["anchor_assignment_probs"]
        anchor_support_scores = prototype["anchor_support_scores"]
        anchor_approach_scores = anchor_assignment_probs
        anchor_leap_scores = z_q.new_zeros(bsz, len(FAMILY_NAMES), anchors)
        anchor_centers = prototype["anchor_centers"]
        anchor_scales = prototype["anchor_scales"]
        anchor_directions = z_q.new_zeros(len(FAMILY_NAMES), anchors, self.anchor_state_dim)
        anchor_leap_scales = z_q.new_ones(len(FAMILY_NAMES), anchors)

        return PrimitiveOutput(
            trajectory_shape_logits=traj_logits,
            attention_pattern_logits=pattern_logits,
            confidence_logits=confidence_logits,
            outcome_logits=outcome_logits,
            trajectory_shape_discrete=traj_disc,
            attention_pattern_discrete=pattern_disc,
            confidence_discrete=conf_disc,
            outcome_discrete=outcome_disc,
            stability_cert=stability_cert,
            support_cert=support_cert,
            success_guard=success_guard,
            success_core=success_core,
            success_residual_weight=zeros_b,
            success_residual=zeros_b,
            success_base=success_base,
            success_affinity=success_affinity,
            success_bonus_weight=zeros_b,
            success_bonus_weight_zero_raw=zeros_b,
            success_bonus=zeros_b,
            success_bonus_zero_raw=zeros_b,
            success_proposal=success_proposal,
            success_cert=success_cert,
            failure_cert=failure_cert,
            ambiguity_cert=ambiguity_cert,
            cert_risk=cert_risk,
            certainty_score=certainty_score,
            fragility_risk=fragility_risk,
            fragility_score=fragility_score,
            effective_certainty=effective_certainty,
            decisiveness_score=decisiveness_score,
            prototype_top_support=prototype_top_support,
            prototype_total_support=prototype_total_support,
            prototype_family_margin=prototype_family_margin,
            prototype_overlap=prototype_overlap,
            prototype_family_switch_rate=prototype_family_switch_rate,
            prototype_anchor_switch_rate=prototype_anchor_switch_rate,
            prototype_nearest_anchor_distance=prototype_nearest_anchor_distance,
            prototype_pull_loss=prototype_pull_loss,
            prototype_usage_loss=prototype_usage_loss,
            prototype_repulsion_loss=prototype_repulsion_loss,
            prototype_family_support_paths=prototype["family_support_paths"],
            prototype_family_distribution_paths=prototype["family_distribution_paths"],
            prototype_order_score=prototype_order_score,
            prototype_order_score_paths=prototype_order_score_paths,
            prototype_order_score_spread=prototype_order_score_spread,
            path_order_score_paths=path_order_score_paths,
            path_boundary_paths=path_boundary_paths,
            path_support_paths=path_support_paths,
            path_outcome_prob_paths=path_outcome_prob_paths,
            path_reconstruction_energy=aggregated["path_reconstruction_energy"],
            trust_support_deficit=trust_support_deficit,
            trust_approach_deficit=trust_approach_deficit,
            trust_leap_cost=trust_leap_cost,
            trust_conflict=trust_conflict,
            trust_agreement_deficit=trust_agreement_deficit,
            trust_margin_deficit=trust_margin_deficit,
            trust_concentration_deficit=trust_concentration_deficit,
            trust_variation_pressure=trust_variation_pressure,
            success_trace_strength=anchor_prev_state.norm(dim=-1),
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
            diffusion_denoise_loss=denoise_mse,
        )


EntropyBottleneckHead = DiffusionBottleneckHead
