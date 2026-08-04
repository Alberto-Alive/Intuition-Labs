"""Diffusion-owned certainty/outcome bottleneck for DIGIT Extrapolation E7.

Architecture:
  Outcome    = mean vote of diffusion reconstructions.
  Certainty  = stability of the latent basin under random noise.
  Geometry   = prototype-family support computed from reconstructed diffusion
               paths, never from a separate clean-latent classifier.
  Commitment = certainty x fragility complement x decisiveness.
"""

from __future__ import annotations

import math
from typing import Dict, NamedTuple

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
    decisiveness_score: torch.Tensor
    prototype_top_support: torch.Tensor
    prototype_family_margin: torch.Tensor
    prototype_anchor_switch_rate: torch.Tensor
    prototype_alignment_loss: torch.Tensor

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


class OrderedThresholdHead(nn.Module):
    """Three-class ordinal head driven by a single scalar risk in [0,1]."""

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
    """Maps diffusion certainty statistics to confidence logits."""

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
        self.cert_risk_proj = nn.Linear(hidden_dim, 1)
        self.confidence_head = OrderedThresholdHead(
            threshold_init=confidence_threshold_init,
            threshold_gap_init=confidence_threshold_gap_init,
            scale_init=scale_init,
        )

    def forward(self, stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(stats)
        cert_risk = torch.sigmoid(self.cert_risk_proj(h)).squeeze(-1)
        confidence_logits = self.confidence_head(cert_risk)
        return confidence_logits, cert_risk


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
    ):
        super().__init__()
        self.state_dim = state_dim
        self.prototypes_per_family = prototypes_per_family
        self.num_families = num_families
        self.support_temperature = support_temperature

        self.centers = nn.Parameter(torch.randn(num_families, prototypes_per_family, state_dim) * 0.02)
        self.raw_scales = nn.Parameter(
            torch.full(
                (num_families, prototypes_per_family, state_dim),
                _inverse_softplus(1.0),
            )
        )

    def forward(self, path_states: torch.Tensor) -> dict[str, torch.Tensor]:
        num_paths, _bsz, _ = path_states.shape
        scales = F.softplus(self.raw_scales).clamp_min(1e-4)

        family_support_paths = []
        family_assignment_paths = []
        family_anchor_support_paths = []

        for family_idx in range(self.num_families):
            centers = self.centers[family_idx]
            family_scales = scales[family_idx]
            delta = (
                path_states.unsqueeze(2) - centers.unsqueeze(0).unsqueeze(0)
            ) / family_scales.unsqueeze(0).unsqueeze(0)
            dist_sq = (delta ** 2).sum(dim=-1)
            anchor_support = torch.exp(-0.5 * dist_sq)
            assignment = F.softmax(
                (-dist_sq) / max(self.support_temperature, 1e-4),
                dim=-1,
            )
            family_support = (assignment * anchor_support).sum(dim=-1)

            family_support_paths.append(family_support)
            family_assignment_paths.append(assignment)
            family_anchor_support_paths.append(anchor_support)

        family_support_paths_t = torch.stack(family_support_paths, dim=-1)
        family_distribution_paths = family_support_paths_t.clamp_min(1e-6)
        family_distribution_paths = family_distribution_paths / family_distribution_paths.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)

        family_support = family_support_paths_t.mean(dim=0)
        family_distribution = family_support.clamp_min(1e-6)
        family_distribution = family_distribution / family_distribution.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        topk = torch.topk(family_support, k=min(2, self.num_families), dim=-1).values
        top_support = topk[:, 0].clamp(min=0.0, max=1.0)
        runner_up = topk[:, 1] if topk.size(-1) > 1 else torch.zeros_like(top_support)
        family_margin = (
            (top_support - runner_up).clamp_min(0.0)
            / (top_support + runner_up).clamp_min(1e-6)
        ).clamp(min=0.0, max=1.0)
        family_conflict = -(
            family_distribution * family_distribution.log()
        ).sum(dim=-1) / math.log(self.num_families)

        top_family_per_path = family_support_paths_t.argmax(dim=-1)
        modal_family = top_family_per_path.mode(dim=0).values
        anchor_switch_rate = (top_family_per_path != modal_family.unsqueeze(0)).float().mean(dim=0)

        anchor_assignment_paths = torch.stack(family_assignment_paths, dim=2)
        anchor_support_paths = torch.stack(family_anchor_support_paths, dim=2)
        anchor_assignment = anchor_assignment_paths.mean(dim=0)
        anchor_support = anchor_support_paths.mean(dim=0)
        top_anchor_ids = anchor_assignment.argmax(dim=-1)

        family_support_std = family_support_paths_t.std(dim=0, unbiased=False)
        family_approach = (
            1.0 - family_support_std / family_support.clamp_min(1e-3)
        ).clamp(min=0.0, max=1.0)
        family_leap_penalty = family_support_std.clamp(min=0.0, max=1.0)

        return {
            "family_support_paths": family_support_paths_t,
            "family_distribution_paths": family_distribution_paths,
            "family_support": family_support,
            "family_distribution": family_distribution,
            "family_approach": family_approach,
            "family_leap_penalty": family_leap_penalty,
            "top_support": top_support,
            "family_margin": family_margin,
            "family_conflict": family_conflict,
            "anchor_switch_rate": anchor_switch_rate,
            "top_family_id": family_support.argmax(dim=-1),
            "top_anchor_ids": top_anchor_ids,
            "anchor_assignment_probs": anchor_assignment,
            "anchor_support_scores": anchor_support,
            "anchor_centers": self.centers,
            "anchor_scales": scales,
        }


class DiffusionBottleneckHead(nn.Module):
    """Diffusion-based epistemic bottleneck for E7."""

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
        perturbation_rank_margin: float = 0.03,
        dropout: float = 0.1,
        **_kwargs,
    ):
        super().__init__()

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
        self.perturbation_rank_margin = perturbation_rank_margin

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
        self.path_outcome_decoder = _make_mlp(d_model, hidden_dim // 2, num_outcome, dropout)
        self.trust_head = DiffusionTrustHead(
            hidden_dim=trust_hidden,
            confidence_threshold_init=confidence_threshold_init,
            confidence_threshold_gap_init=confidence_threshold_gap_init,
            scale_init=monotone_scale_init,
            dropout=dropout,
        )
        self.confidence_head = OrderedThresholdHead(
            threshold_init=confidence_threshold_init,
            threshold_gap_init=confidence_threshold_gap_init,
            scale_init=monotone_scale_init,
        )
        self.prototype_state_proj = _make_mlp(d_model, hidden_dim, anchor_state_dim, dropout)
        self.prototype_bank = DiffusionPrototypeBank(
            state_dim=anchor_state_dim,
            prototypes_per_family=anchors_per_family,
            support_temperature=prototype_support_temperature,
        )
        self.fragility_head = DiffusionFragilityHead(
            input_dim=DiffusionTrustHead.STATS_DIM + 4,
            hidden_dim=fragility_hidden,
            dropout=dropout,
        )

        self._trace_metadata: dict | None = None

    def set_trace_metadata(self, metadata: dict | None) -> None:
        self._trace_metadata = metadata

    def _noise_z(self, z0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_t = self.alpha_bar[t - 1].unsqueeze(-1)
        eps = torch.randn_like(z0)
        z_t = torch.sqrt(alpha_t) * z0 + torch.sqrt(1.0 - alpha_t) * eps
        return z_t, eps

    def _denoise_z(self, z_t: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_t = self.alpha_bar[t - 1].unsqueeze(-1)
        eps_pred = self.denoiser(z_t, t)
        z0_hat = (z_t - torch.sqrt(1.0 - alpha_t) * eps_pred) / torch.sqrt(alpha_t).clamp_min(1e-4)
        return z0_hat, eps_pred

    def _run_random_paths(self, z0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, device = z0.size(0), z0.device
        z_rec_list, out_list, mse_list = [], [], []

        for _ in range(self.num_random_paths):
            t = torch.randint(1, self.T + 1, (bsz,), device=device)
            z_t, eps_true = self._noise_z(z0, t)
            z0_hat, eps_pred = self._denoise_z(z_t, t)
            mse_list.append(F.mse_loss(eps_pred, eps_true, reduction="none").mean(dim=-1))
            z_rec_list.append(z0_hat)
            out_list.append(F.softmax(self.path_outcome_decoder(z0_hat), dim=-1))

        z_rec_stack = torch.stack(z_rec_list, dim=0)
        out_stack = torch.stack(out_list, dim=0)
        denoise_mse = torch.stack(mse_list, dim=0).mean()
        return z_rec_stack, out_stack, denoise_mse

    @staticmethod
    def _compute_certainty_stats(
        z0: torch.Tensor,
        z_rec_stack: torch.Tensor,
        out_stack: torch.Tensor,
        num_outcome: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent_rec_var = z_rec_stack.var(dim=0).mean(dim=-1)
        base_outcome = out_stack.mean(dim=0)
        base_outcome_cls = out_stack.argmax(dim=-1)
        mode_cls = base_outcome_cls.mode(dim=0).values
        answer_agreement = (base_outcome_cls == mode_cls.unsqueeze(0)).float().mean(dim=0)

        safe_base = base_outcome.clamp_min(1e-8)
        answer_entropy = -(safe_base * safe_base.log()).sum(dim=-1)
        answer_entropy_norm = (answer_entropy / math.log(num_outcome)).clamp(0.0, 1.0)

        denoise_energy = (z_rec_stack - z0.unsqueeze(0)).norm(dim=-1).mean(dim=0)
        top_path_cls = out_stack.argmax(dim=-1)
        basin_stability = (top_path_cls == top_path_cls[0].unsqueeze(0)).float().mean(dim=0)

        var_ref = latent_rec_var.detach().mean().clamp_min(1e-4)
        energy_ref = denoise_energy.detach().mean().clamp_min(1e-4)
        latent_var_norm = (latent_rec_var / var_ref).clamp(0.0, 2.0) / 2.0
        denoise_energy_norm = (denoise_energy / energy_ref).clamp(0.0, 2.0) / 2.0

        certainty_stats = torch.stack(
            [
                latent_var_norm,
                answer_agreement,
                answer_entropy_norm,
                denoise_energy_norm,
                basin_stability,
            ],
            dim=-1,
        )
        return certainty_stats, base_outcome

    def _build_joint_latent(self, z_q: torch.Tensor, executor_features: torch.Tensor) -> torch.Tensor:
        return self.joint_latent_proj(torch.cat([z_q, executor_features], dim=-1))

    def _noise_z_with_eps(
        self,
        z0: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        alpha_t = self.alpha_bar[t - 1].unsqueeze(-1)
        return torch.sqrt(alpha_t) * z0 + torch.sqrt(1.0 - alpha_t) * eps

    def _prototype_features(
        self,
        z0: torch.Tensor,
        z_rec_stack: torch.Tensor,
        out_stack: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del z0
        num_paths, bsz, _ = z_rec_stack.shape
        path_states = self.prototype_state_proj(z_rec_stack.reshape(num_paths * bsz, -1)).reshape(
            num_paths,
            bsz,
            self.anchor_state_dim,
        )
        proto = self.prototype_bank(path_states)
        target_family_probs = out_stack[..., [0, 2, 1]].clamp_min(1e-6).detach()
        prototype_alignment_loss = F.kl_div(
            proto["family_distribution_paths"].log(),
            target_family_probs,
            reduction="none",
        ).sum(dim=-1).mean()
        proto["path_states"] = path_states
        proto["prototype_alignment_loss"] = prototype_alignment_loss
        return proto

    def compute_perturbation_aux_losses(
        self,
        z_q: torch.Tensor,
        executor_features: torch.Tensor,
        perturbed_executor_features: torch.Tensor,
        primitives: PrimitiveOutput,
        perturbed_primitives: PrimitiveOutput,
    ) -> dict[str, torch.Tensor]:
        z0_base = self._build_joint_latent(z_q, executor_features)
        z0_perturbed = self._build_joint_latent(z_q, perturbed_executor_features)

        bsz = z0_base.size(0)
        device = z0_base.device
        t = torch.randint(1, self.T + 1, (bsz,), device=device)
        eps = torch.randn_like(z0_base)

        base_z_t = self._noise_z_with_eps(z0_base, t, eps)
        perturbed_z_t = self._noise_z_with_eps(z0_perturbed, t, eps)
        _base_rec, eps_pred_base = self._denoise_z(base_z_t, t)
        _perturbed_rec, eps_pred_perturbed = self._denoise_z(perturbed_z_t, t)

        base_mse = F.mse_loss(eps_pred_base, eps, reduction="none").mean(dim=-1)
        perturbed_mse = F.mse_loss(eps_pred_perturbed, eps, reduction="none").mean(dim=-1)

        base_family_support = torch.stack(
            [
                primitives.success_family_support,
                primitives.failure_family_support,
                primitives.boundary_family_support,
            ],
            dim=-1,
        )
        perturbed_family_support = torch.stack(
            [
                perturbed_primitives.success_family_support,
                perturbed_primitives.failure_family_support,
                perturbed_primitives.boundary_family_support,
            ],
            dim=-1,
        )
        base_top_family = base_family_support.argmax(dim=-1)
        base_top_support = base_family_support.gather(1, base_top_family.unsqueeze(-1)).squeeze(-1)
        perturbed_top_support = perturbed_family_support.gather(1, base_top_family.unsqueeze(-1)).squeeze(-1)

        support_drop = torch.relu(base_top_support - perturbed_top_support)
        conflict_rise = torch.relu(perturbed_primitives.anchor_conflict - primitives.anchor_conflict)
        switch_rise = torch.relu(
            perturbed_primitives.prototype_anchor_switch_rate - primitives.prototype_anchor_switch_rate
        )
        geometry_gate = (support_drop + 0.5 * conflict_rise + 0.5 * switch_rise).clamp(min=0.0, max=1.0)
        perturbation_rank = (
            geometry_gate * torch.relu(base_mse - perturbed_mse + self.perturbation_rank_margin)
        ).mean()
        perturbation_proximity = F.smooth_l1_loss(
            z0_perturbed,
            z0_base.detach(),
            reduction="none",
        ).mean(dim=-1).mean()

        return {
            "perturbation_rank": perturbation_rank,
            "perturbation_proximity": perturbation_proximity,
            "perturbation_geometry_gate": geometry_gate.mean(),
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
    ) -> PrimitiveOutput:
        del path_scales

        bsz, device = z_q.size(0), z_q.device

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

        traj_disc = _discretise(traj_logits, mode, tau)
        pattern_disc = _discretise(pattern_logits, mode, tau)

        z0 = self._build_joint_latent(z_q, executor_features)
        z_rec_stack, out_stack, denoise_mse = self._run_random_paths(z0)
        certainty_stats, base_outcome = self._compute_certainty_stats(z0, z_rec_stack, out_stack, self.num_outcome)
        prototype = self._prototype_features(z0, z_rec_stack, out_stack)

        _cert_logits, cert_risk = self.trust_head(certainty_stats)
        certainty_score = 1.0 - cert_risk

        outcome_probs = base_outcome.clamp_min(1e-6)
        outcome_probs = outcome_probs / outcome_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        outcome_logits = outcome_probs.log()

        latent_var_norm = certainty_stats[:, 0]
        answer_agreement = certainty_stats[:, 1]
        answer_entropy_norm = certainty_stats[:, 2]
        denoise_energy_norm = certainty_stats[:, 3]
        basin_stability = certainty_stats[:, 4]

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
        prototype_family_margin = prototype["family_margin"]
        prototype_anchor_switch_rate = prototype["anchor_switch_rate"]
        anchor_conflict = prototype["family_conflict"]
        prototype_alignment_loss = prototype["prototype_alignment_loss"]

        success_cert = outcome_probs[:, 0]
        ambiguity_cert = outcome_probs[:, 1]
        failure_cert = outcome_probs[:, 2]

        geometry_stats = torch.stack(
            [
                prototype_top_support,
                prototype_family_margin,
                anchor_conflict,
                prototype_anchor_switch_rate,
            ],
            dim=-1,
        )
        fragility_risk = self.fragility_head(torch.cat([certainty_stats, geometry_stats], dim=-1))
        fragility_score = 1.0 - fragility_risk
        confidence_risk = torch.maximum(cert_risk, fragility_risk)
        confidence_logits = self.confidence_head(confidence_risk)

        conf_disc = _discretise(confidence_logits, mode, tau)
        outcome_disc = _discretise(outcome_logits, mode, tau)

        stability_cert = basin_stability
        support_cert = (0.5 * ((1.0 - latent_var_norm).clamp(0.0, 1.0) + prototype_top_support)).clamp(0.0, 1.0)
        success_guard = (
            success_cert
            * (1.0 - 0.35 * torch.maximum(answer_entropy_norm, anchor_conflict))
            * (1.0 - 0.35 * fragility_risk)
        ).clamp(0.0, 1.0)
        success_core = (success_cert * basin_stability * (1.0 - 0.25 * fragility_risk)).clamp(0.0, 1.0)
        success_base = success_cert
        success_affinity = success_family_support
        decisiveness_score = (1.0 - ambiguity_cert).clamp(0.0, 1.0)
        commitment_depth = certainty_score * fragility_score * decisiveness_score

        agreement = executor_feature_column(executor_features, "agreement")
        prob_margin = executor_feature_column(executor_features, "prob_margin")
        max_att_mass = executor_feature_column(executor_features, "max_attention_mass")
        variation_ratio = executor_feature_column(executor_features, "variation_ratio")
        low_margin_flag = executor_feature_column(executor_features, "low_margin_flag")
        unstable_flag = executor_feature_column(executor_features, "unstable_flag")

        trust_support_deficit = (1.0 - prototype_top_support).clamp(0.0, 1.0)
        trust_approach_deficit = (1.0 - answer_agreement).clamp(0.0, 1.0)
        trust_leap_cost = denoise_energy_norm
        trust_conflict = torch.maximum(answer_entropy_norm, anchor_conflict)
        trust_agreement_deficit = (1.0 - agreement).clamp(0.0, 1.0)
        trust_margin_deficit = torch.clamp(1.0 - prob_margin, min=low_margin_flag)
        trust_concentration_deficit = (1.0 - max_att_mass).clamp(0.0, 1.0)
        trust_variation_pressure = torch.clamp(variation_ratio, min=unstable_flag)

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
            success_proposal=success_cert,
            success_cert=success_cert,
            failure_cert=failure_cert,
            ambiguity_cert=ambiguity_cert,
            cert_risk=cert_risk,
            certainty_score=certainty_score,
            fragility_risk=fragility_risk,
            fragility_score=fragility_score,
            decisiveness_score=decisiveness_score,
            prototype_top_support=prototype_top_support,
            prototype_family_margin=prototype_family_margin,
            prototype_anchor_switch_rate=prototype_anchor_switch_rate,
            prototype_alignment_loss=prototype_alignment_loss,
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
