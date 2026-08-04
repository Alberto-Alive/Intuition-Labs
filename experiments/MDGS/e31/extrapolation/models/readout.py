"""Witness/path-agreement statistics and readout modules for E31."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-6)
    return math.log(math.expm1(value))


def bucketize_outcome_logits(outcome_logits: torch.Tensor) -> torch.Tensor:
    """Late-only reporting bucketization for SUCCESS / UNCERTAIN / FAILURE."""

    return outcome_logits.argmax(dim=-1)


def bucketize_hard_commitment_decisions(
    global_margin: torch.Tensor,
    commitment_score: torch.Tensor,
    tau_uncertain: torch.Tensor,
) -> torch.Tensor:
    """Hard evaluation gate: low commitment routes to UNCERTAIN, else sign(M)."""

    uncertain_mask = commitment_score < tau_uncertain
    success_mask = global_margin >= 0
    uncertain = torch.ones_like(global_margin, dtype=torch.long)
    success = torch.zeros_like(global_margin, dtype=torch.long)
    failure = torch.full_like(global_margin, 2, dtype=torch.long)
    return torch.where(uncertain_mask, uncertain, torch.where(success_mask, success, failure))


@dataclass
class WitnessDiagnosticProbes:
    """Optional analysis-only probes, detached from the main decision path."""

    order: torch.Tensor
    boundary: torch.Tensor
    support: torch.Tensor


@dataclass
class RawWitnessGeometry:
    """Raw diffusion-witness geometry measured before the point reader."""

    pairwise_cosine_agreement: torch.Tensor
    pairwise_cosine_agreement_std: torch.Tensor
    mean_squared_distance_to_anchor: torch.Tensor
    witness_latent_variance: torch.Tensor
    witness_latent_norm_mean: torch.Tensor
    witness_latent_norm_std: torch.Tensor
    witness_latent_spread: torch.Tensor
    principal_singular_value_ratio: torch.Tensor


@dataclass
class WitnessSetStats:
    """Witness-derived summary statistics that feed the final readout."""

    witness_margins: torch.Tensor
    pairwise_agreement: torch.Tensor
    coherence_scores: torch.Tensor
    witness_weights: torch.Tensor
    global_margin: torch.Tensor
    vote_disagreement: torch.Tensor
    geometric_disagreement: torch.Tensor
    margin_abs: torch.Tensor
    witness_margin_mean: torch.Tensor
    witness_margin_std: torch.Tensor
    witness_weight_entropy: torch.Tensor
    witness_weight_max: torch.Tensor
    witness_weight_kl_uniform: torch.Tensor
    pairwise_agreement_mean: torch.Tensor
    pairwise_agreement_std: torch.Tensor
    per_witness_path_disagreement: torch.Tensor | None = None
    path_disagreement: torch.Tensor | None = None
    path_weight_eta: torch.Tensor | None = None


@dataclass
class WitnessReadoutResult:
    """Outputs from the strict witness-agreement decision head."""

    margin: torch.Tensor
    uncertainty: torch.Tensor
    outcome_logits: torch.Tensor
    decision_inputs: dict[str, torch.Tensor]


def summarize_raw_witness_geometry(
    anchor: torch.Tensor,
    witness_samples: torch.Tensor,
) -> RawWitnessGeometry:
    """Summarize raw witness geometry before any point-reader compression."""

    if witness_samples.ndim != 3:
        raise ValueError("witness_samples must have shape (num_samples, batch_size, latent_dim)")
    if anchor.ndim != 2:
        raise ValueError("anchor must have shape (batch_size, latent_dim)")

    num_samples, batch_size, latent_dim = witness_samples.shape
    if anchor.shape != (batch_size, latent_dim):
        raise ValueError(
            f"Expected anchor shape {(batch_size, latent_dim)}, got {tuple(anchor.shape)}"
        )

    witness_repr = witness_samples.permute(1, 0, 2)
    centered = witness_repr - witness_repr.mean(dim=1, keepdim=True)
    normalized_repr = F.normalize(witness_repr, p=2, dim=-1, eps=1e-8)
    pairwise = torch.matmul(normalized_repr, normalized_repr.transpose(-1, -2))

    if num_samples > 1:
        offdiag_mask = ~torch.eye(num_samples, dtype=torch.bool, device=pairwise.device)
        offdiag_values = pairwise[:, offdiag_mask].reshape(batch_size, -1)
        pairwise_cosine_agreement = offdiag_values.mean(dim=-1)
        pairwise_cosine_agreement_std = offdiag_values.std(dim=-1, unbiased=False)
        svdvals = torch.linalg.svdvals(centered)
        principal_singular_value_ratio = svdvals[:, 0] / svdvals.sum(dim=-1).clamp_min(1e-8)
    else:
        pairwise_cosine_agreement = torch.ones(
            batch_size, device=witness_samples.device, dtype=witness_samples.dtype
        )
        pairwise_cosine_agreement_std = torch.zeros_like(pairwise_cosine_agreement)
        principal_singular_value_ratio = torch.ones_like(pairwise_cosine_agreement)

    mean_squared_distance_to_anchor = (witness_samples - anchor.unsqueeze(0)).pow(2).sum(
        dim=-1
    ).mean(dim=0)
    witness_latent_variance = centered.pow(2).mean(dim=(1, 2))
    witness_norms = witness_samples.norm(dim=-1)
    witness_latent_norm_mean = witness_norms.mean(dim=0)
    witness_latent_norm_std = witness_norms.std(dim=0, unbiased=False)
    witness_latent_spread = centered.pow(2).sum(dim=-1).mean(dim=-1)

    return RawWitnessGeometry(
        pairwise_cosine_agreement=pairwise_cosine_agreement,
        pairwise_cosine_agreement_std=pairwise_cosine_agreement_std,
        mean_squared_distance_to_anchor=mean_squared_distance_to_anchor,
        witness_latent_variance=witness_latent_variance,
        witness_latent_norm_mean=witness_latent_norm_mean,
        witness_latent_norm_std=witness_latent_norm_std,
        witness_latent_spread=witness_latent_spread,
        principal_singular_value_ratio=principal_singular_value_ratio,
    )


class DetachedProbeHeads(nn.Module):
    """Optional support / boundary / order probes over detached witness summaries."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.order_head = nn.Linear(feature_dim, 1)
        self.boundary_head = nn.Linear(feature_dim, 1)
        self.support_head = nn.Linear(feature_dim, 1)

    def forward(self, features: torch.Tensor) -> WitnessDiagnosticProbes:
        return WitnessDiagnosticProbes(
            order=torch.tanh(self.order_head(features)).squeeze(-1),
            boundary=torch.sigmoid(self.boundary_head(features)).squeeze(-1),
            support=torch.sigmoid(self.support_head(features)).squeeze(-1),
        )


class WitnessAgreementReadout(nn.Module):
    """Implements the E31 cooperative witness/path decision rule."""

    def __init__(
        self,
        latent_dim: int,
        *,
        witness_weight_temperature: float = 1.0,
        use_vote_disagreement: bool = True,
        use_geometric_disagreement: bool = True,
        use_margin_term: bool = True,
        use_coherence_weighting: bool = True,
        gate_mode: str = "soft_commitment_gate",
        commitment_scale_init: float = 1.0,
        tau_uncertain_init: float = 0.0,
        alpha_init: float = 1.0,
        beta_init: float = 1.0,
        gamma_init: float = 1.0,
        bias_init: float = 0.0,
        use_path_disagreement: bool = False,
        use_method_conflict: bool = False,
        use_path_commitment_penalty: bool = False,
        use_path_aware_witness_weights: bool = False,
        path_delta_init: float = 1.0,
        method_rho_init: float = 1.0,
        path_commitment_lambda_init: float = 1.0,
        path_weight_eta_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.margin_head = nn.Linear(latent_dim, 1)
        self.witness_weight_temperature = float(witness_weight_temperature)
        self.use_vote_disagreement = bool(use_vote_disagreement)
        self.use_geometric_disagreement = bool(use_geometric_disagreement)
        self.use_margin_term = bool(use_margin_term)
        self.use_coherence_weighting = bool(use_coherence_weighting)
        self.use_path_disagreement = bool(use_path_disagreement)
        self.use_method_conflict = bool(use_method_conflict)
        self.use_path_commitment_penalty = bool(use_path_commitment_penalty)
        self.use_path_aware_witness_weights = bool(use_path_aware_witness_weights)
        self.gate_mode = gate_mode.strip().lower()
        if self.gate_mode not in {"soft_commitment_gate", "legacy_logits"}:
            raise ValueError("gate_mode must be one of {'soft_commitment_gate', 'legacy_logits'}")
        self.raw_alpha = nn.Parameter(torch.tensor(_inverse_softplus(alpha_init)))
        self.raw_beta = nn.Parameter(torch.tensor(_inverse_softplus(beta_init)))
        self.raw_gamma = nn.Parameter(torch.tensor(_inverse_softplus(gamma_init)))
        self.raw_path_delta = nn.Parameter(torch.tensor(_inverse_softplus(path_delta_init)))
        self.raw_method_rho = nn.Parameter(torch.tensor(_inverse_softplus(method_rho_init)))
        self.raw_lambda_commit = nn.Parameter(torch.tensor(_inverse_softplus(1.0)))
        self.raw_lambda_path = nn.Parameter(torch.tensor(_inverse_softplus(path_commitment_lambda_init)))
        self.raw_path_weight_eta = nn.Parameter(torch.tensor(_inverse_softplus(path_weight_eta_init)))
        self.raw_commitment_scale = nn.Parameter(torch.tensor(_inverse_softplus(commitment_scale_init)))
        self.tau_uncertain = nn.Parameter(torch.tensor(float(tau_uncertain_init)))
        self.bias = nn.Parameter(torch.tensor(float(bias_init)))

    def _resolve_local_margins(
        self,
        margin_samples: torch.Tensor,
        *,
        num_samples: int,
        batch_size: int,
    ) -> torch.Tensor:
        if margin_samples.ndim == 3:
            sample_count, input_batch_size, latent_dim = margin_samples.shape
            if sample_count != num_samples or input_batch_size != batch_size:
                raise ValueError(
                    "margin_samples must align with witness_samples along the sample and batch axes"
                )
            if latent_dim == 1:
                return margin_samples.squeeze(-1).transpose(0, 1)
            flat = margin_samples.reshape(num_samples * batch_size, latent_dim)
            local_margins = self.margin_head(flat)
            return local_margins.reshape(num_samples, batch_size).transpose(0, 1)
        if margin_samples.ndim == 2:
            if margin_samples.shape == (batch_size, num_samples):
                return margin_samples
            if margin_samples.shape == (num_samples, batch_size):
                return margin_samples.transpose(0, 1)
            raise ValueError(
                "margin_samples with ndim=2 must have shape (batch_size, num_samples) or (num_samples, batch_size)"
            )
        raise ValueError(
            "margin_samples must have shape (num_samples, batch_size, latent_dim) or (batch_size, num_samples)"
        )

    def summarize(
        self,
        witness_samples: torch.Tensor,
        *,
        margin_samples: torch.Tensor | None = None,
        per_witness_path_disagreement: torch.Tensor | None = None,
        path_disagreement: torch.Tensor | None = None,
    ) -> WitnessSetStats:
        if witness_samples.ndim != 3:
            raise ValueError(
                "witness_samples must have shape (num_samples, batch_size, latent_dim)"
            )

        num_samples, batch_size, latent_dim = witness_samples.shape
        if num_samples <= 0:
            raise ValueError("witness_samples must include at least one sample")

        if margin_samples is None:
            margin_samples = witness_samples
        local_margins = self._resolve_local_margins(
            margin_samples,
            num_samples=num_samples,
            batch_size=batch_size,
        )
        if per_witness_path_disagreement is not None:
            if per_witness_path_disagreement.shape != (batch_size, num_samples):
                raise ValueError(
                    "per_witness_path_disagreement must have shape "
                    f"{(batch_size, num_samples)}, got {tuple(per_witness_path_disagreement.shape)}"
                )
            per_witness_path_disagreement = per_witness_path_disagreement.to(
                device=witness_samples.device,
                dtype=witness_samples.dtype,
            )
        if path_disagreement is not None:
            if path_disagreement.shape != (batch_size,):
                raise ValueError(
                    f"path_disagreement must have shape {(batch_size,)}, got {tuple(path_disagreement.shape)}"
                )
            path_disagreement = path_disagreement.to(
                device=witness_samples.device,
                dtype=witness_samples.dtype,
            )

        witness_repr = witness_samples.permute(1, 0, 2)
        normalized_repr = F.normalize(witness_repr, p=2, dim=-1, eps=1e-8)
        pairwise_agreement = torch.matmul(normalized_repr, normalized_repr.transpose(-1, -2))

        if num_samples > 1:
            coherence_scores = (pairwise_agreement.sum(dim=-1) - 1.0) / float(num_samples - 1)
            offdiag_mask = ~torch.eye(num_samples, dtype=torch.bool, device=pairwise_agreement.device)
            offdiag_values = pairwise_agreement[:, offdiag_mask].reshape(batch_size, -1)
            pairwise_agreement_mean = offdiag_values.mean(dim=-1)
            pairwise_agreement_std = offdiag_values.std(dim=-1, unbiased=False)
        else:
            coherence_scores = torch.zeros_like(local_margins)
            pairwise_agreement_mean = torch.ones(
                batch_size, device=witness_samples.device, dtype=witness_samples.dtype
            )
            pairwise_agreement_std = torch.zeros(
                batch_size, device=witness_samples.device, dtype=witness_samples.dtype
            )

        path_weight_eta = F.softplus(self.raw_path_weight_eta)
        if self.use_coherence_weighting:
            temperature = max(self.witness_weight_temperature, 1e-6)
            weight_logits = coherence_scores
            if self.use_path_aware_witness_weights and per_witness_path_disagreement is not None:
                weight_logits = weight_logits - path_weight_eta * per_witness_path_disagreement
            witness_weights = F.softmax(weight_logits / temperature, dim=-1)
        else:
            witness_weights = torch.full_like(coherence_scores, 1.0 / float(num_samples))

        global_margin = (witness_weights * local_margins).sum(dim=-1)
        centered = local_margins - global_margin.unsqueeze(-1)
        vote_disagreement = (witness_weights * centered.pow(2)).sum(dim=-1)
        margin_abs = global_margin.abs()
        witness_margin_mean = local_margins.mean(dim=-1)
        witness_margin_std = local_margins.std(dim=-1, unbiased=False)
        witness_weight_max = witness_weights.max(dim=-1).values

        clamped_weights = witness_weights.clamp_min(1e-8)
        weight_entropy = -(clamped_weights * clamped_weights.log()).sum(dim=-1)
        witness_weight_kl_uniform = (
            clamped_weights * (clamped_weights * float(num_samples)).log()
        ).sum(dim=-1)
        if num_samples > 1:
            weight_entropy = weight_entropy / math.log(float(num_samples))
        else:
            weight_entropy = torch.zeros_like(weight_entropy)
            witness_weight_kl_uniform = torch.zeros_like(witness_weight_kl_uniform)

        geometric_disagreement = 1.0 - pairwise_agreement_mean
        return WitnessSetStats(
            witness_margins=local_margins,
            pairwise_agreement=pairwise_agreement,
            coherence_scores=coherence_scores,
            witness_weights=witness_weights,
            global_margin=global_margin,
            vote_disagreement=vote_disagreement,
            geometric_disagreement=geometric_disagreement,
            margin_abs=margin_abs,
            witness_margin_mean=witness_margin_mean,
            witness_margin_std=witness_margin_std,
            witness_weight_entropy=weight_entropy,
            witness_weight_max=witness_weight_max,
            witness_weight_kl_uniform=witness_weight_kl_uniform,
            pairwise_agreement_mean=pairwise_agreement_mean,
            pairwise_agreement_std=pairwise_agreement_std,
            per_witness_path_disagreement=per_witness_path_disagreement,
            path_disagreement=path_disagreement,
            path_weight_eta=path_weight_eta.expand_as(global_margin),
        )

    def forward(self, stats: WitnessSetStats) -> WitnessReadoutResult:
        global_margin = stats.global_margin
        alpha = F.softplus(self.raw_alpha) if self.use_vote_disagreement else global_margin.new_zeros(())
        beta = F.softplus(self.raw_beta) if self.use_geometric_disagreement else global_margin.new_zeros(())
        gamma = F.softplus(self.raw_gamma) if self.use_margin_term else global_margin.new_zeros(())
        use_path = stats.path_disagreement is not None
        path_delta = (
            F.softplus(self.raw_path_delta)
            if self.use_path_disagreement and use_path
            else global_margin.new_zeros(())
        )
        method_rho = (
            F.softplus(self.raw_method_rho)
            if self.use_method_conflict and use_path
            else global_margin.new_zeros(())
        )
        lambda_commit = F.softplus(self.raw_lambda_commit)
        lambda_path = (
            F.softplus(self.raw_lambda_path)
            if self.use_path_commitment_penalty and use_path
            else global_margin.new_zeros(())
        )
        commitment_scale = F.softplus(self.raw_commitment_scale)
        if stats.path_disagreement is None:
            path_disagreement = torch.zeros_like(global_margin)
        else:
            path_disagreement = stats.path_disagreement
        path_confidence = (1.0 - 0.5 * path_disagreement).clamp(0.0, 1.0)
        endpoint_confidence = stats.margin_abs / (stats.margin_abs + 1.0)
        method_conflict = (endpoint_confidence - path_confidence).abs()

        uncertainty = F.softplus(
            alpha * stats.vote_disagreement
            + beta * stats.geometric_disagreement
            + path_delta * path_disagreement
            + method_rho * method_conflict
            - gamma * stats.margin_abs
            + self.bias
        )
        commitment_score = stats.margin_abs - lambda_commit * uncertainty - lambda_path * path_disagreement
        commitment_logit = commitment_scale * (commitment_score - self.tau_uncertain)
        tau_uncertain = self.tau_uncertain.expand_as(commitment_score)
        margin_minus_lambda_u = global_margin - lambda_commit * uncertainty

        if self.gate_mode == "soft_commitment_gate":
            uncertain = F.softplus(-commitment_logit)
            success = commitment_logit + global_margin - uncertain
            failure = commitment_logit - global_margin - uncertain
        elif self.gate_mode == "legacy_logits":
            uncertain = uncertainty
            success = global_margin - uncertainty
            failure = -global_margin - uncertainty
        else:  # pragma: no cover - validated in __init__
            raise ValueError(f"Unsupported gate_mode {self.gate_mode!r}")

        outcome_logits = torch.stack([success, uncertain, failure], dim=-1)
        return WitnessReadoutResult(
            margin=global_margin,
            uncertainty=uncertainty,
            outcome_logits=outcome_logits,
            decision_inputs={
                "global_margin": global_margin,
                "vote_disagreement": stats.vote_disagreement,
                "geometric_disagreement": stats.geometric_disagreement,
                "margin_abs": stats.margin_abs,
                "witness_margin_mean": stats.witness_margin_mean,
                "witness_margin_std": stats.witness_margin_std,
                "witness_weight_entropy": stats.witness_weight_entropy,
                "witness_weight_max": stats.witness_weight_max,
                "witness_weight_kl_uniform": stats.witness_weight_kl_uniform,
                "pairwise_agreement_mean": stats.pairwise_agreement_mean,
                "pairwise_agreement_std": stats.pairwise_agreement_std,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "path_delta": path_delta.expand_as(commitment_score),
                "method_rho": method_rho.expand_as(commitment_score),
                "lambda_commit": lambda_commit.expand_as(commitment_score),
                "lambda_path": lambda_path.expand_as(commitment_score),
                "path_weight_eta": (
                    stats.path_weight_eta
                    if stats.path_weight_eta is not None
                    else torch.zeros_like(commitment_score)
                ),
                "commitment_scale": commitment_scale.expand_as(commitment_score),
                "commitment_score": commitment_score,
                "margin_minus_lambda_u": margin_minus_lambda_u,
                "commitment_logit": commitment_logit,
                "tau_uncertain": tau_uncertain,
                "uncertainty": uncertainty,
                "uncertain_logit": uncertain,
                "path_disagreement": path_disagreement,
                "path_confidence": path_confidence,
                "endpoint_confidence": endpoint_confidence,
                "method_conflict": method_conflict,
            },
        )
