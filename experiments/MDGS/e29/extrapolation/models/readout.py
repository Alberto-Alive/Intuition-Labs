"""Cloud-geometry statistics and readout modules for E29."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-6)
    return math.log(math.expm1(value))


@dataclass
class CloudGeometryStats:
    """Geometry-only statistics of a denoised diffusion cloud."""

    mu: torch.Tensor
    var: torch.Tensor
    radius: torch.Tensor
    agreement: torch.Tensor
    mean_var: torch.Tensor


@dataclass
class GeometryReadoutResult:
    """Outputs from the strict geometry-only decision head."""

    margin: torch.Tensor
    uncertainty: torch.Tensor
    outcome_logits: torch.Tensor
    decision_inputs: dict[str, torch.Tensor]


@dataclass
class CloudDiagnosticProbes:
    """Optional diagnostic probes that are explicitly disconnected from the decision logits."""

    order: torch.Tensor
    boundary: torch.Tensor
    support: torch.Tensor


def compute_cloud_geometry(cloud_samples: torch.Tensor) -> CloudGeometryStats:
    """Summarize a `K x B x D` denoised cloud into geometry-only statistics."""

    if cloud_samples.ndim != 3:
        raise ValueError(
            f"cloud_samples must have shape (num_samples, batch_size, latent_dim), got {tuple(cloud_samples.shape)}"
        )

    mu = cloud_samples.mean(dim=0)
    centered = cloud_samples - mu.unsqueeze(0)
    var = centered.pow(2).mean(dim=0)
    radius = centered.pow(2).sum(dim=-1).mean(dim=0)
    agreement = F.cosine_similarity(
        cloud_samples,
        mu.unsqueeze(0).expand_as(cloud_samples),
        dim=-1,
        eps=1e-8,
    ).mean(dim=0)
    mean_var = var.mean(dim=-1)
    return CloudGeometryStats(
        mu=mu,
        var=var,
        radius=radius,
        agreement=agreement,
        mean_var=mean_var,
    )


def bucketize_outcome_logits(outcome_logits: torch.Tensor) -> torch.Tensor:
    """Late-only reporting bucketization for SUCCESS / UNCERTAIN / FAILURE."""

    return outcome_logits.argmax(dim=-1)


class CloudGeometryReadout(nn.Module):
    """Implements the strict E29 decision rule from cloud geometry only."""

    def __init__(
        self,
        latent_dim: int,
        *,
        use_radius_term: bool = True,
        use_variance_term: bool = True,
        use_margin_term: bool = True,
        alpha_init: float = 1.0,
        beta_init: float = 1.0,
        gamma_init: float = 1.0,
        bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.margin_proj = nn.Linear(latent_dim, 1)
        self.use_radius_term = bool(use_radius_term)
        self.use_variance_term = bool(use_variance_term)
        self.use_margin_term = bool(use_margin_term)
        self.raw_alpha = nn.Parameter(torch.tensor(_inverse_softplus(alpha_init)))
        self.raw_beta = nn.Parameter(torch.tensor(_inverse_softplus(beta_init)))
        self.raw_gamma = nn.Parameter(torch.tensor(_inverse_softplus(gamma_init)))
        self.bias = nn.Parameter(torch.tensor(float(bias_init)))

    def forward(self, stats: CloudGeometryStats) -> GeometryReadoutResult:
        margin = self.margin_proj(stats.mu).squeeze(-1)
        margin_abs = margin.abs()
        alpha = F.softplus(self.raw_alpha) if self.use_radius_term else margin.new_zeros(())
        beta = F.softplus(self.raw_beta) if self.use_variance_term else margin.new_zeros(())
        gamma = F.softplus(self.raw_gamma) if self.use_margin_term else margin.new_zeros(())

        uncertainty = F.softplus(alpha * stats.radius + beta * stats.mean_var - gamma * margin_abs + self.bias)
        outcome_logits = torch.stack(
            [
                margin - uncertainty,
                uncertainty,
                -margin - uncertainty,
            ],
            dim=-1,
        )
        return GeometryReadoutResult(
            margin=margin,
            uncertainty=uncertainty,
            outcome_logits=outcome_logits,
            decision_inputs={
                "mu": stats.mu,
                "radius": stats.radius,
                "mean_var": stats.mean_var,
                "agreement": stats.agreement,
                "margin_abs": margin_abs,
            },
        )


class DiagnosticProbeHeads(nn.Module):
    """Optional order / boundary / support probes for audit-only use."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        feature_dim = latent_dim + 3
        self.order_head = nn.Linear(feature_dim, 1)
        self.boundary_head = nn.Linear(feature_dim, 1)
        self.support_head = nn.Linear(feature_dim, 1)

    def forward(self, stats: CloudGeometryStats) -> CloudDiagnosticProbes:
        features = torch.cat(
            [
                stats.mu,
                stats.radius.unsqueeze(-1),
                stats.mean_var.unsqueeze(-1),
                stats.agreement.unsqueeze(-1),
            ],
            dim=-1,
        )
        return CloudDiagnosticProbes(
            order=torch.tanh(self.order_head(features)).squeeze(-1),
            boundary=torch.sigmoid(self.boundary_head(features)).squeeze(-1),
            support=torch.sigmoid(self.support_head(features)).squeeze(-1),
        )

