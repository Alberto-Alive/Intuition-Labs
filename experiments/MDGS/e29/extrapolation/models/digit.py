"""Strict uncertainty-geometry model for E29."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .diffusion import ConditionalLatentDiffusion
from .encoder import AdultQueryEncoder
from .readout import (
    CloudDiagnosticProbes,
    CloudGeometryReadout,
    CloudGeometryStats,
    DiagnosticProbeHeads,
    compute_cloud_geometry,
)


@dataclass
class E29ForwardResult:
    """Forward outputs for the E29 model."""

    anchor: torch.Tensor
    cloud_samples: torch.Tensor
    cloud_stats: CloudGeometryStats
    margin: torch.Tensor
    uncertainty: torch.Tensor
    outcome_logits: torch.Tensor
    decision_inputs: dict[str, torch.Tensor]
    diffusion_denoise_loss: torch.Tensor
    diagnostic_probes: CloudDiagnosticProbes | None
    trace_target_latent: torch.Tensor | None


class E29Model(nn.Module):
    """Encoder anchor + conditional latent diffusion + strict geometry-only decision head."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.encoder = AdultQueryEncoder(
            field_vocab_sizes=config.field_vocab_sizes,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_encoder_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )
        self.diffusion = ConditionalLatentDiffusion(
            latent_dim=config.d_model,
            num_steps=config.num_diffusion_steps,
            hidden_dim=config.diffusion_hidden_dim,
            target_hidden_dim=config.trace_target_hidden_dim,
            beta_start=config.diffusion_beta_start,
            beta_end=config.diffusion_beta_end,
            dropout=config.diffusion_dropout,
        )
        self.readout = CloudGeometryReadout(
            latent_dim=config.d_model,
            use_radius_term=config.geometry_use_radius_term,
            use_variance_term=config.geometry_use_variance_term,
            use_margin_term=config.geometry_use_margin_term,
            alpha_init=config.uncertainty_alpha_init,
            beta_init=config.uncertainty_beta_init,
            gamma_init=config.uncertainty_gamma_init,
            bias_init=config.uncertainty_bias_init,
        )
        self.probe_heads = DiagnosticProbeHeads(config.d_model) if config.enable_diagnostic_probes else None

    def forward(
        self,
        query_fields: torch.Tensor,
        trace_inputs: torch.Tensor | None = None,
        *,
        num_cloud_samples: int | None = None,
        sampling_noise: torch.Tensor | None = None,
        diffusion_training_noise: torch.Tensor | None = None,
        diffusion_training_timesteps: torch.Tensor | None = None,
    ) -> E29ForwardResult:
        anchor = self.encoder(query_fields)
        diffusion_state = self.diffusion.training_loss(
            anchor,
            trace_inputs,
            noise=diffusion_training_noise,
            timesteps=diffusion_training_timesteps,
        )
        cloud_samples = self.diffusion.sample(
            anchor,
            num_samples=int(num_cloud_samples or self.config.cloud_num_samples),
            noise=sampling_noise,
        )
        cloud_stats = compute_cloud_geometry(cloud_samples)
        readout = self.readout(cloud_stats)
        diagnostic_probes = self.probe_heads(cloud_stats) if self.probe_heads is not None else None
        return E29ForwardResult(
            anchor=anchor,
            cloud_samples=cloud_samples,
            cloud_stats=cloud_stats,
            margin=readout.margin,
            uncertainty=readout.uncertainty,
            outcome_logits=readout.outcome_logits,
            decision_inputs=readout.decision_inputs,
            diffusion_denoise_loss=diffusion_state.loss,
            diagnostic_probes=diagnostic_probes,
            trace_target_latent=diffusion_state.clean_latent,
        )
