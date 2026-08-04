"""Strict witness-agreement model for E30."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .diffusion import ConditionalLatentDiffusion
from .encoder import QueryAnchorEncoder
from .point_reader import build_point_reader
from .readout import (
    DetachedProbeHeads,
    RawWitnessGeometry,
    WitnessDiagnosticProbes,
    WitnessAgreementReadout,
    WitnessReadoutResult,
    WitnessSetStats,
    summarize_raw_witness_geometry,
)


@dataclass
class E30ForwardResult:
    """Forward outputs for the E30 model."""

    anchor: torch.Tensor
    witness_samples: torch.Tensor
    point_reader_output: torch.Tensor
    pre_reader_witness_geometry: RawWitnessGeometry | None
    witness_statistics: WitnessSetStats
    margin: torch.Tensor
    uncertainty: torch.Tensor
    outcome_logits: torch.Tensor
    decision_inputs: dict[str, torch.Tensor]
    diffusion_denoise_loss: torch.Tensor
    diagnostic_probes: WitnessDiagnosticProbes | None
    trace_target_latent: torch.Tensor | None


class E30Model(nn.Module):
    """Encoder anchor + conditional latent diffusion + witness-agreement decision head."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.encoder = QueryAnchorEncoder(
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
            target_hidden_dim=config.diffusion_target_hidden_dim,
            beta_start=config.diffusion_beta_start,
            beta_end=config.diffusion_beta_end,
            dropout=config.diffusion_dropout,
        )
        self.point_reader = build_point_reader(
            config.point_reader_mode,
            latent_dim=config.d_model,
            hidden_dim=config.point_reader_hidden_dim,
            dropout=config.point_reader_dropout,
        )
        self.readout = WitnessAgreementReadout(
            latent_dim=config.d_model,
            witness_weight_temperature=config.witness_weight_temperature,
            use_vote_disagreement=config.use_vote_disagreement,
            use_geometric_disagreement=config.use_geometric_disagreement,
            use_margin_term=config.use_margin_term,
            use_coherence_weighting=config.use_coherence_weighting,
            gate_mode=config.gate_mode,
            commitment_scale_init=config.commitment_scale_init,
            tau_uncertain_init=config.tau_uncertain_init,
            alpha_init=config.uncertainty_alpha_init,
            beta_init=config.uncertainty_beta_init,
            gamma_init=config.uncertainty_gamma_init,
            bias_init=config.uncertainty_bias_init,
        )
        self.probe_heads = (
            DetachedProbeHeads(feature_dim=9) if config.enable_diagnostic_probes else None
        )

    def _probe_features(self, stats: WitnessSetStats) -> torch.Tensor:
        return torch.stack(
            [
                stats.global_margin,
                stats.vote_disagreement,
                stats.geometric_disagreement,
                stats.margin_abs,
                stats.witness_weight_entropy,
                stats.witness_weight_kl_uniform,
                stats.witness_weight_max,
                stats.pairwise_agreement_mean,
                stats.pairwise_agreement_std,
            ],
            dim=-1,
        )

    def forward(
        self,
        query_fields: torch.Tensor,
        trace_inputs: torch.Tensor | None = None,
        *,
        num_cloud_samples: int | None = None,
        sampling_noise: torch.Tensor | None = None,
        diffusion_training_noise: torch.Tensor | None = None,
        diffusion_training_timesteps: torch.Tensor | None = None,
    ) -> E30ForwardResult:
        anchor = self.encoder(query_fields)
        diffusion_state = self.diffusion.training_loss(
            anchor,
            trace_inputs,
            noise=diffusion_training_noise,
            timesteps=diffusion_training_timesteps,
        )
        witness_samples = self.diffusion.sample(
            anchor,
            num_samples=int(num_cloud_samples or self.config.num_cloud_samples),
            noise=sampling_noise,
        )
        pre_reader_witness_geometry = (
            summarize_raw_witness_geometry(anchor, witness_samples)
            if self.config.log_pre_reader_geometry
            else None
        )
        point_reader_output = self.point_reader(witness_samples)
        witness_statistics = self.readout.summarize(
            witness_samples,
            margin_samples=point_reader_output,
        )
        readout: WitnessReadoutResult = self.readout(witness_statistics)

        diagnostic_probes = None
        if self.probe_heads is not None:
            probe_features = self._probe_features(witness_statistics).detach()
            diagnostic_probes = self.probe_heads(probe_features)

        return E30ForwardResult(
            anchor=anchor,
            witness_samples=witness_samples,
            point_reader_output=point_reader_output,
            pre_reader_witness_geometry=pre_reader_witness_geometry,
            witness_statistics=witness_statistics,
            margin=readout.margin,
            uncertainty=readout.uncertainty,
            outcome_logits=readout.outcome_logits,
            decision_inputs=readout.decision_inputs,
            diffusion_denoise_loss=diffusion_state.loss,
            diagnostic_probes=diagnostic_probes,
            trace_target_latent=diffusion_state.clean_latent,
        )
