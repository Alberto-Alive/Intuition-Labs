"""Cooperative witness/path-agreement model for E31."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion import ConditionalLatentDiffusion
from .encoder import QueryAnchorEncoder
from .point_reader import build_point_reader
from .path_uncertainty import LatentPathUncertainty, PathUncertaintyResult
from .readout import (
    DetachedProbeHeads,
    RawWitnessGeometry,
    WitnessDiagnosticProbes,
    WitnessAgreementReadout,
    WitnessReadoutResult,
    WitnessSetStats,
    summarize_raw_witness_geometry,
)


def _inverse_softplus_local(value: float) -> float:
    value = max(float(value), 1e-6)
    return math.log(math.expm1(value))


def _torch_logit_clamped(values: torch.Tensor) -> torch.Tensor:
    return torch.logit(values.clamp(1e-4, 1.0 - 1e-4))


@dataclass
class E31ForwardResult:
    """Forward outputs for the E31 model."""

    anchor: torch.Tensor
    witness_samples: torch.Tensor
    point_reader_output: torch.Tensor
    pre_reader_witness_geometry: RawWitnessGeometry | None
    witness_statistics: WitnessSetStats
    margin: torch.Tensor
    uncertainty: torch.Tensor
    base_outcome_logits: torch.Tensor
    outcome_logits: torch.Tensor
    path_outcome_logits: torch.Tensor | None
    path_failure_logit: torch.Tensor | None
    decision_inputs: dict[str, torch.Tensor]
    diffusion_denoise_loss: torch.Tensor
    path_consistency_loss: torch.Tensor
    path_uncertainty: PathUncertaintyResult | None
    diagnostic_probes: WitnessDiagnosticProbes | None
    trace_target_latent: torch.Tensor | None


class E31Model(nn.Module):
    """Encoder anchor + conditional latent diffusion + cooperative uncertainty head."""

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
        self.path_uncertainty = (
            LatentPathUncertainty(
                latent_dim=config.d_model,
                embedding_dim=config.path_uncertainty_embedding_dim,
                hidden_dim=config.path_uncertainty_hidden_dim,
                dropout=config.path_uncertainty_dropout,
                contrastive_temperature=config.path_contrastive_temperature,
            )
            if config.enable_path_uncertainty
            else None
        )
        self.use_path_disagreement_in_uncertainty = (
            config.use_path_disagreement_in_uncertainty or config.enable_path_signed_evidence
        )
        self.use_method_conflict_in_uncertainty = (
            config.use_method_conflict_in_uncertainty or config.enable_path_signed_evidence
        )
        self.use_path_commitment_penalty = (
            config.use_path_commitment_penalty or config.enable_path_signed_evidence
        )
        self.use_path_aware_witness_weights = (
            config.use_path_aware_witness_weights or config.enable_path_signed_evidence
        )
        self.path_signed_feature_dim = config.path_uncertainty_embedding_dim * 3 + 1
        self.path_outcome_head = None
        self.path_failure_gate_head = None
        self.raw_path_signed_logit_scale = None
        self.raw_path_signed_margin_scale = None
        if config.enable_path_uncertainty and config.enable_path_signed_evidence:
            self.path_outcome_head = nn.Sequential(
                nn.LayerNorm(self.path_signed_feature_dim),
                nn.Linear(
                    self.path_signed_feature_dim,
                    config.path_signed_evidence_hidden_dim,
                ),
                nn.GELU(),
                nn.Dropout(config.path_signed_evidence_dropout),
                nn.Linear(config.path_signed_evidence_hidden_dim, 3),
            )
            self.path_failure_gate_head = nn.Sequential(
                nn.LayerNorm(self.path_signed_feature_dim),
                nn.Linear(
                    self.path_signed_feature_dim,
                    config.path_signed_evidence_hidden_dim,
                ),
                nn.GELU(),
                nn.Dropout(config.path_signed_evidence_dropout),
                nn.Linear(config.path_signed_evidence_hidden_dim, 1),
            )
            self.raw_path_signed_logit_scale = nn.Parameter(
                torch.tensor(
                    _inverse_softplus_local(config.path_signed_logit_scale_init),
                    dtype=torch.float32,
                )
            )
            self.raw_path_signed_margin_scale = nn.Parameter(
                torch.tensor(
                    _inverse_softplus_local(config.path_signed_margin_scale_init),
                    dtype=torch.float32,
                )
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
            use_path_disagreement=self.use_path_disagreement_in_uncertainty,
            use_method_conflict=self.use_method_conflict_in_uncertainty,
            use_path_commitment_penalty=self.use_path_commitment_penalty,
            use_path_aware_witness_weights=self.use_path_aware_witness_weights,
            path_delta_init=config.path_uncertainty_delta_init,
            method_rho_init=config.method_conflict_rho_init,
            path_commitment_lambda_init=config.path_commitment_lambda_init,
            path_weight_eta_init=config.path_weight_eta_init,
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

    def _path_signed_features(self, path_result: PathUncertaintyResult) -> torch.Tensor:
        path_embedding_a = path_result.example_embedding_a
        path_embedding_b = path_result.example_embedding_b
        path_disagreement = path_result.path_disagreement.unsqueeze(-1)
        return torch.cat(
            [
                path_embedding_a,
                path_embedding_b,
                (path_embedding_a - path_embedding_b).abs(),
                path_disagreement,
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
    ) -> E31ForwardResult:
        anchor = self.encoder(query_fields)
        diffusion_state = self.diffusion.training_loss(
            anchor,
            trace_inputs,
            noise=diffusion_training_noise,
            timesteps=diffusion_training_timesteps,
        )
        num_samples = int(num_cloud_samples or self.config.num_cloud_samples)
        path_result = None
        path_consistency_loss = anchor.new_zeros(())
        path_failure_logit = None
        if self.path_uncertainty is not None:
            trajectory_a = self.diffusion.sample_trajectory(
                anchor,
                num_samples=num_samples,
                noise=sampling_noise,
                include_initial=True,
            )
            trajectory_b = self.diffusion.sample_trajectory(
                anchor,
                num_samples=num_samples,
                include_initial=True,
            )
            witness_samples = trajectory_a[-1]
            path_result = self.path_uncertainty(trajectory_a.detach(), trajectory_b.detach())
            path_consistency_loss = path_result.contrastive_loss
            path_disagreement = path_result.path_disagreement.detach()
            per_witness_path_disagreement = path_result.per_witness_path_disagreement.detach()
        else:
            witness_samples = self.diffusion.sample(
                anchor,
                num_samples=num_samples,
                noise=sampling_noise,
            )
            path_disagreement = None
            per_witness_path_disagreement = None
        pre_reader_witness_geometry = (
            summarize_raw_witness_geometry(anchor, witness_samples)
            if self.config.log_pre_reader_geometry
            else None
        )
        point_reader_output = self.point_reader(witness_samples)
        witness_statistics = self.readout.summarize(
            witness_samples,
            margin_samples=point_reader_output,
            per_witness_path_disagreement=per_witness_path_disagreement,
            path_disagreement=path_disagreement,
        )
        readout: WitnessReadoutResult = self.readout(witness_statistics)
        decision_inputs = dict(readout.decision_inputs)
        decision_inputs["base_global_margin"] = readout.decision_inputs["global_margin"]
        decision_inputs["base_commitment_score"] = readout.decision_inputs["commitment_score"]
        decision_inputs["base_commitment_logit"] = readout.decision_inputs["commitment_logit"]

        final_margin = readout.margin
        final_outcome_logits = readout.outcome_logits
        path_outcome_logits = None
        if (
            path_result is not None
            and self.path_outcome_head is not None
            and self.path_failure_gate_head is not None
        ):
            path_features = self._path_signed_features(path_result)
            path_outcome_logits = self.path_outcome_head(path_features)
            path_failure_logit = self.path_failure_gate_head(path_features).squeeze(-1)
            path_failure_probability = torch.sigmoid(path_failure_logit)
            path_failure_threshold = final_margin.new_full(
                final_margin.shape,
                float(self.config.path_failure_gate_threshold),
            )
            path_failure_gate_signal = torch.sigmoid(
                path_failure_logit - _torch_logit_clamped(path_failure_threshold)
            )
            path_logit_scale = F.softplus(self.raw_path_signed_logit_scale)
            path_margin_scale = F.softplus(self.raw_path_signed_margin_scale)
            final_outcome_logits = readout.outcome_logits.clone()
            final_outcome_logits[:, 2] = (
                final_outcome_logits[:, 2] + path_logit_scale * path_failure_gate_signal
            )

            if self.config.use_path_signed_margin_in_hard_gate:
                final_margin = readout.margin - path_margin_scale * path_failure_gate_signal

            lambda_commit = decision_inputs["lambda_commit"]
            lambda_path = decision_inputs["lambda_path"]
            path_disagreement_for_gate = decision_inputs["path_disagreement"]
            final_commitment_score = (
                final_margin.abs()
                - lambda_commit * readout.uncertainty
                - lambda_path * path_disagreement_for_gate
            )
            final_commitment_logit = decision_inputs["commitment_scale"] * (
                final_commitment_score - decision_inputs["tau_uncertain"]
            )

            decision_inputs["global_margin"] = final_margin
            decision_inputs["commitment_score"] = final_commitment_score
            decision_inputs["commitment_logit"] = final_commitment_logit
            decision_inputs["margin_minus_lambda_u"] = (
                final_margin - lambda_commit * readout.uncertainty
            )
            decision_inputs["path_signed_margin"] = path_failure_gate_signal
            decision_inputs["path_signed_margin_abs"] = path_failure_gate_signal.abs()
            decision_inputs["path_signed_logit_scale"] = path_logit_scale.expand_as(final_margin)
            decision_inputs["path_signed_margin_scale"] = path_margin_scale.expand_as(final_margin)
            decision_inputs["path_signed_margin_delta"] = final_margin - readout.margin
            decision_inputs["path_failure_gate_logit"] = path_failure_logit
            decision_inputs["path_failure_gate_probability"] = path_failure_probability
            decision_inputs["path_failure_gate_signal"] = path_failure_gate_signal
            decision_inputs["path_failure_gate_threshold"] = path_failure_threshold
            decision_inputs["path_success_logit"] = path_outcome_logits[:, 0]
            decision_inputs["path_uncertain_logit"] = path_outcome_logits[:, 1]
            decision_inputs["path_failure_logit"] = path_outcome_logits[:, 2]
            decision_inputs["path_success_minus_uncertain"] = (
                path_outcome_logits[:, 0] - path_outcome_logits[:, 1]
            )
            decision_inputs["path_failure_minus_uncertain"] = (
                path_outcome_logits[:, 2] - path_outcome_logits[:, 1]
            )

        diagnostic_probes = None
        if self.probe_heads is not None:
            probe_features = self._probe_features(witness_statistics).detach()
            diagnostic_probes = self.probe_heads(probe_features)

        return E31ForwardResult(
            anchor=anchor,
            witness_samples=witness_samples,
            point_reader_output=point_reader_output,
            pre_reader_witness_geometry=pre_reader_witness_geometry,
            witness_statistics=witness_statistics,
            margin=final_margin,
            uncertainty=readout.uncertainty,
            base_outcome_logits=readout.outcome_logits,
            outcome_logits=final_outcome_logits,
            path_outcome_logits=path_outcome_logits,
            path_failure_logit=path_failure_logit,
            decision_inputs=decision_inputs,
            diffusion_denoise_loss=diffusion_state.loss,
            path_consistency_loss=path_consistency_loss,
            path_uncertainty=path_result,
            diagnostic_probes=diagnostic_probes,
            trace_target_latent=diffusion_state.clean_latent,
        )


E30ForwardResult = E31ForwardResult
E30Model = E31Model
