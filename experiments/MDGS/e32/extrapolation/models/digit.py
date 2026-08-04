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
    bucketize_commitment_decisions,
    summarize_raw_witness_geometry,
)
from ...uncertainty_profile import (
    DisturbanceProfileResult,
    LatentDisturbanceProfile,
    ProfileAggregator,
    build_default_probes,
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
    clean_logits: torch.Tensor | None = None
    disturbance_profile: DisturbanceProfileResult | None = None
    risk_logit: torch.Tensor | None = None
    risk_prob: torch.Tensor | None = None
    calibrated_risk: torch.Tensor | None = None
    commit: torch.Tensor | None = None
    final_commitment_score: torch.Tensor | None = None
    final_commitment_logit: torch.Tensor | None = None
    final_commitment_threshold: torch.Tensor | None = None
    profile_regularization_loss: torch.Tensor | None = None


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
        self.disturbance_profile = None
        if getattr(config, "enable_disturbance_profile", False):
            probes = build_default_probes(
                latent_dim=config.d_model,
                gaussian_k=int(getattr(config, "disturbance_num_samples", 8)),
                mask_k=int(getattr(config, "disturbance_num_samples", 8)),
                boundary_k=int(getattr(config, "disturbance_num_samples", 8)),
                include_learned=bool(getattr(config, "enable_learned_probe", False)),
                enable_boundary_probe=bool(getattr(config, "enable_boundary_probe", True)),
                enable_feature_mask_probe=bool(getattr(config, "enable_feature_mask_probe", True)),
                gaussian_sigma=float(getattr(config, "disturbance_noise_std", 0.05)),
                mask_prob=float(getattr(config, "disturbance_feature_mask_prob", 0.15)),
                boundary_epsilon=float(getattr(config, "disturbance_boundary_epsilon", 0.10)),
                learned_hidden_dim=int(getattr(config, "disturbance_learned_hidden_dim", 128)),
            )
            self.disturbance_profile = LatentDisturbanceProfile(
                latent_dim=config.d_model,
                probes=probes,
                aggregator=ProfileAggregator(
                    hidden_dim=int(getattr(config, "disturbance_profile_hidden_dim", 128)),
                    dropout=float(getattr(config, "disturbance_profile_dropout", 0.1)),
                ),
                config=config,
            )
            # Learned probes are kept inactive until the staged schedule enables them.
            self.disturbance_profile.set_learned_probe_active(False)
        self._collect_disturbance_profile = True

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

    def _disturbance_context_features(
        self,
        stats: WitnessSetStats,
        decision_inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        witness_margins = stats.witness_margins
        dtype = witness_margins.dtype
        positive_fraction = (witness_margins > 0).to(dtype=dtype).mean(dim=-1)
        negative_fraction = (witness_margins < 0).to(dtype=dtype).mean(dim=-1)
        contradiction_fraction = 2.0 * torch.minimum(positive_fraction, negative_fraction)
        signed_balance_abs = (positive_fraction - negative_fraction).abs()
        num_witnesses = max(int(witness_margins.shape[-1]), 1)
        if num_witnesses > 1:
            raw_weight_entropy = stats.witness_weight_entropy * math.log(float(num_witnesses))
            support_effective_fraction = raw_weight_entropy.exp() / float(num_witnesses)
        else:
            support_effective_fraction = torch.ones_like(stats.witness_weight_entropy)

        return {
            "consensus/vote_disagreement": stats.vote_disagreement,
            "consensus/geometric_disagreement": stats.geometric_disagreement,
            "consensus/pairwise_agreement_mean": stats.pairwise_agreement_mean,
            "consensus/pairwise_agreement_std": stats.pairwise_agreement_std,
            "consensus/witness_margin_std": stats.witness_margin_std,
            "boundary/abs_margin": stats.margin_abs,
            "method/path_disagreement": decision_inputs["path_disagreement"],
            "method/path_confidence": decision_inputs["path_confidence"],
            "method/endpoint_confidence": decision_inputs["endpoint_confidence"],
            "method/method_conflict": decision_inputs["method_conflict"],
            "evidence/positive_fraction": positive_fraction,
            "evidence/negative_fraction": negative_fraction,
            "evidence/contradiction_fraction": contradiction_fraction,
            "evidence/signed_balance_abs": signed_balance_abs,
            "evidence/support_concentration": stats.witness_weight_max,
            "evidence/support_effective_fraction": support_effective_fraction,
            "evidence/witness_weight_kl_uniform": stats.witness_weight_kl_uniform,
        }

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

    def _set_requires_grad(self, enabled: bool) -> None:
        for parameter in self.parameters():
            try:
                parameter.requires_grad_(bool(enabled))
            except ValueError:
                # Skip lazy parameters until a forward pass materializes them.
                continue

    def set_training_stage(self, stage: str) -> None:
        stage = stage.strip().lower()
        if self.disturbance_profile is None:
            self._set_requires_grad(True)
            return
        if stage == "base":
            self._set_requires_grad(True)
            self.disturbance_profile.set_requires_grad(False)
            self.disturbance_profile.set_learned_probe_active(False)
            self._collect_disturbance_profile = False
            return
        if stage == "profile":
            self._set_requires_grad(False)
            self.disturbance_profile.set_requires_grad(True)
            self.disturbance_profile.set_learned_probe_active(False)
            self._collect_disturbance_profile = True
            return
        if stage == "finetune":
            self._set_requires_grad(True)
            self.disturbance_profile.set_requires_grad(True)
            self.disturbance_profile.set_learned_probe_active(
                bool(getattr(self.config, "enable_learned_probe", False))
            )
            self._collect_disturbance_profile = True
            return
        raise ValueError("stage must be one of {'base', 'profile', 'finetune'}")

    def lazy_module_status(self) -> dict[str, list[str]]:
        """Report any lazy parameters or buffers that still need materialization."""

        status = {
            "uninitialized_parameters": [],
            "uninitialized_buffers": [],
        }
        for name, parameter in self.named_parameters(recurse=True):
            if parameter.__class__.__name__.startswith("Uninitialized"):
                status["uninitialized_parameters"].append(name)
        for name, buffer in self.named_buffers(recurse=True):
            if buffer.__class__.__name__.startswith("Uninitialized"):
                status["uninitialized_buffers"].append(name)
        return status

    def materialize_lazy_modules(
        self,
        *,
        device: torch.device | None = None,
        sample_batch_size: int = 1,
    ) -> dict[str, list[str]]:
        """
        Force a dummy forward pass so lazy modules are materialized before checkpointing.

        The disturbance-profile aggregator is the only lazy submodule in e32, but this helper
        is defensive: it reports all uninitialized parameters and buffers after the warm-up.
        """

        status_before = self.lazy_module_status()
        if not status_before["uninitialized_parameters"] and not status_before["uninitialized_buffers"]:
            return status_before

        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        previous_training = self.training
        previous_collect = self._collect_disturbance_profile
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = None
        if device.type == "cuda" and torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state_all()

        try:
            self.eval()
            self._collect_disturbance_profile = True
            dummy_queries = torch.zeros(
                sample_batch_size,
                self.config.num_query_fields,
                dtype=torch.long,
                device=device,
            )
            with torch.no_grad():
                _ = self(
                    dummy_queries,
                    trace_inputs=None,
                    num_cloud_samples=min(2, max(1, int(self.config.num_cloud_samples))),
                )
        finally:
            self._collect_disturbance_profile = previous_collect
            self.train(previous_training)
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)

        return self.lazy_module_status()

    def encode(self, query_fields: torch.Tensor) -> torch.Tensor:
        return self.encoder(query_fields)

    def classify_latent(
        self,
        z: torch.Tensor,
        *,
        num_cloud_samples: int | None = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        sampling_noise = None
        if deterministic:
            num_samples = int(num_cloud_samples or self.config.num_cloud_samples)
            sampling_noise = torch.zeros(
                (num_samples, z.shape[0], z.shape[1]),
                device=z.device,
                dtype=z.dtype,
            )
        previous_flag = self._collect_disturbance_profile
        self._collect_disturbance_profile = False
        try:
            result = self.forward_from_anchor(
                z,
                trace_inputs=None,
                num_cloud_samples=num_cloud_samples,
                sampling_noise=sampling_noise,
                collect_disturbance_profile=False,
            )
        finally:
            self._collect_disturbance_profile = previous_flag
        return result.outcome_logits

    def forward_from_anchor(
        self,
        anchor: torch.Tensor,
        trace_inputs: torch.Tensor | None = None,
        *,
        num_cloud_samples: int | None = None,
        sampling_noise: torch.Tensor | None = None,
        diffusion_training_noise: torch.Tensor | None = None,
        diffusion_training_timesteps: torch.Tensor | None = None,
        collect_disturbance_profile: bool = True,
    ) -> E31ForwardResult:
        previous_flag = self._collect_disturbance_profile
        previous_encoder_forward = self.encoder.forward
        self._collect_disturbance_profile = bool(collect_disturbance_profile)
        self.encoder.forward = lambda _query_fields: anchor
        try:
            dummy_query_fields = torch.zeros(
                anchor.shape[0],
                self.config.num_query_fields,
                dtype=torch.long,
                device=anchor.device,
            )
            return self.forward(
                dummy_query_fields,
                trace_inputs=trace_inputs,
                num_cloud_samples=num_cloud_samples,
                sampling_noise=sampling_noise,
                diffusion_training_noise=diffusion_training_noise,
                diffusion_training_timesteps=diffusion_training_timesteps,
            )
        finally:
            self.encoder.forward = previous_encoder_forward
            self._collect_disturbance_profile = previous_flag

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
        decision_inputs["base_commitment_threshold"] = readout.decision_inputs["tau_uncertain"]

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

        disturbance_profile_result = None
        risk_logit = None
        risk_prob = None
        calibrated_risk = None
        risk_adjustment = None
        commit = None
        profile_regularization_loss = None
        if self._collect_disturbance_profile and self.disturbance_profile is not None:
            disturbance_profile_result = self.disturbance_profile(
                model=self,
                clean_z=anchor,
                clean_logits=readout.outcome_logits,
                context_features=self._disturbance_context_features(
                    witness_statistics,
                    decision_inputs,
                ),
            )
            risk_logit = disturbance_profile_result.risk_logit
            risk_prob = disturbance_profile_result.risk_prob
            calibrated_risk = disturbance_profile_result.calibrated_risk
            profile_regularization_loss = disturbance_profile_result.regularization_loss

        final_commitment_score = decision_inputs["commitment_score"]
        final_commitment_threshold = decision_inputs["tau_uncertain"]
        if calibrated_risk is not None and bool(
            getattr(self.config, "use_disturbance_risk_in_commitment_gate", True)
        ):
            lambda_profile_risk = final_commitment_score.new_full(
                final_commitment_score.shape,
                float(getattr(self.config, "lambda_profile_risk", 0.0)),
            )
            risk_adjustment = lambda_profile_risk * calibrated_risk
            final_commitment_score = final_commitment_score - risk_adjustment
            final_commitment_threshold = final_commitment_score.new_full(
                final_commitment_score.shape,
                float(getattr(self.config, "commit_risk_threshold", 0.5)),
            )
            decision_inputs["lambda_profile_risk"] = lambda_profile_risk
            decision_inputs["profile_risk_temperature"] = final_commitment_score.new_full(
                final_commitment_score.shape,
                float(getattr(self.config, "profile_risk_temperature", 1.0)),
            )
            decision_inputs["calibrated_risk"] = calibrated_risk
            decision_inputs["risk_adjustment"] = risk_adjustment
            decision_inputs["final_commit_risk_threshold"] = final_commitment_threshold
        else:
            if self.disturbance_profile is not None:
                decision_inputs["lambda_profile_risk"] = final_commitment_score.new_full(
                    final_commitment_score.shape,
                    float(getattr(self.config, "lambda_profile_risk", 0.0)),
                )
                decision_inputs["profile_risk_temperature"] = final_commitment_score.new_full(
                    final_commitment_score.shape,
                    float(getattr(self.config, "profile_risk_temperature", 1.0)),
                )
                if calibrated_risk is not None:
                    decision_inputs["calibrated_risk"] = calibrated_risk
                    decision_inputs["risk_adjustment"] = final_commitment_score.new_zeros(
                        final_commitment_score.shape
                    )
                    decision_inputs["final_commit_risk_threshold"] = final_commitment_threshold

        decision_inputs["lambda_profile_risk"] = final_commitment_score.new_full(
            final_commitment_score.shape,
            float(getattr(self.config, "lambda_profile_risk", 0.0)),
        )
        decision_inputs["profile_risk_temperature"] = final_commitment_score.new_full(
            final_commitment_score.shape,
            float(getattr(self.config, "profile_risk_temperature", 1.0)),
        )
        if calibrated_risk is None:
            calibrated_risk = final_commitment_score.new_zeros(final_commitment_score.shape)
        if risk_adjustment is None:
            risk_adjustment = final_commitment_score.new_zeros(final_commitment_score.shape)
        decision_inputs["calibrated_risk"] = calibrated_risk
        decision_inputs["risk_adjustment"] = risk_adjustment

        final_commitment_logit = decision_inputs["commitment_scale"] * (
            final_commitment_score - final_commitment_threshold
        )
        decision_inputs["commitment_score"] = final_commitment_score
        decision_inputs["commitment_logit"] = final_commitment_logit
        decision_inputs["final_commitment_score"] = final_commitment_score
        decision_inputs["final_commitment_logit"] = final_commitment_logit
        decision_inputs["final_commitment_threshold"] = final_commitment_threshold
        decision_inputs["final_commit_risk_threshold"] = final_commitment_threshold
        decision_inputs["commit_risk_threshold"] = final_commitment_threshold
        commit = bucketize_commitment_decisions(
            final_margin,
            final_commitment_score,
            final_commitment_threshold,
        )

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
            clean_logits=readout.outcome_logits,
            disturbance_profile=disturbance_profile_result,
            risk_logit=risk_logit,
            risk_prob=risk_prob,
            calibrated_risk=calibrated_risk,
            commit=commit,
            final_commitment_score=final_commitment_score,
            final_commitment_logit=final_commitment_logit,
            final_commitment_threshold=final_commitment_threshold,
            profile_regularization_loss=profile_regularization_loss,
        )


E30ForwardResult = E31ForwardResult
E30Model = E31Model
E32ForwardResult = E31ForwardResult
E32Model = E31Model
