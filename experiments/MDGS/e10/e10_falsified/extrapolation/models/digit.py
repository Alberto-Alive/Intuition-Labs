"""Full DIGIT Extrapolation E10 model with cooperative diffusion evidence."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .bottleneck import DiffusionBottleneckHead
from .decoder import IntuitionDecoder
from .encoder import AdultQueryEncoder
from .executor import EntropyTrajectoryExecutor


class DIGITModel(nn.Module):
    """End-to-end DIGIT model with shared diffusion evidence and prototype geometry regularisation."""

    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab

        self.encoder = AdultQueryEncoder(
            field_vocab_sizes=config.field_vocab_sizes,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_encoder_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )

        self.executor = EntropyTrajectoryExecutor(
            min_group_size=config.min_group_size,
            income_margin=config.income_margin,
            low_agreement_threshold=config.trace_low_agreement_threshold,
            low_margin_threshold=config.trace_low_margin_threshold,
        )
        for param in self.executor.parameters():
            param.requires_grad = False

        self.bottleneck = DiffusionBottleneckHead(
            d_model=config.d_model,
            executor_dim=self.executor.output_dim,
            num_trajectory=config.num_trajectory_classes,
            num_pattern=config.num_pattern_classes,
            num_confidence=config.num_confidence_classes,
            num_outcome=config.num_outcome_classes,
            trace_hidden_dim=config.trace_head_hidden_dim,
            anchor_state_dim=config.anchor_state_dim,
            anchors_per_family=config.anchors_per_family,
            num_diffusion_steps=config.num_diffusion_steps,
            num_random_paths=config.num_random_paths,
            beta_start=config.diffusion_beta_start,
            beta_end=config.diffusion_beta_end,
            denoiser_hidden=config.diffusion_denoiser_hidden,
            trust_hidden=config.diffusion_trust_hidden,
            fragility_hidden=config.diffusion_fragility_hidden,
            confidence_threshold_init=config.monotone_confidence_threshold_init,
            confidence_threshold_gap_init=config.monotone_confidence_threshold_gap_init,
            monotone_scale_init=config.monotone_risk_scale_init,
            prototype_support_temperature=config.prototype_support_temperature,
            prototype_scale_init=config.prototype_scale_init,
            prototype_scale_max=config.prototype_scale_max,
            perturbation_rank_margin=config.perturbation_rank_margin,
            basin_order_margin=config.basin_order_margin,
            ordered_score_margin=config.ordered_score_margin,
            ordered_boundary_margin_scale=config.ordered_boundary_margin_scale,
            ordered_energy_margin=config.ordered_energy_margin,
            ordered_success_support_margin=config.ordered_success_support_margin,
            prototype_repulsion_margin=config.anchor_repulsion_margin,
            family_activity_weight=config.anchor_family_activity_weight,
            family_activity_ratio=config.anchor_family_activity_ratio,
            anchor_domination_weight=config.anchor_domination_weight,
            anchor_domination_max_share=config.anchor_domination_max_share,
            pattern_prior_weight=config.trace_pattern_prior_weight,
            outcome_evidence_scale=config.trace_outcome_evidence_scale,
            outcome_trajectory_scale=config.trace_outcome_trajectory_scale,
            outcome_pattern_scale=config.trace_outcome_pattern_scale,
            dropout=config.dropout,
            head_mode=getattr(config, "head_mode", "mlp"),
        )

        self.decoder = IntuitionDecoder(
            vocab_size=len(vocab),
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_decoder_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            max_output_len=config.max_output_len,
            num_trajectory=config.num_trajectory_classes,
            num_pattern=config.num_pattern_classes,
            num_confidence=config.num_confidence_classes,
            num_outcome=config.num_outcome_classes,
            pad_idx=vocab.pad_idx,
        )

    def set_trace_metadata(self, metadata: Dict | None) -> None:
        self.bottleneck.set_trace_metadata(metadata)

    def forward(
        self,
        query_fields: torch.Tensor,
        private_dataset,
        target_ids: torch.Tensor,
        bottleneck_mode: str = "gumbel",
        tau: float = 1.0,
        skip_decoder: bool = False,
        path_scales: Dict[str, float] | None = None,
        evidence_override: Dict[str, Any] | None = None,
        diffusion_override: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, Any]:
        z_q = self.encoder(query_fields)
        executor_features = self.executor(query_fields, private_dataset)
        use_query_features = not torch.is_tensor(private_dataset)

        primitives = self.bottleneck(
            z_q,
            executor_features,
            mode=bottleneck_mode,
            tau=tau,
            use_query_features=use_query_features,
            path_scales=path_scales,
            evidence_override=evidence_override,
            diffusion_override=diffusion_override,
        )

        decoder_logits = None
        if not skip_decoder:
            decoder_input = target_ids[:, :-1]
            decoder_logits = self.decoder(
                z_q=z_q,
                trajectory_shape_disc=primitives.trajectory_shape_discrete,
                attention_pattern_disc=primitives.attention_pattern_discrete,
                confidence_disc=primitives.confidence_discrete,
                outcome_disc=primitives.outcome_discrete,
                target_ids=decoder_input,
            )

        return {
            "decoder_logits": decoder_logits,
            "primitives": primitives,
            "executor_features": executor_features,
            "z_q": z_q,
            "diffusion_state": getattr(self.bottleneck, "last_diffusion_state", None),
        }

    @torch.no_grad()
    def generate(
        self,
        query_fields: torch.Tensor,
        private_dataset,
        path_scales: Dict[str, float] | None = None,
        evidence_override: Dict[str, Any] | None = None,
        diffusion_override: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, Any]:
        z_q = self.encoder(query_fields)
        executor_features = self.executor(query_fields, private_dataset)
        use_query_features = not torch.is_tensor(private_dataset)

        primitives = self.bottleneck(
            z_q,
            executor_features,
            mode="hard",
            tau=1.0,
            use_query_features=use_query_features,
            path_scales=path_scales,
            evidence_override=evidence_override,
            diffusion_override=diffusion_override,
        )

        token_ids = self.decoder.generate(
            z_q=z_q,
            trajectory_shape_disc=primitives.trajectory_shape_discrete,
            attention_pattern_disc=primitives.attention_pattern_discrete,
            confidence_disc=primitives.confidence_discrete,
            outcome_disc=primitives.outcome_discrete,
            bos_idx=self.vocab.bos_idx,
            eos_idx=self.vocab.eos_idx,
            max_len=self.config.max_output_len,
        )

        return {
            "token_ids": token_ids,
            "primitives": primitives,
            "executor_features": executor_features,
            "diffusion_state": getattr(self.bottleneck, "last_diffusion_state", None),
        }
