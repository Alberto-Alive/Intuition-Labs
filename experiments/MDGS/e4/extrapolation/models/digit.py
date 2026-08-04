"""Full DIGIT Extrapolation model."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .bottleneck import EntropyBottleneckHead
from .decoder import IntuitionDecoder
from .encoder import AdultQueryEncoder
from .executor import EntropyTrajectoryExecutor


class DIGITModel(nn.Module):
    """End-to-end DIGIT model with entropy-trajectory bottleneck."""

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

        self.bottleneck = EntropyBottleneckHead(
            d_model=config.d_model,
            executor_dim=self.executor.output_dim,
            num_trajectory=config.num_trajectory_classes,
            num_pattern=config.num_pattern_classes,
            num_confidence=config.num_confidence_classes,
            num_outcome=config.num_outcome_classes,
            trace_hidden_dim=config.trace_head_hidden_dim,
            monotone_hidden_dim=config.uncertainty_tower_dim,
            anchor_state_dim=config.anchor_state_dim,
            anchors_per_family=config.anchors_per_family,
            anchor_support_temperature=config.anchor_support_temperature,
            anchor_assignment_mode=config.anchor_assignment_mode,
            anchor_use_approach=config.anchor_use_approach,
            anchor_use_leap=config.anchor_use_leap,
            anchor_use_boundary_family=config.anchor_use_boundary_family,
            anchor_use_barrier=config.anchor_use_barrier,
            success_min_support_ratio=config.trace_success_min_support_ratio,
            success_min_group_size_norm=min(config.trace_success_min_group_size / 200.0, 1.0),
            confidence_threshold_init=config.monotone_confidence_threshold_init,
            confidence_threshold_gap_init=config.monotone_confidence_threshold_gap_init,
            outcome_threshold_init=config.monotone_outcome_threshold_init,
            outcome_threshold_gap_init=config.monotone_outcome_threshold_gap_init,
            monotone_scale_init=config.monotone_risk_scale_init,
            dropout=config.dropout,
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
        self.default_path_scales = {
            "shared_trace": float(config.trace_path_scale_shared_trace),
            "success_family": float(config.trace_path_scale_success_trace),
            "failure_family": float(config.trace_path_scale_failure_trace),
            "boundary_family": float(config.trace_path_scale_ambiguity_trace),
            "approach": 1.0 if config.anchor_use_approach else 0.0,
            "leap": 1.0 if config.anchor_use_leap else 0.0,
            "barrier": 1.0 if config.anchor_use_barrier else 0.0,
        }

    def _merge_path_scales(self, path_scales: Dict[str, float] | None = None) -> Dict[str, float]:
        merged = dict(self.default_path_scales)
        if path_scales:
            for key, value in path_scales.items():
                merged[key] = float(value)
        return merged

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
    ) -> Dict[str, Any]:
        z_q = self.encoder(query_fields)
        executor_features = self.executor(query_fields, private_dataset)
        use_query_features = not torch.is_tensor(private_dataset)
        merged_path_scales = self._merge_path_scales(path_scales)
        primitives = self.bottleneck(
            z_q,
            executor_features,
            mode=bottleneck_mode,
            tau=tau,
            use_query_features=use_query_features,
            path_scales=merged_path_scales,
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
        }

    @torch.no_grad()
    def generate(
        self,
        query_fields: torch.Tensor,
        private_dataset,
        path_scales: Dict[str, float] | None = None,
    ) -> Dict[str, Any]:
        z_q = self.encoder(query_fields)
        executor_features = self.executor(query_fields, private_dataset)
        use_query_features = not torch.is_tensor(private_dataset)
        merged_path_scales = self._merge_path_scales(path_scales)
        primitives = self.bottleneck(
            z_q,
            executor_features,
            mode="hard",
            tau=1.0,
            use_query_features=use_query_features,
            path_scales=merged_path_scales,
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
        }
