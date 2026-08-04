"""Full Intuition Model — System C (main model).

Assembles:
  1. Learned Query Encoder
  2. Deterministic Safe Private Executor
  3. Learned Primitive Bottleneck Head
  4. Learned Decoder

Enforces the information bottleneck constraint:
  - Decoder receives ONLY discrete primitives + query encoding.
  - No path from decoder to raw private data or unrestricted aggregates.
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from .query_encoder import QueryEncoder
from .safe_executor import SafePrivateExecutor
from .bottleneck import PrimitiveBottleneckHead, PrimitiveOutput
from .decoder import IntuitionDecoder


class IntuitionModel(nn.Module):
    """End-to-end intuition model with discrete bottleneck."""

    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab

        # Component 1: Query Encoder (learned)
        self.encoder = QueryEncoder(
            num_query_fields=config.num_query_fields,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_encoder_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            max_query_len=config.max_query_len,
        )

        # Component 2: Safe Private Executor (deterministic, not learned)
        self.executor = SafePrivateExecutor(
            min_group_size=config.min_group_size,
            variance_buckets=config.variance_buckets,
            confidence_buckets=config.confidence_buckets,
        )
        # Freeze executor — it has no learnable parameters anyway,
        # but this makes the constraint explicit.
        for p in self.executor.parameters():
            p.requires_grad = False

        # Component 3+4: Primitive Bottleneck Head (learned)
        self.bottleneck = PrimitiveBottleneckHead(
            d_model=config.d_model,
            executor_dim=self.executor.output_dim,
            num_answer=config.num_answer_classes,
            num_support=config.num_support_classes,
            num_confidence=config.num_confidence_classes,
            num_risk=config.num_risk_classes,
        )

        # Component 5: Decoder (learned)
        self.decoder = IntuitionDecoder(
            vocab_size=len(vocab),
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_decoder_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            max_output_len=config.max_output_len,
            num_answer=config.num_answer_classes,
            num_support=config.num_support_classes,
            num_confidence=config.num_confidence_classes,
            num_risk=config.num_risk_classes,
            pad_idx=vocab.pad_idx,
        )

    def forward(
        self,
        query_fields: torch.Tensor,
        group_ids: torch.Tensor,
        private_dataset,
        target_ids: torch.Tensor,
        bottleneck_mode: str = "gumbel",
        tau: float = 1.0,
    ) -> Dict[str, Any]:
        """Full forward pass.

        Returns dict with:
          - decoder_logits: (B, T, V) — token generation logits
          - primitives: PrimitiveOutput
          - executor_features: (B, executor_dim) — safe aggregates
          - z_q: (B, d_model) — query representation
        """
        # 1. Encode query
        z_q = self.encoder(query_fields)

        # 2. Execute safe aggregation (no gradients)
        executor_features = self.executor(query_fields, group_ids, private_dataset)

        # 3. Produce discrete primitives through bottleneck
        primitives = self.bottleneck(
            z_q, executor_features, mode=bottleneck_mode, tau=tau
        )

        # 4. Decode — ONLY from primitives + z_q (bottleneck enforced)
        # Shift target for teacher forcing: input is target[:-1], predict target[1:]
        decoder_input = target_ids[:, :-1]
        decoder_logits = self.decoder(
            z_q=z_q,
            answer_disc=primitives.answer_discrete,
            support_disc=primitives.support_discrete,
            confidence_disc=primitives.confidence_discrete,
            risk_disc=primitives.risk_discrete,
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
        group_ids: torch.Tensor,
        private_dataset,
    ) -> Dict[str, Any]:
        """Generate intuition text (inference mode)."""
        z_q = self.encoder(query_fields)
        executor_features = self.executor(query_fields, group_ids, private_dataset)
        primitives = self.bottleneck(
            z_q, executor_features, mode="hard", tau=1.0
        )

        token_ids = self.decoder.generate(
            z_q=z_q,
            answer_disc=primitives.answer_discrete,
            support_disc=primitives.support_discrete,
            confidence_disc=primitives.confidence_discrete,
            risk_disc=primitives.risk_discrete,
            bos_idx=self.vocab.bos_idx,
            eos_idx=self.vocab.eos_idx,
            max_len=self.config.max_output_len,
        )

        return {
            "token_ids": token_ids,
            "primitives": primitives,
        }
