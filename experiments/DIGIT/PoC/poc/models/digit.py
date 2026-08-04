"""Full DIGIT Model — assembles encoder, executor, bottleneck, decoder.

Adapted for Adult Income: uses categorical query encoder and
real-data executor.
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from .encoder import AdultQueryEncoder
from .executor import AdultSafeExecutor
from .bottleneck import PrimitiveBottleneckHead, PrimitiveOutput
from .decoder import IntuitionDecoder


class DIGITModel(nn.Module):
    """End-to-end DIGIT model with discrete bottleneck."""

    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab

        # Component 1: Query Encoder (learned)
        self.encoder = AdultQueryEncoder(
            field_vocab_sizes=config.field_vocab_sizes,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_encoder_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )

        # Component 2: Safe Private Executor (deterministic, not learned)
        self.executor = AdultSafeExecutor(
            min_group_size=config.min_group_size,
            variance_buckets=config.variance_buckets,
            confidence_buckets=config.confidence_buckets,
        )
        for p in self.executor.parameters():
            p.requires_grad = False

        # Component 3: Primitive Bottleneck Head (learned)
        self.bottleneck = PrimitiveBottleneckHead(
            d_model=config.d_model,
            executor_dim=self.executor.output_dim,
            num_answer=config.num_answer_classes,
            num_support=config.num_support_classes,
            num_confidence=config.num_confidence_classes,
            num_risk=config.num_risk_classes,
        )

        # Component 4: Decoder (learned)
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
        private_dataset,
        target_ids: torch.Tensor,
        bottleneck_mode: str = "gumbel",
        tau: float = 1.0,
    ) -> Dict[str, Any]:
        # 1. Encode query
        z_q = self.encoder(query_fields)

        # 2. Execute safe aggregation (no gradients)
        executor_features = self.executor(query_fields, private_dataset)

        # 3. Produce discrete primitives
        primitives = self.bottleneck(z_q, executor_features,
                                      mode=bottleneck_mode, tau=tau)

        # 4. Decode — ONLY from primitives + z_q
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
    def generate(self, query_fields, private_dataset):
        """Generate intuition text (inference mode)."""
        z_q = self.encoder(query_fields)
        executor_features = self.executor(query_fields, private_dataset)
        primitives = self.bottleneck(z_q, executor_features,
                                      mode="hard", tau=1.0)

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
            "executor_features": executor_features,
        }
