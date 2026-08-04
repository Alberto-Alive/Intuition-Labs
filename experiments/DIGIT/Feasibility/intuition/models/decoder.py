"""Component 5: Decoder.

Architecture: small Transformer decoder.
Input:
  - encoded query representation z_q
  - discrete primitives embedded as categorical embeddings
Output:
  - bounded intuition text (max 2 sentences)

CONSTRAINT: The decoder has NO attention path to raw private data.
Only discrete approved primitives + query encoding may cross the bottleneck.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class IntuitionDecoder(nn.Module):
    """Generates bounded intuition text from primitives + query encoding.

    The decoder receives:
      1. A prefix sequence of embedded primitive tokens (4 tokens: answer,
         support, confidence, risk — each embedded from their discrete
         one-hot representation).
      2. The query encoding z_q projected as an additional context token.
    These form the "memory" that the decoder cross-attends to.

    The decoder then autoregressively generates output tokens.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_output_len: int = 48,
        num_answer: int = 6,
        num_support: int = 4,
        num_confidence: int = 3,
        num_risk: int = 3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_output_len = max_output_len
        self.pad_idx = pad_idx

        # Primitive embeddings: project one-hot discrete primitives to d_model
        self.answer_embed = nn.Linear(num_answer, d_model, bias=False)
        self.support_embed = nn.Linear(num_support, d_model, bias=False)
        self.confidence_embed = nn.Linear(num_confidence, d_model, bias=False)
        self.risk_embed = nn.Linear(num_risk, d_model, bias=False)

        # Query context projection
        self.query_proj = nn.Linear(d_model, d_model)

        # Token embeddings for decoder output
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embed = nn.Embedding(max_output_len, d_model)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def _build_memory(
        self,
        z_q: torch.Tensor,
        answer_disc: torch.Tensor,
        support_disc: torch.Tensor,
        confidence_disc: torch.Tensor,
        risk_disc: torch.Tensor,
    ) -> torch.Tensor:
        """Build the cross-attention memory from primitives + query.

        Returns: (B, 5, d_model) — 4 primitive tokens + 1 query token.
        """
        a = self.answer_embed(answer_disc)        # (B, d_model)
        s = self.support_embed(support_disc)
        c = self.confidence_embed(confidence_disc)
        r = self.risk_embed(risk_disc)
        q = self.query_proj(z_q)                   # (B, d_model)

        # Stack as sequence: (B, 5, d_model)
        memory = torch.stack([q, a, s, c, r], dim=1)
        return memory

    def forward(
        self,
        z_q: torch.Tensor,
        answer_disc: torch.Tensor,
        support_disc: torch.Tensor,
        confidence_disc: torch.Tensor,
        risk_disc: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced forward pass.

        Args:
            z_q: (B, d_model)
            *_disc: discrete one-hot primitives
            target_ids: (B, T) — target token ids (including BOS).

        Returns:
            logits: (B, T, vocab_size)
        """
        memory = self._build_memory(z_q, answer_disc, support_disc,
                                     confidence_disc, risk_disc)

        B, T = target_ids.shape
        positions = torch.arange(T, device=target_ids.device)
        x = self.token_embed(target_ids) + self.pos_embed(positions).unsqueeze(0)

        # Causal mask for autoregressive decoding
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=target_ids.device, dtype=x.dtype
        )

        # Padding mask (same dtype as causal_mask for compatibility)
        tgt_key_padding_mask = (target_ids == self.pad_idx)

        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        x = self.norm(x)
        logits = self.output_proj(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        z_q: torch.Tensor,
        answer_disc: torch.Tensor,
        support_disc: torch.Tensor,
        confidence_disc: torch.Tensor,
        risk_disc: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_len: int = 48,
    ) -> torch.Tensor:
        """Greedy autoregressive generation.

        Returns: (B, <=max_len) token ids.
        """
        memory = self._build_memory(z_q, answer_disc, support_disc,
                                     confidence_disc, risk_disc)
        B = z_q.size(0)
        device = z_q.device

        generated = torch.full((B, 1), bos_idx, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            T = generated.size(1)
            positions = torch.arange(T, device=device)
            x = self.token_embed(generated) + self.pos_embed(positions).unsqueeze(0)

            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                T, device=device
            )
            x = self.decoder(tgt=x, memory=memory, tgt_mask=causal_mask)
            x = self.norm(x)
            logits = self.output_proj(x[:, -1, :])  # (B, vocab)
            next_token = logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if all sequences have produced EOS
            if (next_token.squeeze(-1) == eos_idx).all():
                break

        return generated
