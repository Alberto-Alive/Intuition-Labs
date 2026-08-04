"""Transformer Decoder — generates bounded intuition text.

Identical to the feasibility experiment. The decoder cross-attends to
5 memory tokens: one per primitive embedding + projected query.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntuitionDecoder(nn.Module):
    """Generates bounded intuition text from primitives + query encoding.

    CONSTRAINT: No attention path to raw private data.
    Only discrete primitives + query encoding cross the bottleneck.
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

        # Primitive embeddings
        self.answer_embed = nn.Linear(num_answer, d_model, bias=False)
        self.support_embed = nn.Linear(num_support, d_model, bias=False)
        self.confidence_embed = nn.Linear(num_confidence, d_model, bias=False)
        self.risk_embed = nn.Linear(num_risk, d_model, bias=False)

        # Query context projection
        self.query_proj = nn.Linear(d_model, d_model)

        # Token embeddings
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

    def _build_memory(self, z_q, answer_disc, support_disc,
                      confidence_disc, risk_disc):
        """Build cross-attention memory: (B, 5, d_model)."""
        a = self.answer_embed(answer_disc)
        s = self.support_embed(support_disc)
        c = self.confidence_embed(confidence_disc)
        r = self.risk_embed(risk_disc)
        q = self.query_proj(z_q)
        return torch.stack([q, a, s, c, r], dim=1)

    def forward(self, z_q, answer_disc, support_disc, confidence_disc,
                risk_disc, target_ids):
        """Teacher-forced forward pass.

        Returns: logits (B, T, vocab_size).
        """
        memory = self._build_memory(z_q, answer_disc, support_disc,
                                     confidence_disc, risk_disc)
        B, T = target_ids.shape
        positions = torch.arange(T, device=target_ids.device)
        x = self.token_embed(target_ids) + self.pos_embed(positions).unsqueeze(0)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=target_ids.device, dtype=x.dtype
        )
        tgt_key_padding_mask = (target_ids == self.pad_idx)

        x = self.decoder(tgt=x, memory=memory, tgt_mask=causal_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask)
        x = self.norm(x)
        return self.output_proj(x)

    @torch.no_grad()
    def generate(self, z_q, answer_disc, support_disc, confidence_disc,
                 risk_disc, bos_idx, eos_idx, max_len=48):
        """Greedy autoregressive generation."""
        memory = self._build_memory(z_q, answer_disc, support_disc,
                                     confidence_disc, risk_disc)
        B = z_q.size(0)
        device = z_q.device

        generated = torch.full((B, 1), bos_idx, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            T = generated.size(1)
            positions = torch.arange(T, device=device)
            x = self.token_embed(generated) + self.pos_embed(positions).unsqueeze(0)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
            x = self.decoder(tgt=x, memory=memory, tgt_mask=causal_mask)
            x = self.norm(x)
            logits = self.output_proj(x[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token.squeeze(-1) == eos_idx).all():
                break

        return generated
