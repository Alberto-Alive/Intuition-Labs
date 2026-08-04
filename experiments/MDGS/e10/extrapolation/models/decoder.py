"""Transformer decoder for entropy-gating text generation."""

from __future__ import annotations

import torch
import torch.nn as nn


class IntuitionDecoder(nn.Module):
    """Generates bounded entropy-language from discrete heads + query encoding."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_output_len: int = 48,
        num_trajectory: int = 4,
        num_pattern: int = 3,
        num_confidence: int = 3,
        num_outcome: int = 3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.max_output_len = max_output_len
        self.pad_idx = pad_idx

        self.trajectory_embed = nn.Linear(num_trajectory, d_model, bias=False)
        self.pattern_embed = nn.Linear(num_pattern, d_model, bias=False)
        self.confidence_embed = nn.Linear(num_confidence, d_model, bias=False)
        self.outcome_embed = nn.Linear(num_outcome, d_model, bias=False)
        self.query_proj = nn.Linear(d_model, d_model)

        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embed = nn.Embedding(max_output_len, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, target_len: int, device: torch.device) -> torch.Tensor:
        """Build a boolean causal mask compatible with boolean padding masks."""
        return torch.triu(
            torch.ones((target_len, target_len), device=device, dtype=torch.bool),
            diagonal=1,
        )

    def _build_memory(
        self,
        z_q: torch.Tensor,
        trajectory_shape_disc: torch.Tensor,
        attention_pattern_disc: torch.Tensor,
        confidence_disc: torch.Tensor,
        outcome_disc: torch.Tensor,
    ) -> torch.Tensor:
        trajectory = self.trajectory_embed(trajectory_shape_disc)
        pattern = self.pattern_embed(attention_pattern_disc)
        confidence = self.confidence_embed(confidence_disc)
        outcome = self.outcome_embed(outcome_disc)
        query = self.query_proj(z_q)
        return torch.stack([query, trajectory, pattern, confidence, outcome], dim=1)

    def forward(
        self,
        z_q: torch.Tensor,
        trajectory_shape_disc: torch.Tensor,
        attention_pattern_disc: torch.Tensor,
        confidence_disc: torch.Tensor,
        outcome_disc: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        memory = self._build_memory(
            z_q,
            trajectory_shape_disc,
            attention_pattern_disc,
            confidence_disc,
            outcome_disc,
        )
        _, target_len = target_ids.shape
        positions = torch.arange(target_len, device=target_ids.device)
        x = self.token_embed(target_ids) + self.pos_embed(positions).unsqueeze(0)

        causal_mask = self._causal_mask(target_len, target_ids.device)
        tgt_key_padding_mask = target_ids == self.pad_idx

        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        x = self.norm(x)
        return self.output_proj(x)

    @torch.no_grad()
    def generate(
        self,
        z_q: torch.Tensor,
        trajectory_shape_disc: torch.Tensor,
        attention_pattern_disc: torch.Tensor,
        confidence_disc: torch.Tensor,
        outcome_disc: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_len: int = 48,
    ) -> torch.Tensor:
        memory = self._build_memory(
            z_q,
            trajectory_shape_disc,
            attention_pattern_disc,
            confidence_disc,
            outcome_disc,
        )
        batch_size = z_q.size(0)
        device = z_q.device
        generated = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            target_len = generated.size(1)
            positions = torch.arange(target_len, device=device)
            x = self.token_embed(generated) + self.pos_embed(positions).unsqueeze(0)
            causal_mask = self._causal_mask(target_len, device)
            x = self.decoder(tgt=x, memory=memory, tgt_mask=causal_mask)
            x = self.norm(x)
            logits = self.output_proj(x[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token.squeeze(-1) == eos_idx).all():
                break

        return generated
