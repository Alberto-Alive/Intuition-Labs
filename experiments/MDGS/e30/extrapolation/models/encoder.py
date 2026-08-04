"""Local query encoder for E30."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class QueryAnchorEncoder(nn.Module):
    """Encodes structured categorical query fields into a single anchor latent."""

    def __init__(
        self,
        field_vocab_sizes: List[int],
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_fields = len(field_vocab_sizes)
        self.field_embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size + 1, d_model) for vocab_size in field_vocab_sizes]
        )
        self.pos_embed = nn.Embedding(self.num_fields, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query_fields: torch.Tensor) -> torch.Tensor:
        embeddings = []
        for idx in range(self.num_fields):
            field_ids = query_fields[:, idx].clamp(
                min=0,
                max=self.field_embeddings[idx].num_embeddings - 1,
            )
            embeddings.append(self.field_embeddings[idx](field_ids))

        x = torch.stack(embeddings, dim=1)
        positions = torch.arange(self.num_fields, device=query_fields.device)
        x = x + self.pos_embed(positions).unsqueeze(0)
        x = self.encoder(x)
        x = self.norm(x)
        return x.mean(dim=1)
