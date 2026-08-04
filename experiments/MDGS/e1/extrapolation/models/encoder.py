"""Query Encoder adapted for Adult Income categorical fields."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class AdultQueryEncoder(nn.Module):
    """Encodes structured categorical query fields into a dense vector."""

    def __init__(
        self,
        field_vocab_sizes: List[int],
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_fields = len(field_vocab_sizes)

        self.field_embeddings = nn.ModuleList(
            [nn.Embedding(vs + 1, d_model) for vs in field_vocab_sizes]
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
        for i in range(self.num_fields):
            field_ids = query_fields[:, i].clamp(
                min=0,
                max=self.field_embeddings[i].num_embeddings - 1,
            )
            embeddings.append(self.field_embeddings[i](field_ids))

        x = torch.stack(embeddings, dim=1)
        positions = torch.arange(self.num_fields, device=query_fields.device)
        x = x + self.pos_embed(positions).unsqueeze(0)
        x = self.encoder(x)
        x = self.norm(x)
        return x.mean(dim=1)
