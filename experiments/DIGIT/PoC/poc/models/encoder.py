"""Query Encoder adapted for Adult Income categorical fields.

Each query field is a categorical integer, embedded via per-field
embedding tables and processed by a transformer encoder.
"""

import torch
import torch.nn as nn
from typing import List


class AdultQueryEncoder(nn.Module):
    """Encodes structured categorical query fields into a dense vector z_q."""

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

        # Per-field embedding tables (each includes index 0 = "any/no filter")
        self.field_embeddings = nn.ModuleList([
            nn.Embedding(vs + 1, d_model)
            for vs in field_vocab_sizes
        ])

        # Learnable positional embeddings for field positions
        self.pos_embed = nn.Embedding(self.num_fields, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query_fields: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_fields: (B, num_fields) integer tensor.

        Returns:
            z_q: (B, d_model) latent query representation.
        """
        B = query_fields.size(0)

        # Embed each field independently
        embeddings = []
        for i in range(self.num_fields):
            field_ids = query_fields[:, i].clamp(
                min=0, max=self.field_embeddings[i].num_embeddings - 1
            )
            embeddings.append(self.field_embeddings[i](field_ids))

        # Stack: (B, num_fields, d_model)
        x = torch.stack(embeddings, dim=1)

        # Add positional embeddings
        positions = torch.arange(self.num_fields, device=query_fields.device)
        x = x + self.pos_embed(positions).unsqueeze(0)

        # Encode
        x = self.encoder(x)
        x = self.norm(x)

        # Pool: mean over field tokens
        z_q = x.mean(dim=1)
        return z_q
