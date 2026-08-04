"""Component 1: Query Encoder.

Input: structured query fields (numeric vector).
Architecture: MLP projecting to d_model, then a small Transformer encoder.
Output: latent query representation z_q of shape (batch, d_model).
"""

import torch
import torch.nn as nn
import math


class QueryEncoder(nn.Module):
    def __init__(
        self,
        num_query_fields: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_query_len: int = 64,
    ):
        super().__init__()
        self.d_model = d_model

        # Project each query field into a token embedding.
        # We treat the query as a sequence of field tokens.
        self.field_proj = nn.Linear(1, d_model)

        # Learnable positional embeddings for field positions
        self.pos_embed = nn.Embedding(max_query_len, d_model)

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
            query_fields: (batch, num_fields) — structured numeric query.

        Returns:
            z_q: (batch, d_model) — latent query representation.
        """
        B, F = query_fields.shape

        # Each field becomes a token: (B, F, 1) -> (B, F, d_model)
        x = self.field_proj(query_fields.unsqueeze(-1))

        # Add positional embeddings
        positions = torch.arange(F, device=query_fields.device)
        x = x + self.pos_embed(positions).unsqueeze(0)

        # Encode
        x = self.encoder(x)  # (B, F, d_model)
        x = self.norm(x)

        # Pool: mean over field tokens
        z_q = x.mean(dim=1)  # (B, d_model)
        return z_q
