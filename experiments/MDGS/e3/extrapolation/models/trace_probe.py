"""Small attention probe used to collect real entropy trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _attention_entropy(attn_weights: torch.Tensor) -> torch.Tensor:
    """Compute normalized attention entropy from attention weights."""
    eps = 1e-8
    num_keys = attn_weights.size(-1)
    entropy = -(attn_weights * (attn_weights.clamp(min=eps).log())).sum(dim=-1)
    denom = math.log(float(max(num_keys, 2)))
    return entropy / max(denom, 1.0)


@dataclass
class ProbeOutput:
    logits: torch.Tensor
    attention_entropy_trajectory: torch.Tensor | None = None
    max_attention_mass: torch.Tensor | None = None


class ProbeAttentionBlock(nn.Module):
    """One attention block that exposes per-layer attention weights."""

    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.self_attn(
            x,
            x,
            x,
            need_weights=True,
            average_attn_weights=False,
        )
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x, attn_weights


class QueryTraceProbe(nn.Module):
    """Self-contained probe model that yields answer logits and attention traces."""

    def __init__(
        self,
        field_vocab_sizes: List[int],
        num_answer_classes: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_fields = len(field_vocab_sizes)
        self.field_embeddings = nn.ModuleList(
            [nn.Embedding(vs + 1, d_model) for vs in field_vocab_sizes]
        )
        self.pos_embed = nn.Embedding(self.num_fields, d_model)
        self.blocks = nn.ModuleList(
            [
                ProbeAttentionBlock(d_model=d_model, nhead=nhead, d_ff=d_ff, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_answer_classes)

    def _embed_queries(self, query_fields: torch.Tensor) -> torch.Tensor:
        embeddings = []
        for i in range(self.num_fields):
            field_ids = query_fields[:, i].clamp(
                min=0,
                max=self.field_embeddings[i].num_embeddings - 1,
            )
            embeddings.append(self.field_embeddings[i](field_ids))

        x = torch.stack(embeddings, dim=1)
        positions = torch.arange(self.num_fields, device=query_fields.device)
        return x + self.pos_embed(positions).unsqueeze(0)

    def forward(self, query_fields: torch.Tensor, collect_trace: bool = False) -> ProbeOutput:
        x = self._embed_queries(query_fields)
        entropy_steps = []
        max_attention_steps = []

        for block in self.blocks:
            x, attn_weights = block(x)
            if collect_trace:
                entropy = _attention_entropy(attn_weights).mean(dim=(1, 2))
                max_mass = attn_weights.max(dim=-1).values.mean(dim=(1, 2))
                entropy_steps.append(entropy)
                max_attention_steps.append(max_mass)

        x = self.norm(x)
        logits = self.classifier(x.mean(dim=1))

        if not collect_trace:
            return ProbeOutput(logits=logits)

        return ProbeOutput(
            logits=logits,
            attention_entropy_trajectory=torch.stack(entropy_steps, dim=1),
            max_attention_mass=torch.stack(max_attention_steps, dim=1).mean(dim=1),
        )
