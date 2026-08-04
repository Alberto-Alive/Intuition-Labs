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
    attention_top2_mass: torch.Tensor | None = None
    attention_effective_support: torch.Tensor | None = None
    attention_gini: torch.Tensor | None = None
    attention_concentration_drift: torch.Tensor | None = None


def _attention_topk_mass(attn_weights: torch.Tensor, k: int) -> torch.Tensor:
    """Average the mass captured by the top-k attention entries."""
    topk = torch.topk(attn_weights, k=min(k, attn_weights.size(-1)), dim=-1).values
    return topk.sum(dim=-1).mean(dim=(1, 2))


def _attention_effective_support(attn_weights: torch.Tensor) -> torch.Tensor:
    """Normalize the effective number of attended keys to [0, 1]."""
    eps = 1e-8
    num_keys = attn_weights.size(-1)
    entropy = -(attn_weights * (attn_weights.clamp(min=eps).log())).sum(dim=-1)
    effective_keys = torch.exp(entropy)
    denom = float(max(num_keys - 1, 1))
    return ((effective_keys - 1.0) / denom).clamp(min=0.0, max=1.0).mean(dim=(1, 2))


def _attention_gini(attn_weights: torch.Tensor) -> torch.Tensor:
    """Compute a normalized Gini coefficient over attention weights."""
    sorted_weights, _ = torch.sort(attn_weights, dim=-1)
    num_keys = sorted_weights.size(-1)
    if num_keys <= 1:
        return torch.zeros(sorted_weights.size(0), device=sorted_weights.device, dtype=sorted_weights.dtype)
    ranks = torch.arange(1, num_keys + 1, device=sorted_weights.device, dtype=sorted_weights.dtype)
    weighted = (sorted_weights * ranks.view(*([1] * (sorted_weights.ndim - 1)), -1)).sum(dim=-1)
    total = sorted_weights.sum(dim=-1).clamp(min=1e-8)
    gini = (2.0 * weighted / (num_keys * total)) - ((num_keys + 1.0) / num_keys)
    gini = (gini * (num_keys / float(num_keys - 1))).clamp(min=0.0, max=1.0)
    return gini.mean(dim=(1, 2))


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
        top2_attention_steps = []
        effective_support_steps = []
        gini_steps = []
        concentration_steps = []

        for block in self.blocks:
            x, attn_weights = block(x)
            if collect_trace:
                entropy = _attention_entropy(attn_weights).mean(dim=(1, 2))
                max_mass = attn_weights.max(dim=-1).values.mean(dim=(1, 2))
                top2_mass = _attention_topk_mass(attn_weights, k=2)
                effective_support = _attention_effective_support(attn_weights)
                attention_gini = _attention_gini(attn_weights)
                concentration_strength = (
                    0.45 * max_mass
                    + 0.35 * attention_gini
                    + 0.20 * (1.0 - effective_support)
                ).clamp(min=0.0, max=1.0)
                entropy_steps.append(entropy)
                max_attention_steps.append(max_mass)
                top2_attention_steps.append(top2_mass)
                effective_support_steps.append(effective_support)
                gini_steps.append(attention_gini)
                concentration_steps.append(concentration_strength)

        x = self.norm(x)
        logits = self.classifier(x.mean(dim=1))

        if not collect_trace:
            return ProbeOutput(logits=logits)

        return ProbeOutput(
            logits=logits,
            attention_entropy_trajectory=torch.stack(entropy_steps, dim=1),
            max_attention_mass=torch.stack(max_attention_steps, dim=1).mean(dim=1),
            attention_top2_mass=torch.stack(top2_attention_steps, dim=1).mean(dim=1),
            attention_effective_support=torch.stack(effective_support_steps, dim=1).mean(dim=1),
            attention_gini=torch.stack(gini_steps, dim=1).mean(dim=1),
            attention_concentration_drift=(
                torch.stack(concentration_steps, dim=1).amax(dim=1)
                - torch.stack(concentration_steps, dim=1).amin(dim=1)
            ).clamp(min=0.0, max=1.0),
        )
