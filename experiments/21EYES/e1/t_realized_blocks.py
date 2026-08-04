from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from .t_realized_attention import (
        TRAblation,
        NormalCausalSelfAttention,
        TRealizationMode,
        TRealizedAttention,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from t_realized_attention import (
        TRAblation,
        NormalCausalSelfAttention,
        TRealizationMode,
        TRealizedAttention,
    )


@dataclass(frozen=True)
class TRealizedBlockTransformerConfig:
    vocab_size: int = 64
    seq_len: int = 16
    num_classes: int = 10
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.05
    readout_position: int = -1
    sigma_init_gain: float = 0.35
    t_layers: tuple[int, ...] = ()
    t_mode: TRealizationMode = "full"

    def __post_init__(self) -> None:
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        bad_layers = [layer for layer in self.t_layers if layer < 0 or layer >= self.num_layers]
        if bad_layers:
            raise ValueError(f"t_layers outside model depth: {bad_layers}")


class StandardBlock(nn.Module):
    """Pre-norm causal transformer block with normal self-attention."""

    def __init__(self, config: TRealizedBlockTransformerConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.self_attn = NormalCausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.linear1 = nn.Linear(config.d_model, config.dim_feedforward)
        self.linear2 = nn.Linear(config.dim_feedforward, config.d_model)

    def forward(
        self,
        x: Tensor,
        *,
        ablation: TRAblation = "none",
        return_diagnostics: bool = False,
    ) -> tuple[Tensor, list[dict[str, Tensor]]]:
        del ablation, return_diagnostics
        attn_out, _diagnostics = self.self_attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        mlp_input = self.norm2(x)
        mlp_out = self.linear2(self.dropout(F.gelu(self.linear1(mlp_input))))
        x = x + self.dropout(mlp_out)
        return x, []


class TRealizedBlock(nn.Module):
    """Pre-norm causal transformer block with T-realized self-attention."""

    def __init__(
        self,
        config: TRealizedBlockTransformerConfig,
        *,
        realization_mode: TRealizationMode = "full",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.self_attn = TRealizedAttention(config, realization_mode=realization_mode)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.linear1 = nn.Linear(config.d_model, config.dim_feedforward)
        self.linear2 = nn.Linear(config.dim_feedforward, config.d_model)

    def forward(
        self,
        x: Tensor,
        *,
        ablation: TRAblation = "none",
        return_diagnostics: bool = False,
    ) -> tuple[Tensor, list[dict[str, Tensor]]]:
        attn_out, diagnostics = self.self_attn(
            self.norm1(x),
            ablation=ablation,
            return_diagnostics=return_diagnostics,
        )
        x = x + self.dropout(attn_out)
        mlp_input = self.norm2(x)
        mlp_out = self.linear2(self.dropout(F.gelu(self.linear1(mlp_input))))
        x = x + self.dropout(mlp_out)
        return x, [diagnostics] if diagnostics else []


class TRealizedBlockTransformer(nn.Module):
    """Sequence classifier with normal and T-realized attention blocks."""

    def __init__(self, config: TRealizedBlockTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.t_layers = tuple(sorted(set(config.t_layers)))
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, config.seq_len, config.d_model)
        )
        self.dropout = nn.Dropout(config.dropout)
        t_layer_set = set(self.t_layers)
        self.blocks = nn.ModuleList(
            [
                TRealizedBlock(config, realization_mode=config.t_mode)
                if layer_idx in t_layer_set
                else StandardBlock(config)
                for layer_idx in range(config.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, config.num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)

    def _readout_index(self, seq_len: int) -> int:
        if self.config.readout_position < 0:
            return max(0, seq_len + self.config.readout_position)
        return min(self.config.readout_position, seq_len - 1)

    def forward(
        self,
        tokens: Tensor,
        *,
        ablation: TRAblation = "none",
        return_details: bool = False,
    ) -> Tensor | dict[str, Tensor | list[dict[str, Tensor]]]:
        x = self.token_embedding(tokens)
        x = x + self.positional_embedding[:, : tokens.shape[1]]
        x = self.dropout(x)
        diagnostics: list[dict[str, Tensor]] = []
        for block in self.blocks:
            x, block_diagnostics = block(
                x,
                ablation=ablation,
                return_diagnostics=return_details,
            )
            diagnostics.extend(block_diagnostics)
        x = self.norm(x)
        logits = self.classifier(x[:, self._readout_index(tokens.shape[1])])
        if return_details:
            return {"logits": logits, "tra_diagnostics": diagnostics}
        return logits

    def t_realized_attentions(self) -> Iterable[TRealizedAttention]:
        for module in self.modules():
            if isinstance(module, TRealizedAttention):
                yield module


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    parameters = model.parameters()
    if trainable_only:
        return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def estimate_active_mult_adds(config: TRealizedBlockTransformerConfig) -> int:
    seq_len = config.seq_len
    d_model = config.d_model
    heads = config.nhead
    d_head = config.d_model // config.nhead
    d_ff = config.dim_feedforward
    t_layer_set = set(config.t_layers)

    fixed_attention = (
        4 * seq_len * d_model * d_model
        + 2 * heads * seq_len * seq_len * d_head
    )
    sigma_projection_count = 2 if config.t_mode == "full" else 1
    t_attention = (
        (5 + sigma_projection_count) * seq_len * d_model * d_model
        + heads * seq_len * seq_len * d_head
        + heads * seq_len * seq_len * d_head
        + heads * seq_len * seq_len * d_head
        + sigma_projection_count * heads * seq_len * seq_len * d_head
    )
    mlp = 2 * seq_len * d_model * d_ff

    total = 0
    for layer_idx in range(config.num_layers):
        attention = t_attention if layer_idx in t_layer_set else fixed_attention
        total += attention + mlp
    total += d_model * config.num_classes
    return int(total)
