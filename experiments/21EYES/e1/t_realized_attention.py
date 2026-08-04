from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


TRAblation = Literal[
    "none",
    "t_zero",
    "k_temp_only",
    "v_temp_only",
    "k_only",
    "v_only",
    "shuffle_T",
    "random_T",
    "sigma_zero",
    "fixed_normal_attention",
]

TRealizationMode = Literal["full", "v_only", "k_only"]


@dataclass(frozen=True)
class TRealizedTransformerConfig:
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


class TRealizedAttention(nn.Module):
    """Causal self-attention with pairwise T-realized keys and values."""

    def __init__(
        self,
        config: TRealizedTransformerConfig,
        *,
        realization_mode: TRealizationMode = "full",
    ) -> None:
        super().__init__()
        if config.d_model % config.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = config.d_model
        self.nhead = config.nhead
        self.d_head = config.d_model // config.nhead
        self.sigma_init_gain = config.sigma_init_gain
        self.realization_mode = realization_mode

        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_point_proj = nn.Linear(config.d_model, config.d_model)
        self.k_mu_proj = nn.Linear(config.d_model, config.d_model)
        self.k_sigma_proj = nn.Linear(config.d_model, config.d_model)
        self.v_mu_proj = nn.Linear(config.d_model, config.d_model)
        self.v_sigma_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in [
            self.q_proj,
            self.k_point_proj,
            self.k_mu_proj,
            self.v_mu_proj,
            self.out_proj,
        ]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        for module in [self.k_sigma_proj, self.v_sigma_proj]:
            nn.init.xavier_uniform_(module.weight, gain=self.sigma_init_gain)
            nn.init.zeros_(module.bias)

    def _split_heads(self, value: Tensor) -> Tensor:
        batch, seq_len, _ = value.shape
        return value.view(batch, seq_len, self.nhead, self.d_head).transpose(1, 2)

    def _merge_heads(self, value: Tensor) -> Tensor:
        batch, _heads, seq_len, _d_head = value.shape
        return value.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

    @staticmethod
    def _shuffle_batch(value: Tensor) -> Tensor:
        if value.shape[0] < 2:
            return value
        permutation = torch.randperm(value.shape[0], device=value.device)
        return value[permutation]

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> Tensor:
        return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).triu(1)

    def forward(
        self,
        x: Tensor,
        *,
        ablation: TRAblation = "none",
        return_diagnostics: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        _batch, seq_len, _width = x.shape
        scale = math.sqrt(float(self.d_head))

        Q = self._split_heads(self.q_proj(x))
        K_point = self._split_heads(self.k_point_proj(x))
        K_mu = self._split_heads(self.k_mu_proj(x))
        V_mu = self._split_heads(self.v_mu_proj(x))

        T = torch.tanh(torch.einsum("bhtd,bhjd->bhtj", Q, K_point) / scale)
        if ablation in {"t_zero", "fixed_normal_attention"}:
            T = torch.zeros_like(T)
        elif ablation == "shuffle_T":
            T = self._shuffle_batch(T)
        elif ablation == "random_T":
            T = torch.empty_like(T).uniform_(-1.0, 1.0)
        causal_mask = self._causal_mask(seq_len, x.device).view(1, 1, seq_len, seq_len)
        T = T.masked_fill(causal_mask, 0.0)

        effective_mode = self.realization_mode
        if ablation in {"v_temp_only", "v_only"}:
            effective_mode = "v_only"
        elif ablation in {"k_temp_only", "k_only"}:
            effective_mode = "k_only"

        if effective_mode == "v_only" or ablation in {"sigma_zero", "fixed_normal_attention"}:
            K_sigma = torch.zeros_like(K_mu)
        else:
            K_sigma = self._split_heads(self.k_sigma_proj(x))

        if effective_mode == "k_only" or ablation in {"sigma_zero", "fixed_normal_attention"}:
            V_sigma = torch.zeros_like(V_mu)
        else:
            V_sigma = self._split_heads(self.v_sigma_proj(x))

        K_temp = K_mu.unsqueeze(2) + T.unsqueeze(-1) * K_sigma.unsqueeze(2)
        V_temp = V_mu.unsqueeze(2) + T.unsqueeze(-1) * V_sigma.unsqueeze(2)
        score = torch.einsum("bhtd,bhtjd->bhtj", Q, K_temp) / scale
        score = score.masked_fill(causal_mask, -torch.finfo(score.dtype).max)
        attn = torch.softmax(score, dim=-1)
        output = torch.einsum("bhtj,bhtjd->bhtd", attn, V_temp)
        y = self.out_proj(self._merge_heads(output))

        diagnostics: dict[str, Tensor] = {}
        if return_diagnostics:
            normal_score = torch.einsum("bhtd,bhjd->bhtj", Q, K_mu) / scale
            normal_score = normal_score.masked_fill(
                self._causal_mask(seq_len, x.device).view(1, 1, seq_len, seq_len),
                -torch.finfo(normal_score.dtype).max,
            )
            normal_attn = torch.softmax(normal_score, dim=-1)
            normal_output = torch.einsum("bhtj,bhjd->bhtd", normal_attn, V_mu)
            diagnostics = {
                "T": T,
                "K_temp": K_temp,
                "V_temp": V_temp,
                "K_mu": K_mu,
                "V_mu": V_mu,
                "K_sigma": K_sigma,
                "V_sigma": V_sigma,
                "score": score,
                "attn": attn,
                "output_heads": output,
                "normal_attn": normal_attn,
                "normal_output_heads": normal_output,
            }
        return y, diagnostics


class NormalCausalSelfAttention(nn.Module):
    def __init__(self, config: TRealizedTransformerConfig) -> None:
        super().__init__()
        if config.d_model % config.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = config.d_model
        self.nhead = config.nhead
        self.d_head = config.d_model // config.nhead
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

    def _split_heads(self, value: Tensor) -> Tensor:
        batch, seq_len, _ = value.shape
        return value.view(batch, seq_len, self.nhead, self.d_head).transpose(1, 2)

    def _merge_heads(self, value: Tensor) -> Tensor:
        batch, _heads, seq_len, _d_head = value.shape
        return value.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> Tensor:
        return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).triu(1)

    def forward(
        self,
        x: Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        _batch, seq_len, _width = x.shape
        scale = math.sqrt(float(self.d_head))
        Q = self._split_heads(self.q_proj(x))
        K = self._split_heads(self.k_proj(x))
        V = self._split_heads(self.v_proj(x))
        score = torch.einsum("bhtd,bhjd->bhtj", Q, K) / scale
        score = score.masked_fill(
            self._causal_mask(seq_len, x.device).view(1, 1, seq_len, seq_len),
            -torch.finfo(score.dtype).max,
        )
        attn = torch.softmax(score, dim=-1)
        output = torch.einsum("bhtj,bhjd->bhtd", attn, V)
        y = self.out_proj(self._merge_heads(output))
        diagnostics = {"attn": attn, "output_heads": output} if return_diagnostics else {}
        return y, diagnostics


class TRealizedBlock(nn.Module):
    def __init__(
        self,
        config: TRealizedTransformerConfig,
        *,
        attention_kind: Literal["t_realized", "fixed"],
    ) -> None:
        super().__init__()
        self.attention_kind = attention_kind
        if attention_kind == "t_realized":
            self.self_attn: nn.Module = TRealizedAttention(config)
        else:
            self.self_attn = NormalCausalSelfAttention(config)
        self.norm1 = nn.LayerNorm(config.d_model)
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
        if isinstance(self.self_attn, TRealizedAttention):
            attn_out, diagnostics = self.self_attn(
                x,
                ablation=ablation,
                return_diagnostics=return_diagnostics,
            )
        else:
            attn_out, diagnostics = self.self_attn(
                x,
                return_diagnostics=return_diagnostics,
            )
        x = self.norm1(x + self.dropout(attn_out))
        mlp_out = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        x = self.norm2(x + self.dropout(mlp_out))
        return x, [diagnostics] if diagnostics else []


class _BaseCausalSequenceClassifier(nn.Module):
    def __init__(
        self,
        config: TRealizedTransformerConfig,
        *,
        attention_kind: Literal["t_realized", "fixed"],
    ) -> None:
        super().__init__()
        self.config = config
        self.attention_kind = attention_kind
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, config.seq_len, config.d_model)
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                TRealizedBlock(config, attention_kind=attention_kind)
                for _ in range(config.num_layers)
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


class TRealizedAttentionTransformer(_BaseCausalSequenceClassifier):
    def __init__(self, config: TRealizedTransformerConfig) -> None:
        super().__init__(config, attention_kind="t_realized")


class FixedNormalTransformerClassifier(_BaseCausalSequenceClassifier):
    def __init__(self, config: TRealizedTransformerConfig) -> None:
        super().__init__(config, attention_kind="fixed")


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    parameters = model.parameters()
    if trainable_only:
        return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def estimate_active_mult_adds(
    config: TRealizedTransformerConfig,
    *,
    model_kind: Literal["fixed", "t_realized"],
) -> int:
    seq_len = config.seq_len
    d_model = config.d_model
    d_head = config.d_model // config.nhead
    d_ff = config.dim_feedforward
    heads = config.nhead

    if model_kind == "t_realized":
        projections = 7 * seq_len * d_model * d_model
        pairwise_t = heads * seq_len * seq_len * d_head
        final_scores = heads * seq_len * seq_len * d_head
        weighted_values = heads * seq_len * seq_len * d_head
        temp_realization = 2 * heads * seq_len * seq_len * d_head
        attention = projections + pairwise_t + final_scores + weighted_values + temp_realization
    else:
        projections = 4 * seq_len * d_model * d_model
        score_and_values = 2 * heads * seq_len * seq_len * d_head
        attention = projections + score_and_values

    mlp = 2 * seq_len * d_model * d_ff
    classifier = d_model * config.num_classes
    return config.num_layers * (attention + mlp) + classifier
