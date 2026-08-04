from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


DATABLATION = Literal[
    "none",
    "fixed_point",
    "sigma_zero",
    "shuffle_point",
    "shuffle_attn",
    "shuffle_both",
    "random_point",
    "fixed_attention",
    "fixed_values",
    "no_distribution",
    "score_detached_point",
]

DATPlacement = Literal["MLP_FIRST", "MLP_SECOND"]


@dataclass(frozen=True)
class DistributionAttentionTransformerConfig:
    vocab_size: int = 64
    seq_len: int = 16
    num_classes: int = 10
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.05
    num_distributions: int = 128
    d_key: int = 64
    use_distribution_attention_in: DATPlacement = "MLP_FIRST"
    readout_position: int = 0


class DistributionAttentionLinear(nn.Module):
    """Attention over learned value distributions.

    The same attention score selects both the distribution weight and the point
    inside that distribution:

        score -> softmax(score) = attn
        score -> tanh(score) = point
        mu + point * sigma = realized_values
        sum(attn * realized_values) = output
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        *,
        num_distributions: int,
        d_key: int,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.num_distributions = num_distributions
        self.d_key = d_key
        self.query_proj = nn.Linear(d_in, d_key)
        self.dist_keys = nn.Parameter(torch.empty(num_distributions, d_key))
        self.mu = nn.Parameter(torch.empty(num_distributions, d_out))
        self.sigma = nn.Parameter(torch.empty(num_distributions, d_out))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)
        nn.init.normal_(self.dist_keys, mean=0.0, std=1.0)
        nn.init.normal_(self.mu, mean=0.0, std=1.0 / math.sqrt(self.d_out))
        nn.init.normal_(self.sigma, mean=0.0, std=0.15)

    def _shuffle_batch(self, value: Tensor) -> Tensor:
        if value.shape[0] < 2:
            return value
        perm = torch.randperm(value.shape[0], device=value.device)
        return value[perm]

    def forward(
        self,
        x: Tensor,
        *,
        ablation: DATABLATION = "none",
        return_diagnostics: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        q = self.query_proj(x)
        score = torch.matmul(q, self.dist_keys.t()) / math.sqrt(float(self.d_key))

        attn = torch.softmax(score, dim=-1)
        point_source = score.detach() if ablation == "score_detached_point" else score
        point = torch.tanh(point_source)
        sigma = self.sigma

        if ablation in {"fixed_point", "fixed_values", "no_distribution"}:
            point = torch.zeros_like(point)
        elif ablation in {"shuffle_point", "shuffle_both"}:
            point = self._shuffle_batch(point)
        elif ablation == "random_point":
            point = torch.empty_like(point).uniform_(-1.0, 1.0)

        if ablation in {"sigma_zero", "fixed_values", "no_distribution"}:
            sigma = torch.zeros_like(sigma)

        if ablation in {"shuffle_attn", "shuffle_both"}:
            attn = self._shuffle_batch(attn)
        elif ablation == "fixed_attention":
            attn = torch.full_like(attn, 1.0 / float(self.num_distributions))

        realized_values = (
            self.mu.view(1, 1, self.num_distributions, self.d_out)
            + point.unsqueeze(-1)
            * sigma.view(1, 1, self.num_distributions, self.d_out)
        )
        y = torch.sum(attn.unsqueeze(-1) * realized_values, dim=2)

        diagnostics: dict[str, Tensor] = {}
        if return_diagnostics:
            diagnostics = {
                "score": score,
                "attn": attn,
                "point": point,
                "realized_values": realized_values,
                "point_variance": point.var(unbiased=False),
                "realized_value_variance": realized_values.var(unbiased=False),
                "sigma_norm": sigma.norm(dim=-1).mean(),
            }
        return y, diagnostics


class ParameterAttentionLinear(nn.Module):
    """Fixed-value token-to-parameter attention baseline."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        *,
        num_distributions: int,
        d_key: int,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.num_distributions = num_distributions
        self.d_key = d_key
        self.query_proj = nn.Linear(d_in, d_key)
        self.dist_keys = nn.Parameter(torch.empty(num_distributions, d_key))
        self.values = nn.Parameter(torch.empty(num_distributions, d_out))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)
        nn.init.normal_(self.dist_keys, mean=0.0, std=1.0)
        nn.init.normal_(self.values, mean=0.0, std=1.0 / math.sqrt(self.d_out))

    def forward(self, x: Tensor) -> Tensor:
        q = self.query_proj(x)
        score = torch.matmul(q, self.dist_keys.t()) / math.sqrt(float(self.d_key))
        attn = torch.softmax(score, dim=-1)
        return torch.matmul(attn, self.values)


class HyperNetworkLinear(nn.Module):
    """Small per-token low-rank hypernetwork baseline.

    This is intentionally a baseline only. DAT does not use this module.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        *,
        rank: int = 2,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.base = nn.Linear(d_in, d_out)
        self.context = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
        )
        self.left_head = nn.Linear(hidden_dim, d_out * rank)
        self.right_head = nn.Linear(hidden_dim, rank * d_in)
        self.gain = nn.Parameter(torch.tensor(0.03))
        self.reset_hyper_parameters()

    def reset_hyper_parameters(self) -> None:
        nn.init.normal_(self.left_head.weight, std=0.01)
        nn.init.zeros_(self.left_head.bias)
        nn.init.normal_(self.right_head.weight, std=0.01)
        nn.init.zeros_(self.right_head.bias)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        context = self.context(x)
        left = self.left_head(context).view(batch, seq_len, self.d_out, self.rank)
        right = self.right_head(context).view(batch, seq_len, self.rank, self.d_in)
        dynamic = torch.einsum("btor,btri,bti->bto", left, right, x)
        dynamic = dynamic * self.gain / math.sqrt(float(self.rank))
        return self.base(x) + dynamic


class DistributionAttentionMLP(nn.Module):
    def __init__(
        self,
        config: DistributionAttentionTransformerConfig,
        *,
        kind: Literal["dat", "fixed", "parameter_attention", "hypernetwork"],
    ) -> None:
        super().__init__()
        self.kind = kind
        self.dropout = nn.Dropout(config.dropout)

        if kind == "dat" and config.use_distribution_attention_in == "MLP_FIRST":
            self.linear1: nn.Module = DistributionAttentionLinear(
                config.d_model,
                config.dim_feedforward,
                num_distributions=config.num_distributions,
                d_key=config.d_key,
            )
        elif kind == "parameter_attention":
            self.linear1 = ParameterAttentionLinear(
                config.d_model,
                config.dim_feedforward,
                num_distributions=config.num_distributions,
                d_key=config.d_key,
            )
        elif kind == "hypernetwork":
            self.linear1 = HyperNetworkLinear(
                config.d_model,
                config.dim_feedforward,
                hidden_dim=config.dim_feedforward,
            )
        else:
            self.linear1 = nn.Linear(config.d_model, config.dim_feedforward)

        if kind == "dat" and config.use_distribution_attention_in == "MLP_SECOND":
            self.linear2: nn.Module = DistributionAttentionLinear(
                config.dim_feedforward,
                config.d_model,
                num_distributions=config.num_distributions,
                d_key=config.d_key,
            )
        else:
            self.linear2 = nn.Linear(config.dim_feedforward, config.d_model)

    def _call_linear(
        self,
        module: nn.Module,
        x: Tensor,
        *,
        ablation: DATABLATION,
        return_diagnostics: bool,
    ) -> tuple[Tensor, list[dict[str, Tensor]]]:
        if isinstance(module, DistributionAttentionLinear):
            y, diagnostics = module(
                x,
                ablation=ablation,
                return_diagnostics=return_diagnostics,
            )
            return y, [diagnostics] if diagnostics else []
        return module(x), []

    def forward(
        self,
        x: Tensor,
        *,
        ablation: DATABLATION = "none",
        return_diagnostics: bool = False,
    ) -> tuple[Tensor, list[dict[str, Tensor]]]:
        h, diagnostics = self._call_linear(
            self.linear1,
            x,
            ablation=ablation,
            return_diagnostics=return_diagnostics,
        )
        h = self.dropout(F.gelu(h))
        y, diagnostics2 = self._call_linear(
            self.linear2,
            h,
            ablation=ablation,
            return_diagnostics=return_diagnostics,
        )
        return y, diagnostics + diagnostics2


class DistributionAttentionBlock(nn.Module):
    def __init__(
        self,
        config: DistributionAttentionTransformerConfig,
        *,
        mlp_kind: Literal["dat", "fixed", "parameter_attention", "hypernetwork"],
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            config.d_model,
            config.nhead,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.mlp = DistributionAttentionMLP(config, kind=mlp_kind)

    def forward(
        self,
        x: Tensor,
        *,
        ablation: DATABLATION = "none",
        return_diagnostics: bool = False,
    ) -> tuple[Tensor, list[dict[str, Tensor]]]:
        attn_out, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        mlp_out, diagnostics = self.mlp(
            x,
            ablation=ablation,
            return_diagnostics=return_diagnostics,
        )
        x = self.norm2(x + self.dropout(mlp_out))
        return x, diagnostics


class _BaseSequenceClassifier(nn.Module):
    def __init__(
        self,
        config: DistributionAttentionTransformerConfig,
        *,
        mlp_kind: Literal["dat", "fixed", "parameter_attention", "hypernetwork"],
    ) -> None:
        super().__init__()
        self.config = config
        self.mlp_kind = mlp_kind
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, config.seq_len, config.d_model)
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                DistributionAttentionBlock(config, mlp_kind=mlp_kind)
                for _ in range(config.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, config.num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: Tensor,
        *,
        ablation: DATABLATION = "none",
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
        readout_position = min(self.config.readout_position, x.shape[1] - 1)
        logits = self.classifier(x[:, readout_position])
        if return_details:
            return {"logits": logits, "dat_diagnostics": diagnostics}
        return logits

    def dat_linears(self) -> Iterable[DistributionAttentionLinear]:
        for module in self.modules():
            if isinstance(module, DistributionAttentionLinear):
                yield module


class DistributionAttentionTransformer(_BaseSequenceClassifier):
    def __init__(self, config: DistributionAttentionTransformerConfig) -> None:
        super().__init__(config, mlp_kind="dat")


class FixedTransformerClassifier(_BaseSequenceClassifier):
    def __init__(self, config: DistributionAttentionTransformerConfig) -> None:
        super().__init__(config, mlp_kind="fixed")


class TokenParameterAttentionTransformer(_BaseSequenceClassifier):
    def __init__(self, config: DistributionAttentionTransformerConfig) -> None:
        super().__init__(config, mlp_kind="parameter_attention")


class HyperNetworkTransformer(_BaseSequenceClassifier):
    def __init__(self, config: DistributionAttentionTransformerConfig) -> None:
        super().__init__(config, mlp_kind="hypernetwork")


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    parameters = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in parameters if p.requires_grad)
    return sum(p.numel() for p in parameters)


def estimate_active_mult_adds(
    config: DistributionAttentionTransformerConfig,
    *,
    model_kind: Literal["fixed", "dat", "parameter_attention", "hypernetwork"],
) -> int:
    seq_len = config.seq_len
    d_model = config.d_model
    d_ff = config.dim_feedforward
    num_layers = config.num_layers
    n_dist = config.num_distributions
    d_key = config.d_key

    self_attention = (
        4 * seq_len * d_model * d_model
        + 2 * config.nhead * seq_len * seq_len * (d_model // config.nhead)
    )
    fixed_mlp = 2 * seq_len * d_model * d_ff

    if model_kind == "fixed":
        mlp = fixed_mlp
    elif model_kind == "parameter_attention":
        parameter_attention = (
            seq_len * d_model * d_key
            + seq_len * n_dist * d_key
            + seq_len * n_dist * d_ff
        )
        mlp = parameter_attention + seq_len * d_ff * d_model
    elif model_kind == "hypernetwork":
        rank = 2
        hidden = d_ff
        hyper = (
            seq_len * d_model * hidden
            + seq_len * hidden * (d_ff * rank + rank * d_model)
            + seq_len * d_ff * rank * d_model
        )
        mlp = seq_len * d_model * d_ff + hyper + seq_len * d_ff * d_model
    else:
        dat = (
            seq_len * d_model * d_key
            + seq_len * n_dist * d_key
            + 2 * seq_len * n_dist * d_ff
        )
        mlp = dat + seq_len * d_ff * d_model

    classifier = d_model * config.num_classes
    return int(num_layers * (self_attention + mlp) + classifier)
