from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class WDATransformerConfig:
    vocab_size: int = 64
    seq_len: int = 18
    num_classes: int = 4
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.05
    coord_dim: int = 32
    distribution_groups: int = 16
    low_rank: int = 4
    refinement_steps: int = 2
    controller_hidden_dim: int = 64
    mask_context_token: bool = True
    context_tokens: int = 1
    apply_wda_qkv: bool = True
    readout_position: int = 2


class WeightDistributionParameterization(nn.Module):
    """Low-rank structured distribution directions for a linear weight matrix."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        distribution_groups: int,
        low_rank: int,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.distribution_groups = distribution_groups
        self.low_rank = low_rank
        self.left = nn.Parameter(
            torch.empty(distribution_groups, out_features, low_rank)
        )
        self.right = nn.Parameter(
            torch.empty(distribution_groups, low_rank, in_features)
        )
        self.log_scale = nn.Parameter(torch.full((distribution_groups,), -0.5))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.left, mean=0.0, std=0.12)
        nn.init.normal_(self.right, mean=0.0, std=0.12)

    def delta_weight(self, group_coordinates: Tensor) -> Tensor:
        """Materialize per-example weight offsets.

        group_coordinates has shape [batch, distribution_groups]. The result is
        [batch, out_features, in_features].
        """

        scales = F.softplus(self.log_scale).view(1, -1)
        weighted_coords = group_coordinates * scales
        return torch.einsum(
            "bg,gor,gri->boi",
            weighted_coords,
            self.left,
            self.right,
        ) / math.sqrt(float(self.low_rank))


class DistributionalWeightRealizer(nn.Module):
    def __init__(self, coord_dim: int, distribution_groups: int) -> None:
        super().__init__()
        self.coord_to_group = nn.Linear(coord_dim, distribution_groups)

    def forward(self, coordinates: Tensor) -> Tensor:
        return torch.tanh(self.coord_to_group(coordinates))


class WDALinear(nn.Module):
    """Linear layer whose effective weights are realized from coordinates."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        coord_dim: int,
        distribution_groups: int,
        low_rank: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.realizer = DistributionalWeightRealizer(coord_dim, distribution_groups)
        self.distribution = WeightDistributionParameterization(
            in_features,
            out_features,
            distribution_groups,
            low_rank,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def materialize_weight(
        self,
        coordinates: Tensor,
        *,
        disable_distribution_scale: bool = False,
    ) -> tuple[Tensor, Tensor]:
        group_coordinates = self.realizer(coordinates)
        if disable_distribution_scale:
            delta = torch.zeros(
                coordinates.shape[0],
                self.out_features,
                self.in_features,
                device=coordinates.device,
                dtype=coordinates.dtype,
            )
        else:
            delta = self.distribution.delta_weight(group_coordinates)
        return self.weight_mu.unsqueeze(0) + delta, group_coordinates

    def forward(
        self,
        x: Tensor,
        coordinates: Tensor,
        *,
        disable_distribution_scale: bool = False,
    ) -> Tensor:
        effective_weight, _ = self.materialize_weight(
            coordinates,
            disable_distribution_scale=disable_distribution_scale,
        )
        y = torch.einsum("bti,boi->bto", x, effective_weight)
        if self.bias is not None:
            y = y + self.bias.view(1, 1, -1)
        return y


class DenseHyperLinear(nn.Module):
    """Deterministic low-rank hypernetwork modulation baseline."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        coord_dim: int,
        low_rank: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.low_rank = low_rank
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.left_head = nn.Linear(coord_dim, out_features * low_rank)
        self.right_head = nn.Linear(coord_dim, low_rank * in_features)
        self.gain = nn.Parameter(torch.tensor(0.01))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.normal_(self.left_head.weight, std=0.01)
        nn.init.zeros_(self.left_head.bias)
        nn.init.normal_(self.right_head.weight, std=0.01)
        nn.init.zeros_(self.right_head.bias)

    def materialize_weight(
        self,
        coordinates: Tensor,
        *,
        disable_distribution_scale: bool = False,
    ) -> tuple[Tensor, Tensor]:
        del disable_distribution_scale
        batch = coordinates.shape[0]
        left = self.left_head(coordinates).view(batch, self.out_features, self.low_rank)
        right = self.right_head(coordinates).view(batch, self.low_rank, self.in_features)
        delta = torch.bmm(left, right) * self.gain / math.sqrt(float(self.low_rank))
        return self.weight.unsqueeze(0) + delta, coordinates

    def forward(
        self,
        x: Tensor,
        coordinates: Tensor,
        *,
        disable_distribution_scale: bool = False,
    ) -> Tensor:
        effective_weight, _ = self.materialize_weight(
            coordinates,
            disable_distribution_scale=disable_distribution_scale,
        )
        y = torch.einsum("bti,boi->bto", x, effective_weight)
        if self.bias is not None:
            y = y + self.bias.view(1, 1, -1)
        return y


class CondConvLinear(nn.Module):
    """CondConv-style weighted combination of expert matrices."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        coord_dim: int,
        distribution_groups: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = max(2, min(distribution_groups, 8))
        self.experts = nn.Parameter(
            torch.empty(self.num_experts, out_features, in_features)
        )
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.router = nn.Linear(coord_dim, self.num_experts)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for expert in self.experts:
            nn.init.kaiming_uniform_(expert, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def materialize_weight(
        self,
        coordinates: Tensor,
        *,
        disable_distribution_scale: bool = False,
    ) -> tuple[Tensor, Tensor]:
        del disable_distribution_scale
        gates = torch.softmax(self.router(coordinates), dim=-1)
        weights = torch.einsum("be,eoi->boi", gates, self.experts)
        return weights, gates

    def forward(
        self,
        x: Tensor,
        coordinates: Tensor,
        *,
        disable_distribution_scale: bool = False,
    ) -> Tensor:
        effective_weight, _ = self.materialize_weight(
            coordinates,
            disable_distribution_scale=disable_distribution_scale,
        )
        y = torch.einsum("bti,boi->bto", x, effective_weight)
        if self.bias is not None:
            y = y + self.bias.view(1, 1, -1)
        return y


class DynamicLoRALinear(nn.Module):
    """Input-conditioned LoRA-style low-rank modulation baseline."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        coord_dim: int,
        low_rank: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.low_rank = low_rank
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.left = nn.Parameter(torch.empty(low_rank, out_features))
        self.right = nn.Parameter(torch.empty(low_rank, in_features))
        self.coeff = nn.Linear(coord_dim, low_rank)
        self.gain = nn.Parameter(torch.tensor(0.05))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.normal_(self.left, std=0.02)
        nn.init.normal_(self.right, std=0.02)
        nn.init.zeros_(self.coeff.bias)
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def materialize_weight(
        self,
        coordinates: Tensor,
        *,
        disable_distribution_scale: bool = False,
    ) -> tuple[Tensor, Tensor]:
        del disable_distribution_scale
        coeff = torch.tanh(self.coeff(coordinates))
        delta = torch.einsum("br,ro,ri->boi", coeff, self.left, self.right)
        delta = delta * self.gain / math.sqrt(float(self.low_rank))
        return self.weight.unsqueeze(0) + delta, coeff

    def forward(
        self,
        x: Tensor,
        coordinates: Tensor,
        *,
        disable_distribution_scale: bool = False,
    ) -> Tensor:
        effective_weight, _ = self.materialize_weight(
            coordinates,
            disable_distribution_scale=disable_distribution_scale,
        )
        y = torch.einsum("bti,boi->bto", x, effective_weight)
        if self.bias is not None:
            y = y + self.bias.view(1, 1, -1)
        return y


class MoELinear(nn.Module):
    """Sparse top-k expert-matrix routing baseline."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        coord_dim: int,
        distribution_groups: int,
        bias: bool = True,
        top_k: int = 1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = max(2, min(distribution_groups, 8))
        self.top_k = max(1, min(top_k, self.num_experts))
        self.experts = nn.Parameter(
            torch.empty(self.num_experts, out_features, in_features)
        )
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.router = nn.Linear(coord_dim, self.num_experts)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for expert in self.experts:
            nn.init.kaiming_uniform_(expert, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def materialize_weight(
        self,
        coordinates: Tensor,
        *,
        disable_distribution_scale: bool = False,
    ) -> tuple[Tensor, Tensor]:
        del disable_distribution_scale
        probs = torch.softmax(self.router(coordinates), dim=-1)
        top_values, top_indices = probs.topk(self.top_k, dim=-1)
        gates = torch.zeros_like(probs).scatter(-1, top_indices, top_values)
        gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        weights = torch.einsum("be,eoi->boi", gates, self.experts)
        return weights, gates

    def forward(
        self,
        x: Tensor,
        coordinates: Tensor,
        *,
        disable_distribution_scale: bool = False,
    ) -> Tensor:
        effective_weight, _ = self.materialize_weight(
            coordinates,
            disable_distribution_scale=disable_distribution_scale,
        )
        y = torch.einsum("bti,boi->bto", x, effective_weight)
        if self.bias is not None:
            y = y + self.bias.view(1, 1, -1)
        return y


class TokenParameterAttentionLinear(nn.Module):
    """Approximate TokenFormer-style token-to-parameter attention."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        distribution_groups: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_parameter_tokens = max(2, min(distribution_groups, 16))
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.query = nn.Linear(in_features, self.num_parameter_tokens)
        self.parameter_values = nn.Parameter(
            torch.empty(self.num_parameter_tokens, out_features)
        )
        self.gain = nn.Parameter(torch.tensor(0.05))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.normal_(self.parameter_values, std=0.02)
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def materialize_weight(
        self,
        coordinates: Tensor,
        *,
        disable_distribution_scale: bool = False,
    ) -> tuple[Tensor, Tensor]:
        del disable_distribution_scale
        batch = coordinates.shape[0]
        activity = torch.zeros(
            batch,
            self.num_parameter_tokens,
            dtype=coordinates.dtype,
            device=coordinates.device,
        )
        return self.weight.unsqueeze(0).expand(batch, -1, -1), activity

    def forward(
        self,
        x: Tensor,
        coordinates: Tensor,
        *,
        disable_distribution_scale: bool = False,
    ) -> Tensor:
        del coordinates, disable_distribution_scale
        y = F.linear(x, self.weight, self.bias)
        attn = torch.softmax(self.query(x), dim=-1)
        token_update = torch.matmul(attn, self.parameter_values)
        return y + self.gain * token_update


class AtomAttentionLinear(nn.Module):
    """Previous POF-style weighted selection from learned weight atoms."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        coord_dim: int,
        distribution_groups: int,
        bias: bool = True,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_atoms = max(2, min(distribution_groups, 8))
        self.top_k = max(1, min(top_k, self.num_atoms))
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.atoms = nn.Parameter(torch.empty(self.num_atoms, out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.selector = nn.Linear(coord_dim, self.num_atoms)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.normal_(self.atoms, std=0.02)
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def materialize_weight(
        self,
        coordinates: Tensor,
        *,
        disable_distribution_scale: bool = False,
    ) -> tuple[Tensor, Tensor]:
        del disable_distribution_scale
        probs = torch.softmax(self.selector(coordinates), dim=-1)
        top_values, top_indices = probs.topk(self.top_k, dim=-1)
        weights = torch.zeros_like(probs).scatter(-1, top_indices, top_values)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        delta = torch.einsum("ba,aoi->boi", weights, self.atoms)
        return self.weight.unsqueeze(0) + delta, weights

    def forward(
        self,
        x: Tensor,
        coordinates: Tensor,
        *,
        disable_distribution_scale: bool = False,
    ) -> Tensor:
        effective_weight, _ = self.materialize_weight(
            coordinates,
            disable_distribution_scale=disable_distribution_scale,
        )
        y = torch.einsum("bti,boi->bto", x, effective_weight)
        if self.bias is not None:
            y = y + self.bias.view(1, 1, -1)
        return y


DynamicLinear = (
    WDALinear,
    DenseHyperLinear,
    CondConvLinear,
    DynamicLoRALinear,
    MoELinear,
    TokenParameterAttentionLinear,
    AtomAttentionLinear,
)


def make_dynamic_linear(
    kind: str,
    in_features: int,
    out_features: int,
    config: WDATransformerConfig,
) -> nn.Module:
    if kind == "wda":
        return WDALinear(
            in_features,
            out_features,
            coord_dim=config.coord_dim,
            distribution_groups=config.distribution_groups,
            low_rank=config.low_rank,
        )
    if kind == "dense_hypernetwork":
        return DenseHyperLinear(
            in_features,
            out_features,
            coord_dim=config.coord_dim,
            low_rank=config.low_rank,
        )
    if kind == "condconv":
        return CondConvLinear(
            in_features,
            out_features,
            coord_dim=config.coord_dim,
            distribution_groups=config.distribution_groups,
        )
    if kind == "dynamic_lora":
        return DynamicLoRALinear(
            in_features,
            out_features,
            coord_dim=config.coord_dim,
            low_rank=config.low_rank,
        )
    if kind == "moe":
        return MoELinear(
            in_features,
            out_features,
            coord_dim=config.coord_dim,
            distribution_groups=config.distribution_groups,
        )
    if kind == "token_parameter_attention":
        return TokenParameterAttentionLinear(
            in_features,
            out_features,
            distribution_groups=config.distribution_groups,
        )
    if kind == "weight_atom_attention":
        return AtomAttentionLinear(
            in_features,
            out_features,
            coord_dim=config.coord_dim,
            distribution_groups=config.distribution_groups,
        )
    raise ValueError(f"unknown dynamic linear kind={kind!r}")


class WeightDistributionAttentionController(nn.Module):
    """Small attention controller that emits continuous distribution coordinates."""

    def __init__(
        self,
        *,
        vocab_size: int,
        seq_len: int,
        coord_dim: int,
        hidden_dim: int,
        num_distribution_tokens: int,
        iterative: bool,
        refinement_steps: int,
    ) -> None:
        super().__init__()
        self.coord_dim = coord_dim
        self.iterative = iterative
        self.refinement_steps = refinement_steps
        self.embedding = nn.Embedding(vocab_size + 1, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        self.encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.distribution_tokens = nn.Parameter(
            torch.empty(num_distribution_tokens, hidden_dim)
        )
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.attended_to_coord = nn.Linear(hidden_dim, coord_dim)
        self.context_to_coord = nn.Linear(hidden_dim, coord_dim)
        self.coord_state = nn.Linear(coord_dim, hidden_dim)
        self.refiner = nn.Sequential(
            nn.Linear(hidden_dim * 2 + coord_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, coord_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.distribution_tokens, std=0.05)

    def _context(self, tokens: Tensor) -> Tensor:
        x = self.embedding(tokens) + self.position[:, : tokens.shape[1]]
        pooled = x.mean(dim=1)
        return self.encoder(pooled)

    def _attend(self, context: Tensor, coordinates: Tensor | None = None) -> Tensor:
        if coordinates is not None:
            context = context + self.coord_state(coordinates)
        query = self.query(context)
        logits = torch.matmul(query, self.distribution_tokens.t())
        logits = logits / math.sqrt(float(query.shape[-1]))
        weights = torch.softmax(logits, dim=-1)
        return torch.matmul(weights, self.distribution_tokens)

    def forward(self, tokens: Tensor) -> tuple[Tensor, list[Tensor]]:
        context = self._context(tokens)
        if not self.iterative:
            attended = self._attend(context)
            coordinates = torch.tanh(
                self.context_to_coord(context) + self.attended_to_coord(attended)
            )
            return coordinates, [coordinates]

        coordinates = torch.zeros(
            tokens.shape[0],
            self.coord_dim,
            dtype=context.dtype,
            device=context.device,
        )
        history = [coordinates]
        for step in range(self.refinement_steps):
            attended = self._attend(context, coordinates)
            delta = self.refiner(torch.cat([context, attended, coordinates], dim=-1))
            coordinates = torch.tanh(coordinates + delta / float(step + 1))
            history.append(coordinates)
        return coordinates, history


class FixedTransformerLayer(nn.Module):
    def __init__(self, config: WDATransformerConfig) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            config.d_model,
            config.nhead,
            dropout=config.dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(config.d_model, config.dim_feedforward)
        self.linear2 = nn.Linear(config.dim_feedforward, config.d_model)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        attn_out, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        mlp = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        return self.norm2(x + self.dropout(mlp))


class WDATransformerLayer(nn.Module):
    def __init__(
        self,
        config: WDATransformerConfig,
        *,
        linear_kind: str = "wda",
    ) -> None:
        super().__init__()
        if config.d_model % config.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.config = config
        self.nhead = config.nhead
        self.head_dim = config.d_model // config.nhead
        self.linear_kind = linear_kind

        if config.apply_wda_qkv:
            self.q_proj = make_dynamic_linear(
                linear_kind, config.d_model, config.d_model, config
            )
            self.k_proj = make_dynamic_linear(
                linear_kind, config.d_model, config.d_model, config
            )
            self.v_proj = make_dynamic_linear(
                linear_kind, config.d_model, config.d_model, config
            )
        else:
            self.q_proj = nn.Linear(config.d_model, config.d_model)
            self.k_proj = nn.Linear(config.d_model, config.d_model)
            self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = make_dynamic_linear(
            linear_kind, config.d_model, config.d_model, config
        )
        self.linear1 = make_dynamic_linear(
            linear_kind, config.d_model, config.dim_feedforward, config
        )
        self.linear2 = make_dynamic_linear(
            linear_kind, config.dim_feedforward, config.d_model, config
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def _project(
        self,
        module: nn.Module,
        x: Tensor,
        coordinates: Tensor,
        disable_distribution_scale: bool,
    ) -> Tensor:
        if isinstance(module, DynamicLinear):
            return module(
                x,
                coordinates,
                disable_distribution_scale=disable_distribution_scale,
            )
        return module(x)

    def _attention(
        self,
        x: Tensor,
        coordinates: Tensor,
        disable_distribution_scale: bool,
    ) -> Tensor:
        batch, seq_len, d_model = x.shape
        q = self._project(self.q_proj, x, coordinates, disable_distribution_scale)
        k = self._project(self.k_proj, x, coordinates, disable_distribution_scale)
        v = self._project(self.v_proj, x, coordinates, disable_distribution_scale)
        q = q.view(batch, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch, seq_len, d_model)
        return self.out_proj(
            out,
            coordinates,
            disable_distribution_scale=disable_distribution_scale,
        )

    def forward(
        self,
        x: Tensor,
        coordinates: Tensor,
        *,
        disable_distribution_scale: bool = False,
    ) -> Tensor:
        attn_out = self._attention(x, coordinates, disable_distribution_scale)
        x = self.norm1(x + self.dropout(attn_out))
        mlp = self.linear2(
            self.dropout(
                F.gelu(
                    self.linear1(
                        x,
                        coordinates,
                        disable_distribution_scale=disable_distribution_scale,
                    )
                )
            ),
            coordinates,
            disable_distribution_scale=disable_distribution_scale,
        )
        return self.norm2(x + self.dropout(mlp))


class WeightDistributionAttentionTransformer(nn.Module):
    def __init__(
        self,
        config: WDATransformerConfig,
        *,
        variant: str = "wda_one_shot",
    ) -> None:
        super().__init__()
        self.config = config
        self.variant = variant
        self.mask_token_id = config.vocab_size
        self.embedding = nn.Embedding(config.vocab_size + 1, config.d_model)
        self.position = nn.Parameter(torch.zeros(1, config.seq_len, config.d_model))
        self.dropout = nn.Dropout(config.dropout)
        wda_variants = {
            "wda_one_shot",
            "wda_iterative",
            "wda_small",
            "wda_small_iterative",
            "wda_fixed_coordinates",
            "wda_random_coordinates",
            "wda_shuffled_coordinates",
            "wda_no_distribution_scale",
            "wda_no_controller",
            "wda_frozen_controller",
            "wda_no_iterative_refinement",
        }
        dynamic_kind_by_variant = {
            "wda_dense_hypernetwork_control": "dense_hypernetwork",
            "hypernetwork_transformer": "dense_hypernetwork",
            "condconv_style_transformer": "condconv",
            "dynamic_lora_transformer": "dynamic_lora",
            "moe_transformer": "moe",
            "token_parameter_attention_transformer": "token_parameter_attention",
            "previous_weight_atom_attention_transformer": "weight_atom_attention",
        }
        self.is_wda = variant in wda_variants
        self.linear_kind = "wda" if self.is_wda else dynamic_kind_by_variant.get(variant)
        self.uses_controller = self.linear_kind is not None
        self.iterative = variant in {"wda_iterative", "wda_small_iterative"}

        if self.uses_controller:
            self.controller = WeightDistributionAttentionController(
                vocab_size=config.vocab_size,
                seq_len=config.seq_len,
                coord_dim=config.coord_dim,
                hidden_dim=config.controller_hidden_dim,
                num_distribution_tokens=config.distribution_groups,
                iterative=self.iterative,
                refinement_steps=config.refinement_steps,
            )
            if variant == "wda_frozen_controller":
                for param in self.controller.parameters():
                    param.requires_grad = False
            self.layers = nn.ModuleList(
                [
                    WDATransformerLayer(
                        config,
                        linear_kind=self.linear_kind or "wda",
                    )
                    for _ in range(config.num_layers)
                ]
            )
        else:
            self.controller = None
            self.layers = nn.ModuleList(
                [FixedTransformerLayer(config) for _ in range(config.num_layers)]
            )
        self.norm = nn.LayerNorm(config.d_model)
        if self.uses_controller:
            self.classifier = make_dynamic_linear(
                self.linear_kind or "wda",
                config.d_model,
                config.num_classes,
                config,
            )
        else:
            self.classifier = nn.Linear(config.d_model, config.num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.position, std=0.02)

    def _transformer_tokens(self, tokens: Tensor) -> Tensor:
        if self.config.mask_context_token and self.uses_controller:
            tokens = tokens.clone()
            width = min(self.config.context_tokens, tokens.shape[1])
            tokens[:, :width] = self.mask_token_id
        return tokens

    def _coordinates(
        self,
        tokens: Tensor,
        *,
        coordinate_mode: str,
        fixed_coordinates: Tensor | None,
    ) -> tuple[Tensor | None, list[Tensor]]:
        if self.controller is None:
            return None, []
        if self.variant == "wda_no_controller":
            coordinate_mode = "zero"
        coordinates, history = self.controller(tokens)
        if coordinate_mode == "normal":
            return coordinates, history
        if coordinate_mode == "zero":
            zero = torch.zeros_like(coordinates)
            return zero, [zero for _ in history]
        if coordinate_mode == "fixed":
            if fixed_coordinates is None:
                fixed = coordinates.mean(dim=0, keepdim=True)
            else:
                fixed = fixed_coordinates.to(coordinates.device, coordinates.dtype)
            fixed = fixed.expand_as(coordinates)
            return fixed, [fixed for _ in history]
        if coordinate_mode == "random":
            random = torch.empty_like(coordinates).uniform_(-1.0, 1.0)
            return random, [random for _ in history]
        if coordinate_mode == "shuffled":
            if coordinates.shape[0] < 2:
                return coordinates, history
            perm = torch.randperm(coordinates.shape[0], device=coordinates.device)
            shuffled = coordinates[perm]
            return shuffled, [h[perm] for h in history]
        raise ValueError(f"unknown coordinate_mode={coordinate_mode!r}")

    def forward(
        self,
        tokens: Tensor,
        *,
        coordinate_mode: str = "normal",
        fixed_coordinates: Tensor | None = None,
        disable_distribution_scale: bool = False,
        return_details: bool = False,
    ) -> Tensor | dict[str, Tensor | list[Tensor] | None]:
        coordinates, history = self._coordinates(
            tokens,
            coordinate_mode=coordinate_mode,
            fixed_coordinates=fixed_coordinates,
        )
        x_tokens = self._transformer_tokens(tokens)
        x = self.embedding(x_tokens) + self.position[:, : tokens.shape[1]]
        x = self.dropout(x)
        for layer in self.layers:
            if isinstance(layer, WDATransformerLayer):
                if coordinates is None:
                    raise RuntimeError("WDA layer requires coordinates")
                x = layer(
                    x,
                    coordinates,
                    disable_distribution_scale=disable_distribution_scale,
                )
            else:
                x = layer(x)
        x = self.norm(x)
        readout_position = min(self.config.readout_position, x.shape[1] - 1)
        readout = x[:, readout_position : readout_position + 1]
        if isinstance(self.classifier, DynamicLinear):
            if coordinates is None:
                raise RuntimeError("WDA classifier requires coordinates")
            logits = self.classifier(
                readout,
                coordinates,
                disable_distribution_scale=disable_distribution_scale,
            ).squeeze(1)
        else:
            logits = self.classifier(readout.squeeze(1))
        if return_details:
            return {
                "logits": logits,
                "coordinates": coordinates,
                "coordinate_history": history,
            }
        return logits

    def wda_linears(self) -> Iterable[WDALinear]:
        for module in self.modules():
            if isinstance(module, WDALinear):
                yield module

    def dynamic_linears(self) -> Iterable[nn.Module]:
        for module in self.modules():
            if isinstance(module, DynamicLinear):
                yield module


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def controller_parameter_count(model: nn.Module) -> int:
    controller = getattr(model, "controller", None)
    if controller is None:
        return 0
    return count_parameters(controller)
