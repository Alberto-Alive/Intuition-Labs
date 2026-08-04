from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from data import VOCAB_SIZE


VARIANTS = {
    "baseline_transformer",
    "param_matched_baseline",
    "single_avenue_reference",
    "e8_best_reference",
    "junction_middle_bottleneck",
    "junction_every_layer_bottleneck",
    "junction_every_2_layers_bottleneck",
    "forced_junction_output",
    "scout_best_custom",
}

E8_REFERENCE_VARIANTS = {"single_avenue_reference", "e8_best_reference"}
JUNCTION_VARIANTS = {
    "junction_middle_bottleneck",
    "junction_every_layer_bottleneck",
    "junction_every_2_layers_bottleneck",
    "forced_junction_output",
    "scout_best_custom",
}
AVENUE_VARIANTS = E8_REFERENCE_VARIANTS | JUNCTION_VARIANTS


@dataclass
class TransformerConfig:
    variant: str = "baseline_transformer"
    vocab_size: int = VOCAB_SIZE
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_seq_len: int = 256
    param_adapter_dim: int = 4096
    num_avenues: int = 4
    avenue_dim: int = 128
    avenue_dropout: float = 0.0
    leave_one_prob: float = 0.0
    align_stride: int = 8
    avenue_scale_init: float = 0.1
    junction_tokens: int = 8
    junction_scale_init: float = 0.35
    final_junction_scale_init: float = 0.7
    cross_gated: bool = False
    force_junction_output: bool = False

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TransformerConfig":
        data = dict(raw)
        data.setdefault("vocab_size", VOCAB_SIZE)
        cfg = cls(**data)
        if cfg.variant not in VARIANTS:
            raise ValueError(f"Unknown variant '{cfg.variant}'. Expected one of {sorted(VARIANTS)}")
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if cfg.avenue_dim != cfg.d_model:
            raise ValueError("This scout implementation requires avenue_dim == d_model")
        if cfg.variant == "single_avenue_reference":
            cfg.num_avenues = 1
        if cfg.variant in AVENUE_VARIANTS and cfg.num_avenues <= 0:
            raise ValueError("num_avenues must be positive")
        if cfg.align_stride <= 0:
            raise ValueError("align_stride must be positive")
        if cfg.junction_tokens <= 0:
            raise ValueError("junction_tokens must be positive")
        if cfg.variant in {"forced_junction_output", "scout_best_custom"}:
            cfg.force_junction_output = True
        if cfg.variant == "scout_best_custom":
            cfg.cross_gated = True
        return cfg


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()


def _entropy(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    probs_f = probs.float().clamp_min(1.0e-8)
    return -(probs_f * probs_f.log()).sum(dim=dim)


def _offdiag_mean_cosine(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0 or x.shape[-2] <= 1:
        return x.new_zeros(())
    normed = F.normalize(x.float(), dim=-1)
    cosine = torch.matmul(normed, normed.transpose(-1, -2))
    width = cosine.shape[-1]
    eye = torch.eye(width, dtype=torch.bool, device=x.device).view(1, width, width)
    return cosine.masked_select(~eye.expand(cosine.shape[0], -1, -1)).mean()


def _mean_tensor(values: Sequence[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
    if not values:
        return ref.new_zeros(())
    return torch.stack([value.float() for value in values]).mean()


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ParamMatchedAdapter(nn.Module):
    def __init__(self, d_model: int, adapter_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, adapter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, d_model),
            nn.Dropout(dropout),
        )
        final = self.net[3]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self._last_attention_path = "uninitialized"

    def forward(self, x: torch.Tensor, *, return_diagnostics: bool = False) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bsz, seq_len, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        if not return_diagnostics:
            self._last_attention_path = "sdpa_causal"
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=float(self.cfg.dropout) if self.training else 0.0,
                is_causal=True,
            )
            out = out.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
            return self.out_proj(out), {"sdpa_used": x.new_ones(())}

        self._last_attention_path = "manual_dense"
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)
        causal = _causal_mask(seq_len, x.device)
        logits = logits.masked_fill(~causal.view(1, 1, seq_len, seq_len), torch.finfo(logits.dtype).min)
        weights = F.softmax(logits.float(), dim=-1).to(dtype=x.dtype)
        attn = self.dropout(weights) if self.training else weights
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        token_entropy = _entropy(weights.float(), dim=-1)
        return self.out_proj(out), {
            "attention_entropy": token_entropy.mean(),
            "effective_attended_tokens": token_entropy.exp().mean(),
            "sdpa_used": x.new_zeros(()),
        }


class AvenueMLP(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class E8ReferenceAvenueModule(nn.Module):
    def __init__(self, cfg: TransformerConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.num_avenues = 1 if cfg.variant == "single_avenue_reference" else int(cfg.num_avenues)
        self.avenues = nn.ModuleList([AvenueMLP(cfg.d_model, cfg.dropout) for _ in range(self.num_avenues)])
        self.type_emb = nn.Parameter(torch.randn(self.num_avenues, cfg.d_model) * 0.02)
        self.gate = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, self.num_avenues),
        )
        self.merge = nn.Linear(cfg.d_model, cfg.d_model)
        self.scale = nn.Parameter(torch.tensor(float(cfg.avenue_scale_init)))

    def _raw_avenues(self, source: torch.Tensor) -> torch.Tensor:
        outputs = []
        for idx, avenue in enumerate(self.avenues):
            outputs.append(avenue(source + self.type_emb[idx].to(dtype=source.dtype).view(1, 1, -1)))
        return torch.stack(outputs, dim=2)

    def _training_mask(self, outputs: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.training or self.num_avenues <= 1:
            return None
        bsz = outputs.shape[0]
        device = outputs.device
        if self.cfg.avenue_dropout > 0:
            keep = torch.rand(bsz, self.num_avenues, device=device) >= float(self.cfg.avenue_dropout)
            all_dropped = ~keep.any(dim=-1)
            if bool(all_dropped.any()):
                chosen = torch.randint(0, self.num_avenues, (int(all_dropped.sum().item()),), device=device)
                keep[all_dropped] = False
                keep[all_dropped, chosen] = True
            return keep.float().view(bsz, 1, self.num_avenues, 1)
        return None

    def _apply_eval_control(self, outputs: torch.Tensor, control: Optional[str]) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if control is None or self.num_avenues <= 1:
            return outputs, None
        if control == "avenue_zero":
            return torch.zeros_like(outputs), None
        if control == "avenue_random":
            scale = outputs.detach().float().std().to(dtype=outputs.dtype).clamp_min(1.0e-3)
            return torch.randn_like(outputs) * scale, None
        if control == "avenue_shuffle" and outputs.shape[0] > 1:
            perm = torch.randperm(outputs.shape[0], device=outputs.device)
            return outputs.index_select(0, perm), None
        if control == "avenue_order_shuffle":
            perm = torch.randperm(self.num_avenues, device=outputs.device)
            return outputs.index_select(2, perm), None
        if control.startswith("single_avenue_"):
            idx = max(0, min(self.num_avenues - 1, int(control.rsplit("_", 1)[-1])))
            mask = torch.zeros(1, 1, self.num_avenues, 1, dtype=outputs.dtype, device=outputs.device)
            mask[:, :, idx, :] = 1.0
            return outputs * mask, mask
        if control.startswith("leave_one_"):
            idx = max(0, min(self.num_avenues - 1, int(control.rsplit("_", 1)[-1])))
            mask = torch.ones(1, 1, self.num_avenues, 1, dtype=outputs.dtype, device=outputs.device)
            mask[:, :, idx, :] = 0.0
            return outputs * mask, mask
        return outputs, None

    def forward(
        self,
        x: torch.Tensor,
        *,
        control: Optional[str] = None,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        source = x
        if control == "hidden_state_shuffle" and x.shape[0] > 1:
            perm = torch.randperm(x.shape[0], device=x.device)
            source = x.index_select(0, perm)
        outputs = self._raw_avenues(source)

        train_mask = self._training_mask(outputs)
        if train_mask is not None:
            outputs = outputs * train_mask.to(dtype=outputs.dtype)

        outputs, eval_mask = self._apply_eval_control(outputs, control)
        if self.num_avenues == 1:
            gates = torch.ones(x.shape[0], x.shape[1], 1, dtype=x.dtype, device=x.device)
        elif eval_mask is not None and (control or "").startswith("single_avenue_"):
            gates = eval_mask[..., 0].expand(x.shape[0], x.shape[1], -1)
        else:
            gate_logits = self.gate(x)
            gates = F.softmax(gate_logits.float(), dim=-1).to(dtype=x.dtype)
            active = (outputs.float().norm(dim=-1) > 0).float().to(dtype=x.dtype)
            gates = gates * active
            gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)

        mix = torch.einsum("bta,btad->btd", gates, outputs)
        enriched = x + self.scale.to(dtype=x.dtype) * self.merge(mix)
        sampled = outputs[:, :: max(1, int(self.cfg.align_stride))].reshape(-1, self.num_avenues, self.cfg.d_model)
        pairwise_cos = _offdiag_mean_cosine(sampled)
        stats: Dict[str, torch.Tensor] = {
            "avenue_gate_entropy": _entropy(gates.float(), dim=-1).mean(),
            "avenue_output_norm": outputs.float().norm(dim=-1).mean(),
            "pairwise_avenue_cosine": pairwise_cos,
            "avenue_diversity": 1.0 - pairwise_cos,
            "avenue_scale": self.scale.detach().float(),
        }
        usage = gates.float().mean(dim=(0, 1))
        for idx in range(self.num_avenues):
            stats[f"avenue_usage_{idx}"] = usage[idx]
        for idx in range(self.num_avenues, 4):
            stats[f"avenue_usage_{idx}"] = x.new_zeros(())

        diagnostics = None
        if return_diagnostics:
            diagnostics = {"gates": gates.detach(), "avenues": outputs.detach()}
        return enriched, outputs, stats, diagnostics


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.adapter = (
            ParamMatchedAdapter(cfg.d_model, cfg.param_adapter_dim, cfg.dropout)
            if cfg.variant == "param_matched_baseline"
            else None
        )
        self.avenue = E8ReferenceAvenueModule(cfg, layer_idx) if cfg.variant in E8_REFERENCE_VARIANTS else None

    def forward(
        self,
        x: torch.Tensor,
        *,
        control: Optional[str],
        return_diagnostics: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        attn_out, stats = self.attn(self.ln1(x), return_diagnostics=return_diagnostics)
        x = x + attn_out
        mlp_in = self.ln2(x)
        mlp_out = self.mlp(mlp_in)
        if self.adapter is not None:
            mlp_out = mlp_out + self.adapter(mlp_in)
        x = x + mlp_out
        avenue_outputs: Optional[torch.Tensor] = None
        avenue_diag: Optional[Dict[str, torch.Tensor]] = None
        if self.avenue is not None:
            x, avenue_outputs, avenue_stats, avenue_diag = self.avenue(
                x,
                control=control,
                return_diagnostics=return_diagnostics,
            )
            stats.update(avenue_stats)
        return x, avenue_outputs, stats, avenue_diag


class AvenueTransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor, *, return_diagnostics: bool) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        attn_out, stats = self.attn(self.ln1(x), return_diagnostics=return_diagnostics)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, stats


class PrefixJunction(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.junction_tokens = cfg.junction_tokens
        self.learned_tokens = nn.Parameter(torch.randn(cfg.junction_tokens, cfg.d_model) * 0.02)
        self.context_proj = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def _attention_bias(self, seq_len: int, num_avenues: int, query_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        q_time = torch.arange(seq_len, device=device).repeat_interleave(self.junction_tokens)
        key_time = torch.arange(seq_len, device=device).repeat(num_avenues)
        allowed = key_time.view(1, -1) <= q_time.view(-1, 1)
        bias = torch.zeros(query_len, num_avenues * seq_len, dtype=dtype, device=device)
        return bias.masked_fill(~allowed, torch.finfo(dtype).min)

    def forward(
        self,
        avenue_states: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        bsz, num_avenues, seq_len, d_model = avenue_states.shape
        context = avenue_states.mean(dim=1)
        query = self.context_proj(context).unsqueeze(2) + self.learned_tokens.to(dtype=avenue_states.dtype).view(1, 1, self.junction_tokens, d_model)
        query = query.reshape(bsz, seq_len * self.junction_tokens, d_model)
        keys = avenue_states.reshape(bsz, num_avenues * seq_len, d_model)
        query_len = query.shape[1]
        key_len = keys.shape[1]

        q = self.q_proj(query).view(bsz, query_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(keys).view(bsz, key_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(keys).view(bsz, key_len, self.n_heads, self.d_head).transpose(1, 2)
        bias = self._attention_bias(seq_len, num_avenues, query_len, q.dtype, q.device).view(1, 1, query_len, key_len)

        diagnostics = None
        if return_diagnostics:
            logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)
            weights = F.softmax((logits + bias).float(), dim=-1).to(dtype=q.dtype)
            attn = self.dropout(weights) if self.training else weights
            out = torch.matmul(attn, v)
            source_scores = weights.float().view(bsz, self.n_heads, seq_len, self.junction_tokens, num_avenues, seq_len)
            source_scores = source_scores.mean(dim=(1, 3, 4)).detach()
            diagnostics = {"junction_source_scores": source_scores}
        else:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=bias,
                dropout_p=float(self.cfg.dropout) if self.training else 0.0,
                is_causal=False,
            )
            weights = None

        out = out.transpose(1, 2).contiguous().view(bsz, query_len, d_model)
        out = self.out_proj(out).view(bsz, seq_len, self.junction_tokens, d_model)
        sampled = out[:, :: max(1, int(self.cfg.align_stride))].reshape(-1, self.junction_tokens, d_model)
        diversity = 1.0 - _offdiag_mean_cosine(sampled)
        energy = out.float().norm(dim=-1).mean(dim=(0, 1))
        usage = energy / energy.sum().clamp_min(1.0e-6)
        utilization = _entropy(usage, dim=-1) / math.log(max(2, self.junction_tokens))
        if weights is None:
            entropy = out.new_zeros(())
            effective = out.new_zeros(())
        else:
            entropy_values = _entropy(weights.float(), dim=-1)
            entropy = entropy_values.mean()
            effective = entropy_values.exp().mean()
        stats = {
            "junction_attention_entropy": entropy,
            "avenue_to_junction_attention_entropy": entropy,
            "junction_effective_source_tokens": effective,
            "junction_token_utilization": utilization,
            "junction_norm": out.float().norm(dim=-1).mean(),
            "junction_diversity": diversity,
        }
        return out, stats, diagnostics


class JunctionRead(nn.Module):
    def __init__(self, cfg: TransformerConfig, *, scale_init: Optional[float] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.scale = nn.Parameter(torch.tensor(float(cfg.junction_scale_init if scale_init is None else scale_init)))
        self.gate = (
            nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, 1))
            if cfg.cross_gated
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        junction: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        bsz, seq_len, junction_tokens, d_model = junction.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.d_head)
        k = self.k_proj(junction).view(bsz, seq_len, junction_tokens, self.n_heads, self.d_head)
        v = self.v_proj(junction).view(bsz, seq_len, junction_tokens, self.n_heads, self.d_head)
        logits = torch.einsum("bthd,btjhd->bthj", q, k) / math.sqrt(self.d_head)
        weights = F.softmax(logits.float(), dim=-1).to(dtype=x.dtype)
        context = torch.einsum("bthj,btjhd->bthd", weights, v).reshape(bsz, seq_len, d_model)
        delta = self.out_proj(context)
        if self.gate is not None:
            gate = torch.sigmoid(self.gate(x)).to(dtype=x.dtype)
        else:
            gate = torch.ones(bsz, seq_len, 1, dtype=x.dtype, device=x.device)
        scaled = self.scale.to(dtype=x.dtype) * gate * delta
        stats = {
            "junction_to_avenue_attention_entropy": _entropy(weights.float(), dim=-1).mean(),
            "junction_contribution_norm": scaled.float().norm(dim=-1).mean(),
            "junction_read_gate": gate.float().mean(),
            "junction_read_scale": self.scale.detach().float(),
        }
        return x + scaled, scaled, stats


class JunctionAvenueTransformer(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_avenues = int(cfg.num_avenues)
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.type_emb = nn.Parameter(torch.randn(self.num_avenues, cfg.d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [nn.ModuleList([AvenueTransformerBlock(cfg) for _ in range(self.num_avenues)]) for _ in range(cfg.n_layers)]
        )
        self.junctions = nn.ModuleList([PrefixJunction(cfg) for _ in range(cfg.n_layers)])
        self.reads = nn.ModuleList(
            [nn.ModuleList([JunctionRead(cfg) for _ in range(self.num_avenues)]) for _ in range(cfg.n_layers)]
        )
        self.final_junction = PrefixJunction(cfg)
        self.final_read = JunctionRead(cfg, scale_init=cfg.final_junction_scale_init)
        self.final_fuse = nn.Sequential(
            nn.LayerNorm(2 * cfg.d_model),
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def junction_points(self) -> set[int]:
        if self.cfg.variant == "junction_middle_bottleneck":
            return {max(0, self.cfg.n_layers // 2 - 1)}
        if self.cfg.variant == "junction_every_2_layers_bottleneck":
            return {idx for idx in range(self.cfg.n_layers) if (idx + 1) % 2 == 0}
        return set(range(self.cfg.n_layers))

    def _apply_training_avenue_mask(self, states: torch.Tensor) -> torch.Tensor:
        if not self.training or self.num_avenues <= 1:
            return states
        bsz = states.shape[0]
        device = states.device
        keep = torch.ones(bsz, self.num_avenues, dtype=states.dtype, device=device)
        if self.cfg.avenue_dropout > 0:
            keep = keep * (torch.rand(bsz, self.num_avenues, device=device) >= float(self.cfg.avenue_dropout)).to(dtype=states.dtype)
        if self.cfg.leave_one_prob > 0:
            should_drop = torch.rand(bsz, device=device) < float(self.cfg.leave_one_prob)
            if bool(should_drop.any()):
                drop_idx = torch.randint(0, self.num_avenues, (int(should_drop.sum().item()),), device=device)
                keep[should_drop, drop_idx] = 0.0
        all_dropped = keep.sum(dim=-1) <= 0
        if bool(all_dropped.any()):
            chosen = torch.randint(0, self.num_avenues, (int(all_dropped.sum().item()),), device=device)
            keep[all_dropped] = 0.0
            keep[all_dropped, chosen] = 1.0
        return states * keep.view(bsz, self.num_avenues, 1, 1)

    def _apply_eval_avenue_control(self, states: torch.Tensor, control: Optional[str]) -> torch.Tensor:
        if control is None or self.num_avenues <= 1:
            return states
        if control == "avenue_zero":
            return torch.zeros_like(states)
        if control == "avenue_random":
            scale = states.detach().float().std().to(dtype=states.dtype).clamp_min(1.0e-3)
            return torch.randn_like(states) * scale
        if control == "avenue_shuffle" and states.shape[0] > 1:
            perm = torch.randperm(states.shape[0], device=states.device)
            return states.index_select(0, perm)
        if control == "avenue_order_shuffle":
            perm = torch.randperm(self.num_avenues, device=states.device)
            return states.index_select(1, perm)
        if control == "hidden_state_shuffle" and states.shape[0] > 1:
            perm = torch.randperm(states.shape[0], device=states.device)
            return states.index_select(0, perm)
        if control.startswith("single_avenue_"):
            idx = max(0, min(self.num_avenues - 1, int(control.rsplit("_", 1)[-1])))
            mask = torch.zeros(1, self.num_avenues, 1, 1, dtype=states.dtype, device=states.device)
            mask[:, idx, :, :] = 1.0
            return states * mask
        if control.startswith("leave_one_"):
            idx = max(0, min(self.num_avenues - 1, int(control.rsplit("_", 1)[-1])))
            mask = torch.ones(1, self.num_avenues, 1, 1, dtype=states.dtype, device=states.device)
            mask[:, idx, :, :] = 0.0
            return states * mask
        return states

    def _apply_junction_control(self, junction: torch.Tensor, control: Optional[str], layer_idx: int) -> torch.Tensor:
        if control is None:
            return junction
        if control == "junction_zero" or control == f"junction_point_zero_{layer_idx}":
            return torch.zeros_like(junction)
        if control == "junction_random":
            scale = junction.detach().float().std().to(dtype=junction.dtype).clamp_min(1.0e-3)
            return torch.randn_like(junction) * scale
        if control == "junction_shuffle" and junction.shape[0] > 1:
            perm = torch.randperm(junction.shape[0], device=junction.device)
            return junction.index_select(0, perm)
        return junction

    def _avenue_stats(self, states: torch.Tensor) -> Dict[str, torch.Tensor]:
        sampled = states[:, :, :: max(1, int(self.cfg.align_stride))].permute(0, 2, 1, 3).reshape(-1, self.num_avenues, self.cfg.d_model)
        pairwise_cos = _offdiag_mean_cosine(sampled)
        energy = states.float().norm(dim=-1).mean(dim=2)
        usage = energy / energy.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        stats: Dict[str, torch.Tensor] = {
            "avenue_gate_entropy": _entropy(usage, dim=-1).mean(),
            "avenue_output_norm": states.float().norm(dim=-1).mean(),
            "pairwise_avenue_cosine": pairwise_cos,
            "avenue_diversity": 1.0 - pairwise_cos,
            "avenue_scale": states.new_ones(()),
        }
        usage_mean = usage.mean(dim=0)
        for idx in range(self.num_avenues):
            stats[f"avenue_usage_{idx}"] = usage_mean[idx]
        for idx in range(self.num_avenues, 4):
            stats[f"avenue_usage_{idx}"] = states.new_zeros(())
        return stats

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        return_diagnostics: bool = False,
        control: Optional[str] = None,
    ) -> Dict[str, Any]:
        bsz, seq_len = input_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.cfg.max_seq_len}")
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        base = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))
        states = torch.stack(
            [base + self.type_emb[idx].to(dtype=base.dtype).view(1, 1, -1) for idx in range(self.num_avenues)],
            dim=1,
        )

        layer_stats: List[Dict[str, torch.Tensor]] = []
        avenue_states: List[torch.Tensor] = []
        junction_diags: List[Optional[Dict[str, torch.Tensor]]] = []
        points = self.junction_points()
        last_junction: Optional[torch.Tensor] = None

        for layer_idx, avenue_blocks in enumerate(self.blocks):
            new_states = []
            attn_stats: List[torch.Tensor] = []
            attended_stats: List[torch.Tensor] = []
            sdpa_stats: List[torch.Tensor] = []
            for avenue_idx, block in enumerate(avenue_blocks):
                out, stats = block(states[:, avenue_idx], return_diagnostics=return_diagnostics)
                new_states.append(out)
                if "attention_entropy" in stats:
                    attn_stats.append(stats["attention_entropy"])
                if "effective_attended_tokens" in stats:
                    attended_stats.append(stats["effective_attended_tokens"])
                if "sdpa_used" in stats:
                    sdpa_stats.append(stats["sdpa_used"])
            states = torch.stack(new_states, dim=1)
            states = self._apply_training_avenue_mask(states)
            states = self._apply_eval_avenue_control(states, control)

            stats = self._avenue_stats(states)
            stats["attention_entropy"] = _mean_tensor(attn_stats, states)
            stats["effective_attended_tokens"] = _mean_tensor(attended_stats, states)
            stats["sdpa_used"] = _mean_tensor(sdpa_stats, states)
            diag: Optional[Dict[str, torch.Tensor]] = None

            if layer_idx in points:
                junction, junction_stats, diag = self.junctions[layer_idx](states, return_diagnostics=return_diagnostics)
                junction = self._apply_junction_control(junction, control, layer_idx)
                last_junction = junction
                read_stats: List[Dict[str, torch.Tensor]] = []
                read_states = []
                for avenue_idx, reader in enumerate(self.reads[layer_idx]):
                    read_state, _delta, read_stat = reader(
                        states[:, avenue_idx],
                        junction,
                        return_diagnostics=return_diagnostics,
                    )
                    read_states.append(read_state)
                    read_stats.append(read_stat)
                states = torch.stack(read_states, dim=1)
                stats.update(junction_stats)
                for key in (
                    "junction_to_avenue_attention_entropy",
                    "junction_contribution_norm",
                    "junction_read_gate",
                    "junction_read_scale",
                ):
                    stats[key] = torch.stack([item[key].float() for item in read_stats]).mean()
            else:
                zero = states.new_zeros(())
                stats.update(
                    {
                        "junction_attention_entropy": zero,
                        "avenue_to_junction_attention_entropy": zero,
                        "junction_to_avenue_attention_entropy": zero,
                        "junction_effective_source_tokens": zero,
                        "junction_token_utilization": zero,
                        "junction_norm": zero,
                        "junction_diversity": zero,
                        "junction_contribution_norm": zero,
                        "junction_read_gate": zero,
                        "junction_read_scale": zero,
                    }
                )
            avenue_states.append(states.transpose(1, 2))
            junction_diags.append(diag)
            layer_stats.append(stats)

        x = states.mean(dim=1)
        if self.cfg.force_junction_output:
            if last_junction is None or (self.cfg.n_layers - 1) not in points:
                final_junction, final_stats, final_diag = self.final_junction(states, return_diagnostics=return_diagnostics)
                final_junction = self._apply_junction_control(final_junction, control, self.cfg.n_layers)
                if layer_stats:
                    for key, value in final_stats.items():
                        layer_stats[-1][f"final_{key}"] = value
                if return_diagnostics:
                    junction_diags.append(final_diag)
            else:
                final_junction = last_junction
            enriched, delta, _stats = self.final_read(x, final_junction, return_diagnostics=return_diagnostics)
            x = self.final_fuse(torch.cat([x, enriched - x + delta], dim=-1))

        logits = self.lm_head(self.ln_f(x))
        div_loss = self._diversity_loss(avenue_states)
        return {
            "logits": logits,
            "layer_stats": layer_stats,
            "avenue_attn": None,
            "junction_attn": junction_diags if return_diagnostics else None,
            "avenue_states": avenue_states if return_diagnostics else None,
            "aux_losses": {
                "avenue_alignment_loss": self.token_emb.weight.new_zeros(()),
                "avenue_diversity_loss": div_loss,
            },
            "diagnostics": {
                "predicted_actual_avenue_cosine": self.token_emb.weight.new_zeros(()),
                "predicted_actual_avenue_cosine_by_pair": [],
            },
        }

    def _diversity_loss(self, avenue_states: Sequence[torch.Tensor]) -> torch.Tensor:
        if not avenue_states:
            return self.token_emb.weight.new_zeros(())
        stride = max(1, int(self.cfg.align_stride))
        losses = []
        for avenues in avenue_states:
            sampled = avenues[:, ::stride].reshape(-1, avenues.shape[2], avenues.shape[3])
            losses.append(_offdiag_mean_cosine(sampled).square())
        return torch.stack(losses).mean()

    def attention_kernel_summary(self) -> Dict[str, Any]:
        cuda_flags: Dict[str, Optional[bool]] = {
            "flash_sdp_enabled": None,
            "mem_efficient_sdp_enabled": None,
            "math_sdp_enabled": None,
        }
        if torch.cuda.is_available():
            cuda_flags = {
                "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
                "mem_efficient_sdp_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
                "math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
            }
        return {
            "last_attention_paths": [
                [block.attn._last_attention_path for block in layer] for layer in self.blocks
            ],
            "expected_training_paths": [["sdpa_causal" for _ in layer] for layer in self.blocks],
            "cuda_sdp_flags": cuda_flags,
            "avenue_variant": True,
            "junction_variant": True,
            "junction_points": sorted(self.junction_points()),
            "note": "Avenue streams use causal SDPA; junction tokens use causal prefix bottleneck attention over avenue states.",
        }


class MultiAvenueTransformer(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.variant in JUNCTION_VARIANTS:
            self.impl: nn.Module = JunctionAvenueTransformer(cfg)
            return
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg.n_layers)])
        self.avenue_predictors = nn.ModuleList(
            [nn.Linear(cfg.d_model, cfg.d_model) for _ in range(max(0, cfg.n_layers - 1))]
            if cfg.variant == "e8_best_reference"
            else []
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)
        for block in self.blocks:
            if block.adapter is not None:
                final = block.adapter.net[3]
                assert isinstance(final, nn.Linear)
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

    @property
    def has_avenues(self) -> bool:
        if hasattr(self, "impl"):
            return True
        return self.cfg.variant in AVENUE_VARIANTS

    @property
    def has_junctions(self) -> bool:
        return hasattr(self, "impl")

    @property
    def num_avenues(self) -> int:
        if hasattr(self, "impl"):
            return int(getattr(self.impl, "num_avenues", self.cfg.num_avenues))
        return 1 if self.cfg.variant == "single_avenue_reference" else int(self.cfg.num_avenues)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _alignment_loss(self, avenue_states: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        if len(avenue_states) < 2 or len(self.avenue_predictors) < len(avenue_states) - 1:
            zero = self.token_emb.weight.new_zeros(())
            return zero, zero, []
        stride = max(1, int(self.cfg.align_stride))
        losses: List[torch.Tensor] = []
        sims: List[torch.Tensor] = []
        for idx in range(len(avenue_states) - 1):
            current = avenue_states[idx][:, ::stride]
            target = avenue_states[idx + 1][:, ::stride]
            pred = self.avenue_predictors[idx](current)
            sim = F.cosine_similarity(pred.float(), target.detach().float(), dim=-1).mean()
            losses.append(1.0 - sim)
            sims.append(sim.detach())
        return torch.stack(losses).mean(), torch.stack(sims).mean(), sims

    def _diversity_loss(self, avenue_states: Sequence[torch.Tensor]) -> torch.Tensor:
        if not avenue_states:
            return self.token_emb.weight.new_zeros(())
        stride = max(1, int(self.cfg.align_stride))
        losses = []
        for avenues in avenue_states:
            sampled = avenues[:, ::stride].reshape(-1, avenues.shape[2], avenues.shape[3])
            losses.append(_offdiag_mean_cosine(sampled).square())
        return torch.stack(losses).mean()

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        return_diagnostics: bool = False,
        control: Optional[str] = None,
    ) -> Dict[str, Any]:
        if hasattr(self, "impl"):
            return self.impl(input_ids, return_diagnostics=return_diagnostics, control=control)
        bsz, seq_len = input_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.cfg.max_seq_len}")
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        layer_stats: List[Dict[str, torch.Tensor]] = []
        avenue_states: List[torch.Tensor] = []
        avenue_diags: List[Optional[Dict[str, torch.Tensor]]] = []
        for block in self.blocks:
            x, avenues, stats, diag = block(x, control=control, return_diagnostics=return_diagnostics)
            if avenues is not None:
                avenue_states.append(avenues)
            avenue_diags.append(diag)
            layer_stats.append(stats)

        logits = self.lm_head(self.ln_f(x))
        align_loss, align_sim, align_by_pair = self._alignment_loss(avenue_states)
        div_loss = self._diversity_loss(avenue_states)
        return {
            "logits": logits,
            "layer_stats": layer_stats,
            "avenue_attn": avenue_diags if return_diagnostics else None,
            "junction_attn": None,
            "avenue_states": avenue_states if return_diagnostics else None,
            "aux_losses": {
                "avenue_alignment_loss": align_loss,
                "avenue_diversity_loss": div_loss,
            },
            "diagnostics": {
                "predicted_actual_avenue_cosine": align_sim.detach(),
                "predicted_actual_avenue_cosine_by_pair": [item.detach() for item in align_by_pair],
            },
        }

    def attention_kernel_summary(self) -> Dict[str, Any]:
        if hasattr(self, "impl"):
            return self.impl.attention_kernel_summary()
        cuda_flags: Dict[str, Optional[bool]] = {
            "flash_sdp_enabled": None,
            "mem_efficient_sdp_enabled": None,
            "math_sdp_enabled": None,
        }
        if torch.cuda.is_available():
            cuda_flags = {
                "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
                "mem_efficient_sdp_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
                "math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
            }
        return {
            "last_attention_paths": [block.attn._last_attention_path for block in self.blocks],
            "expected_training_paths": ["sdpa_causal" for _ in self.blocks],
            "cuda_sdp_flags": cuda_flags,
            "avenue_variant": self.has_avenues,
            "junction_variant": False,
            "note": "E8-style references use per-token avenue MLP pathways; baselines use causal SDPA.",
        }


def build_model(raw_config: Dict[str, Any]) -> MultiAvenueTransformer:
    cfg = TransformerConfig.from_dict(raw_config)
    return MultiAvenueTransformer(cfg)


def parameter_count(model: nn.Module, *, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def model_summary(model: nn.Module) -> str:
    total = parameter_count(model)
    trainable = parameter_count(model, trainable_only=True)
    frozen = total - trainable
    return "\n".join(
        [
            repr(model),
            "",
            f"total_parameters: {total}",
            f"trainable_parameters: {trainable}",
            f"frozen_parameters: {frozen}",
        ]
    )
