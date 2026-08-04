from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from data import VOCAB_SIZE


VARIANTS = {
    "baseline_transformer",
    "param_matched_baseline",
    "single_avenue_reference",
    "e8_best_reference",
    "shared_subspace_crossing",
    "gated_subspace_crossing",
    "superposition_crossing",
    "rotation_crossing",
    "pairwise_crossing",
    "crossing_every_2_layers",
    "crossing_every_layer",
    "scout_best_custom",
    "matrix_interwoven_quick",
    "block_sparse_matrix_interweave_quick",
    "activation_pattern_attention_quick",
}

REFERENCE_AVENUE_VARIANTS = {"single_avenue_reference", "e8_best_reference"}
CROSSING_VARIANTS = {
    "shared_subspace_crossing",
    "gated_subspace_crossing",
    "superposition_crossing",
    "rotation_crossing",
    "pairwise_crossing",
    "crossing_every_2_layers",
    "crossing_every_layer",
    "scout_best_custom",
}
MATRIX_INTERWOVEN_VARIANTS = {"matrix_interwoven_quick"}
BLOCK_SPARSE_INTERWEAVE_VARIANTS = {"block_sparse_matrix_interweave_quick"}
PATTERN_ATTENTION_VARIANTS = {"activation_pattern_attention_quick"}
AVENUE_VARIANTS = (
    REFERENCE_AVENUE_VARIANTS
    | CROSSING_VARIANTS
    | MATRIX_INTERWOVEN_VARIANTS
    | BLOCK_SPARSE_INTERWEAVE_VARIANTS
)
CROSSING_KINDS = {"shared", "gated", "superposition", "rotation", "pairwise"}


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
    param_adapter_dim: int = 3072
    num_avenues: int = 4
    avenue_dim: int = 128
    avenue_dropout: float = 0.0
    leave_one_prob: float = 0.0
    align_stride: int = 8
    avenue_scale_init: float = 0.1
    d_cross: int = 64
    gamma_init: float = 0.1
    crossing_kind: str = "shared"
    crossing_interval: int = 2
    crossing_layers: Optional[Sequence[int]] = None
    crossing_layernorm: bool = True
    pair_summary_scale_init: float = 0.25
    interwoven_fused_layers: Optional[Sequence[int]] = None
    interwoven_fused_interval: int = 2
    block_sparse_layers: Optional[Sequence[int]] = None
    block_sparse_interval: int = 2
    block_sparse_rank: int = 16
    block_sparse_alpha_init: float = 0.05
    pattern_group_size: int = 8
    pattern_dim: int = 32
    pattern_heads: int = 4
    pattern_layers: Optional[Sequence[int]] = None
    pattern_interval: int = 1
    pattern_gamma_init: float = 0.1

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
        if cfg.variant in CROSSING_VARIANTS and cfg.num_avenues != 4:
            raise ValueError("E10 crossing scouts use exactly four avenues")
        if cfg.variant in MATRIX_INTERWOVEN_VARIANTS and cfg.num_avenues != 4:
            raise ValueError("E10 matrix-interwoven quick scout uses exactly four avenues")
        if cfg.variant in BLOCK_SPARSE_INTERWEAVE_VARIANTS and cfg.num_avenues != 4:
            raise ValueError("E10 block-sparse interweave quick scout uses exactly four avenues")
        if cfg.align_stride <= 0:
            raise ValueError("align_stride must be positive")
        if cfg.crossing_kind not in CROSSING_KINDS:
            raise ValueError(f"Unknown crossing_kind '{cfg.crossing_kind}'")
        if cfg.crossing_interval <= 0:
            raise ValueError("crossing_interval must be positive")
        if cfg.interwoven_fused_interval <= 0:
            raise ValueError("interwoven_fused_interval must be positive")
        if cfg.block_sparse_interval <= 0:
            raise ValueError("block_sparse_interval must be positive")
        if cfg.block_sparse_rank <= 0:
            raise ValueError("block_sparse_rank must be positive")
        if cfg.pattern_interval <= 0:
            raise ValueError("pattern_interval must be positive")
        if cfg.pattern_group_size <= 0 or cfg.d_model % cfg.pattern_group_size != 0:
            raise ValueError("d_model must be divisible by pattern_group_size")
        if cfg.pattern_dim % cfg.pattern_heads != 0:
            raise ValueError("pattern_dim must be divisible by pattern_heads")
        if cfg.d_cross <= 0:
            raise ValueError("d_cross must be positive")
        if cfg.variant == "gated_subspace_crossing":
            cfg.crossing_kind = "gated"
        elif cfg.variant == "superposition_crossing":
            cfg.crossing_kind = "superposition"
        elif cfg.variant == "rotation_crossing":
            cfg.crossing_kind = "rotation"
        elif cfg.variant == "pairwise_crossing":
            cfg.crossing_kind = "pairwise"
        if cfg.variant == "crossing_every_2_layers":
            cfg.crossing_kind = "shared"
            cfg.crossing_interval = 2
            cfg.crossing_layers = None
        elif cfg.variant == "crossing_every_layer":
            cfg.crossing_kind = "shared"
            cfg.crossing_interval = 1
            cfg.crossing_layers = None
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


def _sampled_pairwise(states: torch.Tensor, stride: int) -> Tuple[torch.Tensor, torch.Tensor]:
    sampled = states[:, :: max(1, int(stride))].reshape(-1, states.shape[2], states.shape[3])
    cosine = _offdiag_mean_cosine(sampled)
    return cosine, 1.0 - cosine


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
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
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
        return torch.stack(
            [
                avenue(source + self.type_emb[idx].to(dtype=source.dtype).view(1, 1, -1))
                for idx, avenue in enumerate(self.avenues)
            ],
            dim=2,
        )

    def _training_mask(self, outputs: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.training or self.num_avenues <= 1:
            return None
        bsz = outputs.shape[0]
        if self.cfg.avenue_dropout <= 0 and self.cfg.leave_one_prob <= 0:
            return None
        keep = torch.ones(bsz, self.num_avenues, dtype=outputs.dtype, device=outputs.device)
        if self.cfg.avenue_dropout > 0:
            keep = keep * (torch.rand(bsz, self.num_avenues, device=outputs.device) >= self.cfg.avenue_dropout).to(
                dtype=outputs.dtype
            )
        if self.cfg.leave_one_prob > 0:
            should_drop = torch.rand(bsz, device=outputs.device) < self.cfg.leave_one_prob
            if bool(should_drop.any()):
                drop_idx = torch.randint(0, self.num_avenues, (int(should_drop.sum().item()),), device=outputs.device)
                keep[should_drop, drop_idx] = 0.0
        all_dropped = keep.sum(dim=-1) <= 0
        if bool(all_dropped.any()):
            chosen = torch.randint(0, self.num_avenues, (int(all_dropped.sum().item()),), device=outputs.device)
            keep[all_dropped] = 0.0
            keep[all_dropped, chosen] = 1.0
        return keep.view(bsz, 1, self.num_avenues, 1)

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
            source = x.index_select(0, torch.randperm(x.shape[0], device=x.device))
        outputs = self._raw_avenues(source)
        train_mask = self._training_mask(outputs)
        if train_mask is not None:
            outputs = outputs * train_mask
        outputs, eval_mask = self._apply_eval_control(outputs, control)
        if self.num_avenues == 1:
            gates = torch.ones(x.shape[0], x.shape[1], 1, dtype=x.dtype, device=x.device)
        elif eval_mask is not None and (control or "").startswith("single_avenue_"):
            gates = eval_mask[..., 0].expand(x.shape[0], x.shape[1], -1)
        else:
            gates = F.softmax(self.gate(x).float(), dim=-1).to(dtype=x.dtype)
            active = (outputs.float().norm(dim=-1) > 0).float().to(dtype=x.dtype)
            gates = gates * active
            gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)

        mix = torch.einsum("bta,btad->btd", gates, outputs)
        enriched = x + self.scale.to(dtype=x.dtype) * self.merge(mix)
        pairwise_cos, diversity = _sampled_pairwise(outputs, self.cfg.align_stride)
        stats: Dict[str, torch.Tensor] = {
            "avenue_gate_entropy": _entropy(gates.float(), dim=-1).mean(),
            "avenue_output_norm": outputs.float().norm(dim=-1).mean(),
            "pairwise_avenue_cosine": pairwise_cos,
            "avenue_diversity": diversity,
            "avenue_scale": self.scale.detach().float(),
        }
        usage = gates.float().mean(dim=(0, 1))
        for idx in range(self.num_avenues):
            stats[f"avenue_usage_{idx}"] = usage[idx]
        for idx in range(self.num_avenues, 4):
            stats[f"avenue_usage_{idx}"] = x.new_zeros(())
        diagnostics = {"gates": gates.detach(), "avenues": outputs.detach()} if return_diagnostics else None
        return enriched, outputs, stats, diagnostics


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.adapter = ParamMatchedAdapter(cfg.d_model, cfg.param_adapter_dim, cfg.dropout) if cfg.variant == "param_matched_baseline" else None
        self.avenue = E8ReferenceAvenueModule(cfg) if cfg.variant in REFERENCE_AVENUE_VARIANTS else None

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
        avenue_states = None
        avenue_diag = None
        if self.avenue is not None:
            x, avenue_states, avenue_stats, avenue_diag = self.avenue(
                x,
                control=control,
                return_diagnostics=return_diagnostics,
            )
            stats.update(avenue_stats)
        return x, avenue_states, stats, avenue_diag


class ActivationPatternAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = int(layer_idx)
        self.group_size = int(cfg.pattern_group_size)
        self.num_groups = int(cfg.d_model // cfg.pattern_group_size)
        self.pattern_dim = int(cfg.pattern_dim)
        self.pattern_heads = int(cfg.pattern_heads)
        self.pattern_head_dim = self.pattern_dim // self.pattern_heads
        self.norm = nn.LayerNorm(cfg.d_model)
        self.in_proj = nn.Linear(self.group_size, self.pattern_dim)
        self.group_emb = nn.Parameter(torch.randn(self.num_groups, self.pattern_dim) * 0.02)
        self.qkv = nn.Linear(self.pattern_dim, 3 * self.pattern_dim)
        self.out_proj = nn.Linear(self.pattern_dim, self.group_size)
        self.dropout = nn.Dropout(cfg.dropout)
        self.gamma = nn.Parameter(torch.tensor(float(cfg.pattern_gamma_init)))

    def forward(
        self,
        x: torch.Tensor,
        *,
        control: Optional[str],
        return_diagnostics: bool,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        bsz, seq_len, dim = x.shape
        source = self.norm(x)
        groups = source.view(bsz * seq_len, self.num_groups, self.group_size)
        pattern = self.in_proj(groups) + self.group_emb.to(dtype=x.dtype).view(1, self.num_groups, self.pattern_dim)
        qkv = self.qkv(pattern)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz * seq_len, self.num_groups, self.pattern_heads, self.pattern_head_dim).transpose(1, 2)
        k = k.view(bsz * seq_len, self.num_groups, self.pattern_heads, self.pattern_head_dim).transpose(1, 2)
        v = v.view(bsz * seq_len, self.num_groups, self.pattern_heads, self.pattern_head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.pattern_head_dim)
        weights = F.softmax(logits.float(), dim=-1).to(dtype=x.dtype)
        attn = self.dropout(weights) if self.training else weights
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz * seq_len, self.num_groups, self.pattern_dim)
        delta = self.out_proj(out).view(bsz, seq_len, dim)
        if control == "pattern_shuffle" and delta.shape[0] > 1:
            delta = delta.index_select(0, torch.randperm(delta.shape[0], device=delta.device))
        if control == "pattern_random":
            scale = delta.detach().float().std().to(dtype=delta.dtype).clamp_min(1.0e-3)
            delta = torch.randn_like(delta) * scale
        contribution = self.gamma.to(dtype=x.dtype) * delta
        if control == "pattern_zero" or control == f"pattern_point_zero_{self.layer_idx}":
            contribution = torch.zeros_like(contribution)
        after = x + contribution
        hidden_norm = x.float().norm(dim=-1).mean()
        delta_norm = contribution.float().norm(dim=-1).mean()
        entropy = _entropy(weights.float(), dim=-1).mean()
        stats: Dict[str, torch.Tensor] = {
            "pattern_attention_applied": x.new_ones(()),
            "pattern_attention_gamma": self.gamma.detach().float(),
            "pattern_attention_entropy": entropy,
            "pattern_attention_delta_norm": delta_norm,
            "pattern_attention_hidden_norm": hidden_norm,
            "pattern_attention_delta_ratio": delta_norm / hidden_norm.clamp_min(1.0e-6),
        }
        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "attention_entropy": entropy.detach(),
                "score": contribution.detach().float().norm(dim=-1),
            }
        return after, stats, diagnostics


class PatternAttentionTransformer(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.patterns = nn.ModuleList([ActivationPatternAttention(cfg, idx) for idx in range(cfg.n_layers)])
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

    def pattern_points(self) -> set[int]:
        if self.cfg.pattern_layers is not None:
            return {int(idx) for idx in self.cfg.pattern_layers if 0 <= int(idx) < self.cfg.n_layers}
        return {idx for idx in range(self.cfg.n_layers) if (idx + 1) % int(self.cfg.pattern_interval) == 0}

    def _empty_pattern_stats(self, x: torch.Tensor, stats: Dict[str, torch.Tensor]) -> None:
        zero = x.new_zeros(())
        stats.update(
            {
                "pattern_attention_applied": zero,
                "pattern_attention_gamma": zero,
                "pattern_attention_entropy": zero,
                "pattern_attention_delta_norm": zero,
                "pattern_attention_hidden_norm": x.float().norm(dim=-1).mean(),
                "pattern_attention_delta_ratio": zero,
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        return_diagnostics: bool = False,
        control: Optional[str] = None,
    ) -> Dict[str, Any]:
        _bsz, seq_len = input_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.cfg.max_seq_len}")
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))
        layer_stats: List[Dict[str, torch.Tensor]] = []
        pattern_diags: List[Optional[Dict[str, torch.Tensor]]] = []
        points = self.pattern_points()
        for layer_idx, block in enumerate(self.blocks):
            x, _avenues, stats, _diag = block(x, control=control, return_diagnostics=return_diagnostics)
            pattern_diag = None
            if layer_idx in points:
                x, pattern_stats, pattern_diag = self.patterns[layer_idx](
                    x,
                    control=control,
                    return_diagnostics=return_diagnostics,
                )
                stats.update(pattern_stats)
            else:
                self._empty_pattern_stats(x, stats)
            layer_stats.append(stats)
            pattern_diags.append(pattern_diag)
        logits = self.lm_head(self.ln_f(x))
        return {
            "logits": logits,
            "layer_stats": layer_stats,
            "avenue_attn": None,
            "crossing_diags": None,
            "pattern_diags": pattern_diags if return_diagnostics else None,
            "avenue_states": None,
            "aux_losses": {
                "avenue_alignment_loss": self.token_emb.weight.new_zeros(()),
                "avenue_diversity_loss": self.token_emb.weight.new_zeros(()),
                "crossing_orthogonality_loss": self.token_emb.weight.new_zeros(()),
            },
            "diagnostics": {
                "predicted_actual_avenue_cosine": self.token_emb.weight.new_zeros(()),
                "predicted_actual_avenue_cosine_by_pair": [],
            },
        }

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
            "last_attention_paths": [block.attn._last_attention_path for block in self.blocks],
            "expected_training_paths": ["sdpa_causal" for _ in self.blocks],
            "cuda_sdp_flags": cuda_flags,
            "avenue_variant": False,
            "crossing_variant": False,
            "pattern_attention_variant": True,
            "pattern_points": sorted(self.pattern_points()),
            "note": "Decoder-only transformer with a second attention over hidden feature-group activation patterns inside each token.",
        }


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


class ActivationCrossing(nn.Module):
    def __init__(self, cfg: TransformerConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.num_avenues = int(cfg.num_avenues)
        self.norm = nn.LayerNorm(cfg.d_model) if cfg.crossing_layernorm else nn.Identity()
        self.in_proj = nn.ModuleList([nn.Linear(cfg.d_model, cfg.d_cross, bias=False) for _ in range(self.num_avenues)])
        self.out_proj = nn.ModuleList([nn.Linear(cfg.d_cross, cfg.d_model, bias=False) for _ in range(self.num_avenues)])
        self.gamma = nn.Parameter(torch.full((self.num_avenues,), float(cfg.gamma_init)))
        self.mix = nn.Parameter(torch.eye(self.num_avenues) + 0.02 * torch.randn(self.num_avenues, self.num_avenues))
        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(cfg.d_model),
                    nn.Linear(cfg.d_model, cfg.d_model),
                    nn.GELU(),
                    nn.Linear(cfg.d_model, self.num_avenues),
                )
                for _ in range(self.num_avenues)
            ]
        )
        self.pair_mix = nn.Parameter(torch.eye(2) + 0.02 * torch.randn(2, 2))
        self.pair_bridge = nn.Parameter(torch.eye(2) + 0.02 * torch.randn(2, 2))
        self.pair_summary_scale = nn.Parameter(torch.tensor(float(cfg.pair_summary_scale_init)))

    def reset_rotation_parameters(self) -> None:
        if self.cfg.crossing_kind != "rotation":
            return
        for module in list(self.in_proj) + list(self.out_proj):
            nn.init.orthogonal_(module.weight)

    def _project_in(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        entering = self.norm(states)
        z = torch.stack([proj(entering[:, :, idx]) for idx, proj in enumerate(self.in_proj)], dim=2)
        return z, entering

    def _project_out(self, z: torch.Tensor) -> torch.Tensor:
        return torch.stack([proj(z[:, :, idx]) for idx, proj in enumerate(self.out_proj)], dim=2)

    def _pair_matrix(self, ref: torch.Tensor) -> torch.Tensor:
        intra = ref.new_zeros((self.num_avenues, self.num_avenues))
        intra[:2, :2] = self.pair_mix.to(dtype=ref.dtype)
        intra[2:, 2:] = self.pair_mix.to(dtype=ref.dtype)
        summarize = ref.new_tensor([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]])
        broadcast = ref.new_tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        bridge = broadcast @ self.pair_bridge.to(dtype=ref.dtype) @ summarize @ intra
        return intra + self.pair_summary_scale.to(dtype=ref.dtype) * bridge

    def _mix_latents(
        self,
        z: torch.Tensor,
        entering: torch.Tensor,
        *,
        identity: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        eye = torch.eye(self.num_avenues, dtype=z.dtype, device=z.device)
        zero = z.new_zeros(())
        if identity:
            return z, eye, zero, eye.float().mean(dim=0)
        if self.cfg.crossing_kind == "gated":
            logits = torch.stack([gate(entering[:, :, idx]) for idx, gate in enumerate(self.gates)], dim=2)
            gates = F.softmax(logits.float(), dim=-1).to(dtype=z.dtype)
            mixed = torch.einsum("btas,btsd->btad", gates, z)
            matrix = gates.float().mean(dim=(0, 1)).to(dtype=z.dtype)
            entropy = _entropy(gates.float(), dim=-1).mean()
            source_usage = gates.float().mean(dim=(0, 1, 2))
            return mixed, matrix, entropy, source_usage
        if self.cfg.crossing_kind == "superposition":
            mixed = z.mean(dim=2, keepdim=True).expand_as(z)
            matrix = torch.full((self.num_avenues, self.num_avenues), 1.0 / self.num_avenues, dtype=z.dtype, device=z.device)
            return mixed, matrix, zero, matrix.float().mean(dim=0)
        if self.cfg.crossing_kind == "pairwise":
            first = torch.empty_like(z)
            first[:, :, :2] = torch.einsum("btad,ca->btcd", z[:, :, :2], self.pair_mix.to(dtype=z.dtype))
            first[:, :, 2:] = torch.einsum("btad,ca->btcd", z[:, :, 2:], self.pair_mix.to(dtype=z.dtype))
            pair_summary = torch.stack([first[:, :, :2].mean(dim=2), first[:, :, 2:].mean(dim=2)], dim=2)
            bridge = torch.einsum("btpd,qp->btqd", pair_summary, self.pair_bridge.to(dtype=z.dtype))
            first[:, :, :2] = first[:, :, :2] + self.pair_summary_scale.to(dtype=z.dtype) * bridge[:, :, :1]
            first[:, :, 2:] = first[:, :, 2:] + self.pair_summary_scale.to(dtype=z.dtype) * bridge[:, :, 1:]
            matrix = self._pair_matrix(z)
            return first, matrix, zero, matrix.float().abs().mean(dim=0)
        matrix = self.mix.to(dtype=z.dtype)
        mixed = torch.einsum("btad,ca->btcd", z, matrix)
        return mixed, matrix, zero, matrix.float().abs().mean(dim=0)

    def orthogonality_penalty(self) -> torch.Tensor:
        if self.cfg.crossing_kind != "rotation":
            return self.gamma.new_zeros(())
        penalties: List[torch.Tensor] = []
        for proj in self.in_proj:
            weight = proj.weight.float()
            gram = weight @ weight.transpose(0, 1)
            eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
            penalties.append((gram - eye).square().mean())
        return torch.stack(penalties).mean()

    def forward(
        self,
        states: torch.Tensor,
        *,
        control: Optional[str],
        return_diagnostics: bool,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        crossing_source = states
        if control == "hidden_state_shuffle" and states.shape[0] > 1:
            crossing_source = states.index_select(0, torch.randperm(states.shape[0], device=states.device))
        z, entering = self._project_in(crossing_source)
        identity = control in {"crossing_identity", "crossing_disable_mixing"}
        z_mix, mix_matrix, gate_entropy, source_usage = self._mix_latents(z, entering, identity=identity)
        if control == "crossing_shuffle" and z_mix.shape[0] > 1:
            z_mix = z_mix.index_select(0, torch.randperm(z_mix.shape[0], device=z_mix.device))
        if control == "crossing_random":
            scale = z_mix.detach().float().std().to(dtype=z_mix.dtype).clamp_min(1.0e-3)
            z_mix = torch.randn_like(z_mix) * scale
        delta = self._project_out(z_mix)
        contribution = self.gamma.to(dtype=states.dtype).view(1, 1, self.num_avenues, 1) * delta
        if control == "crossing_zero" or control == f"crossing_point_zero_{self.layer_idx}":
            contribution = torch.zeros_like(contribution)
        after = states + contribution

        before_cos, before_div = _sampled_pairwise(states, self.cfg.align_stride)
        after_cos, after_div = _sampled_pairwise(after, self.cfg.align_stride)
        latent_cos, latent_div = _sampled_pairwise(z_mix, self.cfg.align_stride)
        hidden_norm = states.float().norm(dim=-1).mean()
        contribution_norm = contribution.float().norm(dim=-1).mean()
        stats: Dict[str, torch.Tensor] = {
            "crossing_applied": states.new_ones(()),
            "crossing_gamma": self.gamma.detach().float().mean(),
            "crossing_contribution_norm": contribution_norm,
            "crossing_hidden_norm": hidden_norm,
            "crossing_contribution_ratio": contribution_norm / hidden_norm.clamp_min(1.0e-6),
            "crossing_activation_norm": z_mix.float().norm(dim=-1).mean(),
            "crossing_latent_pairwise_cosine": latent_cos,
            "crossing_diversity": latent_div,
            "crossing_gate_entropy": gate_entropy.float(),
            "avenue_diversity_before_crossing": before_div,
            "avenue_diversity_after_crossing": after_div,
            "pairwise_avenue_cosine_before_crossing": before_cos,
            "pairwise_avenue_cosine_after_crossing": after_cos,
        }
        for idx in range(self.num_avenues):
            stats[f"crossing_gamma_{idx}"] = self.gamma.detach().float()[idx]
            stats[f"crossing_source_usage_{idx}"] = source_usage.float()[idx]
            for source_idx in range(self.num_avenues):
                stats[f"crossing_mix_{idx}_{source_idx}"] = mix_matrix.detach().float()[idx, source_idx]

        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "score": contribution.detach().float().norm(dim=-1).mean(dim=2),
                "before_context": states.detach().float().mean(dim=2),
                "after_context": after.detach().float().mean(dim=2),
                "mix_matrix": mix_matrix.detach().float(),
            }
        return after, stats, diagnostics


class CrossingAvenueTransformer(nn.Module):
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
        self.crossings = nn.ModuleList([ActivationCrossing(cfg, idx) for idx in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)
        for crossing in self.crossings:
            crossing.reset_rotation_parameters()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def crossing_points(self) -> set[int]:
        if self.cfg.crossing_layers is not None:
            return {int(idx) for idx in self.cfg.crossing_layers if 0 <= int(idx) < self.cfg.n_layers}
        return {idx for idx in range(self.cfg.n_layers) if (idx + 1) % int(self.cfg.crossing_interval) == 0}

    def _training_avenue_mask(self, states: torch.Tensor) -> torch.Tensor:
        if not self.training or self.num_avenues <= 1:
            return states
        if self.cfg.avenue_dropout <= 0 and self.cfg.leave_one_prob <= 0:
            return states
        bsz = states.shape[0]
        keep = torch.ones(bsz, self.num_avenues, dtype=states.dtype, device=states.device)
        if self.cfg.avenue_dropout > 0:
            keep = keep * (torch.rand(bsz, self.num_avenues, device=states.device) >= self.cfg.avenue_dropout).to(
                dtype=states.dtype
            )
        if self.cfg.leave_one_prob > 0:
            should_drop = torch.rand(bsz, device=states.device) < self.cfg.leave_one_prob
            if bool(should_drop.any()):
                drop_idx = torch.randint(0, self.num_avenues, (int(should_drop.sum().item()),), device=states.device)
                keep[should_drop, drop_idx] = 0.0
        all_dropped = keep.sum(dim=-1) <= 0
        if bool(all_dropped.any()):
            chosen = torch.randint(0, self.num_avenues, (int(all_dropped.sum().item()),), device=states.device)
            keep[all_dropped] = 0.0
            keep[all_dropped, chosen] = 1.0
        return states * keep.view(bsz, 1, self.num_avenues, 1)

    def _eval_avenue_control(self, states: torch.Tensor, control: Optional[str]) -> torch.Tensor:
        if control is None:
            return states
        if control == "avenue_zero":
            return torch.zeros_like(states)
        if control == "avenue_random":
            scale = states.detach().float().std().to(dtype=states.dtype).clamp_min(1.0e-3)
            return torch.randn_like(states) * scale
        if control == "avenue_shuffle" and states.shape[0] > 1:
            return states.index_select(0, torch.randperm(states.shape[0], device=states.device))
        if control == "avenue_order_shuffle":
            return states.index_select(2, torch.randperm(self.num_avenues, device=states.device))
        if control.startswith("single_avenue_"):
            idx = max(0, min(self.num_avenues - 1, int(control.rsplit("_", 1)[-1])))
            mask = torch.zeros(1, 1, self.num_avenues, 1, dtype=states.dtype, device=states.device)
            mask[:, :, idx, :] = 1.0
            return states * mask
        if control.startswith("leave_one_"):
            idx = max(0, min(self.num_avenues - 1, int(control.rsplit("_", 1)[-1])))
            mask = torch.ones(1, 1, self.num_avenues, 1, dtype=states.dtype, device=states.device)
            mask[:, :, idx, :] = 0.0
            return states * mask
        return states

    def _avenue_stats(self, states: torch.Tensor) -> Dict[str, torch.Tensor]:
        pairwise_cos, diversity = _sampled_pairwise(states, self.cfg.align_stride)
        energy = states.float().norm(dim=-1).mean(dim=1)
        usage = energy / energy.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        stats: Dict[str, torch.Tensor] = {
            "avenue_gate_entropy": _entropy(usage, dim=-1).mean(),
            "avenue_output_norm": states.float().norm(dim=-1).mean(),
            "pairwise_avenue_cosine": pairwise_cos,
            "avenue_diversity": diversity,
            "avenue_scale": states.new_ones(()),
        }
        usage_mean = usage.mean(dim=0)
        for idx in range(self.num_avenues):
            stats[f"avenue_usage_{idx}"] = usage_mean[idx]
        return stats

    def _empty_crossing_stats(self, states: torch.Tensor, stats: Dict[str, torch.Tensor]) -> None:
        zero = states.new_zeros(())
        pairwise = stats["pairwise_avenue_cosine"]
        diversity = stats["avenue_diversity"]
        stats.update(
            {
                "crossing_applied": zero,
                "crossing_gamma": zero,
                "crossing_contribution_norm": zero,
                "crossing_hidden_norm": states.float().norm(dim=-1).mean(),
                "crossing_contribution_ratio": zero,
                "crossing_activation_norm": zero,
                "crossing_latent_pairwise_cosine": zero,
                "crossing_diversity": zero,
                "crossing_gate_entropy": zero,
                "avenue_diversity_before_crossing": diversity,
                "avenue_diversity_after_crossing": diversity,
                "pairwise_avenue_cosine_before_crossing": pairwise,
                "pairwise_avenue_cosine_after_crossing": pairwise,
            }
        )
        for idx in range(self.num_avenues):
            stats[f"crossing_gamma_{idx}"] = zero
            stats[f"crossing_source_usage_{idx}"] = zero
            for source_idx in range(self.num_avenues):
                stats[f"crossing_mix_{idx}_{source_idx}"] = zero

    def _diversity_loss(self, avenue_states: Sequence[torch.Tensor]) -> torch.Tensor:
        if not avenue_states:
            return self.token_emb.weight.new_zeros(())
        losses = [_sampled_pairwise(states, self.cfg.align_stride)[0].square() for states in avenue_states]
        return torch.stack(losses).mean()

    def _orthogonality_loss(self) -> torch.Tensor:
        losses = [crossing.orthogonality_penalty() for crossing in self.crossings if crossing.layer_idx in self.crossing_points()]
        return torch.stack(losses).mean() if losses else self.token_emb.weight.new_zeros(())

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
            dim=2,
        )
        points = self.crossing_points()
        layer_stats: List[Dict[str, torch.Tensor]] = []
        crossing_diags: List[Optional[Dict[str, torch.Tensor]]] = []
        avenue_states: List[torch.Tensor] = []

        for layer_idx, avenue_blocks in enumerate(self.blocks):
            new_states: List[torch.Tensor] = []
            attn_stats: List[torch.Tensor] = []
            attended_stats: List[torch.Tensor] = []
            sdpa_stats: List[torch.Tensor] = []
            for avenue_idx, block in enumerate(avenue_blocks):
                updated, block_stats = block(states[:, :, avenue_idx], return_diagnostics=return_diagnostics)
                new_states.append(updated)
                if "attention_entropy" in block_stats:
                    attn_stats.append(block_stats["attention_entropy"])
                if "effective_attended_tokens" in block_stats:
                    attended_stats.append(block_stats["effective_attended_tokens"])
                if "sdpa_used" in block_stats:
                    sdpa_stats.append(block_stats["sdpa_used"])
            states = torch.stack(new_states, dim=2)
            states = self._training_avenue_mask(states)
            states = self._eval_avenue_control(states, control)
            stats = self._avenue_stats(states)
            stats["attention_entropy"] = _mean_tensor(attn_stats, states)
            stats["effective_attended_tokens"] = _mean_tensor(attended_stats, states)
            stats["sdpa_used"] = _mean_tensor(sdpa_stats, states)
            diag = None
            if layer_idx in points:
                states, crossing_stats, diag = self.crossings[layer_idx](
                    states,
                    control=control,
                    return_diagnostics=return_diagnostics,
                )
                stats.update(crossing_stats)
            else:
                self._empty_crossing_stats(states, stats)
            avenue_states.append(states)
            layer_stats.append(stats)
            crossing_diags.append(diag)

        logits = self.lm_head(self.ln_f(states.mean(dim=2)))
        return {
            "logits": logits,
            "layer_stats": layer_stats,
            "avenue_attn": None,
            "crossing_diags": crossing_diags if return_diagnostics else None,
            "avenue_states": avenue_states if return_diagnostics else None,
            "aux_losses": {
                "avenue_alignment_loss": self.token_emb.weight.new_zeros(()),
                "avenue_diversity_loss": self._diversity_loss(avenue_states),
                "crossing_orthogonality_loss": self._orthogonality_loss(),
            },
            "diagnostics": {
                "predicted_actual_avenue_cosine": self.token_emb.weight.new_zeros(()),
                "predicted_actual_avenue_cosine_by_pair": [],
            },
        }

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
            "last_attention_paths": [[block.attn._last_attention_path for block in layer] for layer in self.blocks],
            "expected_training_paths": [["sdpa_causal" for _block in layer] for layer in self.blocks],
            "cuda_sdp_flags": cuda_flags,
            "avenue_variant": True,
            "crossing_variant": True,
            "crossing_kind": self.cfg.crossing_kind,
            "crossing_points": sorted(self.crossing_points()),
            "note": "Private avenue streams use causal SDPA; E10 crossing is per-token shared-subspace feature mixing.",
        }


class WideSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("wide d_model must be divisible by n_heads")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_head = self.d_model // self.n_heads
        self.dropout_p = float(dropout)
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(dropout)
        self._last_attention_path = "uninitialized"

    def forward(self, x: torch.Tensor, *, return_diagnostics: bool) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
                dropout_p=self.dropout_p if self.training else 0.0,
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
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        token_entropy = _entropy(weights.float(), dim=-1)
        return self.out_proj(out), {
            "attention_entropy": token_entropy.mean(),
            "effective_attended_tokens": token_entropy.exp().mean(),
            "sdpa_used": x.new_zeros(()),
        }


class WideFeedForward(nn.Module):
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


class FusedAvenueTransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.num_avenues = int(cfg.num_avenues)
        self.d_fused = int(cfg.num_avenues * cfg.d_model)
        self.ln1 = nn.LayerNorm(self.d_fused)
        self.attn = WideSelfAttention(self.d_fused, cfg.n_heads * cfg.num_avenues, cfg.dropout)
        self.ln2 = nn.LayerNorm(self.d_fused)
        self.mlp = WideFeedForward(self.d_fused, cfg.d_ff * cfg.num_avenues, cfg.dropout)

    def forward(self, states: torch.Tensor, *, return_diagnostics: bool) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bsz, seq_len, avenues, dim = states.shape
        x = states.reshape(bsz, seq_len, avenues * dim)
        attn_out, stats = self.attn(self.ln1(x), return_diagnostics=return_diagnostics)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x.view(bsz, seq_len, avenues, dim), stats


class LowRankOffDiagonalInterweave(nn.Module):
    def __init__(self, cfg: TransformerConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = int(layer_idx)
        self.num_avenues = int(cfg.num_avenues)
        self.rank = int(cfg.block_sparse_rank)
        self.norms = nn.ModuleList([nn.LayerNorm(cfg.d_model) for _ in range(self.num_avenues)])
        self.down = nn.ModuleDict()
        self.up = nn.ModuleDict()
        for target in range(self.num_avenues):
            for source in range(self.num_avenues):
                if source == target:
                    continue
                key = f"{target}_{source}"
                self.down[key] = nn.Linear(cfg.d_model, self.rank, bias=False)
                self.up[key] = nn.Linear(self.rank, cfg.d_model, bias=False)
        self.alpha = nn.Parameter(torch.full((self.num_avenues, self.num_avenues), float(cfg.block_sparse_alpha_init)))
        with torch.no_grad():
            self.alpha.fill_(float(cfg.block_sparse_alpha_init))
            self.alpha.diagonal().zero_()

    def forward(
        self,
        states: torch.Tensor,
        *,
        control: Optional[str],
        return_diagnostics: bool,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        source_states = states
        if control == "interweave_shuffle" and states.shape[0] > 1:
            source_states = states.index_select(0, torch.randperm(states.shape[0], device=states.device))
        deltas: List[torch.Tensor] = []
        source_norms: List[torch.Tensor] = []
        for target in range(self.num_avenues):
            delta = torch.zeros_like(states[:, :, target])
            for source in range(self.num_avenues):
                if source == target:
                    continue
                key = f"{target}_{source}"
                transformed = self.up[key](self.down[key](self.norms[source](source_states[:, :, source])))
                weight = self.alpha[target, source].to(dtype=states.dtype)
                contribution = weight * transformed
                delta = delta + contribution
                source_norms.append(contribution.float().norm(dim=-1).mean())
            deltas.append(delta)
        delta_stack = torch.stack(deltas, dim=2)
        if control == "interweave_random":
            scale = delta_stack.detach().float().std().to(dtype=delta_stack.dtype).clamp_min(1.0e-3)
            delta_stack = torch.randn_like(delta_stack) * scale
        if control == "interweave_zero" or control == f"interweave_point_zero_{self.layer_idx}":
            delta_stack = torch.zeros_like(delta_stack)
        after = states + delta_stack
        hidden_norm = states.float().norm(dim=-1).mean()
        delta_norm = delta_stack.float().norm(dim=-1).mean()
        offdiag = self.alpha.detach().float().clone()
        offdiag.fill_diagonal_(0.0)
        stats: Dict[str, torch.Tensor] = {
            "block_sparse_interweave_applied": states.new_ones(()),
            "block_sparse_alpha_mean": offdiag.abs().sum() / max(1, self.num_avenues * (self.num_avenues - 1)),
            "block_sparse_delta_norm": delta_norm,
            "block_sparse_hidden_norm": hidden_norm,
            "block_sparse_delta_ratio": delta_norm / hidden_norm.clamp_min(1.0e-6),
            "block_sparse_source_contribution_norm": _mean_tensor(source_norms, states),
        }
        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "score": delta_stack.detach().float().norm(dim=-1).mean(dim=2),
                "alpha": offdiag,
            }
        return after, stats, diagnostics


class BlockSparseMatrixInterweaveTransformer(nn.Module):
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
        self.interweaves = nn.ModuleList([LowRankOffDiagonalInterweave(cfg, idx) for idx in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)
        with torch.no_grad():
            for interweave in self.interweaves:
                interweave.alpha.fill_(float(cfg.block_sparse_alpha_init))
                interweave.alpha.diagonal().zero_()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def interweave_points(self) -> set[int]:
        if self.cfg.block_sparse_layers is not None:
            return {int(idx) for idx in self.cfg.block_sparse_layers if 0 <= int(idx) < self.cfg.n_layers}
        return {
            idx
            for idx in range(self.cfg.n_layers)
            if (idx + 1) % int(self.cfg.block_sparse_interval) == 0
        }

    def _training_avenue_mask(self, states: torch.Tensor) -> torch.Tensor:
        if not self.training or self.num_avenues <= 1:
            return states
        if self.cfg.avenue_dropout <= 0 and self.cfg.leave_one_prob <= 0:
            return states
        bsz = states.shape[0]
        keep = torch.ones(bsz, self.num_avenues, dtype=states.dtype, device=states.device)
        if self.cfg.avenue_dropout > 0:
            keep = keep * (torch.rand(bsz, self.num_avenues, device=states.device) >= self.cfg.avenue_dropout).to(
                dtype=states.dtype
            )
        if self.cfg.leave_one_prob > 0:
            should_drop = torch.rand(bsz, device=states.device) < self.cfg.leave_one_prob
            if bool(should_drop.any()):
                drop_idx = torch.randint(0, self.num_avenues, (int(should_drop.sum().item()),), device=states.device)
                keep[should_drop, drop_idx] = 0.0
        all_dropped = keep.sum(dim=-1) <= 0
        if bool(all_dropped.any()):
            chosen = torch.randint(0, self.num_avenues, (int(all_dropped.sum().item()),), device=states.device)
            keep[all_dropped] = 0.0
            keep[all_dropped, chosen] = 1.0
        return states * keep.view(bsz, 1, self.num_avenues, 1)

    def _eval_avenue_control(self, states: torch.Tensor, control: Optional[str]) -> torch.Tensor:
        if control is None:
            return states
        if control == "avenue_zero":
            return torch.zeros_like(states)
        if control == "avenue_random":
            scale = states.detach().float().std().to(dtype=states.dtype).clamp_min(1.0e-3)
            return torch.randn_like(states) * scale
        if control == "avenue_shuffle" and states.shape[0] > 1:
            return states.index_select(0, torch.randperm(states.shape[0], device=states.device))
        if control == "avenue_order_shuffle":
            return states.index_select(2, torch.randperm(self.num_avenues, device=states.device))
        if control == "hidden_state_shuffle" and states.shape[0] > 1:
            return states.index_select(0, torch.randperm(states.shape[0], device=states.device))
        if control.startswith("single_avenue_"):
            idx = max(0, min(self.num_avenues - 1, int(control.rsplit("_", 1)[-1])))
            mask = torch.zeros(1, 1, self.num_avenues, 1, dtype=states.dtype, device=states.device)
            mask[:, :, idx, :] = 1.0
            return states * mask
        if control.startswith("leave_one_"):
            idx = max(0, min(self.num_avenues - 1, int(control.rsplit("_", 1)[-1])))
            mask = torch.ones(1, 1, self.num_avenues, 1, dtype=states.dtype, device=states.device)
            mask[:, :, idx, :] = 0.0
            return states * mask
        return states

    def _avenue_stats(self, states: torch.Tensor) -> Dict[str, torch.Tensor]:
        pairwise_cos, diversity = _sampled_pairwise(states, self.cfg.align_stride)
        energy = states.float().norm(dim=-1).mean(dim=1)
        usage = energy / energy.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        stats: Dict[str, torch.Tensor] = {
            "avenue_gate_entropy": _entropy(usage, dim=-1).mean(),
            "avenue_output_norm": states.float().norm(dim=-1).mean(),
            "pairwise_avenue_cosine": pairwise_cos,
            "avenue_diversity": diversity,
            "avenue_scale": states.new_ones(()),
        }
        usage_mean = usage.mean(dim=0)
        for idx in range(self.num_avenues):
            stats[f"avenue_usage_{idx}"] = usage_mean[idx]
        return stats

    def _empty_interweave_stats(self, states: torch.Tensor, stats: Dict[str, torch.Tensor]) -> None:
        zero = states.new_zeros(())
        stats.update(
            {
                "block_sparse_interweave_applied": zero,
                "block_sparse_alpha_mean": zero,
                "block_sparse_delta_norm": zero,
                "block_sparse_hidden_norm": states.float().norm(dim=-1).mean(),
                "block_sparse_delta_ratio": zero,
                "block_sparse_source_contribution_norm": zero,
            }
        )

    def _diversity_loss(self, avenue_states: Sequence[torch.Tensor]) -> torch.Tensor:
        if not avenue_states:
            return self.token_emb.weight.new_zeros(())
        losses = [_sampled_pairwise(states, self.cfg.align_stride)[0].square() for states in avenue_states]
        return torch.stack(losses).mean()

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
            dim=2,
        )
        layer_stats: List[Dict[str, torch.Tensor]] = []
        avenue_states: List[torch.Tensor] = []
        interweave_diags: List[Optional[Dict[str, torch.Tensor]]] = []
        points = self.interweave_points()
        for layer_idx, avenue_blocks in enumerate(self.blocks):
            new_states: List[torch.Tensor] = []
            attn_stats: List[torch.Tensor] = []
            attended_stats: List[torch.Tensor] = []
            sdpa_stats: List[torch.Tensor] = []
            for avenue_idx, block in enumerate(avenue_blocks):
                updated, block_stats = block(states[:, :, avenue_idx], return_diagnostics=return_diagnostics)
                new_states.append(updated)
                if "attention_entropy" in block_stats:
                    attn_stats.append(block_stats["attention_entropy"])
                if "effective_attended_tokens" in block_stats:
                    attended_stats.append(block_stats["effective_attended_tokens"])
                if "sdpa_used" in block_stats:
                    sdpa_stats.append(block_stats["sdpa_used"])
            states = torch.stack(new_states, dim=2)
            states = self._training_avenue_mask(states)
            states = self._eval_avenue_control(states, control)
            stats = self._avenue_stats(states)
            stats["attention_entropy"] = _mean_tensor(attn_stats, states)
            stats["effective_attended_tokens"] = _mean_tensor(attended_stats, states)
            stats["sdpa_used"] = _mean_tensor(sdpa_stats, states)
            diag = None
            if layer_idx in points:
                states, interweave_stats, diag = self.interweaves[layer_idx](
                    states,
                    control=control,
                    return_diagnostics=return_diagnostics,
                )
                stats.update(interweave_stats)
            else:
                self._empty_interweave_stats(states, stats)
            layer_stats.append(stats)
            avenue_states.append(states)
            interweave_diags.append(diag)
        logits = self.lm_head(self.ln_f(states.mean(dim=2)))
        return {
            "logits": logits,
            "layer_stats": layer_stats,
            "avenue_attn": None,
            "crossing_diags": None,
            "interweave_diags": interweave_diags if return_diagnostics else None,
            "avenue_states": avenue_states if return_diagnostics else None,
            "aux_losses": {
                "avenue_alignment_loss": self.token_emb.weight.new_zeros(()),
                "avenue_diversity_loss": self._diversity_loss(avenue_states),
                "crossing_orthogonality_loss": self.token_emb.weight.new_zeros(()),
            },
            "diagnostics": {
                "predicted_actual_avenue_cosine": self.token_emb.weight.new_zeros(()),
                "predicted_actual_avenue_cosine_by_pair": [],
            },
        }

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
            "last_attention_paths": [[block.attn._last_attention_path for block in layer] for layer in self.blocks],
            "expected_training_paths": [["sdpa_causal" for _ in layer] for layer in self.blocks],
            "cuda_sdp_flags": cuda_flags,
            "avenue_variant": True,
            "crossing_variant": False,
            "block_sparse_interweave_variant": True,
            "interweave_points": sorted(self.interweave_points()),
            "rank": self.cfg.block_sparse_rank,
            "note": "Private per-avenue matrices plus low-rank off-diagonal source-to-target matrix corridors.",
        }


class MatrixInterwovenAvenueTransformer(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_avenues = int(cfg.num_avenues)
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.type_emb = nn.Parameter(torch.randn(self.num_avenues, cfg.d_model) * 0.02)
        fused_points = self.fused_points()
        layers: List[nn.Module] = []
        for layer_idx in range(cfg.n_layers):
            if layer_idx in fused_points:
                layers.append(FusedAvenueTransformerBlock(cfg))
            else:
                layers.append(nn.ModuleList([AvenueTransformerBlock(cfg) for _ in range(self.num_avenues)]))
        self.layers = nn.ModuleList(layers)
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

    def fused_points(self) -> set[int]:
        if self.cfg.interwoven_fused_layers is not None:
            return {int(idx) for idx in self.cfg.interwoven_fused_layers if 0 <= int(idx) < self.cfg.n_layers}
        return {
            idx
            for idx in range(self.cfg.n_layers)
            if (idx + 1) % int(self.cfg.interwoven_fused_interval) == 0
        }

    def _training_avenue_mask(self, states: torch.Tensor) -> torch.Tensor:
        if not self.training or self.num_avenues <= 1:
            return states
        if self.cfg.avenue_dropout <= 0 and self.cfg.leave_one_prob <= 0:
            return states
        bsz = states.shape[0]
        keep = torch.ones(bsz, self.num_avenues, dtype=states.dtype, device=states.device)
        if self.cfg.avenue_dropout > 0:
            keep = keep * (torch.rand(bsz, self.num_avenues, device=states.device) >= self.cfg.avenue_dropout).to(
                dtype=states.dtype
            )
        if self.cfg.leave_one_prob > 0:
            should_drop = torch.rand(bsz, device=states.device) < self.cfg.leave_one_prob
            if bool(should_drop.any()):
                drop_idx = torch.randint(0, self.num_avenues, (int(should_drop.sum().item()),), device=states.device)
                keep[should_drop, drop_idx] = 0.0
        all_dropped = keep.sum(dim=-1) <= 0
        if bool(all_dropped.any()):
            chosen = torch.randint(0, self.num_avenues, (int(all_dropped.sum().item()),), device=states.device)
            keep[all_dropped] = 0.0
            keep[all_dropped, chosen] = 1.0
        return states * keep.view(bsz, 1, self.num_avenues, 1)

    def _eval_avenue_control(self, states: torch.Tensor, control: Optional[str]) -> torch.Tensor:
        if control is None:
            return states
        if control == "avenue_zero":
            return torch.zeros_like(states)
        if control == "avenue_random":
            scale = states.detach().float().std().to(dtype=states.dtype).clamp_min(1.0e-3)
            return torch.randn_like(states) * scale
        if control == "avenue_shuffle" and states.shape[0] > 1:
            return states.index_select(0, torch.randperm(states.shape[0], device=states.device))
        if control == "avenue_order_shuffle":
            return states.index_select(2, torch.randperm(self.num_avenues, device=states.device))
        if control == "hidden_state_shuffle" and states.shape[0] > 1:
            return states.index_select(0, torch.randperm(states.shape[0], device=states.device))
        if control.startswith("single_avenue_"):
            idx = max(0, min(self.num_avenues - 1, int(control.rsplit("_", 1)[-1])))
            mask = torch.zeros(1, 1, self.num_avenues, 1, dtype=states.dtype, device=states.device)
            mask[:, :, idx, :] = 1.0
            return states * mask
        if control.startswith("leave_one_"):
            idx = max(0, min(self.num_avenues - 1, int(control.rsplit("_", 1)[-1])))
            mask = torch.ones(1, 1, self.num_avenues, 1, dtype=states.dtype, device=states.device)
            mask[:, :, idx, :] = 0.0
            return states * mask
        return states

    def _avenue_stats(self, states: torch.Tensor, *, fused: bool) -> Dict[str, torch.Tensor]:
        pairwise_cos, diversity = _sampled_pairwise(states, self.cfg.align_stride)
        energy = states.float().norm(dim=-1).mean(dim=1)
        usage = energy / energy.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        stats: Dict[str, torch.Tensor] = {
            "avenue_gate_entropy": _entropy(usage, dim=-1).mean(),
            "avenue_output_norm": states.float().norm(dim=-1).mean(),
            "pairwise_avenue_cosine": pairwise_cos,
            "avenue_diversity": diversity,
            "avenue_scale": states.new_ones(()),
            "matrix_interwoven_applied": states.new_tensor(1.0 if fused else 0.0),
            "split_matrix_applied": states.new_tensor(0.0 if fused else 1.0),
        }
        usage_mean = usage.mean(dim=0)
        for idx in range(self.num_avenues):
            stats[f"avenue_usage_{idx}"] = usage_mean[idx]
        return stats

    def _diversity_loss(self, avenue_states: Sequence[torch.Tensor]) -> torch.Tensor:
        if not avenue_states:
            return self.token_emb.weight.new_zeros(())
        losses = [_sampled_pairwise(states, self.cfg.align_stride)[0].square() for states in avenue_states]
        return torch.stack(losses).mean()

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
            dim=2,
        )
        layer_stats: List[Dict[str, torch.Tensor]] = []
        avenue_states: List[torch.Tensor] = []
        fused_points = self.fused_points()
        for layer_idx in range(self.cfg.n_layers):
            fused = layer_idx in fused_points
            layer = self.layers[layer_idx]
            if fused:
                assert isinstance(layer, FusedAvenueTransformerBlock)
                states, block_stats = layer(states, return_diagnostics=return_diagnostics)
                attn_stats = [block_stats.get("attention_entropy")] if "attention_entropy" in block_stats else []
                attended_stats = [block_stats.get("effective_attended_tokens")] if "effective_attended_tokens" in block_stats else []
                sdpa_stats = [block_stats.get("sdpa_used")] if "sdpa_used" in block_stats else []
            else:
                assert isinstance(layer, nn.ModuleList)
                new_states: List[torch.Tensor] = []
                attn_stats: List[torch.Tensor] = []
                attended_stats: List[torch.Tensor] = []
                sdpa_stats: List[torch.Tensor] = []
                for avenue_idx, block in enumerate(layer):
                    updated, block_stats = block(states[:, :, avenue_idx], return_diagnostics=return_diagnostics)
                    new_states.append(updated)
                    if "attention_entropy" in block_stats:
                        attn_stats.append(block_stats["attention_entropy"])
                    if "effective_attended_tokens" in block_stats:
                        attended_stats.append(block_stats["effective_attended_tokens"])
                    if "sdpa_used" in block_stats:
                        sdpa_stats.append(block_stats["sdpa_used"])
                states = torch.stack(new_states, dim=2)
            states = self._training_avenue_mask(states)
            states = self._eval_avenue_control(states, control)
            stats = self._avenue_stats(states, fused=fused)
            stats["attention_entropy"] = _mean_tensor(attn_stats, states)
            stats["effective_attended_tokens"] = _mean_tensor(attended_stats, states)
            stats["sdpa_used"] = _mean_tensor(sdpa_stats, states)
            layer_stats.append(stats)
            avenue_states.append(states)
        logits = self.lm_head(self.ln_f(states.mean(dim=2)))
        return {
            "logits": logits,
            "layer_stats": layer_stats,
            "avenue_attn": None,
            "crossing_diags": None,
            "avenue_states": avenue_states if return_diagnostics else None,
            "aux_losses": {
                "avenue_alignment_loss": self.token_emb.weight.new_zeros(()),
                "avenue_diversity_loss": self._diversity_loss(avenue_states),
                "crossing_orthogonality_loss": self.token_emb.weight.new_zeros(()),
            },
            "diagnostics": {
                "predicted_actual_avenue_cosine": self.token_emb.weight.new_zeros(()),
                "predicted_actual_avenue_cosine_by_pair": [],
            },
        }

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
        fused_points = self.fused_points()
        paths: List[Any] = []
        expected: List[Any] = []
        for layer_idx in range(self.cfg.n_layers):
            layer = self.layers[layer_idx]
            if layer_idx in fused_points:
                assert isinstance(layer, FusedAvenueTransformerBlock)
                paths.append(layer.attn._last_attention_path)
                expected.append("sdpa_causal")
            else:
                assert isinstance(layer, nn.ModuleList)
                paths.append([block.attn._last_attention_path for block in layer])
                expected.append(["sdpa_causal" for _ in layer])
        return {
            "last_attention_paths": paths,
            "expected_training_paths": expected,
            "cuda_sdp_flags": cuda_flags,
            "avenue_variant": True,
            "crossing_variant": False,
            "matrix_interwoven_variant": True,
            "fused_matrix_layers": sorted(fused_points),
            "note": "Alternates physically split per-avenue matrices with full fused matrices over concatenated avenue channels.",
        }


class MultiAvenueTransformer(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.variant in CROSSING_VARIANTS:
            self.impl: nn.Module = CrossingAvenueTransformer(cfg)
            return
        if cfg.variant in MATRIX_INTERWOVEN_VARIANTS:
            self.impl = MatrixInterwovenAvenueTransformer(cfg)
            return
        if cfg.variant in BLOCK_SPARSE_INTERWEAVE_VARIANTS:
            self.impl = BlockSparseMatrixInterweaveTransformer(cfg)
            return
        if cfg.variant in PATTERN_ATTENTION_VARIANTS:
            self.impl = PatternAttentionTransformer(cfg)
            return
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
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
            return isinstance(
                self.impl,
                (
                    CrossingAvenueTransformer,
                    MatrixInterwovenAvenueTransformer,
                    BlockSparseMatrixInterweaveTransformer,
                ),
            )
        return self.cfg.variant in AVENUE_VARIANTS

    @property
    def has_crossings(self) -> bool:
        return hasattr(self, "impl") and isinstance(self.impl, CrossingAvenueTransformer)

    @property
    def has_interweaves(self) -> bool:
        return hasattr(self, "impl") and isinstance(self.impl, BlockSparseMatrixInterweaveTransformer)

    @property
    def has_patterns(self) -> bool:
        return hasattr(self, "impl") and isinstance(self.impl, PatternAttentionTransformer)

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
        losses: List[torch.Tensor] = []
        sims: List[torch.Tensor] = []
        stride = max(1, int(self.cfg.align_stride))
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
        losses = [_sampled_pairwise(states, self.cfg.align_stride)[0].square() for states in avenue_states]
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
        _bsz, seq_len = input_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.cfg.max_seq_len}")
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))
        layer_stats: List[Dict[str, torch.Tensor]] = []
        avenue_states: List[torch.Tensor] = []
        avenue_diags: List[Optional[Dict[str, torch.Tensor]]] = []
        for block in self.blocks:
            x, avenues, stats, diag = block(x, control=control, return_diagnostics=return_diagnostics)
            if avenues is not None:
                avenue_states.append(avenues)
            layer_stats.append(stats)
            avenue_diags.append(diag)
        logits = self.lm_head(self.ln_f(x))
        align_loss, align_sim, align_by_pair = self._alignment_loss(avenue_states)
        return {
            "logits": logits,
            "layer_stats": layer_stats,
            "avenue_attn": avenue_diags if return_diagnostics else None,
            "crossing_diags": None,
            "avenue_states": avenue_states if return_diagnostics else None,
            "aux_losses": {
                "avenue_alignment_loss": align_loss,
                "avenue_diversity_loss": self._diversity_loss(avenue_states),
                "crossing_orthogonality_loss": self.token_emb.weight.new_zeros(()),
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
            "expected_training_paths": ["sdpa_causal" for _block in self.blocks],
            "cuda_sdp_flags": cuda_flags,
            "avenue_variant": self.has_avenues,
            "crossing_variant": False,
            "note": "Baselines and E8 references use causal SDPA; E8 avenues are residual per-token MLP pathways.",
        }


def build_model(raw_config: Dict[str, Any]) -> MultiAvenueTransformer:
    return MultiAvenueTransformer(TransformerConfig.from_dict(raw_config))


def parameter_count(model: nn.Module, *, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        return sum(param.numel() for param in params if param.requires_grad)
    return sum(param.numel() for param in params)


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
