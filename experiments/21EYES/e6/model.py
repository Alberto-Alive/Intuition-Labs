from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from data import VOCAB_SIZE


VARIANTS = {
    "baseline",
    "param_matched",
    "same_layer_u_gate",
    "inter_layer_route_gate",
    "prev_attn_reuse",
    "random_gate",
    "dense_augmented_qk",
    "block_route_topk_b16_k4",
    "block_route_topk_b16_k2",
    "block_route_topk_b16_k8",
    "block_route_topk_b32_k4",
    "block_route_topk_b32_k2",
    "block_route_topk_b32_k8",
    "coarse_to_fine_block_topk_4",
}


@dataclass
class TransformerConfig:
    variant: str = "baseline"
    vocab_size: int = VOCAB_SIZE
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_seq_len: int = 512
    d_route: int = 32
    gate_bias_init: float = 5.0
    alpha_init: float = 0.0
    beta_init: float = 0.05
    block_size: int = 16
    top_k_blocks: int = 4
    local_window_blocks: int = 2

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TransformerConfig":
        data = dict(raw)
        data.setdefault("vocab_size", VOCAB_SIZE)
        cfg = cls(**data)
        if cfg.variant not in VARIANTS:
            raise ValueError(f"Unknown variant '{cfg.variant}'. Expected one of {sorted(VARIANTS)}")
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if cfg.d_route <= 0:
            raise ValueError("d_route must be positive")
        if cfg.block_size <= 0:
            raise ValueError("block_size must be positive")
        if cfg.top_k_blocks < 0:
            raise ValueError("top_k_blocks must be non-negative")
        if cfg.local_window_blocks <= 0:
            raise ValueError("local_window_blocks must be positive")
        return cfg


def _variant_family(variant: str) -> str:
    if variant.startswith("block_route_topk"):
        return "block_route_topk"
    if variant.startswith("coarse_to_fine"):
        return "coarse_to_fine"
    return variant


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()


def _entropy(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    probs_f = probs.float().clamp_min(1.0e-8)
    return -(probs_f * probs_f.log()).sum(dim=dim)


def _binary_entropy(gate: torch.Tensor) -> torch.Tensor:
    gate_f = gate.float().clamp(1.0e-8, 1.0 - 1.0e-8)
    return -(gate_f * gate_f.log() + (1.0 - gate_f) * (1.0 - gate_f).log())


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
    def __init__(self, d_model: int, d_route: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_route),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_route, d_model),
            nn.Dropout(dropout),
        )
        # Start as the same function as the baseline while still giving the
        # non-routing control comparable trainable capacity.
        final = self.net[3]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RoutedCausalSelfAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.variant = cfg.variant
        self.family = _variant_family(cfg.variant)

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

        needs_route = self.family in {
            "same_layer_u_gate",
            "inter_layer_route_gate",
            "random_gate",
            "dense_augmented_qk",
            "block_route_topk",
            "coarse_to_fine",
        }
        if needs_route:
            self.route_proj = nn.Linear(cfg.d_model, cfg.d_route)
            self.u_proj = nn.Linear(cfg.d_model, cfg.d_route)
            if cfg.variant == "random_gate":
                for param in self.route_proj.parameters():
                    param.requires_grad = False
                for param in self.u_proj.parameters():
                    param.requires_grad = False
        else:
            self.route_proj = None
            self.u_proj = None

        if cfg.variant in {"same_layer_u_gate", "inter_layer_route_gate", "prev_attn_reuse", "random_gate"}:
            self.alpha = nn.Parameter(torch.tensor(float(cfg.alpha_init)))
            self.gate_bias = nn.Parameter(torch.tensor(float(cfg.gate_bias_init)))
        else:
            self.register_parameter("alpha", None)
            self.register_parameter("gate_bias", None)

        if self.family in {"dense_augmented_qk", "coarse_to_fine"}:
            self.beta = nn.Parameter(torch.full((self.n_heads,), float(cfg.beta_init)))
        else:
            self.register_parameter("beta", None)

        self._last_attention_path = "manual"

    def _gate_stats(self, gate: torch.Tensor, causal: torch.Tensor) -> Dict[str, torch.Tensor]:
        valid = gate.float()[:, causal]
        return {
            "gate_mean": valid.mean(),
            "gate_open_rate": (valid > 0.5).float().mean(),
            "gate_entropy": _binary_entropy(valid).mean(),
            "gate_min": valid.min(),
            "gate_max": valid.max(),
        }

    def _route_token_stats(self, route_scores: torch.Tensor, causal: torch.Tensor) -> Dict[str, torch.Tensor]:
        scores = route_scores.float()
        valid_scores = scores[:, causal]
        masked = scores.masked_fill(~causal.view(1, *causal.shape), torch.finfo(scores.dtype).min)
        probs = F.softmax(masked, dim=-1)
        probs = torch.where(causal.view(1, *causal.shape), probs, torch.zeros_like(probs))
        return {
            "route_score_mean": valid_scores.mean(),
            "route_score_std": valid_scores.std(unbiased=False),
            "route_entropy": _entropy(probs, dim=-1).mean(),
        }

    def _route_block_stats(
        self,
        block_scores: torch.Tensor,
        current_blocks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        scores = block_scores.float()
        n_blocks = scores.shape[-1]
        block_ids = torch.arange(n_blocks, device=scores.device)
        valid = block_ids.view(1, 1, n_blocks) <= current_blocks.view(1, -1, 1)
        valid_scores = scores[valid.expand(scores.shape[0], -1, -1)]
        masked = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        probs = F.softmax(masked, dim=-1)
        probs = torch.where(valid, probs, torch.zeros_like(probs))
        return {
            "route_score_mean": valid_scores.mean(),
            "route_score_std": valid_scores.std(unbiased=False),
            "route_entropy": _entropy(probs, dim=-1).mean(),
        }

    def _attention_stats(
        self,
        weights: torch.Tensor,
        *,
        block_ids: Optional[torch.Tensor] = None,
        n_blocks: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        weights_f = weights.float().clamp_min(0.0)
        token_entropy = _entropy(weights_f, dim=-1)
        stats: Dict[str, torch.Tensor] = {
            "attention_entropy": token_entropy.mean(),
            "effective_attended_tokens": token_entropy.exp().mean(),
        }
        if block_ids is not None and n_blocks is not None:
            block_idx = block_ids.view(1, 1, 1, -1).expand(*weights_f.shape[:-1], -1)
            block_weights = torch.zeros(
                *weights_f.shape[:-1],
                n_blocks,
                dtype=weights_f.dtype,
                device=weights_f.device,
            )
            block_weights.scatter_add_(-1, block_idx, weights_f)
            block_entropy = _entropy(block_weights, dim=-1)
            stats["effective_attended_blocks"] = block_entropy.exp().mean()
        return stats

    def _project_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        return q, k, v

    def _merge_heads(self, out: torch.Tensor, d_model: int) -> torch.Tensor:
        return out.transpose(1, 2).contiguous().view(out.shape[0], out.shape[2], d_model)

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: torch.Tensor,
        need_weights: bool,
        force_manual: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        seq_len = q.shape[-2]
        if not need_weights and not force_manual:
            self._last_attention_path = "sdpa_causal"
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=float(self.cfg.dropout) if self.training else 0.0,
                is_causal=True,
            )
            return out, None, {"sdpa_used": torch.ones((), device=q.device)}

        self._last_attention_path = "manual_dense"
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)
        logits = logits.masked_fill(~causal.view(1, 1, seq_len, seq_len), torch.finfo(logits.dtype).min)
        attn_weights = F.softmax(logits.float(), dim=-1).to(dtype=q.dtype)
        attn = self.dropout(attn_weights) if self.training else attn_weights
        out = torch.matmul(attn, v)
        token_blocks = torch.arange(seq_len, device=q.device) // int(self.cfg.block_size)
        stats = self._attention_stats(
            attn_weights,
            block_ids=token_blocks,
            n_blocks=int(token_blocks.max().item()) + 1,
        )
        stats["sdpa_used"] = torch.zeros((), device=q.device)
        return out, attn_weights, stats

    def _dense_augmented_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        x: torch.Tensor,
        *,
        prev_hidden: Optional[torch.Tensor],
        causal: torch.Tensor,
        return_gate_matrices: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        if self.layer_idx == 0 or prev_hidden is None:
            out, attn_weights, stats = self._standard_attention(
                q,
                k,
                v,
                causal=causal,
                need_weights=return_gate_matrices,
            )
            if self.beta is not None:
                stats["beta"] = self.beta.detach().float().mean()
            return out, attn_weights, stats, None

        assert self.route_proj is not None and self.u_proj is not None and self.beta is not None
        bsz, _heads, seq_len, _d_head = q.shape
        route_q = self.route_proj(prev_hidden)
        route_u = self.u_proj(x)

        d_aug = self.d_head + self.cfg.d_route
        base_scale = (float(d_aug) / float(self.d_head)) ** 0.25
        beta = self.beta.to(dtype=q.dtype).view(1, self.n_heads, 1, 1)
        route_q_h = route_q.to(dtype=q.dtype).unsqueeze(1).expand(bsz, self.n_heads, seq_len, self.cfg.d_route)
        route_u_h = route_u.to(dtype=q.dtype).unsqueeze(1).expand(bsz, self.n_heads, seq_len, self.cfg.d_route)
        q_aug = torch.cat((q * base_scale, route_q_h * beta), dim=-1)
        k_aug = torch.cat((k * base_scale, route_u_h * beta), dim=-1)

        stats: Dict[str, torch.Tensor] = {"beta": self.beta.detach().float().mean()}
        if not return_gate_matrices:
            self._last_attention_path = "sdpa_dense_augmented_qk"
            out = F.scaled_dot_product_attention(
                q_aug,
                k_aug,
                v,
                dropout_p=float(self.cfg.dropout) if self.training else 0.0,
                is_causal=True,
            )
            stats["sdpa_used"] = torch.ones((), device=q.device)
            return out, None, stats, None

        self._last_attention_path = "manual_dense_augmented_qk"
        logits = torch.matmul(q_aug, k_aug.transpose(-1, -2)) / math.sqrt(d_aug)
        logits = logits.masked_fill(~causal.view(1, 1, seq_len, seq_len), torch.finfo(logits.dtype).min)
        attn_weights = F.softmax(logits.float(), dim=-1).to(dtype=q.dtype)
        attn = self.dropout(attn_weights) if self.training else attn_weights
        out = torch.matmul(attn, v)
        token_blocks = torch.arange(seq_len, device=q.device) // int(self.cfg.block_size)
        stats.update(
            self._attention_stats(
                attn_weights,
                block_ids=token_blocks,
                n_blocks=int(token_blocks.max().item()) + 1,
            )
        )
        route_scores = torch.matmul(route_q, route_u.transpose(-1, -2)) / math.sqrt(self.cfg.d_route)
        stats.update(self._route_token_stats(route_scores, causal))
        stats["sdpa_used"] = torch.zeros((), device=q.device)
        return out, attn_weights, stats, route_scores

    def _block_descriptors(self, route_u: torch.Tensor, block_size: int) -> torch.Tensor:
        bsz, seq_len, d_route = route_u.shape
        n_blocks = math.ceil(seq_len / block_size)
        padded_len = n_blocks * block_size
        if padded_len != seq_len:
            pad = route_u.new_zeros(bsz, padded_len - seq_len, d_route)
            route_u = torch.cat((route_u, pad), dim=1)
        blocks = route_u.view(bsz, n_blocks, block_size, d_route)
        lengths = torch.full((n_blocks,), block_size, dtype=route_u.dtype, device=route_u.device)
        last_len = seq_len - (n_blocks - 1) * block_size
        lengths[-1] = float(last_len)
        return blocks.sum(dim=2) / lengths.view(1, n_blocks, 1).clamp_min(1.0)

    def _candidate_indices_from_blocks(
        self,
        block_scores: torch.Tensor,
        *,
        seq_len: int,
        block_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, _, n_blocks = block_scores.shape
        device = block_scores.device
        top_k = min(int(self.cfg.top_k_blocks), n_blocks)
        local_window = min(int(self.cfg.local_window_blocks), n_blocks)

        positions = torch.arange(seq_len, device=device)
        current_blocks = positions // block_size
        local_offsets = torch.arange(local_window, device=device)
        local_blocks = current_blocks.view(seq_len, 1) - local_offsets.view(1, local_window)
        local_valid = local_blocks >= 0
        local_blocks = local_blocks.clamp_min(0)

        if top_k > 0:
            block_ids = torch.arange(n_blocks, device=device)
            first_routed_block = current_blocks - local_window + 1
            eligible = block_ids.view(1, 1, n_blocks) < first_routed_block.view(1, seq_len, 1)
            masked_scores = block_scores.masked_fill(~eligible, torch.finfo(block_scores.dtype).min)
            top_scores, top_blocks = torch.topk(masked_scores, k=top_k, dim=-1)
            top_valid = torch.isfinite(top_scores) & eligible.expand(bsz, -1, -1).gather(-1, top_blocks)
        else:
            top_blocks = torch.empty(bsz, seq_len, 0, dtype=torch.long, device=device)
            top_valid = torch.empty(bsz, seq_len, 0, dtype=torch.bool, device=device)

        local_blocks_b = local_blocks.view(1, seq_len, local_window).expand(bsz, -1, -1)
        local_valid_b = local_valid.view(1, seq_len, local_window).expand(bsz, -1, -1)
        block_indices = torch.cat((local_blocks_b, top_blocks), dim=-1)
        block_valid = torch.cat((local_valid_b, top_valid), dim=-1)

        token_offsets = torch.arange(block_size, device=device)
        candidate_indices = block_indices.unsqueeze(-1) * block_size + token_offsets.view(1, 1, 1, block_size)
        query_pos = positions.view(1, seq_len, 1, 1)
        token_valid = (
            block_valid.unsqueeze(-1)
            & (candidate_indices < seq_len)
            & (candidate_indices <= query_pos)
        )
        candidate_indices = candidate_indices.clamp(0, seq_len - 1).reshape(bsz, seq_len, -1)
        token_valid = token_valid.reshape(bsz, seq_len, -1)
        candidate_blocks = block_indices.unsqueeze(-1).expand(-1, -1, -1, block_size).reshape(bsz, seq_len, -1)
        return candidate_indices, token_valid, block_valid, candidate_blocks, current_blocks

    def _gathered_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        candidate_indices: torch.Tensor,
        token_valid: torch.Tensor,
        candidate_blocks: torch.Tensor,
        n_blocks: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bsz, n_heads, seq_len, d_head = q.shape
        max_candidates = candidate_indices.shape[-1]
        gather_idx = candidate_indices.view(bsz, 1, seq_len, max_candidates, 1).expand(
            bsz,
            n_heads,
            seq_len,
            max_candidates,
            d_head,
        )
        k_expanded = k.unsqueeze(2).expand(bsz, n_heads, seq_len, seq_len, d_head)
        v_expanded = v.unsqueeze(2).expand(bsz, n_heads, seq_len, seq_len, d_head)
        k_sel = torch.gather(k_expanded, 3, gather_idx)
        v_sel = torch.gather(v_expanded, 3, gather_idx)
        logits = (q.unsqueeze(3) * k_sel).sum(dim=-1) / math.sqrt(d_head)
        valid_h = token_valid.view(bsz, 1, seq_len, max_candidates)
        logits = logits.masked_fill(~valid_h, torch.finfo(logits.dtype).min)
        weights = F.softmax(logits.float(), dim=-1).to(dtype=q.dtype)
        weights = torch.where(valid_h, weights, torch.zeros_like(weights))
        attn = self.dropout(weights) if self.training else weights
        out = (attn.unsqueeze(-1) * v_sel).sum(dim=3)

        stats = self._attention_stats(weights)
        selected_tokens = token_valid.float().sum(dim=-1)
        stats["selected_token_count"] = selected_tokens.mean()
        block_valid_from_tokens = token_valid.view(bsz, seq_len, -1, int(self.cfg.block_size)).any(dim=-1)
        stats["selected_block_count"] = block_valid_from_tokens.float().sum(dim=-1).mean()
        block_idx = candidate_blocks.view(bsz, 1, seq_len, max_candidates).expand(bsz, n_heads, -1, -1)
        block_weights = torch.zeros(
            bsz,
            n_heads,
            seq_len,
            n_blocks,
            dtype=weights.float().dtype,
            device=weights.device,
        )
        block_weights.scatter_add_(-1, block_idx, weights.float())
        block_entropy = _entropy(block_weights, dim=-1)
        stats["effective_attended_blocks"] = block_entropy.exp().mean()
        return out, stats

    def _block_routed_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        x: torch.Tensor,
        *,
        prev_hidden: Optional[torch.Tensor],
        causal: torch.Tensor,
        return_gate_matrices: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        if self.layer_idx == 0 or prev_hidden is None:
            out, attn_weights, stats = self._standard_attention(
                q,
                k,
                v,
                causal=causal,
                need_weights=return_gate_matrices,
            )
            return out, attn_weights, stats, None

        assert self.route_proj is not None and self.u_proj is not None
        bsz, _heads, seq_len, _d_head = q.shape
        route_q = self.route_proj(prev_hidden)
        route_u = self.u_proj(x)
        block_size = int(self.cfg.block_size)
        u_blocks = self._block_descriptors(route_u, block_size)
        block_scores = torch.matmul(route_q, u_blocks.transpose(-1, -2)) / math.sqrt(self.cfg.d_route)
        n_blocks = block_scores.shape[-1]
        candidate_indices, token_valid, block_valid, candidate_blocks, current_blocks = self._candidate_indices_from_blocks(
            block_scores,
            seq_len=seq_len,
            block_size=block_size,
        )
        self._last_attention_path = "gathered_block_sparse"
        out, stats = self._gathered_attention(
            q,
            k,
            v,
            candidate_indices,
            token_valid,
            candidate_blocks,
            n_blocks,
        )
        stats.update(self._route_block_stats(block_scores, current_blocks))
        stats["sdpa_used"] = torch.zeros((), device=q.device)

        route_scores: Optional[torch.Tensor] = None
        if return_gate_matrices:
            route_scores = torch.matmul(route_q, route_u.transpose(-1, -2)) / math.sqrt(self.cfg.d_route)
            stats.update(self._route_token_stats(route_scores, causal))
        return out, None, stats, route_scores

    def forward(
        self,
        x: torch.Tensor,
        *,
        prev_hidden: Optional[torch.Tensor],
        prev_attn: Optional[torch.Tensor],
        return_gate_matrices: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        bsz, seq_len, d_model = x.shape
        q, k, v = self._project_qkv(x)
        causal = _causal_mask(seq_len, x.device)

        gate_matrix: Optional[torch.Tensor] = None
        stats: Dict[str, torch.Tensor] = {}
        attn_weights: Optional[torch.Tensor] = None

        if self.variant in {"baseline", "param_matched"}:
            out_heads, attn_weights, stats = self._standard_attention(
                q,
                k,
                v,
                causal=causal,
                need_weights=return_gate_matrices,
            )
            out = self._merge_heads(out_heads, d_model)
            out = self.out_proj(out)
            return out, attn_weights, stats, None

        if self.family == "dense_augmented_qk" or (self.family == "coarse_to_fine" and self.training):
            out_heads, attn_weights, stats, gate_matrix = self._dense_augmented_attention(
                q,
                k,
                v,
                x,
                prev_hidden=prev_hidden,
                causal=causal,
                return_gate_matrices=return_gate_matrices,
            )
            out = self._merge_heads(out_heads, d_model)
            out = self.out_proj(out)
            if not return_gate_matrices:
                gate_matrix = None
            return out, attn_weights, stats, gate_matrix

        if self.family in {"block_route_topk", "coarse_to_fine"}:
            out_heads, attn_weights, stats, gate_matrix = self._block_routed_attention(
                q,
                k,
                v,
                x,
                prev_hidden=prev_hidden,
                causal=causal,
                return_gate_matrices=return_gate_matrices,
            )
            if self.beta is not None:
                stats["beta"] = self.beta.detach().float().mean()
            out = self._merge_heads(out_heads, d_model)
            out = self.out_proj(out)
            if not return_gate_matrices:
                gate_matrix = None
            return out, attn_weights, stats, gate_matrix

        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)

        if self.variant == "same_layer_u_gate":
            assert self.route_proj is not None and self.u_proj is not None
            route_source = x
            route_q = self.route_proj(route_source)
            route_u = self.u_proj(x)
            gate_logits = torch.matmul(route_q, route_u.transpose(-1, -2)) / math.sqrt(self.cfg.d_route)
            gate = torch.sigmoid(gate_logits + self.gate_bias)
            gate = torch.where(causal.unsqueeze(0), gate, torch.ones_like(gate))
            logits = logits + self.alpha * torch.log(gate.clamp_min(1.0e-6)).unsqueeze(1)
            gate_matrix = gate
            stats.update(self._gate_stats(gate, causal))
            stats.update(self._route_token_stats(gate_logits, causal))

        elif self.variant in {"inter_layer_route_gate", "random_gate"} and self.layer_idx > 0 and prev_hidden is not None:
            assert self.route_proj is not None and self.u_proj is not None
            route_q = self.route_proj(prev_hidden)
            route_u = self.u_proj(x)
            gate_logits = torch.matmul(route_q, route_u.transpose(-1, -2)) / math.sqrt(self.cfg.d_route)
            gate = torch.sigmoid(gate_logits + self.gate_bias)
            gate = torch.where(causal.unsqueeze(0), gate, torch.ones_like(gate))
            logits = logits + self.alpha * torch.log(gate.clamp_min(1.0e-6)).unsqueeze(1)
            gate_matrix = gate
            stats.update(self._gate_stats(gate, causal))
            stats.update(self._route_token_stats(gate_logits, causal))

        elif self.variant == "prev_attn_reuse" and self.layer_idx > 0 and prev_attn is not None:
            prior = prev_attn.detach().clamp_min(1.0e-8)
            logits = logits + self.alpha * torch.log(prior)
            gate_matrix = prior.mean(dim=1)
            stats.update(self._gate_stats(gate_matrix, causal))

        logits = logits.masked_fill(~causal.view(1, 1, seq_len, seq_len), torch.finfo(logits.dtype).min)
        attn_weights = F.softmax(logits.float(), dim=-1).to(dtype=x.dtype)
        attn = self.dropout(attn_weights) if self.training else attn_weights
        out = torch.matmul(attn, v)
        out = self._merge_heads(out, d_model)
        out = self.out_proj(out)

        token_blocks = torch.arange(seq_len, device=x.device) // int(self.cfg.block_size)
        stats.update(
            self._attention_stats(
                attn_weights,
                block_ids=token_blocks,
                n_blocks=int(token_blocks.max().item()) + 1,
            )
        )
        stats["sdpa_used"] = torch.zeros((), device=x.device)
        if self.alpha is not None:
            stats["alpha"] = self.alpha.detach()
        if self.gate_bias is not None:
            stats["gate_bias"] = self.gate_bias.detach()

        if not return_gate_matrices:
            gate_matrix = None
        return out, attn_weights, stats, gate_matrix


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig, layer_idx: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = RoutedCausalSelfAttention(cfg, layer_idx)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.adapter = ParamMatchedAdapter(cfg.d_model, cfg.d_route, cfg.dropout) if cfg.variant == "param_matched" else None

    def forward(
        self,
        x: torch.Tensor,
        *,
        prev_hidden: Optional[torch.Tensor],
        prev_attn: Optional[torch.Tensor],
        return_gate_matrices: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        attn_in = self.ln1(x)
        attn_out, attn, stats, gate = self.attn(
            attn_in,
            prev_hidden=prev_hidden,
            prev_attn=prev_attn,
            return_gate_matrices=return_gate_matrices,
        )
        x = x + attn_out
        mlp_in = self.ln2(x)
        mlp_out = self.mlp(mlp_in)
        if self.adapter is not None:
            mlp_out = mlp_out + self.adapter(mlp_in)
        x = x + mlp_out
        return x, attn, stats, gate


class ActivationRoutedTransformer(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg.n_layers)])
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

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        return_gate_matrices: bool = False,
    ) -> Dict[str, Any]:
        bsz, seq_len = input_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.cfg.max_seq_len}")
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        prev_hidden: Optional[torch.Tensor] = None
        prev_attn: Optional[torch.Tensor] = None
        layer_stats: List[Dict[str, torch.Tensor]] = []
        gate_matrices: List[Optional[torch.Tensor]] = []

        for block in self.blocks:
            x, attn, stats, gate = block(
                x,
                prev_hidden=prev_hidden,
                prev_attn=prev_attn,
                return_gate_matrices=return_gate_matrices,
            )
            prev_hidden = x
            prev_attn = attn
            layer_stats.append(stats)
            gate_matrices.append(gate)

        logits = self.lm_head(self.ln_f(x))
        return {
            "logits": logits,
            "layer_stats": layer_stats,
            "gate_matrices": gate_matrices if return_gate_matrices else None,
        }

    def alpha_values(self) -> List[Optional[float]]:
        values: List[Optional[float]] = []
        for block in self.blocks:
            alpha = block.attn.alpha
            values.append(None if alpha is None else float(alpha.detach().cpu()))
        return values

    def beta_values(self) -> List[Optional[List[float]]]:
        values: List[Optional[List[float]]] = []
        for block in self.blocks:
            beta = block.attn.beta
            values.append(None if beta is None else [float(v) for v in beta.detach().cpu().tolist()])
        return values

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
        expected_training_paths: List[str] = []
        for block in self.blocks:
            attn = block.attn
            if attn.family == "dense_augmented_qk":
                expected_training_paths.append("sdpa_dense_augmented_qk" if attn.layer_idx > 0 else "sdpa_causal")
            elif attn.family == "coarse_to_fine":
                expected_training_paths.append("sdpa_dense_augmented_qk_train/gathered_block_sparse_eval" if attn.layer_idx > 0 else "sdpa_causal")
            elif attn.family == "block_route_topk":
                expected_training_paths.append("gathered_block_sparse" if attn.layer_idx > 0 else "sdpa_causal")
            elif attn.variant in {"baseline", "param_matched"}:
                expected_training_paths.append("sdpa_causal")
            else:
                expected_training_paths.append("manual_dense")
        return {
            "last_attention_paths": [block.attn._last_attention_path for block in self.blocks],
            "expected_training_paths": expected_training_paths,
            "cuda_sdp_flags": cuda_flags,
            "note": "PyTorch does not expose the exact selected SDPA backend here; flags show enabled backend families and paths show whether SDPA was requested.",
        }


def build_model(raw_config: Dict[str, Any]) -> ActivationRoutedTransformer:
    cfg = TransformerConfig.from_dict(raw_config)
    return ActivationRoutedTransformer(cfg)


def parameter_count(model: nn.Module, *, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def model_summary(model: nn.Module) -> str:
    total = parameter_count(model)
    trainable = parameter_count(model, trainable_only=True)
    frozen = total - trainable
    lines = [
        repr(model),
        "",
        f"total_parameters: {total}",
        f"trainable_parameters: {trainable}",
        f"frozen_parameters: {frozen}",
    ]
    return "\n".join(lines)
