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
    "memory_slot_attention",
    "slot_alignment_loss",
    "predictive_slot_residual",
    "mirrored_slot_attention",
}

SLOT_VARIANTS = {
    "memory_slot_attention",
    "slot_alignment_loss",
    "predictive_slot_residual",
    "mirrored_slot_attention",
}


@dataclass
class TransformerConfig:
    variant: str = "baseline_transformer"
    vocab_size: int = VOCAB_SIZE
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_seq_len: int = 512
    param_adapter_dim: int = 384
    slot_count: int = 64
    d_slot: int = 128
    slot_heads: int = 4
    slot_iters: int = 1
    slot_mlp_mult: int = 2
    slot_update_scale: float = 0.35
    align_stride: int = 8
    mirror_stride: int = 8

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TransformerConfig":
        data = dict(raw)
        data.setdefault("vocab_size", VOCAB_SIZE)
        cfg = cls(**data)
        if cfg.variant not in VARIANTS:
            raise ValueError(f"Unknown variant '{cfg.variant}'. Expected one of {sorted(VARIANTS)}")
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if cfg.d_slot != cfg.d_model:
            raise ValueError("This E7 implementation requires d_slot == d_model for residual slot readback")
        if cfg.slot_count <= 0:
            raise ValueError("slot_count must be positive")
        if cfg.align_stride <= 0 or cfg.mirror_stride <= 0:
            raise ValueError("align_stride and mirror_stride must be positive")
        return cfg


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()


def _entropy(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    probs_f = probs.float().clamp_min(1.0e-8)
    return -(probs_f * probs_f.log()).sum(dim=dim)


def _offdiag_cosine_squared(slots: torch.Tensor) -> torch.Tensor:
    # slots: [N, C, D]
    if slots.numel() == 0 or slots.shape[-2] <= 1:
        return slots.new_zeros(())
    normed = F.normalize(slots.float(), dim=-1)
    cosine = torch.matmul(normed, normed.transpose(-1, -2))
    c = cosine.shape[-1]
    eye = torch.eye(c, dtype=torch.bool, device=slots.device).view(1, c, c)
    offdiag = cosine.masked_select(~eye.expand(cosine.shape[0], -1, -1))
    return offdiag.square().mean()


def _mean_slot_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.float(), b.float(), dim=-1).mean()


def _normalized_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(F.normalize(a.float(), dim=-1), F.normalize(b.float(), dim=-1))


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

        stats: Dict[str, torch.Tensor] = {}
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
        stats["attention_entropy"] = token_entropy.mean()
        stats["effective_attended_tokens"] = token_entropy.exp().mean()
        stats["sdpa_used"] = x.new_zeros(())
        return self.out_proj(out), stats


class SlotProjectionBank(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.token_q = nn.Linear(d_model, d_model)
        self.token_v = nn.Linear(d_model, d_model)
        self.slot_k = nn.Linear(d_model, d_model)
        self.read_q = nn.Linear(d_model, d_model)
        self.read_k = nn.Linear(d_model, d_model)
        self.read_v = nn.Linear(d_model, d_model)
        self.read_out = nn.Linear(d_model, d_model)


class CausalSlotMemory(nn.Module):
    def __init__(
        self,
        cfg: TransformerConfig,
        layer_idx: int,
        *,
        shared_projections: Optional[SlotProjectionBank] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.slot_count = cfg.slot_count
        self.d_slot = cfg.d_slot
        self.slot_update_scale = float(cfg.slot_update_scale)
        self.proj = shared_projections if shared_projections is not None else SlotProjectionBank(cfg.d_model)
        self.owns_proj = shared_projections is None
        self.learned_slots = nn.Parameter(torch.randn(cfg.slot_count, cfg.d_slot) * 0.02)
        self.token_norm = nn.LayerNorm(cfg.d_model)
        self.slot_norm = nn.LayerNorm(cfg.d_slot)
        self.update_gate = nn.Linear(2 * cfg.d_slot, cfg.d_slot)
        hidden = cfg.slot_mlp_mult * cfg.d_slot
        self.slot_mlp = nn.Sequential(
            nn.LayerNorm(cfg.d_slot),
            nn.Linear(cfg.d_slot, hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, cfg.d_slot),
            nn.Dropout(cfg.dropout),
        )
        self.out_norm = nn.LayerNorm(cfg.d_slot)
        self.innovation_mlp: Optional[nn.Module]
        if cfg.variant == "predictive_slot_residual":
            self.innovation_mlp = nn.Sequential(
                nn.Linear(2 * cfg.d_slot, hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(hidden, cfg.d_slot),
            )
        else:
            self.innovation_mlp = None
        self.read_dropout = nn.Dropout(cfg.dropout)

    def _base_slots(
        self,
        x: torch.Tensor,
        prev_slots: Optional[torch.Tensor],
        stable_slots: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        if stable_slots is not None:
            return stable_slots.to(dtype=x.dtype)
        if prev_slots is not None:
            return prev_slots.to(dtype=x.dtype)
        learned = self.learned_slots.to(dtype=x.dtype).view(1, 1, self.slot_count, self.d_slot)
        return learned.expand(bsz, seq_len, self.slot_count, self.d_slot)

    def _slot_control(self, slots: torch.Tensor, control: Optional[str]) -> torch.Tensor:
        if control == "slot_zero":
            return torch.zeros_like(slots)
        if control == "slot_random":
            scale = slots.detach().float().std().to(dtype=slots.dtype).clamp_min(1.0e-3)
            return torch.randn_like(slots) * scale
        if control == "slot_shuffle" and slots.shape[0] > 1:
            perm = torch.randperm(slots.shape[0], device=slots.device)
            return slots.index_select(0, perm)
        return slots

    def forward(
        self,
        x: torch.Tensor,
        *,
        prev_slots: Optional[torch.Tensor],
        stable_slots: Optional[torch.Tensor],
        control: Optional[str] = None,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        source = x
        if control == "hidden_state_shuffle" and x.shape[0] > 1:
            perm = torch.randperm(x.shape[0], device=x.device)
            source = x.index_select(0, perm)

        token = self.token_norm(source)
        base = self._base_slots(x, prev_slots, stable_slots)
        base_n = self.slot_norm(base)
        token_q = self.proj.token_q(token)
        slot_k = self.proj.slot_k(base_n)
        logits = torch.einsum("btd,btcd->btc", token_q, slot_k) / math.sqrt(self.d_slot)
        update_weights = F.softmax(logits.float(), dim=-1).to(dtype=x.dtype)
        token_v = self.proj.token_v(token)
        update = update_weights.unsqueeze(-1) * token_v.unsqueeze(2)
        gate = torch.sigmoid(self.update_gate(torch.cat((base_n, update), dim=-1)))
        raw_slots = base_n + self.slot_update_scale * gate * update
        raw_slots = self.out_norm(raw_slots + self.slot_mlp(raw_slots))

        if stable_slots is not None:
            if self.innovation_mlp is None:
                raise RuntimeError("stable_slots were provided to a non-predictive slot module")
            stable = stable_slots.to(dtype=x.dtype)
            delta = self.innovation_mlp(torch.cat((raw_slots, stable), dim=-1))
            slots = self.out_norm(stable + delta)
            stable_norm = stable.float().norm(dim=-1).mean()
            innovation_norm = delta.float().norm(dim=-1).mean()
        else:
            slots = raw_slots
            stable_norm = x.new_zeros(())
            innovation_norm = x.new_zeros(())

        read_slots = self._slot_control(slots, control)
        read_q = self.proj.read_q(self.token_norm(x))
        read_k = self.proj.read_k(read_slots)
        read_logits = torch.einsum("btd,btcd->btc", read_q, read_k) / math.sqrt(self.d_slot)
        read_weights = F.softmax(read_logits.float(), dim=-1).to(dtype=x.dtype)
        read_v = self.proj.read_v(read_slots)
        slot_context = torch.einsum("btc,btcd->btd", read_weights, read_v)
        slot_out = self.proj.read_out(self.read_dropout(slot_context))
        enriched = x + slot_out

        stats: Dict[str, torch.Tensor] = {
            "slot_norm": slots.float().norm(dim=-1).mean(),
            "slot_diversity": _offdiag_cosine_squared(slots[:, :: max(1, self.cfg.align_stride)].reshape(-1, self.slot_count, self.d_slot)),
            "token_to_slot_entropy": _entropy(update_weights.float(), dim=-1).mean(),
            "slot_read_entropy": _entropy(read_weights.float(), dim=-1).mean(),
            "token_to_slot_max": update_weights.float().amax(dim=-1).mean(),
            "slot_read_max": read_weights.float().amax(dim=-1).mean(),
            "stable_norm": stable_norm,
            "innovation_norm": innovation_norm,
            "innovation_stable_ratio": innovation_norm / stable_norm.clamp_min(1.0e-6),
        }
        if return_diagnostics:
            utilization = (read_weights.float().mean(dim=(0, 1)) > (0.25 / float(self.slot_count))).float().mean()
            mass = update_weights.float().mean(dim=0).transpose(0, 1)
            mass = mass / mass.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
            stats["slot_utilization"] = utilization
            stats["slot_to_token_entropy"] = _entropy(mass, dim=-1).mean()
            diagnostics = {
                "token_to_slot": update_weights.detach(),
                "slot_read": read_weights.detach(),
                "slots": slots.detach(),
            }
        else:
            diagnostics = None
        return enriched, slots, stats, diagnostics


class TransformerBlock(nn.Module):
    def __init__(
        self,
        cfg: TransformerConfig,
        layer_idx: int,
        *,
        shared_slot_projections: Optional[SlotProjectionBank] = None,
    ) -> None:
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
        self.slot_memory = (
            CausalSlotMemory(cfg, layer_idx, shared_projections=shared_slot_projections)
            if cfg.variant in SLOT_VARIANTS
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        prev_slots: Optional[torch.Tensor],
        stable_slots: Optional[torch.Tensor],
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

        slot_diag: Optional[Dict[str, torch.Tensor]] = None
        slots: Optional[torch.Tensor] = None
        if self.slot_memory is not None:
            x, slots, slot_stats, slot_diag = self.slot_memory(
                x,
                prev_slots=prev_slots,
                stable_slots=stable_slots,
                control=control,
                return_diagnostics=return_diagnostics,
            )
            stats.update(slot_stats)
        return x, slots, stats, slot_diag


class SlotStateTransformer(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        shared_slots = SlotProjectionBank(cfg.d_model) if cfg.variant == "mirrored_slot_attention" else None
        self.shared_slot_projections = shared_slots
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    cfg,
                    i,
                    shared_slot_projections=shared_slots if cfg.variant == "mirrored_slot_attention" else None,
                )
                for i in range(cfg.n_layers)
            ]
        )
        needs_slot_predictors = cfg.variant in {"slot_alignment_loss", "predictive_slot_residual"}
        self.slot_predictors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(cfg.d_slot),
                    nn.Linear(cfg.d_slot, cfg.d_slot),
                )
                for _ in range(max(0, cfg.n_layers - 1))
            ]
            if needs_slot_predictors
            else []
        )
        self.mirror_predictors = nn.ModuleList(
            [nn.Linear(cfg.d_model, cfg.d_model) for _ in range(max(0, cfg.n_layers - 1))]
            if cfg.variant == "mirrored_slot_attention"
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
    def has_slots(self) -> bool:
        return self.cfg.variant in SLOT_VARIANTS

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _alignment_losses(self, slot_states: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        if len(slot_states) < 2 or len(self.slot_predictors) < len(slot_states) - 1:
            zero = self.token_emb.weight.new_zeros(())
            return zero, zero, []
        align_losses: List[torch.Tensor] = []
        pred_sims: List[torch.Tensor] = []
        stride = max(1, int(self.cfg.align_stride))
        for idx in range(len(slot_states) - 1):
            current = slot_states[idx][:, ::stride]
            target = slot_states[idx + 1][:, ::stride]
            pred = self.slot_predictors[idx](current)
            align_losses.append(1.0 - F.cosine_similarity(pred.float(), target.detach().float(), dim=-1).mean())
            pred_sims.append(_mean_slot_cosine(pred.detach(), target.detach()))
        return torch.stack(align_losses).mean(), torch.stack(pred_sims).mean(), pred_sims

    def _diversity_loss(self, slot_states: Sequence[torch.Tensor]) -> torch.Tensor:
        if not slot_states:
            return self.token_emb.weight.new_zeros(())
        stride = max(1, int(self.cfg.align_stride))
        losses = [
            _offdiag_cosine_squared(slots[:, ::stride].reshape(-1, self.cfg.slot_count, self.cfg.d_slot))
            for slots in slot_states
        ]
        return torch.stack(losses).mean()

    def _cross_layer_slot_similarity(self, slot_states: Sequence[torch.Tensor]) -> tuple[torch.Tensor, List[torch.Tensor]]:
        if len(slot_states) < 2:
            return self.token_emb.weight.new_zeros(()), []
        sims = []
        stride = max(1, int(self.cfg.align_stride))
        for idx in range(len(slot_states) - 1):
            sims.append(_mean_slot_cosine(slot_states[idx][:, ::stride].detach(), slot_states[idx + 1][:, ::stride].detach()))
        return torch.stack(sims).mean(), sims

    def _mirror_loss(self, hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
        if self.cfg.variant != "mirrored_slot_attention" or len(hidden_states) < 2:
            return self.token_emb.weight.new_zeros(())
        stride = max(1, int(self.cfg.mirror_stride))
        losses = []
        projector = self.shared_slot_projections.token_q if self.shared_slot_projections is not None else None
        if projector is None:
            return self.token_emb.weight.new_zeros(())
        for idx in range(len(hidden_states) - 1):
            current = projector(hidden_states[idx][:, ::stride])
            target = projector(hidden_states[idx + 1][:, ::stride])
            pred = self.mirror_predictors[idx](current)
            losses.append(_normalized_mse(pred, target.detach()))
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
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        prev_slots: Optional[torch.Tensor] = None
        slot_states: List[torch.Tensor] = []
        hidden_states: List[torch.Tensor] = []
        layer_stats: List[Dict[str, torch.Tensor]] = []
        slot_attn: List[Optional[Dict[str, torch.Tensor]]] = []

        for layer_idx, block in enumerate(self.blocks):
            stable_slots: Optional[torch.Tensor] = None
            if (
                self.cfg.variant == "predictive_slot_residual"
                and layer_idx > 0
                and prev_slots is not None
            ):
                stable_slots = self.slot_predictors[layer_idx - 1](prev_slots)
            x, slots, stats, diag = block(
                x,
                prev_slots=prev_slots,
                stable_slots=stable_slots,
                control=control,
                return_diagnostics=return_diagnostics,
            )
            hidden_states.append(x)
            if slots is not None:
                slot_states.append(slots)
                prev_slots = slots
            layer_stats.append(stats)
            slot_attn.append(diag)

        logits = self.lm_head(self.ln_f(x))
        diversity = self._diversity_loss(slot_states)
        alignment, pred_sim, pred_sim_by_pair = self._alignment_losses(slot_states)
        mirror = self._mirror_loss(hidden_states)
        cross_layer, cross_layer_by_pair = self._cross_layer_slot_similarity(slot_states)
        aux_losses = {
            "slot_diversity_loss": diversity,
            "slot_alignment_loss": alignment,
            "mirror_loss": mirror,
        }
        diagnostics = {
            "cross_layer_slot_cosine": cross_layer.detach(),
            "cross_layer_slot_cosine_by_pair": [item.detach() for item in cross_layer_by_pair],
            "predicted_actual_slot_cosine": pred_sim.detach(),
            "predicted_actual_slot_cosine_by_pair": [item.detach() for item in pred_sim_by_pair],
        }
        return {
            "logits": logits,
            "layer_stats": layer_stats,
            "slot_attn": slot_attn if return_diagnostics else None,
            "slot_states": slot_states if return_diagnostics else None,
            "aux_losses": aux_losses,
            "diagnostics": diagnostics,
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
            "slot_variant": self.has_slots,
            "note": "Slot operations are dense tensor projections over causal hidden states; token self-attention uses SDPA during training.",
        }


def build_model(raw_config: Dict[str, Any]) -> SlotStateTransformer:
    cfg = TransformerConfig.from_dict(raw_config)
    return SlotStateTransformer(cfg)


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
