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
    "single_avenue",
    "four_avenue_sum",
    "four_avenue_gated",
    "four_avenue_dropout05",
    "four_avenue_leave_one_out",
    "mirrored_four_avenue",
}

AVENUE_VARIANTS = {
    "single_avenue",
    "four_avenue_sum",
    "four_avenue_gated",
    "four_avenue_dropout05",
    "four_avenue_leave_one_out",
    "mirrored_four_avenue",
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
    max_seq_len: int = 256
    param_adapter_dim: int = 768
    num_avenues: int = 4
    avenue_dim: int = 128
    avenue_dropout: float = 0.0
    leave_one_prob: float = 1.0
    align_stride: int = 8
    avenue_scale_init: float = 0.1

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
        if cfg.variant == "single_avenue":
            cfg.num_avenues = 1
        if cfg.variant in AVENUE_VARIANTS and cfg.num_avenues <= 0:
            raise ValueError("num_avenues must be positive")
        if cfg.align_stride <= 0:
            raise ValueError("align_stride must be positive")
        return cfg


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()


def _entropy(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    probs_f = probs.float().clamp_min(1.0e-8)
    return -(probs_f * probs_f.log()).sum(dim=dim)


def _offdiag_mean_cosine(x: torch.Tensor) -> torch.Tensor:
    # x: [N, A, D]
    if x.numel() == 0 or x.shape[-2] <= 1:
        return x.new_zeros(())
    normed = F.normalize(x.float(), dim=-1)
    cosine = torch.matmul(normed, normed.transpose(-1, -2))
    a = cosine.shape[-1]
    eye = torch.eye(a, dtype=torch.bool, device=x.device).view(1, a, a)
    return cosine.masked_select(~eye.expand(cosine.shape[0], -1, -1)).mean()


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


class MultiAvenueModule(nn.Module):
    def __init__(self, cfg: TransformerConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.num_avenues = 1 if cfg.variant == "single_avenue" else int(cfg.num_avenues)
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
        if self.cfg.variant == "four_avenue_dropout05" and self.cfg.avenue_dropout > 0:
            keep = torch.rand(bsz, self.num_avenues, device=device) >= float(self.cfg.avenue_dropout)
            all_dropped = ~keep.any(dim=-1)
            if bool(all_dropped.any()):
                chosen = torch.randint(0, self.num_avenues, (int(all_dropped.sum().item()),), device=device)
                keep[all_dropped] = False
                keep[all_dropped, chosen] = True
            return keep.float().view(bsz, 1, self.num_avenues, 1)
        if self.cfg.variant == "four_avenue_leave_one_out":
            keep = torch.ones(bsz, self.num_avenues, device=device)
            should_drop = torch.rand(bsz, device=device) < float(self.cfg.leave_one_prob)
            if bool(should_drop.any()):
                drop_idx = torch.randint(0, self.num_avenues, (int(should_drop.sum().item()),), device=device)
                keep[should_drop, drop_idx] = 0.0
            return keep.view(bsz, 1, self.num_avenues, 1)
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
        gate_logits = self.gate(x)
        gates = F.softmax(gate_logits.float(), dim=-1).to(dtype=x.dtype)
        if self.num_avenues == 1:
            gates = torch.ones_like(gates)

        is_sum = self.cfg.variant == "four_avenue_sum"
        if is_sum:
            active = (outputs.float().norm(dim=-1) > 0).float()
            denom = active.sum(dim=-1, keepdim=True).clamp_min(1.0).to(dtype=x.dtype)
            mix = outputs.sum(dim=2) / denom
            diag_gates = active / active.sum(dim=-1, keepdim=True).clamp_min(1.0)
        elif eval_mask is not None and (control or "").startswith("single_avenue_"):
            mix = outputs.sum(dim=2)
            diag_gates = eval_mask[..., 0].expand(x.shape[0], x.shape[1], -1)
        else:
            active = (outputs.float().norm(dim=-1) > 0).float().to(dtype=x.dtype)
            masked_gates = gates * active
            masked_gates = masked_gates / masked_gates.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
            mix = torch.einsum("bta,btad->btd", masked_gates, outputs)
            diag_gates = masked_gates

        enriched = x + self.scale.to(dtype=x.dtype) * self.merge(mix)
        sampled = outputs[:, :: max(1, int(self.cfg.align_stride))].reshape(-1, self.num_avenues, self.cfg.d_model)
        pairwise_cos = _offdiag_mean_cosine(sampled)
        stats: Dict[str, torch.Tensor] = {
            "avenue_gate_entropy": _entropy(diag_gates.float(), dim=-1).mean(),
            "avenue_output_norm": outputs.float().norm(dim=-1).mean(),
            "pairwise_avenue_cosine": pairwise_cos,
            "avenue_diversity": 1.0 - pairwise_cos,
            "avenue_scale": self.scale.detach().float(),
        }
        usage = diag_gates.float().mean(dim=(0, 1))
        for idx in range(self.num_avenues):
            stats[f"avenue_usage_{idx}"] = usage[idx]
        for idx in range(self.num_avenues, 4):
            stats[f"avenue_usage_{idx}"] = x.new_zeros(())

        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "gates": diag_gates.detach(),
                "avenues": outputs.detach(),
            }
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
        self.avenue = MultiAvenueModule(cfg, layer_idx) if cfg.variant in AVENUE_VARIANTS else None

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


class MultiAvenueTransformer(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg.n_layers)])
        self.avenue_predictors = nn.ModuleList(
            [nn.Linear(cfg.d_model, cfg.d_model) for _ in range(max(0, cfg.n_layers - 1))]
            if cfg.variant == "mirrored_four_avenue"
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
        return self.cfg.variant in AVENUE_VARIANTS

    @property
    def num_avenues(self) -> int:
        return 1 if self.cfg.variant == "single_avenue" else int(self.cfg.num_avenues)

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
            "note": "Avenues are per-token learned pathways; token self-attention uses SDPA during training.",
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
