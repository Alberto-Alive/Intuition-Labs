from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerUncertaintyConfig:
    vocab_size: int = 64
    seq_len: int = 18
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    num_refine_layers: int = 1
    dim_feedforward: int = 128
    dropout: float = 0.05
    num_classes: int = 2
    num_views: int = 6
    num_prototypes: int = 16


class MLP(nn.Module):
    def __init__(self, dims, dropout: float = 0.0, last_activation: bool = False):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2 or last_activation:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBackbone(nn.Module):
    def __init__(self, cfg: TransformerUncertaintyConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.cls = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, cfg.seq_len + 1, cfg.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        tok = self.token_emb(x)
        cls = self.cls.expand(bsz, -1, -1)
        h = torch.cat([cls, tok], dim=1) + self.pos[:, : x.shape[1] + 1, :]
        return self.norm(self.encoder(h))


class RefinerBlock(nn.Module):
    """TransformerEncoderLayer equivalent that can return self-attention."""

    def __init__(self, cfg: TransformerUncertaintyConfig):
        super().__init__()
        d = cfg.d_model
        self.self_attn = nn.MultiheadAttention(d, cfg.nhead, dropout=cfg.dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.linear1 = nn.Linear(d, cfg.dim_feedforward)
        self.linear2 = nn.Linear(cfg.dim_feedforward, d)
        self.dropout = nn.Dropout(cfg.dropout)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, need_weights: bool = False) -> Tuple[torch.Tensor, torch.Tensor | None]:
        h = self.norm1(x)
        attn_out, attn = self.self_attn(
            h,
            h,
            h,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        x = x + self.dropout1(attn_out)
        h = self.norm2(x)
        h = self.linear2(self.dropout(F.gelu(self.linear1(h))))
        x = x + self.dropout2(h)
        return x, attn if need_weights else None


class SharedAttentionRefiner(nn.Module):
    def __init__(self, cfg: TransformerUncertaintyConfig):
        super().__init__()
        self.layers = nn.ModuleList([RefinerBlock(cfg) for _ in range(cfg.num_refine_layers)])

    def forward(self, x: torch.Tensor, need_weights: bool = False) -> Tuple[torch.Tensor, torch.Tensor | None]:
        last_attn = None
        for layer in self.layers:
            x, last_attn = layer(x, need_weights=need_weights)
        return x, last_attn


class MDGSExtractor(nn.Module):
    """Extract margin/disagreement/geometry/support from current attention memory.

    Witness queries attend to the same encoded token memory used by prediction.
    The primitive tokens are then allowed to attend to each other.
    """

    def __init__(self, cfg: TransformerUncertaintyConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.witness_tokens = nn.Parameter(torch.randn(cfg.num_views, d) * 0.02)
        self.witness_attn = nn.MultiheadAttention(d, cfg.nhead, dropout=cfg.dropout, batch_first=True)
        self.witness_norm = nn.LayerNorm(d)
        self.view_head = nn.Linear(d, cfg.num_classes)

        self.prototypes = nn.Parameter(torch.randn(cfg.num_prototypes, d) * 0.2)
        self.support_temperature = nn.Parameter(torch.tensor(1.0))

        self.m_embed = MLP([4, d, d], cfg.dropout)
        self.d_embed = MLP([4, d, d], cfg.dropout)
        self.g_embed = MLP([4, d, d], cfg.dropout)
        self.s_embed = MLP([4, d, d], cfg.dropout)

        self.primitive_attn = nn.MultiheadAttention(d, cfg.nhead, dropout=cfg.dropout, batch_first=True)
        self.primitive_norm1 = nn.LayerNorm(d)
        self.primitive_ffn = MLP([d, d * 2, d], cfg.dropout)
        self.primitive_norm2 = nn.LayerNorm(d)

        self.m_scalar_head = MLP([d, d, 1], cfg.dropout)
        self.d_scalar_head = MLP([d, d, 1], cfg.dropout)
        self.g_scalar_head = MLP([d, d, 1], cfg.dropout)
        self.uncertainty_head = MLP([d, d, 1], cfg.dropout)
        self.support_head = MLP([d, d, 1], cfg.dropout)

    @staticmethod
    def _entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return -(probs * (probs + eps).log()).sum(dim=-1)

    def _support_features(self, cls: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_n = F.normalize(cls, dim=-1)
        p_n = F.normalize(self.prototypes, dim=-1)
        cos = h_n @ p_n.t()
        max_cos = cos.max(dim=-1).values
        topk_cos = cos.topk(k=min(3, cos.shape[-1]), dim=-1).values.mean(dim=-1)
        dist2 = torch.cdist(cls, self.prototypes, p=2).pow(2)
        min_dist = dist2.min(dim=-1).values
        top_dist = dist2.topk(k=min(3, dist2.shape[-1]), largest=False, dim=-1).values.mean(dim=-1)
        temp = F.softplus(self.support_temperature) + 1e-4
        support_proto = torch.sigmoid(max_cos - min_dist / temp)
        features = torch.stack([max_cos, topk_cos, -min_dist, -top_dist], dim=-1)
        return features, support_proto

    def forward(self, memory: torch.Tensor, base_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        cls = memory[:, 0, :]
        bsz = memory.shape[0]
        witness_query = cls.unsqueeze(1) + self.witness_tokens.unsqueeze(0)
        views, witness_attn = self.witness_attn(witness_query, memory, memory, need_weights=True)
        views = self.witness_norm(views + witness_query)
        view_logits = self.view_head(views)

        base_probs = F.softmax(base_logits, dim=-1)
        top2 = torch.topk(base_logits, k=2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        confidence = base_probs.max(dim=-1).values
        entropy = self._entropy(base_probs)
        logit_norm = base_logits.norm(dim=-1)
        m_feat = torch.stack([margin, confidence, -entropy, logit_norm], dim=-1)

        view_probs = F.softmax(view_logits, dim=-1)
        mean_probs = view_probs.mean(dim=1)
        prob_var = view_probs.var(dim=1).mean(dim=-1)
        view_entropy_mean = self._entropy(view_probs).mean(dim=1)
        mean_entropy = self._entropy(mean_probs)
        kl = (
            view_probs
            * ((view_probs + 1e-8).log() - (mean_probs.unsqueeze(1) + 1e-8).log())
        ).sum(dim=-1).mean(dim=1)
        d_feat = torch.stack([prob_var, mean_entropy, view_entropy_mean, kl], dim=-1)

        v_n = F.normalize(views, dim=-1)
        cos = torch.matmul(v_n, v_n.transpose(1, 2))
        mask = ~torch.eye(views.shape[1], device=views.device, dtype=torch.bool).unsqueeze(0)
        pair_cos = cos[mask.expand(bsz, -1, -1)].view(bsz, views.shape[1] * (views.shape[1] - 1))
        avg_cos_dist = 1.0 - pair_cos.mean(dim=-1)
        center = views.mean(dim=1, keepdim=True)
        spread = (views - center).pow(2).sum(dim=-1).mean(dim=-1).sqrt()
        norm_std = views.norm(dim=-1).std(dim=1)
        cov_trace = (views - center).pow(2).mean(dim=(1, 2))
        g_feat = torch.stack([avg_cos_dist, spread, norm_std, cov_trace], dim=-1)

        s_feat, support_proto = self._support_features(cls)

        p_m = self.m_embed(m_feat)
        p_d = self.d_embed(d_feat)
        p_g = self.g_embed(g_feat)
        p_s = self.s_embed(s_feat)
        raw_primitive_tokens = torch.stack([p_m, p_d, p_g, p_s], dim=1)

        attn_out, primitive_attn = self.primitive_attn(
            raw_primitive_tokens,
            raw_primitive_tokens,
            raw_primitive_tokens,
            need_weights=True,
        )
        primitive_tokens = self.primitive_norm1(raw_primitive_tokens + attn_out)
        primitive_tokens = self.primitive_norm2(primitive_tokens + self.primitive_ffn(primitive_tokens))
        u = primitive_tokens.mean(dim=1)

        m_scalar = torch.sigmoid(self.m_scalar_head(p_m)).squeeze(-1)
        d_scalar = torch.sigmoid(self.d_scalar_head(p_d)).squeeze(-1)
        g_scalar = torch.sigmoid(self.g_scalar_head(p_g)).squeeze(-1)
        uncertainty = torch.sigmoid(self.uncertainty_head(u)).squeeze(-1)
        support = torch.sigmoid(self.support_head(u)).squeeze(-1)

        return {
            "views": views,
            "view_logits": view_logits,
            "witness_attn": witness_attn,
            "primitive_tokens": primitive_tokens,
            "raw_primitive_tokens": raw_primitive_tokens,
            "primitive_attn": primitive_attn,
            "uncertainty_context": u,
            "uncertainty": uncertainty,
            "support": support,
            "m_scalar": m_scalar,
            "d_scalar": d_scalar,
            "g_scalar": g_scalar,
            "s_proto": support_proto,
            "M_margin": margin,
            "D_disagreement": prob_var,
            "G_spread": spread,
        }


class BaselineTransformer(nn.Module):
    def __init__(self, cfg: TransformerUncertaintyConfig):
        super().__init__()
        self.backbone = TransformerBackbone(cfg)
        self.head = nn.Linear(cfg.d_model, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        memory = self.backbone(x)
        logits = self.head(memory[:, 0, :])
        probs = F.softmax(logits, dim=-1)
        conf = probs.max(dim=-1).values
        return {
            "memory": memory,
            "logits": logits,
            "uncertainty": 1.0 - conf,
            "support": conf,
            "commitment": conf,
        }


class AuxMDGSTransformer(nn.Module):
    """Iteration 1: MDGS improves training as an auxiliary teacher only."""

    def __init__(self, cfg: TransformerUncertaintyConfig, mdgs_mode: str = "real", freeze_mdgs: bool = False):
        super().__init__()
        self.mdgs_mode = mdgs_mode
        self.backbone = TransformerBackbone(cfg)
        self.base_head = nn.Linear(cfg.d_model, cfg.num_classes)
        self.mdgs = MDGSExtractor(cfg)
        if freeze_mdgs:
            for param in self.mdgs.parameters():
                param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        memory = self.backbone(x)
        logits = self.base_head(memory[:, 0, :])
        probs = F.softmax(logits, dim=-1)
        conf = probs.max(dim=-1).values
        if self.mdgs_mode == "none":
            return {
                "memory": memory,
                "logits": logits,
                "uncertainty": 1.0 - conf,
                "support": conf,
                "commitment": conf,
            }

        mdgs_memory = memory.detach() if self.mdgs_mode == "detached" else memory
        mdgs_logits = logits.detach() if self.mdgs_mode == "detached" else logits
        if self.mdgs_mode not in {"real", "detached"}:
            raise ValueError(f"unknown mdgs_mode={self.mdgs_mode!r}")
        mdgs = self.mdgs(mdgs_memory, mdgs_logits)
        final_margin = torch.topk(logits, k=2, dim=-1).values
        final_margin = final_margin[:, 0] - final_margin[:, 1]
        commitment = torch.sigmoid(mdgs["support"] * final_margin - mdgs["uncertainty"])
        return {
            "memory": memory,
            "logits": logits,
            "commitment": commitment,
            **mdgs,
        }


class GatedMDGSTransformer(nn.Module):
    """Iteration 2: MDGS context gates a residual into the prediction state."""

    def __init__(self, cfg: TransformerUncertaintyConfig):
        super().__init__()
        self.backbone = TransformerBackbone(cfg)
        self.base_head = nn.Linear(cfg.d_model, cfg.num_classes)
        self.mdgs = MDGSExtractor(cfg)
        self.adapter = MLP([cfg.d_model, cfg.d_model, cfg.d_model], cfg.dropout)
        self.gate = MLP([cfg.d_model * 2, cfg.d_model, cfg.d_model], cfg.dropout)
        self.final_head = nn.Linear(cfg.d_model, cfg.num_classes)
        if isinstance(self.gate.net[-1], nn.Linear):
            nn.init.constant_(self.gate.net[-1].bias, -2.0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        memory = self.backbone(x)
        cls = memory[:, 0, :]
        base_logits = self.base_head(cls)
        mdgs = self.mdgs(memory, base_logits)
        u = mdgs["uncertainty_context"]
        gate = torch.sigmoid(self.gate(torch.cat([cls, u], dim=-1)))
        cls_final = cls + gate * self.adapter(u)
        logits = self.final_head(cls_final)
        final_margin = torch.topk(logits, k=2, dim=-1).values
        final_margin = final_margin[:, 0] - final_margin[:, 1]
        commitment = torch.sigmoid(mdgs["support"] * final_margin - mdgs["uncertainty"])
        return {
            "memory": memory,
            "base_logits": base_logits,
            "logits": logits,
            "gate": gate.mean(dim=-1),
            "commitment": commitment,
            **mdgs,
        }


class SharedAttentionMDGSTransformer(nn.Module):
    """Iteration 3: prediction and MDGS tokens share a final attention block.

    This is the closest implementation of "uncertainty uses the same attention
    the prediction uses." Primitive reliability tokens join the CLS token and
    content memory, then the final classifier reads the refined CLS state.
    """

    def __init__(self, cfg: TransformerUncertaintyConfig, mdgs_mode: str = "real", freeze_mdgs: bool = False):
        super().__init__()
        self.cfg = cfg
        self.mdgs_mode = mdgs_mode
        self.backbone = TransformerBackbone(cfg)
        self.base_head = nn.Linear(cfg.d_model, cfg.num_classes)
        self.mdgs = MDGSExtractor(cfg)
        if freeze_mdgs:
            for param in self.mdgs.parameters():
                param.requires_grad_(False)
        self.cls_type = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.primitive_type = nn.Parameter(torch.randn(1, 4, cfg.d_model) * 0.02)
        self.content_type = nn.Parameter(torch.randn(1, cfg.seq_len, cfg.d_model) * 0.02)
        self.null_primitive_tokens = nn.Parameter(torch.randn(1, 4, cfg.d_model) * 0.02)
        self.refiner = SharedAttentionRefiner(cfg)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.final_head = nn.Linear(cfg.d_model, cfg.num_classes)
        self.final_uncertainty_head = MLP([cfg.d_model, cfg.d_model, 1], cfg.dropout)
        self.final_support_head = MLP([cfg.d_model, cfg.d_model, 1], cfg.dropout)

    def _primitive_tokens_for_prediction(self, primitive_tokens: torch.Tensor) -> torch.Tensor:
        if self.mdgs_mode == "real":
            return primitive_tokens
        if self.mdgs_mode == "none":
            return self.null_primitive_tokens.expand(primitive_tokens.shape[0], -1, -1)
        if self.mdgs_mode == "random":
            scale = primitive_tokens.detach().std().clamp_min(1e-3)
            mean = primitive_tokens.detach().mean()
            return torch.randn_like(primitive_tokens) * scale + mean
        if self.mdgs_mode == "shuffled":
            if primitive_tokens.shape[0] <= 1:
                return primitive_tokens
            order = torch.randperm(primitive_tokens.shape[0], device=primitive_tokens.device)
            return primitive_tokens[order]
        if self.mdgs_mode == "detached":
            return primitive_tokens.detach()
        raise ValueError(f"unknown mdgs_mode={self.mdgs_mode!r}")

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Dict[str, torch.Tensor]:
        memory = self.backbone(x)
        cls = memory[:, 0:1, :]
        content = memory[:, 1:, :]
        base_logits = self.base_head(cls.squeeze(1))
        mdgs = self.mdgs(memory, base_logits)
        primitive_tokens = self._primitive_tokens_for_prediction(mdgs["primitive_tokens"])

        joint = torch.cat(
            [
                cls + self.cls_type,
                primitive_tokens + self.primitive_type,
                content + self.content_type[:, : content.shape[1], :],
            ],
            dim=1,
        )
        refined, refiner_attn = self.refiner(joint, need_weights=return_attention)
        refined = self.norm(refined)
        cls_final = refined[:, 0, :]
        primitive_final = refined[:, 1:5, :].mean(dim=1)
        logits = self.final_head(cls_final)

        uncertainty = torch.sigmoid(self.final_uncertainty_head(primitive_final)).squeeze(-1)
        support = torch.sigmoid(self.final_support_head(primitive_final)).squeeze(-1)
        final_margin = torch.topk(logits, k=2, dim=-1).values
        final_margin = final_margin[:, 0] - final_margin[:, 1]
        commitment = torch.sigmoid(support * final_margin - uncertainty)

        out = {
            "memory": memory,
            "base_logits": base_logits,
            "logits": logits,
            "joint_tokens": refined,
            "uncertainty": uncertainty,
            "support": support,
            "commitment": commitment,
        }
        if refiner_attn is not None:
            out["refiner_attn"] = refiner_attn
        out.update({k: v for k, v in mdgs.items() if k not in ("uncertainty", "support")})
        return out


def build_model(name: str, cfg: TransformerUncertaintyConfig) -> nn.Module:
    if name == "baseline":
        return BaselineTransformer(cfg)
    if name == "aux":
        return AuxMDGSTransformer(cfg)
    if name == "aux_no_mdgs":
        return AuxMDGSTransformer(cfg, mdgs_mode="none")
    if name == "aux_random_mdgs":
        return AuxMDGSTransformer(cfg)
    if name == "aux_shuffled_mdgs":
        return AuxMDGSTransformer(cfg)
    if name == "aux_detached_mdgs":
        return AuxMDGSTransformer(cfg, mdgs_mode="detached")
    if name == "aux_frozen_mdgs":
        return AuxMDGSTransformer(cfg, freeze_mdgs=True)
    if name == "gated":
        return GatedMDGSTransformer(cfg)
    if name == "gated_anchor":
        return GatedMDGSTransformer(cfg)
    if name == "shared":
        return SharedAttentionMDGSTransformer(cfg)
    if name == "shared_anchor":
        return SharedAttentionMDGSTransformer(cfg)
    if name == "shared_anchor_no_mdgs":
        return SharedAttentionMDGSTransformer(cfg, mdgs_mode="none")
    if name == "shared_anchor_random_mdgs":
        return SharedAttentionMDGSTransformer(cfg, mdgs_mode="random")
    if name == "shared_anchor_shuffled_mdgs":
        return SharedAttentionMDGSTransformer(cfg, mdgs_mode="shuffled")
    if name == "shared_anchor_detached_mdgs":
        return SharedAttentionMDGSTransformer(cfg, mdgs_mode="detached")
    if name == "shared_anchor_frozen_mdgs":
        return SharedAttentionMDGSTransformer(cfg, mdgs_mode="real", freeze_mdgs=True)
    raise ValueError(f"unknown model: {name}")
