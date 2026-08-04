from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    input_dim: int = 2
    hidden_dim: int = 128
    latent_dim: int = 64
    token_dim: int = 64
    num_classes: int = 2
    num_views: int = 6
    num_prototypes: int = 16
    dropout: float = 0.05
    use_jepa: bool = True
    ema_decay: float = 0.99


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

    def forward(self, x):
        return self.net(x)


class BaselineMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.encoder = MLP([cfg.input_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.latent_dim], cfg.dropout)
        self.head = nn.Linear(cfg.latent_dim, cfg.num_classes)

    def forward(
        self,
        x: torch.Tensor,
        jepa_target_x: torch.Tensor | None = None,
        gate_scale: float = 1.0,
        disable_fusion: bool = False,
        disable_primitive_attn: bool = False,
        support_only: bool = False,
    ) -> Dict[str, torch.Tensor | None]:
        h = self.encoder(x)
        logits = self.head(h)
        probs = F.softmax(logits, dim=-1)
        conf = probs.max(dim=-1).values
        return {
            "h": h,
            "logits": logits,
            "base_logits": logits,
            "uncertainty": 1.0 - conf,
            "support": conf,
            "commitment": conf,
            "gate": torch.zeros_like(conf),
        }


class UGLYNet(nn.Module):
    """Uncertainty-Guided Latent Yielding.

    System A: prediction path.
    System B: primitive reliability geometry path.

    System B is inside the forward pass:
        h -> M/D/G/S primitive tokens -> primitive attention -> gated fusion -> final logits
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = MLP([cfg.input_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.latent_dim], cfg.dropout)
        self.base_head = nn.Linear(cfg.latent_dim, cfg.num_classes)

        # Learned view tokens create K internal perspectives over the same latent h.
        self.view_tokens = nn.Parameter(torch.randn(cfg.num_views, cfg.latent_dim) * 0.02)
        self.view_mlp = MLP([cfg.latent_dim, cfg.hidden_dim, cfg.latent_dim], cfg.dropout)
        self.view_norm = nn.LayerNorm(cfg.latent_dim)
        self.view_head = nn.Linear(cfg.latent_dim, cfg.num_classes)

        # Trainable prototypes for support/knownness. These are not class labels;
        # they are latent anchors that should become high-density regions.
        self.prototypes = nn.Parameter(torch.randn(cfg.num_prototypes, cfg.latent_dim) * 0.2)
        self.support_temperature = nn.Parameter(torch.tensor(1.0))

        # Primitive feature embedders. Each primitive becomes a vector token.
        self.M_embed = MLP([4, cfg.token_dim, cfg.token_dim], cfg.dropout)
        self.D_embed = MLP([4, cfg.token_dim, cfg.token_dim], cfg.dropout)
        self.G_embed = MLP([4, cfg.token_dim, cfg.token_dim], cfg.dropout)
        self.S_embed = MLP([4, cfg.token_dim, cfg.token_dim], cfg.dropout)

        self.primitive_attn = nn.MultiheadAttention(
            embed_dim=cfg.token_dim,
            num_heads=4,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.primitive_norm1 = nn.LayerNorm(cfg.token_dim)
        self.primitive_ffn = MLP([cfg.token_dim, cfg.token_dim * 2, cfg.token_dim], cfg.dropout)
        self.primitive_norm2 = nn.LayerNorm(cfg.token_dim)

        self.u_proj = MLP([cfg.token_dim, cfg.latent_dim], cfg.dropout)
        self.gate_net = MLP([cfg.latent_dim + cfg.token_dim, cfg.hidden_dim, cfg.latent_dim], cfg.dropout)
        # Bias the final gate layer negative so System B starts as a small residual.
        if isinstance(self.gate_net.net[-1], nn.Linear):
            nn.init.constant_(self.gate_net.net[-1].bias, -3.0)

        self.final_head = nn.Linear(cfg.latent_dim, cfg.num_classes)
        self.uncertainty_head = MLP([cfg.token_dim, cfg.hidden_dim, 1], cfg.dropout)
        self.support_head = MLP([cfg.token_dim, cfg.hidden_dim, 1], cfg.dropout)

        # JEPA-style target encoder and predictor for latent grounding.
        self.use_jepa = cfg.use_jepa
        self.jepa_predictor = MLP([cfg.latent_dim, cfg.hidden_dim, cfg.latent_dim], cfg.dropout)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_target_encoder(self):
        if not self.use_jepa:
            return
        decay = self.cfg.ema_decay
        for p_online, p_target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            p_target.data.mul_(decay).add_(p_online.data, alpha=1.0 - decay)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def _make_views(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, D]; tokens: [K, D] -> [B, K, D]
        v = h.unsqueeze(1) + self.view_tokens.unsqueeze(0)
        v = self.view_mlp(v)
        v = self.view_norm(v + h.unsqueeze(1))
        return v

    @staticmethod
    def _entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return -(probs * (probs + eps).log()).sum(dim=-1)

    def _support_features(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Cosine support + distance support. Returns feature vector and support scalar.
        h_n = F.normalize(h, dim=-1)
        p_n = F.normalize(self.prototypes, dim=-1)
        cos = h_n @ p_n.t()  # [B, P]
        max_cos = cos.max(dim=-1).values
        topk_cos = cos.topk(k=min(3, cos.shape[-1]), dim=-1).values.mean(dim=-1)

        dist2 = torch.cdist(h, self.prototypes, p=2).pow(2)  # [B, P]
        min_dist = dist2.min(dim=-1).values
        mean_top_dist = dist2.topk(k=min(3, dist2.shape[-1]), largest=False, dim=-1).values.mean(dim=-1)

        temp = F.softplus(self.support_temperature) + 1e-4
        support_scalar = torch.sigmoid(max_cos - min_dist / temp)
        features = torch.stack([
            max_cos,
            topk_cos,
            -min_dist,
            -mean_top_dist,
        ], dim=-1)
        return features, support_scalar

    def _primitive_tokens(
        self,
        h: torch.Tensor,
        base_logits: torch.Tensor,
        views: torch.Tensor,
        view_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, K, D = views.shape
        base_probs = F.softmax(base_logits, dim=-1)
        top2 = torch.topk(base_logits, k=2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        confidence = base_probs.max(dim=-1).values
        entropy = self._entropy(base_probs)
        logit_norm = base_logits.norm(dim=-1)
        M_feat = torch.stack([margin, confidence, -entropy, logit_norm], dim=-1)

        view_probs = F.softmax(view_logits, dim=-1)  # [B, K, C]
        mean_probs = view_probs.mean(dim=1)
        prob_var = view_probs.var(dim=1).mean(dim=-1)
        view_entropy_mean = self._entropy(view_probs).mean(dim=1)
        mean_entropy = self._entropy(mean_probs)
        # Average KL(view || mean)
        kl = (view_probs * ((view_probs + 1e-8).log() - (mean_probs.unsqueeze(1) + 1e-8).log())).sum(dim=-1).mean(dim=1)
        D_feat = torch.stack([prob_var, mean_entropy, view_entropy_mean, kl], dim=-1)

        v_n = F.normalize(views, dim=-1)
        cos = torch.matmul(v_n, v_n.transpose(1, 2))  # [B, K, K]
        mask = ~torch.eye(K, device=views.device, dtype=torch.bool).unsqueeze(0)
        pair_cos = cos[mask.expand(B, -1, -1)].view(B, K * (K - 1))
        avg_cos_dist = 1.0 - pair_cos.mean(dim=-1)
        center = views.mean(dim=1, keepdim=True)
        spread = (views - center).pow(2).sum(dim=-1).mean(dim=-1).sqrt()
        norm_std = views.norm(dim=-1).std(dim=1)
        # Simple covariance trace proxy.
        cov_trace = (views - center).pow(2).mean(dim=(1, 2))
        G_feat = torch.stack([avg_cos_dist, spread, norm_std, cov_trace], dim=-1)

        S_feat, support_proto = self._support_features(h)

        pM = self.M_embed(M_feat)
        pD = self.D_embed(D_feat)
        pG = self.G_embed(G_feat)
        pS = self.S_embed(S_feat)
        tokens = torch.stack([pM, pD, pG, pS], dim=1)  # [B, 4, token_dim]

        scalars = {
            "M_margin": margin,
            "D_disagreement": prob_var,
            "G_spread": spread,
            "S_proto": support_proto,
            "base_confidence": confidence,
            "base_entropy": entropy,
            "view_kl": kl,
            "avg_cos_dist": avg_cos_dist,
        }
        return tokens, scalars

    def forward(
        self,
        x: torch.Tensor,
        jepa_target_x: torch.Tensor | None = None,
        gate_scale: float = 1.0,
        disable_fusion: bool = False,
        disable_primitive_attn: bool = False,
        support_only: bool = False,
    ) -> Dict[str, torch.Tensor | None]:
        h = self.encoder(x)
        base_logits = self.base_head(h)

        views = self._make_views(h)
        view_logits = self.view_head(views)
        primitive_tokens, scalars = self._primitive_tokens(h, base_logits, views, view_logits)
        if support_only:
            primitive_tokens = primitive_tokens[:, 3:4, :]

        if disable_primitive_attn or support_only:
            z = primitive_tokens
            attn_weights = None
            u = z.mean(dim=1)
        else:
            attn_out, attn_weights = self.primitive_attn(primitive_tokens, primitive_tokens, primitive_tokens, need_weights=True)
            z = self.primitive_norm1(primitive_tokens + attn_out)
            z = self.primitive_norm2(z + self.primitive_ffn(z))
            u = z.mean(dim=1)  # [B, token_dim]

        h_unc = self.u_proj(u)
        gate = torch.sigmoid(self.gate_net(torch.cat([h, u], dim=-1)))
        if disable_fusion:
            h_final = h
        else:
            h_final = h + gate_scale * gate * h_unc

        logits = self.final_head(h_final)
        uncertainty = torch.sigmoid(self.uncertainty_head(u)).squeeze(-1)
        support = torch.sigmoid(self.support_head(u)).squeeze(-1)

        probs = F.softmax(logits, dim=-1)
        top2 = torch.topk(logits, k=2, dim=-1).values
        final_margin = top2[:, 0] - top2[:, 1]
        # Commitment is deliberately derived from support, uncertainty, and margin.
        commitment = torch.sigmoid(support * final_margin - uncertainty)

        out = {
            "h": h,
            "h_final": h_final,
            "base_logits": base_logits,
            "view_logits": view_logits,
            "logits": logits,
            "primitive_tokens": primitive_tokens,
            "primitive_attn_weights": attn_weights,
            "uncertainty_context": u,
            "uncertainty": uncertainty,
            "support": support,
            "commitment": commitment,
            "gate": gate.mean(dim=-1),
            "fusion_enabled": torch.tensor(not disable_fusion, device=h.device, dtype=torch.bool),
            **scalars,
        }

        if self.use_jepa and jepa_target_x is not None:
            z_pred = self.jepa_predictor(h)
            with torch.no_grad():
                z_tgt = self.target_encoder(jepa_target_x)
            out["jepa_pred"] = F.normalize(z_pred, dim=-1)
            out["jepa_target"] = F.normalize(z_tgt, dim=-1)

        return out
