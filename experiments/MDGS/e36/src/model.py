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
    evidence_mode: bool = False
    order_temp: float = 6.0
    evidence_dim: int = 0


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
        evidence_mode: bool | None = None,
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
        self.evidence_mode = cfg.evidence_mode
        self.order_temp = cfg.order_temp
        self.evidence_dim = max(0, cfg.evidence_dim)

        self.encoder = MLP([cfg.input_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.latent_dim], cfg.dropout)
        self.base_head = nn.Linear(cfg.latent_dim, cfg.num_classes)
        self.evidence_head = MLP([cfg.latent_dim, cfg.hidden_dim, 3 + self.evidence_dim], cfg.dropout)

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
        # Lightweight scalar heads let the perturbation targets supervise the
        # primitive scores directly, while the raw geometry diagnostics remain
        # available for inspection.
        self.m_scalar_head = MLP([cfg.token_dim, cfg.hidden_dim, 1], cfg.dropout)
        self.d_scalar_head = MLP([cfg.token_dim, cfg.hidden_dim, 1], cfg.dropout)
        self.g_scalar_head = MLP([cfg.token_dim, cfg.hidden_dim, 1], cfg.dropout)

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

    def _evidence_state(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        evidence_raw = self.evidence_head(h)
        order_logit = evidence_raw[:, 0:1]
        boundary_logit = evidence_raw[:, 1:2]
        support_logit = evidence_raw[:, 2:3]
        evidence_feat = evidence_raw[:, 3:] if self.evidence_dim > 0 else None

        order = torch.tanh(order_logit)
        boundary = torch.sigmoid(boundary_logit)
        support = torch.sigmoid(support_logit)

        # Shared evidence mass: support on-manifold, boundary ambiguity, and class order.
        support_mass = support * (1.0 - boundary)
        success_side = torch.sigmoid(self.order_temp * order)
        failure_side = torch.sigmoid(-self.order_temp * order)
        success_evidence = support_mass * success_side
        failure_evidence = support_mass * failure_side
        uncertain_evidence = 1.0 - support_mass

        logits_evidence = torch.log(torch.cat([failure_evidence, success_evidence], dim=-1) + 1e-6)
        probs = F.softmax(logits_evidence, dim=-1)
        pred_conf = probs.max(dim=-1).values

        order_scalar = order.abs().squeeze(-1)
        boundary_scalar = boundary.squeeze(-1)
        support_scalar = support.squeeze(-1)
        uncertainty_scalar = uncertain_evidence.squeeze(-1)
        evidence_conf = support_scalar * (1.0 - boundary_scalar)

        out: Dict[str, torch.Tensor] = {
            "order_logit": order_logit.squeeze(-1),
            "boundary_logit": boundary_logit.squeeze(-1),
            "support_logit": support_logit.squeeze(-1),
            "order": order.squeeze(-1),
            "boundary": boundary_scalar,
            "support": support_scalar,
            "uncertainty": uncertainty_scalar,
            "commitment": evidence_conf * order_scalar,
            "m_scalar": order_scalar,
            "d_scalar": boundary_scalar,
            "g_scalar": boundary_scalar,
            "s_scalar": support_scalar,
            "evidence_logits": logits_evidence,
            "logits_evidence": logits_evidence,
            "logits": logits_evidence,
            "base_logits": logits_evidence,
            "gate": evidence_conf,
            "fusion_enabled": torch.tensor(False, device=h.device, dtype=torch.bool),
            "uncertainty_context": evidence_feat if evidence_feat is not None else evidence_raw,
            "pred_conf": pred_conf,
            "evidence_conf": evidence_conf,
            "primitive_tokens": None,
            "primitive_attn_weights": None,
        }
        if evidence_feat is not None:
            out["evidence_feat"] = evidence_feat
        return out

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
        m_scalar = torch.sigmoid(self.m_scalar_head(pM)).squeeze(-1)
        d_scalar = torch.sigmoid(self.d_scalar_head(pD)).squeeze(-1)
        g_scalar = torch.sigmoid(self.g_scalar_head(pG)).squeeze(-1)

        scalars = {
            "M_margin": margin,
            "D_disagreement": prob_var,
            "G_spread": spread,
            "S_proto": support_proto,
            "m_scalar": m_scalar,
            "d_scalar": d_scalar,
            "g_scalar": g_scalar,
            "base_confidence": confidence,
            "base_entropy": entropy,
            "view_kl": kl,
            "avg_cos_dist": avg_cos_dist,
        }
        return tokens, scalars

    def forward_from_h(
        self,
        h: torch.Tensor,
        gate_scale: float = 1.0,
        disable_fusion: bool = False,
        disable_primitive_attn: bool = False,
        support_only: bool = False,
        evidence_mode: bool = False,
    ) -> Dict[str, torch.Tensor | None]:
        if evidence_mode:
            out: Dict[str, torch.Tensor | None] = {
                "h": h,
                "h_final": h,
                "view_logits": None,
            }
            out.update(self._evidence_state(h))
            return out

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
        out["s_scalar"] = support
        return out

    def forward(
        self,
        x: torch.Tensor,
        jepa_target_x: torch.Tensor | None = None,
        gate_scale: float = 1.0,
        disable_fusion: bool = False,
        disable_primitive_attn: bool = False,
        support_only: bool = False,
        evidence_mode: bool | None = None,
    ) -> Dict[str, torch.Tensor | None]:
        h = self.encoder(x)
        use_evidence_mode = self.evidence_mode if evidence_mode is None else evidence_mode
        out = self.forward_from_h(
            h,
            gate_scale=gate_scale,
            disable_fusion=disable_fusion,
            disable_primitive_attn=disable_primitive_attn,
            support_only=support_only,
            evidence_mode=use_evidence_mode,
        )

        if self.use_jepa and jepa_target_x is not None:
            z_pred = self.jepa_predictor(h)
            with torch.no_grad():
                z_tgt = self.target_encoder(jepa_target_x)
            out["jepa_pred"] = F.normalize(z_pred, dim=-1)
            out["jepa_target"] = F.normalize(z_tgt, dim=-1)

        return out
