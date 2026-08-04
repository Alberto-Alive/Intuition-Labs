from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LatentCoordinatorConfig:
    family: str = "cross_attention"
    input_dim: int = 64
    model_dim: int = 64
    num_heads: int = 2
    num_layers: int = 1
    ff_dim: int = 128
    dropout: float = 0.0


class CoordinatorTokenCrossAttention(nn.Module):
    """Learned coordinator token attends over role-labeled clone activations."""

    def __init__(self, n_roles: int, num_classes: int, config: LatentCoordinatorConfig) -> None:
        super().__init__()
        model_dim = _compatible_dim(config.model_dim, config.num_heads)
        self.input_projection = nn.Linear(config.input_dim, model_dim)
        self.role_embedding = nn.Embedding(n_roles, model_dim)
        self.query = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.layers = nn.ModuleList(
            [_CrossAttentionBlock(model_dim, config.num_heads, config.ff_dim, config.dropout) for _ in range(config.num_layers)]
        )
        self.head = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, num_classes))
        nn.init.normal_(self.query, mean=0.0, std=1.0 / max(1, model_dim) ** 0.5)

    def forward(self, clone_activations: torch.Tensor, role_ids: torch.Tensor) -> torch.Tensor:
        tokens = self.input_projection(clone_activations)
        tokens = tokens + self.role_embedding(role_ids.clamp(min=0, max=self.role_embedding.num_embeddings - 1))
        mask = torch.ones(role_ids.shape, dtype=torch.bool, device=role_ids.device)
        query = self.query.expand(tokens.shape[0], -1, -1)
        for layer in self.layers:
            query = layer(query, tokens, mask)
        return self.head(query[:, 0, :])


class RoleAwareSelfAttentionCoordinator(nn.Module):
    """Permutation-stable self-attention over clone slots plus role embeddings."""

    def __init__(self, n_roles: int, num_classes: int, config: LatentCoordinatorConfig) -> None:
        super().__init__()
        model_dim = _compatible_dim(config.model_dim, config.num_heads)
        self.input_projection = nn.Linear(config.input_dim, model_dim)
        self.role_embedding = nn.Embedding(n_roles, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.head = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, num_classes))
        nn.init.normal_(self.cls_token, mean=0.0, std=1.0 / max(1, model_dim) ** 0.5)

    def forward(self, clone_activations: torch.Tensor, role_ids: torch.Tensor) -> torch.Tensor:
        tokens = self.input_projection(clone_activations)
        tokens = tokens + self.role_embedding(role_ids.clamp(min=0, max=self.role_embedding.num_embeddings - 1))
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.head(encoded[:, 0, :])


class CandidateQueryCrossAttentionCoordinator(nn.Module):
    """Each candidate query attends over role-labeled clone messages and gets one score."""

    requires_candidate_features = True

    def __init__(
        self,
        n_roles: int,
        candidate_feature_dim: int,
        config: LatentCoordinatorConfig,
    ) -> None:
        super().__init__()
        model_dim = _compatible_dim(config.model_dim, config.num_heads)
        self.message_projection = nn.Linear(config.input_dim, model_dim)
        self.candidate_projection = nn.Linear(candidate_feature_dim, model_dim)
        self.role_embedding = nn.Embedding(n_roles, model_dim)
        self.layers = nn.ModuleList(
            [_CrossAttentionBlock(model_dim, config.num_heads, config.ff_dim, config.dropout) for _ in range(config.num_layers)]
        )
        self.score_head = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, 1))

    def forward(
        self,
        clone_activations: torch.Tensor,
        role_ids: torch.Tensor,
        candidate_features: torch.Tensor,
    ) -> torch.Tensor:
        messages = self.message_projection(clone_activations)
        messages = messages + self.role_embedding(role_ids.clamp(min=0, max=self.role_embedding.num_embeddings - 1))
        queries = self.candidate_projection(candidate_features)
        mask = torch.ones(role_ids.shape, dtype=torch.bool, device=role_ids.device)
        for layer in self.layers:
            queries = layer(queries, messages, mask)
        return self.score_head(queries).squeeze(-1)


class CandidateTokenCrossAttentionCoordinator(nn.Module):
    """Each candidate query attends directly over role-labeled token states."""

    requires_candidate_features = True
    requires_token_states = True

    def __init__(
        self,
        n_roles: int,
        candidate_feature_dim: int,
        config: LatentCoordinatorConfig,
    ) -> None:
        super().__init__()
        model_dim = _compatible_dim(config.model_dim, config.num_heads)
        self.token_projection = nn.Linear(config.input_dim, model_dim)
        self.candidate_projection = nn.Linear(candidate_feature_dim, model_dim)
        self.role_embedding = nn.Embedding(n_roles, model_dim)
        self.layers = nn.ModuleList(
            [_CrossAttentionBlock(model_dim, config.num_heads, config.ff_dim, config.dropout) for _ in range(config.num_layers)]
        )
        self.score_head = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, 1))
        self.control_mode = "none"
        self.last_mask_audit: dict[str, object] | None = None
        self.last_attention_stats: dict[str, object] | None = None
        self.last_representation_change: dict[str, object] | None = None

    def forward(
        self,
        clone_activations: torch.Tensor,
        role_ids: torch.Tensor,
        candidate_features: torch.Tensor,
        token_states: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        del clone_activations
        batch, roles, tokens, _dim = token_states.shape
        projected = self.token_projection(token_states)
        role = self.role_embedding(role_ids.clamp(min=0, max=self.role_embedding.num_embeddings - 1)).unsqueeze(2)
        projected = projected + role
        if self.control_mode in {"role_token_ablation", "cross_stream_attention_disabled"}:
            projected = torch.zeros_like(projected)
        flat_tokens = projected.reshape(batch, roles * tokens, projected.shape[-1])
        flat_mask = token_mask.bool().reshape(batch, roles * tokens)
        queries = self.candidate_projection(candidate_features)
        if self.control_mode == "candidate_token_ablation":
            queries = torch.zeros_like(queries)
        for layer in self.layers:
            queries = layer(queries, flat_tokens, flat_mask)
        self.last_mask_audit = {
            "family": "candidate_token_cross_attention",
            "control_mode": self.control_mode,
            "role_tokens_can_see_other_roles": False,
            "role_tokens_can_see_other_roles_initially": False,
            "role_tokens_can_see_candidates": False,
            "candidate_tokens_can_see_role_tokens": self.control_mode not in {"role_token_ablation", "cross_stream_attention_disabled"},
            "candidate_tokens_can_see_gold": False,
            "coordination_tokens_can_see_candidates": False,
            "uses_candidate_position_embedding": False,
            "uses_packed_position_embedding": False,
            "all_role_token_keys_masked": bool(self.control_mode in {"role_token_ablation", "cross_stream_attention_disabled"}),
        }
        self.last_attention_stats = None
        self.last_representation_change = None
        return self.score_head(queries).squeeze(-1)


class BilinearCandidateMessageCoordinator(nn.Module):
    """Scores each candidate from candidate features and role-labeled messages."""

    requires_candidate_features = True

    def __init__(
        self,
        n_roles: int,
        candidate_feature_dim: int,
        config: LatentCoordinatorConfig,
        contrastive: bool = False,
    ) -> None:
        super().__init__()
        model_dim = _compatible_dim(config.model_dim, config.num_heads)
        self.contrastive = bool(contrastive)
        self.message_projection = nn.Linear(config.input_dim, model_dim)
        self.candidate_projection = nn.Linear(candidate_feature_dim, model_dim)
        self.role_embedding = nn.Embedding(n_roles, model_dim)
        self.role_weight = nn.Parameter(torch.zeros(n_roles))
        self.bias_head = nn.Sequential(
            nn.LayerNorm(model_dim + candidate_feature_dim),
            nn.Linear(model_dim + candidate_feature_dim, 1),
        )
        nn.init.zeros_(self.role_weight)

    def forward(
        self,
        clone_activations: torch.Tensor,
        role_ids: torch.Tensor,
        candidate_features: torch.Tensor,
    ) -> torch.Tensor:
        messages = self.message_projection(clone_activations)
        roles = role_ids.clamp(min=0, max=self.role_embedding.num_embeddings - 1)
        messages = messages + self.role_embedding(roles)
        candidates = self.candidate_projection(candidate_features)
        if self.contrastive:
            messages = torch.nn.functional.normalize(messages, dim=-1)
            candidates = torch.nn.functional.normalize(candidates, dim=-1)
        pair_scores = torch.einsum("bcd,brd->bcr", candidates, messages) / max(1.0, float(messages.shape[-1]) ** 0.5)
        weights = torch.softmax(self.role_weight[roles], dim=-1)
        pooled_messages = torch.einsum("br,brd->bd", weights, messages)
        bias = self.bias_head(
            torch.cat(
                [pooled_messages.unsqueeze(1).expand(-1, candidate_features.shape[1], -1), candidate_features],
                dim=-1,
            )
        ).squeeze(-1)
        return (pair_scores * weights.unsqueeze(1)).sum(dim=-1) + bias


class GlobalThenCandidateCoordinator(nn.Module):
    """Global latent first attends over clone messages, then scores candidates."""

    requires_candidate_features = True

    def __init__(
        self,
        n_roles: int,
        candidate_feature_dim: int,
        config: LatentCoordinatorConfig,
        two_round: bool = False,
    ) -> None:
        super().__init__()
        model_dim = _compatible_dim(config.model_dim, config.num_heads)
        self.two_round = bool(two_round)
        self.message_projection = nn.Linear(config.input_dim, model_dim)
        self.candidate_projection = nn.Linear(candidate_feature_dim, model_dim)
        self.role_embedding = nn.Embedding(n_roles, model_dim)
        self.global_query = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.global_layers = nn.ModuleList(
            [_CrossAttentionBlock(model_dim, config.num_heads, config.ff_dim, config.dropout) for _ in range(max(1, config.num_layers))]
        )
        self.candidate_layer = _CrossAttentionBlock(model_dim, config.num_heads, config.ff_dim, config.dropout)
        self.score_head = nn.Sequential(nn.LayerNorm(model_dim * 2), nn.Linear(model_dim * 2, 1))
        nn.init.normal_(self.global_query, mean=0.0, std=1.0 / max(1, model_dim) ** 0.5)

    def forward(
        self,
        clone_activations: torch.Tensor,
        role_ids: torch.Tensor,
        candidate_features: torch.Tensor,
    ) -> torch.Tensor:
        messages = self.message_projection(clone_activations)
        messages = messages + self.role_embedding(role_ids.clamp(min=0, max=self.role_embedding.num_embeddings - 1))
        mask = torch.ones(role_ids.shape, dtype=torch.bool, device=role_ids.device)
        global_latent = self.global_query.expand(messages.shape[0], -1, -1)
        for layer in self.global_layers:
            global_latent = layer(global_latent, messages, mask)
        if self.two_round:
            messages = messages + global_latent.expand(-1, messages.shape[1], -1)
        queries = self.candidate_projection(candidate_features)
        queries = self.candidate_layer(queries, messages, mask)
        global_for_candidates = global_latent.expand(-1, queries.shape[1], -1)
        return self.score_head(torch.cat([queries, global_for_candidates], dim=-1)).squeeze(-1)


class IntegratedCrossViewCandidateCoordinator(nn.Module):
    """Semi-integrated masked multi-stream coordinator over role, router, and candidate tokens.

    The shared text encoder still reads each partial view separately. This module is the
    exploratory bridge: role streams stay separated until explicitly masked cross-view
    summary/router/candidate blocks run, and every candidate is scored from a
    candidate-conditioned latent query rather than from text output.
    """

    requires_candidate_features = True
    requires_token_states = True

    def __init__(
        self,
        n_roles: int,
        candidate_feature_dim: int,
        config: LatentCoordinatorConfig,
        mode: str,
        router_tokens: int = 0,
        use_full_tokens: bool = True,
        use_role_summaries: bool = True,
        early_role_mixing: bool = False,
        role_sees_candidates: bool = False,
        candidates_see_roles: bool = True,
        candidates_see_routers: bool = False,
    ) -> None:
        super().__init__()
        model_dim = _compatible_dim(config.model_dim, config.num_heads)
        self.mode = str(mode)
        self.n_roles = int(n_roles)
        self.router_count = max(0, int(router_tokens))
        self.use_full_tokens = bool(use_full_tokens)
        self.use_role_summaries = bool(use_role_summaries)
        self.early_role_mixing = bool(early_role_mixing)
        self.role_sees_candidates = bool(role_sees_candidates)
        self.candidates_see_roles = bool(candidates_see_roles)
        self.candidates_see_routers = bool(candidates_see_routers)
        self.control_mode = "none"
        self.record_attention = False
        self.token_projection = nn.Linear(config.input_dim, model_dim)
        self.message_projection = nn.Linear(config.input_dim, model_dim)
        self.candidate_projection = nn.Linear(candidate_feature_dim, model_dim)
        self.role_embedding = nn.Embedding(n_roles, model_dim)
        self.router_tokens = nn.Parameter(torch.zeros(1, self.router_count, model_dim)) if self.router_count else None
        self.summary_blocks = nn.ModuleList(
            [_SmallSelfAttentionBlock(model_dim, config.num_heads, config.ff_dim, config.dropout) for _ in range(1 if self.early_role_mixing or self.role_sees_candidates else 0)]
        )
        self.router_blocks = nn.ModuleList(
            [_RecordingCrossAttentionBlock(model_dim, config.num_heads, config.ff_dim, config.dropout) for _ in range(max(1, config.num_layers))]
        )
        self.candidate_blocks = nn.ModuleList(
            [_RecordingCrossAttentionBlock(model_dim, config.num_heads, config.ff_dim, config.dropout) for _ in range(max(1, config.num_layers))]
        )
        self.candidate_self_blocks = nn.ModuleList(
            [_SmallSelfAttentionBlock(model_dim, config.num_heads, config.ff_dim, config.dropout) for _ in range(1 if "guided" in self.mode else 0)]
        )
        self.score_head = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, 1))
        if self.router_tokens is not None:
            nn.init.normal_(self.router_tokens, mean=0.0, std=1.0 / max(1, model_dim) ** 0.5)
        self.last_mask_audit: dict[str, object] | None = None
        self.last_attention_stats: dict[str, object] | None = None
        self.last_representation_change: dict[str, object] | None = None

    def forward(
        self,
        clone_activations: torch.Tensor,
        role_ids: torch.Tensor,
        candidate_features: torch.Tensor,
        token_states: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, roles, tokens, _dim = token_states.shape
        projected_tokens = self.token_projection(token_states)
        projected_messages = self.message_projection(clone_activations)
        role_ids = role_ids.clamp(min=0, max=self.role_embedding.num_embeddings - 1)
        role = self.role_embedding(role_ids)
        projected_tokens = projected_tokens + role.unsqueeze(2)
        projected_messages = projected_messages + role
        mask = token_mask.bool()
        if self.control_mode in {"role_token_ablation", "cross_stream_attention_disabled"}:
            projected_tokens = torch.zeros_like(projected_tokens)
            projected_messages = torch.zeros_like(projected_messages)
        summaries = self._role_summaries(projected_tokens, projected_messages, mask)
        summary_before = summaries
        early_mixing_active = self.early_role_mixing and self.control_mode not in {"early_mixing_disabled", "late_mixing_only_control"}
        if early_mixing_active:
            for block in self.summary_blocks:
                summaries = block(summaries)
        candidates = self.candidate_projection(candidate_features)
        if self.control_mode == "candidate_token_ablation":
            candidates = torch.zeros_like(candidates)
        if self.role_sees_candidates and self.control_mode not in {"early_mixing_disabled", "late_mixing_only_control", "candidate_token_ablation"}:
            candidate_context = candidates.mean(dim=1, keepdim=True).expand(-1, summaries.shape[1], -1)
            summaries = summaries + candidate_context
            for block in self.summary_blocks:
                summaries = block(summaries)
        routers = self._router_values(batch, candidates.device, candidates.dtype)
        memory, memory_mask, memory_roles = self._memory(projected_tokens, summaries, mask)
        router_before = routers
        if routers is not None and self.control_mode != "coordination_token_ablation":
            for block in self.router_blocks:
                routers = block(routers, memory, memory_mask, record=self.record_attention)
            if self.candidates_see_routers:
                router_roles = torch.full((batch, routers.shape[1]), -1, dtype=torch.long, device=role_ids.device)
                memory = torch.cat([memory, routers], dim=1)
                memory_mask = torch.cat([memory_mask, torch.ones(batch, routers.shape[1], dtype=torch.bool, device=memory_mask.device)], dim=1)
                memory_roles = torch.cat([memory_roles, router_roles], dim=1)
        if self.control_mode == "coordination_token_ablation" and self.candidates_see_routers and not self.candidates_see_roles:
            memory = torch.zeros(batch, 1, candidates.shape[-1], dtype=candidates.dtype, device=candidates.device)
            memory_mask = torch.ones(batch, 1, dtype=torch.bool, device=candidates.device)
            memory_roles = torch.full((batch, 1), -1, dtype=torch.long, device=candidates.device)
        if self.control_mode == "cross_stream_attention_disabled" or not self.candidates_see_roles:
            if not (self.candidates_see_routers and routers is not None and self.control_mode != "coordination_token_ablation"):
                memory = torch.zeros(batch, 1, candidates.shape[-1], dtype=candidates.dtype, device=candidates.device)
                memory_mask = torch.ones(batch, 1, dtype=torch.bool, device=candidates.device)
                memory_roles = torch.full((batch, 1), -2, dtype=torch.long, device=candidates.device)
        for block in self.candidate_self_blocks:
            candidates = block(candidates)
        candidate_before = candidates
        for block in self.candidate_blocks:
            candidates = block(candidates, memory, memory_mask, record=self.record_attention)
        self.last_mask_audit = self._mask_audit(memory_mask, memory_roles, roles, tokens)
        self.last_attention_stats = self._attention_stats(memory_roles)
        self.last_representation_change = {
            "summary_l2_change_mean": _tensor_change(summary_before, summaries),
            "router_l2_change_mean": _tensor_change(router_before, routers),
            "candidate_l2_change_mean": _tensor_change(candidate_before, candidates),
        }
        return self.score_head(candidates).squeeze(-1)

    def _role_summaries(self, tokens: torch.Tensor, messages: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        counts = mask.sum(dim=2).clamp(min=1).unsqueeze(-1).to(dtype=tokens.dtype)
        pooled = (tokens * mask.unsqueeze(-1).to(dtype=tokens.dtype)).sum(dim=2) / counts
        return 0.5 * (pooled + messages)

    def _router_values(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self.router_tokens is None or self.router_count <= 0:
            return None
        routers = self.router_tokens.to(device=device, dtype=dtype).expand(batch, -1, -1)
        if self.control_mode == "coordination_token_ablation":
            return torch.zeros_like(routers)
        return routers

    def _memory(
        self,
        tokens: torch.Tensor,
        summaries: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, roles, token_count, dim = tokens.shape
        parts = []
        masks = []
        role_parts = []
        device = tokens.device
        if self.use_full_tokens:
            parts.append(tokens.reshape(batch, roles * token_count, dim))
            masks.append(mask.reshape(batch, roles * token_count))
            role_index = torch.arange(roles, dtype=torch.long, device=device).view(1, roles, 1).expand(batch, roles, token_count)
            role_parts.append(role_index.reshape(batch, roles * token_count))
        if self.use_role_summaries:
            parts.append(summaries)
            masks.append(torch.ones(batch, roles, dtype=torch.bool, device=device))
            role_parts.append(torch.arange(roles, dtype=torch.long, device=device).view(1, roles).expand(batch, roles))
        if not parts:
            return (
                torch.zeros(batch, 1, dim, dtype=tokens.dtype, device=device),
                torch.ones(batch, 1, dtype=torch.bool, device=device),
                torch.full((batch, 1), -2, dtype=torch.long, device=device),
            )
        return torch.cat(parts, dim=1), torch.cat(masks, dim=1), torch.cat(role_parts, dim=1)

    def _mask_audit(self, memory_mask: torch.Tensor, memory_roles: torch.Tensor, roles: int, token_count: int) -> dict[str, object]:
        valid = memory_mask.bool()
        role_key_counts = {
            str(role): int(((memory_roles == role) & valid).sum().detach().cpu().item()) for role in range(roles)
        }
        return {
            "family": self.mode,
            "control_mode": self.control_mode,
            "n_roles": int(roles),
            "tokens_per_role": int(token_count),
            "router_tokens": int(self.router_count),
            "use_full_tokens": bool(self.use_full_tokens),
            "use_role_summaries": bool(self.use_role_summaries),
            "early_role_mixing_configured": bool(self.early_role_mixing),
            "early_role_mixing_active": bool(self.early_role_mixing and self.control_mode not in {"early_mixing_disabled", "late_mixing_only_control"}),
            "role_tokens_can_see_other_roles_initially": False,
            "role_tokens_can_see_candidates": bool(self.role_sees_candidates and self.control_mode not in {"candidate_token_ablation", "early_mixing_disabled", "late_mixing_only_control"}),
            "candidate_tokens_can_see_role_tokens": bool(self.candidates_see_roles and self.control_mode not in {"role_token_ablation", "cross_stream_attention_disabled"}),
            "candidate_tokens_can_see_router_tokens": bool(self.candidates_see_routers and self.router_count > 0 and self.control_mode != "coordination_token_ablation"),
            "candidate_tokens_can_see_gold": False,
            "coordination_tokens_can_see_role_tokens": bool(self.router_count > 0 and self.control_mode != "coordination_token_ablation"),
            "coordination_tokens_can_see_candidates": False,
            "uses_candidate_position_embedding": False,
            "uses_packed_position_embedding": False,
            "role_key_counts": role_key_counts,
            "all_role_token_keys_masked": not bool(self.candidates_see_roles and self.control_mode not in {"role_token_ablation", "cross_stream_attention_disabled"}),
            "mask_finite": bool(torch.isfinite(memory_mask.float()).all().detach().cpu().item()),
        }

    def _attention_stats(self, memory_roles: torch.Tensor) -> dict[str, object] | None:
        candidate_blocks = [block for block in self.candidate_blocks if block.last_weights is not None]
        if not candidate_blocks:
            return None
        weights = candidate_blocks[-1].last_weights
        if weights is None:
            return None
        role_mass: dict[str, float] = {}
        for role in range(self.n_roles):
            role_mask = memory_roles.eq(role).unsqueeze(1).unsqueeze(1).to(dtype=weights.dtype)
            denom = role_mask.sum(dim=-1).clamp(min=1.0)
            role_mass[str(role)] = float(((weights * role_mask).sum(dim=-1) / denom).mean().detach().cpu().item())
        entropy = -(weights.clamp_min(1e-9) * weights.clamp_min(1e-9).log()).sum(dim=-1).mean()
        return {
            "candidate_attention_entropy": float(entropy.detach().cpu().item()),
            "candidate_attention_mean_mass_by_role": role_mass,
            "candidate_attention_max": float(weights.max().detach().cpu().item()),
            "candidate_attention_min": float(weights.min().detach().cpu().item()),
        }


class _RecordingCrossAttentionBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
        )
        self.last_weights: torch.Tensor | None = None

    def forward(self, query: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor, record: bool = False) -> torch.Tensor:
        attended, weights = self.attention(
            query=self.norm1(query),
            key=tokens,
            value=tokens,
            key_padding_mask=~mask.bool(),
            need_weights=record,
            average_attn_weights=False,
        )
        self.last_weights = weights.detach() if record and weights is not None else None
        query = query + attended
        query = query + self.feed_forward(self.norm2(query))
        return query


class _SmallSelfAttentionBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attended, _weights = self.attention(
            query=self.norm1(tokens),
            key=self.norm1(tokens),
            value=tokens,
            need_weights=False,
        )
        tokens = tokens + attended
        tokens = tokens + self.feed_forward(self.norm2(tokens))
        return tokens


def make_latent_coordinator(
    n_roles: int,
    num_classes: int,
    config: LatentCoordinatorConfig,
) -> nn.Module:
    if config.family == "cross_attention":
        return CoordinatorTokenCrossAttention(n_roles=n_roles, num_classes=num_classes, config=config)
    if config.family == "self_attention":
        return RoleAwareSelfAttentionCoordinator(n_roles=n_roles, num_classes=num_classes, config=config)
    raise ValueError(f"unknown latent coordinator family: {config.family}")


def make_candidate_query_coordinator(
    n_roles: int,
    candidate_feature_dim: int,
    config: LatentCoordinatorConfig,
) -> nn.Module:
    if config.family == "candidate_token_cross_attention":
        return CandidateTokenCrossAttentionCoordinator(
            n_roles=n_roles,
            candidate_feature_dim=candidate_feature_dim,
            config=config,
        )
    if config.family == "bilinear_candidate":
        return BilinearCandidateMessageCoordinator(
            n_roles=n_roles,
            candidate_feature_dim=candidate_feature_dim,
            config=config,
            contrastive=False,
        )
    if config.family == "contrastive_candidate":
        return BilinearCandidateMessageCoordinator(
            n_roles=n_roles,
            candidate_feature_dim=candidate_feature_dim,
            config=config,
            contrastive=True,
        )
    if config.family == "global_then_candidate":
        return GlobalThenCandidateCoordinator(
            n_roles=n_roles,
            candidate_feature_dim=candidate_feature_dim,
            config=config,
            two_round=False,
        )
    if config.family == "two_round_message_passing":
        return GlobalThenCandidateCoordinator(
            n_roles=n_roles,
            candidate_feature_dim=candidate_feature_dim,
            config=config,
            two_round=True,
        )
    integrated = _integrated_family_options(config.family)
    if integrated is not None:
        return IntegratedCrossViewCandidateCoordinator(
            n_roles=n_roles,
            candidate_feature_dim=candidate_feature_dim,
            config=config,
            **integrated,
        )
    return CandidateQueryCrossAttentionCoordinator(
        n_roles=n_roles,
        candidate_feature_dim=candidate_feature_dim,
        config=config,
    )


def _integrated_family_options(family: str) -> dict[str, object] | None:
    if family == "integrated_bridge_late":
        return {
            "mode": family,
            "router_tokens": 0,
            "use_full_tokens": True,
            "use_role_summaries": True,
            "early_role_mixing": False,
            "role_sees_candidates": False,
            "candidates_see_roles": True,
            "candidates_see_routers": False,
        }
    if family == "integrated_summary_late":
        return {
            "mode": family,
            "router_tokens": 0,
            "use_full_tokens": False,
            "use_role_summaries": True,
            "early_role_mixing": False,
            "role_sees_candidates": False,
            "candidates_see_roles": True,
            "candidates_see_routers": False,
        }
    if family == "integrated_router_late":
        return {
            "mode": family,
            "router_tokens": 2,
            "use_full_tokens": True,
            "use_role_summaries": True,
            "early_role_mixing": False,
            "role_sees_candidates": False,
            "candidates_see_roles": False,
            "candidates_see_routers": True,
        }
    if family == "integrated_router_plus_tokens":
        return {
            "mode": family,
            "router_tokens": 2,
            "use_full_tokens": True,
            "use_role_summaries": True,
            "early_role_mixing": False,
            "role_sees_candidates": False,
            "candidates_see_roles": True,
            "candidates_see_routers": True,
        }
    if family == "integrated_candidate_guided":
        return {
            "mode": family,
            "router_tokens": 0,
            "use_full_tokens": True,
            "use_role_summaries": True,
            "early_role_mixing": False,
            "role_sees_candidates": True,
            "candidates_see_roles": True,
            "candidates_see_routers": False,
        }
    if family == "integrated_candidate_guided_late":
        return {
            "mode": family,
            "router_tokens": 0,
            "use_full_tokens": True,
            "use_role_summaries": True,
            "early_role_mixing": False,
            "role_sees_candidates": False,
            "candidates_see_roles": True,
            "candidates_see_routers": False,
        }
    if family == "integrated_early_mixing_isolated":
        return {
            "mode": family,
            "router_tokens": 0,
            "use_full_tokens": True,
            "use_role_summaries": True,
            "early_role_mixing": True,
            "role_sees_candidates": False,
            "candidates_see_roles": True,
            "candidates_see_routers": False,
        }
    if family == "integrated_early_mixing":
        return {
            "mode": family,
            "router_tokens": 0,
            "use_full_tokens": True,
            "use_role_summaries": True,
            "early_role_mixing": True,
            "role_sees_candidates": False,
            "candidates_see_roles": True,
            "candidates_see_routers": False,
        }
    if family == "integrated_late_mixing_only":
        return {
            "mode": family,
            "router_tokens": 0,
            "use_full_tokens": True,
            "use_role_summaries": True,
            "early_role_mixing": False,
            "role_sees_candidates": False,
            "candidates_see_roles": True,
            "candidates_see_routers": False,
        }
    return None


class _CrossAttentionBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
        )

    def forward(self, query: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attended, _weights = self.attention(
            query=self.norm1(query),
            key=tokens,
            value=tokens,
            key_padding_mask=~mask.bool(),
            need_weights=False,
        )
        query = query + attended
        query = query + self.feed_forward(self.norm2(query))
        return query


def _compatible_dim(model_dim: int, num_heads: int) -> int:
    model_dim = max(1, int(model_dim))
    num_heads = max(1, int(num_heads))
    remainder = model_dim % num_heads
    if remainder:
        model_dim += num_heads - remainder
    return model_dim


def _tensor_change(before: torch.Tensor | None, after: torch.Tensor | None) -> float:
    if before is None or after is None or before.shape != after.shape or before.numel() == 0:
        return 0.0
    value = (after.detach().float() - before.detach().float()).norm(dim=-1).mean()
    return float(value.cpu().item())
