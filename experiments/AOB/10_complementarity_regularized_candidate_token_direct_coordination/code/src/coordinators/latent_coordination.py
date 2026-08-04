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
    num_avenues: int = 1
    avenue_topk: int = 0


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
        self.model_dim = model_dim
        self.token_projection = nn.Linear(config.input_dim, model_dim)
        self.candidate_projection = nn.Linear(candidate_feature_dim, model_dim)
        self.role_embedding = nn.Embedding(n_roles, model_dim)
        self.avenue_embedding = nn.Embedding(max(1, int(config.num_avenues)), model_dim)
        self.avenue_topk = int(config.avenue_topk)
        self.layers = nn.ModuleList(
            [_CrossAttentionBlock(model_dim, config.num_heads, config.ff_dim, config.dropout) for _ in range(config.num_layers)]
        )
        self.score_head = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, 1))
        self.last_avenue_attention: torch.Tensor | None = None

    def forward(
        self,
        clone_activations: torch.Tensor,
        role_ids: torch.Tensor,
        candidate_features: torch.Tensor,
        token_states: torch.Tensor,
        token_mask: torch.Tensor,
        avenue_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del clone_activations
        self.last_avenue_attention = None
        if token_states.dim() == 5:
            return self._forward_multi_avenue(role_ids, candidate_features, token_states, token_mask, avenue_ids)
        batch, roles, tokens, _dim = token_states.shape
        projected = self.token_projection(token_states)
        role = self.role_embedding(role_ids.clamp(min=0, max=self.role_embedding.num_embeddings - 1)).unsqueeze(2)
        projected = projected + role
        flat_tokens = projected.reshape(batch, roles * tokens, projected.shape[-1])
        flat_mask = token_mask.bool().reshape(batch, roles * tokens)
        queries = self.candidate_projection(candidate_features)
        for layer in self.layers:
            queries = layer(queries, flat_tokens, flat_mask)
        return self.score_head(queries).squeeze(-1)

    def _forward_multi_avenue(
        self,
        role_ids: torch.Tensor,
        candidate_features: torch.Tensor,
        token_states: torch.Tensor,
        token_mask: torch.Tensor,
        avenue_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, roles, avenues, tokens, _dim = token_states.shape
        projected = self.token_projection(token_states)
        role = self.role_embedding(role_ids.clamp(min=0, max=self.role_embedding.num_embeddings - 1)).view(batch, roles, 1, 1, -1)
        if avenue_ids is None:
            ids = torch.arange(avenues, dtype=torch.long, device=token_states.device).view(1, 1, avenues)
            avenue_ids = ids.expand(batch, roles, avenues)
        avenue = self.avenue_embedding(
            avenue_ids.clamp(min=0, max=self.avenue_embedding.num_embeddings - 1)
        ).unsqueeze(3)
        projected = projected + role + avenue
        flat_tokens = projected.reshape(batch, roles * avenues * tokens, projected.shape[-1])
        flat_mask = token_mask.bool().reshape(batch, roles * avenues * tokens)
        if 0 < self.avenue_topk < roles * avenues:
            flat_tokens, flat_mask = self._candidate_agnostic_topk_tokens(projected, token_mask.bool())
        queries = self.candidate_projection(candidate_features)
        for layer in self.layers:
            queries = layer(queries, flat_tokens, flat_mask)
        self._record_last_avenue_attention(batch, roles, avenues, tokens, token_mask.bool())
        return self.score_head(queries).squeeze(-1)

    def _record_last_avenue_attention(
        self,
        batch: int,
        roles: int,
        avenues: int,
        tokens: int,
        token_mask: torch.Tensor,
    ) -> None:
        if 0 < self.avenue_topk < roles * avenues or not self.layers:
            self.last_avenue_attention = None
            return
        weights = getattr(self.layers[-1], "last_attention_weights", None)
        if weights is None:
            self.last_avenue_attention = None
            return
        shaped = weights.reshape(batch, weights.shape[1], weights.shape[2], roles, avenues, tokens)
        mask = token_mask.bool().reshape(batch, 1, 1, roles, avenues, tokens)
        per_avenue = (shaped * mask.to(dtype=shaped.dtype)).sum(dim=(1, 2, 3, 5))
        denom = per_avenue.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        self.last_avenue_attention = per_avenue / denom

    def _candidate_agnostic_topk_tokens(self, projected: torch.Tensor, token_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, roles, avenues, tokens, dim = projected.shape
        valid = token_mask.to(dtype=projected.dtype)
        denom = valid.sum(dim=3).clamp(min=1.0).unsqueeze(-1)
        summaries = (projected * valid.unsqueeze(-1)).sum(dim=3) / denom
        scores = summaries.norm(dim=-1).reshape(batch, roles * avenues)
        topk = min(max(1, self.avenue_topk), roles * avenues)
        selected = scores.topk(topk, dim=1).indices
        gather_tokens = projected.reshape(batch, roles * avenues, tokens, dim)
        gather_mask = token_mask.reshape(batch, roles * avenues, tokens)
        token_index = selected.view(batch, topk, 1, 1).expand(-1, -1, tokens, dim)
        mask_index = selected.view(batch, topk, 1).expand(-1, -1, tokens)
        return gather_tokens.gather(1, token_index).reshape(batch, topk * tokens, dim), gather_mask.gather(1, mask_index).reshape(batch, topk * tokens)


class GatedCandidateTokenCrossAttentionCoordinator(CandidateTokenCrossAttentionCoordinator):
    """Candidate-token attention with learned per-avenue gates before flattening."""

    def __init__(
        self,
        n_roles: int,
        candidate_feature_dim: int,
        config: LatentCoordinatorConfig,
    ) -> None:
        super().__init__(n_roles=n_roles, candidate_feature_dim=candidate_feature_dim, config=config)
        self.gate_head = nn.Sequential(nn.LayerNorm(self.model_dim), nn.Linear(self.model_dim, 1))

    def _forward_multi_avenue(
        self,
        role_ids: torch.Tensor,
        candidate_features: torch.Tensor,
        token_states: torch.Tensor,
        token_mask: torch.Tensor,
        avenue_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, roles, avenues, tokens, _dim = token_states.shape
        projected = self.token_projection(token_states)
        role = self.role_embedding(role_ids.clamp(min=0, max=self.role_embedding.num_embeddings - 1)).view(batch, roles, 1, 1, -1)
        if avenue_ids is None:
            ids = torch.arange(avenues, dtype=torch.long, device=token_states.device).view(1, 1, avenues)
            avenue_ids = ids.expand(batch, roles, avenues)
        avenue = self.avenue_embedding(
            avenue_ids.clamp(min=0, max=self.avenue_embedding.num_embeddings - 1)
        ).unsqueeze(3)
        projected = projected + role + avenue
        valid = token_mask.bool().to(dtype=projected.dtype)
        summaries = (projected * valid.unsqueeze(-1)).sum(dim=3) / valid.sum(dim=3).clamp(min=1.0).unsqueeze(-1)
        gate = torch.softmax(self.gate_head(summaries).squeeze(-1), dim=2)
        self.last_avenue_attention = gate.mean(dim=1)
        projected = projected * gate.unsqueeze(-1).unsqueeze(-1)
        flat_tokens = projected.reshape(batch, roles * avenues * tokens, projected.shape[-1])
        flat_mask = token_mask.bool().reshape(batch, roles * avenues * tokens)
        queries = self.candidate_projection(candidate_features)
        for layer in self.layers:
            queries = layer(queries, flat_tokens, flat_mask)
        return self.score_head(queries).squeeze(-1)


class ProductOfExpertsCandidateTokenCoordinator(CandidateTokenCrossAttentionCoordinator):
    """Scores each avenue independently and combines candidate log-probabilities."""

    def _forward_multi_avenue(
        self,
        role_ids: torch.Tensor,
        candidate_features: torch.Tensor,
        token_states: torch.Tensor,
        token_mask: torch.Tensor,
        avenue_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, roles, avenues, tokens, _dim = token_states.shape
        role = self.role_embedding(role_ids.clamp(min=0, max=self.role_embedding.num_embeddings - 1)).view(batch, roles, 1, -1)
        queries_base = self.candidate_projection(candidate_features)
        per_avenue_logits = []
        for avenue_index in range(avenues):
            projected = self.token_projection(token_states[:, :, avenue_index, :, :])
            if avenue_ids is None:
                avenue_value = torch.full((batch, roles), avenue_index, dtype=torch.long, device=token_states.device)
            else:
                avenue_value = avenue_ids[:, :, avenue_index]
            avenue = self.avenue_embedding(
                avenue_value.clamp(min=0, max=self.avenue_embedding.num_embeddings - 1)
            ).unsqueeze(2)
            projected = projected + role + avenue
            flat_tokens = projected.reshape(batch, roles * tokens, projected.shape[-1])
            flat_mask = token_mask[:, :, avenue_index, :].bool().reshape(batch, roles * tokens)
            valid_avenue = flat_mask.any(dim=1)
            if not bool(valid_avenue.all()):
                flat_tokens = flat_tokens.clone()
                flat_mask = flat_mask.clone()
                flat_tokens[~valid_avenue, 0] = 0.0
                flat_mask[~valid_avenue, 0] = True
            queries = queries_base
            for layer in self.layers:
                queries = layer(queries, flat_tokens, flat_mask)
            logits = self.score_head(queries).squeeze(-1)
            per_avenue_logits.append(torch.where(valid_avenue.unsqueeze(-1), logits, torch.zeros_like(logits)))
        stacked = torch.stack(per_avenue_logits, dim=1)
        confidences = torch.softmax(stacked, dim=-1).amax(dim=-1)
        self.last_avenue_attention = torch.softmax(confidences, dim=-1)
        return torch.log_softmax(stacked, dim=-1).sum(dim=1)


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
    if config.family == "candidate_token_gated_attention":
        return GatedCandidateTokenCrossAttentionCoordinator(
            n_roles=n_roles,
            candidate_feature_dim=candidate_feature_dim,
            config=config,
        )
    if config.family == "candidate_token_product_of_experts":
        return ProductOfExpertsCandidateTokenCoordinator(
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
    return CandidateQueryCrossAttentionCoordinator(
        n_roles=n_roles,
        candidate_feature_dim=candidate_feature_dim,
        config=config,
    )


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
        self.last_attention_weights: torch.Tensor | None = None

    def forward(self, query: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attended, weights = self.attention(
            query=self.norm1(query),
            key=tokens,
            value=tokens,
            key_padding_mask=~mask.bool(),
            need_weights=True,
            average_attn_weights=False,
        )
        self.last_attention_weights = weights
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
