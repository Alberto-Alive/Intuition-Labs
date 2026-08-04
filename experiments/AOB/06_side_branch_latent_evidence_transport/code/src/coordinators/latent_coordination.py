from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

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
    transport_config: Dict[str, object] | None = None


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
        flat_tokens = projected.reshape(batch, roles * tokens, projected.shape[-1])
        flat_mask = token_mask.bool().reshape(batch, roles * tokens)
        queries = self.candidate_projection(candidate_features)
        for layer in self.layers:
            queries = layer(queries, flat_tokens, flat_mask)
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
    if config.family == "latent_evidence_transport":
        from src.models.latent_evidence_transport import (
            LatentEvidenceTransportConfig,
            LatentEvidenceTransportCoordinator,
        )

        return LatentEvidenceTransportCoordinator(
            n_roles=n_roles,
            candidate_feature_dim=candidate_feature_dim,
            input_dim=config.input_dim,
            model_dim=config.model_dim,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
            transport_config=LatentEvidenceTransportConfig.from_mapping(config.transport_config),
        )
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
