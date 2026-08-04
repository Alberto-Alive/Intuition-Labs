from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LatentEvidenceTransportConfig:
    family: str = "candidate_only_refinement"
    refinement_steps: int = 2
    shared_update: bool = True
    gated_residual: bool = True
    norm_position: str = "pre"
    candidate_reads_evidence: bool = True
    update_evidence: bool = False
    stop_gradient_evidence_update: bool = False
    share_candidate_evidence_weights: bool = False
    workspace_slots: int = 0
    workspace_init: str = "none"
    evidence_writes_workspace: bool = True
    candidate_reads_workspace: bool = True
    candidate_writes_workspace: bool = False
    candidate_conditioned_workspace_read: bool = True
    aux_loss: str = "none"
    aux_weight: float = 0.0
    aux_margin: float = 0.05
    energy_source: str = "candidate_state"
    transport_disabled: bool = False
    freeze_transport: bool = False
    random_transport: bool = False

    @classmethod
    def from_mapping(cls, values: Mapping[str, object] | None) -> "LatentEvidenceTransportConfig":
        if values is None:
            return cls()
        allowed = {field.name for field in fields(cls)}
        clean = {key: value for key, value in dict(values).items() if key in allowed}
        return cls(**clean)


class LatentEvidenceTransportCoordinator(nn.Module):
    """Candidate-token-direct scoring with iterative latent evidence transport."""

    requires_candidate_features = True
    requires_token_states = True

    def __init__(
        self,
        n_roles: int,
        candidate_feature_dim: int,
        input_dim: int,
        model_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        transport_config: LatentEvidenceTransportConfig | Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        self.transport_config = (
            transport_config
            if isinstance(transport_config, LatentEvidenceTransportConfig)
            else LatentEvidenceTransportConfig.from_mapping(transport_config)
        )
        model_dim = _compatible_dim(model_dim, num_heads)
        heads = _compatible_heads(model_dim, num_heads)
        steps = max(0, int(self.transport_config.refinement_steps))
        self.model_dim = int(model_dim)
        self.refinement_steps = steps
        self.token_projection = nn.Linear(int(input_dim), model_dim)
        self.candidate_projection = nn.Linear(int(candidate_feature_dim), model_dim)
        self.role_embedding = nn.Embedding(int(n_roles), model_dim)
        self.initial_read = _AttentionUpdateBlock(
            model_dim,
            heads,
            int(ff_dim),
            float(dropout),
            gated_residual=False,
            norm_position="pre",
        )
        block_count = 1 if bool(self.transport_config.shared_update) else max(1, steps)
        self.candidate_blocks = nn.ModuleList(
            [
                _AttentionUpdateBlock(
                    model_dim,
                    heads,
                    int(ff_dim),
                    float(dropout),
                    gated_residual=bool(self.transport_config.gated_residual),
                    norm_position=str(self.transport_config.norm_position),
                )
                for _ in range(block_count)
            ]
        )
        evidence_block_count = (
            1
            if bool(self.transport_config.share_candidate_evidence_weights)
            else (1 if bool(self.transport_config.shared_update) else max(1, steps))
        )
        self.evidence_blocks = nn.ModuleList(
            [
                _AttentionUpdateBlock(
                    model_dim,
                    heads,
                    int(ff_dim),
                    float(dropout),
                    gated_residual=bool(self.transport_config.gated_residual),
                    norm_position=str(self.transport_config.norm_position),
                )
                for _ in range(evidence_block_count)
            ]
        )
        workspace_slots = max(0, int(self.transport_config.workspace_slots))
        self.workspace_slots = workspace_slots
        if workspace_slots:
            self.workspace_seed = nn.Parameter(torch.zeros(1, workspace_slots, model_dim))
            nn.init.normal_(self.workspace_seed, mean=0.0, std=1.0 / max(1, model_dim) ** 0.5)
            self.workspace_from_evidence = _AttentionUpdateBlock(
                model_dim,
                heads,
                int(ff_dim),
                float(dropout),
                gated_residual=bool(self.transport_config.gated_residual),
                norm_position=str(self.transport_config.norm_position),
            )
            self.workspace_from_candidates = _AttentionUpdateBlock(
                model_dim,
                heads,
                int(ff_dim),
                float(dropout),
                gated_residual=bool(self.transport_config.gated_residual),
                norm_position=str(self.transport_config.norm_position),
            )
            self.candidate_from_workspace = _AttentionUpdateBlock(
                model_dim,
                heads,
                int(ff_dim),
                float(dropout),
                gated_residual=bool(self.transport_config.gated_residual),
                norm_position=str(self.transport_config.norm_position),
            )
            self.workspace_pool_gate = _GatedResidual(model_dim)
        else:
            self.register_parameter("workspace_seed", None)
            self.workspace_from_evidence = None
            self.workspace_from_candidates = None
            self.candidate_from_workspace = None
            self.workspace_pool_gate = None
        self.compatibility_scale = nn.Parameter(torch.zeros(()))
        self.score_head = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, 1))
        self._last_trajectory_logits: List[torch.Tensor] = []
        self._last_compatibility_logits: List[torch.Tensor] = []
        self._last_auxiliary_loss = torch.zeros(())
        if bool(self.transport_config.freeze_transport) or bool(self.transport_config.random_transport):
            self._freeze_transport_modules()

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
        evidence = self.token_projection(token_states)
        role = self.role_embedding(role_ids.clamp(min=0, max=self.role_embedding.num_embeddings - 1)).unsqueeze(2)
        evidence = evidence + role
        evidence = evidence.reshape(batch, roles * tokens, self.model_dim)
        evidence_mask = token_mask.bool().reshape(batch, roles * tokens)
        candidate_states = self.candidate_projection(candidate_features)
        candidate_mask = torch.ones(
            candidate_states.shape[:2],
            dtype=torch.bool,
            device=candidate_states.device,
        )

        candidate_states = self.initial_read(candidate_states, evidence, evidence_mask)
        workspace = self._initial_workspace(evidence, evidence_mask)
        self._last_trajectory_logits = []
        self._last_compatibility_logits = []
        self._record_step(candidate_states, evidence, evidence_mask)
        if bool(self.transport_config.transport_disabled) or self.refinement_steps == 0:
            return self._last_trajectory_logits[-1]

        for step in range(self.refinement_steps):
            if workspace is not None:
                workspace = self._update_workspace(workspace, evidence, evidence_mask, candidate_states, candidate_mask)
                candidate_states = self._read_workspace(candidate_states, workspace)
            if bool(self.transport_config.candidate_reads_evidence):
                candidate_states = self._candidate_block(step)(candidate_states, evidence, evidence_mask)
            if bool(self.transport_config.update_evidence):
                source_candidates = candidate_states.detach() if bool(self.transport_config.stop_gradient_evidence_update) else candidate_states
                evidence = self._evidence_block(step)(evidence, source_candidates, candidate_mask)
            self._record_step(candidate_states, evidence, evidence_mask)
        return self._last_trajectory_logits[-1]

    def auxiliary_loss(self, labels: torch.Tensor) -> torch.Tensor:
        if not self._last_trajectory_logits:
            return torch.zeros((), dtype=torch.float32, device=labels.device)
        weight = float(self.transport_config.aux_weight)
        if weight <= 0.0:
            return torch.zeros((), dtype=self._last_trajectory_logits[-1].dtype, device=labels.device)
        mode = str(self.transport_config.aux_loss)
        if mode == "none":
            loss = torch.zeros((), dtype=self._last_trajectory_logits[-1].dtype, device=labels.device)
        elif mode in {"margin_improvement", "energy_margin"}:
            first = _candidate_margin(self._last_trajectory_logits[0], labels)
            final = _candidate_margin(self._last_trajectory_logits[-1], labels)
            target = float(self.transport_config.aux_margin)
            loss = F.relu(target - (final - first)).mean()
        elif mode in {"monotonic_margin", "monotonic_support"}:
            losses = []
            margins = [_candidate_margin(logits, labels) for logits in self._last_trajectory_logits]
            for previous, current in zip(margins, margins[1:]):
                losses.append(F.relu(previous - current).mean())
            loss = torch.stack(losses).mean() if losses else torch.zeros((), dtype=margins[0].dtype, device=labels.device)
        elif mode in {"supervised_contrastive", "compatibility_ce"}:
            logits = self._last_compatibility_logits[-1] if self._last_compatibility_logits else self._last_trajectory_logits[-1]
            loss = F.cross_entropy(logits, labels)
        elif mode in {"triplet_margin", "triplet"}:
            logits = self._last_compatibility_logits[-1] if self._last_compatibility_logits else self._last_trajectory_logits[-1]
            margin = _candidate_margin(logits, labels)
            loss = F.relu(float(self.transport_config.aux_margin) - margin).mean()
        else:
            raise ValueError(f"unknown Stage 7 auxiliary loss: {mode}")
        self._last_auxiliary_loss = loss.detach()
        return loss * weight

    def transport_audit(self) -> Dict[str, object]:
        return {
            **asdict(self.transport_config),
            "model_dim": self.model_dim,
            "workspace_slots": self.workspace_slots,
            "last_auxiliary_loss": float(self._last_auxiliary_loss.detach().float().cpu().item())
            if self._last_auxiliary_loss.numel()
            else 0.0,
        }

    def _record_step(self, candidate_states: torch.Tensor, evidence: torch.Tensor, evidence_mask: torch.Tensor) -> None:
        state_logits = self.score_head(candidate_states).squeeze(-1)
        compatibility = self._compatibility_logits(candidate_states, evidence, evidence_mask)
        if str(self.transport_config.energy_source) == "candidate_evidence_compatibility":
            logits = state_logits + self.compatibility_scale.tanh() * compatibility
        else:
            logits = state_logits
        self._last_trajectory_logits.append(logits)
        self._last_compatibility_logits.append(compatibility)

    def _compatibility_logits(self, candidate_states: torch.Tensor, evidence: torch.Tensor, evidence_mask: torch.Tensor) -> torch.Tensor:
        mask = evidence_mask.to(dtype=evidence.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (evidence * mask).sum(dim=1) / denom
        return torch.einsum("bcd,bd->bc", F.normalize(candidate_states, dim=-1), F.normalize(pooled, dim=-1))

    def _candidate_block(self, step: int) -> nn.Module:
        index = 0 if bool(self.transport_config.shared_update) else min(step, len(self.candidate_blocks) - 1)
        return self.candidate_blocks[index]

    def _evidence_block(self, step: int) -> nn.Module:
        if bool(self.transport_config.share_candidate_evidence_weights):
            return self.candidate_blocks[0]
        index = 0 if bool(self.transport_config.shared_update) else min(step, len(self.evidence_blocks) - 1)
        return self.evidence_blocks[index]

    def _initial_workspace(self, evidence: torch.Tensor, evidence_mask: torch.Tensor) -> torch.Tensor | None:
        if self.workspace_slots <= 0 or self.workspace_seed is None:
            return None
        workspace = self.workspace_seed.expand(evidence.shape[0], -1, -1)
        if str(self.transport_config.workspace_init) == "evidence_pooled":
            mask = evidence_mask.to(dtype=evidence.dtype).unsqueeze(-1)
            pooled = (evidence * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            workspace = workspace + pooled
        return workspace

    def _update_workspace(
        self,
        workspace: torch.Tensor,
        evidence: torch.Tensor,
        evidence_mask: torch.Tensor,
        candidate_states: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.workspace_from_evidence is not None and bool(self.transport_config.evidence_writes_workspace):
            workspace = self.workspace_from_evidence(workspace, evidence, evidence_mask)
        if self.workspace_from_candidates is not None and bool(self.transport_config.candidate_writes_workspace):
            workspace = self.workspace_from_candidates(workspace, candidate_states, candidate_mask)
        return workspace

    def _read_workspace(self, candidate_states: torch.Tensor, workspace: torch.Tensor) -> torch.Tensor:
        if not bool(self.transport_config.candidate_reads_workspace):
            return candidate_states
        mask = torch.ones(workspace.shape[:2], dtype=torch.bool, device=workspace.device)
        if bool(self.transport_config.candidate_conditioned_workspace_read):
            if self.candidate_from_workspace is None:
                return candidate_states
            return self.candidate_from_workspace(candidate_states, workspace, mask)
        pooled = workspace.mean(dim=1, keepdim=True).expand(-1, candidate_states.shape[1], -1)
        if self.workspace_pool_gate is None:
            return candidate_states + pooled
        return self.workspace_pool_gate(candidate_states, pooled)

    def _freeze_transport_modules(self) -> None:
        modules: List[nn.Module | nn.Parameter | None] = [
            self.candidate_blocks,
            self.evidence_blocks,
            self.workspace_seed,
            self.workspace_from_evidence,
            self.workspace_from_candidates,
            self.candidate_from_workspace,
            self.workspace_pool_gate,
        ]
        for item in modules:
            if item is None:
                continue
            if isinstance(item, nn.Parameter):
                item.requires_grad_(False)
                continue
            for parameter in item.parameters():
                parameter.requires_grad_(False)


class _AttentionUpdateBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        gated_residual: bool,
        norm_position: str,
    ) -> None:
        super().__init__()
        self.norm_position = str(norm_position)
        self.attention = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.query_norm = nn.LayerNorm(model_dim)
        self.post_attention_norm = nn.LayerNorm(model_dim)
        self.ff_norm = nn.LayerNorm(model_dim)
        self.post_ff_norm = nn.LayerNorm(model_dim)
        self.attn_gate = _GatedResidual(model_dim) if gated_residual else None
        self.ff_gate = _GatedResidual(model_dim) if gated_residual else None
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
        )

    def forward(self, query: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn_query = self.query_norm(query) if self.norm_position == "pre" else query
        attended, _weights = self.attention(
            query=attn_query,
            key=tokens,
            value=tokens,
            key_padding_mask=~mask.bool(),
            need_weights=False,
        )
        if self.attn_gate is None:
            query = query + attended
        else:
            query = self.attn_gate(query, attended)
        if self.norm_position == "post":
            query = self.post_attention_norm(query)
        ff_input = self.ff_norm(query) if self.norm_position == "pre" else query
        updated = self.feed_forward(ff_input)
        if self.ff_gate is None:
            query = query + updated
        else:
            query = self.ff_gate(query, updated)
        if self.norm_position == "post":
            query = self.post_ff_norm(query)
        return query


class _GatedResidual(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(model_dim * 2, model_dim), nn.Sigmoid())

    def forward(self, state: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([state, update], dim=-1))
        return state + gate * update


def _candidate_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.to(dtype=torch.long, device=logits.device)
    gold = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    wrong = logits.masked_fill(F.one_hot(labels, num_classes=logits.shape[1]).bool(), -1e9).max(dim=1).values
    return gold - wrong


def _compatible_dim(model_dim: int, num_heads: int) -> int:
    model_dim = max(1, int(model_dim))
    num_heads = max(1, int(num_heads))
    remainder = model_dim % num_heads
    if remainder:
        model_dim += num_heads - remainder
    return model_dim


def _compatible_heads(model_dim: int, num_heads: int) -> int:
    model_dim = max(1, int(model_dim))
    num_heads = max(1, int(num_heads))
    while num_heads > 1 and model_dim % num_heads != 0:
        num_heads -= 1
    return max(1, num_heads)

