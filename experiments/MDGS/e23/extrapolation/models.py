"""Model definitions for DIGIT Extrapolation E23."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MemoryLayerForwardResult:
    """Forward outputs exposed by one E23 hidden layer."""

    h_semantic: torch.Tensor
    attention_output: torch.Tensor
    occupancy: int
    occupancy_gate: torch.Tensor
    h_out: torch.Tensor


@dataclass
class DualParameterForwardResult:
    """Forward outputs needed for E23 invariant tests."""

    logits: torch.Tensor
    input_embedding: torch.Tensor
    layer1_semantic: torch.Tensor
    layer2_semantic: torch.Tensor
    layer1_output: torch.Tensor
    layer2_output: torch.Tensor
    layer1_occupancy: int
    layer2_occupancy: int
    layer1_occupancy_gate: torch.Tensor
    layer2_occupancy_gate: torch.Tensor


@dataclass
class BaselineForwardResult:
    """Forward outputs for the matched baseline model."""

    logits: torch.Tensor
    input_embedding: torch.Tensor
    layer1_activation: torch.Tensor
    layer2_activation: torch.Tensor


def normalize_rows(rows: torch.Tensor, eps: float) -> torch.Tensor:
    """Apply L2 normalization row-wise."""
    return F.normalize(rows, dim=-1, eps=eps)


class ExperienceMemoryAttentionLayer(nn.Module):
    """E23 familiarity layer with detached experience-updated prototype buffers."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        key_dim: int,
        num_memory_slots: int,
        memory_lambda: float,
        eps: float,
    ) -> None:
        super().__init__()
        if key_dim % 2 != 0:
            raise ValueError("key_dim must be even")

        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.num_memory_slots = num_memory_slots
        self.memory_lambda = memory_lambda
        self.eps = eps
        self.repulsion_threshold = sqrt(2.0 / num_memory_slots)

        self.semantic = nn.Linear(input_dim, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, key_dim)
        self.k_semantic_proj = nn.Linear(hidden_dim, key_dim // 2)
        self.k_familiarity_proj = nn.Linear(hidden_dim, key_dim // 2)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.register_buffer("prototypes", torch.zeros(num_memory_slots, hidden_dim))
        self.register_buffer("valid_mask", torch.ones(num_memory_slots, dtype=torch.bool))

        self.record_update_inputs = True
        self.last_update_inputs: list[torch.Tensor] = []

    @property
    def occupancy(self) -> int:
        """Return the number of valid memory slots."""
        return int(self.valid_mask.sum().item())

    @torch.no_grad()
    def set_memory(self, memory_slots: torch.Tensor) -> None:
        """Copy the approved initial memory slots and mark all slots valid."""
        if memory_slots.shape != self.prototypes.shape:
            raise ValueError(
                f"Expected memory shape {tuple(self.prototypes.shape)}, got {tuple(memory_slots.shape)}"
            )
        self.prototypes.copy_(memory_slots.to(device=self.prototypes.device, dtype=self.prototypes.dtype))
        self.valid_mask.fill_(True)

    def effective_memory(self) -> torch.Tensor:
        """Return the effective memory used by attention."""
        return self.prototypes[self.valid_mask].detach()

    def forward(self, inputs: torch.Tensor) -> MemoryLayerForwardResult:
        """Run semantic projection and residual memory attention."""
        h_semantic = torch.relu(self.semantic(inputs))
        occupancy_gate = torch.tensor(
            self.occupancy / self.num_memory_slots,
            dtype=h_semantic.dtype,
            device=h_semantic.device,
        )
        attention_output = self._memory_attention(h_semantic)
        h_out = h_semantic + occupancy_gate * attention_output
        return MemoryLayerForwardResult(
            h_semantic=h_semantic,
            attention_output=attention_output,
            occupancy=self.occupancy,
            occupancy_gate=occupancy_gate,
            h_out=h_out,
        )

    @torch.no_grad()
    def update_prototypes(self, semantic_activations: torch.Tensor) -> None:
        """Apply detached nearest-slot EMA updates followed by repulsion."""
        normalized_activations = normalize_rows(
            semantic_activations.detach().to(dtype=self.prototypes.dtype),
            eps=self.eps,
        )
        if self.record_update_inputs:
            self.last_update_inputs = [row.detach().clone() for row in semantic_activations]
        else:
            self.last_update_inputs = []
        self.prototypes.copy_(normalize_rows(self.prototypes, eps=self.eps))

        for normalized_row in normalized_activations:
            distances = torch.norm(self.prototypes - normalized_row.unsqueeze(0), dim=-1)
            nearest_index = distances.argmin()
            updated = self.memory_lambda * self.prototypes[nearest_index] + (1.0 - self.memory_lambda) * normalized_row
            self.prototypes[nearest_index].copy_(normalize_rows(updated.unsqueeze(0), eps=self.eps)[0])
            self._apply_repulsion()

    def _memory_attention(self, semantic_activations: torch.Tensor) -> torch.Tensor:
        """Attend from semantic activations into detached prototype memory."""
        memory = self.effective_memory().to(device=semantic_activations.device, dtype=semantic_activations.dtype)
        query = self.q_proj(semantic_activations)
        keys = torch.cat(
            (
                self.k_semantic_proj(memory),
                self.k_familiarity_proj(memory),
            ),
            dim=-1,
        )
        values = self.v_proj(memory)
        attention_logits = query @ keys.t()
        attention_logits = attention_logits / sqrt(self.key_dim)
        attention_weights = attention_logits.softmax(dim=-1)
        return attention_weights @ values

    @torch.no_grad()
    def _apply_repulsion(self) -> None:
        """Apply one symmetric repulsion pass in normalized space."""
        pairwise_diff = self.prototypes[:, None, :] - self.prototypes[None, :, :]
        pairwise_distance = torch.norm(pairwise_diff, dim=-1)
        upper_triangle = torch.triu(
            torch.ones_like(pairwise_distance, dtype=torch.bool),
            diagonal=1,
        )
        active_pairs = upper_triangle & (pairwise_distance < self.repulsion_threshold)
        safe_distance = pairwise_distance.clamp_min(self.eps)
        pairwise_scale = 0.5 * (self.repulsion_threshold - pairwise_distance).clamp_min(0.0) / safe_distance
        pairwise_delta = pairwise_scale.unsqueeze(-1) * pairwise_diff
        pairwise_delta = pairwise_delta * active_pairs.unsqueeze(-1)
        net_delta = pairwise_delta.sum(dim=1) - pairwise_delta.sum(dim=0)
        self.prototypes.copy_(normalize_rows(self.prototypes + net_delta, eps=self.eps))


class LearnedMemoryAttentionLayer(nn.Module):
    """E23 learned-memory ablation layer with gradient-trained slot parameters."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        key_dim: int,
        num_memory_slots: int,
        eps: float,
    ) -> None:
        super().__init__()
        if key_dim % 2 != 0:
            raise ValueError("key_dim must be even")

        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.num_memory_slots = num_memory_slots
        self.eps = eps

        self.semantic = nn.Linear(input_dim, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, key_dim)
        self.k_semantic_proj = nn.Linear(hidden_dim, key_dim // 2)
        self.k_familiarity_proj = nn.Linear(hidden_dim, key_dim // 2)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.raw_slots = nn.Parameter(torch.zeros(num_memory_slots, hidden_dim))
        self.register_buffer("valid_mask", torch.ones(num_memory_slots, dtype=torch.bool))

    @property
    def occupancy(self) -> int:
        """Return the number of valid memory slots."""
        return int(self.valid_mask.sum().item())

    @torch.no_grad()
    def set_memory(self, memory_slots: torch.Tensor) -> None:
        """Copy the approved initial slot tensor and mark all slots valid."""
        if memory_slots.shape != self.raw_slots.shape:
            raise ValueError(
                f"Expected memory shape {tuple(self.raw_slots.shape)}, got {tuple(memory_slots.shape)}"
            )
        self.raw_slots.copy_(memory_slots.to(device=self.raw_slots.device, dtype=self.raw_slots.dtype))
        self.valid_mask.fill_(True)

    def effective_memory(self) -> torch.Tensor:
        """Return the normalized slot matrix used by attention."""
        return normalize_rows(self.raw_slots[self.valid_mask], eps=self.eps)

    def forward(self, inputs: torch.Tensor) -> MemoryLayerForwardResult:
        """Run semantic projection and residual learned-memory attention."""
        h_semantic = torch.relu(self.semantic(inputs))
        occupancy_gate = torch.tensor(
            self.occupancy / self.num_memory_slots,
            dtype=h_semantic.dtype,
            device=h_semantic.device,
        )
        attention_output = self._memory_attention(h_semantic)
        h_out = h_semantic + occupancy_gate * attention_output
        return MemoryLayerForwardResult(
            h_semantic=h_semantic,
            attention_output=attention_output,
            occupancy=self.occupancy,
            occupancy_gate=occupancy_gate,
            h_out=h_out,
        )

    def _memory_attention(self, semantic_activations: torch.Tensor) -> torch.Tensor:
        """Attend from semantic activations into normalized learned slots."""
        memory = self.effective_memory()
        query = self.q_proj(semantic_activations)
        keys = torch.cat(
            (
                self.k_semantic_proj(memory),
                self.k_familiarity_proj(memory),
            ),
            dim=-1,
        )
        values = self.v_proj(memory)
        attention_logits = query @ keys.t()
        attention_logits = attention_logits / sqrt(self.key_dim)
        attention_weights = attention_logits.softmax(dim=-1)
        return attention_weights @ values


class _InputEncodingMixin:
    """Shared input embedding helper."""

    part_a_embedding: nn.Embedding
    part_b_embedding: nn.Embedding

    def encode_inputs(
        self,
        part_a: torch.Tensor,
        part_b: torch.Tensor,
        part_a_noise: torch.Tensor | None = None,
        part_b_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed both parts, add optional jitter, and concatenate."""
        part_a_embedding = self.part_a_embedding(part_a)
        part_b_embedding = self.part_b_embedding(part_b)

        if part_a_noise is not None:
            part_a_embedding = part_a_embedding + part_a_noise
        if part_b_noise is not None:
            part_b_embedding = part_b_embedding + part_b_noise

        return torch.cat((part_a_embedding, part_b_embedding), dim=-1)


class DualParameterAttentionMLP(nn.Module, _InputEncodingMixin):
    """Model 2 from E23 with detached experience-updated prototype memory."""

    def __init__(
        self,
        *,
        part_a_vocab_size: int,
        part_b_vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        key_dim: int,
        num_classes: int,
        num_memory_slots: int,
        memory_lambda: float,
        eps: float,
    ) -> None:
        super().__init__()
        self.part_a_embedding = nn.Embedding(part_a_vocab_size, embedding_dim)
        self.part_b_embedding = nn.Embedding(part_b_vocab_size, embedding_dim)
        self.layer1 = ExperienceMemoryAttentionLayer(
            input_dim=embedding_dim * 2,
            hidden_dim=hidden_dim,
            key_dim=key_dim,
            num_memory_slots=num_memory_slots,
            memory_lambda=memory_lambda,
            eps=eps,
        )
        self.layer2 = ExperienceMemoryAttentionLayer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            key_dim=key_dim,
            num_memory_slots=num_memory_slots,
            memory_lambda=memory_lambda,
            eps=eps,
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        part_a: torch.Tensor,
        part_b: torch.Tensor,
        part_a_noise: torch.Tensor | None = None,
        part_b_noise: torch.Tensor | None = None,
    ) -> DualParameterForwardResult:
        """Run the approved E23 dual-parameter forward computation."""
        input_embedding = self.encode_inputs(
            part_a,
            part_b,
            part_a_noise=part_a_noise,
            part_b_noise=part_b_noise,
        )
        layer1 = self.layer1(input_embedding)
        layer2 = self.layer2(layer1.h_out)
        logits = self.classifier(layer2.h_out)
        return DualParameterForwardResult(
            logits=logits,
            input_embedding=input_embedding,
            layer1_semantic=layer1.h_semantic,
            layer2_semantic=layer2.h_semantic,
            layer1_output=layer1.h_out,
            layer2_output=layer2.h_out,
            layer1_occupancy=layer1.occupancy,
            layer2_occupancy=layer2.occupancy,
            layer1_occupancy_gate=layer1.occupancy_gate,
            layer2_occupancy_gate=layer2.occupancy_gate,
        )

    @torch.no_grad()
    def set_memory(self, layer1_memory: torch.Tensor, layer2_memory: torch.Tensor) -> None:
        """Set both hidden-layer memories."""
        self.layer1.set_memory(layer1_memory)
        self.layer2.set_memory(layer2_memory)

    @torch.no_grad()
    def update_prototypes(self, layer1_semantic: torch.Tensor, layer2_semantic: torch.Tensor) -> None:
        """Update both familiarity layers from detached semantic activations."""
        self.layer1.update_prototypes(layer1_semantic)
        self.layer2.update_prototypes(layer2_semantic)


class ActiveLearnedMemoryMLP(nn.Module, _InputEncodingMixin):
    """Model 3 from E23 with gradient-trained learned memory slots."""

    def __init__(
        self,
        *,
        part_a_vocab_size: int,
        part_b_vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        key_dim: int,
        num_classes: int,
        num_memory_slots: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.part_a_embedding = nn.Embedding(part_a_vocab_size, embedding_dim)
        self.part_b_embedding = nn.Embedding(part_b_vocab_size, embedding_dim)
        self.layer1 = LearnedMemoryAttentionLayer(
            input_dim=embedding_dim * 2,
            hidden_dim=hidden_dim,
            key_dim=key_dim,
            num_memory_slots=num_memory_slots,
            eps=eps,
        )
        self.layer2 = LearnedMemoryAttentionLayer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            key_dim=key_dim,
            num_memory_slots=num_memory_slots,
            eps=eps,
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        part_a: torch.Tensor,
        part_b: torch.Tensor,
        part_a_noise: torch.Tensor | None = None,
        part_b_noise: torch.Tensor | None = None,
    ) -> DualParameterForwardResult:
        """Run the approved E23 learned-memory ablation forward computation."""
        input_embedding = self.encode_inputs(
            part_a,
            part_b,
            part_a_noise=part_a_noise,
            part_b_noise=part_b_noise,
        )
        layer1 = self.layer1(input_embedding)
        layer2 = self.layer2(layer1.h_out)
        logits = self.classifier(layer2.h_out)
        return DualParameterForwardResult(
            logits=logits,
            input_embedding=input_embedding,
            layer1_semantic=layer1.h_semantic,
            layer2_semantic=layer2.h_semantic,
            layer1_output=layer1.h_out,
            layer2_output=layer2.h_out,
            layer1_occupancy=layer1.occupancy,
            layer2_occupancy=layer2.occupancy,
            layer1_occupancy_gate=layer1.occupancy_gate,
            layer2_occupancy_gate=layer2.occupancy_gate,
        )

    @torch.no_grad()
    def set_memory(self, layer1_memory: torch.Tensor, layer2_memory: torch.Tensor) -> None:
        """Set both hidden-layer learned-memory tensors."""
        self.layer1.set_memory(layer1_memory)
        self.layer2.set_memory(layer2_memory)


class BaselineMLP(nn.Module, _InputEncodingMixin):
    """Model 1 from E23: the matched semantic-only baseline."""

    def __init__(
        self,
        *,
        part_a_vocab_size: int,
        part_b_vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.part_a_embedding = nn.Embedding(part_a_vocab_size, embedding_dim)
        self.part_b_embedding = nn.Embedding(part_b_vocab_size, embedding_dim)
        self.hidden1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        part_a: torch.Tensor,
        part_b: torch.Tensor,
        part_a_noise: torch.Tensor | None = None,
        part_b_noise: torch.Tensor | None = None,
    ) -> BaselineForwardResult:
        """Run the matched baseline forward computation."""
        input_embedding = self.encode_inputs(
            part_a,
            part_b,
            part_a_noise=part_a_noise,
            part_b_noise=part_b_noise,
        )
        layer1_activation = torch.relu(self.hidden1(input_embedding))
        layer2_activation = torch.relu(self.hidden2(layer1_activation))
        logits = self.classifier(layer2_activation)
        return BaselineForwardResult(
            logits=logits,
            input_embedding=input_embedding,
            layer1_activation=layer1_activation,
            layer2_activation=layer2_activation,
        )
