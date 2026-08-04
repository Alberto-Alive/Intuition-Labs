"""Model components for DIGIT Extrapolation E22."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PrototypeAttentionLayerResult:
    """Forward outputs from one dual-parameter prototype-attention layer."""

    h_semantic: torch.Tensor
    h_out: torch.Tensor
    certainty: torch.Tensor
    occupancy: int
    occupancy_gate: torch.Tensor
    attention_output: torch.Tensor


@dataclass
class DualParameterForwardResult:
    """Forward outputs needed for E22 training and reporting."""

    logits: torch.Tensor
    input_embedding: torch.Tensor
    layer1_semantic: torch.Tensor
    layer2_semantic: torch.Tensor
    layer1_output: torch.Tensor
    layer2_output: torch.Tensor
    layer1_certainty: torch.Tensor
    layer2_certainty: torch.Tensor
    layer1_occupancy: int
    layer2_occupancy: int


@dataclass
class BaselineForwardResult:
    """Forward outputs for the matched baseline model."""

    logits: torch.Tensor
    input_embedding: torch.Tensor
    layer1_activation: torch.Tensor
    layer2_activation: torch.Tensor


class PrototypeAttentionLayer(nn.Module):
    """One E22 hidden layer with semantic weights and detached familiarity prototypes."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        key_dim: int,
        num_prototypes: int,
        prototype_lambda: float,
        eps: float,
    ) -> None:
        super().__init__()
        if key_dim % 2 != 0:
            raise ValueError("key_dim must be even")
        if num_prototypes <= 0:
            raise ValueError("num_prototypes must be positive")

        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.num_prototypes = num_prototypes
        self.prototype_lambda = prototype_lambda
        self.eps = eps
        self.repulsion_threshold = sqrt(2.0 / num_prototypes)

        self.semantic = nn.Linear(input_dim, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, key_dim)
        self.k_semantic_proj = nn.Linear(hidden_dim, key_dim // 2)
        self.k_familiarity_proj = nn.Linear(hidden_dim, key_dim // 2)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.register_buffer("prototypes", torch.zeros(num_prototypes, hidden_dim))
        self.register_buffer("valid_mask", torch.zeros(num_prototypes, dtype=torch.bool))

    @property
    def occupancy(self) -> int:
        """Return the number of valid prototypes."""
        return int(self.valid_mask.sum().item())

    def forward(self, inputs: torch.Tensor) -> PrototypeAttentionLayerResult:
        """Compute the semantic state, prototype attention, and detached certainty."""
        h_semantic = torch.relu(self.semantic(inputs))
        occupancy_gate = torch.tensor(
            self.occupancy / self.num_prototypes,
            dtype=h_semantic.dtype,
            device=h_semantic.device,
        )
        attention_output = self._prototype_attention(h_semantic)
        h_out = h_semantic + occupancy_gate * attention_output
        certainty = self.read_only_certainty(h_semantic)
        return PrototypeAttentionLayerResult(
            h_semantic=h_semantic,
            h_out=h_out,
            certainty=certainty,
            occupancy=self.occupancy,
            occupancy_gate=occupancy_gate,
            attention_output=attention_output,
        )

    @torch.no_grad()
    def update_prototypes(self, semantic_activations: torch.Tensor) -> None:
        """Apply the seed-then-EMA update rule with repulsion in normalized space."""
        detached_activations = F.normalize(
            semantic_activations.detach().to(dtype=self.prototypes.dtype),
            dim=-1,
            eps=self.eps,
        )

        for normalized_row in detached_activations:
            invalid_indices = torch.nonzero(~self.valid_mask, as_tuple=False).flatten()
            if invalid_indices.numel() > 0:
                write_index = invalid_indices[0]
                self.prototypes[write_index].copy_(normalized_row)
                self.valid_mask[write_index] = True
            else:
                valid_indices = torch.nonzero(self.valid_mask, as_tuple=False).flatten()
                valid_prototypes = self.prototypes.index_select(0, valid_indices)
                distances = torch.norm(valid_prototypes - normalized_row.unsqueeze(0), dim=-1)
                nearest_index = valid_indices[distances.argmin()]
                updated = (
                    self.prototype_lambda * self.prototypes[nearest_index]
                    + (1.0 - self.prototype_lambda) * normalized_row
                )
                self.prototypes[nearest_index].copy_(F.normalize(updated.unsqueeze(0), dim=-1, eps=self.eps)[0])

            self._apply_repulsion()

    @torch.no_grad()
    def read_only_certainty(self, semantic_activations: torch.Tensor) -> torch.Tensor:
        """Compute detached nearest-prototype max cosine certainty over valid prototypes only."""
        if not torch.any(self.valid_mask):
            return torch.zeros(
                semantic_activations.shape[0],
                dtype=semantic_activations.dtype,
                device=semantic_activations.device,
            )

        normalized_queries = F.normalize(semantic_activations.detach(), dim=-1, eps=self.eps)
        normalized_prototypes = F.normalize(self.prototypes[self.valid_mask].detach(), dim=-1, eps=self.eps)
        similarities = normalized_queries @ normalized_prototypes.t()
        return similarities.max(dim=-1).values

    def _prototype_attention(self, semantic_activations: torch.Tensor) -> torch.Tensor:
        if not torch.any(self.valid_mask):
            return torch.zeros_like(semantic_activations)

        query = self.q_proj(semantic_activations)
        valid_prototypes = self.prototypes[self.valid_mask].detach().to(
            device=semantic_activations.device,
            dtype=semantic_activations.dtype,
        )
        key_semantic = self.k_semantic_proj(valid_prototypes)
        key_familiarity = self.k_familiarity_proj(valid_prototypes)
        keys = torch.cat((key_semantic, key_familiarity), dim=-1)
        values = self.v_proj(valid_prototypes)
        attention_logits = query @ keys.t()
        attention_logits = attention_logits / sqrt(self.key_dim)
        attention_weights = attention_logits.softmax(dim=-1)
        return attention_weights @ values

    @torch.no_grad()
    def _apply_repulsion(self) -> None:
        """Apply one repulsion pass in normalized space without moving buffers off-device."""
        valid_indices = torch.nonzero(self.valid_mask, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            return

        valid_prototypes = self.prototypes.index_select(0, valid_indices)
        pairwise_diff = valid_prototypes[:, None, :] - valid_prototypes[None, :, :]
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
        corrected = F.normalize(valid_prototypes + net_delta, dim=-1, eps=self.eps)
        self.prototypes[valid_indices] = corrected


class _InputEncodingMixin:
    """Shared input embedding helper for the E22 models."""

    part_a_embedding: nn.Embedding
    part_b_embedding: nn.Embedding

    def encode_inputs_with_jitter(
        self,
        part_a: torch.Tensor,
        part_b: torch.Tensor,
        part_a_noise: torch.Tensor | None = None,
        part_b_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed the two categorical parts, add optional jitter, and concatenate."""
        part_a_embedding = self.part_a_embedding(part_a)
        part_b_embedding = self.part_b_embedding(part_b)

        if part_a_noise is not None:
            if part_a_noise.shape != part_a_embedding.shape:
                raise ValueError(
                    f"part_a_noise must have shape {tuple(part_a_embedding.shape)}, got {tuple(part_a_noise.shape)}"
                )
            part_a_embedding = part_a_embedding + part_a_noise

        if part_b_noise is not None:
            if part_b_noise.shape != part_b_embedding.shape:
                raise ValueError(
                    f"part_b_noise must have shape {tuple(part_b_embedding.shape)}, got {tuple(part_b_noise.shape)}"
                )
            part_b_embedding = part_b_embedding + part_b_noise

        return torch.cat((part_a_embedding, part_b_embedding), dim=-1)


class DualParameterAttentionMLP(nn.Module, _InputEncodingMixin):
    """Two-layer E22 model with intrinsic prototype attention at each hidden layer."""

    def __init__(
        self,
        *,
        part_a_vocab_size: int,
        part_b_vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        key_dim: int,
        num_classes: int,
        num_prototypes: int,
        prototype_lambda: float,
        eps: float,
    ) -> None:
        super().__init__()
        self.part_a_embedding = nn.Embedding(part_a_vocab_size, embedding_dim)
        self.part_b_embedding = nn.Embedding(part_b_vocab_size, embedding_dim)
        self.layer1 = PrototypeAttentionLayer(
            input_dim=embedding_dim * 2,
            hidden_dim=hidden_dim,
            key_dim=key_dim,
            num_prototypes=num_prototypes,
            prototype_lambda=prototype_lambda,
            eps=eps,
        )
        self.layer2 = PrototypeAttentionLayer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            key_dim=key_dim,
            num_prototypes=num_prototypes,
            prototype_lambda=prototype_lambda,
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
        """Run the E22 forward path and expose per-layer semantic activations."""
        input_embedding = self.encode_inputs_with_jitter(
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
            layer1_certainty=layer1.certainty,
            layer2_certainty=layer2.certainty,
            layer1_occupancy=layer1.occupancy,
            layer2_occupancy=layer2.occupancy,
        )

    @torch.no_grad()
    def update_prototypes(self, layer1_semantic: torch.Tensor, layer2_semantic: torch.Tensor) -> None:
        """Update both familiarity layers from the current batch's detached semantic activations."""
        self.layer1.update_prototypes(layer1_semantic)
        self.layer2.update_prototypes(layer2_semantic)


class BaselineMLP(nn.Module, _InputEncodingMixin):
    """Matched baseline 2-layer MLP with the same hidden width and training setup."""

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
        """Run the matched baseline forward path."""
        input_embedding = self.encode_inputs_with_jitter(
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
