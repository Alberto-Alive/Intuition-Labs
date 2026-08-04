"""Minimal model components for the DIGIT Extrapolation E14 experiment."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MLPForwardResult:
    """Forward pass outputs needed for training and post-hoc certainty analysis."""

    logits: torch.Tensor
    input_embedding: torch.Tensor
    layer1_activation: torch.Tensor
    layer2_activation: torch.Tensor
    layer1_certainty: torch.Tensor | None = None
    layer2_certainty: torch.Tensor | None = None


class FrozenActivationBankMLP(nn.Module):
    """Two-layer MLP with detached activation-bank buffers for post-hoc certainty."""

    def __init__(
        self,
        part_a_vocab_size: int,
        part_b_vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        bank_capacity: int,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if part_a_vocab_size <= 0 or part_b_vocab_size <= 0:
            raise ValueError("Vocabulary sizes must be positive")
        if embedding_dim <= 0 or hidden_dim <= 0 or num_classes <= 0:
            raise ValueError("embedding_dim, hidden_dim, and num_classes must be positive")
        if bank_capacity <= 0:
            raise ValueError("bank_capacity must be positive")

        self.part_a_vocab_size = part_a_vocab_size
        self.part_b_vocab_size = part_b_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.bank_capacity = bank_capacity
        self.eps = eps

        self.part_a_embedding = nn.Embedding(part_a_vocab_size, embedding_dim)
        self.part_b_embedding = nn.Embedding(part_b_vocab_size, embedding_dim)
        self.hidden1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.register_buffer("layer1_bank", torch.zeros(bank_capacity, hidden_dim))
        self.register_buffer("layer2_bank", torch.zeros(bank_capacity, hidden_dim))
        self.register_buffer("layer1_bank_mask", torch.zeros(bank_capacity, dtype=torch.bool))
        self.register_buffer("layer2_bank_mask", torch.zeros(bank_capacity, dtype=torch.bool))

    def encode_inputs(self, part_a: torch.Tensor, part_b: torch.Tensor) -> torch.Tensor:
        """Embed the two categorical parts and concatenate them."""
        return torch.cat((self.part_a_embedding(part_a), self.part_b_embedding(part_b)), dim=-1)

    def encode_inputs_with_jitter(
        self,
        part_a: torch.Tensor,
        part_b: torch.Tensor,
        part_a_noise: torch.Tensor | None = None,
        part_b_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed the two categorical parts, add optional jitter, and concatenate them."""
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

    def forward(
        self,
        part_a: torch.Tensor,
        part_b: torch.Tensor,
        part_a_noise: torch.Tensor | None = None,
        part_b_noise: torch.Tensor | None = None,
        *,
        return_read_only_certainty: bool = False,
    ) -> MLPForwardResult:
        """Compute logits and, optionally, detached certainty scores."""
        input_embedding = self.encode_inputs_with_jitter(
            part_a,
            part_b,
            part_a_noise=part_a_noise,
            part_b_noise=part_b_noise,
        )
        layer1_activation = torch.relu(self.hidden1(input_embedding))
        layer2_activation = torch.relu(self.hidden2(layer1_activation))
        logits = self.classifier(layer2_activation)

        layer1_certainty = None
        layer2_certainty = None
        if return_read_only_certainty:
            layer1_certainty = self.read_only_max_cosine_certainty(
                layer1_activation,
                self.layer1_bank,
                self.layer1_bank_mask,
            )
            layer2_certainty = self.read_only_max_cosine_certainty(
                layer2_activation,
                self.layer2_bank,
                self.layer2_bank_mask,
            )

        return MLPForwardResult(
            logits=logits,
            input_embedding=input_embedding,
            layer1_activation=layer1_activation,
            layer2_activation=layer2_activation,
            layer1_certainty=layer1_certainty,
            layer2_certainty=layer2_certainty,
        )

    @torch.no_grad()
    def seed_banks(self, layer1_bank: torch.Tensor, layer2_bank: torch.Tensor) -> None:
        """Fill the bank buffers with fixed activation snapshots for testing."""
        if layer1_bank.ndim != 2 or layer1_bank.shape[1] != self.hidden_dim:
            raise ValueError(
                f"layer1_bank must have shape [n, {self.hidden_dim}], got {tuple(layer1_bank.shape)}"
            )
        if layer2_bank.ndim != 2 or layer2_bank.shape[1] != self.hidden_dim:
            raise ValueError(
                f"layer2_bank must have shape [n, {self.hidden_dim}], got {tuple(layer2_bank.shape)}"
            )
        if layer1_bank.shape[0] > self.bank_capacity or layer2_bank.shape[0] > self.bank_capacity:
            raise ValueError("Provided banks exceed the configured bank capacity")

        self.layer1_bank.zero_()
        self.layer2_bank.zero_()
        self.layer1_bank_mask.zero_()
        self.layer2_bank_mask.zero_()

        self.layer1_bank[: layer1_bank.shape[0]].copy_(layer1_bank.detach())
        self.layer2_bank[: layer2_bank.shape[0]].copy_(layer2_bank.detach())
        self.layer1_bank_mask[: layer1_bank.shape[0]].fill_(True)
        self.layer2_bank_mask[: layer2_bank.shape[0]].fill_(True)

    @torch.no_grad()
    def read_only_max_cosine_certainty(
        self,
        activations: torch.Tensor,
        bank: torch.Tensor,
        bank_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute detached max-cosine familiarity against valid bank rows."""
        if activations.ndim != 2 or activations.shape[1] != self.hidden_dim:
            raise ValueError(
                f"activations must have shape [batch, {self.hidden_dim}], got {tuple(activations.shape)}"
            )
        if bank.shape != (self.bank_capacity, self.hidden_dim):
            raise ValueError(
                f"bank must have shape {(self.bank_capacity, self.hidden_dim)}, got {tuple(bank.shape)}"
            )
        if bank_mask.shape != (self.bank_capacity,):
            raise ValueError(f"bank_mask must have shape {(self.bank_capacity,)}, got {tuple(bank_mask.shape)}")

        if not torch.any(bank_mask):
            return torch.zeros(activations.shape[0], device=activations.device, dtype=activations.dtype)

        valid_bank = bank[bank_mask]
        normalized_activations = F.normalize(activations.detach(), dim=-1, eps=self.eps)
        normalized_bank = F.normalize(valid_bank.detach(), dim=-1, eps=self.eps)
        similarities = normalized_activations @ normalized_bank.t()
        return similarities.max(dim=-1).values
