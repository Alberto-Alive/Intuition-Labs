"""Live-bank model components for the DIGIT Extrapolation E15 experiment."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .live_bank import LiveCircularActivationBank


@dataclass
class LiveMLPForwardResult:
    """Forward outputs needed for training and live-certainty diagnostics."""

    logits: torch.Tensor
    input_embedding: torch.Tensor
    layer1_activation: torch.Tensor
    layer2_activation: torch.Tensor
    layer1_certainty: torch.Tensor | None = None
    layer2_certainty: torch.Tensor | None = None


class LiveActivationBankMLP(nn.Module):
    """Two-layer MLP with one live circular activation bank per hidden layer."""

    def __init__(
        self,
        part_a_vocab_size: int,
        part_b_vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        *,
        bank_capacity: int = 4096,
        min_bank_occupancy: int = 410,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.part_a_vocab_size = part_a_vocab_size
        self.part_b_vocab_size = part_b_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.part_a_embedding = nn.Embedding(part_a_vocab_size, embedding_dim)
        self.part_b_embedding = nn.Embedding(part_b_vocab_size, embedding_dim)
        self.hidden1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.layer1_bank = LiveCircularActivationBank(
            capacity=bank_capacity,
            hidden_dim=hidden_dim,
            min_occupancy=min_bank_occupancy,
            eps=eps,
        )
        self.layer2_bank = LiveCircularActivationBank(
            capacity=bank_capacity,
            hidden_dim=hidden_dim,
            min_occupancy=min_bank_occupancy,
            eps=eps,
        )

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

    def forward(
        self,
        part_a: torch.Tensor,
        part_b: torch.Tensor,
        part_a_noise: torch.Tensor | None = None,
        part_b_noise: torch.Tensor | None = None,
        *,
        return_live_certainty: bool = False,
        return_read_only_certainty: bool = False,
    ) -> LiveMLPForwardResult:
        """Compute logits and optionally query live certainty from the current bank state."""
        if return_read_only_certainty and not return_live_certainty:
            return_live_certainty = True

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
        if return_live_certainty:
            layer1_certainty = self.layer1_bank.query(layer1_activation)
            layer2_certainty = self.layer2_bank.query(layer2_activation)

        return LiveMLPForwardResult(
            logits=logits,
            input_embedding=input_embedding,
            layer1_activation=layer1_activation,
            layer2_activation=layer2_activation,
            layer1_certainty=layer1_certainty,
            layer2_certainty=layer2_certainty,
        )

    @torch.no_grad()
    def update_live_banks(self, layer1_activation: torch.Tensor, layer2_activation: torch.Tensor) -> None:
        """Append the current batch activations to the live banks."""
        self.layer1_bank.update(layer1_activation)
        self.layer2_bank.update(layer2_activation)

