"""Forward-gated live-bank model components for DIGIT Extrapolation E16."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from experiments.DIGIT.Extrapolation.e15.extrapolation.live_bank import LiveCircularActivationBank


@dataclass
class GatedMLPForwardResult:
    """Forward outputs needed for E16 training and evaluation."""

    logits: torch.Tensor
    input_embedding: torch.Tensor
    layer1_pre_gate_activation: torch.Tensor
    layer2_pre_gate_activation: torch.Tensor
    layer1_certainty: torch.Tensor
    layer2_certainty: torch.Tensor
    gate_layer1: torch.Tensor
    gate_layer2: torch.Tensor


class ForwardGatedLiveBankMLP(nn.Module):
    """Two-layer MLP with certainty-squared forward gating and live banks."""

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
    ) -> GatedMLPForwardResult:
        """Compute logits with certainty-squared gating from the live banks."""
        input_embedding = self.encode_inputs_with_jitter(
            part_a,
            part_b,
            part_a_noise=part_a_noise,
            part_b_noise=part_b_noise,
        )

        layer1_pre_gate_activation = torch.relu(self.hidden1(input_embedding))
        layer1_certainty = self.layer1_bank.query(layer1_pre_gate_activation)
        gate_layer1 = self._gate_from_certainty(layer1_certainty)
        layer1_gated = layer1_pre_gate_activation * gate_layer1.unsqueeze(-1)

        layer2_pre_gate_activation = torch.relu(self.hidden2(layer1_gated))
        layer2_certainty = self.layer2_bank.query(layer2_pre_gate_activation)
        gate_layer2 = self._gate_from_certainty(layer2_certainty)
        layer2_gated = layer2_pre_gate_activation * gate_layer2.unsqueeze(-1)

        logits = self.classifier(layer2_gated)
        return GatedMLPForwardResult(
            logits=logits,
            input_embedding=input_embedding,
            layer1_pre_gate_activation=layer1_pre_gate_activation,
            layer2_pre_gate_activation=layer2_pre_gate_activation,
            layer1_certainty=layer1_certainty,
            layer2_certainty=layer2_certainty,
            gate_layer1=gate_layer1,
            gate_layer2=gate_layer2,
        )

    @torch.no_grad()
    def update_live_banks(
        self,
        layer1_pre_gate_activation: torch.Tensor,
        layer2_pre_gate_activation: torch.Tensor,
    ) -> None:
        """Append pre-gate activations to the live banks after the optimizer step."""
        self.layer1_bank.update(layer1_pre_gate_activation)
        self.layer2_bank.update(layer2_pre_gate_activation)

    @staticmethod
    def _gate_from_certainty(certainty: torch.Tensor) -> torch.Tensor:
        ones = torch.ones_like(certainty)
        return torch.where(torch.isnan(certainty), ones, certainty.square())

