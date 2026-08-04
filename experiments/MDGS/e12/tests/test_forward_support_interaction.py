from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extrapolation.models import ForwardSupportInteraction


def test_full_relational_operator_matches_frozen_closed_form() -> None:
    interaction = ForwardSupportInteraction(interaction_mode="full")
    with torch.no_grad():
        interaction.vector.copy_(torch.tensor([0.5, -1.0, 2.0]))

    activations = torch.tensor([[1.0, -0.5], [0.25, 0.75]])
    support = torch.tensor([[0.8, 0.3], [0.2, 0.9]])
    context_support = torch.tensor([0.4, 0.7])

    output = interaction(activations, support, context_support)
    expected_gate = torch.sigmoid(0.5 * activations - support + 2.0 * context_support.unsqueeze(-1))
    expected_output = activations * expected_gate

    assert torch.allclose(output, expected_output, atol=1e-6)


def test_scalar_gate_a7_is_exact_elementwise_product() -> None:
    interaction = ForwardSupportInteraction(interaction_mode="a7")
    activations = torch.tensor([[1.0, -0.5], [0.25, 0.75]])
    support = torch.tensor([[0.8, 0.3], [0.2, 0.9]])
    context_support = torch.tensor([0.4, 0.7])

    output = interaction(activations, support, context_support)

    assert torch.allclose(output, activations * support, atol=1e-6)


def test_neutral_interaction_is_identity() -> None:
    interaction = ForwardSupportInteraction(interaction_mode="neutral")
    activations = torch.tensor([[1.0, -0.5], [0.25, 0.75]])
    support = torch.tensor([[0.8, 0.3], [0.2, 0.9]])
    context_support = torch.tensor([0.4, 0.7])

    output = interaction(activations, support, context_support)

    assert torch.allclose(output, activations, atol=1e-6)
