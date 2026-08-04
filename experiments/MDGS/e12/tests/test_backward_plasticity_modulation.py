from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extrapolation.models import BackwardPlasticityModulator


def test_backward_plasticity_matches_frozen_closed_form() -> None:
    modulator = BackwardPlasticityModulator(alpha=10.0, beta=0.1)
    error = torch.tensor([[1.0, -2.0], [0.5, -0.25]])
    support = torch.tensor([[0.0, 0.5], [0.9, 1.0]])

    scaled = modulator.modulate_error(error, support)
    expected = error * (10.0 * support + 0.1 * (1.0 - support))

    assert torch.allclose(scaled, expected, atol=1e-6)
    assert "alpha" not in {name for name, _ in modulator.named_parameters()}
    assert "beta" not in {name for name, _ in modulator.named_parameters()}


def test_backward_plasticity_scales_gradients_without_touching_forward_or_support() -> None:
    modulator = BackwardPlasticityModulator(alpha=10.0, beta=0.1)
    activations = torch.tensor([[1.0, -2.0], [3.0, 4.0]], requires_grad=True)
    support = torch.tensor([[0.0, 1.0], [0.5, 0.25]], requires_grad=True)

    wrapped = modulator.modulate_activation_gradient(activations, support)
    loss = wrapped.sum()
    loss.backward()

    expected_grad = 10.0 * support.detach() + 0.1 * (1.0 - support.detach())

    assert torch.allclose(wrapped.detach(), activations.detach(), atol=1e-6)
    assert torch.allclose(activations.grad, expected_grad, atol=1e-6)
    assert support.grad is None
