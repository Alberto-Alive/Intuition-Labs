from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extrapolation.models import LocalSupportState


def test_local_support_updates_with_exact_ema_from_soft_activation_mass() -> None:
    state = LocalSupportState(num_neurons=3, decay=0.8)
    activations = torch.tensor(
        [
            [[-2.0, 0.0, 2.0], [1.0, -1.0, 0.5]],
            [[0.0, 3.0, -0.5], [-4.0, 0.25, 0.75]],
        ],
        requires_grad=True,
    )

    first_mass = torch.sigmoid(activations.detach()).mean(dim=(0, 1))
    first_expected = 0.2 * first_mass
    updated = state.update(activations)

    assert torch.allclose(updated, first_expected, atol=1e-6)
    assert torch.allclose(state.support, first_expected, atol=1e-6)
    assert state.num_updates.item() == 1

    next_activations = torch.tensor(
        [
            [[1.5, -3.0, 0.0], [0.5, 0.5, 0.5]],
            [[-1.0, 2.0, 4.0], [0.0, -0.5, -2.5]],
        ],
        requires_grad=True,
    )
    second_mass = torch.sigmoid(next_activations.detach()).mean(dim=(0, 1))
    second_expected = 0.8 * first_expected + 0.2 * second_mass
    updated = state.update(next_activations)

    assert torch.allclose(updated, second_expected, atol=1e-6)
    assert torch.allclose(state.support, second_expected, atol=1e-6)
    assert state.num_updates.item() == 2


def test_local_support_is_buffer_only_and_has_no_gradient_path() -> None:
    state = LocalSupportState(num_neurons=2, decay=0.9)
    activations = torch.tensor([[1.0, -1.0], [0.5, -0.5]], requires_grad=True)

    updated = state.update(activations)
    objective = activations.square().sum()
    objective.backward()

    parameter_names = {name for name, _ in state.named_parameters()}
    buffer_names = {name for name, _ in state.named_buffers()}

    assert "support" not in parameter_names
    assert "support" in buffer_names
    assert state.support.requires_grad is False
    assert updated.requires_grad is False
    assert updated.grad_fn is None
    assert state.support.grad is None
    assert activations.grad is not None
