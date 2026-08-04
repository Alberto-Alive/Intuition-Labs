"""Deliverable 1 tests for the E15 live circular-buffer bank."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from experiments.DIGIT.Extrapolation.e15.extrapolation.live_bank import LiveCircularActivationBank


def test_bank_entries_are_detached_from_gradients() -> None:
    """Stored bank rows must remain detached from autograd after updates."""
    bank = LiveCircularActivationBank(capacity=4096, hidden_dim=3, min_occupancy=410)
    projection = nn.Linear(4, 3)
    inputs = torch.randn(2, 4)
    activations = projection(inputs)

    bank.update(activations)
    loss = activations.sum()
    loss.backward()

    assert bank.entries.requires_grad is False
    assert bank.entries.grad is None
    assert bank.entries.grad_fn is None
    assert bank.occupancy == 2
    assert torch.equal(bank.valid_mask[:2], torch.tensor([True, True]))


def test_overwrite_behavior_is_fifo_when_bank_wraps() -> None:
    """Once full, the circular buffer must overwrite the oldest entries first."""
    bank = LiveCircularActivationBank(capacity=4, hidden_dim=2, min_occupancy=1)
    rows = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
        ],
        dtype=torch.float32,
    )

    bank.update(rows[:4])
    bank.update(rows[4:])

    expected_storage = torch.tensor(
        [
            [4.0, 4.0],
            [5.0, 5.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(bank.entries, expected_storage)
    assert torch.equal(bank.valid_mask, torch.ones(4, dtype=torch.bool))
    assert bank.occupancy == 4
    assert bank.next_write_index == 2


def test_certainty_is_nan_below_minimum_occupancy_threshold() -> None:
    """Certainty must be invalid until the bank reaches 10% occupancy."""
    bank = LiveCircularActivationBank(capacity=4096, hidden_dim=3, min_occupancy=410)
    bank.update(torch.randn(409, 3))

    certainty = bank.query(torch.randn(2, 3))

    assert bank.occupancy == 409
    assert torch.isnan(certainty).all()


def test_query_uses_only_filled_slots_and_ignores_empty_ones() -> None:
    """Unfilled rows must not influence certainty, even if their tensors are non-zero."""
    bank = LiveCircularActivationBank(capacity=4, hidden_dim=3, min_occupancy=1)
    bank.update(torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32))

    bank.entries[3].copy_(torch.tensor([0.0, 1.0, 0.0]))
    assert bank.valid_mask.tolist() == [True, False, False, False]

    certainty = bank.query(torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32))

    assert certainty.shape == (1,)
    assert math.isclose(float(certainty.item()), 0.0, abs_tol=1e-6)
