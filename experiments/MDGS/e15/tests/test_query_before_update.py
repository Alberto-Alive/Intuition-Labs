"""Isolated invariant test for E15 live-bank query-before-update ordering."""

from __future__ import annotations

import math

import torch


class MinimalLiveCircularBank:
    """Minimal live bank used only to verify query-before-update ordering."""

    def __init__(self, capacity: int, hidden_dim: int, min_occupancy: int) -> None:
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self.min_occupancy = min_occupancy
        self.bank = torch.zeros(capacity, hidden_dim, dtype=torch.float32)
        self.valid_mask = torch.zeros(capacity, dtype=torch.bool)
        self.write_pointer = 0
        self.valid_count = 0

    def query(self, activations: torch.Tensor) -> torch.Tensor:
        """Return max cosine certainty over valid entries or NaN if underfilled."""
        if self.valid_count < self.min_occupancy:
            return torch.full((activations.shape[0],), float("nan"), dtype=torch.float32)

        valid_bank = self.bank[self.valid_mask]
        normalized_activations = self._normalize(activations)
        normalized_bank = self._normalize(valid_bank)
        similarities = normalized_activations @ normalized_bank.T
        return similarities.max(dim=-1).values

    def update(self, activations: torch.Tensor) -> None:
        """Append detached activations into the circular buffer."""
        detached = activations.detach()
        for row in detached:
            self.bank[self.write_pointer] = row
            self.valid_mask[self.write_pointer] = True
            self.write_pointer = (self.write_pointer + 1) % self.capacity
            self.valid_count = min(self.valid_count + 1, self.capacity)

    @staticmethod
    def _normalize(values: torch.Tensor) -> torch.Tensor:
        return values / values.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def test_query_happens_before_current_batch_is_written() -> None:
    """The current batch must not see itself during certainty lookup."""
    bank = MinimalLiveCircularBank(capacity=4, hidden_dim=3, min_occupancy=1)

    historical_batch = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    current_batch = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)

    bank.update(historical_batch)

    certainty_before_update = bank.query(current_batch)
    assert certainty_before_update.shape == (1,)
    assert math.isclose(float(certainty_before_update.item()), 0.0, abs_tol=1e-6)

    bank.update(current_batch)
    certainty_after_update = bank.query(current_batch)
    assert certainty_after_update.shape == (1,)
    assert math.isclose(float(certainty_after_update.item()), 1.0, abs_tol=1e-6)
