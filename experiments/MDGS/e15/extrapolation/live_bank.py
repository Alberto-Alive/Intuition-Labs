"""Live circular-buffer activation bank for DIGIT Extrapolation E15."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LiveCircularActivationBank(nn.Module):
    """Fixed-size FIFO activation bank with detached writes and masked queries."""

    def __init__(
        self,
        capacity: int = 4096,
        hidden_dim: int = 32,
        *,
        min_occupancy: int | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        resolved_min_occupancy = math.ceil(0.1 * capacity) if min_occupancy is None else min_occupancy
        if resolved_min_occupancy <= 0 or resolved_min_occupancy > capacity:
            raise ValueError("min_occupancy must be in [1, capacity]")

        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self.min_occupancy = resolved_min_occupancy
        self.eps = eps

        self.register_buffer("entries", torch.zeros(capacity, hidden_dim))
        self.register_buffer("valid_mask", torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer("write_pointer", torch.zeros((), dtype=torch.long))
        self.register_buffer("valid_count", torch.zeros((), dtype=torch.long))

    @property
    def occupancy(self) -> int:
        """Current number of valid bank entries."""
        return int(self.valid_count.item())

    @property
    def next_write_index(self) -> int:
        """Next slot that will be overwritten by an update."""
        return int(self.write_pointer.item())

    @torch.no_grad()
    def update(self, activations: torch.Tensor) -> None:
        """Append detached activation rows into the circular buffer."""
        self._validate_activations(activations)
        detached = activations.detach().to(device=self.entries.device, dtype=self.entries.dtype)
        start_index = self.next_write_index

        for offset, row in enumerate(detached):
            slot = (start_index + offset) % self.capacity
            self.entries[slot].copy_(row)
            self.valid_mask[slot] = True

        self.write_pointer.fill_((start_index + detached.shape[0]) % self.capacity)
        self.valid_count.fill_(min(self.capacity, self.occupancy + detached.shape[0]))

    @torch.no_grad()
    def query(self, activations: torch.Tensor) -> torch.Tensor:
        """Return max-cosine certainty or NaN if occupancy is below threshold."""
        self._validate_activations(activations)
        if self.occupancy < self.min_occupancy:
            return torch.full(
                (activations.shape[0],),
                float("nan"),
                device=activations.device,
                dtype=activations.dtype,
            )

        valid_entries = self.entries[self.valid_mask]
        normalized_activations = F.normalize(activations.detach(), dim=-1, eps=self.eps)
        normalized_entries = F.normalize(valid_entries.detach(), dim=-1, eps=self.eps)
        similarities = normalized_activations @ normalized_entries.t()
        return similarities.max(dim=-1).values

    def _validate_activations(self, activations: torch.Tensor) -> None:
        if activations.ndim != 2 or activations.shape[1] != self.hidden_dim:
            raise ValueError(
                f"activations must have shape [batch, {self.hidden_dim}], got {tuple(activations.shape)}"
            )

