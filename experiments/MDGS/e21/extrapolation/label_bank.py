"""Aligned activation, label, and level buffers for DIGIT Extrapolation E21."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FrozenBankMetadata:
    """Frozen bank-aligned metadata buffers after Phase 1."""

    labels: torch.Tensor
    levels: torch.Tensor
    valid_mask: torch.Tensor


class ActivationMetadataBuffer:
    """Minimal aligned storage for labels and levels written with bank rows."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.labels = torch.full((capacity,), -1, dtype=torch.long)
        self.levels = torch.full((capacity,), -1, dtype=torch.long)
        self.valid_mask = torch.zeros(capacity, dtype=torch.bool)
        self.write_pointer = 0
        self.valid_count = 0

    @torch.no_grad()
    def update(self, labels: torch.Tensor, levels: torch.Tensor) -> None:
        if labels.ndim != 1 or levels.ndim != 1 or labels.shape[0] != levels.shape[0]:
            raise ValueError("labels and levels must be aligned one-dimensional tensors")

        start = self.write_pointer
        for offset, (label, level) in enumerate(zip(labels.detach(), levels.detach(), strict=True)):
            slot = (start + offset) % self.capacity
            self.labels[slot] = int(label.item())
            self.levels[slot] = int(level.item())
            self.valid_mask[slot] = True

        self.write_pointer = (start + labels.shape[0]) % self.capacity
        self.valid_count = min(self.capacity, self.valid_count + labels.shape[0])

    def freeze(self) -> FrozenBankMetadata:
        return FrozenBankMetadata(
            labels=self.labels.detach().clone(),
            levels=self.levels.detach().clone(),
            valid_mask=self.valid_mask.detach().clone(),
        )
