"""Partitioned inference-time bank for DIGIT Extrapolation E20."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.DIGIT.Extrapolation.e15.extrapolation.live_bank import LiveCircularActivationBank


class PartitionedActivationBank(nn.Module):
    """Frozen training partition plus live inference partition."""

    def __init__(
        self,
        training_entries: torch.Tensor,
        training_valid_mask: torch.Tensor,
        *,
        inference_capacity: int,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if training_entries.ndim != 2:
            raise ValueError("training_entries must have shape [num_entries, hidden_dim]")
        if training_valid_mask.ndim != 1 or training_valid_mask.shape[0] != training_entries.shape[0]:
            raise ValueError("training_valid_mask must have shape [num_entries]")
        if inference_capacity <= 0:
            raise ValueError("inference_capacity must be positive")

        self.hidden_dim = int(training_entries.shape[1])
        self.eps = eps

        self.register_buffer("training_entries", training_entries.detach().clone())
        self.register_buffer("training_valid_mask", training_valid_mask.detach().clone())
        self.inference_partition = LiveCircularActivationBank(
            capacity=inference_capacity,
            hidden_dim=self.hidden_dim,
            min_occupancy=1,
            eps=eps,
        )

    @classmethod
    def from_live_bank(
        cls,
        live_bank: LiveCircularActivationBank,
        *,
        inference_capacity: int,
    ) -> "PartitionedActivationBank":
        """Build a partitioned bank from a trained live bank snapshot."""
        return cls(
            training_entries=live_bank.entries,
            training_valid_mask=live_bank.valid_mask,
            inference_capacity=inference_capacity,
            eps=live_bank.eps,
        )

    @property
    def training_occupancy(self) -> int:
        return int(self.training_valid_mask.sum().item())

    @property
    def inference_occupancy(self) -> int:
        return self.inference_partition.occupancy

    @property
    def inference_write_pointer(self) -> int:
        return self.inference_partition.next_write_index

    @torch.no_grad()
    def update_training_partition(self, _activations: torch.Tensor) -> None:
        """Reject any attempted inference-time write to the frozen training partition."""
        raise RuntimeError("training partition is frozen and cannot be written during inference")

    @torch.no_grad()
    def update_inference_partition(self, activations: torch.Tensor) -> None:
        """Append detached activations into the live inference partition only."""
        self.inference_partition.update(activations)

    @torch.no_grad()
    def query(self, activations: torch.Tensor) -> torch.Tensor:
        """Return max cosine over the union of training and inference partitions."""
        if activations.ndim != 2 or activations.shape[1] != self.hidden_dim:
            raise ValueError(
                f"activations must have shape [batch, {self.hidden_dim}], got {tuple(activations.shape)}"
            )

        valid_training = self.training_entries[self.training_valid_mask]
        valid_inference = self.inference_partition.entries[self.inference_partition.valid_mask]
        valid_entries = torch.cat((valid_training, valid_inference), dim=0)
        if valid_entries.shape[0] == 0:
            return torch.full(
                (activations.shape[0],),
                float("nan"),
                device=activations.device,
                dtype=activations.dtype,
            )

        normalized_activations = F.normalize(activations.detach(), dim=-1, eps=self.eps)
        normalized_entries = F.normalize(valid_entries.detach().to(activations.device), dim=-1, eps=self.eps)
        similarities = normalized_activations @ normalized_entries.t()
        return similarities.max(dim=-1).values
