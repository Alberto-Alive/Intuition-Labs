"""Backward-modulated model components for DIGIT Extrapolation E17."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from experiments.DIGIT.Extrapolation.e16.extrapolation.models import (
    ForwardGatedLiveBankMLP,
    GatedMLPForwardResult,
)


@dataclass(frozen=True)
class GradientModulationWeights:
    """Detached per-example gradient modulation weights for one forward pass."""

    layer1: torch.Tensor
    layer2: torch.Tensor


class BackwardModulatedLiveBankMLP(ForwardGatedLiveBankMLP):
    """E16 forward path plus certainty-modulated backward hooks."""

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
        alpha: float = 10.0,
        beta: float = 0.1,
        neutral_certainty: float = 0.5,
    ) -> None:
        super().__init__(
            part_a_vocab_size=part_a_vocab_size,
            part_b_vocab_size=part_b_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            bank_capacity=bank_capacity,
            min_bank_occupancy=min_bank_occupancy,
            eps=eps,
        )
        self.alpha = alpha
        self.beta = beta
        self.neutral_certainty = neutral_certainty
        self._active_hook_handles: list[torch.utils.hooks.RemovableHandle] = []

    def build_modulation_weights(
        self,
        losses: torch.Tensor,
        outputs: GatedMLPForwardResult,
    ) -> GradientModulationWeights:
        """Build detached per-example modulation weights for both hidden layers."""
        detached_losses = losses.detach()
        return GradientModulationWeights(
            layer1=self._scale_from_loss_and_certainty(detached_losses, outputs.layer1_certainty),
            layer2=self._scale_from_loss_and_certainty(detached_losses, outputs.layer2_certainty),
        )

    def attach_gradient_modulation_hooks(
        self,
        outputs: GatedMLPForwardResult,
        modulation_weights: GradientModulationWeights,
    ) -> None:
        """Attach per-example scaling hooks to the pre-gate activation tensors."""
        self.clear_gradient_modulation_hooks()
        self._active_hook_handles = [
            outputs.layer1_pre_gate_activation.register_hook(
                lambda grad: grad * modulation_weights.layer1.unsqueeze(-1)
            ),
            outputs.layer2_pre_gate_activation.register_hook(
                lambda grad: grad * modulation_weights.layer2.unsqueeze(-1)
            ),
        ]

    def clear_gradient_modulation_hooks(self) -> None:
        """Remove any active backward hooks from the previous step."""
        for handle in self._active_hook_handles:
            handle.remove()
        self._active_hook_handles.clear()

    def _scale_from_loss_and_certainty(self, losses: torch.Tensor, certainty: torch.Tensor) -> torch.Tensor:
        detached_certainty = certainty.detach()
        if torch.isnan(detached_certainty).any():
            normalized_scale = torch.ones_like(detached_certainty)
        else:
            raw_scale = self.alpha * detached_certainty + self.beta * (1.0 - detached_certainty)
            normalized_scale = raw_scale / raw_scale.mean()
        return losses * normalized_scale
