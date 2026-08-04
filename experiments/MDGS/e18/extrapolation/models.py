"""Targeted backward-modulated model components for DIGIT Extrapolation E18."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from experiments.DIGIT.Extrapolation.e16.extrapolation.models import (
    ForwardGatedLiveBankMLP,
    GatedMLPForwardResult,
)


@dataclass(frozen=True)
class TargetedGradientModulationWeights:
    """Detached per-example gradient row weights for one forward pass."""

    layer1: torch.Tensor
    layer2: torch.Tensor
    layer1_modulated_fraction: float
    layer2_modulated_fraction: float


class TargetedBackwardModulatedLiveBankMLP(ForwardGatedLiveBankMLP):
    """E16 forward path plus threshold-gated backward modulation."""

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
        certainty_threshold: float = 0.95,
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
        self.certainty_threshold = certainty_threshold
        self._active_hook_handles: list[torch.utils.hooks.RemovableHandle] = []

    def build_modulation_weights(
        self,
        losses: torch.Tensor,
        outputs: GatedMLPForwardResult,
    ) -> TargetedGradientModulationWeights:
        """Build detached per-example row weights for both hidden layers."""
        detached_losses = losses.detach()
        layer1_weights, layer1_fraction = self._weights_and_fraction(
            detached_losses,
            outputs.layer1_certainty,
        )
        layer2_weights, layer2_fraction = self._weights_and_fraction(
            detached_losses,
            outputs.layer2_certainty,
        )
        return TargetedGradientModulationWeights(
            layer1=layer1_weights,
            layer2=layer2_weights,
            layer1_modulated_fraction=layer1_fraction,
            layer2_modulated_fraction=layer2_fraction,
        )

    def attach_gradient_modulation_hooks(
        self,
        outputs: GatedMLPForwardResult,
        modulation_weights: TargetedGradientModulationWeights,
    ) -> None:
        """Attach threshold-gated scaling hooks to the pre-gate activation tensors."""
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

    def _weights_and_fraction(
        self,
        losses: torch.Tensor,
        certainty: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        detached_certainty = certainty.detach()
        weights = torch.ones_like(losses)
        low_mask = (~torch.isnan(detached_certainty)) & (detached_certainty < self.certainty_threshold)
        if int(low_mask.sum().item()) == 0:
            return weights, 0.0

        low_certainty = detached_certainty[low_mask]
        raw_scale = self.alpha * low_certainty + self.beta * (1.0 - low_certainty)
        normalized_raw_scale = raw_scale / raw_scale.mean()
        weights[low_mask] = losses[low_mask] * normalized_raw_scale
        return weights, float(low_mask.to(dtype=torch.float32).mean().item())

