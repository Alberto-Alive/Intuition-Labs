"""Losses for E30."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .models.digit import E30ForwardResult


class E30Loss(nn.Module):
    """Combined diffusion denoising + outcome loss."""

    def __init__(
        self,
        config,
        outcome_class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        if outcome_class_weights is not None:
            scaled_weights = outcome_class_weights.clone().float()
            if scaled_weights.numel() >= 3:
                scaled_weights[2] = scaled_weights[2] * float(config.failure_class_weight_scale)
            outcome_class_weights = scaled_weights
        self.outcome_ce = nn.CrossEntropyLoss(weight=outcome_class_weights, reduction="mean")

    def forward(
        self,
        outputs: E30ForwardResult,
        prim_targets: torch.Tensor,
        evidence_targets: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        del evidence_targets
        zero = outputs.outcome_logits.new_zeros(())
        outcome_loss = self.outcome_ce(outputs.outcome_logits, prim_targets[:, 3])
        diffusion_loss = outputs.diffusion_denoise_loss
        total = (
            float(self.config.outcome_loss_weight) * outcome_loss
            + float(self.config.diffusion_loss_weight) * diffusion_loss
        )
        return {
            "outcome": outcome_loss,
            "diffusion_denoise": diffusion_loss,
            "total": total,
            "probe": zero,
        }
