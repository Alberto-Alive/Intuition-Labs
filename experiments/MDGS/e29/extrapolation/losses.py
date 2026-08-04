"""Losses for E29."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models.digit import E29ForwardResult


class E29Loss(nn.Module):
    """Combined denoising + geometry-outcome loss, with optional disconnected probe supervision."""

    def __init__(
        self,
        config,
        outcome_class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.outcome_ce = nn.CrossEntropyLoss(weight=outcome_class_weights, reduction="mean")
        self.probe_regression = nn.SmoothL1Loss(reduction="mean")

    def forward(
        self,
        outputs: E29ForwardResult,
        prim_targets: torch.Tensor,
        evidence_targets: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        zero = outputs.outcome_logits.new_zeros(())
        outcome_loss = self.outcome_ce(outputs.outcome_logits, prim_targets[:, 3])
        diffusion_loss = outputs.diffusion_denoise_loss

        probe_loss = zero
        if (
            outputs.diagnostic_probes is not None
            and evidence_targets is not None
            and float(self.config.probe_loss_weight) != 0.0
        ):
            order_target = (evidence_targets[:, 2] - evidence_targets[:, 3]).clamp(-1.0, 1.0)
            boundary_target = evidence_targets[:, 4]
            support_target = evidence_targets[:, 1]
            probe_loss = (
                self.probe_regression(outputs.diagnostic_probes.order, order_target)
                + self.probe_regression(outputs.diagnostic_probes.boundary, boundary_target)
                + self.probe_regression(outputs.diagnostic_probes.support, support_target)
            ) / 3.0

        total = (
            float(self.config.outcome_loss_weight) * outcome_loss
            + float(self.config.diffusion_loss_weight) * diffusion_loss
            + float(self.config.probe_loss_weight) * probe_loss
        )
        return {
            "outcome": outcome_loss,
            "diffusion_denoise": diffusion_loss,
            "probe": probe_loss,
            "total": total,
        }

