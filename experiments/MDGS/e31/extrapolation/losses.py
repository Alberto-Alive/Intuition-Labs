"""Losses for E31."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models.digit import E31ForwardResult


class E31Loss(nn.Module):
    """Combined diffusion denoising, path consistency, and outcome loss."""

    def __init__(
        self,
        config,
        outcome_class_weights: torch.Tensor | None = None,
        path_failure_pos_weight: torch.Tensor | float | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.path_failure_pos_weight = path_failure_pos_weight
        if outcome_class_weights is not None:
            scaled_weights = outcome_class_weights.clone().float()
            if scaled_weights.numel() >= 3:
                scaled_weights[2] = scaled_weights[2] * float(config.failure_class_weight_scale)
            outcome_class_weights = scaled_weights
        self.outcome_ce = nn.CrossEntropyLoss(weight=outcome_class_weights, reduction="mean")

    def forward(
        self,
        outputs: E31ForwardResult,
        prim_targets: torch.Tensor,
        evidence_targets: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        del evidence_targets
        zero = outputs.outcome_logits.new_zeros(())
        outcome_loss = self.outcome_ce(outputs.outcome_logits, prim_targets[:, 3])
        diffusion_loss = outputs.diffusion_denoise_loss
        path_consistency_loss = outputs.path_consistency_loss
        path_outcome_loss = (
            self.outcome_ce(outputs.path_outcome_logits, prim_targets[:, 3])
            if outputs.path_outcome_logits is not None
            else zero
        )
        path_failure_loss = zero
        if outputs.path_failure_logit is not None:
            failure_targets = (prim_targets[:, 3] == 2).float()
            pos_weight = None
            if self.path_failure_pos_weight is not None:
                pos_weight = torch.as_tensor(
                    self.path_failure_pos_weight,
                    dtype=outputs.path_failure_logit.dtype,
                    device=outputs.path_failure_logit.device,
                )
            raw_failure_loss = F.binary_cross_entropy_with_logits(
                outputs.path_failure_logit,
                failure_targets,
                pos_weight=pos_weight,
                reduction="none",
            )
            focal_gamma = float(getattr(self.config, "path_failure_focal_gamma", 0.0))
            if focal_gamma > 0.0:
                failure_prob = torch.sigmoid(outputs.path_failure_logit)
                pt = torch.where(failure_targets > 0.5, failure_prob, 1.0 - failure_prob)
                raw_failure_loss = raw_failure_loss * (1.0 - pt).clamp_min(1e-6).pow(focal_gamma)
            path_failure_loss = raw_failure_loss.mean()
        total = (
            float(self.config.outcome_loss_weight) * outcome_loss
            + float(self.config.diffusion_loss_weight) * diffusion_loss
            + float(self.config.path_consistency_loss_weight) * path_consistency_loss
            + float(self.config.path_outcome_loss_weight) * path_outcome_loss
            + float(self.config.path_failure_loss_weight) * path_failure_loss
        )
        return {
            "outcome": outcome_loss,
            "diffusion_denoise": diffusion_loss,
            "path_consistency": path_consistency_loss,
            "path_outcome": path_outcome_loss,
            "path_failure": path_failure_loss,
            "total": total,
            "probe": zero,
        }


E30Loss = E31Loss
