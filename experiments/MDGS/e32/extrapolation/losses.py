"""Losses for E32."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models.digit import E31ForwardResult


def build_risk_targets(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    risk_target_mode: str,
) -> torch.Tensor:
    """Build binary risk labels from ground truth and clean/base predictions."""

    labels = labels.detach().reshape(-1).long()
    predictions = predictions.detach().reshape(-1).long()
    mode = str(risk_target_mode).strip().lower()
    if mode == "incorrect_only":
        mode = "incorrect"
    if mode == "incorrect":
        return (predictions != labels).float()
    if mode == "uncertain_label":
        return (labels == 1).float()
    if mode == "incorrect_or_uncertain":
        return torch.maximum(
            (predictions != labels).float(),
            (labels == 1).float(),
        )
    raise ValueError(f"Unknown risk_target_mode {risk_target_mode!r}")


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
        labels = prim_targets[:, 3]
        outcome_loss = self.outcome_ce(outputs.outcome_logits, labels)
        diffusion_loss = outputs.diffusion_denoise_loss
        path_consistency_loss = outputs.path_consistency_loss
        path_outcome_loss = (
            self.outcome_ce(outputs.path_outcome_logits, labels)
            if outputs.path_outcome_logits is not None
            else zero
        )
        path_failure_loss = zero
        if outputs.path_failure_logit is not None:
            failure_targets = (labels == 2).float()
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
        clean_logits = getattr(outputs, "base_outcome_logits", outputs.outcome_logits)
        clean_predictions = clean_logits.argmax(dim=-1)
        risk_target_mode = str(getattr(self.config, "risk_target_mode", "incorrect_only"))
        risk_targets = build_risk_targets(labels, clean_predictions, risk_target_mode)

        risk_loss = zero
        if outputs.risk_logit is not None:
            risk_loss = F.binary_cross_entropy_with_logits(outputs.risk_logit, risk_targets)

        profile_loss = zero
        if outputs.disturbance_profile is not None:
            profile = outputs.disturbance_profile.profile
            low_risk_mask = (1.0 - risk_targets).to(dtype=profile.dtype, device=profile.device)
            profile_energy = profile.pow(2).mean(dim=-1)
            if bool(low_risk_mask.any()):
                profile_loss = (profile_energy * low_risk_mask).sum() / low_risk_mask.sum().clamp_min(1.0)
            profile_loss = profile_loss + outputs.disturbance_profile.regularization_loss
        total = (
            float(self.config.outcome_loss_weight) * outcome_loss
            + float(self.config.diffusion_loss_weight) * diffusion_loss
            + float(self.config.path_consistency_loss_weight) * path_consistency_loss
            + float(self.config.path_outcome_loss_weight) * path_outcome_loss
            + float(self.config.path_failure_loss_weight) * path_failure_loss
            + float(getattr(self.config, "profile_loss_weight", 0.0)) * profile_loss
            + float(getattr(self.config, "risk_loss_weight", 0.0)) * risk_loss
        )
        return {
            "outcome": outcome_loss,
            "diffusion_denoise": diffusion_loss,
            "path_consistency": path_consistency_loss,
            "path_outcome": path_outcome_loss,
            "path_failure": path_failure_loss,
            "profile": profile_loss,
            "risk": risk_loss,
            "total": total,
            "probe": zero,
        }


E30Loss = E31Loss
E32Loss = E31Loss
