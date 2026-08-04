"""Loss functions for the DIGIT Extrapolation model."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .executor_schema import executor_feature_column
from .models.bottleneck import PrimitiveOutput


class DIGITLoss(nn.Module):
    """Combined loss for the entropy-gating smoke test."""

    def __init__(self, config, pad_idx: int = 0, class_weights: Dict[str, torch.Tensor] | None = None):
        super().__init__()
        self.config = config
        self.pad_idx = pad_idx

        if class_weights:
            self.trajectory_ce = nn.CrossEntropyLoss(
                weight=class_weights.get("trajectory_shape"), reduction="mean"
            )
            self.pattern_ce = nn.CrossEntropyLoss(
                weight=class_weights.get("attention_pattern"), reduction="mean"
            )
            self.confidence_ce = nn.CrossEntropyLoss(
                weight=class_weights.get("confidence"), reduction="mean"
            )
            self.outcome_ce = nn.CrossEntropyLoss(
                weight=class_weights.get("outcome"), reduction="mean"
            )
        else:
            self.trajectory_ce = nn.CrossEntropyLoss(reduction="mean")
            self.pattern_ce = nn.CrossEntropyLoss(reduction="mean")
            self.confidence_ce = nn.CrossEntropyLoss(reduction="mean")
            self.outcome_ce = nn.CrossEntropyLoss(reduction="mean")

        self.gen_ce = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="mean")

    def forward(
        self,
        decoder_logits: torch.Tensor | None,
        primitives: PrimitiveOutput,
        target_ids: torch.Tensor,
        prim_targets: torch.Tensor,
        executor_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        losses = {}

        trajectory_loss = self.trajectory_ce(primitives.trajectory_shape_logits, prim_targets[:, 0])
        pattern_loss = self.pattern_ce(primitives.attention_pattern_logits, prim_targets[:, 1])
        confidence_loss = self.confidence_ce(primitives.confidence_logits, prim_targets[:, 2])
        outcome_loss = self.outcome_ce(primitives.outcome_logits, prim_targets[:, 3])
        total_weight = (
            self.config.primitive_trajectory_loss_weight
            + self.config.primitive_pattern_loss_weight
            + self.config.primitive_confidence_loss_weight
            + self.config.primitive_outcome_loss_weight
        )
        primitive_loss = (
            self.config.primitive_trajectory_loss_weight * trajectory_loss
            + self.config.primitive_pattern_loss_weight * pattern_loss
            + self.config.primitive_confidence_loss_weight * confidence_loss
            + self.config.primitive_outcome_loss_weight * outcome_loss
        ) / total_weight
        losses["primitive"] = primitive_loss

        zero = primitives.trajectory_shape_logits.new_tensor(0.0)
        if decoder_logits is None or self.config.lambda_generation == 0.0:
            generation_loss = zero
        else:
            gen_targets = target_ids[:, 1:]
            batch_size, target_len, vocab_size = decoder_logits.shape
            generation_loss = self.gen_ce(
                decoder_logits.reshape(batch_size * target_len, vocab_size),
                gen_targets.reshape(batch_size * target_len),
            )
        losses["generation"] = generation_loss

        policy_loss = (
            self._policy_violation_penalty(primitives, executor_features)
            if self.config.lambda_policy != 0.0
            else zero
        )
        losses["policy"] = policy_loss

        leakage_loss = (
            self._leakage_penalty(decoder_logits, executor_features)
            if decoder_logits is not None and self.config.lambda_leakage != 0.0
            else zero
        )
        losses["leakage"] = leakage_loss

        abstention_loss = (
            self._abstention_encouragement(primitives, executor_features)
            if self.config.lambda_abstention != 0.0
            else zero
        )
        losses["abstention"] = abstention_loss

        losses["total"] = (
            self.config.lambda_primitive * primitive_loss
            + self.config.lambda_generation * generation_loss
            + self.config.lambda_policy * policy_loss
            + self.config.lambda_leakage * leakage_loss
            + self.config.lambda_abstention * abstention_loss
        )

        return losses

    def _policy_violation_penalty(
        self,
        primitives: PrimitiveOutput,
        executor_features: torch.Tensor,
    ) -> torch.Tensor:
        mean_entropy = executor_feature_column(executor_features, "mean_entropy")
        std_entropy = executor_feature_column(executor_features, "std_entropy")
        entropy_range = executor_feature_column(executor_features, "entropy_range")
        low_margin_flag = executor_feature_column(executor_features, "low_margin_flag")
        unstable_flag = executor_feature_column(executor_features, "unstable_flag")

        confidence_probs = F.softmax(primitives.confidence_logits, dim=-1)
        pattern_probs = F.softmax(primitives.attention_pattern_logits, dim=-1)
        outcome_probs = F.softmax(primitives.outcome_logits, dim=-1)

        high_confidence_prob = confidence_probs[:, 2]
        focused_prob = pattern_probs[:, 0]
        success_prob = outcome_probs[:, 0]

        unstable_mask = ((entropy_range >= 0.35) | (std_entropy >= 0.16)).float()
        penalty_high_confidence = unstable_mask * high_confidence_prob
        penalty_focused = (mean_entropy >= 0.65).float() * focused_prob
        unsupported_success_mask = torch.clamp(low_margin_flag + unstable_flag, max=1.0)
        penalty_success = unsupported_success_mask * success_prob

        return (
            penalty_high_confidence.mean()
            + penalty_focused.mean()
            + penalty_success.mean()
        ) / 3.0

    def _leakage_penalty(
        self,
        decoder_logits: torch.Tensor,
        executor_features: torch.Tensor,
    ) -> torch.Tensor:
        decoder_summary = decoder_logits.mean(dim=1)
        k = min(32, decoder_summary.size(-1))
        decoder_reduced = decoder_summary[:, :k]

        d = decoder_reduced - decoder_reduced.mean(dim=0, keepdim=True)
        e = executor_features - executor_features.mean(dim=0, keepdim=True)

        d_std = d.std(dim=0, keepdim=True).clamp(min=1e-6)
        e_std = e.std(dim=0, keepdim=True).clamp(min=1e-6)

        d_norm = d / d_std
        e_norm = e / e_std
        corr = (d_norm.T @ e_norm) / max(d_norm.size(0), 1)
        return (corr ** 2).mean()

    def _abstention_encouragement(
        self,
        primitives: PrimitiveOutput,
        executor_features: torch.Tensor,
    ) -> torch.Tensor:
        mean_entropy = executor_feature_column(executor_features, "mean_entropy")
        agreement = executor_feature_column(executor_features, "agreement")
        outcome_probs = F.softmax(primitives.outcome_logits, dim=-1)
        success_prob = outcome_probs[:, 0]

        low_certainty_mask = (
            (mean_entropy >= 0.60)
            | (agreement < self.config.trace_low_agreement_threshold)
        ).float()
        return (low_certainty_mask * success_prob).mean()
