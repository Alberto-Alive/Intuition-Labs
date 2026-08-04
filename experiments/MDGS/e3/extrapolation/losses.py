"""Loss functions for the DIGIT Extrapolation model."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .executor_schema import executor_feature_column
from .models.bottleneck import PrimitiveOutput


def _success_guard_target_from_evidence_targets(evidence_targets: torch.Tensor) -> torch.Tensor:
    stability_target = evidence_targets[:, 0]
    support_target = evidence_targets[:, 1]
    success_target = evidence_targets[:, 2]
    guard_anchor = torch.clamp(0.65 * stability_target + 0.35 * support_target, min=0.0, max=1.0)
    return torch.minimum(success_target, guard_anchor)


class DIGITLoss(nn.Module):
    """Combined loss for the cooperative-evidence E3 variant."""

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
                weight=class_weights.get("outcome"),
                reduction="mean",
                label_smoothing=float(config.primitive_outcome_label_smoothing),
            )
        else:
            self.trajectory_ce = nn.CrossEntropyLoss(reduction="mean")
            self.pattern_ce = nn.CrossEntropyLoss(reduction="mean")
            self.confidence_ce = nn.CrossEntropyLoss(reduction="mean")
            self.outcome_ce = nn.CrossEntropyLoss(
                reduction="mean",
                label_smoothing=float(config.primitive_outcome_label_smoothing),
            )

        self.gen_ce = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="mean")
        self.evidence_regression = nn.SmoothL1Loss(reduction="mean")

    def forward(
        self,
        decoder_logits: torch.Tensor | None,
        primitives: PrimitiveOutput,
        target_ids: torch.Tensor,
        prim_targets: torch.Tensor,
        executor_features: torch.Tensor,
        evidence_targets: torch.Tensor | None = None,
        perturbed_primitives: PrimitiveOutput | None = None,
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

        evidence_target_loss = (
            self._evidence_target_loss(primitives, evidence_targets)
            if evidence_targets is not None and self.config.lambda_evidence_targets != 0.0
            else zero
        )
        losses["evidence_targets"] = evidence_target_loss

        consistency_loss = (
            self._consistency_loss(primitives)
            if self.config.lambda_consistency != 0.0
            else zero
        )
        losses["consistency"] = consistency_loss

        evidence_mono_loss = (
            self._evidence_monotonicity_loss(primitives, perturbed_primitives)
            if perturbed_primitives is not None and self.config.lambda_evidence_mono != 0.0
            else zero
        )
        losses["evidence_mono"] = evidence_mono_loss

        usage_loss = (
            self._usage_loss(primitives)
            if self.config.lambda_usage != 0.0
            else zero
        )
        losses["usage"] = usage_loss

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
            + self.config.lambda_evidence_targets * evidence_target_loss
            + self.config.lambda_consistency * consistency_loss
            + self.config.lambda_evidence_mono * evidence_mono_loss
            + self.config.lambda_usage * usage_loss
            + self.config.lambda_policy * policy_loss
            + self.config.lambda_leakage * leakage_loss
            + self.config.lambda_abstention * abstention_loss
        )

        return losses

    def _certificate_stack(self, primitives: PrimitiveOutput) -> torch.Tensor:
        return torch.stack(
            [
                primitives.stability_cert,
                primitives.support_cert,
                primitives.success_cert,
                primitives.failure_cert,
                primitives.ambiguity_cert,
            ],
            dim=-1,
        )

    def _evidence_target_loss(self, primitives: PrimitiveOutput, evidence_targets: torch.Tensor) -> torch.Tensor:
        cert_loss = self.evidence_regression(self._certificate_stack(primitives), evidence_targets)
        success_guard = getattr(primitives, "success_guard", None)
        if success_guard is None:
            return cert_loss
        guard_target = _success_guard_target_from_evidence_targets(evidence_targets)
        guard_loss = self.evidence_regression(success_guard, guard_target)
        return cert_loss + 0.5 * guard_loss

    def _consistency_loss(self, primitives: PrimitiveOutput) -> torch.Tensor:
        trajectory_probs = F.softmax(primitives.trajectory_shape_logits, dim=-1).detach()
        pattern_probs = F.softmax(primitives.attention_pattern_logits, dim=-1).detach()
        confidence_probs = F.softmax(primitives.confidence_logits, dim=-1)
        outcome_probs = F.softmax(primitives.outcome_logits, dim=-1)

        stability = primitives.stability_cert
        support = primitives.support_cert
        success_guard = getattr(primitives, "success_guard", None)
        success_base = getattr(primitives, "success_base", primitives.success_cert)
        success_proposal = getattr(primitives, "success_proposal", primitives.success_cert)
        success = primitives.success_cert
        failure = primitives.failure_cert
        ambiguity = primitives.ambiguity_cert

        trajectory_good = torch.maximum(trajectory_probs[:, 0], trajectory_probs[:, 1])
        trajectory_volatile = trajectory_probs[:, 3]
        pattern_focused = pattern_probs[:, 0]
        pattern_diffuse = pattern_probs[:, 2]

        success_anchor = torch.clamp(
            0.45 * stability
            + 0.20 * support
            + 0.20 * pattern_focused
            + 0.15 * trajectory_good,
            min=0.0,
            max=1.0,
        )
        failure_anchor = torch.clamp(
            0.45 * (1.0 - stability)
            + 0.15 * (1.0 - support)
            + 0.20 * pattern_diffuse
            + 0.20 * trajectory_volatile,
            min=0.0,
            max=1.0,
        )
        ambiguity_anchor = torch.clamp(
            0.35 * (1.0 - stability)
            + 0.20 * (1.0 - support)
            + 0.20 * (1.0 - (success - failure).abs())
            + 0.25 * (1.0 - torch.maximum(success, failure)),
            min=0.0,
            max=1.0,
        )

        success_prob = outcome_probs[:, 0]
        uncertain_prob = outcome_probs[:, 1]
        failure_prob = outcome_probs[:, 2]
        high_conf_prob = confidence_probs[:, 2]

        cert_alignment = (
            self.evidence_regression(success, success_anchor)
            + self.evidence_regression(failure, failure_anchor)
            + self.evidence_regression(ambiguity, ambiguity_anchor)
        ) / 3.0

        public_alignment = (
            (high_conf_prob * ambiguity).mean()
            + (success_prob * torch.relu(torch.maximum(failure, ambiguity) - success + 0.05)).mean()
            + (failure_prob * torch.relu(success - failure + 0.05)).mean()
            + (uncertain_prob * torch.relu(torch.abs(success - failure) - 0.20) * support).mean()
        ) / 4.0

        if success_guard is None:
            success_guard = success_anchor
        success_prob_guard_margin = float(self.config.trace_success_prob_guard_margin)
        success_prob_guard_penalty = 0.5 * float(
            self.config.trace_success_prob_guard_penalty_multiplier
        ) * torch.relu(success_prob - success_guard - success_prob_guard_margin).mean()
        success_guard_penalty = (
            0.25 * torch.relu(success_proposal - success_guard - 0.02).mean()
            + 0.25 * torch.relu(success - success_guard - 0.02).mean()
            + success_prob_guard_penalty
        )
        return cert_alignment + public_alignment + 0.25 * success_guard_penalty

    def _evidence_monotonicity_loss(
        self,
        primitives: PrimitiveOutput,
        perturbed_primitives: PrimitiveOutput,
    ) -> torch.Tensor:
        penalties = [
            torch.relu(perturbed_primitives.stability_cert - primitives.stability_cert).mean(),
            torch.relu(perturbed_primitives.support_cert - primitives.support_cert).mean(),
            torch.relu(perturbed_primitives.success_cert - primitives.success_cert).mean(),
            torch.relu(primitives.failure_cert - perturbed_primitives.failure_cert).mean(),
        ]
        return sum(penalties) / len(penalties)

    def _distribution_usage_penalty(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1).mean(dim=0)
        max_share = probs.max()
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum() / math.log(probs.numel())
        share_penalty = torch.relu(max_share - float(self.config.trace_usage_share_threshold))
        return share_penalty * (1.0 - entropy)

    def _usage_loss(self, primitives: PrimitiveOutput) -> torch.Tensor:
        confidence_penalty = self._distribution_usage_penalty(primitives.confidence_logits)
        outcome_penalty = self._distribution_usage_penalty(primitives.outcome_logits)
        return (confidence_penalty + outcome_penalty) / 2.0

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
