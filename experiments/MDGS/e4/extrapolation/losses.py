"""Loss functions for the DIGIT Extrapolation E4 anchor bottleneck."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .executor_schema import executor_feature_column
from .models.bottleneck import PrimitiveOutput


class DIGITLoss(nn.Module):
    """Combined loss for the anchor-structured E4 variant."""

    def __init__(self, config, pad_idx: int = 0, class_weights: Dict[str, torch.Tensor] | None = None):
        super().__init__()
        self.config = config
        self.pad_idx = pad_idx

        if class_weights:
            self.trajectory_ce = nn.CrossEntropyLoss(
                weight=class_weights.get("trajectory_shape"),
                reduction="mean",
            )
            self.pattern_ce = nn.CrossEntropyLoss(
                weight=class_weights.get("attention_pattern"),
                reduction="mean",
            )
            self.confidence_ce = nn.CrossEntropyLoss(
                weight=class_weights.get("confidence"),
                reduction="mean",
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
        losses: Dict[str, torch.Tensor] = {}

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
        ) / max(total_weight, 1e-6)
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

        consistency_loss = self._consistency_loss(primitives) if self.config.lambda_consistency != 0.0 else zero
        losses["consistency"] = consistency_loss

        evidence_mono_loss = (
            self._evidence_monotonicity_loss(primitives, perturbed_primitives)
            if perturbed_primitives is not None and self.config.lambda_evidence_mono != 0.0
            else zero
        )
        losses["evidence_mono"] = evidence_mono_loss

        usage_loss = self._usage_loss(primitives) if self.config.lambda_usage != 0.0 else zero
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

        anchor_pull_loss = (
            self._anchor_pull_loss(primitives, evidence_targets)
            if evidence_targets is not None and self.config.lambda_evidence_targets != 0.0
            else zero
        )
        losses["anchor_pull"] = anchor_pull_loss

        anchor_repulsion_loss = (
            self._anchor_repulsion_loss(primitives)
            if self.config.lambda_evidence_targets != 0.0
            else zero
        )
        losses["anchor_repulsion"] = anchor_repulsion_loss

        anchor_usage_loss = (
            self._anchor_usage_loss(primitives)
            if self.config.lambda_usage != 0.0
            else zero
        )
        losses["anchor_usage"] = anchor_usage_loss

        anchor_approach_alignment_loss = (
            self._anchor_approach_alignment_loss(primitives, evidence_targets)
            if evidence_targets is not None and self.config.anchor_approach_alignment_weight != 0.0
            else zero
        )
        losses["anchor_approach_alignment"] = anchor_approach_alignment_loss

        anchor_family_target_loss = (
            self._anchor_family_target_loss(primitives, evidence_targets)
            if evidence_targets is not None and self.config.anchor_family_target_weight != 0.0
            else zero
        )
        losses["anchor_family_target"] = anchor_family_target_loss

        anchor_family_margin_loss = (
            self._anchor_family_margin_loss(primitives, evidence_targets)
            if evidence_targets is not None and self.config.anchor_family_margin_weight != 0.0
            else zero
        )
        losses["anchor_family_margin"] = anchor_family_margin_loss

        cooperative_cert_order_loss = (
            self._cooperative_certificate_order_loss(primitives, prim_targets)
            if self.config.cooperative_cert_order_weight != 0.0
            else zero
        )
        losses["cooperative_cert_order"] = cooperative_cert_order_loss

        cooperative_guard_target_loss = (
            self._cooperative_guard_target_loss(primitives, evidence_targets)
            if evidence_targets is not None and self.config.cooperative_guard_target_weight != 0.0
            else zero
        )
        losses["cooperative_guard_target"] = cooperative_guard_target_loss

        cooperative_commitment_target_loss = (
            self._cooperative_commitment_target_loss(primitives, evidence_targets)
            if evidence_targets is not None and self.config.cooperative_commitment_target_weight != 0.0
            else zero
        )
        losses["cooperative_commitment_target"] = cooperative_commitment_target_loss

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
            + self.config.anchor_pull_weight * anchor_pull_loss
            + self.config.anchor_repulsion_weight * anchor_repulsion_loss
            + self.config.anchor_usage_weight * anchor_usage_loss
            + self.config.anchor_approach_alignment_weight * anchor_approach_alignment_loss
            + self.config.anchor_family_target_weight * anchor_family_target_loss
            + self.config.anchor_family_margin_weight * anchor_family_margin_loss
            + self.config.cooperative_cert_order_weight * cooperative_cert_order_loss
            + self.config.cooperative_guard_target_weight * cooperative_guard_target_loss
            + self.config.cooperative_commitment_target_weight * cooperative_commitment_target_loss
        )

        return losses

    def _family_support_stack(self, primitives: PrimitiveOutput) -> torch.Tensor:
        return torch.stack(
            [
                primitives.success_family_support,
                primitives.failure_family_support,
                primitives.boundary_family_support,
            ],
            dim=-1,
        )

    def _family_target_state(
        self,
        primitives: PrimitiveOutput,
        evidence_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        family_support = self._family_support_stack(primitives)
        family_targets = evidence_targets[:, 2:5].clone()
        family_enabled = family_support.detach().amax(dim=0) > 1e-8
        if not bool(family_enabled.any()):
            family_enabled = torch.ones_like(family_enabled, dtype=torch.bool)

        family_targets = family_targets * family_enabled.to(family_targets.dtype).unsqueeze(0)
        empty_rows = family_targets.sum(dim=-1) <= 1e-8
        if bool(empty_rows.any()):
            family_targets[empty_rows] = family_enabled.to(family_targets.dtype)

        family_target_dist = family_targets / family_targets.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        target_family = family_target_dist.argmax(dim=-1)
        target_support = family_support.gather(1, target_family.unsqueeze(-1)).squeeze(-1)
        topk = torch.topk(
            family_target_dist,
            k=min(2, family_target_dist.size(-1)),
            dim=-1,
        ).values
        if topk.size(-1) == 1:
            target_margin = topk[:, 0]
        else:
            target_margin = topk[:, 0] - topk[:, 1]
        sample_weight = (
            family_target_dist.gather(1, target_family.unsqueeze(-1)).squeeze(-1)
            + target_margin
        ).detach()
        return family_support, family_enabled, target_family, sample_weight.clamp_min(0.25)

    def _certificate_stack(self, primitives: PrimitiveOutput) -> torch.Tensor:
        return torch.stack(
            [
                primitives.success_cert,
                primitives.failure_cert,
                primitives.ambiguity_cert,
            ],
            dim=-1,
        )

    def _success_guard_target_from_evidence_targets(self, evidence_targets: torch.Tensor) -> torch.Tensor:
        stability_target = evidence_targets[:, 0]
        support_target = evidence_targets[:, 1]
        success_target = evidence_targets[:, 2]
        guard_anchor = torch.clamp(0.65 * stability_target + 0.35 * support_target, min=0.0, max=1.0)
        return torch.minimum(success_target, guard_anchor)

    def _commitment_target_from_evidence_targets(self, evidence_targets: torch.Tensor) -> torch.Tensor:
        stability_target = evidence_targets[:, 0]
        support_target = evidence_targets[:, 1]
        success_target = evidence_targets[:, 2]
        failure_target = evidence_targets[:, 3]
        ambiguity_target = evidence_targets[:, 4]

        decisive_target = torch.maximum(success_target, failure_target)
        decisive_margin = (success_target - failure_target).abs()
        clarity_anchor = torch.clamp(0.50 * stability_target + 0.50 * support_target, min=0.0, max=1.0)
        return torch.clamp(
            (0.70 * decisive_target + 0.30 * clarity_anchor) * (1.0 - 0.50 * ambiguity_target)
            + 0.15 * decisive_margin,
            min=0.0,
            max=1.0,
        )

    def _proxy_stack(self, primitives: PrimitiveOutput) -> torch.Tensor:
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
        return self.evidence_regression(self._proxy_stack(primitives), evidence_targets)

    def _consistency_loss(self, primitives: PrimitiveOutput) -> torch.Tensor:
        confidence_probs = F.softmax(primitives.confidence_logits, dim=-1)
        outcome_probs = F.softmax(primitives.outcome_logits, dim=-1)

        high_conf_prob = confidence_probs[:, 2]
        low_conf_prob = confidence_probs[:, 0]
        success_prob = outcome_probs[:, 0]
        uncertain_prob = outcome_probs[:, 1]
        failure_prob = outcome_probs[:, 2]

        ambiguity = primitives.ambiguity_cert
        success = primitives.success_cert
        failure = primitives.failure_cert
        support = primitives.support_cert
        commitment = primitives.commitment_depth

        cert_alignment = (
            (high_conf_prob * ambiguity).mean()
            + (success_prob * torch.relu(torch.maximum(failure, ambiguity) - success + 0.05)).mean()
            + (failure_prob * torch.relu(success - failure + 0.05)).mean()
            + (uncertain_prob * torch.relu(torch.abs(success - failure) - 0.20) * support).mean()
            + (low_conf_prob * torch.relu(commitment - 0.50)).mean()
        ) / 5.0
        return cert_alignment

    def _evidence_monotonicity_loss(
        self,
        primitives: PrimitiveOutput,
        perturbed_primitives: PrimitiveOutput,
    ) -> torch.Tensor:
        failure_commitment_allowance = 0.5 * torch.relu(
            perturbed_primitives.failure_cert - primitives.failure_cert
        )
        penalties = [
            torch.relu(perturbed_primitives.support_cert - primitives.support_cert).mean(),
            torch.relu(
                perturbed_primitives.commitment_depth
                - primitives.commitment_depth
                - failure_commitment_allowance
            ).mean(),
            torch.relu(primitives.failure_family_support - perturbed_primitives.failure_family_support).mean(),
            torch.relu(primitives.boundary_family_support - perturbed_primitives.boundary_family_support).mean(),
            torch.relu(primitives.anchor_conflict - perturbed_primitives.anchor_conflict).mean(),
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
        outcome_probs = F.softmax(primitives.outcome_logits, dim=-1)

        high_confidence_prob = confidence_probs[:, 2]
        success_prob = outcome_probs[:, 0]

        unstable_mask = ((entropy_range >= 0.35) | (std_entropy >= 0.16)).float()
        unsupported_success_mask = torch.clamp(low_margin_flag + unstable_flag, max=1.0)
        high_conflict_mask = (primitives.anchor_conflict >= 0.60).float()
        return (
            (unstable_mask * high_confidence_prob).mean()
            + (unsupported_success_mask * success_prob).mean()
            + (high_conflict_mask * success_prob).mean()
            + (torch.relu(mean_entropy - 0.65) * high_confidence_prob).mean()
        ) / 4.0

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
            | (primitives.anchor_conflict >= 0.60)
        ).float()
        return (low_certainty_mask * success_prob).mean()

    def _anchor_pull_loss(self, primitives: PrimitiveOutput, evidence_targets: torch.Tensor) -> torch.Tensor:
        anchor_support = primitives.anchor_support_scores
        _, _, target_family, sample_weight = self._family_target_state(primitives, evidence_targets)
        batch_indices = torch.arange(anchor_support.size(0), device=anchor_support.device)
        target_anchor_support = anchor_support[batch_indices, target_family].max(dim=-1).values
        return (sample_weight * (1.0 - target_anchor_support)).mean()

    def _anchor_repulsion_loss(self, primitives: PrimitiveOutput) -> torch.Tensor:
        centers = primitives.anchor_centers
        penalties = []
        margin = float(self.config.anchor_repulsion_margin)
        for lhs in range(centers.size(0)):
            for rhs in range(lhs + 1, centers.size(0)):
                pairwise = torch.cdist(centers[lhs], centers[rhs], p=2)
                penalties.append(torch.relu(margin - pairwise).mean())
        if not penalties:
            return centers.new_tensor(0.0)
        return sum(penalties) / len(penalties)

    def _anchor_usage_loss(self, primitives: PrimitiveOutput) -> torch.Tensor:
        assignment = primitives.anchor_assignment_probs
        family_usage = assignment.mean(dim=0)
        entropy = -(
            family_usage * family_usage.clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(family_usage.size(-1))
        return (1.0 - entropy).mean()

    def _anchor_approach_alignment_loss(self, primitives: PrimitiveOutput, evidence_targets: torch.Tensor) -> torch.Tensor:
        family_approach = torch.stack(
            [
                primitives.success_family_approach,
                primitives.failure_family_approach,
                primitives.boundary_family_approach,
            ],
            dim=-1,
        )
        _, _, target_family, sample_weight = self._family_target_state(primitives, evidence_targets)
        target_approach = family_approach.gather(1, target_family.unsqueeze(-1)).squeeze(-1)
        return (sample_weight * (1.0 - target_approach)).mean()

    def _anchor_family_target_loss(self, primitives: PrimitiveOutput, evidence_targets: torch.Tensor) -> torch.Tensor:
        family_support, family_enabled, target_family, sample_weight = self._family_target_state(
            primitives,
            evidence_targets,
        )
        family_logits = torch.log(family_support.clamp_min(1e-6))
        family_logits = family_logits.masked_fill(~family_enabled.unsqueeze(0), -1e4)
        losses = F.cross_entropy(family_logits, target_family, reduction="none")
        return (sample_weight * losses).mean()

    def _anchor_family_margin_loss(self, primitives: PrimitiveOutput, evidence_targets: torch.Tensor) -> torch.Tensor:
        family_support, family_enabled, target_family, sample_weight = self._family_target_state(
            primitives,
            evidence_targets,
        )
        if int(family_enabled.long().sum().item()) <= 1:
            return family_support.new_tensor(0.0)

        target_support = family_support.gather(1, target_family.unsqueeze(-1)).squeeze(-1)
        target_mask = F.one_hot(target_family, num_classes=family_support.size(-1)).bool()
        active_rest_mask = family_enabled.unsqueeze(0) & ~target_mask
        rest_support = family_support.masked_fill(~active_rest_mask, -1.0).max(dim=-1).values
        margin_loss = torch.relu(
            float(self.config.anchor_family_margin) + rest_support - target_support
        )
        return (sample_weight * margin_loss).mean()

    def _cooperative_certificate_order_loss(
        self,
        primitives: PrimitiveOutput,
        prim_targets: torch.Tensor,
    ) -> torch.Tensor:
        certs = self._certificate_stack(primitives)
        outcome_targets = prim_targets[:, 3]
        cert_target_map = certs.new_tensor([0, 2, 1], dtype=torch.long)
        target_family = cert_target_map[outcome_targets]
        target_cert = certs.gather(1, target_family.unsqueeze(-1)).squeeze(-1)
        target_mask = F.one_hot(target_family, num_classes=certs.size(-1)).bool()
        other_cert = certs.masked_fill(target_mask, -1.0).max(dim=-1).values

        sample_weight = torch.ones_like(target_cert)
        failure_weight = float(self.config.trace_outcome_failure_weight)
        sample_weight = torch.where(
            outcome_targets == 2,
            sample_weight * max(failure_weight, 1.0),
            sample_weight,
        )
        margin = torch.relu(
            float(self.config.cooperative_cert_margin) + other_cert - target_cert
        )
        return (sample_weight * margin).mean()

    def _cooperative_guard_target_loss(
        self,
        primitives: PrimitiveOutput,
        evidence_targets: torch.Tensor,
    ) -> torch.Tensor:
        guard_target = self._success_guard_target_from_evidence_targets(evidence_targets)
        sample_weight = (0.25 + guard_target).detach()
        return (sample_weight * F.smooth_l1_loss(
            primitives.success_guard,
            guard_target,
            reduction="none",
        )).mean()

    def _cooperative_commitment_target_loss(
        self,
        primitives: PrimitiveOutput,
        evidence_targets: torch.Tensor,
    ) -> torch.Tensor:
        commitment_target = self._commitment_target_from_evidence_targets(evidence_targets)
        decisive_target = torch.maximum(evidence_targets[:, 2], evidence_targets[:, 3])
        sample_weight = (0.35 + decisive_target).detach()
        return (sample_weight * F.smooth_l1_loss(
            primitives.commitment_depth,
            commitment_target,
            reduction="none",
        )).mean()
