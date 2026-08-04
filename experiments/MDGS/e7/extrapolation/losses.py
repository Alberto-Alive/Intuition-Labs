"""Loss functions for the DIGIT Extrapolation E7 diffusion variant.

E7 keeps the E6 diffusion-owned outcome path and adds:
  - prototype alignment against detached diffusion outcomes
  - paired base-vs-perturbed denoise ranking
  - latent proximity regularisation for perturbation pairs
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .executor_schema import executor_feature_column
from .models.bottleneck import PrimitiveOutput


class DIGITLoss(nn.Module):
    """Combined loss for the E7 diffusion + prototype geometry scaffold."""

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
        perturbation_kind: str | None = None,
        aux_losses: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        zero = primitives.trajectory_shape_logits.new_tensor(0.0)
        aux_losses = aux_losses or {}

        # ── Primitive classification losses ───────────────────────────────────
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
        losses["primitive"] = primitive_loss if torch.isfinite(primitive_loss) else zero

        # ── Generation ────────────────────────────────────────────────────────
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

        # ── Evidence targets (certificate regression) ─────────────────────────
        evidence_target_loss = (
            self._evidence_target_loss(primitives, evidence_targets)
            if evidence_targets is not None and self.config.lambda_evidence_targets != 0.0
            else zero
        )
        losses["evidence_targets"] = evidence_target_loss

        # ── Internal consistency ──────────────────────────────────────────────
        consistency_loss = (
            self._consistency_loss(primitives)
            if self.config.lambda_consistency != 0.0
            else zero
        )
        losses["consistency"] = consistency_loss

        # ── Evidence monotonicity (perturbation-based) ────────────────────────
        evidence_mono_loss = (
            self._evidence_monotonicity_loss(primitives, perturbed_primitives, perturbation_kind=perturbation_kind)
            if perturbed_primitives is not None and self.config.lambda_evidence_mono != 0.0
            else zero
        )
        losses["evidence_mono"] = evidence_mono_loss

        # ── Usage / distribution collapse ─────────────────────────────────────
        usage_loss = self._usage_loss(primitives) if self.config.lambda_usage != 0.0 else zero
        losses["usage"] = usage_loss

        # ── Policy ────────────────────────────────────────────────────────────
        policy_loss = (
            self._policy_violation_penalty(primitives, executor_features)
            if self.config.lambda_policy != 0.0
            else zero
        )
        losses["policy"] = policy_loss

        # ── Leakage ───────────────────────────────────────────────────────────
        leakage_loss = (
            self._leakage_penalty(decoder_logits, executor_features)
            if decoder_logits is not None and self.config.lambda_leakage != 0.0
            else zero
        )
        losses["leakage"] = leakage_loss

        # ── Abstention ────────────────────────────────────────────────────────
        abstention_loss = (
            self._abstention_encouragement(primitives, executor_features)
            if self.config.lambda_abstention != 0.0
            else zero
        )
        losses["abstention"] = abstention_loss

        # ── E7 auxiliary losses ───────────────────────────────────────────────
        diffusion_denoise_loss = (
            primitives.diffusion_denoise_loss
            if hasattr(primitives, "diffusion_denoise_loss") and torch.isfinite(primitives.diffusion_denoise_loss)
            else zero
        )
        losses["diffusion_denoise"] = diffusion_denoise_loss
        prototype_alignment_loss = (
            primitives.prototype_alignment_loss
            if hasattr(primitives, "prototype_alignment_loss") and torch.isfinite(primitives.prototype_alignment_loss)
            else zero
        )
        losses["prototype_alignment"] = prototype_alignment_loss
        perturbation_rank_loss = aux_losses.get("perturbation_rank", zero)
        perturbation_proximity_loss = aux_losses.get("perturbation_proximity", zero)
        losses["perturbation_rank"] = perturbation_rank_loss
        losses["perturbation_proximity"] = perturbation_proximity_loss

        # ── Total ─────────────────────────────────────────────────────────────
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
            + self.config.lambda_diffusion_denoise * diffusion_denoise_loss
            + self.config.lambda_prototype_alignment * prototype_alignment_loss
            + self.config.lambda_perturbation_rank * perturbation_rank_loss
            + self.config.lambda_perturbation_proximity * perturbation_proximity_loss
        )

        return losses

    # ── Loss helpers (unchanged from E5) ──────────────────────────────────────

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
        *,
        perturbation_kind: str | None = None,
    ) -> torch.Tensor:
        failure_up = torch.relu(perturbed_primitives.failure_cert - primitives.failure_cert)
        ambiguity_up = torch.relu(perturbed_primitives.ambiguity_cert - primitives.ambiguity_cert)
        # Only relax commitment monotonicity when the perturbation shifts mass toward
        # failure without also making the example more ambiguous.
        ambiguity_gate = torch.sigmoid((0.02 - ambiguity_up) * 50.0)
        failure_commitment_allowance = 0.5 * failure_up * ambiguity_gate
        base_outcome_probs = F.softmax(primitives.outcome_logits, dim=-1)
        perturbed_outcome_probs = F.softmax(perturbed_primitives.outcome_logits, dim=-1)
        base_success_prob = base_outcome_probs[:, 0]
        base_uncertain_prob = base_outcome_probs[:, 1]
        base_failure_prob = base_outcome_probs[:, 2]
        perturbed_success_prob = perturbed_outcome_probs[:, 0]
        perturbed_uncertain_prob = perturbed_outcome_probs[:, 1]
        perturbed_failure_prob = perturbed_outcome_probs[:, 2]
        base_nonfailure_prob = base_success_prob + base_uncertain_prob
        perturbed_nonfailure_prob = perturbed_success_prob + perturbed_uncertain_prob
        base_guard_gap = torch.relu(primitives.success_base - primitives.success_guard)
        perturbed_guard_gap = torch.relu(perturbed_primitives.success_base - perturbed_primitives.success_guard)
        failure_context = (primitives.failure_cert > 0.12).float()
        ambiguity_context = (torch.maximum(primitives.ambiguity_cert, base_uncertain_prob) > 0.35).float()
        agreement_failure_context = torch.maximum(
            failure_context,
            (primitives.failure_family_support > 0.10).float(),
        )
        penalties = [
            torch.relu(perturbed_primitives.support_cert - primitives.support_cert).mean(),
            torch.relu(
                perturbed_primitives.commitment_depth
                - primitives.commitment_depth
                - failure_commitment_allowance
            ).mean(),
            torch.relu(perturbed_primitives.success_guard - primitives.success_guard).mean(),
            torch.relu(perturbed_primitives.success_cert - primitives.success_cert).mean(),
            torch.relu(perturbed_success_prob - base_success_prob).mean(),
            torch.relu(perturbed_nonfailure_prob - base_nonfailure_prob).mean(),
            torch.relu(base_guard_gap - perturbed_guard_gap).mean(),
            torch.relu(primitives.failure_family_support - perturbed_primitives.failure_family_support).mean(),
            torch.relu(primitives.boundary_family_support - perturbed_primitives.boundary_family_support).mean(),
            torch.relu(primitives.anchor_conflict - perturbed_primitives.anchor_conflict).mean(),
            (failure_context * torch.relu(primitives.failure_cert - perturbed_primitives.failure_cert)).mean(),
            (failure_context * torch.relu(base_failure_prob - perturbed_failure_prob)).mean(),
        ]
        if hasattr(primitives, "worsening_radius"):
            penalties.append(
                torch.relu(primitives.worsening_radius - perturbed_primitives.worsening_radius).mean()
            )
        if hasattr(primitives, "worsening_ambiguity"):
            penalties.append(
                torch.relu(primitives.worsening_ambiguity - perturbed_primitives.worsening_ambiguity).mean()
            )
        if perturbation_kind in {"lower_attention_concentration", "higher_entropy", "higher_variation_ratio"}:
            penalties.extend(
                [
                    torch.relu(primitives.ambiguity_cert - perturbed_primitives.ambiguity_cert).mean(),
                    torch.relu(base_uncertain_prob - perturbed_uncertain_prob).mean(),
                ]
            )
        if perturbation_kind == "lower_attention_concentration":
            penalties.extend(
                [
                    1.5 * (ambiguity_context * torch.relu(primitives.ambiguity_cert - perturbed_primitives.ambiguity_cert)).mean(),
                    1.25 * (ambiguity_context * torch.relu(base_uncertain_prob - perturbed_uncertain_prob)).mean(),
                    1.25 * (ambiguity_context * torch.relu(perturbed_primitives.commitment_depth - primitives.commitment_depth)).mean(),
                ]
            )
            if hasattr(primitives, "worsening_ambiguity"):
                penalties.append(
                    1.25 * (ambiguity_context * torch.relu(primitives.worsening_ambiguity - perturbed_primitives.worsening_ambiguity)).mean()
                )
        if perturbation_kind in {"lower_agreement", "lower_margin"}:
            penalties.extend(
                [
                    (failure_context * torch.relu(primitives.failure_cert - perturbed_primitives.failure_cert)).mean(),
                    torch.relu(base_failure_prob - perturbed_failure_prob).mean(),
                ]
            )
        if perturbation_kind == "lower_agreement":
            penalties.extend(
                [
                    1.5 * (agreement_failure_context * torch.relu(primitives.failure_cert - perturbed_primitives.failure_cert)).mean(),
                    1.25 * (agreement_failure_context * torch.relu(base_failure_prob - perturbed_failure_prob)).mean(),
                    1.10 * (agreement_failure_context * torch.relu(primitives.failure_family_support - perturbed_primitives.failure_family_support)).mean(),
                ]
            )
            if hasattr(primitives, "worsening_failure"):
                penalties.append(
                    1.10 * (agreement_failure_context * torch.relu(primitives.worsening_failure - perturbed_primitives.worsening_failure)).mean()
                )
        if perturbation_kind == "higher_entropy":
            penalties.extend(
                [
                    1.35 * (failure_context * torch.relu(primitives.failure_cert - perturbed_primitives.failure_cert)).mean(),
                    1.25 * (failure_context * torch.relu(base_failure_prob - perturbed_failure_prob)).mean(),
                    1.10 * (failure_context * torch.relu(perturbed_nonfailure_prob - base_nonfailure_prob)).mean(),
                    1.10 * (failure_context * torch.relu(perturbed_guard_gap - base_guard_gap)).mean(),
                ]
            )
            if hasattr(primitives, "worsening_failure"):
                penalties.append(
                    1.10 * (failure_context * torch.relu(primitives.worsening_failure - perturbed_primitives.worsening_failure)).mean()
                )
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
