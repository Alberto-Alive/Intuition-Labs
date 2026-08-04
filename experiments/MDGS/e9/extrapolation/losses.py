"""Loss functions for the DIGIT Extrapolation E9 diffusion variant.

E9 keeps the E6 diffusion-owned outcome path and adds:
  - direct prototype geometry losses instead of outcome-mimic alignment
  - direct fragility supervision under named perturbations
  - paired base-vs-perturbed denoise ranking
  - path-level family supervision on reconstructed diffusion paths
  - prototype-guided basin-order losses under named perturbations
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
    """Combined loss for the E9 diffusion + prototype geometry scaffold."""

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

        guard_quality_loss = (
            self._guard_quality_loss(primitives, evidence_targets, prim_targets)
            if evidence_targets is not None and self.config.trace_guard_quality_weight != 0.0
            else zero
        )
        losses["guard_quality"] = guard_quality_loss

        family_supervision_loss = (
            self._family_supervision_loss(primitives, prim_targets)
            if float(getattr(self.config, "anchor_family_target_weight", 0.0)) != 0.0
            else zero
        )
        losses["family_supervision"] = family_supervision_loss

        success_prior_loss = (
            self._success_prior_calibration_loss(primitives, prim_targets)
            if self.config.trace_success_prior_weight != 0.0
            else zero
        )
        losses["success_prior"] = success_prior_loss

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

        # ── E9 auxiliary losses ───────────────────────────────────────────────
        diffusion_denoise_loss = (
            primitives.diffusion_denoise_loss
            if hasattr(primitives, "diffusion_denoise_loss") and torch.isfinite(primitives.diffusion_denoise_loss)
            else zero
        )
        losses["diffusion_denoise"] = diffusion_denoise_loss
        prototype_pull_loss = (
            primitives.prototype_pull_loss
            if hasattr(primitives, "prototype_pull_loss") and torch.isfinite(primitives.prototype_pull_loss)
            else zero
        )
        prototype_usage_loss = (
            primitives.prototype_usage_loss
            if hasattr(primitives, "prototype_usage_loss") and torch.isfinite(primitives.prototype_usage_loss)
            else zero
        )
        prototype_repulsion_loss = (
            primitives.prototype_repulsion_loss
            if hasattr(primitives, "prototype_repulsion_loss") and torch.isfinite(primitives.prototype_repulsion_loss)
            else zero
        )
        losses["prototype_pull"] = prototype_pull_loss
        losses["prototype_usage"] = prototype_usage_loss
        losses["prototype_repulsion"] = prototype_repulsion_loss
        perturbation_rank_loss = aux_losses.get("perturbation_rank", zero)
        perturbation_proximity_loss = aux_losses.get("perturbation_proximity", zero)
        basin_order_loss = aux_losses.get("basin_order", zero)
        ordered_geometry_loss = aux_losses.get("ordered_geometry", zero)
        fragility_target_loss = aux_losses.get("fragility_target_loss", zero)
        fragility_rank_loss = aux_losses.get("fragility_rank", zero)
        losses["perturbation_rank"] = perturbation_rank_loss
        losses["perturbation_proximity"] = perturbation_proximity_loss
        losses["basin_order"] = basin_order_loss
        losses["ordered_geometry"] = ordered_geometry_loss
        losses["fragility_target"] = fragility_target_loss
        losses["fragility_rank"] = fragility_rank_loss

        # ── Total ─────────────────────────────────────────────────────────────
        losses["total"] = (
            self.config.lambda_primitive * primitive_loss
            + self.config.lambda_generation * generation_loss
            + self.config.lambda_evidence_targets * evidence_target_loss
            + self.config.lambda_consistency * consistency_loss
            + self.config.trace_guard_quality_weight * guard_quality_loss
            + self.config.trace_success_prior_weight * success_prior_loss
            + self.config.lambda_evidence_mono * evidence_mono_loss
            + self.config.lambda_usage * usage_loss
            + self.config.lambda_policy * policy_loss
            + self.config.lambda_leakage * leakage_loss
            + self.config.lambda_abstention * abstention_loss
            + self.config.lambda_diffusion_denoise * diffusion_denoise_loss
            + self.config.anchor_pull_weight * prototype_pull_loss
            + self.config.anchor_usage_weight * prototype_usage_loss
            + self.config.anchor_repulsion_weight * prototype_repulsion_loss
            + float(getattr(self.config, "anchor_family_target_weight", 0.0)) * family_supervision_loss
            + float(getattr(self.config, "lambda_basin_order", 0.0)) * basin_order_loss
            + float(getattr(self.config, "lambda_ordered_geometry", 0.0)) * ordered_geometry_loss
            + self.config.lambda_perturbation_rank * perturbation_rank_loss
            + self.config.lambda_perturbation_proximity * perturbation_proximity_loss
            + self.config.lambda_fragility_target * (fragility_target_loss + 0.5 * fragility_rank_loss)
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

    @staticmethod
    def _guard_target_from_evidence_targets(evidence_targets: torch.Tensor) -> torch.Tensor:
        stability_target = evidence_targets[:, 0]
        support_target = evidence_targets[:, 1]
        success_target = evidence_targets[:, 2]
        ambiguity_target = evidence_targets[:, 4]
        quality_target = torch.clamp(
            0.45 * stability_target
            + 0.30 * support_target
            + 0.25 * (1.0 - ambiguity_target),
            min=0.0,
            max=1.0,
        )
        return torch.minimum(success_target, 0.90 * quality_target + 0.10 * success_target)

    @staticmethod
    def _high_conflict_mask(primitives: PrimitiveOutput) -> torch.Tensor:
        trust_conflict = getattr(primitives, "trust_conflict", primitives.anchor_conflict)
        prototype_margin = getattr(primitives, "prototype_family_margin", torch.zeros_like(trust_conflict))
        prototype_overlap = getattr(primitives, "prototype_overlap", torch.zeros_like(trust_conflict))
        return (
            (trust_conflict >= 0.78)
            | ((prototype_margin < 0.08) & (prototype_overlap > 0.55))
            | (primitives.anchor_conflict >= 0.92)
        ).float()

    def _guard_quality_loss(
        self,
        primitives: PrimitiveOutput,
        evidence_targets: torch.Tensor,
        prim_targets: torch.Tensor,
    ) -> torch.Tensor:
        guard_target = self._guard_target_from_evidence_targets(evidence_targets)
        success_target = (prim_targets[:, 3] == 0).float()
        non_success_target = 1.0 - success_target
        success_prob = F.softmax(primitives.outcome_logits, dim=-1)[:, 0]
        admissible_success = torch.minimum(primitives.success_base, success_prob)
        high_conflict = self._high_conflict_mask(primitives)
        unsafe_success_excess = torch.relu(
            success_prob
            - primitives.success_guard
            - float(self.config.trace_success_prob_guard_margin)
        )
        blocked_success = torch.relu(primitives.success_guard - admissible_success + 0.02)
        guard_excess = torch.relu(primitives.success_guard - (guard_target + 0.10))
        base_fit = F.smooth_l1_loss(primitives.success_guard, guard_target)
        unsafe_success_penalty = (
            non_success_target
            * (0.35 + 0.65 * high_conflict)
            * unsafe_success_excess
        ).mean()
        success_admission_penalty = (success_target * blocked_success).mean()
        guard_excess_penalty = (
            (0.50 * success_target + non_success_target) * guard_excess
        ).mean()
        return (
            base_fit
            + 0.60 * unsafe_success_penalty
            + 0.35 * success_admission_penalty
            + 0.20 * guard_excess_penalty
        )

    def _family_supervision_loss(
        self,
        primitives: PrimitiveOutput,
        prim_targets: torch.Tensor,
    ) -> torch.Tensor:
        family_support = torch.stack(
            [
                primitives.success_family_support,
                primitives.failure_family_support,
                primitives.boundary_family_support,
            ],
            dim=-1,
        ).clamp_min(1e-6)
        family_log_probs = family_support.log() - family_support.logsumexp(dim=-1, keepdim=True)
        outcome_targets = prim_targets[:, 3].long()
        family_targets = torch.where(
            outcome_targets == 2,
            torch.ones_like(outcome_targets),
            torch.where(outcome_targets == 1, torch.full_like(outcome_targets, 2), outcome_targets),
        )
        ce_loss = F.nll_loss(family_log_probs, family_targets, reduction="mean")

        target_support = family_support.gather(1, family_targets.unsqueeze(-1)).squeeze(-1)
        other_support = family_support.masked_fill(
            F.one_hot(family_targets, num_classes=family_support.size(-1)).bool(),
            float("-inf"),
        ).amax(dim=-1)
        decisive_margin = torch.relu(
            float(self.config.anchor_family_margin) - (target_support - other_support)
        )
        decisive_mask = (family_targets != 2).float()
        boundary_margin = torch.relu(0.04 - (target_support - other_support))
        margin_loss = (
            decisive_mask * decisive_margin + (1.0 - decisive_mask) * boundary_margin
        ).mean()
        top_family_mismatch = (primitives.top_family_id.long() != family_targets).float().mean()
        if hasattr(primitives, "prototype_family_distribution_paths") and hasattr(
            primitives,
            "prototype_family_support_paths",
        ):
            path_family_probs = primitives.prototype_family_distribution_paths.clamp_min(1e-6)
            num_paths = path_family_probs.size(0)
            path_log_probs = path_family_probs.permute(1, 0, 2).reshape(-1, path_family_probs.size(-1)).log()
            path_targets = family_targets.unsqueeze(1).expand(-1, num_paths).reshape(-1)
            path_ce_loss = F.nll_loss(path_log_probs, path_targets, reduction="mean")

            path_support = primitives.prototype_family_support_paths.permute(1, 0, 2)
            target_path_support = path_support.gather(
                2,
                family_targets[:, None, None].expand(-1, num_paths, 1),
            ).squeeze(-1)
            other_path_support = path_support.masked_fill(
                F.one_hot(family_targets, num_classes=path_support.size(-1))[:, None, :].bool(),
                float("-inf"),
            ).amax(dim=-1)
            path_decisive_margin = torch.relu(
                float(self.config.anchor_family_margin) - (target_path_support - other_path_support)
            )
            path_boundary_margin = torch.relu(0.04 - (target_path_support - other_path_support))
            path_margin_loss = (
                decisive_mask[:, None] * path_decisive_margin
                + (1.0 - decisive_mask)[:, None] * path_boundary_margin
            ).mean()
            path_top_family = path_support.argmax(dim=-1)
            path_mismatch = (path_top_family != family_targets[:, None]).float().mean()
        else:
            path_ce_loss = family_support.new_tensor(0.0)
            path_margin_loss = family_support.new_tensor(0.0)
            path_mismatch = family_support.new_tensor(0.0)

        return (
            ce_loss
            + float(getattr(self.config, "anchor_family_margin_weight", 0.0)) * margin_loss
            + 0.25 * top_family_mismatch
            + 0.85 * path_ce_loss
            + 0.60 * path_margin_loss
            + 0.25 * path_mismatch
        )

    def _success_prior_calibration_loss(
        self,
        primitives: PrimitiveOutput,
        prim_targets: torch.Tensor,
    ) -> torch.Tensor:
        success_prob = F.softmax(primitives.outcome_logits, dim=-1)[:, 0]
        target_success_rate = (prim_targets[:, 3] == 0).float().mean()
        success_rate_excess = torch.relu(
            success_prob.mean() - (target_success_rate + float(self.config.trace_success_prior_slack))
        )
        boundary_pressure = torch.relu(primitives.boundary_family_support - primitives.success_family_support)
        uncertain_mask = (prim_targets[:, 3] == 1).float()
        uncertainty_success = (
            uncertain_mask
            * success_prob
            * (0.50 + 0.50 * boundary_pressure + 0.50 * primitives.trust_conflict)
        ).mean()
        return success_rate_excess + 0.50 * uncertainty_success

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
        boundary_context = (
            ambiguity_context
            * (primitives.boundary_family_support > primitives.success_family_support).float()
        )
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
        if hasattr(primitives, "fragility_risk"):
            sharp_violation = (
                torch.relu(perturbed_primitives.commitment_depth - primitives.commitment_depth)
                + torch.relu(perturbed_primitives.certainty_score - primitives.certainty_score)
            ).detach()
            penalties.append(
                (sharp_violation * torch.relu(primitives.fragility_risk - perturbed_primitives.fragility_risk)).mean()
            )
        if hasattr(primitives, "prototype_total_support"):
            penalties.append(
                torch.relu(perturbed_primitives.prototype_total_support - primitives.prototype_total_support).mean()
            )
        if hasattr(primitives, "prototype_overlap"):
            penalties.append(
                torch.relu(perturbed_primitives.prototype_overlap - primitives.prototype_overlap).mean()
            )
        if hasattr(primitives, "prototype_nearest_anchor_distance"):
            penalties.append(
                torch.relu(
                    primitives.prototype_nearest_anchor_distance
                    - perturbed_primitives.prototype_nearest_anchor_distance
                ).mean()
            )
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
                    1.35 * (boundary_context * torch.relu(perturbed_success_prob - base_success_prob)).mean(),
                    1.35 * (boundary_context * torch.relu(perturbed_primitives.commitment_depth - primitives.commitment_depth)).mean(),
                    1.20 * (boundary_context * torch.relu(primitives.fragility_risk - perturbed_primitives.fragility_risk)).mean(),
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
        high_conflict_mask = self._high_conflict_mask(primitives)
        boundary_bias = torch.relu(primitives.boundary_family_support - primitives.success_family_support)
        return (
            (unstable_mask * high_confidence_prob).mean()
            + (unsupported_success_mask * success_prob).mean()
            + (high_conflict_mask * success_prob).mean()
            + (boundary_bias * success_prob).mean()
            + (torch.relu(mean_entropy - 0.65) * high_confidence_prob).mean()
        ) / 5.0

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
            | (self._high_conflict_mask(primitives) > 0.0)
        ).float()
        return (low_certainty_mask * success_prob).mean()
