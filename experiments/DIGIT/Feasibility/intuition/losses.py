"""Loss functions for the Intuition model.

Losses:
  1. Primitive classification loss — cross-entropy on each primitive head
  2. Decoder generation loss — cross-entropy on token generation
  3. Policy violation penalty — penalize when bottleneck disagrees with executor policy flags
  4. Leakage penalty — mutual information upper bound between decoder output and executor features
  5. Abstention encouragement — reward INSUFFICIENT_EVIDENCE / MAYBE for low-support cases
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .models.bottleneck import PrimitiveOutput


class IntuitionLoss(nn.Module):
    """Combined loss for training the full intuition model."""

    def __init__(self, config, pad_idx: int = 0):
        super().__init__()
        self.config = config
        self.pad_idx = pad_idx

        # Primitive classification
        self.prim_ce = nn.CrossEntropyLoss(reduction="mean")

        # Generation
        self.gen_ce = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="mean")

    def forward(
        self,
        decoder_logits: torch.Tensor,
        primitives: PrimitiveOutput,
        target_ids: torch.Tensor,
        prim_targets: torch.Tensor,
        executor_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            decoder_logits: (B, T-1, V) — predicted next-token logits.
            primitives: PrimitiveOutput from bottleneck.
            target_ids: (B, T) — ground truth token ids (with BOS).
            prim_targets: (B, 4) — ground truth [answer, support, confidence, risk].
            executor_features: (B, executor_dim) — from safe executor.

        Returns:
            Dict with individual losses and total.
        """
        losses = {}

        # ── 1. Primitive classification loss ──────────────────────────
        prim_loss = (
            self.prim_ce(primitives.answer_logits, prim_targets[:, 0])
            + self.prim_ce(primitives.support_logits, prim_targets[:, 1])
            + self.prim_ce(primitives.confidence_logits, prim_targets[:, 2])
            + self.prim_ce(primitives.risk_logits, prim_targets[:, 3])
        ) / 4.0
        losses["primitive"] = prim_loss

        # ── 2. Decoder generation loss ────────────────────────────────
        # decoder_logits: (B, T-1, V), targets: (B, T)
        # We predict target[1:] from input target[:-1]
        gen_targets = target_ids[:, 1:]  # (B, T-1)
        B, T_minus_1, V = decoder_logits.shape
        gen_loss = self.gen_ce(
            decoder_logits.reshape(B * T_minus_1, V),
            gen_targets.reshape(B * T_minus_1),
        )
        losses["generation"] = gen_loss

        # ── 3. Policy violation penalty ───────────────────────────────
        # If executor says too_specific (policy flag 0 is active),
        # the answer primitive should be POLICY_BLOCKED (index 5).
        # If executor says min_group_size fails, answer should be
        # INSUFFICIENT_EVIDENCE (index 3).
        policy_loss = self._policy_violation_penalty(primitives, executor_features)
        losses["policy"] = policy_loss

        # ── 4. Leakage penalty ────────────────────────────────────────
        # Penalize if decoder hidden states carry information about
        # executor features beyond what the discrete primitives encode.
        # Approximation: correlation between decoder logits and executor features.
        leakage_loss = self._leakage_penalty(decoder_logits, executor_features)
        losses["leakage"] = leakage_loss

        # ── 5. Abstention encouragement ───────────────────────────────
        # For low-support cases, encourage MAYBE / INSUFFICIENT_EVIDENCE
        abstention_loss = self._abstention_encouragement(
            primitives, executor_features
        )
        losses["abstention"] = abstention_loss

        # ── Total ────────────────────────────────────────────────────
        total = (
            self.config.lambda_primitive * prim_loss
            + self.config.lambda_generation * gen_loss
            + self.config.lambda_policy * policy_loss
            + self.config.lambda_leakage * leakage_loss
            + self.config.lambda_abstention * abstention_loss
        )
        losses["total"] = total

        return losses

    def _policy_violation_penalty(
        self, primitives: PrimitiveOutput, executor_features: torch.Tensor
    ) -> torch.Tensor:
        """Penalize when primitives contradict executor policy signals."""
        # Policy flags are last 3 dims: [too_specific, too_general, normal]
        too_specific = executor_features[:, -3]  # (B,)
        min_pass = executor_features[:, -4]       # (B,)

        # Soft penalty: if too_specific, answer should lean toward POLICY_BLOCKED (idx 5)
        answer_probs = F.softmax(primitives.answer_logits, dim=-1)
        policy_blocked_prob = answer_probs[:, 5]  # prob of POLICY_BLOCKED

        # Penalize when too_specific=1 but policy_blocked_prob is low
        penalty_specific = too_specific * (1.0 - policy_blocked_prob)

        # Penalize when min_pass=0 but INSUFFICIENT_EVIDENCE (idx 3) prob is low
        insuff_prob = answer_probs[:, 3]
        penalty_insuff = (1.0 - min_pass) * (1.0 - insuff_prob)

        return (penalty_specific.mean() + penalty_insuff.mean()) / 2.0

    def _leakage_penalty(
        self, decoder_logits: torch.Tensor, executor_features: torch.Tensor
    ) -> torch.Tensor:
        """Approximate leakage penalty.

        Uses squared correlation between mean decoder logits and executor features
        as a proxy for mutual information leakage.
        """
        # Reduce decoder logits to a summary vector: (B, V) via mean over time
        dec_summary = decoder_logits.mean(dim=1)  # (B, V)

        # Use top-k of dec_summary for efficiency
        k = min(32, dec_summary.size(-1))
        dec_reduced = dec_summary[:, :k]  # (B, k)

        # Compute correlation matrix between dec_reduced and executor_features
        # Normalize
        d = dec_reduced - dec_reduced.mean(dim=0, keepdim=True)
        e = executor_features - executor_features.mean(dim=0, keepdim=True)

        d_std = d.std(dim=0, keepdim=True).clamp(min=1e-6)
        e_std = e.std(dim=0, keepdim=True).clamp(min=1e-6)

        d_norm = d / d_std
        e_norm = e / e_std

        # Cross-correlation: (k, executor_dim)
        corr = (d_norm.T @ e_norm) / max(d_norm.size(0), 1)

        # Penalty: mean squared correlation
        return (corr ** 2).mean()

    def _abstention_encouragement(
        self, primitives: PrimitiveOutput, executor_features: torch.Tensor
    ) -> torch.Tensor:
        """Encourage abstention (MAYBE/INSUFFICIENT) when support is low."""
        support_ratio = executor_features[:, 0]  # (B,)
        low_support_mask = (support_ratio < 0.03).float()

        answer_probs = F.softmax(primitives.answer_logits, dim=-1)
        # Abstention answers: MAYBE(2), INSUFFICIENT(3), INCONSISTENT(4), BLOCKED(5)
        abstention_prob = answer_probs[:, 2:].sum(dim=-1)

        # Penalize confident YES/NO when support is low
        penalty = low_support_mask * (1.0 - abstention_prob)
        return penalty.mean()
