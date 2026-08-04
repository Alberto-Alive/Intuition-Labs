"""Loss functions for the DIGIT model.

Five loss components:
  1. Primitive classification — cross-entropy on each primitive head
  2. Decoder generation — cross-entropy on token generation
  3. Policy violation penalty — penalize bottleneck/executor disagreement
  4. Leakage penalty — mutual information proxy (squared correlation)
  5. Abstention encouragement — reward caution on low-support queries
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .models.bottleneck import PrimitiveOutput


class DIGITLoss(nn.Module):
    """Combined loss for training the DIGIT model."""

    def __init__(self, config, pad_idx: int = 0, class_weights: Dict = None):
        super().__init__()
        self.config = config
        self.pad_idx = pad_idx

        # Weighted cross-entropy for class imbalance
        if class_weights:
            self.answer_ce = nn.CrossEntropyLoss(
                weight=class_weights.get("answer"), reduction="mean"
            )
            self.support_ce = nn.CrossEntropyLoss(
                weight=class_weights.get("support"), reduction="mean"
            )
            self.confidence_ce = nn.CrossEntropyLoss(
                weight=class_weights.get("confidence"), reduction="mean"
            )
            self.risk_ce = nn.CrossEntropyLoss(
                weight=class_weights.get("risk"), reduction="mean"
            )
        else:
            self.answer_ce = nn.CrossEntropyLoss(reduction="mean")
            self.support_ce = nn.CrossEntropyLoss(reduction="mean")
            self.confidence_ce = nn.CrossEntropyLoss(reduction="mean")
            self.risk_ce = nn.CrossEntropyLoss(reduction="mean")

        self.gen_ce = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="mean")

    def forward(
        self,
        decoder_logits: torch.Tensor,
        primitives: PrimitiveOutput,
        target_ids: torch.Tensor,
        prim_targets: torch.Tensor,
        executor_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        losses = {}

        # 1. Primitive classification (weighted per class)
        prim_loss = (
            self.answer_ce(primitives.answer_logits, prim_targets[:, 0])
            + self.support_ce(primitives.support_logits, prim_targets[:, 1])
            + self.confidence_ce(primitives.confidence_logits, prim_targets[:, 2])
            + self.risk_ce(primitives.risk_logits, prim_targets[:, 3])
        ) / 4.0
        losses["primitive"] = prim_loss

        # 2. Decoder generation
        gen_targets = target_ids[:, 1:]
        B, T_minus_1, V = decoder_logits.shape
        gen_loss = self.gen_ce(
            decoder_logits.reshape(B * T_minus_1, V),
            gen_targets.reshape(B * T_minus_1),
        )
        losses["generation"] = gen_loss

        # 3. Policy violation penalty
        policy_loss = self._policy_violation_penalty(primitives, executor_features)
        losses["policy"] = policy_loss

        # 4. Leakage penalty
        leakage_loss = self._leakage_penalty(decoder_logits, executor_features)
        losses["leakage"] = leakage_loss

        # 5. Abstention encouragement
        abstention_loss = self._abstention_encouragement(
            primitives, executor_features
        )
        losses["abstention"] = abstention_loss

        # Total
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
        too_specific = executor_features[:, -3]
        min_pass = executor_features[:, -4]

        answer_probs = F.softmax(primitives.answer_logits, dim=-1)

        # POLICY_BLOCKED when too_specific
        n_answer = primitives.answer_logits.size(-1)
        if n_answer >= 6:
            policy_blocked_prob = answer_probs[:, 5]
            penalty_specific = too_specific * (1.0 - policy_blocked_prob)
        else:
            penalty_specific = torch.tensor(0.0, device=executor_features.device)

        # INSUFFICIENT_EVIDENCE when min_pass fails
        if n_answer >= 4:
            insuff_prob = answer_probs[:, 3]
            penalty_insuff = (1.0 - min_pass) * (1.0 - insuff_prob)
        else:
            penalty_insuff = torch.tensor(0.0, device=executor_features.device)

        return (penalty_specific.mean() + penalty_insuff.mean()) / 2.0

    def _leakage_penalty(
        self, decoder_logits: torch.Tensor, executor_features: torch.Tensor
    ) -> torch.Tensor:
        dec_summary = decoder_logits.mean(dim=1)
        k = min(32, dec_summary.size(-1))
        dec_reduced = dec_summary[:, :k]

        d = dec_reduced - dec_reduced.mean(dim=0, keepdim=True)
        e = executor_features - executor_features.mean(dim=0, keepdim=True)

        d_std = d.std(dim=0, keepdim=True).clamp(min=1e-6)
        e_std = e.std(dim=0, keepdim=True).clamp(min=1e-6)

        d_norm = d / d_std
        e_norm = e / e_std

        corr = (d_norm.T @ e_norm) / max(d_norm.size(0), 1)
        return (corr ** 2).mean()

    def _abstention_encouragement(
        self, primitives: PrimitiveOutput, executor_features: torch.Tensor
    ) -> torch.Tensor:
        income_rate = executor_features[:, 0]
        min_pass = executor_features[:, -4]

        # Low support: group passed min size but income rate is near-zero
        # (almost no records matched)
        low_support_mask = ((min_pass < 0.5)).float()

        answer_probs = F.softmax(primitives.answer_logits, dim=-1)
        # Abstention answers: MAYBE(2)+
        abstention_prob = answer_probs[:, 2:].sum(dim=-1)
        penalty = low_support_mask * (1.0 - abstention_prob)
        return penalty.mean()
