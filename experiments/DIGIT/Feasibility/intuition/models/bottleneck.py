"""Components 3 & 4: Primitive Bottleneck Head.

Input: concatenation of z_q (query representation) and safe executor aggregate features.
Output: discrete constrained primitives.

Implements two modes:
  - Hard discrete: argmax over categorical logits
  - Differentiable: Gumbel-Softmax / straight-through estimator

Primitive types:
  - answer: {YES, NO, MAYBE, INSUFFICIENT_EVIDENCE, INCONSISTENT_SIGNAL, POLICY_BLOCKED}
  - support: {VERY_LOW, LOW, MEDIUM, HIGH}
  - confidence: {LOW, MEDIUM, HIGH}
  - risk: {LOW, MEDIUM, HIGH}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, NamedTuple


class PrimitiveOutput(NamedTuple):
    """Container for bottleneck outputs."""
    answer_logits: torch.Tensor    # (B, 6)
    support_logits: torch.Tensor   # (B, 4)
    confidence_logits: torch.Tensor  # (B, 3)
    risk_logits: torch.Tensor      # (B, 3)
    # Discrete or soft one-hot representations
    answer_discrete: torch.Tensor  # (B, 6)
    support_discrete: torch.Tensor  # (B, 4)
    confidence_discrete: torch.Tensor  # (B, 3)
    risk_discrete: torch.Tensor    # (B, 3)


class PrimitiveBottleneckHead(nn.Module):
    """Produces discrete primitives from query encoding + executor features.

    The bottleneck ensures only discrete approved primitives cross to the decoder.
    """

    def __init__(
        self,
        d_model: int = 256,
        executor_dim: int = 12,
        num_answer: int = 6,
        num_support: int = 4,
        num_confidence: int = 3,
        num_risk: int = 3,
        hidden_dim: int = 256,
    ):
        super().__init__()
        input_dim = d_model + executor_dim

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Per-primitive classification heads
        self.answer_head = nn.Linear(hidden_dim, num_answer)
        self.support_head = nn.Linear(hidden_dim, num_support)
        self.confidence_head = nn.Linear(hidden_dim, num_confidence)
        self.risk_head = nn.Linear(hidden_dim, num_risk)

        self.num_answer = num_answer
        self.num_support = num_support
        self.num_confidence = num_confidence
        self.num_risk = num_risk

    def forward(
        self,
        z_q: torch.Tensor,
        executor_features: torch.Tensor,
        mode: str = "gumbel",
        tau: float = 1.0,
    ) -> PrimitiveOutput:
        """
        Args:
            z_q: (B, d_model) — encoded query.
            executor_features: (B, executor_dim) — safe aggregates.
            mode: "hard" for argmax, "gumbel" for Gumbel-Softmax,
                  "straight_through" for STE.
            tau: temperature for Gumbel-Softmax.

        Returns:
            PrimitiveOutput with logits and discrete representations.
        """
        x = torch.cat([z_q, executor_features], dim=-1)
        h = self.shared(x)

        answer_logits = self.answer_head(h)
        support_logits = self.support_head(h)
        confidence_logits = self.confidence_head(h)
        risk_logits = self.risk_head(h)

        answer_disc = self._discretize(answer_logits, mode, tau)
        support_disc = self._discretize(support_logits, mode, tau)
        confidence_disc = self._discretize(confidence_logits, mode, tau)
        risk_disc = self._discretize(risk_logits, mode, tau)

        return PrimitiveOutput(
            answer_logits=answer_logits,
            support_logits=support_logits,
            confidence_logits=confidence_logits,
            risk_logits=risk_logits,
            answer_discrete=answer_disc,
            support_discrete=support_disc,
            confidence_discrete=confidence_disc,
            risk_discrete=risk_disc,
        )

    def _discretize(
        self, logits: torch.Tensor, mode: str, tau: float
    ) -> torch.Tensor:
        """Convert logits to discrete (or approximately discrete) representation."""
        if mode == "hard":
            # Pure argmax — not differentiable
            idx = logits.argmax(dim=-1)
            return F.one_hot(idx, logits.size(-1)).float()

        elif mode == "gumbel":
            # Gumbel-Softmax with hard=True gives straight-through one-hot
            return F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)

        elif mode == "straight_through":
            # Straight-through estimator: forward uses argmax, backward uses softmax
            soft = F.softmax(logits / tau, dim=-1)
            idx = logits.argmax(dim=-1)
            hard = F.one_hot(idx, logits.size(-1)).float()
            # Straight-through: hard in forward, soft gradients in backward
            return hard - soft.detach() + soft

        elif mode == "soft":
            # Fully differentiable soft version (for debugging)
            return F.softmax(logits / tau, dim=-1)

        else:
            raise ValueError(f"Unknown discretization mode: {mode}")

    def get_primitive_indices(self, prim_output: PrimitiveOutput) -> Dict[str, torch.Tensor]:
        """Extract integer class indices from discrete representations."""
        return {
            "answer": prim_output.answer_discrete.argmax(dim=-1),
            "support": prim_output.support_discrete.argmax(dim=-1),
            "confidence": prim_output.confidence_discrete.argmax(dim=-1),
            "risk": prim_output.risk_discrete.argmax(dim=-1),
        }
