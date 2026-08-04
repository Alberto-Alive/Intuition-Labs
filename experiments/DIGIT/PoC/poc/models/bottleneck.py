"""Primitive Bottleneck Head — the discrete information gate.

Identical to the feasibility experiment. This module is data-agnostic:
it maps (z_q + executor_features) to 4 discrete categorical primitives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, NamedTuple


class PrimitiveOutput(NamedTuple):
    """Container for bottleneck outputs."""
    answer_logits: torch.Tensor     # (B, num_answer)
    support_logits: torch.Tensor    # (B, num_support)
    confidence_logits: torch.Tensor # (B, num_confidence)
    risk_logits: torch.Tensor       # (B, num_risk)
    answer_discrete: torch.Tensor   # (B, num_answer)
    support_discrete: torch.Tensor  # (B, num_support)
    confidence_discrete: torch.Tensor  # (B, num_confidence)
    risk_discrete: torch.Tensor     # (B, num_risk)


class PrimitiveBottleneckHead(nn.Module):
    """Produces discrete primitives from query encoding + executor features."""

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

    def _discretize(self, logits: torch.Tensor, mode: str, tau: float) -> torch.Tensor:
        if mode == "hard":
            idx = logits.argmax(dim=-1)
            return F.one_hot(idx, logits.size(-1)).float()
        elif mode == "gumbel":
            return F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        elif mode == "straight_through":
            soft = F.softmax(logits / tau, dim=-1)
            idx = logits.argmax(dim=-1)
            hard = F.one_hot(idx, logits.size(-1)).float()
            return hard - soft.detach() + soft
        elif mode == "soft":
            return F.softmax(logits / tau, dim=-1)
        else:
            raise ValueError(f"Unknown discretization mode: {mode}")

    def get_primitive_indices(self, prim_output: PrimitiveOutput) -> Dict[str, torch.Tensor]:
        return {
            "answer": prim_output.answer_discrete.argmax(dim=-1),
            "support": prim_output.support_discrete.argmax(dim=-1),
            "confidence": prim_output.confidence_discrete.argmax(dim=-1),
            "risk": prim_output.risk_discrete.argmax(dim=-1),
        }
