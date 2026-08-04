"""Baseline systems A and B.

Baseline A:
  - Deterministic executor
  - Single primitive: {YES, NO, MAYBE}
  - Template decoder (no learning)

Baseline B:
  - Deterministic executor
  - Richer primitive bottleneck (all 4 primitives)
  - Template decoder (no learning)

Both use the SafePrivateExecutor but skip learned components.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List

from .safe_executor import SafePrivateExecutor
from ..data.vocabulary import (
    Vocabulary, ANSWER_LABELS, SUPPORT_LABELS,
    CONFIDENCE_LABELS, RISK_LABELS,
)


# ─── Template decoder (shared by baselines) ──────────────────────────────────

TEMPLATES_A = {
    0: "evidence appears supportive.",           # YES
    1: "evidence appears unsupportive.",          # NO
    2: "signal is inconclusive.",                 # MAYBE
}

TEMPLATES_B = {
    # (answer, support, confidence, risk) -> template
    # Keyed by (answer_idx, support_idx)
    (0, 3): "evidence strongly supports this direction.",
    (0, 2): "evidence appears moderately supportive.",
    (0, 1): "weak positive signal detected. confidence is limited.",
    (0, 0): "weak positive signal detected. confidence is limited.",
    (1, 3): "evidence strongly contradicts this hypothesis.",
    (1, 2): "evidence appears moderately unsupportive.",
    (1, 1): "weak negative signal detected. confidence is limited.",
    (1, 0): "weak negative signal detected. confidence is limited.",
    (2, 3): "mixed findings warrant further investigation.",
    (2, 2): "mixed findings warrant further investigation.",
    (2, 1): "no clear pattern detected. evidence is insufficient.",
    (2, 0): "no clear pattern detected. evidence is insufficient.",
    (3, 0): "not enough records meet privacy threshold.",
    (4, 0): "signal is weak and inconsistent.",
    (5, 0): "the query is too specific to answer safely.",
}

DEFAULT_TEMPLATE = "result is inconclusive."


def _rule_based_answer(executor_features: torch.Tensor) -> torch.Tensor:
    """Simple rule: map executor features to YES/NO/MAYBE (3 classes)."""
    B = executor_features.size(0)
    answers = torch.zeros(B, dtype=torch.long, device=executor_features.device)

    for i in range(B):
        support_ratio = executor_features[i, 0].item()
        min_pass = executor_features[i, -4].item()  # min_group_size_pass

        if min_pass < 0.5:
            answers[i] = 2  # MAYBE (insufficient)
        elif support_ratio > 0.05:
            answers[i] = 0  # YES
        else:
            answers[i] = 1  # NO

    return answers


def _rule_based_full_primitives(
    executor_features: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Rule-based mapping from executor features to all 4 primitives."""
    B = executor_features.size(0)
    device = executor_features.device

    answers = torch.zeros(B, dtype=torch.long, device=device)
    supports = torch.zeros(B, dtype=torch.long, device=device)
    confidences = torch.zeros(B, dtype=torch.long, device=device)
    risks = torch.zeros(B, dtype=torch.long, device=device)

    for i in range(B):
        feat = executor_features[i]
        support_ratio = feat[0].item()
        min_pass = feat[-4].item()
        too_specific = feat[-3].item()
        too_general = feat[-2].item()

        # Determine variance bucket (which one-hot is active)
        var_start = 1
        var_end = var_start + 4
        var_bucket = feat[var_start:var_end].argmax().item()

        # Determine confidence bucket
        conf_start = var_end
        conf_end = conf_start + 3
        conf_bucket = feat[conf_start:conf_end].argmax().item()

        # Policy check
        if too_specific > 0.5:
            answers[i] = 5  # POLICY_BLOCKED
            supports[i] = 0
            confidences[i] = 0
            risks[i] = 2
            continue

        if min_pass < 0.5:
            answers[i] = 3  # INSUFFICIENT_EVIDENCE
            supports[i] = 0
            confidences[i] = 0
            risks[i] = 1
            continue

        if var_bucket >= 3:
            answers[i] = 4  # INCONSISTENT_SIGNAL
            supports[i] = 0
            confidences[i] = 0
            risks[i] = 1
            continue

        # Map support ratio to support level
        if support_ratio > 0.08:
            supports[i] = 3  # HIGH
        elif support_ratio > 0.04:
            supports[i] = 2  # MEDIUM
        elif support_ratio > 0.02:
            supports[i] = 1  # LOW
        else:
            supports[i] = 0  # VERY_LOW

        # Answer based on support
        if supports[i] >= 2:
            answers[i] = 0  # YES
        elif supports[i] == 1:
            answers[i] = 2  # MAYBE
        else:
            answers[i] = 1  # NO

        confidences[i] = conf_bucket
        risks[i] = 0 if supports[i] >= 2 else 1

    return {
        "answer": answers,
        "support": supports,
        "confidence": confidences,
        "risk": risks,
    }


class BaselineA(nn.Module):
    """Baseline A: executor -> 3-class rule -> template."""

    def __init__(self, config, vocab: Vocabulary):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.executor = SafePrivateExecutor(
            min_group_size=config.min_group_size,
            variance_buckets=config.variance_buckets,
            confidence_buckets=config.confidence_buckets,
        )

    @torch.no_grad()
    def forward(
        self,
        query_fields: torch.Tensor,
        group_ids: torch.Tensor,
        private_dataset,
    ) -> Dict[str, Any]:
        executor_features = self.executor(query_fields, group_ids, private_dataset)
        answers = _rule_based_answer(executor_features)

        texts = []
        token_ids_list = []
        for i in range(query_fields.size(0)):
            a = answers[i].item()
            text = TEMPLATES_A.get(a, DEFAULT_TEMPLATE)
            texts.append(text)
            token_ids_list.append(
                torch.tensor(self.vocab.encode(text), dtype=torch.long)
            )

        return {
            "answers": answers,
            "texts": texts,
            "token_ids": token_ids_list,
            "executor_features": executor_features,
        }


class BaselineB(nn.Module):
    """Baseline B: executor -> full rule-based primitives -> template."""

    def __init__(self, config, vocab: Vocabulary):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.executor = SafePrivateExecutor(
            min_group_size=config.min_group_size,
            variance_buckets=config.variance_buckets,
            confidence_buckets=config.confidence_buckets,
        )

    @torch.no_grad()
    def forward(
        self,
        query_fields: torch.Tensor,
        group_ids: torch.Tensor,
        private_dataset,
    ) -> Dict[str, Any]:
        executor_features = self.executor(query_fields, group_ids, private_dataset)
        primitives = _rule_based_full_primitives(executor_features)

        texts = []
        token_ids_list = []
        for i in range(query_fields.size(0)):
            a = primitives["answer"][i].item()
            s = primitives["support"][i].item()
            key = (a, s)
            # Fallback: try answer-only keys for special answers
            text = TEMPLATES_B.get(key)
            if text is None:
                text = TEMPLATES_B.get((a, 0), DEFAULT_TEMPLATE)
            texts.append(text)
            token_ids_list.append(
                torch.tensor(self.vocab.encode(text), dtype=torch.long)
            )

        return {
            "primitives": primitives,
            "texts": texts,
            "token_ids": token_ids_list,
            "executor_features": executor_features,
        }
