"""Baseline systems A and B adapted for Adult Income data.

Baseline A: executor -> 3-class rule (YES/NO/MAYBE) -> template
Baseline B: executor -> full rule-based primitives -> template
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from .executor import AdultSafeExecutor
from ..data.vocabulary import Vocabulary
from ..data.ground_truth import (
    ANSWER_LABELS, SUPPORT_LABELS, TEMPLATE_RESPONSES,
)

TEMPLATES_A = {
    0: "evidence appears supportive of higher income.",
    1: "evidence appears unsupportive of higher income.",
    2: "income signal is inconclusive.",
}

DEFAULT_TEMPLATE = "result is inconclusive."


def _rule_based_answer(executor_features: torch.Tensor) -> torch.Tensor:
    """Map executor features to YES/NO/MAYBE (3 classes)."""
    B = executor_features.size(0)
    answers = torch.zeros(B, dtype=torch.long, device=executor_features.device)

    for i in range(B):
        income_rate = executor_features[i, 0].item()
        min_pass = executor_features[i, -4].item()

        if min_pass < 0.5:
            answers[i] = 2  # MAYBE
        elif income_rate > 0.29:  # above overall rate (~0.24) + margin
            answers[i] = 0  # YES
        elif income_rate < 0.19:  # below overall rate - margin
            answers[i] = 1  # NO
        else:
            answers[i] = 2  # MAYBE
    return answers


def _rule_based_full_primitives(
    executor_features: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Rule-based mapping to all 4 primitives."""
    B = executor_features.size(0)
    device = executor_features.device

    answers = torch.zeros(B, dtype=torch.long, device=device)
    supports = torch.zeros(B, dtype=torch.long, device=device)
    confidences = torch.zeros(B, dtype=torch.long, device=device)
    risks = torch.zeros(B, dtype=torch.long, device=device)

    for i in range(B):
        feat = executor_features[i]
        income_rate = feat[0].item()
        min_pass = feat[-4].item()
        too_specific = feat[-3].item()

        # Variance bucket
        var_start = 1
        var_bucket = feat[var_start:var_start + 4].argmax().item()

        # Confidence bucket
        conf_start = 5
        conf_bucket = feat[conf_start:conf_start + 3].argmax().item()

        # Support ratio from income_rate (approximate)
        # Higher income_rate doesn't directly map to support, but we use
        # confidence bucket as a proxy
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
            risks[i] = 2
            continue

        if var_bucket >= 3:
            answers[i] = 4  # INCONSISTENT_SIGNAL
            supports[i] = 0
            confidences[i] = 0
            risks[i] = 1
            continue

        # Map income_rate to answer
        if income_rate > 0.29:
            answers[i] = 0  # YES
        elif income_rate < 0.19:
            answers[i] = 1  # NO
        else:
            answers[i] = 2  # MAYBE

        # Map confidence bucket to support level
        supports[i] = min(conf_bucket + 1, 3)
        confidences[i] = conf_bucket
        risks[i] = 0 if conf_bucket >= 2 else 1

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
        self.executor = AdultSafeExecutor(
            min_group_size=config.min_group_size,
            variance_buckets=config.variance_buckets,
            confidence_buckets=config.confidence_buckets,
        )

    @torch.no_grad()
    def forward(self, query_fields, private_dataset) -> Dict[str, Any]:
        executor_features = self.executor(query_fields, private_dataset)
        answers = _rule_based_answer(executor_features)

        texts = []
        for i in range(query_fields.size(0)):
            a = answers[i].item()
            texts.append(TEMPLATES_A.get(a, DEFAULT_TEMPLATE))

        return {
            "answers": answers,
            "texts": texts,
            "executor_features": executor_features,
        }


class BaselineB(nn.Module):
    """Baseline B: executor -> full rule-based primitives -> template."""

    def __init__(self, config, vocab: Vocabulary):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.executor = AdultSafeExecutor(
            min_group_size=config.min_group_size,
            variance_buckets=config.variance_buckets,
            confidence_buckets=config.confidence_buckets,
        )

    @torch.no_grad()
    def forward(self, query_fields, private_dataset) -> Dict[str, Any]:
        executor_features = self.executor(query_fields, private_dataset)
        primitives = _rule_based_full_primitives(executor_features)

        texts = []
        for i in range(query_fields.size(0)):
            a = primitives["answer"][i].item()
            s = primitives["support"][i].item()
            answer_label = ANSWER_LABELS[a]
            support_label = SUPPORT_LABELS[s]

            key = f"{answer_label}_{support_label}"
            templates = TEMPLATE_RESPONSES.get(key)
            if templates:
                texts.append(templates[0])
            elif answer_label in TEMPLATE_RESPONSES:
                texts.append(TEMPLATE_RESPONSES[answer_label][0])
            else:
                texts.append(DEFAULT_TEMPLATE)

        return {
            "primitives": primitives,
            "texts": texts,
            "executor_features": executor_features,
        }
