"""Baseline systems for the Extrapolation smoke test."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from ..data.ground_truth import (
    ATTENTION_PATTERN_LABELS,
    GroundTruthComputer,
    OUTCOME_LABELS,
    TEMPLATE_RESPONSES,
)

BASELINE_A_TEMPLATES = {
    0: "attention trajectory suggests likely success.",
    1: "attention trajectory remains uncertain.",
    2: "attention trajectory suggests likely failure.",
}


class BaselineA(nn.Module):
    """Baseline A: direct outcome-only rule mapping."""

    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab

    @torch.no_grad()
    def forward(self, query_fields: torch.Tensor, private_dataset) -> Dict[str, Any]:
        gt = GroundTruthComputer(
            private_data=private_dataset,
            min_group_size=self.config.min_group_size,
            income_margin=self.config.income_margin,
        )

        outcomes = []
        texts = []
        for i in range(query_fields.size(0)):
            derived = gt.compute(query_fields[i], seed=i)
            outcome = derived["outcome"]
            outcomes.append(outcome)
            texts.append(BASELINE_A_TEMPLATES.get(outcome, BASELINE_A_TEMPLATES[1]))

        return {
            "outcomes": torch.tensor(outcomes, dtype=torch.long, device=query_fields.device),
            "texts": texts,
        }


class BaselineB(nn.Module):
    """Baseline B: full rule-based mapping to all four entropy heads."""

    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab

    @torch.no_grad()
    def forward(self, query_fields: torch.Tensor, private_dataset) -> Dict[str, Any]:
        gt = GroundTruthComputer(
            private_data=private_dataset,
            min_group_size=self.config.min_group_size,
            income_margin=self.config.income_margin,
        )

        trajectory_shapes = []
        attention_patterns = []
        confidences = []
        outcomes = []
        texts = []

        for i in range(query_fields.size(0)):
            derived = gt.compute(query_fields[i], seed=i)
            trajectory_shapes.append(derived["trajectory_shape"])
            attention_patterns.append(derived["attention_pattern"])
            confidences.append(derived["confidence"])
            outcomes.append(derived["outcome"])

            outcome_label = OUTCOME_LABELS[derived["outcome"]]
            pattern_label = ATTENTION_PATTERN_LABELS[derived["attention_pattern"]]
            key = f"{outcome_label}_{pattern_label}"
            templates = TEMPLATE_RESPONSES.get(key, TEMPLATE_RESPONSES.get(f"{outcome_label}_MIXED", []))
            texts.append(templates[0] if templates else "attention trajectory remains uncertain.")

        return {
            "primitives": {
                "trajectory_shape": torch.tensor(trajectory_shapes, dtype=torch.long, device=query_fields.device),
                "attention_pattern": torch.tensor(attention_patterns, dtype=torch.long, device=query_fields.device),
                "confidence": torch.tensor(confidences, dtype=torch.long, device=query_fields.device),
                "outcome": torch.tensor(outcomes, dtype=torch.long, device=query_fields.device),
            },
            "texts": texts,
        }
