"""Local schema constants for E31 trace inputs."""

from __future__ import annotations

from typing import Dict


TRAJECTORY_LEN = 6

TRACE_INPUT_NAMES = tuple(
    [f"trajectory_{idx}" for idx in range(TRAJECTORY_LEN)]
    + [
        "max_attention_mass",
        "agreement",
        "prob_margin",
        "max_softmax",
        "predictive_entropy",
        "variation_ratio",
        "support_ratio",
        "n_norm",
        "abs_margin",
        "attention_top2_mass",
        "attention_effective_support",
        "attention_gini",
        "attention_concentration_drift",
    ]
)

TRACE_INPUT_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(TRACE_INPUT_NAMES)}
TRACE_INPUT_DIM = len(TRACE_INPUT_NAMES)
