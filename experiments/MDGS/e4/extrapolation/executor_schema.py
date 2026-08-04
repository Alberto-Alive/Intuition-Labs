"""Named schema for E3 trace inputs and executor features."""

from __future__ import annotations

from typing import Dict

import torch


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
    ]
)

EXECUTOR_FEATURE_NAMES = tuple(
    [f"trajectory_{idx}" for idx in range(TRAJECTORY_LEN)]
    + [
        "mean_entropy",
        "slope",
        "std_entropy",
        "entropy_range",
        "max_attention_mass",
        "agreement",
        "prob_margin",
        "max_softmax",
        "predictive_entropy",
        "variation_ratio",
        "support_ratio",
        "n_norm",
        "abs_margin",
        "low_margin_flag",
        "unstable_flag",
    ]
)

TRACE_INPUT_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(TRACE_INPUT_NAMES)}
EXECUTOR_FEATURE_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(EXECUTOR_FEATURE_NAMES)}

TRACE_INPUT_DIM = len(TRACE_INPUT_NAMES)
EXECUTOR_FEATURE_DIM = len(EXECUTOR_FEATURE_NAMES)
UNCERTAINTY_FEATURE_SLICE = slice(TRAJECTORY_LEN, EXECUTOR_FEATURE_DIM)
UNCERTAINTY_FEATURE_DIM = EXECUTOR_FEATURE_DIM - TRAJECTORY_LEN
TRAJECTORY_SUMMARY_SLICE = slice(0, EXECUTOR_FEATURE_INDEX["max_attention_mass"])


def trace_input_column(values: torch.Tensor, name: str) -> torch.Tensor:
    """Return a named trace-input column."""
    return values[:, TRACE_INPUT_INDEX[name]]


def executor_feature_column(values: torch.Tensor, name: str) -> torch.Tensor:
    """Return a named executor-feature column."""
    return values[:, EXECUTOR_FEATURE_INDEX[name]]
