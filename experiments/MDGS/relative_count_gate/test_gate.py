"""Tests for the monotone relative/count uncertainty gate.

The central guarantee is structural: worsening the trace can never raise the
gate's predicted success probability. We assert it directly against the real
``apply_trace_perturbation`` used by the evaluation suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
MDGS_ROOT = HERE.parent
for path in (str(HERE), str(MDGS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from eval_common import COMMON_MONOTONICITY_CHECKS, apply_trace_perturbation  # noqa: E402

from gate import GateParams, MonotoneRelativeCountGate, _degradation_features, fit_gate  # noqa: E402

ALL_CHECKS = list(COMMON_MONOTONICITY_CHECKS) + ["higher_variation_ratio"]


def _random_traces(n: int = 64, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 15, generator=g, dtype=torch.float64)


def _make_gate() -> MonotoneRelativeCountGate:
    trace = _random_traces(256, seed=1)
    # Synthetic incorrectness correlated with degradation so weights are positive.
    feats = _degradation_features(trace)
    score = (feats.mean(dim=1) - feats.mean()) / feats.std()
    incorrect = (torch.sigmoid(score) > 0.5).to(dtype=torch.long)
    return fit_gate(trace, incorrect)


def test_weights_are_non_negative() -> None:
    gate = _make_gate()
    assert torch.all(gate.params.weights >= 0.0)
    assert gate.params.count_weight >= 0.0


def test_worsening_trace_never_raises_success_prob() -> None:
    gate = _make_gate()
    trace = _random_traces(128, seed=2)
    base = gate.success_prob(trace)
    for kind in ALL_CHECKS:
        perturbed = apply_trace_perturbation(trace, kind)
        worsened = gate.success_prob(perturbed)
        # Allow a tiny numerical tolerance; no example may increase.
        assert torch.all(worsened <= base + 1e-9), f"monotonicity violated under {kind}"


def test_risk_is_monotone_under_each_perturbation() -> None:
    gate = _make_gate()
    trace = _random_traces(128, seed=3)
    base_risk = gate.risk(trace)
    for kind in ALL_CHECKS:
        perturbed = apply_trace_perturbation(trace, kind)
        assert torch.all(gate.risk(perturbed) >= base_risk - 1e-9), f"risk decreased under {kind}"


def test_thresholds_are_ordered() -> None:
    gate = _make_gate()
    assert gate.params.t_low < gate.params.t_high


if __name__ == "__main__":
    test_weights_are_non_negative()
    test_worsening_trace_never_raises_success_prob()
    test_risk_is_monotone_under_each_perturbation()
    test_thresholds_are_ordered()
    print("all gate tests passed")
