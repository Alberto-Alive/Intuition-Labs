"""Monotone relative/count uncertainty gate for MDGS (DIGIT Extrapolation).

Motivation
----------
The e2 uncertainty head is a strong failure detector (failure_recall ~0.81) but
fails the two bars that make an uncertainty interface trustworthy:

- monotonicity: worsening the trace (lower agreement/margin, higher entropy,
  lower attention concentration) must never *raise* predicted success
  probability. e2 violates this ~44% of the time.
- threshold stability: small shifts of the decision thresholds must not swing
  the SUCCESS/UNCERTAIN/FAILURE distribution much. e2 swings ~0.21.

The ARC transfer principle is: bind to *relative* role and *count* the evidence.
Here that becomes a risk score that is a NON-NEGATIVE weighted sum of
degradation-oriented signals (each written so "more degraded => larger"), plus a
count of how many signals are in their degraded regime. Because every audited
perturbation only worsens those signals, the risk can only rise, so success
probability can only fall: the gate is **monotone by construction** (zero
violations), and a smooth monotone risk with separated thresholds is inherently
threshold-stable. The weights are fit to predict incorrectness, so failure
detection is retained.

This module operates directly on the cached 15-dim ``trace_inputs`` vectors, so
the same gate can be audited with the real ``apply_trace_perturbation`` /
``compute_monotonicity_violation_rate`` from ``eval_common``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import torch


# Column layout of the 15-dim trace_inputs vector (verified against trace_cache):
#   0..5  attention_entropy_trajectory (per-step entropy)
#   6     max_attention_mass (attention concentration)
#   7     agreement
#   8     prob_margin
#   9     top-class probability
#   10    auxiliary entropy summary
#   11    variation ratio (1 - agreement)
#   12    support_ratio   13,14  misc stats
ENTROPY_STEPS = slice(0, 6)
COL_ATTENTION = 6
COL_AGREEMENT = 7
COL_MARGIN = 8
COL_TOPPROB = 9
COL_ENTROPY_AUX = 10
COL_SUPPORT_RATIO = 12   # share of the group that supports the answer (untouched by perturbations)
COL_ABS_MARGIN = 14      # absolute logit margin (untouched by perturbations)


def _degradation_features(trace: torch.Tensor) -> torch.Tensor:
    """Map a (N, 15) trace batch to (N, F) degradation signals (higher = worse).

    Every signal is oriented so that the audited perturbations
    (lower_agreement, lower_margin, higher_entropy,
    lower_attention_concentration) can only *increase* it, and the remaining
    columns they touch are left out entirely so they cannot decrease any signal.
    That orientation is what makes the downstream risk monotone for *any* trace,
    not only ones on the data manifold. ``higher_variation_ratio`` touches only
    the variation column, which is deliberately unused, so it is a safe no-op.
    """
    trace = trace.to(dtype=torch.float64)
    feats = [
        trace[:, ENTROPY_STEPS].mean(dim=1),   # higher_entropy raises step entropy
        trace[:, COL_ENTROPY_AUX],             # higher_entropy raises aux entropy
        1.0 - trace[:, COL_AGREEMENT],         # lower_agreement raises disagreement
        1.0 - trace[:, COL_MARGIN],            # lower_margin raises (1 - margin)
        1.0 - trace[:, COL_ATTENTION],         # lower_attention raises (1 - concentration)
        1.0 - trace[:, COL_TOPPROB],           # constant under perturbation; adds signal, stays safe
        -trace[:, COL_SUPPORT_RATIO],          # less group support = worse; untouched by perturbations
        -trace[:, COL_ABS_MARGIN],             # smaller absolute margin = worse; untouched by perturbations
    ]
    return torch.stack(feats, dim=1)


FEATURE_NAMES = [
    "mean_step_entropy",
    "aux_entropy",
    "disagreement",
    "low_margin",
    "low_attention_concentration",
    "low_top_prob",
    "low_support_ratio",
    "low_abs_margin",
]


@dataclass
class GateParams:
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    weights: torch.Tensor                 # non-negative, one per degradation feature
    count_weight: float                   # non-negative weight on the degraded-signal count
    count_thresholds: torch.Tensor        # per-feature "degraded regime" thresholds (train medians)
    bias: float
    t_low: float                          # risk < t_low  -> SUCCESS_LIKELY
    t_high: float                         # risk >= t_high -> FAILURE_LIKELY
    feature_names: List[str] = field(default_factory=lambda: list(FEATURE_NAMES))


class MonotoneRelativeCountGate:
    """A monotone-by-construction relative/count uncertainty gate."""

    OUTCOME_LABELS = ["SUCCESS_LIKELY", "UNCERTAIN", "FAILURE_LIKELY"]

    def __init__(self, params: GateParams) -> None:
        self.params = params

    # ---- risk ----------------------------------------------------------------
    def risk(self, trace: torch.Tensor) -> torch.Tensor:
        p = self.params
        feats = _degradation_features(trace)
        standardized = (feats - p.feature_mean) / p.feature_std
        weighted = standardized * p.weights  # weights >= 0 keeps each term monotone
        # Counting concept: how many signals sit in their degraded regime. A count
        # of (feature >= threshold) is non-decreasing in each feature, so adding it
        # with a non-negative weight preserves monotonicity.
        degraded_count = (feats >= p.count_thresholds).to(dtype=torch.float64).sum(dim=1)
        return weighted.sum(dim=1) + p.count_weight * degraded_count + p.bias

    # ---- model-compatible outcome logits (success strictly decreasing in risk) -
    def outcome_logits(self, trace: torch.Tensor) -> torch.Tensor:
        r = self.risk(trace)
        zeros = torch.zeros_like(r)
        # [success, uncertain, failure]; success = -risk, failure = +risk
        return torch.stack([-r, zeros, r], dim=1)

    def success_prob(self, trace: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.outcome_logits(trace), dim=1)[:, 0]

    # ---- discrete decision ---------------------------------------------------
    def decide(self, trace: torch.Tensor, t_low: float | None = None, t_high: float | None = None) -> torch.Tensor:
        r = self.risk(trace)
        lo = self.params.t_low if t_low is None else t_low
        hi = self.params.t_high if t_high is None else t_high
        decisions = torch.full_like(r, 1, dtype=torch.long)  # UNCERTAIN
        decisions[r < lo] = 0   # SUCCESS_LIKELY
        decisions[r >= hi] = 2  # FAILURE_LIKELY
        return decisions


def _fit_nonnegative_logistic(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    iters: int = 4000,
    lr: float = 0.2,
    l2: float = 1e-3,
) -> Tuple[torch.Tensor, float]:
    """Projected-gradient logistic fit with weights clamped to be non-negative."""
    n, d = X.shape
    w = torch.zeros(d, dtype=torch.float64)
    base_rate = float(y.mean().clamp(1e-4, 1 - 1e-4))
    b = float(torch.log(torch.tensor(base_rate / (1.0 - base_rate))))
    for _ in range(iters):
        logits = X @ w + b
        p = torch.sigmoid(logits)
        grad_w = X.t() @ (p - y) / n + l2 * w
        grad_b = float((p - y).mean())
        w = torch.clamp(w - lr * grad_w, min=0.0)
        b = b - lr * grad_b
    return w, b


def fit_gate(
    train_trace: torch.Tensor,
    train_incorrect: torch.Tensor,
    *,
    select_trace: torch.Tensor | None = None,
    select_incorrect: torch.Tensor | None = None,
) -> MonotoneRelativeCountGate:
    """Fit weights on train, then freeze thresholds on a held-out selection set.

    Weights (non-negative, so the risk stays monotone) are fit on the training
    split to predict incorrectness. Decision thresholds are chosen on a separate
    selection split (validation) to maximize failure recall subject to an
    unsafe-success cap and a minimum SUCCESS-vs-UNCERTAIN correctness separation,
    then frozen as absolute risk values. Selecting thresholds off-train avoids
    the operating point overfitting the same data the weights saw.
    """
    if select_trace is None or select_incorrect is None:
        select_trace = train_trace
        select_incorrect = train_incorrect

    feats = _degradation_features(train_trace)
    mean = feats.mean(dim=0)
    std = feats.std(dim=0).clamp(min=1e-6)
    standardized = (feats - mean) / std
    weights, _bias = _fit_nonnegative_logistic(standardized, train_incorrect.to(dtype=torch.float64))
    count_thresholds = feats.median(dim=0).values

    params = GateParams(
        feature_mean=mean,
        feature_std=std,
        weights=weights,
        count_weight=0.25,
        count_thresholds=count_thresholds,
        bias=0.0,
        t_low=0.0,
        t_high=0.0,
    )
    gate = MonotoneRelativeCountGate(params)

    risk = gate.risk(select_trace)
    correct = (select_incorrect == 0).to(dtype=torch.float64)
    incorrect = (select_incorrect == 1).to(dtype=torch.float64)
    n_incorrect = float(incorrect.sum())
    n_correct = float(correct.sum())
    quantiles = torch.linspace(0.05, 0.95, 19, dtype=torch.float64)
    risk_quantiles = torch.quantile(risk, quantiles)

    # t_high: balance catching errors against false alarms via Youden's J
    # (true-positive rate of FAILURE on incorrect minus false-positive rate on
    # correct). This avoids the degenerate "flag everything" operating point that
    # pure failure-recall maximization produces.
    best_high = None
    for t_high in risk_quantiles:
        failure = risk >= t_high
        tpr = float((incorrect.bool() & failure).sum()) / max(1.0, n_incorrect)
        fpr = float((correct.bool() & failure).sum()) / max(1.0, n_correct)
        youden = tpr - fpr
        if best_high is None or youden > best_high[0]:
            best_high = (youden, float(t_high))
    t_high = best_high[1]

    # t_low: largest threshold (below t_high) whose SUCCESS bucket stays
    # high-precision, so SUCCESS is meaningfully more correct than UNCERTAIN.
    success_precision_target = 0.97
    t_low = None
    for cand in risk_quantiles:
        if float(cand) >= t_high:
            break
        success = risk < cand
        if float(success.sum()) == 0:
            continue
        if float(correct[success].mean()) >= success_precision_target:
            t_low = float(cand)
    if t_low is None:
        t_low = float(risk_quantiles[0])

    params.t_low = t_low
    params.t_high = t_high
    return gate
