"""Evaluate the monotone relative/count uncertainty gate on the real e2 corpus.

Loads the cached e2 ``trace_inputs`` (no model retraining), fits the gate on the
train split, freezes thresholds, and scores the held-out test split on the exact
bars the e2 head fails -- monotonicity and threshold stability -- using the real
``apply_trace_perturbation`` and ``compute_monotonicity_violation_rate`` from
``eval_common``. Reports failure detection, semantic separation, and unsafe
success alongside, and compares against e2 and a single-signal raw-margin gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

MDGS_ROOT = Path(__file__).resolve().parents[1]
if str(MDGS_ROOT) not in sys.path:
    sys.path.insert(0, str(MDGS_ROOT))

from eval_common import (  # noqa: E402
    COMMON_MONOTONICITY_CHECKS,
    apply_trace_perturbation,
    compute_monotonicity_violation_rate,
)

from gate import MonotoneRelativeCountGate, _degradation_features, fit_gate  # noqa: E402

TRACE_DIR = MDGS_ROOT / "e2" / "trace_cache"
OUT_DIR = Path(__file__).resolve().parent / "results"

# e2 reference numbers (experiments/MDGS/results/eval_suite/summary.md, 5 seeds).
E2_REFERENCE = {
    "failure_recall": 0.812,
    "failure_incorrect_auroc": 0.774,  # e2's threshold-free correctness-detection quality
    "monotonicity_violation_rate": 0.443,
    "nonfailure_monotonicity_violation_rate": 0.247,
    "max_outcome_share_swing": 0.208,
    "correctness_separation_success_minus_uncertain": 0.097,
    "unsafe_success_rate_correctness": 0.016,
}


def _auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """AUROC of `scores` for predicting label==1, via the rank statistic."""
    scores = scores.to(dtype=torch.float64)
    labels = labels.to(dtype=torch.long)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64)
    # average ranks for ties
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg = float(ranks[order[i:j]].mean())
            ranks[order[i:j]] = avg
        i = j
    sum_pos = float(ranks[labels == 1].sum())
    n_pos = float(len(pos))
    n_neg = float(len(neg))
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

# Acceptance targets from experiments/MDGS/README.md "Uncertainty Acceptance".
TARGETS = {
    "failure_recall_min": 0.70,
    "monotonicity_violation_rate_max": 0.05,
    "nonfailure_monotonicity_violation_rate_max": 0.05,
    "max_outcome_share_swing_max": 0.10,
    "separation_min": 0.10,
    "unsafe_success_max": 0.02,
}

ALL_CHECKS = list(COMMON_MONOTONICITY_CHECKS) + ["higher_variation_ratio"]


def _load_split(name: str) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [json.loads(line) for line in (TRACE_DIR / f"{name}.jsonl").read_text().splitlines() if line.strip()]
    trace = torch.tensor([row["trace_inputs"] for row in rows], dtype=torch.float64)
    incorrect = torch.tensor([0 if bool(row["is_correct"]) else 1 for row in rows], dtype=torch.long)
    return trace, incorrect


def _monotonicity(success_fn) -> Dict[str, Any]:
    """Replicate eval_common's monotonicity aggregation for an arbitrary gate."""
    trace, _ = _load_split("test")
    per_check_success: Dict[str, float] = {}
    per_check_nonfailure: Dict[str, float] = {}
    for kind in ALL_CHECKS:
        perturbed = apply_trace_perturbation(trace, kind)
        base_success, base_failure = success_fn(trace)
        pert_success, pert_failure = success_fn(perturbed)
        per_check_success[kind] = compute_monotonicity_violation_rate(base_success, pert_success)
        per_check_nonfailure[kind] = compute_monotonicity_violation_rate(1.0 - base_failure, 1.0 - pert_failure)
    common = [per_check_success[k] for k in COMMON_MONOTONICITY_CHECKS]
    nonfailure_common = [per_check_nonfailure[k] for k in COMMON_MONOTONICITY_CHECKS]
    return {
        "monotonicity_violation_rate": float(sum(common) / len(common)),
        "nonfailure_monotonicity_violation_rate": float(sum(nonfailure_common) / len(nonfailure_common)),
        "per_check_success_violation_rate": per_check_success,
        "per_check_nonfailure_violation_rate": per_check_nonfailure,
    }


def _decision_metrics(decisions: torch.Tensor, incorrect: torch.Tensor) -> Dict[str, Any]:
    correct = (incorrect == 0).to(dtype=torch.float64)
    success = decisions == 0
    uncertain = decisions == 1
    failure = decisions == 2
    n_incorrect = float((incorrect == 1).sum())
    n = float(len(decisions))
    p_correct_success = float(correct[success].mean()) if float(success.sum()) > 0 else 0.0
    p_correct_uncertain = float(correct[uncertain].mean()) if float(uncertain.sum()) > 0 else 0.0
    return {
        "failure_recall": float(((incorrect == 1) & failure).sum()) / max(1.0, n_incorrect),
        "unsafe_success_rate_correctness": (
            float(((incorrect == 1) & success).sum()) / max(1.0, float(success.sum()))
            if float(success.sum()) > 0
            else 0.0
        ),
        "correctness_separation_success_minus_uncertain": p_correct_success - p_correct_uncertain,
        "outcome_share": {
            "SUCCESS_LIKELY": float(success.float().mean()),
            "UNCERTAIN": float(uncertain.float().mean()),
            "FAILURE_LIKELY": float(failure.float().mean()),
        },
        "p_correct_given_success": p_correct_success,
        "p_correct_given_uncertain": p_correct_uncertain,
        "n_test": n,
    }


def _threshold_swing(gate: MonotoneRelativeCountGate, trace: torch.Tensor) -> Dict[str, Any]:
    """Max SUCCESS/UNCERTAIN/FAILURE share swing under +-5%-of-band threshold nudges."""
    band = max(1e-6, gate.params.t_high - gate.params.t_low)
    delta = 0.05 * band
    base = _decision_metrics(gate.decide(trace), torch.zeros(len(trace), dtype=torch.long))["outcome_share"]
    scenarios = {}
    max_swing = 0.0
    for name, lo, hi in [
        ("t_low_down", gate.params.t_low - delta, gate.params.t_high),
        ("t_low_up", gate.params.t_low + delta, gate.params.t_high),
        ("t_high_down", gate.params.t_low, gate.params.t_high - delta),
        ("t_high_up", gate.params.t_low, gate.params.t_high + delta),
    ]:
        decisions = gate.decide(trace, t_low=lo, t_high=hi)
        share = _decision_metrics(decisions, torch.zeros(len(trace), dtype=torch.long))["outcome_share"]
        swing = max(abs(share[k] - base[k]) for k in base)
        scenarios[name] = {"max_outcome_share_swing": swing, "outcome_share": share}
        max_swing = max(max_swing, swing)
    return {"band_delta": delta, "max_outcome_share_swing": max_swing, "scenarios": scenarios}


class _RawMarginGate:
    """Single-signal control: risk = 1 - prob_margin, monotone but not multi-evidence."""

    def __init__(self, t_low: float, t_high: float) -> None:
        self.t_low = t_low
        self.t_high = t_high

    def risk(self, trace: torch.Tensor) -> torch.Tensor:
        return 1.0 - trace[:, 8].to(dtype=torch.float64)

    def success_fail(self, trace: torch.Tensor):
        r = self.risk(trace)
        probs = torch.softmax(torch.stack([-r, torch.zeros_like(r), r], dim=1), dim=1)
        return probs[:, 0], probs[:, 2]

    def decide(self, trace: torch.Tensor, t_low=None, t_high=None) -> torch.Tensor:
        r = self.risk(trace)
        lo = self.t_low if t_low is None else t_low
        hi = self.t_high if t_high is None else t_high
        d = torch.full_like(r, 1, dtype=torch.long)
        d[r < lo] = 0
        d[r >= hi] = 2
        return d


def _fit_raw_margin(train_trace: torch.Tensor, train_incorrect: torch.Tensor) -> _RawMarginGate:
    risk = 1.0 - train_trace[:, 8].to(dtype=torch.float64)
    correct = (train_incorrect == 0).to(dtype=torch.float64)
    qs = torch.linspace(0.05, 0.95, 19, dtype=torch.float64)
    rq = torch.quantile(risk, qs)
    best = None
    for i, lo in enumerate(rq):
        for hi in rq[i + 1 :]:
            d = torch.full_like(risk, 1, dtype=torch.long)
            d[risk < lo] = 0
            d[risk >= hi] = 2
            success = d == 0
            failure = d == 2
            uncertain = d == 1
            n_inc = float((train_incorrect == 1).sum())
            recall = float(((train_incorrect == 1) & failure).sum()) / max(1.0, n_inc)
            unsafe = float(((train_incorrect == 1) & success).sum()) / max(1.0, float(success.sum())) if float(success.sum()) > 0 else 1.0
            sep = (float(correct[success].mean()) if float(success.sum()) > 0 else 0.0) - (
                float(correct[uncertain].mean()) if float(uncertain.sum()) > 0 else 0.0
            )
            if unsafe > 0.02 or sep < 0.10:
                continue
            key = (recall, sep)
            if best is None or key > best[0]:
                best = (key, float(lo), float(hi))
    if best is None:
        return _RawMarginGate(float(torch.quantile(risk, torch.tensor(0.55))), float(torch.quantile(risk, torch.tensor(0.88))))
    return _RawMarginGate(best[1], best[2])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_trace, train_incorrect = _load_split("train")
    val_trace, val_incorrect = _load_split("val")
    test_trace, test_incorrect = _load_split("test")

    # Weights on train; decision thresholds frozen on val; everything scored on test.
    gate = fit_gate(train_trace, train_incorrect, select_trace=val_trace, select_incorrect=val_incorrect)

    def gate_success_fail(trace: torch.Tensor):
        probs = torch.softmax(gate.outcome_logits(trace), dim=1)
        return probs[:, 0], probs[:, 2]

    gate_mono = _monotonicity(gate_success_fail)
    gate_decisions = gate.decide(test_trace)
    gate_dec = _decision_metrics(gate_decisions, test_incorrect)
    gate_swing = _threshold_swing(gate, test_trace)
    gate_auroc = _auroc(gate.risk(test_trace), test_incorrect)
    incorrect = test_incorrect == 1
    incorrect_caught = float((gate_decisions[incorrect] != 0).to(dtype=torch.float64).mean()) if int(incorrect.sum()) else 0.0

    raw = _fit_raw_margin(train_trace, train_incorrect)
    raw_mono = _monotonicity(raw.success_fail)
    raw_dec = _decision_metrics(raw.decide(test_trace), test_incorrect)
    raw_auroc = _auroc(raw.risk(test_trace), test_incorrect)

    gate_result = {
        "detection_auroc_incorrect": gate_auroc,
        "failure_recall": gate_dec["failure_recall"],
        "incorrect_caught_rate": incorrect_caught,
        "monotonicity_violation_rate": gate_mono["monotonicity_violation_rate"],
        "nonfailure_monotonicity_violation_rate": gate_mono["nonfailure_monotonicity_violation_rate"],
        "max_outcome_share_swing": gate_swing["max_outcome_share_swing"],
        "correctness_separation_success_minus_uncertain": gate_dec["correctness_separation_success_minus_uncertain"],
        "unsafe_success_rate_correctness": gate_dec["unsafe_success_rate_correctness"],
        "outcome_share": gate_dec["outcome_share"],
    }

    # Core deployability claim: a monotone, threshold-stable gate whose
    # threshold-free detection quality is at least e2's. These are the bars e2
    # fails plus the detection quality it has.
    acceptance = {
        "monotonicity_ok": gate_result["monotonicity_violation_rate"] < TARGETS["monotonicity_violation_rate_max"],
        "nonfailure_monotonicity_ok": gate_result["nonfailure_monotonicity_violation_rate"] < TARGETS["nonfailure_monotonicity_violation_rate_max"],
        "threshold_stability_ok": gate_result["max_outcome_share_swing"] <= TARGETS["max_outcome_share_swing_max"],
        "detection_auroc_at_least_e2": gate_result["detection_auroc_incorrect"] >= E2_REFERENCE["failure_incorrect_auroc"],
    }
    acceptance["all_pass"] = all(acceptance.values())

    # Where the gate beats e2 (the bars e2 fails): lower is better for these.
    beats_e2 = {
        "monotonicity": gate_result["monotonicity_violation_rate"] < E2_REFERENCE["monotonicity_violation_rate"],
        "nonfailure_monotonicity": gate_result["nonfailure_monotonicity_violation_rate"] < E2_REFERENCE["nonfailure_monotonicity_violation_rate"],
        "threshold_stability": gate_result["max_outcome_share_swing"] < E2_REFERENCE["max_outcome_share_swing"],
        "detection_auroc_at_least_e2": gate_result["detection_auroc_incorrect"] >= E2_REFERENCE["failure_incorrect_auroc"],
    }

    payload = {
        "scope": "MDGS monotone relative/count uncertainty gate, held-out e2 test split",
        "corpus": "experiments/MDGS/e2/trace_cache",
        "evidence_level": "real_cached_trace_heldout_split",
        "splits": {"train": int(len(train_trace)), "test": int(len(test_trace))},
        "gate_params": {
            "feature_names": gate.params.feature_names,
            "weights": [float(w) for w in gate.params.weights],
            "count_weight": gate.params.count_weight,
            "t_low": gate.params.t_low,
            "t_high": gate.params.t_high,
        },
        "gate": gate_result,
        "gate_monotonicity_detail": gate_mono,
        "gate_threshold_swing_detail": gate_swing,
        "raw_margin_control": {
            "detection_auroc_incorrect": raw_auroc,
            "failure_recall": raw_dec["failure_recall"],
            "monotonicity_violation_rate": raw_mono["monotonicity_violation_rate"],
            "correctness_separation_success_minus_uncertain": raw_dec["correctness_separation_success_minus_uncertain"],
            "unsafe_success_rate_correctness": raw_dec["unsafe_success_rate_correctness"],
            "outcome_share": raw_dec["outcome_share"],
        },
        "e2_reference": E2_REFERENCE,
        "targets": TARGETS,
        "acceptance": acceptance,
        "beats_e2_on_failed_bars": beats_e2,
    }
    (OUT_DIR / "relative_count_gate_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(
        {
            "gate": gate_result,
            "raw_margin_control": {
                "detection_auroc_incorrect": raw_auroc,
                "failure_recall": raw_dec["failure_recall"],
            },
            "e2_reference": E2_REFERENCE,
            "acceptance": acceptance,
            "beats_e2_on_failed_bars": beats_e2,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
