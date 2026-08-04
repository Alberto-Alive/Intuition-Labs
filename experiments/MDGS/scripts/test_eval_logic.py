"""Logic checks for the Extrapolation evaluation stack."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
E2_ROOT = ROOT / "e2"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(E2_ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_eval_suite
from eval_common import compute_monotonicity_violation_rate, compute_success_safety_metrics, relabel_trace_record
from extrapolation.config import Config
from extrapolation.data.trace_dataset import TraceEntropyDataset, load_trace_records, trace_inputs_from_record
from extrapolation.executor_schema import (
    EXECUTOR_FEATURE_DIM,
    EXECUTOR_FEATURE_INDEX,
    TRACE_INPUT_DIM,
)
from extrapolation.losses import DIGITLoss
from extrapolation.models.bottleneck import EntropyBottleneckHead
from extrapolation.models.bottleneck import PrimitiveOutput
from extrapolation.models.executor import EntropyTrajectoryExecutor
from extrapolation.data.vocabulary import Vocabulary


def _assert_close(lhs: float, rhs: float, *, tol: float = 1e-6) -> None:
    assert math.isclose(lhs, rhs, rel_tol=0.0, abs_tol=tol), f"{lhs} != {rhs}"


def _test_schema_dimensions() -> None:
    config = Config()
    assert config.trace_input_dim == TRACE_INPUT_DIM
    assert config.executor_feature_dim == EXECUTOR_FEATURE_DIM

    vocab = Vocabulary()
    records = load_trace_records(E2_ROOT / "trace_cache" / "train.jsonl")
    dataset = TraceEntropyDataset(records, vocab=vocab, config=config)
    _, trace_inputs, _, _ = dataset[0]
    assert trace_inputs.shape[0] == TRACE_INPUT_DIM
    assert len(records[0]["trace_inputs"]) in {9, TRACE_INPUT_DIM}


def _test_trace_input_upgrade() -> None:
    records = load_trace_records(E2_ROOT / "trace_cache" / "train.jsonl")
    upgraded = trace_inputs_from_record(records[0])
    assert len(upgraded) == TRACE_INPUT_DIM


def _test_executor_mapping() -> None:
    executor = EntropyTrajectoryExecutor(
        low_agreement_threshold=0.55,
        low_margin_threshold=0.15,
    )
    sample = torch.tensor(
        [[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.45, 0.05, 0.80, 0.25, 0.55, 0.12, 0.18, 0.03]],
        dtype=torch.float32,
    )
    features = executor._from_trace_inputs(sample)
    assert features.shape[-1] == EXECUTOR_FEATURE_DIM

    _assert_close(float(features[0, EXECUTOR_FEATURE_INDEX["mean_entropy"]]), 0.35)
    _assert_close(float(features[0, EXECUTOR_FEATURE_INDEX["slope"]]), 0.50)
    _assert_close(float(features[0, EXECUTOR_FEATURE_INDEX["max_attention_mass"]]), 0.70)
    _assert_close(float(features[0, EXECUTOR_FEATURE_INDEX["agreement"]]), 0.45)
    _assert_close(float(features[0, EXECUTOR_FEATURE_INDEX["variation_ratio"]]), 0.55)
    _assert_close(float(features[0, EXECUTOR_FEATURE_INDEX["low_margin_flag"]]), 1.0)
    _assert_close(float(features[0, EXECUTOR_FEATURE_INDEX["unstable_flag"]]), 1.0)


def _test_policy_penalty_uses_named_flags() -> None:
    config = Config()
    criterion = DIGITLoss(config)

    high_conf_logits = torch.tensor([[0.0, 0.0, 8.0]], dtype=torch.float32)
    focused_logits = torch.tensor([[8.0, 0.0, 0.0]], dtype=torch.float32)
    success_logits = torch.tensor([[8.0, 0.0, 0.0]], dtype=torch.float32)
    dummy_logits = torch.tensor([[8.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    primitives = PrimitiveOutput(
        trajectory_shape_logits=dummy_logits,
        attention_pattern_logits=focused_logits,
        confidence_logits=high_conf_logits,
        outcome_logits=success_logits,
        trajectory_shape_discrete=torch.zeros(1, 4),
        attention_pattern_discrete=torch.zeros(1, 3),
        confidence_discrete=torch.zeros(1, 3),
        outcome_discrete=torch.zeros(1, 3),
    )

    safe_features = torch.zeros(1, EXECUTOR_FEATURE_DIM)
    safe_features[0, EXECUTOR_FEATURE_INDEX["mean_entropy"]] = 0.10
    safe_features[0, EXECUTOR_FEATURE_INDEX["std_entropy"]] = 0.01
    safe_features[0, EXECUTOR_FEATURE_INDEX["entropy_range"]] = 0.02

    unsafe_features = safe_features.clone()
    unsafe_features[0, EXECUTOR_FEATURE_INDEX["low_margin_flag"]] = 1.0
    unsafe_features[0, EXECUTOR_FEATURE_INDEX["unstable_flag"]] = 1.0

    safe_penalty = float(criterion._policy_violation_penalty(primitives, safe_features))
    unsafe_penalty = float(criterion._policy_violation_penalty(primitives, unsafe_features))
    assert unsafe_penalty > safe_penalty, f"unsafe={unsafe_penalty} safe={safe_penalty}"


def _test_trace_mode_ignores_query_context() -> None:
    torch.manual_seed(0)
    head = EntropyBottleneckHead(
        d_model=8,
        executor_dim=EXECUTOR_FEATURE_DIM,
        trace_hidden_dim=8,
        hidden_dim=16,
        dropout=0.0,
    )
    executor_features = torch.randn(1, EXECUTOR_FEATURE_DIM)
    z_left = torch.randn(1, 8)
    z_right = torch.randn(1, 8)

    left = head(z_left, executor_features, mode="soft", tau=1.0, use_query_features=False)
    right = head(z_right, executor_features, mode="soft", tau=1.0, use_query_features=False)

    assert torch.allclose(left.trajectory_shape_logits, right.trajectory_shape_logits)
    assert torch.allclose(left.attention_pattern_logits, right.attention_pattern_logits)
    assert torch.allclose(left.confidence_logits, right.confidence_logits)
    assert torch.allclose(left.outcome_logits, right.outcome_logits)


def _test_label_rule_invariants() -> None:
    config = Config()
    with open(E2_ROOT / "trace_cache" / "metadata.json") as f:
        metadata = json.load(f)
    metadata.setdefault("success_min_correctness_prob", float(config.trace_success_correctness_threshold))
    metadata.setdefault("failure_max_correctness_prob", float(config.trace_failure_correctness_threshold))
    metadata.setdefault("success_min_support_ratio", float(config.trace_success_min_support_ratio))
    metadata.setdefault("success_min_group_size", int(config.trace_success_min_group_size))

    confident = {
        "trace_summary": {"mean_entropy": 0.10},
        "max_attention_mass": 0.95,
        "agreement": float(metadata["agreement_high"]),
        "prob_margin": float(metadata["margin_high"]),
        "correctness_probability": float(metadata["success_min_correctness_prob"]) + 0.05,
        "stats": {
            "support_ratio": float(metadata["success_min_support_ratio"]) + 0.05,
            "n": int(metadata["success_min_group_size"]) + 10,
        },
    }
    uncertain = {
        "trace_summary": {"mean_entropy": 0.55},
        "max_attention_mass": 0.50,
        "agreement": float(metadata["agreement_high"]) - 0.01,
        "prob_margin": float(metadata["margin_high"]) - 0.01,
        "correctness_probability": float(metadata["success_min_correctness_prob"]) + 0.05,
        "stats": {
            "support_ratio": float(metadata["success_min_support_ratio"]) + 0.05,
            "n": int(metadata["success_min_group_size"]) + 10,
        },
    }
    wrong = {
        "trace_summary": {"mean_entropy": 0.10},
        "max_attention_mass": 0.95,
        "agreement": float(metadata["agreement_high"]),
        "prob_margin": float(metadata["margin_high"]),
        "correctness_probability": float(metadata["failure_max_correctness_prob"]) - 0.05,
        "stats": {
            "support_ratio": float(metadata["success_min_support_ratio"]) + 0.05,
            "n": int(metadata["success_min_group_size"]) + 10,
        },
    }
    low_support = {
        "trace_summary": {"mean_entropy": 0.10},
        "max_attention_mass": 0.95,
        "agreement": float(metadata["agreement_high"]),
        "prob_margin": float(metadata["margin_high"]),
        "correctness_probability": float(metadata["success_min_correctness_prob"]) + 0.05,
        "stats": {
            "support_ratio": max(float(metadata["success_min_support_ratio"]) - 0.005, 0.0),
            "n": int(metadata["success_min_group_size"]) + 10,
        },
    }

    assert relabel_trace_record(confident, metadata)["outcome"] == 0
    assert relabel_trace_record(uncertain, metadata)["outcome"] == 1
    assert relabel_trace_record(wrong, metadata)["outcome"] == 2
    assert relabel_trace_record(low_support, metadata)["outcome"] == 1


def _test_monotonicity_helper() -> None:
    base = torch.tensor([0.20, 0.50, 0.90], dtype=torch.float32)
    perturbed = torch.tensor([0.10, 0.55, 0.90], dtype=torch.float32)
    _assert_close(compute_monotonicity_violation_rate(base, perturbed), 1.0 / 3.0)


def _test_semantic_probability_metrics() -> None:
    outcome_probabilities = [
        [0.95, 0.04, 0.01],
        [0.10, 0.85, 0.05],
        [0.05, 0.10, 0.85],
    ]
    outcome_predictions = [0, 1, 2]
    outcome_targets = [0, 1, 2]
    is_correct = [1, 1, 0]

    metrics = compute_success_safety_metrics(
        outcome_probabilities,
        outcome_predictions,
        outcome_targets,
        is_correct,
    )
    assert metrics["unsafe_success_rate_correctness"] == 0.0
    assert metrics["success_label_auroc"] > 0.99
    assert metrics["nonfailure_correct_auroc"] > 0.99
    assert metrics["failure_incorrect_auroc"] > 0.99


def _fake_run(seed: int, *, trajectory_acc: float, confidence_macro_f1: float, outcome_macro_f1: float, failure_recall: float,
              unsafe_success_rate_correctness: float, success_label_auroc: float, success_label_brier: float,
              monotonicity_violation_rate: float, nonfailure_monotonicity_violation_rate: float) -> dict:
    return {
        "train_loop_seed": seed,
        "test_metrics": {
            "trajectory_acc": trajectory_acc,
            "trajectory_macro_f1": trajectory_acc,
            "pattern_acc": 0.9,
            "pattern_macro_f1": 0.9,
            "confidence_acc": confidence_macro_f1,
            "confidence_macro_f1": confidence_macro_f1,
            "outcome_acc": outcome_macro_f1,
            "outcome_macro_f1": outcome_macro_f1,
            "failure_recall": failure_recall,
            "failure_likely_recall": failure_recall,
            "joint_acc": 0.5,
            "unsafe_success_rate_label": unsafe_success_rate_correctness,
            "unsafe_success_rate_correctness": unsafe_success_rate_correctness,
            "success_precision_vs_is_correct": 1.0 - unsafe_success_rate_correctness,
            "success_auroc": 0.7,
            "success_auprc": 0.8,
            "success_brier": 0.4,
            "success_ece_10bin": 0.3,
            "success_label_auroc": success_label_auroc,
            "success_label_auprc": success_label_auroc,
            "success_label_brier": success_label_brier,
            "success_label_ece_10bin": success_label_brier,
            "nonfailure_correct_auroc": 0.8,
            "nonfailure_correct_auprc": 0.8,
            "nonfailure_correct_brier": 0.1,
            "nonfailure_correct_ece_10bin": 0.1,
            "failure_incorrect_auroc": 0.8,
            "failure_incorrect_auprc": 0.8,
            "failure_incorrect_brier": 0.1,
            "failure_incorrect_ece_10bin": 0.1,
        },
        "monotonicity": {
            "monotonicity_violation_rate": monotonicity_violation_rate,
            "all_supported_monotonicity_violation_rate": monotonicity_violation_rate,
            "nonfailure_monotonicity_violation_rate": nonfailure_monotonicity_violation_rate,
            "all_supported_nonfailure_monotonicity_violation_rate": nonfailure_monotonicity_violation_rate,
        },
        "acceptance": {
            "trajectory_acc_ok": True,
            "outcome_macro_f1_ok": True,
            "failure_recall_ok": True,
        },
    }


def _test_suite_summary_uses_e2_audit_and_baselines() -> None:
    run_summaries = {
        "e1": [
            _fake_run(
                123,
                trajectory_acc=0.80,
                confidence_macro_f1=0.90,
                outcome_macro_f1=0.70,
                failure_recall=0.40,
                unsafe_success_rate_correctness=0.10,
                success_label_auroc=0.70,
                success_label_brier=0.20,
                monotonicity_violation_rate=0.20,
                nonfailure_monotonicity_violation_rate=0.20,
            )
        ],
        "e2": [
            _fake_run(
                123,
                trajectory_acc=0.82,
                confidence_macro_f1=0.91,
                outcome_macro_f1=0.75,
                failure_recall=0.46,
                unsafe_success_rate_correctness=0.03,
                success_label_auroc=0.82,
                success_label_brier=0.14,
                monotonicity_violation_rate=0.12,
                nonfailure_monotonicity_violation_rate=0.08,
            )
        ],
    }
    variant_details = {
        "e1": {
            "label_audit": {
                "p_is_correct_given_outcome": {
                    "SUCCESS_LIKELY": 1.0,
                    "UNCERTAIN": 1.0,
                    "FAILURE_LIKELY": 0.0,
                },
                "correctness_separation_success_minus_uncertain": 0.0,
            },
            "threshold_sensitivity": {
                "max_outcome_share_swing": 0.05,
                "label_revision_trigger": True,
            },
            "baselines": {
                "correctness": {"logistic_regression": {"auroc": 0.60}},
                "label_prediction": {"entropy_threshold": {"outcome_accuracy": 0.40}},
            },
        },
        "e2": {
            "label_audit": {
                "p_is_correct_given_outcome": {
                    "SUCCESS_LIKELY": 0.98,
                    "UNCERTAIN": 0.85,
                    "FAILURE_LIKELY": 0.30,
                },
                "correctness_separation_success_minus_uncertain": 0.13,
            },
            "threshold_sensitivity": {
                "max_outcome_share_swing": 0.08,
                "label_revision_trigger": False,
            },
            "baselines": {
                "correctness": {"logistic_regression": {"auroc": 0.81}},
                "label_prediction": {"entropy_threshold": {"outcome_accuracy": 0.55}},
            },
        },
    }

    summary = run_eval_suite._build_suite_summary(
        seeds=[123],
        fingerprint="abc123",
        run_summaries=run_summaries,
        variant_details=variant_details,
    )

    assert summary["label_audit"] == variant_details["e2"]["label_audit"]
    assert summary["threshold_sensitivity"]["label_revision_trigger"] is False
    assert summary["baselines"]["correctness"] == variant_details["e2"]["baselines"]["correctness"]
    assert summary["comparison"]["safety_win"] is True
    assert summary["comparison"]["nonfailure_monotonicity_ok"] is False


def main() -> None:
    _test_schema_dimensions()
    _test_trace_input_upgrade()
    _test_executor_mapping()
    _test_policy_penalty_uses_named_flags()
    _test_trace_mode_ignores_query_context()
    _test_label_rule_invariants()
    _test_monotonicity_helper()
    _test_semantic_probability_metrics()
    _test_suite_summary_uses_e2_audit_and_baselines()
    print("Extrapolation evaluation logic checks PASSED")


if __name__ == "__main__":
    main()
