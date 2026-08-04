"""Shared evaluation helpers for DIGIT Extrapolation experiments."""

from __future__ import annotations

import json
import math
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_BENCHMARK_SEEDS = [123, 231, 347, 451, 569]
OUTCOME_LABELS = ["SUCCESS_LIKELY", "UNCERTAIN", "FAILURE_LIKELY"]
ATTENTION_PATTERN_LABELS = ["FOCUSED", "MIXED", "DIFFUSE"]
CONFIDENCE_LABELS = ["LOW", "MEDIUM", "HIGH"]
COMMON_MONOTONICITY_CHECKS = [
    "lower_agreement",
    "lower_margin",
    "higher_entropy",
    "lower_attention_concentration",
]


def json_default(value: Any) -> Any:
    """Convert common tensor / numpy / pathlib types to JSON-safe values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    """Write JSON payload with stable indentation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=json_default)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write newline-delimited JSON rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=json_default))
            f.write("\n")


def summarize_scalar_series(values: Sequence[float]) -> Dict[str, float]:
    """Return mean / std / min / max for a scalar series."""
    arr = np.asarray(list(values), dtype=np.float64)
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return {
            "count": int(arr.size),
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(arr.size),
        "mean": float(valid.mean()),
        "std": float(valid.std(ddof=0)),
        "min": float(valid.min()),
        "max": float(valid.max()),
    }


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _normalized_entropy(probabilities: Sequence[float]) -> float:
    values = [max(float(p), 1e-8) for p in probabilities]
    total = sum(values)
    if total <= 0.0:
        return 0.0
    probs = [value / total for value in values]
    entropy = -sum(prob * math.log(prob) for prob in probs)
    return float(entropy / max(math.log(len(probs)), 1e-8))


def _clip01(value: float) -> float:
    return float(max(min(value, 1.0), 0.0))


def _derived_attention_concentration_features(record: Dict[str, Any]) -> list[float]:
    trajectory = [float(value) for value in record["attention_entropy_trajectory"]]
    mean_entropy = _safe_mean(trajectory)
    entropy_range = (max(trajectory) - min(trajectory)) if trajectory else 0.0
    entropy_std = float(np.asarray(trajectory, dtype=np.float64).std(ddof=0)) if trajectory else 0.0
    max_attention = float(record.get("max_attention_mass", 0.0))
    prob_margin = float(record.get("prob_margin", 0.0))
    probs = [float(value) for value in record.get("prediction_probs", [])]
    predictive_entropy = _normalized_entropy(probs) if probs else _clip01(mean_entropy)

    top2_mass = record.get("attention_top2_mass")
    if top2_mass is None:
        top2_mass = max_attention + (1.0 - max_attention) * _clip01(
            0.40 * (1.0 - mean_entropy) + 0.20 * prob_margin
        )
    effective_support = record.get("attention_effective_support")
    if effective_support is None:
        effective_support = _clip01(
            0.65 * mean_entropy + 0.20 * predictive_entropy + 0.15 * (1.0 - max_attention)
        )
    attention_gini = record.get("attention_gini")
    if attention_gini is None:
        attention_gini = _clip01(
            0.60 * max_attention + 0.30 * (1.0 - mean_entropy) + 0.10 * prob_margin
        )
    concentration_drift = record.get("attention_concentration_drift")
    if concentration_drift is None:
        concentration_drift = _clip01(0.70 * entropy_range + 0.60 * entropy_std)

    return [
        _clip01(max(float(top2_mass), max_attention)),
        _clip01(float(effective_support)),
        _clip01(float(attention_gini)),
        _clip01(float(concentration_drift)),
    ]


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """Compute expected calibration error for binary probabilities."""
    if probabilities.size == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(probabilities.shape[0], 1)
    ece = 0.0
    for idx in range(bins):
        low = edges[idx]
        high = edges[idx + 1]
        if idx == bins - 1:
            mask = (probabilities >= low) & (probabilities <= high)
        else:
            mask = (probabilities >= low) & (probabilities < high)
        if not np.any(mask):
            continue
        confidence = float(probabilities[mask].mean())
        accuracy = float(labels[mask].mean())
        ece += (mask.sum() / total) * abs(confidence - accuracy)
    return float(ece)


def area_under_risk_coverage_curve(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Compute AURC by averaging retained-set risk over confidence-ranked coverage."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.size == 0 or labels.size == 0:
        return float("nan")

    finite_mask = np.isfinite(probabilities)
    if not np.any(finite_mask):
        return float("nan")

    probabilities = probabilities[finite_mask]
    labels = labels[finite_mask]
    order = np.argsort(-probabilities, kind="stable")
    sorted_labels = labels[order].astype(np.float64)
    coverage = np.arange(1, sorted_labels.size + 1, dtype=np.float64)
    cumulative_correct = np.cumsum(sorted_labels)
    risk = 1.0 - (cumulative_correct / coverage)
    return float(risk.mean())


def compute_binary_probability_metrics(probabilities: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute binary probability metrics against 0/1 labels."""
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.size == 0:
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "brier": float("nan"),
            "ece_10bin": float("nan"),
            "aurc": float("nan"),
        }

    unique = np.unique(labels)
    if unique.size < 2:
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = float(roc_auc_score(labels, probabilities))
        auprc = float(average_precision_score(labels, probabilities))

    brier = float(np.mean((probabilities - labels) ** 2))
    ece = expected_calibration_error(probabilities, labels, bins=10)
    aurc = area_under_risk_coverage_curve(probabilities, labels)
    return {
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "ece_10bin": ece,
        "aurc": aurc,
    }


def compute_success_safety_metrics(
    outcome_probabilities: np.ndarray,
    outcome_predictions: np.ndarray,
    outcome_targets: np.ndarray,
    is_correct: np.ndarray,
    success_class_idx: int = 0,
    failure_class_idx: int = 2,
) -> Dict[str, float]:
    """Compute safety metrics for success predictions against labels and correctness."""
    outcome_probabilities = np.asarray(outcome_probabilities, dtype=np.float64)
    outcome_predictions = np.asarray(outcome_predictions, dtype=np.int64)
    outcome_targets = np.asarray(outcome_targets, dtype=np.int64)
    is_correct = np.asarray(is_correct, dtype=np.int64)

    predicted_success = outcome_predictions == success_class_idx
    target_success = outcome_targets == success_class_idx
    predicted_success_count = int(predicted_success.sum())
    label_failures = int(np.logical_and(predicted_success, ~target_success).sum())
    correctness_failures = int(np.logical_and(predicted_success, is_correct == 0).sum())

    success_probabilities = outcome_probabilities[:, success_class_idx]
    failure_probabilities = outcome_probabilities[:, failure_class_idx]
    nonfailure_probabilities = 1.0 - failure_probabilities
    incorrect = 1 - is_correct

    success_correct_metrics = compute_binary_probability_metrics(success_probabilities, is_correct)
    success_label_metrics = compute_binary_probability_metrics(success_probabilities, target_success.astype(np.int64))
    nonfailure_correct_metrics = compute_binary_probability_metrics(nonfailure_probabilities, is_correct)
    failure_incorrect_metrics = compute_binary_probability_metrics(failure_probabilities, incorrect)
    success_precision = float(is_correct[predicted_success].mean()) if predicted_success_count else 0.0

    return {
        "unsafe_success_rate_label": _safe_rate(label_failures, predicted_success_count),
        "unsafe_success_rate_correctness": _safe_rate(correctness_failures, predicted_success_count),
        "success_precision_vs_is_correct": success_precision,
        "success_auroc": success_correct_metrics["auroc"],
        "success_auprc": success_correct_metrics["auprc"],
        "success_brier": success_correct_metrics["brier"],
        "success_ece_10bin": success_correct_metrics["ece_10bin"],
        "success_aurc": success_correct_metrics["aurc"],
        "success_label_auroc": success_label_metrics["auroc"],
        "success_label_auprc": success_label_metrics["auprc"],
        "success_label_brier": success_label_metrics["brier"],
        "success_label_ece_10bin": success_label_metrics["ece_10bin"],
        "success_label_aurc": success_label_metrics["aurc"],
        "nonfailure_correct_auroc": nonfailure_correct_metrics["auroc"],
        "nonfailure_correct_auprc": nonfailure_correct_metrics["auprc"],
        "nonfailure_correct_brier": nonfailure_correct_metrics["brier"],
        "nonfailure_correct_ece_10bin": nonfailure_correct_metrics["ece_10bin"],
        "nonfailure_correct_aurc": nonfailure_correct_metrics["aurc"],
        "failure_incorrect_auroc": failure_incorrect_metrics["auroc"],
        "failure_incorrect_auprc": failure_incorrect_metrics["auprc"],
        "failure_incorrect_brier": failure_incorrect_metrics["brier"],
        "failure_incorrect_ece_10bin": failure_incorrect_metrics["ece_10bin"],
        "failure_incorrect_aurc": failure_incorrect_metrics["aurc"],
        "predicted_success_rate": _safe_rate(predicted_success_count, len(outcome_predictions)),
    }


def _record_trace_inputs(record: Dict[str, Any]) -> list[float]:
    trajectory = [float(value) for value in record["attention_entropy_trajectory"]]
    expected_len = len(trajectory) + 9

    stored = [float(value) for value in record.get("trace_inputs", [])]
    if len(stored) >= expected_len:
        return stored

    probs = [float(value) for value in record.get("prediction_probs", [])]
    agreement = float(record.get("agreement", 0.0))
    stats = record.get("stats", {})
    return trajectory + [
        float(record["max_attention_mass"]),
        agreement,
        float(record["prob_margin"]),
        max(probs, default=0.0),
        _normalized_entropy(probs) if probs else 0.0,
        1.0 - agreement,
        float(stats.get("support_ratio", 0.0)),
        min(float(stats.get("n", 0.0)) / 200.0, 1.0),
        float(stats.get("abs_margin", 0.0)),
    ] + _derived_attention_concentration_features(record)


def _feature_arrays(records: Sequence[Dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    summary = np.array(
        [
            [
                float(record["trace_summary"]["mean_entropy"]),
                float(record["trace_summary"]["slope"]),
                float(record["trace_summary"]["std_entropy"]),
                float(record["trace_summary"]["entropy_range"]),
                float(record["max_attention_mass"]),
                float(record["agreement"]),
                float(record["prob_margin"]),
            ]
            for record in records
        ],
        dtype=np.float32,
    )
    full = np.array(
        [
            _record_trace_inputs(record)
            + [
                float(record["trace_summary"]["mean_entropy"]),
                float(record["trace_summary"]["slope"]),
                float(record["trace_summary"]["std_entropy"]),
                float(record["trace_summary"]["entropy_range"]),
            ]
            for record in records
        ],
        dtype=np.float32,
    )
    return summary, full


def evaluate_correctness_baselines(
    train_records: Sequence[Dict[str, Any]],
    test_records: Sequence[Dict[str, Any]],
    *,
    mlp_max_iter: int = 2000,
    mlp_early_stopping: bool = True,
) -> Dict[str, Dict[str, float | bool | int]]:
    """Train simple correctness predictors on the fixed trace features."""
    x_train_summary, x_train_full = _feature_arrays(train_records)
    x_test_summary, x_test_full = _feature_arrays(test_records)
    y_train = np.array([int(bool(record["is_correct"])) for record in train_records], dtype=np.int64)
    y_test = np.array([int(bool(record["is_correct"])) for record in test_records], dtype=np.int64)

    majority_label = int(Counter(y_train.tolist()).most_common(1)[0][0])
    majority_predictions = np.full_like(y_test, fill_value=majority_label)

    def _binary_summary(predictions: np.ndarray, probabilities: np.ndarray | None = None) -> Dict[str, float]:
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        }
        if probabilities is not None:
            metrics.update(compute_binary_probability_metrics(probabilities, y_test))
        return metrics

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    logistic.fit(x_train_summary, y_train)
    logistic_prob = logistic.predict_proba(x_test_summary)[:, 1]
    logistic_pred = (logistic_prob >= 0.5).astype(np.int64)

    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=mlp_max_iter,
            early_stopping=mlp_early_stopping,
            random_state=42,
        ),
    )
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        mlp.fit(x_train_full, y_train)
    mlp_prob = mlp.predict_proba(x_test_full)[:, 1]
    mlp_pred = (mlp_prob >= 0.5).astype(np.int64)
    mlp_inner = mlp.named_steps["mlpclassifier"]
    converged = not any(issubclass(warning.category, ConvergenceWarning) for warning in caught_warnings)

    return {
        "majority": {
            "positive_label": int(majority_label),
            **_binary_summary(majority_predictions),
        },
        "logistic_regression": _binary_summary(logistic_pred, logistic_prob),
        "mlp": {
            **_binary_summary(mlp_pred, mlp_prob),
            "converged": bool(converged),
            "n_iter": int(mlp_inner.n_iter_),
        },
    }


def _z_score(value: float, mean: float, std: float) -> float:
    denom = max(abs(float(std)), 1e-8)
    return float((float(value) - float(mean)) / denom)


def _label_attention_pattern(mean_entropy: float, max_attention: float, metadata: Dict[str, Any], *, threshold: float) -> int:
    mean_z = _z_score(mean_entropy, metadata["mean_entropy_mean"], metadata["mean_entropy_std"])
    attention_z = _z_score(max_attention, metadata["max_attention_mean"], metadata["max_attention_std"])
    if mean_z <= -threshold and attention_z >= threshold:
        return ATTENTION_PATTERN_LABELS.index("FOCUSED")
    if mean_z >= threshold and attention_z <= -threshold:
        return ATTENTION_PATTERN_LABELS.index("DIFFUSE")
    return ATTENTION_PATTERN_LABELS.index("MIXED")


def _label_confidence(agreement: float, prob_margin: float, metadata: Dict[str, Any]) -> int:
    if agreement >= float(metadata["agreement_high"]) and prob_margin >= float(metadata["margin_high"]):
        return CONFIDENCE_LABELS.index("HIGH")
    if agreement >= float(metadata["agreement_medium"]) and prob_margin >= float(metadata["margin_low"]):
        return CONFIDENCE_LABELS.index("MEDIUM")
    return CONFIDENCE_LABELS.index("LOW")


def _supports_success_outcome(record: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
    stats = record.get("stats", {})
    return (
        float(record.get("agreement", 0.0)) >= float(metadata["agreement_high"])
        and float(record.get("prob_margin", 0.0)) >= float(metadata["margin_high"])
        and float(stats.get("support_ratio", 0.0)) >= float(metadata.get("success_min_support_ratio", 0.0))
        and int(stats.get("n", 0)) >= int(metadata.get("success_min_group_size", 0))
    )


def relabel_trace_record(record: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, int]:
    """Recompute pattern / confidence / outcome under alternate thresholds."""
    pattern = _label_attention_pattern(
        float(record["trace_summary"]["mean_entropy"]),
        float(record["max_attention_mass"]),
        metadata,
        threshold=float(metadata["pattern_z_threshold"]),
    )
    confidence = _label_confidence(
        float(record["agreement"]),
        float(record["prob_margin"]),
        metadata,
    )
    if (
        "correctness_probability" in record
        and "success_min_correctness_prob" in metadata
        and "failure_max_correctness_prob" in metadata
    ):
        probability = float(record["correctness_probability"])
        if probability < float(metadata["failure_max_correctness_prob"]):
            outcome = OUTCOME_LABELS.index("FAILURE_LIKELY")
        elif probability >= float(metadata["success_min_correctness_prob"]) and _supports_success_outcome(record, metadata):
            outcome = OUTCOME_LABELS.index("SUCCESS_LIKELY")
        else:
            outcome = OUTCOME_LABELS.index("UNCERTAIN")
    else:
        if not bool(record["is_correct"]):
            outcome = OUTCOME_LABELS.index("FAILURE_LIKELY")
        elif confidence == CONFIDENCE_LABELS.index("LOW"):
            outcome = OUTCOME_LABELS.index("UNCERTAIN")
        else:
            outcome = OUTCOME_LABELS.index("SUCCESS_LIKELY")
    return {
        "attention_pattern": pattern,
        "confidence": confidence,
        "outcome": outcome,
    }


def compute_label_audit(records: Sequence[Dict[str, Any]], outcome_labels: Sequence[str] = OUTCOME_LABELS) -> Dict[str, Any]:
    """Summarize how well the current outcome labels line up with correctness."""
    total = max(len(records), 1)
    outcome_distribution: Dict[str, int] = {}
    outcome_share: Dict[str, float] = {}
    p_is_correct_given_outcome: Dict[str, float] = {}

    for idx, label in enumerate(outcome_labels):
        subset = [record for record in records if int(record["outcome"]) == idx]
        correct = [int(bool(record["is_correct"])) for record in subset]
        outcome_distribution[label] = len(subset)
        outcome_share[label] = len(subset) / total
        p_is_correct_given_outcome[label] = float(np.mean(correct)) if correct else 0.0

    correctness_separation = (
        p_is_correct_given_outcome.get("SUCCESS_LIKELY", 0.0)
        - p_is_correct_given_outcome.get("UNCERTAIN", 0.0)
    )
    return {
        "num_records": int(len(records)),
        "outcome_distribution": outcome_distribution,
        "outcome_share": outcome_share,
        "p_is_correct_given_outcome": p_is_correct_given_outcome,
        "correctness_separation_success_minus_uncertain": float(correctness_separation),
    }


def compute_threshold_occupancy(records: Sequence[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, float]:
    """Measure how the fixed corpus sits relative to the current thresholds."""
    total = max(len(records), 1)
    agreement_medium = float(metadata["agreement_medium"])
    agreement_high = float(metadata["agreement_high"])
    margin_low = float(metadata["margin_low"])
    margin_high = float(metadata["margin_high"])
    pattern_threshold = float(metadata["pattern_z_threshold"])

    agreement_values = np.array([float(record["agreement"]) for record in records], dtype=np.float64)
    margin_values = np.array([float(record["prob_margin"]) for record in records], dtype=np.float64)

    focusable = 0
    diffuseable = 0
    for record in records:
        mean_entropy = float(record["trace_summary"]["mean_entropy"])
        max_attention = float(record["max_attention_mass"])
        mean_z = _z_score(mean_entropy, metadata["mean_entropy_mean"], metadata["mean_entropy_std"])
        attention_z = _z_score(max_attention, metadata["max_attention_mean"], metadata["max_attention_std"])
        if mean_z <= -pattern_threshold and attention_z >= pattern_threshold:
            focusable += 1
        elif mean_z >= pattern_threshold and attention_z <= -pattern_threshold:
            diffuseable += 1

    return {
        "agreement_below_medium_share": float(np.mean(agreement_values < agreement_medium)),
        "agreement_between_medium_high_share": float(
            np.mean((agreement_values >= agreement_medium) & (agreement_values < agreement_high))
        ),
        "agreement_at_or_above_high_share": float(np.mean(agreement_values >= agreement_high)),
        "margin_below_low_share": float(np.mean(margin_values < margin_low)),
        "margin_between_low_high_share": float(np.mean((margin_values >= margin_low) & (margin_values < margin_high))),
        "margin_at_or_above_high_share": float(np.mean(margin_values >= margin_high)),
        "pattern_focusable_share": _safe_rate(focusable, total),
        "pattern_diffuseable_share": _safe_rate(diffuseable, total),
        "pattern_mixed_region_share": _safe_rate(total - focusable - diffuseable, total),
    }


def compute_threshold_sensitivity(records: Sequence[Dict[str, Any]], metadata: Dict[str, Any], *, relative_delta: float = 0.05) -> Dict[str, Any]:
    """Relabel the fixed corpus under small threshold shifts."""
    baseline_audit = compute_label_audit(records)
    baseline_share = baseline_audit["outcome_share"]
    scenarios: Dict[str, Dict[str, Any]] = {}
    max_outcome_share_swing = 0.0

    for name in ["agreement_medium", "agreement_high", "margin_low", "margin_high", "pattern_z_threshold"]:
        baseline_value = float(metadata[name])
        for direction, factor in (("down", 1.0 - relative_delta), ("up", 1.0 + relative_delta)):
            adjusted = dict(metadata)
            adjusted[name] = baseline_value * factor
            relabeled = [relabel_trace_record(record, adjusted) for record in records]
            relabeled_records = [
                {**record, "attention_pattern": update["attention_pattern"], "confidence": update["confidence"], "outcome": update["outcome"]}
                for record, update in zip(records, relabeled)
            ]
            audit = compute_label_audit(relabeled_records)
            share_swings = {
                label: abs(float(audit["outcome_share"][label]) - float(baseline_share[label]))
                for label in OUTCOME_LABELS
            }
            scenario_key = f"{name}_{direction}"
            scenario_swing = max(share_swings.values(), default=0.0)
            max_outcome_share_swing = max(max_outcome_share_swing, scenario_swing)
            scenarios[scenario_key] = {
                "baseline_value": baseline_value,
                "adjusted_value": float(adjusted[name]),
                "share_swings": share_swings,
                "max_outcome_share_swing": float(scenario_swing),
                "correctness_separation_success_minus_uncertain": float(
                    audit["correctness_separation_success_minus_uncertain"]
                ),
                "outcome_share": audit["outcome_share"],
                "p_is_correct_given_outcome": audit["p_is_correct_given_outcome"],
            }

    return {
        "baseline": baseline_audit,
        "relative_delta": float(relative_delta),
        "max_outcome_share_swing": float(max_outcome_share_swing),
        "label_revision_trigger": bool(
            max_outcome_share_swing > 0.10
            or float(baseline_audit["correctness_separation_success_minus_uncertain"]) < 0.10
        ),
        "scenarios": scenarios,
    }


def build_prediction_records(
    records: Sequence[Dict[str, Any]],
    bundle: Dict[str, Any],
    *,
    run_seed: int,
    label_names: Dict[str, Sequence[str]],
    collapse_audit: Dict[str, Any] | None = None,
) -> list[Dict[str, Any]]:
    """Convert model outputs into JSONL prediction rows."""
    output = []
    predictions = bundle["predictions"].cpu().numpy()
    targets = bundle["targets"].cpu().numpy()
    head_order = list(label_names.keys())
    head_index = {name: idx for idx, name in enumerate(head_order)}

    for idx, record in enumerate(records):
        row = {
            "example_id": record.get("example_id", f"row_{idx:05d}"),
            "split": record.get("split", "unknown"),
            "run_seed": int(run_seed),
            "is_correct": bool(record.get("is_correct", False)),
            "agreement": float(record.get("agreement", 0.0)),
            "prob_margin": float(record.get("prob_margin", 0.0)),
            "max_attention_mass": float(record.get("max_attention_mass", 0.0)),
            "mean_entropy": float(record.get("trace_summary", {}).get("mean_entropy", 0.0)),
        }
        for head_key, labels in label_names.items():
            target_idx = head_index[head_key]
            row[f"{head_key}_target"] = int(targets[idx, target_idx])
            row[f"{head_key}_target_label"] = labels[row[f"{head_key}_target"]]
            row[f"{head_key}_pred"] = int(predictions[idx, target_idx])
            row[f"{head_key}_pred_label"] = labels[row[f"{head_key}_pred"]]
            row[f"{head_key}_logits"] = bundle["logits"][head_key][idx].tolist()
            row[f"{head_key}_probs"] = bundle["probs"][head_key][idx].tolist()
        evidence = bundle.get("evidence")
        if isinstance(evidence, dict):
            for evidence_name, values in evidence.items():
                row[f"{evidence_name}_cert"] = float(values[idx])
            if "success" in evidence and "failure" in evidence:
                row["certificate_margin"] = float(evidence["success"][idx] - evidence["failure"][idx])
        if collapse_audit is not None:
            row["collapse_flags"] = {
                "all_success_mode": bool(collapse_audit.get("all_success_mode", False)),
                "all_uncertain_mode": bool(collapse_audit.get("all_uncertain_mode", False)),
                "all_failure_mode": bool(collapse_audit.get("all_failure_mode", False)),
                "single_confidence_mode": bool(collapse_audit.get("single_confidence_mode", False)),
            }
        output.append(row)
    return output


def compute_monotonicity_violation_rate(base_success_prob: torch.Tensor, perturbed_success_prob: torch.Tensor) -> float:
    """Return the fraction of examples whose success probability increased."""
    base = base_success_prob.detach().cpu()
    perturbed = perturbed_success_prob.detach().cpu()
    return float((perturbed > (base + 1e-6)).float().mean().item())


def apply_trace_perturbation(trace_inputs: torch.Tensor, kind: str) -> torch.Tensor:
    perturbed = trace_inputs.clone()
    if kind == "lower_agreement":
        perturbed[:, 7] = torch.clamp(perturbed[:, 7] * 0.8, min=0.0, max=1.0)
        if perturbed.size(1) > 11:
            perturbed[:, 11] = torch.clamp(1.0 - perturbed[:, 7], min=0.0, max=1.0)
    elif kind == "lower_margin":
        perturbed[:, 8] = torch.clamp(perturbed[:, 8] * 0.8, min=0.0, max=1.0)
    elif kind == "higher_entropy":
        perturbed[:, :6] = torch.clamp(perturbed[:, :6] + 0.15, min=0.0, max=1.0)
        if perturbed.size(1) > 10:
            perturbed[:, 10] = torch.clamp(perturbed[:, 10] + 0.15, min=0.0, max=1.0)
    elif kind == "lower_attention_concentration":
        perturbed[:, 6] = torch.clamp(perturbed[:, 6] * 0.8, min=0.0, max=1.0)
        if perturbed.size(1) > 15:
            perturbed[:, 15] = torch.clamp(perturbed[:, 15] * 0.85, min=0.0, max=1.0)
            perturbed[:, 16] = torch.clamp(perturbed[:, 16] + 0.15, min=0.0, max=1.0)
            perturbed[:, 17] = torch.clamp(perturbed[:, 17] * 0.80, min=0.0, max=1.0)
            perturbed[:, 18] = torch.clamp(perturbed[:, 18] + 0.20, min=0.0, max=1.0)
    elif kind == "higher_variation_ratio" and perturbed.size(1) > 11:
        perturbed[:, 11] = torch.clamp(perturbed[:, 11] + 0.15, min=0.0, max=1.0)
        if perturbed.size(1) > 18:
            perturbed[:, 18] = torch.clamp(perturbed[:, 18] + 0.10, min=0.0, max=1.0)
    return perturbed


def _unpack_trace_eval_batch(batch: Sequence[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Handle legacy 4-tensor batches and E3 5-tensor batches."""
    if len(batch) == 4:
        queries, trace_inputs, _, target_ids = batch
        return queries, trace_inputs, target_ids
    if len(batch) == 5:
        queries, trace_inputs, _, _, target_ids = batch
        return queries, trace_inputs, target_ids
    raise ValueError(f"Unsupported trace batch layout with {len(batch)} items")


def audit_monotonicity(
    model,
    loader,
    device: torch.device,
    path_scales: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Check whether worsening uncertainty proxies raises success probability."""
    supported_checks = list(COMMON_MONOTONICITY_CHECKS)
    first_batch = next(iter(loader))
    trace_input_dim = first_batch[1].shape[-1]
    if trace_input_dim > 11:
        supported_checks.append("higher_variation_ratio")

    deltas: Dict[str, list[float]] = {name: [] for name in supported_checks}
    delta_means: Dict[str, Dict[str, list[float]]] = {
        kind: {
            "success_prob": [],
            "failure_prob": [],
            "uncertain_prob": [],
            "nonfailure_prob": [],
            "commitment_depth": [],
            "stability_cert": [],
            "support_cert": [],
            "success_guard": [],
            "success_base": [],
            "success_base_guard_gap": [],
            "success_cert": [],
            "failure_cert": [],
            "ambiguity_cert": [],
            "worsening_radius": [],
            "worsening_ambiguity": [],
            "worsening_failure": [],
        }
        for kind in supported_checks
    }
    num_examples = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            queries, trace_inputs, target_ids = _unpack_trace_eval_batch(batch)
            queries = queries.to(device)
            trace_inputs = trace_inputs.to(device)
            target_ids = target_ids.to(device)

            base_out = model(
                queries,
                trace_inputs,
                target_ids,
                bottleneck_mode="hard",
                tau=1.0,
                skip_decoder=True,
                path_scales=path_scales,
            )
            base_outcome_probs = F.softmax(base_out["primitives"].outcome_logits, dim=-1)
            base_success = base_outcome_probs[:, 0]
            base_uncertain = base_outcome_probs[:, 1]
            base_failure = base_outcome_probs[:, 2]
            base_nonfailure = 1.0 - base_failure
            base_trajectory_probs = F.softmax(base_out["primitives"].trajectory_shape_logits, dim=-1)
            base_pattern_probs = F.softmax(base_out["primitives"].attention_pattern_logits, dim=-1)
            base_trajectory_good = torch.maximum(base_trajectory_probs[:, 0], base_trajectory_probs[:, 1])
            base_pattern_focused = base_pattern_probs[:, 0]
            num_examples += queries.size(0)

            for kind in supported_checks:
                perturbed_inputs = apply_trace_perturbation(trace_inputs, kind)
                perturbed_out = model(
                    queries,
                    perturbed_inputs,
                    target_ids,
                    bottleneck_mode="hard",
                    tau=1.0,
                    skip_decoder=True,
                    path_scales=path_scales,
                )
                perturbed_outcome_probs = F.softmax(perturbed_out["primitives"].outcome_logits, dim=-1)
                perturbed_success = perturbed_outcome_probs[:, 0]
                perturbed_uncertain = perturbed_outcome_probs[:, 1]
                perturbed_failure = perturbed_outcome_probs[:, 2]
                perturbed_nonfailure = 1.0 - perturbed_failure
                perturbed_trajectory_probs = F.softmax(perturbed_out["primitives"].trajectory_shape_logits, dim=-1)
                perturbed_pattern_probs = F.softmax(perturbed_out["primitives"].attention_pattern_logits, dim=-1)
                perturbed_trajectory_good = torch.maximum(perturbed_trajectory_probs[:, 0], perturbed_trajectory_probs[:, 1])
                perturbed_pattern_focused = perturbed_pattern_probs[:, 0]
                deltas[kind].append(compute_monotonicity_violation_rate(base_success, perturbed_success))
                deltas.setdefault(f"{kind}__nonfailure", []).append(
                    compute_monotonicity_violation_rate(base_nonfailure, perturbed_nonfailure)
                )
                deltas.setdefault(f"{kind}__trajectory_good", []).append(
                    compute_monotonicity_violation_rate(base_trajectory_good, perturbed_trajectory_good)
                )
                deltas.setdefault(f"{kind}__pattern_focused", []).append(
                    compute_monotonicity_violation_rate(base_pattern_focused, perturbed_pattern_focused)
                )
                stats_bucket = delta_means[kind]
                stats_bucket["success_prob"].append(float((perturbed_success - base_success).mean().item()))
                stats_bucket["failure_prob"].append(float((perturbed_failure - base_failure).mean().item()))
                stats_bucket["uncertain_prob"].append(float((perturbed_uncertain - base_uncertain).mean().item()))
                stats_bucket["nonfailure_prob"].append(float((perturbed_nonfailure - base_nonfailure).mean().item()))
                if hasattr(base_out["primitives"], "commitment_depth"):
                    stats_bucket["commitment_depth"].append(
                        float(
                            (
                                perturbed_out["primitives"].commitment_depth
                                - base_out["primitives"].commitment_depth
                            ).mean().item()
                        )
                    )
                if hasattr(base_out["primitives"], "stability_cert"):
                    deltas.setdefault(f"{kind}__stability_cert", []).append(
                        compute_monotonicity_violation_rate(
                            base_out["primitives"].stability_cert,
                            perturbed_out["primitives"].stability_cert,
                        )
                    )
                    stats_bucket["stability_cert"].append(
                        float(
                            (
                                perturbed_out["primitives"].stability_cert
                                - base_out["primitives"].stability_cert
                            ).mean().item()
                        )
                    )
                    deltas.setdefault(f"{kind}__support_cert", []).append(
                        compute_monotonicity_violation_rate(
                            base_out["primitives"].support_cert,
                            perturbed_out["primitives"].support_cert,
                        )
                    )
                    stats_bucket["support_cert"].append(
                        float(
                            (
                                perturbed_out["primitives"].support_cert
                                - base_out["primitives"].support_cert
                            ).mean().item()
                        )
                    )
                    if hasattr(base_out["primitives"], "success_guard"):
                        deltas.setdefault(f"{kind}__success_guard", []).append(
                            compute_monotonicity_violation_rate(
                                base_out["primitives"].success_guard,
                                perturbed_out["primitives"].success_guard,
                            )
                        )
                        stats_bucket["success_guard"].append(
                            float(
                                (
                                    perturbed_out["primitives"].success_guard
                                    - base_out["primitives"].success_guard
                                ).mean().item()
                            )
                        )
                    if hasattr(base_out["primitives"], "success_core"):
                        deltas.setdefault(f"{kind}__success_core", []).append(
                            compute_monotonicity_violation_rate(
                                base_out["primitives"].success_core,
                                perturbed_out["primitives"].success_core,
                            )
                        )
                    if hasattr(base_out["primitives"], "success_base"):
                        deltas.setdefault(f"{kind}__success_base", []).append(
                            compute_monotonicity_violation_rate(
                                base_out["primitives"].success_base,
                                perturbed_out["primitives"].success_base,
                            )
                        )
                        stats_bucket["success_base"].append(
                            float(
                                (
                                    perturbed_out["primitives"].success_base
                                    - base_out["primitives"].success_base
                                ).mean().item()
                            )
                        )
                    if hasattr(base_out["primitives"], "success_base") and hasattr(base_out["primitives"], "success_guard"):
                        base_gap = torch.relu(
                            base_out["primitives"].success_base - base_out["primitives"].success_guard
                        )
                        perturbed_gap = torch.relu(
                            perturbed_out["primitives"].success_base - perturbed_out["primitives"].success_guard
                        )
                        deltas.setdefault(f"{kind}__success_base_guard_gap", []).append(
                            compute_monotonicity_violation_rate(base_gap, perturbed_gap)
                        )
                        stats_bucket["success_base_guard_gap"].append(
                            float((perturbed_gap - base_gap).mean().item())
                        )
                    deltas.setdefault(f"{kind}__success_cert", []).append(
                        compute_monotonicity_violation_rate(
                            base_out["primitives"].success_cert,
                            perturbed_out["primitives"].success_cert,
                        )
                    )
                    stats_bucket["success_cert"].append(
                        float(
                            (
                                perturbed_out["primitives"].success_cert
                                - base_out["primitives"].success_cert
                            ).mean().item()
                        )
                    )
                    deltas.setdefault(f"{kind}__failure_cert_reverse", []).append(
                        compute_monotonicity_violation_rate(
                            1.0 - base_out["primitives"].failure_cert,
                            1.0 - perturbed_out["primitives"].failure_cert,
                        )
                    )
                    stats_bucket["failure_cert"].append(
                        float(
                            (
                                perturbed_out["primitives"].failure_cert
                                - base_out["primitives"].failure_cert
                            ).mean().item()
                        )
                    )
                    if hasattr(base_out["primitives"], "ambiguity_cert"):
                        stats_bucket["ambiguity_cert"].append(
                            float(
                                (
                                    perturbed_out["primitives"].ambiguity_cert
                                    - base_out["primitives"].ambiguity_cert
                                ).mean().item()
                            )
                        )
                    if hasattr(base_out["primitives"], "worsening_radius"):
                        stats_bucket["worsening_radius"].append(
                            float(
                                (
                                    perturbed_out["primitives"].worsening_radius
                                    - base_out["primitives"].worsening_radius
                                ).mean().item()
                            )
                        )
                    if hasattr(base_out["primitives"], "worsening_ambiguity"):
                        stats_bucket["worsening_ambiguity"].append(
                            float(
                                (
                                    perturbed_out["primitives"].worsening_ambiguity
                                    - base_out["primitives"].worsening_ambiguity
                                ).mean().item()
                            )
                        )
                    if hasattr(base_out["primitives"], "worsening_failure"):
                        stats_bucket["worsening_failure"].append(
                            float(
                                (
                                    perturbed_out["primitives"].worsening_failure
                                    - base_out["primitives"].worsening_failure
                                ).mean().item()
                            )
                        )

    per_check = {f"{name}_violation_rate": _safe_mean(values) for name, values in deltas.items()}
    delta_means_by_perturbation = {
        kind: {
            f"{metric}_mean_delta": _safe_mean(values)
            for metric, values in metric_values.items()
            if values
        }
        for kind, metric_values in delta_means.items()
    }
    common_values = [per_check[f"{name}_violation_rate"] for name in COMMON_MONOTONICITY_CHECKS if f"{name}_violation_rate" in per_check]
    supported_values = list(per_check.values())
    nonfailure_common_values = [
        per_check[f"{name}__nonfailure_violation_rate"]
        for name in COMMON_MONOTONICITY_CHECKS
        if f"{name}__nonfailure_violation_rate" in per_check
    ]
    nonfailure_supported_values = [
        value for key, value in per_check.items() if key.endswith("__nonfailure_violation_rate")
    ]
    trajectory_good_values = [
        value for key, value in per_check.items() if key.endswith("__trajectory_good_violation_rate")
    ]
    pattern_focused_values = [
        value for key, value in per_check.items() if key.endswith("__pattern_focused_violation_rate")
    ]
    evidence_stability_values = [
        value for key, value in per_check.items() if key.endswith("__stability_cert_violation_rate")
    ]
    evidence_support_values = [
        value for key, value in per_check.items() if key.endswith("__support_cert_violation_rate")
    ]
    evidence_guard_values = [
        value for key, value in per_check.items() if key.endswith("__success_guard_violation_rate")
    ]
    evidence_core_values = [
        value for key, value in per_check.items() if key.endswith("__success_core_violation_rate")
    ]
    evidence_base_values = [
        value for key, value in per_check.items() if key.endswith("__success_base_violation_rate")
    ]
    evidence_base_gap_values = [
        value for key, value in per_check.items() if key.endswith("__success_base_guard_gap_violation_rate")
    ]
    evidence_success_values = [
        value for key, value in per_check.items() if key.endswith("__success_cert_violation_rate")
    ]
    evidence_failure_values = [
        value for key, value in per_check.items() if key.endswith("__failure_cert_reverse_violation_rate")
    ]
    worst_perturbation_by_metric: Dict[str, Dict[str, float | str]] = {}
    tracked_worst_metrics = {
        "success_violation_rate": [f"{name}_violation_rate" for name in supported_checks],
        "nonfailure_violation_rate": [f"{name}__nonfailure_violation_rate" for name in supported_checks],
        "success_guard_violation_rate": [f"{name}__success_guard_violation_rate" for name in supported_checks],
        "success_base_guard_gap_violation_rate": [f"{name}__success_base_guard_gap_violation_rate" for name in supported_checks],
        "failure_cert_violation_rate": [f"{name}__failure_cert_reverse_violation_rate" for name in supported_checks],
    }
    for metric_name, keys in tracked_worst_metrics.items():
        available = [(key, per_check[key]) for key in keys if key in per_check]
        if not available:
            continue
        worst_key, worst_value = max(available, key=lambda item: item[1])
        perturbation = worst_key
        for suffix in (
            "__failure_cert_reverse_violation_rate",
            "__success_base_guard_gap_violation_rate",
            "__success_guard_violation_rate",
            "__nonfailure_violation_rate",
            "_violation_rate",
        ):
            if perturbation.endswith(suffix):
                perturbation = perturbation[: -len(suffix)]
                break
        worst_perturbation_by_metric[metric_name] = {
            "perturbation": perturbation,
            "value": float(worst_value),
        }
    return {
        "num_examples": int(num_examples),
        "trace_input_dim": int(trace_input_dim),
        "supported_checks": supported_checks,
        **per_check,
        "delta_means_by_perturbation": delta_means_by_perturbation,
        "worst_perturbation_by_metric": worst_perturbation_by_metric,
        "monotonicity_violation_rate": _safe_mean(common_values),
        "all_supported_monotonicity_violation_rate": _safe_mean(supported_values),
        "nonfailure_monotonicity_violation_rate": _safe_mean(nonfailure_common_values),
        "all_supported_nonfailure_monotonicity_violation_rate": _safe_mean(nonfailure_supported_values),
        "trajectory_good_violation_rate": _safe_mean(trajectory_good_values),
        "pattern_focused_violation_rate": _safe_mean(pattern_focused_values),
        "stability_cert_violation_rate": _safe_mean(evidence_stability_values),
        "support_cert_violation_rate": _safe_mean(evidence_support_values),
        "success_guard_violation_rate": _safe_mean(evidence_guard_values),
        "success_core_violation_rate": _safe_mean(evidence_core_values),
        "success_base_violation_rate": _safe_mean(evidence_base_values),
        "success_base_guard_gap_violation_rate": _safe_mean(evidence_base_gap_values),
        "success_cert_violation_rate": _safe_mean(evidence_success_values),
        "failure_cert_violation_rate": _safe_mean(evidence_failure_values),
    }
