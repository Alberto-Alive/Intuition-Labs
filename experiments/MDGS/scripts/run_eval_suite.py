"""Run the Extrapolation safety-first evaluation suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_common import DEFAULT_BENCHMARK_SEEDS, summarize_scalar_series, write_json


LABEL_METRIC_KEYS = [
    "trajectory_acc",
    "trajectory_macro_f1",
    "pattern_acc",
    "pattern_macro_f1",
    "confidence_acc",
    "confidence_macro_f1",
    "outcome_acc",
    "outcome_macro_f1",
    "failure_recall",
    "failure_likely_recall",
    "joint_acc",
]

SAFETY_METRIC_KEYS = [
    "unsafe_success_rate_label",
    "unsafe_success_rate_correctness",
    "success_precision_vs_is_correct",
    "success_auroc",
    "success_auprc",
    "success_brier",
    "success_ece_10bin",
    "success_label_auroc",
    "success_label_auprc",
    "success_label_brier",
    "success_label_ece_10bin",
    "nonfailure_correct_auroc",
    "nonfailure_correct_auprc",
    "nonfailure_correct_brier",
    "nonfailure_correct_ece_10bin",
    "failure_incorrect_auroc",
    "failure_incorrect_auprc",
    "failure_incorrect_brier",
    "failure_incorrect_ece_10bin",
]

MONOTONICITY_KEYS = [
    "monotonicity_violation_rate",
    "all_supported_monotonicity_violation_rate",
    "nonfailure_monotonicity_violation_rate",
    "all_supported_nonfailure_monotonicity_violation_rate",
]

RAW_TRACE_KEYS = (
    "example_id",
    "split",
    "query_fields",
    "gold_answer",
    "pred_answer",
    "prediction_probs",
    "agreement",
    "prob_margin",
    "max_attention_mass",
    "attention_entropy_trajectory",
    "is_correct",
    "stats",
    "is_hard_mined",
)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _parse_seeds(raw: str | None) -> list[int]:
    if not raw:
        return list(DEFAULT_BENCHMARK_SEEDS)
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _trace_identity_fingerprint(trace_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
        digest.update(name.encode("utf-8"))
        digest.update(b"\n")
        with open(trace_dir / name) as f:
            for line in f:
                record = json.loads(line)
                raw_payload = {key: record.get(key) for key in RAW_TRACE_KEYS}
                digest.update(json.dumps(raw_payload, sort_keys=True).encode("utf-8"))
                digest.update(b"\n")
    return digest.hexdigest()[:16]


def _require_matching_corpus() -> str:
    fingerprint_e1 = _trace_identity_fingerprint(ROOT / "e1" / "trace_cache")
    fingerprint_e2 = _trace_identity_fingerprint(ROOT / "e2" / "trace_cache")
    if fingerprint_e1 != fingerprint_e2:
        raise RuntimeError(f"Corpus fingerprint mismatch: {fingerprint_e1} != {fingerprint_e2}")
    return str(fingerprint_e1)


def _format_stat(stats: dict[str, float]) -> str:
    return f"{stats['mean']:.3f} +- {stats['std']:.3f}"


def _metric(summary: dict[str, Any], key: str) -> float:
    if key in summary.get("test_metrics", {}):
        return float(summary["test_metrics"][key])
    if key in summary.get("monotonicity", {}):
        return float(summary["monotonicity"][key])
    raise KeyError(f"Metric '{key}' not found")


def _aggregate_variant(runs: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for key in LABEL_METRIC_KEYS + SAFETY_METRIC_KEYS:
        aggregate[key] = summarize_scalar_series([_metric(run, key) for run in runs])
    for key in MONOTONICITY_KEYS:
        values = []
        for run in runs:
            if key in run.get("monotonicity", {}):
                values.append(float(run["monotonicity"][key]))
        aggregate[key] = summarize_scalar_series(values)
    aggregate["acceptance_rate"] = summarize_scalar_series(
        [1.0 if all(run["acceptance"].values()) else 0.0 for run in runs]
    )
    return aggregate


def _paired_delta(e1_runs: list[dict[str, Any]], e2_runs: list[dict[str, Any]], key: str) -> dict[str, Any]:
    e1_by_seed = {int(run["train_loop_seed"]): run for run in e1_runs}
    e2_by_seed = {int(run["train_loop_seed"]): run for run in e2_runs}
    deltas = []
    for seed in sorted(set(e1_by_seed) & set(e2_by_seed)):
        deltas.append(_metric(e2_by_seed[seed], key) - _metric(e1_by_seed[seed], key))
    return {
        "values": deltas,
        **summarize_scalar_series(deltas),
    }


def _build_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Extrapolation Safety Evaluation",
        "",
        f"Corpus fingerprint: `{summary['corpus_fingerprint']}`",
        f"Seeds: `{', '.join(str(seed) for seed in summary['seeds'])}`",
        "",
        "## Safety Metrics",
        "",
        "| Metric | E1 | E2 | Delta (E2-E1) |",
        "| --- | --- | --- | --- |",
    ]
    for key in SAFETY_METRIC_KEYS:
        lines.append(
            f"| `{key}` | {_format_stat(summary['variants']['e1']['aggregate'][key])} | "
            f"{_format_stat(summary['variants']['e2']['aggregate'][key])} | "
            f"{summary['paired_deltas'][key]['mean']:+.3f} |"
        )

    lines.extend(
        [
            "",
            "## Label Recovery",
            "",
            "| Metric | E1 | E2 | Delta (E2-E1) |",
            "| --- | --- | --- | --- |",
        ]
    )
    for key in LABEL_METRIC_KEYS:
        lines.append(
            f"| `{key}` | {_format_stat(summary['variants']['e1']['aggregate'][key])} | "
            f"{_format_stat(summary['variants']['e2']['aggregate'][key])} | "
            f"{summary['paired_deltas'][key]['mean']:+.3f} |"
        )

    lines.extend(
        [
            "",
            "## Monotonicity",
            "",
            "| Metric | E1 | E2 | Delta (E2-E1) |",
            "| --- | --- | --- | --- |",
        ]
    )
    for key in MONOTONICITY_KEYS:
        lines.append(
            f"| `{key}` | {_format_stat(summary['variants']['e1']['aggregate'][key])} | "
            f"{_format_stat(summary['variants']['e2']['aggregate'][key])} | "
            f"{summary['paired_deltas'][key]['mean']:+.3f} |"
        )

    e1_details = summary["variant_details"]["e1"]
    e2_details = summary["variant_details"]["e2"]
    e1_label_prediction = e1_details["baselines"]["label_prediction"]
    e2_label_prediction = e2_details["baselines"]["label_prediction"]
    correctness = summary["baselines"]["correctness"]
    lines.extend(
        [
            "",
            "## Baselines",
            "",
            f"- E1 entropy-threshold outcome accuracy: `{e1_label_prediction['entropy_threshold']['outcome_accuracy']:.3f}`",
            f"- E1 logistic-to-labels outcome macro-F1: `{e1_label_prediction['logistic_regression']['outcome']['macro_f1']:.3f}`",
            f"- E1 MLP-to-labels outcome macro-F1: `{e1_label_prediction['mlp']['outcome']['macro_f1']:.3f}`",
            f"- E2 entropy-threshold outcome accuracy: `{e2_label_prediction['entropy_threshold']['outcome_accuracy']:.3f}`",
            f"- E2 logistic-to-labels outcome macro-F1: `{e2_label_prediction['logistic_regression']['outcome']['macro_f1']:.3f}`",
            f"- E2 MLP-to-labels outcome macro-F1: `{e2_label_prediction['mlp']['outcome']['macro_f1']:.3f}`",
            f"- Logistic-to-correctness AUROC: `{correctness['logistic_regression']['auroc']:.3f}`",
            f"- MLP-to-correctness AUROC: `{correctness['mlp']['auroc']:.3f}`",
            "",
            "## Label Audit",
            "",
            f"- E1 P(correct | SUCCESS_LIKELY): `{e1_details['label_audit']['p_is_correct_given_outcome']['SUCCESS_LIKELY']:.3f}`",
            f"- E1 P(correct | UNCERTAIN): `{e1_details['label_audit']['p_is_correct_given_outcome']['UNCERTAIN']:.3f}`",
            f"- E1 P(correct | FAILURE_LIKELY): `{e1_details['label_audit']['p_is_correct_given_outcome']['FAILURE_LIKELY']:.3f}`",
            f"- E1 correctness separation (SUCCESS - UNCERTAIN): `{e1_details['label_audit']['correctness_separation_success_minus_uncertain']:.3f}`",
            f"- E1 max outcome-share swing under +-5% threshold shifts: `{e1_details['threshold_sensitivity']['max_outcome_share_swing']:.3f}`",
            f"- E2 P(correct | SUCCESS_LIKELY): `{e2_details['label_audit']['p_is_correct_given_outcome']['SUCCESS_LIKELY']:.3f}`",
            f"- E2 P(correct | UNCERTAIN): `{e2_details['label_audit']['p_is_correct_given_outcome']['UNCERTAIN']:.3f}`",
            f"- E2 P(correct | FAILURE_LIKELY): `{e2_details['label_audit']['p_is_correct_given_outcome']['FAILURE_LIKELY']:.3f}`",
            f"- E2 correctness separation (SUCCESS - UNCERTAIN): `{e2_details['label_audit']['correctness_separation_success_minus_uncertain']:.3f}`",
            f"- E2 max outcome-share swing under +-5% threshold shifts: `{e2_details['threshold_sensitivity']['max_outcome_share_swing']:.3f}`",
            "",
            "## Acceptance",
            "",
        ]
    )
    for key, value in summary["comparison"].items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines)


def _build_suite_summary(
    *,
    seeds: list[int],
    fingerprint: str,
    run_summaries: dict[str, list[dict[str, Any]]],
    variant_details: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    variant_aggregates = {variant: _aggregate_variant(runs) for variant, runs in run_summaries.items()}
    paired_deltas: dict[str, Any] = {}
    for key in LABEL_METRIC_KEYS + SAFETY_METRIC_KEYS + MONOTONICITY_KEYS:
        paired_deltas[key] = _paired_delta(run_summaries["e1"], run_summaries["e2"], key)

    label_audit = variant_details["e2"]["label_audit"]
    threshold_sensitivity = variant_details["e2"]["threshold_sensitivity"]
    baselines = {
        "correctness": variant_details["e2"]["baselines"]["correctness"],
        "e1_label_prediction": variant_details["e1"]["baselines"]["label_prediction"],
        "e2_label_prediction": variant_details["e2"]["baselines"]["label_prediction"],
    }

    e1_mean = variant_aggregates["e1"]
    e2_mean = variant_aggregates["e2"]
    comparison = {
        "absolute_floor_failure_recall_ok": bool(e2_mean["failure_recall"]["mean"] >= 0.30),
        "absolute_floor_outcome_macro_f1_ok": bool(e2_mean["outcome_macro_f1"]["mean"] >= 0.62),
        "absolute_floor_outcome_acc_ok": bool(e2_mean["outcome_acc"]["mean"] >= 0.82),
        "absolute_floor_trajectory_acc_ok": bool(e2_mean["trajectory_acc"]["mean"] >= 0.75),
        "safety_win": bool(
            paired_deltas["unsafe_success_rate_correctness"]["mean"] <= -0.05
            and (
                paired_deltas["failure_likely_recall"]["mean"] >= 0.05
                or (
                    paired_deltas["failure_likely_recall"]["mean"] >= -0.01
                    and paired_deltas["success_label_brier"]["mean"] < 0.0
                    and paired_deltas["success_label_auroc"]["mean"] > 0.0
                )
            )
        ),
        "non_regression_outcome_macro_f1": bool(
            e2_mean["outcome_macro_f1"]["mean"] >= e1_mean["outcome_macro_f1"]["mean"] - 0.02
        ),
        "non_regression_confidence_macro_f1": bool(
            e2_mean["confidence_macro_f1"]["mean"] >= e1_mean["confidence_macro_f1"]["mean"] - 0.02
        ),
        "monotonicity_ok": bool(
            e2_mean["monotonicity_violation_rate"]["mean"] < 0.05
            and e2_mean["monotonicity_violation_rate"]["mean"] < e1_mean["monotonicity_violation_rate"]["mean"]
        ),
        "nonfailure_monotonicity_ok": bool(
            e2_mean["nonfailure_monotonicity_violation_rate"]["mean"] < 0.05
            and e2_mean["nonfailure_monotonicity_violation_rate"]["mean"]
            < e1_mean["nonfailure_monotonicity_violation_rate"]["mean"]
        ),
        "label_revision_trigger": bool(threshold_sensitivity["label_revision_trigger"]),
    }

    return {
        "corpus_fingerprint": fingerprint,
        "seeds": seeds,
        "variants": {
            variant: {
                "runs": runs,
                "aggregate": variant_aggregates[variant],
            }
            for variant, runs in run_summaries.items()
        },
        "paired_deltas": paired_deltas,
        "label_audit": label_audit,
        "threshold_sensitivity": {
            "max_outcome_share_swing": threshold_sensitivity["max_outcome_share_swing"],
            "label_revision_trigger": threshold_sensitivity["label_revision_trigger"],
        },
        "variant_details": {
            variant: {
                "label_audit": variant_details[variant]["label_audit"],
                "threshold_sensitivity": {
                    "max_outcome_share_swing": variant_details[variant]["threshold_sensitivity"]["max_outcome_share_swing"],
                    "label_revision_trigger": variant_details[variant]["threshold_sensitivity"]["label_revision_trigger"],
                },
                "baselines": variant_details[variant]["baselines"],
            }
            for variant in ["e1", "e2"]
        },
        "baselines": baselines,
        "comparison": comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds. Default is the five-seed benchmark.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--skip-logic-tests", action="store_true")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    suite_dir = Path(args.output_dir) if args.output_dir else ROOT / "results" / "eval_suite"
    suite_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = _require_matching_corpus()
    print(f"Using fixed corpus fingerprint: {fingerprint}")

    if not args.skip_logic_tests:
        _run([args.python, str(ROOT / "scripts" / "test_eval_logic.py")], cwd=ROOT)

    run_summaries: dict[str, list[dict[str, Any]]] = {"e1": [], "e2": []}
    for variant in ["e1", "e2"]:
        script_path = ROOT / variant / "scripts" / "run_experiment.py"
        for seed in seeds:
            variant_out = suite_dir / variant / f"seed_{seed}"
            cmd = [
                args.python,
                str(script_path),
                "--train-loop-seed",
                str(seed),
                "--output-dir",
                str(variant_out),
                "--skip-reference-write",
            ]
            _run(cmd, cwd=ROOT / variant)
            summary = _read_json(variant_out / "summary.json")
            run_summaries[variant].append(summary)

    variant_details = {
        variant: _read_json(Path(run_summaries[variant][0]["artifacts"]["metrics"]))
        for variant in ["e1", "e2"]
    }
    suite_summary = _build_suite_summary(
        seeds=seeds,
        fingerprint=fingerprint,
        run_summaries=run_summaries,
        variant_details=variant_details,
    )

    write_json(suite_dir / "summary.json", suite_summary)
    (suite_dir / "summary.md").write_text(_build_markdown_report(suite_summary))
    print(f"Suite summary written to {suite_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
