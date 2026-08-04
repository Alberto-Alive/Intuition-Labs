"""Run outcome-calibration A/B experiments for guard-only E3, optionally in parallel."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
VARIANT_DIR = Path(__file__).resolve().parents[1]
RUN_EXPERIMENT = Path(__file__).with_name("run_experiment.py")

BASE_PATTERN_ONLY_SCALES = {
    "outcome_trajectory_context": 0.0,
    "outcome_pattern_context": 0.8,
}
CONTROL_LABEL = "pattern_guard_x25"

EXPERIMENT_SPECS: dict[str, dict[str, object]] = {
    "pattern_guard_x2": {
        "description": "Current main default: pattern-only outcome context with x2 success-probability guard penalty.",
        "path_scales": dict(BASE_PATTERN_ONLY_SCALES),
        "config_overrides": {
            "success_prob_guard_penalty_multiplier": 2.0,
        },
    },
    "pattern_guard_x25": {
        "description": "Pattern-only context with x2.5 success-probability guard penalty.",
        "path_scales": dict(BASE_PATTERN_ONLY_SCALES),
        "config_overrides": {
            "success_prob_guard_penalty_multiplier": 2.5,
        },
    },
    "pattern_ctx_06": {
        "description": "Keep only pattern context, at strength 0.6.",
        "path_scales": {
            "outcome_trajectory_context": 0.0,
            "outcome_pattern_context": 0.6,
        },
        "config_overrides": {
            "success_prob_guard_penalty_multiplier": 2.0,
        },
    },
    "pattern_ctx_half": {
        "description": "Keep only pattern context, but halve its strength.",
        "path_scales": {
            "outcome_trajectory_context": 0.0,
            "outcome_pattern_context": 0.4,
        },
        "config_overrides": {
            "success_prob_guard_penalty_multiplier": 2.0,
        },
    },
    "pattern_ctx_quarter": {
        "description": "Keep only pattern context, but reduce it to 0.2.",
        "path_scales": {
            "outcome_trajectory_context": 0.0,
            "outcome_pattern_context": 0.2,
        },
        "config_overrides": {
            "success_prob_guard_penalty_multiplier": 2.0,
        },
    },
    "pattern_guard_x3": {
        "description": "Increase the direct success-probability-over-guard penalty to x3.",
        "path_scales": dict(BASE_PATTERN_ONLY_SCALES),
        "config_overrides": {
            "success_prob_guard_penalty_multiplier": 3.0,
        },
    },
    "pattern_guard_x35": {
        "description": "Increase the direct success-probability-over-guard penalty to x3.5.",
        "path_scales": dict(BASE_PATTERN_ONLY_SCALES),
        "config_overrides": {
            "success_prob_guard_penalty_multiplier": 3.5,
        },
    },
    "pattern_guard_x4": {
        "description": "Quadruple the direct success-probability-over-guard penalty.",
        "path_scales": dict(BASE_PATTERN_ONLY_SCALES),
        "config_overrides": {
            "success_prob_guard_penalty_multiplier": 4.0,
        },
    },
    "pattern_guard_x2_margin_003": {
        "description": "Pattern-only context with x2 guard penalty and tighter 0.03 guard margin.",
        "path_scales": dict(BASE_PATTERN_ONLY_SCALES),
        "config_overrides": {
            "success_prob_guard_penalty_multiplier": 2.0,
            "success_prob_guard_margin": 0.03,
        },
    },
    "pattern_guard_x3_margin_003": {
        "description": "Pattern-only context with x3 guard penalty and tighter 0.03 guard margin.",
        "path_scales": dict(BASE_PATTERN_ONLY_SCALES),
        "config_overrides": {
            "success_prob_guard_penalty_multiplier": 3.0,
            "success_prob_guard_margin": 0.03,
        },
    },
    "pattern_guard_x2_smooth_001": {
        "description": "Pattern-only context with x2 guard penalty and light 0.01 outcome label smoothing.",
        "path_scales": dict(BASE_PATTERN_ONLY_SCALES),
        "config_overrides": {
            "success_prob_guard_penalty_multiplier": 2.0,
            "outcome_label_smoothing": 0.01,
        },
    },
    "pattern_guard_x2_smooth_002": {
        "description": "Pattern-only context with x2 guard penalty and 0.02 outcome label smoothing.",
        "path_scales": dict(BASE_PATTERN_ONLY_SCALES),
        "config_overrides": {
            "success_prob_guard_penalty_multiplier": 2.0,
            "outcome_label_smoothing": 0.02,
        },
    },
    "pattern_ctx_06_guard_x3": {
        "description": "Pattern context 0.6 with x3 guard penalty.",
        "path_scales": {
            "outcome_trajectory_context": 0.0,
            "outcome_pattern_context": 0.6,
        },
        "config_overrides": {
            "success_prob_guard_penalty_multiplier": 3.0,
        },
    },
    "pattern_ctx_half_guard_x2": {
        "description": "Half pattern context and double the success-probability guard penalty.",
        "path_scales": {
            "outcome_trajectory_context": 0.0,
            "outcome_pattern_context": 0.4,
        },
        "config_overrides": {
            "success_prob_guard_penalty_multiplier": 2.0,
        },
    },
    "pattern_ctx_06_guard_x2_smooth_001": {
        "description": "Pattern context 0.6 with x2 guard penalty and light 0.01 label smoothing.",
        "path_scales": {
            "outcome_trajectory_context": 0.0,
            "outcome_pattern_context": 0.6,
        },
        "config_overrides": {
            "success_prob_guard_penalty_multiplier": 2.0,
            "outcome_label_smoothing": 0.01,
        },
    },
}

DEFAULT_EXPERIMENTS = list(EXPERIMENT_SPECS)


def _load_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _extract_row(summary: dict) -> dict:
    test_metrics = summary.get("test_metrics", {})
    monotonicity = summary.get("monotonicity", {})
    prediction_audit = summary.get("prediction_audit", {})
    path_audit_summary = summary.get("path_audit_summary", {})

    over_guard_by_pred = prediction_audit.get("success_prob_over_guard_rate_by_predicted_outcome", {})
    excess_by_pred = prediction_audit.get("success_prob_over_guard_excess_by_predicted_outcome", {})
    prob_by_pred = prediction_audit.get("success_prob_mean_by_predicted_outcome", {})
    guard_by_pred = prediction_audit.get("success_guard_mean_by_predicted_outcome", {})
    over_guard_by_correctness = prediction_audit.get(
        "predicted_success_over_guard_rate_by_correctness",
        {},
    )
    excess_by_correctness = prediction_audit.get(
        "predicted_success_over_guard_excess_by_correctness",
        {},
    )

    return {
        "label": summary.get("experiment_label", CONTROL_LABEL),
        "result_dir": summary.get("result_dir"),
        "outcome_acc": test_metrics.get("outcome_acc"),
        "outcome_macro_f1": test_metrics.get("outcome_macro_f1"),
        "trajectory_macro_f1": test_metrics.get("trajectory_macro_f1"),
        "failure_recall": test_metrics.get("failure_recall"),
        "joint_acc": test_metrics.get("joint_acc"),
        "success_brier": test_metrics.get("success_brier"),
        "success_ece_10bin": test_metrics.get("success_ece_10bin"),
        "unsafe_success_rate_correctness": test_metrics.get("unsafe_success_rate_correctness"),
        "success_precision_vs_is_correct": test_metrics.get("success_precision_vs_is_correct"),
        "monotonicity_violation_rate": monotonicity.get("monotonicity_violation_rate"),
        "all_supported_monotonicity_violation_rate": monotonicity.get(
            "all_supported_monotonicity_violation_rate"
        ),
        "success_cert_violation_rate": monotonicity.get("success_cert_violation_rate"),
        "failure_cert_violation_rate": monotonicity.get("failure_cert_violation_rate"),
        "pred_success_over_guard": over_guard_by_pred.get("SUCCESS_LIKELY"),
        "pred_success_guard_excess": excess_by_pred.get("SUCCESS_LIKELY"),
        "pred_success_prob_mean": prob_by_pred.get("SUCCESS_LIKELY"),
        "pred_success_guard_mean": guard_by_pred.get("SUCCESS_LIKELY"),
        "pred_success_over_guard_correct": over_guard_by_correctness.get("correct"),
        "pred_success_over_guard_incorrect": over_guard_by_correctness.get("incorrect"),
        "pred_success_excess_correct": excess_by_correctness.get("correct"),
        "pred_success_excess_incorrect": excess_by_correctness.get("incorrect"),
        "pred_uncertain_over_guard": over_guard_by_pred.get("UNCERTAIN"),
        "pred_failure_over_guard": over_guard_by_pred.get("FAILURE_LIKELY"),
        "largest_single_path_outcome_driver": path_audit_summary.get(
            "largest_single_path_outcome_flip",
            {},
        ).get("scenario"),
        "largest_single_path_outcome_flip": path_audit_summary.get(
            "largest_single_path_outcome_flip",
            {},
        ).get("value"),
        "path_scales": summary.get("path_scales", {}),
        "config_overrides": summary.get("config_overrides", {}),
    }


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _sort_metric(value: float | None, missing: float) -> float:
    return missing if value is None else float(value)


def _recommend(rows: list[dict]) -> dict:
    control = next((row for row in rows if row.get("label") == CONTROL_LABEL), None)
    if control is None:
        pool = rows
    else:
        pool = [
            row
            for row in rows
            if (row.get("outcome_macro_f1") or 0.0) >= (control.get("outcome_macro_f1") or 0.0) - 0.02
            and (row.get("failure_recall") or 0.0) >= (control.get("failure_recall") or 0.0) - 0.06
            and (row.get("joint_acc") or 0.0) >= (control.get("joint_acc") or 0.0) - 0.02
            and (row.get("trajectory_macro_f1") or 0.0) >= (control.get("trajectory_macro_f1") or 0.0) - 0.03
        ]
        if not pool:
            pool = rows
    return min(
        pool,
        key=lambda row: (
            _sort_metric(row.get("pred_success_over_guard_incorrect"), float("inf")),
            _sort_metric(row.get("pred_success_excess_incorrect"), float("inf")),
            _sort_metric(row.get("pred_success_over_guard"), float("inf")),
            _sort_metric(row.get("pred_success_guard_excess"), float("inf")),
            _sort_metric(row.get("success_brier"), float("inf")),
            -_sort_metric(row.get("outcome_macro_f1"), float("-inf")),
            -_sort_metric(row.get("failure_recall"), float("-inf")),
        ),
    )


def _write_markdown(path: Path, rows: list[dict], recommended: dict) -> None:
    headers = [
        "label",
        "out_f1",
        "out_acc",
        "traj_f1",
        "fail_rec",
        "joint",
        "pred_S_over_guard",
        "pred_S_wrong_over",
        "pred_S_excess",
        "brier",
        "ece",
    ]
    lines = [
        "# E3 Outcome Calibration Sweep",
        "",
        f"Recommendation rule: keep runs within a small quality delta of `{CONTROL_LABEL}`, then minimize over-guard on incorrect predicted `SUCCESS`, then overall over-guard, then mean excess, then `success_brier`.",
        "",
        f"Recommended experiment: `{recommended['label']}`",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    _format_metric(row.get("outcome_macro_f1")),
                    _format_metric(row.get("outcome_acc")),
                    _format_metric(row.get("trajectory_macro_f1")),
                    _format_metric(row.get("failure_recall")),
                    _format_metric(row.get("joint_acc")),
                    _format_metric(row.get("pred_success_over_guard")),
                    _format_metric(row.get("pred_success_over_guard_incorrect")),
                    _format_metric(row.get("pred_success_guard_excess")),
                    _format_metric(row.get("success_brier")),
                    _format_metric(row.get("success_ece_10bin")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def _build_command(
    label: str,
    run_dir: Path,
    seed: int,
    write_reference: bool,
    pattern_threshold_override: float | None,
    rebuild_traces: bool,
    path_scales: dict[str, float],
    config_overrides: dict[str, float],
) -> list[str]:
    cmd = [
        sys.executable,
        str(RUN_EXPERIMENT),
        "--train-loop-seed",
        str(seed),
        "--output-dir",
        str(run_dir),
        "--experiment-label",
        label,
    ]
    if not write_reference:
        cmd.append("--skip-reference-write")
    if pattern_threshold_override is not None:
        cmd.extend(["--pattern-threshold-override", str(pattern_threshold_override)])
    if rebuild_traces:
        cmd.append("--rebuild-traces")
    for key, value in path_scales.items():
        cmd.extend(["--path-scale", f"{key}={value}"])
    for key, value in config_overrides.items():
        cmd.extend(["--config-override", f"{key}={value}"])
    return cmd


def _collect_completed(run_dir: Path) -> dict | None:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    return _extract_row(_load_summary(summary_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-loop-seed", type=int, default=123)
    parser.add_argument("--pattern-threshold-override", type=float, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--write-reference", action="store_true")
    parser.add_argument("--rebuild-traces", action="store_true")
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()

    unknown = [name for name in args.experiments if name not in EXPERIMENT_SPECS]
    if unknown:
        raise ValueError(f"Unknown experiments {unknown}; expected subset of {sorted(EXPERIMENT_SPECS)}")

    output_root = (
        Path(args.output_root)
        if args.output_root
        else ROOT_DIR / "results" / VARIANT_DIR.name / "calibration_sweep" / f"seed_{args.train_loop_seed}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    queue = deque(args.experiments)
    running: dict[str, dict[str, object]] = {}
    rows: list[dict] = []
    failures: dict[str, int] = {}

    while queue or running:
        while queue and len(running) < max(int(args.max_parallel), 1):
            label = queue.popleft()
            spec = EXPERIMENT_SPECS[label]
            run_dir = output_root / label
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "console.log"
            log_file = open(log_path, "w")
            cmd = _build_command(
                label=label,
                run_dir=run_dir,
                seed=int(args.train_loop_seed),
                write_reference=bool(args.write_reference),
                pattern_threshold_override=args.pattern_threshold_override,
                rebuild_traces=bool(args.rebuild_traces and not running and not rows),
                path_scales=dict(spec.get("path_scales", {})),
                config_overrides=dict(spec.get("config_overrides", {})),
            )
            print(f"Starting {label}: {spec['description']}")
            print(f"  log: {log_path}")
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            running[label] = {
                "process": process,
                "log_file": log_file,
                "run_dir": run_dir,
            }

        if not running:
            continue

        time.sleep(max(float(args.poll_seconds), 0.5))
        completed: list[str] = []
        for label, payload in running.items():
            process = payload["process"]
            return_code = process.poll()
            if return_code is None:
                continue
            payload["log_file"].close()
            run_dir = payload["run_dir"]
            if return_code != 0:
                failures[label] = int(return_code)
                print(f"FAILED {label}: returncode={return_code}")
            else:
                row = _collect_completed(run_dir)
                if row is None:
                    failures[label] = -1
                    print(f"FAILED {label}: missing summary.json")
                else:
                    rows.append(row)
                    print(
                        f"Finished {label}: "
                        f"outcome_macro_f1={_format_metric(row.get('outcome_macro_f1'))}, "
                        f"pred_success_wrong_over_guard={_format_metric(row.get('pred_success_over_guard_incorrect'))}, "
                        f"pred_success_over_guard={_format_metric(row.get('pred_success_over_guard'))}, "
                        f"success_brier={_format_metric(row.get('success_brier'))}"
                    )
            completed.append(label)
        for label in completed:
            del running[label]

    rows.sort(key=lambda row: args.experiments.index(row["label"]))
    if not rows:
        raise RuntimeError("No successful experiments completed.")

    recommended = _recommend(rows)
    comparison_payload = {
        "variant": VARIANT_DIR.name,
        "train_loop_seed": int(args.train_loop_seed),
        "experiments": rows,
        "failures": failures,
        "recommended_experiment": recommended["label"],
        "recommendation_rule": (
            f"Keep runs near the {CONTROL_LABEL} quality baseline, then minimize over-guard on incorrect "
            "predicted SUCCESS, then overall predicted-success over-guard rate, then mean excess, then success_brier."
        ),
        "specs": {
            label: {
                "description": EXPERIMENT_SPECS[label]["description"],
                "path_scales": EXPERIMENT_SPECS[label]["path_scales"],
                "config_overrides": EXPERIMENT_SPECS[label]["config_overrides"],
            }
            for label in args.experiments
        },
    }
    with open(output_root / "comparison.json", "w") as f:
        json.dump(comparison_payload, f, indent=2)
    _write_markdown(output_root / "comparison.md", rows, recommended)

    print("\n=== Outcome-calibration A/B recommendation ===")
    print(f"Recommended experiment: {recommended['label']}")
    print(f"Comparison JSON: {output_root / 'comparison.json'}")
    print(f"Comparison MD: {output_root / 'comparison.md'}")
    if failures:
        print(f"Failures: {failures}")


if __name__ == "__main__":
    main()
