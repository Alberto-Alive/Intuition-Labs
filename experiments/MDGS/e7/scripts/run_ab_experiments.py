"""Run outcome-head A/B experiments for guard-only E3, optionally in parallel."""

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

EXPERIMENT_SPECS: dict[str, dict[str, object]] = {
    "control": {
        "description": "Current guard-only baseline.",
        "path_scales": {},
    },
    "outcome_ctx_half": {
        "description": "Reduce both outcome context paths by half.",
        "path_scales": {
            "outcome_trajectory_context": 0.4,
            "outcome_pattern_context": 0.4,
        },
    },
    "outcome_ctx_zero": {
        "description": "Remove both outcome context paths.",
        "path_scales": {
            "outcome_trajectory_context": 0.0,
            "outcome_pattern_context": 0.0,
        },
    },
    "traj_ctx_only": {
        "description": "Keep only trajectory context in the final outcome head.",
        "path_scales": {
            "outcome_trajectory_context": 0.8,
            "outcome_pattern_context": 0.0,
        },
    },
    "pattern_ctx_only": {
        "description": "Keep only pattern context in the final outcome head.",
        "path_scales": {
            "outcome_trajectory_context": 0.0,
            "outcome_pattern_context": 0.8,
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
    success_modes = summary.get("success_modes", {})
    prediction_audit = summary.get("prediction_audit", {})
    counterfactual = summary.get("counterfactual_audit", {})
    path_audit_summary = summary.get("path_audit_summary", {})

    def _cf_pair(name: str) -> tuple[float | None, float | None]:
        payload = counterfactual.get(name, {})
        if not payload:
            return None, None
        cert_shift = payload.get("success_cert_mae", 0.0) + payload.get("failure_cert_mae", 0.0)
        outcome_flip = payload.get("outcome_label_flip_rate")
        if outcome_flip is None:
            return None, None
        return float(cert_shift), float(outcome_flip)

    zero_outcome_ctx_cert, zero_outcome_ctx_flip = _cf_pair("zero_outcome_context")
    zero_outcome_traj_cert, zero_outcome_traj_flip = _cf_pair("zero_outcome_trajectory_context")
    zero_outcome_pattern_cert, zero_outcome_pattern_flip = _cf_pair("zero_outcome_pattern_context")
    zero_failure_trace_cert, zero_failure_trace_flip = _cf_pair("zero_failure_trace")
    ctx_breakdown = counterfactual.get("zero_outcome_context", {}).get("outcome_flip_rate_by_base_label", {})

    return {
        "label": summary.get("experiment_label", "control"),
        "result_dir": summary.get("result_dir"),
        "success_core_mode": success_modes.get("success_core_mode", "unknown"),
        "success_residual_mode": success_modes.get("success_residual_mode", "unknown"),
        "trajectory_acc": test_metrics.get("trajectory_acc"),
        "trajectory_macro_f1": test_metrics.get("trajectory_macro_f1"),
        "pattern_macro_f1": test_metrics.get("pattern_macro_f1"),
        "confidence_macro_f1": test_metrics.get("confidence_macro_f1"),
        "outcome_acc": test_metrics.get("outcome_acc"),
        "outcome_macro_f1": test_metrics.get("outcome_macro_f1"),
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
        "success_prob_over_guard_pred_success": prediction_audit.get(
            "success_prob_over_guard_rate_by_predicted_outcome",
            {},
        ).get("SUCCESS_LIKELY"),
        "success_prob_over_guard_pred_uncertain": prediction_audit.get(
            "success_prob_over_guard_rate_by_predicted_outcome",
            {},
        ).get("UNCERTAIN"),
        "success_prob_over_guard_pred_failure": prediction_audit.get(
            "success_prob_over_guard_rate_by_predicted_outcome",
            {},
        ).get("FAILURE_LIKELY"),
        "zero_outcome_context_cert_shift": zero_outcome_ctx_cert,
        "zero_outcome_context_outcome_flip": zero_outcome_ctx_flip,
        "zero_outcome_trajectory_context_cert_shift": zero_outcome_traj_cert,
        "zero_outcome_trajectory_context_outcome_flip": zero_outcome_traj_flip,
        "zero_outcome_pattern_context_cert_shift": zero_outcome_pattern_cert,
        "zero_outcome_pattern_context_outcome_flip": zero_outcome_pattern_flip,
        "zero_failure_trace_cert_shift": zero_failure_trace_cert,
        "zero_failure_trace_outcome_flip": zero_failure_trace_flip,
        "zero_outcome_context_flip_success": ctx_breakdown.get("SUCCESS_LIKELY"),
        "zero_outcome_context_flip_uncertain": ctx_breakdown.get("UNCERTAIN"),
        "zero_outcome_context_flip_failure": ctx_breakdown.get("FAILURE_LIKELY"),
        "largest_single_path_outcome_driver": path_audit_summary.get(
            "largest_single_path_outcome_flip",
            {},
        ).get("scenario"),
        "largest_single_path_outcome_flip": path_audit_summary.get(
            "largest_single_path_outcome_flip",
            {},
        ).get("value"),
    }


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _recommend(rows: list[dict]) -> dict:
    viable = [
        row
        for row in rows
        if (row.get("outcome_acc") or 0.0) >= 0.82 and (row.get("trajectory_acc") or 0.0) >= 0.75
    ]
    pool = viable or rows
    return max(
        pool,
        key=lambda row: (
            -(row.get("success_cert_violation_rate") or float("inf")),
            row.get("outcome_macro_f1") or float("-inf"),
            row.get("failure_recall") or float("-inf"),
            row.get("joint_acc") or float("-inf"),
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
        "succ_cert_v",
        "ctx_flip",
        "traj_ctx_flip",
        "pattern_ctx_flip",
        "pred_S_over_guard",
    ]
    lines = [
        "# E3 Outcome-Context A/B Sweep",
        "",
        "Recommendation rule: among runs with `outcome_acc >= 0.82` and `trajectory_acc >= 0.75`, minimize `success_cert_violation_rate`; break ties by `outcome_macro_f1`, `failure_recall`, then `joint_acc`.",
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
                    _format_metric(row.get("success_cert_violation_rate")),
                    _format_metric(row.get("zero_outcome_context_outcome_flip")),
                    _format_metric(row.get("zero_outcome_trajectory_context_outcome_flip")),
                    _format_metric(row.get("zero_outcome_pattern_context_outcome_flip")),
                    _format_metric(row.get("success_prob_over_guard_pred_success")),
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
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()

    unknown = [name for name in args.experiments if name not in EXPERIMENT_SPECS]
    if unknown:
        raise ValueError(f"Unknown experiments {unknown}; expected subset of {sorted(EXPERIMENT_SPECS)}")

    output_root = (
        Path(args.output_root)
        if args.output_root
        else ROOT_DIR / "results" / VARIANT_DIR.name / "ab_sweep" / f"seed_{args.train_loop_seed}"
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
                "spec": spec,
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
                        f"trajectory_macro_f1={_format_metric(row.get('trajectory_macro_f1'))}, "
                        f"success_cert_violation_rate={_format_metric(row.get('success_cert_violation_rate'))}"
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
            "Among runs with outcome_acc >= 0.82 and trajectory_acc >= 0.75, "
            "minimize success_cert_violation_rate; break ties by outcome_macro_f1, "
            "failure_recall, then joint_acc."
        ),
        "specs": {
            label: {
                "description": EXPERIMENT_SPECS[label]["description"],
                "path_scales": EXPERIMENT_SPECS[label]["path_scales"],
            }
            for label in args.experiments
        },
    }
    with open(output_root / "comparison.json", "w") as f:
        json.dump(comparison_payload, f, indent=2)
    _write_markdown(output_root / "comparison.md", rows, recommended)

    print("\n=== Outcome-context A/B recommendation ===")
    print(f"Recommended experiment: {recommended['label']}")
    print(f"Comparison JSON: {output_root / 'comparison.json'}")
    print(f"Comparison MD: {output_root / 'comparison.md'}")
    if failures:
        print(f"Failures: {failures}")


if __name__ == "__main__":
    main()
