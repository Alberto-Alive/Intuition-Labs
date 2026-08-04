"""Launch and summarize the six-experiment E4 stage-1 anchor sweep."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VARIANT_DIR = Path(__file__).resolve().parents[1]
RUN_EXPERIMENT = VARIANT_DIR / "scripts" / "run_experiment.py"

EXPERIMENT_CONFIGS: dict[str, list[str]] = {
    "anchor_base": [],
    "anchor_no_approach": ["anchor_use_approach=False"],
    "anchor_no_leap": ["anchor_use_leap=False"],
    "anchor_no_boundary_family": ["anchor_use_boundary_family=False"],
    "anchor_hard_assignment": ["anchor_assignment_mode='hard'"],
    "anchor_no_barrier": ["anchor_use_barrier=False"],
}


@dataclass
class SweepJob:
    name: str
    overrides: list[str]
    batch_size: int | None
    retries: int = 0


@dataclass
class ActiveProcess:
    job: SweepJob
    process: subprocess.Popen
    log_path: Path
    output_dir: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=str, default="all")
    parser.add_argument("--max-parallel", type=str, default="auto")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--promote-best-to-stage2", action="store_true")
    parser.add_argument("--batch-size-override", type=int, default=None)
    parser.add_argument("--amp", type=str, default="true")
    parser.add_argument("--train-loop-seed", type=int, default=123)
    return parser.parse_args()


def _selected_experiments(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(EXPERIMENT_CONFIGS)
    names = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [name for name in names if name not in EXPERIMENT_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}")
    return names


def _max_parallel(raw: str) -> int:
    if raw == "auto":
        return 2
    value = int(raw)
    if value <= 0:
        raise ValueError("--max-parallel must be positive")
    return value


def _job_command(job: SweepJob, output_dir: Path, args: argparse.Namespace, stage1_only: bool = True) -> list[str]:
    command = [
        sys.executable,
        str(RUN_EXPERIMENT),
        "--output-dir",
        str(output_dir),
        "--experiment-label",
        job.name,
        "--train-loop-seed",
        str(args.train_loop_seed),
        "--config-override",
        f"use_amp={args.amp.lower() == 'true'}",
        "--config-override",
        "trace_epoch_audit_interval=1",
    ]
    if stage1_only:
        command.extend(["--config-override", "trace_force_stage1_only=True"])
        command.extend(["--config-override", "trace_stage2_epochs=0"])
    else:
        command.extend(["--config-override", "trace_force_stage1_only=False"])
    batch_size = job.batch_size if job.batch_size is not None else args.batch_size_override
    if batch_size is not None:
        command.extend(["--config-override", f"trace_batch_size_override={int(batch_size)}"])
    for override in job.overrides:
        command.extend(["--config-override", override])
    return command


def _spawn(job: SweepJob, output_root: Path, args: argparse.Namespace) -> ActiveProcess:
    output_dir = output_root / job.name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "console.log"
    log_handle = log_path.open("w")
    process = subprocess.Popen(
        _job_command(job, output_dir, args, stage1_only=True),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        cwd=str(VARIANT_DIR),
    )
    log_handle.close()
    return ActiveProcess(job=job, process=process, log_path=log_path, output_dir=output_dir)


def _log_contains_oom(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    text = log_path.read_text(errors="ignore").lower()
    return "out of memory" in text or "cuda oom" in text or "cuda out of memory" in text


def _load_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _comparison_row(name: str, summary: dict[str, Any] | None, status: str) -> dict[str, Any]:
    if not summary:
        return {
            "experiment": name,
            "status": status,
            "eligible": False,
        }
    test_metrics = summary.get("test_metrics", {})
    collapse_audit = summary.get("collapse_audit", {})
    commitment_audit = summary.get("commitment_audit", {})
    eligible = (
        not bool(collapse_audit.get("collapsed_model", True))
        and float(test_metrics.get("trajectory_acc", 0.0)) >= 0.75
    )
    return {
        "experiment": name,
        "status": status,
        "eligible": bool(eligible),
        "trajectory_acc": float(test_metrics.get("trajectory_acc", 0.0)),
        "failure_recall": float(test_metrics.get("failure_recall", 0.0)),
        "outcome_macro_f1": float(test_metrics.get("outcome_macro_f1", 0.0)),
        "commitment_monotonicity_violation_rate": float(
            test_metrics.get("commitment_monotonicity_violation_rate", 1.0)
        ),
        "predicted_success_high_conflict_rate": float(
            test_metrics.get(
                "predicted_success_high_conflict_rate",
                commitment_audit.get("predicted_success_high_conflict_rate", 1.0),
            )
        ),
    }


def _ranking_key(row: dict[str, Any]) -> tuple:
    if not row.get("eligible", False):
        return (1, float("inf"), float("inf"), float("inf"), float("inf"))
    return (
        0,
        float(row.get("commitment_monotonicity_violation_rate", 1.0)),
        float(row.get("predicted_success_high_conflict_rate", 1.0)),
        -float(row.get("failure_recall", 0.0)),
        -float(row.get("outcome_macro_f1", 0.0)),
    )


def _best_available_key(row: dict[str, Any]) -> tuple:
    status_penalty = 0 if row.get("status") == "completed" else 1
    return (
        status_penalty,
        float(row.get("commitment_monotonicity_violation_rate", 1.0)),
        float(row.get("predicted_success_high_conflict_rate", 1.0)),
        -float(row.get("failure_recall", 0.0)),
        -float(row.get("outcome_macro_f1", 0.0)),
    )


def _write_markdown(path: Path, rows: list[dict[str, Any]], best_name: str | None) -> None:
    lines = [
        "# E4 Anchor Sweep",
        "",
        f"Best: `{best_name}`" if best_name else "Best: none",
        "",
        "| Experiment | Status | Eligible | Commitment Mono | Success@Conflict | Failure Recall | Outcome Macro F1 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {experiment} | {status} | {eligible} | {commitment_monotonicity_violation_rate:.4f} | "
            "{predicted_success_high_conflict_rate:.4f} | {failure_recall:.4f} | {outcome_macro_f1:.4f} |".format(
                experiment=row.get("experiment"),
                status=row.get("status"),
                eligible=row.get("eligible"),
                commitment_monotonicity_violation_rate=float(row.get("commitment_monotonicity_violation_rate", 0.0)),
                predicted_success_high_conflict_rate=float(row.get("predicted_success_high_conflict_rate", 0.0)),
                failure_recall=float(row.get("failure_recall", 0.0)),
                outcome_macro_f1=float(row.get("outcome_macro_f1", 0.0)),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def _promote_best(best_row: dict[str, Any], output_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    job = SweepJob(
        name=f"{best_row['experiment']}_promoted_stage2",
        overrides=EXPERIMENT_CONFIGS[best_row["experiment"]],
        batch_size=args.batch_size_override,
        retries=0,
    )
    output_dir = output_root / "promoted_stage2" / best_row["experiment"]
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "console.log"
    with log_path.open("w") as log_handle:
        result = subprocess.run(
            _job_command(job, output_dir, args, stage1_only=False),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(VARIANT_DIR),
            check=False,
        )
    return {
        "experiment": best_row["experiment"],
        "returncode": int(result.returncode),
        "output_dir": str(output_dir),
        "summary_path": str(output_dir / "summary.json"),
    }


def main() -> None:
    args = _parse_args()
    selected = _selected_experiments(args.experiments)
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else (VARIANT_DIR / "results" / "anchor_sweep").resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    pending = [
        SweepJob(name=name, overrides=list(EXPERIMENT_CONFIGS[name]), batch_size=args.batch_size_override)
        for name in selected
    ]
    active: list[ActiveProcess] = []
    completed: dict[str, dict[str, Any]] = {}
    max_parallel = _max_parallel(args.max_parallel)

    while pending or active:
        while pending and len(active) < max_parallel:
            job = pending.pop(0)
            active.append(_spawn(job, output_root, args))

        next_active: list[ActiveProcess] = []
        for item in active:
            returncode = item.process.poll()
            if returncode is None:
                next_active.append(item)
                continue

            if returncode == 0:
                completed[item.job.name] = {
                    "status": "completed",
                    "output_dir": str(item.output_dir),
                    "summary_path": str(item.output_dir / "summary.json"),
                    "log_path": str(item.log_path),
                    "retries": int(item.job.retries),
                }
                continue

            if _log_contains_oom(item.log_path) and item.job.retries == 0 and (item.job.batch_size or args.batch_size_override or 8) > 4:
                pending.append(
                    SweepJob(
                        name=item.job.name,
                        overrides=item.job.overrides,
                        batch_size=4,
                        retries=1,
                    )
                )
                completed[item.job.name] = {
                    "status": "retrying_oom",
                    "output_dir": str(item.output_dir),
                    "log_path": str(item.log_path),
                    "retries": 1,
                }
            else:
                completed[item.job.name] = {
                    "status": "failed",
                    "output_dir": str(item.output_dir),
                    "summary_path": str(item.output_dir / "summary.json"),
                    "log_path": str(item.log_path),
                    "returncode": int(returncode),
                    "retries": int(item.job.retries),
                }
        active = next_active
        if active:
            time.sleep(2.0)

    rows = []
    for name in selected:
        meta = completed.get(name, {"status": "missing"})
        summary = _load_summary(Path(meta.get("summary_path", ""))) if meta.get("summary_path") else None
        rows.append(_comparison_row(name, summary, str(meta.get("status", "missing"))))
    rows.sort(key=_ranking_key)
    best_row = next((row for row in rows if row.get("eligible", False)), None)
    best_available_row = min(rows, key=_best_available_key) if rows else None

    comparison_payload: dict[str, Any] = {
        "selected_experiments": selected,
        "max_parallel": max_parallel,
        "results": rows,
        "best_experiment": best_row.get("experiment") if best_row else None,
        "best_available_experiment": best_available_row.get("experiment") if best_available_row else None,
        "runs": completed,
    }
    if args.promote_best_to_stage2 and best_row is not None:
        comparison_payload["promotion"] = _promote_best(best_row, output_root, args)

    comparison_json = output_root / "comparison.json"
    comparison_md = output_root / "comparison.md"
    comparison_json.write_text(json.dumps(comparison_payload, indent=2) + "\n")
    _write_markdown(comparison_md, rows, comparison_payload.get("best_experiment"))
    print(
        json.dumps(
            {
                "comparison_json": str(comparison_json),
                "best": comparison_payload.get("best_experiment"),
                "best_available": comparison_payload.get("best_available_experiment"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
