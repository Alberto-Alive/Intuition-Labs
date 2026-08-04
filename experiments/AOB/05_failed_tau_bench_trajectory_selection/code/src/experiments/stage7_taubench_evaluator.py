from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence

from src.datasets.taubench_trajectory_selection_dataset import (
    TauBenchTrajectoryCandidate,
    build_taubench_selection_examples,
    load_taubench_candidates_jsonl,
    validate_taubench_selection_examples,
    write_taubench_candidates_jsonl,
)


BENCHMARK = "stage7_taubench_trajectory_selector"
DEFAULT_INPUT = Path("results/stage7a_taubench_smoke_candidates.jsonl")
DEFAULT_OUTPUT = Path("results/stage7a_taubench_smoke_candidates.jsonl")
DEFAULT_EVAL_ROOT = Path("results/stage7_taubench_eval_metadata")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage 7 tau-bench trajectory candidates with the official evaluator.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--eval-root", default=str(DEFAULT_EVAL_ROOT))
    parser.add_argument("--tau2-command", default="tau2")
    parser.add_argument("--tau2-workdir", default=None)
    parser.add_argument("--trust-existing-labels", action="store_true")
    parser.add_argument("--require-labels", action="store_true")
    args = parser.parse_args()

    candidates = load_taubench_candidates_jsonl(args.input)
    evaluated = evaluate_taubench_candidates(
        candidates=candidates,
        eval_root=Path(args.eval_root),
        tau2_command=str(args.tau2_command),
        tau2_workdir=None if args.tau2_workdir is None else Path(args.tau2_workdir),
        trust_existing_labels=bool(args.trust_existing_labels),
    )
    examples = build_taubench_selection_examples(evaluated)
    audit = validate_taubench_selection_examples(examples, require_labels=bool(args.require_labels))
    if bool(args.require_labels) and not bool(audit.get("passes", False)):
        raise SystemExit(json.dumps(audit, indent=2, sort_keys=True))
    write_taubench_candidates_jsonl(args.output, evaluated)
    print(f"stage7 evaluator: wrote {len(evaluated)} evaluated candidates to {args.output}")


def evaluate_taubench_candidates(
    candidates: Sequence[TauBenchTrajectoryCandidate],
    eval_root: Path,
    tau2_command: str = "tau2",
    tau2_workdir: Path | None = None,
    trust_existing_labels: bool = False,
) -> List[TauBenchTrajectoryCandidate]:
    eval_root.mkdir(parents=True, exist_ok=True)
    if trust_existing_labels:
        return list(candidates)
    if not _command_available(tau2_command, tau2_workdir):
        raise RuntimeError("tau2 CLI is not available for official trajectory evaluation")
    out: List[TauBenchTrajectoryCandidate] = []
    for candidate in candidates:
        started = time.perf_counter()
        raw_path = Path(candidate.raw_trajectory_path)
        if not raw_path.exists():
            raise FileNotFoundError(f"raw trajectory missing for {candidate.candidate_id}: {raw_path}")
        command = [tau2_command, "evaluate-trajs", str(raw_path)]
        completed = subprocess.run(
            command,
            cwd=None if tau2_workdir is None else tau2_workdir,
            text=True,
            capture_output=True,
            timeout=900,
        )
        parsed = _parse_evaluate_trajs_output(completed.stdout, completed.stderr)
        success = parsed.get("official_success")
        if success is None:
            success = candidate.official_success
        eval_path = eval_root / candidate.domain / candidate.task_id / f"{candidate.candidate_id}.json"
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.write_text(
            json.dumps(
                {
                    "benchmark": BENCHMARK,
                    "candidate_id": candidate.candidate_id,
                    "task_id": candidate.task_id,
                    "domain": candidate.domain,
                    "raw_trajectory_path": candidate.raw_trajectory_path,
                    "tau2_command": command,
                    "tau2_returncode": int(completed.returncode),
                    "latency_seconds": float(time.perf_counter() - started),
                    "stdout_tail": completed.stdout[-4000:],
                    "stderr_tail": completed.stderr[-4000:],
                    "parsed": parsed,
                    "official_success": success,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        out.append(
            replace(
                candidate,
                official_success=None if success is None else bool(success),
                official_eval_metadata_path=str(eval_path),
                metadata={**candidate.metadata, "official_eval_completed": success is not None},
            )
        )
    return out


def _parse_evaluate_trajs_output(stdout: str, stderr: str) -> Dict[str, object]:
    text = "\n".join(value for value in (stdout, stderr) if value)
    parsed: Dict[str, object] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{") and line.endswith("}"):
            try:
                row = json.loads(line)
            except Exception:
                continue
            parsed.update(row)
    success = _extract_success(parsed)
    if success is None:
        lowered = text.lower()
        if "success" in lowered and ("true" in lowered or "1.0" in lowered):
            success = True
        elif "reward" in lowered and "0.0" in lowered:
            success = False
    parsed["official_success"] = success
    return parsed


def _extract_success(record: Dict[str, object]) -> bool | None:
    for key in ("official_success", "success", "reward", "task_reward", "environment_reward", "passed"):
        if key not in record:
            continue
        value = record[key]
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, (int, float)):
            return float(value) > 0.0
        text = str(value).strip().lower()
        if text in {"true", "success", "passed", "1", "1.0"}:
            return True
        if text in {"false", "failed", "0", "0.0"}:
            return False
    return None


def _command_available(command: str, cwd: Path | None) -> bool:
    try:
        completed = subprocess.run([command, "--help"], cwd=cwd, text=True, capture_output=True, timeout=20)
        return completed.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


if __name__ == "__main__":
    main()
