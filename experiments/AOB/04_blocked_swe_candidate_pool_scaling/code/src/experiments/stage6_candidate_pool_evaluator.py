from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Sequence

from src.datasets.swe_patch_selection_dataset import (
    PatchSelectionExample,
    load_patch_selection_jsonl,
    validate_patch_selection_examples,
    write_patch_selection_jsonl,
)


DEFAULT_INPUT = Path("results/stage6_candidate_pools.jsonl")
DEFAULT_OUTPUT = Path("results/stage6_candidate_pools.jsonl")


@dataclass(frozen=True)
class PatchHarnessResult:
    candidate_id: str
    applied: bool
    passed: bool
    returncode: int
    stdout_tail: str
    stderr_tail: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage 6 fixed candidate patches with a benchmark harness.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--repo-root", default=None, help="Repository checkout to copy before applying each patch.")
    parser.add_argument("--test-command", default=None, help="Command to run inside the temp checkout after applying each patch.")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--trust-existing-labels", action="store_true")
    args = parser.parse_args()

    examples = load_patch_selection_jsonl(args.input)
    if args.trust_existing_labels:
        evaluated = examples
    else:
        if not args.repo_root or not args.test_command:
            raise SystemExit("provide --repo-root and --test-command, or pass --trust-existing-labels")
        evaluated = evaluate_candidate_pool_examples(
            examples,
            repo_root=Path(args.repo_root),
            test_command=args.test_command,
            timeout_seconds=int(args.timeout_seconds),
        )
    audit = validate_patch_selection_examples(evaluated)
    if not bool(audit["passes"]):
        raise SystemExit(str(audit))
    write_patch_selection_jsonl(args.output, evaluated)
    print(f"stage6 candidate pool evaluator: wrote evaluated labels for {len(evaluated)} examples to {args.output}")


def evaluate_candidate_pool_examples(
    examples: Sequence[PatchSelectionExample],
    repo_root: Path,
    test_command: str,
    timeout_seconds: int = 900,
) -> List[PatchSelectionExample]:
    evaluated: List[PatchSelectionExample] = []
    for example in examples:
        labels = []
        results = []
        for candidate in example.candidates:
            result = evaluate_patch_candidate(
                repo_root=repo_root,
                patch_diff=candidate.candidate_diff,
                candidate_id=candidate.candidate_id,
                test_command=test_command,
                timeout_seconds=timeout_seconds,
            )
            labels.append(1 if result.passed else 0)
            results.append(result.__dict__)
        evaluated.append(
            replace(
                example,
                labels_pass_fail=tuple(labels),
                metadata={
                    **example.metadata,
                    "stage6_candidate_evaluation": {
                        "repo_root": str(repo_root),
                        "test_command": test_command,
                        "timeout_seconds": int(timeout_seconds),
                        "results": results,
                    },
                },
            )
        )
    return evaluated


def evaluate_patch_candidate(
    repo_root: Path,
    patch_diff: str,
    candidate_id: str,
    test_command: str,
    timeout_seconds: int,
) -> PatchHarnessResult:
    if not repo_root.exists():
        raise FileNotFoundError(repo_root)
    with tempfile.TemporaryDirectory(prefix="stage6_patch_eval_") as temp:
        worktree = Path(temp) / "repo"
        ignore = shutil.ignore_patterns(".git", ".venv", "__pycache__", ".pytest_cache", "node_modules")
        shutil.copytree(repo_root, worktree, ignore=ignore)
        check = _run_git_apply(worktree, patch_diff, check_only=True, timeout_seconds=timeout_seconds)
        if check.returncode != 0:
            return PatchHarnessResult(
                candidate_id=candidate_id,
                applied=False,
                passed=False,
                returncode=int(check.returncode),
                stdout_tail=_tail(check.stdout),
                stderr_tail=_tail(check.stderr),
            )
        applied = _run_git_apply(worktree, patch_diff, check_only=False, timeout_seconds=timeout_seconds)
        if applied.returncode != 0:
            return PatchHarnessResult(
                candidate_id=candidate_id,
                applied=False,
                passed=False,
                returncode=int(applied.returncode),
                stdout_tail=_tail(applied.stdout),
                stderr_tail=_tail(applied.stderr),
            )
        run = subprocess.run(
            test_command,
            cwd=worktree,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        return PatchHarnessResult(
            candidate_id=candidate_id,
            applied=True,
            passed=run.returncode == 0,
            returncode=int(run.returncode),
            stdout_tail=_tail(run.stdout),
            stderr_tail=_tail(run.stderr),
        )


def _run_git_apply(worktree: Path, patch_diff: str, check_only: bool, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    command = ["git", "apply"]
    if check_only:
        command.append("--check")
    return subprocess.run(
        command,
        input=patch_diff,
        cwd=worktree,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
    )


def _tail(text: str, limit: int = 4000) -> str:
    value = str(text or "")
    return value[-limit:]


if __name__ == "__main__":
    main()
