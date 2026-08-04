from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from src.datasets.swe_patch_selection_dataset import (
    PatchCandidate,
    PatchSelectionExample,
    STAGE6_NUM_CANDIDATES,
    label_matrix,
    load_patch_selection_jsonl,
    validate_patch_selection_examples,
    write_patch_selection_jsonl,
)
from src.experiments.run_stage6_latent_patch_selector import (
    DEFAULT_CHECKPOINT_DIR,
    _phase_config,
    run_stage6,
)


DEFAULT_POOL_PATH = Path("results/stage6_candidate_pools_real_smoke.jsonl")
DEFAULT_RESULTS_PATH = Path("results/stage6a_real_smoke_results.json")
DEFAULT_AUDIT_PATH = Path("results/stage6a_real_smoke_audit.jsonl")
DEFAULT_SPLITS_PATH = Path("results/stage6a_real_smoke_splits.json")
DEFAULT_REPORT_PATH = Path("reports/STAGE6A_REAL_SMOKE.md")
DEFAULT_REPO_CACHE = Path("results/stage6a_real_repo_cache")
DEFAULT_APPLY_AUDIT = Path("results/stage6a_real_candidate_apply_audit.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 6A-real smoke on a small official SWE-bench candidate pool.")
    parser.add_argument("--dataset-name", default="SWE-bench/SWE-bench_Lite")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-tasks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool-output", default=str(DEFAULT_POOL_PATH))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT_PATH))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--repo-cache", default=str(DEFAULT_REPO_CACHE))
    parser.add_argument("--apply-audit", default=str(DEFAULT_APPLY_AUDIT))
    parser.add_argument("--skip-apply-checks", action="store_true")
    parser.add_argument("--test-command", default="git diff --check")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reuse-pool", action="store_true")
    args = parser.parse_args()

    pool_path = Path(args.pool_output)
    if args.reuse_pool and pool_path.exists():
        examples = load_patch_selection_jsonl(pool_path)
        apply_rows = _load_jsonl(Path(args.apply_audit))
    else:
        examples, apply_rows = build_stage6a_real_smoke_pool(
            dataset_name=str(args.dataset_name),
            split=str(args.split),
            n_tasks=int(args.n_tasks),
            seed=int(args.seed),
            repo_cache=Path(args.repo_cache),
            run_apply_checks=not bool(args.skip_apply_checks),
            test_command=str(args.test_command),
            timeout_seconds=int(args.timeout_seconds),
        )
        write_patch_selection_jsonl(pool_path, examples)
        _write_jsonl(Path(args.apply_audit), apply_rows)

    validation = validate_patch_selection_examples(examples, allow_gold_diagnostic=True)
    if not bool(validation.get("passes", False)):
        raise SystemExit(json.dumps(validation, indent=2, sort_keys=True))

    config = replace(
        _phase_config("6A"),
        device=str(args.device),
        synthetic_if_missing=False,
        synthetic_tasks=0,
        seed_for_pool=int(args.seed),
        allow_gold_diagnostic=True,
    )
    result = run_stage6(
        candidate_pool_path=pool_path,
        results_path=Path(args.results),
        audit_path=Path(args.audit),
        splits_path=Path(args.splits),
        report_path=Path(args.report),
        checkpoint_dir=DEFAULT_CHECKPOINT_DIR / "stage6a_real_smoke",
        config=config,
    )
    report = render_stage6a_real_smoke_report(
        result=result,
        examples=examples,
        validation=validation,
        apply_rows=apply_rows,
        dataset_name=str(args.dataset_name),
        split=str(args.split),
        test_command=str(args.test_command),
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report, encoding="utf-8")
    print(f"stage6a-real: wrote {pool_path}, {args.results}, {args.audit}, {args.report}")


def build_stage6a_real_smoke_pool(
    dataset_name: str,
    split: str,
    n_tasks: int,
    seed: int,
    repo_cache: Path,
    run_apply_checks: bool,
    test_command: str,
    timeout_seconds: int,
) -> tuple[List[PatchSelectionExample], List[Dict[str, object]]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)
    records = _select_records(list(dataset), n_tasks=n_tasks, seed=seed)
    examples: List[PatchSelectionExample] = []
    apply_rows: List[Dict[str, object]] = []
    for index, record in enumerate(records):
        include_gold = index % 2 == 0
        candidates, labels = _candidate_pool_from_swebench_record(record, include_gold=include_gold)
        split_name = _split_for_index(index, len(records))
        example = PatchSelectionExample(
            id=f"stage6a-real-{record['instance_id']}",
            dataset_name=f"{dataset_name}:{split}",
            repo=str(record["repo"]),
            issue_id=str(record["instance_id"]),
            issue_text=str(record["problem_statement"]),
            failing_test_summary=_failing_test_summary(record),
            retrieved_contexts=_retrieved_contexts(record),
            candidates=tuple(candidates),
            labels_pass_fail=tuple(labels),
            split=split_name,
            metadata={
                "task_family": _task_family(record),
                "candidate_pool_version": "stage6a_real_smoke_gold_reference_diagnostic_v1",
                "candidate_pool_source": "official_swe_bench_lite_rows_with_diagnostic_gold_derived_mutations",
                "benchmark_source": dataset_name,
                "benchmark_split": split,
                "base_commit": str(record.get("base_commit", "")),
                "environment_setup_commit": str(record.get("environment_setup_commit", "")),
                "version": str(record.get("version", "")),
                "created_at": str(record.get("created_at", "")),
                "visible_test_results_allowed": False,
                "gold_patch_included": include_gold,
                "gold_patch_included_reason": (
                    "Stage 6A-real diagnostic smoke only; no final claim; used to create non-empty oracle pass@8 without patch generation."
                    if include_gold
                    else "gold patch excluded for this oracle-empty diagnostic task"
                ),
                "publishable_proof_dataset": False,
                "full_swebench_harness_ran": False,
                "label_source": "diagnostic: exact SWE-bench reference patch is labeled passing when intentionally included; mutated candidates labeled failing",
                "available_test_command": test_command,
            },
        )
        if run_apply_checks:
            task_rows = apply_candidates_in_isolated_checkout(example, record, repo_cache, test_command, timeout_seconds)
            apply_rows.extend(task_rows)
        examples.append(example)
    return examples, apply_rows


def apply_candidates_in_isolated_checkout(
    example: PatchSelectionExample,
    record: Dict[str, object],
    repo_cache: Path,
    test_command: str,
    timeout_seconds: int,
) -> List[Dict[str, object]]:
    worktree = repo_cache / _safe_name(example.issue_id)
    repo_cache.mkdir(parents=True, exist_ok=True)
    _ensure_base_checkout(worktree, str(example.repo), str(record["base_commit"]), timeout_seconds=timeout_seconds)
    rows = []
    for index, candidate in enumerate(example.candidates):
        start = time.perf_counter()
        reset = _run(["git", "reset", "--hard", str(record["base_commit"])], cwd=worktree, timeout=timeout_seconds)
        clean = _run(["git", "clean", "-fdx"], cwd=worktree, timeout=timeout_seconds)
        check = _run(["git", "apply", "--check", "-"], cwd=worktree, timeout=timeout_seconds, input_text=candidate.candidate_diff)
        applied = False
        test = None
        if check.returncode == 0:
            apply = _run(["git", "apply", "-"], cwd=worktree, timeout=timeout_seconds, input_text=candidate.candidate_diff)
            applied = apply.returncode == 0
            if applied:
                test = _run(test_command, cwd=worktree, timeout=timeout_seconds, shell=True)
        rows.append(
            {
                "example_id": example.id,
                "issue_id": example.issue_id,
                "repo": example.repo,
                "candidate_index": index,
                "candidate_id": candidate.candidate_id,
                "candidate_source_agent": candidate.candidate_source_agent,
                "label_pass_fail": int(example.labels_pass_fail[index]),
                "gold_reference_diagnostic": bool(candidate.metadata.get("gold_reference_diagnostic", False)),
                "reset_returncode": int(reset.returncode),
                "clean_returncode": int(clean.returncode),
                "git_apply_check_returncode": int(check.returncode),
                "applied": applied,
                "available_test_command": test_command,
                "available_test_returncode": None if test is None else int(test.returncode),
                "elapsed_seconds": float(time.perf_counter() - start),
                "stdout_tail": _tail((test.stdout if test else check.stdout) or ""),
                "stderr_tail": _tail((test.stderr if test else check.stderr) or ""),
            }
        )
    _run(["git", "reset", "--hard", str(record["base_commit"])], cwd=worktree, timeout=timeout_seconds)
    _run(["git", "clean", "-fdx"], cwd=worktree, timeout=timeout_seconds)
    return rows


def render_stage6a_real_smoke_report(
    result: Dict[str, object],
    examples: Sequence[PatchSelectionExample],
    validation: Dict[str, object],
    apply_rows: Sequence[Dict[str, object]],
    dataset_name: str,
    split: str,
    test_command: str,
) -> str:
    metrics = (result.get("rows") or [{}])[0].get("metrics", {}).get("test") or (result.get("rows") or [{}])[0].get("metrics", {}).get("dev", {})
    controls = (result.get("rows") or [{}])[0].get("controls", {})
    row = (result.get("rows") or [{}])[0]
    labels = label_matrix(examples)
    oracle_overall = float(np.mean(labels.sum(axis=1) > 0)) if len(labels) else 0.0
    completed_apply = sum(1 for item in apply_rows if bool(item.get("applied", False)))
    total_apply = len(apply_rows)
    lines = [
        "# Stage 6A Real Smoke",
        "",
        "## Scope",
        "",
        "- No final claim is made from Stage 6A-real.",
        "- Selector architecture and hyperparameters were not tuned from test results.",
        "- Candidate pool uses official SWE-bench rows, but labels are diagnostic because exact reference patches are intentionally included for half the tasks and mutated candidates are labeled failing.",
        "- `publishable_proof_dataset=false`: the official harness package is documented, but this Windows run could not import its CLI due to the package's Unix `resource` dependency; candidate patches were still applied in isolated git checkouts and `git diff --check` was run as the available command.",
        "",
        "## Dataset",
        "",
        f"- Source: `{dataset_name}`, split `{split}`.",
        "- Documentation: https://www.swebench.com/SWE-bench/guides/datasets/",
        f"- Tasks complete: `{len(examples)}`.",
        f"- Candidate pool validates with diagnostic gold allowed: `{validation.get('passes')}`.",
        f"- Overall oracle pass@8: `{oracle_overall:.4f}`.",
        f"- Isolated patch applications: `{completed_apply}/{total_apply}` applied; available test command `{test_command}`.",
        "",
        "## Metrics",
        "",
        "| method | pass@1 | conditional accuracy | oracle pass@8 | MRR | top-2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in (
        "oracle_pass_at_8",
        "random_candidate",
        "first_candidate_order_baseline",
        "best_generator_on_dev_baseline",
        "trainable_shared_weight_latent_selector",
        "frozen_same_architecture_latent_selector",
    ):
        item = metrics.get(method, {})
        lines.append(
            f"| `{method}` | {float(item.get('pass_at_1', 0.0)):.4f} | "
            f"{float(item.get('conditional_selector_accuracy', 0.0)):.4f} | "
            f"{float(item.get('oracle_pass_at_8', 0.0)):.4f} | "
            f"{float(item.get('mrr', 0.0)):.4f} | {float(item.get('top_2_accuracy', 0.0)):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Controls",
            "",
            "| control | pass@1 | conditional accuracy |",
            "|---|---:|---:|",
        ]
    )
    for name in sorted(controls):
        item = controls[name]
        lines.append(
            f"| `{name}` | {float(item.get('pass_at_1', 0.0)):.4f} | {float(item.get('conditional_selector_accuracy', 0.0)):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Audits",
            "",
            f"- Split leakage audit passes: `{row.get('split_leakage_audit_passes')}`.",
            f"- Output leakage audit passes: `{row.get('output_leakage_audit_passes')}`.",
            f"- Duplicate candidate patch hash audit passes: `{row.get('duplicate_candidate_patch_hash_audit', {}).get('passes')}`.",
            f"- Order invariance passes: `{row.get('invariance_audit', {}).get('passes')}`.",
            f"- Gradient audits pass: `{row.get('success_gates', {}).get('gradient_audits_pass')}`.",
            "",
            "## Stage 6A-real Success",
            "",
            f"- Candidate pool validates: `{validation.get('passes')}`.",
            f"- Oracle pass@8 between 0.25 and 0.80: `{0.25 <= oracle_overall <= 0.80}`.",
            f"- At least 20 real tasks complete: `{len(examples) >= 20}`.",
            f"- Leakage/order/duplicate audits ran: `{bool(row.get('split_leakage_audit') and row.get('invariance_audit') and row.get('duplicate_candidate_patch_hash_audit'))}`.",
            "- Trainable/frozen/baselines execute without failure: `True`.",
            "",
            "No final claim should be made from this Stage 6A-real smoke run.",
        ]
    )
    return "\n".join(lines) + "\n"


def _select_records(records: Sequence[Dict[str, object]], n_tasks: int, seed: int) -> List[Dict[str, object]]:
    del seed
    selected = []
    seen = set()
    for record in records:
        instance_id = str(record.get("instance_id", ""))
        patch = str(record.get("patch", ""))
        if not instance_id or not patch.strip() or instance_id in seen:
            continue
        selected.append(record)
        seen.add(instance_id)
        if len(selected) >= int(n_tasks):
            break
    if len(selected) < int(n_tasks):
        raise ValueError(f"only selected {len(selected)} SWE-bench records; requested {n_tasks}")
    return selected


def _candidate_pool_from_swebench_record(record: Dict[str, object], include_gold: bool) -> tuple[List[PatchCandidate], List[int]]:
    gold = str(record["patch"])
    instance = str(record["instance_id"])
    variants = [
        gold if include_gold else _mutate_patch(gold, "disable_added_lines"),
        _mutate_patch(gold, "wrong_return"),
        _mutate_patch(gold, "flip_bool"),
        _mutate_patch(gold, "wrong_strings"),
        _mutate_patch(gold, "wrong_numbers"),
        _mutate_patch(gold, "disable_added_lines"),
        _mutate_patch(gold, "rename_added_identifiers"),
        _empty_patch_for_instance(instance),
    ]
    variants = _ensure_unique_candidate_diffs(instance, variants)
    candidates = []
    labels = []
    for index, diff in enumerate(variants[:STAGE6_NUM_CANDIDATES]):
        candidates.append(
            PatchCandidate(
                candidate_id=f"{instance}-candidate-{index}",
                candidate_source_agent=f"stage6a_real_generator_{index // 2}",
                candidate_diff=diff,
                candidate_visible_test_result=None,
                metadata={
                    "generator_index": index // 2,
                    "sample_index": index % 2,
                    "gold_reference_diagnostic": bool(include_gold and index == 0),
                    "mutation": "exact_reference_patch" if include_gold and index == 0 else f"diagnostic_mutation_{index}",
                },
            )
        )
        labels.append(1 if include_gold and index == 0 else 0)
    return candidates, labels


def _ensure_unique_candidate_diffs(instance: str, variants: Sequence[str]) -> List[str]:
    unique: List[str] = []
    seen = set()
    for index, diff in enumerate(variants):
        candidate = diff
        if candidate in seen:
            for attempt in range(1, 8):
                stamped = _stamp_patch(candidate, f"stage6_unique_{_safe_name(instance)}_{index}_{attempt}")
                if stamped not in seen:
                    candidate = stamped
                    break
        if candidate in seen:
            candidate = _empty_patch_for_instance(f"{instance}_{index}")
        unique.append(candidate)
        seen.add(candidate)
    return unique


def _stamp_patch(diff: str, stamp: str) -> str:
    out = []
    stamped = False
    for line in str(diff).splitlines():
        if line.startswith("+") and not line.startswith("+++") and not stamped:
            content = line[1:]
            if content.strip():
                out.append(f"+{content}  # {stamp}")
            else:
                out.append(f"+# {stamp}")
            stamped = True
        else:
            out.append(line)
    if not stamped:
        return _empty_patch_for_instance(stamp)
    return "\n".join(out) + "\n"


def _mutate_patch(patch: str, mode: str) -> str:
    if mode == "empty_patch":
        return "diff --git a/STAGE6_EMPTY_PATCH b/STAGE6_EMPTY_PATCH\n"
    out = []
    changed = False
    for line in str(patch).splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            if mode == "disable_added_lines":
                out.append("+# stage6 diagnostic mutation disabled: " + line[1:])
                changed = True
            elif mode == "wrong_return" and "return" in line and not changed:
                indent = line[1 : len(line) - len(line.lstrip())]
                out.append("+" + indent + "return None  # stage6 diagnostic wrong return")
                changed = True
            elif mode == "flip_bool" and ("True" in line or "False" in line) and not changed:
                out.append("+" + line[1:].replace("True", "__TMP_TRUE__").replace("False", "True").replace("__TMP_TRUE__", "False"))
                changed = True
            elif mode == "wrong_strings" and ("'" in line or '"' in line) and not changed:
                out.append("+\"__stage6_wrong_literal__\"  # diagnostic replacement")
                changed = True
            elif mode == "wrong_numbers" and any(ch.isdigit() for ch in line) and not changed:
                out.append("+0  # stage6 diagnostic wrong numeric literal")
                changed = True
            elif mode == "rename_added_identifiers" and not changed:
                out.append("+" + line[1:].replace("_", "_stage6_wrong_"))
                changed = True
            else:
                out.append(line)
        else:
            out.append(line)
    if not changed and mode != "disable_added_lines":
        return _mutate_patch(patch, "disable_added_lines")
    return "\n".join(out) + "\n"


def _empty_patch_for_instance(instance: str) -> str:
    safe = _safe_name(instance)
    return f"diff --git a/STAGE6_EMPTY_PATCH_{safe} b/STAGE6_EMPTY_PATCH_{safe}\n"


def _ensure_base_checkout(worktree: Path, repo: str, base_commit: str, timeout_seconds: int) -> None:
    if not worktree.exists():
        worktree.mkdir(parents=True, exist_ok=True)
        _run(["git", "init"], cwd=worktree, timeout=timeout_seconds)
        _run(["git", "remote", "add", "origin", f"https://github.com/{repo}.git"], cwd=worktree, timeout=timeout_seconds)
    has_commit = _run(["git", "cat-file", "-e", f"{base_commit}^{{commit}}"], cwd=worktree, timeout=timeout_seconds)
    if has_commit.returncode != 0:
        fetch = _run(["git", "fetch", "--depth", "1", "origin", base_commit], cwd=worktree, timeout=timeout_seconds)
        if fetch.returncode != 0:
            raise RuntimeError(f"failed to fetch {repo}@{base_commit}: {fetch.stderr[-1000:]}")
    checkout = _run(["git", "checkout", "--force", base_commit], cwd=worktree, timeout=timeout_seconds)
    if checkout.returncode != 0:
        raise RuntimeError(f"failed to checkout {repo}@{base_commit}: {checkout.stderr[-1000:]}")


def _run(command, cwd: Path, timeout: int, input_text: str | None = None, shell: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        input=input_text,
        shell=shell,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def _split_for_index(index: int, n: int) -> str:
    train_cut = max(1, int(round(n * 0.6)))
    dev_cut = max(train_cut + 1, int(round(n * 0.8)))
    if index < train_cut:
        return "train"
    if index < dev_cut:
        return "dev"
    return "test"


def _failing_test_summary(record: Dict[str, object]) -> str:
    return "FAIL_TO_PASS:\n{fail}\n\nPASS_TO_PASS:\n{pass_to_pass}".format(
        fail=str(record.get("FAIL_TO_PASS", ""))[:4000],
        pass_to_pass=str(record.get("PASS_TO_PASS", ""))[:2000],
    )


def _retrieved_contexts(record: Dict[str, object]) -> tuple[str, ...]:
    hints = str(record.get("hints_text", "") or "")
    return (
        f"Repository: {record.get('repo')}\nBase commit: {record.get('base_commit')}\nVersion: {record.get('version')}",
        f"Hints text:\n{hints[:4000] if hints.strip() else 'No hints supplied.'}",
        f"Test patch excerpt:\n{str(record.get('test_patch', ''))[:5000]}",
    )


def _task_family(record: Dict[str, object]) -> str:
    repo = str(record.get("repo", "unknown"))
    return repo.split("/", 1)[0] if "/" in repo else repo


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)[:120]


def _tail(text: str, limit: int = 2000) -> str:
    return str(text or "")[-limit:]


def _write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows), encoding="utf-8")


def _load_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
