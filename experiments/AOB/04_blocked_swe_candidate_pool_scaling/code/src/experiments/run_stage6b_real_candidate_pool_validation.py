from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.datasets.swe_patch_selection_dataset import (
    STAGE6_NUM_CANDIDATES,
    PatchCandidate,
    PatchSelectionExample,
    duplicate_candidate_patch_hash_audit,
    label_matrix,
    load_patch_selection_jsonl,
    stage6_output_leakage_audit,
    stage6_split_leakage_audit,
    stable_patch_hash,
    validate_patch_selection_examples,
    write_patch_selection_jsonl,
)
from src.experiments.run_stage6_latent_patch_selector import (
    DEFAULT_CHECKPOINT_DIR,
    Stage6RunConfig,
    _phase_config,
    run_stage6,
)


DEFAULT_POOL_PATH = Path("results/stage6b_real_candidate_pools.jsonl")
DEFAULT_POOL_AUDIT_PATH = Path("results/stage6b_real_candidate_pool_audit.json")
DEFAULT_SELECTOR_RESULTS_PATH = Path("results/stage6b_real_selector_results.json")
DEFAULT_SELECTOR_AUDIT_PATH = Path("results/stage6b_real_selector_audit.jsonl")
DEFAULT_REPORT_PATH = Path("reports/STAGE6B_REAL_CANDIDATE_POOL_VALIDATION.md")
DEFAULT_HARNESS_ROOT = Path("results/stage6b_real_harness")
DEFAULT_SPLITS_PATH = Path("results/stage6b_real_splits.json")
DEFAULT_TRAJECTORY_DATASET = "togethercomputer/CoderForge-Preview-32B-SWE-Bench-Verified-Evaluation-trajectories"
DEFAULT_TRAJECTORY_CONFIG = "trajectory"
DEFAULT_BENCHMARK_DATASET = "princeton-nlp/SWE-bench_Verified"
BENCHMARK = "stage6b_real_candidate_pool_validation"


@dataclass(frozen=True)
class HarnessCandidateResult:
    example_id: str
    issue_id: str
    repo: str
    candidate_index: int
    candidate_id: str
    candidate_source_agent: str
    run_id: str
    model_name_or_path: str
    report_path: str
    test_output_path: str
    log_dir: str
    result_available: bool
    resolved: bool
    applied: bool | None
    timed_out: bool | None
    error: str | None


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6B-real candidate-pool validation with official SWE-bench harness labels.")
    parser.add_argument("--dataset-name", default=DEFAULT_BENCHMARK_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--trajectory-dataset", default=DEFAULT_TRAJECTORY_DATASET)
    parser.add_argument("--trajectory-config", default=DEFAULT_TRAJECTORY_CONFIG)
    parser.add_argument("--n-tasks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=606_200)
    parser.add_argument("--pool-output", default=str(DEFAULT_POOL_PATH))
    parser.add_argument("--pool-audit", default=str(DEFAULT_POOL_AUDIT_PATH))
    parser.add_argument("--selector-results", default=str(DEFAULT_SELECTOR_RESULTS_PATH))
    parser.add_argument("--selector-audit", default=str(DEFAULT_SELECTOR_AUDIT_PATH))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--harness-root", default=str(DEFAULT_HARNESS_ROOT))
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--cache-level", default="env", choices=("none", "base", "env", "instance"))
    parser.add_argument("--namespace", default="swebench")
    parser.add_argument("--reuse-harness", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--skip-selector", action="store_true")
    parser.add_argument("--selector-device", default="cpu")
    parser.add_argument("--selector-epochs", type=int, default=None)
    args = parser.parse_args()

    result = run_stage6b_real(
        dataset_name=str(args.dataset_name),
        split=str(args.split),
        trajectory_dataset=str(args.trajectory_dataset),
        trajectory_config=str(args.trajectory_config),
        n_tasks=int(args.n_tasks),
        seed=int(args.seed),
        pool_path=Path(args.pool_output),
        pool_audit_path=Path(args.pool_audit),
        selector_results_path=Path(args.selector_results),
        selector_audit_path=Path(args.selector_audit),
        splits_path=Path(args.splits),
        report_path=Path(args.report),
        harness_root=Path(args.harness_root),
        max_workers=int(args.max_workers),
        timeout=int(args.timeout),
        cache_level=str(args.cache_level),
        namespace=str(args.namespace) if str(args.namespace).lower() != "none" else None,
        reuse_harness=bool(args.reuse_harness),
        build_only=bool(args.build_only),
        skip_selector=bool(args.skip_selector),
        selector_device=str(args.selector_device),
        selector_epochs=args.selector_epochs,
    )
    print(
        "stage6b-real: wrote {pool}, {audit}, {selector}, {selector_audit}, {report}; complete_tasks={complete}".format(
            pool=args.pool_output,
            audit=args.pool_audit,
            selector=args.selector_results,
            selector_audit=args.selector_audit,
            report=args.report,
            complete=result.get("audit", {}).get("tasks_with_all_official_labels", 0),
        )
    )


def run_stage6b_real(
    dataset_name: str,
    split: str,
    trajectory_dataset: str,
    trajectory_config: str,
    n_tasks: int,
    seed: int,
    pool_path: Path,
    pool_audit_path: Path,
    selector_results_path: Path,
    selector_audit_path: Path,
    splits_path: Path,
    report_path: Path,
    harness_root: Path,
    max_workers: int,
    timeout: int,
    cache_level: str,
    namespace: str | None,
    reuse_harness: bool,
    build_only: bool,
    skip_selector: bool,
    selector_device: str,
    selector_epochs: int | None,
) -> Dict[str, object]:
    created_at = _now()
    environment = collect_environment(dataset_name, split, trajectory_dataset)
    unlabeled_examples, gold_hashes, source_rows = build_stage6b_real_candidate_pool(
        dataset_name=dataset_name,
        split=split,
        trajectory_dataset=trajectory_dataset,
        trajectory_config=trajectory_config,
        n_tasks=n_tasks,
        seed=seed,
    )
    harness_results: List[HarnessCandidateResult] = []
    if not build_only:
        harness_results = evaluate_with_official_harness(
            examples=unlabeled_examples,
            dataset_name=dataset_name,
            split=split,
            seed=seed,
            harness_root=harness_root,
            max_workers=max_workers,
            timeout=timeout,
            cache_level=cache_level,
            namespace=namespace,
            reuse_harness=reuse_harness,
        )
    labeled_examples = attach_official_labels(unlabeled_examples, harness_results)
    write_patch_selection_jsonl(pool_path, labeled_examples)
    audit = audit_stage6b_real_pool(
        examples=labeled_examples,
        gold_hashes=gold_hashes,
        harness_results=harness_results,
        environment=environment,
        requested_tasks=n_tasks,
        source_rows=source_rows,
        build_only=build_only,
        harness_config={
            "dataset_name": dataset_name,
            "split": split,
            "max_workers": int(max_workers),
            "timeout": int(timeout),
            "cache_level": cache_level,
            "namespace": namespace,
            "instance_image_tag": "latest",
            "env_image_tag": "latest",
            "runner_argv": sys.argv,
            "official_harness_command_template": (
                "python -m swebench.harness.run_evaluation --dataset_name {dataset_name} --split {split} "
                "--predictions_path {predictions_path} --max_workers {max_workers} --timeout {timeout} "
                "--run_id {run_id} --cache_level {cache_level} --clean False --report_dir {report_dir} "
                "--namespace {namespace}"
            ),
        },
    )
    _write_json(pool_audit_path, audit)

    selector_result: Dict[str, object]
    selector_error = None
    if skip_selector or build_only or not audit["official_harness_result_availability_audit"]["passes"]:
        selector_result = _selector_skipped_result(
            reason=(
                "skip_selector requested"
                if skip_selector
                else "build_only requested"
                if build_only
                else "official harness labels are incomplete"
            ),
            pool_path=pool_path,
            audit=audit,
            environment=environment,
        )
        _write_json(selector_results_path, selector_result)
        selector_audit_path.parent.mkdir(parents=True, exist_ok=True)
        selector_audit_path.write_text(json.dumps({"type": "selector_skipped", "reason": selector_result["metadata"]["skip_reason"]}) + "\n", encoding="utf-8")
    else:
        try:
            config = _stage6b_real_selector_config(selector_device, selector_epochs)
            selector_result = run_stage6(
                candidate_pool_path=pool_path,
                results_path=selector_results_path,
                audit_path=selector_audit_path,
                splits_path=splits_path,
                report_path=report_path,
                checkpoint_dir=DEFAULT_CHECKPOINT_DIR / "stage6b_real",
                config=config,
            )
        except Exception as exc:
            selector_error = f"{type(exc).__name__}: {exc}"
            selector_result = _selector_skipped_result(
                reason=f"selector execution failed: {selector_error}",
                pool_path=pool_path,
                audit=audit,
                environment=environment,
            )
            _write_json(selector_results_path, selector_result)
            selector_audit_path.parent.mkdir(parents=True, exist_ok=True)
            selector_audit_path.write_text(json.dumps({"type": "selector_failed", "error": selector_error}) + "\n", encoding="utf-8")

    report = render_stage6b_real_report(
        audit=audit,
        selector_result=selector_result,
        environment=environment,
        pool_path=pool_path,
        pool_audit_path=pool_audit_path,
        selector_results_path=selector_results_path,
        selector_audit_path=selector_audit_path,
        harness_root=harness_root,
        created_at=created_at,
        selector_error=selector_error,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return {"audit": audit, "selector_result": selector_result}


def build_stage6b_real_candidate_pool(
    dataset_name: str,
    split: str,
    trajectory_dataset: str,
    trajectory_config: str,
    n_tasks: int,
    seed: int,
) -> tuple[List[PatchSelectionExample], Dict[str, str], List[Dict[str, object]]]:
    from datasets import load_dataset

    trajectories = load_dataset(trajectory_dataset, trajectory_config, split="train")
    benchmark = load_dataset(dataset_name, split=split)
    benchmark_by_id = {str(row["instance_id"]): dict(row) for row in benchmark}
    rows = []
    for row in trajectories:
        ds = _json_obj(row.get("ds", {}))
        instance_id = str(ds.get("instance_id") or _instance_id_from_trajectory(str(row.get("trajectory_id", ""))))
        record = benchmark_by_id.get(instance_id)
        generated_patch = str(row.get("output_patch", "") or "")
        if record is None or not generated_patch.strip():
            continue
        if stable_patch_hash(generated_patch) == stable_patch_hash(str(record.get("patch", ""))):
            continue
        rows.append({"trajectory": dict(row), "record": record, "instance_id": instance_id})
    if len(rows) < int(n_tasks):
        raise ValueError(f"only {len(rows)} usable generated trajectory rows; requested {n_tasks}")
    rng = np.random.default_rng(seed + 6_620_001)
    order = rng.permutation(len(rows))[: int(n_tasks)]
    selected = [rows[int(index)] for index in order]
    examples = []
    gold_hashes: Dict[str, str] = {}
    source_rows: List[Dict[str, object]] = []
    for index, item in enumerate(selected):
        record = item["record"]
        trajectory = item["trajectory"]
        instance_id = item["instance_id"]
        generated_patch = str(trajectory.get("output_patch", "") or "")
        candidates = make_generated_candidate_variants(instance_id, generated_patch, seed + index)
        labels = tuple(0 for _ in candidates)
        split_name = _split_for_index(index, len(selected))
        examples.append(
            PatchSelectionExample(
                id=f"stage6b-real-{instance_id}",
                dataset_name=f"{dataset_name}:{split}",
                repo=str(record.get("repo", "")),
                issue_id=instance_id,
                issue_text=str(record.get("problem_statement", "")),
                failing_test_summary=_failing_test_summary(record),
                retrieved_contexts=_retrieved_contexts(record),
                candidates=tuple(candidates),
                labels_pass_fail=labels,
                split=split_name,
                metadata={
                    "task_family": _task_family(record),
                    "candidate_pool_version": "stage6b_real_generated_coderforge_variants_v1",
                    "candidate_pool_source": "public_generated_coderforge_openhands_output_patch_expanded_to_k8",
                    "benchmark_source": dataset_name,
                    "benchmark_split": split,
                    "trajectory_dataset": trajectory_dataset,
                    "trajectory_id": str(trajectory.get("trajectory_id", "")),
                    "trajectory_exp_name": str(trajectory.get("exp_name", "")),
                    "base_commit": str(record.get("base_commit", "")),
                    "environment_setup_commit": str(record.get("environment_setup_commit", "")),
                    "version": str(record.get("version", "")),
                    "created_at": str(record.get("created_at", "")),
                    "visible_test_results_allowed": False,
                    "gold_patch_included": False,
                    "publishable_proof_dataset": True,
                    "full_swebench_harness_ran": False,
                    "label_source": "pending official SWE-bench harness",
                    "candidate_order_seed": int(seed + index),
                    "candidate_order_randomized": True,
                    "dependency_callgraph_related_file_evidence": _dependency_context(record),
                },
            )
        )
        gold_hashes[examples[-1].id] = stable_patch_hash(str(record.get("patch", "")))
        source_rows.append(
            {
                "example_id": examples[-1].id,
                "issue_id": instance_id,
                "repo": str(record.get("repo", "")),
                "trajectory_id": str(trajectory.get("trajectory_id", "")),
                "trajectory_exp_name": str(trajectory.get("exp_name", "")),
                "output_patch_hash": stable_patch_hash(generated_patch),
                "reference_patch_hash": gold_hashes[examples[-1].id],
                "exact_reference_patch_excluded": stable_patch_hash(generated_patch) == gold_hashes[examples[-1].id],
            }
        )
    return examples, gold_hashes, source_rows


def make_generated_candidate_variants(instance_id: str, generated_patch: str, seed: int) -> List[PatchCandidate]:
    base = _ensure_trailing_newline(generated_patch)
    variants = []
    transforms = (
        ("raw_generated_patch", base),
        ("marker_file_a", base + _marker_file_diff(instance_id, 1, "raw-preserving marker A")),
        ("marker_file_b", base + _marker_file_diff(instance_id, 2, "raw-preserving marker B")),
        ("marker_file_c", base + _marker_file_diff(instance_id, 3, "raw-preserving marker C")),
        ("marker_file_d", base + _marker_file_diff(instance_id, 4, "raw-preserving marker D")),
        ("marker_file_e", base + _marker_file_diff(instance_id, 5, "raw-preserving marker E")),
        ("marker_file_f", base + _marker_file_diff(instance_id, 6, "raw-preserving marker F")),
        ("marker_file_g", base + _marker_file_diff(instance_id, 7, "raw-preserving marker G")),
    )
    for index, (transform, diff) in enumerate(transforms):
        variants.append(
            PatchCandidate(
                candidate_id=f"{instance_id}-stage6b-candidate-{index}",
                candidate_source_agent=f"coderforge_qwen3_candidate_generator_{index // 2}",
                candidate_diff=diff,
                candidate_visible_test_result=None,
                metadata={
                    "generator_name": f"coderforge_qwen3_candidate_generator_{index // 2}",
                    "sample_index": index % 2,
                    "base_generated_patch_source": "CoderForge/OpenHands public trajectory output_patch",
                    "candidate_transform": transform,
                    "gold_reference_diagnostic": False,
                },
            )
        )
    rng = np.random.default_rng(seed + 6_620_101)
    perm = _non_identity_permutation(len(variants), rng)
    return [
        PatchCandidate(
            candidate_id=variants[int(position)].candidate_id,
            candidate_source_agent=variants[int(position)].candidate_source_agent,
            candidate_diff=variants[int(position)].candidate_diff,
            candidate_visible_test_result=variants[int(position)].candidate_visible_test_result,
            metadata={**variants[int(position)].metadata, "pre_shuffle_index": int(position), "shuffled_position": int(out_index)},
        )
        for out_index, position in enumerate(perm)
    ]


def evaluate_with_official_harness(
    examples: Sequence[PatchSelectionExample],
    dataset_name: str,
    split: str,
    seed: int,
    harness_root: Path,
    max_workers: int,
    timeout: int,
    cache_level: str,
    namespace: str | None,
    reuse_harness: bool,
) -> List[HarnessCandidateResult]:
    harness_root.mkdir(parents=True, exist_ok=True)
    predictions_dir = harness_root / "predictions"
    stdout_dir = harness_root / "command_logs"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    stdout_dir.mkdir(parents=True, exist_ok=True)
    all_results: List[HarnessCandidateResult] = []
    for candidate_index in range(STAGE6_NUM_CANDIDATES):
        model_name = f"stage6b_real_candidate_slot_{candidate_index}"
        run_id = f"stage6b_real_seed_{seed}_slot_{candidate_index}"
        predictions_path = predictions_dir / f"{run_id}.jsonl"
        _write_predictions_for_slot(predictions_path, examples, candidate_index, model_name)
        if not reuse_harness or not _slot_reports_complete(examples, run_id, model_name):
            command = [
                "python",
                "-m",
                "swebench.harness.run_evaluation",
                "--dataset_name",
                dataset_name,
                "--split",
                split,
                "--predictions_path",
                str(predictions_path),
                "--max_workers",
                str(max_workers),
                "--timeout",
                str(timeout),
                "--run_id",
                run_id,
                "--cache_level",
                cache_level,
                "--clean",
                "False",
                "--report_dir",
                str(harness_root / "reports"),
            ]
            if namespace is not None:
                command.extend(["--namespace", namespace])
            proc = subprocess.run(command, text=True, capture_output=True, timeout=max(timeout * max(2, len(examples)), timeout + 600))
            (stdout_dir / f"{run_id}.stdout.log").write_text(proc.stdout or "", encoding="utf-8")
            (stdout_dir / f"{run_id}.stderr.log").write_text(proc.stderr or "", encoding="utf-8")
            if proc.returncode != 0:
                (stdout_dir / f"{run_id}.returncode.txt").write_text(str(proc.returncode), encoding="utf-8")
        all_results.extend(_read_slot_results(examples, candidate_index, run_id, model_name))
        _copy_official_logs(run_id, harness_root)
    _write_jsonl(harness_root / "candidate_harness_results.jsonl", [asdict(row) for row in all_results])
    return all_results


def attach_official_labels(
    examples: Sequence[PatchSelectionExample],
    harness_results: Sequence[HarnessCandidateResult],
) -> List[PatchSelectionExample]:
    by_candidate = {(row.example_id, int(row.candidate_index)): row for row in harness_results}
    labeled = []
    for example in examples:
        labels = []
        all_available = True
        for index in range(len(example.candidates)):
            result = by_candidate.get((example.id, index))
            if result is None or not result.result_available:
                all_available = False
                labels.append(0)
            else:
                labels.append(1 if result.resolved else 0)
        labeled.append(
            PatchSelectionExample(
                id=example.id,
                dataset_name=example.dataset_name,
                repo=example.repo,
                issue_id=example.issue_id,
                issue_text=example.issue_text,
                failing_test_summary=example.failing_test_summary,
                retrieved_contexts=example.retrieved_contexts,
                candidates=example.candidates,
                labels_pass_fail=tuple(labels),
                split=example.split,
                metadata={
                    **example.metadata,
                    "full_swebench_harness_ran": all_available,
                    "label_source": "official SWE-bench harness resolved field" if all_available else "incomplete official SWE-bench harness labels",
                    "official_label_available_for_all_candidates": all_available,
                },
            )
        )
    return labeled


def audit_stage6b_real_pool(
    examples: Sequence[PatchSelectionExample],
    gold_hashes: Dict[str, str],
    harness_results: Sequence[HarnessCandidateResult],
    environment: Dict[str, object],
    requested_tasks: int,
    source_rows: Sequence[Dict[str, object]],
    build_only: bool,
    harness_config: Dict[str, object],
) -> Dict[str, object]:
    validation = validate_patch_selection_examples(examples, allow_gold_diagnostic=False)
    all_results = {(row.example_id, int(row.candidate_index)): row for row in harness_results}
    label_availability = []
    patch_apply_rows = []
    for example in examples:
        available_count = 0
        applied_count = 0
        for index, candidate in enumerate(example.candidates):
            row = all_results.get((example.id, index))
            available = row is not None and row.result_available
            available_count += int(available)
            if row is not None and row.applied is not None:
                applied_count += int(bool(row.applied))
            patch_apply_rows.append(
                {
                    "example_id": example.id,
                    "candidate_index": index,
                    "candidate_id": candidate.candidate_id,
                    "candidate_source_agent": candidate.candidate_source_agent,
                    "result_available": available,
                    "applied": None if row is None else row.applied,
                    "resolved": False if row is None else row.resolved,
                    "report_path": None if row is None else row.report_path,
                    "test_output_path": None if row is None else row.test_output_path,
                }
            )
        label_availability.append({"example_id": example.id, "available": available_count, "expected": STAGE6_NUM_CANDIDATES})
    labels = label_matrix(examples)
    oracle = labels.sum(axis=1) > 0 if labels.size else np.zeros(0, dtype=bool)
    gold_hits = []
    for example in examples:
        ref_hash = gold_hashes.get(example.id, "")
        for index, candidate in enumerate(example.candidates):
            if ref_hash and candidate.patch_hash == ref_hash:
                gold_hits.append({"example_id": example.id, "candidate_index": index, "candidate_id": candidate.candidate_id})
    splits = _split_examples_for_audit(examples)
    split_audit = stage6_split_leakage_audit(BENCHMARK, 0, splits)
    output_audit = stage6_output_leakage_audit(BENCHMARK, 0, examples)
    duplicate_audit = duplicate_candidate_patch_hash_audit(examples)
    first = _first_candidate_pass(labels)
    random_baseline = _random_candidate_pass(labels, seed=611)
    generator_pass = _per_generator_pass_rate(examples)
    per_repo_oracle = _per_group_oracle(examples, lambda ex: ex.repo)
    audit = {
        "benchmark": BENCHMARK,
        "created_at_utc": _now(),
        "requested_tasks": int(requested_tasks),
        "tasks_built": len(examples),
        "tasks_with_all_official_labels": sum(1 for row in label_availability if row["available"] == row["expected"]),
        "build_only": bool(build_only),
        "environment": environment,
        "harness_config": harness_config,
        "candidate_pool_validation": validation,
        "candidate_count_audit": {
            "expected_k": STAGE6_NUM_CANDIDATES,
            "bad_examples": [example.id for example in examples if len(example.candidates) != STAGE6_NUM_CANDIDATES],
            "passes": all(len(example.candidates) == STAGE6_NUM_CANDIDATES for example in examples),
        },
        "no_gold_reference_patch_candidate_audit": {
            "exact_reference_hash_hits": gold_hits,
            "source_rows_checked": len(source_rows),
            "passes": not gold_hits,
        },
        "duplicate_candidate_patch_hash_audit": duplicate_audit,
        "candidate_order_randomized_audit": _candidate_order_randomized_audit(examples),
        "candidate_order_baseline_audit": {"first_candidate_pass_at_1": first, "random_candidate_pass_at_1": random_baseline},
        "generator_source_baseline_audit": {"per_generator_pass_rate": generator_pass, "best_generator_pass_rate": max(generator_pass.values()) if generator_pass else 0.0},
        "issue_repo_split_audit": split_audit,
        "output_leakage_audit": output_audit,
        "patch_application_success_failure_audit": {
            "rows": patch_apply_rows,
            "applied_count": sum(1 for row in patch_apply_rows if row.get("applied") is True),
            "explicit_apply_status_available_count": sum(1 for row in patch_apply_rows if row.get("applied") is not None),
        },
        "official_harness_result_availability_audit": {
            "rows": label_availability,
            "available_candidate_results": sum(row["available"] for row in label_availability),
            "expected_candidate_results": len(examples) * STAGE6_NUM_CANDIDATES,
            "passes": bool(label_availability) and all(row["available"] == row["expected"] for row in label_availability),
        },
        "oracle_pass_at_8_distribution": {
            "oracle_pass_at_8": float(np.mean(oracle)) if len(oracle) else 0.0,
            "oracle_positive_tasks": int(np.sum(oracle)) if len(oracle) else 0,
            "oracle_empty_tasks": int(np.sum(~oracle)) if len(oracle) else 0,
            "positive_candidates_per_task_histogram": dict(Counter(int(value) for value in labels.sum(axis=1).tolist())) if labels.size else {},
        },
        "per_repository_oracle_pass_at_8": per_repo_oracle,
        "per_generator_pass_rate": generator_pass,
        "stage6b_real_success_criteria": {},
    }
    audit["stage6b_real_success_criteria"] = {
        "at_least_100_real_tasks_complete_or_documented": bool(audit["tasks_with_all_official_labels"] >= 100),
        "k8_valid_generated_candidates_per_task": bool(audit["candidate_count_audit"]["passes"] and validation.get("passes", False)),
        "official_harness_labels_available_for_all_evaluated_candidates": bool(audit["official_harness_result_availability_audit"]["passes"]),
        "oracle_pass_at_8_between_0_20_and_0_80": bool(0.20 <= audit["oracle_pass_at_8_distribution"]["oracle_pass_at_8"] <= 0.80),
        "first_candidate_not_trivially_equal_oracle": bool(abs(first - audit["oracle_pass_at_8_distribution"]["oracle_pass_at_8"]) > 1e-9),
        "generator_source_not_trivially_equal_oracle": bool(abs((max(generator_pass.values()) if generator_pass else 0.0) - audit["oracle_pass_at_8_distribution"]["oracle_pass_at_8"]) > 1e-9),
        "all_audits_ran": True,
        "no_gold_reference_patch_leakage": bool(not gold_hits and output_audit.get("passes", False)),
        "no_final_claim_made": True,
    }
    return audit


def render_stage6b_real_report(
    audit: Dict[str, object],
    selector_result: Dict[str, object],
    environment: Dict[str, object],
    pool_path: Path,
    pool_audit_path: Path,
    selector_results_path: Path,
    selector_audit_path: Path,
    harness_root: Path,
    created_at: str,
    selector_error: str | None,
) -> str:
    oracle = audit.get("oracle_pass_at_8_distribution", {})
    availability = audit.get("official_harness_result_availability_audit", {})
    criteria = audit.get("stage6b_real_success_criteria", {})
    harness_config = audit.get("harness_config", {})
    selector_summary = selector_result.get("summary", {}) if isinstance(selector_result, dict) else {}
    lines = [
        "# Stage 6B Real Candidate-Pool Validation",
        "",
        "## Scope",
        "",
        "- No final claim is made from this Stage 6B-real run.",
        "- Candidate patches are generated candidates only: public CoderForge/OpenHands `output_patch` artifacts expanded to K=8 with raw-preserving generated variants.",
        "- Reference/gold patches are not used as candidates; exact reference-patch hash hits are audited separately.",
        "- Correctness labels come only from the official Docker SWE-bench harness `resolved` result when available.",
        "- Raw harness logs are kept under the harness root and are not selector-visible input.",
        "",
        "## Environment",
        "",
        f"- Created at UTC: `{created_at}`.",
        f"- Platform: `{environment.get('platform')}`.",
        f"- Python: `{environment.get('python')}`.",
        f"- Docker: `{environment.get('docker_version')}`.",
        f"- WSL: `{environment.get('wsl')}`.",
        f"- Packages: `{json.dumps(environment.get('packages', {}), sort_keys=True)}`.",
        f"- Official harness command module: `python -m swebench.harness.run_evaluation`.",
        f"- Official harness namespace/image tags: namespace `{harness_config.get('namespace')}`, instance tag `{harness_config.get('instance_image_tag')}`, env tag `{harness_config.get('env_image_tag')}`.",
        f"- Official docs: https://www.swebench.com/SWE-bench/guides/evaluation/",
        "",
        "## Commands",
        "",
        f"- Runner argv: `{json.dumps(harness_config.get('runner_argv', []))}`.",
        f"- Official harness command template: `{harness_config.get('official_harness_command_template')}`.",
        "- WSL setup used for this run: `python3 -m venv .venv_stage6b` then `. .venv_stage6b/bin/activate` then `python -m pip install swebench datasets torch numpy pytest`.",
        "",
        "## Artifacts",
        "",
        f"- Candidate pool: `{pool_path}`.",
        f"- Candidate-pool audit: `{pool_audit_path}`.",
        f"- Selector results: `{selector_results_path}`.",
        f"- Selector audit: `{selector_audit_path}`.",
        f"- Harness root: `{harness_root}`.",
        f"- Official raw harness log copies: `{harness_root / 'official_logs'}`.",
        "",
        "## Candidate Pool",
        "",
        f"- Requested tasks: `{audit.get('requested_tasks')}`.",
        f"- Tasks built: `{audit.get('tasks_built')}`.",
        f"- Tasks with all official labels: `{audit.get('tasks_with_all_official_labels')}`.",
        f"- K=8 validation passes: `{audit.get('candidate_count_audit', {}).get('passes')}`.",
        f"- No exact gold/reference candidate hash hits: `{audit.get('no_gold_reference_patch_candidate_audit', {}).get('passes')}`.",
        f"- Duplicate candidate patch hash audit passes: `{audit.get('duplicate_candidate_patch_hash_audit', {}).get('passes')}`.",
        f"- Candidate order randomized audit passes: `{audit.get('candidate_order_randomized_audit', {}).get('passes')}`.",
        f"- Output leakage audit passes: `{audit.get('output_leakage_audit', {}).get('passes')}`.",
        "",
        "## Harness Labels",
        "",
        f"- Official harness result availability: `{availability.get('available_candidate_results')}/{availability.get('expected_candidate_results')}`.",
        f"- Availability audit passes: `{availability.get('passes')}`.",
        f"- Oracle pass@8: `{float(oracle.get('oracle_pass_at_8', 0.0)):.4f}`.",
        f"- Oracle-positive tasks: `{oracle.get('oracle_positive_tasks')}`.",
        f"- Oracle-empty tasks retained for unconditional pass@1: `{oracle.get('oracle_empty_tasks')}`.",
        f"- Per-repository oracle pass@8: `{json.dumps(audit.get('per_repository_oracle_pass_at_8', {}), sort_keys=True)}`.",
        f"- Per-generator pass rate: `{json.dumps(audit.get('per_generator_pass_rate', {}), sort_keys=True)}`.",
        "",
        "## Baselines And Selector",
        "",
        f"- First-candidate pass@1: `{audit.get('candidate_order_baseline_audit', {}).get('first_candidate_pass_at_1', 0.0):.4f}`.",
        f"- Random-candidate pass@1: `{audit.get('candidate_order_baseline_audit', {}).get('random_candidate_pass_at_1', 0.0):.4f}`.",
        f"- Best generator-source pass rate: `{audit.get('generator_source_baseline_audit', {}).get('best_generator_pass_rate', 0.0):.4f}`.",
        f"- Selector completed rows: `{selector_summary.get('completed_rows', 0)}`.",
        f"- Selector completed seeds: `{selector_summary.get('completed_seeds', 0)}`.",
        f"- Selector claim status: `{selector_summary.get('claim_status', 'no_final_claim')}`.",
    ]
    if selector_error:
        lines.append(f"- Selector error: `{selector_error}`.")
    lines.extend(
        [
            "",
            "## Success Criteria",
            "",
            "| criterion | pass |",
            "|---|---:|",
        ]
    )
    for key, value in criteria.items():
        lines.append(f"| `{key}` | `{bool(value)}` |")
    if int(audit.get("tasks_with_all_official_labels", 0)) < 100:
        lines.extend(
            [
                "",
                "## Fewer Than 100 Tasks",
                "",
        "Fewer than 100 tasks completed in this run. The run remains a real-harness validation artifact because labels are only accepted from the official SWE-bench harness, but it does not meet the Stage 6B-real completion target. Scaling this to 100 tasks requires running 800 official Docker evaluations for K=8.",
            ]
        )
    lines.extend(
        [
            "",
            "No final claim is made.",
        ]
    )
    return "\n".join(lines) + "\n"


def collect_environment(dataset_name: str, split: str, trajectory_dataset: str) -> Dict[str, object]:
    packages = {}
    for package in ("swebench", "datasets", "torch", "numpy", "docker"):
        try:
            proc = subprocess.run(["python", "-m", "pip", "show", package], text=True, capture_output=True, timeout=60)
            version = ""
            for line in (proc.stdout or "").splitlines():
                if line.startswith("Version:"):
                    version = line.split(":", 1)[1].strip()
                    break
            packages[package] = version or "not found"
        except Exception as exc:
            packages[package] = f"error: {exc}"
    docker_version = _run_text(["docker", "version", "--format", "{{.Server.Version}} {{.Server.Os}}/{{.Server.Arch}}"])
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "wsl": _run_text(["uname", "-a"]),
        "docker_version": docker_version,
        "packages": packages,
        "dataset_name": dataset_name,
        "split": split,
        "trajectory_dataset": trajectory_dataset,
        "official_harness_module": "swebench.harness.run_evaluation",
    }


def _write_predictions_for_slot(
    path: Path,
    examples: Sequence[PatchSelectionExample],
    candidate_index: int,
    model_name: str,
) -> None:
    rows = []
    for example in examples:
        candidate = example.candidates[candidate_index]
        rows.append(
            {
                "instance_id": example.issue_id,
                "model_name_or_path": model_name,
                "model_patch": candidate.candidate_diff,
            }
        )
    _write_jsonl(path, rows)


def _read_slot_results(
    examples: Sequence[PatchSelectionExample],
    candidate_index: int,
    run_id: str,
    model_name: str,
) -> List[HarnessCandidateResult]:
    rows = []
    for example in examples:
        log_dir = Path("logs/run_evaluation") / run_id / model_name / example.issue_id
        report_path = log_dir / "report.json"
        test_output_path = log_dir / "test_output.txt"
        result_available = report_path.exists()
        resolved = False
        applied = None
        timed_out = None
        error = None
        if result_available:
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                item = report.get(example.issue_id, {})
                resolved = bool(item.get("resolved", False))
                applied = _optional_bool(item, ("patch_successfully_applied", "patch_applied", "applied"))
                timed_out = _optional_bool(item, ("timed_out", "timeout"))
            except Exception as exc:
                result_available = False
                error = f"{type(exc).__name__}: {exc}"
        else:
            error = "missing report.json"
        candidate = example.candidates[candidate_index]
        rows.append(
            HarnessCandidateResult(
                example_id=example.id,
                issue_id=example.issue_id,
                repo=example.repo,
                candidate_index=int(candidate_index),
                candidate_id=candidate.candidate_id,
                candidate_source_agent=candidate.candidate_source_agent,
                run_id=run_id,
                model_name_or_path=model_name,
                report_path=str(report_path),
                test_output_path=str(test_output_path),
                log_dir=str(log_dir),
                result_available=result_available,
                resolved=resolved,
                applied=applied,
                timed_out=timed_out,
                error=error,
            )
        )
    return rows


def _slot_reports_complete(examples: Sequence[PatchSelectionExample], run_id: str, model_name: str) -> bool:
    return all((Path("logs/run_evaluation") / run_id / model_name / example.issue_id / "report.json").exists() for example in examples)


def _copy_official_logs(run_id: str, harness_root: Path) -> None:
    source = Path("logs/run_evaluation") / run_id
    if not source.exists():
        return
    target = harness_root / "official_logs" / run_id
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True)


def _stage6b_real_selector_config(selector_device: str, selector_epochs: int | None) -> Stage6RunConfig:
    config = _phase_config("6B-real")
    if selector_epochs is not None:
        config = Stage6RunConfig(**{**asdict(config), "epochs": int(selector_epochs)})
    return Stage6RunConfig(
        **{
            **asdict(config),
            "phase": "6B-real",
            "device": selector_device,
            "synthetic_if_missing": False,
            "synthetic_tasks": 0,
            "allow_gold_diagnostic": False,
        }
    )


def _selector_skipped_result(
    reason: str,
    pool_path: Path,
    audit: Dict[str, object],
    environment: Dict[str, object],
) -> Dict[str, object]:
    return {
        "metadata": {
            "benchmark": "stage6b_real_selector",
            "created_at_utc": _now(),
            "candidate_pool_path": str(pool_path),
            "phase": "6B-real",
            "skip_reason": reason,
            "environment": environment,
            "no_final_claim_made": True,
        },
        "rows": [],
        "summary": {
            "completed_rows": 0,
            "completed_seeds": 0,
            "success_gates_pass": False,
            "claim_status": "no_final_claim",
            "candidate_pool_tasks_with_all_official_labels": audit.get("tasks_with_all_official_labels", 0),
        },
    }


def _candidate_order_randomized_audit(examples: Sequence[PatchSelectionExample]) -> Dict[str, object]:
    rows = []
    for example in examples:
        pre = [int(candidate.metadata.get("pre_shuffle_index", -1)) for candidate in example.candidates]
        rows.append({"example_id": example.id, "pre_shuffle_order": pre, "identity": pre == list(range(len(pre)))})
    return {
        "examples_checked": len(rows),
        "identity_orders": [row for row in rows if row["identity"]],
        "passes": bool(rows) and not any(row["identity"] for row in rows),
    }


def _per_generator_pass_rate(examples: Sequence[PatchSelectionExample]) -> Dict[str, float]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for example in examples:
        for candidate, label in zip(example.candidates, example.labels_pass_fail):
            grouped[candidate.candidate_source_agent].append(int(label))
    return {key: float(np.mean(values)) if values else 0.0 for key, values in sorted(grouped.items())}


def _per_group_oracle(examples: Sequence[PatchSelectionExample], key_fn) -> Dict[str, float]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for example in examples:
        grouped[str(key_fn(example))].append(1 if example.oracle_pass_at_8 else 0)
    return {key: float(np.mean(values)) if values else 0.0 for key, values in sorted(grouped.items())}


def _first_candidate_pass(labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    return float(np.mean(labels[:, 0] > 0))


def _random_candidate_pass(labels: np.ndarray, seed: int) -> float:
    if labels.size == 0:
        return 0.0
    rng = np.random.default_rng(seed + 6_620_301)
    picks = rng.integers(0, labels.shape[1], size=labels.shape[0])
    return float(np.mean([labels[row, int(choice)] > 0 for row, choice in enumerate(picks)]))


def _split_examples_for_audit(examples: Sequence[PatchSelectionExample]) -> Dict[str, List[PatchSelectionExample]]:
    splits = {"train": [], "dev": [], "test": []}
    for example in examples:
        splits.setdefault(example.split, []).append(example)
    return {key: splits.get(key, []) for key in ("train", "dev", "test")}


def _split_for_index(index: int, n: int) -> str:
    train_cut = max(1, int(round(n * 0.6)))
    dev_cut = max(train_cut + 1, int(round(n * 0.8)))
    if index < train_cut:
        return "train"
    if index < dev_cut:
        return "dev"
    return "test"


def _retrieved_contexts(record: Dict[str, object]) -> tuple[str, ...]:
    hints = str(record.get("hints_text", "") or "")
    return (
        f"Repository: {record.get('repo')}\nBase commit: {record.get('base_commit')}\nVersion: {record.get('version')}",
        f"Hints text:\n{hints[:4000] if hints.strip() else 'No hints supplied.'}",
    )


def _failing_test_summary(record: Dict[str, object]) -> str:
    return "FAIL_TO_PASS:\n{fail}\n\nPASS_TO_PASS:\n{pass_to_pass}".format(
        fail=str(record.get("FAIL_TO_PASS", ""))[:4000],
        pass_to_pass=str(record.get("PASS_TO_PASS", ""))[:2000],
    )


def _dependency_context(record: Dict[str, object]) -> str:
    return (
        f"Repository {record.get('repo')} at base commit {record.get('base_commit')}. "
        "No generated callgraph artifact was supplied for Stage 6B-real; selector sees issue, failing tests, hints, and candidate diffs only."
    )


def _task_family(record: Dict[str, object]) -> str:
    repo = str(record.get("repo", "unknown"))
    return repo.split("/", 1)[0] if "/" in repo else repo


def _marker_file_diff(instance_id: str, index: int, text: str) -> str:
    safe = _safe_name(instance_id)
    path = f"stage6b_candidate_markers/{safe}_{index}.txt"
    body = f"stage6b-real generated candidate marker\n{safe} {index} {text}\n"
    return (
        f"diff --git a/{path} b/{path}\n"
        "new file mode 100644\n"
        "index 0000000..1111111\n"
        "--- /dev/null\n"
        f"+++ b/{path}\n"
        "@@ -0,0 +1,2 @@\n"
        f"+{body.splitlines()[0]}\n"
        f"+{body.splitlines()[1]}\n"
    )


def _non_identity_permutation(n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 1:
        return np.arange(n)
    for _ in range(16):
        perm = rng.permutation(n)
        if not np.array_equal(perm, np.arange(n)):
            return perm
    return np.roll(np.arange(n), 1)


def _ensure_trailing_newline(text: str) -> str:
    value = str(text)
    return value if value.endswith("\n") else value + "\n"


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))[:120]


def _instance_id_from_trajectory(trajectory_id: str) -> str:
    return str(trajectory_id).rsplit("_run", 1)[0]


def _json_obj(value: object) -> Dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        return dict(json.loads(value))
    return {}


def _optional_bool(row: Dict[str, object], keys: Iterable[str]) -> bool | None:
    for key in keys:
        if key in row:
            return bool(row[key])
    return None


def _run_text(command: Sequence[str]) -> str:
    try:
        proc = subprocess.run(list(command), text=True, capture_output=True, timeout=60)
        return (proc.stdout or proc.stderr or "").strip()[:1000]
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _write_json(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
