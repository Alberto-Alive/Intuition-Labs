from __future__ import annotations

import argparse
import json
import platform
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from src.datasets.swe_patch_selection_dataset import (
    STAGE6_NUM_CANDIDATES,
    PatchCandidate,
    PatchSelectionExample,
    duplicate_candidate_patch_hash_audit,
    stable_patch_hash,
    stage6_output_leakage_audit,
    validate_patch_selection_examples,
    write_patch_selection_jsonl,
)
from src.experiments.run_stage6b1_candidate_generation import (
    DEFAULT_DATASET_NAME,
    BenchmarkTask,
    load_benchmark_tasks,
)


BENCHMARK = "stage6b2_official_harness_labeling"
DEFAULT_INPUT_PATH = Path("results/stage6b1b_generated_candidates.jsonl")
DEFAULT_STAGE6B1B_AUDIT_PATH = Path("results/stage6b1b_generation_audit.json")
DEFAULT_LABELED_POOL_PATH = Path("results/stage6b2_labeled_candidate_pools.jsonl")
DEFAULT_AUDIT_PATH = Path("results/stage6b2_official_label_audit.json")
DEFAULT_HARNESS_RESULTS_PATH = Path("results/stage6b2_harness_results.jsonl")
DEFAULT_REPORT_PATH = Path("reports/STAGE6B2_OFFICIAL_HARNESS_LABELING.md")
DEFAULT_HARNESS_ROOT = Path("results/stage6b2_official_harness")
DEFAULT_SEED = 606_292


@dataclass(frozen=True)
class Stage6B2InputCandidate:
    row_index: int
    instance_id: str
    slot_index: int
    candidate_id: str
    generator_name: str
    model_name_or_path: str
    model_patch: str
    seed: int | None
    temperature: float | None
    source_visible_to_selector: bool

    @property
    def patch_hash(self) -> str:
        return stable_patch_hash(self.model_patch)


@dataclass(frozen=True)
class HarnessCandidateResult:
    example_id: str
    issue_id: str
    repo: str
    candidate_index: int
    candidate_id: str
    candidate_source_agent: str
    candidate_patch_hash: str
    run_id: str
    model_name_or_path: str
    report_path: str
    test_output_path: str
    log_dir: str
    result_available: bool
    resolved: bool | None
    applied: bool | None
    timed_out: bool | None
    error: str | None
    official_label_source_field: str
    cache_hit: bool


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6B.2 official SWE-bench Docker harness labeling for a K=8 generated candidate pool.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default="test")
    parser.add_argument("--benchmark-jsonl", default=None)
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--stage6b1b-audit", default=str(DEFAULT_STAGE6B1B_AUDIT_PATH))
    parser.add_argument("--labeled-output", default=str(DEFAULT_LABELED_POOL_PATH))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT_PATH))
    parser.add_argument("--harness-results", default=str(DEFAULT_HARNESS_RESULTS_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--harness-root", default=str(DEFAULT_HARNESS_ROOT))
    parser.add_argument("--k", type=int, default=STAGE6_NUM_CANDIDATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--cache-level", default="env", choices=("none", "base", "env", "instance"))
    parser.add_argument("--namespace", default="swebench")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--build-only", action="store_true", help="Build prediction files and audits without running the official harness.")
    args = parser.parse_args()

    result = run_stage6b2_official_harness_labeling(
        dataset_name=str(args.dataset_name),
        split=str(args.split),
        benchmark_jsonl=None if args.benchmark_jsonl is None else Path(args.benchmark_jsonl),
        input_path=Path(args.input),
        stage6b1b_audit_path=Path(args.stage6b1b_audit),
        labeled_output_path=Path(args.labeled_output),
        audit_path=Path(args.audit),
        harness_results_path=Path(args.harness_results),
        report_path=Path(args.report),
        harness_root=Path(args.harness_root),
        k=int(args.k),
        seed=int(args.seed),
        max_workers=int(args.max_workers),
        timeout=int(args.timeout),
        cache_level=str(args.cache_level),
        namespace=str(args.namespace),
        force_rerun=bool(args.force_rerun),
        build_only=bool(args.build_only),
    )
    audit = result["audit"]
    gates = audit.get("quality_gates", {})
    availability = audit.get("official_label_availability_audit", {})
    oracle = audit.get("oracle_pass_at_8_audit", {})
    print(
        "stage6b2: wrote {pool}, {harness_results}, {audit_path}, {report}; labels={available}/{expected}; "
        "oracle_pass_at_8={oracle:.4f}; gates_pass={gates_pass}".format(
            pool=args.labeled_output,
            harness_results=args.harness_results,
            audit_path=args.audit,
            report=args.report,
            available=availability.get("available_candidate_results", 0),
            expected=availability.get("expected_candidate_results", 0),
            oracle=float(oracle.get("oracle_pass_at_8", 0.0)),
            gates_pass=gates.get("stage6b2_official_label_quality_gates_pass", False),
        )
    )


def run_stage6b2_official_harness_labeling(
    dataset_name: str,
    split: str,
    benchmark_jsonl: Path | None,
    input_path: Path,
    stage6b1b_audit_path: Path,
    labeled_output_path: Path,
    audit_path: Path,
    harness_results_path: Path,
    report_path: Path,
    harness_root: Path,
    k: int,
    seed: int,
    max_workers: int,
    timeout: int,
    cache_level: str,
    namespace: str | None,
    force_rerun: bool,
    build_only: bool,
) -> Dict[str, object]:
    created_at = _now()
    stage6b1b_audit = _read_json_or_empty(stage6b1b_audit_path)
    candidates_by_task, input_audit = load_stage6b1b_candidate_source(input_path=input_path, k=k)
    task_ids = list(candidates_by_task)
    task_records, benchmark_audit = load_benchmark_tasks(
        dataset_name=dataset_name,
        split=split,
        benchmark_jsonl=benchmark_jsonl,
        requested_task_ids=task_ids,
        task_ids_file=None,
        n_tasks=len(task_ids),
        seed=seed,
    )
    examples = build_unlabeled_stage6b2_examples(
        candidates_by_task=candidates_by_task,
        task_records=task_records,
        dataset_name=dataset_name,
        split=split,
        input_path=input_path,
        stage6b1b_audit=stage6b1b_audit,
        seed=seed,
    )
    harness_results: List[HarnessCandidateResult] = []
    if not build_only:
        harness_results = evaluate_with_official_harness(
            examples=examples,
            dataset_name=dataset_name,
            split=split,
            seed=seed,
            harness_root=harness_root,
            k=k,
            max_workers=max_workers,
            timeout=timeout,
            cache_level=cache_level,
            namespace=namespace,
            force_rerun=force_rerun,
        )
    else:
        _write_all_slot_predictions(harness_root=harness_root, examples=examples, seed=seed, k=k)
    labeled_examples = attach_official_labels(examples=examples, harness_results=harness_results, k=k)
    write_patch_selection_jsonl(labeled_output_path, labeled_examples)
    _write_jsonl(harness_results_path, [asdict(row) for row in harness_results])
    environment = collect_environment(dataset_name=dataset_name, split=split)
    audit = build_stage6b2_audit(
        created_at=created_at,
        dataset_name=dataset_name,
        split=split,
        input_path=input_path,
        stage6b1b_audit_path=stage6b1b_audit_path,
        stage6b1b_audit=stage6b1b_audit,
        labeled_output_path=labeled_output_path,
        harness_results_path=harness_results_path,
        harness_root=harness_root,
        examples=labeled_examples,
        candidates_by_task=candidates_by_task,
        task_records=task_records,
        input_audit=input_audit,
        benchmark_audit=benchmark_audit,
        harness_results=harness_results,
        environment=environment,
        k=k,
        seed=seed,
        max_workers=max_workers,
        timeout=timeout,
        cache_level=cache_level,
        namespace=namespace,
        force_rerun=force_rerun,
        build_only=build_only,
    )
    _write_json(audit_path, audit)
    report = render_stage6b2_report(
        audit=audit,
        labeled_output_path=labeled_output_path,
        audit_path=audit_path,
        harness_results_path=harness_results_path,
        harness_root=harness_root,
        report_path=report_path,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return {"audit": audit, "examples": labeled_examples, "harness_results": harness_results}


def load_stage6b1b_candidate_source(input_path: Path, k: int) -> tuple[Dict[str, List[Stage6B2InputCandidate]], Dict[str, object]]:
    rows = []
    errors = []
    if not input_path.exists():
        raise FileNotFoundError(f"Stage 6B.1b candidate source not found: {input_path}")
    for row_index, line in enumerate(input_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception as exc:
            errors.append({"row_index": row_index, "error": f"{type(exc).__name__}: {exc}"})
            continue
        rows.append((row_index, row))
    grouped_raw: Dict[str, List[tuple[int, Dict[str, object]]]] = defaultdict(list)
    for row_index, row in rows:
        grouped_raw[str(row.get("instance_id", ""))].append((row_index, row))
    candidates_by_task: Dict[str, List[Stage6B2InputCandidate]] = {}
    per_task_counts = {}
    duplicate_patch_hash_rows = []
    missing_patch_rows = []
    source_visible_rows = []
    for instance_id, group in grouped_raw.items():
        seen_hashes = set()
        candidates = []
        per_task_counts[instance_id] = len(group)
        for slot_index, (row_index, row) in enumerate(group):
            patch = str(row.get("model_patch", "") or "")
            candidate = Stage6B2InputCandidate(
                row_index=int(row_index),
                instance_id=instance_id,
                slot_index=int(slot_index),
                candidate_id=str(row.get("candidate_id", f"{instance_id}-stage6b1b-candidate-{slot_index:02d}")),
                generator_name=str(row.get("generator_name") or row.get("model_name_or_path") or "unknown_generator"),
                model_name_or_path=str(row.get("model_name_or_path") or row.get("generator_name") or "unknown_model"),
                model_patch=patch,
                seed=_optional_int(row.get("seed")),
                temperature=_optional_float(row.get("temperature")),
                source_visible_to_selector=bool(row.get("source_visible_to_selector", False)),
            )
            if not patch.strip():
                missing_patch_rows.append({"instance_id": instance_id, "row_index": row_index, "candidate_id": candidate.candidate_id})
            if candidate.patch_hash in seen_hashes:
                duplicate_patch_hash_rows.append({"instance_id": instance_id, "row_index": row_index, "candidate_id": candidate.candidate_id, "patch_hash": candidate.patch_hash})
            seen_hashes.add(candidate.patch_hash)
            if candidate.source_visible_to_selector:
                source_visible_rows.append({"instance_id": instance_id, "row_index": row_index, "candidate_id": candidate.candidate_id})
            candidates.append(candidate)
        candidates_by_task[instance_id] = candidates
    audit = {
        "input_path": str(input_path),
        "raw_rows_loaded": len(rows),
        "unique_tasks_loaded": len(candidates_by_task),
        "expected_k": int(k),
        "per_task_candidate_counts": per_task_counts,
        "tasks_with_exactly_k": sum(1 for value in per_task_counts.values() if value == int(k)),
        "bad_task_counts": {key: value for key, value in per_task_counts.items() if value != int(k)},
        "missing_patch_rows": missing_patch_rows,
        "source_visible_to_selector_rows": source_visible_rows,
        "duplicate_patch_hash_rows": duplicate_patch_hash_rows,
        "json_errors": errors,
        "passes": bool(rows)
        and len(candidates_by_task) == 25
        and all(value == int(k) for value in per_task_counts.values())
        and not missing_patch_rows
        and not source_visible_rows
        and not duplicate_patch_hash_rows
        and not errors,
    }
    return dict(candidates_by_task), audit


def build_unlabeled_stage6b2_examples(
    candidates_by_task: Dict[str, List[Stage6B2InputCandidate]],
    task_records: Dict[str, BenchmarkTask],
    dataset_name: str,
    split: str,
    input_path: Path,
    stage6b1b_audit: Dict[str, object],
    seed: int,
) -> List[PatchSelectionExample]:
    examples = []
    randomizable = _stage6b1b_candidate_order_randomizable(stage6b1b_audit)
    for task_index, (instance_id, rows) in enumerate(candidates_by_task.items()):
        task = task_records.get(instance_id)
        record = task.record if task is not None else {"instance_id": instance_id}
        candidates = tuple(
            PatchCandidate(
                candidate_id=row.candidate_id,
                candidate_source_agent=row.generator_name,
                candidate_diff=row.model_patch,
                candidate_visible_test_result=None,
                metadata={
                    "stage6b1b_input_row_index": int(row.row_index),
                    "stage6b1b_input_slot_index": int(row.slot_index),
                    "stage6b1b_input_patch_hash": row.patch_hash,
                    "generator_name": row.generator_name,
                    "model_name_or_path": row.model_name_or_path,
                    "seed": row.seed,
                    "temperature": row.temperature,
                    "source_visible_to_selector": False,
                },
            )
            for row in rows
        )
        examples.append(
            PatchSelectionExample(
                id=f"stage6b2-{instance_id}",
                dataset_name=f"{dataset_name}:{split}",
                repo=str(record.get("repo", "")),
                issue_id=instance_id,
                issue_text=str(record.get("problem_statement", "")),
                failing_test_summary=_failing_test_summary(record),
                retrieved_contexts=_retrieved_contexts(record),
                candidates=candidates,
                labels_pass_fail=(),
                split=_split_for_index(task_index, len(candidates_by_task)),
                metadata={
                    "task_family": _task_family(record),
                    "candidate_pool_version": "stage6b2_official_harness_labeled_stage6b1b_v1",
                    "candidate_pool_source": str(input_path),
                    "benchmark_source": dataset_name,
                    "benchmark_split": split,
                    "base_commit": str(record.get("base_commit", "")),
                    "environment_setup_commit": str(record.get("environment_setup_commit", "")),
                    "version": str(record.get("version", "")),
                    "created_at": str(record.get("created_at", "")),
                    "visible_test_results_allowed": False,
                    "gold_patch_included": False,
                    "raw_preserving_expansion_used": False,
                    "missing_candidate_fabrication_used": False,
                    "candidate_order_source": "stage6b1b_input_order",
                    "candidate_order_seed": int(seed),
                    "candidate_order_randomized": bool(randomizable),
                    "candidate_order_frozen_before_official_labeling": True,
                    "full_swebench_harness_ran": False,
                    "official_labels_complete": False,
                    "label_source": "pending_official_swebench_harness",
                    "publishable_proof_dataset": False,
                    "dependency_callgraph_related_file_evidence": _dependency_context(record),
                },
            )
        )
    return examples


def evaluate_with_official_harness(
    examples: Sequence[PatchSelectionExample],
    dataset_name: str,
    split: str,
    seed: int,
    harness_root: Path,
    k: int,
    max_workers: int,
    timeout: int,
    cache_level: str,
    namespace: str | None,
    force_rerun: bool,
) -> List[HarnessCandidateResult]:
    harness_root.mkdir(parents=True, exist_ok=True)
    (harness_root / "predictions").mkdir(parents=True, exist_ok=True)
    (harness_root / "command_logs").mkdir(parents=True, exist_ok=True)
    all_results: List[HarnessCandidateResult] = []
    for candidate_index in range(int(k)):
        model_name = _slot_model_name(candidate_index)
        run_id = _slot_run_id(seed, candidate_index)
        predictions_path = harness_root / "predictions" / f"{run_id}.jsonl"
        _write_predictions_for_slot(predictions_path, examples, candidate_index, model_name)
        cache_hit = _slot_reports_complete(examples, run_id, model_name, harness_root)
        if force_rerun or not cache_hit:
            run_examples = list(examples)
            command_predictions_path = predictions_path
            if not force_rerun:
                missing_examples = _missing_report_examples(examples, run_id, model_name, harness_root)
                if missing_examples and len(missing_examples) < len(examples):
                    run_examples = missing_examples
                    command_predictions_path = harness_root / "predictions" / f"{run_id}.resume_missing.jsonl"
                    _write_predictions_for_slot(command_predictions_path, run_examples, candidate_index, model_name)
            command = [
                sys.executable,
                "-m",
                "swebench.harness.run_evaluation",
                "--dataset_name",
                dataset_name,
                "--split",
                split,
                "--predictions_path",
                str(command_predictions_path),
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
            proc = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=max(int(timeout) * max(2, len(run_examples)), int(timeout) + 600),
            )
            command_log = harness_root / "command_logs" / run_id
            command_log.mkdir(parents=True, exist_ok=True)
            (command_log / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
            (command_log / "stdout.log").write_text(proc.stdout or "", encoding="utf-8")
            (command_log / "stderr.log").write_text(proc.stderr or "", encoding="utf-8")
            (command_log / "returncode.txt").write_text(str(proc.returncode), encoding="utf-8")
            _copy_harness_summary_report(model_name=model_name, run_id=run_id, harness_root=harness_root)
        _copy_official_logs(run_id=run_id, harness_root=harness_root)
        all_results.extend(_read_slot_results(examples, candidate_index, run_id, model_name, harness_root, cache_hit=cache_hit and not force_rerun))
    _write_jsonl(harness_root / "candidate_harness_results.jsonl", [asdict(row) for row in all_results])
    return all_results


def attach_official_labels(
    examples: Sequence[PatchSelectionExample],
    harness_results: Sequence[HarnessCandidateResult],
    k: int,
) -> List[PatchSelectionExample]:
    by_candidate = {(row.example_id, int(row.candidate_index)): row for row in harness_results}
    labeled = []
    for example in examples:
        labels = []
        all_available = True
        for index in range(len(example.candidates)):
            result = by_candidate.get((example.id, index))
            if result is None or not result.result_available or result.resolved is None:
                all_available = False
            else:
                labels.append(1 if bool(result.resolved) else 0)
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
                labels_pass_fail=tuple(labels) if all_available and len(labels) == int(k) else (),
                split=example.split,
                metadata={
                    **example.metadata,
                    "full_swebench_harness_ran": bool(all_available and len(labels) == int(k)),
                    "official_labels_complete": bool(all_available and len(labels) == int(k)),
                    "label_source": "official_swebench_harness" if all_available and len(labels) == int(k) else "incomplete_official_swebench_harness",
                    "publishable_proof_dataset": bool(all_available and len(labels) == int(k)),
                },
            )
        )
    return labeled


def build_stage6b2_audit(
    created_at: str,
    dataset_name: str,
    split: str,
    input_path: Path,
    stage6b1b_audit_path: Path,
    stage6b1b_audit: Dict[str, object],
    labeled_output_path: Path,
    harness_results_path: Path,
    harness_root: Path,
    examples: Sequence[PatchSelectionExample],
    candidates_by_task: Dict[str, List[Stage6B2InputCandidate]],
    task_records: Dict[str, BenchmarkTask],
    input_audit: Dict[str, object],
    benchmark_audit: Dict[str, object],
    harness_results: Sequence[HarnessCandidateResult],
    environment: Dict[str, object],
    k: int,
    seed: int,
    max_workers: int,
    timeout: int,
    cache_level: str,
    namespace: str | None,
    force_rerun: bool,
    build_only: bool,
) -> Dict[str, object]:
    validation = validate_patch_selection_examples([ex for ex in examples if len(ex.labels_pass_fail) == int(k)], k=int(k), allow_gold_diagnostic=False)
    candidate_count_audit = _candidate_count_audit(examples, k)
    label_availability = _official_label_availability(examples, harness_results, k)
    oracle_audit = _oracle_pass_at_k(examples, k)
    first_baseline = _first_candidate_baseline(examples, k)
    random_baseline = _random_candidate_baseline(examples, seed, k)
    best_generator = _best_generator_on_dev_baseline(examples, k)
    generator_pass = _per_generator_pass_rate(examples, k)
    per_repo_oracle = _per_group_oracle(examples, lambda ex: ex.repo, k)
    reference_audit = _exact_reference_patch_hash_audit(examples, task_records)
    duplicate_audit = _duplicate_patch_hash_audit(examples)
    output_audit = stage6_output_leakage_audit(BENCHMARK, seed, examples)
    source_generator_audit = _source_generator_leakage_audit(examples, best_generator, oracle_audit)
    order_audit = _candidate_order_randomized_audit(examples, stage6b1b_audit)
    patch_integrity = _patch_integrity_audit(examples, candidates_by_task, harness_root, seed, k)
    raw_fabrication_audit = _raw_preserving_and_fabrication_audit(examples, stage6b1b_audit)
    harness_invocation = _harness_invocation_audit(
        harness_root=harness_root,
        seed=seed,
        k=k,
        max_workers=max_workers,
        timeout=timeout,
        cache_level=cache_level,
        namespace=namespace,
        force_rerun=force_rerun,
        build_only=build_only,
    )
    first_equals_oracle = abs(float(first_baseline["first_candidate_pass_at_1"]) - float(oracle_audit["oracle_pass_at_8"])) <= 1e-12
    best_equals_oracle = abs(float(best_generator["best_generator_on_dev_pass_at_1"]) - float(oracle_audit["oracle_pass_at_8"])) <= 1e-12
    audit = {
        "benchmark": BENCHMARK,
        "created_at_utc": created_at,
        "dataset_name": dataset_name,
        "split": split,
        "input_path": str(input_path),
        "stage6b1b_audit_path": str(stage6b1b_audit_path),
        "labeled_output_path": str(labeled_output_path),
        "harness_results_path": str(harness_results_path),
        "harness_root": str(harness_root),
        "requested_tasks": 25,
        "tasks_loaded": len(examples),
        "candidate_count_k": int(k),
        "selector_training_executed": False,
        "official_harness_evaluation_executed": bool(not build_only),
        "official_harness_command_module": "python -m swebench.harness.run_evaluation",
        "final_selector_claim_made": False,
        "environment": environment,
        "benchmark_dataset_audit": benchmark_audit,
        "stage6b1b_input_audit": input_audit,
        "harness_invocation_audit": harness_invocation,
        "candidate_pool_validation": validation,
        "candidate_count_audit": candidate_count_audit,
        "official_label_availability_audit": label_availability,
        "oracle_pass_at_8_audit": oracle_audit,
        "per_repository_oracle_pass_at_8": per_repo_oracle,
        "per_generator_pass_rate": generator_pass,
        "first_candidate_baseline": first_baseline,
        "random_candidate_baseline": random_baseline,
        "best_generator_on_dev_baseline": best_generator,
        "duplicate_candidate_patch_hash_audit": duplicate_audit,
        "exact_reference_patch_hash_audit": reference_audit,
        "candidate_order_randomized_audit": order_audit,
        "patch_integrity_audit": patch_integrity,
        "output_leakage_audit": output_audit,
        "source_generator_leakage_audit": source_generator_audit,
        "raw_preserving_and_fabrication_audit": raw_fabrication_audit,
    }
    quality_gates = {
        "all_25_tasks_preserve_k8_candidates": bool(len(examples) == 25 and candidate_count_audit["passes"]),
        "official_harness_labels_available_for_all_200_candidates": bool(label_availability["passes"] and label_availability["expected_candidate_results"] == 200),
        "oracle_pass_at_8_between_0_20_and_0_80": bool(0.20 <= float(oracle_audit["oracle_pass_at_8"]) <= 0.80),
        "at_least_5_oracle_positive_tasks": bool(int(oracle_audit["oracle_positive_task_count"]) >= 5),
        "first_candidate_baseline_does_not_equal_oracle": bool(not first_equals_oracle),
        "best_generator_source_baseline_does_not_equal_oracle": bool(not best_equals_oracle),
        "no_gold_reference_patch_leakage": bool(reference_audit["passes"] and output_audit.get("passes", False)),
        "all_audits_complete": bool(_all_required_audits_complete(audit)),
        "no_raw_preserving_expansion": bool(raw_fabrication_audit["raw_preserving_expansion_used"] is False),
        "no_fabricated_candidates": bool(raw_fabrication_audit["missing_candidate_fabrication_used"] is False),
        "candidate_patches_not_altered": bool(patch_integrity["passes"]),
        "selector_training_not_executed": True,
        "official_harness_executed_or_reused": bool(not build_only and (len(harness_results) > 0 or _any_slot_command_log(harness_root, seed, k))),
        "no_final_selector_claim_made": True,
    }
    quality_gates["stage6b2_official_label_quality_gates_pass"] = all(bool(value) for value in quality_gates.values())
    audit["quality_gates"] = quality_gates
    return audit


def render_stage6b2_report(
    audit: Dict[str, object],
    labeled_output_path: Path,
    audit_path: Path,
    harness_results_path: Path,
    harness_root: Path,
    report_path: Path,
) -> str:
    availability = audit.get("official_label_availability_audit", {})
    oracle = audit.get("oracle_pass_at_8_audit", {})
    gates = audit.get("quality_gates", {})
    lines = [
        "# Stage 6B.2 Official Harness Labeling",
        "",
        "## Scope",
        "",
        "- Candidate source: `results/stage6b1b_generated_candidates.jsonl`.",
        "- Candidate patches were copied unchanged into slot prediction files.",
        "- Correctness labels are accepted only from the official SWE-bench harness `resolved` field.",
        "- Raw harness logs are stored under the harness root and are not included in selector-visible candidate fields.",
        "- No selector training was executed and no final selector claim is made.",
        "",
        "## Artifacts",
        "",
        f"- Labeled candidate pool: `{labeled_output_path}`.",
        f"- Harness results: `{harness_results_path}`.",
        f"- Official label audit: `{audit_path}`.",
        f"- Harness root: `{harness_root}`.",
        f"- Report: `{report_path}`.",
        "",
        "## Harness",
        "",
        f"- Official command module: `{audit.get('official_harness_command_module')}`.",
        f"- Harness executed: `{audit.get('official_harness_evaluation_executed')}`.",
        f"- Labels available: `{availability.get('available_candidate_results')}/{availability.get('expected_candidate_results')}`.",
        f"- Tasks with complete labels: `{availability.get('tasks_with_all_labels')}/{availability.get('tasks_expected')}`.",
        "",
        "## Label Summary",
        "",
        f"- Tasks loaded: `{audit.get('tasks_loaded')}`.",
        f"- K=8 preserved: `{audit.get('candidate_count_audit', {}).get('passes')}`.",
        f"- Oracle pass@8: `{float(oracle.get('oracle_pass_at_8', 0.0)):.4f}`.",
        f"- Oracle-positive tasks: `{oracle.get('oracle_positive_task_count')}`.",
        f"- Oracle-empty tasks: `{oracle.get('oracle_empty_task_count')}`.",
        f"- Per-repository oracle pass@8: `{json.dumps(audit.get('per_repository_oracle_pass_at_8', {}), sort_keys=True)}`.",
        f"- Per-generator pass rate: `{json.dumps(audit.get('per_generator_pass_rate', {}), sort_keys=True)}`.",
        "",
        "## Baselines",
        "",
        f"- First-candidate pass@1: `{audit.get('first_candidate_baseline', {}).get('first_candidate_pass_at_1', 0.0):.4f}`.",
        f"- Random-candidate pass@1: `{audit.get('random_candidate_baseline', {}).get('random_candidate_pass_at_1', 0.0):.4f}`.",
        f"- Best-generator-on-dev pass@1: `{audit.get('best_generator_on_dev_baseline', {}).get('best_generator_on_dev_pass_at_1', 0.0):.4f}`.",
        f"- Best generator on dev: `{audit.get('best_generator_on_dev_baseline', {}).get('best_generator')}`.",
        "",
        "## Audits",
        "",
        f"- Duplicate candidate patch hash audit passes: `{audit.get('duplicate_candidate_patch_hash_audit', {}).get('passes')}`.",
        f"- Exact reference patch hash hits: `{audit.get('exact_reference_patch_hash_audit', {}).get('hits_total')}`.",
        f"- Candidate order randomized audit passes: `{audit.get('candidate_order_randomized_audit', {}).get('passes')}`.",
        f"- Output leakage audit passes: `{audit.get('output_leakage_audit', {}).get('passes')}`.",
        f"- Source/generator leakage audit passes: `{audit.get('source_generator_leakage_audit', {}).get('passes')}`.",
        f"- Raw-preserving expansion used: `{audit.get('raw_preserving_and_fabrication_audit', {}).get('raw_preserving_expansion_used')}`.",
        f"- Fabricated candidates used: `{audit.get('raw_preserving_and_fabrication_audit', {}).get('missing_candidate_fabrication_used')}`.",
        "",
        "## Quality Gates",
        "",
        "| gate | pass |",
        "|---|---:|",
    ]
    for key, value in gates.items():
        lines.append(f"| `{key}` | `{bool(value)}` |")
    lines.extend(["", "No final selector claim is made.", ""])
    return "\n".join(lines)


def _write_all_slot_predictions(harness_root: Path, examples: Sequence[PatchSelectionExample], seed: int, k: int) -> None:
    for candidate_index in range(int(k)):
        model_name = _slot_model_name(candidate_index)
        run_id = _slot_run_id(seed, candidate_index)
        predictions_path = harness_root / "predictions" / f"{run_id}.jsonl"
        _write_predictions_for_slot(predictions_path, examples, candidate_index, model_name)


def _write_predictions_for_slot(path: Path, examples: Sequence[PatchSelectionExample], candidate_index: int, model_name: str) -> None:
    rows = []
    for example in examples:
        candidate = example.candidates[candidate_index]
        rows.append({"instance_id": example.issue_id, "model_name_or_path": model_name, "model_patch": candidate.candidate_diff})
    _write_jsonl(path, rows)


def _read_slot_results(
    examples: Sequence[PatchSelectionExample],
    candidate_index: int,
    run_id: str,
    model_name: str,
    harness_root: Path,
    cache_hit: bool,
) -> List[HarnessCandidateResult]:
    rows = []
    for example in examples:
        report_path, log_dir = _first_existing_report_path(run_id, model_name, example.issue_id, harness_root)
        test_output_path = log_dir / "test_output.txt"
        result_available = report_path.exists()
        resolved: bool | None = None
        applied = None
        timed_out = None
        error = None
        if result_available:
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                item = report.get(example.issue_id)
                if not isinstance(item, dict):
                    result_available = False
                    error = "report.json missing instance_id row"
                else:
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
                candidate_patch_hash=candidate.patch_hash,
                run_id=run_id,
                model_name_or_path=model_name,
                report_path=str(report_path),
                test_output_path=str(test_output_path),
                log_dir=str(log_dir),
                result_available=result_available,
                resolved=resolved if result_available else None,
                applied=applied,
                timed_out=timed_out,
                error=error,
                official_label_source_field=f"{report_path.name}:{example.issue_id}.resolved",
                cache_hit=bool(cache_hit),
            )
        )
    return rows


def _slot_reports_complete(examples: Sequence[PatchSelectionExample], run_id: str, model_name: str, harness_root: Path) -> bool:
    return all(_first_existing_report_path(run_id, model_name, example.issue_id, harness_root)[0].exists() for example in examples)


def _missing_report_examples(
    examples: Sequence[PatchSelectionExample],
    run_id: str,
    model_name: str,
    harness_root: Path,
) -> List[PatchSelectionExample]:
    return [
        example
        for example in examples
        if not _first_existing_report_path(run_id, model_name, example.issue_id, harness_root)[0].exists()
    ]


def _first_existing_report_path(run_id: str, model_name: str, instance_id: str, harness_root: Path) -> tuple[Path, Path]:
    log_dirs = [
        Path("logs/run_evaluation") / run_id / model_name / instance_id,
        harness_root / "official_logs" / run_id / model_name / instance_id,
    ]
    for log_dir in log_dirs:
        report_path = log_dir / "report.json"
        if report_path.exists():
            return report_path, log_dir
    return log_dirs[0] / "report.json", log_dirs[0]


def _copy_official_logs(run_id: str, harness_root: Path) -> None:
    source = Path("logs/run_evaluation") / run_id
    if not source.exists():
        return
    target = harness_root / "official_logs" / run_id
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True)


def _copy_harness_summary_report(model_name: str, run_id: str, harness_root: Path) -> None:
    source = Path(f"{model_name}.{run_id}.json")
    if not source.exists():
        return
    target = harness_root / "reports" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _official_label_availability(
    examples: Sequence[PatchSelectionExample],
    harness_results: Sequence[HarnessCandidateResult],
    k: int,
) -> Dict[str, object]:
    by_candidate = {(row.example_id, int(row.candidate_index)): row for row in harness_results}
    rows = []
    for example in examples:
        available = 0
        missing = []
        for index in range(int(k)):
            row = by_candidate.get((example.id, index))
            if row is not None and row.result_available and row.resolved is not None:
                available += 1
            else:
                missing.append(index)
        rows.append({"example_id": example.id, "issue_id": example.issue_id, "available": available, "expected": int(k), "missing_candidate_indices": missing})
    expected = len(examples) * int(k)
    available = sum(int(row["available"]) for row in rows)
    return {
        "rows": rows,
        "available_candidate_results": int(available),
        "expected_candidate_results": int(expected),
        "tasks_with_all_labels": sum(1 for row in rows if row["available"] == row["expected"]),
        "tasks_expected": len(examples),
        "passes": bool(rows) and available == expected,
    }


def _candidate_count_audit(examples: Sequence[PatchSelectionExample], k: int) -> Dict[str, object]:
    rows = [{"example_id": example.id, "issue_id": example.issue_id, "candidate_count": len(example.candidates)} for example in examples if len(example.candidates) != int(k)]
    unique_rows = []
    for example in examples:
        hashes = [candidate.patch_hash for candidate in example.candidates]
        if len(set(hashes)) != len(hashes):
            unique_rows.append({"example_id": example.id, "issue_id": example.issue_id, "candidate_count": len(hashes), "unique_patch_hashes": len(set(hashes))})
    return {
        "expected_k": int(k),
        "examples_checked": len(examples),
        "bad_candidate_counts": rows,
        "non_unique_patch_hash_tasks": unique_rows,
        "passes": bool(examples) and not rows and not unique_rows,
    }


def _oracle_pass_at_k(examples: Sequence[PatchSelectionExample], k: int) -> Dict[str, object]:
    complete = [example for example in examples if len(example.labels_pass_fail) == int(k)]
    oracle = [any(int(value) == 1 for value in example.labels_pass_fail) for example in complete]
    positives = int(sum(1 for value in oracle if value))
    empty = int(sum(1 for value in oracle if not value))
    histogram = Counter(sum(int(value) for value in example.labels_pass_fail) for example in complete)
    return {
        "complete_labeled_tasks": len(complete),
        "oracle_pass_at_8": _mean_bool(oracle),
        "oracle_positive_task_count": positives,
        "oracle_empty_task_count": empty,
        "positive_candidates_per_task_histogram": {str(key): int(value) for key, value in sorted(histogram.items())},
    }


def _per_group_oracle(examples: Sequence[PatchSelectionExample], key_fn, k: int) -> Dict[str, float]:
    grouped: Dict[str, List[bool]] = defaultdict(list)
    for example in examples:
        if len(example.labels_pass_fail) != int(k):
            continue
        grouped[str(key_fn(example))].append(any(int(value) == 1 for value in example.labels_pass_fail))
    return {key: _mean_bool(values) for key, values in sorted(grouped.items())}


def _per_generator_pass_rate(examples: Sequence[PatchSelectionExample], k: int) -> Dict[str, float]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for example in examples:
        if len(example.labels_pass_fail) != int(k):
            continue
        for candidate, label in zip(example.candidates, example.labels_pass_fail):
            grouped[candidate.candidate_source_agent].append(int(label))
    return {key: _mean_numeric(values) for key, values in sorted(grouped.items())}


def _first_candidate_baseline(examples: Sequence[PatchSelectionExample], k: int) -> Dict[str, object]:
    values = [bool(example.labels_pass_fail[0]) for example in examples if len(example.labels_pass_fail) == int(k)]
    return {"first_candidate_pass_at_1": _mean_bool(values), "complete_labeled_tasks": len(values)}


def _random_candidate_baseline(examples: Sequence[PatchSelectionExample], seed: int, k: int) -> Dict[str, object]:
    rng = random.Random(seed + 6_292_101)
    values = []
    picks = []
    for example in examples:
        if len(example.labels_pass_fail) != int(k):
            continue
        choice = rng.randrange(int(k))
        values.append(bool(example.labels_pass_fail[choice]))
        picks.append({"example_id": example.id, "issue_id": example.issue_id, "candidate_index": int(choice)})
    return {"random_candidate_pass_at_1": _mean_bool(values), "complete_labeled_tasks": len(values), "seed": int(seed), "picks": picks}


def _best_generator_on_dev_baseline(examples: Sequence[PatchSelectionExample], k: int) -> Dict[str, object]:
    dev_grouped: Dict[str, List[int]] = defaultdict(list)
    for example in examples:
        if example.split != "dev" or len(example.labels_pass_fail) != int(k):
            continue
        for candidate, label in zip(example.candidates, example.labels_pass_fail):
            dev_grouped[candidate.candidate_source_agent].append(int(label))
    dev_rates = {key: _mean_numeric(values) for key, values in sorted(dev_grouped.items())}
    best_generator = None
    if dev_rates:
        best_generator = sorted(dev_rates, key=lambda key: (-dev_rates[key], key))[0]
    values = []
    rows = []
    for example in examples:
        if len(example.labels_pass_fail) != int(k):
            continue
        chosen_index = 0
        if best_generator is not None:
            for index, candidate in enumerate(example.candidates):
                if candidate.candidate_source_agent == best_generator:
                    chosen_index = index
                    break
        label = int(example.labels_pass_fail[chosen_index])
        values.append(bool(label))
        rows.append({"example_id": example.id, "issue_id": example.issue_id, "chosen_index": int(chosen_index), "label": label})
    return {
        "best_generator": best_generator,
        "dev_candidate_pass_rates": dev_rates,
        "best_generator_on_dev_pass_at_1": _mean_bool(values),
        "complete_labeled_tasks": len(values),
        "selection_rows": rows,
    }


def _exact_reference_patch_hash_audit(
    examples: Sequence[PatchSelectionExample],
    task_records: Dict[str, BenchmarkTask],
) -> Dict[str, object]:
    hits = []
    missing_reference_hashes = []
    for example in examples:
        task = task_records.get(example.issue_id)
        ref_hash = task.reference_hash if task is not None else ""
        if not ref_hash:
            missing_reference_hashes.append(example.issue_id)
            continue
        for index, candidate in enumerate(example.candidates):
            if candidate.patch_hash == ref_hash:
                hits.append({"example_id": example.id, "issue_id": example.issue_id, "candidate_index": index, "candidate_id": candidate.candidate_id, "patch_hash": candidate.patch_hash})
    return {
        "tasks_checked": len(examples),
        "missing_reference_hashes": missing_reference_hashes,
        "hits": hits,
        "hits_total": len(hits),
        "passes": not hits and not missing_reference_hashes,
    }


def _duplicate_patch_hash_audit(examples: Sequence[PatchSelectionExample]) -> Dict[str, object]:
    per_task = duplicate_candidate_patch_hash_audit(examples)
    global_counts = Counter(candidate.patch_hash for example in examples for candidate in example.candidates)
    global_duplicates = [
        {"patch_hash": patch_hash, "count": int(count)}
        for patch_hash, count in sorted(global_counts.items())
        if count > 1
    ]
    return {
        **per_task,
        "global_duplicate_patch_hashes": global_duplicates,
        "global_duplicate_patch_hash_count": len(global_duplicates),
        "passes": bool(per_task.get("passes", False)),
    }


def _candidate_order_randomized_audit(examples: Sequence[PatchSelectionExample], stage6b1b_audit: Dict[str, object]) -> Dict[str, object]:
    randomizable = _stage6b1b_candidate_order_randomizable(stage6b1b_audit)
    rows = []
    for example in examples:
        slots = [candidate.metadata.get("stage6b1b_input_slot_index") for candidate in example.candidates]
        rows.append({"example_id": example.id, "issue_id": example.issue_id, "stage6b1b_input_slots": slots, "slot_order_preserved_for_harness": slots == list(range(len(slots)))})
    return {
        "stage6b1b_candidate_order_randomizable": bool(randomizable),
        "order_frozen_before_official_labeling": True,
        "slot_order_preserved_for_harness": all(bool(row["slot_order_preserved_for_harness"]) for row in rows),
        "no_post_label_reordering": True,
        "rows": rows,
        "passes": bool(randomizable) and all(bool(row["slot_order_preserved_for_harness"]) for row in rows),
    }


def _patch_integrity_audit(
    examples: Sequence[PatchSelectionExample],
    candidates_by_task: Dict[str, List[Stage6B2InputCandidate]],
    harness_root: Path,
    seed: int,
    k: int,
) -> Dict[str, object]:
    mismatches = []
    for example in examples:
        source_rows = candidates_by_task.get(example.issue_id, [])
        for index, candidate in enumerate(example.candidates):
            if index >= len(source_rows):
                mismatches.append({"example_id": example.id, "candidate_index": index, "error": "missing input row"})
                continue
            source_hash = source_rows[index].patch_hash
            if candidate.patch_hash != source_hash:
                mismatches.append({"example_id": example.id, "candidate_index": index, "input_hash": source_hash, "output_hash": candidate.patch_hash})
    prediction_mismatches = []
    for candidate_index in range(int(k)):
        predictions_path = harness_root / "predictions" / f"{_slot_run_id(seed, candidate_index)}.jsonl"
        if not predictions_path.exists():
            prediction_mismatches.append({"candidate_index": candidate_index, "error": "missing prediction file", "path": str(predictions_path)})
            continue
        rows = _read_jsonl(predictions_path)
        by_id = {str(row.get("instance_id")): str(row.get("model_patch", "")) for row in rows}
        for example in examples:
            expected_hash = example.candidates[candidate_index].patch_hash
            got_patch = by_id.get(example.issue_id, "")
            if stable_patch_hash(got_patch) != expected_hash:
                prediction_mismatches.append({"issue_id": example.issue_id, "candidate_index": candidate_index, "expected_hash": expected_hash, "prediction_hash": stable_patch_hash(got_patch)})
    return {
        "candidate_pool_patch_mismatches": mismatches,
        "prediction_patch_mismatches": prediction_mismatches,
        "passes": not mismatches and not prediction_mismatches,
    }


def _source_generator_leakage_audit(
    examples: Sequence[PatchSelectionExample],
    best_generator: Dict[str, object],
    oracle_audit: Dict[str, object],
) -> Dict[str, object]:
    forbidden_patterns = (
        "logs/run_evaluation",
        "stage6b2_official_harness",
        "candidate_harness_results",
        "report.json",
        "labels_pass_fail",
        "official_swebench_harness",
        "oracle_pass_at_8",
    )
    visible_hits = []
    for example in examples:
        visible_chunks = [example.issue_text, example.failing_test_summary, "\n".join(example.retrieved_contexts), str(example.metadata.get("dependency_callgraph_related_file_evidence", ""))]
        visible_chunks.extend(f"source_agent: {candidate.candidate_source_agent}\ndiff:\n{candidate.candidate_diff}" for candidate in example.candidates)
        lowered = "\n".join(visible_chunks).lower()
        hits = [pattern for pattern in forbidden_patterns if pattern in lowered]
        if hits:
            visible_hits.append({"example_id": example.id, "issue_id": example.issue_id, "hits": hits})
    best_equals_oracle = abs(float(best_generator.get("best_generator_on_dev_pass_at_1", 0.0)) - float(oracle_audit.get("oracle_pass_at_8", 0.0))) <= 1e-12
    return {
        "selector_visible_forbidden_source_or_label_hits": visible_hits,
        "generator_identity_visible_as_candidate_source_agent": True,
        "best_generator_on_dev_baseline_equals_oracle": bool(best_equals_oracle),
        "passes": not visible_hits and not best_equals_oracle,
    }


def _raw_preserving_and_fabrication_audit(
    examples: Sequence[PatchSelectionExample],
    stage6b1b_audit: Dict[str, object],
) -> Dict[str, object]:
    marker_hits = []
    for example in examples:
        for index, candidate in enumerate(example.candidates):
            diff = candidate.candidate_diff
            if "raw-preserving marker" in diff or "stage6b_candidate_markers/" in diff:
                marker_hits.append({"example_id": example.id, "issue_id": example.issue_id, "candidate_index": index, "candidate_id": candidate.candidate_id})
    stage6b1b_raw = _find_nested_bool(stage6b1b_audit, "raw_preserving_expansion_used")
    stage6b1b_fabricated = _find_nested_bool(stage6b1b_audit, "missing_candidate_fabrication_used")
    return {
        "stage6b1b_raw_preserving_expansion_used": stage6b1b_raw,
        "stage6b1b_missing_candidate_fabrication_used": stage6b1b_fabricated,
        "raw_preserving_marker_hits": marker_hits,
        "raw_preserving_expansion_used": bool(marker_hits) or bool(stage6b1b_raw),
        "missing_candidate_fabrication_used": bool(stage6b1b_fabricated),
        "passes": not marker_hits and not bool(stage6b1b_raw) and not bool(stage6b1b_fabricated),
    }


def _harness_invocation_audit(
    harness_root: Path,
    seed: int,
    k: int,
    max_workers: int,
    timeout: int,
    cache_level: str,
    namespace: str | None,
    force_rerun: bool,
    build_only: bool,
) -> Dict[str, object]:
    slot_rows = []
    for candidate_index in range(int(k)):
        run_id = _slot_run_id(seed, candidate_index)
        command_dir = harness_root / "command_logs" / run_id
        returncode_path = command_dir / "returncode.txt"
        prediction_path = harness_root / "predictions" / f"{run_id}.jsonl"
        slot_rows.append(
            {
                "candidate_index": candidate_index,
                "run_id": run_id,
                "model_name_or_path": _slot_model_name(candidate_index),
                "prediction_path": str(prediction_path),
                "prediction_rows": len(_read_jsonl(prediction_path)) if prediction_path.exists() else 0,
                "returncode": int(returncode_path.read_text(encoding="utf-8").strip()) if returncode_path.exists() else None,
                "command_log_dir": str(command_dir),
            }
        )
    return {
        "official_harness_command_template": (
            "python -m swebench.harness.run_evaluation --dataset_name {dataset_name} --split {split} "
            "--predictions_path {predictions_path} --max_workers {max_workers} --timeout {timeout} "
            "--run_id {run_id} --cache_level {cache_level} --clean False --report_dir {report_dir}"
        ),
        "batch_by_candidate_slot": True,
        "slot_rows": slot_rows,
        "max_workers": int(max_workers),
        "timeout": int(timeout),
        "cache_level": cache_level,
        "namespace": namespace,
        "force_rerun": bool(force_rerun),
        "build_only": bool(build_only),
        "raw_logs_under_harness_root": str(harness_root / "official_logs"),
        "passes": all(row["prediction_rows"] == 25 for row in slot_rows),
    }


def _any_slot_command_log(harness_root: Path, seed: int, k: int) -> bool:
    return any((harness_root / "command_logs" / _slot_run_id(seed, candidate_index)).exists() for candidate_index in range(int(k)))


def _all_required_audits_complete(audit: Dict[str, object]) -> bool:
    required = (
        "official_label_availability_audit",
        "candidate_count_audit",
        "oracle_pass_at_8_audit",
        "per_repository_oracle_pass_at_8",
        "per_generator_pass_rate",
        "first_candidate_baseline",
        "random_candidate_baseline",
        "best_generator_on_dev_baseline",
        "duplicate_candidate_patch_hash_audit",
        "exact_reference_patch_hash_audit",
        "candidate_order_randomized_audit",
        "output_leakage_audit",
        "source_generator_leakage_audit",
        "raw_preserving_and_fabrication_audit",
        "patch_integrity_audit",
    )
    return all(key in audit for key in required)


def collect_environment(dataset_name: str, split: str) -> Dict[str, object]:
    packages = {}
    for package in ("swebench", "datasets", "numpy", "docker"):
        try:
            proc = subprocess.run([sys.executable, "-m", "pip", "show", package], text=True, capture_output=True, timeout=60)
            version = ""
            for line in (proc.stdout or "").splitlines():
                if line.startswith("Version:"):
                    version = line.split(":", 1)[1].strip()
                    break
            packages[package] = version or "not found"
        except Exception as exc:
            packages[package] = f"error: {exc}"
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "wsl": _run_text(["uname", "-a"]),
        "docker_version": _run_text(["docker", "version", "--format", "{{.Server.Version}} {{.Server.Os}}/{{.Server.Arch}}"]),
        "packages": packages,
        "dataset_name": dataset_name,
        "split": split,
    }


def _stage6b1b_candidate_order_randomizable(stage6b1b_audit: Dict[str, object]) -> bool:
    value = _find_nested_bool(stage6b1b_audit, "candidate_order_randomizable")
    return bool(value)


def _find_nested_bool(value: object, key: str) -> bool | None:
    if isinstance(value, dict):
        if key in value and isinstance(value[key], bool):
            return bool(value[key])
        for item in value.values():
            found = _find_nested_bool(item, key)
            if found is not None:
                return found
    if isinstance(value, list):
        for item in value:
            found = _find_nested_bool(item, key)
            if found is not None:
                return found
    return None


def _retrieved_contexts(record: Dict[str, object]) -> tuple[str, ...]:
    hints = str(record.get("hints_text", "") or "")
    return (
        f"Repository: {record.get('repo', '')}\nBase commit: {record.get('base_commit', '')}\nVersion: {record.get('version', '')}",
        f"Hints text:\n{hints[:4000] if hints.strip() else 'No hints supplied.'}",
    )


def _failing_test_summary(record: Dict[str, object]) -> str:
    return "FAIL_TO_PASS:\n{fail}\n\nPASS_TO_PASS:\n{pass_to_pass}".format(
        fail=str(record.get("FAIL_TO_PASS", ""))[:4000],
        pass_to_pass=str(record.get("PASS_TO_PASS", ""))[:2000],
    )


def _dependency_context(record: Dict[str, object]) -> str:
    return (
        f"Repository {record.get('repo', '')} at base commit {record.get('base_commit', '')}. "
        "Selector-visible fields include issue evidence, failing-test names, hints, and candidate diffs only; official harness logs and labels are hidden."
    )


def _task_family(record: Dict[str, object]) -> str:
    repo = str(record.get("repo", "unknown"))
    return repo.split("/", 1)[0] if "/" in repo else repo


def _split_for_index(index: int, n: int) -> str:
    train_cut = max(1, int(round(n * 0.6)))
    dev_cut = max(train_cut + 1, int(round(n * 0.8)))
    if index < train_cut:
        return "train"
    if index < dev_cut:
        return "dev"
    return "test"


def _slot_model_name(candidate_index: int) -> str:
    return f"stage6b2_candidate_slot_{candidate_index}"


def _slot_run_id(seed: int, candidate_index: int) -> str:
    return f"stage6b2_seed_{seed}_slot_{candidate_index}"


def _optional_bool(row: Dict[str, object], keys: Iterable[str]) -> bool | None:
    for key in keys:
        if key in row:
            return bool(row[key])
    return None


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _mean_bool(values: Sequence[bool]) -> float:
    return float(sum(1 for value in values if value) / len(values)) if values else 0.0


def _mean_numeric(values: Sequence[int]) -> float:
    return float(sum(float(value) for value in values) / len(values)) if values else 0.0


def _read_json_or_empty(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return dict(value) if isinstance(value, dict) else {}
    except Exception:
        return {}


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows), encoding="utf-8")


def _run_text(command: Sequence[str]) -> str:
    try:
        proc = subprocess.run(list(command), text=True, capture_output=True, timeout=60)
        return (proc.stdout or proc.stderr or "").strip()[:1000]
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
