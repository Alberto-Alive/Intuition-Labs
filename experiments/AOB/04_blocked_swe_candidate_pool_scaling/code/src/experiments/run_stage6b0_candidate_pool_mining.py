from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np

from src.datasets.swe_patch_selection_dataset import (
    STAGE6_NUM_CANDIDATES,
    PatchCandidate,
    PatchSelectionExample,
    duplicate_candidate_patch_hash_audit,
    label_matrix,
    stage6_output_leakage_audit,
    stage6_split_leakage_audit,
    stable_patch_hash,
    validate_patch_selection_examples,
    write_patch_selection_jsonl,
)


BENCHMARK = "stage6b0_candidate_pool_mining"
DEFAULT_POOL_PATH = Path("results/stage6b0_candidate_pool_mining.jsonl")
DEFAULT_AUDIT_PATH = Path("results/stage6b0_candidate_pool_mining_audit.json")
DEFAULT_REPORT_PATH = Path("reports/STAGE6B0_CANDIDATE_POOL_MINING.md")
DEFAULT_HARNESS_ROOT = Path("results/stage6b0_harness")
DEFAULT_DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
DEFAULT_CODERFORGE_DATASET = "togethercomputer/CoderForge-Preview-32B-SWE-Bench-Verified-Evaluation-trajectories"
DEFAULT_HANSPETER_DATASET = "hanspeterlyngsoeraaschoujensen/swebench-eval-trajectories"
DEFAULT_ANTIEVAL_DATASET = "antieval/swebench-trajectories"
SELECTOR_VISIBLE_SOURCE_AGENT = "generated_candidate_source_blinded"
FORBIDDEN_METADATA_KEYS = re.compile(
    r"(reward|resolved|label|gold|reference|test_output|harness|log|pass_fail|oracle)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class GeneratedPatchRecord:
    instance_id: str
    source_name: str
    generator_name: str
    sample_index: int
    patch_diff: str
    metadata: Dict[str, object]

    @property
    def patch_hash(self) -> str:
        return stable_patch_hash(self.patch_diff)


@dataclass(frozen=True)
class HarnessCandidateResult:
    example_id: str
    issue_id: str
    repo: str
    candidate_index: int
    candidate_id: str
    true_generator_name: str
    true_source_name: str
    run_id: str
    model_name_or_path: str
    report_path: str
    test_output_path: str
    log_dir: str
    result_available: bool
    resolved: bool
    applied: bool | None
    timed_out: bool | None
    failure_kind: str
    report_keys: Tuple[str, ...]
    error: str | None


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine Stage 6B.0 real generated SWE-bench candidate pools.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-tasks", type=int, default=200)
    parser.add_argument("--seed", type=int, default=606_260)
    parser.add_argument("--pool-output", default=str(DEFAULT_POOL_PATH))
    parser.add_argument("--pool-audit", default=str(DEFAULT_AUDIT_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--harness-root", default=str(DEFAULT_HARNESS_ROOT))
    parser.add_argument("--source-jsonl", action="append", default=[])
    parser.add_argument("--source-json", action="append", default=[])
    parser.add_argument("--source-root", action="append", default=[])
    parser.add_argument("--skip-default-hf", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--skip-harness", action="store_true")
    parser.add_argument("--reuse-harness", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--cache-level", default="env", choices=("none", "base", "env", "instance"))
    parser.add_argument("--namespace", default="swebench")
    args = parser.parse_args()

    result = run_stage6b0_candidate_pool_mining(
        dataset_name=str(args.dataset_name),
        split=str(args.split),
        n_tasks=int(args.n_tasks),
        seed=int(args.seed),
        pool_path=Path(args.pool_output),
        audit_path=Path(args.pool_audit),
        report_path=Path(args.report),
        harness_root=Path(args.harness_root),
        source_jsonl=tuple(str(value) for value in args.source_jsonl),
        source_json=tuple(str(value) for value in args.source_json),
        source_roots=tuple(str(value) for value in args.source_root),
        include_default_hf=not bool(args.skip_default_hf),
        build_only=bool(args.build_only),
        skip_harness=bool(args.skip_harness),
        reuse_harness=bool(args.reuse_harness),
        max_workers=int(args.max_workers),
        timeout=int(args.timeout),
        cache_level=str(args.cache_level),
        namespace=None if str(args.namespace).lower() == "none" else str(args.namespace),
    )
    print(
        "stage6b0: wrote {pool}, {audit}, {report}; built={built}; full_labels={labels}".format(
            pool=args.pool_output,
            audit=args.pool_audit,
            report=args.report,
            built=result.get("audit", {}).get("tasks_built", 0),
            labels=result.get("audit", {}).get("tasks_with_all_official_labels", 0),
        )
    )


def run_stage6b0_candidate_pool_mining(
    dataset_name: str,
    split: str,
    n_tasks: int,
    seed: int,
    pool_path: Path,
    audit_path: Path,
    report_path: Path,
    harness_root: Path,
    source_jsonl: Sequence[str],
    source_json: Sequence[str],
    source_roots: Sequence[str],
    include_default_hf: bool,
    build_only: bool,
    skip_harness: bool,
    reuse_harness: bool,
    max_workers: int,
    timeout: int,
    cache_level: str,
    namespace: str | None,
) -> Dict[str, object]:
    created_at = _now()
    environment = collect_environment(dataset_name, split)
    benchmark_by_id, benchmark_audit = load_benchmark_records(dataset_name, split)
    source_audit: Dict[str, object] = {"sources": [], "errors": [], "redacted_metadata_fields": Counter()}
    generated_records: List[GeneratedPatchRecord] = []
    if benchmark_by_id:
        generated_records = mine_generated_patch_records(
            benchmark_by_id=benchmark_by_id,
            include_default_hf=include_default_hf,
            source_jsonl=source_jsonl,
            source_json=source_json,
            source_roots=source_roots,
            source_audit=source_audit,
        )
    examples, gold_hashes, gold_patches, build_audit = build_candidate_pool(
        benchmark_by_id=benchmark_by_id,
        generated_records=generated_records,
        dataset_name=dataset_name,
        split=split,
        n_tasks=n_tasks,
        seed=seed,
    )

    harness_results: List[HarnessCandidateResult] = []
    should_run_harness = bool(examples) and not build_only and not skip_harness
    if should_run_harness:
        harness_results = evaluate_with_official_harness(
            examples=examples,
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
    labeled_examples = attach_official_labels(examples, harness_results)
    write_patch_selection_jsonl(pool_path, labeled_examples)

    audit = audit_candidate_pool(
        examples=labeled_examples,
        gold_hashes=gold_hashes,
        gold_patches=gold_patches,
        generated_records=generated_records,
        benchmark_audit=benchmark_audit,
        source_audit=source_audit,
        build_audit=build_audit,
        harness_results=harness_results,
        environment=environment,
        requested_tasks=n_tasks,
        build_only=build_only,
        skip_harness=skip_harness,
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
    _write_json(audit_path, audit)
    report = render_report(
        audit=audit,
        pool_path=pool_path,
        audit_path=audit_path,
        report_path=report_path,
        harness_root=harness_root,
        created_at=created_at,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return {"audit": audit}


def load_benchmark_records(dataset_name: str, split: str) -> tuple[Dict[str, Dict[str, object]], Dict[str, object]]:
    audit = {"dataset_name": dataset_name, "split": split, "loaded": False, "rows": 0, "error": None}
    try:
        from datasets import load_dataset

        dataset = load_dataset(dataset_name, split=split)
        records = {str(row["instance_id"]): dict(row) for row in dataset}
        audit.update({"loaded": True, "rows": len(records)})
        return records, audit
    except Exception as exc:
        audit["error"] = f"{type(exc).__name__}: {exc}"
        return {}, audit


def mine_generated_patch_records(
    benchmark_by_id: Dict[str, Dict[str, object]],
    include_default_hf: bool,
    source_jsonl: Sequence[str],
    source_json: Sequence[str],
    source_roots: Sequence[str],
    source_audit: Dict[str, object],
) -> List[GeneratedPatchRecord]:
    records: List[GeneratedPatchRecord] = []
    if include_default_hf:
        records.extend(_load_hf_coderforge(benchmark_by_id, source_audit))
        records.extend(_load_hf_message_trajectory_dataset(DEFAULT_HANSPETER_DATASET, benchmark_by_id, source_audit))
        records.extend(_load_hf_message_trajectory_dataset(DEFAULT_ANTIEVAL_DATASET, benchmark_by_id, source_audit))
    for path in source_jsonl:
        records.extend(_load_prediction_file(Path(path), benchmark_by_id, source_audit, forced_kind="jsonl"))
    for path in source_json:
        records.extend(_load_prediction_file(Path(path), benchmark_by_id, source_audit, forced_kind="json"))
    for root in source_roots:
        records.extend(_load_prediction_root(Path(root), benchmark_by_id, source_audit))
    return records


def build_candidate_pool(
    benchmark_by_id: Dict[str, Dict[str, object]],
    generated_records: Sequence[GeneratedPatchRecord],
    dataset_name: str,
    split: str,
    n_tasks: int,
    seed: int,
) -> tuple[List[PatchSelectionExample], Dict[str, str], Dict[str, str], Dict[str, object]]:
    gold_hash_by_instance = {
        instance_id: stable_patch_hash(str(record.get("patch", "") or ""))
        for instance_id, record in benchmark_by_id.items()
    }
    gold_patch_by_instance = {
        instance_id: str(record.get("patch", "") or "")
        for instance_id, record in benchmark_by_id.items()
    }
    grouped: Dict[str, List[GeneratedPatchRecord]] = defaultdict(list)
    source_rows: List[Dict[str, object]] = []
    exact_reference_exclusions = []
    invalid_diff_rows = []
    duplicate_rows = []
    seen_by_instance: Dict[str, set[str]] = defaultdict(set)
    for row in generated_records:
        if row.instance_id not in benchmark_by_id:
            continue
        patch = _clean_patch(row.patch_diff)
        if not _looks_like_unified_diff(patch):
            invalid_diff_rows.append(
                {
                    "instance_id": row.instance_id,
                    "source_name": row.source_name,
                    "generator_name": row.generator_name,
                    "patch_hash": stable_patch_hash(patch),
                }
            )
            continue
        patch_hash = stable_patch_hash(patch)
        ref_hash = gold_hash_by_instance.get(row.instance_id, "")
        if ref_hash and patch_hash == ref_hash:
            exact_reference_exclusions.append(
                {
                    "instance_id": row.instance_id,
                    "source_name": row.source_name,
                    "generator_name": row.generator_name,
                    "patch_hash": patch_hash,
                }
            )
            continue
        if patch_hash in seen_by_instance[row.instance_id]:
            duplicate_rows.append(
                {
                    "instance_id": row.instance_id,
                    "source_name": row.source_name,
                    "generator_name": row.generator_name,
                    "patch_hash": patch_hash,
                }
            )
            continue
        seen_by_instance[row.instance_id].add(patch_hash)
        normalized = GeneratedPatchRecord(
            instance_id=row.instance_id,
            source_name=row.source_name,
            generator_name=row.generator_name,
            sample_index=int(row.sample_index),
            patch_diff=_ensure_trailing_newline(patch),
            metadata=dict(row.metadata),
        )
        grouped[row.instance_id].append(normalized)
        source_rows.append(
            {
                "instance_id": row.instance_id,
                "source_name": row.source_name,
                "generator_name": row.generator_name,
                "sample_index": int(row.sample_index),
                "patch_hash": normalized.patch_hash,
            }
        )

    eligible_ids = sorted(instance_id for instance_id, rows in grouped.items() if len(rows) >= STAGE6_NUM_CANDIDATES)
    rng = np.random.default_rng(seed + 6_260_101)
    if eligible_ids:
        selected_ids = [eligible_ids[int(index)] for index in rng.permutation(len(eligible_ids))[: int(n_tasks)]]
    else:
        selected_ids = []
    examples: List[PatchSelectionExample] = []
    gold_hashes: Dict[str, str] = {}
    gold_patches: Dict[str, str] = {}
    for index, instance_id in enumerate(selected_ids):
        record = benchmark_by_id[instance_id]
        selected = _select_diverse_k(grouped[instance_id], seed + index)
        order_seed = seed + 6_260_201 + index
        perm = _non_identity_permutation(len(selected), np.random.default_rng(order_seed))
        candidates = []
        for shuffled_position, pre_index in enumerate(perm):
            generated = selected[int(pre_index)]
            metadata = {
                **generated.metadata,
                "audit_generator_name": generated.generator_name,
                "audit_source_name": generated.source_name,
                "source_patch_hash": generated.patch_hash,
                "sample_index": int(generated.sample_index),
                "pre_shuffle_index": int(pre_index),
                "shuffled_position": int(shuffled_position),
                "candidate_order_seed": int(order_seed),
                "generator_identity_selector_visible": False,
                "true_source_metadata_visible_to_selector": False,
                "gold_reference_diagnostic": False,
            }
            candidates.append(
                PatchCandidate(
                    candidate_id=f"{instance_id}-stage6b0-candidate-{shuffled_position}",
                    candidate_source_agent=SELECTOR_VISIBLE_SOURCE_AGENT,
                    candidate_diff=generated.patch_diff,
                    candidate_visible_test_result=None,
                    metadata=metadata,
                )
            )
        split_name = _split_for_index(index, len(selected_ids))
        example = PatchSelectionExample(
            id=f"stage6b0-{instance_id}",
            dataset_name=f"{dataset_name}:{split}",
            repo=str(record.get("repo", "")),
            issue_id=instance_id,
            issue_text=str(record.get("problem_statement", "")),
            failing_test_summary=_failing_test_summary(record),
            retrieved_contexts=_retrieved_contexts(record),
            candidates=tuple(candidates),
            labels_pass_fail=tuple(0 for _ in candidates),
            split=split_name,
            metadata={
                "task_family": _task_family(record),
                "candidate_pool_version": "stage6b0_real_generated_patch_mining_v1",
                "candidate_pool_source": "public_generated_patch_artifacts_only",
                "benchmark_source": dataset_name,
                "benchmark_split": split,
                "base_commit": str(record.get("base_commit", "")),
                "environment_setup_commit": str(record.get("environment_setup_commit", "")),
                "version": str(record.get("version", "")),
                "created_at": str(record.get("created_at", "")),
                "visible_test_results_allowed": False,
                "gold_patch_included": False,
                "publishable_proof_dataset": False,
                "full_swebench_harness_ran": False,
                "label_source": "pending official SWE-bench harness",
                "candidate_order_seed": int(order_seed),
                "candidate_order_randomized": True,
                "generator_identity_blinded_in_selector_visible_fields": True,
                "dependency_callgraph_related_file_evidence": _dependency_context(record),
            },
        )
        examples.append(example)
        gold_hashes[example.id] = gold_hash_by_instance.get(instance_id, "")
        gold_patches[example.id] = gold_patch_by_instance.get(instance_id, "")

    candidate_counts = Counter(len(rows) for rows in grouped.values())
    build_audit = {
        "generated_patch_records_loaded": len(generated_records),
        "official_tasks_with_any_generated_candidate": len(grouped),
        "official_tasks_with_at_least_k_unique_generated_candidates": len(eligible_ids),
        "candidate_count_histogram_before_k_filter": {str(key): int(value) for key, value in sorted(candidate_counts.items())},
        "tasks_selected_for_pool": len(examples),
        "source_rows_after_filters": source_rows,
        "exact_reference_patch_exclusions": exact_reference_exclusions,
        "invalid_diff_rows": invalid_diff_rows[:200],
        "invalid_diff_rows_total": len(invalid_diff_rows),
        "duplicate_patch_exclusions": duplicate_rows[:200],
        "duplicate_patch_exclusions_total": len(duplicate_rows),
        "raw_preserving_expansion_used": False,
        "diagnostic_only_expansion_used": False,
        "source_limit_documented": len(eligible_ids) < int(n_tasks) or len(grouped) < 100,
    }
    return examples, gold_hashes, gold_patches, build_audit


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
    command_log_dir = harness_root / "command_logs"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    command_log_dir.mkdir(parents=True, exist_ok=True)
    all_results: List[HarnessCandidateResult] = []
    for candidate_index in range(STAGE6_NUM_CANDIDATES):
        model_name = f"stage6b0_candidate_slot_{candidate_index}"
        run_id = f"stage6b0_seed_{seed}_slot_{candidate_index}"
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
            proc = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=max(timeout * max(2, len(examples)), timeout + 600),
            )
            (command_log_dir / f"{run_id}.stdout.log").write_text(proc.stdout or "", encoding="utf-8")
            (command_log_dir / f"{run_id}.stderr.log").write_text(proc.stderr or "", encoding="utf-8")
            (command_log_dir / f"{run_id}.returncode.txt").write_text(str(proc.returncode), encoding="utf-8")
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
                labels.append(0)
                all_available = False
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


def audit_candidate_pool(
    examples: Sequence[PatchSelectionExample],
    gold_hashes: Dict[str, str],
    gold_patches: Dict[str, str],
    generated_records: Sequence[GeneratedPatchRecord],
    benchmark_audit: Dict[str, object],
    source_audit: Dict[str, object],
    build_audit: Dict[str, object],
    harness_results: Sequence[HarnessCandidateResult],
    environment: Dict[str, object],
    requested_tasks: int,
    build_only: bool,
    skip_harness: bool,
    harness_config: Dict[str, object],
) -> Dict[str, object]:
    validation = validate_patch_selection_examples(examples, allow_gold_diagnostic=False)
    harness_by_candidate = {(row.example_id, int(row.candidate_index)): row for row in harness_results}
    label_availability = []
    patch_apply_rows = []
    for example in examples:
        available_count = 0
        for index, candidate in enumerate(example.candidates):
            row = harness_by_candidate.get((example.id, index))
            available = row is not None and row.result_available
            available_count += int(available)
            patch_apply_rows.append(
                {
                    "example_id": example.id,
                    "issue_id": example.issue_id,
                    "candidate_index": index,
                    "candidate_id": candidate.candidate_id,
                    "true_generator_name": _candidate_true_generator(candidate),
                    "true_source_name": _candidate_true_source(candidate),
                    "result_available": available,
                    "applied": None if row is None else row.applied,
                    "resolved": False if row is None else row.resolved,
                    "timed_out": None if row is None else row.timed_out,
                    "failure_kind": "missing_report" if row is None else row.failure_kind,
                    "report_path": None if row is None else row.report_path,
                    "test_output_path": None if row is None else row.test_output_path,
                }
            )
        label_availability.append({"example_id": example.id, "available": available_count, "expected": STAGE6_NUM_CANDIDATES})

    complete_examples = [
        example
        for example in examples
        if all((example.id, index) in harness_by_candidate and harness_by_candidate[(example.id, index)].result_available for index in range(len(example.candidates)))
    ]
    labels = label_matrix(complete_examples)
    oracle = labels.sum(axis=1) > 0 if labels.size else np.zeros(0, dtype=bool)
    all_labels = label_matrix(examples)
    all_oracle = all_labels.sum(axis=1) > 0 if all_labels.size else np.zeros(0, dtype=bool)
    gold_hits = _exact_reference_hits(examples, gold_hashes)
    near_reference = _near_reference_similarity_audit(examples, gold_patches)
    split_audit = _split_audit(examples)
    output_audit = stage6_output_leakage_audit(BENCHMARK, 0, examples)
    duplicate_audit = duplicate_candidate_patch_hash_audit(examples)
    candidate_order = _candidate_order_randomized_audit(examples)
    first = _first_candidate_pass(labels)
    random = _random_candidate_pass(labels, seed=611)
    generator_pass = _per_generator_pass_rate(complete_examples)
    per_repo_oracle = _per_group_oracle(complete_examples, lambda ex: ex.repo)
    best_generator = _best_generator_on_dev_baseline(complete_examples)
    official_availability = {
        "rows": label_availability,
        "available_candidate_results": sum(row["available"] for row in label_availability),
        "expected_candidate_results": len(examples) * STAGE6_NUM_CANDIDATES,
        "tasks_with_all_official_labels": len(complete_examples),
        "passes": bool(examples) and all(row["available"] == row["expected"] for row in label_availability),
    }
    oracle_distribution = {
        "basis": "complete_labeled_tasks",
        "oracle_pass_at_8": float(np.mean(oracle)) if len(oracle) else 0.0,
        "oracle_positive_tasks": int(np.sum(oracle)) if len(oracle) else 0,
        "oracle_empty_tasks": int(np.sum(~oracle)) if len(oracle) else 0,
        "positive_candidates_per_task_histogram": dict(Counter(int(value) for value in labels.sum(axis=1).tolist())) if labels.size else {},
        "all_built_tasks_oracle_pass_at_8_with_missing_labels_as_fail": float(np.mean(all_oracle)) if len(all_oracle) else 0.0,
    }
    train_positive = sum(1 for example in complete_examples if example.split == "train" and example.oracle_pass_at_8)
    dev_positive = sum(1 for example in complete_examples if example.split == "dev" and example.oracle_pass_at_8)
    source_leakage = {
        "selector_visible_candidate_source_agent_values": sorted({candidate.candidate_source_agent for example in examples for candidate in example.candidates}),
        "true_generator_metadata_selector_visible": False,
        "true_source_metadata_selector_visible": False,
        "best_generator_on_dev_pass_at_1_all": best_generator.get("all_pass_at_1", 0.0),
        "oracle_pass_at_8": oracle_distribution["oracle_pass_at_8"],
        "best_generator_source_trivially_equals_oracle": bool(
            complete_examples and abs(float(best_generator.get("all_pass_at_1", 0.0)) - float(oracle_distribution["oracle_pass_at_8"])) <= 1e-9
        ),
        "passes": bool(examples)
        and sorted({candidate.candidate_source_agent for example in examples for candidate in example.candidates}) == [SELECTOR_VISIBLE_SOURCE_AGENT]
        and not (
            complete_examples and abs(float(best_generator.get("all_pass_at_1", 0.0)) - float(oracle_distribution["oracle_pass_at_8"])) <= 1e-9
        ),
    }
    quality_gates = {
        "at_least_100_real_tasks_attempted_or_documented_limit": bool(
            int(build_audit.get("official_tasks_with_any_generated_candidate", 0)) >= 100 or bool(build_audit.get("source_limit_documented", False))
        ),
        "at_least_50_tasks_with_full_k8_official_labels": bool(len(complete_examples) >= 50),
        "oracle_pass_at_8_between_0_20_and_0_80": bool(0.20 <= float(oracle_distribution["oracle_pass_at_8"]) <= 0.80),
        "at_least_30_oracle_positive_train_examples": bool(train_positive >= 30),
        "at_least_10_oracle_positive_dev_examples": bool(dev_positive >= 10),
        "first_candidate_baseline_not_equal_oracle": bool(
            complete_examples and abs(float(first) - float(oracle_distribution["oracle_pass_at_8"])) > 1e-9
        ),
        "best_generator_source_baseline_not_equal_oracle": bool(
            complete_examples and abs(float(best_generator.get("all_pass_at_1", 0.0)) - float(oracle_distribution["oracle_pass_at_8"])) > 1e-9
        ),
        "no_gold_reference_patch_leakage": bool(not gold_hits and output_audit.get("passes", False)),
        "all_audits_complete": True,
        "no_raw_preserving_expansion": bool(not build_audit.get("raw_preserving_expansion_used", False)),
        "no_final_model_claim_made": True,
    }
    quality_gates["pool_quality_gates_pass"] = all(bool(value) for value in quality_gates.values())
    return {
        "benchmark": BENCHMARK,
        "created_at_utc": _now(),
        "requested_tasks": int(requested_tasks),
        "tasks_built": len(examples),
        "tasks_with_all_official_labels": len(complete_examples),
        "build_only": bool(build_only),
        "skip_harness": bool(skip_harness),
        "selector_training_executed": False,
        "no_final_model_claim_made": True,
        "environment": environment,
        "benchmark_dataset_audit": benchmark_audit,
        "source_mining_audit": _json_ready_source_audit(source_audit),
        "candidate_pool_build_audit": build_audit,
        "harness_config": harness_config,
        "candidate_pool_validation": validation,
        "k8_validation_audit": {
            "expected_k": STAGE6_NUM_CANDIDATES,
            "bad_examples": [example.id for example in examples if len(example.candidates) != STAGE6_NUM_CANDIDATES],
            "passes": bool(examples) and all(len(example.candidates) == STAGE6_NUM_CANDIDATES for example in examples),
        },
        "official_label_availability_audit": official_availability,
        "oracle_pass_at_8_distribution": oracle_distribution,
        "per_generator_pass_rate": generator_pass,
        "per_repository_oracle_pass_at_8": per_repo_oracle,
        "duplicate_patch_hash_audit": duplicate_audit,
        "exact_reference_patch_hash_audit": {
            "candidate_hits": gold_hits,
            "source_exact_reference_exclusions": build_audit.get("exact_reference_patch_exclusions", []),
            "passes": not gold_hits,
        },
        "near_duplicate_reference_similarity_audit": near_reference,
        "candidate_order_randomized_audit": candidate_order,
        "candidate_order_baseline_audit": {
            "first_candidate_pass_at_1": first,
            "random_candidate_pass_at_1": random,
        },
        "first_candidate_baseline": first,
        "best_generator_on_dev_baseline": best_generator,
        "generator_source_leakage_audit": source_leakage,
        "output_leakage_audit": output_audit,
        "train_dev_test_split_audit": split_audit,
        "patch_application_success_failure_audit": {
            "rows": patch_apply_rows,
            "applied_count": sum(1 for row in patch_apply_rows if row.get("applied") is True),
            "explicit_apply_status_available_count": sum(1 for row in patch_apply_rows if row.get("applied") is not None),
            "failure_kind_histogram": dict(Counter(str(row.get("failure_kind")) for row in patch_apply_rows)),
        },
        "official_harness_result_availability_audit": official_availability,
        "pool_quality_gates": quality_gates,
    }


def render_report(
    audit: Dict[str, object],
    pool_path: Path,
    audit_path: Path,
    report_path: Path,
    harness_root: Path,
    created_at: str,
) -> str:
    oracle = audit.get("oracle_pass_at_8_distribution", {})
    availability = audit.get("official_label_availability_audit", {})
    build = audit.get("candidate_pool_build_audit", {})
    quality = audit.get("pool_quality_gates", {})
    lines = [
        "# Stage 6B.0 Candidate Pool Mining",
        "",
        "## Scope",
        "",
        "- No final model claim is made from Stage 6B.0.",
        "- The selector is not trained by this mining script.",
        "- Candidate patches are accepted only from generated public prediction or trajectory artifacts.",
        "- Reference/gold patches are excluded by exact normalized patch hash and audited.",
        "- Raw-preserving patch expansion is not used.",
        "- Correctness labels are accepted only from the official Docker SWE-bench harness `resolved` result.",
        "- Raw harness logs are stored under the harness root and are not selector-visible input.",
        "- True generator/source identities are retained only in audit metadata; selector-visible candidate source is blinded.",
        "",
        "## Environment",
        "",
        f"- Created at UTC: `{created_at}`.",
        f"- Platform: `{audit.get('environment', {}).get('platform')}`.",
        f"- Python: `{audit.get('environment', {}).get('python')}`.",
        f"- Docker: `{audit.get('environment', {}).get('docker_version')}`.",
        f"- WSL: `{audit.get('environment', {}).get('wsl')}`.",
        f"- Packages: `{json.dumps(audit.get('environment', {}).get('packages', {}), sort_keys=True)}`.",
        f"- Official harness command module: `python -m swebench.harness.run_evaluation`.",
        "",
        "## Artifacts",
        "",
        f"- Candidate pool: `{pool_path}`.",
        f"- Candidate-pool audit: `{audit_path}`.",
        f"- Report: `{report_path}`.",
        f"- Harness root: `{harness_root}`.",
        f"- Official raw harness log copies: `{harness_root / 'official_logs'}`.",
        "",
        "## Source Mining",
        "",
        f"- Benchmark dataset loaded: `{audit.get('benchmark_dataset_audit', {}).get('loaded')}`.",
        f"- Benchmark rows: `{audit.get('benchmark_dataset_audit', {}).get('rows')}`.",
        f"- Generated patch records loaded: `{build.get('generated_patch_records_loaded')}`.",
        f"- Official tasks with at least one generated candidate after filters: `{build.get('official_tasks_with_any_generated_candidate')}`.",
        f"- Official tasks with at least K=8 unique generated candidates: `{build.get('official_tasks_with_at_least_k_unique_generated_candidates')}`.",
        f"- Raw-preserving expansion used: `{build.get('raw_preserving_expansion_used')}`.",
        f"- Exact reference patch exclusions from sources: `{len(build.get('exact_reference_patch_exclusions', []))}`.",
        f"- Duplicate patch exclusions: `{build.get('duplicate_patch_exclusions_total')}`.",
        f"- Invalid diff exclusions: `{build.get('invalid_diff_rows_total')}`.",
        "",
        "## Candidate Pool",
        "",
        f"- Requested tasks: `{audit.get('requested_tasks')}`.",
        f"- Tasks built: `{audit.get('tasks_built')}`.",
        f"- K=8 validation passes: `{audit.get('k8_validation_audit', {}).get('passes')}`.",
        f"- Duplicate candidate patch hash audit passes: `{audit.get('duplicate_patch_hash_audit', {}).get('passes')}`.",
        f"- Exact reference candidate audit passes: `{audit.get('exact_reference_patch_hash_audit', {}).get('passes')}`.",
        f"- Candidate order randomized audit passes: `{audit.get('candidate_order_randomized_audit', {}).get('passes')}`.",
        f"- Output leakage audit passes: `{audit.get('output_leakage_audit', {}).get('passes')}`.",
        "",
        "## Official Harness Labels",
        "",
        f"- Candidate result availability: `{availability.get('available_candidate_results')}/{availability.get('expected_candidate_results')}`.",
        f"- Tasks with all official labels: `{availability.get('tasks_with_all_official_labels')}`.",
        f"- Availability audit passes: `{availability.get('passes')}`.",
        f"- Oracle pass@8 on complete labeled tasks: `{float(oracle.get('oracle_pass_at_8', 0.0)):.4f}`.",
        f"- Oracle-positive tasks: `{oracle.get('oracle_positive_tasks')}`.",
        f"- Oracle-empty tasks retained for unconditional pass@1: `{oracle.get('oracle_empty_tasks')}`.",
        f"- Per-repository oracle pass@8: `{json.dumps(audit.get('per_repository_oracle_pass_at_8', {}), sort_keys=True)}`.",
        f"- Per-generator pass rate: `{json.dumps(audit.get('per_generator_pass_rate', {}), sort_keys=True)}`.",
        "",
        "## Baselines",
        "",
        f"- Random candidate pass@1: `{audit.get('candidate_order_baseline_audit', {}).get('random_candidate_pass_at_1', 0.0):.4f}`.",
        f"- First-candidate pass@1: `{audit.get('candidate_order_baseline_audit', {}).get('first_candidate_pass_at_1', 0.0):.4f}`.",
        f"- Best-generator-on-dev baseline: `{json.dumps(audit.get('best_generator_on_dev_baseline', {}), sort_keys=True)}`.",
        "",
        "## Quality Gates",
        "",
        "| gate | pass |",
        "|---|---:|",
    ]
    for key, value in quality.items():
        lines.append(f"| `{key}` | `{bool(value)}` |")
    if not bool(quality.get("pool_quality_gates_pass", False)):
        lines.extend(
            [
                "",
                "## Current Limitation",
                "",
                "The Stage 6B.0 pool quality gates did not pass. This script does not fabricate missing candidates: tasks with fewer than K=8 unique generated public patches are excluded, and raw-preserving expansion remains disabled. Add more public prediction artifacts through `--source-jsonl`, `--source-json`, or `--source-root`, then rerun with the same seed to resume harness labeling.",
            ]
        )
    lines.extend(["", "No final model claim is made.", ""])
    return "\n".join(lines)


def _load_hf_coderforge(
    benchmark_by_id: Dict[str, Dict[str, object]],
    source_audit: Dict[str, object],
) -> List[GeneratedPatchRecord]:
    source_name = f"hf:{DEFAULT_CODERFORGE_DATASET}:trajectory"
    rows: List[GeneratedPatchRecord] = []
    summary = {"source_name": source_name, "loaded": False, "records": 0, "official_records": 0, "error": None}
    try:
        from datasets import load_dataset

        dataset = load_dataset(DEFAULT_CODERFORGE_DATASET, "trajectory", split="train")
        counters: Dict[tuple[str, str], int] = defaultdict(int)
        for row_index, row in enumerate(dataset):
            ds = _json_obj(row.get("ds", {}))
            instance_id = str(ds.get("instance_id") or _instance_id_from_trajectory(str(row.get("trajectory_id", ""))))
            patch = _clean_patch(str(row.get("output_patch", "") or ""))
            if instance_id not in benchmark_by_id or not patch.strip():
                continue
            generator = _clean_generator_name(row.get("exp_name") or row.get("model") or "CoderForge-Preview-32B")
            sample = counters[(instance_id, generator)]
            counters[(instance_id, generator)] += 1
            rows.append(
                GeneratedPatchRecord(
                    instance_id=instance_id,
                    source_name=source_name,
                    generator_name=generator,
                    sample_index=sample,
                    patch_diff=patch,
                    metadata=_redacted_metadata(
                        {
                            "source_dataset": DEFAULT_CODERFORGE_DATASET,
                            "source_config": "trajectory",
                            "source_row_index": row_index,
                            "trajectory_id": row.get("trajectory_id"),
                            "run_id": row.get("run_id"),
                            "exp_name": row.get("exp_name"),
                        },
                        source_audit,
                    ),
                )
            )
        summary.update({"loaded": True, "records": len(dataset), "official_records": len(rows)})
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        source_audit.setdefault("errors", []).append(summary)
    source_audit.setdefault("sources", []).append(summary)
    return rows


def _load_hf_message_trajectory_dataset(
    dataset_name: str,
    benchmark_by_id: Dict[str, Dict[str, object]],
    source_audit: Dict[str, object],
) -> List[GeneratedPatchRecord]:
    source_name = f"hf:{dataset_name}"
    rows: List[GeneratedPatchRecord] = []
    summary = {"source_name": source_name, "loaded": False, "records": 0, "official_records": 0, "patches_extracted": 0, "error": None}
    try:
        from datasets import load_dataset

        dataset = load_dataset(dataset_name, split="train")
        counters: Dict[tuple[str, str], int] = defaultdict(int)
        for row_index, row in enumerate(dataset):
            instance_id = str(row.get("instance_id") or row.get("id") or "")
            if instance_id not in benchmark_by_id:
                continue
            patches = _extract_diff_blocks_from_messages(row.get("messages", ""))
            generator = _clean_generator_name(row.get("model") or row.get("agent") or dataset_name.rsplit("/", 1)[-1])
            for patch in patches:
                sample = counters[(instance_id, generator)]
                counters[(instance_id, generator)] += 1
                rows.append(
                    GeneratedPatchRecord(
                        instance_id=instance_id,
                        source_name=source_name,
                        generator_name=generator,
                        sample_index=sample,
                        patch_diff=patch,
                        metadata=_redacted_metadata(
                            {
                                "source_dataset": dataset_name,
                                "source_row_index": row_index,
                                "model": row.get("model"),
                                "extraction": "messages_diff_block",
                            },
                            source_audit,
                        ),
                    )
                )
        summary.update({"loaded": True, "records": len(dataset), "official_records": len({row.instance_id for row in rows}), "patches_extracted": len(rows)})
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        source_audit.setdefault("errors", []).append(summary)
    source_audit.setdefault("sources", []).append(summary)
    return rows


def _load_prediction_root(
    root: Path,
    benchmark_by_id: Dict[str, Dict[str, object]],
    source_audit: Dict[str, object],
) -> List[GeneratedPatchRecord]:
    rows: List[GeneratedPatchRecord] = []
    if not root.exists():
        source_audit.setdefault("sources", []).append({"source_name": f"root:{root}", "loaded": False, "records": 0, "error": "path does not exist"})
        return rows
    patterns = ("all_preds.jsonl", "preds.json", "predictions.jsonl", "predictions.json", "*.pred", "*.jsonl", "*.json")
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(root.rglob(pattern))
    seen = set()
    for path in sorted(paths):
        if path in seen or path.is_dir() or path.stat().st_size == 0:
            continue
        seen.add(path)
        if path.stat().st_size > 200_000_000:
            source_audit.setdefault("sources", []).append({"source_name": f"file:{path}", "loaded": False, "records": 0, "error": "file too large for mining guardrail"})
            continue
        rows.extend(_load_prediction_file(path, benchmark_by_id, source_audit, forced_kind=None))
    return rows


def _load_prediction_file(
    path: Path,
    benchmark_by_id: Dict[str, Dict[str, object]],
    source_audit: Dict[str, object],
    forced_kind: str | None,
) -> List[GeneratedPatchRecord]:
    source_name = f"file:{path.as_posix()}"
    summary = {"source_name": source_name, "loaded": False, "records": 0, "official_records": 0, "error": None}
    rows: List[GeneratedPatchRecord] = []
    try:
        parsed = list(_iter_prediction_rows(path, forced_kind))
        counters: Dict[tuple[str, str], int] = defaultdict(int)
        for row_index, row in enumerate(parsed):
            for instance_id, patch, generator, metadata in _prediction_row_to_patches(row, path, benchmark_by_id):
                sample = counters[(instance_id, generator)]
                counters[(instance_id, generator)] += 1
                rows.append(
                    GeneratedPatchRecord(
                        instance_id=instance_id,
                        source_name=source_name,
                        generator_name=generator,
                        sample_index=sample,
                        patch_diff=patch,
                        metadata=_redacted_metadata(
                            {**metadata, "source_file": path.as_posix(), "source_row_index": row_index},
                            source_audit,
                        ),
                    )
                )
        summary.update({"loaded": True, "records": len(parsed), "official_records": len(rows)})
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        source_audit.setdefault("errors", []).append(summary)
    source_audit.setdefault("sources", []).append(summary)
    return rows


def _prediction_row_to_patches(
    row: Dict[str, object],
    path: Path,
    benchmark_by_id: Dict[str, Dict[str, object]],
) -> Iterator[tuple[str, str, str, Dict[str, object]]]:
    instance_id = str(row.get("instance_id") or row.get("id") or row.get("task_id") or "")
    if not instance_id and path.stem in benchmark_by_id:
        instance_id = path.stem
    if instance_id not in benchmark_by_id:
        return
    generator = _clean_generator_name(
        row.get("generator_name")
        or row.get("model_name_or_path")
        or row.get("model")
        or row.get("agent")
        or row.get("generator")
        or path.parent.name
        or "public_prediction_artifact"
    )
    patch_values: List[object] = []
    for key in ("model_patch", "output_patch", "generated_patch", "patch", "prediction", "diff"):
        value = row.get(key)
        if isinstance(value, dict):
            for nested in ("model_patch", "output_patch", "generated_patch", "patch", "diff"):
                if value.get(nested):
                    patch_values.append(value.get(nested))
        elif value:
            patch_values.append(value)
    if row.get("messages"):
        patch_values.extend(_extract_diff_blocks_from_messages(row.get("messages")))
    for value in patch_values:
        patch = _clean_patch(str(value or ""))
        if patch.strip():
            yield (
                instance_id,
                patch,
                generator,
                {
                    "source_format": "prediction_artifact",
                    "model_name_or_path": row.get("model_name_or_path"),
                    "generator_name": row.get("generator_name"),
                    "model": row.get("model"),
                    "agent": row.get("agent"),
                    "generator": row.get("generator"),
                    "seed": row.get("seed"),
                    "temperature": row.get("temperature"),
                    "config_name": row.get("config_name"),
                    "trajectory_path": row.get("trajectory_path"),
                    "source_visible_to_selector": row.get("source_visible_to_selector"),
                },
            )


def _iter_prediction_rows(path: Path, forced_kind: str | None) -> Iterator[Dict[str, object]]:
    kind = forced_kind or ("jsonl" if path.suffix.lower() in {".jsonl", ".pred"} else "json")
    if kind == "jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                yield row
    elif isinstance(data, dict):
        if isinstance(data.get("predictions"), list):
            for row in data["predictions"]:
                if isinstance(row, dict):
                    yield row
        elif isinstance(data.get("instances"), list):
            for row in data["instances"]:
                if isinstance(row, dict):
                    yield row
        else:
            for key, value in data.items():
                if isinstance(value, dict):
                    yield {"instance_id": key, **value}
                elif isinstance(value, str):
                    yield {"instance_id": key, "model_patch": value}


def _extract_diff_blocks_from_messages(messages: object) -> List[str]:
    texts: List[str] = []
    if isinstance(messages, str):
        texts = [messages]
    elif isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                content = message.get("content") or message.get("text") or message.get("message") or ""
                if isinstance(content, list):
                    texts.extend(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
                else:
                    texts.append(str(content))
            else:
                texts.append(str(message))
    else:
        texts = [str(messages)]
    patches: List[str] = []
    for text in texts:
        for fenced in re.findall(r"```(?:diff|patch)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
            candidate = _clean_patch(fenced)
            if "diff --git " in candidate:
                patches.append(candidate)
        lines = text.splitlines()
        start_indexes = [index for index, line in enumerate(lines) if line.startswith("diff --git ")]
        for start in start_indexes:
            block = []
            for line in lines[start:]:
                if line.startswith("```"):
                    break
                block.append(line)
            candidate = _clean_patch("\n".join(block))
            if "diff --git " in candidate:
                patches.append(candidate)
    deduped = []
    seen = set()
    for patch in patches:
        patch_hash = stable_patch_hash(patch)
        if patch_hash not in seen:
            seen.add(patch_hash)
            deduped.append(_ensure_trailing_newline(patch))
    return deduped


def _write_predictions_for_slot(
    path: Path,
    examples: Sequence[PatchSelectionExample],
    candidate_index: int,
    model_name: str,
) -> None:
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
        report_keys: Tuple[str, ...] = ()
        failure_kind = "missing_report"
        if result_available:
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                item = report.get(example.issue_id, {})
                report_keys = tuple(sorted(str(key) for key in item.keys()))
                resolved = bool(item.get("resolved", False))
                applied = _optional_bool(item, ("patch_successfully_applied", "patch_applied", "applied"))
                timed_out = _optional_bool(item, ("timed_out", "timeout"))
                failure_kind = _failure_kind(resolved=resolved, applied=applied, timed_out=timed_out, item=item)
            except Exception as exc:
                result_available = False
                error = f"{type(exc).__name__}: {exc}"
                failure_kind = "unreadable_report"
        candidate = example.candidates[candidate_index]
        rows.append(
            HarnessCandidateResult(
                example_id=example.id,
                issue_id=example.issue_id,
                repo=example.repo,
                candidate_index=int(candidate_index),
                candidate_id=candidate.candidate_id,
                true_generator_name=_candidate_true_generator(candidate),
                true_source_name=_candidate_true_source(candidate),
                run_id=run_id,
                model_name_or_path=model_name,
                report_path=str(report_path),
                test_output_path=str(test_output_path),
                log_dir=str(log_dir),
                result_available=result_available,
                resolved=resolved,
                applied=applied,
                timed_out=timed_out,
                failure_kind=failure_kind,
                report_keys=report_keys,
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


def _failure_kind(resolved: bool, applied: bool | None, timed_out: bool | None, item: Dict[str, object]) -> str:
    if resolved:
        return "resolved"
    if applied is False:
        return "patch_apply_failed"
    if timed_out is True:
        return "timeout"
    if item.get("error"):
        return "harness_error"
    return "unresolved_test_or_build_failure"


def _select_diverse_k(records: Sequence[GeneratedPatchRecord], seed: int) -> List[GeneratedPatchRecord]:
    rng = np.random.default_rng(seed + 6_260_301)
    grouped: Dict[str, List[GeneratedPatchRecord]] = defaultdict(list)
    for row in records:
        grouped[f"{row.source_name}|{row.generator_name}"].append(row)
    for rows in grouped.values():
        rng.shuffle(rows)
    group_keys = list(grouped)
    rng.shuffle(group_keys)
    selected: List[GeneratedPatchRecord] = []
    while len(selected) < STAGE6_NUM_CANDIDATES and group_keys:
        next_keys = []
        for key in group_keys:
            if grouped[key] and len(selected) < STAGE6_NUM_CANDIDATES:
                selected.append(grouped[key].pop(0))
            if grouped[key]:
                next_keys.append(key)
        group_keys = next_keys
    if len(selected) < STAGE6_NUM_CANDIDATES:
        remaining = [row for rows in grouped.values() for row in rows]
        rng.shuffle(remaining)
        selected.extend(remaining[: STAGE6_NUM_CANDIDATES - len(selected)])
    return selected[:STAGE6_NUM_CANDIDATES]


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


def _split_audit(examples: Sequence[PatchSelectionExample]) -> Dict[str, object]:
    splits = _split_examples_for_audit(examples)
    base = stage6_split_leakage_audit(BENCHMARK, 0, splits)
    repos = {name: {example.repo for example in rows} for name, rows in splits.items()}
    base.update(
        {
            "split_sizes": {name: len(rows) for name, rows in splits.items()},
            "train_dev_repo_overlap": len(repos.get("train", set()) & repos.get("dev", set())),
            "train_test_repo_overlap": len(repos.get("train", set()) & repos.get("test", set())),
            "dev_test_repo_overlap": len(repos.get("dev", set()) & repos.get("test", set())),
        }
    )
    return base


def _exact_reference_hits(examples: Sequence[PatchSelectionExample], gold_hashes: Dict[str, str]) -> List[Dict[str, object]]:
    hits = []
    for example in examples:
        ref_hash = gold_hashes.get(example.id, "")
        for index, candidate in enumerate(example.candidates):
            if ref_hash and candidate.patch_hash == ref_hash:
                hits.append({"example_id": example.id, "candidate_index": index, "candidate_id": candidate.candidate_id})
    return hits


def _near_reference_similarity_audit(
    examples: Sequence[PatchSelectionExample],
    gold_patches: Dict[str, str],
) -> Dict[str, object]:
    rows = []
    for example in examples:
        reference = str(gold_patches.get(example.id, ""))
        if not reference:
            continue
        for index, candidate in enumerate(example.candidates):
            score = _patch_similarity(candidate.candidate_diff, reference)
            if score >= 0.95:
                rows.append({"example_id": example.id, "candidate_index": index, "similarity": score, "patch_hash": candidate.patch_hash})
    return {
        "method": "SequenceMatcher on normalized patch text; reference patches are used only in this audit and are not written to selector-visible pool records",
        "reference_patches_checked": len([value for value in gold_patches.values() if value]),
        "high_similarity_threshold": 0.95,
        "high_similarity_hits": rows,
        "passes": not rows,
    }


def _per_generator_pass_rate(examples: Sequence[PatchSelectionExample]) -> Dict[str, float]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for example in examples:
        for candidate, label in zip(example.candidates, example.labels_pass_fail):
            grouped[_candidate_true_generator(candidate)].append(int(label))
    return {key: float(np.mean(values)) if values else 0.0 for key, values in sorted(grouped.items())}


def _per_group_oracle(examples: Sequence[PatchSelectionExample], key_fn) -> Dict[str, float]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for example in examples:
        grouped[str(key_fn(example))].append(1 if example.oracle_pass_at_8 else 0)
    return {key: float(np.mean(values)) if values else 0.0 for key, values in sorted(grouped.items())}


def _best_generator_on_dev_baseline(examples: Sequence[PatchSelectionExample]) -> Dict[str, object]:
    dev = [example for example in examples if example.split == "dev"]
    rates: Dict[str, List[int]] = defaultdict(list)
    for example in dev:
        for candidate, label in zip(example.candidates, example.labels_pass_fail):
            rates[_candidate_true_generator(candidate)].append(int(label))
    if not rates:
        return {"selected_generator": None, "dev_candidate_level_rates": {}, "all_pass_at_1": 0.0, "by_split_pass_at_1": {}}
    dev_rates = {key: float(np.mean(values)) for key, values in rates.items()}
    selected = sorted(dev_rates.items(), key=lambda item: (-item[1], item[0]))[0][0]
    by_split: Dict[str, List[int]] = defaultdict(list)
    all_hits = []
    for example in examples:
        pick = next((index for index, candidate in enumerate(example.candidates) if _candidate_true_generator(candidate) == selected), 0)
        hit = int(example.labels_pass_fail[pick])
        by_split[example.split].append(hit)
        all_hits.append(hit)
    return {
        "selected_generator": selected,
        "dev_candidate_level_rates": dict(sorted(dev_rates.items())),
        "all_pass_at_1": float(np.mean(all_hits)) if all_hits else 0.0,
        "by_split_pass_at_1": {key: float(np.mean(values)) for key, values in sorted(by_split.items())},
    }


def _first_candidate_pass(labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    return float(np.mean(labels[:, 0] > 0))


def _random_candidate_pass(labels: np.ndarray, seed: int) -> float:
    if labels.size == 0:
        return 0.0
    rng = np.random.default_rng(seed + 6_260_401)
    picks = rng.integers(0, labels.shape[1], size=labels.shape[0])
    return float(np.mean([labels[row, int(choice)] > 0 for row, choice in enumerate(picks)]))


def collect_environment(dataset_name: str, split: str) -> Dict[str, object]:
    packages = {}
    for package in ("swebench", "datasets", "torch", "numpy", "docker", "pytest"):
        packages[package] = _pip_version(package)
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "wsl": _run_text(["uname", "-a"]),
        "docker_version": _run_text(["docker", "version", "--format", "{{.Server.Version}} {{.Server.Os}}/{{.Server.Arch}}"]),
        "packages": packages,
        "dataset_name": dataset_name,
        "split": split,
        "official_harness_module": "swebench.harness.run_evaluation",
    }


def _pip_version(package: str) -> str:
    try:
        proc = subprocess.run(["python", "-m", "pip", "show", package], text=True, capture_output=True, timeout=60)
        for line in (proc.stdout or "").splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
        return "not found"
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


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
        "No generated callgraph artifact was supplied for Stage 6B.0; selector sees issue, failing tests, hints, and candidate diffs only."
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


def _split_examples_for_audit(examples: Sequence[PatchSelectionExample]) -> Dict[str, List[PatchSelectionExample]]:
    splits = {"train": [], "dev": [], "test": []}
    for example in examples:
        splits.setdefault(example.split, []).append(example)
    return {key: splits.get(key, []) for key in ("train", "dev", "test")}


def _non_identity_permutation(n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 1:
        return np.arange(n, dtype=np.int64)
    for _ in range(16):
        perm = rng.permutation(n)
        if not np.array_equal(perm, np.arange(n)):
            return perm.astype(np.int64, copy=False)
    return np.roll(np.arange(n, dtype=np.int64), 1)


def _candidate_true_generator(candidate: PatchCandidate) -> str:
    return str(candidate.metadata.get("audit_generator_name", "unknown_generator"))


def _candidate_true_source(candidate: PatchCandidate) -> str:
    return str(candidate.metadata.get("audit_source_name", "unknown_source"))


def _json_ready_source_audit(source_audit: Dict[str, object]) -> Dict[str, object]:
    out = dict(source_audit)
    redacted = out.get("redacted_metadata_fields", {})
    if isinstance(redacted, Counter):
        out["redacted_metadata_fields"] = dict(redacted)
    return out


def _redacted_metadata(row: Dict[str, object], source_audit: Dict[str, object]) -> Dict[str, object]:
    redacted_counter = source_audit.setdefault("redacted_metadata_fields", Counter())
    clean: Dict[str, object] = {}
    for key, value in row.items():
        if value is None:
            continue
        if FORBIDDEN_METADATA_KEYS.search(str(key)):
            redacted_counter[str(key)] += 1
            continue
        clean[str(key)] = _json_safe_scalar(value)
    return clean


def _json_safe_scalar(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe_scalar(item) for item in value[:20]]
    if isinstance(value, dict):
        return {str(key): _json_safe_scalar(val) for key, val in list(value.items())[:20] if not FORBIDDEN_METADATA_KEYS.search(str(key))}
    return str(value)


def _patch_similarity(left: str, right: str) -> float:
    left_norm = "\n".join(line.rstrip() for line in str(left).strip().splitlines())
    right_norm = "\n".join(line.rstrip() for line in str(right).strip().splitlines())
    if not left_norm or not right_norm:
        return 0.0
    max_len = 40_000
    return float(SequenceMatcher(None, left_norm[:max_len], right_norm[:max_len]).ratio())


def _clean_patch(value: str) -> str:
    text = str(value or "").replace("\r\n", "\n")
    fenced = re.search(r"```(?:diff|patch)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced and "diff --git " in fenced.group(1):
        text = fenced.group(1)
    if "diff --git " in text:
        text = text[text.find("diff --git ") :]
    return text.strip() + ("\n" if text.strip() else "")


def _looks_like_unified_diff(value: str) -> bool:
    text = str(value or "")
    return "diff --git " in text and ("--- " in text or "+++ " in text)


def _ensure_trailing_newline(text: str) -> str:
    value = str(text)
    return value if value.endswith("\n") else value + "\n"


def _clean_generator_name(value: object) -> str:
    raw = str(value or "unknown_generator")
    cleaned = re.sub(r"[^A-Za-z0-9_.:/@+-]+", "_", raw).strip("_")
    return cleaned[:160] or "unknown_generator"


def _instance_id_from_trajectory(trajectory_id: str) -> str:
    return str(trajectory_id).rsplit("_run", 1)[0]


def _json_obj(value: object) -> Dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except Exception:
            return {}
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
