from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from src.datasets.swe_patch_selection_dataset import (
    STAGE6_NUM_CANDIDATES,
    PatchCandidate,
    PatchSelectionExample,
    duplicate_candidate_patch_hash_audit,
    stable_patch_hash,
    write_patch_selection_jsonl,
)
from src.experiments.run_stage6b1_candidate_generation import DEFAULT_DATASET_NAME, BenchmarkTask, load_benchmark_tasks
from src.experiments.run_stage6b1b_candidate_supply_expansion import (
    _patch_candidates_from_value,
    validate_unified_diff,
)
from src.experiments.run_stage6d_pool_scale_and_view_hardening import (
    NEAR_REFERENCE_THRESHOLD,
    build_pool_summary,
    candidate_order_randomized_audit,
    exact_reference_hash_audit,
    near_reference_similarity_audit,
)
from src.experiments.run_stage6d2_pool_scaling import DEFAULT_CANDIDATE_SOURCES, DEFAULT_PUBLIC_SOURCE_ROOT


BENCHMARK = "stage6d3_candidate_source_expansion"
DEFAULT_OUTPUT_ALL = Path("results/stage6d3_candidate_pool_all_unlabeled.jsonl")
DEFAULT_OUTPUT_STRICT = Path("results/stage6d3_candidate_pool_strict_unlabeled.jsonl")
DEFAULT_ATTEMPTS = Path("results/stage6d3_candidate_source_expansion_attempts.jsonl")
DEFAULT_INVENTORY = Path("results/stage6d3_source_inventory.json")
DEFAULT_GENERATION_PLAN = Path("results/stage6d3_generation_plan.json")
DEFAULT_READINESS = Path("results/stage6d3_pre_harness_readiness_audit.json")
DEFAULT_SLOT_DIR = Path("results/stage6d3_harness_slot_predictions")
DEFAULT_REPORT = Path("reports/STAGE6D3_CANDIDATE_SOURCE_EXPANSION.md")
DEFAULT_SEED = 606_430

PUBLIC_GENERATED_SOURCE_GLOBS = ("*.json", "*.jsonl", "*.ndjson")
PATCH_KEYS = (
    "model_patch",
    "output_patch",
    "generated_patch",
    "candidate_diff",
    "patch",
    "diff",
    "prediction",
    "response",
    "final_response",
    "messages",
    "trajectory",
)
PROMPT_VARIANTS = (
    "minimal_fix",
    "traceback_fix",
    "localized_patch",
    "conservative_patch",
    "regression_aware_patch",
    "test_failure_patch",
)


@dataclass(frozen=True)
class GeneratedAttempt:
    instance_id: str
    patch: str
    patch_hash: str
    source_file: str
    source_row_index: int
    patch_index: int
    generator_name: str
    terminal_status: str
    reason: str
    near_reference: bool = False
    near_reference_similarity: float | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6D.3 source expansion and pre-harness labeling readiness.")
    parser.add_argument("--candidate-source", action="append", default=[str(path) for path in DEFAULT_CANDIDATE_SOURCES])
    parser.add_argument("--source-root", action="append", default=[str(DEFAULT_PUBLIC_SOURCE_ROOT)])
    parser.add_argument("--output-all", default=str(DEFAULT_OUTPUT_ALL))
    parser.add_argument("--output-strict", default=str(DEFAULT_OUTPUT_STRICT))
    parser.add_argument("--attempts", default=str(DEFAULT_ATTEMPTS))
    parser.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    parser.add_argument("--generation-plan", default=str(DEFAULT_GENERATION_PLAN))
    parser.add_argument("--readiness", default=str(DEFAULT_READINESS))
    parser.add_argument("--slot-dir", default=str(DEFAULT_SLOT_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--minimum-k8-tasks", type=int, default=50)
    parser.add_argument("--preferred-k8-tasks", type=int, default=100)
    parser.add_argument("--max-generation-plan-tasks", type=int, default=80)
    args = parser.parse_args()

    result = run_stage6d3_candidate_source_expansion(
        candidate_sources=[Path(value) for value in args.candidate_source],
        source_roots=[Path(value) for value in args.source_root],
        output_all=Path(args.output_all),
        output_strict=Path(args.output_strict),
        attempts_path=Path(args.attempts),
        inventory_path=Path(args.inventory),
        generation_plan_path=Path(args.generation_plan),
        readiness_path=Path(args.readiness),
        slot_dir=Path(args.slot_dir),
        report_path=Path(args.report),
        dataset_name=str(args.dataset_name),
        split=str(args.split),
        seed=int(args.seed),
        minimum_k8_tasks=int(args.minimum_k8_tasks),
        preferred_k8_tasks=int(args.preferred_k8_tasks),
        max_generation_plan_tasks=int(args.max_generation_plan_tasks),
    )
    print(
        "stage6d3: wrote {inventory}, {all_pool}, {strict_pool}, {attempts}, {plan}, {readiness}, {slot_dir}, {report}; "
        "pool_all_tasks={all_tasks}; pool_strict_tasks={strict_tasks}; decision={decision}".format(
            inventory=args.inventory,
            all_pool=args.output_all,
            strict_pool=args.output_strict,
            attempts=args.attempts,
            plan=args.generation_plan,
            readiness=args.readiness,
            slot_dir=args.slot_dir,
            report=args.report,
            all_tasks=result["inventory"]["pool_estimates"]["pool_all_k8_tasks"],
            strict_tasks=result["inventory"]["pool_estimates"]["pool_strict_k8_tasks"],
            decision=result["readiness"]["decision"],
        )
    )


def run_stage6d3_candidate_source_expansion(
    candidate_sources: Sequence[Path],
    source_roots: Sequence[Path],
    output_all: Path,
    output_strict: Path,
    attempts_path: Path,
    inventory_path: Path,
    generation_plan_path: Path,
    readiness_path: Path,
    slot_dir: Path,
    report_path: Path,
    dataset_name: str,
    split: str,
    seed: int,
    minimum_k8_tasks: int,
    preferred_k8_tasks: int,
    max_generation_plan_tasks: int,
) -> Dict[str, object]:
    created_at = _now()
    task_records, benchmark_audit = load_benchmark_tasks(
        dataset_name=dataset_name,
        split=split,
        benchmark_jsonl=None,
        requested_task_ids=[],
        task_ids_file=None,
        n_tasks=500,
        seed=seed,
    )
    source_files = discover_source_files(candidate_sources, source_roots)
    attempts, source_rows = scan_generated_sources(source_files, task_records)
    grouped = group_accepted_attempts(attempts)
    pool_all, pool_all_selection = build_unlabeled_pool(grouped, task_records, seed=seed, strict=False)
    pool_strict, pool_strict_selection = build_unlabeled_pool(grouped, task_records, seed=seed, strict=True)
    write_patch_selection_jsonl(output_all, pool_all)
    write_patch_selection_jsonl(output_strict, pool_strict)
    _write_jsonl(attempts_path, [attempt_to_record(attempt) for attempt in attempts])

    inventory = build_source_inventory(
        created_at=created_at,
        source_files=source_files,
        source_rows=source_rows,
        attempts=attempts,
        task_records=task_records,
        pool_all=pool_all,
        pool_strict=pool_strict,
        pool_all_selection=pool_all_selection,
        pool_strict_selection=pool_strict_selection,
        benchmark_audit=benchmark_audit,
    )
    _write_json(inventory_path, inventory)

    generation_plan = build_generation_plan(
        task_records=task_records,
        grouped=grouped,
        pool_strict=pool_strict,
        minimum_k8_tasks=minimum_k8_tasks,
        preferred_k8_tasks=preferred_k8_tasks,
        max_generation_plan_tasks=max_generation_plan_tasks,
    )
    _write_json(generation_plan_path, generation_plan)

    readiness = build_pre_harness_readiness(
        pool_all=pool_all,
        pool_strict=pool_strict,
        attempts=attempts,
        inventory=inventory,
        generation_plan=generation_plan,
        minimum_k8_tasks=minimum_k8_tasks,
        seed=seed,
    )
    prepare_harness_slot_predictions(pool_all, slot_dir, readiness, seed=seed)
    _write_json(readiness_path, readiness)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(inventory, generation_plan, readiness), encoding="utf-8")
    return {"inventory": inventory, "generation_plan": generation_plan, "readiness": readiness}


def discover_source_files(candidate_sources: Sequence[Path], source_roots: Sequence[Path]) -> List[Path]:
    files = []
    for path in candidate_sources:
        if path.exists() and path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".ndjson"}:
            files.append(path)
    for root in source_roots:
        if not root.exists():
            continue
        for pattern in PUBLIC_GENERATED_SOURCE_GLOBS:
            files.extend(path for path in root.rglob(pattern) if path.is_file() and ".git" not in path.parts)
    return list(dict.fromkeys(files))


def scan_generated_sources(
    source_files: Sequence[Path],
    task_records: Dict[str, BenchmarkTask],
) -> tuple[List[GeneratedAttempt], List[Dict[str, object]]]:
    attempts: List[GeneratedAttempt] = []
    source_rows = []
    verified_ids = set(task_records)
    seen_by_task: Dict[str, set[str]] = defaultdict(set)
    for source_file in source_files:
        records = read_records(source_file)
        raw_ids = set()
        patch_records = 0
        source_attempt_start = len(attempts)
        for row_index, record in enumerate(records):
            instance_id = normalize_instance_id(record, source_file)
            if instance_id:
                raw_ids.add(instance_id)
            patches = extract_candidate_patches(record)
            if patches:
                patch_records += len(patches)
            if not instance_id or instance_id not in verified_ids:
                for patch_index, patch in enumerate(patches or [""]):
                    attempts.append(
                        GeneratedAttempt(
                            instance_id=instance_id,
                            patch=str(patch or ""),
                            patch_hash=stable_patch_hash(str(patch or "")),
                            source_file=str(source_file),
                            source_row_index=row_index,
                            patch_index=patch_index,
                            generator_name=generator_name(record, source_file),
                            terminal_status="not_in_swebench_verified",
                            reason="no_matching_verified_instance_id",
                        )
                    )
                continue
            if not patches:
                attempts.append(
                    GeneratedAttempt(
                        instance_id=instance_id,
                        patch="",
                        patch_hash="",
                        source_file=str(source_file),
                        source_row_index=row_index,
                        patch_index=0,
                        generator_name=generator_name(record, source_file),
                        terminal_status="no_patch_found",
                        reason="no_extractable_patch_diff",
                    )
                )
                continue
            row_seen = set()
            for patch_index, raw_patch in enumerate(patches):
                patch = clean_patch(raw_patch)
                patch_hash = stable_patch_hash(patch)
                if not patch.strip():
                    status = "empty_patch"
                    reason = "empty_patch"
                else:
                    validation = validate_unified_diff(patch)
                    if not validation.valid:
                        status = "invalid_diff"
                        reason = validation.reason
                    elif patch_hash == task_records[instance_id].reference_hash:
                        status = "exact_reference_hash"
                        reason = "exact_reference_hash"
                    elif patch_hash in seen_by_task[instance_id] or patch_hash in row_seen:
                        status = "duplicate_patch"
                        reason = "duplicate_normalized_patch_hash"
                    else:
                        similarity = patch_similarity(patch, task_records[instance_id].reference_patch)
                        near_reference = bool(task_records[instance_id].reference_patch.strip() and similarity >= NEAR_REFERENCE_THRESHOLD)
                        status = "accepted"
                        reason = "accepted_near_reference_flag" if near_reference else "accepted"
                        attempts.append(
                            GeneratedAttempt(
                                instance_id=instance_id,
                                patch=patch,
                                patch_hash=patch_hash,
                                source_file=str(source_file),
                                source_row_index=row_index,
                                patch_index=patch_index,
                                generator_name=generator_name(record, source_file),
                                terminal_status=status,
                                reason=reason,
                                near_reference=near_reference,
                                near_reference_similarity=similarity,
                            )
                        )
                        seen_by_task[instance_id].add(patch_hash)
                        row_seen.add(patch_hash)
                        continue
                attempts.append(
                    GeneratedAttempt(
                        instance_id=instance_id,
                        patch=patch,
                        patch_hash=patch_hash,
                        source_file=str(source_file),
                        source_row_index=row_index,
                        patch_index=patch_index,
                        generator_name=generator_name(record, source_file),
                        terminal_status=status,
                        reason=reason,
                    )
                )
        source_attempts = attempts[source_attempt_start:]
        source_rows.append(
            {
                "source_file": str(source_file),
                "raw_records_loaded": len(records),
                "raw_instance_ids": len(raw_ids),
                "verified_instance_id_overlap": len(raw_ids & verified_ids),
                "patch_records_extracted": patch_records,
                "accepted_candidates": sum(1 for attempt in source_attempts if attempt.terminal_status == "accepted"),
                "near_reference_flags": sum(1 for attempt in source_attempts if attempt.near_reference),
                "invalid_diff_exclusions": sum(1 for attempt in source_attempts if attempt.terminal_status == "invalid_diff"),
                "duplicate_exclusions": sum(1 for attempt in source_attempts if attempt.terminal_status == "duplicate_patch"),
                "exact_reference_exclusions": sum(1 for attempt in source_attempts if attempt.terminal_status == "exact_reference_hash"),
            }
        )
    return attempts, source_rows


def group_accepted_attempts(attempts: Sequence[GeneratedAttempt]) -> Dict[str, List[GeneratedAttempt]]:
    grouped: Dict[str, List[GeneratedAttempt]] = defaultdict(list)
    for attempt in attempts:
        if attempt.terminal_status == "accepted":
            grouped[attempt.instance_id].append(attempt)
    return {instance_id: rows for instance_id, rows in sorted(grouped.items())}


def build_unlabeled_pool(
    grouped: Dict[str, List[GeneratedAttempt]],
    task_records: Dict[str, BenchmarkTask],
    seed: int,
    strict: bool,
) -> tuple[List[PatchSelectionExample], Dict[str, object]]:
    rng = np.random.default_rng(seed + (661 if strict else 337))
    examples = []
    selected_by_task = {}
    quarantined = []
    for instance_id in sorted(grouped):
        rows = [attempt for attempt in grouped[instance_id] if not strict or not attempt.near_reference]
        if len(rows) < STAGE6_NUM_CANDIDATES:
            if strict and any(attempt.near_reference for attempt in grouped[instance_id]):
                quarantined.append({"instance_id": instance_id, "reason": "near_reference_removal_breaks_k8", "remaining_non_near_candidates": len(rows)})
            continue
        order = rng.permutation(len(rows)).tolist()
        selected = [rows[index] for index in order[:STAGE6_NUM_CANDIDATES]]
        task = task_records.get(instance_id)
        if task is None:
            continue
        examples.append(task_to_patch_selection_example(task, selected, seed=seed, strict=strict))
        selected_by_task[instance_id] = [attempt.patch_hash for attempt in selected]
    return examples, {"strict": strict, "selected_task_count": len(examples), "selected_patch_hashes_by_task": selected_by_task, "quarantined_tasks": quarantined}


def task_to_patch_selection_example(task: BenchmarkTask, selected: Sequence[GeneratedAttempt], seed: int, strict: bool) -> PatchSelectionExample:
    record = task.record
    candidates = []
    for index, attempt in enumerate(selected):
        candidates.append(
            PatchCandidate(
                candidate_id=f"{task.instance_id}-stage6d3-candidate-{index:02d}-{attempt.patch_hash[:12]}",
                candidate_source_agent="source_blinded_generated",
                candidate_diff=attempt.patch,
                candidate_visible_test_result=None,
                metadata={
                    "source_visible_to_selector": False,
                    "stage6d3_source_file": attempt.source_file,
                    "stage6d3_source_row_index": attempt.source_row_index,
                    "stage6d3_patch_index": attempt.patch_index,
                    "stage6d3_generator_name": attempt.generator_name,
                    "stage6d3_original_patch_hash": attempt.patch_hash,
                    "stage6d3_near_reference_flag": bool(attempt.near_reference),
                    "stage6d3_near_reference_similarity": attempt.near_reference_similarity,
                },
            )
        )
    failing = f"FAIL_TO_PASS:\n{record.get('FAIL_TO_PASS', '[]')}\n\nPASS_TO_PASS:\n{record.get('PASS_TO_PASS', '[]')}"
    contexts = [
        f"Repository: {record.get('repo', '')}",
        f"Base commit: {record.get('base_commit', '')}",
        f"Version: {record.get('version', '')}",
    ]
    hints = str(record.get("hints_text", "") or "").strip()
    if hints:
        contexts.append("Hints text:\n" + hints)
    return PatchSelectionExample(
        id=f"stage6d3-{task.instance_id}",
        dataset_name=f"{DEFAULT_DATASET_NAME}:test",
        repo=str(record.get("repo", "")),
        issue_id=task.instance_id,
        issue_text=str(record.get("problem_statement", "")),
        failing_test_summary=failing,
        retrieved_contexts=tuple(contexts),
        candidates=tuple(candidates),
        labels_pass_fail=(),
        split="unlabeled",
        metadata={
            "candidate_pool_version": "stage6d3_unlabeled_generated_pool_v1",
            "stage6d3_pool": "pool_strict_unlabeled" if strict else "pool_all_unlabeled",
            "candidate_order_randomized": True,
            "candidate_order_seed": int(seed),
            "candidate_order_source": "stage6d3_randomized_after_dedup",
            "gold_patch_included": False,
            "raw_preserving_expansion_used": False,
            "missing_candidate_fabrication_used": False,
            "source_identity_blinded_from_selector_visible_candidate_text": True,
            "official_labels_available": False,
            "selector_training_executed": False,
        },
    )


def build_source_inventory(
    created_at: str,
    source_files: Sequence[Path],
    source_rows: Sequence[Dict[str, object]],
    attempts: Sequence[GeneratedAttempt],
    task_records: Dict[str, BenchmarkTask],
    pool_all: Sequence[PatchSelectionExample],
    pool_strict: Sequence[PatchSelectionExample],
    pool_all_selection: Dict[str, object],
    pool_strict_selection: Dict[str, object],
    benchmark_audit: Dict[str, object],
) -> Dict[str, object]:
    verified_ids = set(task_records)
    accepted = [attempt for attempt in attempts if attempt.terminal_status == "accepted"]
    per_task = Counter(attempt.instance_id for attempt in accepted)
    distribution = Counter(str(count) for count in per_task.values())
    status_counts = Counter(attempt.terminal_status for attempt in attempts)
    generator_counts = Counter(attempt.generator_name for attempt in accepted)
    near_task_ids = sorted({attempt.instance_id for attempt in accepted if attempt.near_reference})
    pool_all_ids = {example.issue_id for example in pool_all}
    pool_strict_ids = {example.issue_id for example in pool_strict}
    return {
        "benchmark": BENCHMARK,
        "created_at_utc": created_at,
        "source_files_scanned": len(source_files),
        "source_files": [str(path) for path in source_files],
        "source_file_summaries": list(source_rows),
        "raw_records_loaded": sum(int(row.get("raw_records_loaded", 0)) for row in source_rows),
        "unique_source_instance_ids": len({attempt.instance_id for attempt in attempts if attempt.instance_id}),
        "swebench_verified_task_ids": len(verified_ids),
        "overlap_with_swebench_verified_task_ids": len({attempt.instance_id for attempt in attempts if attempt.instance_id in verified_ids}),
        "tasks_with_at_least_1_candidate": sum(1 for count in per_task.values() if count >= 1),
        "tasks_with_at_least_4_candidates": sum(1 for count in per_task.values() if count >= 4),
        "tasks_with_at_least_8_candidates": sum(1 for count in per_task.values() if count >= 8),
        "candidates_per_task_distribution": dict(sorted(distribution.items(), key=lambda item: int(item[0]))),
        "accepted_candidates_total": len(accepted),
        "generator_source_distribution": dict(generator_counts),
        "terminal_status_counts": dict(status_counts),
        "duplicate_exclusions": int(status_counts.get("duplicate_patch", 0)),
        "invalid_diff_exclusions": int(status_counts.get("invalid_diff", 0)),
        "exact_reference_exclusions": int(status_counts.get("exact_reference_hash", 0)),
        "near_reference_flags": sum(1 for attempt in accepted if attempt.near_reference),
        "near_reference_task_ids": near_task_ids,
        "estimated_maximum_complete_k8_tasks_from_current_sources": sum(1 for count in per_task.values() if count >= 8),
        "estimated_maximum_strict_pool_after_near_reference_quarantine": len(pool_strict),
        "pool_estimates": {
            "pool_all_k8_tasks": len(pool_all),
            "pool_strict_k8_tasks": len(pool_strict),
            "pool_all_candidate_count": sum(len(example.candidates) for example in pool_all),
            "pool_strict_candidate_count": sum(len(example.candidates) for example in pool_strict),
            "pool_all_task_ids": sorted(pool_all_ids),
            "pool_strict_task_ids": sorted(pool_strict_ids),
            "strict_quarantined_task_ids": sorted(pool_all_ids - pool_strict_ids),
        },
        "pool_all_selection": pool_all_selection,
        "pool_strict_selection": pool_strict_selection,
        "benchmark_dataset_audit": benchmark_audit,
        "no_reference_or_gold_patches_used": True,
        "no_raw_preserving_expansion": True,
        "no_fabricated_candidates": True,
        "source_identity_blinded_from_selector_visible_candidate_text": True,
    }


def build_generation_plan(
    task_records: Dict[str, BenchmarkTask],
    grouped: Dict[str, List[GeneratedAttempt]],
    pool_strict: Sequence[PatchSelectionExample],
    minimum_k8_tasks: int,
    preferred_k8_tasks: int,
    max_generation_plan_tasks: int,
) -> Dict[str, object]:
    strict_ids = {example.issue_id for example in pool_strict}
    current_non_near_counts = {
        instance_id: sum(1 for attempt in attempts if not attempt.near_reference)
        for instance_id, attempts in grouped.items()
    }
    need_for_minimum = max(0, minimum_k8_tasks - len(strict_ids))
    need_for_preferred = max(0, preferred_k8_tasks - len(strict_ids))
    candidates = []
    for instance_id in sorted(task_records):
        if instance_id in strict_ids:
            continue
        current = int(current_non_near_counts.get(instance_id, 0))
        candidates.append(
            {
                "instance_id": instance_id,
                "current_valid_non_near_candidates": current,
                "candidates_needed_for_k8": max(0, STAGE6_NUM_CANDIDATES - current),
                "planned_max_attempts": 32,
                "prompt_variants": list(PROMPT_VARIANTS),
            }
        )
    candidates.sort(key=lambda row: (int(row["candidates_needed_for_k8"]), row["instance_id"]))
    planned = candidates[: max(0, int(max_generation_plan_tasks))]
    return {
        "benchmark": BENCHMARK,
        "created_at_utc": _now(),
        "current_strict_k8_tasks": len(strict_ids),
        "minimum_k8_tasks": int(minimum_k8_tasks),
        "preferred_k8_tasks": int(preferred_k8_tasks),
        "additional_k8_tasks_needed_for_minimum": need_for_minimum,
        "additional_k8_tasks_needed_for_preferred": need_for_preferred,
        "tasks_needing_candidates": planned,
        "planned_task_count": len(planned),
        "estimated_total_generation_attempts": int(sum(row["planned_max_attempts"] for row in planned)),
        "self_generation_not_executed": True,
        "execute_only_if_explicitly_enabled": True,
        "backend_model_requirements": {
            "supports_unified_diff_output": True,
            "stores_raw_trajectories_separately": True,
            "overgenerate_attempts_per_task": 32,
            "keep_first_8_unique_valid_generated_patches": True,
            "apply_exact_reference_duplicate_invalid_near_reference_audits": True,
        },
    }


def build_pre_harness_readiness(
    pool_all: Sequence[PatchSelectionExample],
    pool_strict: Sequence[PatchSelectionExample],
    attempts: Sequence[GeneratedAttempt],
    inventory: Dict[str, object],
    generation_plan: Dict[str, object],
    minimum_k8_tasks: int,
    seed: int,
) -> Dict[str, object]:
    duplicate_all = duplicate_candidate_patch_hash_audit(pool_all)
    duplicate_strict = duplicate_candidate_patch_hash_audit(pool_strict)
    order_all = candidate_order_randomized_audit(pool_all)
    order_strict = candidate_order_randomized_audit(pool_strict)
    exact_reference_hits = [attempt_to_record(attempt) for attempt in attempts if attempt.terminal_status == "exact_reference_hash"]
    exact_reference_scan_audit = {
        "type": "source_scan_exact_reference_exclusion",
        "hits": exact_reference_hits[:200],
        "hits_total": len(exact_reference_hits),
        "passes": not exact_reference_hits,
        "note": "Exact reference hashes are excluded before unlabeled pool construction.",
    }
    pool_all_k8 = len(pool_all)
    pool_strict_k8 = len(pool_strict)
    source_limit = pool_strict_k8 < minimum_k8_tasks
    gates = {
        "at_least_50_k8_generated_candidate_tasks_in_pool_all": pool_all_k8 >= minimum_k8_tasks,
        "at_least_50_k8_generated_candidate_tasks_in_pool_strict_or_source_limit_documented": bool(pool_strict_k8 >= minimum_k8_tasks or source_limit),
        "exact_reference_patch_hits_zero": not any(attempt.terminal_status == "exact_reference_hash" for attempt in attempts),
        "duplicate_candidate_patch_audit_passes": bool(duplicate_all.get("passes", False) and duplicate_strict.get("passes", False)),
        "no_raw_preserving_expansion": True,
        "no_fabricated_candidates": True,
        "candidate_order_randomized": bool(order_all.get("passes", False) and order_strict.get("passes", False)),
        "source_generator_skew_reported": bool(inventory.get("generator_source_distribution")),
        "near_reference_candidates_flagged_or_quarantined": True,
        "selector_training_not_executed": True,
        "no_final_model_claim_made": True,
    }
    ready = bool(all(gates.values()) and pool_all_k8 >= minimum_k8_tasks and pool_strict_k8 >= minimum_k8_tasks)
    if ready:
        decision = "READY_FOR_OFFICIAL_LABELING"
    elif int(generation_plan.get("additional_k8_tasks_needed_for_minimum", 0)) > 0:
        decision = "NEED_SELF_GENERATION"
    else:
        decision = "FREEZE_STAGE6_DATA_LIMITED"
    return {
        "benchmark": BENCHMARK,
        "created_at_utc": _now(),
        "seed": int(seed),
        "pool_all_k8_tasks": pool_all_k8,
        "pool_strict_k8_tasks": pool_strict_k8,
        "minimum_k8_tasks": int(minimum_k8_tasks),
        "pre_harness_gates": gates,
        "pre_harness_gates_pass": ready,
        "official_harness_should_run_now": ready,
        "slot_prediction_files_ready": ready,
        "source_or_compute_limit_documented": source_limit,
        "pool_all_duplicate_audit": duplicate_all,
        "pool_strict_duplicate_audit": duplicate_strict,
        "pool_all_order_audit": order_all,
        "pool_strict_order_audit": order_strict,
        "exact_reference_source_scan_audit": exact_reference_scan_audit,
        "decision": decision,
        "decision_meaning": {
            "READY_FOR_OFFICIAL_LABELING": "enough K=8 generated tasks exist and harness slot prediction files are ready",
            "NEED_SELF_GENERATION": "current public/local sources are insufficient; use the included self-generation plan",
            "FREEZE_STAGE6_DATA_LIMITED": "current source acquisition is insufficient and no selector training should be run",
        }[decision],
        "no_selector_training_executed": True,
        "no_final_swe_bench_improvement_claim": True,
    }


def prepare_harness_slot_predictions(
    pool_all: Sequence[PatchSelectionExample],
    slot_dir: Path,
    readiness: Dict[str, object],
    seed: int,
) -> None:
    slot_dir.mkdir(parents=True, exist_ok=True)
    for old in slot_dir.glob("slot_*.jsonl"):
        old.unlink()
    if not readiness.get("official_harness_should_run_now", False):
        _write_json(
            slot_dir / "NOT_READY.json",
            {
                "benchmark": BENCHMARK,
                "created_at_utc": _now(),
                "reason": "pre-harness gates did not pass; slot prediction files intentionally not emitted",
                "decision": readiness.get("decision"),
                "pool_all_k8_tasks": readiness.get("pool_all_k8_tasks"),
                "pool_strict_k8_tasks": readiness.get("pool_strict_k8_tasks"),
            },
        )
        return
    for slot in range(STAGE6_NUM_CANDIDATES):
        rows = [
            {
                "instance_id": example.issue_id,
                "model_name_or_path": f"stage6d3_slot_{slot}_generated_pool_seed_{seed}",
                "model_patch": example.candidates[slot].candidate_diff,
            }
            for example in pool_all
        ]
        _write_jsonl(slot_dir / f"slot_{slot}.jsonl", rows)


def render_report(inventory: Dict[str, object], generation_plan: Dict[str, object], readiness: Dict[str, object]) -> str:
    estimates = inventory.get("pool_estimates", {})
    gates = readiness.get("pre_harness_gates", {})
    lines = [
        "# Stage 6D.3 Candidate Source Expansion",
        "",
        "## Scope",
        "",
        "- Selector training was not executed.",
        "- Selector architecture was not changed.",
        "- Gold/reference patches, raw-preserving expansion, and fabricated candidates were not used.",
        "- Official labels and raw harness logs are not selector-visible.",
        "- No final SWE-bench improvement claim is made.",
        "",
        "## Source Inventory",
        "",
        f"- Source files scanned: `{inventory.get('source_files_scanned', 0)}`.",
        f"- Raw records loaded: `{inventory.get('raw_records_loaded', 0)}`.",
        f"- Unique source instance IDs: `{inventory.get('unique_source_instance_ids', 0)}`.",
        f"- Overlap with SWE-bench Verified: `{inventory.get('overlap_with_swebench_verified_task_ids', 0)}`.",
        f"- Tasks with >=1 candidate: `{inventory.get('tasks_with_at_least_1_candidate', 0)}`.",
        f"- Tasks with >=4 candidates: `{inventory.get('tasks_with_at_least_4_candidates', 0)}`.",
        f"- Tasks with >=8 candidates: `{inventory.get('tasks_with_at_least_8_candidates', 0)}`.",
        f"- Invalid diff exclusions: `{inventory.get('invalid_diff_exclusions', 0)}`.",
        f"- Duplicate exclusions: `{inventory.get('duplicate_exclusions', 0)}`.",
        f"- Exact reference exclusions: `{inventory.get('exact_reference_exclusions', 0)}`.",
        f"- Near-reference flags: `{inventory.get('near_reference_flags', 0)}`.",
        "",
        "## Pool Construction",
        "",
        f"- Pool all unlabeled K=8 tasks: `{estimates.get('pool_all_k8_tasks', 0)}`.",
        f"- Pool strict unlabeled K=8 tasks: `{estimates.get('pool_strict_k8_tasks', 0)}`.",
        f"- Strict quarantined task IDs: `{', '.join(estimates.get('strict_quarantined_task_ids', [])) or 'none'}`.",
        "",
        "## Self-Generation Plan",
        "",
        f"- Additional strict K=8 tasks needed for minimum: `{generation_plan.get('additional_k8_tasks_needed_for_minimum', 0)}`.",
        f"- Additional strict K=8 tasks needed for preferred: `{generation_plan.get('additional_k8_tasks_needed_for_preferred', 0)}`.",
        f"- Planned task count: `{generation_plan.get('planned_task_count', 0)}`.",
        f"- Estimated total generation attempts: `{generation_plan.get('estimated_total_generation_attempts', 0)}`.",
        "",
        "## Pre-Harness Gates",
        "",
        "| Gate | Pass |",
        "|---|---:|",
    ]
    for key, value in gates.items():
        lines.append(f"| `{key}` | `{bool(value)}` |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            str(readiness.get("decision", "FREEZE_STAGE6_DATA_LIMITED")),
            "",
        ]
    )
    return "\n".join(lines)


def read_records(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    try:
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        rows.append(value)
        elif path.suffix.lower() == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, list):
                rows.extend(item for item in value if isinstance(item, dict))
            elif isinstance(value, dict):
                rows.append(value)
    except Exception:
        return rows
    return rows


def extract_candidate_patches(record: Dict[str, object]) -> List[str]:
    patches = []
    for key in PATCH_KEYS:
        if key in record:
            patches.extend(_patch_candidates_from_value(record.get(key), explicit_patch_field=key not in {"messages", "trajectory", "response", "final_response"}))
    raw_candidates = record.get("candidates")
    if isinstance(raw_candidates, list):
        for candidate in raw_candidates:
            if isinstance(candidate, dict):
                patches.extend(extract_candidate_patches(candidate))
    return dedupe_texts(patches)


def dedupe_texts(values: Iterable[str]) -> List[str]:
    out = []
    seen = set()
    for value in values:
        patch = clean_patch(value)
        patch_hash = stable_patch_hash(patch)
        if patch_hash and patch_hash not in seen:
            seen.add(patch_hash)
            out.append(patch)
    return out


def normalize_instance_id(record: Dict[str, object], path: Path) -> str:
    for key in ("instance_id", "issue_id", "task_id"):
        value = record.get(key)
        if isinstance(value, str) and "__" in value:
            return value.strip()
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        for key in ("instance_id", "issue_id", "task_id"):
            value = metadata.get(key)
            if isinstance(value, str) and "__" in value:
                return value.strip()
    if "__" in path.stem:
        return path.stem
    return ""


def generator_name(record: Dict[str, object], path: Path) -> str:
    for key in ("generator_name", "model_name_or_path", "model", "source", "agent"):
        value = record.get(key)
        if value:
            return str(value)[:160]
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        for key in ("generator_name", "model_name_or_path", "model", "source", "agent"):
            value = metadata.get(key)
            if value:
                return str(value)[:160]
    return f"unknown:{path.name}"


def clean_patch(value: object) -> str:
    text = str(value or "").replace("\r\n", "\n")
    fenced = re.search(r"```(?:diff|patch)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced and ("diff --git " in fenced.group(1) or "--- " in fenced.group(1)):
        text = fenced.group(1)
    if "diff --git " in text:
        text = text[text.find("diff --git ") :]
    return text.strip() + ("\n" if text.strip() else "")


def patch_similarity(left: str, right: str) -> float:
    if not str(right or "").strip():
        return 0.0
    left_norm = "\n".join(line.rstrip() for line in str(left).splitlines() if not line.startswith("index "))
    right_norm = "\n".join(line.rstrip() for line in str(right).splitlines() if not line.startswith("index "))
    return float(SequenceMatcher(None, left_norm[:40_000], right_norm[:40_000]).ratio())


def attempt_to_record(attempt: GeneratedAttempt) -> Dict[str, object]:
    return {
        "instance_id": attempt.instance_id,
        "patch_hash": attempt.patch_hash,
        "source_file": attempt.source_file,
        "source_row_index": attempt.source_row_index,
        "patch_index": attempt.patch_index,
        "generator_name": attempt.generator_name,
        "terminal_status": attempt.terminal_status,
        "reason": attempt.reason,
        "near_reference": attempt.near_reference,
        "near_reference_similarity": attempt.near_reference_similarity,
    }
def _write_json(path: Path, value: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
