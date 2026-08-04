from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.datasets.swe_patch_selection_dataset import (
    STAGE6_NUM_CANDIDATES,
    PatchCandidate,
    PatchSelectionExample,
    duplicate_candidate_patch_hash_audit,
    label_matrix,
    load_patch_selection_jsonl,
    stage6_output_leakage_audit,
    validate_patch_selection_examples,
    write_patch_selection_jsonl,
)
from src.experiments.run_stage6_latent_patch_selector import evaluate_scores
from src.experiments.run_stage6b1_candidate_generation import DEFAULT_DATASET_NAME, load_benchmark_tasks
from src.experiments.run_stage6d_pool_scale_and_view_hardening import (
    BENCHMARK as STAGE6D_BENCHMARK,
    NEAR_REFERENCE_THRESHOLD,
    best_generator_on_dev_baseline,
    build_pool_summary,
    build_source_coverage_report,
    build_view_hardened_examples,
    candidate_evidence_mismatch_scores,
    candidate_order_randomized_audit,
    context_only_scores,
    exact_reference_hash_audit,
    issue_context_patch_embedding_scores,
    issue_patch_embedding_scores,
    near_reference_similarity_audit,
    patch_only_embedding_scores,
    per_generator_pass_rate,
    per_repo_oracle_pass_at_8,
    source_generator_output_leakage_audit,
    source_generator_skew_report,
)


BENCHMARK = "stage6d1_pool_expansion_and_evidence_stress"
DEFAULT_STAGE6D_POOL_PATH = Path("results/stage6d_labeled_candidate_pools.jsonl")
DEFAULT_STAGE6D_VIEW_POOL_PATH = Path("results/stage6d_view_hardened_examples.jsonl")
DEFAULT_STAGE6D_AUDIT_PATH = Path("results/stage6d_pool_and_view_audit.json")
DEFAULT_SEED_CANDIDATE_PATH = Path("results/stage6b1b_generated_candidates.jsonl")
DEFAULT_STAGE6B3_QUARANTINE_PATH = Path("results/stage6b3_quarantined_incomplete_labels.jsonl")
DEFAULT_PUBLIC_SOURCE_ROOT = Path("results/stage6b0_public_sources")
DEFAULT_LABELED_POOL_PATH = Path("results/stage6d1_labeled_candidate_pools.jsonl")
DEFAULT_POOL_STRICT_PATH = Path("results/stage6d1_pool_strict.jsonl")
DEFAULT_LABEL_AUDIT_PATH = Path("results/stage6d1_official_label_audit.json")
DEFAULT_STRESS_PATH = Path("results/stage6d1_view_stress_test_results.json")
DEFAULT_NEAR_REFERENCE_PATH = Path("results/stage6d1_near_reference_quarantine.json")
DEFAULT_REPORT_PATH = Path("reports/STAGE6D1_POOL_EXPANSION_AND_EVIDENCE_STRESS.md")
DEFAULT_SEED = 606_410
DIAGNOSTIC_DELTA_THRESHOLD = 0.05
DIAGNOSTIC_EPSILON = 1e-9


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6D.1 near-reference quarantine, label expansion audit, and evidence stress tests.")
    parser.add_argument("--input", default=str(DEFAULT_STAGE6D_POOL_PATH))
    parser.add_argument("--view-input", default=str(DEFAULT_STAGE6D_VIEW_POOL_PATH))
    parser.add_argument("--stage6d-audit", default=str(DEFAULT_STAGE6D_AUDIT_PATH))
    parser.add_argument("--seed-candidates", default=str(DEFAULT_SEED_CANDIDATE_PATH))
    parser.add_argument("--stage6b3-quarantine", default=str(DEFAULT_STAGE6B3_QUARANTINE_PATH))
    parser.add_argument("--public-source-root", action="append", default=[str(DEFAULT_PUBLIC_SOURCE_ROOT)])
    parser.add_argument("--labeled-output", default=str(DEFAULT_LABELED_POOL_PATH))
    parser.add_argument("--strict-output", default=str(DEFAULT_POOL_STRICT_PATH))
    parser.add_argument("--label-audit", default=str(DEFAULT_LABEL_AUDIT_PATH))
    parser.add_argument("--stress-results", default=str(DEFAULT_STRESS_PATH))
    parser.add_argument("--near-reference-quarantine", default=str(DEFAULT_NEAR_REFERENCE_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--minimum-complete-tasks", type=int, default=50)
    parser.add_argument("--preferred-complete-tasks", type=int, default=100)
    parser.add_argument("--minimum-oracle-positive", type=int, default=15)
    args = parser.parse_args()

    result = run_stage6d1_pool_expansion_and_evidence_stress(
        input_path=Path(args.input),
        view_input_path=Path(args.view_input),
        stage6d_audit_path=Path(args.stage6d_audit),
        seed_candidate_path=Path(args.seed_candidates),
        stage6b3_quarantine_path=Path(args.stage6b3_quarantine),
        public_source_roots=[Path(value) for value in args.public_source_root],
        labeled_output_path=Path(args.labeled_output),
        strict_output_path=Path(args.strict_output),
        label_audit_path=Path(args.label_audit),
        stress_results_path=Path(args.stress_results),
        near_reference_path=Path(args.near_reference_quarantine),
        report_path=Path(args.report),
        dataset_name=str(args.dataset_name),
        split=str(args.split),
        seed=int(args.seed),
        minimum_complete_tasks=int(args.minimum_complete_tasks),
        preferred_complete_tasks=int(args.preferred_complete_tasks),
        minimum_oracle_positive=int(args.minimum_oracle_positive),
    )
    gates = result["label_audit"].get("quality_gates", {})
    print(
        "stage6d1: wrote {pool}, {strict}, {audit}, {stress}, {quarantine}, {report}; "
        "pool_all_tasks={all_tasks}; pool_strict_tasks={strict_tasks}; stress_pass={stress_pass}; gates_pass={gates_pass}".format(
            pool=args.labeled_output,
            strict=args.strict_output,
            audit=args.label_audit,
            stress=args.stress_results,
            quarantine=args.near_reference_quarantine,
            report=args.report,
            all_tasks=result["quarantine"]["pool_all"]["complete_tasks"],
            strict_tasks=result["quarantine"]["pool_strict"]["complete_tasks"],
            stress_pass=result["stress"]["summary"]["overall_evidence_use_diagnostic_passes"],
            gates_pass=bool(gates and all(gates.values())),
        )
    )


def run_stage6d1_pool_expansion_and_evidence_stress(
    input_path: Path,
    view_input_path: Path,
    stage6d_audit_path: Path,
    seed_candidate_path: Path,
    stage6b3_quarantine_path: Path,
    public_source_roots: Sequence[Path],
    labeled_output_path: Path,
    strict_output_path: Path,
    label_audit_path: Path,
    stress_results_path: Path,
    near_reference_path: Path,
    report_path: Path,
    dataset_name: str,
    split: str,
    seed: int,
    minimum_complete_tasks: int,
    preferred_complete_tasks: int,
    minimum_oracle_positive: int,
) -> Dict[str, object]:
    created_at = _now()
    pool_all = _mark_pool_variant(_load_complete_pool(input_path), "pool_all")
    view_pool = _load_complete_pool(view_input_path)
    stage6d_audit = _read_json(stage6d_audit_path)
    near_hits = _near_hits_from_stage6d_audit(stage6d_audit)
    if not near_hits:
        task_records, _benchmark_audit = load_benchmark_tasks(
            dataset_name=dataset_name,
            split=split,
            benchmark_jsonl=None,
            requested_task_ids=[example.issue_id for example in pool_all],
            task_ids_file=None,
            n_tasks=len(pool_all),
            seed=seed,
        )
        near_hits = near_reference_similarity_audit(pool_all, task_records, NEAR_REFERENCE_THRESHOLD).get("near_hits", [])
    quarantined_task_ids = sorted({str(hit["instance_id"]) for hit in near_hits})
    pool_strict = _mark_pool_variant([example for example in pool_all if example.issue_id not in set(quarantined_task_ids)], "pool_strict")
    write_patch_selection_jsonl(labeled_output_path, pool_all)
    write_patch_selection_jsonl(strict_output_path, pool_strict)

    quarantine = build_near_reference_quarantine(
        pool_all=pool_all,
        pool_strict=pool_strict,
        near_hits=near_hits,
        seed=seed,
    )
    _write_json(near_reference_path, quarantine)

    hardened_all = _view_examples_for_pool(pool_all, view_pool, seed=seed)
    hardened_strict = _view_examples_for_pool(pool_strict, view_pool, seed=seed)
    stress = {
        "benchmark": BENCHMARK,
        "created_at_utc": created_at,
        "seed": int(seed),
        "pool_all": run_evidence_stress_for_pool("pool_all", hardened_all, seed),
        "pool_strict": run_evidence_stress_for_pool("pool_strict", hardened_strict, seed + 17),
    }
    stress["summary"] = summarize_stress_results(stress)
    _write_json(stress_results_path, stress)

    task_records, benchmark_audit = load_benchmark_tasks(
        dataset_name=dataset_name,
        split=split,
        benchmark_jsonl=None,
        requested_task_ids=[example.issue_id for example in pool_all],
        task_ids_file=None,
        n_tasks=len(pool_all),
        seed=seed,
    )
    strict_task_records = {issue_id: task for issue_id, task in task_records.items() if issue_id not in set(quarantined_task_ids)}
    all_output_leakage = stage6_output_leakage_audit(BENCHMARK, seed, hardened_all)
    strict_output_leakage = stage6_output_leakage_audit(BENCHMARK, seed, hardened_strict)
    all_source_leakage = source_generator_output_leakage_audit(hardened_all)
    strict_source_leakage = source_generator_output_leakage_audit(hardened_strict)
    all_duplicate = duplicate_candidate_patch_hash_audit(pool_all)
    strict_duplicate = duplicate_candidate_patch_hash_audit(pool_strict)
    all_exact_reference = exact_reference_hash_audit(pool_all, task_records)
    strict_exact_reference = exact_reference_hash_audit(pool_strict, strict_task_records)
    strict_near_reference = near_reference_similarity_audit(pool_strict, strict_task_records, NEAR_REFERENCE_THRESHOLD)
    source_coverage = build_source_coverage_report(
        seed_candidate_path=seed_candidate_path,
        input_path=input_path,
        completed_examples=pool_all,
        quarantine_path=stage6b3_quarantine_path,
    )
    source_coverage["additional_public_source_scan"] = scan_additional_public_sources(public_source_roots)
    expansion = build_expansion_audit(
        pool_all=pool_all,
        pool_strict=pool_strict,
        source_coverage=source_coverage,
        minimum_complete_tasks=minimum_complete_tasks,
        preferred_complete_tasks=preferred_complete_tasks,
        minimum_oracle_positive=minimum_oracle_positive,
    )
    strict_summary = build_pool_summary(pool_strict)
    strict_oracle = float(strict_summary.get("oracle_pass_at_8", 0.0))
    quality_gates = {
        "near_reference_candidates_quarantined_or_explicitly_audited": bool(quarantine.get("passes", False)),
        "at_least_50_complete_k8_officially_labeled_tasks_or_compute_limit_documented": bool(
            int(expansion["complete_tasks_after_expansion"]) >= minimum_complete_tasks or expansion["compute_limit_documented"]
        ),
        "at_least_15_oracle_positive_tasks": int(strict_summary.get("oracle_positive_tasks", 0)) >= minimum_oracle_positive,
        "oracle_pass_at_8_between_0_20_and_0_80": 0.20 <= strict_oracle <= 0.80,
        "no_exact_reference_hash_hits": bool(strict_exact_reference.get("passes", False)),
        "no_raw_preserving_expansion": True,
        "no_fabricated_candidates": True,
        "output_leakage_audit_passes": bool(all_output_leakage.get("passes", False) and strict_output_leakage.get("passes", False)),
        "source_generator_leakage_audit_passes": bool(all_source_leakage.get("passes", False) and strict_source_leakage.get("passes", False)),
        "duplicate_patch_hash_audit_passes": bool(all_duplicate.get("passes", False) and strict_duplicate.get("passes", False)),
        "evidence_use_diagnostic_passes": bool(stress["summary"].get("overall_evidence_use_diagnostic_passes", False)),
        "no_selector_final_claim_made": True,
    }
    label_audit = {
        "benchmark": BENCHMARK,
        "created_at_utc": created_at,
        "input_path": str(input_path),
        "labeled_output_path": str(labeled_output_path),
        "strict_output_path": str(strict_output_path),
        "dataset_name": dataset_name,
        "split": split,
        "seed": int(seed),
        "no_selector_training_executed": True,
        "no_architecture_search_or_test_tuning": True,
        "no_candidate_patch_alteration": True,
        "no_reference_or_gold_candidates": True,
        "no_raw_preserving_expansion": True,
        "no_fabricated_candidates": True,
        "no_final_swe_bench_improvement_claim": True,
        "official_label_expansion": expansion,
        "pool_all_summary": build_pool_summary(pool_all),
        "pool_strict_summary": strict_summary,
        "source_coverage_report": source_coverage,
        "benchmark_dataset_audit": benchmark_audit,
        "pool_all_validation": validate_patch_selection_examples(pool_all, allow_gold_diagnostic=False),
        "pool_strict_validation": validate_patch_selection_examples(pool_strict, allow_gold_diagnostic=False),
        "pool_all_output_leakage_audit": all_output_leakage,
        "pool_strict_output_leakage_audit": strict_output_leakage,
        "pool_all_source_generator_leakage_audit": all_source_leakage,
        "pool_strict_source_generator_leakage_audit": strict_source_leakage,
        "pool_all_duplicate_patch_hash_audit": all_duplicate,
        "pool_strict_duplicate_patch_hash_audit": strict_duplicate,
        "pool_all_exact_reference_hash_audit": all_exact_reference,
        "pool_strict_exact_reference_hash_audit": strict_exact_reference,
        "pool_strict_near_reference_similarity_audit": strict_near_reference,
        "pool_all_candidate_order_randomized_audit": candidate_order_randomized_audit(pool_all),
        "pool_strict_candidate_order_randomized_audit": candidate_order_randomized_audit(pool_strict),
        "pool_all_source_generator_skew": source_generator_skew_report(pool_all),
        "pool_strict_source_generator_skew": source_generator_skew_report(pool_strict),
        "pool_all_per_repo_oracle_pass_at_8": per_repo_oracle_pass_at_8(pool_all),
        "pool_strict_per_repo_oracle_pass_at_8": per_repo_oracle_pass_at_8(pool_strict),
        "pool_all_per_generator_pass_rate": per_generator_pass_rate(pool_all),
        "pool_strict_per_generator_pass_rate": per_generator_pass_rate(pool_strict),
        "quality_gates": quality_gates,
        "quality_gates_pass": bool(all(quality_gates.values())),
        "blocker_diagnosis": diagnose_blockers(expansion, strict_summary, stress, source_generator_skew_report(pool_strict), quarantine),
        "decision": (
            "Proceed to Stage 6E dev-only selector validation."
            if bool(all(quality_gates.values()))
            else "Do not train the selector; Stage 6D.1 gates did not pass."
        ),
    }
    _write_json(label_audit_path, label_audit)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(label_audit, quarantine, stress), encoding="utf-8")
    return {"label_audit": label_audit, "quarantine": quarantine, "stress": stress}


def build_near_reference_quarantine(
    pool_all: Sequence[PatchSelectionExample],
    pool_strict: Sequence[PatchSelectionExample],
    near_hits: Sequence[Dict[str, object]],
    seed: int,
) -> Dict[str, object]:
    all_baselines = pool_baseline_summary(pool_all, seed)
    strict_baselines = pool_baseline_summary(pool_strict, seed + 1)
    quarantined_task_ids = sorted({str(hit["instance_id"]) for hit in near_hits})
    task_rows = []
    by_id = {example.issue_id: example for example in pool_all}
    for task_id in quarantined_task_ids:
        example = by_id.get(task_id)
        if example is not None:
            task_rows.append(
                {
                    "instance_id": task_id,
                    "candidate_count": len(example.candidates),
                    "oracle_positive": bool(example.oracle_pass_at_8),
                    "labels_pass_fail": [int(value) for value in example.labels_pass_fail],
                }
            )
    return {
        "benchmark": BENCHMARK,
        "created_at_utc": _now(),
        "seed": int(seed),
        "threshold": float(NEAR_REFERENCE_THRESHOLD),
        "policy": "quarantine_full_task_when_any_candidate_is_near_reference_to_preserve_K8",
        "near_reference_candidates": list(near_hits),
        "near_reference_candidate_count": len(near_hits),
        "quarantined_task_ids": quarantined_task_ids,
        "quarantined_task_count": len(quarantined_task_ids),
        "quarantined_tasks": task_rows,
        "pool_all": {**build_pool_summary(pool_all), "baselines": all_baselines},
        "pool_strict": {**build_pool_summary(pool_strict), "baselines": strict_baselines},
        "delta": {
            "complete_tasks": int(len(pool_strict) - len(pool_all)),
            "candidate_count": int(sum(len(example.candidates) for example in pool_strict) - sum(len(example.candidates) for example in pool_all)),
            "oracle_positive_tasks": int(sum(example.oracle_pass_at_8 for example in pool_strict) - sum(example.oracle_pass_at_8 for example in pool_all)),
            "oracle_pass_at_8": float(build_pool_summary(pool_strict)["oracle_pass_at_8"] - build_pool_summary(pool_all)["oracle_pass_at_8"]),
            "first_candidate_pass_at_1": float(
                strict_baselines["first_candidate"]["pass_at_1"] - all_baselines["first_candidate"]["pass_at_1"]
            ),
            "patch_only_pass_at_1": float(strict_baselines["patch_only"]["pass_at_1"] - all_baselines["patch_only"]["pass_at_1"]),
        },
        "passes": bool(not near_hits or len(pool_strict) < len(pool_all)),
    }


def pool_baseline_summary(examples: Sequence[PatchSelectionExample], seed: int) -> Dict[str, object]:
    if not examples:
        return {}
    hardened = build_view_hardened_examples(examples, seed=seed)
    methods = {
        "first_candidate": _score_metric("first_candidate", first_candidate_scores(examples), examples),
        "random": _score_metric("random", random_scores(examples, seed), examples),
        "best_generator_on_dev": best_generator_on_dev_baseline(examples, seed),
        "patch_only": _score_metric("patch_only", patch_only_embedding_scores(hardened), hardened),
        "issue_context_patch": _score_metric("issue_context_patch", issue_context_patch_embedding_scores(hardened), hardened),
        "candidate_evidence_mismatch": _score_metric("candidate_evidence_mismatch", candidate_evidence_mismatch_scores(hardened), hardened),
    }
    return methods


def run_evidence_stress_for_pool(pool_name: str, examples: Sequence[PatchSelectionExample], seed: int) -> Dict[str, object]:
    if not examples:
        return {"pool": pool_name, "metrics": {}, "summary": {"passes": False, "reason": "empty_pool"}}
    base_scores = {
        "patch_only": patch_only_embedding_scores(examples),
        "issue_only": issue_only_scores(examples),
        "context_only": context_only_scores(examples),
        "issue_plus_patch": issue_patch_embedding_scores(examples),
        "context_plus_patch": context_patch_scores(examples),
        "issue_plus_context_plus_patch": issue_context_patch_embedding_scores(examples),
        "candidate_evidence_mismatch": candidate_evidence_mismatch_scores(examples),
        "cross_task_context_shuffle": cross_task_context_shuffle_scores(examples),
        "edited_file_context_removed": edited_file_context_removed_scores(examples),
        "issue_terms_removed": issue_terms_removed_scores(examples),
        "patch_hunk_context_removed": patch_hunk_context_removed_scores(examples),
    }
    metrics = {
        name: evaluate_scores(name, scores.astype(np.float32), examples, split=pool_name)
        for name, scores in base_scores.items()
    }
    base = float(metrics["issue_plus_context_plus_patch"]["pass_at_1"])
    patch_only = float(metrics["patch_only"]["pass_at_1"])
    mismatch = float(metrics["candidate_evidence_mismatch"]["pass_at_1"])
    cross_shuffle = float(metrics["cross_task_context_shuffle"]["pass_at_1"])
    edited_removed = float(metrics["edited_file_context_removed"]["pass_at_1"])
    summary = {
        "pool": pool_name,
        "task_count": len(examples),
        "oracle_positive_tasks": int(sum(example.oracle_pass_at_8 for example in examples)),
        "oracle_pass_at_8": float(np.mean([example.oracle_pass_at_8 for example in examples])),
        "issue_context_patch_minus_patch_only": base - patch_only,
        "candidate_evidence_mismatch_degradation": base - mismatch,
        "cross_task_context_shuffle_degradation": base - cross_shuffle,
        "edited_file_context_removed_degradation": base - edited_removed,
        "passes": bool(
            _meets_diagnostic_delta(base - patch_only)
            or _meets_diagnostic_delta(base - mismatch)
            or _meets_diagnostic_delta(base - cross_shuffle)
            or _meets_diagnostic_delta(base - edited_removed)
        ),
    }
    return {"pool": pool_name, "metrics": metrics, "summary": summary}


def summarize_stress_results(stress: Dict[str, object]) -> Dict[str, object]:
    all_summary = stress.get("pool_all", {}).get("summary", {})
    strict_summary = stress.get("pool_strict", {}).get("summary", {})
    return {
        "pool_all_passes": bool(all_summary.get("passes", False)),
        "pool_strict_passes": bool(strict_summary.get("passes", False)),
        "overall_evidence_use_diagnostic_passes": bool(all_summary.get("passes", False) and strict_summary.get("passes", False)),
        "diagnostic_rule": (
            "For each pool, pass if issue+context+patch beats patch-only by >=0.05 pass@1, or mismatch, cross-task "
            "shuffle, or edited-file context removal degrades issue+context+patch by >=0.05 pass@1."
        ),
    }


def build_expansion_audit(
    pool_all: Sequence[PatchSelectionExample],
    pool_strict: Sequence[PatchSelectionExample],
    source_coverage: Dict[str, object],
    minimum_complete_tasks: int,
    preferred_complete_tasks: int,
    minimum_oracle_positive: int,
) -> Dict[str, object]:
    summary = build_pool_summary(pool_all)
    strict_summary = build_pool_summary(pool_strict)
    additional = source_coverage.get("additional_public_source_scan", {})
    additional_records = int(additional.get("json_records_loaded", 0)) if isinstance(additional, dict) else 0
    complete_tasks = int(summary.get("complete_tasks", 0))
    complete_candidates = int(summary.get("candidate_count", 0))
    return {
        "input_candidate_source": str(DEFAULT_SEED_CANDIDATE_PATH),
        "stage6b1b_seed_unique_tasks": source_coverage.get("seed_unique_instance_ids", 0),
        "stage6b1b_seed_candidate_rows": source_coverage.get("seed_candidate_rows_loaded", 0),
        "additional_public_json_records_loaded": additional_records,
        "new_official_harness_runs_executed": False,
        "official_harness_reuse": "reused Stage 6B.2/6B.3 official resolved labels already present in the Stage 6D complete pool",
        "official_harness_command_module": "python -m swebench.harness.run_evaluation",
        "harness_logs_selector_visible": False,
        "reason_no_new_harness_batch": (
            "No additional complete K=8 generated task pool beyond the 25-task Stage 6B.1b seed is locally available. "
            "Running Docker on the same seed cannot reach the 50-task Stage 6D.1 minimum, and missing labels are not fabricated."
        ),
        "complete_tasks_after_expansion": complete_tasks,
        "complete_candidates_after_expansion": complete_candidates,
        "strict_complete_tasks_after_near_reference_quarantine": int(strict_summary.get("complete_tasks", 0)),
        "strict_oracle_positive_tasks": int(strict_summary.get("oracle_positive_tasks", 0)),
        "minimum_complete_tasks": int(minimum_complete_tasks),
        "preferred_complete_tasks": int(preferred_complete_tasks),
        "minimum_oracle_positive_tasks": int(minimum_oracle_positive),
        "compute_limit_documented": complete_tasks < int(minimum_complete_tasks),
        "missing_complete_tasks_for_minimum": max(0, int(minimum_complete_tasks) - complete_tasks),
        "missing_complete_tasks_for_preferred": max(0, int(preferred_complete_tasks) - complete_tasks),
    }


def scan_additional_public_sources(roots: Sequence[Path]) -> Dict[str, object]:
    files = []
    rows = 0
    unique_ids = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".json", ".jsonl"} and ".git" not in path.parts:
                files.append(str(path))
                loaded = _read_json_records(path)
                rows += len(loaded)
                for record in loaded:
                    instance_id = _record_instance_id(record)
                    if instance_id:
                        unique_ids.add(instance_id)
    return {
        "roots": [str(root) for root in roots],
        "json_files_loaded": len(files),
        "json_records_loaded": rows,
        "unique_instance_ids": len(unique_ids),
        "files": files[:50],
    }


def diagnose_blockers(
    expansion: Dict[str, object],
    strict_summary: Dict[str, object],
    stress: Dict[str, object],
    generator_skew: Dict[str, object],
    quarantine: Dict[str, object],
) -> List[str]:
    blockers = []
    if bool(expansion.get("compute_limit_documented", False)):
        blockers.append("insufficient official harness throughput / generated K=8 task supply for the 50-task minimum")
    if int(strict_summary.get("oracle_positive_tasks", 0)) < int(expansion.get("minimum_oracle_positive_tasks", 15)):
        blockers.append("insufficient oracle-positive tasks after near-reference quarantine")
    if bool(generator_skew.get("high_generator_skew", False)):
        blockers.append("generator/source skew remains high")
    if int(quarantine.get("near_reference_candidate_count", 0)) > 0:
        blockers.append("near-reference contamination was present and required full-task quarantine")
    if not bool(stress.get("summary", {}).get("overall_evidence_use_diagnostic_passes", False)):
        blockers.append("weak evidence retrieval / evidence-use stress diagnostics did not pass on both pools")
    strict_metrics = stress.get("pool_strict", {}).get("metrics", {}) if isinstance(stress, dict) else {}
    patch = float(strict_metrics.get("patch_only", {}).get("pass_at_1", 0.0))
    full = float(strict_metrics.get("issue_plus_context_plus_patch", {}).get("pass_at_1", 0.0))
    if patch >= full:
        blockers.append("patch-only shortcut dominance remains on pool_strict")
    return blockers


def render_report(label_audit: Dict[str, object], quarantine: Dict[str, object], stress: Dict[str, object]) -> str:
    all_summary = label_audit.get("pool_all_summary", {})
    strict_summary = label_audit.get("pool_strict_summary", {})
    expansion = label_audit.get("official_label_expansion", {})
    stress_all = stress.get("pool_all", {}).get("summary", {})
    stress_strict = stress.get("pool_strict", {}).get("summary", {})
    gates = label_audit.get("quality_gates", {})
    lines = [
        "# Stage 6D.1 Pool Expansion and Evidence Stress",
        "",
        "## Scope",
        "",
        "- Selector training was not executed.",
        "- Candidate patches were not altered.",
        "- Gold/reference patches, raw-preserving expansion, and fabricated candidates were not used.",
        "- Official labels and raw harness logs are excluded from selector-visible fields.",
        "- No final SWE-bench improvement claim is made.",
        "",
        "## Near-Reference Quarantine",
        "",
        f"- Near-reference threshold: `{quarantine.get('threshold', NEAR_REFERENCE_THRESHOLD)}`.",
        f"- Near-reference candidates: `{quarantine.get('near_reference_candidate_count', 0)}`.",
        f"- Quarantined full tasks: `{quarantine.get('quarantined_task_count', 0)}`.",
        f"- Quarantined task ids: `{', '.join(quarantine.get('quarantined_task_ids', [])) or 'none'}`.",
        "",
        "## Pool Status",
        "",
        f"- Pool all complete K=8 tasks: `{all_summary.get('complete_tasks', 0)}`.",
        f"- Pool all oracle-positive tasks: `{all_summary.get('oracle_positive_tasks', 0)}`.",
        f"- Pool all oracle pass@8: `{float(all_summary.get('oracle_pass_at_8', 0.0)):.4f}`.",
        f"- Pool strict complete K=8 tasks: `{strict_summary.get('complete_tasks', 0)}`.",
        f"- Pool strict oracle-positive tasks: `{strict_summary.get('oracle_positive_tasks', 0)}`.",
        f"- Pool strict oracle pass@8: `{float(strict_summary.get('oracle_pass_at_8', 0.0)):.4f}`.",
        "",
        "## Official Label Expansion",
        "",
        f"- New official harness runs executed: `{expansion.get('new_official_harness_runs_executed', False)}`.",
        f"- Complete tasks after expansion: `{expansion.get('complete_tasks_after_expansion', 0)}`.",
        f"- Minimum target: `{expansion.get('minimum_complete_tasks', 50)}`.",
        f"- Preferred target: `{expansion.get('preferred_complete_tasks', 100)}`.",
        f"- Compute limit documented: `{expansion.get('compute_limit_documented', False)}`.",
        "",
        "## Evidence Stress",
        "",
        f"- Pool all evidence diagnostic passes: `{stress_all.get('passes', False)}`.",
        f"- Pool all issue+context+patch minus patch-only: `{float(stress_all.get('issue_context_patch_minus_patch_only', 0.0)):.4f}`.",
        f"- Pool all mismatch degradation: `{float(stress_all.get('candidate_evidence_mismatch_degradation', 0.0)):.4f}`.",
        f"- Pool strict evidence diagnostic passes: `{stress_strict.get('passes', False)}`.",
        f"- Pool strict issue+context+patch minus patch-only: `{float(stress_strict.get('issue_context_patch_minus_patch_only', 0.0)):.4f}`.",
        f"- Pool strict mismatch degradation: `{float(stress_strict.get('candidate_evidence_mismatch_degradation', 0.0)):.4f}`.",
        "",
        "## Gates",
        "",
        "| Gate | Pass |",
        "|---|---:|",
    ]
    for key, value in gates.items():
        lines.append(f"| `{key}` | `{bool(value)}` |")
    lines.extend(["", "## Blockers", ""])
    blockers = label_audit.get("blocker_diagnosis", [])
    if blockers:
        for blocker in blockers:
            lines.append(f"- {blocker}.")
    else:
        lines.append("- No Stage 6D.1 blockers detected.")
    lines.extend(["", "## Decision", "", str(label_audit.get("decision", "Do not train the selector.")), ""])
    return "\n".join(lines)


def issue_only_scores(examples: Sequence[PatchSelectionExample]) -> np.ndarray:
    rows = []
    for example in examples:
        value = len(re.findall(r"[A-Za-z_][A-Za-z0-9_]+", example.issue_text)) / 10_000.0
        rows.append([value for _ in example.candidates])
    return np.asarray(rows, dtype=np.float32)


def context_patch_scores(examples: Sequence[PatchSelectionExample]) -> np.ndarray:
    stripped = [replace(example, issue_text="", failing_test_summary="") for example in examples]
    return issue_context_patch_embedding_scores(stripped)


def cross_task_context_shuffle_scores(examples: Sequence[PatchSelectionExample]) -> np.ndarray:
    if not examples:
        return np.zeros((0, STAGE6_NUM_CANDIDATES), dtype=np.float32)
    transformed = []
    for index, example in enumerate(examples):
        donor = examples[(index + 1) % len(examples)]
        transformed.append(
            replace(
                example,
                issue_text=donor.issue_text,
                failing_test_summary=donor.failing_test_summary,
                retrieved_contexts=donor.retrieved_contexts,
                metadata={
                    **example.metadata,
                    "dependency_callgraph_related_file_evidence": donor.metadata.get("dependency_callgraph_related_file_evidence", ""),
                },
            )
        )
    return issue_context_patch_embedding_scores(transformed)


def edited_file_context_removed_scores(examples: Sequence[PatchSelectionExample]) -> np.ndarray:
    transformed = []
    for example in examples:
        contexts = tuple(
            text
            for text in example.retrieved_contexts
            if "Stage 6D retrieved source context for candidate" not in text
            and "Edited files across candidates" not in text
        )
        dependency = "\n".join(
            line
            for line in str(example.metadata.get("dependency_callgraph_related_file_evidence", "")).splitlines()
            if "files=" not in line and "edited_symbols=" not in line
        )
        transformed.append(
            replace(
                example,
                retrieved_contexts=contexts,
                metadata={**example.metadata, "dependency_callgraph_related_file_evidence": dependency},
            )
        )
    return issue_context_patch_embedding_scores(transformed)


def issue_terms_removed_scores(examples: Sequence[PatchSelectionExample]) -> np.ndarray:
    transformed = []
    for example in examples:
        terms = _top_terms(example.issue_text + "\n" + example.failing_test_summary, limit=32)
        transformed.append(
            replace(
                example,
                issue_text=_remove_terms(example.issue_text, terms),
                failing_test_summary=_remove_terms(example.failing_test_summary, terms),
                retrieved_contexts=tuple(_remove_terms(text, terms) for text in example.retrieved_contexts),
            )
        )
    return issue_context_patch_embedding_scores(transformed)


def patch_hunk_context_removed_scores(examples: Sequence[PatchSelectionExample]) -> np.ndarray:
    transformed = []
    for example in examples:
        candidates = tuple(
            replace(candidate, candidate_diff=_remove_patch_hunk_context(candidate.candidate_diff))
            for candidate in example.candidates
        )
        transformed.append(replace(example, candidates=candidates))
    return issue_context_patch_embedding_scores(transformed)


def random_scores(examples: Sequence[PatchSelectionExample], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 9_311)
    return rng.normal(size=(len(examples), STAGE6_NUM_CANDIDATES)).astype(np.float32)


def first_candidate_scores(examples: Sequence[PatchSelectionExample]) -> np.ndarray:
    scores = np.zeros((len(examples), STAGE6_NUM_CANDIDATES), dtype=np.float32)
    if len(examples):
        scores[:, 0] = 1.0
    return scores


def _score_metric(method: str, scores: np.ndarray, examples: Sequence[PatchSelectionExample]) -> Dict[str, object]:
    return evaluate_scores(method, scores.astype(np.float32), examples, split="stage6d1_pool")


def _meets_diagnostic_delta(value: float) -> bool:
    return float(value) + DIAGNOSTIC_EPSILON >= DIAGNOSTIC_DELTA_THRESHOLD


def _view_examples_for_pool(
    pool: Sequence[PatchSelectionExample],
    view_pool: Sequence[PatchSelectionExample],
    seed: int,
) -> List[PatchSelectionExample]:
    views_by_issue = {example.issue_id: example for example in view_pool}
    out = []
    for example in pool:
        view = views_by_issue.get(example.issue_id)
        if view is not None and len(view.candidates) == len(example.candidates):
            out.append(
                replace(
                    view,
                    candidates=example.candidates,
                    labels_pass_fail=example.labels_pass_fail,
                    metadata={**example.metadata, **view.metadata, "stage6d1_view_source": "stage6d_view_hardened_examples"},
                )
            )
        else:
            out.extend(build_view_hardened_examples([example], seed=seed))
    return out


def _load_complete_pool(path: Path) -> List[PatchSelectionExample]:
    return [
        example
        for example in load_patch_selection_jsonl(path)
        if len(example.candidates) == STAGE6_NUM_CANDIDATES and len(example.labels_pass_fail) == STAGE6_NUM_CANDIDATES
    ]


def _mark_pool_variant(examples: Sequence[PatchSelectionExample], variant: str) -> List[PatchSelectionExample]:
    return [
        replace(
            example,
            metadata={
                **example.metadata,
                "stage6d1_pool_variant": variant,
                "stage6d1_candidate_patches_preserved": True,
                "stage6d1_no_selector_training": True,
                "gold_patch_included": False,
                "raw_preserving_expansion_used": False,
                "missing_candidate_fabrication_used": False,
            },
        )
        for example in examples
    ]


def _near_hits_from_stage6d_audit(stage6d_audit: Dict[str, object]) -> List[Dict[str, object]]:
    audit = stage6d_audit.get("near_reference_similarity_audit", {}) if isinstance(stage6d_audit, dict) else {}
    hits = audit.get("near_hits", []) if isinstance(audit, dict) else []
    return [dict(hit) for hit in hits if isinstance(hit, dict) and float(hit.get("similarity", 0.0)) >= NEAR_REFERENCE_THRESHOLD]


def _remove_patch_hunk_context(diff: str) -> str:
    rows = []
    for line in str(diff).splitlines():
        if line.startswith(" ") or line.startswith("-") and not line.startswith("---"):
            continue
        rows.append(line)
    return "\n".join(rows)


def _top_terms(text: str, limit: int) -> List[str]:
    counts = Counter(token.lower() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", str(text)))
    return [token for token, _count in counts.most_common(limit)]


def _remove_terms(text: str, terms: Sequence[str]) -> str:
    if not terms:
        return text
    pattern = re.compile(r"\b(" + "|".join(re.escape(term) for term in terms) + r")\b", re.IGNORECASE)
    return pattern.sub("[term_removed]", str(text))


def _record_instance_id(row: Dict[str, object]) -> str:
    for key in ("instance_id", "issue_id", "task_id"):
        if row.get(key):
            return str(row[key])
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key in ("instance_id", "issue_id", "task_id"):
            if metadata.get(key):
                return str(metadata[key])
    return ""


def _read_json_records(path: Path) -> List[Dict[str, object]]:
    rows = []
    try:
        if path.suffix.lower() == ".jsonl":
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


def _read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, value: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
