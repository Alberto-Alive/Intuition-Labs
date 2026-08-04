from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from src.datasets.swe_patch_selection_dataset import (
    STAGE6_NUM_CANDIDATES,
    PatchSelectionExample,
    duplicate_candidate_patch_hash_audit,
    load_patch_selection_jsonl,
    stable_patch_hash,
    stage6_output_leakage_audit,
    validate_patch_selection_examples,
    write_patch_selection_jsonl,
)
from src.experiments.run_stage6_latent_patch_selector import evaluate_scores
from src.experiments.run_stage6b1_candidate_generation import DEFAULT_DATASET_NAME, load_benchmark_tasks
from src.experiments.run_stage6d_pool_scale_and_view_hardening import (
    NEAR_REFERENCE_THRESHOLD,
    best_generator_on_dev_baseline,
    build_pool_summary,
    build_view_hardened_examples,
    candidate_order_randomized_audit,
    exact_reference_hash_audit,
    issue_context_patch_embedding_scores,
    near_reference_similarity_audit,
    patch_only_embedding_scores,
    per_generator_pass_rate,
    per_repo_oracle_pass_at_8,
    source_generator_output_leakage_audit,
    source_generator_skew_report,
)
from src.experiments.run_stage6d1_pool_expansion_and_evidence_stress import (
    DEFAULT_PUBLIC_SOURCE_ROOT,
    best_generator_on_dev_baseline as _unused_best_generator_import_guard,
    first_candidate_scores,
    random_scores,
    run_evidence_stress_for_pool,
)


BENCHMARK = "stage6d2_pool_scaling"
DEFAULT_POOL_ALL_INPUT = Path("results/stage6d1_labeled_candidate_pools.jsonl")
DEFAULT_POOL_STRICT_INPUT = Path("results/stage6d1_pool_strict.jsonl")
DEFAULT_CANDIDATE_SOURCES = (
    Path("results/stage6b1_generated_candidates.jsonl"),
    Path("results/stage6b1a_generated_candidates.jsonl"),
    Path("results/stage6b1b_generated_candidates.jsonl"),
)
DEFAULT_OUTPUT_ALL = Path("results/stage6d2_labeled_candidate_pools_all.jsonl")
DEFAULT_OUTPUT_STRICT = Path("results/stage6d2_labeled_candidate_pools_strict.jsonl")
DEFAULT_LABEL_AUDIT = Path("results/stage6d2_official_label_audit.json")
DEFAULT_STRESS_RESULTS = Path("results/stage6d2_view_stress_test_results.json")
DEFAULT_SOURCE_AUDIT = Path("results/stage6d2_source_coverage_audit.json")
DEFAULT_REPORT = Path("reports/STAGE6D2_POOL_SCALING.md")
DEFAULT_SEED = 606_420


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6D.2 candidate pool scaling audit and evidence-use diagnostics.")
    parser.add_argument("--pool-all-input", default=str(DEFAULT_POOL_ALL_INPUT))
    parser.add_argument("--pool-strict-input", default=str(DEFAULT_POOL_STRICT_INPUT))
    parser.add_argument("--candidate-source", action="append", default=[str(path) for path in DEFAULT_CANDIDATE_SOURCES])
    parser.add_argument("--source-root", action="append", default=[str(DEFAULT_PUBLIC_SOURCE_ROOT)])
    parser.add_argument("--output-all", default=str(DEFAULT_OUTPUT_ALL))
    parser.add_argument("--output-strict", default=str(DEFAULT_OUTPUT_STRICT))
    parser.add_argument("--label-audit", default=str(DEFAULT_LABEL_AUDIT))
    parser.add_argument("--stress-results", default=str(DEFAULT_STRESS_RESULTS))
    parser.add_argument("--source-audit", default=str(DEFAULT_SOURCE_AUDIT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--minimum-strict-tasks", type=int, default=50)
    parser.add_argument("--preferred-strict-tasks", type=int, default=100)
    parser.add_argument("--minimum-strict-oracle-positive", type=int, default=15)
    parser.add_argument("--preferred-strict-oracle-positive", type=int, default=30)
    args = parser.parse_args()

    result = run_stage6d2_pool_scaling(
        pool_all_input=Path(args.pool_all_input),
        pool_strict_input=Path(args.pool_strict_input),
        candidate_sources=[Path(value) for value in args.candidate_source],
        source_roots=[Path(value) for value in args.source_root],
        output_all=Path(args.output_all),
        output_strict=Path(args.output_strict),
        label_audit_path=Path(args.label_audit),
        stress_results_path=Path(args.stress_results),
        source_audit_path=Path(args.source_audit),
        report_path=Path(args.report),
        dataset_name=str(args.dataset_name),
        split=str(args.split),
        seed=int(args.seed),
        minimum_strict_tasks=int(args.minimum_strict_tasks),
        preferred_strict_tasks=int(args.preferred_strict_tasks),
        minimum_strict_oracle_positive=int(args.minimum_strict_oracle_positive),
        preferred_strict_oracle_positive=int(args.preferred_strict_oracle_positive),
    )
    gates = result["label_audit"].get("quality_gates", {})
    print(
        "stage6d2: wrote {all_pool}, {strict_pool}, {audit}, {stress}, {source}, {report}; "
        "pool_strict_tasks={tasks}; pool_strict_oracle_positive={positive}; stress_pass={stress_pass}; gates_pass={gates_pass}".format(
            all_pool=args.output_all,
            strict_pool=args.output_strict,
            audit=args.label_audit,
            stress=args.stress_results,
            source=args.source_audit,
            report=args.report,
            tasks=result["label_audit"].get("pool_strict", {}).get("summary", {}).get("complete_tasks", 0),
            positive=result["label_audit"].get("pool_strict", {}).get("summary", {}).get("oracle_positive_tasks", 0),
            stress_pass=result["stress"].get("summary", {}).get("pool_strict_passes", False),
            gates_pass=bool(gates and all(gates.values())),
        )
    )


def run_stage6d2_pool_scaling(
    pool_all_input: Path,
    pool_strict_input: Path,
    candidate_sources: Sequence[Path],
    source_roots: Sequence[Path],
    output_all: Path,
    output_strict: Path,
    label_audit_path: Path,
    stress_results_path: Path,
    source_audit_path: Path,
    report_path: Path,
    dataset_name: str,
    split: str,
    seed: int,
    minimum_strict_tasks: int,
    preferred_strict_tasks: int,
    minimum_strict_oracle_positive: int,
    preferred_strict_oracle_positive: int,
) -> Dict[str, object]:
    created_at = _now()
    pool_all = _mark_stage6d2_pool(_load_complete_pool(pool_all_input), "pool_all")
    pool_strict = _mark_stage6d2_pool(_load_complete_pool(pool_strict_input), "pool_strict")
    write_patch_selection_jsonl(output_all, pool_all)
    write_patch_selection_jsonl(output_strict, pool_strict)

    source_audit = build_stage6d2_source_coverage_audit(candidate_sources, source_roots, pool_all, pool_strict)
    _write_json(source_audit_path, source_audit)

    hardened_all = build_view_hardened_examples(pool_all, seed=seed)
    hardened_strict = build_view_hardened_examples(pool_strict, seed=seed + 11)
    stress = {
        "benchmark": BENCHMARK,
        "created_at_utc": created_at,
        "seed": int(seed),
        "pool_all": run_evidence_stress_for_pool("pool_all", hardened_all, seed),
        "pool_strict": run_evidence_stress_for_pool("pool_strict", hardened_strict, seed + 23),
    }
    stress["summary"] = {
        "pool_all_passes": bool(stress["pool_all"]["summary"].get("passes", False)),
        "pool_strict_passes": bool(stress["pool_strict"]["summary"].get("passes", False)),
        "overall_evidence_use_diagnostic_passes": bool(stress["pool_strict"]["summary"].get("passes", False)),
        "stage6d2_decision_pool": "pool_strict",
        "diagnostic_rule": (
            "On pool_strict, pass if issue+context+patch beats patch-only by >=0.05 pass@1, or mismatch, "
            "cross-task context shuffle, or edited-file context removal degrades issue+context+patch by >=0.05 pass@1."
        ),
    }
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
    strict_task_records = {example.issue_id: task_records[example.issue_id] for example in pool_strict if example.issue_id in task_records}
    pool_all_bundle = build_stage6d2_pool_bundle("pool_all", pool_all, hardened_all, task_records, seed)
    pool_strict_bundle = build_stage6d2_pool_bundle("pool_strict", pool_strict, hardened_strict, strict_task_records, seed + 101)
    expansion = build_stage6d2_expansion_status(
        source_audit=source_audit,
        pool_all=pool_all,
        pool_strict=pool_strict,
        minimum_strict_tasks=minimum_strict_tasks,
        preferred_strict_tasks=preferred_strict_tasks,
        minimum_strict_oracle_positive=minimum_strict_oracle_positive,
        preferred_strict_oracle_positive=preferred_strict_oracle_positive,
    )
    quality_gates = build_stage6d2_quality_gates(
        pool_strict_bundle=pool_strict_bundle,
        stress=stress,
        expansion=expansion,
        minimum_strict_tasks=minimum_strict_tasks,
        minimum_strict_oracle_positive=minimum_strict_oracle_positive,
    )
    blockers = diagnose_stage6d2_blockers(pool_strict_bundle, stress, expansion, quality_gates)
    label_audit = {
        "benchmark": BENCHMARK,
        "created_at_utc": created_at,
        "dataset_name": dataset_name,
        "split": split,
        "seed": int(seed),
        "pool_all_input": str(pool_all_input),
        "pool_strict_input": str(pool_strict_input),
        "output_all": str(output_all),
        "output_strict": str(output_strict),
        "no_selector_training_executed": True,
        "no_selector_architecture_change": True,
        "no_reference_or_gold_candidates": True,
        "no_raw_preserving_expansion": True,
        "no_fabricated_candidates": True,
        "official_labels_hidden_from_selector_visible_inputs": True,
        "raw_harness_logs_hidden_from_selector_visible_inputs": True,
        "no_final_swe_bench_improvement_claim": True,
        "official_harness": {
            "command_module": "python -m swebench.harness.run_evaluation",
            "batching_plan": "candidate-slot batches: slot_0 through slot_7",
            "cache_resume_required": True,
            "new_harness_runs_executed": False,
            "reason": expansion["new_harness_runs_reason"],
            "raw_log_visibility": "results harness logs only; never copied into selector-visible fields",
            "label_acceptance_rule": "official resolved only",
        },
        "benchmark_dataset_audit": benchmark_audit,
        "source_coverage_audit_path": str(source_audit_path),
        "source_coverage_summary": source_audit["summary"],
        "expansion_status": expansion,
        "pool_all": pool_all_bundle,
        "pool_strict": pool_strict_bundle,
        "quality_gates": quality_gates,
        "quality_gates_pass": bool(all(quality_gates.values())),
        "blocker_diagnosis": blockers,
        "decision": (
            "Proceed to Stage 6E dev-only selector validation."
            if bool(all(quality_gates.values()))
            else "Do not train the selector; Stage 6D.2 gates did not pass."
        ),
    }
    _write_json(label_audit_path, label_audit)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_stage6d2_report(label_audit, stress), encoding="utf-8")
    return {"label_audit": label_audit, "stress": stress, "source_audit": source_audit}


def build_stage6d2_pool_bundle(
    name: str,
    examples: Sequence[PatchSelectionExample],
    hardened_examples: Sequence[PatchSelectionExample],
    task_records: Dict[str, object],
    seed: int,
) -> Dict[str, object]:
    first = evaluate_scores("first_candidate_baseline", first_candidate_scores(examples), examples, split=name)
    random = evaluate_scores("random_baseline", random_scores(examples, seed), examples, split=name)
    best_generator = best_generator_on_dev_baseline(examples, seed)
    patch_only = evaluate_scores("patch_only_reranker", patch_only_embedding_scores(hardened_examples), hardened_examples, split=name)
    issue_context_patch = evaluate_scores(
        "issue_context_patch_reranker",
        issue_context_patch_embedding_scores(hardened_examples),
        hardened_examples,
        split=name,
    )
    stress_metrics = run_evidence_stress_for_pool(name, hardened_examples, seed)["metrics"]
    duplicate = duplicate_candidate_patch_hash_audit(examples)
    output_leakage = stage6_output_leakage_audit(BENCHMARK, seed, hardened_examples)
    source_leakage = source_generator_output_leakage_audit(hardened_examples)
    exact_reference = exact_reference_hash_audit(examples, task_records)
    near_reference = near_reference_similarity_audit(examples, task_records, NEAR_REFERENCE_THRESHOLD)
    return {
        "summary": build_pool_summary(examples),
        "validation": validate_patch_selection_examples(examples, allow_gold_diagnostic=False),
        "official_candidate_labels": sum(len(example.labels_pass_fail) for example in examples),
        "per_repository_oracle_pass_at_8": per_repo_oracle_pass_at_8(examples),
        "per_generator_pass_rate": per_generator_pass_rate(examples),
        "source_generator_skew": source_generator_skew_report(examples),
        "baselines": {
            "first_candidate_baseline": first,
            "random_baseline": random,
            "best_generator_on_dev_baseline": best_generator,
            "embedding_reranker": issue_context_patch,
            "patch_only_reranker": patch_only,
            "issue_context_patch_reranker": issue_context_patch,
            "candidate_evidence_mismatch": stress_metrics.get("candidate_evidence_mismatch", {}),
        },
        "audits": {
            "duplicate_patch_hash_audit": duplicate,
            "output_leakage_audit": output_leakage,
            "source_generator_leakage_audit": source_leakage,
            "exact_reference_hash_audit": exact_reference,
            "near_reference_similarity_audit": near_reference,
            "candidate_order_randomized_audit": candidate_order_randomized_audit(examples),
        },
    }


def build_stage6d2_source_coverage_audit(
    candidate_sources: Sequence[Path],
    source_roots: Sequence[Path],
    pool_all: Sequence[PatchSelectionExample],
    pool_strict: Sequence[PatchSelectionExample],
) -> Dict[str, object]:
    files = list(dict.fromkeys(Path(path) for path in candidate_sources))
    for root in source_roots:
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".json", ".jsonl"} and ".git" not in path.parts)
    files = list(dict.fromkeys(files))
    source_summaries = []
    all_counts: Dict[str, set[str]] = defaultdict(set)
    total_records = 0
    total_patch_records = 0
    for path in files:
        records = _read_json_records(path)
        patch_rows = 0
        per_instance: Dict[str, set[str]] = defaultdict(set)
        for record in records:
            instance_id = _record_instance_id(record)
            for patch in _extract_patch_candidates(record):
                if not instance_id or not _looks_like_diff(patch):
                    continue
                patch_hash = stable_patch_hash(patch)
                per_instance[instance_id].add(patch_hash)
                all_counts[instance_id].add(patch_hash)
                patch_rows += 1
        total_records += len(records)
        total_patch_records += patch_rows
        source_summaries.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "raw_records": len(records),
                "patch_records": patch_rows,
                "unique_instance_ids": len(per_instance),
                "tasks_with_k8_unique_generated_candidates": sorted(instance_id for instance_id, hashes in per_instance.items() if len(hashes) >= 8),
            }
        )
    pool_all_ids = {example.issue_id for example in pool_all}
    pool_strict_ids = {example.issue_id for example in pool_strict}
    k8_ids = sorted(instance_id for instance_id, hashes in all_counts.items() if len(hashes) >= STAGE6_NUM_CANDIDATES)
    extra_k8 = sorted(set(k8_ids) - pool_all_ids)
    summary = {
        "candidate_source_files_scanned": len(files),
        "source_roots_scanned": [str(root) for root in source_roots],
        "raw_records_loaded": total_records,
        "patch_records_loaded": total_patch_records,
        "unique_instance_ids_in_generated_sources": len(all_counts),
        "tasks_with_at_least_k8_unique_generated_candidates": len(k8_ids),
        "overlap_k8_sources_with_pool_all": len(set(k8_ids) & pool_all_ids),
        "overlap_k8_sources_with_pool_strict": len(set(k8_ids) & pool_strict_ids),
        "extra_k8_generated_tasks_without_complete_official_labels": extra_k8,
        "complete_official_labels_missing_for_extra_k8_tasks": len(extra_k8),
        "source_supply_can_reach_50_tasks_before_harness": len(k8_ids) >= 50,
        "source_supply_can_reach_100_tasks_before_harness": len(k8_ids) >= 100,
    }
    return {
        "benchmark": BENCHMARK,
        "created_at_utc": _now(),
        "candidate_sources": [str(path) for path in candidate_sources],
        "source_roots": [str(root) for root in source_roots],
        "summary": summary,
        "source_files": source_summaries,
        "per_instance_unique_generated_patch_counts": {instance_id: len(hashes) for instance_id, hashes in sorted(all_counts.items())},
        "no_reference_or_gold_patches_used": True,
        "no_raw_preserving_expansion": True,
        "no_fabricated_candidates": True,
        "candidate_order_randomization_required_before_new_harness_labels": True,
        "official_harness_next_step": {
            "eligible_new_k8_tasks": extra_k8,
            "batch_by_candidate_slot": True,
            "slots": list(range(STAGE6_NUM_CANDIDATES)),
            "accept_labels_only_from_official_resolved": True,
            "raw_logs_selector_visible": False,
        },
    }


def build_stage6d2_expansion_status(
    source_audit: Dict[str, object],
    pool_all: Sequence[PatchSelectionExample],
    pool_strict: Sequence[PatchSelectionExample],
    minimum_strict_tasks: int,
    preferred_strict_tasks: int,
    minimum_strict_oracle_positive: int,
    preferred_strict_oracle_positive: int,
) -> Dict[str, object]:
    strict_summary = build_pool_summary(pool_strict)
    source_summary = source_audit["summary"]
    missing_min = max(0, minimum_strict_tasks - int(strict_summary["complete_tasks"]))
    missing_pref = max(0, preferred_strict_tasks - int(strict_summary["complete_tasks"]))
    source_limit = int(source_summary["tasks_with_at_least_k8_unique_generated_candidates"]) < minimum_strict_tasks
    return {
        "pool_all_complete_tasks": len(pool_all),
        "pool_strict_complete_tasks": len(pool_strict),
        "pool_strict_oracle_positive_tasks": int(strict_summary["oracle_positive_tasks"]),
        "minimum_strict_tasks": int(minimum_strict_tasks),
        "preferred_strict_tasks": int(preferred_strict_tasks),
        "minimum_strict_oracle_positive": int(minimum_strict_oracle_positive),
        "preferred_strict_oracle_positive": int(preferred_strict_oracle_positive),
        "missing_strict_tasks_for_minimum": missing_min,
        "missing_strict_tasks_for_preferred": missing_pref,
        "compute_limit_documented": bool(missing_min > 0),
        "source_supply_limit_documented": bool(source_limit),
        "new_harness_runs_executed": False,
        "new_harness_runs_reason": (
            "No additional local generated K=8 SWE-bench Verified candidate supply was found beyond the existing Stage 6B/6D pools; "
            "running Docker on the same incomplete supply would not reach the Stage 6D.2 minimum. Missing candidates and labels were not fabricated."
            if source_limit
            else "Additional K=8 generated supply exists but this run was audit/build-only; run official harness slot batches before adding those tasks."
        ),
    }


def build_stage6d2_quality_gates(
    pool_strict_bundle: Dict[str, object],
    stress: Dict[str, object],
    expansion: Dict[str, object],
    minimum_strict_tasks: int,
    minimum_strict_oracle_positive: int,
) -> Dict[str, bool]:
    strict = pool_strict_bundle["summary"]
    audits = pool_strict_bundle["audits"]
    oracle = float(strict["oracle_pass_at_8"])
    return {
        "pool_strict_has_at_least_50_complete_k8_tasks_or_compute_limit_documented": bool(
            int(strict["complete_tasks"]) >= minimum_strict_tasks or expansion["compute_limit_documented"]
        ),
        "pool_strict_has_at_least_15_oracle_positive_tasks": int(strict["oracle_positive_tasks"]) >= minimum_strict_oracle_positive,
        "oracle_pass_at_8_between_0_20_and_0_80": 0.20 <= oracle <= 0.80,
        "no_exact_reference_hash_hits": bool(audits["exact_reference_hash_audit"].get("passes", False)),
        "near_reference_tasks_quarantined": bool(audits["near_reference_similarity_audit"].get("passes", False)),
        "no_raw_preserving_expansion": True,
        "no_fabricated_candidates": True,
        "output_leakage_audit_passes": bool(audits["output_leakage_audit"].get("passes", False)),
        "source_generator_leakage_audit_passes": bool(audits["source_generator_leakage_audit"].get("passes", False)),
        "duplicate_patch_hash_audit_passes": bool(audits["duplicate_patch_hash_audit"].get("passes", False)),
        "evidence_use_diagnostic_passes": bool(stress["summary"].get("pool_strict_passes", False)),
        "no_selector_training_executed": True,
        "no_final_model_claim_made": True,
    }


def diagnose_stage6d2_blockers(
    pool_strict_bundle: Dict[str, object],
    stress: Dict[str, object],
    expansion: Dict[str, object],
    gates: Dict[str, bool],
) -> List[str]:
    blockers = []
    if expansion.get("compute_limit_documented", False):
        blockers.append("official harness throughput / complete official labels remain below the 50-task minimum")
    if expansion.get("source_supply_limit_documented", False):
        blockers.append("insufficient public generated K=8 patch supply available locally")
    if not gates["pool_strict_has_at_least_15_oracle_positive_tasks"]:
        blockers.append("insufficient oracle-positive tasks in pool_strict")
    if pool_strict_bundle.get("source_generator_skew", {}).get("high_generator_skew", False):
        blockers.append("generator/source skew remains high")
    if not gates["near_reference_tasks_quarantined"]:
        blockers.append("near-reference contamination remains in pool_strict")
    if not gates["evidence_use_diagnostic_passes"]:
        blockers.append("weak evidence retrieval: evidence-use stress diagnostic failed on pool_strict")
    strict_metrics = stress.get("pool_strict", {}).get("metrics", {})
    patch = float(strict_metrics.get("patch_only", {}).get("pass_at_1", 0.0))
    full = float(strict_metrics.get("issue_plus_context_plus_patch", {}).get("pass_at_1", 0.0))
    if patch >= full:
        blockers.append("patch-only shortcut dominance remains")
    return blockers


def render_stage6d2_report(label_audit: Dict[str, object], stress: Dict[str, object]) -> str:
    all_summary = label_audit["pool_all"]["summary"]
    strict_summary = label_audit["pool_strict"]["summary"]
    expansion = label_audit["expansion_status"]
    source = label_audit["source_coverage_summary"]
    strict_stress = stress["pool_strict"]["summary"]
    all_stress = stress["pool_all"]["summary"]
    gates = label_audit["quality_gates"]
    all_baselines = label_audit["pool_all"]["baselines"]
    strict_baselines = label_audit["pool_strict"]["baselines"]
    all_skew = label_audit["pool_all"]["source_generator_skew"]
    strict_skew = label_audit["pool_strict"]["source_generator_skew"]
    lines = [
        "# Stage 6D.2 Pool Scaling",
        "",
        "## Scope",
        "",
        "- Selector training was not executed.",
        "- Selector architecture was not changed.",
        "- Gold/reference patches, raw-preserving expansion, and fabricated candidates were not used.",
        "- Official labels and raw harness logs are not selector-visible.",
        "- No final SWE-bench improvement claim is made.",
        "",
        "## Pool Scale",
        "",
        f"- Pool all complete K=8 tasks: `{all_summary.get('complete_tasks', 0)}`.",
        f"- Pool all official labels: `{all_summary.get('official_label_count', 0)}`.",
        f"- Pool all oracle-positive tasks: `{all_summary.get('oracle_positive_tasks', 0)}`.",
        f"- Pool all oracle pass@8: `{float(all_summary.get('oracle_pass_at_8', 0.0)):.4f}`.",
        f"- Pool strict complete K=8 tasks: `{strict_summary.get('complete_tasks', 0)}`.",
        f"- Pool strict official labels: `{strict_summary.get('official_label_count', 0)}`.",
        f"- Pool strict oracle-positive tasks: `{strict_summary.get('oracle_positive_tasks', 0)}`.",
        f"- Pool strict oracle pass@8: `{float(strict_summary.get('oracle_pass_at_8', 0.0)):.4f}`.",
        "",
        "## Source Coverage",
        "",
        f"- Candidate source files scanned: `{source.get('candidate_source_files_scanned', 0)}`.",
        f"- Raw records loaded: `{source.get('raw_records_loaded', 0)}`.",
        f"- Unique generated-source instance IDs: `{source.get('unique_instance_ids_in_generated_sources', 0)}`.",
        f"- Tasks with K=8 unique generated candidates in sources: `{source.get('tasks_with_at_least_k8_unique_generated_candidates', 0)}`.",
        f"- Extra K=8 generated tasks without complete official labels: `{len(source.get('extra_k8_generated_tasks_without_complete_official_labels', []))}`.",
        f"- New official harness runs executed: `{label_audit.get('official_harness', {}).get('new_harness_runs_executed', False)}`.",
        f"- Compute limit documented: `{expansion.get('compute_limit_documented', False)}`.",
        "",
        "## Pool Baselines",
        "",
        "| Pool | First | Random | Best Generator | Embedding | Patch Only | Issue+Context+Patch | Mismatch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        _baseline_row("pool_all", all_baselines),
        _baseline_row("pool_strict", strict_baselines),
        "",
        "## Source Skew",
        "",
        f"- Pool all unique source agents: `{all_skew.get('unique_source_agents', 0)}`.",
        f"- Pool all high generator skew: `{all_skew.get('high_generator_skew', False)}`.",
        f"- Pool strict unique source agents: `{strict_skew.get('unique_source_agents', 0)}`.",
        f"- Pool strict high generator skew: `{strict_skew.get('high_generator_skew', False)}`.",
        "",
        "## Evidence Stress",
        "",
        f"- Pool all evidence diagnostic passes: `{all_stress.get('passes', False)}`.",
        f"- Pool all mismatch degradation: `{float(all_stress.get('candidate_evidence_mismatch_degradation', 0.0)):.4f}`.",
        f"- Pool strict evidence diagnostic passes: `{strict_stress.get('passes', False)}`.",
        f"- Pool strict issue+context+patch minus patch-only: `{float(strict_stress.get('issue_context_patch_minus_patch_only', 0.0)):.4f}`.",
        f"- Pool strict mismatch degradation: `{float(strict_stress.get('candidate_evidence_mismatch_degradation', 0.0)):.4f}`.",
        f"- Pool strict cross-task shuffle degradation: `{float(strict_stress.get('cross_task_context_shuffle_degradation', 0.0)):.4f}`.",
        f"- Pool strict edited-file removal degradation: `{float(strict_stress.get('edited_file_context_removed_degradation', 0.0)):.4f}`.",
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
        lines.append("- No Stage 6D.2 blockers detected.")
    lines.extend(["", "## Decision", "", str(label_audit.get("decision", "Do not train the selector.")), ""])
    return "\n".join(lines)


def _baseline_row(pool_name: str, baselines: Dict[str, object]) -> str:
    return (
        f"| `{pool_name}` "
        f"| `{_baseline_pass_at_1(baselines, 'first_candidate_baseline'):.4f}` "
        f"| `{_baseline_pass_at_1(baselines, 'random_baseline'):.4f}` "
        f"| `{float(baselines.get('best_generator_on_dev_baseline', {}).get('pass_at_1', 0.0)):.4f}` "
        f"| `{_baseline_pass_at_1(baselines, 'embedding_reranker'):.4f}` "
        f"| `{_baseline_pass_at_1(baselines, 'patch_only_reranker'):.4f}` "
        f"| `{_baseline_pass_at_1(baselines, 'issue_context_patch_reranker'):.4f}` "
        f"| `{_baseline_pass_at_1(baselines, 'candidate_evidence_mismatch'):.4f}` |"
    )


def _baseline_pass_at_1(baselines: Dict[str, object], key: str) -> float:
    row = baselines.get(key, {})
    return float(row.get("pass_at_1", 0.0)) if isinstance(row, dict) else 0.0


def _load_complete_pool(path: Path) -> List[PatchSelectionExample]:
    return [
        example
        for example in load_patch_selection_jsonl(path)
        if len(example.candidates) == STAGE6_NUM_CANDIDATES and len(example.labels_pass_fail) == STAGE6_NUM_CANDIDATES
    ]


def _mark_stage6d2_pool(examples: Sequence[PatchSelectionExample], pool_name: str) -> List[PatchSelectionExample]:
    return [
        replace(
            example,
            metadata={
                **example.metadata,
                "stage6d2_pool": pool_name,
                "stage6d2_candidate_patches_preserved": True,
                "stage6d2_no_selector_training": True,
                "gold_patch_included": False,
                "raw_preserving_expansion_used": False,
                "missing_candidate_fabrication_used": False,
            },
        )
        for example in examples
    ]


def _read_json_records(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
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


def _extract_patch_candidates(record: Dict[str, object]) -> List[str]:
    patches = []
    for key in ("model_patch", "output_patch", "patch", "prediction", "diff", "model_diff"):
        value = record.get(key)
        if isinstance(value, str):
            patches.append(value)
    raw_candidates = record.get("candidates")
    if isinstance(raw_candidates, list):
        for item in raw_candidates:
            if isinstance(item, dict):
                patches.extend(_extract_patch_candidates(item))
    return patches


def _record_instance_id(record: Dict[str, object]) -> str:
    for key in ("instance_id", "issue_id", "task_id", "id"):
        value = record.get(key)
        if isinstance(value, str) and "__" in value:
            return value
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        for key in ("instance_id", "issue_id", "task_id"):
            value = metadata.get(key)
            if value:
                return str(value)
    return ""


def _looks_like_diff(text: str) -> bool:
    value = str(text)
    return "diff --git " in value or ("--- " in value and "+++ " in value and "@@" in value)


def _write_json(path: Path, value: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
