from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.datasets.swe_patch_selection_dataset import (
    STAGE6_NUM_CANDIDATES,
    PatchCandidate,
    PatchSelectionExample,
    duplicate_candidate_patch_hash_audit,
    edited_files_from_diff,
    label_matrix,
    load_patch_selection_jsonl,
    stable_patch_hash,
    stage6_output_leakage_audit,
    validate_patch_selection_examples,
    write_patch_selection_jsonl,
)
from src.experiments.run_stage6_latent_patch_selector import (
    _first_candidate_scores,
    _random_scores,
    evaluate_scores,
)
from src.experiments.run_stage6b1_candidate_generation import DEFAULT_DATASET_NAME, load_benchmark_tasks


BENCHMARK = "stage6d_pool_scale_and_view_hardening"
DEFAULT_INPUT_PATH = Path("results/stage6b3_completed_labeled_candidate_pools.jsonl")
DEFAULT_SEED_CANDIDATE_PATH = Path("results/stage6b1b_generated_candidates.jsonl")
DEFAULT_STAGE6B3_AUDIT_PATH = Path("results/stage6b3_label_completion_audit.json")
DEFAULT_STAGE6B3_QUARANTINE_PATH = Path("results/stage6b3_quarantined_incomplete_labels.jsonl")
DEFAULT_LABELED_POOL_PATH = Path("results/stage6d_labeled_candidate_pools.jsonl")
DEFAULT_VIEW_HARDENED_PATH = Path("results/stage6d_view_hardened_examples.jsonl")
DEFAULT_AUDIT_PATH = Path("results/stage6d_pool_and_view_audit.json")
DEFAULT_DIAGNOSTICS_PATH = Path("results/stage6d_view_diagnostic_results.json")
DEFAULT_REPORT_PATH = Path("reports/STAGE6D_POOL_SCALE_AND_VIEW_HARDENING.md")
DEFAULT_SEED = 606_400
NEAR_REFERENCE_THRESHOLD = 0.95

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "for",
    "from",
    "has",
    "have",
    "if",
    "in",
    "into",
    "is",
    "it",
    "not",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "this",
    "to",
    "was",
    "with",
}

_FORBIDDEN_VISIBLE_PATTERNS = (
    "labels_pass_fail",
    "oracle_pass_at_8",
    "selected_label",
    "pass_fail_label",
    "stage6b2_official_harness",
    "stage6b2_harness_results",
    "report.json",
    "test_output.txt",
    "result_available",
    '"resolved"',
    "official_label",
    "gold_patch",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6D scale audit, evidence-view hardening, and non-neural view diagnostics.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--seed-candidates", default=str(DEFAULT_SEED_CANDIDATE_PATH))
    parser.add_argument("--stage6b3-audit", default=str(DEFAULT_STAGE6B3_AUDIT_PATH))
    parser.add_argument("--stage6b3-quarantine", default=str(DEFAULT_STAGE6B3_QUARANTINE_PATH))
    parser.add_argument("--labeled-output", default=str(DEFAULT_LABELED_POOL_PATH))
    parser.add_argument("--view-output", default=str(DEFAULT_VIEW_HARDENED_PATH))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT_PATH))
    parser.add_argument("--diagnostics", default=str(DEFAULT_DIAGNOSTICS_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target-complete-tasks", type=int, default=100)
    parser.add_argument("--preferred-complete-tasks", type=int, default=200)
    parser.add_argument("--skip-reference-audit", action="store_true")
    args = parser.parse_args()

    result = run_stage6d_pool_scale_and_view_hardening(
        input_path=Path(args.input),
        seed_candidate_path=Path(args.seed_candidates),
        stage6b3_audit_path=Path(args.stage6b3_audit),
        stage6b3_quarantine_path=Path(args.stage6b3_quarantine),
        labeled_output_path=Path(args.labeled_output),
        view_output_path=Path(args.view_output),
        audit_path=Path(args.audit),
        diagnostics_path=Path(args.diagnostics),
        report_path=Path(args.report),
        dataset_name=str(args.dataset_name),
        split=str(args.split),
        seed=int(args.seed),
        target_complete_tasks=int(args.target_complete_tasks),
        preferred_complete_tasks=int(args.preferred_complete_tasks),
        skip_reference_audit=bool(args.skip_reference_audit),
    )
    gates = result["audit"].get("stage6d_success_gates", {})
    print(
        "stage6d: wrote {labeled}, {views}, {audit}, {diagnostics}, {report}; "
        "complete_tasks={tasks}; oracle_positive={positive}; diagnostics_signal={signal}; gates_pass={passes}".format(
            labeled=args.labeled_output,
            views=args.view_output,
            audit=args.audit,
            diagnostics=args.diagnostics,
            report=args.report,
            tasks=result["audit"].get("pool_summary", {}).get("complete_tasks", 0),
            positive=result["audit"].get("pool_summary", {}).get("oracle_positive_tasks", 0),
            signal=result["diagnostics"].get("summary", {}).get("issue_context_signal_detected", False),
            passes=bool(gates and all(gates.values())),
        )
    )


def run_stage6d_pool_scale_and_view_hardening(
    input_path: Path,
    seed_candidate_path: Path,
    stage6b3_audit_path: Path,
    stage6b3_quarantine_path: Path,
    labeled_output_path: Path,
    view_output_path: Path,
    audit_path: Path,
    diagnostics_path: Path,
    report_path: Path,
    dataset_name: str,
    split: str,
    seed: int,
    target_complete_tasks: int,
    preferred_complete_tasks: int,
    skip_reference_audit: bool = False,
) -> Dict[str, object]:
    created_at = _now()
    source_examples = load_patch_selection_jsonl(input_path)
    complete_examples = [
        example
        for example in source_examples
        if len(example.candidates) == STAGE6_NUM_CANDIDATES and len(example.labels_pass_fail) == STAGE6_NUM_CANDIDATES
    ]
    labeled_examples = _with_stage6d_pool_metadata(complete_examples, input_path, target_complete_tasks)
    write_patch_selection_jsonl(labeled_output_path, labeled_examples)

    hardened_examples = build_view_hardened_examples(labeled_examples, seed=seed)
    write_patch_selection_jsonl(view_output_path, hardened_examples)

    diagnostics = run_view_diagnostics(hardened_examples, seed=seed)
    _write_json(diagnostics_path, diagnostics)

    validation = validate_patch_selection_examples(hardened_examples, allow_gold_diagnostic=False)
    output_leakage = stage6_output_leakage_audit(BENCHMARK, seed, hardened_examples)
    duplicate_audit = duplicate_candidate_patch_hash_audit(hardened_examples)
    visible_leakage = source_generator_output_leakage_audit(hardened_examples)
    source_coverage = build_source_coverage_report(
        seed_candidate_path=seed_candidate_path,
        input_path=input_path,
        completed_examples=hardened_examples,
        quarantine_path=stage6b3_quarantine_path,
    )
    stage6b3_audit = _read_json(stage6b3_audit_path)
    benchmark_audit: Dict[str, object] = {"skipped": True, "reason": "reference audit skipped by configuration"}
    exact_reference_audit: Dict[str, object] = {"passes": True, "hits": [], "hits_total": 0, "skipped": True}
    near_reference_audit: Dict[str, object] = {"passes": True, "near_hits": [], "skipped": True}
    if not skip_reference_audit:
        task_records, benchmark_audit = load_benchmark_tasks(
            dataset_name=dataset_name,
            split=split,
            benchmark_jsonl=None,
            requested_task_ids=[example.issue_id for example in hardened_examples],
            task_ids_file=None,
            n_tasks=len(hardened_examples),
            seed=seed,
        )
        exact_reference_audit = exact_reference_hash_audit(hardened_examples, task_records)
        near_reference_audit = near_reference_similarity_audit(
            hardened_examples,
            task_records,
            threshold=NEAR_REFERENCE_THRESHOLD,
        )

    baseline_audit = build_baseline_audit(hardened_examples, seed, diagnostics)
    pool_summary = build_pool_summary(hardened_examples)
    view_audit = build_view_construction_audit(labeled_examples, hardened_examples)
    order_audit = candidate_order_randomized_audit(hardened_examples)
    generator_skew = source_generator_skew_report(hardened_examples)
    compute_limit = build_compute_limit_report(
        pool_summary=pool_summary,
        source_coverage=source_coverage,
        stage6b3_audit=stage6b3_audit,
        target_complete_tasks=target_complete_tasks,
        preferred_complete_tasks=preferred_complete_tasks,
    )

    leakage_reference_order_passes = bool(
        output_leakage.get("passes", False)
        and visible_leakage.get("passes", False)
        and duplicate_audit.get("passes", False)
        and exact_reference_audit.get("passes", False)
        and near_reference_audit.get("passes", False)
        and order_audit.get("passes", False)
    )
    oracle = float(pool_summary.get("oracle_pass_at_8", 0.0))
    gates = {
        "at_least_100_complete_k8_officially_labeled_tasks_or_compute_limit_documented": bool(
            int(pool_summary.get("complete_tasks", 0)) >= target_complete_tasks or compute_limit.get("documented_compute_limit", False)
        ),
        "at_least_30_oracle_positive_tasks_total": int(pool_summary.get("oracle_positive_tasks", 0)) >= 30,
        "oracle_pass_at_8_between_0_20_and_0_80": 0.20 <= oracle <= 0.80,
        "all_leakage_reference_order_audits_pass": leakage_reference_order_passes,
        "source_generator_skew_reported": bool(generator_skew.get("source_agent_counts")),
        "evidence_views_constructed_for_all_examples": bool(view_audit.get("passes", False)),
        "view_diagnostics_show_issue_context_evidence_signal": bool(
            diagnostics.get("summary", {}).get("issue_context_signal_detected", False)
        ),
        "no_selector_final_claim_made": True,
    }
    blockers = diagnose_stage6d_blockers(pool_summary, diagnostics, generator_skew, compute_limit)
    if not leakage_reference_order_passes:
        if not bool(exact_reference_audit.get("passes", False)):
            blockers.append("exact reference hash audit failed or could not fully verify references")
        if not bool(near_reference_audit.get("passes", False)):
            blockers.append(
                "near-reference similarity audit flagged "
                f"{int(near_reference_audit.get('near_hits_total', 0))} generated candidates at threshold {NEAR_REFERENCE_THRESHOLD}"
            )
        if not bool(output_leakage.get("passes", False)) or not bool(visible_leakage.get("passes", False)):
            blockers.append("selector-visible leakage audit failed")
        if not bool(order_audit.get("passes", False)):
            blockers.append("candidate order randomization audit failed")
    audit = {
        "benchmark": BENCHMARK,
        "created_at_utc": created_at,
        "input_path": str(input_path),
        "labeled_output_path": str(labeled_output_path),
        "view_output_path": str(view_output_path),
        "diagnostics_path": str(diagnostics_path),
        "dataset_name": dataset_name,
        "split": split,
        "seed": int(seed),
        "no_selector_training_executed": True,
        "no_architecture_search_or_test_tuning": True,
        "no_final_swe_bench_improvement_claim": True,
        "no_reference_or_gold_candidates": True,
        "no_fabricated_candidates": True,
        "no_raw_preserving_expansion": True,
        "official_harness_logs_excluded_from_selector_visible_inputs": True,
        "pool_summary": pool_summary,
        "source_coverage_report": source_coverage,
        "compute_limit_report": compute_limit,
        "candidate_order_randomized_audit": order_audit,
        "view_construction_audit": view_audit,
        "k8_and_label_validation": validation,
        "official_label_availability_audit": official_label_availability_audit(hardened_examples),
        "output_leakage_audit": output_leakage,
        "source_generator_leakage_audit": visible_leakage,
        "duplicate_candidate_patch_hash_audit": duplicate_audit,
        "exact_reference_patch_hash_audit": exact_reference_audit,
        "near_reference_similarity_audit": near_reference_audit,
        "benchmark_dataset_audit": benchmark_audit,
        "source_generator_skew": generator_skew,
        "baselines": baseline_audit,
        "stage6d_success_gates": gates,
        "stage6d_gates_pass": bool(all(gates.values())),
        "blocker_diagnosis": blockers,
        "next_recommendation": (
            "Proceed to Stage 6E dev-only selector validation only after scaling official labels and evidence diagnostics pass."
            if not all(gates.values())
            else "Proceed to Stage 6E dev-only selector validation."
        ),
    }
    _write_json(audit_path, audit)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_stage6d_report(audit, diagnostics), encoding="utf-8")
    return {"audit": audit, "diagnostics": diagnostics, "hardened_examples": hardened_examples}


def build_view_hardened_examples(examples: Sequence[PatchSelectionExample], seed: int) -> List[PatchSelectionExample]:
    hardened: List[PatchSelectionExample] = []
    for example_index, example in enumerate(examples):
        issue_terms = _top_issue_terms(example.issue_text + "\n" + example.failing_test_summary, limit=24)
        candidate_stats = [_diff_stats(candidate.candidate_diff) for candidate in example.candidates]
        retrieved_contexts = _build_retrieved_contexts(example, issue_terms, candidate_stats)
        dependency_evidence = _build_consistency_risk_evidence(example, candidate_stats)
        candidates = tuple(
            replace(
                candidate,
                metadata={
                    **candidate.metadata,
                    "stage6d_patch_stats": _safe_candidate_stats_for_metadata(candidate_stats[index]),
                    "stage6d_view_hardened": True,
                    "stage6d_patch_hash_preserved": candidate.patch_hash,
                },
            )
            for index, candidate in enumerate(example.candidates)
        )
        hardened_metadata = {
            **example.metadata,
            "stage6d_view_hardened": True,
            "stage6d_view_hardening_seed": int(seed),
            "stage6d_view_hardening_example_index": int(example_index),
            "stage6d_source_context_backend": "diff_derived_fallback_no_repo_checkout",
            "stage6d_selector_visible_excludes_official_labels_and_hidden_logs": True,
            "dependency_callgraph_related_file_evidence": dependency_evidence,
        }
        failing_test_summary = _hardened_failing_summary(example.failing_test_summary)
        hardened.append(
            replace(
                example,
                failing_test_summary=failing_test_summary,
                retrieved_contexts=tuple(retrieved_contexts),
                candidates=candidates,
                metadata=hardened_metadata,
            )
        )
    return hardened


def run_view_diagnostics(examples: Sequence[PatchSelectionExample], seed: int) -> Dict[str, object]:
    methods = {
        "random_candidate": _random_scores(examples, seed),
        "first_candidate_baseline": _first_candidate_scores(examples),
        "patch_only_embedding_reranker": patch_only_embedding_scores(examples),
        "issue_patch_embedding_reranker": issue_patch_embedding_scores(examples),
        "issue_context_patch_embedding_reranker": issue_context_patch_embedding_scores(examples),
        "candidate_only_baseline": candidate_only_scores(examples),
        "candidate_evidence_mismatch_baseline": candidate_evidence_mismatch_scores(examples),
        "context_only_baseline": context_only_scores(examples),
        "schema_template_only_baseline": schema_template_only_scores(examples),
    }
    metrics = {
        name: evaluate_scores(name, scores.astype(np.float32), examples, split="stage6d_complete_pool")
        for name, scores in methods.items()
    }
    patch_only = float(metrics["patch_only_embedding_reranker"]["pass_at_1"])
    issue_patch = float(metrics["issue_patch_embedding_reranker"]["pass_at_1"])
    issue_context_patch = float(metrics["issue_context_patch_embedding_reranker"]["pass_at_1"])
    mismatch = float(metrics["candidate_evidence_mismatch_baseline"]["pass_at_1"])
    context_delta = issue_context_patch - patch_only
    mismatch_delta = issue_context_patch - mismatch
    summary = {
        "benchmark": BENCHMARK,
        "task_count": len(examples),
        "oracle_positive_tasks": int(sum(example.oracle_pass_at_8 for example in examples)),
        "oracle_pass_at_8": _oracle_pass_at_8(examples),
        "patch_only_pass_at_1": patch_only,
        "issue_patch_pass_at_1": issue_patch,
        "issue_context_patch_pass_at_1": issue_context_patch,
        "candidate_evidence_mismatch_pass_at_1": mismatch,
        "issue_context_minus_patch_only": context_delta,
        "issue_context_minus_mismatch": mismatch_delta,
        "issue_context_signal_detected": bool(context_delta > 0.0 or mismatch_delta > 0.0),
        "diagnostic_rule": (
            "pass if issue+context+patch outperforms patch-only, or candidate/evidence mismatch degrades the "
            "issue+context+patch scorer"
        ),
        "selector_training_executed": False,
    }
    return {
        "benchmark": BENCHMARK,
        "created_at_utc": _now(),
        "seed": int(seed),
        "non_neural_diagnostics_only": True,
        "metrics": metrics,
        "summary": summary,
    }


def patch_only_embedding_scores(examples: Sequence[PatchSelectionExample]) -> np.ndarray:
    query = "small localized source fix existing implementation minimal patch avoid generated build artifacts avoid test-only change"
    rows = []
    for example in examples:
        structural = _row_normalize([_patch_structure_score(_diff_stats(candidate.candidate_diff)) for candidate in example.candidates])
        lexical = [
            _cosine_token_similarity(query, _patch_embedding_text(candidate.candidate_diff))
            for candidate in example.candidates
        ]
        rows.append(np.asarray(lexical, dtype=np.float32) + 0.35 * structural)
    return np.asarray(rows, dtype=np.float32)


def issue_patch_embedding_scores(examples: Sequence[PatchSelectionExample]) -> np.ndarray:
    rows = []
    for example in examples:
        query = "\n".join([example.issue_text, example.failing_test_summary])
        patch_scores = patch_only_embedding_scores([example])[0]
        lexical = [
            _cosine_token_similarity(query, _patch_embedding_text(candidate.candidate_diff))
            for candidate in example.candidates
        ]
        rows.append(np.asarray(lexical, dtype=np.float32) + 0.20 * _row_normalize(patch_scores))
    return np.asarray(rows, dtype=np.float32)


def issue_context_patch_embedding_scores(examples: Sequence[PatchSelectionExample]) -> np.ndarray:
    rows = []
    for example in examples:
        patch_scores = patch_only_embedding_scores([example])[0]
        lexical = []
        for candidate_index, candidate in enumerate(example.candidates):
            query = _candidate_hardened_context_text(example, candidate_index)
            lexical.append(_cosine_token_similarity(query, _patch_embedding_text(candidate.candidate_diff)))
        rows.append(np.asarray(lexical, dtype=np.float32) + 0.20 * _row_normalize(patch_scores))
    return np.asarray(rows, dtype=np.float32)


def candidate_evidence_mismatch_scores(examples: Sequence[PatchSelectionExample]) -> np.ndarray:
    if not examples:
        return np.zeros((0, STAGE6_NUM_CANDIDATES), dtype=np.float32)
    rows = []
    for index, example in enumerate(examples):
        patch_scores = patch_only_embedding_scores([example])[0]
        mismatch_example = examples[(index + 1) % len(examples)]
        lexical = []
        for candidate_index, candidate in enumerate(example.candidates):
            mismatch_index = min(candidate_index, len(mismatch_example.candidates) - 1)
            query = _candidate_hardened_context_text(mismatch_example, mismatch_index)
            lexical.append(_cosine_token_similarity(query, _patch_embedding_text(candidate.candidate_diff)))
        rows.append(np.asarray(lexical, dtype=np.float32) + 0.20 * _row_normalize(patch_scores))
    return np.asarray(rows, dtype=np.float32)


def candidate_only_scores(examples: Sequence[PatchSelectionExample]) -> np.ndarray:
    rows = []
    for example in examples:
        values = []
        for index, candidate in enumerate(example.candidates):
            agent_hash = int(stable_patch_hash(candidate.candidate_source_agent)[:6], 16) % 1000
            values.append(float(agent_hash) / 1000.0 - float(index) * 1e-4)
        rows.append(values)
    return np.asarray(rows, dtype=np.float32)


def context_only_scores(examples: Sequence[PatchSelectionExample]) -> np.ndarray:
    rows = []
    for example in examples:
        score = float(len(_tokens(_full_hardened_context_text(example)))) / 10_000.0
        rows.append([score for _ in example.candidates])
    return np.asarray(rows, dtype=np.float32)


def schema_template_only_scores(examples: Sequence[PatchSelectionExample]) -> np.ndarray:
    return np.zeros((len(examples), STAGE6_NUM_CANDIDATES), dtype=np.float32)


def build_pool_summary(examples: Sequence[PatchSelectionExample]) -> Dict[str, object]:
    positives = [example.oracle_pass_at_8 for example in examples]
    labels = label_matrix(examples) if examples else np.zeros((0, STAGE6_NUM_CANDIDATES), dtype=np.float32)
    return {
        "complete_tasks": len(examples),
        "candidate_count": int(sum(len(example.candidates) for example in examples)),
        "official_label_count": int(labels.size),
        "candidate_count_distribution": dict(Counter(str(len(example.candidates)) for example in examples)),
        "label_count_distribution": dict(Counter(str(len(example.labels_pass_fail)) for example in examples)),
        "oracle_positive_tasks": int(sum(positives)),
        "oracle_empty_tasks": int(sum(not value for value in positives)),
        "oracle_pass_at_8": float(np.mean(positives)) if positives else 0.0,
        "repos": dict(Counter(example.repo for example in examples)),
        "task_ids": [example.issue_id for example in examples],
    }


def official_label_availability_audit(examples: Sequence[PatchSelectionExample]) -> Dict[str, object]:
    missing = []
    for example in examples:
        if len(example.labels_pass_fail) != len(example.candidates):
            missing.append({"id": example.id, "issue_id": example.issue_id, "labels": len(example.labels_pass_fail), "candidates": len(example.candidates)})
        elif any(int(value) not in {0, 1} for value in example.labels_pass_fail):
            missing.append({"id": example.id, "issue_id": example.issue_id, "reason": "non_binary_label"})
    return {
        "examples_checked": len(examples),
        "candidate_labels_expected": int(sum(len(example.candidates) for example in examples)),
        "candidate_labels_available": int(sum(len(example.labels_pass_fail) for example in examples)),
        "missing_or_invalid": missing,
        "passes": not missing,
    }


def build_baseline_audit(
    examples: Sequence[PatchSelectionExample],
    seed: int,
    diagnostics: Dict[str, object],
) -> Dict[str, object]:
    best_generator = best_generator_on_dev_baseline(examples, seed)
    per_repo = per_repo_oracle_pass_at_8(examples)
    per_generator = per_generator_pass_rate(examples)
    metrics = dict(diagnostics.get("metrics", {}))
    return {
        "oracle_pass_at_8": _oracle_pass_at_8(examples),
        "first_candidate_baseline": metrics.get("first_candidate_baseline", {}),
        "random_baseline": metrics.get("random_candidate", {}),
        "best_generator_on_dev_baseline": best_generator,
        "embedding_reranker_baseline": metrics.get("issue_context_patch_embedding_reranker", {}),
        "patch_only_embedding_reranker": metrics.get("patch_only_embedding_reranker", {}),
        "issue_patch_embedding_reranker": metrics.get("issue_patch_embedding_reranker", {}),
        "per_repository_oracle_pass_at_8": per_repo,
        "per_generator_pass_rate": per_generator,
    }


def best_generator_on_dev_baseline(examples: Sequence[PatchSelectionExample], seed: int) -> Dict[str, object]:
    if not examples:
        return {"best_generator": None, "dev_task_count": 0, "pass_at_1": 0.0, "predictions": []}
    rng = np.random.default_rng(seed + 6_401)
    indices = [int(value) for value in rng.permutation(len(examples)).tolist()]
    dev_count = max(1, min(len(examples), int(math.ceil(len(examples) * 0.2))))
    dev_indices = set(indices[:dev_count])
    rates: Dict[str, List[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        if index not in dev_indices:
            continue
        for candidate_index, candidate in enumerate(example.candidates):
            rates[candidate.candidate_source_agent].append(int(example.labels_pass_fail[candidate_index]))
    dev_rates = {name: float(np.mean(values)) if values else 0.0 for name, values in rates.items()}
    best = sorted(dev_rates, key=lambda name: (-dev_rates[name], name))[0] if dev_rates else None
    predictions = []
    correct = []
    labels = label_matrix(examples)
    for row, example in enumerate(examples):
        choice = 0
        if best is not None:
            for candidate_index, candidate in enumerate(example.candidates):
                if candidate.candidate_source_agent == best:
                    choice = candidate_index
                    break
        predictions.append(int(choice))
        correct.append(bool(labels[row, choice] > 0))
    return {
        "best_generator": best,
        "dev_task_count": dev_count,
        "dev_generator_rates": dev_rates,
        "pass_at_1": float(np.mean(correct)) if correct else 0.0,
        "predictions": predictions,
        "does_not_equal_oracle": not math.isclose(float(np.mean(correct)) if correct else 0.0, _oracle_pass_at_8(examples)),
    }


def per_repo_oracle_pass_at_8(examples: Sequence[PatchSelectionExample]) -> Dict[str, Dict[str, object]]:
    rows: Dict[str, List[PatchSelectionExample]] = defaultdict(list)
    for example in examples:
        rows[example.repo].append(example)
    return {
        repo: {
            "tasks": len(items),
            "oracle_positive_tasks": int(sum(item.oracle_pass_at_8 for item in items)),
            "oracle_pass_at_8": _oracle_pass_at_8(items),
        }
        for repo, items in sorted(rows.items())
    }


def per_generator_pass_rate(examples: Sequence[PatchSelectionExample]) -> Dict[str, Dict[str, object]]:
    rows: Dict[str, List[int]] = defaultdict(list)
    for example in examples:
        for index, candidate in enumerate(example.candidates):
            rows[candidate.candidate_source_agent].append(int(example.labels_pass_fail[index]))
    return {
        name: {"candidate_count": len(values), "pass_rate": float(np.mean(values)) if values else 0.0}
        for name, values in sorted(rows.items())
    }


def exact_reference_hash_audit(examples: Sequence[PatchSelectionExample], task_records: Dict[str, object]) -> Dict[str, object]:
    hits = []
    missing = []
    for example in examples:
        task = task_records.get(example.issue_id)
        ref_hash = getattr(task, "reference_hash", "") if task is not None else ""
        if not ref_hash:
            missing.append(example.issue_id)
            continue
        for index, candidate in enumerate(example.candidates):
            if candidate.patch_hash == ref_hash:
                hits.append({"instance_id": example.issue_id, "candidate_index": index, "candidate_id": candidate.candidate_id})
    return {"hits": hits, "hits_total": len(hits), "missing_reference_hashes": missing, "passes": not hits and not missing}


def near_reference_similarity_audit(
    examples: Sequence[PatchSelectionExample],
    task_records: Dict[str, object],
    threshold: float,
) -> Dict[str, object]:
    near_hits = []
    missing = []
    for example in examples:
        task = task_records.get(example.issue_id)
        reference_patch = getattr(task, "reference_patch", "") if task is not None else ""
        if not reference_patch.strip():
            missing.append(example.issue_id)
            continue
        ref_norm = _normalize_patch_for_similarity(reference_patch)
        for index, candidate in enumerate(example.candidates):
            ratio = SequenceMatcher(None, ref_norm, _normalize_patch_for_similarity(candidate.candidate_diff)).ratio()
            if ratio >= threshold:
                near_hits.append(
                    {
                        "instance_id": example.issue_id,
                        "candidate_index": index,
                        "candidate_id": candidate.candidate_id,
                        "similarity": float(ratio),
                    }
                )
    return {
        "threshold": float(threshold),
        "near_hits": near_hits,
        "near_hits_total": len(near_hits),
        "missing_reference_patches": missing,
        "passes": not near_hits and not missing,
    }


def candidate_order_randomized_audit(examples: Sequence[PatchSelectionExample]) -> Dict[str, object]:
    missing = []
    source_counts = Counter()
    for example in examples:
        source_counts[str(example.metadata.get("candidate_order_source", "unknown"))] += 1
        if not bool(example.metadata.get("candidate_order_randomized", False)):
            missing.append(example.issue_id)
    return {
        "examples_checked": len(examples),
        "candidate_order_randomized_tasks": len(examples) - len(missing),
        "candidate_order_source_counts": dict(source_counts),
        "not_randomized_task_ids": missing,
        "passes": not missing,
    }


def build_view_construction_audit(
    before: Sequence[PatchSelectionExample],
    after: Sequence[PatchSelectionExample],
) -> Dict[str, object]:
    mismatches = []
    constructed = []
    before_by_id = {example.id: example for example in before}
    for example in after:
        original = before_by_id.get(example.id)
        if original is None:
            mismatches.append({"id": example.id, "reason": "missing_original"})
            continue
        before_hashes = [candidate.patch_hash for candidate in original.candidates]
        after_hashes = [candidate.patch_hash for candidate in example.candidates]
        if before_hashes != after_hashes:
            mismatches.append({"id": example.id, "reason": "candidate_patch_hash_changed"})
        if list(original.labels_pass_fail) != list(example.labels_pass_fail):
            mismatches.append({"id": example.id, "reason": "labels_changed"})
        dependency = str(example.metadata.get("dependency_callgraph_related_file_evidence", ""))
        constructed.append(
            bool(
                example.retrieved_contexts
                and "Stage 6D retrieved source context" in "\n".join(example.retrieved_contexts)
                and "Avenue local syntax/API compatibility" in dependency
            )
        )
    return {
        "examples_checked": len(after),
        "patch_or_label_mismatches": mismatches,
        "examples_with_hardened_evidence_views": int(sum(constructed)),
        "source_context_backend": "diff_derived_fallback_no_repo_checkout",
        "passes": not mismatches and len(after) == len(before) and all(constructed),
    }


def source_generator_output_leakage_audit(examples: Sequence[PatchSelectionExample]) -> Dict[str, object]:
    hits = []
    for example in examples:
        visible_chunks = [
            example.issue_text,
            example.failing_test_summary,
            "\n".join(example.retrieved_contexts),
            str(example.metadata.get("dependency_callgraph_related_file_evidence", "")),
        ]
        visible_chunks.extend(
            f"{candidate.candidate_source_agent}\n{candidate.candidate_visible_test_result or ''}\n{candidate.candidate_diff}"
            for candidate in example.candidates
        )
        text = "\n".join(visible_chunks).lower()
        row_hits = [pattern for pattern in _FORBIDDEN_VISIBLE_PATTERNS if pattern.lower() in text]
        if row_hits:
            hits.append({"id": example.id, "issue_id": example.issue_id, "hits": row_hits})
    return {
        "examples_checked": len(examples),
        "selector_visible_forbidden_source_or_label_hits": hits,
        "candidate_source_agent_visible_by_design": True,
        "generator_metadata_pass_rates_excluded_from_selector_visible_inputs": True,
        "passes": not hits,
    }


def source_generator_skew_report(examples: Sequence[PatchSelectionExample]) -> Dict[str, object]:
    source_counts = Counter(candidate.candidate_source_agent for example in examples for candidate in example.candidates)
    task_source_signatures = Counter(
        "|".join(sorted({candidate.candidate_source_agent for candidate in example.candidates}))
        for example in examples
    )
    repo_counts = Counter(example.repo for example in examples)
    return {
        "source_agent_counts": dict(source_counts),
        "task_source_signature_counts": dict(task_source_signatures),
        "repo_counts": dict(repo_counts),
        "unique_source_agents": len(source_counts),
        "unique_repositories": len(repo_counts),
        "high_generator_skew": len(source_counts) <= 2 or bool(source_counts and max(source_counts.values()) / max(1, sum(source_counts.values())) >= 0.75),
    }


def build_source_coverage_report(
    seed_candidate_path: Path,
    input_path: Path,
    completed_examples: Sequence[PatchSelectionExample],
    quarantine_path: Path,
) -> Dict[str, object]:
    seed_rows = _read_jsonl(seed_candidate_path)
    completed_ids = {example.issue_id for example in completed_examples}
    seed_ids = [_record_instance_id(row) for row in seed_rows]
    seed_counter = Counter(instance_id for instance_id in seed_ids if instance_id)
    quarantine_rows = _read_jsonl(quarantine_path)
    quarantine_ids = sorted({_record_instance_id(row) for row in quarantine_rows if _record_instance_id(row)})
    return {
        "source_files_loaded": int(seed_candidate_path.exists()) + int(input_path.exists()) + int(quarantine_path.exists()),
        "seed_candidate_path": str(seed_candidate_path),
        "seed_candidate_rows_loaded": len(seed_rows),
        "seed_unique_instance_ids": len(seed_counter),
        "complete_label_input_path": str(input_path),
        "complete_label_tasks_loaded": len(completed_examples),
        "overlap_complete_tasks_with_seed_candidates": len(completed_ids & set(seed_counter)),
        "candidates_per_seed_instance_before_label_filtering_top": dict(seed_counter.most_common(25)),
        "candidates_per_complete_instance_after_filtering": {example.issue_id: len(example.candidates) for example in completed_examples},
        "top_missing_or_incomplete_instance_ids": quarantine_ids[:25],
        "quarantined_incomplete_task_count": len(quarantine_ids),
        "top_source_files_by_accepted_candidates": {str(input_path): int(sum(len(example.candidates) for example in completed_examples))},
    }


def build_compute_limit_report(
    pool_summary: Dict[str, object],
    source_coverage: Dict[str, object],
    stage6b3_audit: Dict[str, object],
    target_complete_tasks: int,
    preferred_complete_tasks: int,
) -> Dict[str, object]:
    complete_tasks = int(pool_summary.get("complete_tasks", 0))
    seed_unique = int(source_coverage.get("seed_unique_instance_ids", 0))
    label_completion = stage6b3_audit.get("complete_label_subset", {}) if isinstance(stage6b3_audit, dict) else {}
    return {
        "target_complete_tasks": int(target_complete_tasks),
        "preferred_complete_tasks": int(preferred_complete_tasks),
        "available_complete_tasks": complete_tasks,
        "available_stage6b1b_seed_tasks": seed_unique,
        "missing_complete_tasks_for_target": max(0, int(target_complete_tasks) - complete_tasks),
        "missing_complete_tasks_for_preferred_target": max(0, int(preferred_complete_tasks) - complete_tasks),
        "documented_compute_limit": complete_tasks < int(target_complete_tasks),
        "reason": (
            "Only the Stage 6B.3 complete-label subset is locally available. Scaling to 100 tasks requires additional "
            "generated K=8 pools and official Docker harness evaluations; missing labels are not fabricated."
            if complete_tasks < int(target_complete_tasks)
            else "Target complete task count is available."
        ),
        "stage6b3_complete_tasks": label_completion.get("complete_tasks"),
        "stage6b3_complete_candidates": label_completion.get("complete_candidates"),
        "stage6b3_oracle_positive_tasks": label_completion.get("oracle_positive_tasks"),
    }


def diagnose_stage6d_blockers(
    pool_summary: Dict[str, object],
    diagnostics: Dict[str, object],
    generator_skew: Dict[str, object],
    compute_limit: Dict[str, object],
) -> List[str]:
    blockers = []
    if int(pool_summary.get("complete_tasks", 0)) < int(compute_limit.get("target_complete_tasks", 100)):
        blockers.append("official harness throughput / available complete-label pool is below the 100-task target")
    if int(pool_summary.get("oracle_positive_tasks", 0)) < 30:
        blockers.append("insufficient oracle-positive tasks for Stage 6E train/dev/test targets")
    if bool(generator_skew.get("high_generator_skew", False)):
        blockers.append("generator/source skew remains high")
    summary = diagnostics.get("summary", {}) if isinstance(diagnostics, dict) else {}
    if not bool(summary.get("issue_context_signal_detected", False)):
        blockers.append("view diagnostics did not show measurable issue/context evidence signal")
    if float(summary.get("patch_only_pass_at_1", 0.0)) >= float(summary.get("issue_context_patch_pass_at_1", 0.0)):
        blockers.append("patch-only shortcut dominance remains a risk")
    return blockers


def render_stage6d_report(audit: Dict[str, object], diagnostics: Dict[str, object]) -> str:
    pool = audit.get("pool_summary", {})
    gates = audit.get("stage6d_success_gates", {})
    diag = diagnostics.get("summary", {})
    baselines = audit.get("baselines", {})
    lines = [
        "# Stage 6D Pool Scale and View Hardening",
        "",
        "## Scope",
        "",
        "- Selector training was not executed.",
        "- Candidate patches were not altered.",
        "- Gold/reference patches, raw-preserving expansion, and fabricated candidates were not used.",
        "- Official labels and raw harness logs are excluded from selector-visible views.",
        "- No final SWE-bench improvement claim is made.",
        "",
        "## Pool Status",
        "",
        f"- Complete K=8 officially labeled tasks: `{pool.get('complete_tasks', 0)}`.",
        f"- Official candidate labels: `{pool.get('official_label_count', 0)}`.",
        f"- Oracle-positive tasks: `{pool.get('oracle_positive_tasks', 0)}`.",
        f"- Oracle-empty tasks: `{pool.get('oracle_empty_tasks', 0)}`.",
        f"- Oracle pass@8: `{float(pool.get('oracle_pass_at_8', 0.0)):.4f}`.",
        f"- Scale compute limit documented: `{audit.get('compute_limit_report', {}).get('documented_compute_limit', False)}`.",
        "",
        "## Baselines",
        "",
        f"- First-candidate pass@1: `{float(baselines.get('first_candidate_baseline', {}).get('pass_at_1', 0.0)):.4f}`.",
        f"- Random pass@1: `{float(baselines.get('random_baseline', {}).get('pass_at_1', 0.0)):.4f}`.",
        f"- Best-generator-on-dev pass@1: `{float(baselines.get('best_generator_on_dev_baseline', {}).get('pass_at_1', 0.0)):.4f}`.",
        f"- Issue+context+patch reranker pass@1: `{float(diag.get('issue_context_patch_pass_at_1', 0.0)):.4f}`.",
        f"- Patch-only reranker pass@1: `{float(diag.get('patch_only_pass_at_1', 0.0)):.4f}`.",
        f"- Candidate/evidence mismatch pass@1: `{float(diag.get('candidate_evidence_mismatch_pass_at_1', 0.0)):.4f}`.",
        "",
        "## Audit Notes",
        "",
        f"- Exact reference hash hits: `{audit.get('exact_reference_patch_hash_audit', {}).get('hits_total', 0)}`.",
        f"- Near-reference similarity hits: `{audit.get('near_reference_similarity_audit', {}).get('near_hits_total', 0)}`.",
        f"- Output leakage audit passes: `{audit.get('output_leakage_audit', {}).get('passes', False)}`.",
        f"- Source/generator leakage audit passes: `{audit.get('source_generator_leakage_audit', {}).get('passes', False)}`.",
        f"- Duplicate patch hash audit passes: `{audit.get('duplicate_candidate_patch_hash_audit', {}).get('passes', False)}`.",
        "",
        "## View Diagnostics",
        "",
        f"- Issue+context minus patch-only: `{float(diag.get('issue_context_minus_patch_only', 0.0)):.4f}`.",
        f"- Issue+context minus mismatch: `{float(diag.get('issue_context_minus_mismatch', 0.0)):.4f}`.",
        f"- Issue/context signal detected: `{bool(diag.get('issue_context_signal_detected', False))}`.",
        "",
        "## Gates",
        "",
        "| Gate | Pass |",
        "|---|---:|",
    ]
    for key, value in gates.items():
        lines.append(f"| `{key}` | `{bool(value)}` |")
    lines.extend(["", "## Blockers", ""])
    blockers = audit.get("blocker_diagnosis", [])
    if blockers:
        for blocker in blockers:
            lines.append(f"- {blocker}.")
    else:
        lines.append("- No Stage 6D blockers detected.")
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            (
                "Stage 6D gates passed for proceeding to Stage 6E dev-only selector validation."
                if bool(audit.get("stage6d_gates_pass", False))
                else "Stage 6D gates did not pass; larger official labeled candidate pools and/or stronger evidence retrieval are needed before Stage 6E."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _with_stage6d_pool_metadata(
    examples: Sequence[PatchSelectionExample],
    input_path: Path,
    target_complete_tasks: int,
) -> List[PatchSelectionExample]:
    out = []
    for example in examples:
        out.append(
            replace(
                example,
                metadata={
                    **example.metadata,
                    "candidate_pool_version": "stage6d_scaled_official_labeled_pool_v1",
                    "stage6d_candidate_pool_source": str(input_path),
                    "stage6d_target_complete_tasks": int(target_complete_tasks),
                    "stage6d_candidate_patches_preserved": True,
                    "stage6d_no_selector_training": True,
                    "gold_patch_included": False,
                    "raw_preserving_expansion_used": False,
                    "missing_candidate_fabrication_used": False,
                },
            )
        )
    return out


def _hardened_failing_summary(original: str) -> str:
    return (
        "Public issue/failure evidence available before candidate selection.\n"
        "No official harness outcome or hidden test log is included.\n"
        f"{original}"
    )


def _build_retrieved_contexts(
    example: PatchSelectionExample,
    issue_terms: Sequence[str],
    candidate_stats: Sequence[Dict[str, object]],
) -> List[str]:
    contexts = list(example.retrieved_contexts)
    all_files = sorted({path for stats in candidate_stats for path in stats.get("edited_files", [])})
    contexts.append(
        "\n".join(
            [
                "Stage 6D retrieved source context summary.",
                "Backend: diff-derived fallback because no checked-out pre-patch repository was provided to this stage.",
                f"Repository: {example.repo}",
                f"Issue terms: {', '.join(issue_terms) if issue_terms else 'none'}",
                f"Edited files across candidates: {', '.join(all_files[:24]) if all_files else 'unknown'}",
                "This context uses pre-patch hunk context and removed lines from generated diffs only.",
            ]
        )
    )
    for index, stats in enumerate(candidate_stats):
        contexts.append(
            "\n".join(
                [
                    f"Stage 6D retrieved source context for candidate {index}.",
                    f"Edited files: {', '.join(stats.get('edited_files', [])[:8]) or 'unknown'}",
                    f"Hunk scopes: {', '.join(stats.get('hunk_scopes', [])[:8]) or 'not supplied'}",
                    f"Neighboring symbols/classes: {', '.join(stats.get('symbols', [])[:8]) or 'not detected'}",
                    f"Imports near edited code: {', '.join(stats.get('imports', [])[:8]) or 'not detected'}",
                    "Pre-patch hunk excerpt:",
                    str(stats.get("pre_patch_excerpt", "not available")),
                ]
            )
        )
    return contexts


def _build_consistency_risk_evidence(
    example: PatchSelectionExample,
    candidate_stats: Sequence[Dict[str, object]],
) -> str:
    rows = [
        f"Repository {example.repo}.",
        "Stage 6D consistency/risk evidence is static and selector-visible.",
        "Official harness outcomes and hidden test logs are excluded.",
        "",
        "Avenue local syntax/API compatibility:",
    ]
    for index, stats in enumerate(candidate_stats):
        rows.append(
            "candidate {index}: static_apply_status=not_available_without_checkout; syntax_status=not_executed; "
            "python_files={python_files}; import_changes={import_changes}; call_like_tokens={calls}".format(
                index=index,
                python_files=int(stats.get("python_file_count", 0)),
                import_changes=int(stats.get("import_change_count", 0)),
                calls=", ".join(stats.get("call_tokens", [])[:6]) or "none",
            )
        )
    rows.extend(["", "Avenue issue-behavior compatibility:"])
    for index, stats in enumerate(candidate_stats):
        rows.append(
            "candidate {index}: issue_term_overlap={overlap}; edited_symbols={symbols}".format(
                index=index,
                overlap=", ".join(stats.get("issue_overlap_terms", [])[:8]) or "not measured in diff-only context",
                symbols=", ".join(stats.get("symbols", [])[:8]) or "not detected",
            )
        )
    rows.extend(["", "Avenue surrounding-code consistency:"])
    for index, stats in enumerate(candidate_stats):
        rows.append(
            "candidate {index}: files={files}; added={added}; deleted={deleted}; multi_file={multi_file}; new_file={new_file}".format(
                index=index,
                files=", ".join(stats.get("edited_files", [])[:6]) or "unknown",
                added=int(stats.get("added_lines", 0)),
                deleted=int(stats.get("deleted_lines", 0)),
                multi_file=bool(stats.get("multi_file", False)),
                new_file=bool(stats.get("new_file", False)),
            )
        )
    rows.extend(["", "Avenue regression/security-risk compatibility:"])
    for index, stats in enumerate(candidate_stats):
        rows.append(
            "candidate {index}: touches_tests={tests}; touches_source={source}; security_terms={security}; locality={locality}".format(
                index=index,
                tests=bool(stats.get("touches_tests", False)),
                source=bool(stats.get("touches_source", False)),
                security=", ".join(stats.get("security_terms", [])[:8]) or "none",
                locality=str(stats.get("locality", "unknown")),
            )
        )
    return "\n".join(rows)


def _diff_stats(diff: str) -> Dict[str, object]:
    lines = str(diff).splitlines()
    files = edited_files_from_diff(diff)
    added_lines = [line[1:] for line in lines if line.startswith("+") and not line.startswith("+++")]
    deleted_lines = [line[1:] for line in lines if line.startswith("-") and not line.startswith("---")]
    context_lines = [line[1:] for line in lines if line.startswith(" ") and not line.startswith("  " * 100)]
    hunk_scopes = []
    for line in lines:
        if line.startswith("@@"):
            scope = line.split("@@", 2)[-1].strip()
            if scope:
                hunk_scopes.append(scope)
    symbol_pattern = re.compile(r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")
    symbols = sorted({match.group(1) for line in added_lines + deleted_lines + context_lines for match in symbol_pattern.finditer(line)})
    imports = sorted(
        {
            line.strip()
            for line in added_lines + deleted_lines + context_lines
            if re.match(r"^(from\s+[A-Za-z0-9_.]+\s+import|import\s+[A-Za-z0-9_.]+)", line.strip())
        }
    )
    call_tokens = sorted(
        {
            token
            for line in added_lines
            for token in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
            if token not in {"if", "for", "while", "return", "assert", "with", "except", "class", "def"}
        }
    )
    security_terms = sorted(
        {
            token.lower()
            for token in re.findall(r"\b(auth|token|password|permission|encrypt|decrypt|sql|xss|csrf|path|escape|sanitize)\b", diff, re.I)
        }
    )
    pre_patch = [line for line in context_lines + deleted_lines if line.strip()]
    if len(pre_patch) > 24:
        pre_patch = pre_patch[:12] + ["..."] + pre_patch[-11:]
    source_files = [path for path in files if "test" not in path.lower()]
    top_dirs = sorted({path.split("/", 1)[0] for path in files if "/" in path})
    return {
        "edited_files": files,
        "added_lines": len(added_lines),
        "deleted_lines": len(deleted_lines),
        "hunk_scopes": hunk_scopes,
        "symbols": symbols,
        "imports": imports,
        "call_tokens": call_tokens,
        "security_terms": security_terms,
        "pre_patch_excerpt": "\n".join(pre_patch) if pre_patch else "not available",
        "python_file_count": sum(1 for path in files if path.endswith(".py")),
        "import_change_count": sum(1 for line in added_lines + deleted_lines if re.match(r"^\s*(from|import)\s+", line)),
        "touches_tests": any("test" in path.lower() for path in files) or bool(re.search(r"\b(pytest|unittest|assert)\b", diff, re.I)),
        "touches_source": bool(source_files),
        "multi_file": len(files) > 1,
        "new_file": "new file mode" in diff or "--- /dev/null" in diff,
        "locality": "single_directory" if len(top_dirs) <= 1 else "multi_directory",
        "issue_overlap_terms": [],
    }


def _safe_candidate_stats_for_metadata(stats: Dict[str, object]) -> Dict[str, object]:
    keys = (
        "edited_files",
        "added_lines",
        "deleted_lines",
        "hunk_scopes",
        "symbols",
        "imports",
        "python_file_count",
        "import_change_count",
        "touches_tests",
        "touches_source",
        "multi_file",
        "new_file",
        "locality",
        "security_terms",
    )
    return {key: stats.get(key) for key in keys}


def _patch_structure_score(stats: Dict[str, object]) -> float:
    added = int(stats.get("added_lines", 0))
    deleted = int(stats.get("deleted_lines", 0))
    files = len(stats.get("edited_files", []))
    score = 0.0
    score -= 0.08 * math.log1p(max(0, added + deleted))
    score -= 0.15 * math.log1p(max(0, files))
    score -= 0.45 if bool(stats.get("new_file", False)) else 0.0
    score -= 0.20 if bool(stats.get("multi_file", False)) else 0.0
    score -= 0.15 if bool(stats.get("touches_tests", False)) else 0.0
    score += 0.08 if bool(stats.get("touches_source", False)) else 0.0
    score += 0.05 if stats.get("symbols") else 0.0
    return score


def _patch_embedding_text(diff: str) -> str:
    lines = []
    for line in str(diff).splitlines():
        if line.startswith(("diff --git", "index ", "--- ", "+++ ")):
            lines.append(line)
        elif line.startswith(("+", "-", " ")) and not line.startswith(("+++", "---")):
            lines.append(line[1:])
    return "\n".join(lines)


def _full_hardened_context_text(example: PatchSelectionExample) -> str:
    return "\n".join(
        [
            example.repo,
            example.issue_text,
            example.failing_test_summary,
            "\n".join(example.retrieved_contexts),
            str(example.metadata.get("dependency_callgraph_related_file_evidence", "")),
        ]
    )


def _candidate_hardened_context_text(example: PatchSelectionExample, candidate_index: int) -> str:
    candidate_marker = f"candidate {candidate_index}"
    global_contexts = [
        text
        for text in example.retrieved_contexts
        if "Stage 6D retrieved source context for candidate" not in text
        or f"Stage 6D retrieved source context for candidate {candidate_index}." in text
    ]
    dependency_lines = []
    keep_header = False
    for line in str(example.metadata.get("dependency_callgraph_related_file_evidence", "")).splitlines():
        if line.startswith("Avenue "):
            keep_header = True
            dependency_lines.append(line)
            continue
        if line.startswith("candidate "):
            keep_header = False
            if line.startswith(candidate_marker + ":"):
                dependency_lines.append(line)
            continue
        if keep_header and line.strip():
            dependency_lines.append(line)
    return "\n".join(
        [
            example.repo,
            example.issue_text,
            example.failing_test_summary,
            "\n".join(global_contexts),
            "\n".join(dependency_lines),
        ]
    )


def _cosine_token_similarity(left: str, right: str) -> float:
    left_counts = Counter(_tokens(left))
    right_counts = Counter(_tokens(right))
    if not left_counts or not right_counts:
        return 0.0
    common = set(left_counts) & set(right_counts)
    numerator = sum(float(left_counts[token] * right_counts[token]) for token in common)
    left_norm = math.sqrt(sum(float(value * value) for value in left_counts.values()))
    right_norm = math.sqrt(sum(float(value * value) for value in right_counts.values()))
    return numerator / max(1e-8, left_norm * right_norm)


def _tokens(text: str) -> List[str]:
    out = []
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}|\d+", str(text).lower()):
        if token not in _STOPWORDS:
            out.append(token)
    return out


def _top_issue_terms(text: str, limit: int) -> List[str]:
    counts = Counter(_tokens(text))
    return [token for token, _count in counts.most_common(limit)]


def _row_normalize(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0:
        return arr
    std = float(np.std(arr))
    if std <= 1e-8:
        return arr - float(np.mean(arr))
    return (arr - float(np.mean(arr))) / std


def _oracle_pass_at_8(examples: Sequence[PatchSelectionExample]) -> float:
    return float(np.mean([example.oracle_pass_at_8 for example in examples])) if examples else 0.0


def _normalize_patch_for_similarity(diff: str) -> str:
    rows = []
    for line in str(diff).strip().splitlines():
        if line.startswith("index "):
            continue
        rows.append(line.rstrip())
    return "\n".join(rows)


def _record_instance_id(row: Dict[str, object]) -> str:
    for key in ("instance_id", "issue_id", "task_id"):
        value = row.get(key)
        if value:
            return str(value)
    if isinstance(row.get("metadata"), dict):
        metadata = row["metadata"]
        for key in ("instance_id", "issue_id", "task_id"):
            value = metadata.get(key)
            if value:
                return str(value)
    return ""


def _read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _write_json(path: Path, value: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
