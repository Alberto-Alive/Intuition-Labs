from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from src.datasets.swe_patch_selection_dataset import (
    STAGE6_NUM_CANDIDATES,
    PatchSelectionExample,
    duplicate_candidate_patch_hash_audit,
    label_matrix,
    load_patch_selection_jsonl,
    stage6_output_leakage_audit,
    stage6_split_leakage_audit,
    validate_patch_selection_examples,
)
from src.experiments.run_stage6_latent_patch_selector import (
    BENCHMARK as STAGE6_SELECTOR_BENCHMARK,
    DEFAULT_CHECKPOINT_DIR,
    _phase_architectures,
    _phase_config,
    _run_seed,
    paired_bootstrap_ci,
    paired_permutation_pvalue,
)
from src.experiments.run_stage6b1_candidate_generation import DEFAULT_DATASET_NAME, load_benchmark_tasks


BENCHMARK = "stage6c_mini_selector_validation"
DEFAULT_INPUT_PATH = Path("results/stage6b3_completed_labeled_candidate_pools.jsonl")
DEFAULT_RESULTS_PATH = Path("results/stage6c_mini_selector_results.json")
DEFAULT_AUDIT_PATH = Path("results/stage6c_mini_selector_audit.jsonl")
DEFAULT_SPLITS_PATH = Path("results/stage6c_mini_splits.json")
DEFAULT_REPORT_PATH = Path("reports/STAGE6C_MINI_SELECTOR_VALIDATION.md")
DEFAULT_SEED = 606_300


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6C-mini selector validation on a tiny officially labeled complete-label subset.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT_PATH))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR / "stage6c_mini"))
    args = parser.parse_args()

    result = run_stage6c_mini_selector_validation(
        input_path=Path(args.input),
        results_path=Path(args.results),
        audit_path=Path(args.audit),
        splits_path=Path(args.splits),
        report_path=Path(args.report),
        dataset_name=str(args.dataset_name),
        split=str(args.split),
        seed=int(args.seed),
        folds=int(args.folds),
        device=str(args.device),
        checkpoint_dir=Path(args.checkpoint_dir),
    )
    summary = result.get("summary", {})
    print(
        "stage6c-mini: wrote {results}, {audit}, {splits}, {report}; folds={folds}; "
        "trainable_mean_pass_at_1={trainable:.4f}; success={success}".format(
            results=args.results,
            audit=args.audit,
            splits=args.splits,
            report=args.report,
            folds=summary.get("folds_completed", 0),
            trainable=float(summary.get("mean_metrics", {}).get("trainable_shared_weight_latent_selector", {}).get("pass_at_1", 0.0)),
            success=summary.get("diagnostic_success", False),
        )
    )


def run_stage6c_mini_selector_validation(
    input_path: Path,
    results_path: Path,
    audit_path: Path,
    splits_path: Path,
    report_path: Path,
    dataset_name: str,
    split: str,
    seed: int,
    folds: int,
    device: str,
    checkpoint_dir: Path,
) -> Dict[str, object]:
    created_at = _now()
    examples = [example for example in load_patch_selection_jsonl(input_path) if len(example.labels_pass_fail) == STAGE6_NUM_CANDIDATES]
    validation = validate_patch_selection_examples(examples, allow_gold_diagnostic=False)
    fold_defs = make_stratified_folds(examples, folds=folds, seed=seed)
    split_rows = build_cv_splits(examples, fold_defs)
    _write_json(splits_path, split_rows)
    config = _phase_config("6B-real")
    config = type(config)(**{**config.__dict__, "phase": "6C-mini", "device": device, "run_controls": True, "save_checkpoints": False})
    architecture = _phase_architectures("6B-real")[0]
    fold_rows = []
    audit_rows = []
    for fold in split_rows["folds"]:
        fold_id = int(fold["fold"])
        splits = {
            "train": [examples[index] for index in fold["train_indices"]],
            "dev": [examples[index] for index in fold["dev_indices"]],
            "test": [examples[index] for index in fold["test_indices"]],
        }
        fold_seed = int(seed + fold_id * 101)
        split_leakage = stage6_split_leakage_audit(BENCHMARK, fold_seed, splits)
        output_leakage = stage6_output_leakage_audit(BENCHMARK, fold_seed, [item for rows in splits.values() for item in rows])
        duplicate_audit = duplicate_candidate_patch_hash_audit([item for rows in splits.values() for item in rows])
        row = _run_seed(
            config=config,
            architecture=architecture,
            splits=splits,
            seed=fold_seed,
            checkpoint_dir=checkpoint_dir / f"fold_{fold_id}",
            split_leakage=split_leakage,
            output_leakage=output_leakage,
            duplicate_audit=duplicate_audit,
        )
        row["fold"] = fold_id
        row["fold_split_ids"] = {
            name: [example.id for example in rows]
            for name, rows in splits.items()
        }
        row["fold_oracle_positive_counts"] = {
            name: sum(1 for example in rows if example.oracle_pass_at_8)
            for name, rows in splits.items()
        }
        fold_rows.append(row)
        audit_rows.extend(
            [
                {"benchmark": BENCHMARK, "seed": fold_seed, "fold": fold_id, "type": "split_definition", **row["fold_split_ids"]},
                {"benchmark": BENCHMARK, "seed": fold_seed, "fold": fold_id, "type": "split_leakage", **split_leakage},
                {"benchmark": BENCHMARK, "seed": fold_seed, "fold": fold_id, "type": "output_leakage", **output_leakage},
                {"benchmark": BENCHMARK, "seed": fold_seed, "fold": fold_id, "type": "duplicate_candidate_patch_hash", **duplicate_audit},
                {"benchmark": BENCHMARK, "seed": fold_seed, "fold": fold_id, "type": "trainable_training_audit", **row.get("trainable_audit", {})},
                {"benchmark": BENCHMARK, "seed": fold_seed, "fold": fold_id, "type": "frozen_training_audit", **row.get("frozen_audit", {})},
                {"benchmark": BENCHMARK, "seed": fold_seed, "fold": fold_id, "type": "invariance_audit", **row.get("invariance_audit", {})},
            ]
        )
    task_records, benchmark_audit = load_benchmark_tasks(
        dataset_name=dataset_name,
        split=split,
        benchmark_jsonl=None,
        requested_task_ids=[example.issue_id for example in examples],
        task_ids_file=None,
        n_tasks=len(examples),
        seed=seed,
    )
    external_audits = {
        "dataset_validation": validation,
        "output_leakage_audit": stage6_output_leakage_audit(BENCHMARK, seed, examples),
        "duplicate_candidate_patch_hash_audit": duplicate_candidate_patch_hash_audit(examples),
        "exact_reference_patch_hash_audit": exact_reference_hash_audit(examples, task_records),
        "source_generator_leakage_audit": source_generator_leakage_audit(examples, fold_rows),
    }
    summary = build_summary(examples, fold_rows, external_audits, seed)
    result = {
        "metadata": {
            "benchmark": BENCHMARK,
            "created_at_utc": created_at,
            "input_path": str(input_path),
            "selector_scope": "Stage 6C-mini diagnostic selector-only validation; no patch generation claim",
            "candidate_pool_source": "Stage 6B.3 complete-label subset",
            "architecture": architecture.__dict__,
            "stage6_config": config.__dict__,
            "split_seed": int(seed),
            "fold_count": int(folds),
            "dataset_name": dataset_name,
            "split": split,
            "benchmark_dataset_audit": benchmark_audit,
            "no_architecture_tuning_on_test_folds": True,
            "not_publishable_final_claim": True,
        },
        "folds": fold_rows,
        "audits": external_audits,
        "summary": summary,
    }
    _write_json(results_path, result)
    _write_jsonl(audit_path, audit_rows + [{"benchmark": BENCHMARK, "type": key, **value} for key, value in external_audits.items()])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_stage6c_mini_report(result), encoding="utf-8")
    return result


def make_stratified_folds(examples: Sequence[PatchSelectionExample], folds: int, seed: int) -> List[List[int]]:
    if not examples:
        return []
    fold_count = max(2, min(int(folds), len(examples)))
    positives = [index for index, example in enumerate(examples) if example.oracle_pass_at_8]
    negatives = [index for index, example in enumerate(examples) if not example.oracle_pass_at_8]
    rng = np.random.default_rng(seed + 6_300_101)
    positives = [int(value) for value in rng.permutation(positives).tolist()]
    negatives = [int(value) for value in rng.permutation(negatives).tolist()]
    buckets: List[List[int]] = [[] for _ in range(fold_count)]
    for offset, index in enumerate(positives):
        buckets[offset % fold_count].append(index)
    for offset, index in enumerate(negatives):
        buckets[offset % fold_count].append(index)
    return [sorted(bucket) for bucket in buckets if bucket]


def build_cv_splits(examples: Sequence[PatchSelectionExample], fold_defs: Sequence[Sequence[int]]) -> Dict[str, object]:
    all_indices = set(range(len(examples)))
    rows = []
    fold_count = len(fold_defs)
    for fold_id, test_indices_raw in enumerate(fold_defs):
        test_indices = set(int(value) for value in test_indices_raw)
        dev_indices = set(int(value) for value in fold_defs[(fold_id + 1) % fold_count])
        train_indices = sorted(all_indices - test_indices - dev_indices)
        rows.append(
            {
                "fold": fold_id,
                "train_indices": train_indices,
                "dev_indices": sorted(dev_indices),
                "test_indices": sorted(test_indices),
                "train_ids": [examples[index].id for index in train_indices],
                "dev_ids": [examples[index].id for index in sorted(dev_indices)],
                "test_ids": [examples[index].id for index in sorted(test_indices)],
                "train_oracle_positive": sum(1 for index in train_indices if examples[index].oracle_pass_at_8),
                "dev_oracle_positive": sum(1 for index in dev_indices if examples[index].oracle_pass_at_8),
                "test_oracle_positive": sum(1 for index in test_indices if examples[index].oracle_pass_at_8),
            }
        )
    return {
        "strategy": "5-fold task-level cross-validation with next fold used as dev; no task appears in train and test in the same fold",
        "task_count": len(examples),
        "oracle_positive_tasks": sum(1 for example in examples if example.oracle_pass_at_8),
        "folds": rows,
    }


def build_summary(
    examples: Sequence[PatchSelectionExample],
    fold_rows: Sequence[Dict[str, object]],
    external_audits: Dict[str, object],
    seed: int,
) -> Dict[str, object]:
    method_values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    control_values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    per_task_correct: Dict[str, List[bool]] = defaultdict(list)
    for row in fold_rows:
        metrics = row.get("metrics", {}).get("test", {})
        test_ids = row.get("fold_split_ids", {}).get("test", [])
        test_examples_by_id = {example.id: example for example in examples if example.id in set(test_ids)}
        ordered_test = [test_examples_by_id[item] for item in test_ids if item in test_examples_by_id]
        labels = label_matrix(ordered_test)
        for method, metric in metrics.items():
            for key in ("pass_at_1", "conditional_selector_accuracy", "oracle_pass_at_8", "selection_efficiency", "mrr", "top_2_accuracy"):
                method_values[method][key].append(float(metric.get(key, 0.0)))
            predictions = list(metric.get("predictions", []))
            if len(predictions) == len(ordered_test):
                per_task_correct[method].extend(bool(labels[index, int(choice)] > 0) for index, choice in enumerate(predictions))
        for control, metric in row.get("controls", {}).items():
            for key in ("pass_at_1", "conditional_selector_accuracy", "selection_efficiency", "mrr", "top_2_accuracy"):
                control_values[control][key].append(float(metric.get(key, 0.0)))
        randomized = metrics.get("randomized_labels_trainable_latent_selector")
        if randomized:
            control_values["randomized_labels"]["pass_at_1"].append(float(randomized.get("pass_at_1", 0.0)))
    mean_metrics = {
        method: {key: _mean(values) for key, values in keys.items()}
        for method, keys in sorted(method_values.items())
    }
    mean_controls = {
        control: {key: _mean(values) for key, values in keys.items()}
        for control, keys in sorted(control_values.items())
    }
    simple_baselines = ("random_candidate", "first_candidate_order_baseline", "best_generator_on_dev_baseline", "visible_test_heuristic")
    non_oracle_methods = [name for name in mean_metrics if name not in {"trainable_shared_weight_latent_selector", "oracle_pass_at_8"}]
    best_simple = max(simple_baselines, key=lambda name: mean_metrics.get(name, {}).get("pass_at_1", 0.0))
    best_non_oracle = max(non_oracle_methods, key=lambda name: mean_metrics.get(name, {}).get("pass_at_1", 0.0)) if non_oracle_methods else None
    trainable_correct = np.asarray(per_task_correct.get("trainable_shared_weight_latent_selector", []), dtype=np.float64)
    best_simple_correct = np.asarray(per_task_correct.get(best_simple, []), dtype=np.float64)
    paired = {}
    if trainable_correct.size and trainable_correct.size == best_simple_correct.size:
        diff = trainable_correct - best_simple_correct
        paired = {
            "comparison": f"trainable_shared_weight_latent_selector_vs_{best_simple}",
            "mean_delta": float(np.mean(diff)),
            "paired_bootstrap_95ci": list(paired_bootstrap_ci(diff, seed=seed + 17)),
            "paired_permutation_pvalue": paired_permutation_pvalue(trainable_correct, best_simple_correct, seed=seed + 18),
        }
    diagnostic_gates = diagnostic_success_gates(mean_metrics, mean_controls, fold_rows, external_audits, best_simple)
    return {
        "task_count": len(examples),
        "oracle_positive_tasks": sum(1 for example in examples if example.oracle_pass_at_8),
        "oracle_empty_tasks": sum(1 for example in examples if not example.oracle_pass_at_8),
        "folds_completed": len(fold_rows),
        "mean_metrics": mean_metrics,
        "mean_controls": mean_controls,
        "best_simple_non_oracle_baseline": best_simple,
        "best_non_oracle_method": best_non_oracle,
        "paired_tests_vs_best_simple_baseline": paired,
        "diagnostic_gates": diagnostic_gates,
        "diagnostic_success": bool(all(diagnostic_gates.values())),
        "interpretation": (
            "Stage 6C-mini shows preliminary selector signal on a tiny officially labeled SWE-bench candidate-selection subset."
            if all(diagnostic_gates.values())
            else "Stage 6C-mini did not show reliable selector signal; larger official labeled candidate pools or stronger candidate/view construction are needed."
        ),
        "failure_diagnosis": failure_diagnosis(examples, mean_metrics, mean_controls, diagnostic_gates),
    }


def diagnostic_success_gates(
    mean_metrics: Dict[str, Dict[str, float]],
    mean_controls: Dict[str, Dict[str, float]],
    fold_rows: Sequence[Dict[str, object]],
    external_audits: Dict[str, object],
    best_simple: str,
) -> Dict[str, bool]:
    trainable = float(mean_metrics.get("trainable_shared_weight_latent_selector", {}).get("pass_at_1", 0.0))
    frozen = float(mean_metrics.get("frozen_same_architecture_latent_selector", {}).get("pass_at_1", 0.0))
    random_value = float(mean_metrics.get("random_candidate", {}).get("pass_at_1", 0.0))
    best_simple_value = float(mean_metrics.get(best_simple, {}).get("pass_at_1", 0.0))
    randomized = float(mean_controls.get("randomized_labels", {}).get("pass_at_1", 1.0))
    mismatch = float(mean_controls.get("candidate_evidence_mismatch", {}).get("pass_at_1", trainable))
    invariance = [row.get("invariance_audit", {}) for row in fold_rows]
    return {
        "trainable_latent_executes_on_all_folds": bool(fold_rows and all(row.get("status") == "completed" for row in fold_rows)),
        "trainable_beats_frozen_mean_pass_at_1": trainable > frozen,
        "trainable_beats_best_simple_non_oracle_baseline_mean_pass_at_1": trainable > best_simple_value,
        "trainable_beats_random_mean_pass_at_1": trainable > random_value,
        "randomized_labels_collapse_toward_chance": randomized <= max(random_value + 0.10, 0.20),
        "candidate_evidence_mismatch_degrades_substantially": mismatch <= max(0.0, trainable - 0.05),
        "candidate_order_invariance_passes": bool(invariance and all(item.get("candidate_order_invariance_passes", False) for item in invariance)),
        "physical_order_invariance_passes": bool(invariance and all(item.get("physical_order_invariance_passes", False) for item in invariance)),
        "no_leakage_audits_fail": bool(
            external_audits.get("output_leakage_audit", {}).get("passes", False)
            and external_audits.get("source_generator_leakage_audit", {}).get("passes", False)
            and external_audits.get("duplicate_candidate_patch_hash_audit", {}).get("passes", False)
            and external_audits.get("exact_reference_patch_hash_audit", {}).get("passes", False)
        ),
        "trainable_gradient_audits_pass": bool(
            fold_rows
            and all(float(row.get("trainable_audit", {}).get("agent_grad_norm_mean") or 0.0) > 0.0 for row in fold_rows)
            and all(float(row.get("trainable_audit", {}).get("agent_parameter_delta") or 0.0) > 0.0 for row in fold_rows)
            and all(float(row.get("trainable_audit", {}).get("coordinator_grad_norm_mean") or 0.0) > 0.0 for row in fold_rows)
            and all(float(row.get("trainable_audit", {}).get("coordinator_parameter_delta") or 0.0) > 0.0 for row in fold_rows)
        ),
        "frozen_gradient_audits_pass": bool(
            fold_rows
            and all(float(row.get("frozen_audit", {}).get("agent_grad_norm_mean") or 0.0) == 0.0 for row in fold_rows)
            and all(float(row.get("frozen_audit", {}).get("agent_parameter_delta") or 0.0) == 0.0 for row in fold_rows)
            and all(float(row.get("frozen_audit", {}).get("coordinator_grad_norm_mean") or 0.0) > 0.0 for row in fold_rows)
        ),
        "shared_identity_and_no_detach_audits_pass": bool(
            fold_rows
            and all(bool(row.get("trainable_audit", {}).get("shared_parameter_identity")) for row in fold_rows)
            and all(bool(row.get("trainable_audit", {}).get("activation_requires_grad_before_coordinator")) for row in fold_rows)
            and all(bool(row.get("trainable_audit", {}).get("no_detach_between_clone_activations_and_loss")) for row in fold_rows)
        ),
    }


def failure_diagnosis(
    examples: Sequence[PatchSelectionExample],
    mean_metrics: Dict[str, Dict[str, float]],
    mean_controls: Dict[str, Dict[str, float]],
    gates: Dict[str, bool],
) -> List[str]:
    if all(gates.values()):
        return []
    reasons = []
    positives = sum(1 for example in examples if example.oracle_pass_at_8)
    if positives < 10:
        reasons.append("too few oracle-positive train/dev/test examples for stable selector validation")
    if len(examples) < 50:
        reasons.append("complete-label pool is tiny")
    generator_counts = Counter(candidate.candidate_source_agent for example in examples for candidate in example.candidates)
    if len(generator_counts) <= 2:
        reasons.append("generator/source skew is high")
    trainable = mean_metrics.get("trainable_shared_weight_latent_selector", {}).get("pass_at_1", 0.0)
    patch_only = mean_controls.get("patch_only", {}).get("pass_at_1", 0.0)
    candidate_only = mean_controls.get("candidate_only", {}).get("pass_at_1", 0.0)
    if patch_only >= trainable or candidate_only >= trainable:
        reasons.append("candidate/view construction may let patch-only or candidate-only views capture most signal")
    if not gates.get("candidate_evidence_mismatch_degrades_substantially", True):
        reasons.append("candidate/evidence mismatch did not degrade enough, suggesting weak issue-context use")
    if not gates.get("trainable_beats_frozen_mean_pass_at_1", True):
        reasons.append("trainable architecture did not beat frozen same-architecture comparator")
    if not gates.get("trainable_beats_best_simple_non_oracle_baseline_mean_pass_at_1", True):
        reasons.append("simple baseline matched or beat trainable selector")
    return reasons


def exact_reference_hash_audit(
    examples: Sequence[PatchSelectionExample],
    task_records: Dict[str, object],
) -> Dict[str, object]:
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


def source_generator_leakage_audit(
    examples: Sequence[PatchSelectionExample],
    fold_rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    forbidden = ("logs/run_evaluation", "stage6b2_official_harness", "report.json", "test_output.txt", "labels_pass_fail", "oracle_pass_at_8")
    hits = []
    for example in examples:
        visible_chunks = [
            example.issue_text,
            example.failing_test_summary,
            "\n".join(example.retrieved_contexts),
            str(example.metadata.get("dependency_callgraph_related_file_evidence", "")),
        ]
        visible_chunks.extend(f"{candidate.candidate_source_agent}\n{candidate.candidate_visible_test_result}\n{candidate.candidate_diff}" for candidate in example.candidates)
        text = "\n".join(visible_chunks).lower()
        row_hits = [pattern for pattern in forbidden if pattern in text]
        if row_hits:
            hits.append({"example_id": example.id, "issue_id": example.issue_id, "hits": row_hits})
    best_generator_equals_oracle = []
    best_values = []
    oracle_values = []
    for row in fold_rows:
        metrics = row.get("metrics", {}).get("test", {})
        best = float(metrics.get("best_generator_on_dev_baseline", {}).get("pass_at_1", 0.0))
        oracle = float(metrics.get("oracle_pass_at_8", {}).get("oracle_pass_at_8", 0.0))
        best_values.append(best)
        oracle_values.append(oracle)
        if abs(best - oracle) <= 1e-12 and oracle > 0.0:
            best_generator_equals_oracle.append({"fold": row.get("fold"), "best_generator_pass_at_1": best, "oracle_pass_at_8": oracle})
    aggregate_best = _mean(best_values)
    aggregate_oracle = _mean(oracle_values)
    return {
        "selector_visible_forbidden_source_or_label_hits": hits,
        "best_generator_baseline_equals_oracle_fold_warnings": best_generator_equals_oracle,
        "aggregate_best_generator_pass_at_1": aggregate_best,
        "aggregate_oracle_pass_at_8": aggregate_oracle,
        "aggregate_best_generator_equals_oracle": bool(abs(aggregate_best - aggregate_oracle) <= 1e-12 and aggregate_oracle > 0.0),
        "generator_identity_visible_as_candidate_source_agent": True,
        "passes": not hits and not (abs(aggregate_best - aggregate_oracle) <= 1e-12 and aggregate_oracle > 0.0),
    }


def render_stage6c_mini_report(result: Dict[str, object]) -> str:
    summary = result.get("summary", {})
    mean_metrics = summary.get("mean_metrics", {})
    mean_controls = summary.get("mean_controls", {})
    gates = summary.get("diagnostic_gates", {})
    paired = summary.get("paired_tests_vs_best_simple_baseline", {})
    lines = [
        "# Stage 6C-mini Selector Validation",
        "",
        "## Scope",
        "",
        "- Diagnostic selector-only validation on a tiny officially labeled complete-label subset.",
        "- Candidate patches are fixed generated candidates from Stage 6B.1b/6B.3.",
        "- The existing Stage 6 candidate-token-direct architecture is used unchanged.",
        "- No final SWE-bench improvement, publishable result, or end-to-end patch generation claim is made.",
        "",
        "## Dataset",
        "",
        f"- Complete tasks: `{summary.get('task_count')}`.",
        f"- Oracle-positive tasks: `{summary.get('oracle_positive_tasks')}`.",
        f"- Oracle-empty tasks: `{summary.get('oracle_empty_tasks')}`.",
        f"- Folds completed: `{summary.get('folds_completed')}`.",
        "",
        "## Mean Metrics",
        "",
        "| method | pass@1 | conditional accuracy | oracle pass@8 | efficiency | MRR | top-2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, metrics in mean_metrics.items():
        lines.append(
            "| `{}` | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                method,
                float(metrics.get("pass_at_1", 0.0)),
                float(metrics.get("conditional_selector_accuracy", 0.0)),
                float(metrics.get("oracle_pass_at_8", 0.0)),
                float(metrics.get("selection_efficiency", 0.0)),
                float(metrics.get("mrr", 0.0)),
                float(metrics.get("top_2_accuracy", 0.0)),
            )
        )
    lines.extend(["", "## Controls", "", "| control | pass@1 | conditional accuracy | efficiency |", "|---|---:|---:|---:|"])
    for control, metrics in mean_controls.items():
        lines.append(
            "| `{}` | {:.4f} | {:.4f} | {:.4f} |".format(
                control,
                float(metrics.get("pass_at_1", 0.0)),
                float(metrics.get("conditional_selector_accuracy", 0.0)),
                float(metrics.get("selection_efficiency", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Paired Test",
            "",
            f"- Comparison: `{paired.get('comparison')}`.",
            f"- Mean delta: `{float(paired.get('mean_delta', 0.0)):.4f}`.",
            f"- Bootstrap 95% CI: `{paired.get('paired_bootstrap_95ci', [])}`.",
            f"- Paired permutation p-value: `{paired.get('paired_permutation_pvalue')}`.",
            "",
            "## Diagnostic Gates",
            "",
            "| gate | pass |",
            "|---|---:|",
        ]
    )
    for key, value in gates.items():
        lines.append(f"| `{key}` | `{bool(value)}` |")
    lines.extend(["", "## Diagnosis", ""])
    diagnosis = summary.get("failure_diagnosis", [])
    if diagnosis:
        for item in diagnosis:
            lines.append(f"- {item}.")
    else:
        lines.append("- Diagnostic gates passed.")
    lines.extend(["", "## Conclusion", "", str(summary.get("interpretation", "No final selector claim is made.")), ""])
    return "\n".join(lines)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


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
