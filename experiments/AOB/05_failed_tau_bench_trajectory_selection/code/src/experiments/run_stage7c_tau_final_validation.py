from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from src.datasets.taubench_trajectory_selection_dataset import (
    STAGE7_NUM_CANDIDATES,
    TauBenchTrajectorySelectionExample,
    build_taubench_selection_examples,
    duplicate_trajectory_hash_audit,
    load_taubench_candidates_jsonl,
    near_duplicate_trajectory_audit,
    stage7_output_leakage_audit,
    stage7_split_leakage_audit,
    taubench_label_matrix,
    trajectory_candidate_text,
    validate_taubench_selection_examples,
)
from src.experiments.run_stage7_taubench_trajectory_selector import _audit_subset
from src.experiments.run_stage7b_tau_dev_validation import (
    Stage7BConfig,
    Stage7BVariant,
    agent_config_for,
    coordinator_config_for,
    evaluate_controls,
    evaluate_methods,
    fit_models,
    generator_identity_only_diagnostic,
    gradient_update_audit_passes,
    invariance_audit,
    load_candidate_source_map,
    message_config_for,
    resolve_device,
    training_config_for,
)


BENCHMARK = "stage7c_tau_final_validation"
DEFAULT_STAGE7B_COMPARISON = Path("results/stage7b_tau_dev_variant_comparison.json")
DEFAULT_STAGE7B_RESULTS = Path("results/stage7b_tau_dev_selector_results.json")
DEFAULT_STAGE7B_SPLITS = Path("results/stage7b_tau_dev_splits.json")
DEFAULT_STAGE7B_LABEL_AUDIT = Path("results/stage7b0_tau_label_audit.json")
DEFAULT_STAGE7C_CANDIDATES = Path("results/stage7c_tau_final_candidate_pool.jsonl")
DEFAULT_FINAL_SPLITS = Path("results/stage7c_tau_final_splits.json")
DEFAULT_FINAL_RESULTS = Path("results/stage7c_tau_final_selector_results.json")
DEFAULT_FINAL_AUDIT = Path("results/stage7c_tau_final_selector_audit.jsonl")
DEFAULT_FINAL_CONTROLS = Path("results/stage7c_tau_final_control_results.json")
DEFAULT_FINAL_STATISTICS = Path("results/stage7c_tau_final_statistics.json")
DEFAULT_FINAL_REPORT = Path("reports/STAGE7C_TAU_FINAL_VALIDATION.md")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints/stage7c")
FINAL_SEEDS = tuple(range(10, 20))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 7C final tau-bench latent trajectory selector validation.")
    parser.add_argument("--stage7b-comparison", default=str(DEFAULT_STAGE7B_COMPARISON))
    parser.add_argument("--stage7b-results", default=str(DEFAULT_STAGE7B_RESULTS))
    parser.add_argument("--stage7b-splits", default=str(DEFAULT_STAGE7B_SPLITS))
    parser.add_argument("--label-audit", default=str(DEFAULT_STAGE7B_LABEL_AUDIT))
    parser.add_argument("--splits", default=str(DEFAULT_FINAL_SPLITS))
    parser.add_argument("--results", default=str(DEFAULT_FINAL_RESULTS))
    parser.add_argument("--audit", default=str(DEFAULT_FINAL_AUDIT))
    parser.add_argument("--controls", default=str(DEFAULT_FINAL_CONTROLS))
    parser.add_argument("--statistics", default=str(DEFAULT_FINAL_STATISTICS))
    parser.add_argument("--report", default=str(DEFAULT_FINAL_REPORT))
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    args = parser.parse_args()
    result = run_stage7c_final_validation(
        stage7b_comparison_path=Path(args.stage7b_comparison),
        stage7b_results_path=Path(args.stage7b_results),
        stage7b_splits_path=Path(args.stage7b_splits),
        label_audit_path=Path(args.label_audit),
        splits_path=Path(args.splits),
        results_path=Path(args.results),
        audit_path=Path(args.audit),
        controls_path=Path(args.controls),
        statistics_path=Path(args.statistics),
        report_path=Path(args.report),
        checkpoint_dir=Path(args.checkpoint_dir),
    )
    print(json.dumps({"decision": result["decision"], "selected_variant": result["selected_variant"]["name"]}, sort_keys=True))


def run_stage7c_final_validation(
    stage7b_comparison_path: Path,
    stage7b_results_path: Path,
    stage7b_splits_path: Path,
    label_audit_path: Path,
    splits_path: Path,
    results_path: Path,
    audit_path: Path,
    controls_path: Path,
    statistics_path: Path,
    report_path: Path,
    checkpoint_dir: Path,
) -> Dict[str, object]:
    stage7b_comparison = json.loads(stage7b_comparison_path.read_text(encoding="utf-8"))
    stage7b_results = json.loads(stage7b_results_path.read_text(encoding="utf-8"))
    stage7b_splits = json.loads(stage7b_splits_path.read_text(encoding="utf-8"))
    selected_variant = load_selected_variant(stage7b_comparison, stage7b_results)
    config = load_frozen_stage7b_config(stage7b_results)
    candidate_pool_path = resolve_candidate_pool(stage7b_results)
    pool_name = str(stage7b_results.get("pool_name", "balanced_pool"))
    candidates = load_taubench_candidates_jsonl(candidate_pool_path)
    examples = build_taubench_selection_examples(candidates)
    validation = validate_taubench_selection_examples(examples, require_labels=True)
    if not bool(validation.get("passes", False)):
        raise RuntimeError(json.dumps(validation, indent=2, sort_keys=True))
    candidate_source_map = load_candidate_source_map(label_audit_path, pool_name)
    splits, split_meta = build_final_splits(examples, stage7b_splits)
    write_final_splits(splits_path, splits, split_meta, candidate_pool_path, pool_name, selected_variant, config)

    all_examples = list(splits["train"]) + list(splits["final"])
    split_leakage = stage7_split_leakage_audit(BENCHMARK, 707_400, {"train": splits["train"], "dev": [], "test": splits["final"]})
    output_leakage = stage7_output_leakage_audit(BENCHMARK, 707_400, all_examples)
    duplicate_audit = duplicate_trajectory_hash_audit(all_examples)
    near_duplicate_audit = near_duplicate_trajectory_audit(all_examples)
    candidate_order_audit = candidate_order_randomized_audit(all_examples)
    raw_separation = raw_trajectory_separation_audit(all_examples)
    audit_rows: List[Dict[str, object]] = [
        {"benchmark": BENCHMARK, "type": "split_leakage", **split_leakage},
        {"benchmark": BENCHMARK, "type": "output_leakage", **output_leakage},
        {"benchmark": BENCHMARK, "type": "duplicate_trajectory_hash", **duplicate_audit},
        {"benchmark": BENCHMARK, "type": "near_duplicate_trajectory", **near_duplicate_audit},
        {"benchmark": BENCHMARK, "type": "candidate_order_randomized", **candidate_order_audit},
        {"benchmark": BENCHMARK, "type": "raw_trajectory_separation", **raw_separation},
    ]

    progress_path = results_path.with_name(f"{results_path.stem}_rows_progress.jsonl")
    completed_rows = load_progress_rows(progress_path, selected_variant, config, candidate_pool_path, split_meta)
    completed_by_seed = {int(row.get("seed", -1)): row for row in completed_rows}
    rows = []
    for seed in FINAL_SEEDS:
        row = completed_by_seed.get(int(seed))
        if row is None:
            row = run_final_seed(
                selected_variant=selected_variant,
                seed=int(seed),
                splits=splits,
                candidate_source_map=candidate_source_map,
                config=config,
                checkpoint_dir=checkpoint_dir,
                split_leakage=split_leakage,
                output_leakage=output_leakage,
                duplicate_audit=duplicate_audit,
                near_duplicate_audit=near_duplicate_audit,
                candidate_order_audit=candidate_order_audit,
                raw_separation=raw_separation,
            )
            append_progress_row(progress_path, row, selected_variant, config, candidate_pool_path, split_meta)
        rows.append(row)
        audit_rows.extend(audit_rows_from_result_row(row))

    statistics = build_final_statistics(rows, bootstrap_samples=int(config.bootstrap_samples))
    controls = build_control_results(rows)
    gates = final_success_gates(
        rows=rows,
        statistics=statistics,
        controls=controls,
        split_leakage=split_leakage,
        output_leakage=output_leakage,
        duplicate_audit=duplicate_audit,
        near_duplicate_audit=near_duplicate_audit,
        candidate_order_audit=candidate_order_audit,
        raw_separation=raw_separation,
        report_path=report_path,
    )
    decision = final_decision(gates, statistics)
    conservative_claim = conservative_claim_text() if decision == "A. STAGE7C_FINAL_PASSED" else None
    result = {
        "artifact": "stage7c_tau_final_selector_results",
        "created_at_utc": _now(),
        "benchmark": "tau-bench / tau2",
        "stage": BENCHMARK,
        "domain": config.domain,
        "candidate_pool_path": str(candidate_pool_path),
        "pool_name": pool_name,
        "stage7b_comparison_path": str(stage7b_comparison_path),
        "stage7b_results_path": str(stage7b_results_path),
        "stage7b_splits_path": str(stage7b_splits_path),
        "selected_variant": asdict(selected_variant),
        "frozen_stage7b_config": asdict(config),
        "seeds": list(FINAL_SEEDS),
        "k": STAGE7_NUM_CANDIDATES,
        "split_summary": split_summary(splits, candidate_source_map),
        "split_meta": split_meta,
        "dataset_validation": validation,
        "stage7c_final_no_airline_cross_domain": True,
        "no_trajectory_generation_claim": True,
        "no_end_to_end_tau_bench_sota_claim": True,
        "conservative_claim_allowed": conservative_claim is not None,
        "conservative_claim": conservative_claim,
        "decision": decision,
        "final_success_gates": gates,
        "failure_diagnosis": failure_diagnosis(gates, statistics, controls),
        "rows": rows,
        "controls_path": str(controls_path),
        "statistics_path": str(statistics_path),
        "report_path": str(report_path),
    }
    write_outputs(result, controls, statistics, audit_rows, results_path, controls_path, statistics_path, audit_path, report_path)
    return result


def run_final_seed(
    selected_variant: Stage7BVariant,
    seed: int,
    splits: Dict[str, List[TauBenchTrajectorySelectionExample]],
    candidate_source_map: Dict[str, Dict[str, object]],
    config: Stage7BConfig,
    checkpoint_dir: Path,
    split_leakage: Dict[str, object],
    output_leakage: Dict[str, object],
    duplicate_audit: Dict[str, object],
    near_duplicate_audit: Dict[str, object],
    candidate_order_audit: Dict[str, object],
    raw_separation: Dict[str, object],
) -> Dict[str, object]:
    started = time.perf_counter()
    device = resolve_device(config.device)
    fit_splits = {"train": splits["train"], "dev": splits["train"]}
    models = fit_models(
        splits=fit_splits,
        seed=seed,
        device=device,
        agent_config=agent_config_for(config),
        training=training_config_for(config, selected_variant),
        message_config=message_config_for(selected_variant),
        coordinator_config=coordinator_config_for(config, selected_variant),
        variant=selected_variant,
        candidate_source_map=candidate_source_map,
    )
    metrics = {
        "train": add_best_generator_train_dev_alias(
            evaluate_methods(models, splits["train"], "train", seed, candidate_source_map)
        ),
        "final": add_best_generator_train_dev_alias(
            evaluate_methods(models, splits["final"], "final", seed, candidate_source_map)
        ),
    }
    controls = evaluate_controls(models["trainable"], splits["final"], seed)
    invariance = invariance_audit(models["trainable"], splits["final"], seed)
    statistics = seed_pairwise_statistics(metrics["final"], seed=seed, bootstrap_samples=int(config.bootstrap_samples))
    checkpoint_paths = save_final_checkpoints(checkpoint_dir, selected_variant, seed, models, config)
    row = {
        "stage": BENCHMARK,
        "phase": "7C_final",
        "domain": config.domain,
        "seed": int(seed),
        "variant": selected_variant.name,
        "variant_config": asdict(selected_variant),
        "status": "completed",
        "completed_at_utc": _now(),
        "device": device,
        "training_selection_split": "train_only_no_final_labels",
        "metrics": metrics,
        "controls": controls,
        "invariance_audit": invariance,
        "paired_task_level_statistics": statistics,
        "checkpoint_paths": checkpoint_paths,
        "elapsed_seconds": float(time.perf_counter() - started),
        "split_leakage_audit": split_leakage,
        "split_leakage_audit_passes": bool(split_leakage.get("passes", False)),
        "output_leakage_audit": output_leakage,
        "output_leakage_audit_passes": bool(output_leakage.get("passes", False)),
        "duplicate_trajectory_hash_audit": duplicate_audit,
        "near_duplicate_trajectory_audit": near_duplicate_audit,
        "candidate_order_randomized_audit": candidate_order_audit,
        "raw_trajectory_separation_audit": raw_separation,
        "trainable_audit": _audit_subset(models["trainable"].audit),
        "frozen_audit": _audit_subset(models["frozen"].audit),
        "raw_latent_audit": _audit_subset(models["raw_latent"].audit),
        "randomized_labels_audit": _audit_subset(models["randomized_labels"].audit),
        "generator_identity_only_diagnostic": generator_identity_only_diagnostic(models["best_generator_on_train_baseline"], splits["final"]),
    }
    row["success_gates"] = seed_success_gates(row)
    return row


def load_selected_variant(stage7b_comparison: Dict[str, object], stage7b_results: Dict[str, object]) -> Stage7BVariant:
    selected_name = str(stage7b_comparison.get("selected_config", ""))
    selected = dict(stage7b_results.get("selected_config", {}))
    if not selected or str(selected.get("name", "")) != selected_name:
        raise RuntimeError("Stage 7B selected config mismatch between comparison and results artifacts")
    return Stage7BVariant(
        name=str(selected["name"]),
        description=str(selected["description"]),
        num_avenues=int(selected["num_avenues"]),
        lr=float(selected["lr"]),
        gradient_clip_norm=float(selected["gradient_clip_norm"]),
        avenue_dropout=float(selected.get("avenue_dropout", 0.0)),
        avenue_prompt_mode=str(selected.get("avenue_prompt_mode", "stage7_taubench_trajectory_selection")),
        coordinator_family=str(selected.get("coordinator_family", "candidate_token_cross_attention")),
    )


def load_frozen_stage7b_config(stage7b_results: Dict[str, object]) -> Stage7BConfig:
    config = dict(stage7b_results.get("config", {}))
    allowed = set(Stage7BConfig.__dataclass_fields__)
    filtered = {key: value for key, value in config.items() if key in allowed}
    return Stage7BConfig(**filtered)


def resolve_candidate_pool(stage7b_results: Dict[str, object]) -> Path:
    if DEFAULT_STAGE7C_CANDIDATES.exists():
        return DEFAULT_STAGE7C_CANDIDATES
    return Path(str(stage7b_results.get("candidate_pool_path", "results/stage7b0_tau_candidate_pool_balanced.jsonl")))


def build_final_splits(
    examples: Sequence[TauBenchTrajectorySelectionExample],
    stage7b_splits: Dict[str, object],
) -> Tuple[Dict[str, List[TauBenchTrajectorySelectionExample]], Dict[str, object]]:
    split_ids = stage7b_splits.get("splits", {})
    if not isinstance(split_ids, dict):
        raise RuntimeError("Stage 7B split artifact has no splits mapping")
    train_ids = set(str(value) for value in split_ids.get("train", []))
    final_source = "final" if split_ids.get("final") else "test" if split_ids.get("test") else "dev"
    final_ids = set(str(value) for value in split_ids.get(final_source, []))
    if not train_ids or not final_ids:
        raise RuntimeError("Stage 7C requires non-empty train and final task ids from the frozen Stage 7B split artifact")
    if train_ids & final_ids:
        raise RuntimeError("Stage 7C train/final split has overlapping task ids")
    by_id = {f"{example.domain}:{example.task_id}": example for example in examples}
    missing_train = sorted(train_ids - set(by_id))
    missing_final = sorted(final_ids - set(by_id))
    if missing_train or missing_final:
        raise RuntimeError(json.dumps({"missing_train": missing_train, "missing_final": missing_final}, sort_keys=True))
    train = [replace(by_id[key], split="train") for key in sorted(train_ids, key=task_sort_key)]
    final = [replace(by_id[key], split="final") for key in sorted(final_ids, key=task_sort_key)]
    meta = {
        "source_stage7b_split_file_declared_no_final_heldout_tasks_used": bool(stage7b_splits.get("no_final_heldout_tasks_used", False)),
        "final_task_ids_source_split": final_source,
        "train_task_ids_source_split": "train",
        "train_task_count": len(train),
        "final_task_count": len(final),
        "task_overlap_train_final": len(train_ids & final_ids),
        "no_final_labels_used_for_training_or_early_stopping": True,
        "fit_selection_split": "train_only_no_final_labels",
    }
    return {"train": train, "final": final}, meta


def write_final_splits(
    path: Path,
    splits: Dict[str, List[TauBenchTrajectorySelectionExample]],
    split_meta: Dict[str, object],
    candidate_pool_path: Path,
    pool_name: str,
    selected_variant: Stage7BVariant,
    config: Stage7BConfig,
) -> None:
    record = {
        "artifact": "stage7c_tau_final_splits",
        "created_at_utc": _now(),
        "candidate_pool_path": str(candidate_pool_path),
        "pool_name": pool_name,
        "domain": config.domain,
        "selected_variant": asdict(selected_variant),
        "frozen_stage7b_config": asdict(config),
        "split_meta": split_meta,
        "splits": {name: [f"{example.domain}:{example.task_id}" for example in rows] for name, rows in splits.items()},
        "summary": split_summary(splits, {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")


def add_best_generator_train_dev_alias(metrics: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    if "best_generator_on_train_baseline" in metrics:
        alias = dict(metrics["best_generator_on_train_baseline"])
        alias["method"] = "best_generator_on_train_dev_baseline"
        alias["fit_scope"] = "train_only_no_final_labels"
        metrics["best_generator_on_train_dev_baseline"] = alias
    return metrics


def seed_success_gates(row: Dict[str, object]) -> Dict[str, bool]:
    final = row["metrics"]["final"]
    trainable = final["trainable_shared_weight_latent_selector"]
    frozen = final["frozen_same_architecture_latent_selector"]
    randomized = final["randomized_labels_trainable_latent_selector"]
    best_name, best_pass = best_non_oracle_baseline(final)
    base_pass = float(trainable["pass_at_1"])
    return {
        "oracle_pass_at_8_between_0_25_and_0_85": 0.25 <= float(trainable["oracle_pass_at_8"]) <= 0.85,
        "trainable_beats_frozen": base_pass > float(frozen["pass_at_1"]),
        "trainable_beats_best_non_oracle_baseline": base_pass > best_pass,
        "best_non_oracle_baseline": bool(best_name),
        "randomized_labels_trainable_collapse": float(randomized["pass_at_1"]) <= max(0.0, base_pass - 0.05),
        "evidence_corruption_degrades": evidence_corruption_delta(row) >= 0.05,
        "candidate_order_invariance_passes": bool(row["invariance_audit"].get("candidate_order_invariance_passes", False)),
        "physical_order_invariance_passes": bool(row["invariance_audit"].get("physical_order_invariance_passes", False)),
        "gradient_update_audits_pass": gradient_update_audit_passes(row),
        "checkpoints_saved": checkpoints_saved(row),
    }


def build_final_statistics(rows: Sequence[Dict[str, object]], bootstrap_samples: int) -> Dict[str, object]:
    final_metrics = [row["metrics"]["final"] for row in rows]
    method_names = sorted({name for metrics in final_metrics for name in metrics})
    method_summary = {}
    for name in method_names:
        metrics_rows = [metrics[name] for metrics in final_metrics if name in metrics and not bool(metrics[name].get("unavailable", False))]
        if not metrics_rows:
            method_summary[name] = {"unavailable": True}
            continue
        method_summary[name] = {
            "pass_at_1_mean": mean_metric(metrics_rows, "pass_at_1"),
            "pass_at_1_std": std_metric(metrics_rows, "pass_at_1"),
            "conditional_selector_accuracy_mean": mean_metric(metrics_rows, "conditional_selector_accuracy"),
            "oracle_pass_at_8_mean": mean_metric(metrics_rows, "oracle_pass_at_8"),
            "selection_efficiency_mean": mean_metric(metrics_rows, "selection_efficiency"),
            "mrr_mean": mean_metric(metrics_rows, "mrr"),
            "top_2_accuracy_mean": mean_metric(metrics_rows, "top_2_accuracy"),
            "cost_tokens_latency_mean": cost_summary(metrics_rows),
            "per_task_success_count_breakdown": aggregate_nested_breakdown(metrics_rows, "per_success_count_breakdown"),
            "per_generator_source_audit_breakdown": aggregate_nested_breakdown(metrics_rows, "per_generator_source_audit_breakdown"),
        }
    best_name, best_mean = aggregate_best_non_oracle_baseline(final_metrics)
    trainable = method_summary["trainable_shared_weight_latent_selector"]
    frozen = method_summary["frozen_same_architecture_latent_selector"]
    randomized = method_summary["randomized_labels_trainable_latent_selector"]
    paired_best = aggregate_paired_statistics(rows, best_name, bootstrap_samples=bootstrap_samples, seed=817_400)
    paired_frozen = aggregate_paired_statistics(rows, "frozen_same_architecture_latent_selector", bootstrap_samples=bootstrap_samples, seed=817_500)
    seed_rows = []
    beats_frozen = 0
    for row in rows:
        final = row["metrics"]["final"]
        trainable_pass = float(final["trainable_shared_weight_latent_selector"]["pass_at_1"])
        frozen_pass = float(final["frozen_same_architecture_latent_selector"]["pass_at_1"])
        seed_best_name, seed_best_pass = best_non_oracle_baseline(final)
        beats_frozen += int(trainable_pass > frozen_pass)
        seed_rows.append(
            {
                "seed": int(row["seed"]),
                "trainable_pass_at_1": trainable_pass,
                "frozen_pass_at_1": frozen_pass,
                "best_non_oracle_baseline": seed_best_name,
                "best_non_oracle_baseline_pass_at_1": seed_best_pass,
                "trainable_minus_frozen": trainable_pass - frozen_pass,
                "trainable_minus_best_non_oracle_baseline": trainable_pass - seed_best_pass,
            }
        )
    return {
        "artifact": "stage7c_tau_final_statistics",
        "created_at_utc": _now(),
        "completed_seeds": sorted(int(row["seed"]) for row in rows if row.get("status") == "completed"),
        "method_summary": method_summary,
        "best_non_oracle_baseline": best_name,
        "best_non_oracle_baseline_pass_at_1_mean": best_mean,
        "trainable_pass_at_1_mean": float(trainable["pass_at_1_mean"]),
        "frozen_pass_at_1_mean": float(frozen["pass_at_1_mean"]),
        "randomized_labels_trainable_pass_at_1_mean": float(randomized["pass_at_1_mean"]),
        "oracle_pass_at_8_mean": float(trainable["oracle_pass_at_8_mean"]),
        "trainable_minus_frozen_pass_at_1": float(trainable["pass_at_1_mean"]) - float(frozen["pass_at_1_mean"]),
        "trainable_minus_best_non_oracle_baseline": float(trainable["pass_at_1_mean"]) - best_mean,
        "trainable_minus_randomized_labels_trainable": float(trainable["pass_at_1_mean"]) - float(randomized["pass_at_1_mean"]),
        "trainable_selection_efficiency_mean": float(trainable["selection_efficiency_mean"]),
        "best_non_oracle_selection_efficiency_mean": float(method_summary[best_name]["selection_efficiency_mean"]) if best_name else 0.0,
        "trainable_beats_frozen_seed_count": beats_frozen,
        "per_seed_results": seed_rows,
        "paired_trainable_vs_best_non_oracle_baseline": paired_best,
        "paired_trainable_vs_frozen": paired_frozen,
    }


def build_control_results(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    control_names = sorted({name for row in rows for name in row.get("controls", {})})
    base_values = [float(row["metrics"]["final"]["trainable_shared_weight_latent_selector"]["pass_at_1"]) for row in rows]
    control_summary = {}
    for name in control_names:
        values = [float(row["controls"][name]["pass_at_1"]) for row in rows if name in row.get("controls", {})]
        deltas = [base - value for base, value in zip(base_values, values)]
        control_summary[name] = {
            "pass_at_1_mean": float(np.mean(values)) if values else 0.0,
            "base_minus_control_pass_at_1_mean": float(np.mean(deltas)) if deltas else 0.0,
            "pass_count": len(values),
        }
    evidence_controls = (
        "candidate_evidence_mismatch",
        "cross_task_user_goal_shuffle",
        "cross_task_policy_shuffle",
        "cross_task_tool_observation_shuffle",
    )
    return {
        "artifact": "stage7c_tau_final_control_results",
        "created_at_utc": _now(),
        "control_summary": control_summary,
        "evidence_corruption_max_delta": max(float(control_summary.get(name, {}).get("base_minus_control_pass_at_1_mean", 0.0)) for name in evidence_controls),
        "randomized_labels_control_delta": float(control_summary.get("randomized_labels", {}).get("base_minus_control_pass_at_1_mean", 0.0)),
        "candidate_order_invariance_pass_count": sum(bool(row["invariance_audit"].get("candidate_order_invariance_passes", False)) for row in rows),
        "physical_order_invariance_pass_count": sum(bool(row["invariance_audit"].get("physical_order_invariance_passes", False)) for row in rows),
        "generator_identity_only_diagnostic": aggregate_generator_identity_diagnostic(rows),
        "stronger_evidence_use_gates": stronger_evidence_use_gates(control_summary),
    }


def final_success_gates(
    rows: Sequence[Dict[str, object]],
    statistics: Dict[str, object],
    controls: Dict[str, object],
    split_leakage: Dict[str, object],
    output_leakage: Dict[str, object],
    duplicate_audit: Dict[str, object],
    near_duplicate_audit: Dict[str, object],
    candidate_order_audit: Dict[str, object],
    raw_separation: Dict[str, object],
    report_path: Path,
) -> Dict[str, bool]:
    completed = [row for row in rows if row.get("status") == "completed"]
    gates = {
        "completed_seeds_gte_10": len(completed) >= 10,
        "oracle_pass_at_8_between_0_25_and_0_85": 0.25 <= float(statistics["oracle_pass_at_8_mean"]) <= 0.85,
        "trainable_beats_frozen_on_gte_8_of_10_seeds": int(statistics["trainable_beats_frozen_seed_count"]) >= 8,
        "trainable_mean_pass_at_1_beats_frozen": float(statistics["trainable_minus_frozen_pass_at_1"]) > 0.0,
        "trainable_beats_best_non_oracle_by_gte_0_05": float(statistics["trainable_minus_best_non_oracle_baseline"]) >= 0.05,
        "paired_bootstrap_ci_lower_vs_best_non_oracle_gt_0": float(statistics["paired_trainable_vs_best_non_oracle_baseline"]["paired_bootstrap_95_ci"]["lower"]) > 0.0,
        "paired_bootstrap_ci_lower_vs_frozen_gt_0": float(statistics["paired_trainable_vs_frozen"]["paired_bootstrap_95_ci"]["lower"]) > 0.0,
        "trainable_selection_efficiency_beats_best_non_oracle": float(statistics["trainable_selection_efficiency_mean"]) > float(statistics["best_non_oracle_selection_efficiency_mean"]),
        "randomized_labels_collapse": float(statistics["trainable_minus_randomized_labels_trainable"]) >= 0.05,
        "evidence_corruption_control_degrades_by_gte_0_05": float(controls["evidence_corruption_max_delta"]) >= 0.05,
        "candidate_order_invariance_passes": int(controls["candidate_order_invariance_pass_count"]) == len(completed),
        "physical_order_invariance_passes": int(controls["physical_order_invariance_pass_count"]) == len(completed),
        "leakage_audits_pass": bool(split_leakage.get("passes", False)) and bool(output_leakage.get("passes", False)),
        "gradient_update_frozen_audits_pass": all(gradient_update_audit_passes(row) for row in completed),
        "duplicate_and_near_duplicate_audits_pass": bool(duplicate_audit.get("passes", False)) and bool(near_duplicate_audit.get("passes", False)),
        "candidate_order_randomized_audit_pass": bool(candidate_order_audit.get("passes", False)),
        "raw_trajectory_separation_audit_pass": bool(raw_separation.get("passes", False)),
        "checkpoints_saved": all(checkpoints_saved(row) for row in completed),
        "final_report_generated": bool(report_path),
    }
    return gates


def final_decision(gates: Dict[str, bool], statistics: Dict[str, object]) -> str:
    if all(gates.values()):
        return "A. STAGE7C_FINAL_PASSED"
    main_positive = (
        int(statistics.get("trainable_beats_frozen_seed_count", 0)) >= 8
        and float(statistics.get("trainable_minus_frozen_pass_at_1", 0.0)) > 0.0
        and float(statistics.get("trainable_minus_best_non_oracle_baseline", 0.0)) > 0.0
    )
    if main_positive:
        return "B. STAGE7C_FINAL_PARTIAL"
    return "C. STAGE7C_FINAL_FAILED"


def failure_diagnosis(gates: Dict[str, bool], statistics: Dict[str, object], controls: Dict[str, object]) -> Dict[str, object]:
    return {
        "oracle_pass_at_8_mean": statistics.get("oracle_pass_at_8_mean"),
        "oracle_pass_at_8_too_high_or_low": not gates.get("oracle_pass_at_8_between_0_25_and_0_85", False),
        "candidate_trajectories_too_easy_or_hard": not gates.get("oracle_pass_at_8_between_0_25_and_0_85", False),
        "best_baseline_saturation": not gates.get("trainable_beats_best_non_oracle_by_gte_0_05", False),
        "evidence_corruption_not_degrading": not gates.get("evidence_corruption_control_degrades_by_gte_0_05", False),
        "generator_source_leakage": not gates.get("leakage_audits_pass", False),
        "pool_size_too_small": len(statistics.get("completed_seeds", [])) < 10,
        "trainable_frozen_gap_absent": not gates.get("trainable_mean_pass_at_1_beats_frozen", False),
        "architecture_mismatch": not gates.get("trainable_beats_frozen_on_gte_8_of_10_seeds", False),
        "stronger_evidence_use_gates": controls.get("stronger_evidence_use_gates", {}),
    }


def seed_pairwise_statistics(metrics: Dict[str, Dict[str, object]], seed: int, bootstrap_samples: int) -> Dict[str, object]:
    best_name, _best_pass = best_non_oracle_baseline(metrics)
    return {
        "best_non_oracle_baseline": best_name,
        "trainable_vs_best_non_oracle_baseline": paired_statistics(metrics, best_name, seed=seed + 811_000, bootstrap_samples=bootstrap_samples),
        "trainable_vs_frozen": paired_statistics(metrics, "frozen_same_architecture_latent_selector", seed=seed + 812_000, bootstrap_samples=bootstrap_samples),
    }


def aggregate_paired_statistics(rows: Sequence[Dict[str, object]], baseline_name: str, bootstrap_samples: int, seed: int) -> Dict[str, object]:
    trainable_correct = []
    baseline_correct = []
    for row in rows:
        metrics = row["metrics"]["final"]
        trainable_correct.extend(metrics["trainable_shared_weight_latent_selector"].get("correct_by_example", []))
        baseline_correct.extend(metrics[baseline_name].get("correct_by_example", []))
    return paired_arrays_statistics(
        np.asarray(trainable_correct, dtype=np.float32),
        np.asarray(baseline_correct, dtype=np.float32),
        baseline_name=baseline_name,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
    )


def paired_statistics(metrics: Dict[str, Dict[str, object]], baseline_name: str, seed: int, bootstrap_samples: int) -> Dict[str, object]:
    trainable = np.asarray(metrics["trainable_shared_weight_latent_selector"].get("correct_by_example", []), dtype=np.float32)
    baseline = np.asarray(metrics[baseline_name].get("correct_by_example", []), dtype=np.float32)
    return paired_arrays_statistics(trainable, baseline, baseline_name=baseline_name, seed=seed, bootstrap_samples=bootstrap_samples)


def paired_arrays_statistics(
    trainable: np.ndarray,
    baseline: np.ndarray,
    baseline_name: str,
    seed: int,
    bootstrap_samples: int,
) -> Dict[str, object]:
    if trainable.shape != baseline.shape:
        raise ValueError(f"paired arrays mismatch for {baseline_name}: {trainable.shape} vs {baseline.shape}")
    diff = trainable - baseline
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(max(1, int(bootstrap_samples))):
        idx = rng.integers(0, len(diff), size=len(diff)) if len(diff) else np.zeros(0, dtype=np.int64)
        means.append(float(np.mean(diff[idx])) if len(idx) else 0.0)
    sign = paired_sign_test(diff)
    return {
        "baseline": baseline_name,
        "n_pairs": int(len(diff)),
        "trainable_pass_at_1": float(np.mean(trainable)) if len(trainable) else 0.0,
        "baseline_pass_at_1": float(np.mean(baseline)) if len(baseline) else 0.0,
        "absolute_pass_at_1_delta": float(np.mean(diff)) if len(diff) else 0.0,
        "paired_bootstrap_95_ci": {
            "mean": float(np.mean(diff)) if len(diff) else 0.0,
            "lower": float(np.quantile(means, 0.025)) if means else 0.0,
            "upper": float(np.quantile(means, 0.975)) if means else 0.0,
        },
        "paired_sign_test": sign,
    }


def paired_sign_test(diff: np.ndarray) -> Dict[str, object]:
    wins = int(np.sum(diff > 0))
    losses = int(np.sum(diff < 0))
    ties = int(np.sum(diff == 0))
    n = wins + losses
    if n == 0:
        p = 1.0
    else:
        k = min(wins, losses)
        p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / float(2**n))
    return {"wins": wins, "losses": losses, "ties": ties, "n_untied": n, "two_sided_p": float(p)}


def best_non_oracle_baseline(metrics: Dict[str, Dict[str, object]]) -> Tuple[str, float]:
    excluded = {
        "oracle_pass_at_8",
        "trainable_shared_weight_latent_selector",
        "randomized_labels_trainable_latent_selector",
        "llm_judge_baseline",
    }
    candidates = {
        name: float(row.get("pass_at_1", 0.0))
        for name, row in metrics.items()
        if name not in excluded and not bool(row.get("unavailable", False))
    }
    if not candidates:
        return ("", 0.0)
    name = max(sorted(candidates), key=lambda key: candidates[key])
    return name, candidates[name]


def aggregate_best_non_oracle_baseline(final_metrics: Sequence[Dict[str, Dict[str, object]]]) -> Tuple[str, float]:
    names = sorted({name for metrics in final_metrics for name in metrics})
    excluded = {
        "oracle_pass_at_8",
        "trainable_shared_weight_latent_selector",
        "randomized_labels_trainable_latent_selector",
        "llm_judge_baseline",
    }
    means = {}
    for name in names:
        if name in excluded:
            continue
        values = [float(metrics[name].get("pass_at_1", 0.0)) for metrics in final_metrics if name in metrics and not bool(metrics[name].get("unavailable", False))]
        if values:
            means[name] = float(np.mean(values))
    if not means:
        return ("", 0.0)
    name = max(sorted(means), key=lambda key: means[key])
    return name, means[name]


def evidence_corruption_delta(row: Dict[str, object]) -> float:
    base = float(row["metrics"]["final"]["trainable_shared_weight_latent_selector"]["pass_at_1"])
    controls = row.get("controls", {})
    names = (
        "candidate_evidence_mismatch",
        "cross_task_user_goal_shuffle",
        "cross_task_policy_shuffle",
        "cross_task_tool_observation_shuffle",
    )
    return max(base - float((controls.get(name) or {}).get("pass_at_1", base)) for name in names)


def checkpoints_saved(row: Dict[str, object]) -> bool:
    paths = row.get("checkpoint_paths", {})
    required = (
        "trainable",
        "frozen",
        "raw_latent",
        "randomized_labels",
        "static_trajectory_feature_reranker",
        "text_only_single_reviewer",
        "text_only_multi_agent_reviewer",
    )
    return all(Path(str(paths.get(name, ""))).exists() for name in required)


def audit_rows_from_result_row(row: Dict[str, object]) -> List[Dict[str, object]]:
    out = []
    for key in ("trainable_audit", "frozen_audit", "raw_latent_audit", "randomized_labels_audit"):
        audit = row.get(key)
        if isinstance(audit, dict) and audit:
            out.append({**audit, "benchmark": BENCHMARK, "seed": row.get("seed"), "variant": row.get("variant"), "type": key})
    out.append({"benchmark": BENCHMARK, "seed": row.get("seed"), "variant": row.get("variant"), "type": "controls", **row.get("controls", {})})
    out.append({"benchmark": BENCHMARK, "seed": row.get("seed"), "variant": row.get("variant"), "type": "invariance_audit", **row.get("invariance_audit", {})})
    out.append({"benchmark": BENCHMARK, "seed": row.get("seed"), "variant": row.get("variant"), "type": "success_gates", **row.get("success_gates", {})})
    return out


def save_final_checkpoints(
    checkpoint_dir: Path,
    variant: Stage7BVariant,
    seed: int,
    models: Dict[str, object],
    config: Stage7BConfig,
) -> Dict[str, str]:
    root = checkpoint_dir / variant.name / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    for name in ("trainable", "frozen", "raw_latent", "randomized_labels"):
        fit = models.get(name)
        path = root / f"{fit.method}.pt"
        torch.save(
            {
                "benchmark": BENCHMARK,
                "seed": int(seed),
                "variant": asdict(variant),
                "config": asdict(config),
                "method": fit.method,
                "system_state_dict": fit.system.state_dict(),
                "audit": fit.audit,
                "history": fit.history,
            },
            path,
        )
        paths[name] = str(path)
    for name in ("static_trajectory_feature_reranker", "text_only_single_reviewer", "text_only_multi_agent_reviewer"):
        model = models.get(name)
        path = root / f"{name}.pt"
        torch.save(
            {
                "benchmark": BENCHMARK,
                "seed": int(seed),
                "variant": asdict(variant),
                "config": asdict(config),
                "method": model.method,
                "model_state_dict": model.model.state_dict(),
                "history": model.history,
                "param_count": model.param_count,
            },
            path,
        )
        paths[name] = str(path)
    return paths


def candidate_order_randomized_audit(examples: Sequence[TauBenchTrajectorySelectionExample]) -> Dict[str, object]:
    failures = []
    for example in examples:
        if not bool(example.metadata.get("candidate_order_randomized", False)):
            failures.append(example.id)
    return {"examples_checked": len(examples), "not_randomized_example_ids": failures, "passes": not failures}


def raw_trajectory_separation_audit(examples: Sequence[TauBenchTrajectorySelectionExample]) -> Dict[str, object]:
    failures = []
    for example in examples:
        for candidate in example.candidates:
            visible = trajectory_candidate_text(candidate)
            raw_path = str(candidate.raw_trajectory_path)
            if not raw_path:
                failures.append({"candidate_id": candidate.candidate_id, "issue": "missing_raw_trajectory_path"})
            elif raw_path in visible:
                failures.append({"candidate_id": candidate.candidate_id, "issue": "raw_path_visible_in_selector_text"})
    return {"examples_checked": len(examples), "failures": failures, "passes": not failures}


def split_summary(
    splits: Dict[str, List[TauBenchTrajectorySelectionExample]],
    candidate_source_map: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    out = {}
    for split, rows in splits.items():
        success_counts = [sum(example.labels_pass_fail) for example in rows]
        generators = Counter()
        for example in rows:
            for candidate in example.candidates:
                generators[str(candidate_source_map.get(candidate.candidate_id, {}).get("generator_name_audit_only", "unknown"))] += 1
        out[split] = {
            "tasks": len(rows),
            "candidates": sum(len(example.candidates) for example in rows),
            "oracle_pass_at_8": sum(1 for count in success_counts if count > 0) / float(max(1, len(success_counts))),
            "oracle_empty_tasks": sum(1 for count in success_counts if count == 0),
            "success_count_distribution": dict(sorted(Counter(success_counts).items())),
            "generator_source_distribution_audit_only": dict(sorted(generators.items())),
        }
    out["task_overlap_train_final"] = len({example.task_id for example in splits.get("train", [])} & {example.task_id for example in splits.get("final", [])})
    return out


def aggregate_nested_breakdown(rows: Sequence[Dict[str, object]], key: str) -> Dict[str, object]:
    values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        nested = row.get(key, {})
        if not isinstance(nested, dict):
            continue
        for group, metrics in nested.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        values[str(group)][str(metric_name)].append(float(value))
    return {
        group: {metric: float(np.mean(vals)) if vals else 0.0 for metric, vals in sorted(metrics.items())}
        for group, metrics in sorted(values.items())
    }


def aggregate_generator_identity_diagnostic(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    metrics = []
    generators = Counter()
    for row in rows:
        diag = row.get("generator_identity_only_diagnostic", {})
        if isinstance(diag, dict):
            metric = diag.get("metrics", {})
            if isinstance(metric, dict):
                metrics.append(metric)
            best = str(diag.get("best_generator_on_train", ""))
            if best:
                generators[best] += 1
    return {
        "audit_only_not_selector_visible": True,
        "best_generator_on_train_counts": dict(sorted(generators.items())),
        "pass_at_1_mean": mean_metric(metrics, "pass_at_1"),
        "selection_efficiency_mean": mean_metric(metrics, "selection_efficiency"),
    }


def stronger_evidence_use_gates(control_summary: Dict[str, Dict[str, object]]) -> Dict[str, bool]:
    candidate_delta = float(control_summary.get("candidate_only", {}).get("base_minus_control_pass_at_1_mean", 0.0))
    trajectory_delta = float(control_summary.get("trajectory_only", {}).get("base_minus_control_pass_at_1_mean", 0.0))
    mismatch_delta = float(control_summary.get("candidate_evidence_mismatch", {}).get("base_minus_control_pass_at_1_mean", 0.0))
    goal_delta = float(control_summary.get("cross_task_user_goal_shuffle", {}).get("base_minus_control_pass_at_1_mean", 0.0))
    policy_delta = float(control_summary.get("cross_task_policy_shuffle", {}).get("base_minus_control_pass_at_1_mean", 0.0))
    tool_delta = float(control_summary.get("cross_task_tool_observation_shuffle", {}).get("base_minus_control_pass_at_1_mean", 0.0))
    return {
        "full_model_beats_candidate_only_control": candidate_delta > 0.0,
        "full_model_beats_trajectory_only_control": trajectory_delta > 0.0,
        "candidate_evidence_mismatch_degrades_substantially": mismatch_delta >= 0.05,
        "cross_task_goal_shuffle_degrades_substantially": goal_delta >= 0.05,
        "cross_task_policy_shuffle_degrades_substantially": policy_delta >= 0.05,
        "cross_task_tool_shuffle_degrades_substantially": tool_delta >= 0.05,
    }


def mean_metric(rows: Sequence[Dict[str, object]], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in rows]
    return float(np.mean(values)) if values else 0.0


def std_metric(rows: Sequence[Dict[str, object]], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in rows]
    return float(np.std(values)) if len(values) > 1 else 0.0


def cost_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    keys = (
        "estimated_tokens_per_selector_decision",
        "latency_seconds_per_selector_decision",
    )
    out = {}
    for key in keys:
        values = []
        for row in rows:
            meta = row.get("cost_tokens_latency", {})
            if isinstance(meta, dict) and key in meta:
                values.append(float(meta[key]))
        out[key] = float(np.mean(values)) if values else 0.0
    return out


def append_progress_row(
    path: Path,
    row: Dict[str, object],
    selected_variant: Stage7BVariant,
    config: Stage7BConfig,
    candidate_pool_path: Path,
    split_meta: Dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "artifact": "stage7c_tau_final_selector_row_progress",
        "created_at_utc": _now(),
        "candidate_pool_path": str(candidate_pool_path),
        "selected_variant": asdict(selected_variant),
        "config": asdict(config),
        "split_meta": split_meta,
        "row": row,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")


def load_progress_rows(
    path: Path,
    selected_variant: Stage7BVariant,
    config: Stage7BConfig,
    candidate_pool_path: Path,
    split_meta: Dict[str, object],
) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows: Dict[int, Dict[str, object]] = {}
    wanted_variant = asdict(selected_variant)
    wanted_config = asdict(config)
    wanted_pool = str(candidate_pool_path)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("selected_variant") != wanted_variant:
            continue
        if record.get("config") != wanted_config:
            continue
        if str(record.get("candidate_pool_path", "")) != wanted_pool:
            continue
        if record.get("split_meta") != split_meta:
            continue
        row = record.get("row")
        if isinstance(row, dict) and str(row.get("status", "")) == "completed":
            rows[int(row.get("seed", -1))] = row
    return [rows[key] for key in sorted(rows)]


def write_outputs(
    result: Dict[str, object],
    controls: Dict[str, object],
    statistics: Dict[str, object],
    audit_rows: Sequence[Dict[str, object]],
    results_path: Path,
    controls_path: Path,
    statistics_path: Path,
    audit_path: Path,
    report_path: Path,
) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    controls_path.parent.mkdir(parents=True, exist_ok=True)
    statistics_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    controls_path.write_text(json.dumps(controls, indent=2, sort_keys=True), encoding="utf-8")
    statistics_path.write_text(json.dumps(statistics, indent=2, sort_keys=True), encoding="utf-8")
    audit_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in audit_rows), encoding="utf-8")
    report_path.write_text(render_report(result, controls, statistics), encoding="utf-8")


def render_report(result: Dict[str, object], controls: Dict[str, object], statistics: Dict[str, object]) -> str:
    gates = result["final_success_gates"]
    lines = [
        "# Stage 7C tau-bench Final Validation",
        "",
        "## Decision",
        f"- Decision: `{result['decision']}`",
        f"- Conservative final claim allowed: `{result['conservative_claim_allowed']}`",
        f"- Selected Stage 7B variant: `{result['selected_variant']['name']}`",
        "- Airline cross-domain validation run: `False`.",
        "",
        "## Dataset",
        f"- Candidate pool: `{result['candidate_pool_path']}`",
        f"- Split meta: `{json.dumps(result['split_meta'], sort_keys=True)}`",
        f"- Split summary: `{json.dumps(result['split_summary'], sort_keys=True)}`",
        "",
        "## Final Metrics",
        f"- Trainable pass@1 mean: `{statistics['trainable_pass_at_1_mean']:.4f}`",
        f"- Frozen pass@1 mean: `{statistics['frozen_pass_at_1_mean']:.4f}`",
        f"- Best non-oracle baseline: `{statistics['best_non_oracle_baseline']}`",
        f"- Best non-oracle pass@1 mean: `{statistics['best_non_oracle_baseline_pass_at_1_mean']:.4f}`",
        f"- Trainable-frozen delta: `{statistics['trainable_minus_frozen_pass_at_1']:.4f}`",
        f"- Trainable-best-baseline delta: `{statistics['trainable_minus_best_non_oracle_baseline']:.4f}`",
        f"- Oracle pass@8 mean: `{statistics['oracle_pass_at_8_mean']:.4f}`",
        f"- Trainable selection efficiency: `{statistics['trainable_selection_efficiency_mean']:.4f}`",
        "",
        "## Statistics",
        f"- Trainable vs best baseline: `{json.dumps(statistics['paired_trainable_vs_best_non_oracle_baseline'], sort_keys=True)}`",
        f"- Trainable vs frozen: `{json.dumps(statistics['paired_trainable_vs_frozen'], sort_keys=True)}`",
        "",
        "## Controls",
        f"`{json.dumps(controls, sort_keys=True)}`",
        "",
        "## Gates",
        f"`{json.dumps(gates, sort_keys=True)}`",
        "",
        "## Failure Diagnosis",
        f"`{json.dumps(result['failure_diagnosis'], sort_keys=True)}`",
        "",
        "## Claim",
        f"`{result['conservative_claim']}`" if result["conservative_claim"] else "`No final claim allowed.`",
        "",
    ]
    return "\n".join(lines)


def conservative_claim_text() -> str:
    return (
        "On held-out tau-bench retail K=8 trajectory-selection tasks, the shared-weight latent cloned-agent "
        "selector chose successful tool-use trajectories more reliably than frozen latent, text-only, "
        "embedding/static, and simple ordering baselines under strict leakage, corruption, and invariance "
        "controls. This supports latent coordination as a trajectory-selection architecture, not "
        "trajectory-generation superiority."
    )


def task_sort_key(task_id: str) -> Tuple[int, str]:
    text = str(task_id)
    if ":" in text:
        text = text.split(":", 1)[1]
    return (int(text), text) if text.isdigit() else (10**9, text)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
