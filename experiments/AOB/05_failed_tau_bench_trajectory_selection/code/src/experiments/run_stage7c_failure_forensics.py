from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.datasets.taubench_trajectory_selection_dataset import (
    build_taubench_selection_examples,
    load_taubench_candidates_jsonl,
)
from src.experiments.run_stage7b_tau_dev_validation import load_candidate_source_map


DEFAULT_STAGE7C_RESULTS = Path("results/stage7c_tau_final_selector_results.json")
DEFAULT_STAGE7C_AUDIT = Path("results/stage7c_tau_final_selector_audit.jsonl")
DEFAULT_STAGE7C_CONTROLS = Path("results/stage7c_tau_final_control_results.json")
DEFAULT_STAGE7C_STATISTICS = Path("results/stage7c_tau_final_statistics.json")
DEFAULT_STAGE7C_REPORT = Path("reports/STAGE7C_TAU_FINAL_VALIDATION.md")
DEFAULT_STAGE7C_SPLITS = Path("results/stage7c_tau_final_splits.json")
DEFAULT_STAGE7B_RESULTS = Path("results/stage7b_tau_dev_selector_results.json")
DEFAULT_STAGE7B_COMPARISON = Path("results/stage7b_tau_dev_variant_comparison.json")
DEFAULT_LABEL_AUDIT = Path("results/stage7b0_tau_label_audit.json")
DEFAULT_OUTPUT_JSON = Path("results/stage7c_failure_forensics.json")
DEFAULT_OUTPUT_REPORT = Path("reports/STAGE7C_FAILURE_FORENSICS.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Produce Stage 7C failure forensics without rerunning training.")
    parser.add_argument("--stage7c-results", default=str(DEFAULT_STAGE7C_RESULTS))
    parser.add_argument("--stage7c-audit", default=str(DEFAULT_STAGE7C_AUDIT))
    parser.add_argument("--stage7c-controls", default=str(DEFAULT_STAGE7C_CONTROLS))
    parser.add_argument("--stage7c-statistics", default=str(DEFAULT_STAGE7C_STATISTICS))
    parser.add_argument("--stage7c-report", default=str(DEFAULT_STAGE7C_REPORT))
    parser.add_argument("--stage7c-splits", default=str(DEFAULT_STAGE7C_SPLITS))
    parser.add_argument("--stage7b-results", default=str(DEFAULT_STAGE7B_RESULTS))
    parser.add_argument("--stage7b-comparison", default=str(DEFAULT_STAGE7B_COMPARISON))
    parser.add_argument("--label-audit", default=str(DEFAULT_LABEL_AUDIT))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-report", default=str(DEFAULT_OUTPUT_REPORT))
    args = parser.parse_args()
    result = run_forensics(
        stage7c_results_path=Path(args.stage7c_results),
        stage7c_audit_path=Path(args.stage7c_audit),
        stage7c_controls_path=Path(args.stage7c_controls),
        stage7c_statistics_path=Path(args.stage7c_statistics),
        stage7c_report_path=Path(args.stage7c_report),
        stage7c_splits_path=Path(args.stage7c_splits),
        stage7b_results_path=Path(args.stage7b_results),
        stage7b_comparison_path=Path(args.stage7b_comparison),
        label_audit_path=Path(args.label_audit),
        output_json_path=Path(args.output_json),
        output_report_path=Path(args.output_report),
    )
    print(json.dumps({"recommendation": result["decision_recommendation"]}, sort_keys=True))


def run_forensics(
    stage7c_results_path: Path,
    stage7c_audit_path: Path,
    stage7c_controls_path: Path,
    stage7c_statistics_path: Path,
    stage7c_report_path: Path,
    stage7c_splits_path: Path,
    stage7b_results_path: Path,
    stage7b_comparison_path: Path,
    label_audit_path: Path,
    output_json_path: Path,
    output_report_path: Path,
) -> Dict[str, object]:
    stage7c_results = read_json(stage7c_results_path)
    stage7c_controls = read_json(stage7c_controls_path)
    stage7c_statistics = read_json(stage7c_statistics_path)
    stage7c_splits = read_json(stage7c_splits_path)
    stage7b_results = read_json(stage7b_results_path)
    stage7b_comparison = read_json(stage7b_comparison_path)
    audit_rows = read_jsonl(stage7c_audit_path)
    selected_variant = str(stage7c_results.get("selected_variant", {}).get("name", ""))
    selected_dev_rows = [row for row in stage7b_results.get("rows", []) if str(row.get("variant", "")) == selected_variant]
    candidate_pool_path = Path(str(stage7c_results.get("candidate_pool_path", "")))
    pool_name = str(stage7c_results.get("pool_name", "balanced_pool"))
    candidate_source_map = load_candidate_source_map(label_audit_path, pool_name)

    final_failure = summarize_final_failure(stage7c_results, stage7c_statistics)
    dev_final = compare_dev_final(stage7b_results, stage7b_comparison, selected_dev_rows, stage7c_statistics, stage7c_controls)
    evidence = evidence_use_analysis(selected_dev_rows, stage7c_controls, stage7c_statistics)
    baselines = baseline_saturation_analysis(stage7c_statistics)
    pool = pool_difficulty_analysis(stage7c_results, stage7c_splits, candidate_pool_path, candidate_source_map)
    training_audit = training_audit_sanity(stage7c_results, audit_rows)
    categories = classify_failure(final_failure, dev_final, evidence, baselines, pool, training_audit, stage7c_results)
    recommendation = choose_recommendation(categories, final_failure, evidence, baselines, pool)
    result = {
        "artifact": "stage7c_failure_forensics",
        "created_at_utc": now(),
        "inputs": {
            "stage7c_results": str(stage7c_results_path),
            "stage7c_audit": str(stage7c_audit_path),
            "stage7c_controls": str(stage7c_controls_path),
            "stage7c_statistics": str(stage7c_statistics_path),
            "stage7c_report": str(stage7c_report_path),
            "stage7c_splits": str(stage7c_splits_path),
            "stage7b_results": str(stage7b_results_path),
            "stage7b_comparison": str(stage7b_comparison_path),
        },
        "constraints": {
            "training_rerun": False,
            "hyperparameters_changed": False,
            "architecture_changed": False,
            "candidate_pool_changed": False,
            "splits_changed": False,
            "controls_weakened": False,
            "candidate_trajectories_altered": False,
            "labels_altered": False,
            "final_benchmark_claim_made": False,
        },
        "selected_stage7b_config": stage7c_results.get("selected_variant", {}),
        "final_failure_summary": final_failure,
        "dev_vs_final_comparison": dev_final,
        "failure_categories": categories,
        "evidence_use_analysis": evidence,
        "baseline_saturation_analysis": baselines,
        "pool_difficulty_analysis": pool,
        "training_audit_sanity": training_audit,
        "likely_blocker": likely_blocker(categories),
        "decision_recommendation": recommendation,
        "conclusion": conclusion_text(recommendation, categories),
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    output_report_path.write_text(render_report(result), encoding="utf-8")
    return result


def summarize_final_failure(stage7c_results: Dict[str, object], stats: Dict[str, object]) -> Dict[str, object]:
    per_seed = list(stats.get("per_seed_results", []))
    wins_vs_frozen = sum(1 for row in per_seed if float(row.get("trainable_minus_frozen", 0.0)) > 0.0)
    losses_vs_frozen = sum(1 for row in per_seed if float(row.get("trainable_minus_frozen", 0.0)) < 0.0)
    ties_vs_frozen = len(per_seed) - wins_vs_frozen - losses_vs_frozen
    wins_vs_best = sum(1 for row in per_seed if float(row.get("trainable_minus_best_non_oracle_baseline", 0.0)) > 0.0)
    losses_vs_best = sum(1 for row in per_seed if float(row.get("trainable_minus_best_non_oracle_baseline", 0.0)) < 0.0)
    ties_vs_best = len(per_seed) - wins_vs_best - losses_vs_best
    gates = dict(stage7c_results.get("final_success_gates", {}))
    failed_gates = sorted(key for key, value in gates.items() if not bool(value))
    return {
        "stage7c_decision": stage7c_results.get("decision"),
        "trainable_latent_mean_pass_at_1": stats.get("trainable_pass_at_1_mean"),
        "frozen_latent_mean_pass_at_1": stats.get("frozen_pass_at_1_mean"),
        "best_non_oracle_baseline": stats.get("best_non_oracle_baseline"),
        "best_non_oracle_baseline_pass_at_1": stats.get("best_non_oracle_baseline_pass_at_1_mean"),
        "oracle_pass_at_8": stats.get("oracle_pass_at_8_mean"),
        "trainable_vs_frozen_delta": stats.get("trainable_minus_frozen_pass_at_1"),
        "trainable_vs_best_baseline_delta": stats.get("trainable_minus_best_non_oracle_baseline"),
        "seeds_won_lost_vs_frozen": {"won": wins_vs_frozen, "lost": losses_vs_frozen, "tied": ties_vs_frozen},
        "seeds_won_lost_vs_best_non_oracle": {"won": wins_vs_best, "lost": losses_vs_best, "tied": ties_vs_best},
        "ci_vs_frozen": stats.get("paired_trainable_vs_frozen", {}).get("paired_bootstrap_95_ci", {}),
        "ci_vs_best_baseline": stats.get("paired_trainable_vs_best_non_oracle_baseline", {}).get("paired_bootstrap_95_ci", {}),
        "paired_sign_test_vs_frozen": stats.get("paired_trainable_vs_frozen", {}).get("paired_sign_test", {}),
        "paired_sign_test_vs_best_baseline": stats.get("paired_trainable_vs_best_non_oracle_baseline", {}).get("paired_sign_test", {}),
        "failed_gates": failed_gates,
    }


def compare_dev_final(
    stage7b_results: Dict[str, object],
    stage7b_comparison: Dict[str, object],
    selected_dev_rows: Sequence[Dict[str, object]],
    final_stats: Dict[str, object],
    final_controls: Dict[str, object],
) -> Dict[str, object]:
    selected_name = str(stage7b_comparison.get("selected_config", ""))
    dev_variant = dict(stage7b_comparison.get("variant_metrics", {}).get(selected_name, {}))
    dev_controls = aggregate_dev_controls(selected_dev_rows)
    final_control_summary = dict(final_controls.get("control_summary", {}))
    return {
        "selected_variant": selected_name,
        "dev_trainable_pass_at_1": dev_variant.get("trainable_pass_at_1_mean"),
        "final_trainable_pass_at_1": final_stats.get("trainable_pass_at_1_mean"),
        "dev_frozen_pass_at_1": dev_variant.get("frozen_pass_at_1_mean"),
        "final_frozen_pass_at_1": final_stats.get("frozen_pass_at_1_mean"),
        "dev_best_non_oracle_baseline_pass_at_1": dev_variant.get("best_non_oracle_baseline_pass_at_1_mean"),
        "final_best_non_oracle_baseline": final_stats.get("best_non_oracle_baseline"),
        "final_best_non_oracle_baseline_pass_at_1": final_stats.get("best_non_oracle_baseline_pass_at_1_mean"),
        "dev_oracle_pass_at_8": dev_variant.get("oracle_pass_at_8_mean"),
        "final_oracle_pass_at_8": final_stats.get("oracle_pass_at_8_mean"),
        "evidence_corruption_degradation_dev": max_control_delta(
            dev_controls,
            (
                "candidate_evidence_mismatch",
                "cross_task_user_goal_shuffle",
                "cross_task_policy_shuffle",
                "cross_task_tool_observation_shuffle",
            ),
        ),
        "evidence_corruption_degradation_final": final_controls.get("evidence_corruption_max_delta"),
        "candidate_only_dev": dev_controls.get("candidate_only", {}),
        "candidate_only_final": final_control_summary.get("candidate_only", {}),
        "trajectory_only_dev": dev_controls.get("trajectory_only", {}),
        "trajectory_only_final": final_control_summary.get("trajectory_only", {}),
        "stage7b_success_gates": stage7b_results.get("stage7b_success_gates", {}),
    }


def evidence_use_analysis(
    selected_dev_rows: Sequence[Dict[str, object]],
    final_controls: Dict[str, object],
    final_stats: Dict[str, object],
) -> Dict[str, object]:
    del selected_dev_rows
    summary = dict(final_controls.get("control_summary", {}))
    method_summary = final_stats.get("method_summary", {})
    full = {
        "pass_at_1_mean": final_stats.get("trainable_pass_at_1_mean"),
        "conditional_accuracy_mean": method_summary.get("trainable_shared_weight_latent_selector", {}).get("conditional_selector_accuracy_mean"),
        "selection_efficiency_mean": final_stats.get("trainable_selection_efficiency_mean"),
    }
    names = (
        "candidate_only",
        "trajectory_only",
        "user_goal_only",
        "policy_only",
        "tool_observations_only",
        "candidate_evidence_mismatch",
        "cross_task_user_goal_shuffle",
        "cross_task_policy_shuffle",
        "cross_task_tool_observation_shuffle",
        "hidden_states_shuffled_across_examples",
        "schema_template_only",
    )
    controls = {name: summary.get(name, {}) for name in names}
    mismatch_delta = float(summary.get("candidate_evidence_mismatch", {}).get("base_minus_control_pass_at_1_mean", 0.0))
    cross_task_deltas = {
        name: float(summary.get(name, {}).get("base_minus_control_pass_at_1_mean", 0.0))
        for name in (
            "cross_task_user_goal_shuffle",
            "cross_task_policy_shuffle",
            "cross_task_tool_observation_shuffle",
        )
    }
    candidate_only_delta = float(summary.get("candidate_only", {}).get("base_minus_control_pass_at_1_mean", 0.0))
    trajectory_only_delta = float(summary.get("trajectory_only", {}).get("base_minus_control_pass_at_1_mean", 0.0))
    return {
        "full_model": full,
        "controls": controls,
        "did_evidence_mismatch_degrade": mismatch_delta >= 0.05,
        "candidate_evidence_mismatch_delta": mismatch_delta,
        "did_cross_task_shuffles_degrade": max(cross_task_deltas.values()) >= 0.05,
        "cross_task_shuffle_deltas": cross_task_deltas,
        "did_candidate_only_match_or_exceed_full_model": candidate_only_delta <= 0.0,
        "candidate_only_delta": candidate_only_delta,
        "did_trajectory_only_match_or_exceed_full_model": trajectory_only_delta <= 0.0,
        "trajectory_only_delta": trajectory_only_delta,
        "interpretation": (
            "The final model does not show reliable task-evidence use: evidence mismatch and cross-task shuffles "
            "do not degrade by the required margin, and candidate-only slightly exceeds the full model."
        ),
    }


def baseline_saturation_analysis(stats: Dict[str, object]) -> Dict[str, object]:
    method_summary = dict(stats.get("method_summary", {}))
    rows = []
    for name, row in sorted(method_summary.items()):
        if name in {"oracle_pass_at_8", "trainable_shared_weight_latent_selector"}:
            continue
        if bool(row.get("unavailable", False)):
            rows.append({"method": name, "unavailable": True})
            continue
        rows.append(
            {
                "method": name,
                "unavailable": False,
                "pass_at_1": row.get("pass_at_1_mean"),
                "conditional_accuracy": row.get("conditional_selector_accuracy_mean"),
                "selection_efficiency": row.get("selection_efficiency_mean"),
                "mrr": row.get("mrr_mean"),
                "top_2_accuracy": row.get("top_2_accuracy_mean"),
            }
        )
    available_rows = [row for row in rows if not bool(row.get("unavailable", False))]
    strongest = max(available_rows, key=lambda row: float(row.get("pass_at_1") or 0.0)) if available_rows else {}
    return {
        "baselines": rows,
        "strongest_baseline": strongest,
        "strongest_baseline_uses_similar_information_as_trainable_latent": strongest.get("method") in {
            "frozen_same_architecture_latent_selector",
            "raw_latent_selector",
            "text_only_single_reviewer",
            "text_only_multi_agent_reviewer",
            "static_trajectory_feature_reranker",
            "embedding_reranker",
        },
        "strongest_baseline_may_exploit_trajectory_or_source_artifacts": strongest.get("method")
        in {"best_generator_on_train_baseline", "best_generator_on_train_dev_baseline", "static_trajectory_feature_reranker"},
        "oracle_pass_at_8_leaves_room_for_improvement": 0.0 < float(stats.get("oracle_pass_at_8_mean", 0.0)) - float(strongest.get("pass_at_1") or 0.0),
        "oracle_minus_strongest_baseline": float(stats.get("oracle_pass_at_8_mean", 0.0)) - float(strongest.get("pass_at_1") or 0.0),
    }


def pool_difficulty_analysis(
    stage7c_results: Dict[str, object],
    stage7c_splits: Dict[str, object],
    candidate_pool_path: Path,
    candidate_source_map: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    candidates = load_taubench_candidates_jsonl(candidate_pool_path)
    examples = build_taubench_selection_examples(candidates)
    final_ids = set(str(value) for value in stage7c_splits.get("splits", {}).get("final", []))
    final_examples = [example for example in examples if f"{example.domain}:{example.task_id}" in final_ids]
    success_counts = [sum(example.labels_pass_fail) for example in final_examples]
    oracle_positive = sum(1 for count in success_counts if count > 0)
    oracle_empty = sum(1 for count in success_counts if count == 0)
    all_success = sum(1 for count in success_counts if count == 8)
    all_fail = oracle_empty
    generator_labels: Dict[str, List[int]] = defaultdict(list)
    task_type_counts = Counter()
    task_type_pass = Counter()
    for example in final_examples:
        task_type = str(example.metadata.get("task_type", "unknown"))
        task_type_counts[task_type] += 1
        task_type_pass[task_type] += int(any(example.labels_pass_fail))
        for candidate, label in zip(example.candidates, example.labels_pass_fail):
            generator = str(candidate_source_map.get(candidate.candidate_id, {}).get("generator_name_audit_only", "unknown"))
            generator_labels[generator].append(int(label))
    expected_random = float(np.mean([count / 8.0 for count in success_counts])) if success_counts else 0.0
    random_observed = [
        float(row.get("metrics", {}).get("final", {}).get("random_trajectory", {}).get("pass_at_1", 0.0))
        for row in stage7c_results.get("rows", [])
    ]
    near = stage7c_results.get("rows", [{}])[0].get("near_duplicate_trajectory_audit", {})
    duplicate = stage7c_results.get("rows", [{}])[0].get("duplicate_trajectory_hash_audit", {})
    return {
        "final_task_count": len(final_examples),
        "oracle_positive_count": oracle_positive,
        "oracle_empty_count": oracle_empty,
        "all_success_tasks": all_success,
        "all_fail_tasks": all_fail,
        "successful_candidates_per_task_distribution": dict(sorted(Counter(success_counts).items())),
        "successful_candidates_per_oracle_positive_task_distribution": dict(sorted(Counter(count for count in success_counts if count > 0).items())),
        "random_baseline_expected_per_success_count": expected_random,
        "random_baseline_observed_pass_at_1_mean": float(np.mean(random_observed)) if random_observed else 0.0,
        "per_task_type_breakdown": {
            key: {
                "tasks": int(task_type_counts[key]),
                "oracle_positive_rate": float(task_type_pass[key] / max(1, task_type_counts[key])),
            }
            for key in sorted(task_type_counts)
        },
        "per_generator_success_rates_audit_only": {
            generator: {
                "candidates": len(labels),
                "success_rate": float(np.mean(labels)) if labels else 0.0,
            }
            for generator, labels in sorted(generator_labels.items())
        },
        "near_duplicate_trajectory_rate": len(near.get("near_duplicates", [])) / float(max(1, sum(len(example.candidates) for example in final_examples))),
        "near_duplicate_audit": near,
        "duplicate_trajectory_audit": duplicate,
    }


def training_audit_sanity(stage7c_results: Dict[str, object], audit_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    rows = list(stage7c_results.get("rows", []))
    per_seed = []
    for row in rows:
        trainable = row.get("trainable_audit", {})
        frozen = row.get("frozen_audit", {})
        invariance = row.get("invariance_audit", {})
        per_seed.append(
            {
                "seed": row.get("seed"),
                "trainable_grad_norm_gt_0": float(trainable.get("agent_grad_norm_mean", 0.0)) > 0.0,
                "trainable_parameter_delta_gt_0": float(trainable.get("agent_parameter_delta", 0.0)) > 0.0,
                "frozen_agent_grad_delta_zero": float(frozen.get("agent_grad_norm_mean", 1.0)) == 0.0 and float(frozen.get("agent_parameter_delta", 1.0)) == 0.0,
                "coordinator_grad_delta_gt_0": float(trainable.get("coordinator_grad_norm_mean", 0.0)) > 0.0
                and float(trainable.get("coordinator_parameter_delta", 0.0)) > 0.0,
                "shared_parameter_identity": bool(trainable.get("shared_parameter_identity", False)),
                "activation_requires_grad_before_coordinator": bool(trainable.get("activation_requires_grad_before_coordinator", False)),
                "no_detach_in_main_path": bool(trainable.get("no_detach_between_clone_activations_and_loss", False)),
                "checkpoints_saved": all(Path(str(path)).exists() for path in row.get("checkpoint_paths", {}).values()),
                "candidate_order_invariance": bool(invariance.get("candidate_order_invariance_passes", False)),
                "physical_order_invariance": bool(invariance.get("physical_order_invariance_passes", False)),
                "leakage_audits_pass": bool(row.get("split_leakage_audit_passes", False)) and bool(row.get("output_leakage_audit_passes", False)),
            }
        )
    checks = {
        key: all(bool(row.get(key, False)) for row in per_seed)
        for key in (
            "trainable_grad_norm_gt_0",
            "trainable_parameter_delta_gt_0",
            "frozen_agent_grad_delta_zero",
            "coordinator_grad_delta_gt_0",
            "shared_parameter_identity",
            "activation_requires_grad_before_coordinator",
            "no_detach_in_main_path",
            "checkpoints_saved",
            "candidate_order_invariance",
            "physical_order_invariance",
            "leakage_audits_pass",
        )
    }
    return {
        "all_checks_pass": all(checks.values()),
        "aggregate_checks": checks,
        "per_seed": per_seed,
        "audit_row_count": len(audit_rows),
        "audit_types": dict(Counter(str(row.get("type", "")) for row in audit_rows)),
    }


def classify_failure(
    final_failure: Dict[str, object],
    dev_final: Dict[str, object],
    evidence: Dict[str, object],
    baselines: Dict[str, object],
    pool: Dict[str, object],
    training_audit: Dict[str, object],
    stage7c_results: Dict[str, object],
) -> Dict[str, Dict[str, object]]:
    final_gates = dict(stage7c_results.get("final_success_gates", {}))
    dev_trainable = float(dev_final.get("dev_trainable_pass_at_1") or 0.0)
    final_trainable = float(dev_final.get("final_trainable_pass_at_1") or 0.0)
    dev_delta_best = float(dev_final.get("dev_trainable_pass_at_1") or 0.0) - float(dev_final.get("dev_best_non_oracle_baseline_pass_at_1") or 0.0)
    final_delta_best = float(final_failure.get("trainable_vs_best_baseline_delta") or 0.0)
    categories = {
        "dev_overfitting": {
            "present": dev_trainable - final_trainable >= 0.10 or (dev_delta_best > final_delta_best + 0.05),
            "evidence": {
                "dev_trainable": dev_trainable,
                "final_trainable": final_trainable,
                "dev_trainable_minus_best": dev_delta_best,
                "final_trainable_minus_best": final_delta_best,
            },
        },
        "final_pool_distribution_shift": {
            "present": abs(float(dev_final.get("dev_oracle_pass_at_8") or 0.0) - float(dev_final.get("final_oracle_pass_at_8") or 0.0)) >= 0.05,
            "evidence": {
                "dev_oracle": dev_final.get("dev_oracle_pass_at_8"),
                "final_oracle": dev_final.get("final_oracle_pass_at_8"),
                "dev_final_trainable_gap": dev_trainable - final_trainable,
                "note": "Stage 7C reused the frozen held-out task IDs from the Stage 7B split, so the observed score drop is not attributed to oracle-rate distribution shift.",
            },
        },
        "oracle_pass_at_8_too_high_or_too_low": {
            "present": not (0.25 <= float(final_failure.get("oracle_pass_at_8") or 0.0) <= 0.85),
            "evidence": {"oracle_pass_at_8": final_failure.get("oracle_pass_at_8")},
        },
        "best_baseline_saturation": {
            "present": float(final_failure.get("trainable_vs_best_baseline_delta") or 0.0) < 0.0,
            "evidence": {
                "best_baseline": final_failure.get("best_non_oracle_baseline"),
                "best_baseline_pass_at_1": final_failure.get("best_non_oracle_baseline_pass_at_1"),
                "oracle_minus_best_baseline": baselines.get("oracle_minus_strongest_baseline"),
            },
        },
        "trainable_frozen_gap_absent": {
            "present": float(final_failure.get("trainable_vs_frozen_delta") or 0.0) <= 0.0,
            "evidence": {
                "trainable_vs_frozen_delta": final_failure.get("trainable_vs_frozen_delta"),
                "seeds_won_lost_vs_frozen": final_failure.get("seeds_won_lost_vs_frozen"),
            },
        },
        "evidence_controls_failed": {
            "present": not bool(final_gates.get("evidence_corruption_control_degrades_by_gte_0_05", False)),
            "evidence": {
                "evidence_corruption_final": dev_final.get("evidence_corruption_degradation_final"),
                "evidence_answers": {
                    "mismatch_degraded": evidence.get("did_evidence_mismatch_degrade"),
                    "cross_task_degraded": evidence.get("did_cross_task_shuffles_degrade"),
                },
            },
        },
        "candidate_only_or_trajectory_only_explains_result": {
            "present": bool(evidence.get("did_candidate_only_match_or_exceed_full_model")) or bool(evidence.get("did_trajectory_only_match_or_exceed_full_model")),
            "evidence": {
                "candidate_only_delta": evidence.get("candidate_only_delta"),
                "trajectory_only_delta": evidence.get("trajectory_only_delta"),
            },
        },
        "generator_identity_source_leakage": {
            "present": not bool(final_gates.get("leakage_audits_pass", False)),
            "evidence": {"leakage_gates_pass": final_gates.get("leakage_audits_pass")},
        },
        "hidden_evaluator_leakage": {
            "present": not bool(final_gates.get("leakage_audits_pass", False)),
            "evidence": {"leakage_gates_pass": final_gates.get("leakage_audits_pass")},
        },
        "insufficient_final_task_count": {
            "present": int(pool.get("final_task_count", 0)) < 50,
            "evidence": {"final_task_count": pool.get("final_task_count")},
        },
        "optimizer_training_instability": {
            "present": not bool(training_audit.get("all_checks_pass", False)),
            "evidence": training_audit.get("aggregate_checks", {}),
        },
        "architecture_mismatch": {
            "present": float(final_failure.get("trainable_vs_frozen_delta") or 0.0) <= 0.0
            and not bool(evidence.get("did_cross_task_shuffles_degrade")),
            "evidence": {
                "trainable_vs_frozen_delta": final_failure.get("trainable_vs_frozen_delta"),
                "evidence_corruption_final": dev_final.get("evidence_corruption_degradation_final"),
            },
        },
        "text_embedding_baselines_stronger_than_expected": {
            "present": any(
                float(row.get("pass_at_1") or 0.0) >= float(final_failure.get("trainable_latent_mean_pass_at_1") or 0.0)
                for row in baselines.get("baselines", [])
                if row.get("method") in {"embedding_reranker", "text_only_single_reviewer", "text_only_multi_agent_reviewer"}
            ),
            "evidence": {
                row["method"]: row.get("pass_at_1")
                for row in baselines.get("baselines", [])
                if row.get("method") in {"embedding_reranker", "text_only_single_reviewer", "text_only_multi_agent_reviewer"}
            },
        },
    }
    return categories


def choose_recommendation(
    categories: Dict[str, Dict[str, object]],
    final_failure: Dict[str, object],
    evidence: Dict[str, object],
    baselines: Dict[str, object],
    pool: Dict[str, object],
) -> str:
    del evidence
    baseline_dominates = bool(categories["best_baseline_saturation"]["present"]) and float(final_failure.get("trainable_vs_best_baseline_delta") or 0.0) < -0.05
    architecture_failed = bool(categories["architecture_mismatch"]["present"]) and bool(categories["trainable_frozen_gap_absent"]["present"])
    if baseline_dominates and architecture_failed:
        return "A. PAUSE_STAGE7"
    if bool(categories["insufficient_final_task_count"]["present"]) and not baseline_dominates:
        return "C. SCALE_STAGE7_POOL_FIRST"
    if bool(categories["evidence_controls_failed"]["present"]) or bool(categories["candidate_only_or_trajectory_only_explains_result"]["present"]):
        return "B. REDO_STAGE7_WITH_NEW_PREREGISTERED_DESIGN"
    if float(baselines.get("oracle_minus_strongest_baseline") or 0.0) <= 0.10 and int(pool.get("final_task_count", 0)) >= 50:
        return "D. RETURN_TO_STAGE4_5_PAPER"
    return "A. PAUSE_STAGE7"


def likely_blocker(categories: Dict[str, Dict[str, object]]) -> str:
    present = [key for key, row in categories.items() if bool(row.get("present", False))]
    if "architecture_mismatch" in present and "evidence_controls_failed" in present:
        return "architecture_and_view_design"
    if "best_baseline_saturation" in present:
        return "baseline_saturation"
    if "insufficient_final_task_count" in present:
        return "small_final_pool"
    return "mixed"


def conclusion_text(recommendation: str, categories: Dict[str, Dict[str, object]]) -> str:
    present = ", ".join(key for key, row in categories.items() if bool(row.get("present", False)))
    return (
        f"{recommendation}. Stage 7C failed as a final benchmark result. The dominant diagnosed categories are: "
        f"{present}. No Stage 7 success or final benchmark claim is supported by these artifacts."
    )


def aggregate_dev_controls(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    names = sorted({name for row in rows for name in row.get("controls", {})})
    out = {}
    for name in names:
        values = []
        deltas = []
        for row in rows:
            base = float(row["metrics"]["dev"]["trainable_shared_weight_latent_selector"]["pass_at_1"])
            control = row.get("controls", {}).get(name)
            if not isinstance(control, dict):
                continue
            value = float(control.get("pass_at_1", 0.0))
            values.append(value)
            deltas.append(base - value)
        out[name] = {
            "pass_at_1_mean": float(np.mean(values)) if values else 0.0,
            "base_minus_control_pass_at_1_mean": float(np.mean(deltas)) if deltas else 0.0,
            "pass_count": len(values),
        }
    return out


def max_control_delta(control_summary: Dict[str, Dict[str, float]], names: Iterable[str]) -> float:
    return max(float(control_summary.get(name, {}).get("base_minus_control_pass_at_1_mean", 0.0)) for name in names)


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_report(result: Dict[str, object]) -> str:
    final_failure = result["final_failure_summary"]
    dev_final = result["dev_vs_final_comparison"]
    evidence = result["evidence_use_analysis"]
    baselines = result["baseline_saturation_analysis"]
    pool = result["pool_difficulty_analysis"]
    audit = result["training_audit_sanity"]
    categories = result["failure_categories"]
    lines = [
        "# Stage 7C Failure Forensics",
        "",
        "## Conclusion",
        result["conclusion"],
        "",
        "No model, architecture, pool, split, hyperparameter, control, trajectory, or label was changed. No final benchmark claim is made.",
        "",
        "## 1. Final Failure",
        f"- Trainable latent mean pass@1: `{fmt(final_failure['trainable_latent_mean_pass_at_1'])}`",
        f"- Frozen latent mean pass@1: `{fmt(final_failure['frozen_latent_mean_pass_at_1'])}`",
        f"- Best non-oracle baseline: `{final_failure['best_non_oracle_baseline']}` at `{fmt(final_failure['best_non_oracle_baseline_pass_at_1'])}`",
        f"- Oracle pass@8: `{fmt(final_failure['oracle_pass_at_8'])}`",
        f"- Trainable vs frozen delta: `{fmt(final_failure['trainable_vs_frozen_delta'])}`",
        f"- Trainable vs best baseline delta: `{fmt(final_failure['trainable_vs_best_baseline_delta'])}`",
        f"- Seeds won/lost vs frozen: `{json.dumps(final_failure['seeds_won_lost_vs_frozen'], sort_keys=True)}`",
        f"- Seeds won/lost vs best baseline: `{json.dumps(final_failure['seeds_won_lost_vs_best_non_oracle'], sort_keys=True)}`",
        f"- CI vs frozen: `{json.dumps(final_failure['ci_vs_frozen'], sort_keys=True)}`",
        f"- CI vs best baseline: `{json.dumps(final_failure['ci_vs_best_baseline'], sort_keys=True)}`",
        f"- Failed gates: `{json.dumps(final_failure['failed_gates'])}`",
        "",
        "## 2. Stage 7B Dev vs Stage 7C Final",
        f"- Dev trainable: `{fmt(dev_final['dev_trainable_pass_at_1'])}`; final trainable: `{fmt(dev_final['final_trainable_pass_at_1'])}`",
        f"- Dev frozen: `{fmt(dev_final['dev_frozen_pass_at_1'])}`; final frozen: `{fmt(dev_final['final_frozen_pass_at_1'])}`",
        f"- Dev best baseline: `{fmt(dev_final['dev_best_non_oracle_baseline_pass_at_1'])}`; final best baseline: `{dev_final['final_best_non_oracle_baseline']}` at `{fmt(dev_final['final_best_non_oracle_baseline_pass_at_1'])}`",
        f"- Dev oracle pass@8: `{fmt(dev_final['dev_oracle_pass_at_8'])}`; final oracle pass@8: `{fmt(dev_final['final_oracle_pass_at_8'])}`",
        f"- Evidence corruption degradation dev vs final: `{fmt(dev_final['evidence_corruption_degradation_dev'])}` vs `{fmt(dev_final['evidence_corruption_degradation_final'])}`",
        f"- Candidate-only dev/final: `{json.dumps(dev_final['candidate_only_dev'], sort_keys=True)}` / `{json.dumps(dev_final['candidate_only_final'], sort_keys=True)}`",
        f"- Trajectory-only dev/final: `{json.dumps(dev_final['trajectory_only_dev'], sort_keys=True)}` / `{json.dumps(dev_final['trajectory_only_final'], sort_keys=True)}`",
        "",
        "## 3. Failure Categories",
    ]
    for key, row in categories.items():
        lines.append(f"- `{key}`: `{row['present']}` evidence=`{json.dumps(row['evidence'], sort_keys=True)}`")
    lines.extend(
        [
            "",
            "## 4. Evidence-Use Analysis",
            f"- Full model: `{json.dumps(evidence['full_model'], sort_keys=True)}`",
            f"- Controls: `{json.dumps(evidence['controls'], sort_keys=True)}`",
            f"- Did evidence mismatch degrade? `{evidence['did_evidence_mismatch_degrade']}` delta=`{fmt(evidence['candidate_evidence_mismatch_delta'])}`",
            f"- Did cross-task shuffles degrade? `{evidence['did_cross_task_shuffles_degrade']}` deltas=`{json.dumps(evidence['cross_task_shuffle_deltas'], sort_keys=True)}`",
            f"- Candidate-only matched/exceeded full model? `{evidence['did_candidate_only_match_or_exceed_full_model']}` delta=`{fmt(evidence['candidate_only_delta'])}`",
            f"- Trajectory-only matched/exceeded full model? `{evidence['did_trajectory_only_match_or_exceed_full_model']}` delta=`{fmt(evidence['trajectory_only_delta'])}`",
            f"- Interpretation: {evidence['interpretation']}",
            "",
            "## 5. Baseline Saturation",
        ]
    )
    for row in baselines["baselines"]:
        if bool(row.get("unavailable", False)):
            lines.append(f"- `{row['method']}` unavailable.")
            continue
        lines.append(
            f"- `{row['method']}` pass@1=`{fmt(row['pass_at_1'])}` conditional=`{fmt(row['conditional_accuracy'])}` "
            f"efficiency=`{fmt(row['selection_efficiency'])}` MRR=`{fmt(row['mrr'])}` top-2=`{fmt(row['top_2_accuracy'])}`"
        )
    lines.extend(
        [
            f"- Strongest baseline: `{json.dumps(baselines['strongest_baseline'], sort_keys=True)}`",
            f"- Strongest baseline may exploit trajectory/source artifacts: `{baselines['strongest_baseline_may_exploit_trajectory_or_source_artifacts']}`",
            f"- Oracle minus strongest baseline: `{fmt(baselines['oracle_minus_strongest_baseline'])}`",
            "",
            "## 6. Pool Difficulty",
            f"- Final task count: `{pool['final_task_count']}`",
            f"- Oracle-positive/empty: `{pool['oracle_positive_count']}` / `{pool['oracle_empty_count']}`",
            f"- All-success/all-fail tasks: `{pool['all_success_tasks']}` / `{pool['all_fail_tasks']}`",
            f"- Success-count distribution: `{json.dumps(pool['successful_candidates_per_task_distribution'], sort_keys=True)}`",
            f"- Oracle-positive success-count distribution: `{json.dumps(pool['successful_candidates_per_oracle_positive_task_distribution'], sort_keys=True)}`",
            f"- Random expected pass@1: `{fmt(pool['random_baseline_expected_per_success_count'])}`",
            f"- Per-task-type breakdown: `{json.dumps(pool['per_task_type_breakdown'], sort_keys=True)}`",
            f"- Per-generator success rates: `{json.dumps(pool['per_generator_success_rates_audit_only'], sort_keys=True)}`",
            f"- Near-duplicate trajectory rate: `{fmt(pool['near_duplicate_trajectory_rate'])}`",
            "",
            "## 7. Training/Audit Sanity",
            f"- Aggregate checks: `{json.dumps(audit['aggregate_checks'], sort_keys=True)}`",
            f"- Audit rows: `{audit['audit_row_count']}` types=`{json.dumps(audit['audit_types'], sort_keys=True)}`",
            "",
            "## 8. Decision Recommendation",
            f"`{result['decision_recommendation']}`",
            "",
            "The failure should not be softened: the final trainable selector underperformed frozen and non-oracle baselines, evidence controls did not support task-evidence use, and no Stage 7 final claim is supported.",
            "",
        ]
    )
    return "\n".join(lines)


def fmt(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


if __name__ == "__main__":
    main()
