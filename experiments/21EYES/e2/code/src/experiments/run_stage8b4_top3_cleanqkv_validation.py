from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from src.experiments.run_stage8b3_micro_evolution_tournament import (
    _evaluate_seed_n as evaluate_stage8b4_seed_n,
    _examples,
)
from src.experiments.stage8_architecture_registry import architecture_config_from_dict, architecture_config_to_dict
from src.experiments.stage8_capacity_evaluator import Stage8EvaluationConfig
from src.models.latent_attention_variants import Stage8ArchitectureConfig, estimate_stage8_compute


RESULTS_PATH = Path("results/stage8b4_top3_and_cleanqkv_results.json")
CONTROLS_AUDIT_PATH = Path("results/stage8b4_controls_audit.jsonl")
CLEANQKV_MEMORY_AUDIT_PATH = Path("results/stage8b4_cleanqkv_memory_audit.json")
VIEW_DIVERSITY_AUDIT_PATH = Path("results/stage8b4_view_diversity_audit.json")
COMPUTE_AUDIT_PATH = Path("results/stage8b4_compute_audit.json")
CAPACITY_CURVES_PATH = Path("results/stage8b4_capacity_curves.csv")
CANDIDATE_FOR_STAGE8C_PATH = Path("results/stage8b4_candidate_for_stage8c.yaml")
FREEZE_RECOMMENDATION_PATH = Path("results/stage8b4_freeze_recommendation.json")
REPORT_PATH = Path("reports/STAGE8B4_TOP3_VS_CLEANQKV_CHALLENGER.md")
STAGE8B3_FREEZE_MANIFEST_PATH = Path("results/stage8b3_top3_freeze_manifest.json")
BASELINE_CAPACITY_PATH = Path("results/stage8_baseline_capacity.json")

TOP3_CONTROLS: Tuple[str, ...] = (
    "randomized_labels",
    "candidate_order_shuffle_with_label_remap",
    "evidence_block_order_shuffle",
    "candidate_evidence_mismatch",
    "cross_task_evidence_shuffle",
    "cross_task_query_shuffle",
    "candidate_only",
    "query_only",
    "evidence_only",
    "distractor_only",
    "schema_template_only",
    "hidden_state_shuffle",
    "role_permutation",
    "avenue_permutation",
)

CLEANQKV_EXTRA_CONTROLS: Tuple[str, ...] = (
    "memory_slot_permutation",
    "memory_shuffle_across_examples",
    "memory_disabled",
    "memory_gate_forced_closed",
    "memory_gate_forced_open",
)

TASK_FAMILY_LABELS = (
    "Needle Binding",
    "Multi-Hop Binding",
    "Constraint Satisfaction",
    "Conflict Resolution",
    "Compositional Role Evidence",
    "Sparse Relevant Evidence",
)


@dataclass(frozen=True)
class Stage8B4Budget:
    n_values: Tuple[int, ...] = (8, 16, 32, 64, 128, 256)
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    train_examples: int = 48
    eval_examples: int = 48
    workers: int = 1
    include_n512: bool = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 8B.4 top-3 versus Clean-QKV challenger validation.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--train-examples", type=int, default=48)
    parser.add_argument("--eval-examples", type=int, default=48)
    parser.add_argument("--ten-seeds", action="store_true")
    parser.add_argument("--include-n512", action="store_true")
    args = parser.parse_args()
    seeds = tuple(range(10)) if args.ten_seeds else (0, 1, 2, 3, 4)
    n_values = (8, 16, 32, 64, 128, 256, 512) if args.include_n512 else (8, 16, 32, 64, 128, 256)
    budget = Stage8B4Budget(
        n_values=n_values,
        seeds=seeds,
        train_examples=int(args.train_examples),
        eval_examples=int(args.eval_examples),
        workers=max(1, int(args.workers)),
        include_n512=bool(args.include_n512),
    )
    payload = run_stage8b4(budget)
    print(f"stage8b4: wrote {RESULTS_PATH}, {FREEZE_RECOMMENDATION_PATH}, {REPORT_PATH}; decision={payload['decision']}")


def run_stage8b4(budget: Stage8B4Budget | None = None) -> Dict[str, object]:
    budget = budget or Stage8B4Budget()
    _reset_outputs()
    started = time.perf_counter()
    baseline_capacity = _load_stage8a_baseline_capacity()
    top3_configs = _load_stage8b3_top3_configs()
    challenger = _cleanqkv_challenger_config()
    architectures = [("stage8b3_top3", config) for config in top3_configs] + [("cleanqkv_challenger", challenger)]
    rows = _evaluate_architectures(architectures, budget, baseline_capacity)
    baselines = _evaluate_baselines(budget)
    ranked = _rank_architectures(rows)
    selected = _select_candidate_for_stage8c(ranked)
    decision = _decision(ranked, selected)
    payload = {
        "stage": "8B.4",
        "status": "completed",
        "decision": decision,
        "no_stage8c_run": True,
        "no_10x_attention_capacity_claim": True,
        "frozen_top3_source": str(STAGE8B3_FREEZE_MANIFEST_PATH),
        "frozen_top3_config_ids": [config.config_id for config in top3_configs],
        "cleanqkv_challenger_config_id": challenger.config_id,
        "budget": asdict(budget),
        "n_schedule": list(budget.n_values),
        "seeds": list(budget.seeds),
        "task_families": list(TASK_FAMILY_LABELS),
        "baseline_capacity_from_stage8a": baseline_capacity,
        "architectures": ranked,
        "baselines": baselines,
        "selected_candidate_for_stage8c": selected,
        "wall_clock_seconds": time.perf_counter() - started,
        "recommendation": _recommendation(decision, selected),
    }
    _dump_json(RESULTS_PATH, payload)
    _write_controls_audit(ranked)
    _dump_json(CLEANQKV_MEMORY_AUDIT_PATH, _cleanqkv_memory_audit(ranked))
    _dump_json(VIEW_DIVERSITY_AUDIT_PATH, _view_diversity_audit(ranked))
    _dump_json(COMPUTE_AUDIT_PATH, _compute_audit(ranked, baselines))
    _write_capacity_curves(ranked, baselines)
    _write_candidate_yaml(selected)
    _dump_json(FREEZE_RECOMMENDATION_PATH, _freeze_recommendation(payload, selected))
    _write_report(payload)
    return payload


def _evaluate_architectures(
    architectures: Sequence[Tuple[str, Stage8ArchitectureConfig]],
    budget: Stage8B4Budget,
    baseline_capacity: int,
) -> List[Dict[str, object]]:
    tasks = [(group, config, budget, baseline_capacity) for group, config in architectures]
    if budget.workers <= 1:
        return [_evaluate_architecture_task(task) for task in tasks]
    with ThreadPoolExecutor(max_workers=budget.workers) as executor:
        return list(executor.map(_evaluate_architecture_task, tasks))


def _evaluate_architecture_task(task) -> Dict[str, object]:
    group, config, budget, baseline_capacity = task
    controls = TOP3_CONTROLS + (CLEANQKV_EXTRA_CONTROLS if config.model_kind == "clean_qkv_activation_memory" else ())
    eval_config = Stage8EvaluationConfig(
        train_examples=budget.train_examples,
        eval_examples=budget.eval_examples,
        k_candidates=8,
        min_stable_seeds=len(budget.seeds),
        controls=tuple(controls),
    )
    started = time.perf_counter()
    rows = [
        evaluate_stage8b4_seed_n(config, int(n_blocks), int(seed), eval_config)
        for seed in budget.seeds
        for n_blocks in budget.n_values
    ]
    return _summarize_architecture(group, config, rows, controls, baseline_capacity, time.perf_counter() - started)


def _summarize_architecture(
    group: str,
    config: Stage8ArchitectureConfig,
    rows: Sequence[Mapping[str, object]],
    controls: Sequence[str],
    baseline_capacity: int,
    wall_clock: float,
) -> Dict[str, object]:
    controls_summary = _control_summary(rows)
    comparator_summary = _comparator_summary(rows)
    diversity = _diversity_summary(rows)
    accuracy_by_n = _mean_by_n(rows, "heldout_template_dev_accuracy")
    dev_accuracy_by_n = _mean_by_n(rows, "dev_accuracy")
    train_accuracy_by_n = _mean_by_n(rows, "train_accuracy")
    std_by_n = _std_by_n(rows, "heldout_template_dev_accuracy")
    min_by_n = _min_by_n(rows, "heldout_template_dev_accuracy")
    family = _task_family_summary(rows)
    capacity = _capacity_from_rows(rows, controls_summary)
    compute = estimate_stage8_compute(config, max(int(row.get("n_blocks", 0)) for row in rows), 8)
    heldout_mean = _mean(float(row.get("heldout_template_dev_accuracy", 0.0)) for row in rows)
    summary: Dict[str, object] = {
        "group": group,
        "config_id": config.config_id,
        "architecture_name": config.name,
        "model_kind": config.model_kind,
        "architecture_parameters": architecture_config_to_dict(config),
        "frozen_from_stage8b3": group == "stage8b3_top3",
        "cleanqkv_challenger": config.model_kind == "clean_qkv_activation_memory",
        "n_schedule": sorted({int(row.get("n_blocks", 0)) for row in rows}),
        "seeds": sorted({int(row.get("seed", 0)) for row in rows}),
        "train_accuracy": _mean(float(row.get("train_accuracy", 0.0)) for row in rows),
        "dev_template_accuracy": _mean(float(row.get("dev_accuracy", 0.0)) for row in rows),
        "held_out_template_dev_accuracy": heldout_mean,
        "train_dev_gap": _mean(float(row.get("train_accuracy", 0.0)) - float(row.get("heldout_template_dev_accuracy", 0.0)) for row in rows),
        "accuracy_by_N": accuracy_by_n,
        "dev_accuracy_by_N": dev_accuracy_by_n,
        "train_accuracy_by_N": train_accuracy_by_n,
        "std_accuracy_by_N": std_by_n,
        "min_accuracy_by_N": min_by_n,
        "accuracy_by_task_family": family,
        "capacity_C": capacity,
        "capacity_ratio_vs_stage8a_best_matched_baseline": float(capacity / baseline_capacity) if baseline_capacity > 0 else 0.0,
        "early_capacity_ratio": float(capacity / baseline_capacity) if baseline_capacity > 0 else 0.0,
        "capacity_ratio_note": "Stage 8A best matched baseline capacity is zero; ratio is recorded as 0.0, not as a 10x claim." if baseline_capacity <= 0 else "ratio_vs_stage8a_best_matched_baseline",
        "compute_normalized_capacity": capacity / max(1.0, float(compute.get("estimated_forward_compute", 1.0))),
        "randomized_label_accuracy": _control_metric(controls_summary, "randomized_labels", "mean_accuracy"),
        "candidate_evidence_mismatch_degradation": _first_control_metric(controls_summary, ("candidate_evidence_mismatch", "evidence_candidate_mismatch"), "mean_degradation"),
        "cross_task_evidence_shuffle_degradation": _control_metric(controls_summary, "cross_task_evidence_shuffle", "mean_degradation"),
        "candidate_only_accuracy": comparator_summary.get("candidate_only_accuracy", 0.0),
        "query_only_accuracy": comparator_summary.get("query_only_accuracy", 0.0),
        "evidence_only_accuracy": comparator_summary.get("evidence_only_accuracy", 0.0),
        "frozen_comparator_accuracy": comparator_summary.get("frozen_same_architecture_accuracy", 0.0),
        "frozen_comparator_gap": heldout_mean - comparator_summary.get("frozen_same_architecture_accuracy", 0.0),
        "raw_latent_comparator_accuracy": comparator_summary.get("raw_latent_accuracy", 0.0),
        "hidden_state_shuffle_accuracy": _control_metric(controls_summary, "hidden_state_shuffle", "mean_accuracy"),
        "hidden_state_shuffle_degradation": _control_metric(controls_summary, "hidden_state_shuffle", "mean_degradation"),
        "role_entropy": diversity["role_entropy"],
        "avenue_entropy": diversity["avenue_entropy"],
        "routing_entropy": diversity["routing_entropy"],
        "hidden_state_similarity": diversity["pairwise_hidden_state_similarity"],
        "memory_slot_entropy": diversity["memory_slot_entropy"],
        "memory_gate_calibration": diversity["memory_gate_calibration"],
        "parameter_count": int(compute.get("parameter_count_estimate", 0)),
        "estimated_forward_compute": compute,
        "wall_clock_time": wall_clock,
        "memory_usage": {"estimated_peak_bytes": 0, "source": "not_measured_cpu_stage8b4"},
        "controls_run": list(controls),
        "controls_summary": controls_summary,
        "comparator_summary": comparator_summary,
        "rows": list(rows),
    }
    summary["hard_disqualifiers"] = _hard_disqualifiers(summary)
    summary["controls_pass"] = not summary["hard_disqualifiers"]
    summary["ready_for_stage8c_freeze"] = _ready_for_stage8c(summary)
    return summary


def _evaluate_baselines(budget: Stage8B4Budget) -> Dict[str, object]:
    base = Stage8ArchitectureConfig(name="stage8b4_baseline", epochs=2, lr=0.01, hidden_dim=32, max_train_examples=budget.train_examples)
    configs = [
        replace(base, name="stage8b4_random_candidate", model_kind="random_candidate"),
        replace(base, name="stage8b4_candidate_only", model_kind="candidate_only"),
        replace(base, name="stage8b4_query_only", model_kind="query_only"),
        replace(base, name="stage8b4_evidence_only", model_kind="evidence_only"),
        replace(base, name="stage8b4_retrieval_topk", model_kind="retrieval_topk", top_k_views=4),
        replace(base, name="stage8b4_true_monolithic_transformer", model_kind="monolithic_transformer", roles=1, avenues=1),
        replace(base, name="stage8b4_legacy_hashed_feature_baseline", model_kind="retrieval_topk", top_k_views=1),
        replace(base, name="stage8b4_raw_latent_comparator", model_kind="raw_latent_selector"),
    ]
    eval_config = Stage8EvaluationConfig(train_examples=budget.train_examples, eval_examples=budget.eval_examples, controls=())
    baseline_rows = []
    for config in configs:
        rows = [
            evaluate_stage8b4_seed_n(config, int(n_blocks), int(seed), eval_config)
            for seed in budget.seeds
            for n_blocks in budget.n_values
        ]
        baseline_rows.append(
            {
                "architecture_name": config.name,
                "model_kind": config.model_kind,
                "accuracy_by_N": _mean_by_n(rows, "heldout_template_dev_accuracy"),
                "held_out_template_dev_accuracy": _mean(float(row.get("heldout_template_dev_accuracy", 0.0)) for row in rows),
                "accuracy_by_task_family": _task_family_summary(rows),
                "parameter_count": estimate_stage8_compute(config, max(budget.n_values), 8).get("parameter_count_estimate", 0),
                "estimated_forward_compute": estimate_stage8_compute(config, max(budget.n_values), 8),
                "rows": rows,
            }
        )
    return {"n_schedule": list(budget.n_values), "seeds": list(budget.seeds), "rows": baseline_rows}


def _load_stage8b3_top3_configs() -> List[Stage8ArchitectureConfig]:
    if not STAGE8B3_FREEZE_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"missing Stage 8B.3 freeze manifest: {STAGE8B3_FREEZE_MANIFEST_PATH}")
    payload = json.loads(STAGE8B3_FREEZE_MANIFEST_PATH.read_text(encoding="utf-8"))
    configs = []
    for row in payload.get("top3_full_configs", []):
        if isinstance(row, Mapping):
            configs.append(architecture_config_from_dict(dict(row)))
    if len(configs) != 3:
        raise RuntimeError(f"expected exactly 3 Stage 8B.3 top configs, found {len(configs)}")
    return configs


def _cleanqkv_challenger_config() -> Stage8ArchitectureConfig:
    return Stage8ArchitectureConfig(
        name="stage8b4_clean_qkv_activation_memory_challenger",
        model_kind="clean_qkv_activation_memory",
        roles=8,
        avenues=2,
        chunking="activation_memory_writer",
        candidate_query="clean_current_candidate_query",
        coordinator="clean_qkv_activation_memory_reader",
        view_sharing="memory_type_and_provenance_embeddings",
        memory_compression="activation_memory_slots",
        attention_routing="top_k_memory_routing",
        training_loss="multi_positive_cross_entropy",
        regularizers=("memory_gate_calibration", "memory_slot_load_balance"),
        curriculum="mixed_n",
        vector_dim=128,
        hidden_dim=32,
        coord_hops=1,
        top_k_views=4,
        epochs=4,
        lr=0.05,
        dropout=0.0,
        max_train_examples=128,
    )


def _hard_disqualifiers(row: Mapping[str, object]) -> List[str]:
    failures: List[str] = []
    heldout = float(row.get("held_out_template_dev_accuracy", 0.0))
    if float(row.get("randomized_label_accuracy", 1.0)) > 0.35:
        failures.append("randomized_labels_do_not_collapse")
    if float(row.get("candidate_only_accuracy", 1.0)) >= max(0.35, heldout - 0.10):
        failures.append("candidate_only_explains_result")
    if float(row.get("query_only_accuracy", 1.0)) >= max(0.35, heldout - 0.10):
        failures.append("query_only_explains_result")
    if float(row.get("candidate_evidence_mismatch_degradation", 0.0)) < 0.10:
        failures.append("evidence_mismatch_does_not_degrade")
    if float(row.get("frozen_comparator_gap", 0.0)) <= 0.05:
        failures.append("trainable_does_not_beat_frozen")
    if float(row.get("hidden_state_similarity", 1.0)) >= 0.85:
        failures.append("severe_view_or_memory_collapse")
    if float(row.get("dev_template_accuracy", 0.0)) - heldout > 0.20:
        failures.append("same_template_only_generalization")
    if bool(row.get("cleanqkv_challenger")):
        controls = row.get("controls_summary", {})
        memory_entropy = float(row.get("memory_slot_entropy", 0.0))
        if memory_entropy <= 0.10:
            failures.append("severe_memory_slot_collapse")
        if _control_metric(controls, "memory_disabled", "mean_degradation") < 0.10:
            failures.append("memory_disabled_does_not_degrade")
        if _control_metric(controls, "memory_shuffle_across_examples", "mean_degradation") < 0.10:
            failures.append("memory_shuffle_does_not_degrade")
        if _control_metric(controls, "memory_gate_forced_closed", "mean_degradation") < 0.10:
            failures.append("forced_closed_gate_does_not_degrade")
        slot_perm_delta = abs(_control_metric(controls, "memory_slot_permutation", "mean_delta_from_base"))
        if slot_perm_delta > 0.05:
            failures.append("memory_slot_permutation_not_invariant")
    return failures


def _ready_for_stage8c(row: Mapping[str, object]) -> bool:
    return bool(
        not row.get("hard_disqualifiers")
        and int(row.get("capacity_C", 0)) >= 64
        and _accuracy_at(row, 64) >= 0.85
        and float(row.get("cross_task_evidence_shuffle_degradation", 0.0)) >= 0.10
        and float(row.get("candidate_evidence_mismatch_degradation", 0.0)) >= 0.10
        and float(row.get("frozen_comparator_gap", 0.0)) > 0.05
    )


def _capacity_from_rows(rows: Sequence[Mapping[str, object]], controls: Mapping[str, object]) -> int:
    capacity = 0
    control_gate = (
        _first_control_metric(controls, ("candidate_evidence_mismatch", "evidence_candidate_mismatch"), "mean_degradation") >= 0.10
        and _control_metric(controls, "cross_task_evidence_shuffle", "mean_degradation") >= 0.10
        and _control_metric(controls, "randomized_labels", "mean_accuracy") <= 0.35
    )
    grouped: Dict[int, List[float]] = {}
    for row in rows:
        grouped.setdefault(int(row.get("n_blocks", 0)), []).append(float(row.get("heldout_template_dev_accuracy", 0.0)))
    for n_blocks, values in sorted(grouped.items()):
        if _mean(values) >= 0.85 and min(values) >= 0.75 and (pstdev(values) if len(values) > 1 else 0.0) <= 0.08 and control_gate:
            capacity = max(capacity, n_blocks)
    return capacity


def _rank_architectures(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    output = []
    for row in rows:
        copy = dict(row)
        stability = max(0.0, 1.0 - _mean(float(value) for value in copy.get("std_accuracy_by_N", {}).values())) if isinstance(copy.get("std_accuracy_by_N"), Mapping) else 0.0
        collapse_score = max(0.0, 1.0 - float(copy.get("hidden_state_similarity", 1.0)))
        copy["selection_tuple"] = [
            int(copy.get("capacity_C", 0)),
            1 if copy.get("controls_pass") else 0,
            float(copy.get("held_out_template_dev_accuracy", 0.0)),
            float(copy.get("frozen_comparator_gap", 0.0)),
            collapse_score,
            float(copy.get("compute_normalized_capacity", 0.0)),
            stability,
        ]
        output.append(copy)
    output.sort(key=lambda row: tuple(row["selection_tuple"]), reverse=True)
    return output


def _select_candidate_for_stage8c(rows: Sequence[Mapping[str, object]]) -> Dict[str, object] | None:
    ready = [row for row in rows if row.get("ready_for_stage8c_freeze")]
    if not ready:
        return None
    selected = dict(ready[0])
    return {
        "architecture_name": selected.get("architecture_name"),
        "config_id": selected.get("config_id"),
        "model_kind": selected.get("model_kind"),
        "capacity_C": selected.get("capacity_C"),
        "accuracy_by_N": selected.get("accuracy_by_N"),
        "held_out_template_dev_accuracy": selected.get("held_out_template_dev_accuracy"),
        "dev_template_accuracy": selected.get("dev_template_accuracy"),
        "controls_pass": selected.get("controls_pass"),
        "hard_disqualifiers": selected.get("hard_disqualifiers"),
        "candidate_evidence_mismatch_degradation": selected.get("candidate_evidence_mismatch_degradation"),
        "cross_task_evidence_shuffle_degradation": selected.get("cross_task_evidence_shuffle_degradation"),
        "frozen_comparator_gap": selected.get("frozen_comparator_gap"),
        "hidden_state_similarity": selected.get("hidden_state_similarity"),
        "ready_for_stage8c_freeze": selected.get("ready_for_stage8c_freeze"),
        "architecture_parameters": selected.get("architecture_parameters"),
        "freeze_scope": "candidate_for_stage8c_only_stage8c_not_run",
    }


def _decision(rows: Sequence[Mapping[str, object]], selected: Mapping[str, object] | None) -> str:
    clean = next((row for row in rows if row.get("cleanqkv_challenger")), None)
    top3_rows = [row for row in rows if row.get("frozen_from_stage8b3")]
    best_top3 = top3_rows[0] if top3_rows else None
    if selected is None:
        return "STAGE8B4_NO_ARCHITECTURE_READY"
    if clean is not None and best_top3 is not None:
        clean_pass = bool(clean.get("controls_pass")) and _cleanqkv_memory_controls_pass(clean)
        clean_capacity = int(clean.get("capacity_C", 0))
        top3_capacity = int(best_top3.get("capacity_C", 0))
        clean_heldout = float(clean.get("held_out_template_dev_accuracy", 0.0))
        top3_heldout = float(best_top3.get("held_out_template_dev_accuracy", 0.0))
        if clean_pass and (
            clean_capacity > top3_capacity
            or clean_heldout > top3_heldout + 0.0001
        ):
            return "STAGE8B4_CLEANQKV_WINS_DEV"
        if clean_pass and abs(clean_heldout - top3_heldout) <= 0.05:
            return "STAGE8B4_CLEANQKV_COMPETITIVE"
        if best_top3.get("ready_for_stage8c_freeze") and (
            not clean_pass
            or top3_capacity > clean_capacity
            or top3_heldout > clean_heldout + 0.05
        ):
            return "STAGE8B4_TOP3_WIN_CLEANQKV_NOT_COMPETITIVE"
    return "STAGE8B4_READY_TO_FREEZE_FOR_STAGE8C"


def _cleanqkv_memory_controls_pass(row: Mapping[str, object]) -> bool:
    controls = row.get("controls_summary", {})
    if not isinstance(controls, Mapping):
        return False
    return (
        _control_metric(controls, "memory_disabled", "mean_degradation") >= 0.10
        and _control_metric(controls, "memory_shuffle_across_examples", "mean_degradation") >= 0.10
        and _control_metric(controls, "memory_gate_forced_closed", "mean_degradation") >= 0.10
        and abs(_control_metric(controls, "memory_slot_permutation", "mean_delta_from_base")) <= 0.05
    )


def _recommendation(decision: str, selected: Mapping[str, object] | None) -> str:
    if selected and decision in {"STAGE8B4_READY_TO_FREEZE_FOR_STAGE8C", "STAGE8B4_CLEANQKV_WINS_DEV"}:
        return f"Freeze candidate for Stage 8C consideration: {selected.get('architecture_name')}. Stage 8C has not been run."
    if decision == "STAGE8B4_CLEANQKV_COMPETITIVE":
        return "Clean-QKV is competitive; run an additional Stage 8B.4b challenger-focused validation before any Stage 8C freeze."
    if decision == "STAGE8B4_TOP3_WIN_CLEANQKV_NOT_COMPETITIVE":
        return "Keep the best frozen Stage 8B.3 architecture as the only Stage 8C candidate; do not promote Clean-QKV."
    return "No architecture should be frozen for Stage 8C from this run."


def _control_summary(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    grouped: Dict[str, List[Mapping[str, object]]] = {}
    for row in rows:
        controls = row.get("controls", {})
        if isinstance(controls, Mapping):
            for name, payload in controls.items():
                if isinstance(payload, Mapping):
                    grouped.setdefault(str(name), []).append(payload)
    return {
        name: {
            "mean_accuracy": _mean(float(row.get("accuracy", 0.0)) for row in values),
            "mean_degradation": _mean(float(row.get("degradation", 0.0)) for row in values),
            "mean_delta_from_base": _mean(float(row.get("delta_from_base", 0.0)) for row in values),
            "pass_rate": _mean(1.0 if row.get("passes") else 0.0 for row in values),
            "count": len(values),
        }
        for name, values in sorted(grouped.items())
    }


def _comparator_summary(rows: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = {}
    for row in rows:
        comparators = row.get("comparators", {})
        if isinstance(comparators, Mapping):
            for name, value in comparators.items():
                grouped.setdefault(str(name), []).append(float(value))
    return {name: _mean(values) for name, values in sorted(grouped.items())}


def _diversity_summary(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    role_entropies = []
    avenue_entropies = []
    routing = []
    pairwise = []
    memory_entropy = []
    gate_values = []
    gate_relevant = []
    gate_irrelevant = []
    for row in rows:
        diag = row.get("diagnostics", {})
        if not isinstance(diag, Mapping):
            continue
        role = diag.get("role_usage_distribution")
        avenue = diag.get("avenue_usage_distribution")
        if isinstance(role, list):
            role_entropies.append(_entropy(role) / math.log(max(2, len(role))))
        if isinstance(avenue, list):
            avenue_entropies.append(_entropy(avenue) / math.log(max(2, len(avenue))))
        if "routing_entropy_mean" in diag:
            routing.append(float(diag["routing_entropy_mean"]))
        if "pairwise_hidden_state_similarity" in diag:
            pairwise.append(float(diag["pairwise_hidden_state_similarity"]))
        if "memory_slot_entropy_mean" in diag:
            memory_entropy.append(float(diag["memory_slot_entropy_mean"]))
        if "memory_gate_mean" in diag:
            gate_values.append(float(diag["memory_gate_mean"]))
        if "memory_gate_relevant_mean" in diag:
            gate_relevant.append(float(diag["memory_gate_relevant_mean"]))
        if "memory_gate_irrelevant_mean" in diag:
            gate_irrelevant.append(float(diag["memory_gate_irrelevant_mean"]))
    return {
        "role_entropy": _mean(role_entropies),
        "avenue_entropy": _mean(avenue_entropies),
        "routing_entropy": _mean(routing),
        "pairwise_hidden_state_similarity": _mean(pairwise) if pairwise else 1.0,
        "memory_slot_entropy": _mean(memory_entropy),
        "memory_gate_calibration": {
            "mean_gate": _mean(gate_values),
            "relevant_mean_gate": _mean(gate_relevant),
            "irrelevant_mean_gate": _mean(gate_irrelevant),
        },
    }


def _task_family_summary(rows: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = {}
    for row in rows:
        values = row.get("accuracy_by_task_family", {})
        if isinstance(values, Mapping):
            for family, accuracy in values.items():
                grouped.setdefault(str(family), []).append(float(accuracy))
    return {family: _mean(values) for family, values in sorted(grouped.items())}


def _write_controls_audit(rows: Sequence[Mapping[str, object]]) -> None:
    CONTROLS_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONTROLS_AUDIT_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "stage": "8B.4",
                        "config_id": row.get("config_id"),
                        "architecture_name": row.get("architecture_name"),
                        "model_kind": row.get("model_kind"),
                        "controls_summary": row.get("controls_summary"),
                        "hard_disqualifiers": row.get("hard_disqualifiers"),
                        "controls_pass": row.get("controls_pass"),
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def _cleanqkv_memory_audit(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    clean = [row for row in rows if row.get("cleanqkv_challenger")]
    return {
        "stage": "8B.4",
        "rows": [
            {
                "config_id": row.get("config_id"),
                "architecture_name": row.get("architecture_name"),
                "memory_slot_entropy": row.get("memory_slot_entropy"),
                "memory_gate_calibration": row.get("memory_gate_calibration"),
                "memory_disabled_degradation": _control_metric(row.get("controls_summary", {}), "memory_disabled", "mean_degradation"),
                "memory_shuffle_across_examples_degradation": _control_metric(row.get("controls_summary", {}), "memory_shuffle_across_examples", "mean_degradation"),
                "memory_slot_permutation_delta": _control_metric(row.get("controls_summary", {}), "memory_slot_permutation", "mean_delta_from_base"),
                "memory_gate_forced_closed_degradation": _control_metric(row.get("controls_summary", {}), "memory_gate_forced_closed", "mean_degradation"),
                "memory_gate_forced_open_accuracy": _control_metric(row.get("controls_summary", {}), "memory_gate_forced_open", "mean_accuracy"),
                "memory_controls_pass": _cleanqkv_memory_controls_pass(row),
            }
            for row in clean
        ],
    }


def _view_diversity_audit(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    return {
        "stage": "8B.4",
        "stage8b1_reference_mean_pairwise_similarity": 0.7086284708390153,
        "rows": [
            {
                "config_id": row.get("config_id"),
                "architecture_name": row.get("architecture_name"),
                "model_kind": row.get("model_kind"),
                "role_entropy": row.get("role_entropy"),
                "avenue_entropy": row.get("avenue_entropy"),
                "routing_entropy": row.get("routing_entropy"),
                "hidden_state_similarity": row.get("hidden_state_similarity"),
                "memory_slot_entropy": row.get("memory_slot_entropy"),
                "severe_collapse": float(row.get("hidden_state_similarity", 1.0)) >= 0.85,
            }
            for row in rows
        ],
    }


def _compute_audit(rows: Sequence[Mapping[str, object]], baselines: Mapping[str, object]) -> Dict[str, object]:
    return {
        "stage": "8B.4",
        "architectures": [
            {
                "config_id": row.get("config_id"),
                "architecture_name": row.get("architecture_name"),
                "capacity_C": row.get("capacity_C"),
                "parameter_count": row.get("parameter_count"),
                "estimated_forward_compute": row.get("estimated_forward_compute"),
                "compute_normalized_capacity": row.get("compute_normalized_capacity"),
                "wall_clock_time": row.get("wall_clock_time"),
                "memory_usage": row.get("memory_usage"),
            }
            for row in rows
        ],
        "baselines": [
            {
                "architecture_name": row.get("architecture_name"),
                "model_kind": row.get("model_kind"),
                "parameter_count": row.get("parameter_count"),
                "estimated_forward_compute": row.get("estimated_forward_compute"),
            }
            for row in baselines.get("rows", [])
        ] if isinstance(baselines, Mapping) else [],
    }


def _write_capacity_curves(rows: Sequence[Mapping[str, object]], baselines: Mapping[str, object]) -> None:
    CAPACITY_CURVES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CAPACITY_CURVES_PATH.open("w", encoding="utf-8") as handle:
        handle.write("kind,config_id,architecture_name,model_kind,n_blocks,seed,dev_accuracy,heldout_template_dev_accuracy,capacity_C,estimated_compute\n")
        for row in rows:
            for seed_row in row.get("rows", []):  # type: ignore[union-attr]
                if isinstance(seed_row, Mapping):
                    _write_curve_row(handle, "architecture", row, seed_row)
        for baseline in baselines.get("rows", []) if isinstance(baselines, Mapping) else []:
            for seed_row in baseline.get("rows", []):  # type: ignore[union-attr]
                if isinstance(seed_row, Mapping):
                    _write_curve_row(handle, "baseline", baseline, seed_row)


def _write_curve_row(handle, kind: str, row: Mapping[str, object], seed_row: Mapping[str, object]) -> None:
    compute = seed_row.get("compute", {})
    handle.write(
        "{kind},{config_id},{name},{model_kind},{n},{seed},{dev:.6f},{heldout:.6f},{capacity},{compute:.3f}\n".format(
            kind=kind,
            config_id=row.get("config_id", ""),
            name=str(row.get("architecture_name", "")).replace(",", "_"),
            model_kind=row.get("model_kind", ""),
            n=int(seed_row.get("n_blocks", 0)),
            seed=int(seed_row.get("seed", 0)),
            dev=float(seed_row.get("dev_accuracy", 0.0)),
            heldout=float(seed_row.get("heldout_template_dev_accuracy", 0.0)),
            capacity=int(row.get("capacity_C", 0)),
            compute=float(compute.get("estimated_forward_compute", 0.0)) if isinstance(compute, Mapping) else 0.0,
        )
    )


def _write_candidate_yaml(selected: Mapping[str, object] | None) -> None:
    CANDIDATE_FOR_STAGE8C_PATH.parent.mkdir(parents=True, exist_ok=True)
    if selected is None:
        CANDIDATE_FOR_STAGE8C_PATH.write_text(
            "stage: \"8B.4\"\nfreeze_candidate: null\nstage8c_not_run: true\nno_10x_attention_capacity_claim: true\n",
            encoding="utf-8",
        )
        return
    lines = [
        "stage: \"8B.4\"",
        "stage8c_not_run: true",
        "no_10x_attention_capacity_claim: true",
        f"freeze_candidate_name: {json.dumps(selected.get('architecture_name'))}",
        f"freeze_candidate_config_id: {json.dumps(selected.get('config_id'))}",
        f"capacity_C: {json.dumps(selected.get('capacity_C'))}",
        "architecture:",
    ]
    params = selected.get("architecture_parameters", {})
    if isinstance(params, Mapping):
        for key, value in params.items():
            lines.append(f"  {key}: {json.dumps(value)}")
    CANDIDATE_FOR_STAGE8C_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _freeze_recommendation(payload: Mapping[str, object], selected: Mapping[str, object] | None) -> Dict[str, object]:
    freeze_decisions = {
        "STAGE8B4_READY_TO_FREEZE_FOR_STAGE8C",
        "STAGE8B4_CLEANQKV_WINS_DEV",
        "STAGE8B4_TOP3_WIN_CLEANQKV_NOT_COMPETITIVE",
    }
    return {
        "stage": "8B.4",
        "decision": payload.get("decision"),
        "recommend_freeze_for_stage8c": selected is not None and payload.get("decision") in freeze_decisions,
        "selected_candidate": selected,
        "stage8c_not_run": True,
        "no_10x_attention_capacity_claim": True,
        "freeze_conditions": {
            "reaches_N64_accuracy_ge_085": bool(selected and _accuracy_at(selected, 64) >= 0.85),
            "controls_pass": bool(selected and selected.get("controls_pass")),
            "evidence_controls_degrade": bool(selected and selected.get("candidate_evidence_mismatch_degradation", 0.0) >= 0.10 and selected.get("cross_task_evidence_shuffle_degradation", 0.0) >= 0.10),
            "frozen_comparator_lower": bool(selected and selected.get("frozen_comparator_gap", 0.0) > 0.05),
        },
    }


def _write_report(payload: Mapping[str, object]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = payload.get("architectures", []) if isinstance(payload.get("architectures"), list) else []
    selected = payload.get("selected_candidate_for_stage8c")
    lines = [
        "# Stage 8B.4 Top-3 Larger-Scale Validation vs Clean-QKV Challenger",
        "",
        "## Executive Summary",
        "",
        f"- Decision: {payload.get('decision')}",
        "- Stage 8C was not run.",
        "- No 10x attention-capacity claim is made.",
        f"- Selected Stage 8C candidate: {selected.get('architecture_name') if isinstance(selected, Mapping) else 'none'}",
        f"- Recommendation: {payload.get('recommendation')}",
        "",
        "## Validation Design",
        "",
        f"- N schedule: {payload.get('n_schedule')}",
        f"- Seeds: {payload.get('seeds')}",
        "- Splits: dev templates and held-out-template dev only; final Stage 8C templates were not used.",
        "- Frozen Stage 8B.3 top-three configs were loaded from the freeze manifest without modification.",
        "",
        "## Architecture Results",
        "",
        "| Architecture | Kind | C | Heldout dev | N64 | Mismatch degrade | Cross-task degrade | Frozen gap | Collapse sim | Ready |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {kind} | {capacity} | {heldout:.4f} | {n64:.4f} | {mismatch:.4f} | {cross:.4f} | {frozen:.4f} | {collapse:.4f} | {ready} |".format(
                name=row.get("architecture_name"),
                kind=row.get("model_kind"),
                capacity=row.get("capacity_C"),
                heldout=float(row.get("held_out_template_dev_accuracy", 0.0)),
                n64=_accuracy_at(row, 64),
                mismatch=float(row.get("candidate_evidence_mismatch_degradation", 0.0)),
                cross=float(row.get("cross_task_evidence_shuffle_degradation", 0.0)),
                frozen=float(row.get("frozen_comparator_gap", 0.0)),
                collapse=float(row.get("hidden_state_similarity", 1.0)),
                ready=row.get("ready_for_stage8c_freeze"),
            )
        )
    lines.extend(
        [
            "",
            "## Clean-QKV Memory Controls",
            "",
            json.dumps(_cleanqkv_memory_audit(rows), indent=2, sort_keys=True),
            "",
            "## Baseline Comparison",
            "",
            json.dumps(_baseline_brief(payload.get("baselines", {})), indent=2, sort_keys=True),
            "",
            "## Failure and Disqualification Notes",
            "",
        ]
    )
    for row in rows:
        lines.append(f"- {row.get('architecture_name')}: {row.get('hard_disqualifiers')}")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Results: `{RESULTS_PATH}`",
            f"- Controls audit: `{CONTROLS_AUDIT_PATH}`",
            f"- Clean-QKV memory audit: `{CLEANQKV_MEMORY_AUDIT_PATH}`",
            f"- View diversity audit: `{VIEW_DIVERSITY_AUDIT_PATH}`",
            f"- Compute audit: `{COMPUTE_AUDIT_PATH}`",
            f"- Capacity curves: `{CAPACITY_CURVES_PATH}`",
            f"- Candidate for Stage 8C: `{CANDIDATE_FOR_STAGE8C_PATH}`",
            f"- Freeze recommendation: `{FREEZE_RECOMMENDATION_PATH}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _baseline_brief(baselines: object) -> Dict[str, object]:
    if not isinstance(baselines, Mapping):
        return {}
    return {
        "n_schedule": baselines.get("n_schedule"),
        "seeds": baselines.get("seeds"),
        "rows": [
            {
                "architecture_name": row.get("architecture_name"),
                "model_kind": row.get("model_kind"),
                "held_out_template_dev_accuracy": row.get("held_out_template_dev_accuracy"),
                "accuracy_by_N": row.get("accuracy_by_N"),
            }
            for row in baselines.get("rows", [])
        ],
    }


def _load_stage8a_baseline_capacity() -> int:
    if not BASELINE_CAPACITY_PATH.exists():
        return 0
    payload = json.loads(BASELINE_CAPACITY_PATH.read_text(encoding="utf-8"))
    best = payload.get("best_monolithic_capacity", {})
    if isinstance(best, Mapping):
        return int(best.get("capacity", 0))
    return 0


def _mean_by_n(rows: Sequence[Mapping[str, object]], key: str) -> Dict[str, float]:
    grouped: Dict[int, List[float]] = {}
    for row in rows:
        grouped.setdefault(int(row.get("n_blocks", 0)), []).append(float(row.get(key, 0.0)))
    return {str(n): _mean(values) for n, values in sorted(grouped.items())}


def _std_by_n(rows: Sequence[Mapping[str, object]], key: str) -> Dict[str, float]:
    grouped: Dict[int, List[float]] = {}
    for row in rows:
        grouped.setdefault(int(row.get("n_blocks", 0)), []).append(float(row.get(key, 0.0)))
    return {str(n): (pstdev(values) if len(values) > 1 else 0.0) for n, values in sorted(grouped.items())}


def _min_by_n(rows: Sequence[Mapping[str, object]], key: str) -> Dict[str, float]:
    grouped: Dict[int, List[float]] = {}
    for row in rows:
        grouped.setdefault(int(row.get("n_blocks", 0)), []).append(float(row.get(key, 0.0)))
    return {str(n): min(values) if values else 0.0 for n, values in sorted(grouped.items())}


def _control_metric(controls: object, control: str, key: str) -> float:
    if isinstance(controls, Mapping):
        row = controls.get(control, {})
        if isinstance(row, Mapping):
            return float(row.get(key, 0.0))
    return 0.0


def _first_control_metric(controls: Mapping[str, object], names: Sequence[str], key: str) -> float:
    for name in names:
        if name in controls:
            return _control_metric(controls, name, key)
    return 0.0


def _accuracy_at(row: Mapping[str, object], n_blocks: int) -> float:
    values = row.get("accuracy_by_N", {})
    if isinstance(values, Mapping):
        return float(values.get(str(n_blocks), values.get(n_blocks, 0.0)))
    return 0.0


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return mean(rows) if rows else 0.0


def _entropy(values: Sequence[float]) -> float:
    total = sum(float(value) for value in values)
    if total <= 0:
        return 0.0
    return -sum((float(value) / total) * math.log(max(1e-12, float(value) / total)) for value in values if value > 0)


def _reset_outputs() -> None:
    for path in (CONTROLS_AUDIT_PATH, CAPACITY_CURVES_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


def _dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
