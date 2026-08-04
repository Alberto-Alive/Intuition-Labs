from __future__ import annotations

import json
import hashlib
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from src.datasets.latent_attention_capacity_dataset import SCALE_SCHEDULE, TASK_FAMILIES
from src.experiments.stage8_architecture_registry import (
    architecture_config_to_dict,
    default_latent_config,
    generate_stage8_search_space,
)
from src.experiments.stage8_capacity_evaluator import (
    Stage8EvaluationConfig,
    capacity_ratio,
    dump_json,
    evaluate_stage8_architecture,
    paired_bootstrap_ci,
    write_capacity_curves_csv,
)
from src.experiments.stage8_controls import REQUIRED_STAGE8_CONTROLS
from src.models.latent_attention_variants import Stage8ArchitectureConfig, estimate_stage8_compute


BASELINE_CAPACITY_PATH = Path("results/stage8_baseline_capacity.json")
STAGE8A_RESULTS_PATH = Path("results/stage8_stage8a_results.jsonl")
STAGE8A_CONTROLS_AUDIT_PATH = Path("results/stage8_stage8a_controls_audit.jsonl")
SEARCH_DATABASE_PATH = Path("results/stage8_search_database.jsonl")
PROMOTED_CONFIGS_PATH = Path("results/stage8_promoted_configs.json")
FINALIST_RESULTS_PATH = Path("results/stage8_finalist_results.json")
FINAL_VALIDATION_RESULTS_PATH = Path("results/stage8_final_validation_results.json")
CONTROLS_AUDIT_PATH = Path("results/stage8_controls_audit.jsonl")
STAGE8B_CONTROLS_AUDIT_PATH = Path("results/stage8_stage8b_controls_audit.jsonl")
SHORTCUT_AUDIT_PATH = Path("results/stage8_shortcut_audit.json")
COMPUTE_AUDIT_PATH = Path("results/stage8_compute_audit.json")
STAGE8B_COMPUTE_AUDIT_PATH = Path("results/stage8_stage8b_compute_audit.json")
CAPACITY_CURVES_PATH = Path("results/stage8_capacity_curves.csv")
STAGE8B_CAPACITY_CURVES_PATH = Path("results/stage8_stage8b_capacity_curves.csv")
BEST_CONFIG_PATH = Path("results/stage8_best_config.yaml")
CONFIG_FREEZE_MANIFEST_PATH = Path("results/stage8_config_freeze_manifest.json")
CHECKPOINT_DIR = Path("results/stage8_checkpoints")
STAGE8A_REPORT_PATH = Path("reports/STAGE8A_BASELINE_CAPACITY.md")
STAGE8B_REPORT_PATH = Path("reports/STAGE8B_ARCHITECTURE_SEARCH.md")


@dataclass(frozen=True)
class Stage8SearchBudget:
    max_architecture_configs: int = 50
    max_promoted_configs: int = 8
    max_finalists: int = 2
    max_wall_clock_seconds: float = 1800.0
    max_n: int = 128
    stage8a_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    stage8b_small_seeds: Tuple[int, ...] = (0,)
    stage8b_promoted_seeds: Tuple[int, ...] = (0, 1, 2)
    stage8b_finalist_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    stage8c_seeds: Tuple[int, ...] = (10, 11, 12, 13, 14, 15, 16, 17, 18, 19)
    train_examples: int = 192
    eval_examples: int = 96
    final_examples: int = 128
    run_controls: bool = True


def run_stage8a_baseline_validation(
    budget: Stage8SearchBudget,
    eval_config: Stage8EvaluationConfig | None = None,
) -> Dict[str, object]:
    eval_config = _stage8a_eval_config(eval_config or _eval_config(budget), budget)
    STAGE8A_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STAGE8A_RESULTS_PATH.touch(exist_ok=True)
    STAGE8A_CONTROLS_AUDIT_PATH.touch(exist_ok=True)
    n_values = _n_values(budget.max_n)
    rows = _load_database(STAGE8A_RESULTS_PATH)
    completed = {str(row.get("config_id")) for row in rows if row.get("status") == "completed"}
    for baseline in _stage8a_required_baselines(default_latent_config()):
        if baseline.config_id in completed:
            continue
        result = evaluate_stage8_architecture(
            baseline,
            n_values=n_values,
            seeds=budget.stage8a_seeds,
            eval_config=eval_config,
            eval_split="dev",
            run_controls=budget.run_controls,
        )
        rows.append(_database_row("stage8a", baseline, result, baseline_capacity=0))
        _append_jsonl(SEARCH_DATABASE_PATH, rows[-1])
        _append_jsonl(STAGE8A_RESULTS_PATH, rows[-1])
        _append_controls(result)
        _append_stage8a_controls(result)
    best_monolithic = _best_capacity(rows, model_kinds={"monolithic_transformer"})
    best_non_oracle = _best_capacity(rows, exclude_model_kinds={"oracle_evidence_location"})
    decision = _stage8a_decision(rows, best_monolithic, best_non_oracle)
    payload = {
        "stage": "8A",
        "status": "completed",
        "decision": decision,
        "n_values": n_values,
        "seeds": list(budget.stage8a_seeds),
        "baselines": rows,
        "best_monolithic_capacity": best_monolithic,
        "best_non_oracle_capacity": best_non_oracle,
        "capacity_by_task_family": _capacity_by_task_family(rows, eval_config),
        "note": "Stage 8A validates benchmarks and baselines only; it does not make a latent-attention success claim.",
        "explicit_no_10x_claim": "NO_10X_CLAIM",
    }
    dump_json(BASELINE_CAPACITY_PATH, payload)
    _write_compute_audit(rows)
    write_capacity_curves_csv(CAPACITY_CURVES_PATH, rows)
    _write_stage8a_report(payload, eval_config)
    return payload


def run_stage8b_architecture_search(
    budget: Stage8SearchBudget,
    eval_config: Stage8EvaluationConfig | None = None,
    baseline_payload: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    base_eval_config = eval_config or _eval_config(budget)
    small_eval_config = _stage8b_small_eval_config(base_eval_config)
    promoted_eval_config = _stage8b_promoted_eval_config(base_eval_config)
    baseline_payload = baseline_payload or _load_json(BASELINE_CAPACITY_PATH, {})
    baseline_capacity = int(baseline_payload.get("best_monolithic_capacity", {}).get("capacity", 0)) if isinstance(baseline_payload.get("best_monolithic_capacity"), Mapping) else 0
    target_capacity, target_n = _stage8b_target_from_baseline(baseline_capacity, budget.max_n)
    budget_signature = _stage8b_budget_signature(budget, baseline_capacity, target_n)
    STAGE8B_CONTROLS_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    STAGE8B_CONTROLS_AUDIT_PATH.touch(exist_ok=True)
    started = time.perf_counter()
    small_n = [value for value in (8, 16, 32, 64) if value <= budget.max_n] or [min(8, budget.max_n)]
    promoted_n = [value for value in (64, 128) if value <= budget.max_n] or [max(small_n)]
    completed = _completed_config_phases(SEARCH_DATABASE_PATH, budget_signature=budget_signature)
    configs = generate_stage8_search_space(max_configs=budget.max_architecture_configs, min_configs=min(50, budget.max_architecture_configs))
    small_results: List[Dict[str, object]] = []

    for config in configs:
        if _time_exhausted(started, budget):
            break
        if ("stage8b_small", config.config_id) in completed:
            continue
        result = evaluate_stage8_architecture(
            config,
            n_values=small_n,
            seeds=budget.stage8b_small_seeds,
            eval_config=small_eval_config,
            eval_split="dev",
            run_controls=budget.run_controls,
        )
        row = _database_row("stage8b_small", config, result, baseline_capacity=baseline_capacity)
        row["budget_signature"] = budget_signature
        row["target_capacity"] = target_capacity
        row["target_n"] = target_n
        row["promotion_checks"] = _promotion_checks(config, result, baseline_payload, small_eval_config, budget, promoted_n[:1])
        small_results.append(row)
        _append_jsonl(SEARCH_DATABASE_PATH, row)
        _append_controls(result)
        _append_stage8b_controls(result, budget_signature)

    all_small = [
        row for row in _load_database(SEARCH_DATABASE_PATH)
        if row.get("phase") == "stage8b_small" and row.get("budget_signature") == budget_signature
    ]
    promoted = _select_promoted(all_small, budget.max_promoted_configs, require_promotion_pass=True)
    dump_json(
        PROMOTED_CONFIGS_PATH,
        {
            "stage": "8B",
            "promoted": promoted,
            "num_promoted": len(promoted),
            "minimum_promoted_requested": min(10, budget.max_promoted_configs),
            "budget": asdict(budget),
            "budget_signature": budget_signature,
            "target_capacity": target_capacity,
            "target_n": target_n,
        },
    )

    promoted_results: List[Dict[str, object]] = []
    config_by_id = {config.config_id: config for config in configs}
    for row in promoted:
        config = config_by_id.get(str(row["config_id"]))
        if config is None or _time_exhausted(started, budget):
            continue
        if ("stage8b_promoted", config.config_id) in completed:
            continue
        result = evaluate_stage8_architecture(
            config,
            n_values=promoted_n,
            seeds=budget.stage8b_promoted_seeds,
            eval_config=promoted_eval_config,
            eval_split="dev",
            run_controls=budget.run_controls,
        )
        promoted_row = _database_row("stage8b_promoted", config, result, baseline_capacity=baseline_capacity)
        promoted_row["budget_signature"] = budget_signature
        promoted_row["target_capacity"] = target_capacity
        promoted_row["target_n"] = target_n
        promoted_results.append(promoted_row)
        _append_jsonl(SEARCH_DATABASE_PATH, promoted_row)
        _append_controls(result)
        _append_stage8b_controls(result, budget_signature)

    all_promoted = [
        row for row in _load_database(SEARCH_DATABASE_PATH)
        if row.get("phase") == "stage8b_promoted" and row.get("budget_signature") == budget_signature
    ]
    finalists = _select_promoted(all_promoted or promoted, budget.max_finalists, require_promotion_pass=False)
    finalist_rows: List[Dict[str, object]] = []
    finalist_n = _n_values(budget.max_n)
    for row in finalists:
        config = config_by_id.get(str(row["config_id"]))
        if config is None or _time_exhausted(started, budget):
            continue
        if ("stage8b_finalist", config.config_id) in completed:
            continue
        result = evaluate_stage8_architecture(
            config,
            n_values=finalist_n,
            seeds=budget.stage8b_finalist_seeds,
            eval_config=promoted_eval_config,
            eval_split="dev",
            run_controls=budget.run_controls,
        )
        finalist_row = _database_row("stage8b_finalist", config, result, baseline_capacity=baseline_capacity)
        finalist_row["budget_signature"] = budget_signature
        finalist_row["target_capacity"] = target_capacity
        finalist_row["target_n"] = target_n
        finalist_rows.append(finalist_row)
        _append_jsonl(SEARCH_DATABASE_PATH, finalist_row)
        _append_controls(result)
        _append_stage8b_controls(result, budget_signature)

    all_finalists = [
        row for row in _load_database(SEARCH_DATABASE_PATH)
        if row.get("phase") == "stage8b_finalist" and row.get("budget_signature") == budget_signature
    ]
    selected = _select_promoted(all_finalists or finalists, 1, require_promotion_pass=False)
    if selected:
        _write_best_config(config_by_id.get(str(selected[0]["config_id"])), selected[0])
        _write_freeze_manifest(
            config_by_id.get(str(selected[0]["config_id"])),
            selected[0],
            baseline_capacity=baseline_capacity,
            target_capacity=target_capacity,
            target_n=target_n,
            budget=budget,
            budget_signature=budget_signature,
        )
    decision = _stage8b_decision(selected[0] if selected else None, promoted, all_small, baseline_capacity, budget)
    payload = {
        "stage": "8B",
        "status": "completed" if not _time_exhausted(started, budget) else "budget_exhausted",
        "decision": decision,
        "small_results_count": len(all_small),
        "evaluated_config_count": len(all_small),
        "minimum_config_count_requested": 50,
        "promoted": promoted,
        "promoted_count": len(promoted),
        "minimum_promoted_requested": min(10, budget.max_promoted_configs),
        "finalists": all_finalists,
        "selected_final_architecture": selected[0] if selected else None,
        "baseline_capacity": baseline_capacity,
        "target_capacity": target_capacity,
        "target_n": target_n,
        "budget_signature": budget_signature,
        "budget": asdict(budget),
        "note": "Stage 8B ranks dev architectures only; no final held-out claim is made here.",
        "explicit_no_stage8c_claim": "NO_STAGE8C_CLAIM",
    }
    dump_json(FINALIST_RESULTS_PATH, payload)
    database = [
        row for row in _load_database(SEARCH_DATABASE_PATH)
        if row.get("phase", "").startswith("stage8b") and row.get("budget_signature") == budget_signature
    ]
    _write_compute_audit(database)
    _write_stage8b_compute_audit(database, budget_signature)
    write_capacity_curves_csv(CAPACITY_CURVES_PATH, database)
    write_capacity_curves_csv(STAGE8B_CAPACITY_CURVES_PATH, database)
    _write_stage8b_report(payload, database)
    return payload


def run_stage8c_final_validation(
    best_config: Stage8ArchitectureConfig,
    budget: Stage8SearchBudget,
    eval_config: Stage8EvaluationConfig | None = None,
    baseline_payload: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    base_eval_config = eval_config or _eval_config(budget)
    eval_config = Stage8EvaluationConfig(
        train_examples=base_eval_config.train_examples,
        eval_examples=base_eval_config.eval_examples,
        final_examples=base_eval_config.final_examples,
        k_candidates=base_eval_config.k_candidates,
        accuracy_threshold=base_eval_config.accuracy_threshold,
        chance_margin=base_eval_config.chance_margin,
        min_stable_seeds=base_eval_config.min_stable_seeds,
        max_seed_std=base_eval_config.max_seed_std,
        min_seed_accuracy=base_eval_config.min_seed_accuracy,
        degradation_threshold=base_eval_config.degradation_threshold,
        invariance_tolerance=base_eval_config.invariance_tolerance,
        controls=REQUIRED_STAGE8_CONTROLS,
    )
    baseline_payload = baseline_payload or _load_json(BASELINE_CAPACITY_PATH, {})
    baseline_capacity = int(baseline_payload.get("best_monolithic_capacity", {}).get("capacity", 0)) if isinstance(baseline_payload.get("best_monolithic_capacity"), Mapping) else 0
    n_values = _n_values(budget.max_n)
    result = evaluate_stage8_architecture(
        best_config,
        n_values=n_values,
        seeds=budget.stage8c_seeds,
        eval_config=eval_config,
        eval_split="final",
        run_controls=True,
    )
    row = _database_row("stage8c_final", best_config, result, baseline_capacity=baseline_capacity)
    _append_jsonl(SEARCH_DATABASE_PATH, row)
    _append_controls(result)
    target_n = max(n_values)
    frozen_config = replace(best_config, name=f"stage8c_frozen_{best_config.name}", model_kind="frozen_latent", frozen=True, epochs=0)
    frozen_result = evaluate_stage8_architecture(
        frozen_config,
        n_values=(target_n,),
        seeds=budget.stage8c_seeds,
        eval_config=eval_config,
        eval_split="final",
        run_controls=False,
    )
    non_oracle_config = replace(default_latent_config(), name="stage8c_retrieval_topk_non_oracle", model_kind="retrieval_topk", top_k_views=4)
    non_oracle_result = evaluate_stage8_architecture(
        non_oracle_config,
        n_values=(target_n,),
        seeds=budget.stage8c_seeds,
        eval_config=eval_config,
        eval_split="final",
        run_controls=False,
    )
    ratio = capacity_ratio(int(result["capacity"]["capacity"]), baseline_capacity) if isinstance(result.get("capacity"), Mapping) else 0.0
    gate_details = _stage8c_gate_details(result, frozen_result, non_oracle_result, target_n, ratio)
    critical_controls_pass = bool(gate_details["all_critical_gates_pass"])
    decision = _decision_output(ratio, critical_controls_pass, completed_seeds=len(budget.stage8c_seeds), budget=budget, result=result)
    payload = {
        "stage": "8C",
        "status": "completed",
        "decision": decision,
        "final_architecture": architecture_config_to_dict(best_config),
        "held_out_capacity": result["capacity"],
        "baseline_capacity": baseline_capacity,
        "capacity_ratio": ratio,
        "critical_controls_pass": critical_controls_pass,
        "stage8c_gate_details": gate_details,
        "frozen_same_architecture_result": _database_row("stage8c_frozen_comparator", frozen_config, frozen_result, baseline_capacity=baseline_capacity),
        "best_non_oracle_high_n_result": _database_row("stage8c_non_oracle_comparator", non_oracle_config, non_oracle_result, baseline_capacity=baseline_capacity),
        "result": row,
        "note": "Architecture and hyperparameters are frozen for this validation. Do not tune on these results.",
    }
    dump_json(FINAL_VALIDATION_RESULTS_PATH, payload)
    database = _load_database(SEARCH_DATABASE_PATH)
    _write_compute_audit(database)
    write_capacity_curves_csv(CAPACITY_CURVES_PATH, database)
    return payload


def _eval_config(budget: Stage8SearchBudget) -> Stage8EvaluationConfig:
    return Stage8EvaluationConfig(
        train_examples=budget.train_examples,
        eval_examples=budget.eval_examples,
        final_examples=budget.final_examples,
        min_stable_seeds=5,
    )


def _stage8b_target_from_baseline(baseline_capacity: int, max_n: int) -> tuple[int, int]:
    target_capacity = int(10 * max(0, baseline_capacity))
    schedule = _n_values(max_n)
    if target_capacity <= 0:
        return target_capacity, schedule[0]
    for value in schedule:
        if value >= target_capacity:
            return target_capacity, value
    for value in (1024, 2048, 4096):
        if value >= target_capacity:
            return target_capacity, min(value, max_n if max_n >= value else value)
    return target_capacity, max(schedule)


def _stage8b_budget_signature(budget: Stage8SearchBudget, baseline_capacity: int, target_n: int) -> str:
    payload = {
        "stage": "8B",
        "max_architecture_configs": budget.max_architecture_configs,
        "max_promoted_configs": budget.max_promoted_configs,
        "max_finalists": budget.max_finalists,
        "max_n": budget.max_n,
        "stage8b_small_seeds": list(budget.stage8b_small_seeds),
        "stage8b_promoted_seeds": list(budget.stage8b_promoted_seeds),
        "stage8b_finalist_seeds": list(budget.stage8b_finalist_seeds),
        "train_examples": budget.train_examples,
        "eval_examples": budget.eval_examples,
        "baseline_capacity": baseline_capacity,
        "target_n": target_n,
    }
    return hashlib.blake2b(json.dumps(payload, sort_keys=True).encode("utf-8"), digest_size=8).hexdigest()


def _stage8a_eval_config(eval_config: Stage8EvaluationConfig, budget: Stage8SearchBudget) -> Stage8EvaluationConfig:
    return Stage8EvaluationConfig(
        train_examples=eval_config.train_examples,
        eval_examples=eval_config.eval_examples,
        final_examples=eval_config.final_examples,
        k_candidates=eval_config.k_candidates,
        accuracy_threshold=eval_config.accuracy_threshold,
        chance_margin=eval_config.chance_margin,
        min_stable_seeds=max(1, len(tuple(budget.stage8a_seeds))),
        max_seed_std=eval_config.max_seed_std,
        min_seed_accuracy=eval_config.min_seed_accuracy,
        degradation_threshold=eval_config.degradation_threshold,
        invariance_tolerance=eval_config.invariance_tolerance,
        controls=(
            "randomized_labels",
            "randomized_candidate_order_with_label_remap",
            "randomized_evidence_block_order",
            "evidence_candidate_mismatch",
            "cross_task_evidence_shuffle",
            "cross_task_query_shuffle",
            "distractor_only",
            "schema_template_only",
        ),
    )


def _stage8b_small_eval_config(eval_config: Stage8EvaluationConfig) -> Stage8EvaluationConfig:
    return Stage8EvaluationConfig(
        train_examples=eval_config.train_examples,
        eval_examples=eval_config.eval_examples,
        final_examples=eval_config.final_examples,
        k_candidates=eval_config.k_candidates,
        accuracy_threshold=eval_config.accuracy_threshold,
        chance_margin=eval_config.chance_margin,
        min_stable_seeds=1,
        max_seed_std=eval_config.max_seed_std,
        min_seed_accuracy=eval_config.min_seed_accuracy,
        degradation_threshold=eval_config.degradation_threshold,
        invariance_tolerance=eval_config.invariance_tolerance,
        controls=(
            "randomized_labels",
            "randomized_candidate_order_with_label_remap",
            "randomized_evidence_block_order",
            "evidence_candidate_mismatch",
            "cross_task_evidence_shuffle",
        ),
    )


def _stage8b_promoted_eval_config(eval_config: Stage8EvaluationConfig) -> Stage8EvaluationConfig:
    return Stage8EvaluationConfig(
        train_examples=eval_config.train_examples,
        eval_examples=eval_config.eval_examples,
        final_examples=eval_config.final_examples,
        k_candidates=eval_config.k_candidates,
        accuracy_threshold=eval_config.accuracy_threshold,
        chance_margin=eval_config.chance_margin,
        min_stable_seeds=3,
        max_seed_std=eval_config.max_seed_std,
        min_seed_accuracy=eval_config.min_seed_accuracy,
        degradation_threshold=eval_config.degradation_threshold,
        invariance_tolerance=eval_config.invariance_tolerance,
        controls=(
            "randomized_labels",
            "randomized_candidate_order_with_label_remap",
            "randomized_evidence_block_order",
            "evidence_candidate_mismatch",
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
        ),
    )


def _stage8a_required_baselines(base: Stage8ArchitectureConfig) -> List[Stage8ArchitectureConfig]:
    return [
        replace(base, name="random_candidate_baseline", model_kind="random_candidate"),
        replace(base, name="candidate_only_baseline", model_kind="candidate_only"),
        replace(base, name="evidence_only_baseline", model_kind="evidence_only"),
        replace(base, name="query_only_baseline", model_kind="query_only"),
        replace(base, name="retrieval_topk_baseline", model_kind="retrieval_topk", top_k_views=4),
        replace(base, name="standard_monolithic_transformer_full_context", model_kind="monolithic_transformer", roles=1, avenues=1),
        replace(base, name="same_parameter_count_monolithic_transformer", model_kind="monolithic_transformer", roles=1, avenues=1, hidden_dim=base.hidden_dim),
        replace(base, name="same_compute_budget_monolithic_transformer", model_kind="monolithic_transformer", roles=1, avenues=1, hidden_dim=max(16, base.hidden_dim // 2)),
        replace(base, name="initial_latent_baseline_diagnostic", model_kind="trainable_latent"),
    ]


def _database_row(
    phase: str,
    config: Stage8ArchitectureConfig,
    result: Mapping[str, object],
    baseline_capacity: int,
) -> Dict[str, object]:
    rows = list(result.get("rows", [])) if isinstance(result.get("rows"), list) else []
    accuracy_by_n: Dict[str, List[float]] = {}
    control_results: Dict[str, object] = {}
    shortcut_audits: List[object] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        n = str(row.get("n_blocks", ""))
        accuracy_by_n.setdefault(n, []).append(float(row.get("accuracy", 0.0)))
        if row.get("controls"):
            control_results.setdefault(n, []).append(row.get("controls"))
        if row.get("shortcut_audit"):
            shortcut_audits.append(row.get("shortcut_audit"))
    capacity = int(result.get("capacity", {}).get("capacity", 0)) if isinstance(result.get("capacity"), Mapping) else 0
    compute = result.get("compute", {})
    seeds = sorted({int(row.get("seed", 0)) for row in rows if isinstance(row, Mapping)})
    checkpoint_path = _write_run_checkpoint(phase, config.config_id, result)
    return {
        "phase": phase,
        "config_id": config.config_id,
        "architecture_name": config.name,
        "model_kind": config.model_kind,
        "architecture_parameters": architecture_config_to_dict(config),
        "task_families": list(TASK_FAMILIES),
        "train_n_schedule": _schedule_from_rows(rows, "num_train_examples"),
        "eval_n_schedule": sorted(int(key) for key in accuracy_by_n if str(key).isdigit()),
        "seed": seeds[0] if len(seeds) == 1 else None,
        "seeds": seeds,
        "accuracy_by_n": {key: mean(value) for key, value in accuracy_by_n.items()},
        "seed_accuracy_by_n": accuracy_by_n,
        "capacity_C": capacity,
        "capacity_ratio_vs_baseline": capacity_ratio(capacity, baseline_capacity),
        "compute_estimate": compute,
        "parameter_count": int(compute.get("parameter_count_estimate", 0)) if isinstance(compute, Mapping) else 0,
        "wall_clock_time": float(result.get("wall_clock_seconds", 0.0)),
        "memory_usage": {"estimated_peak_bytes": 0, "source": "not_measured_cpu_harness"},
        "control_results": control_results,
        "shortcut_audits": shortcut_audits[:5],
        "failure_reason": None,
        "checkpoint_path": str(checkpoint_path),
        "status": result.get("status", "completed"),
        "capacity": result.get("capacity", {}),
        "rows": rows,
    }


def _promotion_checks(
    config: Stage8ArchitectureConfig,
    result: Mapping[str, object],
    baseline_payload: Mapping[str, object],
    eval_config: Stage8EvaluationConfig,
    budget: Stage8SearchBudget,
    gate_n: Sequence[int],
) -> Dict[str, object]:
    if not gate_n:
        return {"passes": False, "reason": "no_gate_n"}
    gate_accuracy = _mean_accuracy_at(result, gate_n[-1])
    baseline_accuracy = _baseline_accuracy_at(baseline_payload, gate_n[-1])
    frozen_config = replace(config, name=f"frozen_{config.name}", model_kind="frozen_latent", frozen=True, epochs=0)
    frozen = evaluate_stage8_architecture(
        frozen_config,
        n_values=gate_n,
        seeds=budget.stage8b_small_seeds,
        eval_config=eval_config,
        eval_split="dev",
        run_controls=False,
    )
    candidate_only = evaluate_stage8_architecture(
        replace(config, name=f"candidate_only_for_{config.name}", model_kind="candidate_only"),
        n_values=gate_n,
        seeds=budget.stage8b_small_seeds,
        eval_config=eval_config,
        eval_split="dev",
        run_controls=False,
    )
    frozen_accuracy = _mean_accuracy_at(frozen, gate_n[-1])
    candidate_accuracy = _mean_accuracy_at(candidate_only, gate_n[-1])
    controls_pass = _soft_controls_pass(result)
    shortcut_pass = _shortcut_rows_pass(result)
    baseline_gate = gate_accuracy > baseline_accuracy
    passes = baseline_gate and gate_accuracy > frozen_accuracy and gate_accuracy > candidate_accuracy and controls_pass and shortcut_pass
    return {
        "passes": passes,
        "gate_n": gate_n[-1],
        "gate_accuracy": gate_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "beats_best_matched_monolithic": baseline_gate,
        "frozen_accuracy": frozen_accuracy,
        "candidate_only_accuracy": candidate_accuracy,
        "controls_pass": controls_pass,
        "shortcut_pass": shortcut_pass,
    }


def _select_promoted(rows: Sequence[Mapping[str, object]], top_k: int, require_promotion_pass: bool = False) -> List[Dict[str, object]]:
    ranked = []
    for row in rows:
        checks = row.get("promotion_checks", {})
        check_bonus = 1.0 if isinstance(checks, Mapping) and checks.get("passes") else 0.0
        if require_promotion_pass and check_bonus <= 0.0:
            continue
        capacity = float(row.get("capacity_C", 0))
        max_accuracy = max((float(value) for value in row.get("accuracy_by_n", {}).values()), default=0.0) if isinstance(row.get("accuracy_by_n"), Mapping) else 0.0
        compute = row.get("compute_estimate", {})
        estimated_compute = float(compute.get("estimated_forward_compute", 1.0)) if isinstance(compute, Mapping) else 1.0
        stability = _row_stability_score(row)
        score = check_bonus * 10_000.0 + capacity * 100.0 + max_accuracy + stability - 1e-12 * estimated_compute
        ranked.append((score, row))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [dict(row) for _, row in ranked[:top_k]]


def _baseline_accuracy_at(baseline_payload: Mapping[str, object], n_blocks: int) -> float:
    best = baseline_payload.get("best_monolithic_capacity", {})
    if isinstance(best, Mapping):
        accuracy_by_n = best.get("accuracy_by_n", {})
        if isinstance(accuracy_by_n, Mapping):
            return float(accuracy_by_n.get(str(n_blocks), 0.0))
    return 0.0


def _shortcut_rows_pass(result: Mapping[str, object]) -> bool:
    rows = result.get("rows", [])
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        shortcut = row.get("shortcut_audit", {})
        if isinstance(shortcut, Mapping) and not bool(shortcut.get("passes", False)):
            return False
    return True


def _row_stability_score(row: Mapping[str, object]) -> float:
    capacity = row.get("capacity", {})
    if not isinstance(capacity, Mapping):
        return 0.0
    summaries = capacity.get("n_summaries", {})
    if not isinstance(summaries, Mapping):
        return 0.0
    score = 0.0
    for summary in summaries.values():
        if isinstance(summary, Mapping):
            score -= float(summary.get("std_accuracy", 0.0))
            score += 0.01 * float(summary.get("seed_count", 0))
    return score


def _best_capacity(rows: Sequence[Mapping[str, object]], model_kinds: set[str] | None = None, exclude_model_kinds: set[str] | None = None) -> Dict[str, object]:
    best: Mapping[str, object] | None = None
    for row in rows:
        kind = str(row.get("architecture_parameters", {}).get("model_kind", "")) if isinstance(row.get("architecture_parameters"), Mapping) else ""
        if model_kinds and kind not in model_kinds:
            continue
        if exclude_model_kinds and kind in exclude_model_kinds:
            continue
        if best is None or int(row.get("capacity_C", 0)) > int(best.get("capacity_C", 0)):
            best = row
    if best is None:
        return {"capacity": 0}
    return {
        "config_id": best.get("config_id"),
        "architecture_name": best.get("architecture_name"),
        "model_kind": best.get("architecture_parameters", {}).get("model_kind") if isinstance(best.get("architecture_parameters"), Mapping) else None,
        "capacity": int(best.get("capacity_C", 0)),
        "accuracy_by_n": best.get("accuracy_by_n", {}),
    }


def _capacity_by_task_family(rows: Sequence[Mapping[str, object]], eval_config: Stage8EvaluationConfig) -> Dict[str, object]:
    output: Dict[str, object] = {}
    for row in rows:
        by_family_n: Dict[str, Dict[int, List[float]]] = {}
        for seed_row in row.get("rows", []):  # type: ignore[union-attr]
            if not isinstance(seed_row, Mapping):
                continue
            n_blocks = int(seed_row.get("n_blocks", 0))
            family_acc = seed_row.get("accuracy_by_task_family", {})
            if not isinstance(family_acc, Mapping):
                continue
            for family, accuracy in family_acc.items():
                by_family_n.setdefault(str(family), {}).setdefault(n_blocks, []).append(float(accuracy))
        family_capacity: Dict[str, object] = {}
        for family, n_rows in by_family_n.items():
            capacity = 0
            summaries: Dict[str, object] = {}
            for n_blocks, values in sorted(n_rows.items()):
                seed_count = len(values)
                acc_mean = mean(values) if values else 0.0
                eligible = seed_count >= eval_config.min_stable_seeds and acc_mean >= eval_config.accuracy_threshold
                if eligible:
                    capacity = max(capacity, n_blocks)
                summaries[str(n_blocks)] = {
                    "mean_accuracy": acc_mean,
                    "seed_count": seed_count,
                    "eligible": eligible,
                }
            family_capacity[family] = {"capacity": capacity, "n_summaries": summaries}
        output[str(row.get("architecture_name"))] = family_capacity
    return output


def _stage8a_decision(
    rows: Sequence[Mapping[str, object]],
    best_monolithic: Mapping[str, object],
    best_non_oracle: Mapping[str, object],
) -> str:
    if not rows or not best_monolithic or not best_non_oracle:
        return "STAGE8A_BASELINE_INCOMPLETE"
    required_kinds = {
        "random_candidate",
        "candidate_only",
        "evidence_only",
        "query_only",
        "retrieval_topk",
        "monolithic_transformer",
    }
    present = {
        str(row.get("architecture_parameters", {}).get("model_kind"))
        for row in rows
        if isinstance(row.get("architecture_parameters"), Mapping)
    }
    if not required_kinds.issubset(present):
        return "STAGE8A_BASELINE_INCOMPLETE"
    shortcut_failures = []
    randomized_label_failures = []
    for row in rows:
        for seed_row in row.get("rows", []):  # type: ignore[union-attr]
            if not isinstance(seed_row, Mapping):
                continue
            shortcut = seed_row.get("shortcut_audit", {})
            if isinstance(shortcut, Mapping) and not bool(shortcut.get("passes", False)):
                shortcut_failures.append((row.get("architecture_name"), seed_row.get("seed"), seed_row.get("n_blocks"), shortcut.get("failures")))
            controls = seed_row.get("controls", {})
            if isinstance(controls, Mapping):
                randomized = controls.get("randomized_labels")
                if isinstance(randomized, Mapping) and float(randomized.get("accuracy", 1.0)) > 0.35:
                    randomized_label_failures.append((row.get("architecture_name"), seed_row.get("seed"), seed_row.get("n_blocks")))
    if shortcut_failures or randomized_label_failures:
        return "STAGE8A_BENCHMARK_BROKEN"
    return "STAGE8A_BASELINE_ESTABLISHED"


def _write_stage8a_report(payload: Mapping[str, object], eval_config: Stage8EvaluationConfig) -> None:
    STAGE8A_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    baselines = payload.get("baselines", [])
    lines = [
        "# Stage 8A Baseline Capacity",
        "",
        f"Decision: {payload.get('decision', 'UNKNOWN')}",
        "",
        "Explicit statement: NO_10X_CLAIM. Stage 8A establishes baseline capacity only and does not run architecture search.",
        "",
        "## Capacity By Baseline",
        "",
        "| Baseline | Kind | Capacity C | Capacity/Compute |",
        "|---|---:|---:|---:|",
    ]
    for row in baselines if isinstance(baselines, list) else []:
        if not isinstance(row, Mapping):
            continue
        compute = row.get("compute_estimate", {})
        estimated = float(compute.get("estimated_forward_compute", 0.0)) if isinstance(compute, Mapping) else 0.0
        capacity = int(row.get("capacity_C", 0))
        per_compute = capacity / estimated if estimated > 0 else 0.0
        kind = row.get("architecture_parameters", {}).get("model_kind", "") if isinstance(row.get("architecture_parameters"), Mapping) else ""
        lines.append(f"| {row.get('architecture_name')} | {kind} | {capacity} | {per_compute:.8f} |")
    lines.extend(
        [
            "",
            "## Matched Baseline Summary",
            "",
            f"- Best matched monolithic baseline capacity: {_capacity_report_text(payload.get('best_monolithic_capacity'))}",
            f"- Best non-oracle baseline capacity: {_capacity_report_text(payload.get('best_non_oracle_capacity'))}",
            f"- Chance level: {1.0 / max(1, eval_config.k_candidates):.3f}",
            "",
            "## Accuracy Vs N",
            "",
        ]
    )
    for row in baselines if isinstance(baselines, list) else []:
        if isinstance(row, Mapping):
            lines.append(f"- {row.get('architecture_name')}: {row.get('accuracy_by_n', {})}")
    lines.extend(["", "## Compute Vs N", ""])
    for row in baselines if isinstance(baselines, list) else []:
        if isinstance(row, Mapping):
            lines.append(f"- {row.get('architecture_name')}: {_compute_by_n_for_report(row)}")
    lines.extend(["", "## Capacity By Task Family", ""])
    capacity_by_family = payload.get("capacity_by_task_family", {})
    if isinstance(capacity_by_family, Mapping):
        for baseline, family_rows in capacity_by_family.items():
            lines.append(f"### {baseline}")
            if isinstance(family_rows, Mapping):
                for family, family_payload in family_rows.items():
                    capacity = family_payload.get("capacity", 0) if isinstance(family_payload, Mapping) else 0
                    lines.append(f"- {family}: C={capacity}")
    lines.extend(["", "## Controls And Shortcut Audits", ""])
    for row in baselines if isinstance(baselines, list) else []:
        if isinstance(row, Mapping):
            lines.append(f"- {row.get('architecture_name')}: {_control_summary_for_report(row)}")
    lines.append("")
    lines.append(f"- Controls audit JSONL: `{STAGE8A_CONTROLS_AUDIT_PATH}`")
    lines.append(f"- Shortcut audit JSON: `{SHORTCUT_AUDIT_PATH}`")
    lines.append(f"- Capacity curves CSV: `{CAPACITY_CURVES_PATH}`")
    lines.append(f"- Compute audit JSON: `{COMPUTE_AUDIT_PATH}`")
    lines.append("")
    lines.append("Failed baselines, zero-capacity baselines, and failed controls are retained in the JSON artifacts.")
    STAGE8A_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_stage8b_report(payload: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> None:
    STAGE8B_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    selected = payload.get("selected_final_architecture")
    selected_name = selected.get("architecture_name") if isinstance(selected, Mapping) else "none"
    selected_capacity = selected.get("capacity_C") if isinstance(selected, Mapping) else "n/a"
    selected_ratio = selected.get("capacity_ratio_vs_baseline") if isinstance(selected, Mapping) else "n/a"
    lines = [
        "# Stage 8B Architecture Search",
        "",
        f"Decision: {payload.get('decision', 'UNKNOWN')}",
        "",
        "Explicit statement: NO_STAGE8C_CLAIM. Stage 8B is dev-only architecture search and does not validate a final 10x claim.",
        "",
        "## Baseline And Target",
        "",
        f"- Stage 8A matched monolithic baseline capacity: {payload.get('baseline_capacity')}",
        f"- Target capacity: {payload.get('target_capacity')}",
        f"- Stage 8B target N: {payload.get('target_n')}",
        f"- Budget signature: {payload.get('budget_signature')}",
        "",
        "## Search Summary",
        "",
        f"- Evaluated configs: {payload.get('evaluated_config_count')}",
        f"- Promoted configs: {payload.get('promoted_count')}",
        f"- Finalist rows: {len(payload.get('finalists', [])) if isinstance(payload.get('finalists'), list) else 0}",
        f"- Selected config: {selected_name}",
        f"- Selected dev capacity: {selected_capacity}",
        f"- Selected dev capacity ratio: {selected_ratio}",
        "",
        "## Ranked Rows",
        "",
        "| Phase | Config | Capacity | Ratio | Max Accuracy | Compute | Controls |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in sorted(rows, key=lambda item: (str(item.get("phase")), -float(max(item.get("accuracy_by_n", {}).values(), default=0.0) if isinstance(item.get("accuracy_by_n"), Mapping) else 0.0)))[:80]:
        accuracy_by_n = row.get("accuracy_by_n", {})
        max_accuracy = max((float(value) for value in accuracy_by_n.values()), default=0.0) if isinstance(accuracy_by_n, Mapping) else 0.0
        compute = row.get("compute_estimate", {})
        estimated = float(compute.get("estimated_forward_compute", 0.0)) if isinstance(compute, Mapping) else 0.0
        controls = _control_summary_for_report(row)
        control_text = "tracked" if controls else "not_run"
        lines.append(
            f"| {row.get('phase')} | {row.get('architecture_name')} | {row.get('capacity_C')} | "
            f"{row.get('capacity_ratio_vs_baseline')} | {max_accuracy:.4f} | {estimated:.1f} | {control_text} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Search database: `{SEARCH_DATABASE_PATH}`",
            f"- Promoted configs: `{PROMOTED_CONFIGS_PATH}`",
            f"- Finalists: `{FINALIST_RESULTS_PATH}`",
            f"- Frozen best config: `{BEST_CONFIG_PATH}`",
            f"- Freeze manifest: `{CONFIG_FREEZE_MANIFEST_PATH}`",
            f"- Controls audit: `{STAGE8B_CONTROLS_AUDIT_PATH}`",
            f"- Compute audit: `{STAGE8B_COMPUTE_AUDIT_PATH}`",
            f"- Capacity curves: `{STAGE8B_CAPACITY_CURVES_PATH}`",
            "",
            "Failed configs and zero-capacity configs are retained in the JSONL database.",
        ]
    )
    STAGE8B_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stage8b_decision(
    selected: Mapping[str, object] | None,
    promoted: Sequence[Mapping[str, object]],
    small_rows: Sequence[Mapping[str, object]],
    baseline_capacity: int,
    budget: Stage8SearchBudget,
) -> str:
    if len(small_rows) < min(50, budget.max_architecture_configs):
        return "STAGE8B_SEARCH_NEEDS_MORE_COMPUTE"
    if not promoted:
        return "STAGE8B_HYPOTHESIS_NOT_SUPPORTED_ON_DEV"
    if selected is None or not CONFIG_FREEZE_MANIFEST_PATH.exists():
        return "STAGE8B_SEARCH_NEEDS_MORE_COMPUTE"
    ratio = float(selected.get("capacity_ratio_vs_baseline", 0.0))
    capacity = int(selected.get("capacity_C", 0))
    controls = _control_summary_for_report(selected)
    controls_pass = bool(controls) and all(str(value).startswith(str(value).split("/")[1].split()[0] + "/") for value in controls.values())
    if baseline_capacity <= 0 and capacity > 0 and controls:
        return "STAGE8B_TARGET_REACHED_ON_DEV"
    if ratio >= 10.0 and controls:
        return "STAGE8B_TARGET_REACHED_ON_DEV"
    if capacity > baseline_capacity and controls:
        return "STAGE8B_PROMISING_BUT_NOT_10X"
    return "STAGE8B_HYPOTHESIS_NOT_SUPPORTED_ON_DEV"


def _capacity_report_text(row) -> str:
    if not isinstance(row, Mapping):
        return "not measured"
    return f"C={row.get('capacity', 0)} ({row.get('architecture_name', 'unknown')})"


def _compute_by_n_for_report(row: Mapping[str, object]) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for seed_row in row.get("rows", []):  # type: ignore[union-attr]
        if not isinstance(seed_row, Mapping):
            continue
        n_blocks = str(seed_row.get("n_blocks"))
        if n_blocks in output:
            continue
        compute = seed_row.get("compute", {})
        if isinstance(compute, Mapping):
            output[n_blocks] = float(compute.get("estimated_forward_compute", 0.0))
    return dict(sorted(output.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else 0))


def _control_summary_for_report(row: Mapping[str, object]) -> Dict[str, object]:
    counts: Dict[str, List[bool]] = {}
    for seed_row in row.get("rows", []):  # type: ignore[union-attr]
        if not isinstance(seed_row, Mapping):
            continue
        controls = seed_row.get("controls", {})
        if not isinstance(controls, Mapping):
            continue
        for name, control_row in controls.items():
            if isinstance(control_row, Mapping):
                counts.setdefault(str(name), []).append(bool(control_row.get("passes", False)))
    return {
        name: f"{sum(values)}/{len(values)} pass"
        for name, values in sorted(counts.items())
        if values
    }


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            timeout=5,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _critical_controls_pass(result: Mapping[str, object]) -> bool:
    capacity = result.get("capacity", {})
    if not isinstance(capacity, Mapping):
        return False
    summaries = capacity.get("n_summaries", {})
    if not isinstance(summaries, Mapping):
        return False
    for summary in summaries.values():
        if isinstance(summary, Mapping) and summary.get("eligible"):
            controls = summary.get("controls", {})
            return bool(isinstance(controls, Mapping) and controls.get("passes"))
    return False


def _stage8c_gate_details(
    trainable_result: Mapping[str, object],
    frozen_result: Mapping[str, object],
    non_oracle_result: Mapping[str, object],
    target_n: int,
    ratio: float,
) -> Dict[str, object]:
    trainable_rows = _rows_at_n(trainable_result, target_n)
    frozen_rows = _rows_at_n(frozen_result, target_n)
    non_oracle_rows = _rows_at_n(non_oracle_result, target_n)
    frozen_by_seed = {int(row.get("seed", -1)): float(row.get("accuracy", 0.0)) for row in frozen_rows}
    non_oracle_by_seed = {int(row.get("seed", -1)): float(row.get("accuracy", 0.0)) for row in non_oracle_rows}
    trainable_beats_frozen = 0
    trainable_beats_non_oracle = 0
    for row in trainable_rows:
        seed = int(row.get("seed", -1))
        accuracy = float(row.get("accuracy", 0.0))
        if accuracy > frozen_by_seed.get(seed, 1.0):
            trainable_beats_frozen += 1
        if accuracy > non_oracle_by_seed.get(seed, 1.0):
            trainable_beats_non_oracle += 1

    trainable_predictions: List[int] = []
    non_oracle_predictions: List[int] = []
    labels: List[int] = []
    for trainable_row, non_oracle_row in zip(trainable_rows, non_oracle_rows):
        trainable_predictions.extend(int(value) for value in trainable_row.get("predictions", []))
        non_oracle_predictions.extend(int(value) for value in non_oracle_row.get("predictions", []))
        labels.extend(int(value) for value in trainable_row.get("labels", []))
    ci = paired_bootstrap_ci(trainable_predictions, non_oracle_predictions, labels, seed=target_n, samples=1000)
    controls_pass = _critical_controls_pass(trainable_result)
    ratio_gate = ratio >= 10.0
    frozen_gate = trainable_beats_frozen >= 8
    non_oracle_gate = trainable_beats_non_oracle >= 8
    ci_gate = float(ci["lower"]) > 0.0
    return {
        "target_n": int(target_n),
        "capacity_ratio_gate": ratio_gate,
        "capacity_ratio": ratio,
        "trainable_beats_frozen_seed_count": trainable_beats_frozen,
        "trainable_beats_frozen_gate": frozen_gate,
        "trainable_beats_non_oracle_seed_count": trainable_beats_non_oracle,
        "trainable_beats_best_non_oracle_high_n_gate": non_oracle_gate,
        "paired_bootstrap_ci_vs_non_oracle": ci,
        "paired_bootstrap_ci_lower_bound_gate": ci_gate,
        "critical_controls_gate": controls_pass,
        "all_critical_gates_pass": bool(ratio_gate and frozen_gate and non_oracle_gate and ci_gate and controls_pass),
    }


def _rows_at_n(result: Mapping[str, object], n_blocks: int) -> List[Mapping[str, object]]:
    rows = result.get("rows", [])
    return [
        row
        for row in rows
        if isinstance(row, Mapping) and int(row.get("n_blocks", -1)) == int(n_blocks)
    ]


def _decision_output(
    ratio: float,
    controls_pass: bool,
    completed_seeds: int,
    budget: Stage8SearchBudget,
    result: Mapping[str, object],
) -> str:
    capacity = int(result.get("capacity", {}).get("capacity", 0)) if isinstance(result.get("capacity"), Mapping) else 0
    if ratio >= 10.0 and controls_pass and completed_seeds >= 10:
        return "STAGE8_TARGET_ACHIEVED"
    if capacity > 0 and ratio > 1.0:
        return "STAGE8_PROMISING_BUT_NOT_10X"
    if budget.max_architecture_configs < 50 or budget.max_n < 128:
        return "STAGE8_SEARCH_NEEDS_MORE_COMPUTE"
    return "STAGE8_HYPOTHESIS_NOT_SUPPORTED"


def _n_values(max_n: int) -> List[int]:
    return [value for value in SCALE_SCHEDULE if value <= max_n] or [8]


def _time_exhausted(started: float, budget: Stage8SearchBudget) -> bool:
    return (time.perf_counter() - started) >= budget.max_wall_clock_seconds


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _append_jsonl(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_run_checkpoint(phase: str, config_id: str, result: Mapping[str, object]) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"{phase}_{config_id}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _load_database(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _completed_config_phases(path: Path, budget_signature: str | None = None) -> set[tuple[str, str]]:
    return {
        (str(row.get("phase")), str(row.get("config_id")))
        for row in _load_database(path)
        if row.get("status") == "completed"
        and (budget_signature is None or row.get("budget_signature") == budget_signature)
    }


def _append_controls(result: Mapping[str, object]) -> None:
    for row in result.get("rows", []):  # type: ignore[union-attr]
        if isinstance(row, Mapping) and row.get("controls"):
            _append_jsonl(
                CONTROLS_AUDIT_PATH,
                {
                    "config_id": result.get("config_id"),
                    "architecture_name": result.get("architecture_name"),
                    "seed": row.get("seed"),
                    "n_blocks": row.get("n_blocks"),
                    "controls": row.get("controls"),
                },
            )
    shortcuts = [
        row.get("shortcut_audit")
        for row in result.get("rows", [])  # type: ignore[union-attr]
        if isinstance(row, Mapping) and row.get("shortcut_audit")
    ]
    if shortcuts:
        dump_json(SHORTCUT_AUDIT_PATH, {"latest_config_id": result.get("config_id"), "shortcut_audits": shortcuts[:20]})


def _append_stage8a_controls(result: Mapping[str, object]) -> None:
    for row in result.get("rows", []):  # type: ignore[union-attr]
        if isinstance(row, Mapping):
            _append_jsonl(
                STAGE8A_CONTROLS_AUDIT_PATH,
                {
                    "config_id": result.get("config_id"),
                    "architecture_name": result.get("architecture_name"),
                    "model_kind": result.get("model_kind"),
                    "seed": row.get("seed"),
                    "n_blocks": row.get("n_blocks"),
                    "controls": row.get("controls", {}),
                    "shortcut_audit": row.get("shortcut_audit", {}),
                },
            )


def _append_stage8b_controls(result: Mapping[str, object], budget_signature: str) -> None:
    for row in result.get("rows", []):  # type: ignore[union-attr]
        if isinstance(row, Mapping):
            _append_jsonl(
                STAGE8B_CONTROLS_AUDIT_PATH,
                {
                    "budget_signature": budget_signature,
                    "config_id": result.get("config_id"),
                    "architecture_name": result.get("architecture_name"),
                    "model_kind": result.get("model_kind"),
                    "seed": row.get("seed"),
                    "n_blocks": row.get("n_blocks"),
                    "controls": row.get("controls", {}),
                    "shortcut_audit": row.get("shortcut_audit", {}),
                    "accuracy_by_task_family": row.get("accuracy_by_task_family", {}),
                },
            )


def _write_compute_audit(rows: Sequence[Mapping[str, object]]) -> None:
    audit = []
    for row in rows:
        compute = row.get("compute_estimate") or row.get("compute") or {}
        audit.append(
            {
                "config_id": row.get("config_id"),
                "architecture_name": row.get("architecture_name"),
                "capacity_C": row.get("capacity_C"),
                "capacity_ratio_vs_baseline": row.get("capacity_ratio_vs_baseline"),
                "compute": compute,
                "capacity_per_compute": (
                    float(row.get("capacity_C", 0)) / float(compute.get("estimated_forward_compute", 1.0))
                    if isinstance(compute, Mapping) and float(compute.get("estimated_forward_compute", 0.0)) > 0
                    else 0.0
                ),
            }
        )
    dump_json(COMPUTE_AUDIT_PATH, {"rows": audit})


def _write_stage8b_compute_audit(rows: Sequence[Mapping[str, object]], budget_signature: str) -> None:
    audit = []
    for row in rows:
        compute = row.get("compute_estimate") or row.get("compute") or {}
        audit.append(
            {
                "budget_signature": budget_signature,
                "phase": row.get("phase"),
                "config_id": row.get("config_id"),
                "architecture_name": row.get("architecture_name"),
                "capacity_C": row.get("capacity_C"),
                "capacity_ratio_vs_baseline": row.get("capacity_ratio_vs_baseline"),
                "parameter_count": row.get("parameter_count"),
                "wall_clock_time": row.get("wall_clock_time"),
                "compute": compute,
                "capacity_per_compute": (
                    float(row.get("capacity_C", 0)) / float(compute.get("estimated_forward_compute", 1.0))
                    if isinstance(compute, Mapping) and float(compute.get("estimated_forward_compute", 0.0)) > 0
                    else 0.0
                ),
            }
        )
    dump_json(STAGE8B_COMPUTE_AUDIT_PATH, {"budget_signature": budget_signature, "rows": audit})


def _write_best_config(config: Stage8ArchitectureConfig | None, row: Mapping[str, object]) -> None:
    BEST_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if config is None:
        BEST_CONFIG_PATH.write_text("# no selected Stage 8 config\n", encoding="utf-8")
        return
    lines = ["# Frozen Stage 8B-selected architecture. Do not edit after Stage 8C starts."]
    for key, value in architecture_config_to_dict(config).items():
        lines.append(f"{key}: {json.dumps(value)}")
    lines.append(f"stage8b_capacity_C: {json.dumps(row.get('capacity_C', 0))}")
    lines.append(f"stage8b_capacity_ratio_vs_baseline: {json.dumps(row.get('capacity_ratio_vs_baseline', 0.0))}")
    BEST_CONFIG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_freeze_manifest(
    config: Stage8ArchitectureConfig | None,
    row: Mapping[str, object],
    baseline_capacity: int,
    target_capacity: int,
    target_n: int,
    budget: Stage8SearchBudget,
    budget_signature: str,
) -> None:
    if config is None:
        return
    manifest = {
        "selected_config_id": config.config_id,
        "full_architecture_config": architecture_config_to_dict(config),
        "baseline_capacity_from_stage8a": int(baseline_capacity),
        "target_capacity": int(target_capacity),
        "stage8b_target_n": int(target_n),
        "achieved_dev_capacity": int(row.get("capacity_C", 0)),
        "achieved_dev_capacity_ratio": row.get("capacity_ratio_vs_baseline", 0.0),
        "selection_metric": {
            "priority": [
                "highest_dev_capacity_ratio",
                "controls_pass",
                "lowest_compute_among_ties",
                "best_stability_across_seeds",
                "best_held_out_dev_template_performance",
            ],
            "selected_row_capacity": row.get("capacity_C", 0),
            "selected_row_accuracy_by_n": row.get("accuracy_by_n", {}),
        },
        "dev_seeds_used": row.get("seeds", []),
        "task_families_used": row.get("task_families", []),
        "n_schedule_used": row.get("eval_n_schedule", []),
        "controls_summary": _control_summary_for_report(row),
        "compute_summary": row.get("compute_estimate", {}),
        "git_commit_hash": _git_commit_hash(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "budget": asdict(budget),
        "budget_signature": budget_signature,
        "freeze_statement": "This config is frozen before Stage 8C. No architecture or hyperparameter changes are allowed after this point.",
    }
    dump_json(CONFIG_FREEZE_MANIFEST_PATH, manifest)


def _schedule_from_rows(rows: Sequence[object], key: str) -> List[int]:
    del key
    values = sorted({int(row.get("n_blocks", 0)) for row in rows if isinstance(row, Mapping)})
    return values


def _mean_accuracy_at(result: Mapping[str, object], n_blocks: int) -> float:
    rows = result.get("rows", [])
    values = [
        float(row.get("accuracy", 0.0))
        for row in rows
        if isinstance(row, Mapping) and int(row.get("n_blocks", -1)) == int(n_blocks)
    ]
    return mean(values) if values else 0.0


def _soft_controls_pass(result: Mapping[str, object]) -> bool:
    rows = result.get("rows", [])
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        controls = row.get("controls", {})
        if not isinstance(controls, Mapping):
            continue
        randomized = controls.get("randomized_labels")
        mismatch = controls.get("evidence_candidate_mismatch")
        if isinstance(randomized, Mapping) and isinstance(mismatch, Mapping):
            if float(randomized.get("accuracy", 1.0)) <= 0.30 and float(mismatch.get("degradation", 0.0)) >= 0.05:
                return True
    return False
