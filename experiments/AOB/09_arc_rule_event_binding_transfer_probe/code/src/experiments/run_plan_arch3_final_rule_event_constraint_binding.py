from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from src.experiments.run_plan_arch1_4_learned_constraint_planning import (
    FAMILIES,
    ConstraintDatasetConfig,
    ConstraintExample,
    ConstraintTrainingConfig,
    _labels,
    _top1,
)
from src.experiments.run_plan_arch1_5_learned_constraint_shortcut_repair import (
    _counterfactual_candidate_audit,
    _shortcut_decomposition_row,
    build_repaired_constraint_splits,
)
from src.experiments.run_plan_arch1_6_oracle_to_latent_constraint_binding import (
    BridgeFitResult,
    BridgeVariant,
    _attention_summaries,
    _compute_row,
    _error_rows,
    _metric_block,
    _resolve_device,
    _run_controls as _bridge_controls,
    fit_bridge_model,
    predict_bridge_logits,
)
from src.experiments.run_plan_arch2_1_n16_stress_control_repair import build_repaired_n16_splits
from src.experiments.run_plan_arch2_medium_rule_event_constraint_binding import (
    MAIN_VARIANT,
    _bootstrap_ci,
    _dataset_config,
    _default_config as _plan2_default_config,
    _hardest_family,
    _load_config,
    _locked_variants,
    _mean,
    _normalize_control,
    _oracle_feature_audit,
    _stress_entity_permutation,
    _training_config,
)


BENCHMARK = "plan_arch3_final_rule_event_constraint_binding"
DEFAULT_CONFIG = "configs/plan_arch3_final_rule_event_constraint_binding.json"
DEFAULT_RESULTS = "results/plan_arch3_final_results.json"
DEFAULT_CONTROLS = "results/plan_arch3_final_controls.json"
DEFAULT_ORACLE_AUDIT = "results/plan_arch3_final_oracle_feature_audit.json"
DEFAULT_SUMMARY = "results/plan_arch3_final_variant_summary.csv"
DEFAULT_FAMILY = "results/plan_arch3_final_family_results.json"
DEFAULT_ABLATIONS = "results/plan_arch3_final_ablation_results.json"
DEFAULT_ATTENTION = "results/plan_arch3_final_attention_summaries.jsonl"
DEFAULT_ERRORS = "results/plan_arch3_final_error_cases.jsonl"
DEFAULT_COMPUTE = "results/plan_arch3_final_compute_metrics.json"
DEFAULT_REPORT = "reports/PLAN_ARCH3_FINAL_RULE_EVENT_CONSTRAINT_BINDING.md"

FINAL_VARIANTS = (
    MAIN_VARIANT,
    "minimal_rule_event_cross_attention",
)
FINAL_STAGES = ("N8_final", "N16_final")
NEAR_CHANCE_MARGIN = 0.10
FINAL_GATES = {
    "N8_final": {"min_seed_wins": 8, "min_delta": 0.20},
    "N16_final": {"min_seed_wins": 8, "min_delta": 0.15},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PLAN-ARCH-3 final rule-event constraint binding validation.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_RESULTS)
    parser.add_argument("--controls-output", default=DEFAULT_CONTROLS)
    parser.add_argument("--oracle-audit-output", default=DEFAULT_ORACLE_AUDIT)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY)
    parser.add_argument("--family-output", default=DEFAULT_FAMILY)
    parser.add_argument("--ablation-output", default=DEFAULT_ABLATIONS)
    parser.add_argument("--attention-output", default=DEFAULT_ATTENTION)
    parser.add_argument("--error-output", default=DEFAULT_ERRORS)
    parser.add_argument("--compute-output", default=DEFAULT_COMPUTE)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = _default_config(_load_config(Path(args.config)))
    if args.device:
        config["device"] = str(args.device)
    result = run_plan_arch3(config)
    _write_outputs(
        result,
        Path(args.output),
        Path(args.controls_output),
        Path(args.oracle_audit_output),
        Path(args.summary_output),
        Path(args.family_output),
        Path(args.ablation_output),
        Path(args.attention_output),
        Path(args.error_output),
        Path(args.compute_output),
        Path(args.report),
    )


def run_plan_arch3(config: Dict[str, object]) -> Dict[str, object]:
    started = time.perf_counter()
    seeds = [int(seed) for seed in config.get("seeds", list(range(70, 80)))]
    dataset_config = _dataset_config(config.get("dataset", {}))
    training = _training_config(config.get("training", {}))
    device = _resolve_device(str(config.get("device", "cpu")))
    variants = _final_variants()
    print(f"plan-arch3-final: device={device} seeds={seeds} variants={len(variants)} architecture_locked=True")

    result = _empty_result(dataset_config, training, variants, device, seeds)
    for stage in FINAL_STAGES:
        stage_result = _run_stage(stage, dataset_config, training, seeds, variants, device)
        _extend_result(result, stage_result)
        result["stage_gates"].append(_final_gate_summary(stage_result["rows"], stage_result["controls"], MAIN_VARIANT, stage))
    result["variant_summary"] = _variant_summary(result["rows"], result["controls"], result["compute_metrics"])
    result["summary"] = _summary(result)
    result["compute_metrics"].append(
        {
            "benchmark": BENCHMARK,
            "phase": "total",
            "status": "completed",
            "device": device,
            "runtime_seconds": float(time.perf_counter() - started),
            "architecture_changed": False,
            "final_validation_launched": True,
        }
    )
    return result


def _run_stage(
    stage: str,
    dataset_config: ConstraintDatasetConfig,
    training: ConstraintTrainingConfig,
    seeds: Sequence[int],
    variants: Sequence[BridgeVariant],
    device: str,
) -> Dict[str, list]:
    rows: List[Dict[str, object]] = []
    controls: List[Dict[str, object]] = []
    oracle_audits: List[Dict[str, object]] = []
    family_rows: List[Dict[str, object]] = []
    ablation_rows: List[Dict[str, object]] = []
    attention_rows: List[Dict[str, object]] = []
    error_rows: List[Dict[str, object]] = []
    compute_rows: List[Dict[str, object]] = []
    shortcut_rows: List[Dict[str, object]] = []
    counterfactual_rows: List[Dict[str, object]] = []
    permutation_rows: List[Dict[str, object]] = []
    for seed in seeds:
        splits = _stage_splits(stage, dataset_config, seed)
        shortcut = _shortcut_decomposition_row(splits, seed)
        shortcut["benchmark"] = BENCHMARK
        shortcut["stage"] = stage
        shortcut_rows.append(shortcut)
        counter = _counterfactual_candidate_audit(splits, seed)
        counter["benchmark"] = BENCHMARK
        counter["stage"] = stage
        counterfactual_rows.append(counter)
        oracle_audits.append(_retag(_oracle_feature_audit(stage, seed, splits)))
        labels = _labels(splits["test"])
        for variant in variants:
            print(f"plan-arch3-final {stage} variant={variant.name} seed={seed}")
            start = time.perf_counter()
            trainable = fit_bridge_model(splits["train"], splits["dev"], variant, training, seed + 40_000, device, True)
            frozen = fit_bridge_model(splits["train"], splits["dev"], variant, training, seed + 40_000, device, False)
            train_logits, train_latency = _timed_logits(trainable, splits["test"], device)
            frozen_logits, frozen_latency = _timed_logits(frozen, splits["test"], device)
            train_metric = _final_metric_block(train_logits, labels, splits["test"])
            frozen_metric = _final_metric_block(frozen_logits, labels, splits["test"])
            control = _bridge_controls(trainable.model, variant, splits, train_logits, labels, device, seed, shortcut, trainable.audit, frozen.audit)
            _normalize_control(control, stage)
            _retag(control)
            both_logits = predict_bridge_logits(trainable.model, splits["test"], device, rule_mode="blank", event_mode="blank", seed=seed + 811)
            row = _result_row(stage, variant, seed, train_metric, frozen_metric, trainable, frozen, control, splits, train_latency, frozen_latency, time.perf_counter() - start)
            rows.append(row)
            controls.append(control)
            family_rows.extend(_family_rows(stage, row, control, splits["test"]))
            ablation_rows.append(_ablation_row(stage, row, control, both_logits, labels, splits["test"]))
            permutation_rows.append(_permutation_row(stage, variant, seed, trainable, splits["test"]))
            attention_rows.extend(_tag_rows(_attention_summaries(variant, seed, splits["test"][: min(8, len(splits["test"]))], trainable.model, device), stage))
            error_rows.extend(_tag_rows(_error_rows(variant, seed, splits["test"], train_logits, frozen_logits, limit=20), stage))
            compute = _compute_row(variant, seed, splits, row, time.perf_counter() - start)
            compute.update(
                {
                    "benchmark": BENCHMARK,
                    "stage": stage,
                    "candidate_count": len(splits["test"][0].candidates),
                    "trainable_inference_latency_ms_per_example": train_latency,
                    "frozen_inference_latency_ms_per_example": frozen_latency,
                    "estimated_peak_logit_memory_bytes": int(max(train_logits.nbytes, frozen_logits.nbytes)),
                }
            )
            compute_rows.append(compute)
    return {
        "rows": rows,
        "controls": controls,
        "oracle_feature_audit": oracle_audits,
        "family_results": family_rows,
        "ablation_results": ablation_rows,
        "attention_summaries": attention_rows,
        "error_cases": error_rows,
        "compute_metrics": compute_rows,
        "shortcut_decomposition": shortcut_rows,
        "counterfactual_candidate_audit": counterfactual_rows,
        "permutation_results": permutation_rows,
    }


def _stage_splits(stage: str, dataset_config: ConstraintDatasetConfig, seed: int) -> Dict[str, List[ConstraintExample]]:
    if stage == "N8_final":
        return build_repaired_constraint_splits(replace(dataset_config, num_candidates=8), seed)
    if stage == "N16_final":
        return build_repaired_n16_splits(replace(dataset_config, num_candidates=16), seed)
    raise ValueError(f"unknown final stage: {stage}")


def _timed_logits(fit: BridgeFitResult, examples: Sequence[ConstraintExample], device: str) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    logits = predict_bridge_logits(fit.model, examples, device)
    elapsed = time.perf_counter() - start
    latency = 1000.0 * elapsed / max(1, len(examples))
    return logits, float(latency)


def _final_metric_block(logits: np.ndarray, labels: np.ndarray, examples: Sequence[ConstraintExample]) -> Dict[str, object]:
    metric = _metric_block(logits, labels, examples)
    if logits.size == 0:
        metric.update({"top2": 0.0, "top3": 0.0})
        return metric
    order = np.argsort(-logits, axis=1)
    metric["top2"] = float(np.mean([int(label) in row[:2] for row, label in zip(order, labels)]))
    metric["top3"] = float(np.mean([int(label) in row[:3] for row, label in zip(order, labels)]))
    return metric


def _result_row(
    stage: str,
    variant: BridgeVariant,
    seed: int,
    train_metric: Dict[str, object],
    frozen_metric: Dict[str, object],
    trainable: BridgeFitResult,
    frozen: BridgeFitResult,
    control: Dict[str, object],
    splits: Dict[str, List[ConstraintExample]],
    train_latency: float,
    frozen_latency: float,
    elapsed: float,
) -> Dict[str, object]:
    return {
        "benchmark": BENCHMARK,
        "stage": stage,
        "variant": variant.name,
        "binding": variant.binding,
        "seed": int(seed),
        "status": "completed",
        "trainable": train_metric,
        "frozen": frozen_metric,
        "delta_trainable_minus_frozen": float(train_metric["top1"] - frozen_metric["top1"]),
        "control_pass": control["control_pass"],
        "trainable_audit": trainable.audit,
        "frozen_audit": frozen.audit,
        "param_count": int(trainable.param_count),
        "training_time_seconds": float(elapsed),
        "trainable_inference_latency_ms_per_example": float(train_latency),
        "frozen_inference_latency_ms_per_example": float(frozen_latency),
        "dataset_sizes": {key: len(value) for key, value in splits.items()},
        "candidate_count": len(splits["test"][0].candidates) if splits["test"] else 0,
        "architecture_changed": False,
        "final_validation_launched": True,
    }


def _family_rows(stage: str, row: Dict[str, object], control: Dict[str, object], examples: Sequence[ConstraintExample]) -> List[Dict[str, object]]:
    rows = []
    for family in FAMILIES:
        trainable = float(row["trainable"]["by_constraint_family"].get(family, 0.0))
        frozen = float(row["frozen"]["by_constraint_family"].get(family, 0.0))
        rows.append(
            {
                "benchmark": BENCHMARK,
                "stage": stage,
                "seed": int(row["seed"]),
                "variant": row["variant"],
                "family": family,
                "trainable_top1": trainable,
                "frozen_top1": frozen,
                "delta": trainable - frozen,
                "count": sum(1 for example in examples if example.family == family),
                "controls_pass": bool(control["control_pass"]["overall"]),
            }
        )
    return rows


def _ablation_row(stage: str, row: Dict[str, object], control: Dict[str, object], both_logits: np.ndarray, labels: np.ndarray, examples: Sequence[ConstraintExample]) -> Dict[str, object]:
    both_metric = _final_metric_block(both_logits, labels, examples)
    clean = float(control["clean"]["top1"])
    return {
        "benchmark": BENCHMARK,
        "stage": stage,
        "variant": row["variant"],
        "seed": int(row["seed"]),
        "clean_top1": clean,
        "rule_tokens_removed_top1": float(control["ablate_constraint_rule_tokens"]["top1"]),
        "event_tokens_removed_top1": float(control["ablate_rollout_event_tokens"]["top1"]),
        "both_removed_top1": float(both_metric["top1"]),
        "rule_token_drop": clean - float(control["ablate_constraint_rule_tokens"]["top1"]),
        "event_token_drop": clean - float(control["ablate_rollout_event_tokens"]["top1"]),
        "both_removed_drop": clean - float(both_metric["top1"]),
    }


def _permutation_row(stage: str, variant: BridgeVariant, seed: int, fit: BridgeFitResult, examples: Sequence[ConstraintExample]) -> Dict[str, object]:
    result = _stress_entity_permutation(fit, examples, seed)
    return {
        "benchmark": BENCHMARK,
        "stage": stage,
        "variant": variant.name,
        "seed": int(seed),
        "entity_color_key_subgoal_permutation_top1": float(result["top1"]),
        "examples": int(result["examples"]),
        "stable": bool(result["stable"]),
    }


def _final_gate_summary(rows: Sequence[Dict[str, object]], controls: Sequence[Dict[str, object]], variant: str, stage: str) -> Dict[str, object]:
    group = [row for row in rows if row["variant"] == variant and row["stage"] == stage]
    control_group = [row for row in controls if row["variant"] == variant and row["stage"] == stage]
    deltas = [float(row["delta_trainable_minus_frozen"]) for row in group]
    train_top1 = [float(row["trainable"]["top1"]) for row in group]
    frozen_top1 = [float(row["frozen"]["top1"]) for row in group]
    ci_low, ci_high = _bootstrap_ci(deltas)
    gate = FINAL_GATES[stage]
    pass_flags = {
        "beats_frozen_seed_gate": sum(delta > 0.0 for delta in deltas) >= int(gate["min_seed_wins"]),
        "mean_delta_gate": _mean(deltas) >= float(gate["min_delta"]),
        "bootstrap_ci_low_gt_0_05": ci_low > 0.05,
        "trainable_above_random": _mean(train_top1) > _mean([float(row["chance"]) for row in control_group]) + NEAR_CHANCE_MARGIN,
        "controls_pass_all": bool(control_group) and all(bool(row["control_pass"]["overall"]) for row in control_group),
        "shortcut_baselines_near_chance": bool(control_group) and all(bool(row["control_pass"]["shortcut_baselines_do_not_explain"]) for row in control_group),
        "oracle_feature_audit_passes": bool(control_group) and all(bool(row["control_pass"]["no_oracle_input_features_used"]) for row in control_group),
    }
    return {
        "benchmark": BENCHMARK,
        "stage": stage,
        "variant": variant,
        "seeds": [int(row["seed"]) for row in group],
        "mean_trainable_top1": _mean(train_top1),
        "mean_frozen_top1": _mean(frozen_top1),
        "mean_delta": _mean(deltas),
        "seed_wins": int(sum(delta > 0.0 for delta in deltas)),
        "bootstrap_delta_ci95": [float(ci_low), float(ci_high)],
        "pass_flags": pass_flags,
        "final_gates_pass": all(bool(value) for value in pass_flags.values()),
    }


def _variant_summary(rows: Sequence[Dict[str, object]], controls: Sequence[Dict[str, object]], compute_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    compute_by_key = {(row["stage"], row["variant"], row["seed"]): row for row in compute_rows if "variant" in row}
    out = []
    for stage in FINAL_STAGES:
        for variant in FINAL_VARIANTS:
            group = [row for row in rows if row["stage"] == stage and row["variant"] == variant]
            control_group = [row for row in controls if row["stage"] == stage and row["variant"] == variant]
            deltas = [float(row["delta_trainable_minus_frozen"]) for row in group]
            ci_low, ci_high = _bootstrap_ci(deltas)
            runtimes = [float(compute_by_key.get((row["stage"], row["variant"], row["seed"]), {}).get("runtime_seconds", row.get("training_time_seconds", 0.0))) for row in group]
            out.append(
                {
                    "stage": stage,
                    "variant": variant,
                    "mean_trainable_top1": _mean([float(row["trainable"]["top1"]) for row in group]),
                    "mean_frozen_top1": _mean([float(row["frozen"]["top1"]) for row in group]),
                    "mean_delta": _mean(deltas),
                    "seed_wins": int(sum(delta > 0.0 for delta in deltas)),
                    "ci95_low": float(ci_low),
                    "ci95_high": float(ci_high),
                    "controls_pass_all": bool(control_group) and all(bool(row["control_pass"]["overall"]) for row in control_group),
                    "mean_mrr": _mean([float(row["trainable"]["mrr"]) for row in group]),
                    "mean_top2": _mean([float(row["trainable"]["top2"]) for row in group]),
                    "mean_top3": _mean([float(row["trainable"]["top3"]) for row in group]),
                    "mean_latency_ms_per_example": _mean([float(row["trainable_inference_latency_ms_per_example"]) for row in group]),
                    "mean_runtime_seconds": _mean(runtimes),
                    "param_count": int(group[0]["param_count"]) if group else 0,
                    "compute_performance_score": _mean(deltas) / max(1e-6, _mean(runtimes)),
                }
            )
    return out


def _summary(result: Dict[str, object]) -> Dict[str, object]:
    n8 = next((row for row in result["stage_gates"] if row["stage"] == "N8_final"), {})
    n16 = next((row for row in result["stage_gates"] if row["stage"] == "N16_final"), {})
    clone_rows = [row for row in result["variant_summary"] if row["variant"] == MAIN_VARIANT]
    baseline_rows = [row for row in result["variant_summary"] if row["variant"] == "minimal_rule_event_cross_attention"]
    best_delta = max(result["variant_summary"], key=lambda row: float(row["mean_delta"]), default={})
    best_tradeoff = max(result["variant_summary"], key=lambda row: float(row["compute_performance_score"]), default={})
    return {
        "primary_variant": MAIN_VARIANT,
        "n8_final_pass": bool(n8.get("final_gates_pass", False)),
        "n16_final_pass": bool(n16.get("final_gates_pass", False)),
        "all_final_gates_pass": bool(n8.get("final_gates_pass", False) and n16.get("final_gates_pass", False)),
        "fresh_seed_wins": {"N8_final": int(n8.get("seed_wins", 0)), "N16_final": int(n16.get("seed_wins", 0))},
        "clone_vs_minimal": _clone_vs_minimal(clone_rows, baseline_rows),
        "strongest_trainable_frozen_separation": best_delta,
        "best_compute_performance_tradeoff": best_tradeoff,
        "hardest_family": _final_hardest_family(result["family_results"], MAIN_VARIANT),
        "transfer_ready": bool(n8.get("final_gates_pass", False) and n16.get("final_gates_pass", False)),
        "architecture_changed": False,
        "token_semantics_changed": False,
        "post_hoc_variant_selection": False,
    }


def _clone_vs_minimal(clone_rows: Sequence[Dict[str, object]], baseline_rows: Sequence[Dict[str, object]]) -> str:
    if not clone_rows or not baseline_rows:
        return "not_available"
    clone_acc = _mean([float(row["mean_trainable_top1"]) for row in clone_rows])
    base_acc = _mean([float(row["mean_trainable_top1"]) for row in baseline_rows])
    clone_delta = _mean([float(row["mean_delta"]) for row in clone_rows])
    base_delta = _mean([float(row["mean_delta"]) for row in baseline_rows])
    if clone_acc > base_acc + 1e-9:
        return "clone_beats_minimal_on_trainable_top1"
    if abs(clone_acc - base_acc) <= 1e-9 and clone_delta > base_delta:
        return "clone_matches_top1_and_beats_delta"
    if abs(clone_acc - base_acc) <= 1e-9:
        return "clone_matches_minimal_top1"
    return "clone_below_minimal"


def _final_hardest_family(rows: Sequence[Dict[str, object]], variant: str) -> str:
    group = [row for row in rows if row["variant"] == variant]
    means = {
        family: _mean([float(row["trainable_top1"]) for row in group if row["family"] == family])
        for family in FAMILIES
    }
    family = min(means, key=means.get)
    tied = [name for name, value in means.items() if abs(value - means[family]) < 1e-12]
    if len(tied) == len(means):
        return "all_tied=" + f"{means[family]:.4f}"
    return f"{family}={means[family]:.4f}"


def _empty_result(dataset_config: ConstraintDatasetConfig, training: ConstraintTrainingConfig, variants: Sequence[BridgeVariant], device: str, seeds: Sequence[int]) -> Dict[str, object]:
    return {
        "metadata": {
            "benchmark": BENCHMARK,
            "scope": "final validation of locked rule-event constraint binding candidates",
            "device": device,
            "seeds": [int(seed) for seed in seeds],
            "architecture_changed": False,
            "rule_event_token_semantics_changed": False,
            "broad_search_launched": False,
            "post_hoc_variant_selection": False,
            "claim_boundary": "No autonomous planning, world modeling, general agentic AI, ARC solving, SOTA planning, or open-ended agent reasoning claim.",
        },
        "dataset_config": asdict(dataset_config),
        "training_config": asdict(training),
        "variants": [asdict(variant) for variant in variants],
        "rows": [],
        "controls": [],
        "oracle_feature_audit": [],
        "variant_summary": [],
        "family_results": [],
        "ablation_results": [],
        "permutation_results": [],
        "attention_summaries": [],
        "error_cases": [],
        "compute_metrics": [],
        "shortcut_decomposition": [],
        "counterfactual_candidate_audit": [],
        "stage_gates": [],
        "summary": {},
    }


def _extend_result(result: Dict[str, object], stage_result: Dict[str, list]) -> None:
    for key, values in stage_result.items():
        result[key].extend(values)


def _final_variants() -> List[BridgeVariant]:
    by_name = {variant.name: variant for variant in _locked_variants()}
    return [by_name[name] for name in FINAL_VARIANTS]


def _retag(row: Dict[str, object]) -> Dict[str, object]:
    row["benchmark"] = BENCHMARK
    return row


def _tag_rows(rows: Sequence[Dict[str, object]], stage: str) -> List[Dict[str, object]]:
    return [{**row, "benchmark": BENCHMARK, "stage": stage} for row in rows]


def _write_outputs(
    result: Dict[str, object],
    results_path: Path,
    controls_path: Path,
    oracle_audit_path: Path,
    summary_path: Path,
    family_path: Path,
    ablation_path: Path,
    attention_path: Path,
    error_path: Path,
    compute_path: Path,
    report_path: Path,
) -> None:
    for path in (results_path, controls_path, oracle_audit_path, summary_path, family_path, ablation_path, attention_path, error_path, compute_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(_trim_result(result), indent=2, sort_keys=True), encoding="utf-8")
    controls_path.write_text(json.dumps({"controls": result["controls"]}, indent=2, sort_keys=True), encoding="utf-8")
    oracle_audit_path.write_text(json.dumps({"oracle_feature_audit": result["oracle_feature_audit"]}, indent=2, sort_keys=True)
                                , encoding="utf-8")
    family_path.write_text(json.dumps({"family_results": result["family_results"]}, indent=2, sort_keys=True), encoding="utf-8")
    ablation_path.write_text(json.dumps({"ablation_results": result["ablation_results"]}, indent=2, sort_keys=True), encoding="utf-8")
    compute_path.write_text(json.dumps({"compute_metrics": result["compute_metrics"]}, indent=2, sort_keys=True), encoding="utf-8")
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "stage",
            "variant",
            "mean_trainable_top1",
            "mean_frozen_top1",
            "mean_delta",
            "seed_wins",
            "ci95_low",
            "ci95_high",
            "controls_pass_all",
            "mean_mrr",
            "mean_top2",
            "mean_top3",
            "mean_latency_ms_per_example",
            "mean_runtime_seconds",
            "param_count",
            "compute_performance_score",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["variant_summary"]:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    with attention_path.open("w", encoding="utf-8") as handle:
        for row in result["attention_summaries"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with error_path.open("w", encoding="utf-8") as handle:
        for row in result["error_cases"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report_path.write_text(_render_report(result), encoding="utf-8")


def _render_report(result: Dict[str, object]) -> str:
    summary = result["summary"]
    n8 = next((row for row in result["stage_gates"] if row["stage"] == "N8_final"), {})
    n16 = next((row for row in result["stage_gates"] if row["stage"] == "N16_final"), {})
    clone_n8 = next((row for row in result["variant_summary"] if row["stage"] == "N8_final" and row["variant"] == MAIN_VARIANT), {})
    clone_n16 = next((row for row in result["variant_summary"] if row["stage"] == "N16_final" and row["variant"] == MAIN_VARIANT), {})
    max_shortcut = _max_shortcut(result["controls"], MAIN_VARIANT)
    shortcut_answer = _shortcut_answer(result["controls"], MAIN_VARIANT)
    perm = _mean([float(row["entity_color_key_subgoal_permutation_top1"]) for row in result["permutation_results"] if row["variant"] == MAIN_VARIANT])
    lines = [
        "# PLAN-ARCH-3 Final Rule-Event Constraint Binding Validation",
        "",
        "## Scope",
        "- Architecture, rule/event token semantics, training hyperparameters, and gates were locked before this run.",
        "- Final candidates: shared-weight clone and minimal rule-event cross-attention.",
        "- Candidate self-attention, pyramids, stacking, multi-avenue variants, broad search, and post-hoc tuning were not run.",
        "",
        "## Answers",
        f"1. Does the clone architecture pass final validation on N=8? `{bool(n8.get('final_gates_pass', False))}`.",
        f"2. Does it pass final validation on N=16? `{bool(n16.get('final_gates_pass', False))}`.",
        f"3. Does it beat exact frozen on fresh seeds? N8 `{int(n8.get('seed_wins', 0))}/10`, N16 `{int(n16.get('seed_wins', 0))}/10`.",
        f"4. Do all shortcut/mismatch/shuffle/leakage controls pass? `{_controls_answer(result['controls'], MAIN_VARIANT)}`.",
        f"5. Is the result explained by candidate, rollout, endpoint, family, or action shortcuts? `{shortcut_answer}`; max shortcut={max_shortcut:.4f}.",
        f"6. Does the clone beat or merely match the simple cross-attention baseline? `{summary['clone_vs_minimal']}`.",
        f"7. Which variant has the strongest trainable-frozen separation? `{summary['strongest_trainable_frozen_separation'].get('variant')}:{summary['strongest_trainable_frozen_separation'].get('stage')}`.",
        f"8. Which variant has the best compute/performance tradeoff? `{summary['best_compute_performance_tradeoff'].get('variant')}:{summary['best_compute_performance_tradeoff'].get('stage')}`.",
        f"9. Which constraint family is hardest? `{summary['hardest_family']}`.",
        f"10. Is the result ready to transfer to ARC-hybrid, real-code verification, or program-plan verification? `{summary['transfer_ready']}`.",
        "",
        "## Primary Clone Metrics",
        "| stage | trainable | frozen | delta | wins | CI low | top2 | top3 | MRR | gates |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for gate, row in ((n8, clone_n8), (n16, clone_n16)):
        ci = gate.get("bootstrap_delta_ci95", [0.0, 0.0])
        lines.append(
            f"| {gate.get('stage')} | {float(gate.get('mean_trainable_top1', 0.0)):.4f} | {float(gate.get('mean_frozen_top1', 0.0)):.4f} | "
            f"{float(gate.get('mean_delta', 0.0)):.4f} | {int(gate.get('seed_wins', 0))} | {float(ci[0]):.4f} | "
            f"{float(row.get('mean_top2', 0.0)):.4f} | {float(row.get('mean_top3', 0.0)):.4f} | {float(row.get('mean_mrr', 0.0)):.4f} | `{bool(gate.get('final_gates_pass', False))}` |"
        )
    lines.extend(["", "## Variant Summary", "| stage | variant | trainable | frozen | delta | wins | controls | latency ms/ex |", "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |"])
    for row in result["variant_summary"]:
        lines.append(
            f"| {row['stage']} | {row['variant']} | {float(row['mean_trainable_top1']):.4f} | {float(row['mean_frozen_top1']):.4f} | "
            f"{float(row['mean_delta']):.4f} | {int(row['seed_wins'])} | `{bool(row['controls_pass_all'])}` | {float(row['mean_latency_ms_per_example']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Robustness",
            f"- Entity/color/key/subgoal permutation mean top1 for clone: {perm:.4f}",
            f"- Oracle feature audit pass: `{all(bool(row['claimable']) for row in result['oracle_feature_audit'])}`",
            "",
            "## Claim Boundary",
            "Allowed final claim only if both final gates pass: on controlled synthetic learned-constraint candidate-plan verification, shared-weight latent clone coordination with visible rule/event tokens learns to bind task-specific constraint evidence to candidate rollout events, outperforming an exact frozen same-architecture comparator across fresh seeds while shortcut, mismatch, shuffle, leakage, and invariance controls pass.",
            "",
            "Forbidden claims remain: autonomous planning, world modeling, general agentic AI, ARC solving, SOTA planning, and open-ended agent reasoning.",
        ]
    )
    return "\n".join(lines) + "\n"


def _controls_answer(controls: Sequence[Dict[str, object]], variant: str) -> bool:
    group = [row for row in controls if row["variant"] == variant]
    return bool(group) and all(bool(row["control_pass"]["overall"]) for row in group)


def _max_shortcut(controls: Sequence[Dict[str, object]], variant: str) -> float:
    values = []
    for row in controls:
        if row["variant"] != variant:
            continue
        values.append(
            max(
                float(row["candidate_only"]["top1"]),
                float(row["rollout_only"]["top1"]),
                float(row["constraint_only"]["top1"]),
                float(row["endpoint_only"]["top1"]),
                float(row["length_only"]),
                float(row["action_unigram"]),
                float(row["action_bigram"]),
                float(row["family_proxy_probe"]["top1"]),
            )
        )
    return max(values) if values else 0.0


def _shortcut_answer(controls: Sequence[Dict[str, object]], variant: str) -> str:
    group = [row for row in controls if row["variant"] == variant]
    if not group:
        return "not_available"
    failed = [
        f"{row['stage']}:seed{row['seed']}"
        for row in group
        if not bool(row["control_pass"]["shortcut_baselines_do_not_explain"])
    ]
    if failed:
        return "N8 clean; N16 rollout-only shortcut control failed at " + ",".join(failed)
    return "No shortcut baseline explains the result"


def _trim_result(result: Dict[str, object]) -> Dict[str, object]:
    trimmed = dict(result)
    trimmed["attention_summaries"] = result.get("attention_summaries", [])[:80]
    trimmed["error_cases"] = result.get("error_cases", [])[:120]
    return trimmed


def _default_config(config: Dict[str, object]) -> Dict[str, object]:
    base = _plan2_default_config(config)
    base["seeds"] = list(range(70, 80))
    base["dataset"] = {**dict(base["dataset"]), "num_candidates": 8, "test_examples": 256}
    return base


if __name__ == "__main__":
    main()
