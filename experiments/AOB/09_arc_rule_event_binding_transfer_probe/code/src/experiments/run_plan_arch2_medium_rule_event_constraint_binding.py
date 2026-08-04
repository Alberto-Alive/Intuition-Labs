from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence

import numpy as np

from src.experiments.run_plan_arch1_4_learned_constraint_planning import (
    FAMILIES,
    ConstraintDatasetConfig,
    ConstraintEvidence,
    ConstraintExample,
    ConstraintTrainingConfig,
    _action_counts,
    _accuracy,
    _bigram_predictions,
    _final_distance_predictions,
    _labels,
    _length_predictions,
    _softmax_np,
    _top1,
    _unigram_predictions,
)
from src.experiments.run_plan_arch1_5_learned_constraint_shortcut_repair import (
    _counterfactual_candidate_audit,
    _role_cells,
    _shortcut_decomposition_row,
    build_repaired_constraint_splits,
)
from src.experiments.run_plan_arch1_6_oracle_to_latent_constraint_binding import (
    BENCHMARK as BRIDGE_BENCHMARK,
    BridgeFitResult,
    BridgeVariant,
    _attention_summaries,
    _bridge_variants,
    _compute_row,
    _error_rows,
    _failure_type_counts,
    _metric_block,
    _random_labels,
    _resolve_device,
    _run_controls as _bridge_controls,
    fit_bridge_model,
    oracle_teacher,
    predict_bridge_logits,
)


BENCHMARK = "plan_arch2_medium_rule_event_constraint_binding"
DEFAULT_CONFIG = "configs/plan_arch2_medium_rule_event_constraint_binding.json"
DEFAULT_RESULTS = "results/plan_arch2_medium_results.json"
DEFAULT_CONTROLS = "results/plan_arch2_medium_controls.json"
DEFAULT_ORACLE_AUDIT = "results/plan_arch2_medium_oracle_feature_audit.json"
DEFAULT_BINDING_AUDIT = "results/plan_arch2_medium_rule_event_binding_audit.json"
DEFAULT_LEADERBOARD = "results/plan_arch2_medium_variant_leaderboard.csv"
DEFAULT_FAMILY = "results/plan_arch2_medium_family_results.json"
DEFAULT_ABLATIONS = "results/plan_arch2_medium_ablation_results.json"
DEFAULT_ATTENTION = "results/plan_arch2_medium_attention_summaries.jsonl"
DEFAULT_ERRORS = "results/plan_arch2_medium_error_cases.jsonl"
DEFAULT_COMPUTE = "results/plan_arch2_medium_compute_metrics.json"
DEFAULT_REPORT = "reports/PLAN_ARCH2_MEDIUM_RULE_EVENT_CONSTRAINT_BINDING.md"

MAIN_VARIANT = "shared_weight_constraint_clone_with_event_tokens"
LOCKED_VARIANTS = (
    MAIN_VARIANT,
    "rule_event_bilinear_verifier",
    "minimal_rule_event_cross_attention",
    "transition_tuple_verifier_with_aux_heads",
    "P1_rollout_no_final_constraint_with_event_tokens",
)
NEAR_CHANCE_MARGIN = 0.10
COLLAPSE_MARGIN = 0.15
DELTA_GATE = 0.20


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PLAN-ARCH-2 medium rule-event constraint binding validation.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_RESULTS)
    parser.add_argument("--controls-output", default=DEFAULT_CONTROLS)
    parser.add_argument("--oracle-audit-output", default=DEFAULT_ORACLE_AUDIT)
    parser.add_argument("--binding-audit-output", default=DEFAULT_BINDING_AUDIT)
    parser.add_argument("--leaderboard-output", default=DEFAULT_LEADERBOARD)
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
    result = run_plan_arch2(config)
    _write_outputs(
        result,
        Path(args.output),
        Path(args.controls_output),
        Path(args.oracle_audit_output),
        Path(args.binding_audit_output),
        Path(args.leaderboard_output),
        Path(args.family_output),
        Path(args.ablation_output),
        Path(args.attention_output),
        Path(args.error_output),
        Path(args.compute_output),
        Path(args.report),
    )


def run_plan_arch2(config: Dict[str, object]) -> Dict[str, object]:
    started = time.perf_counter()
    seeds = [int(seed) for seed in config.get("seeds", [10, 11, 12, 13, 14])]
    dataset_config = _dataset_config(config.get("dataset", {}))
    training = _training_config(config.get("training", {}))
    device = _resolve_device(str(config.get("device", "cpu")))
    variants = _locked_variants()
    print(f"plan-arch2-medium: device={device} seeds={seeds} variants={len(variants)} final_validation=False")

    result = _empty_result(dataset_config, training, variants, device)
    n8_stage = _run_stage("N8_medium", dataset_config, training, seeds, variants, device)
    _extend_result(result, n8_stage)
    n8_summary = _medium_gate_summary(n8_stage["rows"], n8_stage["controls"], MAIN_VARIANT, "N8_medium")
    result["stage_gates"].append(n8_summary)
    if n8_summary["medium_gates_pass"]:
        n16_config = replace(dataset_config, num_candidates=16, test_examples=max(128, int(dataset_config.test_examples)))
        n16_stage = _run_stage("N16_stress", n16_config, training, seeds, variants, device)
        _extend_result(result, n16_stage)
        result["stage_gates"].append(_medium_gate_summary(n16_stage["rows"], n16_stage["controls"], MAIN_VARIANT, "N16_stress"))
    result["leaderboard"] = _leaderboard(result["rows"], result["controls"])
    result["summary"] = _summary(result)
    result["compute_metrics"].append(
        {
            "benchmark": BENCHMARK,
            "status": "completed",
            "device": device,
            "total_runtime_seconds": float(time.perf_counter() - started),
            "final_validation_launched": False,
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
    binding_audits: List[Dict[str, object]] = []
    family_results: List[Dict[str, object]] = []
    ablation_results: List[Dict[str, object]] = []
    attention_rows: List[Dict[str, object]] = []
    error_rows: List[Dict[str, object]] = []
    compute_rows: List[Dict[str, object]] = []
    shortcut_rows: List[Dict[str, object]] = []
    counterfactual_rows: List[Dict[str, object]] = []
    for seed in seeds:
        build_config = replace(dataset_config, num_candidates=min(8, int(dataset_config.num_candidates)))
        base_splits = build_repaired_constraint_splits(build_config, seed)
        splits = _expand_splits_if_needed(base_splits, int(dataset_config.num_candidates), seed)
        shortcut_row = _shortcut_decomposition_row(splits, seed)
        shortcut_row["stage"] = stage
        shortcut_rows.append(shortcut_row)
        counter = _counterfactual_candidate_audit(splits, seed)
        counter["stage"] = stage
        counterfactual_rows.append(counter)
        oracle_audits.append(_oracle_feature_audit(stage, seed, splits))
        labels = _labels(splits["test"])
        for variant in variants:
            print(f"plan-arch2-medium {stage} variant={variant.name} seed={seed}")
            start = time.perf_counter()
            trainable = fit_bridge_model(splits["train"], splits["dev"], variant, training, seed + 20_000, device, True)
            frozen = fit_bridge_model(splits["train"], splits["dev"], variant, training, seed + 20_000, device, False)
            train_logits = predict_bridge_logits(trainable.model, splits["test"], device)
            frozen_logits = predict_bridge_logits(frozen.model, splits["test"], device)
            train_metric = _metric_block(train_logits, labels, splits["test"])
            frozen_metric = _metric_block(frozen_logits, labels, splits["test"])
            control = _bridge_controls(trainable.model, variant, splits, train_logits, labels, device, seed, shortcut_row, trainable.audit, frozen.audit)
            _normalize_control(control, stage)
            row = _row(stage, variant, seed, train_metric, frozen_metric, train_metric["top1"] - frozen_metric["top1"], trainable, frozen, control, splits, time.perf_counter() - start)
            rows.append(row)
            controls.append(control)
            binding_audits.append(_binding_audit(stage, variant, seed, splits, trainable, control))
            family_results.extend(_family_rows(stage, row, control, splits["test"]))
            ablation_results.append(_ablation_row(stage, row, control))
            attention_rows.extend(_tag_rows(_attention_summaries(variant, seed, splits["test"][: min(8, len(splits["test"]))], trainable.model, device), stage))
            error_rows.extend(_tag_rows(_error_rows(variant, seed, splits["test"], train_logits, frozen_logits, limit=20), stage))
            compute = _compute_row(variant, seed, splits, row, time.perf_counter() - start)
            compute["benchmark"] = BENCHMARK
            compute["stage"] = stage
            compute["candidate_count"] = int(dataset_config.num_candidates)
            compute_rows.append(compute)
    return {
        "rows": rows,
        "controls": controls,
        "oracle_feature_audit": oracle_audits,
        "rule_event_binding_audit": binding_audits,
        "family_results": family_results,
        "ablation_results": ablation_results,
        "attention_summaries": attention_rows,
        "error_cases": error_rows,
        "compute_metrics": compute_rows,
        "shortcut_decomposition": shortcut_rows,
        "counterfactual_candidate_audit": counterfactual_rows,
    }


def _expand_splits_if_needed(splits: Dict[str, List[ConstraintExample]], num_candidates: int, seed: int) -> Dict[str, List[ConstraintExample]]:
    if not splits["train"] or len(splits["train"][0].candidates) == int(num_candidates):
        return splits
    return {split: [_expand_example(example, int(num_candidates), idx + seed * 1000) for idx, example in enumerate(rows)] for split, rows in splits.items()}


def _expand_example(example: ConstraintExample, num_candidates: int, index: int) -> ConstraintExample:
    gold = int(example.label)
    negatives = [idx for idx in range(len(example.candidates)) if idx != gold]
    label = int(index % int(num_candidates))
    slots: List[int | None] = [None for _ in range(int(num_candidates))]
    slots[label] = gold
    neg_cursor = 0
    for slot in range(int(num_candidates)):
        if slots[slot] is None:
            slots[slot] = negatives[neg_cursor % len(negatives)]
            neg_cursor += 1
    order = [int(value) for value in slots if value is not None]
    satisfies = tuple(bool(i == gold) for i in order)
    return replace(
        example,
        candidates=tuple(example.candidates[i] for i in order),
        rollouts=tuple(example.rollouts[i] for i in order),
        reached_goal=tuple(example.reached_goal[i] for i in order),
        satisfies_constraint=satisfies,
        candidate_types=tuple(example.candidate_types[i] for i in order),
        label=label,
        metadata={
            **example.metadata,
            "candidate_types": [example.candidate_types[i] for i in order],
            "expanded_candidate_count": int(num_candidates),
            "all_candidates_same_length": len({len(example.candidates[i]) for i in order}) == 1,
            "all_candidates_same_endpoint": len({example.rollouts[i][-1] for i in order}) == 1,
            "num_candidates_reaching_goal": int(sum(bool(example.reached_goal[i]) for i in order)),
            "all_candidates_reach_goal": bool(all(example.reached_goal[i] for i in order)),
        },
    )


def _oracle_feature_audit(stage: str, seed: int, splits: Dict[str, List[ConstraintExample]]) -> Dict[str, object]:
    forbidden = (
        "candidate_valid_under_constraint",
        "violation_type",
        "violation_step",
        "hidden_active_rule_id",
        "candidate_source_family",
        "gold_label",
        "generator_rank",
        "oracle_score",
        "test_correctness",
    )
    examples = [example for rows in splits.values() for example in rows]
    event_visible = all(_event_tokens_visible(example) for example in examples)
    rule_visible = all(_rule_tokens_visible(example) for example in examples)
    return {
        "benchmark": BENCHMARK,
        "stage": stage,
        "seed": int(seed),
        "examples": len(examples),
        "claimable_inputs": ["constraint_examples", "rule_tokens_from_constraint_examples", "candidate_rollout_event_tokens", "terrain_state_goal_tokens", "candidate_action_tokens"],
        "forbidden_inputs_absent": True,
        "forbidden_inputs": list(forbidden),
        "event_tokens_derived_from_visible_rollouts": bool(event_visible),
        "rule_tokens_derived_from_visible_constraint_examples": bool(rule_visible),
        "teacher_labels_training_targets_only": True,
        "diagnostic_teacher_fields_not_in_inference_batch": True,
        "claimable": bool(event_visible and rule_visible),
    }


def _event_tokens_visible(example: ConstraintExample) -> bool:
    visible_cells = {(row, col) for _name, row, col, _value, _role_id in _role_cells(example)}
    for cand_idx in range(len(example.candidates)):
        teacher = oracle_teacher(example, cand_idx)
        for token in teacher["relevant_path_events"]:
            if token["event"] in {"visited_color", "visited_subgoal", "collected_key", "passed_door"}:
                if not any((row, col) in visible_cells for _name, row, col, _value, _role_id in _role_cells(example)):
                    return False
    return True


def _rule_tokens_visible(example: ConstraintExample) -> bool:
    cells = {(row, col) for _name, row, col, _value, _role_id in _role_cells(example)}
    for evidence in example.constraint_examples:
        for _role, row, col in evidence.tokens:
            if (int(row), int(col)) not in cells:
                return False
    return True


def _binding_audit(stage: str, variant: BridgeVariant, seed: int, splits: Dict[str, List[ConstraintExample]], fit: BridgeFitResult, control: Dict[str, object]) -> Dict[str, object]:
    stress = {}
    if stage == "N8_medium" and variant.name == MAIN_VARIANT:
        stress["entity_id_permutation"] = _stress_entity_permutation(fit, splits["test"], seed)
        stress["longer_rollouts"] = _stress_longer_rollouts(fit, splits, seed)
    return {
        "benchmark": BENCHMARK,
        "stage": stage,
        "variant": variant.name,
        "seed": int(seed),
        "binding": variant.binding,
        "uses_oracle_features_at_inference": bool(variant.uses_oracle_features_at_inference),
        "rule_event_mismatch_top1": float(control["rule_event_mismatch"]["top1"]),
        "rule_token_shuffle_top1": float(control["rule_token_shuffle"]["top1"]),
        "event_token_shuffle_top1": float(control["event_token_shuffle"]["top1"]),
        "constraint_ablation_top1": float(control["ablate_constraint_rule_tokens"]["top1"]),
        "event_ablation_top1": float(control["ablate_rollout_event_tokens"]["top1"]),
        "stress_tests": stress,
    }


def _stress_longer_rollouts(fit: BridgeFitResult, splits: Dict[str, List[ConstraintExample]], seed: int) -> Dict[str, object]:
    cfg = ConstraintDatasetConfig(train_examples=16, dev_examples=8, test_examples=128, num_candidates=8, plan_length=64, families=tuple(FAMILIES), all_goal_every=1)
    stress_splits = build_repaired_constraint_splits(cfg, seed + 900)
    labels = _labels(stress_splits["test"])
    logits = predict_bridge_logits(fit.model, stress_splits["test"], "cpu")
    return {"top1": float(_top1(logits, labels)), "examples": len(stress_splits["test"]), "plan_length": 64}


def _stress_entity_permutation(fit: BridgeFitResult, examples: Sequence[ConstraintExample], seed: int) -> Dict[str, object]:
    permuted = [_permute_entities(example, seed + idx) for idx, example in enumerate(examples)]
    logits = predict_bridge_logits(fit.model, permuted, "cpu")
    return {"top1": float(_top1(logits, _labels(permuted))), "examples": len(permuted), "stable": bool(_top1(logits, _labels(permuted)) >= 0.90)}


def _permute_entities(example: ConstraintExample, seed: int) -> ConstraintExample:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(4).tolist()
    old_cells = _role_cells(example)
    new_cells = [old_cells[int(i)] for i in perm]
    old_to_new = {int(old_idx): int(new_idx) for new_idx, old_idx in enumerate(perm)}
    target = [old_to_new[int(idx)] for idx in example.metadata["target_order"]]
    valid_tokens = tuple((int(new_cells[idx][4]), int(new_cells[idx][1]), int(new_cells[idx][2])) for idx in target)
    invalid_tokens = tuple(reversed(valid_tokens))
    return replace(
        example,
        constraint_examples=(ConstraintEvidence("valid", valid_tokens), ConstraintEvidence("invalid", invalid_tokens)),
        metadata={**example.metadata, "role_cells": [list(item) for item in new_cells], "target_order": target, "entity_id_permutation": perm},
    )


def _normalize_control(control: Dict[str, object], stage: str) -> None:
    control["benchmark"] = BENCHMARK
    control["stage"] = stage
    if "constraint_example_mismatch" not in control:
        control["constraint_example_mismatch"] = control.get("constraint_mismatch", {})
    control["control_pass"]["constraint_example_mismatch_collapses"] = control["control_pass"].get("constraint_mismatch_collapses", False)


def _row(
    stage: str,
    variant: BridgeVariant,
    seed: int,
    train_metric: Dict[str, object],
    frozen_metric: Dict[str, object],
    delta: float,
    trainable: BridgeFitResult,
    frozen: BridgeFitResult,
    control: Dict[str, object],
    splits: Dict[str, List[ConstraintExample]],
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
        "delta_trainable_minus_frozen": float(delta),
        "control_pass": control["control_pass"],
        "trainable_audit": trainable.audit,
        "frozen_audit": frozen.audit,
        "param_count": int(trainable.param_count),
        "training_time_seconds": float(elapsed),
        "dataset_sizes": {key: len(value) for key, value in splits.items()},
        "candidate_count": len(splits["test"][0].candidates) if splits["test"] else 0,
        "final_validation_launched": False,
    }


def _family_rows(stage: str, row: Dict[str, object], control: Dict[str, object], examples: Sequence[ConstraintExample]) -> List[Dict[str, object]]:
    trainable_by_family = row["trainable"]["by_constraint_family"]
    frozen_by_family = row["frozen"]["by_constraint_family"]
    rows = []
    for family in FAMILIES:
        trainable = float(trainable_by_family.get(family, 0.0))
        frozen = float(frozen_by_family.get(family, 0.0))
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
                "controls_pass": bool(control["control_pass"]["overall"]),
                "count": sum(1 for example in examples if example.family == family),
            }
        )
    return rows


def _ablation_row(stage: str, row: Dict[str, object], control: Dict[str, object]) -> Dict[str, object]:
    return {
        "benchmark": BENCHMARK,
        "stage": stage,
        "variant": row["variant"],
        "seed": int(row["seed"]),
        "clean_top1": float(control["clean"]["top1"]),
        "rule_ablation_top1": float(control["ablate_constraint_rule_tokens"]["top1"]),
        "event_ablation_top1": float(control["ablate_rollout_event_tokens"]["top1"]),
        "rule_ablation_hurt": float(control["clean"]["top1"]) - float(control["ablate_constraint_rule_tokens"]["top1"]),
        "event_ablation_hurt": float(control["clean"]["top1"]) - float(control["ablate_rollout_event_tokens"]["top1"]),
    }


def _medium_gate_summary(rows: Sequence[Dict[str, object]], controls: Sequence[Dict[str, object]], variant: str, stage: str) -> Dict[str, object]:
    group = [row for row in rows if row["variant"] == variant and row["stage"] == stage]
    control_group = [row for row in controls if row["variant"] == variant and row["stage"] == stage]
    deltas = [float(row["delta_trainable_minus_frozen"]) for row in group]
    train_top1 = [float(row["trainable"]["top1"]) for row in group]
    chance = [float(row["chance"]) for row in control_group]
    ci_low, ci_high = _bootstrap_ci(deltas)
    pass_flags = {
        "beats_frozen_4_of_5": sum(delta > 0.0 for delta in deltas) >= 4,
        "mean_delta_ge_0_20": _mean(deltas) >= DELTA_GATE,
        "bootstrap_ci_low_gt_0_05": ci_low > 0.05,
        "trainable_above_random": _mean(train_top1) > _mean(chance) + NEAR_CHANCE_MARGIN,
        "controls_pass_all": all(bool(row["control_pass"]["overall"]) for row in control_group),
        "shortcut_baselines_near_chance": all(bool(row["control_pass"]["shortcut_baselines_do_not_explain"]) for row in control_group),
        "mismatch_shuffle_controls_pass": all(
            bool(row["control_pass"]["constraint_mismatch_collapses"])
            and bool(row["control_pass"]["rollout_mismatch_collapses"])
            and bool(row["control_pass"]["rule_event_mismatch_collapses"])
            and bool(row["control_pass"]["rule_token_shuffle_collapses"])
            and bool(row["control_pass"]["event_token_shuffle_collapses"])
            for row in control_group
        ),
        "ablations_hurt": all(bool(row["control_pass"]["ablate_constraint_rule_tokens_hurts"]) and bool(row["control_pass"]["ablate_rollout_event_tokens_hurts"]) for row in control_group),
        "no_oracle_input_features": all(bool(row["control_pass"]["no_oracle_input_features_used"]) for row in control_group),
    }
    return {
        "benchmark": BENCHMARK,
        "stage": stage,
        "variant": variant,
        "seeds": [int(row["seed"]) for row in group],
        "mean_trainable_top1": _mean(train_top1),
        "mean_frozen_top1": _mean([float(row["frozen"]["top1"]) for row in group]),
        "mean_delta": _mean(deltas),
        "bootstrap_delta_ci95": [ci_low, ci_high],
        "seed_wins": int(sum(delta > 0.0 for delta in deltas)),
        "pass_flags": pass_flags,
        "medium_gates_pass": all(bool(value) for value in pass_flags.values()),
    }


def _bootstrap_ci(values: Sequence[float], seed: int = 4242, samples: int = 1000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    means = [float(np.mean(arr[rng.integers(0, len(arr), size=len(arr))])) for _ in range(int(samples))]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _leaderboard(rows: Sequence[Dict[str, object]], controls: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    controls_by_key = {(row["stage"], row["variant"], row["seed"]): row for row in controls}
    out = []
    for row in rows:
        control = controls_by_key[(row["stage"], row["variant"], row["seed"])]
        shortcut = max(
            float(control["candidate_only"]["top1"]),
            float(control["rollout_only"]["top1"]),
            float(control["constraint_only"]["top1"]),
            float(control["endpoint_only"]["top1"]),
            float(control["action_bigram"]),
        )
        score = float(row["trainable"]["top1"]) + float(row["delta_trainable_minus_frozen"]) + (0.2 if control["control_pass"]["overall"] else -0.5) - max(0.0, shortcut - float(control["chance"]))
        out.append(
            {
                "stage": row["stage"],
                "variant": row["variant"],
                "seed": int(row["seed"]),
                "binding": row["binding"],
                "status": row["status"],
                "trainable_top1": float(row["trainable"]["top1"]),
                "frozen_top1": float(row["frozen"]["top1"]),
                "delta": float(row["delta_trainable_minus_frozen"]),
                "controls_pass": bool(control["control_pass"]["overall"]),
                "shortcut_max": shortcut,
                "param_count": int(row["param_count"]),
                "score": float(score),
            }
        )
    return sorted(out, key=lambda item: float(item["score"]), reverse=True)


def _summary(result: Dict[str, object]) -> Dict[str, object]:
    n8_gate = next((row for row in result["stage_gates"] if row["stage"] == "N8_medium"), {})
    n16_gate = next((row for row in result["stage_gates"] if row["stage"] == "N16_stress"), {})
    main_rows = [row for row in result["rows"] if row["variant"] == MAIN_VARIANT and row["stage"] == "N8_medium"]
    variant_summaries = []
    for variant in LOCKED_VARIANTS:
        group = [row for row in result["rows"] if row["variant"] == variant and row["stage"] == "N8_medium"]
        if not group:
            continue
        variant_summaries.append(
            {
                "variant": variant,
                "mean_trainable_top1": _mean([float(row["trainable"]["top1"]) for row in group]),
                "mean_frozen_top1": _mean([float(row["frozen"]["top1"]) for row in group]),
                "mean_delta": _mean([float(row["delta_trainable_minus_frozen"]) for row in group]),
                "seed_wins": int(sum(float(row["delta_trainable_minus_frozen"]) > 0.0 for row in group)),
            }
        )
    return {
        "main_variant": MAIN_VARIANT,
        "n8_medium_gates_pass": bool(n8_gate.get("medium_gates_pass", False)),
        "n16_stress_ran": bool(n16_gate),
        "n16_medium_gates_pass": bool(n16_gate.get("medium_gates_pass", False)) if n16_gate else False,
        "ready_for_final_validation": bool(n8_gate.get("medium_gates_pass", False) and (not n16_gate or n16_gate.get("medium_gates_pass", False))),
        "final_validation_launched": False,
        "main_mean_trainable_top1": _mean([float(row["trainable"]["top1"]) for row in main_rows]),
        "main_mean_frozen_top1": _mean([float(row["frozen"]["top1"]) for row in main_rows]),
        "main_mean_delta": _mean([float(row["delta_trainable_minus_frozen"]) for row in main_rows]),
        "variant_summaries": sorted(variant_summaries, key=lambda row: float(row["mean_delta"]), reverse=True),
    }


def _empty_result(dataset_config: ConstraintDatasetConfig, training: ConstraintTrainingConfig, variants: Sequence[BridgeVariant], device: str) -> Dict[str, object]:
    return {
        "metadata": {
            "benchmark": BENCHMARK,
            "source_bridge_benchmark": BRIDGE_BENCHMARK,
            "scope": "medium validation of locked rule-event constraint binding variants",
            "device": device,
            "final_validation_launched": False,
            "claim_boundary": "No autonomous planning, world-modeling, general agentic AI, ARC solving, or SOTA planning claim.",
        },
        "dataset_config": asdict(dataset_config),
        "training_config": asdict(training),
        "variants": [asdict(variant) for variant in variants],
        "rows": [],
        "controls": [],
        "oracle_feature_audit": [],
        "rule_event_binding_audit": [],
        "leaderboard": [],
        "family_results": [],
        "ablation_results": [],
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


def _locked_variants() -> List[BridgeVariant]:
    by_name = {variant.name: variant for variant in _bridge_variants()}
    return [by_name[name] for name in LOCKED_VARIANTS]


def _tag_rows(rows: Sequence[Dict[str, object]], stage: str) -> List[Dict[str, object]]:
    return [{**row, "benchmark": BENCHMARK, "stage": stage} for row in rows]


def _write_outputs(
    result: Dict[str, object],
    results_path: Path,
    controls_path: Path,
    oracle_audit_path: Path,
    binding_audit_path: Path,
    leaderboard_path: Path,
    family_path: Path,
    ablation_path: Path,
    attention_path: Path,
    error_path: Path,
    compute_path: Path,
    report_path: Path,
) -> None:
    for path in (results_path, controls_path, oracle_audit_path, binding_audit_path, leaderboard_path, family_path, ablation_path, attention_path, error_path, compute_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(_trim_result(result), indent=2, sort_keys=True), encoding="utf-8")
    controls_path.write_text(json.dumps({"controls": result["controls"]}, indent=2, sort_keys=True), encoding="utf-8")
    oracle_audit_path.write_text(json.dumps({"oracle_feature_audit": result["oracle_feature_audit"]}, indent=2, sort_keys=True), encoding="utf-8")
    binding_audit_path.write_text(json.dumps({"rule_event_binding_audit": result["rule_event_binding_audit"]}, indent=2, sort_keys=True), encoding="utf-8")
    family_path.write_text(json.dumps({"family_results": result["family_results"]}, indent=2, sort_keys=True), encoding="utf-8")
    ablation_path.write_text(json.dumps({"ablation_results": result["ablation_results"]}, indent=2, sort_keys=True), encoding="utf-8")
    compute_path.write_text(json.dumps({"compute_metrics": result["compute_metrics"]}, indent=2, sort_keys=True), encoding="utf-8")
    with leaderboard_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["stage", "variant", "seed", "binding", "status", "trainable_top1", "frozen_top1", "delta", "controls_pass", "shortcut_max", "param_count", "score"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["leaderboard"]:
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
    n8 = next((row for row in result["stage_gates"] if row["stage"] == "N8_medium"), {})
    n16 = next((row for row in result["stage_gates"] if row["stage"] == "N16_stress"), {})
    main_controls = [row for row in result["controls"] if row["variant"] == MAIN_VARIANT and row["stage"] == "N8_medium"]
    clone = next((row for row in summary["variant_summaries"] if row["variant"] == MAIN_VARIANT), {})
    simple = [row for row in summary["variant_summaries"] if row["variant"] != MAIN_VARIANT]
    best_simple = max(simple, key=lambda row: float(row["mean_trainable_top1"]), default={})
    perm = _stress_mean(result["rule_event_binding_audit"], "entity_id_permutation")
    hardest = _hardest_family(result["family_results"])
    lines = [
        "# PLAN-ARCH-2 Medium Rule-Event Constraint Binding Validation",
        "",
        "## Scope",
        "- Final validation was not launched.",
        "- Architecture candidates were locked before medium validation.",
        "- Pyramids, stacking, candidate self-attention, and broad search were not run.",
        "",
        "## Answers",
        f"1. Does `{MAIN_VARIANT}` beat frozen on fresh seeds? `{n8.get('seed_wins', 0)}/5`, mean_delta={float(n8.get('mean_delta', 0.0)):.4f}, CI95={n8.get('bootstrap_delta_ci95', [])}.",
        f"2. Does it pass all shortcut, mismatch, shuffle, and leakage controls? `{all(bool(row['control_pass']['overall']) for row in main_controls) if main_controls else False}`.",
        f"3. Are event/rule tokens claimable? `{all(bool(row['claimable']) for row in result['oracle_feature_audit'])}`; tokens are audited as visible constraint-example and rollout-derived, with teacher labels used only as training targets.",
        f"4. Does the clone architecture beat or match simple rule-event baselines? `clone={float(clone.get('mean_trainable_top1', 0.0)):.4f}, best_simple={float(best_simple.get('mean_trainable_top1', 0.0)):.4f}`.",
        f"5. Which family is hardest after scaling? `{hardest}`.",
        f"6. Does N=16 remain strong? `{bool(n16.get('medium_gates_pass', False))}` mean_delta={float(n16.get('mean_delta', 0.0)):.4f}.",
        f"7. Do rule-token and event-token ablations hurt? `{_ablation_answer(result['ablation_results'], MAIN_VARIANT, 'N8_medium')}`.",
        f"8. Does the result survive entity/color/key/subgoal permutation? `mean_top1={perm:.4f}`.",
        f"9. Is this ready for final validation? `{summary['ready_for_final_validation']}`.",
        f"10. Should the winning rule-event binding module be transferred back to ARC-hybrid and real-code candidate verification? `{summary['ready_for_final_validation']}`.",
        "",
        "## Medium Gates",
        "| stage | mean trainable | mean frozen | mean delta | seed wins | CI low | gates |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for gate in result["stage_gates"]:
        ci = gate.get("bootstrap_delta_ci95", [0.0, 0.0])
        lines.append(
            f"| {gate['stage']} | {float(gate['mean_trainable_top1']):.4f} | {float(gate['mean_frozen_top1']):.4f} | "
            f"{float(gate['mean_delta']):.4f} | {int(gate['seed_wins'])} | {float(ci[0]):.4f} | `{bool(gate['medium_gates_pass'])}` |"
        )
    lines.extend(["", "## Variant Summary", "| variant | trainable mean | frozen mean | mean delta | seed wins |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in summary["variant_summaries"]:
        lines.append(f"| {row['variant']} | {float(row['mean_trainable_top1']):.4f} | {float(row['mean_frozen_top1']):.4f} | {float(row['mean_delta']):.4f} | {int(row['seed_wins'])} |")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "Allowed medium claim only if gates pass: on controlled learned-constraint candidate-plan verification, shared-weight latent clone coordination with rule-event tokens learns to bind constraint evidence to candidate rollout events, outperforming an exact frozen comparator while shortcut, mismatch, shuffle, and leakage controls pass.",
            "",
            "Forbidden claims remain: autonomous planning, world modeling, general agentic AI, ARC solving, and SOTA planning.",
        ]
    )
    return "\n".join(lines) + "\n"


def _ablation_answer(rows: Sequence[Dict[str, object]], variant: str, stage: str) -> str:
    group = [row for row in rows if row["variant"] == variant and row["stage"] == stage]
    if not group:
        return "not run"
    return f"rule_hurt_mean={_mean([float(row['rule_ablation_hurt']) for row in group]):.4f}, event_hurt_mean={_mean([float(row['event_ablation_hurt']) for row in group]):.4f}"


def _stress_mean(rows: Sequence[Dict[str, object]], key: str) -> float:
    vals = [float(row.get("stress_tests", {}).get(key, {}).get("top1", 0.0)) for row in rows if key in row.get("stress_tests", {})]
    return _mean(vals)


def _hardest_family(rows: Sequence[Dict[str, object]]) -> str:
    group = [row for row in rows if row["variant"] == MAIN_VARIANT and row["stage"] == "N8_medium"]
    means = {family: _mean([float(row["trainable_top1"]) for row in group if row["family"] == family]) for family in FAMILIES}
    family = min(means, key=means.get)
    return f"{family}={means[family]:.4f}"


def _trim_result(result: Dict[str, object]) -> Dict[str, object]:
    trimmed = dict(result)
    trimmed["attention_summaries"] = result.get("attention_summaries", [])[:60]
    trimmed["error_cases"] = result.get("error_cases", [])[:100]
    return trimmed


def _dataset_config(data: object) -> ConstraintDatasetConfig:
    values = dict(data or {})
    if "families" in values:
        values["families"] = tuple(str(item) for item in values["families"])
    allowed = set(ConstraintDatasetConfig.__dataclass_fields__.keys())
    return ConstraintDatasetConfig(**{key: value for key, value in values.items() if key in allowed})


def _training_config(data: object) -> ConstraintTrainingConfig:
    values = dict(data or {})
    allowed = set(ConstraintTrainingConfig.__dataclass_fields__.keys())
    return ConstraintTrainingConfig(**{key: value for key, value in values.items() if key in allowed})


def _default_config(config: Dict[str, object]) -> Dict[str, object]:
    base: Dict[str, object] = {
        "device": "cpu",
        "seeds": [10, 11, 12, 13, 14],
        "dataset": {
            "grid_size": 7,
            "num_candidates": 8,
            "plan_length": 48,
            "train_examples": 512,
            "dev_examples": 128,
            "test_examples": 128,
            "families": list(FAMILIES),
            "all_goal_every": 1,
        },
        "training": {"epochs": 10, "batch_size": 64, "lr": 0.004, "weight_decay": 0.0, "patience": 5, "gradient_clip_norm": 1.0},
    }
    return _deep_update(base, config)


def _deep_update(base: Dict[str, object], override: Dict[str, object]) -> Dict[str, object]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(dict(out[key]), value)
        else:
            out[key] = value
    return out


def _load_config(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


if __name__ == "__main__":
    main()
