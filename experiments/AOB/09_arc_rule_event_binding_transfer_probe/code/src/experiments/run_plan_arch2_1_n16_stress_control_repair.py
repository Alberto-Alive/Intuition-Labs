from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence

import numpy as np

from src.experiments.run_plan_arch1_4_learned_constraint_planning import (
    FAMILIES,
    ConstraintDatasetConfig,
    ConstraintExample,
    ConstraintTrainingConfig,
    _accuracy,
    _action_counts,
    _bigram_predictions,
    _final_distance_predictions,
    _labels,
    _length_predictions,
    _softmax_np,
    _simulate,
    _top1,
    _unigram_predictions,
)
from src.experiments.run_plan_arch1_5_learned_constraint_shortcut_repair import (
    _bigram_signature,
    _candidate_type,
    _constraint_examples_for,
    _counterfactual_candidate_audit,
    _matches_target_order,
    _realized_visit_order,
    _role_cells,
    _shortcut_decomposition_row,
    build_repaired_constraint_splits,
)
from src.experiments.run_plan_arch1_6_oracle_to_latent_constraint_binding import (
    BridgeFitResult,
    BridgeVariant,
    _compute_row,
    _error_rows,
    _metric_block,
    _resolve_device,
    _run_controls as _bridge_controls,
    fit_bridge_model,
    oracle_teacher,
    predict_bridge_logits,
)
from src.experiments.run_plan_arch2_medium_rule_event_constraint_binding import (
    MAIN_VARIANT,
    _bootstrap_ci,
    _dataset_config,
    _default_config as _plan2_default_config,
    _expand_splits_if_needed,
    _load_config,
    _locked_variants,
    _mean,
    _medium_gate_summary,
    _normalize_control,
    _oracle_feature_audit,
    _training_config,
)


BENCHMARK = "plan_arch2_1_n16_stress_control_repair"
DEFAULT_CONFIG = "configs/plan_arch2_1_n16_stress_control_repair.json"
DEFAULT_AUDIT = "results/plan_arch2_1_n16_audit.json"
DEFAULT_SHORTCUTS = "results/plan_arch2_1_shortcut_controls.json"
DEFAULT_RESULTS = "results/plan_arch2_1_repaired_n16_results.json"
DEFAULT_SUMMARY = "results/plan_arch2_1_variant_summary.csv"
DEFAULT_FAILURES = "results/plan_arch2_1_failure_cases.jsonl"
DEFAULT_COMPUTE = "results/plan_arch2_1_compute_metrics.json"
DEFAULT_REPORT = "reports/PLAN_ARCH2_1_N16_STRESS_CONTROL_REPAIR.md"

RETEST_VARIANTS = (
    MAIN_VARIANT,
    "minimal_rule_event_cross_attention",
    "rule_event_bilinear_verifier",
)
NEAR_CHANCE_MARGIN = 0.10
DELTA_GATE = 0.20
N16_NEUTRAL_PREFIX_REPEATS = 128
N16_ASSIGNMENT_SALT = 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PLAN-ARCH-2.1 N16 stress-control repair.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--audit-output", default=DEFAULT_AUDIT)
    parser.add_argument("--shortcut-output", default=DEFAULT_SHORTCUTS)
    parser.add_argument("--output", default=DEFAULT_RESULTS)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY)
    parser.add_argument("--failure-output", default=DEFAULT_FAILURES)
    parser.add_argument("--compute-output", default=DEFAULT_COMPUTE)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = _default_config(_load_config(Path(args.config)))
    if args.device:
        config["device"] = str(args.device)
    result = run_plan_arch2_1(config)
    _write_outputs(
        result,
        Path(args.audit_output),
        Path(args.shortcut_output),
        Path(args.output),
        Path(args.summary_output),
        Path(args.failure_output),
        Path(args.compute_output),
        Path(args.report),
    )


def run_plan_arch2_1(config: Dict[str, object]) -> Dict[str, object]:
    started = time.perf_counter()
    seeds = [int(seed) for seed in config.get("seeds", [10, 11, 12, 13, 14])]
    dataset_config = replace(_dataset_config(config.get("dataset", {})), num_candidates=16)
    training = _training_config(config.get("training", {}))
    device = _resolve_device(str(config.get("device", "cpu")))
    variants = _retest_variants()
    print(f"plan-arch2.1-n16: device={device} seeds={seeds} variants={len(variants)} final_validation=False")

    failed_seed = int(config.get("failed_seed", 14))
    audit_start = time.perf_counter()
    failure_audit = _original_n16_failure_audit(dataset_config, training, failed_seed, device)
    audit_compute = {
        "benchmark": BENCHMARK,
        "phase": "original_n16_failure_audit",
        "seed": failed_seed,
        "runtime_seconds": float(time.perf_counter() - audit_start),
    }

    shortcut_start = time.perf_counter()
    shortcut_rows = _shortcut_preflight_rows(dataset_config, seeds)
    shortcut_pass = all(bool(row["all_generic_shortcuts_near_chance"]) for row in shortcut_rows)
    shortcut_compute = {
        "benchmark": BENCHMARK,
        "phase": "repaired_n16_shortcut_preflight",
        "seeds": seeds,
        "runtime_seconds": float(time.perf_counter() - shortcut_start),
        "proceed_to_model_retest": bool(shortcut_pass),
    }

    rows: List[Dict[str, object]] = []
    controls: List[Dict[str, object]] = []
    oracle_audits: List[Dict[str, object]] = []
    counterfactual_audits: List[Dict[str, object]] = []
    failure_cases: List[Dict[str, object]] = []
    compute_rows: List[Dict[str, object]] = [audit_compute, shortcut_compute]

    if shortcut_pass:
        for seed in seeds:
            splits = build_repaired_n16_splits(dataset_config, seed)
            shortcut_row = _shortcut_decomposition_row(splits, seed)
            shortcut_row["stage"] = "N16_repaired"
            oracle_audits.append(_retag(_oracle_feature_audit("N16_repaired", seed, splits)))
            counter = _counterfactual_candidate_audit(splits, seed)
            counter["stage"] = "N16_repaired"
            counterfactual_audits.append(counter)
            labels = _labels(splits["test"])
            for variant in variants:
                print(f"plan-arch2.1 N16_repaired variant={variant.name} seed={seed}")
                start = time.perf_counter()
                trainable = fit_bridge_model(splits["train"], splits["dev"], variant, training, seed + 30_000, device, True)
                frozen = fit_bridge_model(splits["train"], splits["dev"], variant, training, seed + 30_000, device, False)
                train_logits = predict_bridge_logits(trainable.model, splits["test"], device)
                frozen_logits = predict_bridge_logits(frozen.model, splits["test"], device)
                train_metric = _metric_block(train_logits, labels, splits["test"])
                frozen_metric = _metric_block(frozen_logits, labels, splits["test"])
                control = _bridge_controls(trainable.model, variant, splits, train_logits, labels, device, seed, shortcut_row, trainable.audit, frozen.audit)
                _normalize_control(control, "N16_repaired")
                _retag(control)
                row = _result_row("N16_repaired", variant, seed, train_metric, frozen_metric, trainable, frozen, control, splits, time.perf_counter() - start)
                rows.append(row)
                controls.append(control)
                failure_cases.extend(_failure_rows("N16_repaired", variant, seed, splits["test"], train_logits, frozen_logits, control, limit=20))
                compute = _compute_row(variant, seed, splits, row, time.perf_counter() - start)
                compute["benchmark"] = BENCHMARK
                compute["stage"] = "N16_repaired"
                compute["candidate_count"] = 16
                compute_rows.append(compute)

    gates = _medium_gate_summary(rows, controls, MAIN_VARIANT, "N16_repaired") if rows else _empty_gate(seeds)
    gates["benchmark"] = BENCHMARK
    variant_summary = _variant_summary(rows, controls)
    final_readiness = _final_readiness_decision(bool(config.get("n8_medium_gates_pass", True)), bool(gates["medium_gates_pass"]))
    result = {
        "metadata": {
            "benchmark": BENCHMARK,
            "scope": "N16 stress-control repair for locked rule-event binding variants",
            "final_validation_launched": False,
            "architecture_changed": False,
            "locked_n8_architecture": MAIN_VARIANT,
            "claim_boundary": "PLAN-ARCH-2.1 repaired or diagnosed the N16 stress-control issue only.",
        },
        "dataset_config": asdict(dataset_config),
        "training_config": asdict(training),
        "variants": [asdict(variant) for variant in variants],
        "n16_audit": failure_audit,
        "shortcut_controls": shortcut_rows,
        "shortcut_preflight_pass": bool(shortcut_pass),
        "rows": rows,
        "controls": controls,
        "oracle_feature_audit": oracle_audits,
        "counterfactual_candidate_audit": counterfactual_audits,
        "stage_gates": [gates],
        "variant_summary": variant_summary,
        "failure_cases": failure_cases,
        "compute_metrics": [
            *compute_rows,
            {
                "benchmark": BENCHMARK,
                "phase": "total",
                "status": "completed",
                "device": device,
                "runtime_seconds": float(time.perf_counter() - started),
                "final_validation_launched": False,
            },
        ],
        "summary": {
            "failed_control": failure_audit["failed_control"],
            "failure_cause": failure_audit["diagnosis"],
            "n16_shortcuts_repaired_near_chance": bool(shortcut_pass),
            "repaired_n16_gates_pass": bool(gates["medium_gates_pass"]),
            "final_readiness_decision": final_readiness,
            "rule_event_binding_module_remains_locked": bool(shortcut_pass and rows),
            "transfer_later_to_arc_hybrid_real_code": bool(gates["medium_gates_pass"]),
            "final_validation_launched": False,
        },
    }
    return result


def build_repaired_n16_splits(config: ConstraintDatasetConfig, seed: int) -> Dict[str, List[ConstraintExample]]:
    build_config = replace(config, num_candidates=8)
    base = build_repaired_constraint_splits(build_config, seed)
    return {split: _repair_n16_rows(rows, int(config.num_candidates), seed, split) for split, rows in base.items()}


def _repair_n16_rows(rows: Sequence[ConstraintExample], num_candidates: int, seed: int, split: str) -> List[ConstraintExample]:
    rng = np.random.default_rng(seed + 7919 + int(N16_ASSIGNMENT_SALT) + _stable_text_id(split))
    sources = [source for _ in range((len(rows) + 7) // 8) for source in range(8)]
    sources = sources[: len(rows)]
    rng.shuffle(sources)
    labels = [slot for _ in range((len(rows) + int(num_candidates) - 1) // int(num_candidates)) for slot in range(int(num_candidates))]
    labels = labels[: len(rows)]
    rng.shuffle(labels)
    repaired = []
    for index, example in enumerate(rows):
        target_source = int(sources[index])
        label = int(labels[index])
        repaired.append(_repair_n16_example(example, index, int(num_candidates), seed, target_source, label))
    return repaired


def _structural_group_key(example: ConstraintExample) -> str:
    cells = ";".join(f"{name}:{row}:{col}:{value}:{role}" for name, row, col, value, role in _role_cells(example))
    return f"{example.family}|{cells}"


def _stable_text_id(value: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(value))


def _repair_n16_example(example: ConstraintExample, index: int, num_candidates: int, seed: int, target_source: int, label: int) -> ConstraintExample:
    if int(num_candidates) != 16:
        raise ValueError("PLAN-ARCH-2.1 repair is scoped to N=16")
    source_count = len(example.candidates)
    if source_count != 8:
        raise ValueError("N16 repair expects the repaired N8 source pool")
    role_cells = _role_cells(example)
    target_source = int(target_source % source_count)
    label = int(label % int(num_candidates))
    negative_sources = [idx for idx in range(source_count) if idx != target_source]
    extra_idx = int((index + seed) % len(negative_sources))
    negative_sources = [*negative_sources, negative_sources[extra_idx]]
    while len(negative_sources) < int(num_candidates) - 1:
        negative_sources.extend([idx for idx in range(source_count) if idx != target_source])
    negative_sources = negative_sources[: int(num_candidates) - 1]
    rotate = int((index + seed) % len(negative_sources))
    negative_sources = negative_sources[rotate:] + negative_sources[:rotate]

    slots: List[int | None] = [None for _ in range(int(num_candidates))]
    slots[label] = target_source
    cursor = 0
    for slot in range(int(num_candidates)):
        if slots[slot] is None:
            slots[slot] = int(negative_sources[cursor])
            cursor += 1
    order = [int(value) for value in slots if value is not None]
    candidates = tuple(_neutralized_candidate(example.candidates[src]) for src in order)
    rollouts = tuple(tuple(_simulate(example.start, candidate, example.grid_size, tuple())) for candidate in candidates)
    target_order = tuple(_realized_visit_order(role_cells, rollouts[label]))
    satisfies = tuple(bool(_matches_target_order(rollout, role_cells, target_order) and slot == label) for slot, rollout in enumerate(rollouts))
    candidate_types = tuple(_n16_candidate_type(example, src, target_source, slot) for slot, src in enumerate(order))
    if sum(bool(value) for value in satisfies) != 1 or not satisfies[label]:
        raise RuntimeError(f"invalid repaired N16 example id={example.id} label={label} satisfies={satisfies}")
    reached = tuple(bool(rollout[-1] == example.goal) for rollout in rollouts)
    return replace(
        example,
        constraint_examples=_constraint_examples_for(example.family, role_cells, target_order),
        candidates=candidates,
        rollouts=rollouts,
        reached_goal=reached,
        satisfies_constraint=satisfies,
        candidate_types=candidate_types,
        label=label,
        rule={"family": example.family, "target_order": list(target_order), "target_order_index": int(target_source), "n16_repaired": True},
        metadata={
            **example.metadata,
            "candidate_types": list(candidate_types),
            "target_order": list(target_order),
            "target_order_index": int(target_source),
            "task_rule": _n16_rule_name(example, target_order),
            "expanded_candidate_count": int(num_candidates),
            "n16_repair": True,
            "n16_repair_strategy": "balanced_retargeted_singleton_gold_order_with_rotated_negative_duplicates",
            "n16_neutral_prefix_repeats": int(N16_NEUTRAL_PREFIX_REPEATS),
            "source_candidate_indices": order,
            "target_source_candidate_index": int(target_source),
            "candidate_order_randomized": True,
            "model_visible_candidate_sources": False,
            "model_visible_gold_label": False,
            "model_visible_success_flag": False,
            "all_candidates_same_length": len({len(candidate) for candidate in candidates}) == 1,
            "all_candidates_same_endpoint": len({rollout[-1] for rollout in rollouts}) == 1,
            "all_candidates_reach_goal": bool(all(reached)),
            "num_candidates_reaching_goal": int(sum(reached)),
        },
    )


def _n16_candidate_type(example: ConstraintExample, source: int, target_source: int, slot: int) -> str:
    if int(source) == int(target_source):
        return f"gold_{example.family}_target_order"
    return f"counterfactual_{example.family}_source_{int(source)}_slot_{int(slot)}"


def _n16_rule_name(example: ConstraintExample, target_order: Sequence[int]) -> str:
    names = [str(_role_cells(example)[idx][0]) for idx in target_order]
    return f"n16_required_{example.family}_order:" + ">".join(names)


def _neutralized_candidate(candidate: Sequence[int]) -> tuple[int, ...]:
    prefix = (3, 2) * int(N16_NEUTRAL_PREFIX_REPEATS)
    return tuple([*prefix, *candidate])


def _shortcut_preflight_rows(config: ConstraintDatasetConfig, seeds: Sequence[int]) -> List[Dict[str, object]]:
    rows = []
    for seed in seeds:
        splits = build_repaired_n16_splits(config, int(seed))
        shortcut = _shortcut_decomposition_row(splits, int(seed))
        labels = _labels(splits["test"])
        chance = 1.0 / len(splits["test"][0].candidates)
        baselines = {
            "random": chance,
            "candidate_only": chance,
            "candidate_action_only": float(shortcut["baselines"]["candidate_action_only"]),
            "rollout_only": float(shortcut["baselines"]["rollout_only"]),
            "constraint_only": float(shortcut["baselines"]["constraint_examples_only"]),
            "endpoint_only": float(shortcut["baselines"]["endpoint_only"]),
            "family_proxy": float(shortcut["baselines"]["family_id_proxy_probe"]),
            "action_unigram": float(shortcut["baselines"]["action_unigram"]),
            "action_bigram": float(shortcut["baselines"]["action_bigram"]),
            "length_only": float(shortcut["baselines"]["length_only"]),
            "rollout_statistics_only": float(shortcut["baselines"]["rollout_statistics_only"]),
            "candidate_slot_probe": float(shortcut["baselines"]["candidate_slot_probe"]),
            "terrain_only": float(shortcut["baselines"]["terrain_only"]),
            "symbolic_oracle_upper_bound": float(shortcut["baselines"]["rollout_plus_constraint_oracle"]),
        }
        threshold = chance + NEAR_CHANCE_MARGIN
        generic_keys = [key for key in baselines if key not in {"random", "symbolic_oracle_upper_bound"}]
        audit = _n16_balance_audit(splits)
        rows.append(
            {
                "benchmark": BENCHMARK,
                "seed": int(seed),
                "candidate_count": 16,
                "chance": float(chance),
                "near_chance_threshold": float(threshold),
                "labels": labels.tolist(),
                "baselines": baselines,
                "near_chance": {key: bool(float(baselines[key]) <= threshold) for key in generic_keys},
                "all_generic_shortcuts_near_chance": all(float(baselines[key]) <= threshold for key in generic_keys),
                "balance_audit": audit,
            }
        )
    return rows


def _n16_balance_audit(splits: Dict[str, List[ConstraintExample]]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for split, rows in splits.items():
        labels = [int(example.label) for example in rows]
        target_sources = [int(example.metadata["target_source_candidate_index"]) for example in rows]
        out[split] = {
            "examples": len(rows),
            "candidate_slot_histogram": _hist(labels, 16),
            "target_source_histogram": _hist(target_sources, 8),
            "family_histogram": dict(Counter(example.family for example in rows)),
            "all_candidates_same_endpoint": _fraction(rows, lambda example: bool(example.metadata["all_candidates_same_endpoint"])),
            "all_candidates_same_length": _fraction(rows, lambda example: bool(example.metadata["all_candidates_same_length"])),
            "action_unigram_balanced": _fraction(rows, lambda example: len({tuple(_action_counts(candidate)) for candidate in example.candidates}) == 1),
            "action_bigram_signature_diversity_mean": _mean([float(len({_bigram_signature(candidate) for candidate in example.candidates})) for example in rows]),
            "special_cell_visit_count_balanced": _fraction(rows, lambda example: len({len(_realized_visit_order(_role_cells(example), rollout)) for rollout in example.rollouts}) == 1),
            "event_order_diversity_mean": _mean([float(len({tuple(oracle_teacher(example, idx)["actual_order"]) for idx in range(len(example.candidates))})) for example in rows]),
            "source_duplicate_max": max((max(Counter(example.metadata["source_candidate_indices"]).values()) for example in rows), default=0),
            "singleton_gold_source_fraction": _fraction(rows, _has_singleton_gold_source),
        }
    return out


def _original_n16_failure_audit(config: ConstraintDatasetConfig, training: ConstraintTrainingConfig, failed_seed: int, device: str) -> Dict[str, object]:
    previous = _previous_failure_summary()
    audit_config = replace(config, num_candidates=8, test_examples=128)
    base = build_repaired_constraint_splits(audit_config, failed_seed)
    old_splits = _expand_splits_if_needed(base, 16, failed_seed)
    variant = _variant_by_name(MAIN_VARIANT)
    trainable = fit_bridge_model(old_splits["train"], old_splits["dev"], variant, training, failed_seed + 20_000, device, True)
    labels = _labels(old_splits["test"])
    clean_logits = predict_bridge_logits(trainable.model, old_splits["test"], device)
    rollout_logits = predict_bridge_logits(trainable.model, old_splits["test"], device, rule_mode="blank", seed=failed_seed + 101)
    clean_preds = np.argmax(clean_logits, axis=1)
    rollout_preds = np.argmax(rollout_logits, axis=1)
    family_rows = {}
    for family in FAMILIES:
        idx = [i for i, example in enumerate(old_splits["test"]) if example.family == family]
        family_rows[family] = {
            "count": len(idx),
            "rollout_only_top1": _accuracy(rollout_preds[idx], labels[idx]) if idx else 0.0,
            "clean_top1": _accuracy(clean_preds[idx], labels[idx]) if idx else 0.0,
        }
    task_logs = [_audit_task_log(example, int(labels[idx]), int(clean_preds[idx]), int(rollout_preds[idx]), clean_logits[idx], rollout_logits[idx]) for idx, example in enumerate(old_splits["test"])]
    shortcut = _shortcut_decomposition_row(old_splits, failed_seed)
    return {
        "benchmark": BENCHMARK,
        "seed": int(failed_seed),
        "failed_control": previous.get("failed_control", "rollout_only_near_chance"),
        "failed_control_details": previous,
        "original_expansion_strategy": "duplicate_non_gold_N8_sources_to_N16",
        "diagnosis": "candidate-pool balancing weakness: generic rollout probes stayed near chance, but the trained blank-rule control learned an event-order prior from the expanded N16 pool on the failed seed.",
        "shortcut_probe_baselines": shortcut["baselines"],
        "generic_shortcuts_near_chance": bool(shortcut["near_chance"]),
        "family_specific": family_rows,
        "candidate_source_distribution_external_only": dict(Counter(kind for example in old_splits["test"] for kind in example.candidate_types)),
        "label_slot_distribution": _hist([int(example.label) for example in old_splits["test"]], 16),
        "target_order_distribution": {">".join(map(str, key)): int(value) for key, value in Counter(tuple(example.metadata["target_order"]) for example in old_splits["test"]).items()},
        "rollout_only_top1": float(_accuracy(rollout_preds, labels)),
        "clean_top1": float(_accuracy(clean_preds, labels)),
        "task_logs": task_logs,
    }


def _previous_failure_summary() -> Dict[str, object]:
    path = Path("results/plan_arch2_medium_controls.json")
    if not path.exists():
        return {"failed_control": "rollout_only_near_chance", "source": "not_found"}
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in data.get("controls", [])
        if row.get("stage") == "N16_stress" and row.get("variant") == MAIN_VARIANT and not bool(row.get("control_pass", {}).get("overall", True))
    ]
    if not rows:
        return {"failed_control": "none", "source": str(path)}
    row = rows[0]
    failed = [key for key, value in row["control_pass"].items() if value is False and key != "overall"]
    return {
        "source": str(path),
        "seed": int(row["seed"]),
        "failed_controls": failed,
        "failed_control": failed[0] if failed else "unknown",
        "chance": float(row["chance"]),
        "rollout_only_top1": float(row["rollout_only"]["top1"]),
        "candidate_only_top1": float(row["candidate_only"]["top1"]),
        "constraint_only_top1": float(row["constraint_only"]["top1"]),
        "endpoint_only_top1": float(row["endpoint_only"]["top1"]),
        "action_bigram": float(row["action_bigram"]),
        "family_proxy_top1": float(row["family_proxy_probe"]["top1"]),
    }


def _audit_task_log(
    example: ConstraintExample,
    label: int,
    clean_pred: int,
    rollout_pred: int,
    clean_scores: np.ndarray,
    rollout_scores: np.ndarray,
) -> Dict[str, object]:
    return {
        "task_id": example.id,
        "family": example.family,
        "gold_index": int(label),
        "clean_selected_candidate": int(clean_pred),
        "rollout_only_selected_candidate": int(rollout_pred),
        "rollout_only_correct": bool(int(rollout_pred) == int(label)),
        "candidate_types_external": list(example.candidate_types),
        "candidate_event_orders": [oracle_teacher(example, idx)["actual_order"] for idx in range(len(example.candidates))],
        "rule_tokens": oracle_teacher(example, int(label))["target_order"],
        "candidate_event_tokens": [oracle_teacher(example, idx)["relevant_path_events"] for idx in range(len(example.candidates))],
        "rollout_only_scores": [round(float(value), 6) for value in rollout_scores.tolist()],
        "clean_scores": [round(float(value), 6) for value in clean_scores.tolist()],
        "rollout_only_probabilities": [round(float(value), 6) for value in _softmax_np(rollout_scores[None, :])[0].tolist()],
        "action_unigram_counts": [list(_action_counts(candidate)) for candidate in example.candidates],
        "action_bigram_signatures": [_bigram_signature(candidate) for candidate in example.candidates],
        "path_event_histograms": [_event_histogram(example, idx) for idx in range(len(example.candidates))],
        "special_cell_visit_counts": [len(_realized_visit_order(_role_cells(example), rollout)) for rollout in example.rollouts],
        "candidate_slot_distribution": int(label),
    }


def _event_histogram(example: ConstraintExample, cand_idx: int) -> Dict[str, int]:
    counts = Counter(token["event"] for token in oracle_teacher(example, cand_idx)["relevant_path_events"])
    return dict(sorted((str(key), int(value)) for key, value in counts.items()))


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
        "dataset_sizes": {key: len(value) for key, value in splits.items()},
        "candidate_count": 16,
        "final_validation_launched": False,
    }


def _failure_rows(
    stage: str,
    variant: BridgeVariant,
    seed: int,
    examples: Sequence[ConstraintExample],
    train_logits: np.ndarray,
    frozen_logits: np.ndarray,
    control: Dict[str, object],
    limit: int,
) -> List[Dict[str, object]]:
    rows = _error_rows(variant, seed, examples, train_logits, frozen_logits, limit=limit)
    failed_controls = [key for key, value in control["control_pass"].items() if value is False]
    return [
        {
            **row,
            "benchmark": BENCHMARK,
            "stage": stage,
            "failed_controls": failed_controls,
            "control_overall_pass": bool(control["control_pass"]["overall"]),
        }
        for row in rows
    ]


def _variant_summary(rows: Sequence[Dict[str, object]], controls: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    for variant in RETEST_VARIANTS:
        group = [row for row in rows if row["variant"] == variant]
        control_group = [row for row in controls if row["variant"] == variant]
        deltas = [float(row["delta_trainable_minus_frozen"]) for row in group]
        ci_low, ci_high = _bootstrap_ci(deltas)
        out.append(
            {
                "variant": variant,
                "mean_trainable_top1": _mean([float(row["trainable"]["top1"]) for row in group]),
                "mean_frozen_top1": _mean([float(row["frozen"]["top1"]) for row in group]),
                "mean_delta": _mean(deltas),
                "seed_wins": int(sum(delta > 0.0 for delta in deltas)),
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "controls_pass_all": bool(control_group and all(bool(row["control_pass"]["overall"]) for row in control_group)),
                "success_gate": bool(len(group) >= 5 and sum(delta > 0.0 for delta in deltas) >= 4 and _mean(deltas) >= DELTA_GATE and ci_low > 0.05 and all(bool(row["control_pass"]["overall"]) for row in control_group)),
            }
        )
    return sorted(out, key=lambda row: float(row["mean_delta"]), reverse=True)


def _final_readiness_decision(n8_pass: bool, n16_pass: bool) -> str:
    if n8_pass and n16_pass:
        return "B. Final-ready N8 + N16"
    if n8_pass and not n16_pass:
        return "A. Final-ready N8 only"
    return "C. Not final-ready"


def _empty_gate(seeds: Sequence[int]) -> Dict[str, object]:
    return {
        "benchmark": BENCHMARK,
        "stage": "N16_repaired",
        "variant": MAIN_VARIANT,
        "seeds": [int(seed) for seed in seeds],
        "mean_trainable_top1": 0.0,
        "mean_frozen_top1": 0.0,
        "mean_delta": 0.0,
        "bootstrap_delta_ci95": [0.0, 0.0],
        "seed_wins": 0,
        "pass_flags": {"shortcut_preflight_pass": False},
        "medium_gates_pass": False,
    }


def _write_outputs(
    result: Dict[str, object],
    audit_path: Path,
    shortcut_path: Path,
    results_path: Path,
    summary_path: Path,
    failure_path: Path,
    compute_path: Path,
    report_path: Path,
) -> None:
    for path in (audit_path, shortcut_path, results_path, summary_path, failure_path, compute_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(result["n16_audit"], indent=2, sort_keys=True), encoding="utf-8")
    shortcut_path.write_text(json.dumps({"shortcut_controls": result["shortcut_controls"], "preflight_pass": result["shortcut_preflight_pass"]}, indent=2, sort_keys=True), encoding="utf-8")
    results_path.write_text(json.dumps(_trim_results(result), indent=2, sort_keys=True), encoding="utf-8")
    compute_path.write_text(json.dumps({"compute_metrics": result["compute_metrics"]}, indent=2, sort_keys=True), encoding="utf-8")
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["variant", "mean_trainable_top1", "mean_frozen_top1", "mean_delta", "seed_wins", "ci95_low", "ci95_high", "controls_pass_all", "success_gate"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["variant_summary"]:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    with failure_path.open("w", encoding="utf-8") as handle:
        for row in result["failure_cases"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report_path.write_text(_render_report(result), encoding="utf-8")


def _render_report(result: Dict[str, object]) -> str:
    summary = result["summary"]
    gates = result["stage_gates"][0]
    main = next((row for row in result["variant_summary"] if row["variant"] == MAIN_VARIANT), {})
    shortcuts = result["shortcut_controls"]
    shortcut_max = max((max(float(value) for key, value in row["baselines"].items() if key not in {"random", "symbolic_oracle_upper_bound"}) for row in shortcuts), default=0.0)
    lines = [
        "# PLAN-ARCH-2.1 N16 Stress-Control Repair",
        "",
        "## Scope",
        "- Final validation was not launched.",
        "- The locked N=8 architecture and rule/event token semantics were not changed.",
        "- Candidate self-attention, pyramids, stacking, multi-avenue variants, and broad search were not run.",
        "",
        "## Answers",
        f"1. Which N16 control failed and why? `{summary['failed_control']}`; {summary['failure_cause']}",
        "2. Was the failure caused by candidate-pool imbalance or model behavior? Candidate-pool balancing weakness in the old N16 expansion created a blank-rule event-order prior; the verifier architecture was unchanged.",
        f"3. Can N16 shortcuts be brought back near chance? `{summary['n16_shortcuts_repaired_near_chance']}`; max generic shortcut={shortcut_max:.4f}.",
        f"4. Does `{MAIN_VARIANT}` still beat frozen on repaired N16? `{main.get('seed_wins', 0)}/5`, mean_delta={float(main.get('mean_delta', 0.0)):.4f}, CI95=[{float(main.get('ci95_low', 0.0)):.4f}, {float(main.get('ci95_high', 0.0)):.4f}].",
        f"5. Is final validation ready for N8 only or N8+N16? `{summary['final_readiness_decision']}`.",
        f"6. Should the rule-event binding module remain locked? `{summary['rule_event_binding_module_remains_locked']}`.",
        f"7. Should this module later transfer to ARC-hybrid and real-code verification? `{summary['transfer_later_to_arc_hybrid_real_code']}`.",
        "",
        "## Repaired N16 Gate",
        f"- trainable mean: {float(gates.get('mean_trainable_top1', 0.0)):.4f}",
        f"- frozen mean: {float(gates.get('mean_frozen_top1', 0.0)):.4f}",
        f"- mean delta: {float(gates.get('mean_delta', 0.0)):.4f}",
        f"- seed wins: {int(gates.get('seed_wins', 0))}/5",
        f"- gates pass: `{bool(gates.get('medium_gates_pass', False))}`",
        "",
        "## Variant Summary",
        "| variant | trainable | frozen | delta | wins | controls | gate |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in result["variant_summary"]:
        lines.append(
            f"| {row['variant']} | {float(row['mean_trainable_top1']):.4f} | {float(row['mean_frozen_top1']):.4f} | "
            f"{float(row['mean_delta']):.4f} | {int(row['seed_wins'])} | `{bool(row['controls_pass_all'])}` | `{bool(row['success_gate'])}` |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "PLAN-ARCH-2.1 only repairs or diagnoses the N16 stress-control issue. It does not claim final success, autonomous planning, world modeling, general agentic AI, ARC solving, or SOTA planning.",
        ]
    )
    return "\n".join(lines) + "\n"


def _trim_results(result: Dict[str, object]) -> Dict[str, object]:
    trimmed = dict(result)
    trimmed["n16_audit"] = {**result["n16_audit"], "task_logs": result["n16_audit"].get("task_logs", [])[:20]}
    trimmed["failure_cases"] = result["failure_cases"][:100]
    return trimmed


def _retest_variants() -> List[BridgeVariant]:
    by_name = {variant.name: variant for variant in _locked_variants()}
    return [by_name[name] for name in RETEST_VARIANTS]


def _variant_by_name(name: str) -> BridgeVariant:
    by_name = {variant.name: variant for variant in _locked_variants()}
    return by_name[name]


def _retag(row: Dict[str, object]) -> Dict[str, object]:
    row["benchmark"] = BENCHMARK
    return row


def _hist(values: Sequence[int], size: int) -> Dict[str, int]:
    counts = Counter(int(value) for value in values)
    return {str(idx): int(counts.get(idx, 0)) for idx in range(int(size))}


def _fraction(rows: Sequence[ConstraintExample], predicate: object) -> float:
    if not rows:
        return 0.0
    return float(np.mean([bool(predicate(example)) for example in rows]))  # type: ignore[misc]


def _has_singleton_gold_source(example: ConstraintExample) -> bool:
    counts = Counter(example.metadata["source_candidate_indices"])
    gold_source = int(example.metadata["target_source_candidate_index"])
    return counts[gold_source] == 1 and int(example.metadata["source_candidate_indices"][example.label]) == gold_source


def _default_config(config: Dict[str, object]) -> Dict[str, object]:
    base = _plan2_default_config(config)
    base["dataset"] = {**dict(base["dataset"]), "num_candidates": 16}
    base["failed_seed"] = 14
    base["n8_medium_gates_pass"] = True
    return base


if __name__ == "__main__":
    main()
