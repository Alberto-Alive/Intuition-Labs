from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


BENCHMARK = "plan_arch1_empirical_architecture_discovery"
STAGE_NAME = "PLAN-ARCH-1_EMPIRICAL_ARCHITECTURE_DISCOVERY"
LOCKED_BASELINE = "P1_rollout_no_final"
DEFAULT_CONFIG = "configs/plan_arch1_empirical_architecture_discovery_smoke.json"
DEFAULT_RESULTS = "results/plan_arch1_results.json"
DEFAULT_CONTROLS = "results/plan_arch1_controls.json"
DEFAULT_LEADERBOARD = "results/plan_arch1_variant_leaderboard.csv"
DEFAULT_FAILURES = "results/plan_arch1_failure_taxonomy.json"
DEFAULT_OVERFIT = "results/plan_arch1_overfit_curves.json"
DEFAULT_ABLATIONS = "results/plan_arch1_ablation_results.json"
DEFAULT_ATTENTION = "results/plan_arch1_attention_summaries.jsonl"
DEFAULT_ERRORS = "results/plan_arch1_error_cases.jsonl"
DEFAULT_COMPUTE = "results/plan_arch1_compute_metrics.json"
DEFAULT_REPORT = "reports/PLAN_ARCH1_EMPIRICAL_ARCHITECTURE_DISCOVERY.md"

FIELD_VOCABS = (16, 16, 16, 8, 16, 16, 16, 16)
ROLE_IDS = {
    "summary": 0,
    "state": 1,
    "goal": 2,
    "obstacle": 3,
    "action": 4,
    "rollout": 5,
    "transition": 6,
    "risk": 7,
    "progress": 8,
    "counterfactual": 9,
    "transition": 10,
}
ACTIONS = ("U", "D", "L", "R")
ACTION_DELTAS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}


@dataclass(frozen=True)
class PlanDatasetConfig:
    grid_size: int = 8
    num_candidates: int = 8
    plan_length: int = 12
    obstacle_count: int = 8
    train_examples: int = 128
    dev_examples: int = 48
    test_examples: int = 48
    max_generation_attempts: int = 200
    require_exactly_one_success: bool = True
    shortcut_repaired_candidate_pools: bool = False
    shortcut_pool_attempts: int = 1600
    learnability_level: int = -1


@dataclass(frozen=True)
class PlanTrainingConfig:
    epochs: int = 8
    batch_size: int = 32
    lr: float = 0.0007
    weight_decay: float = 0.0001
    patience: int = 3
    gradient_clip_norm: float = 1.0
    objective: str = "cross_entropy"
    margin: float = 0.25
    aux_weight: float = 0.25


@dataclass(frozen=True)
class FeatureTrainingConfig:
    epochs: int = 20
    batch_size: int = 32
    lr: float = 0.003
    weight_decay: float = 0.0001
    patience: int = 4
    hidden_dim: int = 64


@dataclass(frozen=True)
class PlanVariant:
    name: str
    family: str
    description: str
    view_names: Tuple[str, ...]
    representation: str = "hybrid_sparse_dense"
    query_mechanism: str = "candidate_token_direct_attention"
    objective: str = "cross_entropy"
    model_dim: int = 48
    num_heads: int = 4
    ff_dim: int = 96
    evidence_layers: int = 1
    candidate_layers: int = 1
    coordination_blocks: int = 1
    candidate_self_attention: bool = False
    evidence_self_attention: bool = False
    bidirectional: bool = False
    pyramid: bool = False
    residual_level_scoring: bool = False
    avenues_per_view: int = 1
    include_final_state: bool = False
    include_outcome_tokens: bool = False
    diagnostic_only: bool = False


@dataclass(frozen=True)
class PlanExample:
    id: str
    split: str
    grid_size: int
    start: Tuple[int, int]
    goal: Tuple[int, int]
    obstacles: Tuple[Tuple[int, int], ...]
    candidates: Tuple[Tuple[int, ...], ...]
    rollouts: Tuple[Tuple[Tuple[int, int], ...], ...]
    collisions: Tuple[Tuple[int, ...], ...]
    goal_reached: Tuple[bool, ...]
    progress: Tuple[Tuple[int, ...], ...]
    label: int
    metadata: Dict[str, object]


@dataclass
class PlanFitResult:
    method: str
    model: "PlanLatentVerifier"
    variant: PlanVariant
    trainable_shared: bool
    audit: Dict[str, object]
    history: List[Dict[str, float]]
    training_time_seconds: float
    param_count: int


@dataclass
class FeatureFitResult:
    method: str
    model: "PlanFeatureMLP"
    audit: Dict[str, object]
    history: List[Dict[str, float]]
    training_time_seconds: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PLAN-ARCH-1 bounded architecture search.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_RESULTS)
    parser.add_argument("--controls-output", default=DEFAULT_CONTROLS)
    parser.add_argument("--leaderboard-output", default=DEFAULT_LEADERBOARD)
    parser.add_argument("--failure-output", default=DEFAULT_FAILURES)
    parser.add_argument("--overfit-output", default=DEFAULT_OVERFIT)
    parser.add_argument("--ablation-output", default=DEFAULT_ABLATIONS)
    parser.add_argument("--attention-output", default=DEFAULT_ATTENTION)
    parser.add_argument("--error-output", default=DEFAULT_ERRORS)
    parser.add_argument("--compute-output", default=DEFAULT_COMPUTE)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-variants", type=int, default=0)
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    if args.device:
        config["device"] = str(args.device)
    result = run_plan_arch1_search(config, max_variants=int(args.max_variants))
    _write_outputs(
        result,
        Path(args.output),
        Path(args.controls_output),
        Path(args.leaderboard_output),
        Path(args.failure_output),
        Path(args.overfit_output),
        Path(args.ablation_output),
        Path(args.attention_output),
        Path(args.error_output),
        Path(args.compute_output),
        Path(args.report),
    )


def run_plan_arch1_search(config: Dict[str, object], max_variants: int = 0) -> Dict[str, object]:
    dataset_config = _dataset_config(config.get("dataset", {}))
    cheap_dataset_config = _dataset_config({**asdict(dataset_config), **dict(config.get("cheap_dataset", {}))})
    medium_dataset_config = _dataset_config({**asdict(dataset_config), **dict(config.get("medium_dataset", {}))})
    training = _training_config(config.get("training", {}))
    overfit_training = _training_config({**dict(config.get("training", {})), **dict(config.get("overfit_training", {}))})
    feature_training = _feature_training_config(config.get("feature_training", {}))
    cheap_seeds = [int(seed) for seed in config.get("cheap_seeds", config.get("seeds", [0]))]
    medium_seeds = [int(seed) for seed in config.get("medium_seeds", [10, 11, 12, 13, 14])]
    run_medium = bool(config.get("run_medium", True))
    run_overfit = bool(config.get("run_overfit_gates", True))
    max_medium_variants = int(config.get("max_medium_variants", 3))
    device = _resolve_device(str(config.get("device", "auto")))
    variants = _variant_plan(config.get("variants"))
    if max_variants:
        variants = variants[:max_variants]
    if int(config.get("max_variants", 0)):
        variants = variants[: int(config.get("max_variants", 0))]
    micro_gates = _micro_gates(config.get("micro_overfit_gates"))

    rows: List[Dict[str, object]] = []
    controls: List[Dict[str, object]] = []
    overfit_rows: List[Dict[str, object]] = []
    ablation_rows: List[Dict[str, object]] = []
    attention_rows: List[Dict[str, object]] = []
    error_rows: List[Dict[str, object]] = []
    compute_rows: List[Dict[str, object]] = []
    phase0_rows: List[Dict[str, object]] = []

    print(f"plan-arch1: device={device} variants={len(variants)} cheap_seeds={cheap_seeds}")
    baseline_cache: Dict[int, float] = {}
    split_cache: Dict[Tuple[str, int, str], Dict[str, List[PlanExample]]] = {}
    for variant in variants:
        for seed in cheap_seeds:
            print(f"plan-arch1 cheap variant={variant.name} seed={seed}: phase0/overfit")
            cache_key = ("cheap", int(seed), json.dumps(asdict(cheap_dataset_config), sort_keys=True))
            if cache_key not in split_cache:
                split_cache[cache_key] = build_plan_splits(cheap_dataset_config, seed=seed)
            splits = split_cache[cache_key]
            phase0_rows.append(_phase0_sanity(variant, splits, seed))
            gate_rows = (
                _run_overfit_gates(variant, dataset_config, overfit_training, seed, device, micro_gates)
                if run_overfit and not variant.diagnostic_only
                else [{"variant": variant.name, "seed": seed, "gate": "skipped", "pass": True, "gate_required": False}]
            )
            overfit_rows.extend(gate_rows)
            gates_pass = all(bool(row.get("pass")) for row in gate_rows if row.get("gate_required", True))
            row_base = _row_base("cheap", variant, seed, gates_pass)
            if not gates_pass:
                rows.append({**row_base, "status": "failed_overfit_gate", "failure_reason": _first_failed_gate(gate_rows)})
                continue
            row, control_row, ablations, attn, errors, compute = _fit_and_evaluate_variant(
                phase="cheap",
                variant=variant,
                splits=splits,
                training=replace(training, objective=variant.objective),
                feature_training=feature_training,
                seed=seed,
                device=device,
                baseline_cache=baseline_cache,
            )
            rows.append(row)
            controls.append(control_row)
            ablation_rows.extend(ablations)
            attention_rows.extend(attn)
            error_rows.extend(errors)
            compute_rows.append(compute)
            _clear_cuda()

    cheap_summary = _phase_summary(rows, controls, phase="cheap", min_seed_wins=2, min_delta=0.15)
    medium_rows: List[Dict[str, object]] = []
    medium_controls: List[Dict[str, object]] = []
    medium_summary: Dict[str, object] = {"ran": False, "reason": "no cheap variant passed gates"}
    top_variants = _select_medium_variants(cheap_summary, variants, max_medium_variants)
    if run_medium and top_variants:
        print(f"plan-arch1 medium: selected {[variant.name for variant in top_variants]}")
        for variant in top_variants:
            for seed in medium_seeds:
                cache_key = ("medium", int(seed), json.dumps(asdict(medium_dataset_config), sort_keys=True))
                if cache_key not in split_cache:
                    split_cache[cache_key] = build_plan_splits(medium_dataset_config, seed=seed)
                splits = split_cache[cache_key]
                row, control_row, ablations, attn, errors, compute = _fit_and_evaluate_variant(
                    phase="medium",
                    variant=variant,
                    splits=splits,
                    training=replace(training, objective=variant.objective),
                    feature_training=feature_training,
                    seed=seed,
                    device=device,
                    baseline_cache={},
                )
                rows.append(row)
                controls.append(control_row)
                medium_rows.append(row)
                medium_controls.append(control_row)
                ablation_rows.extend(ablations)
                attention_rows.extend(attn)
                error_rows.extend(errors)
                compute_rows.append(compute)
                _clear_cuda()
        medium_summary = _phase_summary(rows, controls, phase="medium", min_seed_wins=4, min_delta=0.15)
        medium_summary["ran"] = True
    elif not run_medium:
        medium_summary = {"ran": False, "reason": "disabled by config"}

    leaderboard = _leaderboard(rows, controls)
    failure_taxonomy = _failure_taxonomy(rows, controls, overfit_rows)
    summary = _overall_summary(rows, controls, leaderboard, cheap_summary, medium_summary, ablation_rows)
    return {
        "metadata": {
            "benchmark": BENCHMARK,
            "stage_name": STAGE_NAME,
            "created_at_utc": _now(),
            "device": device,
            "scope": "bounded architecture discovery for latent candidate-plan evaluation; not final validation",
            "locked_baseline": LOCKED_BASELINE,
            "reference_motivation": {
                "P1_rollout_no_final_trainable_mean_top1": 0.2812,
                "P1_rollout_no_final_frozen_mean_top1": 0.1198,
                "P1_rollout_no_final_delta": 0.1615,
                "P1_rollout_no_final_controls": True,
            },
            "claim_boundary": "No autonomous planning, world-modeling, general agentic AI, or final validation claim.",
            "final_validation_launched": False,
        },
        "dataset_config": asdict(dataset_config),
        "cheap_dataset_config": asdict(cheap_dataset_config),
        "medium_dataset_config": asdict(medium_dataset_config),
        "training_config": asdict(training),
        "feature_training_config": asdict(feature_training),
        "variant_plan": [asdict(variant) for variant in variants],
        "phase0_sanity": phase0_rows,
        "rows": rows,
        "controls": controls,
        "leaderboard": leaderboard,
        "failure_taxonomy": failure_taxonomy,
        "overfit_curves": overfit_rows,
        "ablation_results": ablation_rows,
        "attention_summaries": attention_rows,
        "error_cases": error_rows,
        "compute_metrics": compute_rows,
        "cheap_summary": cheap_summary,
        "medium_summary": medium_summary,
        "summary": summary,
    }


def _fit_and_evaluate_variant(
    phase: str,
    variant: PlanVariant,
    splits: Dict[str, List[PlanExample]],
    training: PlanTrainingConfig,
    feature_training: FeatureTrainingConfig,
    seed: int,
    device: str,
    baseline_cache: Dict[int, float],
) -> Tuple[Dict[str, object], Dict[str, object], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    print(f"plan-arch1 {phase} variant={variant.name} seed={seed}: trainable/frozen")
    start = time.perf_counter()
    trainable = fit_plan_verifier(splits["train"], splits["dev"], variant, training, seed + 10_001, device, True, f"{phase}_trainable__{variant.name}")
    frozen = fit_plan_verifier(splits["train"], splits["dev"], variant, training, seed + 10_001, device, False, f"{phase}_frozen__{variant.name}")
    train_logits = predict_plan_logits(trainable.model, splits["test"], variant, training.batch_size, device)
    frozen_logits = predict_plan_logits(frozen.model, splits["test"], variant, training.batch_size, device)
    labels = _labels(splits["test"])
    train_metric = _metric_block(train_logits, labels, splits["test"])
    frozen_metric = _metric_block(frozen_logits, labels, splits["test"])
    baselines = _baseline_block(splits["train"], splits["dev"], splits["test"], feature_training, seed)
    if variant.name == LOCKED_BASELINE:
        baseline_cache[seed] = float(train_metric["top1"])
    p1_reference = baseline_cache.get(seed, _find_p1_reference(phase, seed))
    control_row = _run_controls(trainable.model, variant, splits["test"], train_logits, labels, training.batch_size, device, seed)
    control_row["phase"] = phase
    ablations = _run_ablations(trainable.model, variant, splits["test"], train_logits, labels, training.batch_size, device, seed)
    attention = _attention_summaries(trainable.model, variant, splits["test"][: min(8, len(splits["test"]))], training.batch_size, device, seed)
    errors = _error_rows(phase, variant, seed, splits["test"], train_logits, frozen_logits, baselines, limit=20)
    elapsed = time.perf_counter() - start
    row = {
        **_row_base(phase, variant, seed, True),
        "status": "completed",
        "trainable": train_metric,
        "frozen": frozen_metric,
        "delta_trainable_minus_frozen": float(train_metric["top1"] - frozen_metric["top1"]),
        "delta_vs_locked_p1_reference": float(train_metric["top1"] - p1_reference) if p1_reference is not None else None,
        "locked_p1_reference_top1": p1_reference,
        "baselines": baselines,
        "control_pass": control_row["control_pass"],
        "trainable_audit": trainable.audit,
        "frozen_audit": frozen.audit,
        "training_time_seconds": float(elapsed),
        "param_count": int(trainable.param_count),
        "final_validation_launched": False,
    }
    compute = {
        "benchmark": BENCHMARK,
        "phase": phase,
        "seed": seed,
        "variant": variant.name,
        "candidate_count": len(splits["test"][0].candidates) if splits["test"] else 0,
        "view_count": len(_expanded_views(variant)),
        "coordination_blocks": int(variant.coordination_blocks),
        "train_examples": len(splits["train"]),
        "dev_examples": len(splits["dev"]),
        "test_examples": len(splits["test"]),
        "training_time_seconds": float(elapsed),
        "trainable_param_count": int(trainable.param_count),
        "frozen_param_count": int(frozen.param_count),
        "approx_attention_cost": int(len(splits["test"])) * int(len(_expanded_views(variant))) * int(variant.coordination_blocks) * int(len(splits["test"][0].candidates) if splits["test"] else 0),
    }
    return row, control_row, ablations, attention, errors, compute


def build_plan_splits(config: PlanDatasetConfig, seed: int) -> Dict[str, List[PlanExample]]:
    rng = np.random.default_rng(seed)
    counts = {"train": config.train_examples, "dev": config.dev_examples, "test": config.test_examples}
    splits: Dict[str, List[PlanExample]] = {}
    offset = 0
    for split, count in counts.items():
        rows = []
        for index in range(int(count)):
            rows.append(_generate_example(config, split, index, int(rng.integers(0, 2_000_000_000)) + offset))
        splits[split] = rows
        offset += 100_000
    return splits


def _generate_example(config: PlanDatasetConfig, split: str, index: int, seed: int) -> PlanExample:
    if int(config.learnability_level) >= 0:
        return _generate_ladder_example(config, split, index, seed)
    if bool(config.shortcut_repaired_candidate_pools):
        return _generate_shortcut_repaired_example(config, split, index, seed)

    rng = np.random.default_rng(seed)
    for attempt in range(int(config.max_generation_attempts)):
        grid = int(config.grid_size)
        start = (int(rng.integers(0, grid)), int(rng.integers(0, grid)))
        goal = (int(rng.integers(0, grid)), int(rng.integers(0, grid)))
        if start == goal:
            continue
        obstacles = _sample_obstacles(grid, start, goal, int(config.obstacle_count), rng)
        path = _bfs_path(grid, start, goal, obstacles)
        if path is None or len(path) - 1 > int(config.plan_length):
            continue
        if (int(config.plan_length) - (len(path) - 1)) % 2 != 0:
            continue
        gold = _pad_plan(_path_to_actions(path), int(config.plan_length), grid, goal, obstacles)
        candidates = [tuple(gold)]
        seen = {tuple(gold)}
        attempts = 0
        while len(candidates) < int(config.num_candidates) and attempts < 300:
            attempts += 1
            cand = tuple(_make_distractor(gold, grid, start, goal, obstacles, rng, attempts))
            if cand in seen:
                continue
            rollout, collisions, reached, progress = _simulate(grid, start, goal, obstacles, cand)
            if bool(reached) and bool(config.require_exactly_one_success):
                continue
            seen.add(cand)
            candidates.append(cand)
        if len(candidates) < int(config.num_candidates):
            continue
        order = rng.permutation(len(candidates))
        candidates_shuffled = tuple(candidates[int(i)] for i in order)
        label = int(np.where(order == 0)[0][0])
        rollouts = []
        collisions_all = []
        reached_all = []
        progress_all = []
        for cand in candidates_shuffled:
            rollout, collisions, reached, progress = _simulate(grid, start, goal, obstacles, cand)
            rollouts.append(tuple(rollout))
            collisions_all.append(tuple(collisions))
            reached_all.append(bool(reached))
            progress_all.append(tuple(progress))
        metadata = {
            "candidate_order_randomized": True,
            "model_visible_candidate_sources": False,
            "model_visible_generator_rank": False,
            "model_visible_gold_label": False,
            "model_visible_simulator_success_flag": False,
            "explicit_final_state_tokens_in_locked_baseline": False,
            "explicit_outcome_tokens_in_locked_baseline": False,
            "candidate_lengths": [len(c) for c in candidates_shuffled],
            "candidate_action_counts": [_action_counts(c) for c in candidates_shuffled],
            "offline_goal_reached": list(reached_all),
            "offline_collision_counts": [int(sum(c)) for c in collisions_all],
            "obstacle_count": len(obstacles),
            "manhattan_start_goal": _manhattan(start, goal),
        }
        return PlanExample(
            id=f"{split}_{index}",
            split=split,
            grid_size=grid,
            start=start,
            goal=goal,
            obstacles=tuple(sorted(obstacles)),
            candidates=candidates_shuffled,
            rollouts=tuple(rollouts),
            collisions=tuple(collisions_all),
            goal_reached=tuple(reached_all),
            progress=tuple(progress_all),
            label=label,
            metadata=metadata,
        )
    raise RuntimeError(f"could not generate gridworld example after {config.max_generation_attempts} attempts")


def _generate_ladder_example(config: PlanDatasetConfig, split: str, index: int, seed: int) -> PlanExample:
    rng = np.random.default_rng(seed)
    level = int(config.learnability_level)
    for _attempt in range(int(config.max_generation_attempts)):
        grid = int(config.grid_size)
        start = (int(rng.integers(0, grid)), int(rng.integers(0, grid)))
        goal = (int(rng.integers(0, grid)), int(rng.integers(0, grid)))
        if start == goal:
            continue
        obstacles = _sample_obstacles(grid, start, goal, int(config.obstacle_count), rng)
        gold = _sample_ladder_gold(level, grid, start, goal, obstacles, int(config.plan_length), rng)
        if gold is None:
            continue
        gold_rollout, gold_collisions, gold_reached, gold_progress = _simulate(grid, start, goal, obstacles, gold)
        if not gold_reached or sum(gold_collisions):
            continue
        gold_signature = _candidate_shortcut_signature(start, goal, gold, gold_collisions, gold_progress)
        negative_pool: List[Tuple[float, Tuple[int, ...], str]] = []
        seen = {tuple(gold)}
        for _pool_attempt in range(int(config.shortcut_pool_attempts)):
            cand = _sample_ladder_negative(level, grid, start, goal, obstacles, int(config.plan_length), rng)
            if cand is None or tuple(cand) in seen:
                continue
            rollout, collisions, reached, progress = _simulate(grid, start, goal, obstacles, cand)
            if bool(reached) and bool(config.require_exactly_one_success):
                continue
            if level in {2, 3, 4, 5} and sum(collisions):
                continue
            if level in {0, 1} and _manhattan(rollout[-1], goal) < 3 and sum(collisions) == 0:
                continue
            if level == 2 and _manhattan(rollout[-1], goal) < 1:
                continue
            if level == 3 and not (1 <= _manhattan(rollout[-1], goal) <= 2):
                continue
            if level in {4, 5} and _manhattan(rollout[-2], goal) != 1:
                continue
            signature = _candidate_shortcut_signature(start, goal, cand, collisions, progress)
            score = _candidate_balance_distance(gold_signature, signature)
            negative_pool.append((score, tuple(cand), f"ladder_level_{level}_balanced_path"))
            seen.add(tuple(cand))
        if len(negative_pool) < int(config.num_candidates) - 1:
            continue
        negatives = _select_balanced_negatives(negative_pool, int(config.num_candidates) - 1, rng)
        candidate_records = [(tuple(gold), f"ladder_level_{level}_balanced_path"), *[(cand, family) for cand, family in negatives]]
        ordered_records, label = _balanced_candidate_order(candidate_records, int(config.num_candidates), index, rng)
        candidates_shuffled = tuple(cand for cand, _family in ordered_records)
        source_families = [family for _cand, family in ordered_records]

        rollouts = []
        collisions_all = []
        reached_all = []
        progress_all = []
        for cand in candidates_shuffled:
            rollout, collisions, reached, progress = _simulate(grid, start, goal, obstacles, cand)
            rollouts.append(tuple(rollout))
            collisions_all.append(tuple(collisions))
            reached_all.append(bool(reached))
            progress_all.append(tuple(progress))
        if sum(bool(value) for value in reached_all) != 1 or not reached_all[label]:
            continue
        metadata = {
            "candidate_order_randomized": True,
            "candidate_order_balanced_by_index": True,
            "model_visible_candidate_sources": False,
            "model_visible_generator_rank": False,
            "model_visible_gold_label": False,
            "model_visible_simulator_success_flag": False,
            "explicit_final_state_tokens_in_locked_baseline": False,
            "explicit_outcome_tokens_in_locked_baseline": False,
            "candidate_pool_strategy": f"clean_learnability_ladder_level_{level}",
            "learnability_level": level,
            "candidate_source_families": source_families,
            "candidate_lengths": [len(c) for c in candidates_shuffled],
            "candidate_action_counts": [_action_counts(c) for c in candidates_shuffled],
            "candidate_turn_counts": [_turn_count(c) for c in candidates_shuffled],
            "candidate_bigram_reversal": [_bigram_reversal_rate(c) for c in candidates_shuffled],
            "offline_goal_reached": list(reached_all),
            "offline_collision_counts": [int(sum(c)) for c in collisions_all],
            "visible_last_manhattan": [int(p[-2]) if len(p) > 1 else int(p[0]) for p in progress_all],
            "final_manhattan": [int(p[-1]) if p else 0 for p in progress_all],
            "obstacle_count": len(obstacles),
            "manhattan_start_goal": _manhattan(start, goal),
        }
        return PlanExample(
            id=f"{split}_{index}",
            split=split,
            grid_size=grid,
            start=start,
            goal=goal,
            obstacles=tuple(sorted(obstacles)),
            candidates=candidates_shuffled,
            rollouts=tuple(rollouts),
            collisions=tuple(collisions_all),
            goal_reached=tuple(reached_all),
            progress=tuple(progress_all),
            label=label,
            metadata=metadata,
        )
    raise RuntimeError(f"could not generate ladder example after {config.max_generation_attempts} attempts")


def _sample_ladder_gold(
    level: int,
    grid: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: Sequence[Tuple[int, int]] | set[Tuple[int, int]],
    length: int,
    rng: np.random.Generator,
) -> Tuple[int, ...] | None:
    if int(level) in {4, 5}:
        neighbors = _goal_neighbor_actions(grid, goal, obstacles)
        if not neighbors:
            return None
        return _sample_success_plan_via_goal_neighbor(grid, start, goal, obstacles, length, neighbors, rng)
    path = _sample_no_collision_path_to_target(grid, start, goal, obstacles, length, rng)
    return tuple(path) if path is not None else None


def _sample_ladder_negative(
    level: int,
    grid: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: Sequence[Tuple[int, int]] | set[Tuple[int, int]],
    length: int,
    rng: np.random.Generator,
) -> Tuple[int, ...] | None:
    if int(level) in {4, 5}:
        neighbors = _goal_neighbor_actions(grid, goal, obstacles)
        return _sample_no_collision_near_goal_negative(grid, start, goal, obstacles, length, neighbors, rng) if neighbors else None
    target = _sample_ladder_target(level, grid, start, goal, obstacles, rng)
    if target is None:
        return None
    path = _sample_no_collision_path_to_target(grid, start, target, obstacles, length, rng)
    return tuple(path) if path is not None else None


def _sample_ladder_target(
    level: int,
    grid: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: Sequence[Tuple[int, int]] | set[Tuple[int, int]],
    rng: np.random.Generator,
) -> Tuple[int, int] | None:
    obstacle_set = set(obstacles)
    cells = []
    for row in range(int(grid)):
        for col in range(int(grid)):
            cell = (row, col)
            if cell in obstacle_set or cell in {start, goal}:
                continue
            distance = _manhattan(cell, goal)
            if int(level) in {0, 1} and distance >= 3:
                cells.append(cell)
            elif int(level) == 2 and 1 <= distance <= max(2, grid - 2):
                cells.append(cell)
            elif int(level) == 3 and 1 <= distance <= 2:
                cells.append(cell)
    if not cells:
        return None
    return cells[int(rng.integers(0, len(cells)))]


def _generate_shortcut_repaired_example(config: PlanDatasetConfig, split: str, index: int, seed: int) -> PlanExample:
    rng = np.random.default_rng(seed)
    for _attempt in range(int(config.max_generation_attempts)):
        grid = int(config.grid_size)
        start = (int(rng.integers(0, grid)), int(rng.integers(0, grid)))
        goal = (int(rng.integers(0, grid)), int(rng.integers(0, grid)))
        if start == goal:
            continue
        obstacles = _sample_obstacles(grid, start, goal, int(config.obstacle_count), rng)
        neighbors = _goal_neighbor_actions(grid, goal, obstacles)
        if not neighbors:
            continue

        gold = _sample_success_plan_via_goal_neighbor(grid, start, goal, obstacles, int(config.plan_length), neighbors, rng)
        if gold is None:
            continue
        gold_rollout, gold_collisions, gold_reached, gold_progress = _simulate(grid, start, goal, obstacles, gold)
        if not gold_reached or sum(gold_collisions):
            continue
        gold_signature = _candidate_shortcut_signature(start, goal, gold, gold_collisions, gold_progress)

        negative_pool: List[Tuple[float, Tuple[int, ...], str]] = []
        seen = {tuple(gold)}
        for _pool_attempt in range(int(config.shortcut_pool_attempts)):
            cand = _sample_no_collision_near_goal_negative(grid, start, goal, obstacles, int(config.plan_length), neighbors, rng)
            if cand is None or cand in seen:
                continue
            rollout, collisions, reached, progress = _simulate(grid, start, goal, obstacles, cand)
            if reached or sum(collisions):
                continue
            signature = _candidate_shortcut_signature(start, goal, cand, collisions, progress)
            score = _candidate_balance_distance(gold_signature, signature)
            if float(signature["direct_count_distance"]) <= float(gold_signature["direct_count_distance"]):
                score -= 1.0
            if int(signature["final_action"]) == int(gold_signature["final_action"]):
                score -= 0.25
            negative_pool.append((score, tuple(cand), "balanced_neighbor_path"))
            seen.add(tuple(cand))
        if len(negative_pool) < int(config.num_candidates) - 1:
            continue

        negatives = _select_balanced_negatives(negative_pool, int(config.num_candidates) - 1, rng)
        candidate_records = [(tuple(gold), "balanced_neighbor_path"), *[(cand, family) for cand, family in negatives]]
        ordered_records, label = _balanced_candidate_order(candidate_records, int(config.num_candidates), index, rng)
        candidates_shuffled = tuple(cand for cand, _family in ordered_records)
        source_families = [family for _cand, family in ordered_records]

        rollouts = []
        collisions_all = []
        reached_all = []
        progress_all = []
        for cand in candidates_shuffled:
            rollout, collisions, reached, progress = _simulate(grid, start, goal, obstacles, cand)
            rollouts.append(tuple(rollout))
            collisions_all.append(tuple(collisions))
            reached_all.append(bool(reached))
            progress_all.append(tuple(progress))
        if sum(bool(value) for value in reached_all) != 1 or not reached_all[label]:
            continue
        metadata = {
            "candidate_order_randomized": True,
            "candidate_order_balanced_by_index": True,
            "model_visible_candidate_sources": False,
            "model_visible_generator_rank": False,
            "model_visible_gold_label": False,
            "model_visible_simulator_success_flag": False,
            "explicit_final_state_tokens_in_locked_baseline": False,
            "explicit_outcome_tokens_in_locked_baseline": False,
            "candidate_pool_strategy": "shortcut_repaired_neighbor_prefix_v1",
            "candidate_source_families": source_families,
            "candidate_lengths": [len(c) for c in candidates_shuffled],
            "candidate_action_counts": [_action_counts(c) for c in candidates_shuffled],
            "candidate_turn_counts": [_turn_count(c) for c in candidates_shuffled],
            "candidate_bigram_reversal": [_bigram_reversal_rate(c) for c in candidates_shuffled],
            "offline_goal_reached": list(reached_all),
            "offline_collision_counts": [int(sum(c)) for c in collisions_all],
            "visible_last_manhattan": [int(p[-2]) if len(p) > 1 else int(p[0]) for p in progress_all],
            "obstacle_count": len(obstacles),
            "manhattan_start_goal": _manhattan(start, goal),
        }
        return PlanExample(
            id=f"{split}_{index}",
            split=split,
            grid_size=grid,
            start=start,
            goal=goal,
            obstacles=tuple(sorted(obstacles)),
            candidates=candidates_shuffled,
            rollouts=tuple(rollouts),
            collisions=tuple(collisions_all),
            goal_reached=tuple(reached_all),
            progress=tuple(progress_all),
            label=label,
            metadata=metadata,
        )
    raise RuntimeError(f"could not generate repaired gridworld example after {config.max_generation_attempts} attempts")


def _goal_neighbor_actions(
    grid: int,
    goal: Tuple[int, int],
    obstacles: Sequence[Tuple[int, int]] | set[Tuple[int, int]],
) -> List[Tuple[Tuple[int, int], int]]:
    obstacle_set = set(obstacles)
    out = []
    for action, (dr, dc) in ACTION_DELTAS.items():
        neighbor = (goal[0] - dr, goal[1] - dc)
        if _in_bounds(neighbor, grid) and neighbor not in obstacle_set:
            out.append((neighbor, int(action)))
    return out


def _sample_success_plan_via_goal_neighbor(
    grid: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: Sequence[Tuple[int, int]] | set[Tuple[int, int]],
    length: int,
    neighbors: Sequence[Tuple[Tuple[int, int], int]],
    rng: np.random.Generator,
) -> Tuple[int, ...] | None:
    pool: List[Tuple[float, Tuple[int, ...]]] = []
    for _attempt in range(96):
        order = rng.permutation(len(neighbors)).tolist()
        for idx in order:
            neighbor, final_action = neighbors[int(idx)]
            prefix = _sample_no_collision_path_to_target(grid, start, neighbor, obstacles, int(length) - 1, rng)
            if prefix is None:
                continue
            candidate = tuple([*prefix, int(final_action)])
            rollout, collisions, reached, progress = _simulate(grid, start, goal, obstacles, candidate)
            if not reached or sum(collisions):
                continue
            signature = _candidate_shortcut_signature(start, goal, candidate, collisions, progress)
            loopiness = float(signature["direct_count_distance"]) + 0.25 * float(signature["turns"]) + 2.0 * float(signature["bigram_reversal_rate"])
            pool.append((loopiness, candidate))
    if not pool:
        return None
    pool.sort(key=lambda row: row[0])
    start_idx = min(len(pool) - 1, max(0, int(len(pool) * 0.45)))
    tail = pool[start_idx:]
    return tail[int(rng.integers(0, len(tail)))][1]


def _sample_no_collision_near_goal_negative(
    grid: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: Sequence[Tuple[int, int]] | set[Tuple[int, int]],
    length: int,
    neighbors: Sequence[Tuple[Tuple[int, int], int]],
    rng: np.random.Generator,
) -> Tuple[int, ...] | None:
    obstacle_set = set(obstacles)
    order = rng.permutation(len(neighbors)).tolist()
    for idx in order:
        neighbor, goal_action = neighbors[int(idx)]
        prefix = _sample_no_collision_path_to_target(grid, start, neighbor, obstacles, int(length) - 1, rng)
        if prefix is None:
            continue
        wrong_actions = []
        for action, (dr, dc) in ACTION_DELTAS.items():
            if int(action) == int(goal_action):
                continue
            nxt = (neighbor[0] + dr, neighbor[1] + dc)
            if _in_bounds(nxt, grid) and nxt not in obstacle_set and nxt != goal:
                wrong_actions.append(int(action))
        if wrong_actions:
            return tuple([*prefix, int(rng.choice(wrong_actions))])
    return None


def _sample_no_collision_path_to_target(
    grid: int,
    start: Tuple[int, int],
    target: Tuple[int, int],
    obstacles: Sequence[Tuple[int, int]] | set[Tuple[int, int]],
    length: int,
    rng: np.random.Generator,
) -> List[int] | None:
    if length < 0:
        return None
    obstacle_set = set(obstacles)
    if target in obstacle_set:
        return None
    for _attempt in range(160):
        pos = start
        actions: List[int] = []
        for step in range(int(length)):
            remaining = int(length) - step - 1
            valid: List[Tuple[int, Tuple[int, int], int]] = []
            for action, (dr, dc) in ACTION_DELTAS.items():
                nxt = (pos[0] + dr, pos[1] + dc)
                if not _in_bounds(nxt, grid) or nxt in obstacle_set:
                    continue
                distance = _manhattan(nxt, target)
                if distance <= remaining and (remaining - distance) % 2 == 0:
                    valid.append((int(action), nxt, int(distance)))
            if not valid:
                break
            valid.sort(key=lambda row: row[2])
            weights = np.asarray([0.65 / (1.0 + row[2]) + 0.35 for row in valid], dtype=np.float64)
            weights = weights / weights.sum()
            choice = int(rng.choice(len(valid), p=weights))
            action, pos, _distance = valid[choice]
            actions.append(int(action))
        if len(actions) == int(length) and pos == target:
            return actions
    return None


def _candidate_shortcut_signature(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    actions: Sequence[int],
    collisions: Sequence[int],
    progress: Sequence[int],
) -> Dict[str, float]:
    direct = _direct_action_counts(start, goal, len(actions))
    counts = _action_counts(actions)
    visible_last = float(progress[-2]) if len(progress) > 1 else float(progress[0] if progress else 0)
    start_dist = float(progress[0]) if progress else float(_manhattan(start, goal))
    return {
        "direct_count_distance": float(np.sum(np.abs(np.asarray(counts) - np.asarray(direct)))),
        "bigram_reversal_rate": float(_bigram_reversal_rate(actions)),
        "turns": float(_turn_count(actions)),
        "collisions": float(sum(collisions)),
        "visible_last": visible_last,
        "visible_progress": float(start_dist - visible_last),
        "final_action": float(actions[-1]) if actions else 0.0,
    }


def _candidate_balance_distance(gold: Dict[str, float], candidate: Dict[str, float]) -> float:
    weights = {
        "direct_count_distance": 1.0,
        "bigram_reversal_rate": 4.0,
        "turns": 0.4,
        "collisions": 4.0,
        "visible_last": 3.0,
        "visible_progress": 0.5,
    }
    return float(sum(weight * abs(float(gold[key]) - float(candidate[key])) for key, weight in weights.items()))


def _select_balanced_negatives(
    negative_pool: Sequence[Tuple[float, Tuple[int, ...], str]],
    count: int,
    rng: np.random.Generator,
) -> List[Tuple[Tuple[int, ...], str]]:
    ranked = sorted(negative_pool, key=lambda row: (row[0], rng.random()))
    top = ranked[: max(int(count), min(len(ranked), int(count) * 4))]
    order = rng.permutation(len(top)).tolist()
    selected: List[Tuple[Tuple[int, ...], str]] = []
    for idx in order:
        _score, cand, family = top[int(idx)]
        selected.append((cand, family))
        if len(selected) == int(count):
            break
    if len(selected) < int(count):
        for _score, cand, family in ranked:
            if all(cand != existing for existing, _family in selected):
                selected.append((cand, family))
            if len(selected) == int(count):
                break
    return selected


def _balanced_candidate_order(
    records: Sequence[Tuple[Tuple[int, ...], str]],
    num_candidates: int,
    index: int,
    rng: np.random.Generator,
) -> Tuple[List[Tuple[Tuple[int, ...], str]], int]:
    label = int(index % max(1, int(num_candidates)))
    gold = records[0]
    negatives = [records[int(i)] for i in rng.permutation(len(records) - 1) + 1]
    ordered: List[Tuple[Tuple[int, ...], str] | None] = [None for _ in range(int(num_candidates))]
    ordered[label] = gold
    neg_iter = iter(negatives)
    for slot in range(int(num_candidates)):
        if ordered[slot] is None:
            ordered[slot] = next(neg_iter)
    return [record for record in ordered if record is not None], label


def _sample_obstacles(grid: int, start: Tuple[int, int], goal: Tuple[int, int], count: int, rng: np.random.Generator) -> set[Tuple[int, int]]:
    obstacles: set[Tuple[int, int]] = set()
    attempts = 0
    while len(obstacles) < int(count) and attempts < int(count) * 20:
        attempts += 1
        cell = (int(rng.integers(0, grid)), int(rng.integers(0, grid)))
        if cell not in {start, goal}:
            obstacles.add(cell)
    return obstacles


def _bfs_path(grid: int, start: Tuple[int, int], goal: Tuple[int, int], obstacles: set[Tuple[int, int]]) -> List[Tuple[int, int]] | None:
    queue = [start]
    parent: Dict[Tuple[int, int], Tuple[int, int] | None] = {start: None}
    for cell in queue:
        if cell == goal:
            break
        for action in range(4):
            dr, dc = ACTION_DELTAS[action]
            nxt = (cell[0] + dr, cell[1] + dc)
            if not _in_bounds(nxt, grid) or nxt in obstacles or nxt in parent:
                continue
            parent[nxt] = cell
            queue.append(nxt)
    if goal not in parent:
        return None
    path = []
    cur: Tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return list(reversed(path))


def _path_to_actions(path: Sequence[Tuple[int, int]]) -> List[int]:
    actions = []
    reverse = {delta: action for action, delta in ACTION_DELTAS.items()}
    for a, b in zip(path, path[1:]):
        actions.append(reverse[(b[0] - a[0], b[1] - a[1])])
    return actions


def _pad_plan(
    actions: Sequence[int],
    length: int,
    grid: int | None = None,
    terminal: Tuple[int, int] | None = None,
    obstacles: Sequence[Tuple[int, int]] | set[Tuple[int, int]] | None = None,
) -> List[int]:
    out = list(actions)
    cycle = [2, 3]
    if grid is not None and terminal is not None:
        obstacle_set = set(obstacles or [])
        for action, reverse in ((0, 1), (1, 0), (2, 3), (3, 2)):
            dr, dc = ACTION_DELTAS[action]
            nxt = (terminal[0] + dr, terminal[1] + dc)
            if _in_bounds(nxt, int(grid)) and nxt not in obstacle_set:
                cycle = [action, reverse]
                break
    while len(out) + 1 < int(length):
        out.extend(cycle)
    return out[: int(length)]


def _make_distractor(
    gold: Sequence[int],
    grid: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: set[Tuple[int, int]],
    rng: np.random.Generator,
    salt: int,
) -> List[int]:
    if salt % 3 != 2:
        obstacle_set = set(obstacles)
        for _attempt in range(40):
            target = (int(rng.integers(0, grid)), int(rng.integers(0, grid)))
            if target in {start, goal} or target in obstacle_set:
                continue
            path = _bfs_path(grid, start, target, obstacle_set)
            if path is None or len(path) - 1 > len(gold):
                continue
            if (len(gold) - (len(path) - 1)) % 2 != 0:
                continue
            return _pad_plan(_path_to_actions(path), len(gold), grid, target, obstacle_set)
    cand = list(gold)
    mode = salt % 6
    if mode == 0 and len(cand) > 3:
        i, j = sorted(rng.choice(len(cand), size=2, replace=False))
        cand[i], cand[j] = cand[j], cand[i]
    elif mode == 1 and len(cand) > 4:
        i = int(rng.integers(0, len(cand) - 3))
        cand[i : i + 3] = reversed(cand[i : i + 3])
    elif mode == 2:
        i = int(rng.integers(0, len(cand)))
        cand[i] = int((cand[i] + int(rng.choice([1, 2, 3]))) % 4)
        j = int(rng.integers(0, len(cand)))
        cand[j] = int((cand[j] + int(rng.choice([1, 3]))) % 4)
    elif mode == 3:
        rng.shuffle(cand)
    elif mode == 4 and len(cand) > 5:
        shift = int(rng.integers(1, len(cand)))
        cand = cand[shift:] + cand[:shift]
    else:
        counts = _action_counts(gold)
        pool = [action for action, count in enumerate(counts) for _ in range(int(count))]
        if len(pool) == len(cand):
            rng.shuffle(pool)
            cand = [int(action) for action in pool]
    return cand


def _simulate(
    grid: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: Sequence[Tuple[int, int]] | set[Tuple[int, int]],
    actions: Sequence[int],
) -> Tuple[List[Tuple[int, int]], List[int], bool, List[int]]:
    obstacle_set = set(obstacles)
    pos = start
    states = [pos]
    collisions = []
    progress = [_manhattan(pos, goal)]
    for action in actions:
        dr, dc = ACTION_DELTAS[int(action)]
        nxt = (pos[0] + dr, pos[1] + dc)
        collide = int((not _in_bounds(nxt, grid)) or nxt in obstacle_set)
        if not collide:
            pos = nxt
        states.append(pos)
        collisions.append(collide)
        progress.append(_manhattan(pos, goal))
    return states, collisions, pos == goal and sum(collisions) == 0, progress


def apply_plan_control(examples: Sequence[PlanExample], control: str, seed: int) -> List[PlanExample]:
    rng = np.random.default_rng(seed)
    if control == "candidate_order_shuffle_with_gold_remap":
        return [_candidate_order_shuffle(example, rng) for example in examples]
    if control == "candidate_identity_shuffle":
        return [replace(shuffled, metadata={**shuffled.metadata, "control": control}) for shuffled in (_candidate_order_shuffle(example, rng) for example in examples)]
    if control == "candidate_list_composition_control":
        return _candidate_list_composition_control(examples, rng)
    if control == "candidate_only":
        return [replace(example, start=(0, 0), goal=(0, 0), obstacles=tuple(), rollouts=_blank_rollouts(example), collisions=_blank_collisions(example), progress=_blank_progress(example), metadata={**example.metadata, "control": control}) for example in examples]
    if control == "state_plan_mismatch":
        return _cross_example_replace(examples, rng, replace_state=True, replace_goal=False, replace_rollout=True, control=control)
    if control == "goal_shuffle":
        return _cross_example_replace(examples, rng, replace_state=False, replace_goal=True, replace_rollout=False, control=control)
    if control == "rollout_mismatch":
        return _cross_example_replace(examples, rng, replace_state=False, replace_goal=False, replace_rollout=True, control=control)
    if control == "final_state_mismatch":
        return _final_state_mismatch(examples, rng)
    if control == "role_order_shuffle":
        return [replace(example, metadata={**example.metadata, "control": control, "role_order_shuffled": True}) for example in examples]
    if control == "randomized_labels":
        return [replace(example, label=int(rng.integers(0, len(example.candidates))), metadata={**example.metadata, "control": control}) for example in examples]
    raise ValueError(f"unknown plan control: {control}")


def _candidate_order_shuffle(example: PlanExample, rng: np.random.Generator) -> PlanExample:
    order = rng.permutation(len(example.candidates))
    if len(order) > 1 and np.array_equal(order, np.arange(len(order))):
        order = np.roll(order, 1)
    label = int(np.where(order == int(example.label))[0][0])
    metadata = dict(example.metadata)
    for key in (
        "candidate_lengths",
        "candidate_action_counts",
        "candidate_turn_counts",
        "candidate_bigram_reversal",
        "candidate_source_families",
        "offline_goal_reached",
        "offline_collision_counts",
        "visible_last_manhattan",
        "final_manhattan",
    ):
        values = metadata.get(key)
        if isinstance(values, list) and len(values) == len(order):
            metadata[key] = [values[int(i)] for i in order]
    metadata["control"] = "candidate_order_shuffle_with_gold_remap"
    return replace(
        example,
        candidates=tuple(example.candidates[int(i)] for i in order),
        rollouts=tuple(example.rollouts[int(i)] for i in order),
        collisions=tuple(example.collisions[int(i)] for i in order),
        goal_reached=tuple(example.goal_reached[int(i)] for i in order),
        progress=tuple(example.progress[int(i)] for i in order),
        label=label,
        metadata=metadata,
    )


def _candidate_list_composition_control(examples: Sequence[PlanExample], rng: np.random.Generator) -> List[PlanExample]:
    if len(examples) < 2:
        return list(examples)
    out = []
    for index, example in enumerate(examples):
        other_index = int(rng.integers(0, len(examples) - 1))
        if other_index >= index:
            other_index += 1
        other = examples[other_index]
        if len(other.candidates) != len(example.candidates):
            out.append(example)
            continue
        out.append(
            replace(
                example,
                candidates=other.candidates,
                rollouts=other.rollouts,
                collisions=other.collisions,
                goal_reached=other.goal_reached,
                progress=other.progress,
                metadata={**example.metadata, "control": "candidate_list_composition_control", "candidate_list_from_example": other.id},
            )
        )
    return out


def _final_state_mismatch(examples: Sequence[PlanExample], rng: np.random.Generator) -> List[PlanExample]:
    if len(examples) < 2:
        return list(examples)
    out = []
    for index, example in enumerate(examples):
        other_index = int(rng.integers(0, len(examples) - 1))
        if other_index >= index:
            other_index += 1
        other = examples[other_index]
        if len(other.rollouts) != len(example.rollouts):
            out.append(example)
            continue
        rollouts = []
        progress = []
        for cand_index, rollout in enumerate(example.rollouts):
            if not rollout:
                rollouts.append(rollout)
                progress.append(example.progress[cand_index])
                continue
            replacement = other.rollouts[cand_index][-1] if other.rollouts[cand_index] else rollout[-1]
            patched = tuple([*rollout[:-1], replacement])
            rollouts.append(patched)
            progress.append(tuple(_manhattan(pos, example.goal) for pos in patched))
        out.append(
            replace(
                example,
                rollouts=tuple(rollouts),
                progress=tuple(progress),
                metadata={**example.metadata, "control": "final_state_mismatch", "mismatch_from_example": other.id},
            )
        )
    return out


def _cross_example_replace(
    examples: Sequence[PlanExample],
    rng: np.random.Generator,
    replace_state: bool,
    replace_goal: bool,
    replace_rollout: bool,
    control: str,
) -> List[PlanExample]:
    if len(examples) < 2:
        return list(examples)
    out = []
    for index, example in enumerate(examples):
        other_index = int(rng.integers(0, len(examples) - 1))
        if other_index >= index:
            other_index += 1
        other = examples[other_index]
        out.append(
            replace(
                example,
                start=other.start if replace_state else example.start,
                obstacles=other.obstacles if replace_state else example.obstacles,
                goal=other.goal if replace_goal else example.goal,
                rollouts=other.rollouts if replace_rollout and len(other.rollouts) == len(example.rollouts) else example.rollouts,
                collisions=other.collisions if replace_rollout and len(other.collisions) == len(example.collisions) else example.collisions,
                progress=other.progress if replace_rollout and len(other.progress) == len(example.progress) else example.progress,
                metadata={**example.metadata, "control": control, "mismatch_from_example": other.id},
            )
        )
    return out


class PlanLatentVerifier(nn.Module):
    def __init__(self, variant: PlanVariant) -> None:
        super().__init__()
        self.variant = variant
        dim = _compatible_dim(variant.model_dim, variant.num_heads)
        self.dim = dim
        self.token_encoder = PlanTokenEncoder(dim)
        layer = nn.TransformerEncoderLayer(dim, int(variant.num_heads), int(variant.ff_dim), dropout=0.0, batch_first=True, activation="gelu", norm_first=True)
        cand_layer = nn.TransformerEncoderLayer(dim, int(variant.num_heads), int(variant.ff_dim), dropout=0.0, batch_first=True, activation="gelu", norm_first=True)
        self.evidence_encoder = nn.TransformerEncoder(layer, num_layers=int(variant.evidence_layers))
        self.candidate_encoder = nn.TransformerEncoder(cand_layer, num_layers=int(variant.candidate_layers))
        self.candidate_self = (
            nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, int(variant.num_heads), int(variant.ff_dim), batch_first=True, activation="gelu", norm_first=True), num_layers=1)
            if variant.candidate_self_attention
            else None
        )
        self.blocks = nn.ModuleList([PlanCrossBlock(dim, int(variant.num_heads), int(variant.ff_dim)) for _ in range(int(variant.coordination_blocks))])
        self.reverse_blocks = nn.ModuleList([PlanCrossBlock(dim, int(variant.num_heads), int(variant.ff_dim)) for _ in range(int(variant.coordination_blocks))])
        self.candidate_to_evidence = nn.Linear(dim, dim)
        self.pyramid_projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.listwise = nn.MultiheadAttention(dim, int(variant.num_heads), batch_first=True)
        self.score_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.aux_collision = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.aux_success = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def configure_shared_trainable(self, trainable: bool) -> None:
        shared_ids = {id(parameter) for _name, parameter in self.shared_parameter_items()}
        for parameter in self.parameters():
            parameter.requires_grad = True
        if not trainable:
            for parameter in self.parameters():
                if id(parameter) in shared_ids:
                    parameter.requires_grad = False

    def shared_parameter_items(self) -> List[Tuple[str, nn.Parameter]]:
        modules = ("token_encoder", "evidence_encoder", "candidate_encoder")
        return [(name, parameter) for name, parameter in self.named_parameters() if name.split(".", 1)[0] in modules]

    def coordinator_parameter_items(self) -> List[Tuple[str, nn.Parameter]]:
        modules = ("candidate_self", "blocks", "reverse_blocks", "candidate_to_evidence", "pyramid_projection", "listwise", "score_head", "aux_collision", "aux_success")
        return [(name, parameter) for name, parameter in self.named_parameters() if name.split(".", 1)[0] in modules]

    def forward(self, batch: Dict[str, torch.Tensor], hidden_state_shuffle: bool = False, shuffle_seed: int = 0, return_attention: bool = False) -> Dict[str, object]:
        evidence = batch["evidence_fields"]
        evidence_mask = batch["evidence_mask"].bool()
        candidate = batch["candidate_fields"]
        candidate_mask = batch["candidate_mask"].bool()
        bsz, cand_count, view_count, evid_tokens, _fields = evidence.shape
        cand_tokens = candidate.shape[2]
        evid_emb = self.token_encoder(evidence.reshape(bsz * cand_count * view_count, evid_tokens, -1))
        evid_encoded = self.evidence_encoder(evid_emb, src_key_padding_mask=~evidence_mask.reshape(bsz * cand_count * view_count, evid_tokens))
        evid_encoded = evid_encoded.reshape(bsz, cand_count, view_count * evid_tokens, -1)
        evid_mask = evidence_mask.reshape(bsz, cand_count, view_count * evid_tokens)
        if hidden_state_shuffle and bsz > 1:
            generator = torch.Generator(device=evid_encoded.device)
            generator.manual_seed(int(shuffle_seed))
            order = torch.randperm(bsz, generator=generator, device=evid_encoded.device)
            evid_encoded = evid_encoded.index_select(0, order)
            evid_mask = evid_mask.index_select(0, order)

        cand_emb = self.token_encoder(candidate.reshape(bsz * cand_count, cand_tokens, -1))
        cand_encoded = self.candidate_encoder(cand_emb, src_key_padding_mask=~candidate_mask.reshape(bsz * cand_count, cand_tokens))
        if self.candidate_self is not None:
            cand_encoded = self.candidate_self(cand_encoded, src_key_padding_mask=~candidate_mask.reshape(bsz * cand_count, cand_tokens))
        attention_rows = []
        flat_evid = evid_encoded.reshape(bsz * cand_count, view_count * evid_tokens, -1)
        flat_evid_mask = evid_mask.reshape(bsz * cand_count, view_count * evid_tokens)
        if self.variant.pyramid:
            pooled_views = _masked_mean(evid_encoded.reshape(bsz * cand_count * view_count, evid_tokens, -1), evidence_mask.reshape(bsz * cand_count * view_count, evid_tokens))
            pooled_views = pooled_views.reshape(bsz * cand_count, view_count, -1)
            composed = self.pyramid_projection(pooled_views.mean(dim=1, keepdim=True))
            flat_evid = torch.cat([flat_evid, composed], dim=1)
            flat_evid_mask = torch.cat([flat_evid_mask, torch.ones((bsz * cand_count, 1), dtype=torch.bool, device=flat_evid.device)], dim=1)
        for block_index, block in enumerate(self.blocks):
            if self.variant.query_mechanism == "candidate_guided_evidence_processing":
                pooled_candidate = _masked_mean(cand_encoded, candidate_mask.reshape(bsz * cand_count, cand_tokens))
                guided = flat_evid + self.candidate_to_evidence(pooled_candidate).unsqueeze(1)
            else:
                guided = flat_evid
            if self.variant.bidirectional or self.variant.query_mechanism == "bidirectional_candidate_evidence":
                reverse, _ = self.reverse_blocks[block_index](guided, cand_encoded, candidate_mask.reshape(bsz * cand_count, cand_tokens), return_attention=False)
                guided = reverse
            cand_encoded, weights = block(cand_encoded, guided, flat_evid_mask, return_attention=return_attention)
            if weights is not None:
                attention_rows.append(weights.reshape(bsz, cand_count, *weights.shape[1:]))
        pooled = _masked_mean(cand_encoded, candidate_mask.reshape(bsz * cand_count, cand_tokens)).reshape(bsz, cand_count, -1)
        if self.variant.query_mechanism in {"pairwise_candidate_comparison", "listwise_candidate_ranking"}:
            listwise_out, _weights = self.listwise(pooled, pooled, pooled, need_weights=False)
            pooled = pooled + listwise_out
        logits = self.score_head(pooled).squeeze(-1)
        aux_collision = self.aux_collision(pooled).squeeze(-1)
        aux_success = self.aux_success(pooled).squeeze(-1)
        return {"logits": logits, "aux_collision": aux_collision, "aux_success": aux_success, "attention_weights": attention_rows}


class PlanTokenEncoder(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(size, model_dim) for size in FIELD_VOCABS])
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        fields = fields.long()
        out = torch.zeros((*fields.shape[:2], self.embeddings[0].embedding_dim), dtype=torch.float32, device=fields.device)
        for index, embedding in enumerate(self.embeddings):
            values = fields[..., index].clamp(min=0, max=embedding.num_embeddings - 1)
            out = out + embedding(values)
        return self.norm(out)


class PlanCrossBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, ff_dim: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(model_dim, num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(model_dim)
        self.norm_ff = nn.LayerNorm(model_dim)
        self.ff = nn.Sequential(nn.Linear(model_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, model_dim))

    def forward(self, query: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor, return_attention: bool) -> Tuple[torch.Tensor, torch.Tensor | None]:
        attended, weights = self.attn(
            self.norm_q(query),
            context,
            context,
            key_padding_mask=~context_mask.bool(),
            need_weights=return_attention,
            average_attn_weights=False,
        )
        query = query + attended
        query = query + self.ff(self.norm_ff(query))
        return query, weights if return_attention else None


def fit_plan_verifier(
    train_examples: Sequence[PlanExample],
    dev_examples: Sequence[PlanExample],
    variant: PlanVariant,
    training: PlanTrainingConfig,
    seed: int,
    device: str,
    trainable_shared: bool,
    method: str,
) -> PlanFitResult:
    _set_seed(seed)
    model = PlanLatentVerifier(variant).to(device)
    model.configure_shared_trainable(trainable_shared)
    initial_shared = _flat_params(model.shared_parameter_items())
    initial_coord = _flat_params(model.coordinator_parameter_items())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(training.lr), weight_decay=float(training.weight_decay))
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_dev = -1.0
    bad = 0
    history: List[Dict[str, float]] = []
    shared_grad_norms = []
    coord_grad_norms = []
    start = time.perf_counter()
    for epoch in range(int(training.epochs)):
        model.train()
        losses = []
        order = np.random.default_rng(seed + epoch).permutation(len(train_examples)).tolist()
        for batch_ids in _chunks(order, int(training.batch_size)):
            batch_examples = [train_examples[i] for i in batch_ids]
            batch = collate_plan_batch(batch_examples, variant, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss = _candidate_loss(out, batch, training)
            loss.backward()
            shared_grad_norms.append(_grad_norm([p for _n, p in model.shared_parameter_items()]))
            coord_grad_norms.append(_grad_norm([p for _n, p in model.coordinator_parameter_items()]))
            if float(training.gradient_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(training.gradient_clip_norm))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_logits = predict_plan_logits(model, dev_examples, variant, training.batch_size, device)
        dev_acc = _top1(dev_logits, _labels(dev_examples))
        history.append({"epoch": float(epoch), "loss": _mean(losses), "dev_top1": float(dev_acc)})
        if dev_acc > best_dev:
            best_dev = float(dev_acc)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= int(training.patience):
            break
    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    final_shared = _flat_params(model.shared_parameter_items())
    final_coord = _flat_params(model.coordinator_parameter_items())
    audit = {
        "method": method,
        "initialization_seed": int(seed),
        "shared_model_trainable": bool(trainable_shared),
        "same_architecture_comparator": True,
        "shared_parameter_delta": _l2_delta(initial_shared, final_shared),
        "coordinator_parameter_delta": _l2_delta(initial_coord, final_coord),
        "shared_grad_norm_mean": _mean(shared_grad_norms),
        "coordinator_grad_norm_mean": _mean(coord_grad_norms),
        "frozen_shared_model_zero_delta": bool((not trainable_shared) and _l2_delta(initial_shared, final_shared) == 0.0),
        "frozen_shared_model_zero_grad": bool((not trainable_shared) and _mean(shared_grad_norms) == 0.0),
        "trainable_shared_model_changed": bool(trainable_shared and _l2_delta(initial_shared, final_shared) > 0.0),
        "trainable_shared_model_received_gradients": bool(trainable_shared and _mean(shared_grad_norms) > 0.0),
    }
    return PlanFitResult(method, model, variant, trainable_shared, audit, history, float(time.perf_counter() - start), sum(p.numel() for p in model.parameters()))


def predict_plan_logits(
    model: PlanLatentVerifier,
    examples: Sequence[PlanExample],
    variant: PlanVariant,
    batch_size: int,
    device: str,
    condition: str = "none",
    seed: int = 0,
    view_mask: str | None = None,
    return_attention: bool = False,
) -> np.ndarray | Tuple[np.ndarray, List[Dict[str, object]]]:
    model.eval()
    out_logits = []
    attention_rows: List[Dict[str, object]] = []
    eval_variant = variant
    if condition == "role_order_shuffle":
        views = list(_expanded_views(variant))
        rng = np.random.default_rng(seed)
        rng.shuffle(views)
        eval_variant = replace(variant, view_names=tuple(views), avenues_per_view=1)
    with torch.no_grad():
        for offset, batch_examples in enumerate(_batched(examples, int(batch_size))):
            batch = collate_plan_batch(batch_examples, eval_variant, device, view_mask=view_mask)
            output = model(
                batch,
                hidden_state_shuffle=condition == "hidden_state_shuffle",
                shuffle_seed=seed + offset,
                return_attention=return_attention,
            )
            logits = output["logits"].detach().cpu().numpy()
            out_logits.append(logits)
            if return_attention:
                attention_rows.extend(_summarize_attention(output, batch_examples, eval_variant, logits))
    logits = np.concatenate(out_logits, axis=0) if out_logits else np.zeros((0, 0), dtype=np.float32)
    if return_attention:
        return logits, attention_rows
    return logits


def collate_plan_batch(examples: Sequence[PlanExample], variant: PlanVariant, device: str, view_mask: str | None = None) -> Dict[str, torch.Tensor]:
    max_evidence_tokens = 72
    max_candidate_tokens = 48
    views = _expanded_views(variant)
    evidence_rows = []
    evidence_masks = []
    candidate_rows = []
    candidate_masks = []
    labels = []
    collision_labels = []
    success_labels = []
    for example in examples:
        ex_evidence = []
        ex_evidence_mask = []
        ex_candidates = []
        ex_candidate_mask = []
        ex_collision = []
        ex_success = []
        for cand_index, actions in enumerate(example.candidates):
            per_view = []
            per_mask = []
            for view_id, view in enumerate(views):
                tokens = [] if view_mask == _canonical_view(view) else _evidence_tokens(example, cand_index, view, view_id, variant)
                fields, mask = _pack_tokens(tokens, max_evidence_tokens)
                per_view.append(fields)
                per_mask.append(mask)
            cand_tokens = _candidate_tokens(example, cand_index, actions, variant)
            cand_fields, cand_mask = _pack_tokens(cand_tokens, max_candidate_tokens)
            ex_evidence.append(np.stack(per_view))
            ex_evidence_mask.append(np.stack(per_mask))
            ex_candidates.append(cand_fields)
            ex_candidate_mask.append(cand_mask)
            ex_collision.append(float(sum(example.collisions[cand_index]) > 0))
            ex_success.append(float(example.goal_reached[cand_index]))
        evidence_rows.append(np.stack(ex_evidence))
        evidence_masks.append(np.stack(ex_evidence_mask))
        candidate_rows.append(np.stack(ex_candidates))
        candidate_masks.append(np.stack(ex_candidate_mask))
        labels.append(int(example.label))
        collision_labels.append(ex_collision)
        success_labels.append(ex_success)
    return {
        "evidence_fields": torch.as_tensor(np.stack(evidence_rows), dtype=torch.long, device=device),
        "evidence_mask": torch.as_tensor(np.stack(evidence_masks), dtype=torch.bool, device=device),
        "candidate_fields": torch.as_tensor(np.stack(candidate_rows), dtype=torch.long, device=device),
        "candidate_mask": torch.as_tensor(np.stack(candidate_masks), dtype=torch.bool, device=device),
        "labels": torch.as_tensor(labels, dtype=torch.long, device=device),
        "collision_labels": torch.as_tensor(collision_labels, dtype=torch.float32, device=device),
        "success_labels": torch.as_tensor(success_labels, dtype=torch.float32, device=device),
    }


def _evidence_tokens(example: PlanExample, cand_index: int, view: str, view_id: int, variant: PlanVariant) -> List[List[int]]:
    view = _canonical_view(view)
    tokens: List[List[int]] = []
    if view in {"state", "state_goal", "goal_relevance_state"}:
        tokens.append(_token("state", example.start[0], example.start[1], 0, 0, example.grid_size, view_id, 0))
    if view in {"goal", "state_goal", "goal_relevance_state"}:
        tokens.append(_token("goal", example.goal[0], example.goal[1], 0, 0, example.grid_size, view_id, 0))
    if view in {"obstacle", "risk", "risk_adjusted_future", "goal_relevance_state"}:
        for row, col in example.obstacles:
            tokens.append(_token("obstacle", row, col, 0, 0, example.grid_size, view_id, 0))
    if view in {"rollout", "dense_rollout", "partial_rollout", "rollout_short", "rollout_long", "candidate_future", "risk_adjusted_future", "progress", "counterfactual"}:
        states = list(example.rollouts[cand_index])
        collisions = list(example.collisions[cand_index])
        progress = list(example.progress[cand_index])
        usable_states = states if variant.include_final_state else states[:-1]
        if view in {"partial_rollout", "rollout_short"}:
            usable_states = usable_states[: max(1, len(usable_states) // 2)]
        elif view == "rollout_long":
            usable_states = usable_states[len(usable_states) // 2 :]
        for step, pos in enumerate(usable_states):
            delta = 0
            if view == "progress" and step < len(progress) - 1:
                delta = int(np.sign(progress[step] - progress[step + 1]) + 1)
            tokens.append(_token("rollout", pos[0], pos[1], 0, step, example.grid_size, view_id, delta))
            if view in {"risk", "risk_adjusted_future", "counterfactual"} and step < len(collisions) and collisions[step]:
                tokens.append(_token("risk", pos[0], pos[1], 1, step, example.grid_size, view_id, 1))
            if view == "progress" and step < len(progress):
                tokens.append(_token("progress", pos[0], pos[1], min(15, progress[step]), step, example.grid_size, view_id, delta))
    if view in {"transition", "transition_tuple", "action_conditioned_transition", "hybrid_transition_summary"}:
        states = list(example.rollouts[cand_index])
        actions = list(example.candidates[cand_index])
        collisions = list(example.collisions[cand_index])
        max_step = min(len(actions), max(0, len(states) - 1))
        if not variant.include_final_state:
            max_step = max(0, max_step - 1)
        for step in range(max_step):
            state_t = states[step]
            state_next = states[step + 1]
            action = int(actions[step])
            dr = state_next[0] - state_t[0]
            dc = state_next[1] - state_t[1]
            delta_code = (dr + 1) * 3 + (dc + 1)
            contact = int(collisions[step]) if step < len(collisions) else 0
            goal_distance = _manhattan(state_next, example.goal)
            tokens.append(_token("transition", state_t[0], state_t[1], action, step, example.grid_size, view_id, min(15, goal_distance)))
            tokens.append(_token("transition", state_next[0], state_next[1], int(delta_code), step, example.grid_size, view_id, contact))
            if view in {"action_conditioned_transition", "hybrid_transition_summary"}:
                tokens.append(_token("action", state_t[0], state_t[1], action, step, example.grid_size, view_id, int(delta_code)))
        if states and actions:
            pre_final = states[-2] if len(states) > 1 else states[-1]
            final_action = int(actions[-1])
            final_state = states[-1]
            if variant.include_final_state:
                tokens.append(_token("summary", final_state[0], final_state[1], min(15, _manhattan(final_state, example.goal)), len(actions), example.grid_size, view_id, 0))
            tokens.append(_token("summary", pre_final[0], pre_final[1], final_action, max(0, len(actions) - 1), example.grid_size, view_id, min(15, _manhattan(pre_final, example.goal))))
    if view in {"action", "action_sequence", "candidate_future", "plan_value"}:
        for step, action in enumerate(example.candidates[cand_index]):
            tokens.append(_token("action", 0, 0, int(action), step, example.grid_size, view_id, 0))
    if view in {"cost", "plan_value"}:
        tokens.append(_token("summary", 0, 0, min(15, len(example.candidates[cand_index])), 0, example.grid_size, view_id, 0))
    if variant.include_outcome_tokens:
        tokens.append(_token("summary", 0, 0, int(example.goal_reached[cand_index]), len(example.candidates[cand_index]), example.grid_size, view_id, int(sum(example.collisions[cand_index]) > 0)))
    if not tokens:
        tokens.append(_token("summary", 0, 0, 0, 0, example.grid_size, view_id, 0))
    return tokens


def _candidate_tokens(example: PlanExample, cand_index: int, actions: Sequence[int], variant: PlanVariant) -> List[List[int]]:
    tokens = [_token("summary", example.start[0], example.start[1], 0, 0, example.grid_size, 0, 0)]
    if variant.include_outcome_tokens:
        tokens.append(_token("summary", 0, 0, int(example.goal_reached[cand_index]), 0, example.grid_size, 0, int(sum(example.collisions[cand_index]) > 0)))
    for step, action in enumerate(actions):
        prev_action = int(actions[step - 1]) if step > 0 else 4
        tokens.append(_token("action", prev_action, int(action), int(action), step, example.grid_size, 0, 0))
    return tokens


def _token(role: str, row: int, col: int, value: int, step: int, grid_size: int, view_id: int, aux: int) -> List[int]:
    raw = [ROLE_IDS.get(role, 0), row, col, value, step, grid_size, view_id, aux]
    return [max(0, min(FIELD_VOCABS[i] - 1, int(v))) for i, v in enumerate(raw)]


def _pack_tokens(tokens: Sequence[Sequence[int]], limit: int) -> Tuple[np.ndarray, np.ndarray]:
    fields = np.zeros((int(limit), len(FIELD_VOCABS)), dtype=np.int64)
    mask = np.zeros((int(limit),), dtype=bool)
    if not tokens:
        tokens = [_token("summary", 0, 0, 0, 0, 8, 0, 0)]
    count = min(int(limit), len(tokens))
    fields[:count] = np.asarray(tokens[:count], dtype=np.int64)
    mask[:count] = True
    return fields, mask


def _candidate_loss(output: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], training: PlanTrainingConfig) -> torch.Tensor:
    logits = output["logits"]
    labels = batch["labels"]
    objective = str(training.objective)
    gold = logits.gather(1, labels.view(-1, 1))
    wrong_mask = torch.ones_like(logits, dtype=torch.bool)
    wrong_mask.scatter_(1, labels.view(-1, 1), False)
    wrong = logits.masked_select(wrong_mask).view(logits.shape[0], -1)
    if objective == "pairwise_ranking":
        loss = F.softplus(wrong - gold).mean()
    elif objective == "margin_ranking":
        loss = F.relu(float(training.margin) - gold + wrong).mean()
    elif objective == "contrastive_candidate_scoring":
        loss = F.cross_entropy(logits - logits.mean(dim=1, keepdim=True), labels)
    else:
        loss = F.cross_entropy(logits, labels)
    if objective in {"aux_collision", "aux_goal_reached", "aux_progress", "aux_plan_value"}:
        if objective == "aux_collision":
            aux = F.binary_cross_entropy_with_logits(output["aux_collision"], 1.0 - batch["collision_labels"])
        else:
            aux = F.binary_cross_entropy_with_logits(output["aux_success"], batch["success_labels"])
        loss = loss + float(training.aux_weight) * aux
    return loss


class PlanFeatureMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        bsz, cand, dim = features.shape
        return self.net(features.reshape(bsz * cand, dim)).reshape(bsz, cand)


def fit_feature_mlp(train_examples: Sequence[PlanExample], dev_examples: Sequence[PlanExample], config: FeatureTrainingConfig, seed: int) -> FeatureFitResult:
    _set_seed(seed)
    model = PlanFeatureMLP(_feature_dim(), int(config.hidden_dim))
    initial = _flat_feature_params(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_dev = -1.0
    bad = 0
    history: List[Dict[str, float]] = []
    start = time.perf_counter()
    for epoch in range(int(config.epochs)):
        losses = []
        order = np.random.default_rng(seed + epoch).permutation(len(train_examples)).tolist()
        for batch_ids in _chunks(order, int(config.batch_size)):
            examples = [train_examples[i] for i in batch_ids]
            features, labels = _feature_batch(examples)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        dev_logits = predict_feature_logits(model, dev_examples)
        dev_acc = _top1(dev_logits, _labels(dev_examples))
        history.append({"epoch": float(epoch), "loss": _mean(losses), "dev_top1": float(dev_acc)})
        if dev_acc > best_dev:
            best_dev = float(dev_acc)
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= int(config.patience):
            break
    model.load_state_dict(best_state)
    final = _flat_feature_params(model)
    return FeatureFitResult("stats_mlp", model, {"parameter_delta": _l2_delta(initial, final), "trainable": True}, history, float(time.perf_counter() - start))


def predict_feature_logits(model: PlanFeatureMLP, examples: Sequence[PlanExample]) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        features, _labels_t = _feature_batch(examples)
        return model(features).detach().cpu().numpy()


def _feature_batch(examples: Sequence[PlanExample]) -> Tuple[torch.Tensor, torch.Tensor]:
    rows = [[_candidate_features(example, cand_index) for cand_index in range(len(example.candidates))] for example in examples]
    return torch.as_tensor(rows, dtype=torch.float32), torch.as_tensor([example.label for example in examples], dtype=torch.long)


def _candidate_features(example: PlanExample, cand_index: int) -> List[float]:
    actions = example.candidates[cand_index]
    counts = np.asarray(_action_counts(actions), dtype=np.float32) / max(1, len(actions))
    return [
        len(actions) / 16.0,
        *_as_list(counts),
        _bigram_reversal_rate(actions),
        _turn_count(actions) / max(1, len(actions) - 1),
    ]


def _feature_dim() -> int:
    return 7


def _baseline_block(
    train_examples: Sequence[PlanExample],
    dev_examples: Sequence[PlanExample],
    test_examples: Sequence[PlanExample],
    feature_training: FeatureTrainingConfig,
    seed: int,
) -> Dict[str, object]:
    labels = _labels(test_examples)
    random = 1.0 / max(1, len(test_examples[0].candidates) if test_examples else 1)
    feature_fit = fit_feature_mlp(train_examples, dev_examples, feature_training, seed + 303)
    stats_logits = predict_feature_logits(feature_fit.model, test_examples)
    return {
        "random": random,
        "candidate_only_index0": _accuracy(np.zeros(len(test_examples), dtype=np.int64), labels),
        "length_only": _accuracy(_length_only_predictions(test_examples), labels),
        "unigram_action_stat": _accuracy(_unigram_predictions(test_examples), labels),
        "bigram_action_stat": _accuracy(_bigram_predictions(test_examples), labels),
        "candidate_source_metadata_only": _candidate_source_metadata_only_accuracy(test_examples),
        "candidate_pair_artifact_baseline": _candidate_pair_artifact_accuracy(test_examples),
        "rollout_collision_heuristic": _accuracy(_collision_heuristic_predictions(test_examples), labels),
        "progress_heuristic": _accuracy(_progress_heuristic_predictions(test_examples), labels),
        "stats_mlp": _top1(stats_logits, labels),
        "stats_mlp_audit": feature_fit.audit,
    }


def _candidate_source_metadata_only_accuracy(examples: Sequence[PlanExample]) -> float:
    if not examples:
        return 0.0
    family_gold: Dict[str, int] = {}
    family_total: Dict[str, int] = {}
    for example in examples:
        families = example.metadata.get("candidate_source_families")
        if not isinstance(families, list) or len(families) != len(example.candidates):
            continue
        for idx, family in enumerate(families):
            key = str(family)
            family_total[key] = family_total.get(key, 0) + 1
            if idx == int(example.label):
                family_gold[key] = family_gold.get(key, 0) + 1
    if not family_total:
        return _accuracy(np.zeros(len(examples), dtype=np.int64), _labels(examples))
    family_rate = {family: family_gold.get(family, 0) / max(1, total) for family, total in family_total.items()}
    preds = []
    for example in examples:
        families = example.metadata.get("candidate_source_families")
        if not isinstance(families, list) or len(families) != len(example.candidates):
            preds.append(0)
            continue
        scores = [family_rate.get(str(family), 0.0) for family in families]
        preds.append(int(np.argmax(scores)))
    return _accuracy(np.asarray(preds, dtype=np.int64), _labels(examples))


def _candidate_pair_artifact_accuracy(examples: Sequence[PlanExample]) -> float:
    if not examples:
        return 0.0
    labels = _labels(examples)
    medoid_preds = []
    for example in examples:
        rows = []
        final_actions = [int(c[-1]) if c else 0 for c in example.candidates]
        final_counts = {action: final_actions.count(action) for action in range(4)}
        for cand in example.candidates:
            actions = list(cand)
            counts = np.asarray(_action_counts(actions), dtype=np.float32) / max(1, len(actions))
            rows.append(
                np.asarray(
                    [
                        *counts.tolist(),
                        _bigram_reversal_rate(actions),
                        _turn_count(actions) / max(1, len(actions) - 1),
                        final_counts.get(int(actions[-1]) if actions else 0, 0) / max(1, len(example.candidates)),
                    ],
                    dtype=np.float32,
                )
            )
        matrix = np.stack(rows)
        center = matrix.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(matrix - center, axis=1)
        medoid_preds.append(int(np.argmin(distances)))
    return _accuracy(np.asarray(medoid_preds, dtype=np.int64), labels)


def _candidate_order_stress(
    model: PlanLatentVerifier,
    variant: PlanVariant,
    examples: Sequence[PlanExample],
    clean_logits: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: str,
    seed: int,
    permutations: int = 4,
) -> Dict[str, object]:
    clean_top1 = float(_top1(clean_logits, labels))
    values = []
    for idx in range(int(permutations)):
        controlled = apply_plan_control(examples, "candidate_order_shuffle_with_gold_remap", seed + idx)
        logits = predict_plan_logits(model, controlled, variant, batch_size, device)
        values.append(float(_top1(logits, _labels(controlled))))
    deltas = [abs(value - clean_top1) for value in values]
    return {
        "permutations": int(permutations),
        "top1_values": values,
        "mean_top1": float(mean(values)) if values else 0.0,
        "max_abs_top1_delta": float(max(deltas)) if deltas else 0.0,
    }


def _run_controls(
    model: PlanLatentVerifier,
    variant: PlanVariant,
    examples: Sequence[PlanExample],
    clean_logits: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: str,
    seed: int,
) -> Dict[str, object]:
    chance = 1.0 / max(1, len(examples[0].candidates) if examples else 1)
    controls: Dict[str, object] = {
        "benchmark": BENCHMARK,
        "variant": variant.name,
        "seed": seed,
        "chance": chance,
        "candidate_source_metadata_only_accuracy": _candidate_source_metadata_only_accuracy(examples),
        "candidate_pair_artifact_baseline": _candidate_pair_artifact_accuracy(examples),
        "length_only": _accuracy(_length_only_predictions(examples), labels),
        "unigram_action_stat": _accuracy(_unigram_predictions(examples), labels),
        "bigram_action_stat": _accuracy(_bigram_predictions(examples), labels),
    }
    for name in (
        "candidate_only",
        "state_plan_mismatch",
        "goal_shuffle",
        "rollout_mismatch",
        "candidate_order_shuffle_with_gold_remap",
        "candidate_identity_shuffle",
        "candidate_list_composition_control",
        "randomized_labels",
    ):
        controlled = apply_plan_control(examples, name, seed + 17_000 + len(controls))
        logits = predict_plan_logits(model, controlled, variant, batch_size, device)
        controls[name] = _metric_block(logits, _labels(controlled), controlled)
    has_final_state_evidence = bool(variant.include_final_state) or any(_canonical_view(view) in {"transition", "transition_tuple", "action_conditioned_transition", "hybrid_transition_summary"} for view in _expanded_views(variant))
    if has_final_state_evidence:
        controlled = apply_plan_control(examples, "final_state_mismatch", seed + 72_000)
        logits = predict_plan_logits(model, controlled, variant, batch_size, device)
        controls["final_state_mismatch"] = _metric_block(logits, _labels(controlled), controlled)
    if bool(variant.candidate_self_attention):
        controlled = apply_plan_control(examples, "candidate_only", seed + 81_000)
        logits = predict_plan_logits(model, controlled, variant, batch_size, device)
        controls["candidate_self_attention_only"] = _metric_block(logits, _labels(controlled), controlled)
    role_logits = predict_plan_logits(model, examples, variant, batch_size, device, condition="role_order_shuffle", seed=seed + 44_000)
    hidden_logits = predict_plan_logits(model, examples, variant, batch_size, device, condition="hidden_state_shuffle", seed=seed + 55_000)
    controls["role_order_shuffle"] = _metric_block(role_logits, labels, examples)
    controls["hidden_state_shuffle"] = _metric_block(hidden_logits, labels, examples)
    controls["candidate_order_invariance_delta"] = abs(float(_top1(clean_logits, labels)) - float(controls["candidate_order_shuffle_with_gold_remap"]["top1"]))
    controls["candidate_identity_invariance_delta"] = abs(float(_top1(clean_logits, labels)) - float(controls["candidate_identity_shuffle"]["top1"]))
    controls["candidate_order_stress"] = _candidate_order_stress(model, variant, examples, clean_logits, labels, batch_size, device, seed + 91_000)
    controls["role_order_invariance_delta"] = abs(float(_top1(clean_logits, labels)) - float(_top1(role_logits, labels)))
    clean_top1 = float(_top1(clean_logits, labels))
    shortcut_values = [
        float(controls["candidate_only"]["top1"]),
        float(controls["length_only"]),
        float(controls["unigram_action_stat"]),
        float(controls["bigram_action_stat"]),
        float(controls["candidate_source_metadata_only_accuracy"]),
    ]
    pair_artifact_near_chance = float(controls["candidate_pair_artifact_baseline"]) <= chance + 0.10
    if "candidate_self_attention_only" in controls:
        shortcut_values.append(float(controls["candidate_self_attention_only"]["top1"]))
        shortcut_values.append(float(controls["candidate_pair_artifact_baseline"]))
    controls["control_pass"] = {
        "candidate_only_near_chance": float(controls["candidate_only"]["top1"]) <= chance + 0.10,
        "length_only_near_chance": float(controls["length_only"]) <= chance + 0.10,
        "action_unigram_near_chance": float(controls["unigram_action_stat"]) <= chance + 0.10,
        "action_bigram_near_chance": float(controls["bigram_action_stat"]) <= chance + 0.10,
        "candidate_source_metadata_near_chance": float(controls["candidate_source_metadata_only_accuracy"]) <= chance + 0.10,
        "candidate_pair_artifact_near_chance": pair_artifact_near_chance if bool(variant.candidate_self_attention) else True,
        "shortcut_baselines_near_chance": max(shortcut_values) <= chance + 0.10,
        "state_plan_mismatch_collapses": float(controls["state_plan_mismatch"]["top1"]) <= chance + 0.15,
        "goal_shuffle_collapses": float(controls["goal_shuffle"]["top1"]) <= chance + 0.15,
        "rollout_mismatch_collapses": float(controls["rollout_mismatch"]["top1"]) <= chance + 0.15,
        "candidate_list_composition_collapses": float(controls["candidate_list_composition_control"]["top1"]) <= chance + 0.15,
        "final_state_mismatch_collapses": (float(controls["final_state_mismatch"]["top1"]) <= chance + 0.15) if "final_state_mismatch" in controls else True,
        "candidate_order_remap_invariance": float(controls["candidate_order_invariance_delta"]) <= 0.08,
        "candidate_identity_shuffle_invariance": float(controls["candidate_identity_invariance_delta"]) <= 0.08,
        "candidate_order_stress_invariance": float(controls["candidate_order_stress"]["max_abs_top1_delta"]) <= 0.08,
        "role_order_invariance": float(controls["role_order_invariance_delta"]) <= 0.08,
        "hidden_state_shuffle_collapses": float(controls["hidden_state_shuffle"]["top1"]) <= max(chance + 0.20, clean_top1 - 0.05),
        "randomized_labels_collapses": float(controls["randomized_labels"]["top1"]) <= chance + 0.15,
    }
    if "candidate_self_attention_only" in controls:
        controls["control_pass"]["candidate_self_attention_only_near_chance"] = float(controls["candidate_self_attention_only"]["top1"]) <= chance + 0.10
    controls["control_pass"]["overall"] = all(bool(value) for value in controls["control_pass"].values())
    return controls


def _run_ablations(
    model: PlanLatentVerifier,
    variant: PlanVariant,
    examples: Sequence[PlanExample],
    clean_logits: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: str,
    seed: int,
) -> List[Dict[str, object]]:
    clean_top1 = _top1(clean_logits, labels)
    rows = []
    for view in sorted(set(_canonical_view(view) for view in _expanded_views(variant))):
        logits = predict_plan_logits(model, examples, variant, batch_size, device, view_mask=view, seed=seed + 61_000)
        top1 = _top1(logits, labels)
        rows.append(
            {
                "benchmark": BENCHMARK,
                "variant": variant.name,
                "seed": seed,
                "view": view,
                "clean_top1": float(clean_top1),
                "ablated_top1": float(top1),
                "drop": float(clean_top1 - top1),
                "meaningful_drop": bool(clean_top1 - top1 >= 0.05),
            }
        )
    return rows


def _run_overfit_gates(
    variant: PlanVariant,
    dataset_config: PlanDatasetConfig,
    training: PlanTrainingConfig,
    seed: int,
    device: str,
    gates: Sequence[Tuple[str, int, int, float]],
) -> List[Dict[str, object]]:
    rows = []
    for gate_name, candidates, examples, threshold in gates:
        gate_config = replace(dataset_config, num_candidates=int(candidates), train_examples=max(int(examples), 4), dev_examples=4, test_examples=4)
        splits = build_plan_splits(gate_config, seed=seed + 1000 + int(candidates))
        train_examples = splits["train"][: int(examples)]
        gate_variant = replace(variant)
        fit = fit_plan_verifier(
            train_examples,
            train_examples,
            gate_variant,
            replace(training, batch_size=min(int(training.batch_size), max(1, len(train_examples))), objective=variant.objective),
            seed + 91_000 + int(candidates),
            device,
            True,
            f"overfit__{variant.name}__{gate_name}",
        )
        logits = predict_plan_logits(fit.model, train_examples, gate_variant, training.batch_size, device)
        acc = _top1(logits, _labels(train_examples))
        rows.append(
            {
                "variant": variant.name,
                "seed": seed,
                "gate": gate_name,
                "candidate_count": int(candidates),
                "examples": len(train_examples),
                "train_accuracy": float(acc),
                "threshold": float(threshold),
                "pass": bool(acc >= float(threshold)),
                "gate_required": True,
                "history": fit.history,
                "audit": fit.audit,
            }
        )
        if acc < float(threshold):
            break
    return rows


def _phase0_sanity(variant: PlanVariant, splits: Dict[str, Sequence[PlanExample]], seed: int) -> Dict[str, object]:
    examples = [example for part in splits.values() for example in part[: min(8, len(part))]]
    remapped = apply_plan_control(examples[:4], "candidate_order_shuffle_with_gold_remap", seed + 500)
    return {
        "benchmark": BENCHMARK,
        "variant": variant.name,
        "seed": seed,
        "inputs_have_candidates": all(len(example.candidates) > 1 for example in examples),
        "candidate_order_remap_preserves_gold": all(example.goal_reached[example.label] for example in remapped),
        "no_source_metadata_visible": all(not bool(example.metadata.get("model_visible_candidate_sources", True)) for example in examples),
        "no_gold_label_as_input": all(not bool(example.metadata.get("model_visible_gold_label", True)) for example in examples),
        "no_simulator_success_flag_as_input": all(not bool(example.metadata.get("model_visible_simulator_success_flag", True)) for example in examples),
        "locked_baseline_has_no_final_state_tokens": not bool(variant.include_final_state) if variant.name == LOCKED_BASELINE else True,
        "locked_baseline_has_no_outcome_tokens": not bool(variant.include_outcome_tokens) if variant.name == LOCKED_BASELINE else True,
        "view_count": len(_expanded_views(variant)),
        "candidate_count": sorted({len(example.candidates) for example in examples}),
        "pass": True,
    }


def _metric_block(logits: np.ndarray, labels: np.ndarray, examples: Sequence[PlanExample]) -> Dict[str, object]:
    if logits.size == 0:
        return {"top1": 0.0, "top2": 0.0, "top3": 0.0, "mrr": 0.0, "mean_gold_rank": 0.0, "gold_ranks": []}
    order = np.argsort(-logits, axis=1)
    ranks = [int(np.where(row == int(label))[0][0]) + 1 for row, label in zip(order, labels)]
    preds = order[:, 0]
    margins = []
    for i, label in enumerate(labels):
        wrong = np.delete(logits[i], int(label))
        margins.append(float(logits[i, int(label)] - (np.max(wrong) if wrong.size else logits[i, int(label)])))
    return {
        "top1": _accuracy(preds, labels),
        "top2": float(np.mean([rank <= 2 for rank in ranks])) if ranks else 0.0,
        "top3": float(np.mean([rank <= 3 for rank in ranks])) if ranks else 0.0,
        "mrr": float(np.mean([1.0 / rank for rank in ranks])) if ranks else 0.0,
        "mean_gold_rank": float(np.mean(ranks)) if ranks else 0.0,
        "gold_ranks": ranks,
        "mean_gold_vs_best_wrong_logit_margin": float(np.mean(margins)) if margins else 0.0,
        "min_gold_vs_best_wrong_logit_margin": float(np.min(margins)) if margins else 0.0,
        "selected_plan_success_rate": float(np.mean([example.goal_reached[int(pred)] for pred, example in zip(preds, examples)])) if examples else 0.0,
        "by_obstacle_count": _accuracy_by_key(preds, labels, [str(len(e.obstacles)) for e in examples]),
        "by_start_goal_distance": _accuracy_by_key(preds, labels, [str(e.metadata.get("manhattan_start_goal", 0)) for e in examples]),
        "by_collision_count_selected": _accuracy_by_key(preds, labels, [str(sum(e.collisions[int(pred)])) for pred, e in zip(preds, examples)]),
    }


def _attention_summaries(model: PlanLatentVerifier, variant: PlanVariant, examples: Sequence[PlanExample], batch_size: int, device: str, seed: int) -> List[Dict[str, object]]:
    if not examples:
        return []
    output = predict_plan_logits(model, examples, variant, batch_size, device, seed=seed, return_attention=True)
    if not isinstance(output, tuple):
        return []
    _logits, rows = output
    for row in rows:
        row["benchmark"] = BENCHMARK
        row["variant"] = variant.name
        row["seed"] = seed
    return rows


def _summarize_attention(output: Dict[str, object], examples: Sequence[PlanExample], variant: PlanVariant, logits: np.ndarray) -> List[Dict[str, object]]:
    weights = output.get("attention_weights", [])
    if not weights:
        return []
    views = _expanded_views(variant)
    rows = []
    preds = np.argmax(logits, axis=1)
    for ex_idx, example in enumerate(examples):
        blocks = []
        for block_idx, block_weights in enumerate(weights):
            w = block_weights[ex_idx].detach().float().cpu().numpy()
            # candidates, heads, query tokens, evidence tokens
            mass_per_candidate = w.sum(axis=(1, 2, 3)) / max(1, w.shape[1] * w.shape[2])
            entropy = float(-np.sum(w.reshape(-1) * np.log(np.clip(w.reshape(-1), 1e-9, 1.0))) / max(1, w.shape[0]))
            blocks.append(
                {
                    "block": int(block_idx),
                    "attention_mass_per_candidate": {str(i): float(mass_per_candidate[i]) for i in range(len(mass_per_candidate))},
                    "attention_entropy": entropy,
                    "views": list(views),
                }
            )
        rows.append(
            {
                "type": "plan_arch1_attention_summary",
                "example_id": example.id,
                "predicted_candidate": int(preds[ex_idx]),
                "gold_candidate": int(example.label),
                "candidate_probabilities": _softmax_np(logits[ex_idx]).tolist(),
                "blocks": blocks,
            }
        )
    return rows


def _error_rows(
    phase: str,
    variant: PlanVariant,
    seed: int,
    examples: Sequence[PlanExample],
    train_logits: np.ndarray,
    frozen_logits: np.ndarray,
    baselines: Dict[str, object],
    limit: int,
) -> List[Dict[str, object]]:
    labels = _labels(examples)
    train_pred = np.argmax(train_logits, axis=1)
    frozen_pred = np.argmax(frozen_logits, axis=1)
    rows = []
    for index, example in enumerate(examples):
        if len(rows) >= int(limit):
            break
        category = None
        if train_pred[index] == labels[index] and frozen_pred[index] != labels[index]:
            category = "trainable_right_frozen_wrong"
        elif train_pred[index] != labels[index] and frozen_pred[index] == labels[index]:
            category = "frozen_right_trainable_wrong"
        elif train_pred[index] != labels[index]:
            category = "all_models_wrong"
        if category is None:
            continue
        rows.append(
            {
                "type": "plan_arch1_error_case",
                "phase": phase,
                "category": category,
                "variant": variant.name,
                "seed": seed,
                "example_id": example.id,
                "start": list(example.start),
                "goal": list(example.goal),
                "obstacles": [list(cell) for cell in example.obstacles],
                "gold_candidate_index": int(example.label),
                "predicted_candidate_index": int(train_pred[index]),
                "frozen_candidate_index": int(frozen_pred[index]),
                "candidate_probabilities": _softmax_np(train_logits[index]).tolist(),
                "selected_plan_success": bool(example.goal_reached[int(train_pred[index])]),
                "gold_plan_success": bool(example.goal_reached[int(example.label)]),
                "baseline_snapshot": {key: value for key, value in baselines.items() if isinstance(value, (int, float, str))},
            }
        )
    return rows


def _leaderboard(rows: Sequence[Dict[str, object]], controls: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    control_by_key = {(row["phase"], row["variant"], row["seed"]): row for row in controls}
    leaders = []
    for row in rows:
        if row.get("status") != "completed":
            leaders.append({"phase": row.get("phase"), "variant": row.get("variant"), "seed": row.get("seed"), "status": row.get("status"), "score": -999.0})
            continue
        control = control_by_key.get((row["phase"], row["variant"], row["seed"]), {})
        trainable = float(row["trainable"]["top1"])
        frozen = float(row["frozen"]["top1"])
        delta = float(row["delta_trainable_minus_frozen"])
        candidate_only = float(control.get("candidate_only", {}).get("top1", 1.0))
        controls_pass = bool(control.get("control_pass", {}).get("overall", False))
        baseline_best = _best_shortcut_baseline(row.get("baselines", {}))
        score = trainable + delta - 0.5 * candidate_only + (0.2 if controls_pass else -1.0) - max(0.0, baseline_best - trainable)
        leaders.append(
            {
                "phase": row["phase"],
                "variant": row["variant"],
                "family": row.get("family"),
                "seed": row["seed"],
                "status": row["status"],
                "trainable_top1": trainable,
                "frozen_top1": frozen,
                "delta": delta,
                "selected_plan_success": float(row["trainable"].get("selected_plan_success_rate", 0.0)),
                "candidate_only": candidate_only,
                "shortcut_best": baseline_best,
                "controls_pass": controls_pass,
                "delta_vs_locked_p1_reference": row.get("delta_vs_locked_p1_reference"),
                "param_count": int(row.get("param_count", 0)),
                "training_time_seconds": float(row.get("training_time_seconds", 0.0)),
                "score": score,
            }
        )
    leaders.sort(key=lambda item: float(item.get("score", -999.0)), reverse=True)
    return leaders


def _phase_summary(rows: Sequence[Dict[str, object]], controls: Sequence[Dict[str, object]], phase: str, min_seed_wins: int, min_delta: float) -> Dict[str, object]:
    phase_rows = [row for row in rows if row.get("phase") == phase and row.get("status") == "completed"]
    by_variant: Dict[str, List[Dict[str, object]]] = {}
    for row in phase_rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    summaries = []
    for variant, variant_rows in by_variant.items():
        deltas = [float(row["delta_trainable_minus_frozen"]) for row in variant_rows]
        trainable = [float(row["trainable"]["top1"]) for row in variant_rows]
        frozen = [float(row["frozen"]["top1"]) for row in variant_rows]
        variant_controls = [row for row in controls if row.get("phase") == phase and row.get("variant") == variant]
        controls_pass = all(bool(row.get("control_pass", {}).get("overall", False)) for row in variant_controls)
        summaries.append(
            {
                "variant": variant,
                "phase": phase,
                "seeds": [int(row["seed"]) for row in variant_rows],
                "mean_trainable_top1": _mean(trainable),
                "std_trainable_top1": _std(trainable),
                "mean_frozen_top1": _mean(frozen),
                "mean_delta": _mean(deltas),
                "bootstrap_ci_delta": _bootstrap_ci(deltas),
                "seeds_trainable_beats_frozen": int(sum(delta > 0.0 for delta in deltas)),
                "controls_pass_all": controls_pass,
                "screen_passed": bool(sum(delta > 0.0 for delta in deltas) >= int(min_seed_wins) and _mean(deltas) >= float(min_delta) and controls_pass),
            }
        )
    summaries.sort(key=lambda item: float(item["mean_delta"]), reverse=True)
    return {"phase": phase, "variant_summaries": summaries, "ran": bool(phase_rows)}


def _select_medium_variants(summary: Dict[str, object], variants: Sequence[PlanVariant], max_count: int) -> List[PlanVariant]:
    passed = [row for row in summary.get("variant_summaries", []) if row.get("screen_passed")]
    passed.sort(key=lambda row: float(row.get("mean_delta", 0.0)), reverse=True)
    names = {str(row["variant"]) for row in passed[: int(max_count)]}
    return [variant for variant in variants if variant.name in names]


def _failure_taxonomy(rows: Sequence[Dict[str, object]], controls: Sequence[Dict[str, object]], overfit_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    counts: Dict[str, int] = {}
    for row in rows:
        if row.get("status") == "failed_overfit_gate":
            key = f"failed_overfit::{row.get('failure_reason')}"
            counts[key] = counts.get(key, 0) + 1
        elif row.get("status") == "completed":
            if float(row.get("delta_trainable_minus_frozen", 0.0)) <= 0.0:
                counts["did_not_beat_frozen"] = counts.get("did_not_beat_frozen", 0) + 1
            if float(row.get("frozen", {}).get("top1", 0.0)) >= float(row.get("trainable", {}).get("top1", 0.0)) and float(row.get("frozen", {}).get("top1", 0.0)) > 0.30:
                counts["frozen_suspiciously_high"] = counts.get("frozen_suspiciously_high", 0) + 1
            if _best_shortcut_baseline(row.get("baselines", {})) >= float(row.get("trainable", {}).get("top1", 0.0)):
                counts["shortcut_baseline_explains_result"] = counts.get("shortcut_baseline_explains_result", 0) + 1
    for row in controls:
        if not bool(row.get("control_pass", {}).get("overall", False)):
            counts["control_failure"] = counts.get("control_failure", 0) + 1
    return {
        "counts": dict(sorted(counts.items())),
        "failed_overfit_gates": [row for row in overfit_rows if not row.get("pass")],
        "failure_reasons": {
            "failed_overfit": "variant did not satisfy micro-overfit gate",
            "did_not_beat_frozen": "trainable shared model did not improve over exact frozen same architecture",
            "control_failure": "shortcut/mismatch/shuffle/invariance control failed",
            "shortcut_baseline_explains_result": "simple candidate/action/rollout heuristic matched or exceeded the model",
            "frozen_suspiciously_high": "frozen same-architecture comparator was high enough to quarantine the mechanism",
        },
    }


def _overall_summary(
    rows: Sequence[Dict[str, object]],
    controls: Sequence[Dict[str, object]],
    leaderboard: Sequence[Dict[str, object]],
    cheap_summary: Dict[str, object],
    medium_summary: Dict[str, object],
    ablations: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    best = next((row for row in leaderboard if row.get("status") == "completed"), {})
    medium_gates = _medium_gates(rows, controls, ablations)
    return {
        "best_variant": best.get("variant"),
        "best_phase": best.get("phase"),
        "best_trainable_top1": float(best.get("trainable_top1", 0.0)) if best else 0.0,
        "best_delta": float(best.get("delta", 0.0)) if best else 0.0,
        "cheap_summary": cheap_summary,
        "medium_summary": medium_summary,
        "medium_ready_variant": _medium_ready_variant(medium_gates),
        "medium_gates": medium_gates,
        "final_validation_launched": False,
        "allowed_claim_if_final_gates_eventually_pass": (
            "A searched variant of shared-weight latent candidate evaluation improves controlled candidate-plan ranking "
            "over the locked P1 baseline and exact frozen comparator while passing shortcut, mismatch, shuffle, leakage, and invariance controls."
        ),
    }


def _medium_gates(rows: Sequence[Dict[str, object]], controls: Sequence[Dict[str, object]], ablations: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, bool]]:
    medium_rows = [row for row in rows if row.get("phase") == "medium" and row.get("status") == "completed"]
    if not medium_rows:
        return {}
    by_variant: Dict[str, List[Dict[str, object]]] = {}
    for row in medium_rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    gates = {}
    for variant, variant_rows in by_variant.items():
        deltas = [float(row["delta_trainable_minus_frozen"]) for row in variant_rows]
        controls_v = [row for row in controls if row.get("phase") == "medium" and row.get("variant") == variant]
        ablations_v = [row for row in ablations if row.get("variant") == variant]
        chance = 1.0 / max(1, int(variant_rows[0].get("candidate_count", 8)))
        ci = _bootstrap_ci(deltas)
        gates[variant] = {
            "trainable_beats_frozen_on_at_least_4_of_5_seeds": len(deltas) >= 5 and sum(delta > 0.0 for delta in deltas) >= 4,
            "mean_trainable_frozen_delta_at_least_0_15": _mean(deltas) >= 0.15,
            "bootstrap_ci_lower_bound_gt_0": float(ci[0]) > 0.0,
            "top1_above_random": _mean([float(row["trainable"]["top1"]) for row in variant_rows]) > chance,
            "candidate_only_near_chance": all(bool(row.get("control_pass", {}).get("candidate_only_near_chance", False)) for row in controls_v),
            "shortcut_baselines_near_chance": all(bool(row.get("control_pass", {}).get("shortcut_baselines_near_chance", False)) for row in controls_v),
            "state_plan_mismatch_collapses": all(bool(row.get("control_pass", {}).get("state_plan_mismatch_collapses", False)) for row in controls_v),
            "goal_shuffle_collapses": all(bool(row.get("control_pass", {}).get("goal_shuffle_collapses", False)) for row in controls_v),
            "rollout_mismatch_collapses": all(bool(row.get("control_pass", {}).get("rollout_mismatch_collapses", False)) for row in controls_v),
            "candidate_list_composition_collapses": all(bool(row.get("control_pass", {}).get("candidate_list_composition_collapses", False)) for row in controls_v),
            "candidate_order_remap_passes": all(bool(row.get("control_pass", {}).get("candidate_order_remap_invariance", False)) for row in controls_v),
            "candidate_order_stress_passes": all(bool(row.get("control_pass", {}).get("candidate_order_stress_invariance", False)) for row in controls_v),
            "role_order_remap_passes": all(bool(row.get("control_pass", {}).get("role_order_invariance", False)) for row in controls_v),
            "hidden_state_shuffle_collapses": all(bool(row.get("control_pass", {}).get("hidden_state_shuffle_collapses", False)) for row in controls_v),
            "randomized_labels_collapse": all(bool(row.get("control_pass", {}).get("randomized_labels_collapses", False)) for row in controls_v),
            "ablation_meaningful": any(bool(row.get("meaningful_drop", False)) for row in ablations_v),
            "no_shortcut_baseline_explains_result": all(_best_shortcut_baseline(row.get("baselines", {})) < float(row["trainable"]["top1"]) for row in variant_rows),
        }
    return gates


def _write_outputs(
    result: Dict[str, object],
    results_path: Path,
    controls_path: Path,
    leaderboard_path: Path,
    failure_path: Path,
    overfit_path: Path,
    ablation_path: Path,
    attention_path: Path,
    error_path: Path,
    compute_path: Path,
    report_path: Path,
) -> None:
    for path in (results_path, controls_path, leaderboard_path, failure_path, overfit_path, ablation_path, attention_path, error_path, compute_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(_without_large_logs(result), indent=2, sort_keys=True), encoding="utf-8")
    controls_path.write_text(json.dumps({"controls": result["controls"], "medium_gates": result["summary"].get("medium_gates", {})}, indent=2, sort_keys=True), encoding="utf-8")
    failure_path.write_text(json.dumps(result["failure_taxonomy"], indent=2, sort_keys=True), encoding="utf-8")
    overfit_path.write_text(json.dumps({"overfit_curves": result["overfit_curves"]}, indent=2, sort_keys=True), encoding="utf-8")
    ablation_path.write_text(json.dumps({"ablation_results": result["ablation_results"]}, indent=2, sort_keys=True), encoding="utf-8")
    attention_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in result["attention_summaries"]) + ("\n" if result["attention_summaries"] else ""), encoding="utf-8")
    error_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in result["error_cases"]) + ("\n" if result["error_cases"] else ""), encoding="utf-8")
    compute_path.write_text(json.dumps({"compute_metrics": result["compute_metrics"]}, indent=2, sort_keys=True), encoding="utf-8")
    with leaderboard_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = sorted({key for row in result["leaderboard"] for key in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["leaderboard"])
    report_path.write_text(_render_report(result), encoding="utf-8")


def _render_report(result: Dict[str, object]) -> str:
    leaderboard = result.get("leaderboard", [])
    failures = result.get("failure_taxonomy", {}).get("counts", {})
    lines = [
        "# PLAN-ARCH-1 Empirical Architecture Discovery",
        "",
        "## Scope",
        "",
        "- Bounded architecture search for latent candidate-plan evaluation on synthetic gridworld candidate-plan verification.",
        "- This is not final validation, autonomous planning, ARC solving, world modeling, or general agentic AI.",
        "- Exact frozen same-architecture comparators, shortcut baselines, mismatch controls, candidate-order remap, role-order remap, and hidden-state shuffle controls are required.",
        "- Final validation was not launched.",
        "",
        "## Leaderboard",
        "",
        "| rank | phase | variant | trainable | frozen | delta | success | shortcut | controls |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(leaderboard[:30], start=1):
        lines.append(
            f"| {rank} | {row.get('phase')} | {row.get('variant')} | {float(row.get('trainable_top1', 0.0)):.4f} | "
            f"{float(row.get('frozen_top1', 0.0)):.4f} | {float(row.get('delta', 0.0)):.4f} | "
            f"{float(row.get('selected_plan_success', 0.0)):.4f} | {float(row.get('shortcut_best', 0.0)):.4f} | `{bool(row.get('controls_pass', False))}` |"
        )
    lines.extend(["", "## Failure Taxonomy", "", "```json", json.dumps(failures, indent=2, sort_keys=True), "```", ""])
    lines.extend(["## Medium Gates", "", "```json", json.dumps(result.get("summary", {}).get("medium_gates", {}), indent=2, sort_keys=True), "```", ""])
    lines.extend(
        [
            "## Required Answers",
            "",
            f"1. Which architecture variant beats P1_rollout_no_final? `{_answer_beats_p1(result)}`.",
            f"2. Which mechanism mattered most? `{_answer_mechanism(result)}`.",
            f"3. Did any variant improve trainable-frozen delta? `{_answer_delta(result)}`.",
            f"4. Did any variant improve absolute selected-plan success? `{_answer_success(result)}`.",
            f"5. Did any variant reduce frozen performance while increasing trainable performance? `{_answer_reduce_frozen(result)}`.",
            f"6. Did any variant fail because frozen was suspiciously high? `{bool(failures.get('frozen_suspiciously_high', 0))}`.",
            f"7. Did pyramid become valid after controls? `{_answer_pyramid(result)}`.",
            f"8. Did stacking help or overfit? `{_answer_stacking(result)}`.",
            f"9. Did multi-avenue clones help? `{_answer_multi_avenue(result)}`.",
            f"10. Which variants failed and why? `{failures}`.",
            f"11. Is there a medium-ready variant? `{result.get('summary', {}).get('medium_ready_variant')}`.",
            "12. Should the winning mechanism be transferred back to real-code or ARC-hybrid? `only after medium gates pass; current smoke results are architecture-search diagnostics`.",
            "",
            "## Claim Boundary",
            "",
            "- Do not claim autonomous planning.",
            "- Do not claim world modeling.",
            "- Do not claim general agentic AI.",
            f"- Allowed claim only if final gates eventually pass: {result.get('summary', {}).get('allowed_claim_if_final_gates_eventually_pass')}",
        ]
    )
    return "\n".join(lines) + "\n"


def _variant_plan(configured: object | None = None) -> List[PlanVariant]:
    variants = [
        PlanVariant(
            name=LOCKED_BASELINE,
            family="locked_baseline",
            description="Shared-weight state/goal/obstacle/action/rollout verifier; rollout omits final state and outcome tokens.",
            view_names=("state", "goal", "obstacle", "action", "rollout"),
        ),
        PlanVariant(
            name="P1_rollout_with_final_state",
            family="locked_baseline",
            description="Shared-weight state/goal/obstacle/action/rollout verifier with final rollout state visible and no outcome flag.",
            view_names=("state", "goal", "obstacle", "action", "rollout"),
            include_final_state=True,
        ),
        PlanVariant(
            name="transition_tuple_verifier",
            family="transition_representation",
            description="Sparse transition tuple evidence: state_t, action_t, state_t_plus_1, delta, and visible contact.",
            view_names=("state", "goal", "obstacle", "transition_tuple"),
            include_final_state=True,
        ),
        PlanVariant(
            name="candidate_token_direct_transition_tokens",
            family="transition_representation",
            description="Candidate-token-direct verifier augmented with rollout and transition tuple tokens.",
            view_names=("state", "goal", "obstacle", "action", "rollout", "transition_tuple"),
            include_final_state=True,
        ),
        PlanVariant(
            name="simple_cross_attention_verifier_baseline",
            family="cross_attention_baseline",
            description="Small simple cross-attention verifier baseline using rollout and transition summary evidence.",
            view_names=("state", "goal", "rollout", "hybrid_transition_summary"),
            model_dim=32,
            num_heads=2,
            ff_dim=64,
            include_final_state=True,
        ),
        PlanVariant(
            name="oracle_success_flag_sanity",
            family="diagnostic_oracle",
            description="Diagnostic-only exact success flag sanity check.",
            view_names=("state", "goal", "rollout"),
            include_final_state=True,
            include_outcome_tokens=True,
            diagnostic_only=True,
        ),
        PlanVariant(
            name="Q_candidate_self_attention",
            family="candidate_query",
            description="Candidate self-attention before evidence query.",
            view_names=("state", "goal", "obstacle", "action", "rollout"),
            candidate_self_attention=True,
        ),
        PlanVariant(
            name="Q_candidate_self_attention_plus_rollout_no_final",
            family="candidate_query",
            description="Candidate self-attention with the locked no-final-state rollout evidence set.",
            view_names=("state", "goal", "obstacle", "action", "rollout"),
            candidate_self_attention=True,
        ),
        PlanVariant(
            name="Q_candidate_self_attention_with_stronger_mismatch_controls",
            family="candidate_query",
            description="Candidate self-attention retest under the PLAN-ARCH-1.1 stronger mismatch and list-composition controls.",
            view_names=("state", "goal", "obstacle", "action", "rollout"),
            candidate_self_attention=True,
        ),
        PlanVariant(
            name="Q_bidirectional_candidate_evidence",
            family="candidate_query",
            description="Bidirectional candidate/evidence blocks.",
            view_names=("state", "goal", "obstacle", "action", "rollout"),
            query_mechanism="bidirectional_candidate_evidence",
            bidirectional=True,
        ),
        PlanVariant(
            name="S_stack2_rollout",
            family="stacking",
            description="Two candidate/evidence coordination blocks.",
            view_names=("state", "goal", "obstacle", "action", "rollout"),
            coordination_blocks=2,
        ),
        PlanVariant(
            name="S_stack3_rollout",
            family="stacking",
            description="Three candidate/evidence coordination blocks.",
            view_names=("state", "goal", "obstacle", "action", "rollout"),
            coordination_blocks=3,
        ),
        PlanVariant(
            name="V_risk_progress_views",
            family="clone_view",
            description="Adds risk and progress clones without explicit success/outcome tokens.",
            view_names=("state", "goal", "obstacle", "action", "rollout", "risk", "progress"),
        ),
        PlanVariant(
            name="M_two_avenues_per_view",
            family="multi_avenue",
            description="Two avenues per state/goal/obstacle/action/rollout view.",
            view_names=("state", "goal", "obstacle", "action", "rollout"),
            avenues_per_view=2,
        ),
        PlanVariant(
            name="P_leaf_plus_composed",
            family="pyramid",
            description="Pyramidal leaf plus composed latent state query. Quarantined unless frozen stays low and controls pass.",
            view_names=("state_goal", "candidate_future", "risk_adjusted_future", "plan_value"),
            pyramid=True,
            residual_level_scoring=True,
        ),
        PlanVariant(
            name="L_pairwise_ranking",
            family="loss",
            description="Pairwise ranking objective over candidates.",
            view_names=("state", "goal", "obstacle", "action", "rollout"),
            objective="pairwise_ranking",
        ),
        PlanVariant(
            name="L_aux_collision",
            family="loss",
            description="Cross-entropy plus auxiliary no-collision prediction generated by simulator labels, not inputs.",
            view_names=("state", "goal", "obstacle", "action", "rollout", "risk"),
            objective="aux_collision",
        ),
        PlanVariant(
            name="R_partial_rollout_prefix",
            family="rollout_representation",
            description="Uses rollout prefix only.",
            view_names=("state", "goal", "obstacle", "action", "partial_rollout"),
        ),
        PlanVariant(
            name="R_long_short_rollout",
            family="rollout_representation",
            description="Separate short-term and long-term rollout avenues.",
            view_names=("state", "goal", "obstacle", "action", "rollout_short", "rollout_long"),
        ),
    ]
    if configured:
        names = {str(name) for name in configured if isinstance(name, str)}
        variants = [variant for variant in variants if variant.name in names] or variants
    return variants


def _row_base(phase: str, variant: PlanVariant, seed: int, gates_pass: bool) -> Dict[str, object]:
    return {
        "benchmark": BENCHMARK,
        "stage_name": STAGE_NAME,
        "phase": phase,
        "variant": variant.name,
        "family": variant.family,
        "description": variant.description,
        "seed": seed,
        "view_names": list(variant.view_names),
        "expanded_view_names": list(_expanded_views(variant)),
        "representation": variant.representation,
        "query_mechanism": variant.query_mechanism,
        "objective": variant.objective,
        "coordination_blocks": int(variant.coordination_blocks),
        "avenues_per_view": int(variant.avenues_per_view),
        "pyramid": bool(variant.pyramid),
        "diagnostic_only": bool(variant.diagnostic_only),
        "overfit_gates_pass": bool(gates_pass),
    }


def _expanded_views(variant: PlanVariant) -> Tuple[str, ...]:
    views = []
    for view in variant.view_names:
        for avenue in range(max(1, int(variant.avenues_per_view))):
            views.append(str(view) if int(variant.avenues_per_view) == 1 else f"{view}:avenue{avenue}")
    return tuple(views)


def _canonical_view(view: str) -> str:
    return str(view).split(":", 1)[0]


def _dataset_config(data: object) -> PlanDatasetConfig:
    values = dict(data or {})
    allowed = set(PlanDatasetConfig.__dataclass_fields__.keys())
    return PlanDatasetConfig(**{key: value for key, value in values.items() if key in allowed})


def _training_config(data: object) -> PlanTrainingConfig:
    values = dict(data or {})
    allowed = set(PlanTrainingConfig.__dataclass_fields__.keys())
    return PlanTrainingConfig(**{key: value for key, value in values.items() if key in allowed})


def _feature_training_config(data: object) -> FeatureTrainingConfig:
    values = dict(data or {})
    allowed = set(FeatureTrainingConfig.__dataclass_fields__.keys())
    return FeatureTrainingConfig(**{key: value for key, value in values.items() if key in allowed})


def _micro_gates(data: object | None) -> List[Tuple[str, int, int, float]]:
    if not data:
        return [("4_examples_N2", 2, 4, 0.95), ("16_examples_N4", 4, 16, 0.90), ("64_examples_N8", 8, 64, 0.80)]
    rows = []
    for row in data:
        if isinstance(row, dict):
            rows.append((str(row.get("name", "gate")), int(row["candidate_count"]), int(row["examples"]), float(row["threshold"])))
    return rows


def _load_config(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _clear_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _labels(examples: Sequence[PlanExample]) -> np.ndarray:
    return np.asarray([example.label for example in examples], dtype=np.int64)


def _length_only_predictions(examples: Sequence[PlanExample]) -> np.ndarray:
    return np.asarray([int(np.argmin([len(c) for c in example.candidates])) for example in examples], dtype=np.int64)


def _unigram_predictions(examples: Sequence[PlanExample]) -> np.ndarray:
    preds = []
    for example in examples:
        direct = _direct_action_counts(example.start, example.goal, len(example.candidates[0]))
        scores = [float(np.sum(np.abs(np.asarray(_action_counts(c)) - np.asarray(direct)))) for c in example.candidates]
        preds.append(int(np.argmin(scores)))
    return np.asarray(preds, dtype=np.int64)


def _bigram_predictions(examples: Sequence[PlanExample]) -> np.ndarray:
    preds = []
    for example in examples:
        scores = [_bigram_reversal_rate(c) for c in example.candidates]
        preds.append(int(np.argmin(scores)))
    return np.asarray(preds, dtype=np.int64)


def _collision_heuristic_predictions(examples: Sequence[PlanExample]) -> np.ndarray:
    return np.asarray([int(np.argmin([sum(c) for c in example.collisions])) for example in examples], dtype=np.int64)


def _progress_heuristic_predictions(examples: Sequence[PlanExample]) -> np.ndarray:
    preds = []
    for example in examples:
        scores = []
        for progress in example.progress:
            visible_final = progress[-2] if len(progress) > 1 else progress[-1]
            scores.append(visible_final)
        preds.append(int(np.argmin(scores)))
    return np.asarray(preds, dtype=np.int64)


def _best_shortcut_baseline(baselines: Dict[str, object]) -> float:
    keys = (
        "candidate_only_index0",
        "length_only",
        "unigram_action_stat",
        "bigram_action_stat",
        "candidate_source_metadata_only",
        "rollout_collision_heuristic",
        "progress_heuristic",
        "stats_mlp",
    )
    values = [float(baselines.get(key, 0.0)) for key in keys if isinstance(baselines.get(key, 0.0), (int, float))]
    return max(values) if values else 0.0


def _accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(predictions.astype(np.int64) == labels.astype(np.int64))) if len(labels) else 0.0


def _top1(logits: np.ndarray, labels: np.ndarray) -> float:
    if logits.size == 0:
        return 0.0
    return _accuracy(np.argmax(logits, axis=1), labels)


def _accuracy_by_key(predictions: np.ndarray, labels: np.ndarray, keys: Sequence[str]) -> Dict[str, float]:
    out = {}
    for key in sorted(set(keys)):
        idx = [i for i, value in enumerate(keys) if value == key]
        out[str(key)] = _accuracy(np.asarray([predictions[i] for i in idx]), np.asarray([labels[i] for i in idx]))
    return out


def _softmax_np(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    values = values - np.max(values)
    exp = np.exp(values)
    return exp / np.clip(exp.sum(), 1e-12, None)


def _bootstrap_ci(values: Sequence[float], iterations: int = 1000) -> List[float]:
    if not values:
        return [0.0, 0.0]
    rng = np.random.default_rng(12345)
    arr = np.asarray(values, dtype=np.float64)
    means = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(iterations)]
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _chunks(values: Sequence[int], size: int) -> Iterable[List[int]]:
    for index in range(0, len(values), max(1, int(size))):
        yield list(values[index : index + max(1, int(size))])


def _batched(values: Sequence[PlanExample], size: int) -> Iterable[List[PlanExample]]:
    for index in range(0, len(values), max(1, int(size))):
        yield list(values[index : index + max(1, int(size))])


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


def _flat_params(items: Sequence[Tuple[str, nn.Parameter]]) -> torch.Tensor:
    tensors = [parameter.detach().float().cpu().reshape(-1) for _name, parameter in items]
    return torch.cat(tensors) if tensors else torch.zeros(0)


def _flat_feature_params(model: nn.Module) -> torch.Tensor:
    tensors = [parameter.detach().float().reshape(-1) for parameter in model.parameters()]
    return torch.cat(tensors) if tensors else torch.zeros(0)


def _l2_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or a.numel() != b.numel():
        return 0.0
    return float(torch.linalg.vector_norm(a - b).item())


def _grad_norm(parameters: Sequence[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().pow(2).sum().cpu())
    return float(math.sqrt(total)) if total > 0 else 0.0


def _compatible_dim(dim: int, heads: int) -> int:
    dim = int(dim)
    heads = max(1, int(heads))
    if dim % heads == 0:
        return dim
    return int(math.ceil(dim / heads) * heads)


def _in_bounds(cell: Tuple[int, int], grid: int) -> bool:
    return 0 <= cell[0] < grid and 0 <= cell[1] < grid


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _action_counts(actions: Sequence[int]) -> List[int]:
    return [int(sum(1 for action in actions if int(action) == i)) for i in range(4)]


def _direct_action_counts(start: Tuple[int, int], goal: Tuple[int, int], length: int) -> List[int]:
    counts = [0, 0, 0, 0]
    vertical = goal[0] - start[0]
    horizontal = goal[1] - start[1]
    counts[1 if vertical > 0 else 0] += abs(vertical)
    counts[3 if horizontal > 0 else 2] += abs(horizontal)
    remaining = max(0, int(length) - sum(counts))
    counts[2] += remaining // 2
    counts[3] += remaining - remaining // 2
    return counts


def _bigram_reversal_rate(actions: Sequence[int]) -> float:
    if len(actions) < 2:
        return 0.0
    reverses = {(0, 1), (1, 0), (2, 3), (3, 2)}
    return float(sum((int(a), int(b)) in reverses for a, b in zip(actions, actions[1:]))) / float(len(actions) - 1)


def _turn_count(actions: Sequence[int]) -> int:
    if len(actions) < 2:
        return 0
    return int(sum(1 for a, b in zip(actions, actions[1:]) if int(a) != int(b)))


def _as_list(values: np.ndarray) -> List[float]:
    return [float(value) for value in values.tolist()]


def _blank_rollouts(example: PlanExample) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    return tuple(tuple((0, 0) for _ in rollout) for rollout in example.rollouts)


def _blank_collisions(example: PlanExample) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(0 for _ in collisions) for collisions in example.collisions)


def _blank_progress(example: PlanExample) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(0 for _ in progress) for progress in example.progress)


def _find_p1_reference(phase: str, seed: int) -> float | None:
    del phase, seed
    return None


def _first_failed_gate(rows: Sequence[Dict[str, object]]) -> str:
    for row in rows:
        if row.get("gate_required", True) and not row.get("pass"):
            return str(row.get("gate"))
    return "unknown"


def _medium_ready_variant(gates: Dict[str, Dict[str, bool]]) -> str | None:
    for variant, gate_map in gates.items():
        if all(bool(value) for value in gate_map.values()):
            return variant
    return None


def _without_large_logs(result: Dict[str, object]) -> Dict[str, object]:
    return {key: value for key, value in result.items() if key not in {"attention_summaries", "error_cases", "compute_metrics"}}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _answer_beats_p1(result: Dict[str, object]) -> str:
    rows = [row for row in result.get("rows", []) if row.get("status") == "completed"]
    p1 = [row for row in rows if row.get("variant") == LOCKED_BASELINE]
    if not p1:
        return "no_locked_p1_run"
    p1_best = max(float(row["trainable"]["top1"]) for row in p1)
    better = [row for row in rows if row.get("variant") != LOCKED_BASELINE and float(row["trainable"]["top1"]) > p1_best and bool(row.get("control_pass", {}).get("overall", False))]
    return better[0]["variant"] if better else "none"


def _answer_mechanism(result: Dict[str, object]) -> str:
    completed = [row for row in result.get("leaderboard", []) if row.get("status") == "completed"]
    return str(completed[0].get("family", "not_established")) if completed else "not_run"


def _answer_delta(result: Dict[str, object]) -> str:
    completed = [row for row in result.get("leaderboard", []) if row.get("status") == "completed"]
    if not completed:
        return "not_run"
    best = max(completed, key=lambda row: float(row.get("delta", 0.0)))
    return f"{float(best.get('delta', 0.0)) > 0.0} best={best.get('variant')} delta={float(best.get('delta', 0.0)):.4f}"


def _answer_success(result: Dict[str, object]) -> str:
    completed = [row for row in result.get("leaderboard", []) if row.get("status") == "completed"]
    if not completed:
        return "not_run"
    best = max(completed, key=lambda row: float(row.get("selected_plan_success", 0.0)))
    return f"{best.get('variant')} success={float(best.get('selected_plan_success', 0.0)):.4f}"


def _answer_reduce_frozen(result: Dict[str, object]) -> str:
    rows = [row for row in result.get("rows", []) if row.get("status") == "completed" and row.get("variant") != LOCKED_BASELINE]
    p1 = [row for row in result.get("rows", []) if row.get("status") == "completed" and row.get("variant") == LOCKED_BASELINE]
    if not rows or not p1:
        return "not_established"
    p1_frozen = _mean([float(row["frozen"]["top1"]) for row in p1])
    p1_train = _mean([float(row["trainable"]["top1"]) for row in p1])
    hits = [row["variant"] for row in rows if float(row["frozen"]["top1"]) < p1_frozen and float(row["trainable"]["top1"]) > p1_train]
    return ", ".join(hits) if hits else "none"


def _answer_pyramid(result: Dict[str, object]) -> str:
    rows = [row for row in result.get("leaderboard", []) if str(row.get("variant", "")).startswith("P_")]
    if not rows:
        return "not_run"
    valid = [row for row in rows if float(row.get("delta", 0.0)) > 0.0 and bool(row.get("controls_pass", False))]
    return valid[0]["variant"] if valid else "no"


def _answer_stacking(result: Dict[str, object]) -> str:
    rows = [row for row in result.get("leaderboard", []) if str(row.get("variant", "")).startswith("S_")]
    if not rows:
        return "not_run"
    best = max(rows, key=lambda row: float(row.get("delta", 0.0)))
    return f"{best.get('variant')} delta={float(best.get('delta', 0.0)):.4f} controls={bool(best.get('controls_pass', False))}"


def _answer_multi_avenue(result: Dict[str, object]) -> str:
    rows = [row for row in result.get("leaderboard", []) if str(row.get("variant", "")).startswith("M_")]
    if not rows:
        return "not_run"
    best = rows[0]
    return f"{best.get('variant')} delta={float(best.get('delta', 0.0)):.4f} controls={bool(best.get('controls_pass', False))}"


if __name__ == "__main__":
    main()
