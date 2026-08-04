from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, replace
from itertools import permutations
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.experiments.run_plan_arch1_4_learned_constraint_planning import (
    ACTION_DELTAS,
    FAMILIES,
    ROLE_IDS,
    ConstraintCloneVerifier,
    ConstraintDatasetConfig,
    ConstraintEvidence,
    ConstraintExample,
    ConstraintFitResult,
    ConstraintTrainingConfig,
    ConstraintVariant,
    _accuracy,
    _action_counts,
    _actions_via_waypoints,
    _balanced_order,
    _batched,
    _bigram_predictions,
    _chunks,
    _clear_cuda,
    _compatible_dim,
    _final_distance_predictions,
    _final_position_predictions,
    _flat_params,
    _grad_norm,
    _labels,
    _length_predictions,
    _l2_delta,
    _manhattan,
    _masked_mean,
    _pack_tokens,
    _set_seed,
    _simulate,
    _softmax_np,
    _top1,
    _unigram_predictions,
    _variant_plan,
)


BENCHMARK = "plan_arch1_5_learned_constraint_shortcut_repair"
DEFAULT_CONFIG = "configs/plan_arch1_5_learned_constraint_shortcut_repair.json"
DEFAULT_RESULTS = "results/plan_arch1_5_results.json"
DEFAULT_CONTROLS = "results/plan_arch1_5_controls.json"
DEFAULT_SHORTCUTS = "results/plan_arch1_5_shortcut_decomposition.json"
DEFAULT_COUNTERFACTUAL = "results/plan_arch1_5_counterfactual_candidate_audit.json"
DEFAULT_FAMILY = "results/plan_arch1_5_family_results.json"
DEFAULT_LEADERBOARD = "results/plan_arch1_5_variant_leaderboard.csv"
DEFAULT_FAILURES = "results/plan_arch1_5_failure_taxonomy.json"
DEFAULT_ATTENTION = "results/plan_arch1_5_attention_summaries.jsonl"
DEFAULT_ERRORS = "results/plan_arch1_5_error_cases.jsonl"
DEFAULT_COMPUTE = "results/plan_arch1_5_compute_metrics.json"
DEFAULT_REPORT = "reports/PLAN_ARCH1_5_LEARNED_CONSTRAINT_SHORTCUT_REPAIR.md"

FIELD_VOCABS = (24, 16, 16, 32, 96, 8, 16, 16)
FAMILY_IDS = {"color_zone": 1, "ordered_subgoal": 2, "key_door": 3}
NEAR_CHANCE_MARGIN = 0.10
COLLAPSE_MARGIN = 0.15
HURT_MARGIN = 0.08
DELTA_GATE = 0.10

BASE_SPECIAL_CELLS: Tuple[Tuple[int, int], ...] = ((1, 1), (1, 5), (5, 1), (5, 5))


class ProbeMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        bsz, cand_count, dim = features.shape
        return self.net(features.reshape(bsz * cand_count, dim)).reshape(bsz, cand_count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PLAN-ARCH-1.5 learned-constraint shortcut repair benchmark.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_RESULTS)
    parser.add_argument("--controls-output", default=DEFAULT_CONTROLS)
    parser.add_argument("--shortcut-output", default=DEFAULT_SHORTCUTS)
    parser.add_argument("--counterfactual-output", default=DEFAULT_COUNTERFACTUAL)
    parser.add_argument("--family-output", default=DEFAULT_FAMILY)
    parser.add_argument("--leaderboard-output", default=DEFAULT_LEADERBOARD)
    parser.add_argument("--failure-output", default=DEFAULT_FAILURES)
    parser.add_argument("--attention-output", default=DEFAULT_ATTENTION)
    parser.add_argument("--error-output", default=DEFAULT_ERRORS)
    parser.add_argument("--compute-output", default=DEFAULT_COMPUTE)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = _default_config(_load_config(Path(args.config)))
    if args.device:
        config["device"] = str(args.device)
    result = run_plan_arch1_5(config)
    _write_outputs(
        result,
        Path(args.output),
        Path(args.controls_output),
        Path(args.shortcut_output),
        Path(args.counterfactual_output),
        Path(args.family_output),
        Path(args.leaderboard_output),
        Path(args.failure_output),
        Path(args.attention_output),
        Path(args.error_output),
        Path(args.compute_output),
        Path(args.report),
    )


def run_plan_arch1_5(config: Dict[str, object]) -> Dict[str, object]:
    started = time.perf_counter()
    dataset_config = _dataset_config(config.get("dataset", {}))
    training = _training_config(config.get("training", {}))
    micro_training = _training_config({**dict(config.get("training", {})), **dict(config.get("micro_overfit_training", {}))})
    seeds = [int(seed) for seed in config.get("seeds", [0, 1, 2])]
    device = _resolve_device(str(config.get("device", "cpu")))
    variants = _variant_plan()
    print(f"plan-arch1.5: device={device} seeds={seeds} variants={len(variants)} no_medium_validation=True")

    rows: List[Dict[str, object]] = []
    controls: List[Dict[str, object]] = []
    shortcuts: List[Dict[str, object]] = []
    counterfactual_audits: List[Dict[str, object]] = []
    family_rows: List[Dict[str, object]] = []
    attention_rows: List[Dict[str, object]] = []
    error_rows: List[Dict[str, object]] = []
    compute_rows: List[Dict[str, object]] = []
    micro_rows: List[Dict[str, object]] = []

    for variant in variants:
        print(f"plan-arch1.5 micro variant={variant.name}")
        micro_rows.extend(_run_micro_overfit_gates_15(variant, dataset_config, micro_training, seeds[0] if seeds else 0, device))

    for seed in seeds:
        splits = build_repaired_constraint_splits(dataset_config, seed)
        shortcut_row = _shortcut_decomposition_row(splits, seed)
        shortcuts.append(shortcut_row)
        counterfactual_audits.append(_counterfactual_candidate_audit(splits, seed))
        family_rows.extend(_family_shortcut_rows(splits, shortcut_row, seed))
        labels = _labels(splits["test"])
        for variant in variants:
            print(f"plan-arch1.5 cheap variant={variant.name} seed={seed}")
            start = time.perf_counter()
            if variant.minimal:
                fit = fit_constraint_verifier_15(splits["train"], splits["dev"], variant, training, seed + 30_101, device, True, f"minimal15__{variant.name}")
                logits = predict_constraint_logits_15(fit.model, splits["test"], variant, training.batch_size, device)
                metric = _metric_block_15(logits, labels, splits["test"])
                control = _run_controls_15(fit.model, variant, splits, logits, labels, training.batch_size, device, seed, shortcut_row)
                row = _result_row_15("cheap", variant, seed, metric, None, None, fit, control, splits, time.perf_counter() - start)
            else:
                trainable = fit_constraint_verifier_15(splits["train"], splits["dev"], variant, training, seed + 10_101, device, True, f"trainable15__{variant.name}")
                frozen = fit_constraint_verifier_15(splits["train"], splits["dev"], variant, training, seed + 10_101, device, False, f"frozen15__{variant.name}")
                train_logits = predict_constraint_logits_15(trainable.model, splits["test"], variant, training.batch_size, device)
                frozen_logits = predict_constraint_logits_15(frozen.model, splits["test"], variant, training.batch_size, device)
                train_metric = _metric_block_15(train_logits, labels, splits["test"])
                frozen_metric = _metric_block_15(frozen_logits, labels, splits["test"])
                control = _run_controls_15(trainable.model, variant, splits, train_logits, labels, training.batch_size, device, seed, shortcut_row)
                row = _result_row_15("cheap", variant, seed, train_metric, frozen_metric, train_metric["top1"] - frozen_metric["top1"], trainable, control, splits, time.perf_counter() - start)
                row["frozen_audit"] = frozen.audit
                error_rows.extend(_error_rows_15(variant, seed, splits["test"], train_logits, frozen_logits, limit=20))
                attention_rows.extend(_attention_summaries_15(trainable.model, variant, splits["test"][: min(8, len(splits["test"]))], training.batch_size, device, seed))
            controls.append(control)
            rows.append(row)
            compute_rows.append(_compute_row_15(variant, seed, splits, row, time.perf_counter() - start))
            family_rows.extend(_family_variant_rows(row, control, splits["test"]))
            _clear_cuda()

    leaderboard = _leaderboard_15(rows, controls)
    failure_taxonomy = _failure_taxonomy_15(rows, controls, shortcuts, micro_rows)
    summary = _summary_15(rows, controls, shortcuts, leaderboard)
    compute_rows.append(
        {
            "benchmark": BENCHMARK,
            "status": "completed",
            "device": device,
            "total_runtime_seconds": float(time.perf_counter() - started),
            "medium_validation_launched": False,
        }
    )
    return {
        "metadata": {
            "benchmark": BENCHMARK,
            "scope": "learned-constraint shortcut repair benchmark; no medium validation launched",
            "device": device,
            "families": list(dataset_config.families),
            "claim_boundary": "No planning claim. No architecture improvement claim. This stage repairs the learned-constraint benchmark.",
            "medium_validation_launched": False,
        },
        "dataset_config": asdict(dataset_config),
        "training_config": asdict(training),
        "micro_training_config": asdict(micro_training),
        "variants": [asdict(variant) for variant in variants],
        "micro_overfit": micro_rows,
        "rows": rows,
        "controls": controls,
        "shortcut_decomposition": shortcuts,
        "counterfactual_candidate_audit": counterfactual_audits,
        "family_results": family_rows,
        "leaderboard": leaderboard,
        "failure_taxonomy": failure_taxonomy,
        "attention_summaries": attention_rows,
        "error_cases": error_rows,
        "compute_metrics": compute_rows,
        "summary": summary,
    }


def build_repaired_constraint_splits(config: ConstraintDatasetConfig, seed: int) -> Dict[str, List[ConstraintExample]]:
    counts = {"train": int(config.train_examples), "dev": int(config.dev_examples), "test": int(config.test_examples)}
    splits: Dict[str, List[ConstraintExample]] = {}
    rng = np.random.default_rng(seed)
    offset = 0
    for split, count in counts.items():
        rows = []
        for index in range(count):
            rows.append(_generate_repaired_example(config, split, index, int(rng.integers(0, 2_000_000_000)) + offset))
        splits[split] = rows
        offset += 100_000
    return splits


def _generate_repaired_example(config: ConstraintDatasetConfig, split: str, index: int, seed: int) -> ConstraintExample:
    rng = np.random.default_rng(seed)
    family = str(config.families[index % len(config.families)])
    if family not in FAMILIES:
        raise ValueError(f"unknown constraint family: {family}")
    grid = int(config.grid_size)
    start, goal = (0, 0), (grid - 1, grid - 1)
    role_cells = _role_cells_for_family(family, index)
    terrain = tuple((row, col, value - 1) for _name, row, col, value, _role_id in role_cells)
    templates = _candidate_templates(start, goal, role_cells, int(config.num_candidates), index, grid)
    target_idx = int((index // max(1, len(config.families)) + int(rng.integers(0, len(templates)))) % len(templates))
    target_order = templates[target_idx][1]
    records = []
    for order_idx, (actions, realized_order) in enumerate(templates):
        records.append((tuple(actions), _candidate_type(family, order_idx, realized_order, target_order), bool(realized_order == target_order)))
    balanced = _balance_repaired_records(records, start, int(config.plan_length), grid)
    ordered, label = _balanced_order(balanced, int(config.num_candidates), index, rng)
    candidates = tuple(record[0] for record in ordered)
    candidate_types = tuple(record[1] for record in ordered)
    rollouts = tuple(tuple(_simulate(start, actions, grid, tuple())) for actions in candidates)
    reached = tuple(bool(rollout[-1] == goal) for rollout in rollouts)
    satisfies = tuple(_matches_target_order(rollout, role_cells, target_order) and reached[idx] for idx, rollout in enumerate(rollouts))
    if sum(bool(value) for value in satisfies) != 1 or not satisfies[int(label)]:
        raise RuntimeError(f"invalid repaired example family={family} index={index} satisfies={satisfies} label={label}")
    constraint_examples = _constraint_examples_for(family, role_cells, target_order)
    metadata = {
        "candidate_order_randomized": True,
        "model_visible_candidate_sources": False,
        "model_visible_gold_label": False,
        "model_visible_success_flag": False,
        "all_candidates_same_length": len(set(len(c) for c in candidates)) == 1,
        "num_candidates_reaching_goal": int(sum(reached)),
        "all_candidates_reach_goal": bool(all(reached)),
        "all_candidates_same_endpoint": bool(len({rollout[-1] for rollout in rollouts}) == 1),
        "candidate_types": list(candidate_types),
        "role_cells": [list(item) for item in role_cells],
        "target_order": list(target_order),
        "target_order_index": int(target_idx),
        "task_rule": _rule_name(family, role_cells, target_order),
        "paired_counterfactual": True,
    }
    subgoal_a = (role_cells[0][1], role_cells[0][2]) if family == "ordered_subgoal" else None
    subgoal_b = (role_cells[1][1], role_cells[1][2]) if family == "ordered_subgoal" else None
    key = next(((row, col) for name, row, col, _value, _role in role_cells if name.startswith("key")), None)
    door = next(((row, col) for name, row, col, _value, _role in role_cells if name.startswith("door")), None)
    return ConstraintExample(
        id=f"{split}_{index}",
        split=split,
        family=family,
        grid_size=grid,
        start=start,
        goal=goal,
        obstacles=tuple(),
        terrain=terrain,
        subgoal_a=subgoal_a,
        subgoal_b=subgoal_b,
        key=key,
        door=door,
        rule={"family": family, "target_order": list(target_order), "target_order_index": int(target_idx)},
        constraint_examples=constraint_examples,
        candidates=candidates,
        rollouts=rollouts,
        reached_goal=reached,
        satisfies_constraint=satisfies,
        candidate_types=candidate_types,
        label=int(label),
        metadata=metadata,
    )


def _role_cells_for_family(family: str, index: int) -> Tuple[Tuple[str, int, int, int, int], ...]:
    # Rotate the same structural map so terrain/family probes cannot identify
    # the active rule while the rollout still exposes task-local evidence.
    shift = int(index % len(BASE_SPECIAL_CELLS))
    cells = tuple(BASE_SPECIAL_CELLS[(i + shift) % len(BASE_SPECIAL_CELLS)] for i in range(len(BASE_SPECIAL_CELLS)))
    if family == "color_zone":
        return tuple((f"zone_{i}", row, col, i + 1, ROLE_IDS["terrain"]) for i, (row, col) in enumerate(cells))
    if family == "ordered_subgoal":
        roles = (ROLE_IDS["subgoal_a"], ROLE_IDS["subgoal_b"], ROLE_IDS["subgoal_a"], ROLE_IDS["subgoal_b"])
        return tuple((f"subgoal_{chr(65 + i)}", row, col, i + 1, roles[i]) for i, (row, col) in enumerate(cells))
    if family == "key_door":
        names = ("key_a", "door_a", "key_b", "door_b")
        roles = (ROLE_IDS["key"], ROLE_IDS["door"], ROLE_IDS["key"], ROLE_IDS["door"])
        return tuple((names[i], row, col, i + 1, roles[i]) for i, (row, col) in enumerate(cells))
    raise ValueError(f"unknown family: {family}")


def _candidate_templates(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    role_cells: Sequence[Tuple[str, int, int, int, int]],
    num_candidates: int,
    index: int,
    grid: int,
) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    rows: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
    seen: set[Tuple[int, ...]] = set()
    for order in permutations(range(len(role_cells))):
        for horizontal_first in (bool(index % 2), not bool(index % 2)):
            waypoints = [(role_cells[i][1], role_cells[i][2]) for i in order]
            actions = tuple(_actions_via_waypoints(start, [*waypoints, goal], horizontal_first=horizontal_first))
            rollout = _simulate(start, actions, int(grid), tuple())
            realized_order = tuple(_realized_visit_order(role_cells, rollout))
            if len(realized_order) != len(role_cells) or realized_order in seen:
                continue
            rows.append((actions, realized_order))
            seen.add(realized_order)
            if len(rows) >= int(num_candidates):
                return rows
    raise RuntimeError(f"could not construct {num_candidates} unique paired-counterfactual candidates")


def _constraint_examples_for(
    family: str,
    role_cells: Sequence[Tuple[str, int, int, int, int]],
    target_order: Sequence[int],
) -> Tuple[ConstraintEvidence, ...]:
    valid = tuple((int(role_cells[i][4]), int(role_cells[i][1]), int(role_cells[i][2])) for i in target_order)
    invalid_order = tuple(reversed(tuple(target_order)))
    invalid = tuple((int(role_cells[i][4]), int(role_cells[i][1]), int(role_cells[i][2])) for i in invalid_order)
    return (
        ConstraintEvidence("valid", valid),
        ConstraintEvidence("invalid", invalid),
    )


def _candidate_type(family: str, order_idx: int, order: Sequence[int], target_order: Sequence[int]) -> str:
    if tuple(order) == tuple(target_order):
        return f"gold_{family}_target_order"
    return f"counterfactual_{family}_order_{order_idx}"


def _rule_name(family: str, role_cells: Sequence[Tuple[str, int, int, int, int]], target_order: Sequence[int]) -> str:
    names = [str(role_cells[i][0]) for i in target_order]
    if family == "color_zone":
        return "required_zone_order:" + ">".join(names)
    if family == "ordered_subgoal":
        return "required_subgoal_order:" + ">".join(names)
    if family == "key_door":
        return "required_key_door_context:" + ">".join(names)
    return ">".join(names)


def _balance_repaired_records(
    records: Sequence[Tuple[Tuple[int, ...], str, bool]],
    start: Tuple[int, int],
    plan_length: int,
    grid: int,
) -> List[Tuple[Tuple[int, ...], str, bool]]:
    counts = [_action_counts(actions) for actions, _ctype, _valid in records]
    max_u = max(row[0] for row in counts)
    max_d = max(row[1] for row in counts)
    max_l = max(row[2] for row in counts)
    max_r = max(row[3] for row in counts)
    target_d = max(max_d, max_u + (grid - 1))
    target_u = target_d - (grid - 1)
    target_r = max(max_r, max_l + (grid - 1))
    target_l = target_r - (grid - 1)
    target = [target_u, target_d, target_l, target_r]
    while sum(target) < int(plan_length):
        target[0] += 1
        target[1] += 1
        target[2] += 1
        target[3] += 1
    balanced = []
    for actions, ctype, valid in records:
        current = _action_counts(actions)
        add_vertical = max(0, min(target[0] - current[0], target[1] - current[1]))
        add_horizontal = max(0, min(target[2] - current[2], target[3] - current[3]))
        prefix: List[int] = []
        prefix.extend([1, 0] * add_vertical)
        prefix.extend([3, 2] * add_horizontal)
        out = [*prefix, *actions]
        # Add neutral loops at the start so all candidates share one length and
        # endpoint without altering the order in which special cells are reached.
        loop = [3, 2] if start[1] + 1 < grid else [1, 0]
        while len(out) < sum(target):
            out = [*loop, *out]
        balanced.append((tuple(out[: sum(target)]), ctype, valid))
    return balanced


def _matches_target_order(
    rollout: Sequence[Tuple[int, int]],
    role_cells: Sequence[Tuple[str, int, int, int, int]],
    target_order: Sequence[int],
) -> bool:
    return tuple(_realized_visit_order(role_cells, rollout)) == tuple(int(i) for i in target_order)


def _realized_visit_order(
    role_cells: Sequence[Tuple[str, int, int, int, int]],
    rollout: Sequence[Tuple[int, int]],
) -> List[int]:
    cell_to_idx = {(int(row), int(col)): idx for idx, (_name, row, col, _value, _role_id) in enumerate(role_cells)}
    seen: List[int] = []
    for pos in rollout:
        idx = cell_to_idx.get((int(pos[0]), int(pos[1])))
        if idx is not None and (not seen or seen[-1] != idx):
            if idx not in seen:
                seen.append(idx)
    return seen


def fit_constraint_verifier_15(
    train_examples: Sequence[ConstraintExample],
    dev_examples: Sequence[ConstraintExample],
    variant: ConstraintVariant,
    training: ConstraintTrainingConfig,
    seed: int,
    device: str,
    trainable_shared: bool,
    method: str,
) -> ConstraintFitResult:
    _set_seed(seed)
    model = ConstraintCloneVerifier(replace(variant, model_dim=_compatible_dim(variant.model_dim, variant.num_heads))).to(device)
    model.configure_shared_trainable(trainable_shared)
    initial_shared = _flat_params(model.shared_parameter_items())
    initial_coord = _flat_params(model.coordinator_parameter_items())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(training.lr), weight_decay=float(training.weight_decay))
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_dev = -1.0
    bad = 0
    history: List[Dict[str, float]] = []
    shared_grad_norms: List[float] = []
    coord_grad_norms: List[float] = []
    start = time.perf_counter()
    for epoch in range(int(training.epochs)):
        model.train()
        losses = []
        order = np.random.default_rng(seed + epoch).permutation(len(train_examples)).tolist()
        for batch_ids in _chunks(order, int(training.batch_size)):
            batch_examples = [train_examples[i] for i in batch_ids]
            batch = collate_constraint_batch_15(batch_examples, variant, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss = F.cross_entropy(out["logits"], batch["labels"])
            loss.backward()
            shared_grad_norms.append(_grad_norm([p for _n, p in model.shared_parameter_items()]))
            coord_grad_norms.append(_grad_norm([p for _n, p in model.coordinator_parameter_items()]))
            if float(training.gradient_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(training.gradient_clip_norm))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_logits = predict_constraint_logits_15(model, dev_examples, variant, training.batch_size, device)
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
        "best_dev_top1": float(best_dev),
    }
    return ConstraintFitResult(method, model, variant, trainable_shared, audit, history, float(time.perf_counter() - start), sum(p.numel() for p in model.parameters()))


def predict_constraint_logits_15(
    model: nn.Module,
    examples: Sequence[ConstraintExample],
    variant: ConstraintVariant,
    batch_size: int,
    device: str,
    condition: str = "none",
    seed: int = 0,
    allowed_views: set[str] | None = None,
    blocked_views: set[str] | None = None,
    candidate_token_mode: str = "action",
    return_attention: bool = False,
) -> np.ndarray | Tuple[np.ndarray, List[Dict[str, object]]]:
    eval_variant = variant
    if condition == "role_order_shuffle":
        views = list(variant.view_names)
        rng = np.random.default_rng(seed)
        rng.shuffle(views)
        eval_variant = replace(variant, view_names=tuple(views))
    model.eval()
    logits_rows = []
    attention_rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for offset, batch_examples in enumerate(_batched(examples, int(batch_size))):
            batch = collate_constraint_batch_15(
                batch_examples,
                eval_variant,
                device,
                allowed_views=allowed_views,
                blocked_views=blocked_views,
                candidate_token_mode=candidate_token_mode,
            )
            output = model(
                batch,
                hidden_state_shuffle=condition == "hidden_state_shuffle",
                shuffle_seed=seed + offset,
                return_attention=return_attention,
            )
            logits = output["logits"].detach().cpu().numpy()
            logits_rows.append(logits)
            if return_attention:
                attention_rows.extend(_summarize_attention_15(output, batch_examples, eval_variant, logits))
    logits_np = np.concatenate(logits_rows, axis=0) if logits_rows else np.zeros((0, 0), dtype=np.float32)
    return (logits_np, attention_rows) if return_attention else logits_np


def collate_constraint_batch_15(
    examples: Sequence[ConstraintExample],
    variant: ConstraintVariant,
    device: str,
    allowed_views: set[str] | None = None,
    blocked_views: set[str] | None = None,
    candidate_token_mode: str = "action",
) -> Dict[str, torch.Tensor]:
    max_evidence_tokens = 96
    max_candidate_tokens = 72
    evidence_rows = []
    evidence_masks = []
    candidate_rows = []
    candidate_masks = []
    labels = []
    for example in examples:
        ex_evidence = []
        ex_masks = []
        ex_candidates = []
        ex_candidate_masks = []
        for cand_index, actions in enumerate(example.candidates):
            per_view = []
            per_mask = []
            for view in variant.view_names:
                canonical = _canonical_view(view)
                if (allowed_views is not None and canonical not in allowed_views) or (blocked_views is not None and canonical in blocked_views):
                    tokens: List[List[int]] = []
                else:
                    tokens = _evidence_tokens_15(example, cand_index, canonical, variant)
                fields, mask = _pack_tokens_15(tokens, max_evidence_tokens)
                per_view.append(fields)
                per_mask.append(mask)
            cand_tokens = _candidate_tokens_15(example, cand_index, actions, candidate_token_mode)
            cand_fields, cand_mask = _pack_tokens_15(cand_tokens, max_candidate_tokens)
            ex_evidence.append(np.stack(per_view))
            ex_masks.append(np.stack(per_mask))
            ex_candidates.append(cand_fields)
            ex_candidate_masks.append(cand_mask)
        evidence_rows.append(np.stack(ex_evidence))
        evidence_masks.append(np.stack(ex_masks))
        candidate_rows.append(np.stack(ex_candidates))
        candidate_masks.append(np.stack(ex_candidate_masks))
        labels.append(int(example.label))
    return {
        "evidence_fields": torch.as_tensor(np.stack(evidence_rows), dtype=torch.long, device=device),
        "evidence_mask": torch.as_tensor(np.stack(evidence_masks), dtype=torch.bool, device=device),
        "candidate_fields": torch.as_tensor(np.stack(candidate_rows), dtype=torch.long, device=device),
        "candidate_mask": torch.as_tensor(np.stack(candidate_masks), dtype=torch.bool, device=device),
        "labels": torch.as_tensor(labels, dtype=torch.long, device=device),
    }


def _evidence_tokens_15(example: ConstraintExample, cand_index: int, view: str, variant: ConstraintVariant) -> List[List[int]]:
    tokens: List[List[int]] = []
    family_id = FAMILY_IDS[example.family]
    if view == "state":
        tokens.append(_token_15("state", example.start[0], example.start[1], 0, 0, family_id, 0, 0))
    elif view == "goal":
        tokens.append(_token_15("goal", example.goal[0], example.goal[1], 0, 0, family_id, 0, 0))
    elif view == "terrain":
        for _name, row, col, value, role_id in _role_cells(example):
            role = _role_name(int(role_id))
            tokens.append(_token_15(role, int(row), int(col), int(value), 0, family_id, 0, 0))
    elif view == "action":
        for step, action in enumerate(example.candidates[cand_index]):
            tokens.append(_token_15("action", 0, 0, int(action), step, family_id, 0, 0))
    elif view == "rollout":
        states = list(example.rollouts[cand_index])
        usable = states if variant.include_final_state else states[:-1]
        for step, pos in enumerate(usable):
            tokens.append(_token_15("rollout", pos[0], pos[1], _cell_value_15(example, pos), step, family_id, 0, int(pos == example.goal)))
    elif view == "transition":
        states = list(example.rollouts[cand_index])
        actions = list(example.candidates[cand_index])
        max_step = min(len(actions), max(0, len(states) - 1))
        if not variant.include_final_state:
            max_step = max(0, max_step - 1)
        for step in range(max_step):
            state_t = states[step]
            state_next = states[step + 1]
            action = int(actions[step])
            tokens.append(_token_15("transition", state_t[0], state_t[1], action, step, family_id, 0, _cell_value_15(example, state_t)))
            tokens.append(_token_15("transition", state_next[0], state_next[1], _cell_value_15(example, state_next), step, family_id, 0, int(state_next == example.goal)))
    elif view in {"constraint", "violation"}:
        for evidence_idx, evidence in enumerate(example.constraint_examples):
            if view == "constraint" and evidence.label == "invalid":
                continue
            if view == "violation" and evidence.label == "valid":
                continue
            role = "constraint_valid" if evidence.label == "valid" else "constraint_invalid"
            tokens.append(_token_15("rule", 0, 0, _rule_value_15(example), evidence_idx, family_id, 0, 0))
            for step, (raw_role, row, col) in enumerate(evidence.tokens):
                tokens.append(_token_15(role, row, col, _role_value_15(int(raw_role), example, int(row), int(col)), step, family_id, 0, int(evidence.label == "valid")))
    if not tokens:
        tokens.append(_token_15("summary", 0, 0, 0, 0, family_id, 0, 0))
    return tokens


def _candidate_tokens_15(example: ConstraintExample, cand_index: int, actions: Sequence[int], mode: str) -> List[List[int]]:
    family_id = FAMILY_IDS[example.family]
    if mode == "blank":
        return [_token_15("summary", example.start[0], example.start[1], len(actions), 0, family_id, 0, 0)]
    tokens = [_token_15("summary", example.start[0], example.start[1], len(actions), 0, family_id, 0, 0)]
    for step, action in enumerate(actions):
        prev = int(actions[step - 1]) if step > 0 else 4
        tokens.append(_token_15("action", prev, int(action), int(action), step, family_id, 0, 0))
    return tokens


def _pack_tokens_15(tokens: Sequence[Sequence[int]], limit: int) -> Tuple[np.ndarray, np.ndarray]:
    fields = np.zeros((int(limit), len(FIELD_VOCABS)), dtype=np.int64)
    mask = np.zeros((int(limit),), dtype=bool)
    if not tokens:
        tokens = [_token_15("summary", 0, 0, 0, 0, 0, 0, 0)]
    count = min(int(limit), len(tokens))
    fields[:count] = np.asarray(tokens[:count], dtype=np.int64)
    mask[:count] = True
    return fields, mask


def _token_15(role: str, row: int, col: int, value: int, step: int, family_id: int, view_id: int, aux: int) -> List[int]:
    raw = [ROLE_IDS.get(role, 0), row, col, value, step, family_id, view_id, aux]
    return [max(0, min(FIELD_VOCABS[index] - 1, int(value))) for index, value in enumerate(raw)]


def _role_cells(example: ConstraintExample) -> Tuple[Tuple[str, int, int, int, int], ...]:
    return tuple((str(name), int(row), int(col), int(value), int(role_id)) for name, row, col, value, role_id in example.metadata.get("role_cells", []))


def _role_name(role_id: int) -> str:
    for name, value in ROLE_IDS.items():
        if int(value) == int(role_id):
            return name
    return "terrain"


def _cell_value_15(example: ConstraintExample, pos: Tuple[int, int]) -> int:
    for _name, row, col, value, _role_id in _role_cells(example):
        if (int(row), int(col)) == (int(pos[0]), int(pos[1])):
            return int(value)
    return 0


def _rule_value_15(example: ConstraintExample) -> int:
    target = [int(v) for v in example.metadata.get("target_order", [])]
    value = int(example.metadata.get("target_order_index", 0)) + 1
    return value + 8 * FAMILY_IDS.get(example.family, 0) + (target[0] if target else 0)


def _role_value_15(raw_role: int, example: ConstraintExample, row: int, col: int) -> int:
    for _name, r, c, value, role_id in _role_cells(example):
        if int(role_id) == int(raw_role) and int(r) == int(row) and int(c) == int(col):
            return int(value)
    return int(raw_role)


def _canonical_view(view: str) -> str:
    return str(view).split(":", 1)[0]


def _run_micro_overfit_gates_15(
    variant: ConstraintVariant,
    dataset_config: ConstraintDatasetConfig,
    training: ConstraintTrainingConfig,
    seed: int,
    device: str,
) -> List[Dict[str, object]]:
    gates = [("4_examples_N2", 2, 4, 0.95), ("16_examples_N4", 4, 16, 0.90), ("64_examples_N8", 8, 64, 0.85)]
    rows = []
    for name, candidates, examples, threshold in gates:
        gate_config = replace(dataset_config, num_candidates=int(candidates), train_examples=max(4, int(examples)), dev_examples=4, test_examples=4)
        splits = build_repaired_constraint_splits(gate_config, seed + 1000 + int(candidates))
        train_examples = splits["train"][: int(examples)]
        fit = fit_constraint_verifier_15(train_examples, train_examples, variant, replace(training, batch_size=min(int(training.batch_size), len(train_examples))), seed + 91_500 + int(candidates), device, True, f"micro15__{variant.name}__{name}")
        logits = predict_constraint_logits_15(fit.model, train_examples, variant, training.batch_size, device)
        acc = _top1(logits, _labels(train_examples))
        rows.append(
            {
                "benchmark": BENCHMARK,
                "variant": variant.name,
                "gate": name,
                "candidate_count": int(candidates),
                "examples": len(train_examples),
                "train_accuracy": float(acc),
                "threshold": float(threshold),
                "pass": bool(acc >= threshold),
                "history": fit.history,
                "audit": fit.audit,
            }
        )
        if acc < threshold:
            break
    return rows


def _run_controls_15(
    model: nn.Module,
    variant: ConstraintVariant,
    splits: Dict[str, List[ConstraintExample]],
    clean_logits: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: str,
    seed: int,
    shortcut_row: Dict[str, object],
) -> Dict[str, object]:
    examples = splits["test"]
    chance = 1.0 / max(1, len(examples[0].candidates) if examples else 1)

    def predict(control_examples: Sequence[ConstraintExample], **kwargs: object) -> np.ndarray:
        return predict_constraint_logits_15(model, control_examples, variant, batch_size, device, seed=seed + 44_500, **kwargs)  # type: ignore[arg-type]

    clean_top1 = float(_top1(clean_logits, labels))
    controls: Dict[str, object] = {
        "benchmark": BENCHMARK,
        "variant": variant.name,
        "seed": int(seed),
        "chance": float(chance),
        "clean": _metric_block_15(clean_logits, labels, examples),
        "candidate_action_only": _metric_block_15(predict(examples, allowed_views=set()), labels, examples),
        "rollout_only": _metric_block_15(predict(examples, allowed_views={"rollout", "transition", "action"}), labels, examples),
        "constraint_only": _metric_block_15(predict(examples, allowed_views={"constraint", "violation"}, candidate_token_mode="blank"), labels, examples),
        "endpoint_only": {"top1": _accuracy(_final_distance_predictions(examples), labels)},
        "length_only": _accuracy(_length_predictions(examples), labels),
        "action_unigram": _accuracy(_unigram_predictions(examples), labels),
        "action_bigram": _accuracy(_bigram_predictions(examples), labels),
        "family_proxy_probe": {"top1": float(shortcut_row["baselines"]["family_id_proxy_probe"])},
    }
    for name in (
        "candidate_evidence_mismatch",
        "constraint_example_mismatch",
        "rollout_mismatch",
        "candidate_order_shuffle_with_gold_remap",
        "randomized_labels",
    ):
        controlled = apply_repair_control(examples, name, seed + 50_500 + len(controls))
        logits = predict(controlled)
        controls[name] = _metric_block_15(logits, _labels(controlled), controlled)
    role_logits = predict_constraint_logits_15(model, examples, variant, batch_size, device, condition="role_order_shuffle", seed=seed + 60_500)
    hidden_logits = predict_constraint_logits_15(model, examples, variant, batch_size, device, condition="hidden_state_shuffle", seed=seed + 61_500)
    no_constraint = predict(examples, blocked_views={"constraint", "violation"})
    no_rollout = predict(examples, blocked_views={"rollout", "transition"})
    controls["role_order_shuffle"] = _metric_block_15(role_logits, labels, examples)
    controls["hidden_state_shuffle"] = _metric_block_15(hidden_logits, labels, examples)
    controls["ablate_constraint_view"] = _metric_block_15(no_constraint, labels, examples)
    controls["ablate_rollout_transition_view"] = _metric_block_15(no_rollout, labels, examples)
    controls["candidate_order_delta"] = abs(clean_top1 - float(controls["candidate_order_shuffle_with_gold_remap"]["top1"]))
    controls["role_order_delta"] = abs(clean_top1 - float(controls["role_order_shuffle"]["top1"]))
    shortcut_values = [
        float(controls["candidate_action_only"]["top1"]),
        float(controls["rollout_only"]["top1"]),
        float(controls["constraint_only"]["top1"]),
        float(controls["endpoint_only"]["top1"]),
        float(controls["length_only"]),
        float(controls["action_unigram"]),
        float(controls["action_bigram"]),
        float(controls["family_proxy_probe"]["top1"]),
    ]
    controls["control_pass"] = {
        "candidate_only_near_chance": float(controls["candidate_action_only"]["top1"]) <= chance + NEAR_CHANCE_MARGIN,
        "rollout_only_near_chance": float(controls["rollout_only"]["top1"]) <= chance + NEAR_CHANCE_MARGIN,
        "constraint_only_near_chance": float(controls["constraint_only"]["top1"]) <= chance + NEAR_CHANCE_MARGIN,
        "endpoint_only_near_chance": float(controls["endpoint_only"]["top1"]) <= chance + NEAR_CHANCE_MARGIN,
        "length_action_stat_near_chance": max(float(controls["length_only"]), float(controls["action_unigram"]), float(controls["action_bigram"])) <= chance + NEAR_CHANCE_MARGIN,
        "family_proxy_probe_near_chance": float(controls["family_proxy_probe"]["top1"]) <= chance + NEAR_CHANCE_MARGIN,
        "generic_shortcuts_near_chance": max(shortcut_values) <= chance + NEAR_CHANCE_MARGIN,
        "constraint_mismatch_collapses": float(controls["constraint_example_mismatch"]["top1"]) <= chance + COLLAPSE_MARGIN,
        "rollout_mismatch_collapses": float(controls["rollout_mismatch"]["top1"]) <= chance + COLLAPSE_MARGIN,
        "candidate_evidence_mismatch_collapses": float(controls["candidate_evidence_mismatch"]["top1"]) <= chance + COLLAPSE_MARGIN,
        "candidate_order_remap_passes": float(controls["candidate_order_delta"]) <= 0.08,
        "role_order_remap_passes": float(controls["role_order_delta"]) <= 0.08,
        "hidden_state_shuffle_collapses": float(controls["hidden_state_shuffle"]["top1"]) <= max(chance + 0.20, clean_top1 - 0.05),
        "randomized_labels_collapses": float(controls["randomized_labels"]["top1"]) <= chance + COLLAPSE_MARGIN,
        "ablate_constraint_view_hurts": clean_top1 - float(controls["ablate_constraint_view"]["top1"]) >= HURT_MARGIN,
        "ablate_rollout_transition_view_hurts": clean_top1 - float(controls["ablate_rollout_transition_view"]["top1"]) >= HURT_MARGIN,
    }
    controls["control_pass"]["overall"] = all(bool(value) for value in controls["control_pass"].values())
    return controls


def apply_repair_control(examples: Sequence[ConstraintExample], control: str, seed: int) -> List[ConstraintExample]:
    rng = np.random.default_rng(seed)
    if control == "candidate_order_shuffle_with_gold_remap":
        return [_candidate_order_shuffle(example, rng) for example in examples]
    if control == "randomized_labels":
        return [replace(example, label=int(rng.integers(0, len(example.candidates))), metadata={**example.metadata, "control": control}) for example in examples]
    out = []
    for idx, example in enumerate(examples):
        other = _other_example(examples, idx, rng, same_family=control in {"constraint_example_mismatch", "rollout_mismatch"})
        if control == "constraint_example_mismatch":
            out.append(replace(example, rule=other.rule, constraint_examples=other.constraint_examples, metadata={**example.metadata, "control": control, "mismatched_rule": other.metadata.get("task_rule")}))
        elif control == "rollout_mismatch":
            out.append(replace(example, rollouts=other.rollouts if len(other.rollouts) == len(example.rollouts) else example.rollouts, metadata={**example.metadata, "control": control}))
        elif control == "candidate_evidence_mismatch":
            out.append(
                replace(
                    example,
                    rollouts=other.rollouts if len(other.rollouts) == len(example.rollouts) else example.rollouts,
                    rule=other.rule,
                    constraint_examples=other.constraint_examples,
                    metadata={**example.metadata, "control": control, "mismatched_rule": other.metadata.get("task_rule")},
                )
            )
        else:
            raise ValueError(f"unknown control: {control}")
    return out


def _other_example(examples: Sequence[ConstraintExample], idx: int, rng: np.random.Generator, same_family: bool) -> ConstraintExample:
    example = examples[idx]
    candidates = [j for j, other in enumerate(examples) if j != idx and ((other.family == example.family) if same_family else True)]
    if not candidates:
        candidates = [j for j in range(len(examples)) if j != idx]
    if not candidates:
        return example
    return examples[int(candidates[int(rng.integers(0, len(candidates)))])]


def _candidate_order_shuffle(example: ConstraintExample, rng: np.random.Generator) -> ConstraintExample:
    order = rng.permutation(len(example.candidates))
    if len(order) > 1 and np.array_equal(order, np.arange(len(order))):
        order = np.roll(order, 1)
    label = int(np.where(order == int(example.label))[0][0])
    return replace(
        example,
        candidates=tuple(example.candidates[int(i)] for i in order),
        rollouts=tuple(example.rollouts[int(i)] for i in order),
        reached_goal=tuple(example.reached_goal[int(i)] for i in order),
        satisfies_constraint=tuple(example.satisfies_constraint[int(i)] for i in order),
        candidate_types=tuple(example.candidate_types[int(i)] for i in order),
        label=label,
        metadata={**example.metadata, "control": "candidate_order_shuffle_with_gold_remap"},
    )


def _shortcut_decomposition_row(splits: Dict[str, List[ConstraintExample]], seed: int) -> Dict[str, object]:
    train, dev, test = splits["train"], splits["dev"], splits["test"]
    labels = _labels(test)
    modes = (
        "candidate_action_only",
        "rollout_only",
        "terrain_only",
        "constraint_examples_only",
        "family_id_proxy_probe",
        "rollout_statistics_only",
        "candidate_slot_probe",
    )
    baselines: Dict[str, float] = {
        "random": 1.0 / max(1, len(test[0].candidates) if test else 1),
        "endpoint_only": _accuracy(_final_distance_predictions(test), labels),
        "length_only": _accuracy(_length_predictions(test), labels),
        "action_unigram": _accuracy(_unigram_predictions(test), labels),
        "action_bigram": _accuracy(_bigram_predictions(test), labels),
    }
    for mode_idx, mode in enumerate(modes):
        model = _fit_probe_mlp(train, dev, seed + 1_700 + mode_idx, mode)
        logits = _predict_probe_mlp(model, test, mode)
        baselines[mode] = _top1(logits, labels)
    baselines["family_specific_heuristics"] = _family_specific_oracle(test)
    baselines["rollout_plus_constraint_oracle"] = _rollout_plus_constraint_oracle(test)
    shortcut_max = max(float(baselines[key]) for key in baselines if key not in {"family_specific_heuristics", "rollout_plus_constraint_oracle"})
    by_family = {}
    for family in FAMILIES:
        family_examples = [example for example in test if example.family == family]
        family_labels = _labels(family_examples)
        by_family[family] = {
            "count": len(family_examples),
            "candidate_action_only": _top1(_predict_probe_mlp(_fit_probe_mlp(train, dev, seed + 1_900 + FAMILY_IDS[family], "candidate_action_only"), family_examples, "candidate_action_only"), family_labels) if family_examples else 0.0,
            "rollout_statistics_only": _top1(_predict_probe_mlp(_fit_probe_mlp(train, dev, seed + 2_000 + FAMILY_IDS[family], "rollout_statistics_only"), family_examples, "rollout_statistics_only"), family_labels) if family_examples else 0.0,
            "endpoint_only": _accuracy(_final_distance_predictions(family_examples), family_labels) if family_examples else 0.0,
            "family_specific_heuristic": _family_specific_oracle(family_examples),
        }
    return {
        "benchmark": BENCHMARK,
        "seed": int(seed),
        "chance": baselines["random"],
        "baselines": baselines,
        "by_family": by_family,
        "shortcut_max_controlled": float(shortcut_max),
        "near_chance": bool(shortcut_max <= baselines["random"] + NEAR_CHANCE_MARGIN),
        "plan_arch1_4_shortcut_explanation": _plan14_shortcut_explanation(),
    }


def _fit_probe_mlp(train: Sequence[ConstraintExample], dev: Sequence[ConstraintExample], seed: int, mode: str) -> ProbeMLP:
    _set_seed(seed)
    train_x, train_y = _probe_feature_batch(train, mode)
    dev_x, dev_y = _probe_feature_batch(dev, mode)
    model = ProbeMLP(train_x.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0001)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_dev = -1.0
    bad = 0
    for epoch in range(10):
        order = np.random.default_rng(seed + epoch).permutation(len(train)).tolist()
        for batch_ids in _chunks(order, 64):
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_x[batch_ids])
            loss = F.cross_entropy(logits, train_y[batch_ids])
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            acc = _top1(model(dev_x).detach().numpy(), dev_y.numpy())
        if acc > best_dev:
            best_dev = acc
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= 4:
            break
    model.load_state_dict(best_state)
    return model


def _predict_probe_mlp(model: ProbeMLP, examples: Sequence[ConstraintExample], mode: str) -> np.ndarray:
    if not examples:
        return np.zeros((0, 0), dtype=np.float32)
    x, _y = _probe_feature_batch(examples, mode)
    with torch.no_grad():
        return model(x).detach().numpy()


def _probe_feature_batch(examples: Sequence[ConstraintExample], mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor([[_probe_features(example, idx, mode) for idx in range(len(example.candidates))] for example in examples], dtype=torch.float32),
        torch.as_tensor([example.label for example in examples], dtype=torch.long),
    )


def _probe_features(example: ConstraintExample, cand_index: int, mode: str) -> List[float]:
    candidate = example.candidates[cand_index]
    rollout = example.rollouts[cand_index]
    if mode == "candidate_slot_probe":
        return [1.0 if cand_index == i else 0.0 for i in range(len(example.candidates))]
    if mode == "terrain_only":
        return _terrain_proxy_features(example)
    if mode == "constraint_examples_only":
        return _constraint_proxy_features(example)
    if mode == "family_id_proxy_probe":
        return _family_proxy_features(example)
    if mode == "candidate_action_only":
        return _action_sequence_features(candidate)
    if mode == "rollout_only":
        return _rollout_sequence_features(example, rollout)
    if mode == "rollout_statistics_only":
        return _rollout_statistics_features(example, rollout, candidate)
    raise ValueError(f"unknown probe mode: {mode}")


def _action_sequence_features(candidate: Sequence[int]) -> List[float]:
    counts = np.asarray(_action_counts(candidate), dtype=np.float32) / max(1, len(candidate))
    bigrams = [0.0] * 16
    for a, b in zip(candidate, candidate[1:]):
        bigrams[int(a) * 4 + int(b)] += 1.0
    denom = max(1, len(candidate) - 1)
    bigrams = [value / denom for value in bigrams]
    turns = sum(1 for a, b in zip(candidate, candidate[1:]) if int(a) != int(b)) / denom
    reversals = sum(1 for a, b in zip(candidate, candidate[1:]) if {int(a), int(b)} in ({0, 1}, {2, 3})) / denom
    return [len(candidate) / 64.0, *counts.tolist(), *bigrams, turns, reversals]


def _rollout_sequence_features(example: ConstraintExample, rollout: Sequence[Tuple[int, int]]) -> List[float]:
    order = _special_visit_order(example, rollout)
    onehot = []
    for slot in range(4):
        value = order[slot] if slot < len(order) else -1
        onehot.extend([1.0 if value == i else 0.0 for i in range(4)])
    hist = [0.0] * 5
    for pos in rollout:
        hist[_cell_value_15(example, pos)] += 1.0
    total = max(1.0, sum(hist))
    return [value / total for value in hist] + onehot


def _rollout_statistics_features(example: ConstraintExample, rollout: Sequence[Tuple[int, int]], candidate: Sequence[int]) -> List[float]:
    order = _special_visit_order(example, rollout)
    repeated = len(rollout) - len(set(rollout))
    special_count = len(order)
    row_span = max(pos[0] for pos in rollout) - min(pos[0] for pos in rollout)
    col_span = max(pos[1] for pos in rollout) - min(pos[1] for pos in rollout)
    manhattan_total = sum(_manhattan(a, b) for a, b in zip(rollout, rollout[1:]))
    pair_order = []
    for a in range(4):
        for b in range(4):
            pair_order.append(1.0 if a in order and b in order and order.index(a) < order.index(b) else 0.0)
    return [
        len(candidate) / 64.0,
        repeated / max(1, len(rollout)),
        special_count / 4.0,
        row_span / max(1, example.grid_size - 1),
        col_span / max(1, example.grid_size - 1),
        manhattan_total / max(1, len(rollout)),
        *pair_order,
    ]


def _terrain_proxy_features(example: ConstraintExample) -> List[float]:
    role_cells = _role_cells(example)
    values = [0.0] * 16
    for _name, row, col, value, role_id in role_cells:
        values[int(value) % 8] += 1.0
        values[8 + (int(role_id) % 8)] += 1.0
    return [FAMILY_IDS[example.family] / 4.0, len(role_cells) / 8.0, *values]


def _constraint_proxy_features(example: ConstraintExample) -> List[float]:
    target = [int(v) for v in example.metadata.get("target_order", [])]
    hist = [0.0] * 8
    for value in target:
        hist[value % 8] += 1.0
    return [FAMILY_IDS[example.family] / 4.0, int(example.metadata.get("target_order_index", 0)) / 8.0, *hist]


def _family_proxy_features(example: ConstraintExample) -> List[float]:
    role_cells = _role_cells(example)
    colors = [0.0] * 8
    role_hist = [0.0] * 16
    for _name, _row, _col, value, role_id in role_cells:
        colors[int(value) % 8] += 1.0
        role_hist[int(role_id) % 16] += 1.0
    return [FAMILY_IDS[example.family] / 4.0, len(role_cells) / 8.0, *colors, *role_hist]


def _special_visit_order(example: ConstraintExample, rollout: Sequence[Tuple[int, int]]) -> List[int]:
    cell_to_idx = {(row, col): idx for idx, (_name, row, col, _value, _role_id) in enumerate(_role_cells(example))}
    order: List[int] = []
    for pos in rollout:
        idx = cell_to_idx.get((int(pos[0]), int(pos[1])))
        if idx is not None and idx not in order:
            order.append(idx)
    return order


def _family_specific_oracle(examples: Sequence[ConstraintExample]) -> float:
    if not examples:
        return 0.0
    preds = []
    for example in examples:
        scores = [1.0 if valid else 0.0 for valid in example.satisfies_constraint]
        preds.append(int(np.argmax(scores)))
    return _accuracy(np.asarray(preds, dtype=np.int64), _labels(examples))


def _rollout_plus_constraint_oracle(examples: Sequence[ConstraintExample]) -> float:
    return _family_specific_oracle(examples)


def _counterfactual_candidate_audit(splits: Dict[str, List[ConstraintExample]], seed: int) -> Dict[str, object]:
    examples = [example for rows in splits.values() for example in rows]
    test = splits["test"]

    def frac(predicate: object, rows: Sequence[ConstraintExample]) -> float:
        values = [bool(predicate(example)) for example in rows]  # type: ignore[misc]
        return float(np.mean(values)) if values else 0.0

    labels = [example.label for example in test]
    label_hist = {str(idx): int(sum(1 for label in labels if int(label) == idx)) for idx in range(max(1, len(test[0].candidates) if test else 0))}
    return {
        "benchmark": BENCHMARK,
        "seed": int(seed),
        "examples": len(examples),
        "one_valid_candidate": frac(lambda e: sum(bool(v) for v in e.satisfies_constraint) == 1 and e.satisfies_constraint[e.label], examples),
        "all_candidates_same_endpoint": frac(lambda e: bool(e.metadata["all_candidates_same_endpoint"]), examples),
        "all_candidates_reach_goal": frac(lambda e: bool(e.metadata["all_candidates_reach_goal"]), examples),
        "all_candidates_same_length": frac(lambda e: bool(e.metadata["all_candidates_same_length"]), examples),
        "action_unigram_balanced": frac(lambda e: len({tuple(_action_counts(c)) for c in e.candidates}) == 1, examples),
        "action_bigram_balanced": frac(lambda e: len({_bigram_signature(c) for c in e.candidates}) == 1, examples),
        "special_cell_count_balanced": frac(lambda e: len({len(_special_visit_order(e, r)) for r in e.rollouts}) == 1, examples),
        "candidate_slot_histogram_test": label_hist,
    }


def _bigram_signature(candidate: Sequence[int]) -> Tuple[int, ...]:
    bigrams = [0] * 16
    for a, b in zip(candidate, candidate[1:]):
        bigrams[int(a) * 4 + int(b)] += 1
    return tuple(bigrams)


def _metric_block_15(logits: np.ndarray, labels: np.ndarray, examples: Sequence[ConstraintExample]) -> Dict[str, object]:
    if logits.size == 0:
        return {"top1": 0.0, "mrr": 0.0, "gold_ranks": []}
    order = np.argsort(-logits, axis=1)
    preds = order[:, 0]
    ranks = [int(np.where(row == int(label))[0][0]) + 1 for row, label in zip(order, labels)]
    margins = []
    for idx, label in enumerate(labels):
        wrong = np.delete(logits[idx], int(label))
        margins.append(float(logits[idx, int(label)] - (np.max(wrong) if wrong.size else logits[idx, int(label)])))
    return {
        "top1": _accuracy(preds, labels),
        "mrr": float(np.mean([1.0 / rank for rank in ranks])) if ranks else 0.0,
        "gold_ranks": ranks,
        "mean_gold_vs_best_wrong_logit_margin": float(np.mean(margins)) if margins else 0.0,
        "min_gold_vs_best_wrong_logit_margin": float(np.min(margins)) if margins else 0.0,
        "by_constraint_family": _accuracy_by_key(preds, labels, [example.family for example in examples]),
        "by_target_order": _accuracy_by_key(preds, labels, [str(example.metadata.get("target_order_index", 0)) for example in examples]),
        "all_candidates_reach_goal": _accuracy_on_subset(preds, labels, [bool(example.metadata["all_candidates_reach_goal"]) for example in examples]),
        "all_candidates_same_endpoint": _accuracy_on_subset(preds, labels, [bool(example.metadata["all_candidates_same_endpoint"]) for example in examples]),
        "failure_types": _failure_type_counts(preds, labels, examples),
    }


def _accuracy_by_key(predictions: np.ndarray, labels: np.ndarray, keys: Sequence[str]) -> Dict[str, float]:
    out = {}
    for key in sorted(set(keys)):
        idx = [i for i, value in enumerate(keys) if value == key]
        out[str(key)] = _accuracy(np.asarray([predictions[i] for i in idx]), np.asarray([labels[i] for i in idx]))
    return out


def _accuracy_on_subset(predictions: np.ndarray, labels: np.ndarray, mask: Sequence[bool]) -> Dict[str, object]:
    idx = [i for i, value in enumerate(mask) if bool(value)]
    return {"count": len(idx), "top1": _accuracy(np.asarray([predictions[i] for i in idx]), np.asarray([labels[i] for i in idx])) if idx else 0.0}


def _failure_type_counts(preds: np.ndarray, labels: np.ndarray, examples: Sequence[ConstraintExample]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for pred, label, example in zip(preds, labels, examples):
        if int(pred) == int(label):
            continue
        key = str(example.candidate_types[int(pred)])
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _attention_summaries_15(
    model: nn.Module,
    variant: ConstraintVariant,
    examples: Sequence[ConstraintExample],
    batch_size: int,
    device: str,
    seed: int,
) -> List[Dict[str, object]]:
    output = predict_constraint_logits_15(model, examples, variant, batch_size, device, seed=seed, return_attention=True)
    if not isinstance(output, tuple):
        return []
    _logits, rows = output
    for row in rows:
        row["benchmark"] = BENCHMARK
        row["variant"] = variant.name
        row["seed"] = int(seed)
    return rows


def _summarize_attention_15(output: Dict[str, object], examples: Sequence[ConstraintExample], variant: ConstraintVariant, logits: np.ndarray) -> List[Dict[str, object]]:
    weights = output.get("attention_weights")
    if weights is None:
        return []
    views = list(variant.view_names)
    w = weights.detach().float().cpu().numpy()
    bsz = len(examples)
    cand_count = len(examples[0].candidates) if examples else 0
    token_count = 96
    rows = []
    preds = np.argmax(logits, axis=1)
    for ex_idx, example in enumerate(examples):
        candidate_rows = []
        for cand_idx in range(cand_count):
            flat_idx = ex_idx * cand_count + cand_idx
            per_view = {}
            for view_idx, view in enumerate(views):
                start = view_idx * token_count
                end = start + token_count
                per_view[_canonical_view(view)] = float(w[flat_idx, :, :, start:end].sum())
            total = max(1e-8, sum(per_view.values()))
            candidate_rows.append({key: value / total for key, value in per_view.items()})
        rows.append(
            {
                "type": "plan_arch1_5_attention_summary",
                "example_id": example.id,
                "family": example.family,
                "target_order": example.metadata.get("target_order"),
                "gold_candidate": int(example.label),
                "predicted_candidate": int(preds[ex_idx]),
                "constraint_attention_mass_gold": candidate_rows[int(example.label)].get("constraint", 0.0) + candidate_rows[int(example.label)].get("violation", 0.0),
                "rollout_attention_mass_gold": candidate_rows[int(example.label)].get("rollout", 0.0) + candidate_rows[int(example.label)].get("transition", 0.0),
                "candidate_view_attention": candidate_rows,
            }
        )
    return rows


def _result_row_15(
    phase: str,
    variant: ConstraintVariant,
    seed: int,
    train_metric: Dict[str, object],
    frozen_metric: Dict[str, object] | None,
    delta: float | None,
    fit: ConstraintFitResult,
    control: Dict[str, object],
    splits: Dict[str, List[ConstraintExample]],
    elapsed: float,
) -> Dict[str, object]:
    return {
        "benchmark": BENCHMARK,
        "phase": phase,
        "variant": variant.name,
        "family": variant.family,
        "seed": int(seed),
        "status": "completed",
        "trainable": train_metric,
        "frozen": frozen_metric,
        "delta_trainable_minus_frozen": float(delta) if delta is not None else None,
        "control_pass": control["control_pass"],
        "trainable_audit": fit.audit,
        "param_count": int(fit.param_count),
        "training_time_seconds": float(elapsed),
        "dataset_sizes": {key: len(value) for key, value in splits.items()},
        "medium_validation_launched": False,
    }


def _compute_row_15(variant: ConstraintVariant, seed: int, splits: Dict[str, List[ConstraintExample]], row: Dict[str, object], elapsed: float) -> Dict[str, object]:
    return {
        "benchmark": BENCHMARK,
        "variant": variant.name,
        "seed": int(seed),
        "train_examples": len(splits["train"]),
        "dev_examples": len(splits["dev"]),
        "test_examples": len(splits["test"]),
        "candidate_count": len(splits["test"][0].candidates) if splits["test"] else 0,
        "view_count": len(variant.view_names),
        "param_count": int(row.get("param_count", 0)),
        "training_time_seconds": float(elapsed),
        "medium_validation_launched": False,
    }


def _leaderboard_15(rows: Sequence[Dict[str, object]], controls: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    control_by_key = {(row["variant"], row["seed"]): row for row in controls}
    leaders = []
    for row in rows:
        control = control_by_key.get((row["variant"], row["seed"]), {})
        train_top1 = float(row.get("trainable", {}).get("top1", 0.0))
        frozen = row.get("frozen")
        frozen_top1 = float(frozen.get("top1", 0.0)) if isinstance(frozen, dict) else 0.0
        delta = float(row.get("delta_trainable_minus_frozen", 0.0) or 0.0)
        controls_pass = bool(control.get("control_pass", {}).get("overall", False))
        shortcut_max = _shortcut_max_15(control)
        score = train_top1 + delta - max(0.0, shortcut_max - float(control.get("chance", 0.125))) + (0.2 if controls_pass else -0.5)
        leaders.append(
            {
                "variant": row["variant"],
                "seed": int(row["seed"]),
                "family": row["family"],
                "status": row["status"],
                "trainable_top1": train_top1,
                "frozen_top1": frozen_top1,
                "delta": delta,
                "controls_pass": controls_pass,
                "shortcut_max": shortcut_max,
                "param_count": int(row.get("param_count", 0)),
                "score": float(score),
            }
        )
    return sorted(leaders, key=lambda item: float(item["score"]), reverse=True)


def _shortcut_max_15(control: Dict[str, object]) -> float:
    values = [
        float(control.get("candidate_action_only", {}).get("top1", 0.0)),
        float(control.get("rollout_only", {}).get("top1", 0.0)),
        float(control.get("constraint_only", {}).get("top1", 0.0)),
        float(control.get("endpoint_only", {}).get("top1", 0.0)),
        float(control.get("length_only", 0.0)),
        float(control.get("action_unigram", 0.0)),
        float(control.get("action_bigram", 0.0)),
        float(control.get("family_proxy_probe", {}).get("top1", 0.0)),
    ]
    return max(values)


def _failure_taxonomy_15(
    rows: Sequence[Dict[str, object]],
    controls: Sequence[Dict[str, object]],
    shortcuts: Sequence[Dict[str, object]],
    micro_rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    counts: Dict[str, int] = {}
    for row in micro_rows:
        if not bool(row.get("pass")):
            counts["micro_overfit_failure"] = counts.get("micro_overfit_failure", 0) + 1
    for row in rows:
        if row.get("frozen") and float(row.get("frozen", {}).get("top1", 0.0)) >= 0.50:
            counts["frozen_high_suspicious"] = counts.get("frozen_high_suspicious", 0) + 1
        if row.get("delta_trainable_minus_frozen") is not None and float(row.get("delta_trainable_minus_frozen", 0.0)) <= 0.0:
            counts["did_not_beat_frozen"] = counts.get("did_not_beat_frozen", 0) + 1
    for row in controls:
        if not bool(row.get("control_pass", {}).get("overall", False)):
            counts["control_failure"] = counts.get("control_failure", 0) + 1
        if _shortcut_max_15(row) > float(row.get("chance", 0.125)) + NEAR_CHANCE_MARGIN:
            counts["shortcut_above_chance"] = counts.get("shortcut_above_chance", 0) + 1
    for row in shortcuts:
        if not bool(row.get("near_chance", False)):
            counts["shortcut_decomposition_above_chance"] = counts.get("shortcut_decomposition_above_chance", 0) + 1
    return {
        "counts": dict(sorted(counts.items())),
        "failure_reasons": {
            "micro_overfit_failure": "variant could not memorize small repaired learned-constraint sets",
            "control_failure": "one or more shortcut, mismatch, shuffle, audit, or ablation gates failed",
            "shortcut_above_chance": "model-side shortcut control exceeded chance window",
            "shortcut_decomposition_above_chance": "standalone shortcut probe exceeded chance window",
            "frozen_high_suspicious": "same-architecture frozen comparator solved too much of the task",
            "did_not_beat_frozen": "trainable shared model did not beat exact frozen comparator",
        },
    }


def _summary_15(
    rows: Sequence[Dict[str, object]],
    controls: Sequence[Dict[str, object]],
    shortcuts: Sequence[Dict[str, object]],
    leaderboard: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    by_variant: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    control_by_key = {(row["variant"], row["seed"]): row for row in controls}
    variant_summaries = []
    for variant, group in by_variant.items():
        deltas = [float(row.get("delta_trainable_minus_frozen", 0.0) or 0.0) for row in group if row.get("delta_trainable_minus_frozen") is not None]
        controls_pass = all(bool(control_by_key.get((row["variant"], row["seed"]), {}).get("control_pass", {}).get("overall", False)) for row in group)
        variant_summaries.append(
            {
                "variant": variant,
                "mean_trainable_top1": _mean([float(row["trainable"]["top1"]) for row in group]),
                "mean_frozen_top1": _mean([float(row.get("frozen", {}).get("top1", 0.0)) for row in group if isinstance(row.get("frozen"), dict)]),
                "mean_delta": _mean(deltas),
                "seeds_trainable_beats_frozen": int(sum(delta > 0.0 for delta in deltas)),
                "controls_pass_all": controls_pass,
                "success_gate": bool(deltas and sum(delta > 0.0 for delta in deltas) >= 2 and _mean(deltas) >= DELTA_GATE and controls_pass),
            }
        )
    chance_values = [float(row["chance"]) for row in shortcuts]
    baseline_mean = _mean(chance_values)
    shortcut_means = _mean([float(row["shortcut_max_controlled"]) for row in shortcuts])
    medium_ready = any(bool(row["success_gate"]) for row in variant_summaries)
    return {
        "best_variant": leaderboard[0].get("variant") if leaderboard else None,
        "random_mean": baseline_mean,
        "shortcut_max_mean": shortcut_means,
        "shortcut_decomposition_near_chance": all(bool(row.get("near_chance", False)) for row in shortcuts),
        "variant_summaries": sorted(variant_summaries, key=lambda row: float(row["mean_trainable_top1"]), reverse=True),
        "medium_ready": bool(medium_ready),
        "medium_validation_launched": False,
        "architecture_search_should_resume": bool(medium_ready),
    }


def _family_shortcut_rows(splits: Dict[str, List[ConstraintExample]], shortcut_row: Dict[str, object], seed: int) -> List[Dict[str, object]]:
    rows = []
    for family, values in shortcut_row["by_family"].items():
        rows.append({"benchmark": BENCHMARK, "seed": int(seed), "family": family, "kind": "shortcut", **values})
    return rows


def _family_variant_rows(row: Dict[str, object], control: Dict[str, object], examples: Sequence[ConstraintExample]) -> List[Dict[str, object]]:
    trainable_by_family = row.get("trainable", {}).get("by_constraint_family", {})
    frozen_by_family = row.get("frozen", {}).get("by_constraint_family", {}) if isinstance(row.get("frozen"), dict) else {}
    rows = []
    for family in FAMILIES:
        trainable = float(trainable_by_family.get(family, 0.0))
        frozen = float(frozen_by_family.get(family, 0.0)) if frozen_by_family else 0.0
        rows.append(
            {
                "benchmark": BENCHMARK,
                "seed": int(row["seed"]),
                "family": family,
                "kind": "variant",
                "variant": row["variant"],
                "trainable_top1": trainable,
                "frozen_top1": frozen,
                "delta": trainable - frozen if frozen_by_family else None,
                "controls_pass": bool(control.get("control_pass", {}).get("overall", False)),
                "count": sum(1 for example in examples if example.family == family),
            }
        )
    return rows


def _error_rows_15(variant: ConstraintVariant, seed: int, examples: Sequence[ConstraintExample], train_logits: np.ndarray, frozen_logits: np.ndarray, limit: int) -> List[Dict[str, object]]:
    labels = _labels(examples)
    train_pred = np.argmax(train_logits, axis=1)
    frozen_pred = np.argmax(frozen_logits, axis=1)
    rows = []
    for index, example in enumerate(examples):
        if len(rows) >= limit:
            break
        if train_pred[index] == labels[index]:
            continue
        rows.append(
            {
                "type": "plan_arch1_5_error_case",
                "variant": variant.name,
                "seed": int(seed),
                "example_id": example.id,
                "family": example.family,
                "target_order": example.metadata.get("target_order"),
                "failure_type": example.candidate_types[int(train_pred[index])],
                "gold_candidate_index": int(example.label),
                "predicted_candidate_index": int(train_pred[index]),
                "frozen_candidate_index": int(frozen_pred[index]),
                "gold_vs_best_negative_margin": float(train_logits[index, int(example.label)] - np.max(np.delete(train_logits[index], int(example.label)))),
                "candidate_probabilities": _softmax_np(train_logits[index]).tolist(),
                "candidate_types": list(example.candidate_types),
            }
        )
    return rows


def _write_outputs(
    result: Dict[str, object],
    results_path: Path,
    controls_path: Path,
    shortcut_path: Path,
    counterfactual_path: Path,
    family_path: Path,
    leaderboard_path: Path,
    failure_path: Path,
    attention_path: Path,
    error_path: Path,
    compute_path: Path,
    report_path: Path,
) -> None:
    for path in (results_path, controls_path, shortcut_path, counterfactual_path, family_path, leaderboard_path, failure_path, attention_path, error_path, compute_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(_trim_result(result), indent=2, sort_keys=True), encoding="utf-8")
    controls_path.write_text(json.dumps({"controls": result["controls"]}, indent=2, sort_keys=True), encoding="utf-8")
    shortcut_path.write_text(json.dumps({"shortcut_decomposition": result["shortcut_decomposition"]}, indent=2, sort_keys=True), encoding="utf-8")
    counterfactual_path.write_text(json.dumps({"counterfactual_candidate_audit": result["counterfactual_candidate_audit"]}, indent=2, sort_keys=True), encoding="utf-8")
    family_path.write_text(json.dumps({"family_results": result["family_results"]}, indent=2, sort_keys=True), encoding="utf-8")
    failure_path.write_text(json.dumps(result["failure_taxonomy"], indent=2, sort_keys=True), encoding="utf-8")
    compute_path.write_text(json.dumps({"compute_metrics": result["compute_metrics"]}, indent=2, sort_keys=True), encoding="utf-8")
    with leaderboard_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["variant", "seed", "family", "status", "trainable_top1", "frozen_top1", "delta", "controls_pass", "shortcut_max", "param_count", "score"]
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
    shortcuts = result["shortcut_decomposition"]
    shortcut_means = _mean_baselines(shortcuts)
    best_minimal = _variant_summary(summary, "minimal_transition_cross_attention")
    clones = [row for row in summary["variant_summaries"] if row["variant"] != "minimal_transition_cross_attention"]
    best_clone = max(clones, key=lambda row: float(row["mean_trainable_top1"]), default={})
    lines = [
        "# PLAN-ARCH-1.5 Learned-Constraint Shortcut Repair",
        "",
        "## Scope",
        "- No planning claim.",
        "- No architecture improvement claim.",
        "- Medium validation was not launched.",
        "- Candidate self-attention, pyramids, stacking, and multi-avenue variants were not run.",
        "",
        "## Answers",
        f"1. What shortcut explained the PLAN-ARCH-1.4 scores? `{shortcuts[0]['plan_arch1_4_shortcut_explanation'] if shortcuts else 'not available'}`.",
        f"2. Does the repaired dataset make endpoint, action, rollout-only, and family-proxy baselines near chance? `{summary['shortcut_decomposition_near_chance']}` (shortcut_max_mean={summary['shortcut_max_mean']:.4f}, random={summary['random_mean']:.4f}).",
        f"3. Are constraint examples necessary? `{_necessity_answer(result['controls'], 'constraint_mismatch_collapses', 'ablate_constraint_view_hurts')}`.",
        f"4. Is rollout evidence necessary? `{_necessity_answer(result['controls'], 'rollout_mismatch_collapses', 'ablate_rollout_transition_view_hurts')}`.",
        f"5. Does rollout + constraint evidence outperform either alone? `{_two_key_answer(shortcut_means)}`.",
        f"6. Which constraint family remains learnable after repair? `{_family_answer(result['rows'])}`.",
        f"7. Does trainable beat frozen? `{_trainable_beats_answer(summary)}`.",
        f"8. Does the clone architecture beat minimal cross-attention? `{float(best_clone.get('mean_trainable_top1', 0.0)) > float(best_minimal.get('mean_trainable_top1', 0.0))}`.",
        f"9. Are any variants valid enough for medium validation? `{summary['medium_ready']}`.",
        f"10. Should architecture search resume after shortcut repair? `{summary['architecture_search_should_resume']}`.",
        "",
        "## Shortcut Decomposition",
        "| baseline | mean top1 |",
        "| --- | ---: |",
    ]
    for key, value in shortcut_means.items():
        lines.append(f"| {key} | {value:.4f} |")
    lines.extend(["", "## Leaderboard", "| variant | seed | trainable | frozen | delta | shortcut | controls |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"])
    for row in result["leaderboard"][:20]:
        lines.append(
            f"| {row['variant']} | {row['seed']} | {float(row['trainable_top1']):.4f} | {float(row['frozen_top1']):.4f} | "
            f"{float(row['delta']):.4f} | {float(row['shortcut_max']):.4f} | `{bool(row['controls_pass'])}` |"
        )
    lines.extend(["", "## Variant Summary", "| variant | trainable mean | frozen mean | mean delta | seed wins | controls | gate |", "| --- | ---: | ---: | ---: | ---: | --- | --- |"])
    for row in summary["variant_summaries"]:
        lines.append(
            f"| {row['variant']} | {float(row['mean_trainable_top1']):.4f} | {float(row['mean_frozen_top1']):.4f} | "
            f"{float(row['mean_delta']):.4f} | {int(row['seeds_trainable_beats_frozen'])} | `{bool(row['controls_pass_all'])}` | `{bool(row['success_gate'])}` |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "This stage repairs the learned-constraint benchmark and tests whether constraint-conditioned candidate-plan verification is cleanly learnable. It does not claim planning or an architecture improvement.",
        ]
    )
    return "\n".join(lines) + "\n"


def _mean_baselines(shortcuts: Sequence[Dict[str, object]]) -> Dict[str, float]:
    keys = sorted({key for row in shortcuts for key in row.get("baselines", {}).keys()})
    return {key: _mean([float(row["baselines"][key]) for row in shortcuts if key in row.get("baselines", {})]) for key in keys}


def _two_key_answer(shortcut_means: Dict[str, float]) -> str:
    rollout = float(shortcut_means.get("rollout_only", 0.0))
    constraint = float(shortcut_means.get("constraint_examples_only", 0.0))
    both = float(shortcut_means.get("rollout_plus_constraint_oracle", 0.0))
    chance = float(shortcut_means.get("random", 0.125))
    return f"rollout_only={rollout:.4f}, constraint_only={constraint:.4f}, rollout_plus_constraint={both:.4f}, pass={rollout <= chance + NEAR_CHANCE_MARGIN and constraint <= chance + NEAR_CHANCE_MARGIN and both > max(rollout, constraint) + NEAR_CHANCE_MARGIN}"


def _variant_summary(summary: Dict[str, object], name: str) -> Dict[str, object]:
    return next((row for row in summary["variant_summaries"] if row["variant"] == name), {})


def _ablation_answer(controls: Sequence[Dict[str, object]], key: str) -> str:
    values = [bool(row.get("control_pass", {}).get(key, False)) for row in controls]
    return f"{sum(values)}/{len(values)} pass" if values else "not run"


def _necessity_answer(controls: Sequence[Dict[str, object]], mismatch_key: str, ablation_key: str) -> str:
    if not controls:
        return "not run"
    mismatch = [bool(row.get("control_pass", {}).get(mismatch_key, False)) for row in controls]
    ablation = [bool(row.get("control_pass", {}).get(ablation_key, False)) for row in controls]
    clean_above_chance = [float(row.get("clean", {}).get("top1", 0.0)) > float(row.get("chance", 0.125)) + NEAR_CHANCE_MARGIN for row in controls]
    established = all(mismatch) and all(ablation) and all(clean_above_chance)
    if established:
        return f"yes ({sum(mismatch)}/{len(mismatch)} mismatch collapse, {sum(ablation)}/{len(ablation)} ablation hurt)"
    return f"not established (mismatch={sum(mismatch)}/{len(mismatch)}, ablation_hurts={sum(ablation)}/{len(ablation)}, clean_above_chance={sum(clean_above_chance)}/{len(clean_above_chance)})"


def _trainable_beats_answer(summary: Dict[str, object]) -> str:
    winners = [row["variant"] for row in summary["variant_summaries"] if int(row["seeds_trainable_beats_frozen"]) >= 2]
    return ", ".join(winners) if winners else "none"


def _family_answer(rows: Sequence[Dict[str, object]]) -> str:
    best: Dict[str, float] = {}
    for row in rows:
        metric = row.get("trainable", {})
        by_family = metric.get("by_constraint_family", {}) if isinstance(metric, dict) else {}
        for family, value in by_family.items():
            best[str(family)] = max(best.get(str(family), 0.0), float(value))
    return "; ".join(f"{family}={value:.3f}" for family, value in sorted(best.items())) if best else "none"


def _plan14_shortcut_explanation() -> str:
    path = Path("results/plan_arch1_4_controls.json")
    if not path.exists():
        return "PLAN-ARCH-1.4 controls unavailable; expected shortcut was candidate/source and family construction artifacts."
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        controls = data.get("controls", [])
        fields = {
            "candidate_only": lambda row: float(row.get("candidate_only", {}).get("top1", 0.0)),
            "state_goal_only": lambda row: float(row.get("state_goal_only", {}).get("top1", 0.0)),
            "constraint_examples_only": lambda row: float(row.get("constraint_examples_only", {}).get("top1", 0.0)),
            "action_bigram": lambda row: float(row.get("action_bigram", 0.0)),
        }
        means = {name: _mean([fn(row) for row in controls]) for name, fn in fields.items()}
        winner = max(means, key=means.get)
        return f"{winner} was highest among recorded 1.4 shortcut controls (means={{{', '.join(f'{k}:{v:.4f}' for k, v in means.items())}}}); this points to candidate/source-family construction artifacts, not endpoint solving."
    except Exception as exc:
        return f"could not parse PLAN-ARCH-1.4 controls ({exc}); expected shortcut was candidate/source and family construction artifacts."


def _trim_result(result: Dict[str, object]) -> Dict[str, object]:
    trimmed = dict(result)
    trimmed["attention_summaries"] = result.get("attention_summaries", [])[:40]
    trimmed["error_cases"] = result.get("error_cases", [])[:80]
    return trimmed


def _default_config(config: Dict[str, object]) -> Dict[str, object]:
    base: Dict[str, object] = {
        "device": "cpu",
        "seeds": [0, 1, 2],
        "dataset": {
            "grid_size": 7,
            "num_candidates": 8,
            "plan_length": 48,
            "train_examples": 256,
            "dev_examples": 64,
            "test_examples": 64,
            "families": list(FAMILIES),
            "all_goal_every": 1,
        },
        "training": {"epochs": 8, "batch_size": 64, "lr": 0.003, "weight_decay": 0.0, "patience": 4, "gradient_clip_norm": 1.0},
        "micro_overfit_training": {"epochs": 40, "batch_size": 32, "lr": 0.003, "weight_decay": 0.0, "patience": 40, "gradient_clip_norm": 1.0},
    }
    return _deep_update(base, config)


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


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


if __name__ == "__main__":
    main()
