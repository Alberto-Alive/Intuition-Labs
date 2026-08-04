from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


BENCHMARK = "plan_arch1_4_learned_constraint_planning"
DEFAULT_CONFIG = "configs/plan_arch1_4_learned_constraint_planning.json"
DEFAULT_RESULTS = "results/plan_arch1_4_results.json"
DEFAULT_CONTROLS = "results/plan_arch1_4_controls.json"
DEFAULT_HEURISTICS = "results/plan_arch1_4_heuristic_baselines.json"
DEFAULT_LEADERBOARD = "results/plan_arch1_4_variant_leaderboard.csv"
DEFAULT_FAILURES = "results/plan_arch1_4_failure_taxonomy.json"
DEFAULT_ATTENTION = "results/plan_arch1_4_attention_summaries.jsonl"
DEFAULT_ERRORS = "results/plan_arch1_4_error_cases.jsonl"
DEFAULT_COMPUTE = "results/plan_arch1_4_compute_metrics.json"
DEFAULT_REPORT = "reports/PLAN_ARCH1_4_LEARNED_CONSTRAINT_PLANNING.md"

ACTIONS = ("U", "D", "L", "R")
ACTION_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
REVERSE_ACTION = {0: 1, 1: 0, 2: 3, 3: 2}
FAMILIES = ("color_zone", "ordered_subgoal", "key_door")

ROLE_IDS = {
    "pad": 0,
    "state": 1,
    "goal": 2,
    "obstacle": 3,
    "terrain": 4,
    "subgoal_a": 5,
    "subgoal_b": 6,
    "key": 7,
    "door": 8,
    "rollout": 9,
    "transition": 10,
    "action": 11,
    "constraint_valid": 12,
    "constraint_invalid": 13,
    "rule": 14,
    "summary": 15,
}
FAMILY_IDS = {"color_zone": 1, "ordered_subgoal": 2, "key_door": 3}
FIELD_VOCABS = (24, 16, 16, 32, 64, 8, 16, 16)


@dataclass(frozen=True)
class ConstraintDatasetConfig:
    grid_size: int = 7
    num_candidates: int = 8
    plan_length: int = 32
    train_examples: int = 256
    dev_examples: int = 64
    test_examples: int = 64
    families: Tuple[str, ...] = FAMILIES
    all_goal_every: int = 4


@dataclass(frozen=True)
class ConstraintTrainingConfig:
    epochs: int = 8
    batch_size: int = 64
    lr: float = 0.003
    weight_decay: float = 0.0
    patience: int = 4
    gradient_clip_norm: float = 1.0


@dataclass(frozen=True)
class ConstraintVariant:
    name: str
    family: str
    description: str
    view_names: Tuple[str, ...]
    include_final_state: bool = True
    model_dim: int = 32
    num_heads: int = 2
    ff_dim: int = 64
    minimal: bool = False


@dataclass(frozen=True)
class ConstraintEvidence:
    label: str
    tokens: Tuple[Tuple[int, int, int], ...]


@dataclass(frozen=True)
class ConstraintExample:
    id: str
    split: str
    family: str
    grid_size: int
    start: Tuple[int, int]
    goal: Tuple[int, int]
    obstacles: Tuple[Tuple[int, int], ...]
    terrain: Tuple[Tuple[int, int, int], ...]
    subgoal_a: Tuple[int, int] | None
    subgoal_b: Tuple[int, int] | None
    key: Tuple[int, int] | None
    door: Tuple[int, int] | None
    rule: Dict[str, object]
    constraint_examples: Tuple[ConstraintEvidence, ...]
    candidates: Tuple[Tuple[int, ...], ...]
    rollouts: Tuple[Tuple[Tuple[int, int], ...], ...]
    reached_goal: Tuple[bool, ...]
    satisfies_constraint: Tuple[bool, ...]
    candidate_types: Tuple[str, ...]
    label: int
    metadata: Dict[str, object]


@dataclass
class ConstraintFitResult:
    method: str
    model: nn.Module
    variant: ConstraintVariant
    trainable_shared: bool
    audit: Dict[str, object]
    history: List[Dict[str, float]]
    training_time_seconds: float
    param_count: int


class ConstraintTokenEncoder(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(size, model_dim) for size in FIELD_VOCABS])
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        fields = fields.long()
        out = torch.zeros((*fields.shape[:-1], self.embeddings[0].embedding_dim), dtype=torch.float32, device=fields.device)
        for index, embedding in enumerate(self.embeddings):
            values = fields[..., index].clamp(min=0, max=embedding.num_embeddings - 1)
            out = out + embedding(values)
        return self.norm(out)


class ConstraintCloneVerifier(nn.Module):
    def __init__(self, variant: ConstraintVariant) -> None:
        super().__init__()
        self.variant = variant
        dim = _compatible_dim(variant.model_dim, variant.num_heads)
        self.dim = dim
        self.token_encoder = ConstraintTokenEncoder(dim)
        self.evidence_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.candidate_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.attn = nn.MultiheadAttention(dim, int(variant.num_heads), batch_first=True, dropout=0.0)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, int(variant.ff_dim)), nn.GELU(), nn.Linear(int(variant.ff_dim), dim))
        self.score_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def shared_parameter_items(self) -> List[Tuple[str, nn.Parameter]]:
        modules = ("token_encoder", "evidence_proj", "candidate_proj", "attn", "ff")
        return [(name, parameter) for name, parameter in self.named_parameters() if name.split(".", 1)[0] in modules]

    def coordinator_parameter_items(self) -> List[Tuple[str, nn.Parameter]]:
        return [(name, parameter) for name, parameter in self.named_parameters() if name.split(".", 1)[0] == "score_head"]

    def configure_shared_trainable(self, trainable: bool) -> None:
        shared_ids = {id(parameter) for _name, parameter in self.shared_parameter_items()}
        for parameter in self.parameters():
            parameter.requires_grad = True
        if not trainable:
            for parameter in self.parameters():
                if id(parameter) in shared_ids:
                    parameter.requires_grad = False

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        hidden_state_shuffle: bool = False,
        shuffle_seed: int = 0,
        return_attention: bool = False,
    ) -> Dict[str, object]:
        evidence = batch["evidence_fields"]
        evidence_mask = batch["evidence_mask"].bool()
        candidate = batch["candidate_fields"]
        candidate_mask = batch["candidate_mask"].bool()
        bsz, cand_count, view_count, evid_tokens, fields = evidence.shape
        cand_tokens = candidate.shape[2]

        evid_emb = self.token_encoder(evidence.reshape(bsz * cand_count * view_count, evid_tokens, fields))
        evid_emb = self.evidence_proj(evid_emb)
        evid_emb = evid_emb.reshape(bsz, cand_count, view_count * evid_tokens, -1)
        evid_mask = evidence_mask.reshape(bsz, cand_count, view_count * evid_tokens)
        if hidden_state_shuffle and bsz > 1:
            generator = torch.Generator(device=evid_emb.device)
            generator.manual_seed(int(shuffle_seed))
            order = torch.randperm(bsz, generator=generator, device=evid_emb.device)
            evid_emb = evid_emb.index_select(0, order)
            evid_mask = evid_mask.index_select(0, order)

        cand_emb = self.token_encoder(candidate.reshape(bsz * cand_count, cand_tokens, fields))
        cand_emb = self.candidate_proj(cand_emb)
        cand_mask = candidate_mask.reshape(bsz * cand_count, cand_tokens)
        query = _masked_mean(cand_emb, cand_mask).unsqueeze(1)
        flat_evid = evid_emb.reshape(bsz * cand_count, view_count * evid_tokens, -1)
        flat_mask = evid_mask.reshape(bsz * cand_count, view_count * evid_tokens)
        attended, weights = self.attn(query, flat_evid, flat_evid, key_padding_mask=~flat_mask, need_weights=return_attention, average_attn_weights=False)
        fused = query.squeeze(1) + attended.squeeze(1)
        fused = fused + self.ff(fused)
        pooled = fused.reshape(bsz, cand_count, -1)
        return {"logits": self.score_head(pooled).squeeze(-1), "attention_weights": weights, "context_mask": flat_mask}


class CandidateOnlyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        bsz, cand_count, dim = features.shape
        return self.net(features.reshape(bsz * cand_count, dim)).reshape(bsz, cand_count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PLAN-ARCH-1.4 learned constraint planning benchmark.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_RESULTS)
    parser.add_argument("--controls-output", default=DEFAULT_CONTROLS)
    parser.add_argument("--heuristics-output", default=DEFAULT_HEURISTICS)
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
    result = run_plan_arch1_4(config)
    _write_outputs(
        result,
        Path(args.output),
        Path(args.controls_output),
        Path(args.heuristics_output),
        Path(args.leaderboard_output),
        Path(args.failure_output),
        Path(args.attention_output),
        Path(args.error_output),
        Path(args.compute_output),
        Path(args.report),
    )


def run_plan_arch1_4(config: Dict[str, object]) -> Dict[str, object]:
    started = time.perf_counter()
    dataset_config = _dataset_config(config.get("dataset", {}))
    training = _training_config(config.get("training", {}))
    micro_training = _training_config({**dict(config.get("training", {})), **dict(config.get("micro_overfit_training", {}))})
    seeds = [int(seed) for seed in config.get("seeds", [0, 1, 2])]
    device = _resolve_device(str(config.get("device", "cpu")))
    variants = _variant_plan()
    print(f"plan-arch1.4: device={device} seeds={seeds} variants={len(variants)} no_medium_validation=True")

    rows: List[Dict[str, object]] = []
    controls: List[Dict[str, object]] = []
    heuristic_rows: List[Dict[str, object]] = []
    attention_rows: List[Dict[str, object]] = []
    error_rows: List[Dict[str, object]] = []
    compute_rows: List[Dict[str, object]] = []
    micro_rows: List[Dict[str, object]] = []

    for variant in variants:
        print(f"plan-arch1.4 micro variant={variant.name}")
        micro_rows.extend(_run_micro_overfit_gates(variant, dataset_config, micro_training, seeds[0] if seeds else 0, device))

    for seed in seeds:
        splits = build_constraint_splits(dataset_config, seed)
        heuristic_rows.append(_heuristic_baseline_row(splits, seed, config))
        for variant in variants:
            print(f"plan-arch1.4 cheap variant={variant.name} seed={seed}")
            start = time.perf_counter()
            if variant.minimal:
                fit = fit_constraint_verifier(splits["train"], splits["dev"], variant, training, seed + 30_001, device, True, f"minimal__{variant.name}")
                logits = predict_constraint_logits(fit.model, splits["test"], variant, training.batch_size, device)
                labels = _labels(splits["test"])
                metric = _metric_block(logits, labels, splits["test"])
                control = _run_controls(fit.model, variant, splits["test"], logits, labels, training.batch_size, device, seed)
                row = _result_row("cheap", variant, seed, metric, None, None, fit, control, splits, time.perf_counter() - start)
            else:
                trainable = fit_constraint_verifier(splits["train"], splits["dev"], variant, training, seed + 10_001, device, True, f"trainable__{variant.name}")
                frozen = fit_constraint_verifier(splits["train"], splits["dev"], variant, training, seed + 10_001, device, False, f"frozen__{variant.name}")
                train_logits = predict_constraint_logits(trainable.model, splits["test"], variant, training.batch_size, device)
                frozen_logits = predict_constraint_logits(frozen.model, splits["test"], variant, training.batch_size, device)
                labels = _labels(splits["test"])
                train_metric = _metric_block(train_logits, labels, splits["test"])
                frozen_metric = _metric_block(frozen_logits, labels, splits["test"])
                control = _run_controls(trainable.model, variant, splits["test"], train_logits, labels, training.batch_size, device, seed)
                row = _result_row("cheap", variant, seed, train_metric, frozen_metric, train_metric["top1"] - frozen_metric["top1"], trainable, control, splits, time.perf_counter() - start)
                row["frozen_audit"] = frozen.audit
                error_rows.extend(_error_rows(variant, seed, splits["test"], train_logits, frozen_logits, limit=20))
                attention_rows.extend(_attention_summaries(trainable.model, variant, splits["test"][: min(8, len(splits["test"]))], training.batch_size, device, seed))
            controls.append(control)
            rows.append(row)
            compute_rows.append(_compute_row(variant, seed, splits, row, time.perf_counter() - start))
            _clear_cuda()

    leaderboard = _leaderboard(rows, controls)
    failure_taxonomy = _failure_taxonomy(rows, controls, heuristic_rows, micro_rows)
    summary = _summary(rows, controls, heuristic_rows, leaderboard)
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
            "scope": "learned constraint candidate-plan verification benchmark; no medium validation launched",
            "device": device,
            "families": list(dataset_config.families),
            "claim_boundary": "No autonomous planning claim. No world-model claim. No architecture improvement claim unless gates pass.",
            "medium_validation_launched": False,
        },
        "dataset_config": asdict(dataset_config),
        "training_config": asdict(training),
        "micro_training_config": asdict(micro_training),
        "variants": [asdict(variant) for variant in variants],
        "micro_overfit": micro_rows,
        "rows": rows,
        "controls": controls,
        "heuristic_baselines": heuristic_rows,
        "leaderboard": leaderboard,
        "failure_taxonomy": failure_taxonomy,
        "attention_summaries": attention_rows,
        "error_cases": error_rows,
        "compute_metrics": compute_rows,
        "summary": summary,
    }


def build_constraint_splits(config: ConstraintDatasetConfig, seed: int) -> Dict[str, List[ConstraintExample]]:
    counts = {"train": int(config.train_examples), "dev": int(config.dev_examples), "test": int(config.test_examples)}
    splits: Dict[str, List[ConstraintExample]] = {}
    rng = np.random.default_rng(seed)
    offset = 0
    for split, count in counts.items():
        rows = []
        for index in range(count):
            rows.append(_generate_example(config, split, index, int(rng.integers(0, 2_000_000_000)) + offset))
        splits[split] = rows
        offset += 100_000
    return splits


def _generate_example(config: ConstraintDatasetConfig, split: str, index: int, seed: int) -> ConstraintExample:
    rng = np.random.default_rng(seed)
    family = str(config.families[index % len(config.families)])
    if family == "color_zone":
        return _generate_color_example(config, split, index, rng)
    if family == "ordered_subgoal":
        return _generate_ordered_example(config, split, index, rng)
    if family == "key_door":
        return _generate_key_door_example(config, split, index, rng)
    raise ValueError(f"unknown constraint family: {family}")


def _generate_color_example(config: ConstraintDatasetConfig, split: str, index: int, rng: np.random.Generator) -> ConstraintExample:
    grid = int(config.grid_size)
    start, goal = (0, 0), (grid - 1, grid - 1)
    waypoints = [(1, grid - 2), (grid - 2, 1), (grid // 2, grid - 1), (grid - 1, grid // 2)]
    required_color = int(index % 4)
    terrain = tuple((row, col, color) for color, (row, col) in enumerate(waypoints))
    records = []
    for color, waypoint in enumerate(waypoints):
        actions = _actions_via_waypoints(start, [waypoint, goal], horizontal_first=bool((index + color) % 2))
        records.append((tuple(actions), "gold_required_color" if color == required_color else "goal_wrong_color", color == required_color))
    lure_color = int((required_color + 2) % 4)
    records.append((tuple(_with_detour(_actions_via_waypoints(start, [waypoints[lure_color], goal], horizontal_first=True), index)), "endpoint_lure_wrong_color", False))
    for salt in range(3):
        color = int((required_color + 1 + salt) % 4)
        waypoint = waypoints[color]
        actions = _actions_via_waypoints(start, [waypoint, goal], horizontal_first=bool(salt % 2))
        records.append((tuple(_with_detour(actions, salt)), "goal_wrong_color_distractor", False))
    return _finalize_example(
        config,
        split,
        index,
        "color_zone",
        start,
        goal,
        tuple(),
        terrain,
        None,
        None,
        None,
        None,
        {"family": "color_zone", "required_color": required_color},
        (
            ConstraintEvidence("valid", ((ROLE_IDS["terrain"], waypoints[required_color][0], waypoints[required_color][1]),)),
            ConstraintEvidence("invalid", ((ROLE_IDS["terrain"], waypoints[(required_color + 1) % 4][0], waypoints[(required_color + 1) % 4][1]),)),
        ),
        records,
        rng,
    )


def _generate_ordered_example(config: ConstraintDatasetConfig, split: str, index: int, rng: np.random.Generator) -> ConstraintExample:
    grid = int(config.grid_size)
    start, goal = (0, 0), (grid - 1, grid - 1)
    pairs = [((1, grid - 2), (grid - 2, 1)), ((2, grid - 2), (grid - 3, 1)), ((1, grid // 2), (grid - 2, grid // 2))]
    subgoal_a, subgoal_b = pairs[index % len(pairs)]
    order_ab = bool((index // len(pairs)) % 2 == 0)
    first, second = (subgoal_a, subgoal_b) if order_ab else (subgoal_b, subgoal_a)
    records = [
        (tuple(_actions_via_waypoints(start, [first, second, goal], horizontal_first=False)), "gold_ordered_subgoals", True),
        (tuple(_actions_via_waypoints(start, [second, first, goal], horizontal_first=True)), "wrong_subgoal_order", False),
        (tuple(_actions_via_waypoints(start, [first, goal], horizontal_first=False)), "skip_second_subgoal", False),
        (tuple(_actions_via_waypoints(start, [second, goal], horizontal_first=True)), "skip_first_subgoal", False),
        (tuple(_actions_via_waypoints(start, [goal], horizontal_first=False)), "direct_goal_skip_subgoals", False),
        (tuple(_with_detour(_actions_via_waypoints(start, [second, first, goal], horizontal_first=True), index)), "endpoint_lure_wrong_subgoal_order", False),
        (tuple(_actions_via_waypoints(start, [(grid // 2, 1), goal], horizontal_first=False)), "similar_stats_distractor", False),
        (tuple(_with_detour(_actions_via_waypoints(start, [second, first, goal], horizontal_first=False), index)), "wrong_order_distractor", False),
    ]
    return _finalize_example(
        config,
        split,
        index,
        "ordered_subgoal",
        start,
        goal,
        tuple(),
        tuple(),
        subgoal_a,
        subgoal_b,
        None,
        None,
        {"family": "ordered_subgoal", "order": "A_BEFORE_B" if order_ab else "B_BEFORE_A"},
        (
            ConstraintEvidence("valid", ((_subgoal_role(first, subgoal_a), first[0], first[1]), (_subgoal_role(second, subgoal_a), second[0], second[1]))),
            ConstraintEvidence("invalid", ((_subgoal_role(second, subgoal_a), second[0], second[1]), (_subgoal_role(first, subgoal_a), first[0], first[1]))),
        ),
        records,
        rng,
    )


def _generate_key_door_example(config: ConstraintDatasetConfig, split: str, index: int, rng: np.random.Generator) -> ConstraintExample:
    grid = int(config.grid_size)
    start, goal = (0, 0), (grid - 1, grid - 1)
    pairs = [((1, grid - 2), (grid - 2, 1)), ((2, grid - 2), (grid - 2, 2)), ((1, grid - 3), (grid - 3, 1))]
    key, door = pairs[index % len(pairs)]
    records = [
        (tuple(_actions_via_waypoints(start, [key, door, goal], horizontal_first=False)), "gold_key_before_door", True),
        (tuple(_actions_via_waypoints(start, [door, key, goal], horizontal_first=True)), "key_door_violation", False),
        (tuple(_actions_via_waypoints(start, [door, goal], horizontal_first=False)), "door_without_key", False),
        (tuple(_actions_via_waypoints(start, [key, goal], horizontal_first=True)), "key_without_door", False),
        (tuple(_actions_via_waypoints(start, [goal], horizontal_first=False)), "direct_goal_no_keydoor", False),
        (tuple(_with_detour(_actions_via_waypoints(start, [door, key, goal], horizontal_first=True), index)), "endpoint_lure_keydoor_violation", False),
        (tuple(_actions_via_waypoints(start, [(grid // 2, grid // 2), goal], horizontal_first=False)), "similar_stats_distractor", False),
        (tuple(_with_detour(_actions_via_waypoints(start, [door, key, goal], horizontal_first=False), index)), "door_first_distractor", False),
    ]
    return _finalize_example(
        config,
        split,
        index,
        "key_door",
        start,
        goal,
        tuple(),
        tuple(),
        None,
        None,
        key,
        door,
        {"family": "key_door", "rule": "KEY_BEFORE_DOOR"},
        (
            ConstraintEvidence("valid", ((ROLE_IDS["key"], key[0], key[1]), (ROLE_IDS["door"], door[0], door[1]))),
            ConstraintEvidence("invalid", ((ROLE_IDS["door"], door[0], door[1]), (ROLE_IDS["key"], key[0], key[1]))),
        ),
        records,
        rng,
    )


def _finalize_example(
    config: ConstraintDatasetConfig,
    split: str,
    index: int,
    family: str,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: Tuple[Tuple[int, int], ...],
    terrain: Tuple[Tuple[int, int, int], ...],
    subgoal_a: Tuple[int, int] | None,
    subgoal_b: Tuple[int, int] | None,
    key: Tuple[int, int] | None,
    door: Tuple[int, int] | None,
    rule: Dict[str, object],
    constraint_examples: Tuple[ConstraintEvidence, ...],
    records: Sequence[Tuple[Tuple[int, ...], str, bool]],
    rng: np.random.Generator,
) -> ConstraintExample:
    selected_records = _select_candidate_records(records, int(config.num_candidates), family, index)
    padded = _balance_action_records(selected_records, start, int(config.plan_length), int(config.grid_size))
    ordered, label = _balanced_order(padded, int(config.num_candidates), index, rng)
    candidates = tuple(record[0] for record in ordered)
    candidate_types = tuple(record[1] for record in ordered)
    rollouts = tuple(tuple(_simulate(start, actions, int(config.grid_size), obstacles)) for actions in candidates)
    reached = tuple(bool(rollout[-1] == goal) for rollout in rollouts)
    satisfies = tuple(bool(_satisfies_family(family, rollout, rule, terrain, subgoal_a, subgoal_b, key, door) and reached[idx]) for idx, rollout in enumerate(rollouts))
    if sum(bool(value) for value in satisfies) != 1 or not satisfies[label]:
        # Deterministic templates should already satisfy this. Raising here catches malformed benchmark definitions.
        raise RuntimeError(f"invalid generated constraint example family={family} index={index} satisfies={satisfies} label={label}")
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
        **rule,
    }
    return ConstraintExample(
        id=f"{split}_{index}",
        split=split,
        family=family,
        grid_size=int(config.grid_size),
        start=start,
        goal=goal,
        obstacles=obstacles,
        terrain=terrain,
        subgoal_a=subgoal_a,
        subgoal_b=subgoal_b,
        key=key,
        door=door,
        rule=rule,
        constraint_examples=constraint_examples,
        candidates=candidates,
        rollouts=rollouts,
        reached_goal=reached,
        satisfies_constraint=satisfies,
        candidate_types=candidate_types,
        label=int(label),
        metadata=metadata,
    )


def _select_candidate_records(
    records: Sequence[Tuple[Tuple[int, ...], str, bool]],
    num_candidates: int,
    family: str,
    index: int,
) -> List[Tuple[Tuple[int, ...], str, bool]]:
    gold_records = [record for record in records if bool(record[2])]
    if len(gold_records) != 1:
        raise RuntimeError(f"invalid candidate template family={family} index={index} gold_count={len(gold_records)}")
    negatives = [record for record in records if not bool(record[2])]
    if int(num_candidates) < 2 or len(negatives) < int(num_candidates) - 1:
        raise RuntimeError(f"not enough candidate templates family={family} index={index} num_candidates={num_candidates}")
    return [gold_records[0], *negatives[: int(num_candidates) - 1]]


def _variant_plan() -> List[ConstraintVariant]:
    return [
        ConstraintVariant(
            name="transition_tuple_verifier",
            family="clone_transition",
            description="State/goal/terrain/transition/constraint clone verifier.",
            view_names=("state", "goal", "terrain", "transition", "constraint"),
        ),
        ConstraintVariant(
            name="candidate_token_direct_transition_tokens",
            family="clone_transition",
            description="Candidate-token-direct verifier with action, rollout, transition, and constraint views.",
            view_names=("state", "goal", "terrain", "action", "rollout", "transition", "constraint"),
        ),
        ConstraintVariant(
            name="P1_rollout_no_final_constraint",
            family="adapted_p1",
            description="P1 rollout-no-final adapted to constraint examples.",
            view_names=("state", "goal", "terrain", "action", "rollout", "constraint"),
            include_final_state=False,
        ),
        ConstraintVariant(
            name="shared_weight_constraint_clone_verifier",
            family="constraint_clone",
            description="Shared-weight latent clone verifier with constraint and counterexample views.",
            view_names=("state", "goal", "terrain", "rollout", "transition", "constraint", "violation"),
        ),
        ConstraintVariant(
            name="minimal_transition_cross_attention",
            family="minimal_non_clone",
            description="Tiny non-clone transition cross-attention baseline over pooled candidate and evidence tokens.",
            view_names=("state", "goal", "terrain", "rollout", "transition", "constraint"),
            minimal=True,
        ),
    ]


def fit_constraint_verifier(
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
    model = ConstraintCloneVerifier(variant).to(device)
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
            batch = collate_constraint_batch(batch_examples, variant, device)
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
        dev_logits = predict_constraint_logits(model, dev_examples, variant, training.batch_size, device)
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
    return ConstraintFitResult(method, model, variant, trainable_shared, audit, history, float(time.perf_counter() - start), sum(p.numel() for p in model.parameters()))


def predict_constraint_logits(
    model: nn.Module,
    examples: Sequence[ConstraintExample],
    variant: ConstraintVariant,
    batch_size: int,
    device: str,
    condition: str = "none",
    seed: int = 0,
    allowed_views: set[str] | None = None,
    view_mask: str | None = None,
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
            batch = collate_constraint_batch(batch_examples, eval_variant, device, allowed_views=allowed_views, view_mask=view_mask)
            output = model(
                batch,
                hidden_state_shuffle=condition == "hidden_state_shuffle",
                shuffle_seed=seed + offset,
                return_attention=return_attention,
            )
            logits = output["logits"].detach().cpu().numpy()
            logits_rows.append(logits)
            if return_attention:
                attention_rows.extend(_summarize_attention(output, batch_examples, eval_variant, logits))
    logits_np = np.concatenate(logits_rows, axis=0) if logits_rows else np.zeros((0, 0), dtype=np.float32)
    return (logits_np, attention_rows) if return_attention else logits_np


def collate_constraint_batch(
    examples: Sequence[ConstraintExample],
    variant: ConstraintVariant,
    device: str,
    allowed_views: set[str] | None = None,
    view_mask: str | None = None,
) -> Dict[str, torch.Tensor]:
    max_evidence_tokens = 80
    max_candidate_tokens = 48
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
            for view_id, view in enumerate(variant.view_names):
                canonical = _canonical_view(view)
                if view_mask == "all_evidence" or view_mask == canonical or (allowed_views is not None and canonical not in allowed_views):
                    tokens: List[List[int]] = []
                else:
                    tokens = _evidence_tokens(example, cand_index, canonical, variant)
                fields, mask = _pack_tokens(tokens, max_evidence_tokens)
                per_view.append(fields)
                per_mask.append(mask)
            cand_tokens = _candidate_tokens(example, cand_index, actions)
            cand_fields, cand_mask = _pack_tokens(cand_tokens, max_candidate_tokens)
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


def _evidence_tokens(example: ConstraintExample, cand_index: int, view: str, variant: ConstraintVariant) -> List[List[int]]:
    tokens: List[List[int]] = []
    family_id = FAMILY_IDS[example.family]
    if view == "state":
        tokens.append(_token("state", example.start[0], example.start[1], 0, 0, family_id, 0, 0))
    elif view == "goal":
        tokens.append(_token("goal", example.goal[0], example.goal[1], 0, 0, family_id, 0, 0))
    elif view == "terrain":
        for row, col in example.obstacles:
            tokens.append(_token("obstacle", row, col, 0, 0, family_id, 0, 0))
        for row, col, color in example.terrain:
            tokens.append(_token("terrain", row, col, int(color), 0, family_id, 0, 0))
        if example.subgoal_a is not None:
            tokens.append(_token("subgoal_a", example.subgoal_a[0], example.subgoal_a[1], 0, 0, family_id, 0, 0))
        if example.subgoal_b is not None:
            tokens.append(_token("subgoal_b", example.subgoal_b[0], example.subgoal_b[1], 1, 0, family_id, 0, 0))
        if example.key is not None:
            tokens.append(_token("key", example.key[0], example.key[1], 0, 0, family_id, 0, 0))
        if example.door is not None:
            tokens.append(_token("door", example.door[0], example.door[1], 1, 0, family_id, 0, 0))
    elif view == "action":
        for step, action in enumerate(example.candidates[cand_index]):
            tokens.append(_token("action", 0, 0, int(action), step, family_id, 0, 0))
    elif view == "rollout":
        states = list(example.rollouts[cand_index])
        usable = states if variant.include_final_state else states[:-1]
        for step, pos in enumerate(usable):
            tokens.append(_token("rollout", pos[0], pos[1], _cell_value(example, pos), step, family_id, 0, int(pos == example.goal)))
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
            tokens.append(_token("transition", state_t[0], state_t[1], action, step, family_id, 0, _cell_value(example, state_t)))
            tokens.append(_token("transition", state_next[0], state_next[1], _cell_value(example, state_next), step, family_id, 0, int(state_next == example.goal)))
    elif view in {"constraint", "violation"}:
        for evidence_idx, evidence in enumerate(example.constraint_examples):
            if view == "constraint" and evidence.label == "invalid":
                continue
            if view == "violation" and evidence.label == "valid":
                continue
            role = "constraint_valid" if evidence.label == "valid" else "constraint_invalid"
            tokens.append(_token("rule", 0, 0, _rule_value(example), evidence_idx, family_id, 0, 0))
            for step, (raw_role, row, col) in enumerate(evidence.tokens):
                tokens.append(_token(role, row, col, _role_value(int(raw_role), example), step, family_id, 0, int(evidence.label == "valid")))
    if not tokens:
        tokens.append(_token("summary", 0, 0, 0, 0, family_id, 0, 0))
    return tokens


def _candidate_tokens(example: ConstraintExample, cand_index: int, actions: Sequence[int]) -> List[List[int]]:
    family_id = FAMILY_IDS[example.family]
    tokens = [_token("summary", example.start[0], example.start[1], len(actions), 0, family_id, 0, 0)]
    for step, action in enumerate(actions):
        prev = int(actions[step - 1]) if step > 0 else 4
        tokens.append(_token("action", prev, int(action), int(action), step, family_id, 0, 0))
    return tokens


def _token(role: str, row: int, col: int, value: int, step: int, family_id: int, view_id: int, aux: int) -> List[int]:
    raw = [ROLE_IDS.get(role, 0), row, col, value, step, family_id, view_id, aux]
    return [max(0, min(FIELD_VOCABS[index] - 1, int(value))) for index, value in enumerate(raw)]


def _pack_tokens(tokens: Sequence[Sequence[int]], limit: int) -> Tuple[np.ndarray, np.ndarray]:
    fields = np.zeros((int(limit), len(FIELD_VOCABS)), dtype=np.int64)
    mask = np.zeros((int(limit),), dtype=bool)
    if not tokens:
        tokens = [_token("summary", 0, 0, 0, 0, 0, 0, 0)]
    count = min(int(limit), len(tokens))
    fields[:count] = np.asarray(tokens[:count], dtype=np.int64)
    mask[:count] = True
    return fields, mask


def _run_micro_overfit_gates(
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
        splits = build_constraint_splits(gate_config, seed + 1000 + int(candidates))
        train_examples = splits["train"][: int(examples)]
        fit = fit_constraint_verifier(train_examples, train_examples, variant, replace(training, batch_size=min(int(training.batch_size), len(train_examples))), seed + 91_000 + int(candidates), device, True, f"micro__{variant.name}__{name}")
        logits = predict_constraint_logits(fit.model, train_examples, variant, training.batch_size, device)
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


def _heuristic_baseline_row(splits: Dict[str, List[ConstraintExample]], seed: int, config: Dict[str, object]) -> Dict[str, object]:
    train, dev, test = splits["train"], splits["dev"], splits["test"]
    labels = _labels(test)
    candidate_mlp = _fit_candidate_feature_mlp(train, dev, seed + 700, "candidate")
    zone_mlp = _fit_candidate_feature_mlp(train, dev, seed + 701, "zone_hist")
    cand_logits = _predict_feature_mlp(candidate_mlp, test, "candidate")
    zone_logits = _predict_feature_mlp(zone_mlp, test, "zone_hist")
    heuristics = {
        "random": 1.0 / max(1, len(test[0].candidates) if test else 1),
        "final_position_equals_goal": _accuracy(_final_position_predictions(test), labels),
        "final_distance_to_goal": _accuracy(_final_distance_predictions(test), labels),
        "length_only": _accuracy(_length_predictions(test), labels),
        "action_unigram": _accuracy(_unigram_predictions(test), labels),
        "action_bigram": _accuracy(_bigram_predictions(test), labels),
        "candidate_only_mlp": _top1(cand_logits, labels),
        "endpoint_only": _accuracy(_final_distance_predictions(test), labels),
        "terrain_count": _accuracy(_terrain_count_predictions(test), labels),
        "visited_zone_histogram_mlp": _top1(zone_logits, labels),
        "ordered_subgoal_oracle": _family_oracle_accuracy(test, "ordered_subgoal"),
        "key_door_rule_oracle": _family_oracle_accuracy(test, "key_door"),
    }
    by_family = {}
    for family in FAMILIES:
        family_examples = [example for example in test if example.family == family]
        family_labels = _labels(family_examples)
        by_family[family] = {
            "count": len(family_examples),
            "final_position_equals_goal": _accuracy(_final_position_predictions(family_examples), family_labels),
            "endpoint_only": _accuracy(_final_distance_predictions(family_examples), family_labels),
            "family_oracle": _family_oracle_accuracy(family_examples, family),
        }
    return {
        "benchmark": BENCHMARK,
        "seed": int(seed),
        "heuristics": heuristics,
        "by_family": by_family,
        "diagnostic_note": "Family-specific ordered/key heuristics are oracle upper-bound diagnostics, not controlled model inputs.",
    }


def _run_controls(
    model: nn.Module,
    variant: ConstraintVariant,
    examples: Sequence[ConstraintExample],
    clean_logits: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: str,
    seed: int,
) -> Dict[str, object]:
    chance = 1.0 / max(1, len(examples[0].candidates) if examples else 1)

    def predict(control_examples: Sequence[ConstraintExample], **kwargs: object) -> np.ndarray:
        return predict_constraint_logits(model, control_examples, variant, batch_size, device, seed=seed + 44_000, **kwargs)  # type: ignore[arg-type]

    controls: Dict[str, object] = {
        "benchmark": BENCHMARK,
        "variant": variant.name,
        "seed": int(seed),
        "chance": float(chance),
        "clean": _metric_block(clean_logits, labels, examples),
        "candidate_only": _metric_block(predict(examples, view_mask="all_evidence"), labels, examples),
        "endpoint_only": {"top1": _accuracy(_final_distance_predictions(examples), labels)},
        "length_only": _accuracy(_length_predictions(examples), labels),
        "action_unigram": _accuracy(_unigram_predictions(examples), labels),
        "action_bigram": _accuracy(_bigram_predictions(examples), labels),
        "state_goal_only": _metric_block(predict(examples, allowed_views={"state", "goal"}), labels, examples),
        "constraint_examples_only": _metric_block(predict(examples, allowed_views={"constraint", "violation"}), labels, examples),
    }
    for name in (
        "candidate_evidence_mismatch",
        "constraint_example_mismatch",
        "goal_shuffle",
        "terrain_zone_shuffle",
        "rollout_mismatch",
        "candidate_order_shuffle_with_gold_remap",
        "randomized_labels",
    ):
        controlled = apply_constraint_control(examples, name, seed + 50_000 + len(controls))
        logits = predict(controlled)
        controls[name] = _metric_block(logits, _labels(controlled), controlled)
    role_logits = predict_constraint_logits(model, examples, variant, batch_size, device, condition="role_order_shuffle", seed=seed + 60_000)
    hidden_logits = predict_constraint_logits(model, examples, variant, batch_size, device, condition="hidden_state_shuffle", seed=seed + 61_000)
    controls["role_order_shuffle"] = _metric_block(role_logits, labels, examples)
    controls["hidden_state_shuffle"] = _metric_block(hidden_logits, labels, examples)
    controls["candidate_order_delta"] = abs(float(_top1(clean_logits, labels)) - float(controls["candidate_order_shuffle_with_gold_remap"]["top1"]))
    controls["role_order_delta"] = abs(float(_top1(clean_logits, labels)) - float(controls["role_order_shuffle"]["top1"]))
    shortcut_values = [
        float(controls["candidate_only"]["top1"]),
        float(controls["endpoint_only"]["top1"]),
        float(controls["length_only"]),
        float(controls["action_unigram"]),
        float(controls["action_bigram"]),
        float(controls["state_goal_only"]["top1"]),
        float(controls["constraint_examples_only"]["top1"]),
    ]
    clean_top1 = float(_top1(clean_logits, labels))
    controls["control_pass"] = {
        "candidate_only_near_chance": float(controls["candidate_only"]["top1"]) <= chance + 0.10,
        "endpoint_only_near_chance": float(controls["endpoint_only"]["top1"]) <= chance + 0.10,
        "length_only_near_chance": float(controls["length_only"]) <= chance + 0.10,
        "action_stat_near_chance": max(float(controls["action_unigram"]), float(controls["action_bigram"])) <= chance + 0.10,
        "state_goal_only_near_chance": float(controls["state_goal_only"]["top1"]) <= chance + 0.10,
        "constraint_examples_only_near_chance": float(controls["constraint_examples_only"]["top1"]) <= chance + 0.10,
        "generic_shortcuts_near_chance": max(shortcut_values) <= chance + 0.10,
        "candidate_evidence_mismatch_collapses": float(controls["candidate_evidence_mismatch"]["top1"]) <= chance + 0.15,
        "constraint_example_mismatch_collapses": float(controls["constraint_example_mismatch"]["top1"]) <= chance + 0.15,
        "goal_shuffle_collapses": float(controls["goal_shuffle"]["top1"]) <= chance + 0.15,
        "terrain_zone_shuffle_collapses": float(controls["terrain_zone_shuffle"]["top1"]) <= chance + 0.15,
        "rollout_mismatch_collapses": float(controls["rollout_mismatch"]["top1"]) <= chance + 0.15,
        "candidate_order_remap_passes": float(controls["candidate_order_delta"]) <= 0.08,
        "role_order_remap_passes": float(controls["role_order_delta"]) <= 0.08,
        "hidden_state_shuffle_collapses": float(controls["hidden_state_shuffle"]["top1"]) <= max(chance + 0.20, clean_top1 - 0.05),
        "randomized_labels_collapses": float(controls["randomized_labels"]["top1"]) <= chance + 0.15,
    }
    controls["control_pass"]["overall"] = all(bool(value) for value in controls["control_pass"].values())
    return controls


def apply_constraint_control(examples: Sequence[ConstraintExample], control: str, seed: int) -> List[ConstraintExample]:
    rng = np.random.default_rng(seed)
    if control == "candidate_order_shuffle_with_gold_remap":
        return [_candidate_order_shuffle(example, rng) for example in examples]
    if control == "randomized_labels":
        return [replace(example, label=int(rng.integers(0, len(example.candidates))), metadata={**example.metadata, "control": control}) for example in examples]
    out = []
    for idx, example in enumerate(examples):
        other = examples[(idx + int(rng.integers(1, max(2, len(examples))))) % len(examples)] if len(examples) > 1 else example
        if control == "candidate_evidence_mismatch":
            out.append(replace(example, rollouts=other.rollouts if len(other.rollouts) == len(example.rollouts) else example.rollouts, terrain=other.terrain, rule=other.rule, constraint_examples=other.constraint_examples, metadata={**example.metadata, "control": control}))
        elif control == "constraint_example_mismatch":
            out.append(replace(example, rule=other.rule, constraint_examples=other.constraint_examples, metadata={**example.metadata, "control": control}))
        elif control == "goal_shuffle":
            wrong_goal = (0, example.grid_size - 1) if example.goal != (0, example.grid_size - 1) else (example.grid_size - 1, 0)
            out.append(replace(example, goal=wrong_goal, metadata={**example.metadata, "control": control}))
        elif control == "terrain_zone_shuffle":
            terrain = tuple((row, col, int((color + 1 + idx) % 4)) for row, col, color in example.terrain)
            out.append(
                replace(
                    example,
                    terrain=terrain,
                    subgoal_a=example.subgoal_b if example.subgoal_b is not None else example.subgoal_a,
                    subgoal_b=example.subgoal_a if example.subgoal_a is not None else example.subgoal_b,
                    key=example.door if example.door is not None else example.key,
                    door=example.key if example.key is not None else example.door,
                    metadata={**example.metadata, "control": control},
                )
            )
        elif control == "rollout_mismatch":
            out.append(replace(example, rollouts=other.rollouts if len(other.rollouts) == len(example.rollouts) else example.rollouts, metadata={**example.metadata, "control": control}))
        else:
            raise ValueError(f"unknown control: {control}")
    return out


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


def _attention_summaries(
    model: nn.Module,
    variant: ConstraintVariant,
    examples: Sequence[ConstraintExample],
    batch_size: int,
    device: str,
    seed: int,
) -> List[Dict[str, object]]:
    output = predict_constraint_logits(model, examples, variant, batch_size, device, seed=seed, return_attention=True)
    if not isinstance(output, tuple):
        return []
    _logits, rows = output
    for row in rows:
        row["benchmark"] = BENCHMARK
        row["variant"] = variant.name
        row["seed"] = int(seed)
    return rows


def _summarize_attention(output: Dict[str, object], examples: Sequence[ConstraintExample], variant: ConstraintVariant, logits: np.ndarray) -> List[Dict[str, object]]:
    weights = output.get("attention_weights")
    if weights is None:
        return []
    views = list(variant.view_names)
    w = weights.detach().float().cpu().numpy()
    # Shape: B*C, heads, query=1, V*T
    bsz = len(examples)
    cand_count = len(examples[0].candidates) if examples else 0
    token_count = 80
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
                "type": "plan_arch1_4_attention_summary",
                "example_id": example.id,
                "family": example.family,
                "gold_candidate": int(example.label),
                "predicted_candidate": int(preds[ex_idx]),
                "constraint_attention_mass_gold": candidate_rows[int(example.label)].get("constraint", 0.0) + candidate_rows[int(example.label)].get("violation", 0.0),
                "transition_attention_mass_gold": candidate_rows[int(example.label)].get("transition", 0.0) + candidate_rows[int(example.label)].get("rollout", 0.0),
                "candidate_view_attention": candidate_rows,
            }
        )
    return rows


def _result_row(
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


def _compute_row(variant: ConstraintVariant, seed: int, splits: Dict[str, List[ConstraintExample]], row: Dict[str, object], elapsed: float) -> Dict[str, object]:
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


def _leaderboard(rows: Sequence[Dict[str, object]], controls: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    control_by_key = {(row["variant"], row["seed"]): row for row in controls}
    leaders = []
    for row in rows:
        control = control_by_key.get((row["variant"], row["seed"]), {})
        train_top1 = float(row.get("trainable", {}).get("top1", 0.0))
        frozen = row.get("frozen")
        frozen_top1 = float(frozen.get("top1", 0.0)) if isinstance(frozen, dict) else 0.0
        delta = float(row.get("delta_trainable_minus_frozen", 0.0) or 0.0)
        controls_pass = bool(control.get("control_pass", {}).get("overall", False))
        shortcut_max = _generic_shortcut_max(control)
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
                "generic_shortcut_max": shortcut_max,
                "param_count": int(row.get("param_count", 0)),
                "score": float(score),
            }
        )
    return sorted(leaders, key=lambda item: float(item["score"]), reverse=True)


def _failure_taxonomy(
    rows: Sequence[Dict[str, object]],
    controls: Sequence[Dict[str, object]],
    heuristic_rows: Sequence[Dict[str, object]],
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
        if _generic_shortcut_max(row) > float(row.get("chance", 0.125)) + 0.10:
            counts["generic_shortcut_above_chance"] = counts.get("generic_shortcut_above_chance", 0) + 1
    for row in heuristic_rows:
        if float(row.get("heuristics", {}).get("final_position_equals_goal", 1.0)) > float(row.get("heuristics", {}).get("random", 0.125)) + 0.10:
            counts["endpoint_heuristic_solves"] = counts.get("endpoint_heuristic_solves", 0) + 1
    return {
        "counts": dict(sorted(counts.items())),
        "failure_reasons": {
            "micro_overfit_failure": "variant could not memorize small learned-constraint training sets",
            "control_failure": "one or more shortcut, mismatch, shuffle, or invariance controls failed",
            "generic_shortcut_above_chance": "candidate-only, endpoint, length, action-stat, state-goal, or constraint-only control exceeded chance window",
            "endpoint_heuristic_solves": "endpoint heuristic is still too predictive",
            "frozen_high_suspicious": "same-architecture frozen comparator solved too much of the task",
            "did_not_beat_frozen": "trainable shared model did not beat exact frozen comparator",
        },
    }


def _summary(
    rows: Sequence[Dict[str, object]],
    controls: Sequence[Dict[str, object]],
    heuristic_rows: Sequence[Dict[str, object]],
    leaderboard: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    best = leaderboard[0] if leaderboard else {}
    by_variant: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    variant_summaries = []
    control_by_key = {(row["variant"], row["seed"]): row for row in controls}
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
                "success_gate": bool(deltas and sum(delta > 0.0 for delta in deltas) >= 2 and _mean(deltas) >= 0.15 and controls_pass),
            }
        )
    endpoint_values = [float(row["heuristics"]["final_position_equals_goal"]) for row in heuristic_rows]
    random_values = [float(row["heuristics"]["random"]) for row in heuristic_rows]
    medium_ready = any(bool(row["success_gate"]) for row in variant_summaries)
    return {
        "best_variant": best.get("variant"),
        "endpoint_heuristic_mean": _mean(endpoint_values),
        "random_mean": _mean(random_values),
        "endpoint_heuristic_near_chance": _mean(endpoint_values) <= _mean(random_values) + 0.10 if endpoint_values else False,
        "variant_summaries": sorted(variant_summaries, key=lambda row: float(row["mean_trainable_top1"]), reverse=True),
        "medium_ready": bool(medium_ready),
        "medium_validation_launched": False,
        "architecture_search_should_resume": bool(medium_ready),
    }


def _metric_block(logits: np.ndarray, labels: np.ndarray, examples: Sequence[ConstraintExample]) -> Dict[str, object]:
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
        "by_plan_length": _accuracy_by_key(preds, labels, [str(len(example.candidates[0])) for example in examples]),
        "by_num_candidates_reaching_goal": _accuracy_by_key(preds, labels, [str(example.metadata["num_candidates_reaching_goal"]) for example in examples]),
        "all_candidates_reach_goal": _accuracy_on_subset(preds, labels, [bool(example.metadata["all_candidates_reach_goal"]) for example in examples]),
        "all_candidates_same_endpoint": _accuracy_on_subset(preds, labels, [bool(example.metadata["all_candidates_same_endpoint"]) for example in examples]),
        "failure_types": _failure_type_counts(preds, labels, examples),
    }


def _write_outputs(
    result: Dict[str, object],
    results_path: Path,
    controls_path: Path,
    heuristics_path: Path,
    leaderboard_path: Path,
    failure_path: Path,
    attention_path: Path,
    error_path: Path,
    compute_path: Path,
    report_path: Path,
) -> None:
    for path in (results_path, controls_path, heuristics_path, leaderboard_path, failure_path, attention_path, error_path, compute_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(_trim_result(result), indent=2, sort_keys=True), encoding="utf-8")
    controls_path.write_text(json.dumps({"controls": result["controls"]}, indent=2, sort_keys=True), encoding="utf-8")
    heuristics_path.write_text(json.dumps({"heuristic_baselines": result["heuristic_baselines"]}, indent=2, sort_keys=True), encoding="utf-8")
    failure_path.write_text(json.dumps(result["failure_taxonomy"], indent=2, sort_keys=True), encoding="utf-8")
    compute_path.write_text(json.dumps({"compute_metrics": result["compute_metrics"]}, indent=2, sort_keys=True), encoding="utf-8")
    with leaderboard_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["variant", "seed", "family", "status", "trainable_top1", "frozen_top1", "delta", "controls_pass", "generic_shortcut_max", "param_count", "score"]
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
    best_minimal = _variant_summary(summary, "minimal_transition_cross_attention")
    best_clone = max([row for row in summary["variant_summaries"] if row["variant"] != "minimal_transition_cross_attention"], key=lambda row: float(row["mean_trainable_top1"]), default={})
    lines = [
        "# PLAN-ARCH-1.4 Learned Constraint Planning Benchmark",
        "",
        "## Scope",
        "- No autonomous planning claim.",
        "- No world-model claim.",
        "- No architecture improvement claim unless gates pass.",
        "- Medium validation was not launched.",
        "- Candidate self-attention, pyramid, stacking, and multi-avenue search were not run.",
        "",
        "## Answers",
        f"1. Does learned-constraint planning avoid endpoint-heuristic solving? `{summary['endpoint_heuristic_near_chance']}` (endpoint={summary['endpoint_heuristic_mean']:.4f}, random={summary['random_mean']:.4f}).",
        f"2. Can trainable latent coordination beat frozen? `{_trainable_beats_answer(summary)}`.",
        f"3. Which constraint families are learnable? `{_family_answer(result['rows'])}`.",
        f"4. Does the model use constraint examples? `{_ablation_answer(result['controls'], 'constraint_example_mismatch_collapses')}`.",
        f"5. Does it use rollout/transition evidence? `{_ablation_answer(result['controls'], 'rollout_mismatch_collapses')}`.",
        f"6. Are shortcut baselines near chance? `{_shortcut_answer(result['controls'])}`.",
        f"7. Does minimal transition cross-attention outperform clone architecture? `{float(best_minimal.get('mean_trainable_top1', 0.0)) > float(best_clone.get('mean_trainable_top1', 0.0))}`.",
        f"8. Is the clone architecture adding value over simple baselines? `{_clone_value_answer(summary)}`.",
        f"9. Is there a medium-ready learned-constraint variant? `{summary['medium_ready']}`.",
        f"10. Should architecture search resume after this benchmark is established? `{summary['architecture_search_should_resume']}`.",
        "",
        "## Leaderboard",
        "| variant | seed | trainable | frozen | delta | shortcut | controls |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in result["leaderboard"][:20]:
        lines.append(
            f"| {row['variant']} | {row['seed']} | {float(row['trainable_top1']):.4f} | {float(row['frozen_top1']):.4f} | "
            f"{float(row['delta']):.4f} | {float(row['generic_shortcut_max']):.4f} | `{bool(row['controls_pass'])}` |"
        )
    lines.extend(
        [
            "",
            "## Variant Summary",
            "| variant | trainable mean | frozen mean | mean delta | seed wins | controls | gate |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in summary["variant_summaries"]:
        lines.append(
            f"| {row['variant']} | {float(row['mean_trainable_top1']):.4f} | {float(row['mean_frozen_top1']):.4f} | "
            f"{float(row['mean_delta']):.4f} | {int(row['seeds_trainable_beats_frozen'])} | `{bool(row['controls_pass_all'])}` | `{bool(row['success_gate'])}` |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "This stage establishes a better candidate-plan verification benchmark. It does not claim autonomous planning, world-modeling, or an architecture improvement.",
        ]
    )
    return "\n".join(lines) + "\n"


# Geometry, validation, and feature helpers.


def _actions_via_waypoints(start: Tuple[int, int], waypoints: Sequence[Tuple[int, int]], horizontal_first: bool = False) -> List[int]:
    actions: List[int] = []
    cur = start
    for target in waypoints:
        actions.extend(_actions_between(cur, target, horizontal_first=horizontal_first))
        cur = target
        horizontal_first = not horizontal_first
    return actions


def _actions_between(start: Tuple[int, int], target: Tuple[int, int], horizontal_first: bool = False) -> List[int]:
    actions: List[int] = []
    vertical = [1 if target[0] > start[0] else 0 for _ in range(abs(target[0] - start[0]))]
    horizontal = [3 if target[1] > start[1] else 2 for _ in range(abs(target[1] - start[1]))]
    if horizontal_first:
        actions.extend(horizontal)
        actions.extend(vertical)
    else:
        actions.extend(vertical)
        actions.extend(horizontal)
    return actions


def _with_detour(actions: Sequence[int], salt: int) -> List[int]:
    loop = [3, 2] if salt % 2 == 0 else [1, 0]
    insert_at = min(len(actions), 2 + (salt % max(1, len(actions) - 1)))
    return [*actions[:insert_at], *loop, *actions[insert_at:]]


def _pad_to_length(actions: List[int], start: Tuple[int, int], target_length: int, grid: int) -> List[int]:
    out = list(actions)
    loop = [3, 2] if start[1] + 1 < grid else [1, 0]
    if (target_length - len(out)) % 2 != 0:
        out = _with_detour(out, 0)
    while len(out) < int(target_length):
        out = [*loop, *out]
    return out[: int(target_length)]


def _balance_action_records(
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
    # All candidates are constructed to end at the same nominal goal. Preserve that net
    # displacement while equalizing candidate-only unigram statistics.
    target_d = max(max_d, max_u + 6)
    target_u = target_d - 6
    target_r = max(max_r, max_l + 6)
    target_l = target_r - 6
    target = [target_u, target_d, target_l, target_r]
    while sum(target) < int(plan_length):
        target[0] += 1
        target[1] += 1
        if sum(target) < int(plan_length):
            target[2] += 1
            target[3] += 1
    balanced = []
    for actions, ctype, valid in records:
        current = _action_counts(actions)
        add_vertical = max(0, min(target[0] - current[0], target[1] - current[1]))
        add_horizontal = max(0, min(target[2] - current[2], target[3] - current[3]))
        prefix = []
        prefix.extend([1, 0] * add_vertical)
        prefix.extend([3, 2] * add_horizontal)
        out = [*prefix, *actions]
        if len(out) < sum(target):
            out = _pad_to_length(out, start, sum(target), grid)
        balanced.append((tuple(out[: sum(target)]), ctype, valid))
    return balanced


def _simulate(start: Tuple[int, int], actions: Sequence[int], grid: int, obstacles: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    pos = start
    obstacle_set = set(obstacles)
    states = [pos]
    for action in actions:
        dr, dc = ACTION_DELTAS[int(action)]
        nxt = (pos[0] + dr, pos[1] + dc)
        if _in_bounds(nxt, grid) and nxt not in obstacle_set:
            pos = nxt
        states.append(pos)
    return states


def _satisfies_family(
    family: str,
    rollout: Sequence[Tuple[int, int]],
    rule: Dict[str, object],
    terrain: Sequence[Tuple[int, int, int]],
    subgoal_a: Tuple[int, int] | None,
    subgoal_b: Tuple[int, int] | None,
    key: Tuple[int, int] | None,
    door: Tuple[int, int] | None,
) -> bool:
    if family == "color_zone":
        required = int(rule["required_color"])
        terrain_by_cell = {(row, col): int(color) for row, col, color in terrain}
        first_color = next((terrain_by_cell[pos] for pos in rollout if pos in terrain_by_cell), None)
        return first_color == required
    if family == "ordered_subgoal":
        order = str(rule["order"])
        first, second = (subgoal_a, subgoal_b) if order == "A_BEFORE_B" else (subgoal_b, subgoal_a)
        return _first_index(rollout, first) < _first_index(rollout, second) < math.inf
    if family == "key_door":
        return _first_index(rollout, key) < _first_index(rollout, door) < math.inf
    return False


def _balanced_order(records: Sequence[Tuple[Tuple[int, ...], str, bool]], num_candidates: int, index: int, rng: np.random.Generator) -> Tuple[List[Tuple[Tuple[int, ...], str, bool]], int]:
    gold = next(record for record in records if bool(record[2]))
    negatives = [record for record in records if not bool(record[2])]
    rng.shuffle(negatives)
    label = int(index % int(num_candidates))
    ordered: List[Tuple[Tuple[int, ...], str, bool] | None] = [None for _ in range(int(num_candidates))]
    ordered[label] = gold
    iterator = iter(negatives)
    for slot in range(int(num_candidates)):
        if ordered[slot] is None:
            ordered[slot] = next(iterator)
    return [record for record in ordered if record is not None], label


def _subgoal_role(pos: Tuple[int, int], subgoal_a: Tuple[int, int]) -> int:
    return ROLE_IDS["subgoal_a"] if pos == subgoal_a else ROLE_IDS["subgoal_b"]


def _role_value(raw_role: int, example: ConstraintExample) -> int:
    if raw_role == ROLE_IDS["terrain"]:
        return int(example.rule.get("required_color", 0))
    if raw_role == ROLE_IDS["subgoal_a"]:
        return 1
    if raw_role == ROLE_IDS["subgoal_b"]:
        return 2
    if raw_role == ROLE_IDS["key"]:
        return 3
    if raw_role == ROLE_IDS["door"]:
        return 4
    return 0


def _rule_value(example: ConstraintExample) -> int:
    if example.family == "color_zone":
        return int(example.rule.get("required_color", 0))
    if example.family == "ordered_subgoal":
        return 8 if example.rule.get("order") == "A_BEFORE_B" else 9
    if example.family == "key_door":
        return 10
    return 0


def _cell_value(example: ConstraintExample, pos: Tuple[int, int]) -> int:
    for row, col, color in example.terrain:
        if (row, col) == pos:
            return int(color) + 1
    if example.subgoal_a == pos:
        return 8
    if example.subgoal_b == pos:
        return 9
    if example.key == pos:
        return 10
    if example.door == pos:
        return 11
    return 0


def _first_index(rollout: Sequence[Tuple[int, int]], cell: Tuple[int, int] | None) -> float:
    if cell is None:
        return math.inf
    for index, pos in enumerate(rollout):
        if pos == cell:
            return float(index)
    return math.inf


def _final_position_predictions(examples: Sequence[ConstraintExample]) -> np.ndarray:
    preds = []
    for example in examples:
        scores = [1.0 if rollout[-1] == example.goal else 0.0 for rollout in example.rollouts]
        preds.append(int(np.argmax(scores)))
    return np.asarray(preds, dtype=np.int64)


def _final_distance_predictions(examples: Sequence[ConstraintExample]) -> np.ndarray:
    return np.asarray([int(np.argmin([_manhattan(rollout[-1], example.goal) for rollout in example.rollouts])) for example in examples], dtype=np.int64)


def _length_predictions(examples: Sequence[ConstraintExample]) -> np.ndarray:
    return np.asarray([int(np.argmin([len(candidate) for candidate in example.candidates])) for example in examples], dtype=np.int64)


def _unigram_predictions(examples: Sequence[ConstraintExample]) -> np.ndarray:
    preds = []
    for example in examples:
        center = np.mean([np.asarray(_action_counts(candidate), dtype=np.float32) for candidate in example.candidates], axis=0)
        scores = [float(np.linalg.norm(np.asarray(_action_counts(candidate), dtype=np.float32) - center)) for candidate in example.candidates]
        preds.append(int(np.argmin(scores)))
    return np.asarray(preds, dtype=np.int64)


def _bigram_predictions(examples: Sequence[ConstraintExample]) -> np.ndarray:
    preds = []
    for example in examples:
        scores = []
        for candidate in example.candidates:
            reversals = sum(1 for a, b in zip(candidate, candidate[1:]) if REVERSE_ACTION[int(a)] == int(b))
            scores.append(reversals / max(1, len(candidate) - 1))
        preds.append(int(np.argmin(scores)))
    return np.asarray(preds, dtype=np.int64)


def _terrain_count_predictions(examples: Sequence[ConstraintExample]) -> np.ndarray:
    preds = []
    for example in examples:
        terrain_cells = {(row, col) for row, col, _color in example.terrain}
        scores = [sum(1 for pos in rollout if pos in terrain_cells) for rollout in example.rollouts]
        preds.append(int(np.argmax(scores)))
    return np.asarray(preds, dtype=np.int64)


def _family_oracle_accuracy(examples: Sequence[ConstraintExample], family: str) -> float:
    relevant = [example for example in examples if example.family == family]
    if not relevant:
        return 0.0
    preds = []
    for example in relevant:
        scores = [1.0 if valid else 0.0 for valid in example.satisfies_constraint]
        preds.append(int(np.argmax(scores)))
    return _accuracy(np.asarray(preds, dtype=np.int64), _labels(relevant))


def _fit_candidate_feature_mlp(train: Sequence[ConstraintExample], dev: Sequence[ConstraintExample], seed: int, mode: str) -> CandidateOnlyMLP:
    _set_seed(seed)
    train_x, train_y = _feature_batch(train, mode)
    dev_x, dev_y = _feature_batch(dev, mode)
    model = CandidateOnlyMLP(train_x.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0001)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_dev = -1.0
    bad = 0
    for epoch in range(8):
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
        if bad >= 3:
            break
    model.load_state_dict(best_state)
    return model


def _predict_feature_mlp(model: CandidateOnlyMLP, examples: Sequence[ConstraintExample], mode: str) -> np.ndarray:
    x, _y = _feature_batch(examples, mode)
    with torch.no_grad():
        return model(x).detach().numpy()


def _feature_batch(examples: Sequence[ConstraintExample], mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.as_tensor([[_features(example, idx, mode) for idx in range(len(example.candidates))] for example in examples], dtype=torch.float32), torch.as_tensor([example.label for example in examples], dtype=torch.long)


def _features(example: ConstraintExample, cand_index: int, mode: str) -> List[float]:
    candidate = example.candidates[cand_index]
    counts = np.asarray(_action_counts(candidate), dtype=np.float32) / max(1, len(candidate))
    turns = sum(1 for a, b in zip(candidate, candidate[1:]) if int(a) != int(b)) / max(1, len(candidate) - 1)
    if mode == "candidate":
        return [len(candidate) / 32.0, *counts.tolist(), turns]
    hist = [0.0] * 12
    for pos in example.rollouts[cand_index]:
        hist[_cell_value(example, pos)] += 1.0
    total = max(1.0, sum(hist))
    return [value / total for value in hist]


def _error_rows(variant: ConstraintVariant, seed: int, examples: Sequence[ConstraintExample], train_logits: np.ndarray, frozen_logits: np.ndarray, limit: int) -> List[Dict[str, object]]:
    labels = _labels(examples)
    train_pred = np.argmax(train_logits, axis=1)
    frozen_pred = np.argmax(frozen_logits, axis=1)
    rows = []
    for index, example in enumerate(examples):
        if len(rows) >= limit:
            break
        if train_pred[index] == labels[index]:
            continue
        pred_type = example.candidate_types[int(train_pred[index])]
        rows.append(
            {
                "type": "plan_arch1_4_error_case",
                "variant": variant.name,
                "seed": int(seed),
                "example_id": example.id,
                "family": example.family,
                "failure_type": pred_type,
                "gold_candidate_index": int(example.label),
                "predicted_candidate_index": int(train_pred[index]),
                "frozen_candidate_index": int(frozen_pred[index]),
                "gold_vs_best_negative_margin": float(train_logits[index, int(example.label)] - np.max(np.delete(train_logits[index], int(example.label)))),
                "candidate_probabilities": _softmax_np(train_logits[index]).tolist(),
                "candidate_types": list(example.candidate_types),
            }
        )
    return rows


def _failure_type_counts(preds: np.ndarray, labels: np.ndarray, examples: Sequence[ConstraintExample]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for pred, label, example in zip(preds, labels, examples):
        if int(pred) == int(label):
            continue
        key = str(example.candidate_types[int(pred)])
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _trim_result(result: Dict[str, object]) -> Dict[str, object]:
    trimmed = dict(result)
    trimmed["attention_summaries"] = result.get("attention_summaries", [])[:40]
    trimmed["error_cases"] = result.get("error_cases", [])[:80]
    return trimmed


def _generic_shortcut_max(control: Dict[str, object]) -> float:
    values = [
        float(control.get("candidate_only", {}).get("top1", 0.0)),
        float(control.get("endpoint_only", {}).get("top1", 0.0)),
        float(control.get("length_only", 0.0)),
        float(control.get("action_unigram", 0.0)),
        float(control.get("action_bigram", 0.0)),
        float(control.get("state_goal_only", {}).get("top1", 0.0)),
        float(control.get("constraint_examples_only", {}).get("top1", 0.0)),
    ]
    return max(values)


def _variant_summary(summary: Dict[str, object], name: str) -> Dict[str, object]:
    return next((row for row in summary["variant_summaries"] if row["variant"] == name), {})


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


def _ablation_answer(controls: Sequence[Dict[str, object]], key: str) -> str:
    values = [bool(row.get("control_pass", {}).get(key, False)) for row in controls]
    return f"{sum(values)}/{len(values)} pass" if values else "not run"


def _shortcut_answer(controls: Sequence[Dict[str, object]]) -> str:
    values = [bool(row.get("control_pass", {}).get("generic_shortcuts_near_chance", False)) for row in controls]
    return f"{sum(values)}/{len(values)} pass" if values else "not run"


def _clone_value_answer(summary: Dict[str, object]) -> str:
    minimal = float(_variant_summary(summary, "minimal_transition_cross_attention").get("mean_trainable_top1", 0.0))
    clones = [row for row in summary["variant_summaries"] if row["variant"] != "minimal_transition_cross_attention"]
    best_clone = max([float(row["mean_trainable_top1"]) for row in clones], default=0.0)
    return f"best_clone={best_clone:.4f}, minimal={minimal:.4f}, adds_value={best_clone > minimal}"


# Generic utilities.


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
        "seeds": [0, 1, 2],
        "dataset": {
            "grid_size": 7,
            "num_candidates": 8,
            "plan_length": 32,
            "train_examples": 256,
            "dev_examples": 64,
            "test_examples": 64,
            "families": list(FAMILIES),
            "all_goal_every": 4,
        },
        "training": {"epochs": 8, "batch_size": 64, "lr": 0.003, "weight_decay": 0.0, "patience": 4, "gradient_clip_norm": 1.0},
        "micro_overfit_training": {"epochs": 40, "batch_size": 32, "lr": 0.003, "weight_decay": 0.0, "patience": 40, "gradient_clip_norm": 1.0},
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


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _clear_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _labels(examples: Sequence[ConstraintExample]) -> np.ndarray:
    return np.asarray([example.label for example in examples], dtype=np.int64)


def _accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(predictions.astype(np.int64) == labels.astype(np.int64))) if len(labels) else 0.0


def _top1(logits: np.ndarray, labels: np.ndarray) -> float:
    return _accuracy(np.argmax(logits, axis=1), labels) if logits.size else 0.0


def _accuracy_by_key(predictions: np.ndarray, labels: np.ndarray, keys: Sequence[str]) -> Dict[str, float]:
    out = {}
    for key in sorted(set(keys)):
        idx = [i for i, value in enumerate(keys) if value == key]
        out[str(key)] = _accuracy(np.asarray([predictions[i] for i in idx]), np.asarray([labels[i] for i in idx]))
    return out


def _accuracy_on_subset(predictions: np.ndarray, labels: np.ndarray, mask: Sequence[bool]) -> Dict[str, object]:
    idx = [i for i, value in enumerate(mask) if bool(value)]
    return {"count": len(idx), "top1": _accuracy(np.asarray([predictions[i] for i in idx]), np.asarray([labels[i] for i in idx])) if idx else 0.0}


def _mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0


def _softmax_np(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    values = values - np.max(values)
    exp = np.exp(values)
    return exp / np.clip(exp.sum(), 1e-12, None)


def _chunks(values: Sequence[int], size: int) -> Iterable[List[int]]:
    for index in range(0, len(values), max(1, int(size))):
        yield list(values[index : index + max(1, int(size))])


def _batched(values: Sequence[ConstraintExample], size: int) -> Iterable[List[ConstraintExample]]:
    for index in range(0, len(values), max(1, int(size))):
        yield list(values[index : index + max(1, int(size))])


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


def _flat_params(items: Sequence[Tuple[str, nn.Parameter]]) -> torch.Tensor:
    tensors = [parameter.detach().float().cpu().reshape(-1) for _name, parameter in items]
    return torch.cat(tensors) if tensors else torch.zeros(0)


def _l2_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0 or a.numel() != b.numel():
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
    return dim if dim % heads == 0 else int(math.ceil(dim / heads) * heads)


def _canonical_view(view: str) -> str:
    return str(view).split(":", 1)[0]


def _in_bounds(cell: Tuple[int, int], grid: int) -> bool:
    return 0 <= int(cell[0]) < int(grid) and 0 <= int(cell[1]) < int(grid)


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _action_counts(actions: Sequence[int]) -> List[int]:
    return [int(sum(1 for action in actions if int(action) == i)) for i in range(4)]


if __name__ == "__main__":
    main()
