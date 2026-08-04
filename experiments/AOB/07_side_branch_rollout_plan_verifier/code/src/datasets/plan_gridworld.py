from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


Position = Tuple[int, int]
ActionSeq = Tuple[int, ...]

ACTION_NAMES = ("U", "D", "L", "R")
ACTION_TO_DELTA: Dict[int, Position] = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}


@dataclass(frozen=True)
class PlanGridworldDatasetConfig:
    num_candidates: int = 8
    train_examples: int = 192
    dev_examples: int = 64
    test_examples: int = 64
    grid_size: int = 7
    obstacle_density: float = 0.16
    plan_horizon: int = 16
    min_goal_distance: int = 4
    level: int = 4
    max_generation_attempts: int = 400
    length_balanced: bool = True


@dataclass(frozen=True)
class CandidateRollout:
    actions: ActionSeq
    rollout: Tuple[Position, ...]
    final_position: Position
    first_goal_step: int
    reaches_goal: bool
    collision: bool
    wrong_goal: bool
    near_miss: bool
    valid: bool
    family: str

    @property
    def effective_cost(self) -> int:
        return int(self.first_goal_step) if self.reaches_goal else 10_000


@dataclass(frozen=True)
class PlanGridworldExample:
    id: str
    split: str
    grid_size: int
    start: Position
    goal: Position
    obstacles: Tuple[Position, ...]
    candidates: Tuple[CandidateRollout, ...]
    label: int
    public_source_tags: Tuple[str, ...]
    metadata: Dict[str, object]


def build_plan_gridworld_splits(
    config: PlanGridworldDatasetConfig,
    seed: int,
) -> Dict[str, List[PlanGridworldExample]]:
    rng = np.random.default_rng(seed)
    splits: Dict[str, List[PlanGridworldExample]] = {"train": [], "dev": [], "test": []}
    counts = {
        "train": int(config.train_examples),
        "dev": int(config.dev_examples),
        "test": int(config.test_examples),
    }
    for split, count in counts.items():
        for index in range(count):
            example_seed = int(rng.integers(0, 2**31 - 1))
            example_rng = np.random.default_rng(example_seed)
            splits[split].append(_generate_example(config, split, index, example_seed, example_rng))
    return splits


def simulate_candidate(
    grid_size: int,
    start: Position,
    goal: Position,
    obstacles: Sequence[Position],
    actions: Sequence[int],
    family: str = "unknown",
) -> CandidateRollout:
    obstacle_set = set(tuple(pos) for pos in obstacles)
    pos = tuple(start)
    rollout: List[Position] = [pos]
    reaches_goal = pos == tuple(goal)
    first_goal_step = 0 if reaches_goal else 10_000
    collision = False
    stopped = reaches_goal
    for step, action in enumerate(actions, start=1):
        if stopped:
            rollout.append(tuple(goal))
            continue
        dr, dc = ACTION_TO_DELTA[int(action)]
        nxt = (pos[0] + dr, pos[1] + dc)
        if not _in_bounds(nxt, grid_size) or nxt in obstacle_set:
            collision = True
            rollout.append(pos)
            stopped = True
            continue
        pos = nxt
        rollout.append(pos)
        if pos == tuple(goal):
            reaches_goal = True
            first_goal_step = step
            stopped = True
    while len(rollout) < len(tuple(actions)) + 1:
        rollout.append(rollout[-1])
    final_position = tuple(rollout[-1])
    near_miss = (not reaches_goal) and _manhattan(final_position, tuple(goal)) == 1 and not collision
    wrong_goal = (not reaches_goal) and _manhattan(final_position, tuple(goal)) >= 2 and not collision
    valid = bool(reaches_goal and not collision)
    return CandidateRollout(
        actions=tuple(int(a) for a in actions),
        rollout=tuple(rollout),
        final_position=final_position,
        first_goal_step=int(first_goal_step),
        reaches_goal=bool(reaches_goal),
        collision=bool(collision),
        wrong_goal=bool(wrong_goal),
        near_miss=bool(near_miss),
        valid=valid,
        family=str(family),
    )


def apply_plan_control(
    examples: Sequence[PlanGridworldExample],
    control: str,
    seed: int,
) -> List[PlanGridworldExample]:
    rng = np.random.default_rng(seed)
    if control == "candidate_plan_only":
        return [_candidate_plan_only(example) for example in examples]
    if control == "state_goal_only":
        return [_state_goal_only(example) for example in examples]
    if control == "state_plan_mismatch":
        return _state_plan_mismatch(examples, rng)
    if control == "goal_shuffle":
        return _goal_shuffle(examples, rng)
    if control == "obstacle_constraint_shuffle":
        return _obstacle_shuffle(examples, rng)
    if control == "rollout_mismatch":
        return _rollout_mismatch(examples, rng)
    if control == "final_state_mismatch":
        return _final_state_mismatch(examples, rng)
    if control == "final_state_outcome_only_mismatch":
        return _final_state_outcome_only_mismatch(examples, rng)
    if control == "candidate_order_shuffle_with_gold_remap":
        return [_candidate_order_shuffle(example, rng) for example in examples]
    if control == "action_symbol_permutation":
        return [_horizontal_reflection(example) for example in examples]
    if control == "map_rotation_reflection":
        return [_vertical_reflection(example) for example in examples]
    if control == "randomized_labels":
        return [_randomized_label(example, rng) for example in examples]
    raise ValueError(f"unknown PLAN-1 control: {control}")


def leakage_audit_rows(examples: Sequence[PlanGridworldExample]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for example in examples:
        lengths = [len(candidate.actions) for candidate in example.candidates]
        gold_count = sum(
            _candidate_score(candidate) == _candidate_score(example.candidates[int(example.label)])
            for candidate in example.candidates
        )
        source_visible = any(tag != "candidate_plan" for tag in example.public_source_tags)
        rows.append(
            {
                "type": "plan1_leakage_audit",
                "id": example.id,
                "split": example.split,
                "candidate_count": len(example.candidates),
                "candidate_order_label": int(example.label),
                "exactly_one_best_candidate": bool(gold_count == 1),
                "all_candidate_lengths_equal": bool(len(set(lengths)) == 1),
                "candidate_source_ids_visible_to_model": bool(source_visible),
                "simulator_success_visible_as_metadata": False,
                "gold_flag_visible": False,
                "reward_or_cost_visible_as_raw_metadata": False,
                "pass": bool(gold_count == 1 and len(set(lengths)) == 1 and not source_visible),
            }
        )
    return rows


def candidate_source_metadata_only_accuracy(examples: Sequence[PlanGridworldExample]) -> float:
    if not examples:
        return 0.0
    predictions = [0 for _example in examples]
    return float(np.mean([int(pred == int(example.label)) for pred, example in zip(predictions, examples)]))


def dataset_summary(splits: Dict[str, Sequence[PlanGridworldExample]]) -> Dict[str, object]:
    all_examples = [example for rows in splits.values() for example in rows]
    return {
        "num_examples": {split: len(rows) for split, rows in splits.items()},
        "candidate_count": sorted({len(example.candidates) for example in all_examples}),
        "grid_sizes": _count_strings([str(example.grid_size) for example in all_examples]),
        "obstacle_counts": _count_strings([str(len(example.obstacles)) for example in all_examples]),
        "goal_distances": _count_strings([str(int(example.metadata.get("goal_distance", 0))) for example in all_examples]),
        "all_length_balanced": all(len({len(candidate.actions) for candidate in example.candidates}) == 1 for example in all_examples),
        "exactly_one_best_candidate": all(_exactly_one_best(example) for example in all_examples),
        "candidate_families_hidden": all(all(tag == "candidate_plan" for tag in example.public_source_tags) for example in all_examples),
        "candidate_family_counts": _count_strings([candidate.family for example in all_examples for candidate in example.candidates]),
    }


def selected_plan_metrics(examples: Sequence[PlanGridworldExample], predictions: Sequence[int]) -> Dict[str, object]:
    selected: List[CandidateRollout] = []
    for example, pred in zip(examples, predictions):
        index = max(0, min(len(example.candidates) - 1, int(pred)))
        selected.append(example.candidates[index])
    if not selected:
        return {}
    valid_selected = [candidate for candidate in selected if candidate.valid]
    return {
        "selected_plan_success_rate": float(np.mean([candidate.reaches_goal and not candidate.collision for candidate in selected])),
        "selected_valid_plan_average_cost": float(np.mean([candidate.effective_cost for candidate in valid_selected])) if valid_selected else 0.0,
        "selected_collision_rate": float(np.mean([candidate.collision for candidate in selected])),
        "selected_constraint_violation_rate": float(np.mean([candidate.collision for candidate in selected])),
        "selected_wrong_goal_rate": float(np.mean([candidate.wrong_goal for candidate in selected])),
        "selected_near_miss_rate": float(np.mean([candidate.near_miss for candidate in selected])),
        "selected_valid_but_suboptimal_rate": float(np.mean([candidate.valid and candidate.family != "valid_shortest" for candidate in selected])),
        "selected_family_distribution": _count_strings([candidate.family for candidate in selected]),
    }


def _generate_example(
    config: PlanGridworldDatasetConfig,
    split: str,
    index: int,
    example_seed: int,
    rng: np.random.Generator,
) -> PlanGridworldExample:
    for attempt in range(int(config.max_generation_attempts)):
        grid_size = int(config.grid_size)
        start, goal, obstacles = _sample_world(config, rng)
        shortest = _shortest_path(grid_size, start, goal, obstacles)
        if shortest is None:
            continue
        if len(shortest) < int(config.min_goal_distance) or len(shortest) >= int(config.plan_horizon) - 2:
            continue
        candidates = _generate_candidates(config, start, goal, obstacles, shortest, rng)
        if len(candidates) != int(config.num_candidates):
            continue
        scores = [_candidate_score(candidate) for candidate in candidates]
        best = max(scores)
        if scores.count(best) != 1:
            continue
        label = int(scores.index(best))
        if candidates[label].family != "valid_shortest":
            continue
        return PlanGridworldExample(
            id=f"{split}_{index:05d}",
            split=split,
            grid_size=grid_size,
            start=start,
            goal=goal,
            obstacles=tuple(sorted(obstacles)),
            candidates=tuple(candidates),
            label=label,
            public_source_tags=tuple("candidate_plan" for _ in candidates),
            metadata={
                "generator_version": "plan1_gridworld_length_balanced_v1",
                "candidate_count": int(config.num_candidates),
                "plan_horizon": int(config.plan_horizon),
                "level": int(config.level),
                "obstacle_density": float(config.obstacle_density),
                "goal_distance": int(len(shortest)),
                "model_visible_candidate_sources": False,
                "example_seed": int(example_seed),
                "generation_attempt": int(attempt),
            },
        )
    raise RuntimeError(f"failed to generate PLAN-1 gridworld example after {config.max_generation_attempts} attempts")


def _sample_world(config: PlanGridworldDatasetConfig, rng: np.random.Generator) -> Tuple[Position, Position, Tuple[Position, ...]]:
    grid_size = int(config.grid_size)
    cells = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    start = tuple(cells[int(rng.integers(0, len(cells)))])
    goal = tuple(cells[int(rng.integers(0, len(cells)))])
    while goal == start or _manhattan(start, goal) < int(config.min_goal_distance):
        goal = tuple(cells[int(rng.integers(0, len(cells)))])
    obstacle_budget = int(round(float(config.obstacle_density) * grid_size * grid_size))
    available = [cell for cell in cells if cell not in {start, goal}]
    rng.shuffle(available)
    obstacles = tuple(sorted(available[:obstacle_budget]))
    if _shortest_path(grid_size, start, goal, obstacles) is None:
        obstacles = tuple(sorted(obstacles[: max(0, len(obstacles) // 2)]))
    return start, goal, obstacles


def _generate_candidates(
    config: PlanGridworldDatasetConfig,
    start: Position,
    goal: Position,
    obstacles: Tuple[Position, ...],
    shortest: Sequence[int],
    rng: np.random.Generator,
) -> List[CandidateRollout]:
    grid_size = int(config.grid_size)
    horizon = int(config.plan_horizon)
    target_count = int(config.num_candidates)
    families = _families_for_level(int(config.level), target_count)
    candidates: List[CandidateRollout] = []
    seen: set[ActionSeq] = set()

    gold_actions = _pad_actions(shortest, horizon, rng)
    gold = simulate_candidate(grid_size, start, goal, obstacles, gold_actions, "valid_shortest")
    candidates.append(gold)
    seen.add(gold.actions)

    attempts = 0
    while len(candidates) < target_count and attempts < target_count * 160:
        family = families[(len(candidates) - 1 + attempts) % max(1, len(families))]
        attempts += 1
        actions = _candidate_actions_for_family_balanced(
            family,
            grid_size,
            start,
            goal,
            obstacles,
            shortest,
            gold.actions,
            horizon,
            rng,
        )
        if actions is None:
            actions = _candidate_actions_for_family(family, grid_size, start, goal, obstacles, shortest, horizon, rng)
        if actions is None:
            continue
        rollout = simulate_candidate(grid_size, start, goal, obstacles, actions, family)
        if rollout.actions in seen:
            continue
        if _candidate_score(rollout) >= _candidate_score(gold):
            continue
        seen.add(rollout.actions)
        candidates.append(rollout)

    while len(candidates) < target_count:
        actions = tuple(int(a) for a in rng.integers(0, 4, size=horizon).tolist())
        rollout = simulate_candidate(grid_size, start, goal, obstacles, actions, "random_walk")
        if rollout.actions in seen or _candidate_score(rollout) >= _candidate_score(gold):
            continue
        seen.add(rollout.actions)
        candidates.append(rollout)

    order = rng.permutation(len(candidates))
    return [candidates[int(i)] for i in order]


def _candidate_actions_for_family_balanced(
    family: str,
    grid_size: int,
    start: Position,
    goal: Position,
    obstacles: Tuple[Position, ...],
    shortest: Sequence[int],
    gold_actions: Sequence[int],
    horizon: int,
    rng: np.random.Generator,
) -> ActionSeq | None:
    if not shortest:
        return None
    if family == "invalid_collision":
        return _balanced_collision_actions(grid_size, start, goal, obstacles, shortest, gold_actions, rng)
    if family == "valid_but_long":
        return _balanced_valid_long_actions(grid_size, start, goal, obstacles, shortest, gold_actions, horizon, rng)
    if family == "near_miss":
        return _matched_near_goal_path_actions(grid_size, start, goal, obstacles, shortest, gold_actions, rng)
    if family == "wrong_goal":
        return _matched_decoy_path_actions(grid_size, start, goal, obstacles, shortest, gold_actions, rng)
    if family == "deceptive_prefix":
        return _matched_decoy_path_actions(grid_size, start, goal, obstacles, shortest, gold_actions, rng, share_prefix=True)
    if family == "invalid_dead_end":
        return _matched_decoy_path_actions(grid_size, start, goal, obstacles, shortest, gold_actions, rng)
    return None


def _matched_decoy_path_actions(
    grid_size: int,
    start: Position,
    goal: Position,
    obstacles: Tuple[Position, ...],
    shortest: Sequence[int],
    gold_actions: Sequence[int],
    rng: np.random.Generator,
    share_prefix: bool = False,
) -> ActionSeq | None:
    obstacle_set = set(obstacles)
    cells = [(r, c) for r in range(grid_size) for c in range(grid_size) if (r, c) not in obstacle_set and (r, c) != goal and (r, c) != start]
    rng.shuffle(cells)
    target_len = len(shortest)
    for target in cells:
        if _manhattan(target, goal) < 2:
            continue
        path = _shortest_path(grid_size, start, target, obstacles)
        if path is None or abs(len(path) - target_len) > 1:
            continue
        if share_prefix and target_len > 1 and tuple(path[: max(1, target_len // 3)]) != tuple(shortest[: max(1, target_len // 3)]):
            continue
        candidate = list(path) + list(gold_actions[len(path) :])
        rollout = simulate_candidate(grid_size, start, goal, obstacles, candidate[: len(gold_actions)], "wrong_goal")
        if not rollout.reaches_goal:
            return tuple(candidate[: len(gold_actions)])
    return None


def _matched_near_goal_path_actions(
    grid_size: int,
    start: Position,
    goal: Position,
    obstacles: Tuple[Position, ...],
    shortest: Sequence[int],
    gold_actions: Sequence[int],
    rng: np.random.Generator,
) -> ActionSeq | None:
    obstacle_set = set(obstacles)
    targets = [pos for pos, _action in _legal_neighbors(grid_size, goal, obstacles) if pos != start and pos not in obstacle_set]
    rng.shuffle(targets)
    target_len = len(shortest)
    for target in targets:
        path = _shortest_path(grid_size, start, target, obstacles)
        if path is None or abs(len(path) - target_len) > 1:
            continue
        candidate = list(path) + list(gold_actions[len(path) :])
        rollout = simulate_candidate(grid_size, start, goal, obstacles, candidate[: len(gold_actions)], "near_miss")
        if not rollout.reaches_goal:
            return tuple(candidate[: len(gold_actions)])
    return _matched_decoy_path_actions(grid_size, start, goal, obstacles, shortest, gold_actions, rng)


def _balanced_collision_actions(
    grid_size: int,
    start: Position,
    goal: Position,
    obstacles: Tuple[Position, ...],
    shortest: Sequence[int],
    gold_actions: Sequence[int],
    rng: np.random.Generator,
) -> ActionSeq | None:
    del goal
    positions = _positions_from_actions(start, shortest)
    obstacle_set = set(obstacles)
    order = list(range(len(shortest)))
    rng.shuffle(order)
    for step_index in order:
        pos = positions[step_index]
        bad_actions = []
        for action, (dr, dc) in ACTION_TO_DELTA.items():
            nxt = (pos[0] + dr, pos[1] + dc)
            if action == int(shortest[step_index]):
                continue
            if not _in_bounds(nxt, grid_size) or nxt in obstacle_set:
                bad_actions.append(action)
        if bad_actions:
            actions = list(gold_actions)
            actions[step_index] = int(bad_actions[int(rng.integers(0, len(bad_actions)))])
            return tuple(actions)
    return None


def _balanced_valid_long_actions(
    grid_size: int,
    start: Position,
    goal: Position,
    obstacles: Tuple[Position, ...],
    shortest: Sequence[int],
    gold_actions: Sequence[int],
    horizon: int,
    rng: np.random.Generator,
) -> ActionSeq | None:
    path_positions = _positions_from_actions(start, shortest)
    order = list(range(max(1, len(path_positions) - 1)))
    rng.shuffle(order)
    for insert_at in order:
        if len(shortest) + 2 >= horizon:
            return None
        pos = path_positions[insert_at]
        legal = _legal_neighbors(grid_size, pos, obstacles)
        next_pos = path_positions[min(insert_at + 1, len(path_positions) - 1)]
        legal = [item for item in legal if item[0] != next_pos and item[0] != goal]
        if not legal:
            continue
        detour_pos, out_action = legal[int(rng.integers(0, len(legal)))]
        back_action = _action_between(detour_pos, pos)
        prefix = list(shortest[:insert_at]) + [out_action, back_action] + list(shortest[insert_at:])
        actions = prefix + list(gold_actions[len(prefix) :])
        if len(actions) < horizon:
            actions.extend(list(gold_actions[len(actions) :]))
        rollout = simulate_candidate(grid_size, start, goal, obstacles, actions[:horizon], "valid_but_long")
        if rollout.valid and rollout.first_goal_step > len(shortest):
            return tuple(actions[:horizon])
    return None


def _balanced_last_step_miss_actions(
    grid_size: int,
    start: Position,
    goal: Position,
    obstacles: Tuple[Position, ...],
    shortest: Sequence[int],
    gold_actions: Sequence[int],
    rng: np.random.Generator,
) -> ActionSeq | None:
    if len(shortest) < 1:
        return None
    return _balanced_wrong_branch_actions(grid_size, start, goal, obstacles, shortest, gold_actions, rng, force_last=True)


def _balanced_wrong_branch_actions(
    grid_size: int,
    start: Position,
    goal: Position,
    obstacles: Tuple[Position, ...],
    shortest: Sequence[int],
    gold_actions: Sequence[int],
    rng: np.random.Generator,
    prefer_late: bool = False,
    force_last: bool = False,
) -> ActionSeq | None:
    positions = _positions_from_actions(start, shortest)
    if force_last:
        order = [len(shortest) - 1]
    else:
        lo = max(1, len(shortest) // 2) if prefer_late else 0
        order = list(range(lo, len(shortest)))
        rng.shuffle(order)
    for step_index in order:
        pos = positions[step_index]
        next_gold = positions[min(step_index + 1, len(positions) - 1)]
        choices = _legal_neighbors(grid_size, pos, obstacles)
        choices = [item for item in choices if item[0] != next_gold and item[0] != goal]
        if not choices:
            continue
        _nxt, action = choices[int(rng.integers(0, len(choices)))]
        actions = list(gold_actions)
        actions[step_index] = int(action)
        return tuple(actions)
    return None


def _balanced_two_mutation_actions(
    grid_size: int,
    start: Position,
    goal: Position,
    obstacles: Tuple[Position, ...],
    shortest: Sequence[int],
    gold_actions: Sequence[int],
    rng: np.random.Generator,
) -> ActionSeq | None:
    first = _balanced_wrong_branch_actions(grid_size, start, goal, obstacles, shortest, gold_actions, rng)
    if first is None:
        return None
    actions = list(first)
    positions = _positions_from_actions(start, shortest)
    indices = list(range(len(shortest)))
    rng.shuffle(indices)
    for step_index in indices:
        pos = positions[step_index]
        choices = _legal_neighbors(grid_size, pos, obstacles)
        choices = [item for item in choices if int(item[1]) != int(shortest[step_index])]
        if not choices:
            continue
        _nxt, action = choices[int(rng.integers(0, len(choices)))]
        actions[step_index] = int(action)
        return tuple(actions)
    return tuple(actions)


def _families_for_level(level: int, target_count: int) -> Tuple[str, ...]:
    if level <= 1:
        return ("invalid_collision",)[: max(1, target_count - 1)]
    if level == 2:
        return ("invalid_collision", "wrong_goal", "near_miss")
    if level == 3:
        return ("wrong_goal", "near_miss", "deceptive_prefix", "valid_but_long", "invalid_collision")
    if level == 4:
        return ("wrong_goal", "near_miss", "deceptive_prefix", "invalid_dead_end", "valid_but_long", "invalid_collision")
    return (
        "wrong_goal",
        "near_miss",
        "deceptive_prefix",
        "invalid_dead_end",
        "valid_but_long",
        "invalid_collision",
        "random_walk",
    )


def _candidate_actions_for_family(
    family: str,
    grid_size: int,
    start: Position,
    goal: Position,
    obstacles: Tuple[Position, ...],
    shortest: Sequence[int],
    horizon: int,
    rng: np.random.Generator,
) -> ActionSeq | None:
    if family == "valid_but_long":
        return _valid_long_actions(grid_size, start, goal, obstacles, shortest, horizon, rng)
    if family == "invalid_collision":
        return _collision_actions(grid_size, start, goal, obstacles, shortest, horizon, rng)
    if family == "wrong_goal":
        return _wrong_goal_actions(grid_size, start, goal, obstacles, horizon, rng)
    if family == "near_miss":
        return _near_miss_actions(grid_size, start, goal, obstacles, horizon, rng)
    if family == "deceptive_prefix":
        return _deceptive_prefix_actions(grid_size, start, goal, obstacles, shortest, horizon, rng)
    if family == "invalid_dead_end":
        return _dead_end_actions(grid_size, start, goal, obstacles, horizon, rng)
    if family == "random_walk":
        return tuple(int(a) for a in rng.integers(0, 4, size=horizon).tolist())
    return None


def _valid_long_actions(
    grid_size: int,
    start: Position,
    goal: Position,
    obstacles: Tuple[Position, ...],
    shortest: Sequence[int],
    horizon: int,
    rng: np.random.Generator,
) -> ActionSeq | None:
    path_positions = _positions_from_actions(start, shortest)
    for _ in range(40):
        if len(shortest) + 2 >= horizon:
            return None
        insert_at = int(rng.integers(0, max(1, len(path_positions) - 1)))
        pos = path_positions[insert_at]
        legal = _legal_neighbors(grid_size, pos, obstacles)
        next_pos = path_positions[min(insert_at + 1, len(path_positions) - 1)]
        legal = [item for item in legal if item[0] != next_pos and item[0] != goal]
        if not legal:
            continue
        detour_pos, out_action = legal[int(rng.integers(0, len(legal)))]
        back_action = _action_between(detour_pos, pos)
        actions = list(shortest[:insert_at]) + [out_action, back_action] + list(shortest[insert_at:])
        rollout = simulate_candidate(grid_size, start, goal, obstacles, actions, "valid_but_long")
        if rollout.valid and rollout.first_goal_step > len(shortest):
            return _pad_actions(actions, horizon, rng)
    return None


def _collision_actions(
    grid_size: int,
    start: Position,
    goal: Position,
    obstacles: Tuple[Position, ...],
    shortest: Sequence[int],
    horizon: int,
    rng: np.random.Generator,
) -> ActionSeq | None:
    positions = _positions_from_actions(start, shortest)
    obstacle_set = set(obstacles)
    for _ in range(40):
        step_index = int(rng.integers(0, max(1, len(positions) - 1)))
        pos = positions[step_index]
        bad_actions = []
        for action, (dr, dc) in ACTION_TO_DELTA.items():
            nxt = (pos[0] + dr, pos[1] + dc)
            if not _in_bounds(nxt, grid_size) or nxt in obstacle_set:
                bad_actions.append(action)
        if not bad_actions:
            continue
        actions = list(shortest[:step_index]) + [int(bad_actions[int(rng.integers(0, len(bad_actions)))])]
        return _pad_actions(actions, horizon, rng)
    return None


def _wrong_goal_actions(
    grid_size: int,
    start: Position,
    goal: Position,
    obstacles: Tuple[Position, ...],
    horizon: int,
    rng: np.random.Generator,
) -> ActionSeq | None:
    cells = [(r, c) for r in range(grid_size) for c in range(grid_size) if (r, c) != goal and (r, c) not in set(obstacles)]
    rng.shuffle(cells)
    for target in cells[:40]:
        if _manhattan(target, goal) < 2:
            continue
        path = _shortest_path(grid_size, start, target, obstacles)
        if path is not None and len(path) <= horizon and _shortest_path(grid_size, target, goal, obstacles) is not None:
            return _pad_actions(path, horizon, rng)
    return None


def _near_miss_actions(
    grid_size: int,
    start: Position,
    goal: Position,
    obstacles: Tuple[Position, ...],
    horizon: int,
    rng: np.random.Generator,
) -> ActionSeq | None:
    neighbors = [pos for pos, _action in _legal_neighbors(grid_size, goal, obstacles) if pos != start]
    rng.shuffle(neighbors)
    for target in neighbors:
        path = _shortest_path(grid_size, start, target, obstacles)
        if path is not None and len(path) <= horizon:
            return _pad_actions(path, horizon, rng)
    return None


def _deceptive_prefix_actions(
    grid_size: int,
    start: Position,
    goal: Position,
    obstacles: Tuple[Position, ...],
    shortest: Sequence[int],
    horizon: int,
    rng: np.random.Generator,
) -> ActionSeq | None:
    if len(shortest) < 3:
        return None
    prefix_len = int(rng.integers(1, max(2, len(shortest) - 1)))
    prefix = list(shortest[:prefix_len])
    pos = _positions_from_actions(start, prefix)[-1]
    choices = _legal_neighbors(grid_size, pos, obstacles)
    next_gold_pos = _positions_from_actions(start, shortest[: prefix_len + 1])[-1]
    choices = [item for item in choices if item[0] != next_gold_pos and item[0] != goal]
    if not choices:
        return _collision_actions(grid_size, start, goal, obstacles, shortest, horizon, rng)
    _nxt, action = choices[int(rng.integers(0, len(choices)))]
    return _pad_actions(prefix + [action], horizon, rng)


def _dead_end_actions(
    grid_size: int,
    start: Position,
    goal: Position,
    obstacles: Tuple[Position, ...],
    horizon: int,
    rng: np.random.Generator,
) -> ActionSeq | None:
    actions: List[int] = []
    pos = start
    for _ in range(min(horizon, max(4, horizon // 2))):
        choices = _legal_neighbors(grid_size, pos, obstacles)
        choices = [item for item in choices if _manhattan(item[0], goal) >= _manhattan(pos, goal)]
        if not choices:
            break
        pos, action = choices[int(rng.integers(0, len(choices)))]
        actions.append(int(action))
        if pos == goal:
            return None
    return _pad_actions(actions, horizon, rng) if actions else None


def _candidate_plan_only(example: PlanGridworldExample) -> PlanGridworldExample:
    blank_candidates = tuple(
        simulate_candidate(
            example.grid_size,
            (0, 0),
            (0, 0),
            tuple(),
            candidate.actions,
            candidate.family,
        )
        for candidate in example.candidates
    )
    return replace(
        example,
        start=(0, 0),
        goal=(0, 0),
        obstacles=tuple(),
        candidates=blank_candidates,
        metadata={**example.metadata, "control": "candidate_plan_only"},
    )


def _state_goal_only(example: PlanGridworldExample) -> PlanGridworldExample:
    neutral_actions = tuple([0, 1] * ((len(example.candidates[0].actions) + 1) // 2))[: len(example.candidates[0].actions)]
    candidates = tuple(
        simulate_candidate(example.grid_size, example.start, example.goal, example.obstacles, neutral_actions, "candidate_hidden")
        for _ in example.candidates
    )
    return replace(example, candidates=candidates, metadata={**example.metadata, "control": "state_goal_only"})


def _state_plan_mismatch(examples: Sequence[PlanGridworldExample], rng: np.random.Generator) -> List[PlanGridworldExample]:
    out = []
    for index, example in enumerate(examples):
        other = _other_example(examples, index, rng)
        candidates = tuple(
            simulate_candidate(other.grid_size, other.start, other.goal, other.obstacles, candidate.actions, candidate.family)
            for candidate in example.candidates
        )
        out.append(
            replace(
                example,
                grid_size=other.grid_size,
                start=other.start,
                goal=other.goal,
                obstacles=other.obstacles,
                candidates=candidates,
                metadata={**example.metadata, "control": "state_plan_mismatch", "state_from_id": other.id},
            )
        )
    return out


def _goal_shuffle(examples: Sequence[PlanGridworldExample], rng: np.random.Generator) -> List[PlanGridworldExample]:
    out = []
    for index, example in enumerate(examples):
        other = _other_example(examples, index, rng)
        candidates = tuple(
            simulate_candidate(example.grid_size, example.start, other.goal, example.obstacles, candidate.actions, candidate.family)
            for candidate in example.candidates
        )
        out.append(
            replace(example, goal=other.goal, candidates=candidates, metadata={**example.metadata, "control": "goal_shuffle", "goal_from_id": other.id})
        )
    return out


def _obstacle_shuffle(examples: Sequence[PlanGridworldExample], rng: np.random.Generator) -> List[PlanGridworldExample]:
    out = []
    for index, example in enumerate(examples):
        other = _other_example(examples, index, rng)
        obstacles = tuple(pos for pos in other.obstacles if pos not in {example.start, example.goal})
        candidates = tuple(
            simulate_candidate(example.grid_size, example.start, example.goal, obstacles, candidate.actions, candidate.family)
            for candidate in example.candidates
        )
        out.append(
            replace(
                example,
                obstacles=tuple(sorted(obstacles)),
                candidates=candidates,
                metadata={**example.metadata, "control": "obstacle_constraint_shuffle", "obstacles_from_id": other.id},
            )
        )
    return out


def _rollout_mismatch(examples: Sequence[PlanGridworldExample], rng: np.random.Generator) -> List[PlanGridworldExample]:
    out = []
    for index, example in enumerate(examples):
        other = _other_example(examples, index, rng)
        replacements = list(other.candidates)
        if len(replacements) < len(example.candidates):
            replacements = replacements * (len(example.candidates) // max(1, len(replacements)) + 1)
        candidates = []
        for candidate, replacement_rollout in zip(example.candidates, replacements):
            candidates.append(
                replace(
                    candidate,
                    rollout=replacement_rollout.rollout,
                    final_position=replacement_rollout.final_position,
                )
            )
        out.append(replace(example, candidates=tuple(candidates), metadata={**example.metadata, "control": "rollout_mismatch"}))
    return out


def _final_state_mismatch(examples: Sequence[PlanGridworldExample], rng: np.random.Generator) -> List[PlanGridworldExample]:
    """Break final-state evidence everywhere it is model-visible.

    PLAN-1 consumes final-state information from both the explicit outcome view
    and the last token of the rollout view. The original control changed only
    ``final_position``, so rollout-enabled models could still read the correct
    final state from ``candidate.rollout[-1]``. This repaired control changes
    both pieces while preserving the candidate action sequence.
    """

    out = []
    for index, example in enumerate(examples):
        other = _other_example(examples, index, rng)
        replacements = list(other.candidates)
        if len(replacements) < len(example.candidates):
            replacements = replacements * (len(example.candidates) // max(1, len(replacements)) + 1)
        candidates = []
        for candidate_index, candidate in enumerate(example.candidates):
            wrong_final = _different_final_position(
                candidate.final_position,
                replacements[candidate_index:],
                example.grid_size,
            )
            candidates.append(
                replace(
                    candidate,
                    rollout=_replace_rollout_final(candidate.rollout, wrong_final),
                    final_position=wrong_final,
                )
            )
        out.append(replace(example, candidates=tuple(candidates), metadata={**example.metadata, "control": "final_state_mismatch"}))
    return out


def _final_state_outcome_only_mismatch(examples: Sequence[PlanGridworldExample], rng: np.random.Generator) -> List[PlanGridworldExample]:
    out = []
    for index, example in enumerate(examples):
        other = _other_example(examples, index, rng)
        replacements = list(other.candidates)
        if len(replacements) < len(example.candidates):
            replacements = replacements * (len(example.candidates) // max(1, len(replacements)) + 1)
        candidates = []
        for candidate_index, candidate in enumerate(example.candidates):
            wrong_final = _different_final_position(
                candidate.final_position,
                replacements[candidate_index:],
                example.grid_size,
            )
            candidates.append(replace(candidate, final_position=wrong_final))
        out.append(replace(example, candidates=tuple(candidates), metadata={**example.metadata, "control": "final_state_outcome_only_mismatch"}))
    return out


def _candidate_order_shuffle(example: PlanGridworldExample, rng: np.random.Generator) -> PlanGridworldExample:
    order = rng.permutation(len(example.candidates))
    if len(order) > 1 and np.array_equal(order, np.arange(len(order))):
        order = np.roll(order, 1)
    candidates = tuple(example.candidates[int(i)] for i in order)
    tags = tuple(example.public_source_tags[int(i)] for i in order)
    label = int(np.where(order == int(example.label))[0][0])
    return replace(
        example,
        candidates=candidates,
        public_source_tags=tags,
        label=label,
        metadata={**example.metadata, "control": "candidate_order_shuffle_with_gold_remap"},
    )


def _horizontal_reflection(example: PlanGridworldExample) -> PlanGridworldExample:
    size = example.grid_size
    transform_pos = lambda pos: (pos[0], size - 1 - pos[1])
    mapping = {0: 0, 1: 1, 2: 3, 3: 2}
    return _geometric_transform(example, transform_pos, mapping, "action_symbol_permutation")


def _vertical_reflection(example: PlanGridworldExample) -> PlanGridworldExample:
    size = example.grid_size
    transform_pos = lambda pos: (size - 1 - pos[0], pos[1])
    mapping = {0: 1, 1: 0, 2: 2, 3: 3}
    return _geometric_transform(example, transform_pos, mapping, "map_rotation_reflection")


def _geometric_transform(
    example: PlanGridworldExample,
    transform_pos,
    action_mapping: Dict[int, int],
    control_name: str,
) -> PlanGridworldExample:
    start = transform_pos(example.start)
    goal = transform_pos(example.goal)
    obstacles = tuple(sorted(transform_pos(pos) for pos in example.obstacles))
    candidates = tuple(
        simulate_candidate(
            example.grid_size,
            start,
            goal,
            obstacles,
            tuple(action_mapping[int(action)] for action in candidate.actions),
            candidate.family,
        )
        for candidate in example.candidates
    )
    return replace(
        example,
        start=start,
        goal=goal,
        obstacles=obstacles,
        candidates=candidates,
        metadata={**example.metadata, "control": control_name},
    )


def _randomized_label(example: PlanGridworldExample, rng: np.random.Generator) -> PlanGridworldExample:
    return replace(example, label=int(rng.integers(0, len(example.candidates))), metadata={**example.metadata, "control": "randomized_labels"})


def _replace_rollout_final(rollout: Sequence[Position], final_position: Position) -> Tuple[Position, ...]:
    if not rollout:
        return (tuple(final_position),)
    rows = list(tuple(pos) for pos in rollout)
    rows[-1] = tuple(final_position)
    return tuple(rows)


def _different_final_position(
    original: Position,
    replacements: Sequence[CandidateRollout],
    grid_size: int,
) -> Position:
    for replacement in replacements:
        candidate = tuple(replacement.final_position)
        if candidate != tuple(original):
            return candidate
    # Degenerate tasks can have many candidates ending at the same cell. The
    # fallback keeps the control active without exposing reward/success flags.
    return ((int(original[0]) + 1) % max(1, int(grid_size)), int(original[1]))


def _other_example(examples: Sequence[PlanGridworldExample], index: int, rng: np.random.Generator) -> PlanGridworldExample:
    if len(examples) < 2:
        return examples[index]
    other_index = int(rng.integers(0, len(examples) - 1))
    if other_index >= index:
        other_index += 1
    return examples[other_index]


def _shortest_path(
    grid_size: int,
    start: Position,
    goal: Position,
    obstacles: Sequence[Position],
) -> Tuple[int, ...] | None:
    obstacle_set = set(obstacles)
    queue: deque[Position] = deque([start])
    parent: Dict[Position, Tuple[Position, int]] = {}
    seen = {start}
    while queue:
        pos = queue.popleft()
        if pos == goal:
            actions: List[int] = []
            while pos != start:
                prev, action = parent[pos]
                actions.append(action)
                pos = prev
            actions.reverse()
            return tuple(actions)
        for nxt, action in _legal_neighbors(grid_size, pos, obstacle_set):
            if nxt in seen:
                continue
            seen.add(nxt)
            parent[nxt] = (pos, action)
            queue.append(nxt)
    return None


def _legal_neighbors(grid_size: int, pos: Position, obstacles: Sequence[Position]) -> List[Tuple[Position, int]]:
    obstacle_set = set(obstacles)
    out = []
    for action, (dr, dc) in ACTION_TO_DELTA.items():
        nxt = (pos[0] + dr, pos[1] + dc)
        if _in_bounds(nxt, grid_size) and nxt not in obstacle_set:
            out.append((nxt, int(action)))
    return out


def _pad_actions(actions: Sequence[int], horizon: int, rng: np.random.Generator) -> ActionSeq:
    out = list(int(a) for a in actions[:horizon])
    while len(out) < int(horizon):
        out.append(int(rng.integers(0, 4)))
    return tuple(out[:horizon])


def _positions_from_actions(start: Position, actions: Sequence[int]) -> List[Position]:
    pos = start
    positions = [pos]
    for action in actions:
        dr, dc = ACTION_TO_DELTA[int(action)]
        pos = (pos[0] + dr, pos[1] + dc)
        positions.append(pos)
    return positions


def _action_between(a: Position, b: Position) -> int:
    delta = (b[0] - a[0], b[1] - a[1])
    for action, action_delta in ACTION_TO_DELTA.items():
        if action_delta == delta:
            return int(action)
    raise ValueError(f"positions are not adjacent: {a} -> {b}")


def _candidate_score(candidate: CandidateRollout) -> Tuple[int, int]:
    if candidate.valid:
        return (1, -int(candidate.first_goal_step))
    return (0, -10_000 + int(candidate.near_miss) - int(candidate.collision))


def _exactly_one_best(example: PlanGridworldExample) -> bool:
    scores = [_candidate_score(candidate) for candidate in example.candidates]
    return bool(scores.count(max(scores)) == 1)


def _in_bounds(pos: Position, grid_size: int) -> bool:
    return 0 <= pos[0] < int(grid_size) and 0 <= pos[1] < int(grid_size)


def _manhattan(a: Position, b: Position) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _count_strings(values: Iterable[str]) -> Dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))
