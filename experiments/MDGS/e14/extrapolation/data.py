"""Synthetic compositional dataset specification for DIGIT Extrapolation E14."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
import random
from typing import Iterable


NUM_PART_A_VALUES = 8
NUM_PART_B_VALUES = 8
NUM_CLASSES = 4
TOTAL_COMBINATIONS = NUM_PART_A_VALUES * NUM_PART_B_VALUES
DEFAULT_WITHHELD_COUNT = 16
MIN_SEEN_COMBINATIONS_PER_PART = 6
DEFAULT_HOLDOUT_SEED = 0


@dataclass(frozen=True)
class CombinationRecord:
    """One `(part_a, part_b)` combination and its metadata."""

    part_a: int
    part_b: int
    class_id: int
    is_withheld: bool


def all_combinations() -> tuple[tuple[int, int], ...]:
    """Return the full 8x8 grid of `(part_a, part_b)` combinations."""
    return tuple(
        (part_a, part_b)
        for part_a in range(NUM_PART_A_VALUES)
        for part_b in range(NUM_PART_B_VALUES)
    )


def fixed_class_id(part_a: int, part_b: int) -> int:
    """Balanced conjunction-dependent class mapping shared across all seeds."""
    _validate_part_value(part_a, upper=NUM_PART_A_VALUES, name="part_a")
    _validate_part_value(part_b, upper=NUM_PART_B_VALUES, name="part_b")
    return (part_a + part_b) % NUM_CLASSES


def sample_withheld_combinations(
    seed: int,
    *,
    withheld_count: int = DEFAULT_WITHHELD_COUNT,
    min_seen_per_part: int = MIN_SEEN_COMBINATIONS_PER_PART,
) -> tuple[tuple[int, int], ...]:
    """Sample a seed-reproducible holdout uniformly over every valid 16-cell subset.

    Under the frozen E14 constraints:

    - 8 values for `part_a`
    - 8 values for `part_b`
    - 16 withheld combinations
    - at least 6 seen combinations per `part_a` and per `part_b`

    every valid holdout must contain exactly 2 withheld cells in each row and
    exactly 2 withheld cells in each column.

    This function samples uniformly from that valid set by using dynamic
    programming to count the number of completions from each partial row-wise
    assignment, then drawing each row's column-pair proportionally to the number
    of exact completions that remain.
    """
    if withheld_count <= 0 or withheld_count >= TOTAL_COMBINATIONS:
        raise ValueError("withheld_count must be positive and smaller than the total number of combinations")
    if not 0 <= min_seen_per_part <= min(NUM_PART_A_VALUES, NUM_PART_B_VALUES):
        raise ValueError("min_seen_per_part must lie within the feasible row/column range")

    withheld_per_row = NUM_PART_B_VALUES - min_seen_per_part
    withheld_per_col = NUM_PART_A_VALUES - min_seen_per_part
    expected_withheld = NUM_PART_A_VALUES * withheld_per_row
    if withheld_count != expected_withheld or withheld_count != NUM_PART_B_VALUES * withheld_per_col:
        raise ValueError(
            "The requested withheld_count is incompatible with the minimum seen-count "
            "constraint for the current 8x8 experiment"
        )

    @lru_cache(maxsize=None)
    def count_completions(
        row_index: int,
        remaining_column_capacity: tuple[int, ...],
    ) -> int:
        rows_remaining = NUM_PART_A_VALUES - row_index
        required_cells = rows_remaining * withheld_per_row
        if sum(remaining_column_capacity) != required_cells:
            return 0
        if row_index == NUM_PART_A_VALUES:
            return 1 if all(value == 0 for value in remaining_column_capacity) else 0

        total = 0
        available_columns = [index for index, value in enumerate(remaining_column_capacity) if value > 0]
        for column_choice in combinations(available_columns, withheld_per_row):
            next_capacity = list(remaining_column_capacity)
            for column_index in column_choice:
                next_capacity[column_index] -= 1
            total += count_completions(row_index + 1, tuple(next_capacity))
        return total

    rng = random.Random(seed)
    remaining_column_capacity = [withheld_per_col for _ in range(NUM_PART_B_VALUES)]
    withheld: list[tuple[int, int]] = []

    for part_a in range(NUM_PART_A_VALUES):
        options: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []
        available_columns = [index for index, value in enumerate(remaining_column_capacity) if value > 0]
        for column_choice in combinations(available_columns, withheld_per_row):
            next_capacity = list(remaining_column_capacity)
            for column_index in column_choice:
                next_capacity[column_index] -= 1
            completion_count = count_completions(part_a + 1, tuple(next_capacity))
            if completion_count > 0:
                options.append((column_choice, tuple(next_capacity), completion_count))

        total_weight = sum(weight for _, _, weight in options)
        if total_weight <= 0:
            raise RuntimeError(f"No valid holdout completions remain for seed {seed} at row {part_a}")

        draw = rng.randrange(total_weight)
        cumulative = 0
        chosen_columns: tuple[int, ...] | None = None
        chosen_next_capacity: tuple[int, ...] | None = None
        for column_choice, next_capacity, weight in options:
            cumulative += weight
            if draw < cumulative:
                chosen_columns = column_choice
                chosen_next_capacity = next_capacity
                break

        if chosen_columns is None or chosen_next_capacity is None:
            raise RuntimeError(f"Failed to sample a valid holdout choice for seed {seed} at row {part_a}")

        withheld.extend((part_a, part_b) for part_b in chosen_columns)
        remaining_column_capacity = list(chosen_next_capacity)

    result = tuple(sorted(withheld))
    if not _withheld_satisfies_seen_constraint(result, min_seen_per_part=min_seen_per_part):
        raise RuntimeError(f"Sampled holdout for seed {seed} violates the seen-count constraint")
    return result


def build_combination_records(
    *,
    seed: int = DEFAULT_HOLDOUT_SEED,
    withheld_combinations: Iterable[tuple[int, int]] | None = None,
) -> tuple[CombinationRecord, ...]:
    """Build the full 8x8 combination table with fixed classes and holdout flags."""
    withheld = _normalize_withheld_combinations(
        seed=seed,
        withheld_combinations=withheld_combinations,
    )
    return tuple(
        CombinationRecord(
            part_a=part_a,
            part_b=part_b,
            class_id=fixed_class_id(part_a, part_b),
            is_withheld=(part_a, part_b) in withheld,
        )
        for part_a in range(NUM_PART_A_VALUES)
        for part_b in range(NUM_PART_B_VALUES)
    )


def class_matrix() -> tuple[tuple[int, ...], ...]:
    """Return the fixed 8x8 class lookup table."""
    return tuple(
        tuple(fixed_class_id(part_a, part_b) for part_b in range(NUM_PART_B_VALUES))
        for part_a in range(NUM_PART_A_VALUES)
    )


def withheld_matrix(
    *,
    seed: int = DEFAULT_HOLDOUT_SEED,
    withheld_combinations: Iterable[tuple[int, int]] | None = None,
) -> tuple[tuple[bool, ...], ...]:
    """Return the 8x8 holdout mask as a boolean matrix."""
    withheld = _normalize_withheld_combinations(
        seed=seed,
        withheld_combinations=withheld_combinations,
    )
    matrix: list[list[bool]] = [[False for _ in range(NUM_PART_B_VALUES)] for _ in range(NUM_PART_A_VALUES)]
    for part_a, part_b in withheld:
        matrix[part_a][part_b] = True
    return tuple(tuple(row) for row in matrix)


def seen_combination_counts_by_part(
    *,
    seed: int = DEFAULT_HOLDOUT_SEED,
    withheld_combinations: Iterable[tuple[int, int]] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Count how many seen combinations remain for each part value."""
    records = build_combination_records(seed=seed, withheld_combinations=withheld_combinations)
    part_a_counts = [0 for _ in range(NUM_PART_A_VALUES)]
    part_b_counts = [0 for _ in range(NUM_PART_B_VALUES)]
    for record in records:
        if record.is_withheld:
            continue
        part_a_counts[record.part_a] += 1
        part_b_counts[record.part_b] += 1
    return tuple(part_a_counts), tuple(part_b_counts)


def class_counts(
    *,
    seed: int = DEFAULT_HOLDOUT_SEED,
    withheld_combinations: Iterable[tuple[int, int]] | None = None,
    withheld_only: bool | None = None,
) -> tuple[int, ...]:
    """Count combinations per class for the full table, seen-only, or withheld-only."""
    records = build_combination_records(seed=seed, withheld_combinations=withheld_combinations)
    counts = [0 for _ in range(NUM_CLASSES)]
    for record in records:
        if withheld_only is True and not record.is_withheld:
            continue
        if withheld_only is False and record.is_withheld:
            continue
        counts[record.class_id] += 1
    return tuple(counts)


def render_combination_matrix(
    *,
    seed: int = DEFAULT_HOLDOUT_SEED,
    withheld_combinations: Iterable[tuple[int, int]] | None = None,
) -> str:
    """Render the 8x8 grid with class labels and withheld markers."""
    matrix = class_matrix()
    holdout_mask = withheld_matrix(seed=seed, withheld_combinations=withheld_combinations)
    lines = ["part_a\\part_b | 0  1  2  3  4  5  6  7", "-------------+------------------------"]
    for part_a in range(NUM_PART_A_VALUES):
        cells = []
        for part_b in range(NUM_PART_B_VALUES):
            marker = "*" if holdout_mask[part_a][part_b] else " "
            cells.append(f"{matrix[part_a][part_b]}{marker}")
        lines.append(f"{part_a:<12} | " + " ".join(cells))
    return "\n".join(lines)


def _normalize_withheld_combinations(
    *,
    seed: int,
    withheld_combinations: Iterable[tuple[int, int]] | None,
) -> frozenset[tuple[int, int]]:
    raw_withheld = sample_withheld_combinations(seed) if withheld_combinations is None else tuple(withheld_combinations)
    normalized: set[tuple[int, int]] = set()
    for part_a, part_b in raw_withheld:
        _validate_part_value(part_a, upper=NUM_PART_A_VALUES, name="part_a")
        _validate_part_value(part_b, upper=NUM_PART_B_VALUES, name="part_b")
        normalized.add((part_a, part_b))

    if len(normalized) != DEFAULT_WITHHELD_COUNT:
        raise ValueError(
            f"Expected exactly {DEFAULT_WITHHELD_COUNT} unique withheld combinations, got {len(normalized)}"
        )
    if not _withheld_satisfies_seen_constraint(normalized, min_seen_per_part=MIN_SEEN_COMBINATIONS_PER_PART):
        raise ValueError(
            f"Provided withheld combinations violate the minimum seen count of "
            f"{MIN_SEEN_COMBINATIONS_PER_PART} per part value"
        )
    return frozenset(normalized)


def _withheld_satisfies_seen_constraint(
    withheld: Iterable[tuple[int, int]],
    *,
    min_seen_per_part: int,
) -> bool:
    part_a_seen = [NUM_PART_B_VALUES for _ in range(NUM_PART_A_VALUES)]
    part_b_seen = [NUM_PART_A_VALUES for _ in range(NUM_PART_B_VALUES)]
    for part_a, part_b in withheld:
        part_a_seen[part_a] -= 1
        part_b_seen[part_b] -= 1
    return min(part_a_seen) >= min_seen_per_part and min(part_b_seen) >= min_seen_per_part


def _validate_part_value(value: int, *, upper: int, name: str) -> None:
    if not 0 <= value < upper:
        raise ValueError(f"{name} must lie in [0, {upper}), got {value}")
