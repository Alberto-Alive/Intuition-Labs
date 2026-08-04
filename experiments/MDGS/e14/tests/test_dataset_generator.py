from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extrapolation.data import (
    DEFAULT_HOLDOUT_SEED,
    DEFAULT_WITHHELD_COUNT,
    MIN_SEEN_COMBINATIONS_PER_PART,
    NUM_CLASSES,
    TOTAL_COMBINATIONS,
    all_combinations,
    build_combination_records,
    class_counts,
    class_matrix,
    sample_withheld_combinations,
    seen_combination_counts_by_part,
)


def test_seeded_holdout_is_reproducible_and_has_expected_size() -> None:
    seed_zero_first = sample_withheld_combinations(DEFAULT_HOLDOUT_SEED)
    seed_zero_second = sample_withheld_combinations(DEFAULT_HOLDOUT_SEED)
    seed_one = sample_withheld_combinations(1)

    assert len(seed_zero_first) == DEFAULT_WITHHELD_COUNT
    assert seed_zero_first == seed_zero_second
    assert seed_zero_first != seed_one
    assert len(set(seed_zero_first)) == DEFAULT_WITHHELD_COUNT
    assert set(seed_zero_first).issubset(set(all_combinations()))


def test_seeded_holdout_keeps_every_part_value_frequent_in_seen_set() -> None:
    records = build_combination_records(seed=DEFAULT_HOLDOUT_SEED)
    part_a_seen_counts, part_b_seen_counts = seen_combination_counts_by_part(seed=DEFAULT_HOLDOUT_SEED)

    assert len(records) == TOTAL_COMBINATIONS
    assert sum(record.is_withheld for record in records) == DEFAULT_WITHHELD_COUNT
    assert min(part_a_seen_counts) >= MIN_SEEN_COMBINATIONS_PER_PART
    assert min(part_b_seen_counts) >= MIN_SEEN_COMBINATIONS_PER_PART


def test_fixed_class_lookup_is_balanced_across_all_combinations() -> None:
    full_counts = class_counts(seed=DEFAULT_HOLDOUT_SEED)

    assert full_counts == (16, 16, 16, 16)
    assert sum(full_counts) == TOTAL_COMBINATIONS
    assert len(class_matrix()) == 8
    assert all(len(row) == 8 for row in class_matrix())
    assert len(full_counts) == NUM_CLASSES


def test_class_lookup_depends_on_both_parts() -> None:
    matrix = class_matrix()

    # No row is constant in class, so `part_a` alone does not determine the label.
    assert all(len(set(row)) > 1 for row in matrix)

    # No column is constant in class, so `part_b` alone does not determine the label.
    for column_index in range(8):
        column_values = {matrix[row_index][column_index] for row_index in range(8)}
        assert len(column_values) > 1
