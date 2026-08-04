"""Invariant tests for the frozen E19 frequency-level assignment."""

from __future__ import annotations

from collections import Counter, defaultdict


def build_level_assignment() -> dict[tuple[int, int], int]:
    """Construct the frozen E19 level assignment from the matching definition."""
    level_to_matchings = {
        1: (0, 5),
        2: (1, 6),
        3: (2, 7),
        4: (3, 4),
    }
    assignment: dict[tuple[int, int], int] = {}
    for level, matching_ids in level_to_matchings.items():
        for matching_id in matching_ids:
            for part_a in range(8):
                part_b = (part_a + matching_id) % 8
                assignment[(part_a, part_b)] = level
    return assignment


def test_each_level_contains_exactly_16_combinations() -> None:
    """Each of the four levels must contain exactly 16 combinations."""
    assignment = build_level_assignment()
    counts = Counter(assignment.values())

    assert counts == {1: 16, 2: 16, 3: 16, 4: 16}


def test_each_part_a_value_appears_exactly_twice_per_level() -> None:
    """Every part_a value must appear exactly twice in every frequency level."""
    assignment = build_level_assignment()
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for (part_a, _part_b), level in assignment.items():
        counts[level][part_a] += 1

    for level in (1, 2, 3, 4):
        assert counts[level] == {part_a: 2 for part_a in range(8)}


def test_each_part_b_value_appears_exactly_twice_per_level() -> None:
    """Every part_b value must appear exactly twice in every frequency level."""
    assignment = build_level_assignment()
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for (_part_a, part_b), level in assignment.items():
        counts[level][part_b] += 1

    for level in (1, 2, 3, 4):
        assert counts[level] == {part_b: 2 for part_b in range(8)}
