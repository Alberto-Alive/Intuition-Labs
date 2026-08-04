"""Fixed four-level familiarity dataset specification for DIGIT Extrapolation E19."""

from __future__ import annotations

from dataclasses import dataclass


NUM_PART_A_VALUES = 8
NUM_PART_B_VALUES = 8
NUM_CLASSES = 4
TOTAL_COMBINATIONS = NUM_PART_A_VALUES * NUM_PART_B_VALUES

LEVEL_CORE = 1
LEVEL_FAMILIAR = 2
LEVEL_RARE = 3
LEVEL_NOVEL = 4
LEVEL_ORDER = (LEVEL_CORE, LEVEL_FAMILIAR, LEVEL_RARE, LEVEL_NOVEL)
LEVEL_NAMES = {
    LEVEL_CORE: "core",
    LEVEL_FAMILIAR: "familiar",
    LEVEL_RARE: "rare",
    LEVEL_NOVEL: "novel",
}
LEVEL_TO_MATCHINGS = {
    LEVEL_CORE: (0, 5),
    LEVEL_FAMILIAR: (1, 6),
    LEVEL_RARE: (2, 7),
    LEVEL_NOVEL: (3, 4),
}


@dataclass(frozen=True)
class FrequencyLevelRecord:
    """One `(part_a, part_b)` combination and its frozen familiarity level."""

    part_a: int
    part_b: int
    class_id: int
    level: int
    level_name: str


def fixed_class_id(part_a: int, part_b: int) -> int:
    """Balanced conjunction-dependent class mapping shared across all seeds."""
    return (part_a + part_b) % NUM_CLASSES


def build_level_assignment() -> dict[tuple[int, int], int]:
    """Construct the frozen E19 level assignment from the matching definition."""
    assignment: dict[tuple[int, int], int] = {}
    for level, matching_ids in LEVEL_TO_MATCHINGS.items():
        for matching_id in matching_ids:
            for part_a in range(NUM_PART_A_VALUES):
                part_b = (part_a + matching_id) % NUM_PART_B_VALUES
                assignment[(part_a, part_b)] = level
    return assignment


def build_frequency_level_records() -> tuple[FrequencyLevelRecord, ...]:
    """Return the full 8x8 grid with fixed class ids and familiarity levels."""
    assignment = build_level_assignment()
    return tuple(
        FrequencyLevelRecord(
            part_a=part_a,
            part_b=part_b,
            class_id=fixed_class_id(part_a, part_b),
            level=assignment[(part_a, part_b)],
            level_name=LEVEL_NAMES[assignment[(part_a, part_b)]],
        )
        for part_a in range(NUM_PART_A_VALUES)
        for part_b in range(NUM_PART_B_VALUES)
    )


def render_level_assignment_table() -> str:
    """Render the frozen level table as a simple 8x8 matrix."""
    assignment = build_level_assignment()
    symbol = {
        LEVEL_CORE: "C",
        LEVEL_FAMILIAR: "F",
        LEVEL_RARE: "R",
        LEVEL_NOVEL: "N",
    }
    lines = ["part_a\\part_b | 0  1  2  3  4  5  6  7", "-------------+------------------------"]
    for part_a in range(NUM_PART_A_VALUES):
        cells = [symbol[assignment[(part_a, part_b)]] for part_b in range(NUM_PART_B_VALUES)]
        lines.append(f"{part_a:<12} | " + "  ".join(cells))
    return "\n".join(lines)
