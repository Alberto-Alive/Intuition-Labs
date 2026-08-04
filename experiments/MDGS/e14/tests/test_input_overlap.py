from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extrapolation.config import E14Config
from extrapolation.experiment import build_dataset, compute_exact_seen_input_overlap_fraction


def test_seen_test_examples_have_zero_exact_raw_input_overlap_with_training() -> None:
    dataset = build_dataset(E14Config(seed=0))
    overlap_fraction = compute_exact_seen_input_overlap_fraction(dataset)
    assert overlap_fraction == 0.0
