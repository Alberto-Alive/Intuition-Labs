from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extrapolation.config import E12Config
from extrapolation.experiment import verify_neutral_certainty


def test_neutral_certainty_matches_baseline_transformer_dynamics() -> None:
    verification = verify_neutral_certainty(E12Config())

    assert verification["logits_max_abs_diff"] < 1e-8
    assert verification["grad_max_abs_diff"] < 1e-8
    assert verification["parameter_step_max_abs_diff"] < 1e-8
