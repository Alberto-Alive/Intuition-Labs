from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "src"
    / "experiments"
    / "run_stage12c_coordination_subspace.py"
)
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("stage12c_coordination_subspace", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_leave_one_out_residualize_uses_same_example_negative_baseline() -> None:
    features = np.asarray(
        [
            [3.0, 1.0],  # ex0 seed31 positive
            [1.0, 1.0],  # ex0 seed30 negative
            [2.0, 2.0],  # ex1 seed31 negative
            [4.0, 2.0],  # ex1 seed37 positive
            [1.0, 2.0],  # ex1 seed30 negative
        ],
        dtype=np.float32,
    )
    metadata = [
        {"example_id": "ex0", "gain_positive": True},
        {"example_id": "ex0", "gain_positive": False},
        {"example_id": "ex1", "gain_positive": False},
        {"example_id": "ex1", "gain_positive": True},
        {"example_id": "ex1", "gain_positive": False},
    ]
    residual = MODULE._leave_one_out_residualize(features, metadata)
    assert np.allclose(residual[0], np.asarray([2.0, 0.0], dtype=np.float32))
    assert np.allclose(residual[1], np.asarray([-2.0, 0.0], dtype=np.float32))
    assert np.allclose(residual[3], np.asarray([2.5, 0.0], dtype=np.float32))


def test_inject_by_example_avenue_only_modifies_selected_avenue() -> None:
    token_states = torch.zeros(2, 3, 4, 5, 6, dtype=torch.float32)
    avenues = torch.as_tensor([1, 3], dtype=torch.long)
    injection = torch.arange(2 * 3 * 6, dtype=torch.float32).reshape(2, 3, 6)
    out = MODULE._inject_by_example_avenue(token_states, avenues, injection)
    assert torch.allclose(out[0, :, 0], torch.zeros_like(out[0, :, 0]))
    assert torch.allclose(out[1, :, 2], torch.zeros_like(out[1, :, 2]))
    expected0 = injection[0].unsqueeze(1).expand(-1, 5, -1)
    expected1 = injection[1].unsqueeze(1).expand(-1, 5, -1)
    assert torch.allclose(out[0, :, 1], expected0)
    assert torch.allclose(out[1, :, 3], expected1)


def test_mean_difference_direction_and_threshold_separate_simple_case() -> None:
    train_x = np.asarray(
        [
            [2.0, 2.0],
            [1.5, 1.0],
            [-1.0, -1.0],
            [-2.0, -1.0],
        ],
        dtype=np.float32,
    )
    train_y = np.asarray([1, 1, 0, 0], dtype=np.int64)
    direction = MODULE._mean_difference_direction(train_x, train_y)
    scores = train_x @ direction["unit_direction"]
    threshold = MODULE._score_threshold(scores, train_y)
    metrics = MODULE._score_metrics(scores, train_y, threshold)
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
