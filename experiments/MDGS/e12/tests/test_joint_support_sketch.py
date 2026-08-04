from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extrapolation.models import JointSupportSketch


def test_joint_support_sketch_updates_mean_and_variance_by_exact_ema() -> None:
    projection = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    sketch = JointSupportSketch(
        input_dim=4,
        sketch_dim=2,
        mean_decay=0.8,
        variance_decay=0.6,
        eps=1e-9,
        projection=projection,
    )
    activations = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ],
        requires_grad=True,
    )

    sketches = sketch.update(activations)
    expected_projected = torch.tensor([[1.0, 2.0], [5.0, 6.0]])
    expected_mean = torch.tensor([0.6, 0.8])
    expected_variance = torch.tensor([5.8, 8.6])

    assert torch.allclose(sketches, expected_projected, atol=1e-6)
    assert torch.allclose(sketch.mean, expected_mean, atol=1e-6)
    assert torch.allclose(sketch.variance, expected_variance, atol=1e-6)
    assert sketch.num_updates.item() == 1
    assert sketch.projection.requires_grad is False
    assert "projection" not in {name for name, _ in sketch.named_parameters()}


def test_joint_support_distance_is_higher_for_unseen_joint_configuration() -> None:
    projection = torch.tensor(
        [
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0, -1.0],
        ]
    )
    sketch = JointSupportSketch(
        input_dim=4,
        sketch_dim=2,
        mean_decay=0.8,
        variance_decay=0.8,
        eps=1e-6,
        projection=projection,
    )
    familiar_batch = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.1, 0.9, -0.1, 0.1],
            [-0.1, 0.1, 1.1, 0.9],
        ]
    )

    for _ in range(12):
        sketch.update(familiar_batch)

    familiar_distance = sketch.novelty_distance(torch.tensor([[1.0, 1.0, 0.0, 0.0]])).item()
    unseen_distance = sketch.novelty_distance(torch.tensor([[1.0, 0.0, 1.0, 0.0]])).item()

    assert unseen_distance > familiar_distance
