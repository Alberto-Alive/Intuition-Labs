"""Frozen configuration for DIGIT Extrapolation E26."""

from __future__ import annotations

from dataclasses import dataclass

from experiments.DIGIT.Extrapolation.e19.extrapolation.data import (
    NUM_CLASSES,
    NUM_PART_A_VALUES,
    NUM_PART_B_VALUES,
)


@dataclass(frozen=True)
class E26Config:
    seed: int = 0
    device: str = "cpu"

    num_part_a_values: int = NUM_PART_A_VALUES
    num_part_b_values: int = NUM_PART_B_VALUES
    num_classes: int = NUM_CLASSES

    task_a_part_b_values: tuple[int, ...] = (0, 1, 2, 3)
    task_b_part_b_values: tuple[int, ...] = (4, 5, 6, 7)

    task_a_train_repeats_per_combo: int = 200
    task_b_train_repeats_per_combo: int = 200
    task_a_probe_repeats_per_combo: int = 32
    task_b_test_repeats_per_combo: int = 32
    batch_size: int = 64

    embedding_dim: int = 16
    hidden_dim: int = 32
    key_dim: int = 32
    num_prototypes: int = 32
    prototype_lambda: float = 0.99
    prototype_eps: float = 1e-8

    input_jitter_std: float = 0.05
    train_noise_seed_offset: int = 10_000
    probe_noise_seed_offset: int = 20_000
    test_noise_seed_offset: int = 30_000

    learning_rate: float = 1e-3
    max_epochs: int = 200
    loss_improvement_tolerance: float = 1e-4
    loss_patience_epochs: int = 20

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
