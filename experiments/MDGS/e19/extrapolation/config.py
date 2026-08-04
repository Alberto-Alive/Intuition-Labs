"""Frozen training configuration for DIGIT Extrapolation E19."""

from __future__ import annotations

from dataclasses import dataclass

from experiments.DIGIT.Extrapolation.e14.extrapolation.data import (
    NUM_CLASSES,
    NUM_PART_A_VALUES,
    NUM_PART_B_VALUES,
)


@dataclass(frozen=True)
class E19Config:
    seed: int = 0
    device: str = "cpu"

    num_part_a_values: int = NUM_PART_A_VALUES
    num_part_b_values: int = NUM_PART_B_VALUES
    num_classes: int = NUM_CLASSES

    core_repeats_per_combo: int = 200
    familiar_repeats_per_combo: int = 20
    rare_repeats_per_combo: int = 2
    novel_repeats_per_combo: int = 0
    test_repeats_per_combo: int = 32
    batch_size: int = 64

    embedding_dim: int = 16
    hidden_dim: int = 32
    bank_capacity: int = 4096
    min_bank_occupancy: int = 410
    input_jitter_std: float = 0.05
    train_noise_seed_offset: int = 10_000
    test_noise_seed_offset: int = 20_000

    learning_rate: float = 1e-3
    max_epochs: int = 200
    loss_improvement_tolerance: float = 1e-4
    loss_patience_epochs: int = 20

