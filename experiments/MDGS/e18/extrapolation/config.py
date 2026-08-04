"""Frozen training configuration for DIGIT Extrapolation E18."""

from __future__ import annotations

from dataclasses import dataclass

from experiments.DIGIT.Extrapolation.e14.extrapolation.data import (
    DEFAULT_WITHHELD_COUNT,
    NUM_CLASSES,
    NUM_PART_A_VALUES,
    NUM_PART_B_VALUES,
)


@dataclass(frozen=True)
class E18Config:
    seed: int = 0
    device: str = "cpu"

    num_part_a_values: int = NUM_PART_A_VALUES
    num_part_b_values: int = NUM_PART_B_VALUES
    num_classes: int = NUM_CLASSES
    withheld_count: int = DEFAULT_WITHHELD_COUNT

    train_repeats_per_seen_combo: int = 128
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
    fixed_budget_epochs: int = 59

    alpha: float = 10.0
    beta: float = 0.1
    certainty_threshold: float = 0.95

