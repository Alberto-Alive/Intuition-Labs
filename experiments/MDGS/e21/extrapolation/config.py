"""Frozen configuration for DIGIT Extrapolation E21."""

from __future__ import annotations

from dataclasses import dataclass

from experiments.DIGIT.Extrapolation.e14.extrapolation.data import (
    NUM_CLASSES,
    NUM_PART_A_VALUES,
    NUM_PART_B_VALUES,
)


@dataclass(frozen=True)
class E21Config:
    seed: int = 0
    device: str = "cpu"

    num_part_a_values: int = NUM_PART_A_VALUES
    num_part_b_values: int = NUM_PART_B_VALUES
    num_classes: int = NUM_CLASSES

    phase1_core_repeats_per_combo: int = 200
    phase1_familiar_repeats_per_combo: int = 20
    phase1_rare_repeats_per_combo: int = 0
    phase1_novel_repeats_per_combo: int = 0
    phase2_repeats_per_combo: int = 20
    phase2_epochs: int = 50
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

    retrieval_threshold: float = 0.95
    retrieval_bottom_fraction: float = 0.25
    retrieved_neighbor_count: int = 5
    phase2_accuracy_threshold: float = 0.75
