"""Frozen configuration constants for DIGIT Extrapolation E23."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class E23Config:
    """Configuration needed for E23 model construction and initialisation tests."""

    seed: int = 0
    device: str = "cuda"
    num_part_a_values: int = 8
    num_part_b_values: int = 8
    num_classes: int = 4

    train_repeats_per_combo: int = 200
    test_repeats_per_combo: int = 128
    batch_size: int = 64

    embedding_dim: int = 16
    hidden_dim: int = 32
    key_dim: int = 32
    num_memory_slots: int = 32
    memory_lambda: float = 0.99
    memory_eps: float = 1e-8
    memory_init_std: float = 0.01

    input_jitter_std: float = 0.05
    train_noise_seed_offset: int = 10_000
    test_noise_seed_offset: int = 20_000

    learning_rate: float = 1e-3
    max_epochs: int = 200
    loss_improvement_tolerance: float = 1e-4
    loss_patience_epochs: int = 20

    seed0_report_epochs: tuple[int, ...] = (10, 20, 50)
