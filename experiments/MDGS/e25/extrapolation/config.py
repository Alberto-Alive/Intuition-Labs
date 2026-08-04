"""Frozen configuration for DIGIT Extrapolation E25.

Inherits training setup from E22 exactly. Adds E25-specific constants
for the LTH procedure and checkpoint schedule.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class E25Config:
    seed: int = 0
    device: str = "cuda"

    # Dataset — identical to E22/E19
    num_part_a_values: int = 8
    num_part_b_values: int = 8
    num_classes: int = 4
    core_repeats_per_combo: int = 200
    familiar_repeats_per_combo: int = 20
    rare_repeats_per_combo: int = 2
    novel_repeats_per_combo: int = 0
    test_repeats_per_combo: int = 32

    # Model — identical to E22
    embedding_dim: int = 16
    hidden_dim: int = 32
    key_dim: int = 32
    num_prototypes: int = 32
    prototype_lambda: float = 0.99
    prototype_eps: float = 1e-8

    # Training — identical to E22
    batch_size: int = 64
    learning_rate: float = 1e-3
    max_epochs: int = 200
    loss_improvement_tolerance: float = 1e-4
    loss_patience_epochs: int = 20
    input_jitter_std: float = 0.05
    train_noise_seed_offset: int = 10_000
    test_noise_seed_offset: int = 20_000

    # E25-specific: familiarity exposure checkpoint epochs
    exposure_checkpoint_epochs: tuple[int, ...] = (5, 10, 20)

    # E25-specific: LTH procedure
    lth_sparsity: float = 0.5          # global sparsity target
    lth_performance_threshold: float = 0.99  # pruned/dense accuracy ratio

    # E25-specific: numerical stability
    exposure_eps: float = 1e-8
