"""Frozen configuration for the E12 Regime 2 experiment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class E12Config:
    seed: int = 7
    device: str = "cpu"

    num_colors: int = 8
    num_shapes: int = 8
    num_classes: int = 4
    sequence_length: int = 3
    vocab_size: int = 17
    cls_token_id: int = 0
    shape_token_offset: int = 9
    withheld_modulus: int = 4
    withheld_remainder: int = 0

    train_repeats_per_seen_combo: int = 20
    val_repeats_per_seen_combo: int = 4
    test_repeats_per_combo: int = 6
    batch_size: int = 16

    d_model: int = 32
    nhead: int = 4
    num_layers: int = 2
    d_ff: int = 64
    dropout: float = 0.0

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 12

    local_support_decay: float = 0.9
    joint_mean_decay: float = 0.9
    joint_variance_decay: float = 0.9
    joint_sketch_dim: int = 16
    support_eps: float = 1e-6

    local_stabilization_window: int = 500
    local_stabilization_mae_threshold: float = 0.01
    joint_stabilization_auc_threshold: float = 0.65
    warmup_probe_size: int = 128

    backward_alpha: float = 10.0
    backward_beta: float = 0.1

