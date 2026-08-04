"""Configuration dataclass for the DIGIT Extrapolation experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class Config:
    # Model dimensions
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1

    # Query encoding
    num_query_fields: int = 13
    field_vocab_sizes: List[int] = field(
        default_factory=lambda: [15, 17, 3, 6, 7, 9, 8, 6, 21, 3, 3, 7, 13]
    )

    # Vocabulary / sequences
    max_output_len: int = 48
    trajectory_len: int = 6

    # Bottleneck
    num_trajectory_classes: int = 4
    num_pattern_classes: int = 3
    num_confidence_classes: int = 3
    num_outcome_classes: int = 3
    gumbel_tau_start: float = 2.0
    gumbel_tau_end: float = 0.5
    gumbel_anneal_steps: int = 10000

    # Ground-truth thresholds
    min_group_size: int = 10
    income_margin: float = 0.10

    # Runtime defaults
    batch_size: int = 16
    device: str = "cuda"
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Loss weights
    lambda_primitive: float = 1.0
    lambda_generation: float = 1.0
    lambda_policy: float = 0.5
    lambda_leakage: float = 1.0
    lambda_abstention: float = 0.3

    # Smoke-test dataset sizes
    num_train_queries: int = 64
    num_val_queries: int = 32
    num_test_queries: int = 32
    data_split_seed: int = 0
    train_loop_num_train_queries: int = 512
    train_loop_num_val_queries: int = 128
    train_loop_num_test_queries: int = 32
    train_loop_epochs: int = 12
    train_loop_patience: int = 4
    train_loop_seed: int = 123

    # Trace collection / probe model
    trace_dir: str = "trace_cache"
    probe_num_answer_classes: int = 3
    probe_num_layers: int = 6
    probe_batch_size: int = 64
    probe_train_examples: int = 1280
    probe_val_examples: int = 320
    probe_test_examples: int = 320
    probe_pool_multiplier: int = 8
    probe_learning_rate: float = 3e-4
    probe_weight_decay: float = 0.01
    probe_epochs: int = 10
    probe_patience: int = 3
    probe_mc_passes: int = 8
    trace_low_agreement_threshold: float = 0.55
    trace_low_margin_threshold: float = 0.15
    primitive_trajectory_loss_weight: float = 1.0
    primitive_pattern_loss_weight: float = 1.0
    primitive_confidence_loss_weight: float = 1.0
    primitive_outcome_loss_weight: float = 1.0
    trace_stage1_epochs: int = 25
    trace_stage2_epochs: int = 8
    trace_stage1_patience: int = 6
    trace_stage1_selection_tolerance: float = 0.01
    trace_primary_metric: str = "primary_score"
    trace_class_weight_power: float = 0.5
    trace_class_weight_clip: float = 4.0
    trace_use_weighted_sampler: bool = False
    trace_head_hidden_dim: int = 64
    trace_hard_case_pool_multiplier: int = 4
    trace_train_hard_fraction: float = 0.30
    trace_failure_target_share: float = 0.12
    trace_outcome_failure_weight: float = 4.0
    trace_primary_metric_failure_weight: float = 0.30
    trace_pattern_z_candidates: List[float] = field(
        default_factory=lambda: [0.75, 0.60, 0.50]
    )
    trace_pattern_min_share: float = 0.10
    trace_pattern_threshold_override: Optional[float] = None
    baseline_mlp_max_iter: int = 2000
    baseline_mlp_early_stopping: bool = True

    @property
    def executor_feature_dim(self) -> int:
        return self.trajectory_len + 4 + 3 + 2

    @property
    def total_primitive_classes(self) -> int:
        return (
            self.num_trajectory_classes
            + self.num_pattern_classes
            + self.num_confidence_classes
            + self.num_outcome_classes
        )

    @property
    def max_bits_per_query(self) -> float:
        return math.log2(
            self.num_trajectory_classes
            * self.num_pattern_classes
            * self.num_confidence_classes
            * self.num_outcome_classes
        )

    @classmethod
    def from_yaml(cls, path: str, base_path: Optional[str] = None) -> "Config":
        """Load config from YAML. If base_path is provided, load base first."""
        base_values = {}
        if base_path:
            with open(base_path) as f:
                base_values = yaml.safe_load(f) or {}

        with open(path) as f:
            overrides = yaml.safe_load(f) or {}

        merged = {**base_values, **overrides}
        return cls(**{k: v for k, v in merged.items() if hasattr(cls, k)})
