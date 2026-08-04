"""Configuration dataclass for the DIGIT PoC experiment."""

from __future__ import annotations

import math
import yaml
from dataclasses import dataclass, field
from typing import List, Optional


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
    vocab_size: int = 2000

    # Bottleneck
    num_answer_classes: int = 6
    num_support_classes: int = 4
    num_confidence_classes: int = 3
    num_risk_classes: int = 3
    gumbel_tau_start: float = 2.0
    gumbel_tau_end: float = 0.5
    gumbel_anneal_steps: int = 10000

    # Safe executor
    min_group_size: int = 10
    variance_buckets: int = 4
    confidence_buckets: int = 3

    # Ground truth thresholds
    income_margin: float = 0.05
    support_thresholds: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.15])
    confidence_thresholds: List[int] = field(default_factory=lambda: [30, 200])
    risk_variance_thresholds: List[float] = field(default_factory=lambda: [0.20, 0.24])

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 80
    warmup_steps: int = 500
    grad_clip: float = 1.0
    patience: int = 15

    # Loss weights
    lambda_primitive: float = 1.0
    lambda_generation: float = 1.0
    lambda_policy: float = 0.5
    lambda_leakage: float = 1.0
    lambda_abstention: float = 0.3

    # Data
    num_train_queries: int = 8000
    num_val_queries: int = 2000
    num_test_queries: int = 2000

    # Experiment
    seeds: List[int] = field(default_factory=lambda: [42, 137, 256, 512, 1024])
    data_split_seed: int = 0
    device: str = "cuda"

    # DP baseline
    dp_epsilons: List[float] = field(
        default_factory=lambda: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    )

    @property
    def executor_feature_dim(self) -> int:
        return 1 + self.variance_buckets + self.confidence_buckets + 1 + 3

    @property
    def total_primitive_classes(self) -> int:
        return (
            self.num_answer_classes
            + self.num_support_classes
            + self.num_confidence_classes
            + self.num_risk_classes
        )

    @property
    def max_bits_per_query(self) -> float:
        return math.log2(
            self.num_answer_classes
            * self.num_support_classes
            * self.num_confidence_classes
            * self.num_risk_classes
        )

    @classmethod
    def from_yaml(cls, path: str, base_path: Optional[str] = None) -> Config:
        """Load config from YAML. If base_path is provided, load base first."""
        base_values = {}
        if base_path:
            with open(base_path) as f:
                base_values = yaml.safe_load(f) or {}

        with open(path) as f:
            overrides = yaml.safe_load(f) or {}

        merged = {**base_values, **overrides}
        return cls(**{k: v for k, v in merged.items() if hasattr(cls, k)})
