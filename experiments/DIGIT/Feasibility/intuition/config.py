"""Configuration loader for Intuition-Labs."""

import yaml
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Model dimensions
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1

    # Vocabulary / sequences
    max_query_len: int = 64
    max_output_len: int = 48
    vocab_size: int = 2000

    # Bottleneck
    gumbel_tau_start: float = 2.0
    gumbel_tau_end: float = 0.5
    gumbel_anneal_steps: int = 10000

    # Safe executor
    min_group_size: int = 5
    variance_buckets: int = 4
    confidence_buckets: int = 3

    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    num_epochs: int = 50
    warmup_steps: int = 500
    grad_clip: float = 1.0

    # Loss weights
    lambda_primitive: float = 1.0
    lambda_generation: float = 1.0
    lambda_policy: float = 0.5
    lambda_leakage: float = 1.0
    lambda_abstention: float = 0.3

    # Data
    num_train_samples: int = 10000
    num_val_samples: int = 1000
    num_private_records: int = 500
    num_query_fields: int = 8

    # Device
    device: str = "cuda"
    seed: int = 42

    # Derived: number of categories per primitive
    num_answer_classes: int = 6   # YES, NO, MAYBE, INSUFFICIENT_EVIDENCE, INCONSISTENT_SIGNAL, POLICY_BLOCKED
    num_support_classes: int = 4  # VERY_LOW, LOW, MEDIUM, HIGH
    num_confidence_classes: int = 3  # LOW, MEDIUM, HIGH
    num_risk_classes: int = 3     # LOW, MEDIUM, HIGH

    # Safe executor output dimension (computed from buckets + flags)
    @property
    def executor_feature_dim(self) -> int:
        # support_ratio(1) + variance_bucket(var_buckets) + confidence_bucket(conf_buckets)
        # + min_group_pass(1) + policy_flags(3)
        return 1 + self.variance_buckets + self.confidence_buckets + 1 + 3

    @property
    def total_primitive_classes(self) -> int:
        return (self.num_answer_classes + self.num_support_classes
                + self.num_confidence_classes + self.num_risk_classes)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
