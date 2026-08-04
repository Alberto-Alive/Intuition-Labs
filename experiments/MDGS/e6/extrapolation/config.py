"""Configuration dataclass for the DIGIT Extrapolation E6 experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import yaml

from .executor_schema import EXECUTOR_FEATURE_DIM, TRACE_INPUT_DIM, UNCERTAINTY_FEATURE_DIM


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
    gumbel_tau_start: float = 1.2
    gumbel_tau_end: float = 0.5
    gumbel_anneal_steps: int = 10000

    # Ground-truth thresholds
    min_group_size: int = 10
    income_margin: float = 0.10

    # Runtime defaults
    batch_size: int = 8
    device: str = "cuda"
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    use_amp: bool = True

    # Loss weights
    lambda_primitive: float = 1.0
    lambda_generation: float = 1.0
    lambda_policy: float = 0.5
    lambda_leakage: float = 1.0
    lambda_abstention: float = 0.3
    lambda_evidence_targets: float = 0.8
    lambda_consistency: float = 0.25
    lambda_evidence_mono: float = 0.50
    lambda_usage: float = 1.20

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
    trace_dir: str = "../e3/trace_cache"
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
    primitive_outcome_label_smoothing: float = 0.0
    trace_stage1_epochs: int = 25
    trace_stage2_epochs: int = 8
    trace_stage1_head_warmup_epochs: int = 4
    trace_stage1_patience: int = 10
    trace_stage1_selection_tolerance: float = 0.01
    trace_stage1_frontier_size: int = 8
    trace_stage1_frontier_outcome_slack: float = 0.06
    trace_stage1_frontier_failure_slack: float = 0.10
    trace_primary_metric: str = "primary_score"
    trace_class_weight_power: float = 0.5
    trace_class_weight_clip: float = 4.0
    trace_use_weighted_sampler: bool = False
    trace_head_hidden_dim: int = 64
    uncertainty_tower_dim: int = 96
    anchor_state_dim: int = 48
    anchors_per_family: int = 4
    anchor_support_temperature: float = 0.25
    anchor_assignment_mode: str = "soft"
    anchor_use_approach: bool = True
    anchor_use_leap: bool = True
    anchor_use_boundary_family: bool = True
    anchor_use_barrier: bool = True
    anchor_pull_weight: float = 0.30
    anchor_repulsion_weight: float = 0.10
    anchor_usage_weight: float = 0.08
    anchor_approach_alignment_weight: float = 0.12
    anchor_family_target_weight: float = 0.25
    anchor_family_margin_weight: float = 0.15
    anchor_family_margin: float = 0.08
    cooperative_cert_order_weight: float = 0.20
    cooperative_cert_margin: float = 0.08
    cooperative_guard_target_weight: float = 0.15
    cooperative_commitment_target_weight: float = 0.12
    anchor_repulsion_margin: float = 0.75
    monotone_confidence_threshold_init: float = 0.70
    monotone_confidence_threshold_gap_init: float = 0.45
    monotone_risk_scale_init: float = 4.0
    trace_hard_case_pool_multiplier: int = 4
    trace_train_hard_fraction: float = 0.30
    trace_failure_target_share: float = 0.12
    trace_outcome_failure_weight: float = 8.0
    trace_primary_metric_failure_weight: float = 0.30
    trace_evidence_target_version: str = "v1"
    trace_usage_share_threshold: float = 0.95
    trace_selection_collapse_share_threshold: float = 0.90
    trace_selection_min_active_families: int = 2
    trace_selection_min_family_share: float = 0.03
    trace_selection_require_success_and_failure: bool = True
    trace_selection_min_commitment_std: float = 0.03
    trace_epoch_audit_interval: int = 5
    trace_success_prob_guard_penalty_multiplier: float = 2.5
    trace_success_prob_guard_margin: float = 0.05
    trace_barrier_warmup_epochs: int = 4
    trace_barrier_activation_min_families: int = 2
    trace_barrier_activation_min_family_share: float = 0.05
    trace_barrier_activation_max_family_share: float = 0.90
    trace_barrier_activation_require_success_and_failure: bool = True
    trace_barrier_activation_min_commitment_std: float = 0.03
    trace_force_stage1_only: bool = False
    trace_batch_size_override: Optional[int] = None
    trace_path_scale_shared_trace: float = 1.0
    trace_path_scale_success_trace: float = 1.0
    trace_path_scale_success_core: float = 1.0
    trace_path_scale_success_residual: float = 1.0
    trace_path_scale_success_base: float = 1.0
    trace_path_scale_success_bonus: float = 1.0
    trace_path_scale_failure_trace: float = 1.0
    trace_path_scale_ambiguity_trace: float = 1.0
    trace_path_scale_cert_context: float = 1.0
    trace_path_scale_outcome_context: float = 0.0
    trace_path_scale_outcome_trajectory_context: float = 0.0
    trace_path_scale_outcome_pattern_context: float = 0.80
    trace_path_scale_success_raw: float = 0.50
    trace_path_scale_failure_raw: float = 1.0
    trace_path_scale_ambiguity_raw: float = 1.0
    trace_correctness_cv_folds: int = 5
    trace_success_correctness_threshold: float = 0.90
    trace_failure_correctness_threshold: float = 0.60
    trace_success_min_support_ratio: float = 0.01
    trace_success_min_group_size: int = 30
    trace_pattern_z_candidates: List[float] = field(
        default_factory=lambda: [0.75, 0.60, 0.50]
    )
    trace_pattern_min_share: float = 0.10
    trace_pattern_threshold_override: Optional[float] = None
    baseline_mlp_max_iter: int = 2000
    baseline_mlp_early_stopping: bool = True

    # ── Diffusion bottleneck (E6) ─────────────────────────────────────────────
    # Number of forward-diffusion / denoising steps per path.
    num_diffusion_steps: int = 8
    # Random paths used to measure latent-basin stability (certainty).
    num_random_paths: int = 6
    # Linear noise schedule endpoints.
    diffusion_beta_start: float = 0.02
    diffusion_beta_end: float = 0.20
    # Hidden dim of the LatentDenoiser MLP.
    diffusion_denoiser_hidden: int = 256
    # Hidden dim of the DiffusionTrustHead MLP.
    diffusion_trust_hidden: int = 128
    # Loss weight for the denoising MSE objective.
    lambda_diffusion_denoise: float = 1.0

    @property
    def executor_feature_dim(self) -> int:
        return EXECUTOR_FEATURE_DIM

    @property
    def trace_input_dim(self) -> int:
        return TRACE_INPUT_DIM

    @property
    def uncertainty_feature_dim(self) -> int:
        return UNCERTAINTY_FEATURE_DIM

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
