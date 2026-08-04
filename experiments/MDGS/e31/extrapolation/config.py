"""Configuration for the E31 cooperative path/witness uncertainty experiment."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class E31Config:
    """Standalone E31 config with only the fields used by the local experiment."""

    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1

    num_query_fields: int = 13
    field_vocab_sizes: list[int] = field(
        default_factory=lambda: [15, 17, 3, 6, 7, 9, 8, 6, 21, 3, 3, 7, 13]
    )
    max_output_len: int = 48

    experiment_name: str = "e31"
    trace_dir: str = "trace_cache"
    device: str = "cpu"
    use_amp: bool = False
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    train_loop_epochs: int = 12
    train_loop_patience: int = 4
    train_loop_seed: int = 123
    output_root: str = "experiments/results"

    trace_class_weight_power: float = 0.5
    trace_class_weight_clip: float = 4.0
    baseline_mlp_max_iter: int = 2000
    baseline_mlp_early_stopping: bool = True

    num_diffusion_steps: int = 8
    diffusion_beta_start: float = 0.02
    diffusion_beta_end: float = 0.20
    num_cloud_samples: int = 8
    diffusion_hidden_dim: int = 256
    diffusion_target_hidden_dim: int = 128
    diffusion_dropout: float = 0.0

    point_reader_mode: str = "identity"
    point_reader_hidden_dim: int = 256
    point_reader_dropout: float = 0.0

    outcome_loss_weight: float = 1.0
    diffusion_loss_weight: float = 1.0
    failure_class_weight_scale: float = 1.0

    enable_diagnostic_probes: bool = False
    log_pre_reader_geometry: bool = True
    log_grouped_geometry: bool = True
    log_weighting_diagnostics: bool = True
    witness_weight_temperature: float = 1.0
    use_vote_disagreement: bool = True
    use_geometric_disagreement: bool = True
    use_margin_term: bool = True
    use_coherence_weighting: bool = True
    gate_mode: str = "soft_commitment_gate"
    use_hard_eval_gate: bool = True
    primary_metric: str = "balanced_margin_primary"
    commitment_scale_init: float = 1.0
    tau_uncertain_init: float = 0.0

    uncertainty_alpha_init: float = 1.0
    uncertainty_beta_init: float = 1.0
    uncertainty_gamma_init: float = 1.0
    uncertainty_bias_init: float = 0.0

    enable_path_uncertainty: bool = True
    path_uncertainty_embedding_dim: int = 64
    path_uncertainty_hidden_dim: int = 128
    path_uncertainty_dropout: float = 0.0
    path_contrastive_temperature: float = 0.2
    path_consistency_loss_weight: float = 0.05
    use_path_disagreement_in_uncertainty: bool = False
    use_method_conflict_in_uncertainty: bool = False
    use_path_commitment_penalty: bool = False
    use_path_aware_witness_weights: bool = False
    path_uncertainty_delta_init: float = 1.0
    method_conflict_rho_init: float = 1.0
    path_commitment_lambda_init: float = 1.0
    path_weight_eta_init: float = 1.0

    enable_path_signed_evidence: bool = False
    use_path_signed_margin_in_hard_gate: bool = True
    path_signed_evidence_hidden_dim: int = 128
    path_signed_evidence_dropout: float = 0.0
    path_signed_logit_scale_init: float = 0.50
    path_signed_margin_scale_init: float = 0.75
    path_failure_loss_weight: float = 0.20
    path_failure_positive_weight_scale: float = 1.0
    path_failure_focal_gamma: float = 1.0
    path_failure_gate_threshold: float = 0.50
    path_outcome_loss_weight: float = 0.10


E30Config = E31Config
