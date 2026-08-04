"""Configuration for the E29 strict uncertainty-geometry experiment."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class E29Config:
    """Standalone E29 config with only the fields used by the local experiment."""

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

    experiment_name: str = "e29"
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
    output_root: str = "experiments/DIGIT/Extrapolation/results/e29"

    trace_class_weight_power: float = 0.5
    trace_class_weight_clip: float = 4.0
    baseline_mlp_max_iter: int = 2000
    baseline_mlp_early_stopping: bool = True

    num_diffusion_steps: int = 8
    diffusion_beta_start: float = 0.02
    diffusion_beta_end: float = 0.20
    cloud_num_samples: int = 8
    diffusion_hidden_dim: int = 256
    trace_target_hidden_dim: int = 128
    diffusion_dropout: float = 0.0

    outcome_loss_weight: float = 1.0
    diffusion_loss_weight: float = 1.0
    probe_loss_weight: float = 0.0

    enable_diagnostic_probes: bool = False
    geometry_use_radius_term: bool = True
    geometry_use_variance_term: bool = True
    geometry_use_margin_term: bool = True

    uncertainty_alpha_init: float = 1.0
    uncertainty_beta_init: float = 1.0
    uncertainty_gamma_init: float = 1.0
    uncertainty_bias_init: float = 0.0
