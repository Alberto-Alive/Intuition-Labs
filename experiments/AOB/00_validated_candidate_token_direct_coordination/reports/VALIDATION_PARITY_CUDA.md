# Activation-Aware Coordination Report

## Protocol

- Train labels are used only to fit supervised coordinators.
- Dev labels are used for early stopping and method inspection.
- Test predictions are computed only after all methods for each seed are fit.
- Coordinators receive visible candidate outputs and, for activation-aware methods, agent telemetry.
- The deterministic synthetic task labels are never given to coordinators at prediction time.

## Run Metadata

```json
{
  "agent_config": {
    "activation_noise": 0.85,
    "activation_signal_strength": 1.25,
    "agent_quality_signal_strength": 0.25,
    "agent_signal_strength": 1.0,
    "confidence_noise": 0.28,
    "hidden_dim": 16,
    "hidden_label_signal_strength": 0.0,
    "n_agents": 5,
    "num_layers": 3,
    "reasoning_mode_signal_strength": 0.35,
    "stuckness_signal_strength": 0.25,
    "task_signal_strength": 1.0
  },
  "config": {
    "agents": {
      "activation_noise": 0.85,
      "activation_signal_strength": 1.25,
      "agent_quality_signal_strength": 0.25,
      "agent_signal_strength": 1.0,
      "confidence_noise": 0.28,
      "hidden_dim": 16,
      "hidden_label_signal_strength": 0.0,
      "n_agents": 5,
      "num_layers": 3,
      "reasoning_mode_signal_strength": 0.35,
      "stuckness_signal_strength": 0.25,
      "task_signal_strength": 1.0
    },
    "controls": [
      "none"
    ],
    "cuda_memory_budget_gb": 16.0,
    "dataset": {
      "feature_mod": 997,
      "n_dev": 80,
      "n_test": 80,
      "n_train": 240,
      "num_classes": 4,
      "num_tasks": 4
    },
    "deterministic": true,
    "device": "cuda",
    "label_reduced_agents": {
      "agent_quality_signal_strength": 0.35,
      "agent_signal_strength": 1.0,
      "hidden_label_signal_strength": 0.0,
      "reasoning_mode_signal_strength": 0.45,
      "stuckness_signal_strength": 0.35,
      "task_signal_strength": 0.0
    },
    "output_path": "results/validation_parity_cuda.json",
    "report_path": "reports/VALIDATION_PARITY_CUDA.md",
    "run_diagnostics": true,
    "run_label_signal_ablation": true,
    "run_randomized_label_leakage": false,
    "run_strict_coordination": true,
    "run_telemetry_channel_ablations": false,
    "seeds": [
      0
    ],
    "strict_agents": {
      "activation_noise": 0.65,
      "confidence_noise": 0.2,
      "evidence_signal_strength": 1.35,
      "hidden_dim": 16,
      "n_agents": 5,
      "num_layers": 3,
      "reasoning_mode_signal_strength": 0.35,
      "uncertainty_signal_strength": 0.45
    },
    "strict_dataset": {
      "n_dev": 80,
      "n_evidence_bits": 4,
      "n_test": 80,
      "n_train": 240,
      "num_classes": 4,
      "num_tasks": 1
    },
    "training": {
      "batch_size": 128,
      "clusters": 6,
      "epochs": 8,
      "hidden_dims": [
        48,
        24
      ],
      "lr": 0.003,
      "patience": 4,
      "pca_components": 8,
      "weight_decay": 0.0001
    }
  },
  "config_path": "configs\\validation_parity_cuda.json",
  "dataset_config": {
    "feature_mod": 997,
    "n_dev": 80,
    "n_test": 80,
    "n_train": 240,
    "num_classes": 4,
    "num_tasks": 4
  },
  "device_used": "cuda",
  "hardware": {
    "cuda_available": true,
    "cuda_device": "NVIDIA GeForce RTX 5070 Ti",
    "cuda_memory_budget_gb": 16.0,
    "cuda_memory_fraction": 1.0,
    "cuda_total_gb": 15.92047119140625,
    "deterministic": true,
    "torch_version": "2.11.0+cu128"
  },
  "label_reduced_agent_config": {
    "activation_noise": 0.85,
    "activation_signal_strength": 1.25,
    "agent_quality_signal_strength": 0.35,
    "agent_signal_strength": 1.0,
    "confidence_noise": 0.28,
    "hidden_dim": 16,
    "hidden_label_signal_strength": 0.0,
    "n_agents": 5,
    "num_layers": 3,
    "reasoning_mode_signal_strength": 0.45,
    "stuckness_signal_strength": 0.35,
    "task_signal_strength": 0.0
  },
  "strict_agent_config": {
    "activation_noise": 0.65,
    "confidence_noise": 0.2,
    "evidence_signal_strength": 1.35,
    "hidden_dim": 16,
    "n_agents": 5,
    "num_layers": 3,
    "reasoning_mode_signal_strength": 0.35,
    "uncertainty_signal_strength": 0.45
  },
  "strict_dataset_config": {
    "n_dev": 80,
    "n_evidence_bits": 4,
    "n_test": 80,
    "n_train": 240,
    "num_classes": 4,
    "num_tasks": 1
  },
  "training_config": {
    "batch_size": 128,
    "epochs": 8,
    "hidden_dims": [
      48,
      24
    ],
    "lr": 0.003,
    "patience": 4,
    "weight_decay": 0.0001
  }
}
```

## Aggregate Accuracy

| benchmark | split | condition | method | mean acc | std | runs | params |
|---|---|---|---|---:|---:|---:|---:|
| stage0_label_reduced | dev | none | activation_cluster_router | 0.8875 | 0.0000 | 1 | 0 |
| stage0_label_reduced | dev | none | activation_pca_mlp | 0.4875 | 0.0000 | 1 | 4444 |
| stage0_label_reduced | dev | none | activation_pool_mlp | 0.4750 | 0.0000 | 1 | 7900 |
| stage0_label_reduced | dev | none | capacity_matched_text_only | 0.5875 | 0.0000 | 1 | 7900 |
| stage0_label_reduced | dev | none | independent_swarm | 0.5125 | 0.0000 | 1 | 0 |
| stage0_label_reduced | dev | none | output_only_oracle_probe | 0.5125 | 0.0000 | 1 | 4444 |
| stage0_label_reduced | dev | none | single_agent | 0.4750 | 0.0000 | 1 | 0 |
| stage0_label_reduced | dev | none | telemetry_only_label_probe | 0.2125 | 0.0000 | 1 | 1708 |
| stage0_label_reduced | dev | none | text_only_coordinator | 0.5625 | 0.0000 | 1 | 4060 |
| stage0_label_reduced | test | none | activation_cluster_router | 0.9250 | 0.0000 | 1 | 0 |
| stage0_label_reduced | test | none | activation_pca_mlp | 0.6125 | 0.0000 | 1 | 4444 |
| stage0_label_reduced | test | none | activation_pool_mlp | 0.6125 | 0.0000 | 1 | 7900 |
| stage0_label_reduced | test | none | capacity_matched_text_only | 0.6125 | 0.0000 | 1 | 7900 |
| stage0_label_reduced | test | none | independent_swarm | 0.5750 | 0.0000 | 1 | 0 |
| stage0_label_reduced | test | none | output_only_oracle_probe | 0.6125 | 0.0000 | 1 | 4444 |
| stage0_label_reduced | test | none | single_agent | 0.5500 | 0.0000 | 1 | 0 |
| stage0_label_reduced | test | none | telemetry_only_label_probe | 0.2875 | 0.0000 | 1 | 1708 |
| stage0_label_reduced | test | none | text_only_coordinator | 0.7000 | 0.0000 | 1 | 4060 |
| stage0_original | dev | none | activation_cluster_router | 0.7625 | 0.0000 | 1 | 0 |
| stage0_original | dev | none | activation_pca_mlp | 0.4125 | 0.0000 | 1 | 4444 |
| stage0_original | dev | none | activation_pool_mlp | 0.4125 | 0.0000 | 1 | 7900 |
| stage0_original | dev | none | capacity_matched_text_only | 0.5500 | 0.0000 | 1 | 7900 |
| stage0_original | dev | none | independent_swarm | 0.5125 | 0.0000 | 1 | 0 |
| stage0_original | dev | none | output_only_oracle_probe | 0.5500 | 0.0000 | 1 | 4444 |
| stage0_original | dev | none | single_agent | 0.4750 | 0.0000 | 1 | 0 |
| stage0_original | dev | none | telemetry_only_label_probe | 0.2875 | 0.0000 | 1 | 1708 |
| stage0_original | dev | none | text_only_coordinator | 0.5875 | 0.0000 | 1 | 4060 |
| stage0_original | test | none | activation_cluster_router | 0.7625 | 0.0000 | 1 | 0 |
| stage0_original | test | none | activation_pca_mlp | 0.5125 | 0.0000 | 1 | 4444 |
| stage0_original | test | none | activation_pool_mlp | 0.5250 | 0.0000 | 1 | 7900 |
| stage0_original | test | none | capacity_matched_text_only | 0.6000 | 0.0000 | 1 | 7900 |
| stage0_original | test | none | independent_swarm | 0.5250 | 0.0000 | 1 | 0 |
| stage0_original | test | none | output_only_oracle_probe | 0.5125 | 0.0000 | 1 | 4444 |
| stage0_original | test | none | single_agent | 0.4875 | 0.0000 | 1 | 0 |
| stage0_original | test | none | telemetry_only_label_probe | 0.3125 | 0.0000 | 1 | 1708 |
| stage0_original | test | none | text_only_coordinator | 0.5125 | 0.0000 | 1 | 4060 |
| strict_coordination | dev | none | activation_cluster_router | 0.1750 | 0.0000 | 1 | 0 |
| strict_coordination | dev | none | activation_pca_mlp | 0.4500 | 0.0000 | 1 | 4300 |
| strict_coordination | dev | none | activation_pool_mlp | 0.3750 | 0.0000 | 1 | 7756 |
| strict_coordination | dev | none | capacity_matched_text_only | 0.3000 | 0.0000 | 1 | 7756 |
| strict_coordination | dev | none | independent_swarm | 0.1625 | 0.0000 | 1 | 0 |
| strict_coordination | dev | none | output_only_oracle_probe | 0.3375 | 0.0000 | 1 | 4300 |
| strict_coordination | dev | none | single_agent | 0.2875 | 0.0000 | 1 | 0 |
| strict_coordination | dev | none | telemetry_only_label_probe | 0.5875 | 0.0000 | 1 | 1708 |
| strict_coordination | dev | none | text_only_coordinator | 0.3125 | 0.0000 | 1 | 3916 |
| strict_coordination | test | none | activation_cluster_router | 0.2250 | 0.0000 | 1 | 0 |
| strict_coordination | test | none | activation_pca_mlp | 0.4375 | 0.0000 | 1 | 4300 |
| strict_coordination | test | none | activation_pool_mlp | 0.4000 | 0.0000 | 1 | 7756 |
| strict_coordination | test | none | capacity_matched_text_only | 0.2625 | 0.0000 | 1 | 7756 |
| strict_coordination | test | none | independent_swarm | 0.1500 | 0.0000 | 1 | 0 |
| strict_coordination | test | none | output_only_oracle_probe | 0.2625 | 0.0000 | 1 | 4300 |
| strict_coordination | test | none | single_agent | 0.2625 | 0.0000 | 1 | 0 |
| strict_coordination | test | none | telemetry_only_label_probe | 0.5000 | 0.0000 | 1 | 1708 |
| strict_coordination | test | none | text_only_coordinator | 0.2750 | 0.0000 | 1 | 3916 |

## Main Test Comparison

| benchmark | method | test mean acc | std | runs |
|---|---|---:|---:|---:|
| stage0_label_reduced | activation_cluster_router | 0.9250 | 0.0000 | 1 |
| stage0_label_reduced | text_only_coordinator | 0.7000 | 0.0000 | 1 |
| stage0_label_reduced | capacity_matched_text_only | 0.6125 | 0.0000 | 1 |
| stage0_label_reduced | activation_pool_mlp | 0.6125 | 0.0000 | 1 |
| stage0_label_reduced | activation_pca_mlp | 0.6125 | 0.0000 | 1 |
| stage0_label_reduced | output_only_oracle_probe | 0.6125 | 0.0000 | 1 |
| stage0_label_reduced | independent_swarm | 0.5750 | 0.0000 | 1 |
| stage0_label_reduced | single_agent | 0.5500 | 0.0000 | 1 |
| stage0_label_reduced | telemetry_only_label_probe | 0.2875 | 0.0000 | 1 |
| stage0_original | activation_cluster_router | 0.7625 | 0.0000 | 1 |
| stage0_original | capacity_matched_text_only | 0.6000 | 0.0000 | 1 |
| stage0_original | independent_swarm | 0.5250 | 0.0000 | 1 |
| stage0_original | activation_pool_mlp | 0.5250 | 0.0000 | 1 |
| stage0_original | text_only_coordinator | 0.5125 | 0.0000 | 1 |
| stage0_original | activation_pca_mlp | 0.5125 | 0.0000 | 1 |
| stage0_original | output_only_oracle_probe | 0.5125 | 0.0000 | 1 |
| stage0_original | single_agent | 0.4875 | 0.0000 | 1 |
| stage0_original | telemetry_only_label_probe | 0.3125 | 0.0000 | 1 |
| strict_coordination | telemetry_only_label_probe | 0.5000 | 0.0000 | 1 |
| strict_coordination | activation_pca_mlp | 0.4375 | 0.0000 | 1 |
| strict_coordination | activation_pool_mlp | 0.4000 | 0.0000 | 1 |
| strict_coordination | text_only_coordinator | 0.2750 | 0.0000 | 1 |
| strict_coordination | single_agent | 0.2625 | 0.0000 | 1 |
| strict_coordination | capacity_matched_text_only | 0.2625 | 0.0000 | 1 |
| strict_coordination | output_only_oracle_probe | 0.2625 | 0.0000 | 1 |
| strict_coordination | activation_cluster_router | 0.2250 | 0.0000 | 1 |
| strict_coordination | independent_swarm | 0.1500 | 0.0000 | 1 |

## Confidence Intervals

| benchmark | quantity | mean | 95% bootstrap CI | n |
|---|---|---:|---|---:|
| stage0_label_reduced | activation_pca_mlp accuracy | 0.6125 | [0.6125, 0.6125] | 1 |
| stage0_label_reduced | output_only_oracle_probe accuracy | 0.6125 | [0.6125, 0.6125] | 1 |
| stage0_label_reduced | telemetry_only_label_probe accuracy | 0.2875 | [0.2875, 0.2875] | 1 |
| stage0_original | activation_pca_mlp accuracy | 0.5125 | [0.5125, 0.5125] | 1 |
| stage0_original | output_only_oracle_probe accuracy | 0.5125 | [0.5125, 0.5125] | 1 |
| stage0_original | telemetry_only_label_probe accuracy | 0.3125 | [0.3125, 0.3125] | 1 |
| strict_coordination | activation_pca_mlp accuracy | 0.4375 | [0.4375, 0.4375] | 1 |
| strict_coordination | output_only_oracle_probe accuracy | 0.2625 | [0.2625, 0.2625] | 1 |
| strict_coordination | telemetry_only_label_probe accuracy | 0.5000 | [0.5000, 0.5000] | 1 |
| stage0_label_reduced | correctness AUC activation_only | 0.9752 | [0.9552, 0.9870] | 5 |
| stage0_label_reduced | correctness AUC text_output_only | 0.6014 | [0.4954, 0.6878] | 5 |
| stage0_label_reduced | correctness AUC text_plus_activation | 0.9822 | [0.9764, 0.9880] | 5 |
| stage0_original | correctness AUC activation_only | 0.9684 | [0.9627, 0.9749] | 5 |
| stage0_original | correctness AUC text_output_only | 0.5658 | [0.5163, 0.6254] | 5 |
| stage0_original | correctness AUC text_plus_activation | 0.9706 | [0.9604, 0.9813] | 5 |
| strict_coordination | correctness AUC activation_only | 0.4765 | [0.4356, 0.5265] | 5 |
| strict_coordination | correctness AUC text_output_only | 0.5336 | [0.4898, 0.5716] | 5 |
| strict_coordination | correctness AUC text_plus_activation | 0.4765 | [0.4503, 0.5055] | 5 |

## Anti-Cheat Controls

| benchmark | condition | method | test mean acc | std | runs |
|---|---|---|---:|---:|---:|
| stage0_label_reduced | none | activation_cluster_router | 0.9250 | 0.0000 | 1 |
| stage0_label_reduced | none | activation_pca_mlp | 0.6125 | 0.0000 | 1 |
| stage0_label_reduced | none | activation_pool_mlp | 0.6125 | 0.0000 | 1 |
| stage0_original | none | activation_cluster_router | 0.7625 | 0.0000 | 1 |
| stage0_original | none | activation_pca_mlp | 0.5125 | 0.0000 | 1 |
| stage0_original | none | activation_pool_mlp | 0.5250 | 0.0000 | 1 |
| strict_coordination | none | activation_cluster_router | 0.2250 | 0.0000 | 1 |
| strict_coordination | none | activation_pca_mlp | 0.4375 | 0.0000 | 1 |
| strict_coordination | none | activation_pool_mlp | 0.4000 | 0.0000 | 1 |

## Randomized-Label Leakage Test

Randomized-label leakage test was disabled.

## Label Permutation Sanity

| benchmark | method | normal test acc | permuted-label test acc | drop |
|---|---|---:|---:|---:|

## Train/Test Split Audit

Test access violations before the test gate: `0`.

| benchmark | seed | train/dev overlap | train/test overlap | dev/test overlap |
|---|---:|---:|---:|---:|
| stage0_label_reduced | 0 | 0 | 0 | 0 |
| stage0_original | 0 | 0 | 0 | 0 |
| strict_coordination | 0 | 0 | 0 | 0 |

## Diagnostic Label Probes

| benchmark | method | test mean acc | std | runs | params |
|---|---|---:|---:|---:|---:|
| stage0_label_reduced | activation_pca_mlp | 0.6125 | 0.0000 | 1 | 4444 |
| stage0_label_reduced | capacity_matched_text_only | 0.6125 | 0.0000 | 1 | 7900 |
| stage0_label_reduced | output_only_oracle_probe | 0.6125 | 0.0000 | 1 | 4444 |
| stage0_label_reduced | telemetry_only_label_probe | 0.2875 | 0.0000 | 1 | 1708 |
| stage0_label_reduced | text_only_coordinator | 0.7000 | 0.0000 | 1 | 4060 |
| stage0_original | activation_pca_mlp | 0.5125 | 0.0000 | 1 | 4444 |
| stage0_original | capacity_matched_text_only | 0.6000 | 0.0000 | 1 | 7900 |
| stage0_original | output_only_oracle_probe | 0.5125 | 0.0000 | 1 | 4444 |
| stage0_original | telemetry_only_label_probe | 0.3125 | 0.0000 | 1 | 1708 |
| stage0_original | text_only_coordinator | 0.5125 | 0.0000 | 1 | 4060 |
| strict_coordination | activation_pca_mlp | 0.4375 | 0.0000 | 1 | 4300 |
| strict_coordination | capacity_matched_text_only | 0.2625 | 0.0000 | 1 | 7756 |
| strict_coordination | output_only_oracle_probe | 0.2625 | 0.0000 | 1 | 4300 |
| strict_coordination | telemetry_only_label_probe | 0.5000 | 0.0000 | 1 | 1708 |
| strict_coordination | text_only_coordinator | 0.2750 | 0.0000 | 1 | 3916 |

## Agent Correctness Probe

| benchmark | feature mode | mean acc | mean AUC | runs |
|---|---|---:|---:|---:|
| stage0_label_reduced | activation_only | 0.9100 | 0.9752 | 5 |
| stage0_label_reduced | text_output_only | 0.6225 | 0.6014 | 5 |
| stage0_label_reduced | text_plus_activation | 0.8975 | 0.9822 | 5 |
| stage0_original | activation_only | 0.8825 | 0.9684 | 5 |
| stage0_original | text_output_only | 0.5375 | 0.5658 | 5 |
| stage0_original | text_plus_activation | 0.8925 | 0.9706 | 5 |
| strict_coordination | activation_only | 0.7675 | 0.4765 | 5 |
| strict_coordination | text_output_only | 0.7675 | 0.5336 | 5 |
| strict_coordination | text_plus_activation | 0.7650 | 0.4765 | 5 |

## Redundancy Probe

| benchmark | metric | mean value | std | runs |
|---|---|---:|---:|---:|
| stage0_label_reduced | cluster_purity_by_answer_label | 0.3150 | 0.0000 | 1 |
| stage0_label_reduced | cluster_purity_by_correctness | 0.9275 | 0.0000 | 1 |
| stage0_original | cluster_purity_by_answer_label | 0.3300 | 0.0000 | 1 |
| stage0_original | cluster_purity_by_correctness | 0.8200 | 0.0000 | 1 |
| strict_coordination | cluster_purity_by_answer_label | 0.3050 | 0.0000 | 1 |
| strict_coordination | cluster_purity_by_correctness | 0.7675 | 0.0000 | 1 |

## Telemetry Channel Ablations

Telemetry channel ablations were not run.

## Probe Access Matrix

| artifact | visible_agent_answers | visible_confidence | visible_traces | hidden_activations | train_labels_for_fit | dev_labels_for_early_stopping | test_labels_or_eval_feedback |
|---|---|---|---|---|---|---|---|
| single_agent | True | False | False | False | False | False | False |
| independent_swarm | True | True | False | False | False | False | False |
| text_only_coordinator | True | True | False | False | True | True | False |
| output_only_oracle_probe | True | True | True | False | True | True | False |
| telemetry_only_label_probe | False | False | False | True | True | True | False |
| activation_pca_mlp | True | True | False | True | True | True | False |
| agent_correctness_probe | varies_by_feature_mode | varies_by_feature_mode | varies_by_feature_mode | varies_by_feature_mode | True | True | False |

## CPU/CUDA Parity

CPU/CUDA parity artifact not attached to this report.

## All Runs

| benchmark | seed | split | condition | method | accuracy | params |
|---|---:|---|---|---|---:|---:|
| stage0_label_reduced | 0 | dev | none | activation_cluster_router | 0.8875 | 0 |
| stage0_label_reduced | 0 | dev | none | activation_pca_mlp | 0.4875 | 4444 |
| stage0_label_reduced | 0 | dev | none | activation_pool_mlp | 0.4750 | 7900 |
| stage0_label_reduced | 0 | dev | none | capacity_matched_text_only | 0.5875 | 7900 |
| stage0_label_reduced | 0 | dev | none | independent_swarm | 0.5125 | 0 |
| stage0_label_reduced | 0 | dev | none | output_only_oracle_probe | 0.5125 | 4444 |
| stage0_label_reduced | 0 | dev | none | single_agent | 0.4750 | 0 |
| stage0_label_reduced | 0 | dev | none | telemetry_only_label_probe | 0.2125 | 1708 |
| stage0_label_reduced | 0 | dev | none | text_only_coordinator | 0.5625 | 4060 |
| stage0_label_reduced | 0 | test | none | activation_cluster_router | 0.9250 | 0 |
| stage0_label_reduced | 0 | test | none | activation_pca_mlp | 0.6125 | 4444 |
| stage0_label_reduced | 0 | test | none | activation_pool_mlp | 0.6125 | 7900 |
| stage0_label_reduced | 0 | test | none | capacity_matched_text_only | 0.6125 | 7900 |
| stage0_label_reduced | 0 | test | none | independent_swarm | 0.5750 | 0 |
| stage0_label_reduced | 0 | test | none | output_only_oracle_probe | 0.6125 | 4444 |
| stage0_label_reduced | 0 | test | none | single_agent | 0.5500 | 0 |
| stage0_label_reduced | 0 | test | none | telemetry_only_label_probe | 0.2875 | 1708 |
| stage0_label_reduced | 0 | test | none | text_only_coordinator | 0.7000 | 4060 |
| stage0_original | 0 | dev | none | activation_cluster_router | 0.7625 | 0 |
| stage0_original | 0 | dev | none | activation_pca_mlp | 0.4125 | 4444 |
| stage0_original | 0 | dev | none | activation_pool_mlp | 0.4125 | 7900 |
| stage0_original | 0 | dev | none | capacity_matched_text_only | 0.5500 | 7900 |
| stage0_original | 0 | dev | none | independent_swarm | 0.5125 | 0 |
| stage0_original | 0 | dev | none | output_only_oracle_probe | 0.5500 | 4444 |
| stage0_original | 0 | dev | none | single_agent | 0.4750 | 0 |
| stage0_original | 0 | dev | none | telemetry_only_label_probe | 0.2875 | 1708 |
| stage0_original | 0 | dev | none | text_only_coordinator | 0.5875 | 4060 |
| stage0_original | 0 | test | none | activation_cluster_router | 0.7625 | 0 |
| stage0_original | 0 | test | none | activation_pca_mlp | 0.5125 | 4444 |
| stage0_original | 0 | test | none | activation_pool_mlp | 0.5250 | 7900 |
| stage0_original | 0 | test | none | capacity_matched_text_only | 0.6000 | 7900 |
| stage0_original | 0 | test | none | independent_swarm | 0.5250 | 0 |
| stage0_original | 0 | test | none | output_only_oracle_probe | 0.5125 | 4444 |
| stage0_original | 0 | test | none | single_agent | 0.4875 | 0 |
| stage0_original | 0 | test | none | telemetry_only_label_probe | 0.3125 | 1708 |
| stage0_original | 0 | test | none | text_only_coordinator | 0.5125 | 4060 |
| strict_coordination | 0 | dev | none | activation_cluster_router | 0.1750 | 0 |
| strict_coordination | 0 | dev | none | activation_pca_mlp | 0.4500 | 4300 |
| strict_coordination | 0 | dev | none | activation_pool_mlp | 0.3750 | 7756 |
| strict_coordination | 0 | dev | none | capacity_matched_text_only | 0.3000 | 7756 |
| strict_coordination | 0 | dev | none | independent_swarm | 0.1625 | 0 |
| strict_coordination | 0 | dev | none | output_only_oracle_probe | 0.3375 | 4300 |
| strict_coordination | 0 | dev | none | single_agent | 0.2875 | 0 |
| strict_coordination | 0 | dev | none | telemetry_only_label_probe | 0.5875 | 1708 |
| strict_coordination | 0 | dev | none | text_only_coordinator | 0.3125 | 3916 |
| strict_coordination | 0 | test | none | activation_cluster_router | 0.2250 | 0 |
| strict_coordination | 0 | test | none | activation_pca_mlp | 0.4375 | 4300 |
| strict_coordination | 0 | test | none | activation_pool_mlp | 0.4000 | 7756 |
| strict_coordination | 0 | test | none | capacity_matched_text_only | 0.2625 | 7756 |
| strict_coordination | 0 | test | none | independent_swarm | 0.1500 | 0 |
| strict_coordination | 0 | test | none | output_only_oracle_probe | 0.2625 | 4300 |
| strict_coordination | 0 | test | none | single_agent | 0.2625 | 0 |
| strict_coordination | 0 | test | none | telemetry_only_label_probe | 0.5000 | 1708 |
| strict_coordination | 0 | test | none | text_only_coordinator | 0.2750 | 3916 |

## Mechanism Distinction

- Stage 0 mechanism: activation telemetry is most consistent with a reliability/correctness signal. Activation-only correctness AUC is 0.9684 on `stage0_original` and 0.9752 on `stage0_label_reduced`, while Stage 0 telemetry-only final-label accuracy is 0.3125.
- Strict coordination mechanism: the task is constructed around distributed private evidence. Telemetry-only final-label accuracy is 0.5000, while activation-only correctness AUC is 0.4765; this supports an evidence-combination mechanism, not correctness estimation.
- These are synthetic mechanisms. They validate the harness and controls, but they should not be generalized to real transformer agents without repeating the same diagnostics on captured model activations.

## Interpretation

- `stage0_label_reduced`: `activation_pca_mlp` scored 0.6125; strongest output-only reference scored 0.7000; delta -0.0875.
- `stage0_label_reduced`: telemetry-only label accuracy was 0.2875, 0.3250 below activation+output; this argues against simple final-label decoding.
- `stage0_original`: `activation_pca_mlp` scored 0.5125; strongest output-only reference scored 0.6000; delta -0.0875.
- `stage0_original`: telemetry-only label accuracy was 0.3125, 0.2000 below activation+output; this argues against simple final-label decoding.
- `strict_coordination`: `activation_pca_mlp` scored 0.4375; strongest output-only reference scored 0.2750; delta 0.1625.
- `strict_coordination`: telemetry-only label accuracy was 0.5000; in this benchmark the combined telemetry contains distributed private evidence. This should not be interpreted as correctness estimation.
- `stage0_label_reduced`: agent-correctness AUC text=0.6014, activation=0.9752, text+activation=0.9822.
- `stage0_original`: agent-correctness AUC text=0.5658, activation=0.9684, text+activation=0.9706.
- `strict_coordination`: agent-correctness AUC text=0.5336, activation=0.4765, text+activation=0.4765.
- `stage0_label_reduced`: activation-cluster purity by answer label=0.3150, by correctness=0.9275; low answer purity argues against simple answer-label clustering.
- `stage0_original`: activation-cluster purity by answer label=0.3300, by correctness=0.8200; low answer purity argues against simple answer-label clustering.
- `strict_coordination`: activation-cluster purity by answer label=0.3050, by correctness=0.7675; low answer purity argues against simple answer-label clustering.
- Recommended next experiment: port the strict partial-evidence benchmark to a small local transformer with hidden-state hooks and preserve the same validation diagnostics.
