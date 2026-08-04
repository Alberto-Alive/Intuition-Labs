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
  "config": {
    "agent_view_log_path": "results/transformer_strict_parity_cpu_agent_views.jsonl",
    "controls": [
      "none"
    ],
    "coordinator_visible_log_path": "results/transformer_strict_parity_cpu_visible.jsonl",
    "cuda_memory_budget_gb": 16.0,
    "deterministic": true,
    "device": "cpu",
    "hidden_state_npz_path": "results/transformer_strict_parity_cpu_hidden_states.npz",
    "hidden_state_probe_name": "hidden_state_only_probe",
    "output_path": "results/transformer_strict_parity_cpu.json",
    "report_path": "reports/TRANSFORMER_STRICT_PARITY_CPU.md",
    "run_diagnostics": true,
    "run_individual_hidden_label_probe": false,
    "run_layer_position_ablations": false,
    "run_prompt_evidence_controls": false,
    "run_randomized_label_leakage": false,
    "run_stage1_mechanism_diagnostics": false,
    "run_telemetry_channel_ablations": false,
    "seeds": [
      0
    ],
    "strict_dataset": {
      "n_dev": 80,
      "n_evidence_bits": 4,
      "n_test": 120,
      "n_train": 160,
      "num_classes": 4,
      "num_tasks": 1
    },
    "training": {
      "batch_size": 64,
      "clusters": 4,
      "epochs": 5,
      "hidden_dims": [
        32,
        16
      ],
      "lr": 0.003,
      "patience": 3,
      "pca_components": 8,
      "weight_decay": 0.0001
    },
    "transformer": {
      "batch_size": 16,
      "layers": [
        1,
        5
      ],
      "max_length": 96,
      "model_name_or_path": "EleutherAI/pythia-70m-deduped",
      "token_positions": [
        "evidence",
        "final",
        "prompt_mean"
      ]
    }
  },
  "config_path": "configs\\transformer_strict_parity_cpu.json",
  "device_used": "cpu",
  "hardware": {
    "cuda_available": true,
    "deterministic": true,
    "fallback_reason": "cuda unavailable or not requested",
    "torch_version": "2.11.0+cu128"
  },
  "stage": "stage1_transformer_strict",
  "strict_dataset_config": {
    "n_dev": 80,
    "n_evidence_bits": 4,
    "n_test": 120,
    "n_train": 160,
    "num_classes": 4,
    "num_tasks": 1
  },
  "training_config": {
    "batch_size": 64,
    "epochs": 5,
    "hidden_dims": [
      32,
      16
    ],
    "lr": 0.003,
    "patience": 3,
    "weight_decay": 0.0001
  },
  "transformer_config": {
    "answer_format": "default",
    "batch_size": 16,
    "device": "cpu",
    "evidence_tokens": [
      "BIT_ZERO",
      "BIT_ONE"
    ],
    "layers": [
      1,
      5
    ],
    "mask_token": "BIT_MASK",
    "max_length": 96,
    "model_name_or_path": "EleutherAI/pythia-70m-deduped",
    "prompt_template": "default",
    "token_positions": [
      "evidence",
      "final",
      "prompt_mean"
    ],
    "visible_evidence": false
  }
}
```

## Aggregate Accuracy

| benchmark | split | condition | method | mean acc | std | runs | params |
|---|---|---|---|---:|---:|---:|---:|
| transformer_strict | dev | none | activation_cluster_router | 0.2125 | 0.0000 | 1 | 0 |
| transformer_strict | dev | none | activation_pca_mlp | 0.3625 | 0.0000 | 1 | 2324 |
| transformer_strict | dev | none | activation_pool_mlp | 0.2875 | 0.0000 | 1 | 67604 |
| transformer_strict | dev | none | capacity_matched_text_only | 0.2750 | 0.0000 | 1 | 67604 |
| transformer_strict | dev | none | hidden_state_only_probe | 0.2875 | 0.0000 | 1 | 884 |
| transformer_strict | dev | none | independent_swarm | 0.2125 | 0.0000 | 1 | 0 |
| transformer_strict | dev | none | output_only_oracle_probe | 0.2875 | 0.0000 | 1 | 2324 |
| transformer_strict | dev | none | single_agent | 0.2125 | 0.0000 | 1 | 0 |
| transformer_strict | dev | none | text_only_coordinator | 0.2750 | 0.0000 | 1 | 2068 |
| transformer_strict | test | none | activation_cluster_router | 0.2167 | 0.0000 | 1 | 0 |
| transformer_strict | test | none | activation_pca_mlp | 0.3583 | 0.0000 | 1 | 2324 |
| transformer_strict | test | none | activation_pool_mlp | 0.2750 | 0.0000 | 1 | 67604 |
| transformer_strict | test | none | capacity_matched_text_only | 0.2833 | 0.0000 | 1 | 67604 |
| transformer_strict | test | none | hidden_state_only_probe | 0.2750 | 0.0000 | 1 | 884 |
| transformer_strict | test | none | independent_swarm | 0.2167 | 0.0000 | 1 | 0 |
| transformer_strict | test | none | output_only_oracle_probe | 0.2750 | 0.0000 | 1 | 2324 |
| transformer_strict | test | none | single_agent | 0.2167 | 0.0000 | 1 | 0 |
| transformer_strict | test | none | text_only_coordinator | 0.2833 | 0.0000 | 1 | 2068 |

## Main Test Comparison

| benchmark | method | test mean acc | std | runs |
|---|---|---:|---:|---:|
| transformer_strict | activation_pca_mlp | 0.3583 | 0.0000 | 1 |
| transformer_strict | text_only_coordinator | 0.2833 | 0.0000 | 1 |
| transformer_strict | capacity_matched_text_only | 0.2833 | 0.0000 | 1 |
| transformer_strict | activation_pool_mlp | 0.2750 | 0.0000 | 1 |
| transformer_strict | hidden_state_only_probe | 0.2750 | 0.0000 | 1 |
| transformer_strict | output_only_oracle_probe | 0.2750 | 0.0000 | 1 |
| transformer_strict | single_agent | 0.2167 | 0.0000 | 1 |
| transformer_strict | independent_swarm | 0.2167 | 0.0000 | 1 |
| transformer_strict | activation_cluster_router | 0.2167 | 0.0000 | 1 |

## Confidence Intervals

| benchmark | quantity | mean | 95% bootstrap CI | n |
|---|---|---:|---|---:|
| transformer_strict | activation_pca_mlp accuracy | 0.3583 | [0.3583, 0.3583] | 1 |
| transformer_strict | hidden_state_only_probe accuracy | 0.2750 | [0.2750, 0.2750] | 1 |
| transformer_strict | output_only_oracle_probe accuracy | 0.2750 | [0.2750, 0.2750] | 1 |
| transformer_strict | correctness AUC activation_only | 0.4986 | [0.4797, 0.5121] | 4 |
| transformer_strict | correctness AUC text_output_only | 0.5000 | [0.5000, 0.5000] | 4 |
| transformer_strict | correctness AUC text_plus_activation | 0.4986 | [0.4797, 0.5121] | 4 |

## Anti-Cheat Controls

| benchmark | condition | method | test mean acc | std | runs |
|---|---|---|---:|---:|---:|
| transformer_strict | none | activation_cluster_router | 0.2167 | 0.0000 | 1 |
| transformer_strict | none | activation_pca_mlp | 0.3583 | 0.0000 | 1 |
| transformer_strict | none | activation_pool_mlp | 0.2750 | 0.0000 | 1 |

## Randomized-Label Leakage Test

Randomized-label leakage test was disabled.

## Label Permutation Sanity

| benchmark | method | normal test acc | permuted-label test acc | drop |
|---|---|---:|---:|---:|

## Train/Test Split Audit

Test access violations before the test gate: `0`.

| benchmark | seed | train/dev overlap | train/test overlap | dev/test overlap |
|---|---:|---:|---:|---:|
| transformer_strict | 0 | 0 | 0 | 0 |

## Diagnostic Label Probes

| benchmark | method | test mean acc | std | runs | params |
|---|---|---:|---:|---:|---:|
| transformer_strict | activation_pca_mlp | 0.3583 | 0.0000 | 1 | 2324 |
| transformer_strict | capacity_matched_text_only | 0.2833 | 0.0000 | 1 | 67604 |
| transformer_strict | hidden_state_only_probe | 0.2750 | 0.0000 | 1 | 884 |
| transformer_strict | output_only_oracle_probe | 0.2750 | 0.0000 | 1 | 2324 |
| transformer_strict | text_only_coordinator | 0.2833 | 0.0000 | 1 | 2068 |

## Stage 1 Mechanism Validation

- Stage 1 supports distributed private-evidence recovery from real transformer hidden states in this controlled benchmark. It does not yet prove general open-ended real-agent coordination.
- Combined hidden-state-only final-label accuracy is 0.2750; strongest visible-output reference is 0.2833; output-only oracle final-label accuracy is 0.2750.
- `activation_pca_mlp` hidden+visible accuracy is 0.3583; interpret it alongside hidden-only and stability diagnostics because PCA/pooling variance remains material.

## Stage 3 Robustness

| method | test mean acc | std | 95% bootstrap CI | runs |
|---|---:|---:|---|---:|
| hidden_state_only_probe | 0.2750 | 0.0000 | [0.2750, 0.2750] | 1 |
| activation_pca_mlp | 0.3583 | 0.0000 | [0.3583, 0.3583] | 1 |
| output_only_oracle_probe | 0.2750 | 0.0000 | [0.2750, 0.2750] | 1 |
| text_only_coordinator | 0.2833 | 0.0000 | [0.2833, 0.2833] | 1 |

- Claim update: Stage 3 tests robustness of the controlled hidden-evidence mechanism across more seeds and variants; it still does not establish open-ended real-agent generalization.

## Per-Evidence-Bit Probes

Private-evidence-bit probes were not run.

## Agent-Count Curve

Agent-count curve was not run.

## Layer And Token Position Probes

Layer/token-position hidden-state probes were not run.

## Prompt Robustness

Prompt robustness diagnostics were not run.

## Model Robustness

Model robustness diagnostics were not run.

## Evidence Masking And Shuffling

Evidence masking/shuffling controls were not run.

## Output Leakage Audit

- Output-only oracle final-label accuracy is 0.2750.

## activation_pca_mlp Stability

activation_pca_mlp stability diagnostics were not run.

## Explicit Evidence-Sharing Baseline

Explicit evidence-sharing baseline was not run.

## Agent Correctness Probe

| benchmark | feature mode | mean acc | mean AUC | runs |
|---|---|---:|---:|---:|
| transformer_strict | activation_only | 0.7833 | 0.4986 | 4 |
| transformer_strict | text_output_only | 0.7833 | 0.5000 | 4 |
| transformer_strict | text_plus_activation | 0.7833 | 0.4986 | 4 |

## Redundancy Probe

| benchmark | metric | mean value | std | runs |
|---|---|---:|---:|---:|
| transformer_strict | cluster_purity_by_answer_label | 1.0000 | 0.0000 | 1 |
| transformer_strict | cluster_purity_by_correctness | 0.7833 | 0.0000 | 1 |

## Telemetry Channel Ablations

Telemetry channel ablations were not run.

## Transformer Hidden-State Ablations

Transformer hidden-state slice ablations were not run.

## Individual Hidden-State Label Probe

Individual hidden-state label probe was not run.

## Probe Access Matrix

| artifact | visible_agent_answers | visible_confidence | visible_traces | hidden_activations | private_prompt_evidence | coordinator_input | train_labels_for_fit | dev_labels_for_early_stopping | test_labels_or_eval_feedback |
|---|---|---|---|---|---|---|---|---|---|
| single_agent | True | False | False | False | False | False | False | False | False |
| independent_swarm | True | True | False | False | False | False | False | False | False |
| text_only_coordinator | True | True | False | False | False | False | True | True | False |
| output_only_oracle_probe | True | True | True | False | False | False | True | True | False |
| telemetry_only_label_probe | False | False | False | True | False | False | True | True | False |
| hidden_state_only_probe | False | False | False | True | False | False | True | True | False |
| activation_pca_mlp | True | True | False | True | False | False | True | True | False |
| agent_correctness_probe | varies_by_feature_mode | varies_by_feature_mode | varies_by_feature_mode | varies_by_feature_mode | False | False | True | True | False |
| transformer_private_agent_view_log | False | False | False | False | True | False | False | False | False |
| transformer_coordinator_visible_input_log | True | True | True | False | False | True | False | False | False |

## CPU/CUDA Parity

CPU/CUDA parity artifact not attached to this report.

## All Runs

| benchmark | seed | split | condition | method | accuracy | params |
|---|---:|---|---|---|---:|---:|
| transformer_strict | 0 | dev | none | activation_cluster_router | 0.2125 | 0 |
| transformer_strict | 0 | dev | none | activation_pca_mlp | 0.3625 | 2324 |
| transformer_strict | 0 | dev | none | activation_pool_mlp | 0.2875 | 67604 |
| transformer_strict | 0 | dev | none | capacity_matched_text_only | 0.2750 | 67604 |
| transformer_strict | 0 | dev | none | hidden_state_only_probe | 0.2875 | 884 |
| transformer_strict | 0 | dev | none | independent_swarm | 0.2125 | 0 |
| transformer_strict | 0 | dev | none | output_only_oracle_probe | 0.2875 | 2324 |
| transformer_strict | 0 | dev | none | single_agent | 0.2125 | 0 |
| transformer_strict | 0 | dev | none | text_only_coordinator | 0.2750 | 2068 |
| transformer_strict | 0 | test | none | activation_cluster_router | 0.2167 | 0 |
| transformer_strict | 0 | test | none | activation_pca_mlp | 0.3583 | 2324 |
| transformer_strict | 0 | test | none | activation_pool_mlp | 0.2750 | 67604 |
| transformer_strict | 0 | test | none | capacity_matched_text_only | 0.2833 | 67604 |
| transformer_strict | 0 | test | none | hidden_state_only_probe | 0.2750 | 884 |
| transformer_strict | 0 | test | none | independent_swarm | 0.2167 | 0 |
| transformer_strict | 0 | test | none | output_only_oracle_probe | 0.2750 | 2324 |
| transformer_strict | 0 | test | none | single_agent | 0.2167 | 0 |
| transformer_strict | 0 | test | none | text_only_coordinator | 0.2833 | 2068 |

## Mechanism Distinction

- Transformer strict mechanism: captured local-transformer hidden states give hidden-only accuracy 0.2750 and hidden+output accuracy 0.3583, while output-only oracle accuracy is 0.2750. Mean individual-agent hidden label accuracy is 0.0000; this distinguishes combined private-evidence aggregation from single-agent final-label decoding.

## Synthetic Vs Transformer Hidden States

- Stage 1 source: local Hugging Face model `EleutherAI/pythia-70m-deduped`, layers `[1, 5]`, token positions `['evidence', 'final', 'prompt_mean']`.
- `transformer_strict`: hidden-only=0.2750, hidden+output=0.3583, output-only oracle=0.2750.
- Synthetic telemetry channels are constructed additive signals. Transformer hidden states are captured by forward hooks from a local model and do not expose semantic channel labels; only layer/token-position ablations are available.
- Claims remain limited to this controlled partial-evidence setup unless the same controls reproduce on richer real-agent tasks.

## Interpretation

- `transformer_strict`: `activation_pca_mlp` scored 0.3583; strongest output-only reference scored 0.2833; delta 0.0750.
- `transformer_strict`: combined hidden-state-only label accuracy was 0.2750; mean individual-agent hidden label accuracy was 0.0000. This indicates label-predictive distributed private evidence across agents, not single-agent final-label decoding.
- `transformer_strict`: agent-correctness AUC text=0.5000, activation=0.4986, text+activation=0.4986.
- `transformer_strict`: hidden-state cluster purity by visible answer label=1.0000, by correctness=0.7833. Visible answers are intentionally sanitized and constant, so answer-label purity is not evidence of reasoning-mode clustering here.
- Recommended next experiment: make the transformer strict task less templated while keeping private evidence hidden from visible outputs, then repeat the same access audit, randomized-label sanity check, output-only baselines, and individual-vs-combined hidden-state probes.
