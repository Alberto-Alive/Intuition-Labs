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
    "adapter_hidden_dim": 48,
    "agent_mode": "tiny_transformer",
    "allow_tiny_fallback": true,
    "hidden_dim": 48,
    "local_files_only": true,
    "lora_alpha": 8.0,
    "lora_rank": 4,
    "lora_target_modules": [
      "query_key_value",
      "dense",
      "dense_h_to_4h",
      "dense_4h_to_h"
    ],
    "max_length": 128,
    "model_name_or_path": "EleutherAI/pythia-70m-deduped",
    "tiny_ff_dim": 96,
    "tiny_heads": 2,
    "tiny_layers": 1,
    "tiny_vocab_size": 4096,
    "train_top_layers": 0
  },
  "baseline_training_config": {
    "batch_size": 32,
    "epochs": 8,
    "hidden_dims": [
      32
    ],
    "lr": 0.002,
    "patience": 4,
    "weight_decay": 0.0001
  },
  "config": {
    "agent": {
      "adapter_hidden_dim": 48,
      "agent_mode": "tiny_transformer",
      "hidden_dim": 48,
      "max_length": 128,
      "tiny_ff_dim": 96,
      "tiny_heads": 2,
      "tiny_layers": 1,
      "tiny_vocab_size": 4096
    },
    "audit_log_path": "results/real_shared_weight_latent_coordination_curriculum4_audit.jsonl",
    "baseline_training": {
      "batch_size": 32,
      "epochs": 8,
      "hidden_dims": [
        32
      ],
      "lr": 0.002,
      "patience": 4,
      "weight_decay": 0.0001
    },
    "controls": [
      "none",
      "randomized_labels",
      "view_masked",
      "view_shuffled",
      "hidden_states_shuffled_across_examples",
      "role_labels_shuffled",
      "physical_order_shuffled_roles_preserved",
      "candidate_order_shuffled"
    ],
    "coordinator": {
      "dropout": 0.0,
      "family": "cross_attention",
      "ff_dim": 128,
      "input_dim": 48,
      "model_dim": 64,
      "num_heads": 2,
      "num_layers": 1
    },
    "coordinator_families": [
      "cross_attention"
    ],
    "dataset": {
      "include_private_signal_tokens": true,
      "max_files": 40,
      "n_dev": 96,
      "n_test": 128,
      "n_train": 256,
      "n_views": 4,
      "num_candidates": 4,
      "snippet_radius": 2,
      "source_roots": [
        "src",
        "tests"
      ]
    },
    "deterministic": true,
    "device": "cpu",
    "output_path": "results/real_shared_weight_latent_coordination_curriculum4_results.json",
    "positive_control_training": {
      "batch_size": 32,
      "epochs": 40,
      "hidden_dims": [
        64,
        32
      ],
      "lr": 0.005,
      "patience": 10,
      "weight_decay": 0.0001
    },
    "report_path": "reports/REAL_SHARED_WEIGHT_CURRICULUM4.md",
    "seeds": [
      0
    ],
    "text_feature_dim": 128,
    "training": {
      "batch_size": 32,
      "epochs": 14,
      "gradient_accumulation_steps": 1,
      "lr": 0.002,
      "patience": 6,
      "weight_decay": 0.0001
    }
  },
  "config_path": "configs\\real_shared_weight_latent_coordination_cpu_curriculum4.json",
  "coordinator_config": {
    "dropout": 0.0,
    "family": "cross_attention",
    "ff_dim": 128,
    "input_dim": 48,
    "model_dim": 64,
    "num_heads": 2,
    "num_layers": 1
  },
  "dataset_config": {
    "dataset_source": "constructed_local_code_patch_selection",
    "generator_version": "balanced_distributed_tuple_v2",
    "include_private_signal_tokens": true,
    "max_files": 40,
    "n_dev": 96,
    "n_test": 128,
    "n_train": 256,
    "n_views": 4,
    "num_candidates": 4,
    "snippet_radius": 2,
    "source_roots": [
      "src",
      "tests"
    ]
  },
  "dataset_summary_by_seed": {
    "0": {
      "constructed_fallback": true,
      "dataset_source": "constructed_local_code_patch_selection",
      "num_candidates": 4,
      "num_views": 4,
      "problem_families_by_split": {
        "dev": [
          "date_bucket",
          "retry_budget",
          "token_counter"
        ],
        "test": [
          "inventory_delta",
          "pagination_cursor",
          "permission_merge"
        ],
        "train": [
          "cache_lookup",
          "json_flag",
          "list_window",
          "metric_average",
          "path_joiner",
          "string_normalizer"
        ]
      },
      "publishable_proof_dataset": false,
      "source_file_examples": [
        "cache.py",
        "dates.py",
        "flags.py",
        "inventory.py",
        "metrics.py",
        "normalizers.py",
        "pagination.py",
        "paths.py",
        "permissions.py",
        "retries.py"
      ],
      "source_files_used": 12,
      "split_sizes": {
        "dev": 96,
        "test": 128,
        "train": 256
      }
    }
  },
  "device_requested": "cpu",
  "device_used": "cpu",
  "environment": {
    "cuda_available": true,
    "cuda_device_name": "NVIDIA GeForce RTX 5070 Ti",
    "cuda_total_memory": 17094475776,
    "device": "cpu",
    "python": "3.14.0 (tags/v3.14.0:ebf955d, Oct  7 2025, 10:15:03) [MSC v.1944 64 bit (AMD64)]",
    "torch": "2.11.0+cu128"
  },
  "previous_run_summary": {
    "control_accuracy": {
      "candidate_order_shuffled": 0.2578125,
      "hidden_states_shuffled_across_examples": 0.265625,
      "physical_order_shuffled_roles_preserved": 0.2890625,
      "randomized_labels": 0.265625,
      "role_labels_shuffled": 0.2734375,
      "view_masked": 0.2578125,
      "view_shuffled": 0.2421875
    },
    "gate_e_controls_collapse": false,
    "gate_f_dataset_not_trivial": false,
    "path": "results\\real_shared_weight_latent_coordination_curriculum4_results.json",
    "test_accuracy": {
      "bag_of_words_candidate_patch_only": 0.25,
      "candidate_order_baseline": 0.25,
      "coordinator_only_probe": 0.2265625,
      "explicit_evidence_oracle": 1.0,
      "frozen_shared_agent_latent_coordinator": 0.25,
      "larger_tiny_transformer_full_context": 0.28125,
      "majority_class_baseline": 0.25,
      "role_labels_only_baseline": 0.25,
      "single_agent_full_context": 0.1875,
      "single_agent_partial_view": 0.25,
      "single_view_text_role_0": 0.25,
      "single_view_text_role_1": 0.2578125,
      "single_view_text_role_2": 0.234375,
      "single_view_text_role_3": 0.25,
      "text_only_partial_view_baseline": 0.25,
      "text_output_only_multi_agent_coordinator": 0.2421875,
      "trainable_shared_agent_latent_coordinator": 0.2890625
    },
    "trivial_baseline_accuracy": {
      "bag_of_words_candidate_patch_only": 0.25,
      "candidate_order_baseline": 0.25,
      "majority_class_baseline": 0.25,
      "masked_evidence_text_baseline": 0.234375,
      "role_labels_only_baseline": 0.25,
      "single_view_text_role_0": 0.25,
      "single_view_text_role_1": 0.2578125,
      "single_view_text_role_2": 0.234375,
      "single_view_text_role_3": 0.25,
      "text_only_partial_view_baseline": 0.25
    }
  },
  "stage": "real_shared_weight_latent_coordination",
  "training_config": {
    "batch_size": 32,
    "epochs": 14,
    "gradient_accumulation_steps": 1,
    "lr": 0.002,
    "patience": 6,
    "weight_decay": 0.0001
  }
}
```

## Aggregate Accuracy

| benchmark | split | condition | method | mean acc | std | runs | params |
|---|---|---|---|---:|---:|---:|---:|
| real_shared_weight_latent_coordination | dev | candidate_order_shuffled | trainable_shared_agent_latent_coordinator | 0.2083 | 0.0000 | 1 | 263924 |
| real_shared_weight_latent_coordination | dev | explicit_evidence | explicit_evidence_neural_tuple_model | 1.0000 | 0.0000 | 1 | 3265 |
| real_shared_weight_latent_coordination | dev | explicit_evidence | raw_structured_evidence_coordinator | 1.0000 | 0.0000 | 1 | 3201 |
| real_shared_weight_latent_coordination | dev | hidden_states_shuffled_across_examples | trainable_shared_agent_latent_coordinator | 0.2083 | 0.0000 | 1 | 263924 |
| real_shared_weight_latent_coordination | dev | none | bag_of_words_candidate_patch_only | 0.2708 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | dev | none | candidate_order_baseline | 0.2500 | 0.0000 | 1 | 0 |
| real_shared_weight_latent_coordination | dev | none | coordinator_only_probe | 0.2708 | 0.0000 | 1 | 420 |
| real_shared_weight_latent_coordination | dev | none | explicit_evidence_oracle | 1.0000 | 0.0000 | 1 | 0 |
| real_shared_weight_latent_coordination | dev | none | frozen_shared_agent_latent_coordinator | 0.2500 | 0.0000 | 1 | 37316 |
| real_shared_weight_latent_coordination | dev | none | larger_tiny_transformer_full_context | 0.2917 | 0.0000 | 1 | 574468 |
| real_shared_weight_latent_coordination | dev | none | majority_class_baseline | 0.2500 | 0.0000 | 1 | 0 |
| real_shared_weight_latent_coordination | dev | none | role_labels_only_baseline | 0.2500 | 0.0000 | 1 | 1188 |
| real_shared_weight_latent_coordination | dev | none | single_agent_full_context | 0.2500 | 0.0000 | 1 | 226804 |
| real_shared_weight_latent_coordination | dev | none | single_agent_partial_view | 0.2500 | 0.0000 | 1 | 226804 |
| real_shared_weight_latent_coordination | dev | none | single_view_text_role_0 | 0.2500 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | dev | none | single_view_text_role_1 | 0.2708 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | dev | none | single_view_text_role_2 | 0.2500 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | dev | none | single_view_text_role_3 | 0.2812 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | dev | none | text_only_partial_view_baseline | 0.2708 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | dev | none | text_output_only_multi_agent_coordinator | 0.2917 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | dev | none | trainable_shared_agent_latent_coordinator | 0.3021 | 0.0000 | 1 | 263924 |
| real_shared_weight_latent_coordination | dev | physical_order_shuffled_roles_preserved | trainable_shared_agent_latent_coordinator | 0.3021 | 0.0000 | 1 | 263924 |
| real_shared_weight_latent_coordination | dev | randomized_labels | trainable_shared_agent_latent_coordinator | 0.2500 | 0.0000 | 1 | 263924 |
| real_shared_weight_latent_coordination | dev | role_labels_shuffled | trainable_shared_agent_latent_coordinator | 0.2396 | 0.0000 | 1 | 263924 |
| real_shared_weight_latent_coordination | dev | view_masked | masked_evidence_text_baseline | 0.2604 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | dev | view_masked | trainable_shared_agent_latent_coordinator | 0.1979 | 0.0000 | 1 | 263924 |
| real_shared_weight_latent_coordination | dev | view_shuffled | trainable_shared_agent_latent_coordinator | 0.2396 | 0.0000 | 1 | 263924 |
| real_shared_weight_latent_coordination | dev | visible_explicit_evidence | trainable_shared_agent_latent_visible_explicit_evidence | 0.3021 | 0.0000 | 1 | 263924 |
| real_shared_weight_latent_coordination | test | candidate_order_shuffled | trainable_shared_agent_latent_coordinator | 0.2578 | 0.0000 | 1 | 263924 |
| real_shared_weight_latent_coordination | test | explicit_evidence | explicit_evidence_neural_tuple_model | 1.0000 | 0.0000 | 1 | 3265 |
| real_shared_weight_latent_coordination | test | explicit_evidence | raw_structured_evidence_coordinator | 1.0000 | 0.0000 | 1 | 3201 |
| real_shared_weight_latent_coordination | test | hidden_states_shuffled_across_examples | trainable_shared_agent_latent_coordinator | 0.2578 | 0.0000 | 1 | 263924 |
| real_shared_weight_latent_coordination | test | none | bag_of_words_candidate_patch_only | 0.2500 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | test | none | candidate_order_baseline | 0.2500 | 0.0000 | 1 | 0 |
| real_shared_weight_latent_coordination | test | none | coordinator_only_probe | 0.2812 | 0.0000 | 1 | 420 |
| real_shared_weight_latent_coordination | test | none | explicit_evidence_oracle | 1.0000 | 0.0000 | 1 | 0 |
| real_shared_weight_latent_coordination | test | none | frozen_shared_agent_latent_coordinator | 0.2500 | 0.0000 | 1 | 37316 |
| real_shared_weight_latent_coordination | test | none | larger_tiny_transformer_full_context | 0.2422 | 0.0000 | 1 | 574468 |
| real_shared_weight_latent_coordination | test | none | majority_class_baseline | 0.2500 | 0.0000 | 1 | 0 |
| real_shared_weight_latent_coordination | test | none | role_labels_only_baseline | 0.2500 | 0.0000 | 1 | 1188 |
| real_shared_weight_latent_coordination | test | none | single_agent_full_context | 0.1875 | 0.0000 | 1 | 226804 |
| real_shared_weight_latent_coordination | test | none | single_agent_partial_view | 0.2500 | 0.0000 | 1 | 226804 |
| real_shared_weight_latent_coordination | test | none | single_view_text_role_0 | 0.2500 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | test | none | single_view_text_role_1 | 0.2578 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | test | none | single_view_text_role_2 | 0.2578 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | test | none | single_view_text_role_3 | 0.2500 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | test | none | text_only_partial_view_baseline | 0.2500 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | test | none | text_output_only_multi_agent_coordinator | 0.2422 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | test | none | trainable_shared_agent_latent_coordinator | 0.2500 | 0.0000 | 1 | 263924 |
| real_shared_weight_latent_coordination | test | physical_order_shuffled_roles_preserved | trainable_shared_agent_latent_coordinator | 0.2500 | 0.0000 | 1 | 263924 |
| real_shared_weight_latent_coordination | test | randomized_labels | trainable_shared_agent_latent_coordinator | 0.2578 | 0.0000 | 1 | 263924 |
| real_shared_weight_latent_coordination | test | role_labels_shuffled | trainable_shared_agent_latent_coordinator | 0.2891 | 0.0000 | 1 | 263924 |
| real_shared_weight_latent_coordination | test | view_masked | masked_evidence_text_baseline | 0.2344 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | test | view_masked | trainable_shared_agent_latent_coordinator | 0.2812 | 0.0000 | 1 | 263924 |
| real_shared_weight_latent_coordination | test | view_shuffled | trainable_shared_agent_latent_coordinator | 0.2812 | 0.0000 | 1 | 263924 |
| real_shared_weight_latent_coordination | test | visible_explicit_evidence | trainable_shared_agent_latent_visible_explicit_evidence | 0.2422 | 0.0000 | 1 | 263924 |

## Main Test Comparison

| benchmark | method | test mean acc | std | runs |
|---|---|---:|---:|---:|
| real_shared_weight_latent_coordination | explicit_evidence_oracle | 1.0000 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | coordinator_only_probe | 0.2812 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | single_view_text_role_1 | 0.2578 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | single_view_text_role_2 | 0.2578 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | frozen_shared_agent_latent_coordinator | 0.2500 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | trainable_shared_agent_latent_coordinator | 0.2500 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | majority_class_baseline | 0.2500 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | candidate_order_baseline | 0.2500 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | bag_of_words_candidate_patch_only | 0.2500 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | text_only_partial_view_baseline | 0.2500 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | single_view_text_role_0 | 0.2500 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | single_view_text_role_3 | 0.2500 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | role_labels_only_baseline | 0.2500 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | single_agent_partial_view | 0.2500 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | text_output_only_multi_agent_coordinator | 0.2422 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | larger_tiny_transformer_full_context | 0.2422 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | single_agent_full_context | 0.1875 | 0.0000 | 1 |

## Confidence Intervals

| benchmark | quantity | mean | 95% bootstrap CI | n |
|---|---|---:|---|---:|

## Anti-Cheat Controls

| benchmark | condition | method | test mean acc | std | runs |
|---|---|---|---:|---:|---:|

## Randomized-Label Leakage Test

Randomized-label leakage test was disabled.

## Label Permutation Sanity

| benchmark | method | normal test acc | permuted-label test acc | drop |
|---|---|---:|---:|---:|

## Train/Test Split Audit

Test access violations before the test gate: `0`.

| benchmark | seed | train/dev overlap | train/test overlap | dev/test overlap |
|---|---:|---:|---:|---:|

## Diagnostic Label Probes

| benchmark | method | test mean acc | std | runs | params |
|---|---|---:|---:|---:|---:|

## Stage 1 Mechanism Validation

Stage 1 transformer_strict diagnostics have not been run.

## Stage 3 Robustness

Stage 3 robustness metrics are not available.

## Stage 4 Hidden-State Coordinators

Stage 4 hidden-state coordinators were not run.

## Stage 4 Agent-Count Curve

Stage 4 agent-count curve was not run.

## Stage 4 Controls

Stage 4 controls were not run.

## Stage 5 Role-Aware QKV Coordinators

Stage 5 role-aware QKV coordinators were not run.

## Stage 5 Agent-Count Curve

Stage 5 agent-count curve was not run.

## Stage 5 Controls

Stage 5 controls were not run.

## Shared-Weight Cloned-Agent Training

Stage 6A shared-weight cloned-agent training was not run.

## Benchmark Validity Diagnostics

Failed shortcuts found:
- old masked and shuffled view controls stayed above chance
- old role-label shuffle stayed above chance because textual role cues were available
- old single partial-view baseline matched the trainable latent method
- old candidate index distribution was not exactly balanced in small splits

Dataset fixes applied:
- candidate patch order randomized with balanced gold position
- eight semantically close candidates generated from a balanced four-attribute codeword set
- each view carries only one generic high/low cue
- role text removed from clone prompts so role embeddings carry slot identity
- problem families are disjoint across train/dev/test
- masked control removes all distributed evidence cues

Trivial baseline table:
| method | test acc | std | runs |
|---|---:|---:|---:|
| bag_of_words_candidate_patch_only | 0.2500 | 0.0000 | 1 |
| candidate_order_baseline | 0.2500 | 0.0000 | 1 |
| majority_class_baseline | 0.2500 | 0.0000 | 1 |
| role_labels_only_baseline | 0.2500 | 0.0000 | 1 |
| single_view_text_role_0 | 0.2500 | 0.0000 | 1 |
| single_view_text_role_1 | 0.2578 | 0.0000 | 1 |
| single_view_text_role_2 | 0.2578 | 0.0000 | 1 |
| single_view_text_role_3 | 0.2500 | 0.0000 | 1 |
| text_only_partial_view_baseline | 0.2500 | 0.0000 | 1 |
| masked_evidence_text_baseline | 0.2344 | 0.0000 | 1 |

Control table before/after:
| condition | before acc | after acc |
|---|---:|---:|
| candidate_order_shuffled | 0.2578 | 0.2578 |
| hidden_states_shuffled_across_examples | 0.2656 | 0.2578 |
| physical_order_shuffled_roles_preserved | 0.2891 | 0.2500 |
| randomized_labels | 0.2656 | 0.2578 |
| role_labels_shuffled | 0.2734 | 0.2891 |
| view_masked | 0.2578 | 0.2812 |
| view_shuffled | 0.2422 | 0.2812 |

Label and split audit:
| split | label counts | problem families |
|---|---|---|
| dev | `{'0': 24, '1': 24, '2': 24, '3': 24}` | `['date_bucket', 'retry_budget', 'token_counter']` |
| test | `{'0': 32, '1': 32, '2': 32, '3': 32}` | `['inventory_delta', 'pagination_cursor', 'permission_merge']` |
| train | `{'0': 64, '1': 64, '2': 64, '3': 64}` | `['cache_lookup', 'json_flag', 'list_window', 'metric_average', 'path_joiner', 'string_normalizer']` |

Correct examples under corrupted controls:
| condition | acc | inspected correct examples |
|---|---:|---|
| hidden_states_shuffled_across_examples | 0.2578 | `['test-00001', 'test-00002', 'test-00003', 'test-00005', 'test-00009']` |
| role_labels_shuffled | 0.2891 | `['test-00001', 'test-00005', 'test-00009', 'test-00017', 'test-00021']` |
| view_masked | 0.2812 | `['test-00001', 'test-00003', 'test-00005', 'test-00013', 'test-00015']` |
| view_shuffled | 0.2812 | `['test-00003', 'test-00005', 'test-00013', 'test-00014', 'test-00019']` |

Final gate status:
| gate | pass |
|---|---|
| gate_a_valid_training_graph | True |
| gate_b_frozen_control | True |
| gate_c_trainable_beats_frozen | False |
| gate_d_trainable_beats_text_only | True |
| gate_e_controls_collapse | True |
| gate_f_dataset_not_trivial | True |
| gate_positive_control_learnable | True |

## Learnability Positive Controls

These controls test whether the repaired benchmark is learnable before tuning latent architecture.

| method | condition | test acc | std | runs | params |
|---|---|---:|---:|---:|---:|
| explicit_evidence_neural_tuple_model | explicit_evidence | 1.0000 | 0.0000 | 1 | 3265 |
| raw_structured_evidence_coordinator | explicit_evidence | 1.0000 | 0.0000 | 1 | 3201 |
| larger_tiny_transformer_full_context | none | 0.2422 | 0.0000 | 1 | 574468 |
| trainable_shared_agent_latent_visible_explicit_evidence | visible_explicit_evidence | 0.2422 | 0.0000 | 1 | 263924 |
| single_agent_full_context | none | 0.1875 | 0.0000 | 1 | 226804 |

Learnability gate `gate_positive_control_learnable`: `True`; best learned positive-control test accuracy `1.0000`.
At least one learned positive control solved the benchmark, so subsequent failures are more likely model/architecture/training issues than label construction issues.

## Real Shared-Weight Latent Coordination

This section is governed by the real shared-weight latent coordination specification. The available local data did not include SWE-bench or issue-patch artifacts, so the run uses a constructed local-code patch-selection fallback and is not treated as publishable real-world proof.

Architecture and data:
- Agent model mode: `tiny_transformer`; model path: `EleutherAI/pythia-70m-deduped`; hidden dim: `48`.
- Trainable parameters: shared agent coordination parameters plus coordinator in trainable runs; frozen runs keep shared agent deltas at zero.
- Dataset source: `constructed_local_code_patch_selection`; split sizes: `{'train': 256, 'dev': 96, 'test': 128}`; source files used: `12`.
- Batch settings: batch size `32`, epochs `14`, grad accumulation `1`.

Main comparison:
| method | test mean acc | std | runs | params |
|---|---:|---:|---:|---:|
| explicit_evidence_oracle | 1.0000 | 0.0000 | 1 | 0 |
| coordinator_only_probe | 0.2812 | 0.0000 | 1 | 420 |
| single_view_text_role_1 | 0.2578 | 0.0000 | 1 | 4260 |
| single_view_text_role_2 | 0.2578 | 0.0000 | 1 | 4260 |
| bag_of_words_candidate_patch_only | 0.2500 | 0.0000 | 1 | 4260 |
| candidate_order_baseline | 0.2500 | 0.0000 | 1 | 0 |
| frozen_shared_agent_latent_coordinator | 0.2500 | 0.0000 | 1 | 37316 |
| majority_class_baseline | 0.2500 | 0.0000 | 1 | 0 |
| role_labels_only_baseline | 0.2500 | 0.0000 | 1 | 1188 |
| single_agent_partial_view | 0.2500 | 0.0000 | 1 | 226804 |
| single_view_text_role_0 | 0.2500 | 0.0000 | 1 | 4260 |
| single_view_text_role_3 | 0.2500 | 0.0000 | 1 | 4260 |
| text_only_partial_view_baseline | 0.2500 | 0.0000 | 1 | 4260 |
| trainable_shared_agent_latent_coordinator | 0.2500 | 0.0000 | 1 | 263924 |
| larger_tiny_transformer_full_context | 0.2422 | 0.0000 | 1 | 574468 |
| text_output_only_multi_agent_coordinator | 0.2422 | 0.0000 | 1 | 4260 |
| single_agent_full_context | 0.1875 | 0.0000 | 1 | 226804 |

Audit table:
| method | condition | shared params | M grad | M delta | C grad | C delta | activations grad | no detach | per-clone grad | memory bytes |
|---|---|---|---:|---:|---:|---:|---|---|---|---:|
| frozen_shared_agent_latent_coordinator | none | pass | 0.000000 | 0.000000 | 1.957447 | 1.318636 | fail | fail | fail | 0 |
| trainable_shared_agent_latent_coordinator | none | pass | 0.368175 | 4.412677 | 1.688092 | 2.006457 | pass | pass | pass | 0 |
| trainable_shared_agent_latent_coordinator | randomized_labels | pass | 0.115919 | 2.632085 | 1.597752 | 1.603456 | pass | pass | pass | 0 |
| trainable_shared_agent_latent_visible_explicit_evidence | visible_explicit_evidence | pass | 0.234069 | 8.630353 | 1.622290 | 2.949070 | pass | pass | pass | 0 |

Controls:
| condition | test mean acc | std | runs |
|---|---:|---:|---:|
| candidate_order_shuffled | 0.2578 | 0.0000 | 1 |
| hidden_states_shuffled_across_examples | 0.2578 | 0.0000 | 1 |
| physical_order_shuffled_roles_preserved | 0.2500 | 0.0000 | 1 |
| randomized_labels | 0.2578 | 0.0000 | 1 |
| role_labels_shuffled | 0.2891 | 0.0000 | 1 |
| view_masked | 0.2812 | 0.0000 | 1 |
| view_shuffled | 0.2812 | 0.0000 | 1 |

Learning curves:
| method | condition | split | first epoch acc | final epoch acc | epochs |
|---|---|---|---:|---:|---:|
| frozen_shared_agent_latent_coordinator | none | dev | 0.2500 | 0.2500 | 7 |
| frozen_shared_agent_latent_coordinator | none | train | 0.2500 | 0.3164 | 7 |
| trainable_shared_agent_latent_coordinator | none | dev | 0.2500 | 0.1979 | 9 |
| trainable_shared_agent_latent_coordinator | none | train | 0.2500 | 0.7070 | 9 |
| trainable_shared_agent_latent_visible_explicit_evidence | visible_explicit_evidence | dev | 0.2500 | 0.2188 | 13 |
| trainable_shared_agent_latent_visible_explicit_evidence | visible_explicit_evidence | train | 0.2500 | 0.5859 | 13 |

Split and output leakage audit:
| seed | id overlaps train/dev/test | candidate hash overlaps train/dev/test | output leakage passes |
|---:|---|---|---|
| 0 | 0/0/0 | 0/0/0 | True |

Acceptance gates:
| gate | pass |
|---|---|
| gate_a_valid_training_graph | True |
| gate_b_frozen_control | True |
| gate_c_trainable_beats_frozen | False |
| gate_d_trainable_beats_text_only | True |
| gate_e_controls_collapse | True |
| gate_f_dataset_not_trivial | True |

Checklist:
1. Exact model: `tiny_transformer` with `EleutherAI/pythia-70m-deduped` when pretrained loading is enabled.
2. Trainable parameters: `['tiny_embedding.weight', 'tiny_position.weight', 'tiny_encoder.layers.0.self_attn.in_proj_weight', 'tiny_encoder.layers.0.self_attn.in_proj_bias', 'tiny_encoder.layers.0.self_attn.out_proj.weight', 'tiny_encoder.layers.0.self_attn.out_proj.bias', 'tiny_encoder.layers.0.linear1.weight', 'tiny_encoder.layers.0.linear1.bias', 'tiny_encoder.layers.0.linear2.weight', 'tiny_encoder.layers.0.linear2.bias', 'tiny_encoder.layers.0.norm1.weight', 'tiny_encoder.layers.0.norm1.bias', 'tiny_encoder.layers.0.norm2.weight', 'tiny_encoder.layers.0.norm2.bias', 'tiny_norm.weight', 'tiny_norm.bias', 'activation_adapter.0.weight', 'activation_adapter.0.bias', 'activation_adapter.1.weight', 'activation_adapter.1.bias', 'activation_adapter.3.weight', 'activation_adapter.3.bias']`.
3. Shared across clones: `True`.
4. Nonzero shared gradients: `True`.
5. Shared parameter delta: `4.412677`.
6. Trainable beat frozen: `False`.
7. Trainable beat text-only: `True`.
8. Controls collapse: inspect the controls table; this fallback run does not by itself authorize a positive real-world claim.
9. Physical order shuffle with roles preserved: inspect `physical_order_shuffled_roles_preserved`.
10. Role-label shuffle: inspect `role_labels_shuffled`.
11. Single partial-view accuracy: `0.25`.
12. Output leakage: `True`.
13. Train/dev/test leakage: `True`.
14. GPU memory: `0` bytes.
15. Failed variants tried: all trained coordinator-family variants are shown in the main table and audit table.
16. Most conservative valid claim: constructed-fallback graph and audit plumbing are implemented; real-world latent coordination remains unproven without a real issue-patch dataset and stronger trained-model performance.

The current architecture/training setup does not demonstrate shared-weight cloned-agent cooperation. Further work should diagnose model capacity, activation extraction, dataset difficulty, and coordinator design.

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

Output leakage audit was not run.

## activation_pca_mlp Stability

activation_pca_mlp stability diagnostics were not run.

## Explicit Evidence-Sharing Baseline

Explicit evidence-sharing baseline was not run.

## Agent Correctness Probe

Agent correctness probe was not run.

## Redundancy Probe

Redundancy probe was not run.

## Telemetry Channel Ablations

Telemetry channel ablations were not run.

## Transformer Hidden-State Ablations

Transformer hidden-state slice ablations were not run.

## Individual Hidden-State Label Probe

Individual hidden-state label probe was not run.

## Probe Access Matrix

Probe access matrix was not recorded.

## CPU/CUDA Parity

CPU/CUDA parity artifact not attached to this report.

## All Runs

| benchmark | seed | split | condition | method | accuracy | params |
|---|---:|---|---|---|---:|---:|
| real_shared_weight_latent_coordination | 0 | dev | candidate_order_shuffled | trainable_shared_agent_latent_coordinator | 0.2083 | 263924 |
| real_shared_weight_latent_coordination | 0 | dev | explicit_evidence | explicit_evidence_neural_tuple_model | 1.0000 | 3265 |
| real_shared_weight_latent_coordination | 0 | dev | explicit_evidence | raw_structured_evidence_coordinator | 1.0000 | 3201 |
| real_shared_weight_latent_coordination | 0 | dev | hidden_states_shuffled_across_examples | trainable_shared_agent_latent_coordinator | 0.2083 | 263924 |
| real_shared_weight_latent_coordination | 0 | dev | none | bag_of_words_candidate_patch_only | 0.2708 | 4260 |
| real_shared_weight_latent_coordination | 0 | dev | none | candidate_order_baseline | 0.2500 | 0 |
| real_shared_weight_latent_coordination | 0 | dev | none | coordinator_only_probe | 0.2708 | 420 |
| real_shared_weight_latent_coordination | 0 | dev | none | explicit_evidence_oracle | 1.0000 | 0 |
| real_shared_weight_latent_coordination | 0 | dev | none | frozen_shared_agent_latent_coordinator | 0.2500 | 37316 |
| real_shared_weight_latent_coordination | 0 | dev | none | larger_tiny_transformer_full_context | 0.2917 | 574468 |
| real_shared_weight_latent_coordination | 0 | dev | none | majority_class_baseline | 0.2500 | 0 |
| real_shared_weight_latent_coordination | 0 | dev | none | role_labels_only_baseline | 0.2500 | 1188 |
| real_shared_weight_latent_coordination | 0 | dev | none | single_agent_full_context | 0.2500 | 226804 |
| real_shared_weight_latent_coordination | 0 | dev | none | single_agent_partial_view | 0.2500 | 226804 |
| real_shared_weight_latent_coordination | 0 | dev | none | single_view_text_role_0 | 0.2500 | 4260 |
| real_shared_weight_latent_coordination | 0 | dev | none | single_view_text_role_1 | 0.2708 | 4260 |
| real_shared_weight_latent_coordination | 0 | dev | none | single_view_text_role_2 | 0.2500 | 4260 |
| real_shared_weight_latent_coordination | 0 | dev | none | single_view_text_role_3 | 0.2812 | 4260 |
| real_shared_weight_latent_coordination | 0 | dev | none | text_only_partial_view_baseline | 0.2708 | 4260 |
| real_shared_weight_latent_coordination | 0 | dev | none | text_output_only_multi_agent_coordinator | 0.2917 | 4260 |
| real_shared_weight_latent_coordination | 0 | dev | none | trainable_shared_agent_latent_coordinator | 0.3021 | 263924 |
| real_shared_weight_latent_coordination | 0 | dev | physical_order_shuffled_roles_preserved | trainable_shared_agent_latent_coordinator | 0.3021 | 263924 |
| real_shared_weight_latent_coordination | 0 | dev | randomized_labels | trainable_shared_agent_latent_coordinator | 0.2500 | 263924 |
| real_shared_weight_latent_coordination | 0 | dev | role_labels_shuffled | trainable_shared_agent_latent_coordinator | 0.2396 | 263924 |
| real_shared_weight_latent_coordination | 0 | dev | view_masked | masked_evidence_text_baseline | 0.2604 | 4260 |
| real_shared_weight_latent_coordination | 0 | dev | view_masked | trainable_shared_agent_latent_coordinator | 0.1979 | 263924 |
| real_shared_weight_latent_coordination | 0 | dev | view_shuffled | trainable_shared_agent_latent_coordinator | 0.2396 | 263924 |
| real_shared_weight_latent_coordination | 0 | dev | visible_explicit_evidence | trainable_shared_agent_latent_visible_explicit_evidence | 0.3021 | 263924 |
| real_shared_weight_latent_coordination | 0 | test | candidate_order_shuffled | trainable_shared_agent_latent_coordinator | 0.2578 | 263924 |
| real_shared_weight_latent_coordination | 0 | test | explicit_evidence | explicit_evidence_neural_tuple_model | 1.0000 | 3265 |
| real_shared_weight_latent_coordination | 0 | test | explicit_evidence | raw_structured_evidence_coordinator | 1.0000 | 3201 |
| real_shared_weight_latent_coordination | 0 | test | hidden_states_shuffled_across_examples | trainable_shared_agent_latent_coordinator | 0.2578 | 263924 |
| real_shared_weight_latent_coordination | 0 | test | none | bag_of_words_candidate_patch_only | 0.2500 | 4260 |
| real_shared_weight_latent_coordination | 0 | test | none | candidate_order_baseline | 0.2500 | 0 |
| real_shared_weight_latent_coordination | 0 | test | none | coordinator_only_probe | 0.2812 | 420 |
| real_shared_weight_latent_coordination | 0 | test | none | explicit_evidence_oracle | 1.0000 | 0 |
| real_shared_weight_latent_coordination | 0 | test | none | frozen_shared_agent_latent_coordinator | 0.2500 | 37316 |
| real_shared_weight_latent_coordination | 0 | test | none | larger_tiny_transformer_full_context | 0.2422 | 574468 |
| real_shared_weight_latent_coordination | 0 | test | none | majority_class_baseline | 0.2500 | 0 |
| real_shared_weight_latent_coordination | 0 | test | none | role_labels_only_baseline | 0.2500 | 1188 |
| real_shared_weight_latent_coordination | 0 | test | none | single_agent_full_context | 0.1875 | 226804 |
| real_shared_weight_latent_coordination | 0 | test | none | single_agent_partial_view | 0.2500 | 226804 |
| real_shared_weight_latent_coordination | 0 | test | none | single_view_text_role_0 | 0.2500 | 4260 |
| real_shared_weight_latent_coordination | 0 | test | none | single_view_text_role_1 | 0.2578 | 4260 |
| real_shared_weight_latent_coordination | 0 | test | none | single_view_text_role_2 | 0.2578 | 4260 |
| real_shared_weight_latent_coordination | 0 | test | none | single_view_text_role_3 | 0.2500 | 4260 |
| real_shared_weight_latent_coordination | 0 | test | none | text_only_partial_view_baseline | 0.2500 | 4260 |
| real_shared_weight_latent_coordination | 0 | test | none | text_output_only_multi_agent_coordinator | 0.2422 | 4260 |
| real_shared_weight_latent_coordination | 0 | test | none | trainable_shared_agent_latent_coordinator | 0.2500 | 263924 |
| real_shared_weight_latent_coordination | 0 | test | physical_order_shuffled_roles_preserved | trainable_shared_agent_latent_coordinator | 0.2500 | 263924 |
| real_shared_weight_latent_coordination | 0 | test | randomized_labels | trainable_shared_agent_latent_coordinator | 0.2578 | 263924 |
| real_shared_weight_latent_coordination | 0 | test | role_labels_shuffled | trainable_shared_agent_latent_coordinator | 0.2891 | 263924 |
| real_shared_weight_latent_coordination | 0 | test | view_masked | masked_evidence_text_baseline | 0.2344 | 4260 |
| real_shared_weight_latent_coordination | 0 | test | view_masked | trainable_shared_agent_latent_coordinator | 0.2812 | 263924 |
| real_shared_weight_latent_coordination | 0 | test | view_shuffled | trainable_shared_agent_latent_coordinator | 0.2812 | 263924 |
| real_shared_weight_latent_coordination | 0 | test | visible_explicit_evidence | trainable_shared_agent_latent_visible_explicit_evidence | 0.2422 | 263924 |

## Mechanism Distinction


## Synthetic Vs Transformer Hidden States

No captured-transformer hidden-state stage is present in this report.

## Interpretation

- Recommended next experiment: make the transformer strict task less templated while keeping private evidence hidden from visible outputs, then repeat the same access audit, randomized-label sanity check, output-only baselines, and individual-vs-combined hidden-state probes.
