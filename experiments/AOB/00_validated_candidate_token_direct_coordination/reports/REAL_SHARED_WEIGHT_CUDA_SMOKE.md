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
    "adapter_hidden_dim": 64,
    "agent_mode": "pretrained_adapter",
    "allow_tiny_fallback": false,
    "hidden_dim": 64,
    "local_files_only": true,
    "lora_alpha": 4.0,
    "lora_rank": 2,
    "lora_target_modules": [
      "query_key_value",
      "dense",
      "dense_h_to_4h",
      "dense_4h_to_h"
    ],
    "max_length": 128,
    "model_name_or_path": "EleutherAI/pythia-70m-deduped",
    "tiny_ff_dim": 128,
    "tiny_heads": 2,
    "tiny_layers": 1,
    "tiny_vocab_size": 4096,
    "train_top_layers": 0
  },
  "baseline_training_config": {
    "batch_size": 16,
    "epochs": 2,
    "hidden_dims": [
      32
    ],
    "lr": 0.002,
    "patience": 1,
    "weight_decay": 0.0001
  },
  "config": {
    "agent": {
      "adapter_hidden_dim": 64,
      "agent_mode": "pretrained_adapter",
      "allow_tiny_fallback": false,
      "hidden_dim": 64,
      "local_files_only": true,
      "lora_alpha": 4.0,
      "lora_rank": 2,
      "lora_target_modules": [
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h"
      ],
      "max_length": 128,
      "model_name_or_path": "EleutherAI/pythia-70m-deduped",
      "train_top_layers": 0
    },
    "audit_log_path": "results/real_shared_weight_latent_coordination_cuda_smoke_audit.jsonl",
    "baseline_training": {
      "batch_size": 16,
      "epochs": 2,
      "hidden_dims": [
        32
      ],
      "lr": 0.002,
      "patience": 1,
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
      "input_dim": 64,
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
      "n_dev": 32,
      "n_test": 32,
      "n_train": 64,
      "n_views": 4,
      "num_candidates": 4,
      "snippet_radius": 2,
      "source_roots": [
        "src",
        "tests"
      ]
    },
    "deterministic": false,
    "device": "cuda",
    "large_full_context_ff_dim": 128,
    "large_full_context_heads": 2,
    "large_full_context_hidden_dim": 64,
    "large_full_context_layers": 1,
    "output_path": "results/real_shared_weight_latent_coordination_cuda_smoke_results.json",
    "positive_control_training": {
      "batch_size": 16,
      "epochs": 10,
      "hidden_dims": [
        64,
        32
      ],
      "lr": 0.005,
      "patience": 3,
      "weight_decay": 0.0001
    },
    "report_path": "reports/REAL_SHARED_WEIGHT_CUDA_SMOKE.md",
    "seeds": [
      0
    ],
    "text_feature_dim": 128,
    "training": {
      "batch_size": 2,
      "epochs": 1,
      "gradient_accumulation_steps": 4,
      "lr": 0.0005,
      "patience": 1,
      "weight_decay": 0.0001
    }
  },
  "config_path": "configs\\real_shared_weight_latent_coordination_cuda_smoke.json",
  "coordinator_config": {
    "dropout": 0.0,
    "family": "cross_attention",
    "ff_dim": 128,
    "input_dim": 64,
    "model_dim": 64,
    "num_heads": 2,
    "num_layers": 1
  },
  "dataset_config": {
    "dataset_source": "constructed_local_code_patch_selection",
    "generator_version": "balanced_distributed_tuple_v2",
    "include_private_signal_tokens": true,
    "max_files": 40,
    "n_dev": 32,
    "n_test": 32,
    "n_train": 64,
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
        "dev": 32,
        "test": 32,
        "train": 64
      }
    }
  },
  "device_requested": "cuda",
  "device_used": "cuda",
  "environment": {
    "cuda_available": true,
    "cuda_device_name": "NVIDIA GeForce RTX 5070 Ti",
    "cuda_total_memory": 17094475776,
    "device": "cuda",
    "python": "3.14.0 (tags/v3.14.0:ebf955d, Oct  7 2025, 10:15:03) [MSC v.1944 64 bit (AMD64)]",
    "torch": "2.11.0+cu128"
  },
  "previous_run_summary": {},
  "stage": "real_shared_weight_latent_coordination",
  "training_config": {
    "batch_size": 2,
    "epochs": 1,
    "gradient_accumulation_steps": 4,
    "lr": 0.0005,
    "patience": 1,
    "weight_decay": 0.0001
  }
}
```

## Aggregate Accuracy

| benchmark | split | condition | method | mean acc | std | runs | params |
|---|---|---|---|---:|---:|---:|---:|
| real_shared_weight_latent_coordination | dev | candidate_order_shuffled | trainable_shared_agent_latent_coordinator | 0.2500 | 0.0000 | 1 | 174660 |
| real_shared_weight_latent_coordination | dev | explicit_evidence | explicit_evidence_neural_tuple_model | 1.0000 | 0.0000 | 1 | 3265 |
| real_shared_weight_latent_coordination | dev | explicit_evidence | raw_structured_evidence_coordinator | 1.0000 | 0.0000 | 1 | 3201 |
| real_shared_weight_latent_coordination | dev | hidden_states_shuffled_across_examples | trainable_shared_agent_latent_coordinator | 0.2500 | 0.0000 | 1 | 174660 |
| real_shared_weight_latent_coordination | dev | none | bag_of_words_candidate_patch_only | 0.2500 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | dev | none | candidate_order_baseline | 0.2500 | 0.0000 | 1 | 0 |
| real_shared_weight_latent_coordination | dev | none | coordinator_only_probe | 0.2500 | 0.0000 | 1 | 420 |
| real_shared_weight_latent_coordination | dev | none | explicit_evidence_oracle | 1.0000 | 0.0000 | 1 | 0 |
| real_shared_weight_latent_coordination | dev | none | frozen_shared_agent_latent_coordinator | 0.2500 | 0.0000 | 1 | 38340 |
| real_shared_weight_latent_coordination | dev | none | larger_tiny_transformer_full_context | 0.2500 | 0.0000 | 1 | 136580 |
| real_shared_weight_latent_coordination | dev | none | majority_class_baseline | 0.2500 | 0.0000 | 1 | 0 |
| real_shared_weight_latent_coordination | dev | none | role_labels_only_baseline | 0.2500 | 0.0000 | 1 | 1188 |
| real_shared_weight_latent_coordination | dev | none | single_agent_full_context | 0.2812 | 0.0000 | 1 | 136580 |
| real_shared_weight_latent_coordination | dev | none | single_agent_partial_view | 0.2500 | 0.0000 | 1 | 136580 |
| real_shared_weight_latent_coordination | dev | none | single_view_text_role_0 | 0.2812 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | dev | none | single_view_text_role_1 | 0.2500 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | dev | none | single_view_text_role_2 | 0.2500 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | dev | none | single_view_text_role_3 | 0.2812 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | dev | none | text_only_partial_view_baseline | 0.2812 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | dev | none | text_output_only_multi_agent_coordinator | 0.2500 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | dev | none | trainable_shared_agent_latent_coordinator | 0.2500 | 0.0000 | 1 | 174660 |
| real_shared_weight_latent_coordination | dev | physical_order_shuffled_roles_preserved | trainable_shared_agent_latent_coordinator | 0.2500 | 0.0000 | 1 | 174660 |
| real_shared_weight_latent_coordination | dev | randomized_labels | trainable_shared_agent_latent_coordinator | 0.2500 | 0.0000 | 1 | 174660 |
| real_shared_weight_latent_coordination | dev | role_labels_shuffled | trainable_shared_agent_latent_coordinator | 0.2500 | 0.0000 | 1 | 174660 |
| real_shared_weight_latent_coordination | dev | view_masked | masked_evidence_text_baseline | 0.3125 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | dev | view_masked | trainable_shared_agent_latent_coordinator | 0.2500 | 0.0000 | 1 | 174660 |
| real_shared_weight_latent_coordination | dev | view_shuffled | trainable_shared_agent_latent_coordinator | 0.2500 | 0.0000 | 1 | 174660 |
| real_shared_weight_latent_coordination | dev | visible_explicit_evidence | trainable_shared_agent_latent_visible_explicit_evidence | 0.2500 | 0.0000 | 1 | 174660 |
| real_shared_weight_latent_coordination | test | candidate_order_shuffled | trainable_shared_agent_latent_coordinator | 0.1250 | 0.0000 | 1 | 174660 |
| real_shared_weight_latent_coordination | test | explicit_evidence | explicit_evidence_neural_tuple_model | 1.0000 | 0.0000 | 1 | 3265 |
| real_shared_weight_latent_coordination | test | explicit_evidence | raw_structured_evidence_coordinator | 1.0000 | 0.0000 | 1 | 3201 |
| real_shared_weight_latent_coordination | test | hidden_states_shuffled_across_examples | trainable_shared_agent_latent_coordinator | 0.2500 | 0.0000 | 1 | 174660 |
| real_shared_weight_latent_coordination | test | none | bag_of_words_candidate_patch_only | 0.2500 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | test | none | candidate_order_baseline | 0.2500 | 0.0000 | 1 | 0 |
| real_shared_weight_latent_coordination | test | none | coordinator_only_probe | 0.2500 | 0.0000 | 1 | 420 |
| real_shared_weight_latent_coordination | test | none | explicit_evidence_oracle | 1.0000 | 0.0000 | 1 | 0 |
| real_shared_weight_latent_coordination | test | none | frozen_shared_agent_latent_coordinator | 0.2500 | 0.0000 | 1 | 38340 |
| real_shared_weight_latent_coordination | test | none | larger_tiny_transformer_full_context | 0.2500 | 0.0000 | 1 | 136580 |
| real_shared_weight_latent_coordination | test | none | majority_class_baseline | 0.2500 | 0.0000 | 1 | 0 |
| real_shared_weight_latent_coordination | test | none | role_labels_only_baseline | 0.2500 | 0.0000 | 1 | 1188 |
| real_shared_weight_latent_coordination | test | none | single_agent_full_context | 0.2500 | 0.0000 | 1 | 136580 |
| real_shared_weight_latent_coordination | test | none | single_agent_partial_view | 0.2500 | 0.0000 | 1 | 136580 |
| real_shared_weight_latent_coordination | test | none | single_view_text_role_0 | 0.2812 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | test | none | single_view_text_role_1 | 0.2500 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | test | none | single_view_text_role_2 | 0.2500 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | test | none | single_view_text_role_3 | 0.2500 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | test | none | text_only_partial_view_baseline | 0.2500 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | test | none | text_output_only_multi_agent_coordinator | 0.2500 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | test | none | trainable_shared_agent_latent_coordinator | 0.2500 | 0.0000 | 1 | 174660 |
| real_shared_weight_latent_coordination | test | physical_order_shuffled_roles_preserved | trainable_shared_agent_latent_coordinator | 0.2500 | 0.0000 | 1 | 174660 |
| real_shared_weight_latent_coordination | test | randomized_labels | trainable_shared_agent_latent_coordinator | 0.2500 | 0.0000 | 1 | 174660 |
| real_shared_weight_latent_coordination | test | role_labels_shuffled | trainable_shared_agent_latent_coordinator | 0.2500 | 0.0000 | 1 | 174660 |
| real_shared_weight_latent_coordination | test | view_masked | masked_evidence_text_baseline | 0.2500 | 0.0000 | 1 | 4260 |
| real_shared_weight_latent_coordination | test | view_masked | trainable_shared_agent_latent_coordinator | 0.2500 | 0.0000 | 1 | 174660 |
| real_shared_weight_latent_coordination | test | view_shuffled | trainable_shared_agent_latent_coordinator | 0.2500 | 0.0000 | 1 | 174660 |
| real_shared_weight_latent_coordination | test | visible_explicit_evidence | trainable_shared_agent_latent_visible_explicit_evidence | 0.2500 | 0.0000 | 1 | 174660 |

## Main Test Comparison

| benchmark | method | test mean acc | std | runs |
|---|---|---:|---:|---:|
| real_shared_weight_latent_coordination | explicit_evidence_oracle | 1.0000 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | single_view_text_role_0 | 0.2812 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | frozen_shared_agent_latent_coordinator | 0.2500 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | trainable_shared_agent_latent_coordinator | 0.2500 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | text_output_only_multi_agent_coordinator | 0.2500 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | majority_class_baseline | 0.2500 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | candidate_order_baseline | 0.2500 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | bag_of_words_candidate_patch_only | 0.2500 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | text_only_partial_view_baseline | 0.2500 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | single_view_text_role_1 | 0.2500 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | single_view_text_role_2 | 0.2500 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | single_view_text_role_3 | 0.2500 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | role_labels_only_baseline | 0.2500 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | single_agent_full_context | 0.2500 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | larger_tiny_transformer_full_context | 0.2500 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | single_agent_partial_view | 0.2500 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | coordinator_only_probe | 0.2500 | 0.0000 | 1 |

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
| single_view_text_role_0 | 0.2812 | 0.0000 | 1 |
| single_view_text_role_1 | 0.2500 | 0.0000 | 1 |
| single_view_text_role_2 | 0.2500 | 0.0000 | 1 |
| single_view_text_role_3 | 0.2500 | 0.0000 | 1 |
| text_only_partial_view_baseline | 0.2500 | 0.0000 | 1 |
| masked_evidence_text_baseline | 0.2500 | 0.0000 | 1 |

Control table before/after:
| condition | before acc | after acc |
|---|---:|---:|
| candidate_order_shuffled | n/a | 0.1250 |
| hidden_states_shuffled_across_examples | n/a | 0.2500 |
| physical_order_shuffled_roles_preserved | n/a | 0.2500 |
| randomized_labels | n/a | 0.2500 |
| role_labels_shuffled | n/a | 0.2500 |
| view_masked | n/a | 0.2500 |
| view_shuffled | n/a | 0.2500 |

Label and split audit:
| split | label counts | problem families |
|---|---|---|
| dev | `{'0': 8, '1': 8, '2': 8, '3': 8}` | `['date_bucket', 'retry_budget', 'token_counter']` |
| test | `{'0': 8, '1': 8, '2': 8, '3': 8}` | `['inventory_delta', 'pagination_cursor', 'permission_merge']` |
| train | `{'0': 16, '1': 16, '2': 16, '3': 16}` | `['cache_lookup', 'json_flag', 'list_window', 'metric_average', 'path_joiner', 'string_normalizer']` |

Correct examples under corrupted controls:
| condition | acc | inspected correct examples |
|---|---:|---|
| hidden_states_shuffled_across_examples | 0.2500 | `['test-00001', 'test-00005', 'test-00009', 'test-00013', 'test-00017']` |
| role_labels_shuffled | 0.2500 | `['test-00001', 'test-00005', 'test-00009', 'test-00013', 'test-00017']` |
| view_masked | 0.2500 | `['test-00001', 'test-00005', 'test-00009', 'test-00013', 'test-00017']` |
| view_shuffled | 0.2500 | `['test-00001', 'test-00005', 'test-00009', 'test-00013', 'test-00017']` |

Final gate status:
| gate | pass |
|---|---|
| gate_a_valid_training_graph | True |
| gate_b_frozen_control | True |
| gate_c_trainable_beats_frozen | False |
| gate_d_trainable_beats_text_only | False |
| gate_e_controls_collapse | True |
| gate_f_dataset_not_trivial | True |
| gate_positive_control_learnable | True |

## Learnability Positive Controls

These controls test whether the repaired benchmark is learnable before tuning latent architecture.

| method | condition | test acc | std | runs | params |
|---|---|---:|---:|---:|---:|
| explicit_evidence_neural_tuple_model | explicit_evidence | 1.0000 | 0.0000 | 1 | 3265 |
| raw_structured_evidence_coordinator | explicit_evidence | 1.0000 | 0.0000 | 1 | 3201 |
| larger_tiny_transformer_full_context | none | 0.2500 | 0.0000 | 1 | 136580 |
| single_agent_full_context | none | 0.2500 | 0.0000 | 1 | 136580 |
| trainable_shared_agent_latent_visible_explicit_evidence | visible_explicit_evidence | 0.2500 | 0.0000 | 1 | 174660 |

Learnability gate `gate_positive_control_learnable`: `True`; best learned positive-control test accuracy `1.0000`.
At least one learned positive control solved the benchmark, so subsequent failures are more likely model/architecture/training issues than label construction issues.

## Real Shared-Weight Latent Coordination

This section is governed by the real shared-weight latent coordination specification. The available local data did not include SWE-bench or issue-patch artifacts, so the run uses a constructed local-code patch-selection fallback and is not treated as publishable real-world proof.

Architecture and data:
- Agent model mode: `pretrained_adapter`; model path: `EleutherAI/pythia-70m-deduped`; hidden dim: `64`.
- Trainable parameters: shared agent coordination parameters plus coordinator in trainable runs; frozen runs keep shared agent deltas at zero.
- Dataset source: `constructed_local_code_patch_selection`; split sizes: `{'train': 64, 'dev': 32, 'test': 32}`; source files used: `12`.
- Batch settings: batch size `2`, epochs `1`, grad accumulation `4`.

Main comparison:
| method | test mean acc | std | runs | params |
|---|---:|---:|---:|---:|
| explicit_evidence_oracle | 1.0000 | 0.0000 | 1 | 0 |
| single_view_text_role_0 | 0.2812 | 0.0000 | 1 | 4260 |
| bag_of_words_candidate_patch_only | 0.2500 | 0.0000 | 1 | 4260 |
| candidate_order_baseline | 0.2500 | 0.0000 | 1 | 0 |
| coordinator_only_probe | 0.2500 | 0.0000 | 1 | 420 |
| frozen_shared_agent_latent_coordinator | 0.2500 | 0.0000 | 1 | 38340 |
| larger_tiny_transformer_full_context | 0.2500 | 0.0000 | 1 | 136580 |
| majority_class_baseline | 0.2500 | 0.0000 | 1 | 0 |
| role_labels_only_baseline | 0.2500 | 0.0000 | 1 | 1188 |
| single_agent_full_context | 0.2500 | 0.0000 | 1 | 136580 |
| single_agent_partial_view | 0.2500 | 0.0000 | 1 | 136580 |
| single_view_text_role_1 | 0.2500 | 0.0000 | 1 | 4260 |
| single_view_text_role_2 | 0.2500 | 0.0000 | 1 | 4260 |
| single_view_text_role_3 | 0.2500 | 0.0000 | 1 | 4260 |
| text_only_partial_view_baseline | 0.2500 | 0.0000 | 1 | 4260 |
| text_output_only_multi_agent_coordinator | 0.2500 | 0.0000 | 1 | 4260 |
| trainable_shared_agent_latent_coordinator | 0.2500 | 0.0000 | 1 | 174660 |

Audit table:
| method | condition | shared params | M grad | M delta | C grad | C delta | activations grad | no detach | per-clone grad | memory bytes |
|---|---|---|---:|---:|---:|---:|---|---|---|---:|
| frozen_shared_agent_latent_coordinator | none | pass | 0.000000 | 0.000000 | 3.708388 | 0.229353 | fail | fail | fail | 428324352 |
| trainable_shared_agent_latent_coordinator | none | pass | 3.295086 | 0.546476 | 3.905973 | 0.213011 | pass | pass | pass | 520116736 |
| trainable_shared_agent_latent_coordinator | randomized_labels | pass | 3.469192 | 0.526848 | 3.148672 | 0.268912 | pass | pass | pass | 612128256 |
| trainable_shared_agent_latent_visible_explicit_evidence | visible_explicit_evidence | pass | 3.664792 | 0.541025 | 3.798757 | 0.289119 | pass | pass | pass | 701649408 |

Controls:
| condition | test mean acc | std | runs |
|---|---:|---:|---:|
| candidate_order_shuffled | 0.1250 | 0.0000 | 1 |
| hidden_states_shuffled_across_examples | 0.2500 | 0.0000 | 1 |
| physical_order_shuffled_roles_preserved | 0.2500 | 0.0000 | 1 |
| randomized_labels | 0.2500 | 0.0000 | 1 |
| role_labels_shuffled | 0.2500 | 0.0000 | 1 |
| view_masked | 0.2500 | 0.0000 | 1 |
| view_shuffled | 0.2500 | 0.0000 | 1 |

Learning curves:
| method | condition | split | first epoch acc | final epoch acc | epochs |
|---|---|---|---:|---:|---:|
| frozen_shared_agent_latent_coordinator | none | dev | 0.2500 | 0.2500 | 1 |
| frozen_shared_agent_latent_coordinator | none | train | 0.2500 | 0.2500 | 1 |
| trainable_shared_agent_latent_coordinator | none | dev | 0.2500 | 0.2500 | 1 |
| trainable_shared_agent_latent_coordinator | none | train | 0.2500 | 0.2500 | 1 |
| trainable_shared_agent_latent_visible_explicit_evidence | visible_explicit_evidence | dev | 0.2500 | 0.2500 | 1 |
| trainable_shared_agent_latent_visible_explicit_evidence | visible_explicit_evidence | train | 0.2500 | 0.2500 | 1 |

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
| gate_d_trainable_beats_text_only | False |
| gate_e_controls_collapse | True |
| gate_f_dataset_not_trivial | True |

Checklist:
1. Exact model: `pretrained_adapter` with `EleutherAI/pythia-70m-deduped` when pretrained loading is enabled.
2. Trainable parameters: `['pretrained_model.layers.0.attention.query_key_value.lora_a.weight', 'pretrained_model.layers.0.attention.query_key_value.lora_b.weight', 'pretrained_model.layers.0.attention.dense.lora_a.weight', 'pretrained_model.layers.0.attention.dense.lora_b.weight', 'pretrained_model.layers.0.mlp.dense_h_to_4h.lora_a.weight', 'pretrained_model.layers.0.mlp.dense_h_to_4h.lora_b.weight', 'pretrained_model.layers.0.mlp.dense_4h_to_h.lora_a.weight', 'pretrained_model.layers.0.mlp.dense_4h_to_h.lora_b.weight', 'pretrained_model.layers.1.attention.query_key_value.lora_a.weight', 'pretrained_model.layers.1.attention.query_key_value.lora_b.weight', 'pretrained_model.layers.1.attention.dense.lora_a.weight', 'pretrained_model.layers.1.attention.dense.lora_b.weight', 'pretrained_model.layers.1.mlp.dense_h_to_4h.lora_a.weight', 'pretrained_model.layers.1.mlp.dense_h_to_4h.lora_b.weight', 'pretrained_model.layers.1.mlp.dense_4h_to_h.lora_a.weight', 'pretrained_model.layers.1.mlp.dense_4h_to_h.lora_b.weight', 'pretrained_model.layers.2.attention.query_key_value.lora_a.weight', 'pretrained_model.layers.2.attention.query_key_value.lora_b.weight', 'pretrained_model.layers.2.attention.dense.lora_a.weight', 'pretrained_model.layers.2.attention.dense.lora_b.weight', 'pretrained_model.layers.2.mlp.dense_h_to_4h.lora_a.weight', 'pretrained_model.layers.2.mlp.dense_h_to_4h.lora_b.weight', 'pretrained_model.layers.2.mlp.dense_4h_to_h.lora_a.weight', 'pretrained_model.layers.2.mlp.dense_4h_to_h.lora_b.weight', 'pretrained_model.layers.3.attention.query_key_value.lora_a.weight', 'pretrained_model.layers.3.attention.query_key_value.lora_b.weight', 'pretrained_model.layers.3.attention.dense.lora_a.weight', 'pretrained_model.layers.3.attention.dense.lora_b.weight', 'pretrained_model.layers.3.mlp.dense_h_to_4h.lora_a.weight', 'pretrained_model.layers.3.mlp.dense_h_to_4h.lora_b.weight', 'pretrained_model.layers.3.mlp.dense_4h_to_h.lora_a.weight', 'pretrained_model.layers.3.mlp.dense_4h_to_h.lora_b.weight', 'pretrained_model.layers.4.attention.query_key_value.lora_a.weight', 'pretrained_model.layers.4.attention.query_key_value.lora_b.weight', 'pretrained_model.layers.4.attention.dense.lora_a.weight', 'pretrained_model.layers.4.attention.dense.lora_b.weight', 'pretrained_model.layers.4.mlp.dense_h_to_4h.lora_a.weight', 'pretrained_model.layers.4.mlp.dense_h_to_4h.lora_b.weight', 'pretrained_model.layers.4.mlp.dense_4h_to_h.lora_a.weight', 'pretrained_model.layers.4.mlp.dense_4h_to_h.lora_b.weight', 'pretrained_model.layers.5.attention.query_key_value.lora_a.weight', 'pretrained_model.layers.5.attention.query_key_value.lora_b.weight', 'pretrained_model.layers.5.attention.dense.lora_a.weight', 'pretrained_model.layers.5.attention.dense.lora_b.weight', 'pretrained_model.layers.5.mlp.dense_h_to_4h.lora_a.weight', 'pretrained_model.layers.5.mlp.dense_h_to_4h.lora_b.weight', 'pretrained_model.layers.5.mlp.dense_4h_to_h.lora_a.weight', 'pretrained_model.layers.5.mlp.dense_4h_to_h.lora_b.weight', 'activation_adapter.0.weight', 'activation_adapter.0.bias', 'activation_adapter.1.weight', 'activation_adapter.1.bias', 'activation_adapter.3.weight', 'activation_adapter.3.bias']`.
3. Shared across clones: `True`.
4. Nonzero shared gradients: `True`.
5. Shared parameter delta: `0.546476`.
6. Trainable beat frozen: `False`.
7. Trainable beat text-only: `False`.
8. Controls collapse: inspect the controls table; this fallback run does not by itself authorize a positive real-world claim.
9. Physical order shuffle with roles preserved: inspect `physical_order_shuffled_roles_preserved`.
10. Role-label shuffle: inspect `role_labels_shuffled`.
11. Single partial-view accuracy: `0.25`.
12. Output leakage: `True`.
13. Train/dev/test leakage: `True`.
14. GPU memory: `701649408` bytes.
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
| real_shared_weight_latent_coordination | 0 | dev | candidate_order_shuffled | trainable_shared_agent_latent_coordinator | 0.2500 | 174660 |
| real_shared_weight_latent_coordination | 0 | dev | explicit_evidence | explicit_evidence_neural_tuple_model | 1.0000 | 3265 |
| real_shared_weight_latent_coordination | 0 | dev | explicit_evidence | raw_structured_evidence_coordinator | 1.0000 | 3201 |
| real_shared_weight_latent_coordination | 0 | dev | hidden_states_shuffled_across_examples | trainable_shared_agent_latent_coordinator | 0.2500 | 174660 |
| real_shared_weight_latent_coordination | 0 | dev | none | bag_of_words_candidate_patch_only | 0.2500 | 4260 |
| real_shared_weight_latent_coordination | 0 | dev | none | candidate_order_baseline | 0.2500 | 0 |
| real_shared_weight_latent_coordination | 0 | dev | none | coordinator_only_probe | 0.2500 | 420 |
| real_shared_weight_latent_coordination | 0 | dev | none | explicit_evidence_oracle | 1.0000 | 0 |
| real_shared_weight_latent_coordination | 0 | dev | none | frozen_shared_agent_latent_coordinator | 0.2500 | 38340 |
| real_shared_weight_latent_coordination | 0 | dev | none | larger_tiny_transformer_full_context | 0.2500 | 136580 |
| real_shared_weight_latent_coordination | 0 | dev | none | majority_class_baseline | 0.2500 | 0 |
| real_shared_weight_latent_coordination | 0 | dev | none | role_labels_only_baseline | 0.2500 | 1188 |
| real_shared_weight_latent_coordination | 0 | dev | none | single_agent_full_context | 0.2812 | 136580 |
| real_shared_weight_latent_coordination | 0 | dev | none | single_agent_partial_view | 0.2500 | 136580 |
| real_shared_weight_latent_coordination | 0 | dev | none | single_view_text_role_0 | 0.2812 | 4260 |
| real_shared_weight_latent_coordination | 0 | dev | none | single_view_text_role_1 | 0.2500 | 4260 |
| real_shared_weight_latent_coordination | 0 | dev | none | single_view_text_role_2 | 0.2500 | 4260 |
| real_shared_weight_latent_coordination | 0 | dev | none | single_view_text_role_3 | 0.2812 | 4260 |
| real_shared_weight_latent_coordination | 0 | dev | none | text_only_partial_view_baseline | 0.2812 | 4260 |
| real_shared_weight_latent_coordination | 0 | dev | none | text_output_only_multi_agent_coordinator | 0.2500 | 4260 |
| real_shared_weight_latent_coordination | 0 | dev | none | trainable_shared_agent_latent_coordinator | 0.2500 | 174660 |
| real_shared_weight_latent_coordination | 0 | dev | physical_order_shuffled_roles_preserved | trainable_shared_agent_latent_coordinator | 0.2500 | 174660 |
| real_shared_weight_latent_coordination | 0 | dev | randomized_labels | trainable_shared_agent_latent_coordinator | 0.2500 | 174660 |
| real_shared_weight_latent_coordination | 0 | dev | role_labels_shuffled | trainable_shared_agent_latent_coordinator | 0.2500 | 174660 |
| real_shared_weight_latent_coordination | 0 | dev | view_masked | masked_evidence_text_baseline | 0.3125 | 4260 |
| real_shared_weight_latent_coordination | 0 | dev | view_masked | trainable_shared_agent_latent_coordinator | 0.2500 | 174660 |
| real_shared_weight_latent_coordination | 0 | dev | view_shuffled | trainable_shared_agent_latent_coordinator | 0.2500 | 174660 |
| real_shared_weight_latent_coordination | 0 | dev | visible_explicit_evidence | trainable_shared_agent_latent_visible_explicit_evidence | 0.2500 | 174660 |
| real_shared_weight_latent_coordination | 0 | test | candidate_order_shuffled | trainable_shared_agent_latent_coordinator | 0.1250 | 174660 |
| real_shared_weight_latent_coordination | 0 | test | explicit_evidence | explicit_evidence_neural_tuple_model | 1.0000 | 3265 |
| real_shared_weight_latent_coordination | 0 | test | explicit_evidence | raw_structured_evidence_coordinator | 1.0000 | 3201 |
| real_shared_weight_latent_coordination | 0 | test | hidden_states_shuffled_across_examples | trainable_shared_agent_latent_coordinator | 0.2500 | 174660 |
| real_shared_weight_latent_coordination | 0 | test | none | bag_of_words_candidate_patch_only | 0.2500 | 4260 |
| real_shared_weight_latent_coordination | 0 | test | none | candidate_order_baseline | 0.2500 | 0 |
| real_shared_weight_latent_coordination | 0 | test | none | coordinator_only_probe | 0.2500 | 420 |
| real_shared_weight_latent_coordination | 0 | test | none | explicit_evidence_oracle | 1.0000 | 0 |
| real_shared_weight_latent_coordination | 0 | test | none | frozen_shared_agent_latent_coordinator | 0.2500 | 38340 |
| real_shared_weight_latent_coordination | 0 | test | none | larger_tiny_transformer_full_context | 0.2500 | 136580 |
| real_shared_weight_latent_coordination | 0 | test | none | majority_class_baseline | 0.2500 | 0 |
| real_shared_weight_latent_coordination | 0 | test | none | role_labels_only_baseline | 0.2500 | 1188 |
| real_shared_weight_latent_coordination | 0 | test | none | single_agent_full_context | 0.2500 | 136580 |
| real_shared_weight_latent_coordination | 0 | test | none | single_agent_partial_view | 0.2500 | 136580 |
| real_shared_weight_latent_coordination | 0 | test | none | single_view_text_role_0 | 0.2812 | 4260 |
| real_shared_weight_latent_coordination | 0 | test | none | single_view_text_role_1 | 0.2500 | 4260 |
| real_shared_weight_latent_coordination | 0 | test | none | single_view_text_role_2 | 0.2500 | 4260 |
| real_shared_weight_latent_coordination | 0 | test | none | single_view_text_role_3 | 0.2500 | 4260 |
| real_shared_weight_latent_coordination | 0 | test | none | text_only_partial_view_baseline | 0.2500 | 4260 |
| real_shared_weight_latent_coordination | 0 | test | none | text_output_only_multi_agent_coordinator | 0.2500 | 4260 |
| real_shared_weight_latent_coordination | 0 | test | none | trainable_shared_agent_latent_coordinator | 0.2500 | 174660 |
| real_shared_weight_latent_coordination | 0 | test | physical_order_shuffled_roles_preserved | trainable_shared_agent_latent_coordinator | 0.2500 | 174660 |
| real_shared_weight_latent_coordination | 0 | test | randomized_labels | trainable_shared_agent_latent_coordinator | 0.2500 | 174660 |
| real_shared_weight_latent_coordination | 0 | test | role_labels_shuffled | trainable_shared_agent_latent_coordinator | 0.2500 | 174660 |
| real_shared_weight_latent_coordination | 0 | test | view_masked | masked_evidence_text_baseline | 0.2500 | 4260 |
| real_shared_weight_latent_coordination | 0 | test | view_masked | trainable_shared_agent_latent_coordinator | 0.2500 | 174660 |
| real_shared_weight_latent_coordination | 0 | test | view_shuffled | trainable_shared_agent_latent_coordinator | 0.2500 | 174660 |
| real_shared_weight_latent_coordination | 0 | test | visible_explicit_evidence | trainable_shared_agent_latent_visible_explicit_evidence | 0.2500 | 174660 |

## Mechanism Distinction


## Synthetic Vs Transformer Hidden States

No captured-transformer hidden-state stage is present in this report.

## Interpretation

- Recommended next experiment: make the transformer strict task less templated while keeping private evidence hidden from visible outputs, then repeat the same access audit, randomized-label sanity check, output-only baselines, and individual-vs-combined hidden-state probes.
