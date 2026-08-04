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
    "audit_log_path": "results/real_shared_weight_latent_coordination_audit.jsonl",
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
      "cross_attention",
      "self_attention"
    ],
    "dataset": {
      "include_private_signal_tokens": true,
      "max_files": 40,
      "n_dev": 96,
      "n_test": 128,
      "n_train": 256,
      "n_views": 4,
      "num_candidates": 8,
      "snippet_radius": 2,
      "source_roots": [
        "src",
        "tests"
      ]
    },
    "deterministic": true,
    "device": "cpu",
    "message_channel": {
      "aux_decay_epochs": 4,
      "aux_loss_weight": 0.05,
      "aux_warmup_epochs": 2,
      "message_dim": 64,
      "msg_position": "prepend"
    },
    "output_path": "results/real_shared_weight_latent_coordination_results.json",
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
    "report_path": "reports/REPORT.md",
    "run_layer_token_sweep_variants": true,
    "run_message_channel_variants": true,
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
  "config_path": "configs\\real_shared_weight_latent_coordination_cpu_debug.json",
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
    "num_candidates": 8,
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
      "num_candidates": 8,
      "num_views": 4,
      "private_cue_style": "explicit_signal_token",
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
  "message_channel_config": {
    "active_message_dropout": 0.0,
    "active_message_ff_dim": 128,
    "active_message_heads": 2,
    "active_message_layers": [
      -1
    ],
    "aux_decay_epochs": 4,
    "aux_loss_weight": 0.05,
    "aux_warmup_epochs": 2,
    "coordinator_family": "latent",
    "message_dim": 64,
    "msg_position": "prepend",
    "readout_source": "msg",
    "use_message_head": false,
    "use_msg_token": false,
    "use_private_cue_aux": false
  },
  "previous_run_summary": {
    "control_accuracy": {
      "candidate_order_shuffled": 0.1484375,
      "hidden_states_shuffled_across_examples": 0.1484375,
      "physical_order_shuffled_roles_preserved": 0.1484375,
      "randomized_labels": 0.1171875,
      "role_labels_shuffled": 0.1484375,
      "view_masked": 0.1484375,
      "view_shuffled": 0.1484375
    },
    "gate_e_controls_collapse": true,
    "gate_f_dataset_not_trivial": true,
    "path": "results\\real_shared_weight_latent_coordination_results.json",
    "test_accuracy": {
      "bag_of_words_candidate_patch_only": 0.125,
      "candidate_order_baseline": 0.125,
      "coordinator_only_probe": 0.1484375,
      "explicit_evidence_oracle": 1.0,
      "frozen_shared_agent_evidence_token_msg_head_aux_candidate_query": 1.0,
      "frozen_shared_agent_latent_coordinator": 0.125,
      "frozen_shared_agent_latent_coordinator_self_attention": 0.125,
      "frozen_shared_agent_msg_head_aux_candidate_query": 0.1484375,
      "larger_tiny_transformer_full_context": 0.09375,
      "majority_class_baseline": 0.125,
      "role_labels_only_baseline": 0.125,
      "single_agent_full_context": 0.1171875,
      "single_agent_partial_view": 0.125,
      "single_view_text_role_0": 0.1171875,
      "single_view_text_role_1": 0.125,
      "single_view_text_role_2": 0.140625,
      "single_view_text_role_3": 0.125,
      "text_only_partial_view_baseline": 0.125,
      "text_output_only_multi_agent_coordinator": 0.125,
      "trainable_shared_agent_append_msg_head_aux_candidate_query": 0.125,
      "trainable_shared_agent_evidence_token_msg_head_aux_candidate_query": 1.0,
      "trainable_shared_agent_latent_coordinator": 0.15625,
      "trainable_shared_agent_latent_coordinator_self_attention": 0.1328125,
      "trainable_shared_agent_msg_head": 0.125,
      "trainable_shared_agent_msg_head_aux": 0.125,
      "trainable_shared_agent_msg_head_aux_candidate_query": 0.1484375,
      "trainable_shared_agent_msg_head_candidate_query_no_aux": 0.1484375,
      "trainable_shared_agent_msg_token_only": 0.140625
    },
    "trivial_baseline_accuracy": {
      "bag_of_words_candidate_patch_only": 0.125,
      "candidate_order_baseline": 0.125,
      "majority_class_baseline": 0.125,
      "masked_evidence_text_baseline": 0.125,
      "role_labels_only_baseline": 0.125,
      "single_view_text_role_0": 0.1171875,
      "single_view_text_role_1": 0.125,
      "single_view_text_role_2": 0.140625,
      "single_view_text_role_3": 0.125,
      "text_only_partial_view_baseline": 0.125
    }
  },
  "proposed_frozen_method": "frozen_shared_agent_active_msg_head_aux_candidate_query",
  "proposed_latent_method": "trainable_shared_agent_active_msg_head_aux_candidate_query",
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
| real_shared_weight_latent_coordination | dev | candidate_order_shuffled | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.1250 | 0.0000 | 1 | 295059 |
| real_shared_weight_latent_coordination | dev | explicit_evidence | explicit_evidence_neural_tuple_model | 1.0000 | 0.0000 | 1 | 3521 |
| real_shared_weight_latent_coordination | dev | explicit_evidence | raw_structured_evidence_coordinator | 1.0000 | 0.0000 | 1 | 3457 |
| real_shared_weight_latent_coordination | dev | hidden_states_shuffled_across_examples | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.1250 | 0.0000 | 1 | 295059 |
| real_shared_weight_latent_coordination | dev | none | bag_of_words_candidate_patch_only | 0.1250 | 0.0000 | 1 | 4392 |
| real_shared_weight_latent_coordination | dev | none | candidate_order_baseline | 0.1250 | 0.0000 | 1 | 0 |
| real_shared_weight_latent_coordination | dev | none | coordinator_only_probe | 0.1354 | 0.0000 | 1 | 552 |
| real_shared_weight_latent_coordination | dev | none | explicit_evidence_oracle | 1.0000 | 0.0000 | 1 | 0 |
| real_shared_weight_latent_coordination | dev | none | frozen_shared_agent_active_msg_head_aux_candidate_query | 0.1146 | 0.0000 | 1 | 68451 |
| real_shared_weight_latent_coordination | dev | none | frozen_shared_agent_evidence_token_diagnostic_upper_bound | 1.0000 | 0.0000 | 1 | 46051 |
| real_shared_weight_latent_coordination | dev | none | frozen_shared_agent_latent_coordinator | 0.1250 | 0.0000 | 1 | 37576 |
| real_shared_weight_latent_coordination | dev | none | frozen_shared_agent_latent_coordinator_self_attention | 0.1250 | 0.0000 | 1 | 37576 |
| real_shared_weight_latent_coordination | dev | none | larger_tiny_transformer_full_context | 0.1562 | 0.0000 | 1 | 574856 |
| real_shared_weight_latent_coordination | dev | none | majority_class_baseline | 0.1250 | 0.0000 | 1 | 0 |
| real_shared_weight_latent_coordination | dev | none | role_labels_only_baseline | 0.1250 | 0.0000 | 1 | 1320 |
| real_shared_weight_latent_coordination | dev | none | single_agent_full_context | 0.1667 | 0.0000 | 1 | 227000 |
| real_shared_weight_latent_coordination | dev | none | single_agent_partial_view | 0.1250 | 0.0000 | 1 | 227000 |
| real_shared_weight_latent_coordination | dev | none | single_view_text_role_0 | 0.1458 | 0.0000 | 1 | 4392 |
| real_shared_weight_latent_coordination | dev | none | single_view_text_role_1 | 0.1562 | 0.0000 | 1 | 4392 |
| real_shared_weight_latent_coordination | dev | none | single_view_text_role_2 | 0.1250 | 0.0000 | 1 | 4392 |
| real_shared_weight_latent_coordination | dev | none | single_view_text_role_3 | 0.1250 | 0.0000 | 1 | 4392 |
| real_shared_weight_latent_coordination | dev | none | text_only_partial_view_baseline | 0.1250 | 0.0000 | 1 | 4392 |
| real_shared_weight_latent_coordination | dev | none | text_output_only_multi_agent_coordinator | 0.1562 | 0.0000 | 1 | 4392 |
| real_shared_weight_latent_coordination | dev | none | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.1250 | 0.0000 | 1 | 295059 |
| real_shared_weight_latent_coordination | dev | none | trainable_shared_agent_active_msg_head_candidate_query_no_aux | 0.2812 | 0.0000 | 1 | 294801 |
| real_shared_weight_latent_coordination | dev | none | trainable_shared_agent_evidence_token_diagnostic_upper_bound | 1.0000 | 0.0000 | 1 | 272659 |
| real_shared_weight_latent_coordination | dev | none | trainable_shared_agent_latent_coordinator | 0.1458 | 0.0000 | 1 | 264184 |
| real_shared_weight_latent_coordination | dev | none | trainable_shared_agent_latent_coordinator_self_attention | 0.1562 | 0.0000 | 1 | 264184 |
| real_shared_weight_latent_coordination | dev | physical_order_shuffled_roles_preserved | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.1250 | 0.0000 | 1 | 295059 |
| real_shared_weight_latent_coordination | dev | randomized_labels | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.1042 | 0.0000 | 1 | 295059 |
| real_shared_weight_latent_coordination | dev | randomized_labels | trainable_shared_agent_latent_coordinator | 0.1250 | 0.0000 | 1 | 264184 |
| real_shared_weight_latent_coordination | dev | role_labels_shuffled | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.1250 | 0.0000 | 1 | 295059 |
| real_shared_weight_latent_coordination | dev | view_masked | masked_evidence_text_baseline | 0.1250 | 0.0000 | 1 | 4392 |
| real_shared_weight_latent_coordination | dev | view_masked | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.1250 | 0.0000 | 1 | 295059 |
| real_shared_weight_latent_coordination | dev | view_shuffled | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.1250 | 0.0000 | 1 | 295059 |
| real_shared_weight_latent_coordination | dev | visible_explicit_evidence | trainable_shared_agent_latent_visible_explicit_evidence | 0.1250 | 0.0000 | 1 | 264184 |
| real_shared_weight_latent_coordination | test | candidate_order_shuffled | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.0781 | 0.0000 | 1 | 295059 |
| real_shared_weight_latent_coordination | test | explicit_evidence | explicit_evidence_neural_tuple_model | 1.0000 | 0.0000 | 1 | 3521 |
| real_shared_weight_latent_coordination | test | explicit_evidence | raw_structured_evidence_coordinator | 1.0000 | 0.0000 | 1 | 3457 |
| real_shared_weight_latent_coordination | test | hidden_states_shuffled_across_examples | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.0781 | 0.0000 | 1 | 295059 |
| real_shared_weight_latent_coordination | test | none | bag_of_words_candidate_patch_only | 0.1250 | 0.0000 | 1 | 4392 |
| real_shared_weight_latent_coordination | test | none | candidate_order_baseline | 0.1250 | 0.0000 | 1 | 0 |
| real_shared_weight_latent_coordination | test | none | coordinator_only_probe | 0.1562 | 0.0000 | 1 | 552 |
| real_shared_weight_latent_coordination | test | none | explicit_evidence_oracle | 1.0000 | 0.0000 | 1 | 0 |
| real_shared_weight_latent_coordination | test | none | frozen_shared_agent_active_msg_head_aux_candidate_query | 0.1641 | 0.0000 | 1 | 68451 |
| real_shared_weight_latent_coordination | test | none | frozen_shared_agent_evidence_token_diagnostic_upper_bound | 0.9297 | 0.0000 | 1 | 46051 |
| real_shared_weight_latent_coordination | test | none | frozen_shared_agent_latent_coordinator | 0.1250 | 0.0000 | 1 | 37576 |
| real_shared_weight_latent_coordination | test | none | frozen_shared_agent_latent_coordinator_self_attention | 0.1250 | 0.0000 | 1 | 37576 |
| real_shared_weight_latent_coordination | test | none | larger_tiny_transformer_full_context | 0.0703 | 0.0000 | 1 | 574856 |
| real_shared_weight_latent_coordination | test | none | majority_class_baseline | 0.1250 | 0.0000 | 1 | 0 |
| real_shared_weight_latent_coordination | test | none | role_labels_only_baseline | 0.1250 | 0.0000 | 1 | 1320 |
| real_shared_weight_latent_coordination | test | none | single_agent_full_context | 0.1172 | 0.0000 | 1 | 227000 |
| real_shared_weight_latent_coordination | test | none | single_agent_partial_view | 0.1250 | 0.0000 | 1 | 227000 |
| real_shared_weight_latent_coordination | test | none | single_view_text_role_0 | 0.1172 | 0.0000 | 1 | 4392 |
| real_shared_weight_latent_coordination | test | none | single_view_text_role_1 | 0.1172 | 0.0000 | 1 | 4392 |
| real_shared_weight_latent_coordination | test | none | single_view_text_role_2 | 0.1250 | 0.0000 | 1 | 4392 |
| real_shared_weight_latent_coordination | test | none | single_view_text_role_3 | 0.1250 | 0.0000 | 1 | 4392 |
| real_shared_weight_latent_coordination | test | none | text_only_partial_view_baseline | 0.1250 | 0.0000 | 1 | 4392 |
| real_shared_weight_latent_coordination | test | none | text_output_only_multi_agent_coordinator | 0.1250 | 0.0000 | 1 | 4392 |
| real_shared_weight_latent_coordination | test | none | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.0781 | 0.0000 | 1 | 295059 |
| real_shared_weight_latent_coordination | test | none | trainable_shared_agent_active_msg_head_candidate_query_no_aux | 0.3047 | 0.0000 | 1 | 294801 |
| real_shared_weight_latent_coordination | test | none | trainable_shared_agent_evidence_token_diagnostic_upper_bound | 1.0000 | 0.0000 | 1 | 272659 |
| real_shared_weight_latent_coordination | test | none | trainable_shared_agent_latent_coordinator | 0.1328 | 0.0000 | 1 | 264184 |
| real_shared_weight_latent_coordination | test | none | trainable_shared_agent_latent_coordinator_self_attention | 0.1250 | 0.0000 | 1 | 264184 |
| real_shared_weight_latent_coordination | test | physical_order_shuffled_roles_preserved | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.0781 | 0.0000 | 1 | 295059 |
| real_shared_weight_latent_coordination | test | randomized_labels | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.1172 | 0.0000 | 1 | 295059 |
| real_shared_weight_latent_coordination | test | randomized_labels | trainable_shared_agent_latent_coordinator | 0.1172 | 0.0000 | 1 | 264184 |
| real_shared_weight_latent_coordination | test | role_labels_shuffled | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.0781 | 0.0000 | 1 | 295059 |
| real_shared_weight_latent_coordination | test | view_masked | masked_evidence_text_baseline | 0.1250 | 0.0000 | 1 | 4392 |
| real_shared_weight_latent_coordination | test | view_masked | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.0781 | 0.0000 | 1 | 295059 |
| real_shared_weight_latent_coordination | test | view_shuffled | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.0781 | 0.0000 | 1 | 295059 |
| real_shared_weight_latent_coordination | test | visible_explicit_evidence | trainable_shared_agent_latent_visible_explicit_evidence | 0.1250 | 0.0000 | 1 | 264184 |

## Main Test Comparison

| benchmark | method | test mean acc | std | runs |
|---|---|---:|---:|---:|
| real_shared_weight_latent_coordination | trainable_shared_agent_evidence_token_diagnostic_upper_bound | 1.0000 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | explicit_evidence_oracle | 1.0000 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | frozen_shared_agent_evidence_token_diagnostic_upper_bound | 0.9297 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | trainable_shared_agent_active_msg_head_candidate_query_no_aux | 0.3047 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | frozen_shared_agent_active_msg_head_aux_candidate_query | 0.1641 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | coordinator_only_probe | 0.1562 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | trainable_shared_agent_latent_coordinator | 0.1328 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | frozen_shared_agent_latent_coordinator | 0.1250 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | frozen_shared_agent_latent_coordinator_self_attention | 0.1250 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | trainable_shared_agent_latent_coordinator_self_attention | 0.1250 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | text_output_only_multi_agent_coordinator | 0.1250 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | majority_class_baseline | 0.1250 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | candidate_order_baseline | 0.1250 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | bag_of_words_candidate_patch_only | 0.1250 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | text_only_partial_view_baseline | 0.1250 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | single_view_text_role_2 | 0.1250 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | single_view_text_role_3 | 0.1250 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | role_labels_only_baseline | 0.1250 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | single_agent_partial_view | 0.1250 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | single_view_text_role_0 | 0.1172 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | single_view_text_role_1 | 0.1172 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | single_agent_full_context | 0.1172 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.0781 | 0.0000 | 1 |
| real_shared_weight_latent_coordination | larger_tiny_transformer_full_context | 0.0703 | 0.0000 | 1 |

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
| bag_of_words_candidate_patch_only | 0.1250 | 0.0000 | 1 |
| candidate_order_baseline | 0.1250 | 0.0000 | 1 |
| majority_class_baseline | 0.1250 | 0.0000 | 1 |
| role_labels_only_baseline | 0.1250 | 0.0000 | 1 |
| single_view_text_role_0 | 0.1172 | 0.0000 | 1 |
| single_view_text_role_1 | 0.1172 | 0.0000 | 1 |
| single_view_text_role_2 | 0.1250 | 0.0000 | 1 |
| single_view_text_role_3 | 0.1250 | 0.0000 | 1 |
| text_only_partial_view_baseline | 0.1250 | 0.0000 | 1 |
| masked_evidence_text_baseline | 0.1250 | 0.0000 | 1 |

Control table before/after:
| condition | before acc | after acc |
|---|---:|---:|
| candidate_order_shuffled | 0.1484 | 0.0781 |
| hidden_states_shuffled_across_examples | 0.1484 | 0.0781 |
| physical_order_shuffled_roles_preserved | 0.1484 | 0.0781 |
| randomized_labels | 0.1172 | 0.1172 |
| role_labels_shuffled | 0.1484 | 0.0781 |
| view_masked | 0.1484 | 0.0781 |
| view_shuffled | 0.1484 | 0.0781 |

Label and split audit:
| split | label counts | problem families |
|---|---|---|
| dev | `{'0': 12, '1': 12, '2': 12, '3': 12, '4': 12, '5': 12, '6': 12, '7': 12}` | `['date_bucket', 'retry_budget', 'token_counter']` |
| test | `{'0': 16, '1': 16, '2': 16, '3': 16, '4': 16, '5': 16, '6': 16, '7': 16}` | `['inventory_delta', 'pagination_cursor', 'permission_merge']` |
| train | `{'0': 32, '1': 32, '2': 32, '3': 32, '4': 32, '5': 32, '6': 32, '7': 32}` | `['cache_lookup', 'json_flag', 'list_window', 'metric_average', 'path_joiner', 'string_normalizer']` |

Correct examples under corrupted controls:
| condition | acc | inspected correct examples |
|---|---:|---|
| hidden_states_shuffled_across_examples | 0.0781 | `['test-00001', 'test-00004', 'test-00013', 'test-00059', 'test-00088']` |
| role_labels_shuffled | 0.0781 | `['test-00001', 'test-00004', 'test-00013', 'test-00059', 'test-00088']` |
| view_masked | 0.0781 | `['test-00001', 'test-00004', 'test-00013', 'test-00059', 'test-00088']` |
| view_shuffled | 0.0781 | `['test-00001', 'test-00004', 'test-00013', 'test-00059', 'test-00088']` |

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
| explicit_evidence_neural_tuple_model | explicit_evidence | 1.0000 | 0.0000 | 1 | 3521 |
| raw_structured_evidence_coordinator | explicit_evidence | 1.0000 | 0.0000 | 1 | 3457 |
| trainable_shared_agent_latent_visible_explicit_evidence | visible_explicit_evidence | 0.1250 | 0.0000 | 1 | 264184 |
| single_agent_full_context | none | 0.1172 | 0.0000 | 1 | 227000 |
| larger_tiny_transformer_full_context | none | 0.0703 | 0.0000 | 1 | 574856 |

Learnability gate `gate_positive_control_learnable`: `True`; best learned positive-control test accuracy `1.0000`.
At least one learned positive control solved the benchmark, so subsequent failures are more likely model/architecture/training issues than label construction issues.

## Latent Message Channel Diagnostics

This section records active message-readout variants. Test probe rows are computed only after the test gate is opened; architecture decisions should use train/dev behavior.

Architecture tried:
| variant | method | architecture | why tried | diagnostics trigger |
|---|---|---|---|---|
| A | trainable_shared_agent_latent_coordinator | Current raw pooled hidden activation coordinator. | Required baseline for the failure diagnosis; it preserves the prior main path. | Required variant A. |
| B | trainable_shared_agent_active_msg_head_aux_candidate_query | Learned shared message query plus role embedding cross-attends over selected token hidden states, then passes through the shared message head and candidate-query coordinator. | Primary active-message readout test after passive [MSG] outputs collapsed. | Required active message readout architecture. |
| C | frozen_shared_agent_active_msg_head_aux_candidate_query | Frozen shared agent with the same active message readout, shared message head, auxiliary cue loss, and candidate-query coordinator. | Tests whether active readout extracts information already present in frozen token states. | Required frozen comparator with the exact same active readout. |
| D | trainable_shared_agent_active_msg_head_candidate_query_no_aux | Active message readout with shared message head and candidate-query coordinator, without private-cue auxiliary loss. | Ablates whether the auxiliary cue objective is necessary for active message formation. | Required first message-channel variant. |
| H | trainable_shared_agent_evidence_token_diagnostic_upper_bound | Evidence-token final-layer readout plus shared message head, auxiliary cue loss, and candidate-query cross-attention. | Diagnostic upper bound showing what direct cue-token extraction can solve. | Evidence-token diagnostic retained only as an upper bound. |
| I | frozen_shared_agent_evidence_token_diagnostic_upper_bound | Frozen shared agent with evidence-token readout and the same message head, auxiliary loss, and candidate-query coordinator. | Frozen comparator for the evidence-token upper bound; solving here means frozen token extraction, not shared-agent learning. | Evidence-token diagnostic retained only as an upper bound. |

Main result table:
| method | test acc | std | runs | params |
|---|---:|---:|---:|---:|
| trainable_shared_agent_latent_coordinator | 0.1328 | 0.0000 | 1 | 264184 |
| trainable_shared_agent_active_msg_head_aux_candidate_query | 0.0781 | 0.0000 | 1 | 295059 |
| frozen_shared_agent_active_msg_head_aux_candidate_query | 0.1641 | 0.0000 | 1 | 68451 |
| trainable_shared_agent_active_msg_head_candidate_query_no_aux | 0.3047 | 0.0000 | 1 | 294801 |
| trainable_shared_agent_evidence_token_diagnostic_upper_bound | 1.0000 | 0.0000 | 1 | 272659 |
| frozen_shared_agent_evidence_token_diagnostic_upper_bound | 0.9297 | 0.0000 | 1 | 46051 |

Active trainable-vs-frozen comparison:
| trainable method | frozen method | trainable acc | frozen acc | delta | interpretation |
|---|---|---:|---:|---:|---|
| trainable_shared_agent_active_msg_head_aux_candidate_query | frozen_shared_agent_active_msg_head_aux_candidate_query | 0.0781 | 0.1641 | -0.0859 | active_readout_not_established |

Control table:
| method | condition | test acc | std | runs |
|---|---|---:|---:|---:|
| trainable_shared_agent_active_msg_head_aux_candidate_query | candidate_order_shuffled | 0.0781 | 0.0000 | 1 |
| trainable_shared_agent_active_msg_head_aux_candidate_query | hidden_states_shuffled_across_examples | 0.0781 | 0.0000 | 1 |
| trainable_shared_agent_active_msg_head_aux_candidate_query | physical_order_shuffled_roles_preserved | 0.0781 | 0.0000 | 1 |
| trainable_shared_agent_active_msg_head_aux_candidate_query | randomized_labels | 0.1172 | 0.0000 | 1 |
| trainable_shared_agent_latent_coordinator | randomized_labels | 0.1172 | 0.0000 | 1 |
| trainable_shared_agent_active_msg_head_aux_candidate_query | role_labels_shuffled | 0.0781 | 0.0000 | 1 |
| trainable_shared_agent_active_msg_head_aux_candidate_query | view_masked | 0.0781 | 0.0000 | 1 |
| trainable_shared_agent_active_msg_head_aux_candidate_query | view_shuffled | 0.0781 | 0.0000 | 1 |

Mechanism probe table:
| probe | method | feature source | test mean acc | std | runs |
|---|---|---|---:|---:|---:|
| private cue | frozen_shared_agent_active_msg_head_aux_candidate_query | active_message_readout | 0.5137 | 0.0304 | 4 |
| private cue | frozen_shared_agent_active_msg_head_aux_candidate_query | message_head_output | 0.5137 | 0.0304 | 4 |
| private cue | frozen_shared_agent_active_msg_head_aux_candidate_query | raw_pooled_activation | 0.4844 | 0.0391 | 4 |
| private cue | frozen_shared_agent_evidence_token_diagnostic_upper_bound | evidence_token_activation | 0.9805 | 0.0338 | 4 |
| private cue | frozen_shared_agent_evidence_token_diagnostic_upper_bound | message_head_output | 0.9785 | 0.0372 | 4 |
| private cue | frozen_shared_agent_evidence_token_diagnostic_upper_bound | msg_token | 0.5391 | 0.0544 | 4 |
| private cue | frozen_shared_agent_evidence_token_diagnostic_upper_bound | raw_pooled_activation | 0.4941 | 0.0433 | 4 |
| private cue | trainable_shared_agent_active_msg_head_aux_candidate_query | active_message_readout | 0.5137 | 0.0304 | 4 |
| private cue | trainable_shared_agent_active_msg_head_aux_candidate_query | message_head_output | 0.5137 | 0.0304 | 4 |
| private cue | trainable_shared_agent_active_msg_head_aux_candidate_query | raw_pooled_activation | 0.4844 | 0.0544 | 4 |
| private cue | trainable_shared_agent_active_msg_head_candidate_query_no_aux | active_message_readout | 0.8594 | 0.1777 | 4 |
| private cue | trainable_shared_agent_active_msg_head_candidate_query_no_aux | message_head_output | 0.8594 | 0.1776 | 4 |
| private cue | trainable_shared_agent_active_msg_head_candidate_query_no_aux | raw_pooled_activation | 0.4805 | 0.0511 | 4 |
| private cue | trainable_shared_agent_evidence_token_diagnostic_upper_bound | evidence_token_activation | 0.9902 | 0.0169 | 4 |
| private cue | trainable_shared_agent_evidence_token_diagnostic_upper_bound | message_head_output | 0.9902 | 0.0169 | 4 |
| private cue | trainable_shared_agent_evidence_token_diagnostic_upper_bound | msg_token | 0.5078 | 0.0317 | 4 |
| private cue | trainable_shared_agent_evidence_token_diagnostic_upper_bound | raw_pooled_activation | 0.4785 | 0.0570 | 4 |
| private cue | trainable_shared_agent_latent_coordinator | raw_pooled_activation | 0.4941 | 0.0186 | 4 |
| combined final-label | frozen_shared_agent_active_msg_head_aux_candidate_query | combined_activation | 0.1250 | 0.0000 | 1 |
| combined final-label | frozen_shared_agent_active_msg_head_aux_candidate_query | combined_message | 0.1250 | 0.0000 | 1 |
| combined final-label | frozen_shared_agent_evidence_token_diagnostic_upper_bound | combined_activation | 0.1484 | 0.0000 | 1 |
| combined final-label | frozen_shared_agent_evidence_token_diagnostic_upper_bound | combined_message | 0.0703 | 0.0000 | 1 |
| combined final-label | trainable_shared_agent_active_msg_head_aux_candidate_query | combined_activation | 0.1484 | 0.0000 | 1 |
| combined final-label | trainable_shared_agent_active_msg_head_aux_candidate_query | combined_message | 0.1250 | 0.0000 | 1 |
| combined final-label | trainable_shared_agent_active_msg_head_candidate_query_no_aux | combined_activation | 0.1719 | 0.0000 | 1 |
| combined final-label | trainable_shared_agent_active_msg_head_candidate_query_no_aux | combined_message | 0.0625 | 0.0000 | 1 |
| combined final-label | trainable_shared_agent_evidence_token_diagnostic_upper_bound | combined_activation | 0.1094 | 0.0000 | 1 |
| combined final-label | trainable_shared_agent_evidence_token_diagnostic_upper_bound | combined_message | 0.0938 | 0.0000 | 1 |
| combined final-label | trainable_shared_agent_latent_coordinator | combined_activation | 0.1406 | 0.0000 | 1 |

Message collapse statistics:
| method | feature source | variance | norm mean | norm std | mean cosine | within-label cosine | between-label cosine |
|---|---|---:|---:|---:|---:|---:|---:|
| frozen_shared_agent_active_msg_head_aux_candidate_query | active_message_readout | 0.000018 | 15.2133 | 0.0052 | 1.0000 | 1.0000 | 1.0000 |
| frozen_shared_agent_active_msg_head_aux_candidate_query | message_head_output | 0.000001 | 4.7631 | 0.0018 | 1.0000 | 1.0000 | 1.0000 |
| frozen_shared_agent_active_msg_head_aux_candidate_query | raw_pooled_activation | 0.017086 | 2.8617 | 0.1827 | 0.6005 | 0.5969 | 0.6010 |
| frozen_shared_agent_evidence_token_diagnostic_upper_bound | evidence_token_activation | 0.014735 | 2.6571 | 0.1956 | 0.5991 | 0.6049 | 0.5983 |
| frozen_shared_agent_evidence_token_diagnostic_upper_bound | message_head_output | 0.168806 | 7.0007 | 0.8377 | 0.1250 | 0.1454 | 0.1223 |
| frozen_shared_agent_evidence_token_diagnostic_upper_bound | msg_token | 0.000009 | 2.0139 | 0.0044 | 0.9996 | 0.9996 | 0.9996 |
| frozen_shared_agent_evidence_token_diagnostic_upper_bound | raw_pooled_activation | 0.014074 | 2.9585 | 0.1553 | 0.6914 | 0.6896 | 0.6916 |
| trainable_shared_agent_active_msg_head_aux_candidate_query | active_message_readout | 0.000020 | 14.0327 | 0.0045 | 1.0000 | 1.0000 | 1.0000 |
| trainable_shared_agent_active_msg_head_aux_candidate_query | message_head_output | 0.000001 | 4.0129 | 0.0021 | 1.0000 | 1.0000 | 1.0000 |
| trainable_shared_agent_active_msg_head_aux_candidate_query | raw_pooled_activation | 0.013392 | 3.3006 | 0.1914 | 0.7641 | 0.7614 | 0.7645 |
| trainable_shared_agent_active_msg_head_candidate_query_no_aux | active_message_readout | 4.378211 | 52.0331 | 1.0202 | 0.6879 | 0.7125 | 0.6846 |
| trainable_shared_agent_active_msg_head_candidate_query_no_aux | message_head_output | 0.168515 | 13.7613 | 0.8113 | 0.7723 | 0.7919 | 0.7696 |
| trainable_shared_agent_active_msg_head_candidate_query_no_aux | raw_pooled_activation | 0.084487 | 11.9769 | 1.5423 | 0.9027 | 0.9012 | 0.9029 |
| trainable_shared_agent_evidence_token_diagnostic_upper_bound | evidence_token_activation | 0.206178 | 7.0607 | 0.6856 | 0.2024 | 0.2134 | 0.2009 |
| trainable_shared_agent_evidence_token_diagnostic_upper_bound | message_head_output | 0.166693 | 7.1581 | 0.4906 | 0.1593 | 0.1788 | 0.1567 |
| trainable_shared_agent_evidence_token_diagnostic_upper_bound | msg_token | 0.000022 | 9.0731 | 0.0144 | 1.0000 | 0.9999 | 1.0000 |
| trainable_shared_agent_evidence_token_diagnostic_upper_bound | raw_pooled_activation | 0.033415 | 3.6328 | 0.2351 | 0.5124 | 0.5042 | 0.5135 |
| trainable_shared_agent_latent_coordinator | raw_pooled_activation | 0.155695 | 6.2306 | 0.9654 | 0.2516 | 0.2527 | 0.2514 |

Failure diagnosis:
The trainable message model did not beat the frozen same-coordinator comparator, so this does not support shared-agent learning.
Active-message interpretation: `active_readout_still_collapses_diagnose_message_readout_optimization_before_changing_coordinator`.
Private cues are not reliably decodable from active message-head outputs; diagnose message-readout optimization before changing the coordinator.
Evidence-token states decode private cues, so that branch remains an upper-bound token-extraction diagnostic rather than evidence of shared-agent learning.
Message-head outputs encode private cues better than chance, but final answer accuracy remains weak; message formation and coordination/composition should be separated.

Next branch recommendation:
Diagnose active message-readout optimization before changing the coordinator.

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
| trainable_shared_agent_evidence_token_diagnostic_upper_bound | 1.0000 | 0.0000 | 1 | 272659 |
| frozen_shared_agent_evidence_token_diagnostic_upper_bound | 0.9297 | 0.0000 | 1 | 46051 |
| trainable_shared_agent_active_msg_head_candidate_query_no_aux | 0.3047 | 0.0000 | 1 | 294801 |
| frozen_shared_agent_active_msg_head_aux_candidate_query | 0.1641 | 0.0000 | 1 | 68451 |
| coordinator_only_probe | 0.1562 | 0.0000 | 1 | 552 |
| trainable_shared_agent_latent_coordinator | 0.1328 | 0.0000 | 1 | 264184 |
| bag_of_words_candidate_patch_only | 0.1250 | 0.0000 | 1 | 4392 |
| candidate_order_baseline | 0.1250 | 0.0000 | 1 | 0 |
| frozen_shared_agent_latent_coordinator | 0.1250 | 0.0000 | 1 | 37576 |
| frozen_shared_agent_latent_coordinator_self_attention | 0.1250 | 0.0000 | 1 | 37576 |
| majority_class_baseline | 0.1250 | 0.0000 | 1 | 0 |
| role_labels_only_baseline | 0.1250 | 0.0000 | 1 | 1320 |
| single_agent_partial_view | 0.1250 | 0.0000 | 1 | 227000 |
| single_view_text_role_2 | 0.1250 | 0.0000 | 1 | 4392 |
| single_view_text_role_3 | 0.1250 | 0.0000 | 1 | 4392 |
| text_only_partial_view_baseline | 0.1250 | 0.0000 | 1 | 4392 |
| text_output_only_multi_agent_coordinator | 0.1250 | 0.0000 | 1 | 4392 |
| trainable_shared_agent_latent_coordinator_self_attention | 0.1250 | 0.0000 | 1 | 264184 |
| single_agent_full_context | 0.1172 | 0.0000 | 1 | 227000 |
| single_view_text_role_0 | 0.1172 | 0.0000 | 1 | 4392 |
| single_view_text_role_1 | 0.1172 | 0.0000 | 1 | 4392 |
| trainable_shared_agent_active_msg_head_aux_candidate_query | 0.0781 | 0.0000 | 1 | 295059 |
| larger_tiny_transformer_full_context | 0.0703 | 0.0000 | 1 | 574856 |

Audit table:
| method | condition | shared params | M grad | M delta | msg-head grad | msg-head delta | aux-head grad | aux-head delta | C grad | C delta | activations grad | no detach | per-clone grad | memory bytes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|
| frozen_shared_agent_active_msg_head_aux_candidate_query | none | pass | 0.000000 | 0.000000 | 0.036882 | 0.535494 | 0.015783 | 0.058424 | 0.338974 | 1.294649 | pass | pass | pass | 0 |
| frozen_shared_agent_evidence_token_diagnostic_upper_bound | none | pass | 0.000000 | 0.000000 | 0.447880 | 1.746412 | 0.006814 | 0.315992 | 1.155480 | 2.632396 | pass | pass | pass | 0 |
| frozen_shared_agent_latent_coordinator | none | pass | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.985572 | 1.277836 | fail | fail | fail | 0 |
| frozen_shared_agent_latent_coordinator_self_attention | none | pass | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.828536 | 1.252514 | fail | fail | fail | 0 |
| trainable_shared_agent_active_msg_head_aux_candidate_query | none | pass | 0.000881 | 1.314529 | 0.025935 | 0.426271 | 0.017095 | 0.075667 | 0.363988 | 1.121949 | pass | pass | pass | 0 |
| trainable_shared_agent_active_msg_head_aux_candidate_query | randomized_labels | pass | 0.000965 | 1.330433 | 0.034509 | 0.424498 | 0.020171 | 0.059151 | 0.314115 | 1.270122 | pass | pass | pass | 0 |
| trainable_shared_agent_active_msg_head_candidate_query_no_aux | none | pass | 0.436492 | 9.154579 | 0.154592 | 2.382320 | 0.000000 | 0.000000 | 0.617379 | 5.479206 | pass | pass | pass | 0 |
| trainable_shared_agent_evidence_token_diagnostic_upper_bound | none | pass | 0.208371 | 5.340530 | 0.335143 | 1.254255 | 0.006730 | 0.241956 | 1.147856 | 3.668050 | pass | pass | pass | 0 |
| trainable_shared_agent_latent_coordinator | none | pass | 0.304545 | 4.433984 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.788738 | 2.187122 | pass | pass | pass | 0 |
| trainable_shared_agent_latent_coordinator | randomized_labels | pass | 0.628998 | 14.689618 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.658842 | 3.160315 | pass | pass | pass | 0 |
| trainable_shared_agent_latent_coordinator_self_attention | none | pass | 1.569516 | 6.583943 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 2.090473 | 3.045460 | pass | pass | pass | 0 |
| trainable_shared_agent_latent_visible_explicit_evidence | visible_explicit_evidence | pass | 0.393997 | 1.925960 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 2.133899 | 1.235574 | pass | pass | pass | 0 |

Controls:
| condition | test mean acc | std | runs |
|---|---:|---:|---:|
| candidate_order_shuffled | 0.0781 | 0.0000 | 1 |
| hidden_states_shuffled_across_examples | 0.0781 | 0.0000 | 1 |
| physical_order_shuffled_roles_preserved | 0.0781 | 0.0000 | 1 |
| randomized_labels | 0.1172 | 0.0000 | 1 |
| role_labels_shuffled | 0.0781 | 0.0000 | 1 |
| view_masked | 0.0781 | 0.0000 | 1 |
| view_shuffled | 0.0781 | 0.0000 | 1 |

Learning curves:
| method | condition | split | first epoch acc | final epoch acc | epochs |
|---|---|---|---:|---:|---:|
| frozen_shared_agent_active_msg_head_aux_candidate_query | none | dev | 0.1146 | 0.1042 | 7 |
| frozen_shared_agent_active_msg_head_aux_candidate_query | none | train | 0.1562 | 0.1641 | 7 |
| frozen_shared_agent_evidence_token_diagnostic_upper_bound | none | dev | 0.1146 | 1.0000 | 9 |
| frozen_shared_agent_evidence_token_diagnostic_upper_bound | none | train | 0.1953 | 0.9961 | 9 |
| frozen_shared_agent_latent_coordinator | none | dev | 0.1250 | 0.1250 | 7 |
| frozen_shared_agent_latent_coordinator | none | train | 0.1250 | 0.2383 | 7 |
| frozen_shared_agent_latent_coordinator_self_attention | none | dev | 0.1250 | 0.1250 | 7 |
| frozen_shared_agent_latent_coordinator_self_attention | none | train | 0.1250 | 0.2539 | 7 |
| trainable_shared_agent_active_msg_head_aux_candidate_query | none | dev | 0.1250 | 0.1042 | 7 |
| trainable_shared_agent_active_msg_head_aux_candidate_query | none | train | 0.1680 | 0.1562 | 7 |
| trainable_shared_agent_active_msg_head_aux_candidate_query | randomized_labels | dev | 0.1042 | 0.1146 | 7 |
| trainable_shared_agent_active_msg_head_aux_candidate_query | randomized_labels | train | 0.1641 | 0.1484 | 7 |
| trainable_shared_agent_active_msg_head_candidate_query_no_aux | none | dev | 0.1042 | 0.2604 | 14 |
| trainable_shared_agent_active_msg_head_candidate_query_no_aux | none | train | 0.1562 | 0.2500 | 14 |
| trainable_shared_agent_evidence_token_diagnostic_upper_bound | none | dev | 0.0938 | 1.0000 | 11 |
| trainable_shared_agent_evidence_token_diagnostic_upper_bound | none | train | 0.1289 | 1.0000 | 11 |
| trainable_shared_agent_latent_coordinator | none | dev | 0.1250 | 0.1250 | 9 |
| trainable_shared_agent_latent_coordinator | none | train | 0.1250 | 0.2578 | 9 |
| trainable_shared_agent_latent_coordinator_self_attention | none | dev | 0.1250 | 0.1354 | 11 |
| trainable_shared_agent_latent_coordinator_self_attention | none | train | 0.1250 | 0.2539 | 11 |
| trainable_shared_agent_latent_visible_explicit_evidence | visible_explicit_evidence | dev | 0.1250 | 0.0938 | 7 |
| trainable_shared_agent_latent_visible_explicit_evidence | visible_explicit_evidence | train | 0.1484 | 0.3984 | 7 |

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
1. Exact model: `tiny_transformer` with `EleutherAI/pythia-70m-deduped` when pretrained loading is enabled.
2. Trainable parameters: `['tiny_embedding.weight', 'tiny_position.weight', 'tiny_encoder.layers.0.self_attn.in_proj_weight', 'tiny_encoder.layers.0.self_attn.in_proj_bias', 'tiny_encoder.layers.0.self_attn.out_proj.weight', 'tiny_encoder.layers.0.self_attn.out_proj.bias', 'tiny_encoder.layers.0.linear1.weight', 'tiny_encoder.layers.0.linear1.bias', 'tiny_encoder.layers.0.linear2.weight', 'tiny_encoder.layers.0.linear2.bias', 'tiny_encoder.layers.0.norm1.weight', 'tiny_encoder.layers.0.norm1.bias', 'tiny_encoder.layers.0.norm2.weight', 'tiny_encoder.layers.0.norm2.bias', 'tiny_norm.weight', 'tiny_norm.bias', 'activation_adapter.0.weight', 'activation_adapter.0.bias', 'activation_adapter.1.weight', 'activation_adapter.1.bias', 'activation_adapter.3.weight', 'activation_adapter.3.bias']`.
3. Shared across clones: `True`.
4. Nonzero shared gradients: `True`.
5. Shared parameter delta: `1.314529`.
6. Trainable beat frozen: `False`.
7. Trainable beat text-only: `False`.
8. Controls collapse: inspect the controls table; this fallback run does not by itself authorize a positive real-world claim.
9. Physical order shuffle with roles preserved: inspect `physical_order_shuffled_roles_preserved`.
10. Role-label shuffle: inspect `role_labels_shuffled`.
11. Single partial-view accuracy: `0.125`.
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
| real_shared_weight_latent_coordination | 0 | dev | candidate_order_shuffled | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.1250 | 295059 |
| real_shared_weight_latent_coordination | 0 | dev | explicit_evidence | explicit_evidence_neural_tuple_model | 1.0000 | 3521 |
| real_shared_weight_latent_coordination | 0 | dev | explicit_evidence | raw_structured_evidence_coordinator | 1.0000 | 3457 |
| real_shared_weight_latent_coordination | 0 | dev | hidden_states_shuffled_across_examples | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.1250 | 295059 |
| real_shared_weight_latent_coordination | 0 | dev | none | bag_of_words_candidate_patch_only | 0.1250 | 4392 |
| real_shared_weight_latent_coordination | 0 | dev | none | candidate_order_baseline | 0.1250 | 0 |
| real_shared_weight_latent_coordination | 0 | dev | none | coordinator_only_probe | 0.1354 | 552 |
| real_shared_weight_latent_coordination | 0 | dev | none | explicit_evidence_oracle | 1.0000 | 0 |
| real_shared_weight_latent_coordination | 0 | dev | none | frozen_shared_agent_active_msg_head_aux_candidate_query | 0.1146 | 68451 |
| real_shared_weight_latent_coordination | 0 | dev | none | frozen_shared_agent_evidence_token_diagnostic_upper_bound | 1.0000 | 46051 |
| real_shared_weight_latent_coordination | 0 | dev | none | frozen_shared_agent_latent_coordinator | 0.1250 | 37576 |
| real_shared_weight_latent_coordination | 0 | dev | none | frozen_shared_agent_latent_coordinator_self_attention | 0.1250 | 37576 |
| real_shared_weight_latent_coordination | 0 | dev | none | larger_tiny_transformer_full_context | 0.1562 | 574856 |
| real_shared_weight_latent_coordination | 0 | dev | none | majority_class_baseline | 0.1250 | 0 |
| real_shared_weight_latent_coordination | 0 | dev | none | role_labels_only_baseline | 0.1250 | 1320 |
| real_shared_weight_latent_coordination | 0 | dev | none | single_agent_full_context | 0.1667 | 227000 |
| real_shared_weight_latent_coordination | 0 | dev | none | single_agent_partial_view | 0.1250 | 227000 |
| real_shared_weight_latent_coordination | 0 | dev | none | single_view_text_role_0 | 0.1458 | 4392 |
| real_shared_weight_latent_coordination | 0 | dev | none | single_view_text_role_1 | 0.1562 | 4392 |
| real_shared_weight_latent_coordination | 0 | dev | none | single_view_text_role_2 | 0.1250 | 4392 |
| real_shared_weight_latent_coordination | 0 | dev | none | single_view_text_role_3 | 0.1250 | 4392 |
| real_shared_weight_latent_coordination | 0 | dev | none | text_only_partial_view_baseline | 0.1250 | 4392 |
| real_shared_weight_latent_coordination | 0 | dev | none | text_output_only_multi_agent_coordinator | 0.1562 | 4392 |
| real_shared_weight_latent_coordination | 0 | dev | none | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.1250 | 295059 |
| real_shared_weight_latent_coordination | 0 | dev | none | trainable_shared_agent_active_msg_head_candidate_query_no_aux | 0.2812 | 294801 |
| real_shared_weight_latent_coordination | 0 | dev | none | trainable_shared_agent_evidence_token_diagnostic_upper_bound | 1.0000 | 272659 |
| real_shared_weight_latent_coordination | 0 | dev | none | trainable_shared_agent_latent_coordinator | 0.1458 | 264184 |
| real_shared_weight_latent_coordination | 0 | dev | none | trainable_shared_agent_latent_coordinator_self_attention | 0.1562 | 264184 |
| real_shared_weight_latent_coordination | 0 | dev | physical_order_shuffled_roles_preserved | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.1250 | 295059 |
| real_shared_weight_latent_coordination | 0 | dev | randomized_labels | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.1042 | 295059 |
| real_shared_weight_latent_coordination | 0 | dev | randomized_labels | trainable_shared_agent_latent_coordinator | 0.1250 | 264184 |
| real_shared_weight_latent_coordination | 0 | dev | role_labels_shuffled | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.1250 | 295059 |
| real_shared_weight_latent_coordination | 0 | dev | view_masked | masked_evidence_text_baseline | 0.1250 | 4392 |
| real_shared_weight_latent_coordination | 0 | dev | view_masked | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.1250 | 295059 |
| real_shared_weight_latent_coordination | 0 | dev | view_shuffled | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.1250 | 295059 |
| real_shared_weight_latent_coordination | 0 | dev | visible_explicit_evidence | trainable_shared_agent_latent_visible_explicit_evidence | 0.1250 | 264184 |
| real_shared_weight_latent_coordination | 0 | test | candidate_order_shuffled | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.0781 | 295059 |
| real_shared_weight_latent_coordination | 0 | test | explicit_evidence | explicit_evidence_neural_tuple_model | 1.0000 | 3521 |
| real_shared_weight_latent_coordination | 0 | test | explicit_evidence | raw_structured_evidence_coordinator | 1.0000 | 3457 |
| real_shared_weight_latent_coordination | 0 | test | hidden_states_shuffled_across_examples | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.0781 | 295059 |
| real_shared_weight_latent_coordination | 0 | test | none | bag_of_words_candidate_patch_only | 0.1250 | 4392 |
| real_shared_weight_latent_coordination | 0 | test | none | candidate_order_baseline | 0.1250 | 0 |
| real_shared_weight_latent_coordination | 0 | test | none | coordinator_only_probe | 0.1562 | 552 |
| real_shared_weight_latent_coordination | 0 | test | none | explicit_evidence_oracle | 1.0000 | 0 |
| real_shared_weight_latent_coordination | 0 | test | none | frozen_shared_agent_active_msg_head_aux_candidate_query | 0.1641 | 68451 |
| real_shared_weight_latent_coordination | 0 | test | none | frozen_shared_agent_evidence_token_diagnostic_upper_bound | 0.9297 | 46051 |
| real_shared_weight_latent_coordination | 0 | test | none | frozen_shared_agent_latent_coordinator | 0.1250 | 37576 |
| real_shared_weight_latent_coordination | 0 | test | none | frozen_shared_agent_latent_coordinator_self_attention | 0.1250 | 37576 |
| real_shared_weight_latent_coordination | 0 | test | none | larger_tiny_transformer_full_context | 0.0703 | 574856 |
| real_shared_weight_latent_coordination | 0 | test | none | majority_class_baseline | 0.1250 | 0 |
| real_shared_weight_latent_coordination | 0 | test | none | role_labels_only_baseline | 0.1250 | 1320 |
| real_shared_weight_latent_coordination | 0 | test | none | single_agent_full_context | 0.1172 | 227000 |
| real_shared_weight_latent_coordination | 0 | test | none | single_agent_partial_view | 0.1250 | 227000 |
| real_shared_weight_latent_coordination | 0 | test | none | single_view_text_role_0 | 0.1172 | 4392 |
| real_shared_weight_latent_coordination | 0 | test | none | single_view_text_role_1 | 0.1172 | 4392 |
| real_shared_weight_latent_coordination | 0 | test | none | single_view_text_role_2 | 0.1250 | 4392 |
| real_shared_weight_latent_coordination | 0 | test | none | single_view_text_role_3 | 0.1250 | 4392 |
| real_shared_weight_latent_coordination | 0 | test | none | text_only_partial_view_baseline | 0.1250 | 4392 |
| real_shared_weight_latent_coordination | 0 | test | none | text_output_only_multi_agent_coordinator | 0.1250 | 4392 |
| real_shared_weight_latent_coordination | 0 | test | none | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.0781 | 295059 |
| real_shared_weight_latent_coordination | 0 | test | none | trainable_shared_agent_active_msg_head_candidate_query_no_aux | 0.3047 | 294801 |
| real_shared_weight_latent_coordination | 0 | test | none | trainable_shared_agent_evidence_token_diagnostic_upper_bound | 1.0000 | 272659 |
| real_shared_weight_latent_coordination | 0 | test | none | trainable_shared_agent_latent_coordinator | 0.1328 | 264184 |
| real_shared_weight_latent_coordination | 0 | test | none | trainable_shared_agent_latent_coordinator_self_attention | 0.1250 | 264184 |
| real_shared_weight_latent_coordination | 0 | test | physical_order_shuffled_roles_preserved | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.0781 | 295059 |
| real_shared_weight_latent_coordination | 0 | test | randomized_labels | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.1172 | 295059 |
| real_shared_weight_latent_coordination | 0 | test | randomized_labels | trainable_shared_agent_latent_coordinator | 0.1172 | 264184 |
| real_shared_weight_latent_coordination | 0 | test | role_labels_shuffled | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.0781 | 295059 |
| real_shared_weight_latent_coordination | 0 | test | view_masked | masked_evidence_text_baseline | 0.1250 | 4392 |
| real_shared_weight_latent_coordination | 0 | test | view_masked | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.0781 | 295059 |
| real_shared_weight_latent_coordination | 0 | test | view_shuffled | trainable_shared_agent_active_msg_head_aux_candidate_query | 0.0781 | 295059 |
| real_shared_weight_latent_coordination | 0 | test | visible_explicit_evidence | trainable_shared_agent_latent_visible_explicit_evidence | 0.1250 | 264184 |

## Mechanism Distinction


## Synthetic Vs Transformer Hidden States

No captured-transformer hidden-state stage is present in this report.

## Interpretation

- Recommended next experiment: make the transformer strict task less templated while keeping private evidence hidden from visible outputs, then repeat the same access audit, randomized-label sanity check, output-only baselines, and individual-vs-combined hidden-state probes.
