# Stage 8B.4 Top-3 Larger-Scale Validation vs Clean-QKV Challenger

## Executive Summary

- Decision: STAGE8B4_TOP3_WIN_CLEANQKV_NOT_COMPETITIVE
- Stage 8C was not run.
- No 10x attention-capacity claim is made.
- Selected Stage 8C candidate: stage8b3_j_explicit_view_objective_assignment_03
- Recommendation: Keep the best frozen Stage 8B.3 architecture as the only Stage 8C candidate; do not promote Clean-QKV.

## Validation Design

- N schedule: [8, 16, 32, 64, 128, 256]
- Seeds: [0, 1, 2, 3, 4]
- Splits: dev templates and held-out-template dev only; final Stage 8C templates were not used.
- Frozen Stage 8B.3 top-three configs were loaded from the freeze manifest without modification.

## Architecture Results

| Architecture | Kind | C | Heldout dev | N64 | Mismatch degrade | Cross-task degrade | Frozen gap | Collapse sim | Ready |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| stage8b3_j_explicit_view_objective_assignment_03 | evidence_feature_latent | 64 | 0.9444 | 1.0000 | 0.8111 | 0.8160 | 0.8160 | 0.0052 | True |
| stage8b3_f_memory_slot_attention_02 | evidence_feature_latent | 64 | 0.9444 | 1.0000 | 0.8111 | 0.8160 | 0.8160 | 0.0066 | True |
| stage8b3_j_explicit_view_objective_assignment_02 | evidence_feature_latent | 64 | 0.9444 | 1.0000 | 0.8111 | 0.8160 | 0.8160 | 0.0066 | True |
| stage8b4_clean_qkv_activation_memory_challenger | clean_qkv_activation_memory | 0 | 0.5243 | 0.4958 | 0.3924 | 0.3965 | 0.3958 | 0.6136 | False |

## Clean-QKV Memory Controls

{
  "rows": [
    {
      "architecture_name": "stage8b4_clean_qkv_activation_memory_challenger",
      "config_id": "3ae869210307d7af",
      "memory_controls_pass": true,
      "memory_disabled_degradation": 0.41041666666666665,
      "memory_gate_calibration": {
        "irrelevant_mean_gate": 0.9932136098224036,
        "mean_gate": 0.9909273365285596,
        "relevant_mean_gate": 0.9887193032416122
      },
      "memory_gate_forced_closed_degradation": 0.41041666666666665,
      "memory_gate_forced_open_accuracy": 0.5229166666666667,
      "memory_shuffle_across_examples_degradation": 0.4,
      "memory_slot_entropy": 1.3199093653510015,
      "memory_slot_permutation_delta": 0.0006944444444444401
    }
  ],
  "stage": "8B.4"
}

## Baseline Comparison

{
  "n_schedule": [
    8,
    16,
    32,
    64,
    128,
    256
  ],
  "rows": [
    {
      "accuracy_by_N": {
        "128": 0.09583333333333333,
        "16": 0.1125,
        "256": 0.1125,
        "32": 0.09583333333333333,
        "64": 0.15,
        "8": 0.1375
      },
      "architecture_name": "stage8b4_random_candidate",
      "held_out_template_dev_accuracy": 0.11736111111111111,
      "model_kind": "random_candidate"
    },
    {
      "accuracy_by_N": {
        "128": 0.125,
        "16": 0.09583333333333334,
        "256": 0.1625,
        "32": 0.15,
        "64": 0.08333333333333333,
        "8": 0.1625
      },
      "architecture_name": "stage8b4_candidate_only",
      "held_out_template_dev_accuracy": 0.12986111111111112,
      "model_kind": "candidate_only"
    },
    {
      "accuracy_by_N": {
        "128": 0.1375,
        "16": 0.19166666666666668,
        "256": 0.15,
        "32": 0.1875,
        "64": 0.10416666666666667,
        "8": 0.10833333333333332
      },
      "architecture_name": "stage8b4_query_only",
      "held_out_template_dev_accuracy": 0.14652777777777778,
      "model_kind": "query_only"
    },
    {
      "accuracy_by_N": {
        "128": 0.1375,
        "16": 0.11666666666666667,
        "256": 0.1375,
        "32": 0.15,
        "64": 0.11666666666666667,
        "8": 0.1125
      },
      "architecture_name": "stage8b4_evidence_only",
      "held_out_template_dev_accuracy": 0.1284722222222222,
      "model_kind": "evidence_only"
    },
    {
      "accuracy_by_N": {
        "128": 0.2791666666666667,
        "16": 0.4041666666666667,
        "256": 0.14166666666666666,
        "32": 0.2916666666666667,
        "64": 0.2791666666666667,
        "8": 0.575
      },
      "architecture_name": "stage8b4_retrieval_topk",
      "held_out_template_dev_accuracy": 0.3284722222222222,
      "model_kind": "retrieval_topk"
    },
    {
      "accuracy_by_N": {
        "128": 0.09166666666666667,
        "16": 0.12916666666666668,
        "256": 0.09583333333333333,
        "32": 0.12083333333333333,
        "64": 0.14166666666666666,
        "8": 0.12916666666666668
      },
      "architecture_name": "stage8b4_true_monolithic_transformer",
      "held_out_template_dev_accuracy": 0.11805555555555555,
      "model_kind": "monolithic_transformer"
    },
    {
      "accuracy_by_N": {
        "128": 0.17916666666666667,
        "16": 0.38333333333333336,
        "256": 0.14166666666666666,
        "32": 0.2791666666666667,
        "64": 0.2625,
        "8": 0.44166666666666665
      },
      "architecture_name": "stage8b4_legacy_hashed_feature_baseline",
      "held_out_template_dev_accuracy": 0.28125,
      "model_kind": "retrieval_topk"
    },
    {
      "accuracy_by_N": {
        "128": 0.2375,
        "16": 0.375,
        "256": 0.19583333333333333,
        "32": 0.2833333333333333,
        "64": 0.25833333333333336,
        "8": 0.525
      },
      "architecture_name": "stage8b4_raw_latent_comparator",
      "held_out_template_dev_accuracy": 0.3125,
      "model_kind": "raw_latent_selector"
    }
  ],
  "seeds": [
    0,
    1,
    2,
    3,
    4
  ]
}

## Failure and Disqualification Notes

- stage8b3_j_explicit_view_objective_assignment_03: []
- stage8b3_f_memory_slot_attention_02: []
- stage8b3_j_explicit_view_objective_assignment_02: []
- stage8b4_clean_qkv_activation_memory_challenger: []

## Artifacts

- Results: `results\stage8b4_top3_and_cleanqkv_results.json`
- Controls audit: `results\stage8b4_controls_audit.jsonl`
- Clean-QKV memory audit: `results\stage8b4_cleanqkv_memory_audit.json`
- View diversity audit: `results\stage8b4_view_diversity_audit.json`
- Compute audit: `results\stage8b4_compute_audit.json`
- Capacity curves: `results\stage8b4_capacity_curves.csv`
- Candidate for Stage 8C: `results\stage8b4_candidate_for_stage8c.yaml`
- Freeze recommendation: `results\stage8b4_freeze_recommendation.json`
