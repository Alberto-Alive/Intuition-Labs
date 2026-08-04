# Stage 3.8 Full Hard Validation

## Scope/Config

- Architecture: `topk_attention_no_head`
- Locked architecture changes: none
- Locked setup: shared-weight cloned agent; active/top-k token readout; no message head; no private-cue auxiliary loss; candidate-query coordinator; exact frozen same-architecture comparator.
- Dataset/control representation: `balanced_categories_v3` from `real_import_restore_candidate_balanced_34b`.
- Normalized-view-format variant: not used.
- Device requested/used: `cuda`
- CUDA device: `NVIDIA GeForce RTX 5070 Ti`
- Mixed precision: `bf16`
- Split sizes: train `64`, dev `32`, test `64`; candidates `8`.
- Seeds requested: `[0]`
- Batch size fallback: `[32, 16, 8]`

## Schema-Aware Role Policy

- Role-specific schema fields are preserved and treated as legitimate program-analysis evidence.
- `role_embedding_shuffle_diagnostic` is diagnostic only and is not a primary corruption gate.
- Primary gates are compatibility-breaking controls that preserve schema where appropriate while breaking evidence/candidate compatibility.

## Pre-Run Shortcut Diagnostics

| gate | pass | value | threshold |
|---|---|---:|---:|
| candidate_only | `True` | 0.1250 | 0.1800 |
| candidate_metadata_only | `True` | 0.1250 | 0.1800 |
| view_masked_candidates_visible | `True` | 0.1250 | 0.1800 |
| role_pair_only | `True` | 0.1250 | 0.1800 |
| lexical_overlap | `True` | 0.1250 | 0.1800 |
| static_frequency | `True` | 0.0625 | 0.2000 |
| schema_only_baseline | `True` | 0.1250 | 0.1800 |
| null_evidence_values_baseline | `True` | 0.1250 | 0.1800 |
| evidence_only_no_candidates_baseline | `True` | 0.1250 | 0.1800 |
| all_role_oracle | `True` | 1.0000 | 0.9000 |

## Positive-Control Learnability

- Method: `candidate_pair_compatibility_mlp`
- Seed 0 test accuracy: `1.0000`
- Learnability confirmed: `True`

## Per-Seed Accuracy

| seed | batch | accum | max CUDA GB | trainable | frozen | delta | text | raw | majority | cand-order | max single-view | full ctx | candidate-pair | oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 32 | 1 | 0.074 | 0.1094 | 0.1719 | -0.0625 | 0.1250 | 0.1406 | 0.1250 | 0.1250 | 0.1406 | 0.1094 | 1.0000 | 1.0000 |

## Mean/Std/Bootstrap CI

- Completed seeds: `[0]`
- Mean trainable accuracy: `0.1094`
- Mean frozen accuracy: `0.1719`
- Mean trainable-frozen delta: `-0.0625`
- Delta std: `0.0000`
- Bootstrap 95% CI for delta: `[-0.0625, -0.0625]`

## Baselines Table

| method | mean | std |
|---|---:|---:|
| trainable | 0.1094 | 0.0000 |
| frozen | 0.1719 | 0.0000 |
| text_only | 0.1250 | 0.0000 |
| raw_latent | 0.1406 | 0.0000 |
| majority_baseline | 0.1250 | 0.0000 |
| candidate_order_baseline | 0.1250 | 0.0000 |
| single_agent_full_context | 0.1094 | 0.0000 |
| candidate_pair_compatibility_mlp | 1.0000 | 0.0000 |
| all_role_oracle | 1.0000 | 0.0000 |
| single_view_text_role_0 | 0.1250 | 0.0000 |
| single_view_text_role_1 | 0.1250 | 0.0000 |
| single_view_text_role_2 | 0.1250 | 0.0000 |
| single_view_text_role_3 | 0.1406 | 0.0000 |

## Schema-Aware Control Table

| control | mean | std | max | pass |
|---|---:|---:|---:|---|
| candidate_only | 0.0312 | 0.0000 | 0.0312 | `True` |
| view_masked_candidates_visible | 0.1250 | 0.0000 | 0.1250 | `True` |
| schema_only | 0.0469 | 0.0000 | 0.0469 | `True` |
| value_shuffle_within_schema | 0.0781 | 0.0000 | 0.0781 | `True` |
| cross_example_view_bundle_shuffle | 0.0938 | 0.0000 | 0.0938 | `True` |
| candidate_evidence_mismatch | 0.1406 | 0.0000 | 0.1406 | `True` |
| schema_preserved_role_value_shuffle | 0.0781 | 0.0000 | 0.0781 | `True` |
| null_evidence_values | 0.0938 | 0.0000 | 0.0938 | `True` |
| evidence_only_no_candidates | 0.1250 | 0.0000 | 0.1250 | `True` |
| randomized_labels | 0.0000 | 0.0000 | 0.0000 | `True` |
| hidden_states_shuffled_across_examples | 0.1250 | 0.0000 | 0.1250 | `True` |

## Invariance Control Table

| control | mean | delta from trainable | tolerance | pass |
|---|---:|---:|---:|---|
| physical_order_shuffled_roles_preserved | 0.1094 | 0.0000 | 0.0800 | `True` |
| candidate_order_shuffled_with_gold_remap | 0.1094 | 0.0000 | 0.0800 | `True` |

## Role-Embedding Shuffle Diagnostic

| seed | trainable | role_embedding_shuffle_diagnostic | delta | gating? |
|---:|---:|---:|---:|---|
| 0 | 0.1094 | 0.0938 | -0.0156 | `False` |

## Audit Summary

| seed | split leak | output leak | train grad | train delta | frozen grad | frozen delta | checkpoints |
|---:|---|---|---:|---:|---:|---:|---|
| 0 | pass | pass | 0.1120 | 1.1145 | 0.0000 | 0.0000 | pass |

## Acceptance Gates

| criterion | pass | value |
|---|---|---|
| completed seeds >= 10 | `False` | `1` |
| trainable beats frozen on at least 8/10 seeds | `False` | `0` |
| mean trainable-frozen delta >= +0.20 | `False` | `-0.0625` |
| bootstrap 95% CI lower bound for delta > +0.05 | `False` | `[-0.0625, -0.0625]` |
| trainable beats text-only, raw-latent, majority, candidate-order, and full-context baselines by mean accuracy | `False` | `{"candidate_order_baseline": 0.125, "majority_baseline": 0.125, "raw_latent": 0.140625, "single_agent_full_context": 0.109375, "text_only": 0.125, "trainable_mean": 0.109375}` |
| trainable beats every single-view baseline by mean accuracy | `False` | `{"single_view_text_role_0": 0.125, "single_view_text_role_1": 0.125, "single_view_text_role_2": 0.125, "single_view_text_role_3": 0.140625}` |
| primary corruption controls collapse near chance | `True` | `{"candidate_evidence_mismatch": 0.140625, "candidate_only": 0.03125, "cross_example_view_bundle_shuffle": 0.09375, "evidence_only_no_candidates": 0.125, "hidden_states_shuffled_across_examples": 0.125, "null_evidence_values": 0.09375, "randomized_labels": 0.0, "schema_only": 0.046875, "schema_preserved_role_value_shuffle": 0.078125, "value_shuffle_within_schema": 0.078125, "view_masked_candidates_visible": 0.125}` |
| invariance controls preserve accuracy | `True` | `{"candidate_order_shuffled_with_gold_remap": {"mean_accuracy": 0.109375, "mean_delta_from_trainable": 0.0}, "physical_order_shuffled_roles_preserved": {"mean_accuracy": 0.109375, "mean_delta_from_trainable": 0.0}}` |
| pre-run shortcut diagnostics and positive controls pass | `True` | `{"all_role_oracle": true, "candidate_metadata_only": true, "candidate_only": true, "evidence_only_no_candidates_baseline": true, "lexical_overlap": true, "null_evidence_values_baseline": true, "positive_control_learnability": true, "role_pair_only": true, "schema_only_baseline": true, "static_frequency": true, "view_masked_candidates_visible": true}` |
| leakage audits pass | `True` | `"split_candidate_patch_hash_and_output"` |
| shared trainable model receives gradients and changes | `True` | `"all_completed_seeds"` |
| frozen comparator remains frozen | `True` | `"all_completed_seeds"` |
| checkpoints saved for every seed and required method | `True` | `["trainable", "frozen", "randomized_labels", "text_only", "raw_latent", "majority_baseline", "candidate_order_baseline", "single_agent_full_context", "single_view_text_role_0", "single_view_text_role_1", "single_view_text_role_2", "single_view_text_role_3", "candidate_pair_compatibility_mlp", "explicit_evidence_oracle"]` |

## Failed Seeds/OOMs

- Failed/interrupted: `[]`
- OOM retries: `[]`

## Conservative Interpretation

- Full Stage 3.8 gates passed: `False`
- Do not interpret `role_embedding_shuffle_diagnostic` as a primary corruption failure; schema text legitimately preserves role identity.
- This report only supports a Stage 3.8 benchmark conclusion if every gate above passes, checkpoints exist, and leakage/audit checks pass.
