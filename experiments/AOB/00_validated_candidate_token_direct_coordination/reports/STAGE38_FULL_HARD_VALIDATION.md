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
- Split sizes: train `2048`, dev `512`, test `1024`; candidates `8`.
- Seeds requested: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`
- Batch size fallback: `[32, 16, 8]`

## Schema-Aware Role Policy

- Role-specific schema fields are preserved and treated as legitimate program-analysis evidence.
- `role_embedding_shuffle_diagnostic` is diagnostic only and is not a primary corruption gate.
- Primary gates are compatibility-breaking controls that preserve schema where appropriate while breaking evidence/candidate compatibility.

## Pre-Run Shortcut Diagnostics

| gate | pass | value | threshold |
|---|---|---:|---:|
| candidate_only | `True` | 0.1231 | 0.1800 |
| candidate_metadata_only | `True` | 0.1249 | 0.1800 |
| view_masked_candidates_visible | `True` | 0.1263 | 0.1800 |
| role_pair_only | `True` | 0.1250 | 0.1800 |
| lexical_overlap | `True` | 0.1250 | 0.1800 |
| static_frequency | `True` | 0.1381 | 0.2000 |
| schema_only_baseline | `True` | 0.1259 | 0.1800 |
| null_evidence_values_baseline | `True` | 0.1251 | 0.1800 |
| evidence_only_no_candidates_baseline | `True` | 0.1250 | 0.1800 |
| all_role_oracle | `True` | 1.0000 | 0.9000 |

## Positive-Control Learnability

- Method: `candidate_pair_compatibility_mlp`
- Seed 0 test accuracy: `1.0000`
- Learnability confirmed: `True`

## Per-Seed Accuracy

| seed | batch | accum | max CUDA GB | trainable | frozen | delta | text | raw | majority | cand-order | max single-view | full ctx | candidate-pair | oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 32 | 1 | 0.105 | 1.0000 | 1.0000 | 0.0000 | 0.1250 | 0.1172 | 0.1250 | 0.1250 | 0.1250 | 0.2002 | 1.0000 | 1.0000 |
| 1 | 32 | 1 | 0.105 | 1.0000 | 0.4883 | 0.5117 | 0.1318 | 0.1816 | 0.1250 | 0.1250 | 0.1250 | 0.2012 | 1.0000 | 1.0000 |
| 2 | 32 | 1 | 0.105 | 0.5410 | 0.5674 | -0.0264 | 0.1221 | 0.1338 | 0.1250 | 0.1250 | 0.1299 | 0.2363 | 1.0000 | 1.0000 |
| 3 | 32 | 1 | 0.105 | 1.0000 | 0.4434 | 0.5566 | 0.1162 | 0.1250 | 0.1250 | 0.1250 | 0.1250 | 0.2266 | 1.0000 | 1.0000 |
| 4 | 32 | 1 | 0.105 | 0.4463 | 0.5088 | -0.0625 | 0.1299 | 0.1250 | 0.1250 | 0.1250 | 0.1357 | 0.2197 | 1.0000 | 1.0000 |
| 5 | 32 | 1 | 0.105 | 1.0000 | 0.9980 | 0.0020 | 0.1260 | 0.1230 | 0.1250 | 0.1250 | 0.1250 | 0.1973 | 1.0000 | 1.0000 |
| 6 | 32 | 1 | 0.105 | 1.0000 | 0.4863 | 0.5137 | 0.1328 | 0.1855 | 0.1250 | 0.1250 | 0.1338 | 0.2080 | 1.0000 | 1.0000 |
| 7 | 32 | 1 | 0.105 | 0.4873 | 0.4854 | 0.0020 | 0.1250 | 0.1875 | 0.1250 | 0.1250 | 0.1279 | 0.2148 | 1.0000 | 1.0000 |
| 8 | 32 | 1 | 0.105 | 0.5557 | 0.5127 | 0.0430 | 0.1279 | 0.1387 | 0.1250 | 0.1250 | 0.1250 | 0.2100 | 1.0000 | 1.0000 |
| 9 | 32 | 1 | 0.105 | 0.3760 | 0.3955 | -0.0195 | 0.1250 | 0.1289 | 0.1250 | 0.1250 | 0.1250 | 0.2041 | 1.0000 | 1.0000 |

## Mean/Std/Bootstrap CI

- Completed seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`
- Mean trainable accuracy: `0.7406`
- Mean frozen accuracy: `0.5886`
- Mean trainable-frozen delta: `0.1521`
- Delta std: `0.2472`
- Bootstrap 95% CI for delta: `[0.0012, 0.3161]`

## Baselines Table

| method | mean | std |
|---|---:|---:|
| trainable | 0.7406 | 0.2635 |
| frozen | 0.5886 | 0.2095 |
| text_only | 0.1262 | 0.0046 |
| raw_latent | 0.1446 | 0.0270 |
| majority_baseline | 0.1250 | 0.0000 |
| candidate_order_baseline | 0.1250 | 0.0000 |
| single_agent_full_context | 0.2118 | 0.0119 |
| candidate_pair_compatibility_mlp | 1.0000 | 0.0000 |
| all_role_oracle | 1.0000 | 0.0000 |
| single_view_text_role_0 | 0.1238 | 0.0022 |
| single_view_text_role_1 | 0.1248 | 0.0048 |
| single_view_text_role_2 | 0.1235 | 0.0036 |
| single_view_text_role_3 | 0.1241 | 0.0064 |

## Schema-Aware Control Table

| control | mean | std | max | pass |
|---|---:|---:|---:|---|
| candidate_only | 0.1409 | 0.0359 | 0.2207 | `True` |
| view_masked_candidates_visible | 0.1402 | 0.0337 | 0.1963 | `True` |
| schema_only | 0.1399 | 0.0344 | 0.2051 | `True` |
| value_shuffle_within_schema | 0.1296 | 0.0104 | 0.1494 | `True` |
| cross_example_view_bundle_shuffle | 0.1385 | 0.0207 | 0.1846 | `True` |
| candidate_evidence_mismatch | 0.1297 | 0.0080 | 0.1445 | `True` |
| schema_preserved_role_value_shuffle | 0.1352 | 0.0084 | 0.1494 | `True` |
| null_evidence_values | 0.1421 | 0.0337 | 0.2041 | `True` |
| evidence_only_no_candidates | 0.1250 | 0.0000 | 0.1250 | `True` |
| randomized_labels | 0.1207 | 0.0519 | 0.2031 | `True` |
| hidden_states_shuffled_across_examples | 0.1478 | 0.0214 | 0.1904 | `True` |

## Invariance Control Table

| control | mean | delta from trainable | tolerance | pass |
|---|---:|---:|---:|---|
| physical_order_shuffled_roles_preserved | 0.7406 | 0.0000 | 0.0800 | `True` |
| candidate_order_shuffled_with_gold_remap | 0.7018 | -0.0389 | 0.0800 | `True` |

## Role-Embedding Shuffle Diagnostic

| seed | trainable | role_embedding_shuffle_diagnostic | delta | gating? |
|---:|---:|---:|---:|---|
| 0 | 1.0000 | 0.1572 | -0.8428 | `False` |
| 1 | 1.0000 | 0.2510 | -0.7490 | `False` |
| 2 | 0.5410 | 0.1543 | -0.3867 | `False` |
| 3 | 1.0000 | 0.2627 | -0.7373 | `False` |
| 4 | 0.4463 | 0.1699 | -0.2764 | `False` |
| 5 | 1.0000 | 0.2842 | -0.7158 | `False` |
| 6 | 1.0000 | 0.2012 | -0.7988 | `False` |
| 7 | 0.4873 | 0.2773 | -0.2100 | `False` |
| 8 | 0.5557 | 0.2686 | -0.2871 | `False` |
| 9 | 0.3760 | 0.2422 | -0.1338 | `False` |

## Audit Summary

| seed | split leak | output leak | train grad | train delta | frozen grad | frozen delta | checkpoints |
|---:|---|---|---:|---:|---:|---:|---|
| 0 | pass | pass | 0.2997 | 13.8380 | 0.0000 | 0.0000 | pass |
| 1 | pass | pass | 0.1341 | 9.6679 | 0.0000 | 0.0000 | pass |
| 2 | pass | pass | 0.0993 | 11.5711 | 0.0000 | 0.0000 | pass |
| 3 | pass | pass | 0.1688 | 10.9799 | 0.0000 | 0.0000 | pass |
| 4 | pass | pass | 0.2931 | 8.9785 | 0.0000 | 0.0000 | pass |
| 5 | pass | pass | 0.2180 | 12.7805 | 0.0000 | 0.0000 | pass |
| 6 | pass | pass | 0.2249 | 13.6557 | 0.0000 | 0.0000 | pass |
| 7 | pass | pass | 0.2259 | 7.9967 | 0.0000 | 0.0000 | pass |
| 8 | pass | pass | 0.1239 | 10.7018 | 0.0000 | 0.0000 | pass |
| 9 | pass | pass | 0.0802 | 11.0697 | 0.0000 | 0.0000 | pass |

## Acceptance Gates

| criterion | pass | value |
|---|---|---|
| completed seeds >= 10 | `True` | `10` |
| trainable beats frozen on at least 8/10 seeds | `False` | `6` |
| mean trainable-frozen delta >= +0.20 | `False` | `0.15205078125` |
| bootstrap 95% CI lower bound for delta > +0.05 | `False` | `[0.001154785156250001, 0.3161181640625]` |
| trainable beats text-only, raw-latent, majority, candidate-order, and full-context baselines by mean accuracy | `True` | `{"candidate_order_baseline": 0.125, "majority_baseline": 0.125, "raw_latent": 0.14462890625, "single_agent_full_context": 0.21181640625, "text_only": 0.126171875, "trainable_mean": 0.740625}` |
| trainable beats every single-view baseline by mean accuracy | `True` | `{"single_view_text_role_0": 0.123828125, "single_view_text_role_1": 0.1248046875, "single_view_text_role_2": 0.12353515625, "single_view_text_role_3": 0.12412109375}` |
| primary corruption controls collapse near chance | `True` | `{"candidate_evidence_mismatch": 0.1296875, "candidate_only": 0.14091796875, "cross_example_view_bundle_shuffle": 0.1384765625, "evidence_only_no_candidates": 0.125, "hidden_states_shuffled_across_examples": 0.14775390625, "null_evidence_values": 0.14208984375, "randomized_labels": 0.120703125, "schema_only": 0.13994140625, "schema_preserved_role_value_shuffle": 0.13515625, "value_shuffle_within_schema": 0.12958984375, "view_masked_candidates_visible": 0.140234375}` |
| invariance controls preserve accuracy | `True` | `{"candidate_order_shuffled_with_gold_remap": {"mean_accuracy": 0.7017578125, "mean_delta_from_trainable": -0.038867187499999956}, "physical_order_shuffled_roles_preserved": {"mean_accuracy": 0.740625, "mean_delta_from_trainable": 0.0}}` |
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
