# Stage 3 GPU Hard Validation

## Scope

- Architecture: `topk_attention_no_head`
- Architecture changes: forbidden
- Dataset: real program-analysis import-restoration benchmark
- Comparator: exact frozen same-architecture comparator required
- Device requested/used: `cuda`
- CUDA device: `NVIDIA GeForce RTX 5070 Ti`
- CUDA total memory: `17094475776` bytes
- Mixed precision: `bf16`
- Batch size: `32` with fallback `[32, 16, 8]`; effective batch target `32`
- Split sizes: train `2048`, dev `512`, test `1024`
- Seeds requested: `[7]`

## Completion

- Completed seeds: `[7]`
- Failed or interrupted seeds: `[]`
- OOM retries: `[]`

## Per-Seed Accuracy

| seed | batch | accum | max CUDA GB | trainable | frozen | delta | text | raw | majority | cand-order | max single-view | full ctx | random | hidden shuffle | view masked | view shuffled | role shuffle | physical order | cand-order shuffle | oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 32 | 1 | 0.105 | 0.7070 | 0.8877 | -0.1807 | 0.1328 | 0.1367 | 0.1250 | 0.1250 | 0.1396 | 0.0996 | 0.0771 | 0.1357 | 0.1143 | 0.1396 | 0.2510 | 0.7070 | 0.7031 | 1.0000 |

## Single-View Baselines

| seed | role 0 | role 1 | role 2 | role 3 | max single-view | trainable |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.1396 | 0.1240 | 0.1318 | 0.1289 | 0.1396 | 0.7070 |

## Delta Summary

- Mean trainable accuracy: `0.7070`
- Mean frozen accuracy: `0.8877`
- Mean delta: `-0.1807`
- Delta std: `0.0000`
- Bootstrap 95% CI: `[-0.1807, -0.1807]`

## Corrected Gate Table

| gate | expected behavior | mean test accuracy | max test accuracy | mean delta from trainable | pass |
|---|---|---:|---:|---:|---|
| randomized_labels | mean near chance <= 0.2250 | 0.0771 | 0.0771 | -0.6299 | True |
| hidden_states_shuffled_across_examples | mean near chance <= 0.2250 | 0.1357 | 0.1357 | -0.5713 | True |
| view_masked | mean near chance <= 0.2250 | 0.1143 | 0.1143 | -0.5928 | True |
| view_shuffled | mean near chance <= 0.2250 | 0.1396 | 0.1396 | -0.5674 | True |
| role_labels_shuffled | mean near chance <= 0.2250 | 0.2510 | 0.2510 | -0.4561 | False |
| candidate_order_baseline | mean near chance <= 0.2250 | 0.1250 | 0.1250 | -0.5820 | True |
| majority_baseline | mean near chance <= 0.2250 | 0.1250 | 0.1250 | -0.5820 | True |
| physical_order_shuffled_roles_preserved | preserve accuracy | 0.7070 | 0.7070 | 0.0000 | True |
| candidate_order_shuffled | preserve accuracy | 0.7031 | 0.7031 | -0.0039 | True |

## Audit Summary

| seed | split leak | output leak | shared id | train grad | train delta | frozen grad | frozen delta | active variance | active cosine | one-role failure |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 7 | pass | pass | pass | 0.3084 | 13.4756 | 0.0000 | 0.0000 | 0.3114 | 0.7009 | pass |

## Per-Role Ablation

| seed | base | role 0 masked | role 1 masked | role 2 masked | role 3 masked | one-role failure |
|---:|---:|---:|---:|---:|---:|---|
| 7 | 0.7070 | 0.3018 | 0.3975 | 0.7080 | 0.5928 | False |

## Per-Family Accuracy

| seed | test_real_import_restore_1 | test_real_import_restore_2 | test_real_import_restore_3 | test_real_import_restore_5 | test_real_import_restore_6 |
|---:|---:|---:|---:|---:|---:|
| 7 | 0.7391 | 0.0000 | 0.6856 | 0.6545 | 0.7791 |

## Locked Criteria

| criterion | pass | value |
|---|---|---|
| completed Stage 3 seeds >= 10 | False | `1` |
| trainable beats frozen on at least 8/10 seeds | False | `0` |
| mean trainable-frozen delta >= +0.20 | False | `-0.1806640625` |
| bootstrap 95% CI lower bound for delta > +0.05 | False | `[-0.1806640625, -0.1806640625]` |
| trainable beats text-only, raw-latent, majority, and candidate-order baselines by mean accuracy | True | `{'trainable_mean': 0.70703125, 'text_only_mean': 0.1328125, 'raw_latent_mean': 0.13671875, 'majority_mean': 0.125, 'candidate_order_mean': 0.125}` |
| trainable beats every single-view baseline by mean accuracy | True | `{'single_view_text_role_0': 0.1396484375, 'single_view_text_role_1': 0.1240234375, 'single_view_text_role_2': 0.1318359375, 'single_view_text_role_3': 0.12890625}` |
| corruption controls plus majority/candidate-order remain near chance | False | `{'randomized_labels': {'mean_accuracy': 0.0771484375, 'max_accuracy': 0.0771484375}, 'hidden_states_shuffled_across_examples': {'mean_accuracy': 0.1357421875, 'max_accuracy': 0.1357421875}, 'view_masked': {'mean_accuracy': 0.1142578125, 'max_accuracy': 0.1142578125}, 'view_shuffled': {'mean_accuracy': 0.1396484375, 'max_accuracy': 0.1396484375}, 'role_labels_shuffled': {'mean_accuracy': 0.2509765625, 'max_accuracy': 0.2509765625}, 'candidate_order_baseline': {'mean_accuracy': 0.125, 'max_accuracy': 0.125}, 'majority_baseline': {'mean_accuracy': 0.125, 'max_accuracy': 0.125}}` |
| invariance controls preserve accuracy | True | `{'physical_order_shuffled_roles_preserved': {'mean_accuracy': 0.70703125, 'mean_delta_from_trainable': 0.0}, 'candidate_order_shuffled': {'mean_accuracy': 0.703125, 'mean_delta_from_trainable': -0.00390625}}` |
| role-label shuffle drops substantially or remains near chance | True | `{'mean_accuracy': 0.2509765625, 'max_accuracy': 0.2509765625, 'mean_delta_from_trainable': -0.4560546875}` |
| no single role explains the result | True | `0` |
| leakage audits pass | True | `split_candidate_patch_hash_and_output` |
| checkpoints saved for every seed and required method | True | `['trainable', 'frozen', 'text_only', 'raw_latent', 'majority_baseline', 'candidate_order_baseline', 'single_agent_full_context', 'single_view_text_role_0', 'single_view_text_role_1', 'single_view_text_role_2', 'single_view_text_role_3', 'explicit_evidence_oracle']` |
| shared trainable model receives gradients and changes | True | `all_completed_seeds` |
| frozen comparator remains frozen | True | `all_completed_seeds` |

## Conservative Interpretation

Stage 3 locked success criteria pass: `False`.
Do not claim Stage 3 success. Treat this as an incomplete or failed hard-validation result until every locked criterion above passes with no hidden failed seeds.
