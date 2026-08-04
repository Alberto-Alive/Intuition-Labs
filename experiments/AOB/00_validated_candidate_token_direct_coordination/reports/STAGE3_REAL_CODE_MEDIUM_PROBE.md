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
- Batch size: `16` with fallback `[16, 8, 4]`; effective batch target `16`
- Split sizes: train `512`, dev `128`, test `256`
- Seeds requested: `[0]`

## Completion

- Completed seeds: `[0]`
- Failed or interrupted seeds: `[]`
- OOM retries: `[]`

## Per-Seed Accuracy

| seed | batch | accum | max CUDA GB | trainable | frozen | delta | text | raw | majority | cand-order | max single-view | full ctx | random | hidden shuffle | view masked | view shuffled | role shuffle | physical order | cand-order shuffle | oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 16 | 1 | 0.089 | 1.0000 | 0.8984 | 0.1016 | 0.1250 | 0.1289 | 0.1250 | 0.1250 | 0.1250 | 0.0859 | 0.0000 | 0.2031 | 0.0117 | 0.1562 | 0.1914 | 1.0000 | 1.0000 | 1.0000 |

## Single-View Baselines

| seed | role 0 | role 1 | role 2 | role 3 | max single-view | trainable |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1250 | 0.1250 | 0.1250 | 0.1172 | 0.1250 | 1.0000 |

## Delta Summary

- Mean trainable accuracy: `1.0000`
- Mean frozen accuracy: `0.8984`
- Mean delta: `0.1016`
- Delta std: `0.0000`
- Bootstrap 95% CI: `[0.1016, 0.1016]`

## Corrected Gate Table

| gate | expected behavior | mean test accuracy | max test accuracy | mean delta from trainable | pass |
|---|---|---:|---:|---:|---|
| randomized_labels | mean near chance <= 0.2250 | 0.0000 | 0.0000 | -1.0000 | True |
| hidden_states_shuffled_across_examples | mean near chance <= 0.2250 | 0.2031 | 0.2031 | -0.7969 | True |
| view_masked | mean near chance <= 0.2250 | 0.0117 | 0.0117 | -0.9883 | True |
| view_shuffled | mean near chance <= 0.2250 | 0.1562 | 0.1562 | -0.8438 | True |
| role_labels_shuffled | mean near chance <= 0.2250 | 0.1914 | 0.1914 | -0.8086 | True |
| candidate_order_baseline | mean near chance <= 0.2250 | 0.1250 | 0.1250 | -0.8750 | True |
| majority_baseline | mean near chance <= 0.2250 | 0.1250 | 0.1250 | -0.8750 | True |
| physical_order_shuffled_roles_preserved | preserve accuracy | 1.0000 | 1.0000 | 0.0000 | True |
| candidate_order_shuffled | preserve accuracy | 1.0000 | 1.0000 | 0.0000 | True |

## Audit Summary

| seed | split leak | output leak | shared id | train grad | train delta | frozen grad | frozen delta | active variance | active cosine | one-role failure |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 0 | pass | pass | pass | 0.4243 | 10.5552 | 0.0000 | 0.0000 | 0.3348 | 0.6773 | pass |

## Per-Role Ablation

| seed | base | role 0 masked | role 1 masked | role 2 masked | role 3 masked | one-role failure |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 1.0000 | 0.4570 | 0.3555 | 1.0000 | 0.5625 | False |

## Per-Family Accuracy

| seed | test_real_import_restore_1 | test_real_import_restore_2 | test_real_import_restore_3 | test_real_import_restore_4 |
|---:|---:|---:|---:|---:|
| 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Locked Criteria

| criterion | pass | value |
|---|---|---|
| completed Stage 3 seeds >= 10 | False | `1` |
| trainable beats frozen on at least 8/10 seeds | False | `1` |
| mean trainable-frozen delta >= +0.20 | False | `0.1015625` |
| bootstrap 95% CI lower bound for delta > +0.05 | True | `[0.1015625, 0.1015625]` |
| trainable beats text-only, raw-latent, majority, and candidate-order baselines by mean accuracy | True | `{'trainable_mean': 1.0, 'text_only_mean': 0.125, 'raw_latent_mean': 0.12890625, 'majority_mean': 0.125, 'candidate_order_mean': 0.125}` |
| trainable beats every single-view baseline by mean accuracy | True | `{'single_view_text_role_0': 0.125, 'single_view_text_role_1': 0.125, 'single_view_text_role_2': 0.125, 'single_view_text_role_3': 0.1171875}` |
| corruption controls plus majority/candidate-order remain near chance | True | `{'randomized_labels': {'mean_accuracy': 0.0, 'max_accuracy': 0.0}, 'hidden_states_shuffled_across_examples': {'mean_accuracy': 0.203125, 'max_accuracy': 0.203125}, 'view_masked': {'mean_accuracy': 0.01171875, 'max_accuracy': 0.01171875}, 'view_shuffled': {'mean_accuracy': 0.15625, 'max_accuracy': 0.15625}, 'role_labels_shuffled': {'mean_accuracy': 0.19140625, 'max_accuracy': 0.19140625}, 'candidate_order_baseline': {'mean_accuracy': 0.125, 'max_accuracy': 0.125}, 'majority_baseline': {'mean_accuracy': 0.125, 'max_accuracy': 0.125}}` |
| invariance controls preserve accuracy | True | `{'physical_order_shuffled_roles_preserved': {'mean_accuracy': 1.0, 'mean_delta_from_trainable': 0.0}, 'candidate_order_shuffled': {'mean_accuracy': 1.0, 'mean_delta_from_trainable': 0.0}}` |
| role-label shuffle drops substantially or remains near chance | True | `{'mean_accuracy': 0.19140625, 'max_accuracy': 0.19140625, 'mean_delta_from_trainable': -0.80859375}` |
| no single role explains the result | True | `0` |
| leakage audits pass | True | `split_candidate_patch_hash_and_output` |
| checkpoints saved for every seed and required method | True | `['trainable', 'frozen', 'text_only', 'raw_latent', 'majority_baseline', 'candidate_order_baseline', 'single_agent_full_context', 'single_view_text_role_0', 'single_view_text_role_1', 'single_view_text_role_2', 'single_view_text_role_3', 'explicit_evidence_oracle']` |
| shared trainable model receives gradients and changes | True | `all_completed_seeds` |
| frozen comparator remains frozen | True | `all_completed_seeds` |

## Conservative Interpretation

Stage 3 locked success criteria pass: `False`.
Do not claim Stage 3 success. Treat this as an incomplete or failed hard-validation result until every locked criterion above passes with no hidden failed seeds.
