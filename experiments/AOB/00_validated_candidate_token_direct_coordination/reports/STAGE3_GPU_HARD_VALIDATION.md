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
- Seeds requested: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`

## Completion

- Completed seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`
- Failed or interrupted seeds: `[]`
- OOM retries: `[]`

## Per-Seed Accuracy

| seed | batch | accum | max CUDA GB | trainable | frozen | delta | text | raw | majority | cand-order | max single-view | full ctx | random | hidden shuffle | view masked | view shuffled | role shuffle | physical order | cand-order shuffle | oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 32 | 1 | 0.105 | 1.0000 | 0.8594 | 0.1406 | 0.1299 | 0.1553 | 0.1250 | 0.1250 | 0.1299 | 0.1289 | 0.1211 | 0.2275 | 0.0078 | 0.1621 | 0.2646 | 1.0000 | 1.0000 | 1.0000 |
| 1 | 32 | 1 | 0.105 | 0.9570 | 0.4648 | 0.4922 | 0.1221 | 0.1240 | 0.1250 | 0.1250 | 0.1309 | 0.1309 | 0.1572 | 0.1885 | 0.0801 | 0.1582 | 0.2266 | 0.9570 | 0.9531 | 1.0000 |
| 2 | 32 | 1 | 0.105 | 1.0000 | 1.0000 | 0.0000 | 0.1162 | 0.1143 | 0.1250 | 0.1250 | 0.1260 | 0.0977 | 0.2881 | 0.1465 | 0.1270 | 0.1250 | 0.2920 | 1.0000 | 1.0000 | 1.0000 |
| 3 | 32 | 1 | 0.105 | 0.8584 | 0.7090 | 0.1494 | 0.1348 | 0.1143 | 0.1250 | 0.1250 | 0.1250 | 0.1162 | 0.0762 | 0.1953 | 0.0957 | 0.1162 | 0.1807 | 0.8584 | 0.8613 | 1.0000 |
| 4 | 32 | 1 | 0.105 | 1.0000 | 0.6494 | 0.3506 | 0.1299 | 0.1270 | 0.1250 | 0.1250 | 0.1279 | 0.1240 | 0.0986 | 0.1523 | 0.1045 | 0.1279 | 0.1973 | 1.0000 | 1.0000 | 1.0000 |
| 5 | 32 | 1 | 0.105 | 0.8838 | 0.7217 | 0.1621 | 0.1250 | 0.1221 | 0.1250 | 0.1250 | 0.1357 | 0.1221 | 0.1377 | 0.1602 | 0.1396 | 0.1758 | 0.1758 | 0.8838 | 0.8779 | 1.0000 |
| 6 | 32 | 1 | 0.105 | 1.0000 | 0.7305 | 0.2695 | 0.1250 | 0.1250 | 0.1250 | 0.1250 | 0.1406 | 0.1152 | 0.1035 | 0.1406 | 0.1035 | 0.1289 | 0.2324 | 1.0000 | 1.0000 | 1.0000 |
| 7 | 32 | 1 | 0.105 | 0.7070 | 0.8877 | -0.1807 | 0.1328 | 0.1367 | 0.1250 | 0.1250 | 0.1396 | 0.0996 | 0.0771 | 0.1357 | 0.1143 | 0.1396 | 0.2510 | 0.7070 | 0.7031 | 1.0000 |
| 8 | 32 | 1 | 0.105 | 0.8193 | 0.7588 | 0.0605 | 0.1250 | 0.1211 | 0.1250 | 0.1250 | 0.1377 | 0.1387 | 0.1113 | 0.1514 | 0.2471 | 0.1504 | 0.2344 | 0.8193 | 0.8242 | 1.0000 |
| 9 | 32 | 1 | 0.105 | 1.0000 | 0.6299 | 0.3701 | 0.1240 | 0.1152 | 0.1250 | 0.1250 | 0.1250 | 0.1318 | 0.0000 | 0.1445 | 0.1650 | 0.1475 | 0.1592 | 1.0000 | 1.0000 | 1.0000 |

## Single-View Baselines

| seed | role 0 | role 1 | role 2 | role 3 | max single-view | trainable |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1182 | 0.1250 | 0.1299 | 0.1221 | 0.1299 | 1.0000 |
| 1 | 0.1113 | 0.1309 | 0.1201 | 0.1250 | 0.1309 | 0.9570 |
| 2 | 0.1250 | 0.1230 | 0.1260 | 0.1172 | 0.1260 | 1.0000 |
| 3 | 0.1172 | 0.1250 | 0.1191 | 0.1240 | 0.1250 | 0.8584 |
| 4 | 0.1279 | 0.1230 | 0.1250 | 0.1182 | 0.1279 | 1.0000 |
| 5 | 0.1357 | 0.1094 | 0.1230 | 0.1172 | 0.1357 | 0.8838 |
| 6 | 0.1201 | 0.1406 | 0.1250 | 0.1318 | 0.1406 | 1.0000 |
| 7 | 0.1396 | 0.1309 | 0.1318 | 0.1289 | 0.1396 | 0.7070 |
| 8 | 0.1279 | 0.1377 | 0.1172 | 0.1191 | 0.1377 | 0.8193 |
| 9 | 0.1182 | 0.1064 | 0.1250 | 0.1250 | 0.1250 | 1.0000 |

## Delta Summary

- Mean trainable accuracy: `0.9226`
- Mean frozen accuracy: `0.7411`
- Mean delta: `0.1814`
- Delta std: `0.1872`
- Bootstrap 95% CI: `[0.0750, 0.2957]`

## Corrected Gate Table

| gate | expected behavior | mean test accuracy | max test accuracy | mean delta from trainable | pass |
|---|---|---:|---:|---:|---|
| randomized_labels | mean near chance <= 0.2250 | 0.1171 | 0.2881 | -0.8055 | True |
| hidden_states_shuffled_across_examples | mean near chance <= 0.2250 | 0.1643 | 0.2275 | -0.7583 | True |
| view_masked | mean near chance <= 0.2250 | 0.1185 | 0.2471 | -0.8041 | True |
| view_shuffled | mean near chance <= 0.2250 | 0.1432 | 0.1758 | -0.7794 | True |
| role_labels_shuffled | mean near chance <= 0.2250 | 0.2214 | 0.2920 | -0.7012 | True |
| candidate_order_baseline | mean near chance <= 0.2250 | 0.1250 | 0.1250 | -0.7976 | True |
| majority_baseline | mean near chance <= 0.2250 | 0.1250 | 0.1250 | -0.7976 | True |
| physical_order_shuffled_roles_preserved | preserve accuracy | 0.9226 | 1.0000 | 0.0000 | True |
| candidate_order_shuffled | preserve accuracy | 0.9220 | 1.0000 | -0.0006 | True |

## Audit Summary

| seed | split leak | output leak | shared id | train grad | train delta | frozen grad | frozen delta | active variance | active cosine | one-role failure |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 0 | pass | pass | pass | 0.2823 | 10.7433 | 0.0000 | 0.0000 | 0.1374 | 0.8676 | pass |
| 1 | pass | pass | pass | 0.3566 | 14.9005 | 0.0000 | 0.0000 | 0.3618 | 0.6493 | pass |
| 2 | pass | pass | pass | 0.2806 | 13.1869 | 0.0000 | 0.0000 | 0.4695 | 0.5449 | pass |
| 3 | pass | pass | pass | 0.3140 | 12.5649 | 0.0000 | 0.0000 | 0.3636 | 0.6357 | pass |
| 4 | pass | pass | pass | 0.2695 | 13.5892 | 0.0000 | 0.0000 | 0.4113 | 0.6003 | pass |
| 5 | pass | pass | pass | 0.2671 | 14.8487 | 0.0000 | 0.0000 | 0.4367 | 0.5719 | pass |
| 6 | pass | pass | pass | 0.2029 | 12.1480 | 0.0000 | 0.0000 | 0.4148 | 0.6135 | pass |
| 7 | pass | pass | pass | 0.3524 | 13.4756 | 0.0000 | 0.0000 | 0.3114 | 0.7009 | pass |
| 8 | pass | pass | pass | 0.2815 | 14.1405 | 0.0000 | 0.0000 | 0.3279 | 0.6831 | pass |
| 9 | pass | pass | pass | 0.2013 | 13.2010 | 0.0000 | 0.0000 | 0.3684 | 0.6413 | pass |

## Per-Role Ablation

| seed | base | role 0 masked | role 1 masked | role 2 masked | role 3 masked | one-role failure |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 1.0000 | 0.7100 | 0.6787 | 0.3887 | 1.0000 | False |
| 1 | 0.9570 | 0.5996 | 0.2627 | 0.7705 | 0.7002 | False |
| 2 | 1.0000 | 0.5449 | 0.6328 | 1.0000 | 0.3174 | False |
| 3 | 0.8584 | 0.6006 | 0.4014 | 0.6719 | 0.6895 | False |
| 4 | 1.0000 | 0.5098 | 0.5186 | 0.5713 | 1.0000 | False |
| 5 | 0.8838 | 0.5938 | 0.3174 | 0.7139 | 0.6611 | False |
| 6 | 1.0000 | 0.6602 | 0.6016 | 0.5195 | 1.0000 | False |
| 7 | 0.7070 | 0.3018 | 0.3975 | 0.7080 | 0.5928 | False |
| 8 | 0.8193 | 0.6797 | 0.3398 | 0.7578 | 0.8184 | False |
| 9 | 1.0000 | 0.4199 | 0.3877 | 0.4883 | 0.9990 | False |

## Per-Family Accuracy

| seed | test_real_import_restore_0 | test_real_import_restore_1 | test_real_import_restore_2 | test_real_import_restore_3 | test_real_import_restore_4 | test_real_import_restore_5 | test_real_import_restore_6 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 1 | 0.8398 | 0.0000 | 0.8800 | 0.9684 | 0.0000 | 0.9985 | 0.0000 |
| 2 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 3 | 1.0000 | 1.0000 | 0.0000 | 0.9010 | 0.0000 | 0.0000 | 0.0000 |
| 4 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| 5 | 0.9943 | 0.0000 | 0.0000 | 0.8078 | 1.0000 | 0.0000 | 0.0000 |
| 6 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| 7 | 0.0000 | 0.7391 | 0.0000 | 0.6856 | 0.0000 | 0.6545 | 0.7791 |
| 8 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.7365 | 0.0000 |
| 9 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

## Locked Criteria

| criterion | pass | value |
|---|---|---|
| completed Stage 3 seeds >= 10 | True | `10` |
| trainable beats frozen on at least 8/10 seeds | True | `8` |
| mean trainable-frozen delta >= +0.20 | False | `0.1814453125` |
| bootstrap 95% CI lower bound for delta > +0.05 | True | `[0.0749853515625, 0.29571044921874995]` |
| trainable beats text-only, raw-latent, majority, and candidate-order baselines by mean accuracy | True | `{'trainable_mean': 0.92255859375, 'text_only_mean': 0.12646484375, 'raw_latent_mean': 0.12548828125, 'majority_mean': 0.125, 'candidate_order_mean': 0.125}` |
| trainable beats every single-view baseline by mean accuracy | True | `{'single_view_text_role_0': 0.12412109375, 'single_view_text_role_1': 0.1251953125, 'single_view_text_role_2': 0.12421875, 'single_view_text_role_3': 0.1228515625}` |
| corruption controls plus majority/candidate-order remain near chance | True | `{'randomized_labels': {'mean_accuracy': 0.11708984375, 'max_accuracy': 0.2880859375}, 'hidden_states_shuffled_across_examples': {'mean_accuracy': 0.1642578125, 'max_accuracy': 0.2275390625}, 'view_masked': {'mean_accuracy': 0.11845703125, 'max_accuracy': 0.2470703125}, 'view_shuffled': {'mean_accuracy': 0.1431640625, 'max_accuracy': 0.17578125}, 'role_labels_shuffled': {'mean_accuracy': 0.22138671875, 'max_accuracy': 0.2919921875}, 'candidate_order_baseline': {'mean_accuracy': 0.125, 'max_accuracy': 0.125}, 'majority_baseline': {'mean_accuracy': 0.125, 'max_accuracy': 0.125}}` |
| invariance controls preserve accuracy | True | `{'physical_order_shuffled_roles_preserved': {'mean_accuracy': 0.92255859375, 'mean_delta_from_trainable': 0.0}, 'candidate_order_shuffled': {'mean_accuracy': 0.92197265625, 'mean_delta_from_trainable': -0.0005859375000000222}}` |
| role-label shuffle drops substantially or remains near chance | True | `{'mean_accuracy': 0.22138671875, 'max_accuracy': 0.2919921875, 'mean_delta_from_trainable': -0.701171875}` |
| no single role explains the result | True | `0` |
| leakage audits pass | True | `split_candidate_patch_hash_and_output` |
| checkpoints saved for every seed and required method | True | `['trainable', 'frozen', 'text_only', 'raw_latent', 'majority_baseline', 'candidate_order_baseline', 'single_agent_full_context', 'single_view_text_role_0', 'single_view_text_role_1', 'single_view_text_role_2', 'single_view_text_role_3', 'explicit_evidence_oracle']` |
| shared trainable model receives gradients and changes | True | `all_completed_seeds` |
| frozen comparator remains frozen | True | `all_completed_seeds` |

## Conservative Interpretation

Stage 3 locked success criteria pass: `False`.
Do not claim Stage 3 success. Treat this as an incomplete or failed hard-validation result until every locked criterion above passes with no hidden failed seeds.
