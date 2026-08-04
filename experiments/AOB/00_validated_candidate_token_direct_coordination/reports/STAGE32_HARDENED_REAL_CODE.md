# Stage 3.2 Hardened Real-Code Import-Restoration Validation

## Scope/Config

- Architecture: `topk_attention_no_head`
- Locked architecture: topk_attention_no_head, shared-weight cloned agent, active/top-k token readout, no message head, no private-cue auxiliary loss, candidate-query coordinator.
- Comparator: exact frozen same-architecture comparator.
- Dataset mode: `real_import_restore_hardened` / `import_restore_redacted_v1`
- Device requested/used: `cuda`
- CUDA device: `NVIDIA GeForce RTX 5070 Ti`
- Mixed precision: `bf16`
- Split sizes: train `2048`, dev `512`, test `1024`
- Seeds requested: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`

## Dataset Hardening

- Private views replace exact symbol, module, import statement, target path, and candidate text with non-identifying typed placeholders.
- Candidate text is redacted with stable non-semantic IDs; exact candidate identity is carried only through structured candidate-query features and audit metadata.
- Distractors are sampled from real local import pairs by same package family, symbol type, module category, provider-name parity, usage pattern, and import-slot parity where possible.
- Primary benchmark is not cross-file-only or two-hop-only.

## Pre-Training Shortcut Diagnostics

- Dataset validity passed: `True`
- Candidate lexical-overlap baseline: `0.1023`
- Static import-frequency baseline: `0.1242`
- Majority baseline: `0.1250`
- Candidate-order baseline: `0.1250`
- Max single-view baseline: `0.1318`
- Explicit structured oracle: `1.0000`

## Completion

- Completed seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`
- Failed or interrupted seeds: `[]`
- OOM retries: `[]`

## Per-Seed Accuracy

| seed | batch | max CUDA GB | trainable | frozen | delta | text | raw | majority | cand-order | max single-view | full ctx | random | hidden shuffle | view masked | view shuffled | role shuffle | physical order | cand-order shuffle | oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 32 | 0.105 | 0.3057 | 0.1982 | 0.1074 | 0.1240 | 0.1250 | 0.1250 | 0.1250 | 0.1250 | 0.1250 | 0.0000 | 0.1602 | 0.1211 | 0.1465 | 0.2578 | 0.3057 | 0.3047 | 1.0000 |
| 1 | 32 | 0.105 | 0.4014 | 0.3975 | 0.0039 | 0.1260 | 0.1250 | 0.1250 | 0.1250 | 0.1250 | 0.1250 | 0.1064 | 0.2002 | 0.2178 | 0.2002 | 0.2139 | 0.4014 | 0.4014 | 1.0000 |
| 2 | 32 | 0.105 | 0.5771 | 0.1045 | 0.4727 | 0.1250 | 0.1250 | 0.1250 | 0.1250 | 0.1396 | 0.1250 | 0.1670 | 0.1387 | 0.1855 | 0.1436 | 0.1328 | 0.5771 | 0.5762 | 1.0000 |
| 3 | 32 | 0.105 | 0.2021 | 0.2930 | -0.0908 | 0.1230 | 0.1250 | 0.1250 | 0.1250 | 0.1289 | 0.1250 | 0.0947 | 0.0898 | 0.1348 | 0.0957 | 0.2021 | 0.2021 | 0.2031 | 1.0000 |
| 4 | 32 | 0.105 | 0.5449 | 0.3721 | 0.1729 | 0.1250 | 0.1250 | 0.1250 | 0.1250 | 0.1250 | 0.1250 | 0.1982 | 0.1377 | 0.1865 | 0.1240 | 0.1709 | 0.5449 | 0.5449 | 1.0000 |
| 5 | 32 | 0.105 | 0.2568 | 0.1895 | 0.0674 | 0.1289 | 0.1250 | 0.1250 | 0.1250 | 0.1279 | 0.1250 | 0.1182 | 0.1406 | 0.1191 | 0.1318 | 0.2510 | 0.2568 | 0.2568 | 1.0000 |
| 6 | 32 | 0.105 | 1.0000 | 0.0830 | 0.9170 | 0.1279 | 0.1250 | 0.1250 | 0.1250 | 0.1250 | 0.1250 | 0.1504 | 0.1406 | 0.1221 | 0.1289 | 0.1484 | 1.0000 | 1.0000 | 1.0000 |
| 7 | 32 | 0.105 | 0.9990 | 0.1924 | 0.8066 | 0.1357 | 0.1250 | 0.1250 | 0.1250 | 0.1309 | 0.1250 | 0.1494 | 0.1484 | 0.1260 | 0.1240 | 0.3271 | 0.9990 | 0.9990 | 1.0000 |
| 8 | 32 | 0.105 | 0.9990 | 0.1465 | 0.8525 | 0.1250 | 0.1250 | 0.1250 | 0.1250 | 0.1318 | 0.1250 | 0.2695 | 0.1904 | 0.1553 | 0.1240 | 0.1699 | 0.9990 | 1.0000 | 1.0000 |
| 9 | 32 | 0.105 | 1.0000 | 0.4609 | 0.5391 | 0.1250 | 0.1250 | 0.1250 | 0.1250 | 0.1279 | 0.1250 | 0.1650 | 0.1396 | 0.1426 | 0.1416 | 0.2646 | 1.0000 | 1.0000 | 1.0000 |

## Mean/CI

- Mean trainable accuracy: `0.6286`
- Mean frozen accuracy: `0.2437`
- Mean delta: `0.3849`
- Delta std: `0.3613`
- Bootstrap 95% CI: `[0.1653, 0.6018]`

## Corruption Controls

| gate | mean accuracy | max accuracy | pass |
|---|---:|---:|---|
| randomized_labels | 0.1419 | 0.2695 | True |
| hidden_states_shuffled_across_examples | 0.1486 | 0.2002 | True |
| view_masked | 0.1511 | 0.2178 | True |
| view_shuffled | 0.1360 | 0.2002 | True |
| role_labels_shuffled | 0.2139 | 0.3271 | True |
| candidate_order_baseline | 0.1250 | 0.1250 | True |
| majority_baseline | 0.1250 | 0.1250 | True |

## Invariance Controls

| gate | mean accuracy | mean delta from trainable | pass |
|---|---:|---:|---|
| physical_order_shuffled_roles_preserved | 0.6286 | 0.0000 | True |
| candidate_order_shuffled | 0.6286 | 0.0000 | True |

## Single-View Baselines

| seed | role 0 | role 1 | role 2 | role 3 |
|---:|---:|---:|---:|---:|
| 0 | 0.1211 | 0.1240 | 0.1240 | 0.1250 |
| 1 | 0.1250 | 0.1240 | 0.1250 | 0.1221 |
| 2 | 0.1260 | 0.1250 | 0.1396 | 0.1250 |
| 3 | 0.1250 | 0.1172 | 0.1289 | 0.1221 |
| 4 | 0.1201 | 0.1162 | 0.1250 | 0.1250 |
| 5 | 0.1250 | 0.1250 | 0.1250 | 0.1279 |
| 6 | 0.1221 | 0.1250 | 0.1084 | 0.1240 |
| 7 | 0.1289 | 0.1309 | 0.1250 | 0.1250 |
| 8 | 0.1318 | 0.1250 | 0.1250 | 0.1279 |
| 9 | 0.1182 | 0.1143 | 0.1279 | 0.1279 |

## Per-Role Ablation

| seed | base | role 0 masked | role 1 masked | role 2 masked | role 3 masked | one-role failure |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.3057 | 0.1182 | 0.2139 | 0.2129 | 0.2139 | True |
| 1 | 0.4014 | 0.3027 | 0.0762 | 0.2939 | 0.2754 | True |
| 2 | 0.5771 | 0.2988 | 0.5771 | 0.2959 | 0.5781 | False |
| 3 | 0.2021 | 0.2012 | 0.2012 | 0.2012 | 0.1348 | False |
| 4 | 0.5449 | 0.4688 | 0.3369 | 0.2539 | 0.5449 | False |
| 5 | 0.2568 | 0.2539 | 0.3262 | 0.2363 | 0.1377 | False |
| 6 | 1.0000 | 0.4980 | 0.4609 | 0.3789 | 1.0000 | False |
| 7 | 0.9990 | 0.9990 | 0.4561 | 0.5234 | 0.5039 | False |
| 8 | 0.9990 | 0.5449 | 0.9990 | 0.5811 | 0.5908 | False |
| 9 | 1.0000 | 0.5439 | 0.5674 | 1.0000 | 0.6162 | False |

## Per-Family Accuracy

| seed | test_real_import_restore_0 | test_real_import_restore_1 | test_real_import_restore_2 | test_real_import_restore_3 | test_real_import_restore_4 | test_real_import_restore_5 | test_real_import_restore_6 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0000 | 0.2341 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7033 |
| 1 | 0.0672 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.4869 | 0.0000 |
| 2 | 0.7285 | 0.0000 | 0.0000 | 0.2296 | 0.0000 | 0.0000 | 0.5803 |
| 3 | 0.0000 | 0.2222 | 0.2584 | 0.1693 | 0.0000 | 0.0000 | 0.0000 |
| 4 | 0.5980 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4310 | 0.6390 |
| 5 | 0.1034 | 0.0000 | 0.0000 | 0.2606 | 0.3602 | 0.0000 | 0.0000 |
| 6 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| 7 | 0.0000 | 1.0000 | 1.0000 | 0.9975 | 0.0000 | 1.0000 | 1.0000 |
| 8 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.9986 | 0.0000 |
| 9 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

## Audit Summary

| seed | split leak | output leak | shared id | train grad | train delta | frozen grad | frozen delta | active variance | active cosine |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 0 | pass | pass | pass | 0.6983 | 9.0630 | 0.0000 | 0.0000 | 0.2469 | 0.7577 |
| 1 | pass | pass | pass | 0.3722 | 14.9547 | 0.0000 | 0.0000 | 0.2080 | 0.7992 |
| 2 | pass | pass | pass | 0.2579 | 12.5596 | 0.0000 | 0.0000 | 0.2663 | 0.7384 |
| 3 | pass | pass | pass | 0.1878 | 14.5074 | 0.0000 | 0.0000 | 0.1295 | 0.8726 |
| 4 | pass | pass | pass | 0.3686 | 23.1545 | 0.0000 | 0.0000 | 0.6081 | 0.3708 |
| 5 | pass | pass | pass | 0.2435 | 10.8977 | 0.0000 | 0.0000 | 0.2001 | 0.8197 |
| 6 | pass | pass | pass | 0.5699 | 20.6482 | 0.0000 | 0.0000 | 0.2607 | 0.7357 |
| 7 | pass | pass | pass | 0.5894 | 16.6283 | 0.0000 | 0.0000 | 0.5498 | 0.4511 |
| 8 | pass | pass | pass | 1.2700 | 23.9859 | 0.0000 | 0.0000 | 0.1873 | 0.8076 |
| 9 | pass | pass | pass | 0.1874 | 9.1533 | 0.0000 | 0.0000 | 0.4660 | 0.5525 |

## Gate Table

| criterion | pass | value |
|---|---|---|
| pre-training shortcut diagnostics passed | True | `[{'gate': 'candidate lexical-overlap baseline <= 0.2250', 'pass': True, 'values': [0.11328125, 0.09765625, 0.0859375, 0.125, 0.126953125, 0.125, 0.125, 0.1328125, 0.05078125, 0.041015625]}, {'gate': 'candidate-order baseline near chance <= 0.2250', 'pass': True, 'values': [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]}, {'gate': 'exact candidate patch text leakage == 0.0000', 'pass': True, 'values': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}, {'gate': 'exact gold full import leakage == 0.0000', 'pass': True, 'values': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}, {'gate': 'exact gold module path leakage == 0.0000', 'pass': True, 'values': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}, {'gate': 'exact gold symbol leakage == 0.0000', 'pass': True, 'values': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}, {'gate': 'explicit structured oracle >= 0.9000', 'pass': True, 'values': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}, {'gate': 'majority baseline near chance <= 0.2250', 'pass': True, 'values': [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]}, {'gate': 'single-view text baselines <= 0.2250', 'pass': True, 'values': [{'single_view_text_role_0': 0.123046875, 'single_view_text_role_1': 0.125, 'single_view_text_role_2': 0.1318359375, 'single_view_text_role_3': 0.125}, {'single_view_text_role_0': 0.125, 'single_view_text_role_1': 0.1220703125, 'single_view_text_role_2': 0.1220703125, 'single_view_text_role_3': 0.1162109375}, {'single_view_text_role_0': 0.125, 'single_view_text_role_1': 0.125, 'single_view_text_role_2': 0.125, 'single_view_text_role_3': 0.1318359375}, {'single_view_text_role_0': 0.125, 'single_view_text_role_1': 0.125, 'single_view_text_role_2': 0.1240234375, 'single_view_text_role_3': 0.125}, {'single_view_text_role_0': 0.125, 'single_view_text_role_1': 0.125, 'single_view_text_role_2': 0.1240234375, 'single_view_text_role_3': 0.125}, {'single_view_text_role_0': 0.119140625, 'single_view_text_role_1': 0.125, 'single_view_text_role_2': 0.125, 'single_view_text_role_3': 0.12890625}, {'single_view_text_role_0': 0.125, 'single_view_text_role_1': 0.125, 'single_view_text_role_2': 0.125, 'single_view_text_role_3': 0.1220703125}, {'single_view_text_role_0': 0.125, 'single_view_text_role_1': 0.125, 'single_view_text_role_2': 0.103515625, 'single_view_text_role_3': 0.125}, {'single_view_text_role_0': 0.125, 'single_view_text_role_1': 0.1240234375, 'single_view_text_role_2': 0.125, 'single_view_text_role_3': 0.1240234375}, {'single_view_text_role_0': 0.1259765625, 'single_view_text_role_1': 0.125, 'single_view_text_role_2': 0.125, 'single_view_text_role_3': 0.125}]}, {'gate': 'static import-frequency baseline <= 0.3000', 'pass': True, 'values': [0.158203125, 0.021484375, 0.0654296875, 0.134765625, 0.228515625, 0.06640625, 0.1494140625, 0.1484375, 0.080078125, 0.189453125]}]` |
| completed Stage 3.2 seeds >= 10 | True | `10` |
| trainable beats frozen on at least 8/10 seeds | True | `9` |
| mean trainable-frozen delta >= +0.20 | True | `0.38486328125` |
| bootstrap 95% CI lower bound for delta > +0.05 | True | `[0.16528564453125003, 0.6017871093749999]` |
| trainable beats text-only, raw-latent, majority, candidate-order, and full-context baselines by mean accuracy | True | `{'trainable_mean': 0.62861328125, 'text_only_mean': 0.1265625, 'raw_latent_mean': 0.125, 'majority_mean': 0.125, 'candidate_order_mean': 0.125, 'full_context_mean': 0.125}` |
| trainable beats every single-view baseline by mean accuracy | True | `{'single_view_text_role_0': 0.12431640625, 'single_view_text_role_1': 0.12265625, 'single_view_text_role_2': 0.125390625, 'single_view_text_role_3': 0.1251953125}` |
| corruption controls plus majority/candidate-order remain near chance | True | `{'randomized_labels': {'mean_accuracy': 0.14189453125, 'max_accuracy': 0.26953125}, 'hidden_states_shuffled_across_examples': {'mean_accuracy': 0.1486328125, 'max_accuracy': 0.2001953125}, 'view_masked': {'mean_accuracy': 0.15107421875, 'max_accuracy': 0.2177734375}, 'view_shuffled': {'mean_accuracy': 0.13603515625, 'max_accuracy': 0.2001953125}, 'role_labels_shuffled': {'mean_accuracy': 0.2138671875, 'max_accuracy': 0.3271484375}, 'candidate_order_baseline': {'mean_accuracy': 0.125, 'max_accuracy': 0.125}, 'majority_baseline': {'mean_accuracy': 0.125, 'max_accuracy': 0.125}}` |
| invariance controls preserve accuracy | True | `{'physical_order_shuffled_roles_preserved': {'mean_accuracy': 0.62861328125, 'mean_delta_from_trainable': 0.0}, 'candidate_order_shuffled': {'mean_accuracy': 0.62861328125, 'mean_delta_from_trainable': 0.0}}` |
| role-label shuffle drops substantially or remains near chance | True | `{'mean_accuracy': 0.2138671875, 'max_accuracy': 0.3271484375, 'mean_delta_from_trainable': -0.41474609375000004}` |
| explicit structured oracle remains high | True | `1.0` |
| no single role explains the result | False | `2` |
| leakage audits pass | True | `split_candidate_patch_hash_and_output` |
| shared trainable model receives gradients and changes | True | `all_completed_seeds` |
| frozen comparator remains frozen | True | `all_completed_seeds` |

## Conservative Interpretation

Stage 3.2 hardened criteria pass: `False`.
Do not claim real-code latent coordination success. Treat the result as invalid or incomplete until shortcut diagnostics, training deltas, controls, invariances, and audits all pass.
