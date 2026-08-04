# Stage 3.2 Hardened Real-Code Import-Restoration Validation

## Scope/Config

- Architecture: `topk_attention_no_head`
- Locked architecture: topk_attention_no_head, shared-weight cloned agent, active/top-k token readout, no message head, no private-cue auxiliary loss, candidate-query coordinator.
- Comparator: exact frozen same-architecture comparator.
- Dataset mode: `real_import_restore_hardened` / `import_restore_redacted_v1`
- Device requested/used: `cuda`
- CUDA device: `NVIDIA GeForce RTX 5070 Ti`
- Mixed precision: `bf16`
- Split sizes: train `256`, dev `128`, test `256`
- Seeds requested: `[0]`

## Dataset Hardening

- Private views replace exact symbol, module, import statement, target path, and candidate text with non-identifying typed placeholders.
- Candidate text is redacted with stable non-semantic IDs; exact candidate identity is carried only through structured candidate-query features and audit metadata.
- Distractors are sampled from real local import pairs by same package family, symbol type, module category, provider-name parity, usage pattern, and import-slot parity where possible.
- Primary benchmark is not cross-file-only or two-hop-only.

## Pre-Training Shortcut Diagnostics

- Dataset validity passed: `True`
- Candidate lexical-overlap baseline: `0.1133`
- Static import-frequency baseline: `0.0898`
- Majority baseline: `0.1250`
- Candidate-order baseline: `0.1250`
- Max single-view baseline: `0.1250`
- Explicit structured oracle: `1.0000`

## Completion

- Completed seeds: `[0]`
- Failed or interrupted seeds: `[]`
- OOM retries: `[]`

## Per-Seed Accuracy

| seed | batch | max CUDA GB | trainable | frozen | delta | text | raw | majority | cand-order | max single-view | full ctx | random | hidden shuffle | view masked | view shuffled | role shuffle | physical order | cand-order shuffle | oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 32 | 0.105 | 0.1562 | 0.1758 | -0.0195 | 0.1250 | 0.1094 | 0.1250 | 0.1250 | 0.1250 | 0.1250 | 0.1328 | 0.1562 | 0.1562 | 0.1562 | 0.1562 | 0.1562 | 0.1562 | 1.0000 |

## Mean/CI

- Mean trainable accuracy: `0.1562`
- Mean frozen accuracy: `0.1758`
- Mean delta: `-0.0195`
- Delta std: `0.0000`
- Bootstrap 95% CI: `[-0.0195, -0.0195]`

## Corruption Controls

| gate | mean accuracy | max accuracy | pass |
|---|---:|---:|---|
| randomized_labels | 0.1328 | 0.1328 | True |
| hidden_states_shuffled_across_examples | 0.1562 | 0.1562 | True |
| view_masked | 0.1562 | 0.1562 | True |
| view_shuffled | 0.1562 | 0.1562 | True |
| role_labels_shuffled | 0.1562 | 0.1562 | True |
| candidate_order_baseline | 0.1250 | 0.1250 | True |
| majority_baseline | 0.1250 | 0.1250 | True |

## Invariance Controls

| gate | mean accuracy | mean delta from trainable | pass |
|---|---:|---:|---|
| physical_order_shuffled_roles_preserved | 0.1562 | 0.0000 | True |
| candidate_order_shuffled | 0.1562 | 0.0000 | True |

## Single-View Baselines

| seed | role 0 | role 1 | role 2 | role 3 |
|---:|---:|---:|---:|---:|
| 0 | 0.1250 | 0.1250 | 0.1250 | 0.1250 |

## Per-Role Ablation

| seed | base | role 0 masked | role 1 masked | role 2 masked | role 3 masked | one-role failure |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.1562 | 0.1562 | 0.1562 | 0.1562 | 0.1562 | False |

## Per-Family Accuracy

| seed | test_real_import_restore_1 | test_real_import_restore_2 | test_real_import_restore_3 | test_real_import_restore_6 |
|---:|---:|---:|---:|---:|
| 0 | 0.2564 | 0.0000 | 0.0000 | 0.0000 |

## Audit Summary

| seed | split leak | output leak | shared id | train grad | train delta | frozen grad | frozen delta | active variance | active cosine |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 0 | pass | pass | pass | 0.2190 | 2.7857 | 0.0000 | 0.0000 | 0.0288 | 0.9716 |

## Gate Table

| criterion | pass | value |
|---|---|---|
| pre-training shortcut diagnostics passed | True | `[{'gate': 'candidate lexical-overlap baseline <= 0.2250', 'pass': True, 'values': [0.11328125]}, {'gate': 'candidate-order baseline near chance <= 0.2250', 'pass': True, 'values': [0.125]}, {'gate': 'exact candidate patch text leakage == 0.0000', 'pass': True, 'values': [0.0]}, {'gate': 'exact gold full import leakage == 0.0000', 'pass': True, 'values': [0.0]}, {'gate': 'exact gold module path leakage == 0.0000', 'pass': True, 'values': [0.0]}, {'gate': 'exact gold symbol leakage == 0.0000', 'pass': True, 'values': [0.0]}, {'gate': 'explicit structured oracle >= 0.9000', 'pass': True, 'values': [1.0]}, {'gate': 'majority baseline near chance <= 0.2250', 'pass': True, 'values': [0.125]}, {'gate': 'single-view text baselines <= 0.2250', 'pass': True, 'values': [{'single_view_text_role_0': 0.125, 'single_view_text_role_1': 0.125, 'single_view_text_role_2': 0.125, 'single_view_text_role_3': 0.1015625}]}, {'gate': 'static import-frequency baseline <= 0.3000', 'pass': True, 'values': [0.08984375]}]` |
| completed Stage 3.2 seeds >= 10 | False | `1` |
| trainable beats frozen on at least 8/10 seeds | False | `0` |
| mean trainable-frozen delta >= +0.20 | False | `-0.01953125` |
| bootstrap 95% CI lower bound for delta > +0.05 | False | `[-0.01953125, -0.01953125]` |
| trainable beats text-only, raw-latent, majority, candidate-order, and full-context baselines by mean accuracy | True | `{'trainable_mean': 0.15625, 'text_only_mean': 0.125, 'raw_latent_mean': 0.109375, 'majority_mean': 0.125, 'candidate_order_mean': 0.125, 'full_context_mean': 0.125}` |
| trainable beats every single-view baseline by mean accuracy | True | `{'single_view_text_role_0': 0.125, 'single_view_text_role_1': 0.125, 'single_view_text_role_2': 0.125, 'single_view_text_role_3': 0.125}` |
| corruption controls plus majority/candidate-order remain near chance | True | `{'randomized_labels': {'mean_accuracy': 0.1328125, 'max_accuracy': 0.1328125}, 'hidden_states_shuffled_across_examples': {'mean_accuracy': 0.15625, 'max_accuracy': 0.15625}, 'view_masked': {'mean_accuracy': 0.15625, 'max_accuracy': 0.15625}, 'view_shuffled': {'mean_accuracy': 0.15625, 'max_accuracy': 0.15625}, 'role_labels_shuffled': {'mean_accuracy': 0.15625, 'max_accuracy': 0.15625}, 'candidate_order_baseline': {'mean_accuracy': 0.125, 'max_accuracy': 0.125}, 'majority_baseline': {'mean_accuracy': 0.125, 'max_accuracy': 0.125}}` |
| invariance controls preserve accuracy | True | `{'physical_order_shuffled_roles_preserved': {'mean_accuracy': 0.15625, 'mean_delta_from_trainable': 0.0}, 'candidate_order_shuffled': {'mean_accuracy': 0.15625, 'mean_delta_from_trainable': 0.0}}` |
| role-label shuffle drops substantially or remains near chance | True | `{'mean_accuracy': 0.15625, 'max_accuracy': 0.15625, 'mean_delta_from_trainable': 0.0}` |
| explicit structured oracle remains high | True | `1.0` |
| no single role explains the result | True | `0` |
| leakage audits pass | True | `split_candidate_patch_hash_and_output` |
| shared trainable model receives gradients and changes | True | `all_completed_seeds` |
| frozen comparator remains frozen | True | `all_completed_seeds` |

## Conservative Interpretation

Stage 3.2 hardened criteria pass: `False`.
Do not claim real-code latent coordination success. Treat the result as invalid or incomplete until shortcut diagnostics, training deltas, controls, invariances, and audits all pass.
