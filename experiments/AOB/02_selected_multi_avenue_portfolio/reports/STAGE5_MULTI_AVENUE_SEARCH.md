# Stage 5 Multi-Avenue Search

## Scope

- Fixed benchmark: Stage 3.8 schema-aware balanced_categories_v3.
- Dataset labels, oracle metadata separation, leakage controls, schema-aware corruption controls, and frozen comparator policy are unchanged.
- Baseline reference: Stage 4 `candidate_token_direct_lr3e4_clip1`, trainable `0.7892`, frozen `0.1518`, delta `+0.6374`.
- Selection uses dev metrics from cheap and medium phases only; final test seeds are not used for tuning.

## Search Space

| variant | avenues | prompt | topk | avenue dropout | lr | clip |
|---|---:|---|---:|---:|---:|---:|
| baseline_4x1_candidate_token_direct_lr3e4_clip1 | 1 | single | 0 | 0.00 | 0.0003 | 1.00 |
| avenue2_goal_direct_lr3e4_clip1 | 2 | goal | 0 | 0.00 | 0.0003 | 1.00 |
| avenue4_goal_direct_lr3e4_clip1 | 4 | goal | 0 | 0.00 | 0.0003 | 1.00 |
| avenue4_goal_dropout05_lr3e4_clip1 | 4 | goal | 0 | 0.05 | 0.0003 | 1.00 |
| avenue4_goal_topk4_lr3e4_clip1 | 4 | goal | 4 | 0.00 | 0.0003 | 1.00 |
| avenue2_schema_value_direct_lr3e4_clip1 | 2 | schema_value_split | 0 | 0.00 | 0.0003 | 1.00 |

## Pre-Run Controls

- Source: `computed_for_stage5`
- Shortcut diagnostics pass: `True`

| gate | pass |
|---|---|
| all_role_oracle | `True` |
| candidate_metadata_only | `True` |
| candidate_only | `True` |
| evidence_only_no_candidates_baseline | `True` |
| lexical_overlap | `True` |
| null_evidence_values_baseline | `True` |
| positive_control_learnability | `True` |
| role_pair_only | `True` |
| schema_only_baseline | `True` |
| static_frequency | `True` |
| view_masked_candidates_visible | `True` |

## Phase Summaries

### Cheap

| variant | n | dev train | dev frozen | dev delta | min dev | weak <0.60 | controls | invariance | selected-valid |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| baseline_4x1_candidate_token_direct_lr3e4_clip1 | 3 | 1.0000 | 0.1510 | 0.8490 | 1.0000 | 0 | `False` | `True` | `False` |
| avenue2_goal_direct_lr3e4_clip1 | 3 | 0.5833 | 0.1562 | 0.4271 | 0.1211 | 1 | `True` | `True` | `False` |
| avenue4_goal_direct_lr3e4_clip1 | 3 | 0.5703 | 0.1771 | 0.3932 | 0.1016 | 1 | `True` | `True` | `False` |
| avenue4_goal_dropout05_lr3e4_clip1 | 3 | 0.7031 | 0.1732 | 0.5299 | 0.5898 | 1 | `True` | `True` | `True` |
| avenue4_goal_topk4_lr3e4_clip1 | 3 | 0.4102 | 0.1823 | 0.2279 | 0.0977 | 3 | `True` | `True` | `False` |
| avenue2_schema_value_direct_lr3e4_clip1 | 3 | 0.4987 | 0.1562 | 0.3424 | 0.1211 | 2 | `True` | `True` | `False` |

### Final

| variant | n | dev train | dev frozen | dev delta | min dev | weak <0.60 | controls | invariance | selected-valid |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| avenue4_goal_dropout05_lr3e4_clip1 | 10 | 0.9498 | 0.2873 | 0.6625 | 0.5605 | 1 | `True` | `True` | `True` |

### Medium

| variant | n | dev train | dev frozen | dev delta | min dev | weak <0.60 | controls | invariance | selected-valid |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| avenue4_goal_dropout05_lr3e4_clip1 | 5 | 0.9809 | 0.1781 | 0.8027 | 0.9043 | 0 | `True` | `True` | `True` |

## Selected Variants

- Selected after cheap: `['avenue4_goal_dropout05_lr3e4_clip1']`
- Selected after medium: `['avenue4_goal_dropout05_lr3e4_clip1']`

## Final Validation

| seed | variant | trainable | frozen | delta | physical-order | candidate-order | max single avenue |
|---:|---|---:|---:|---:|---:|---:|---:|
| 30 | avenue4_goal_dropout05_lr3e4_clip1 | 1.0000 | 0.5225 | 0.4775 | 1.0000 | 1.0000 | 0.9961 |
| 31 | avenue4_goal_dropout05_lr3e4_clip1 | 1.0000 | 0.5635 | 0.4365 | 1.0000 | 1.0000 | 0.5342 |
| 32 | avenue4_goal_dropout05_lr3e4_clip1 | 1.0000 | 0.1299 | 0.8701 | 1.0000 | 1.0000 | 0.9854 |
| 33 | avenue4_goal_dropout05_lr3e4_clip1 | 0.9375 | 0.1719 | 0.7656 | 0.9375 | 0.9375 | 0.9277 |
| 34 | avenue4_goal_dropout05_lr3e4_clip1 | 1.0000 | 0.1387 | 0.8613 | 1.0000 | 1.0000 | 0.9561 |
| 35 | avenue4_goal_dropout05_lr3e4_clip1 | 0.3047 | 0.1885 | 0.1162 | 0.3047 | 0.2939 | 0.4258 |
| 36 | avenue4_goal_dropout05_lr3e4_clip1 | 0.9014 | 0.1006 | 0.8008 | 0.9014 | 0.9023 | 0.9014 |
| 37 | avenue4_goal_dropout05_lr3e4_clip1 | 1.0000 | 0.5195 | 0.4805 | 1.0000 | 1.0000 | 0.7979 |
| 38 | avenue4_goal_dropout05_lr3e4_clip1 | 1.0000 | 0.1777 | 0.8223 | 1.0000 | 1.0000 | 1.0000 |
| 39 | avenue4_goal_dropout05_lr3e4_clip1 | 1.0000 | 0.1406 | 0.8594 | 1.0000 | 1.0000 | 0.9648 |

## Final Gates

| criterion | pass | value |
|---|---|---|
| completed seeds >= 10 | `True` | `10` |
| trainable beats frozen on at least 8/10 seeds | `True` | `10` |
| mean trainable-frozen delta >= +0.20 | `True` | `0.6490234375` |
| bootstrap 95% CI lower bound > +0.05 | `True` | `[0.49666748046875003, 0.790234375]` |
| trainable beats text/raw/majority/candidate-order/full-context by mean accuracy | `True` | `{"candidate_order_baseline": 0.125, "majority_baseline": 0.125, "raw_latent": 0.17626953125, "single_agent_full_context": 0.201171875, "text_only": 0.1240234375, "trainable": 0.91435546875}` |
| trainable beats every single-view baseline | `True` | `{"single_view_text_role_0": 0.12568359375, "single_view_text_role_1": 0.12666015625, "single_view_text_role_2": 0.124609375, "single_view_text_role_3": 0.1240234375}` |
| all Stage 4 controls pass | `True` | `{"candidate_evidence_mismatch": 0.1287109375, "candidate_metadata_only": 0.1240234375, "candidate_only": 0.14609375, "cross_example_view_bundle_shuffle": 0.1470703125, "evidence_only_no_candidates": 0.125, "hidden_states_shuffled_across_examples": 0.1625, "null_evidence_values": 0.122265625, "randomized_labels": 0.1169921875, "schema_only": 0.13349609375, "schema_preserved_role_value_shuffle": 0.123828125, "value_shuffle_within_schema": 0.13115234375, "view_masked_candidates_visible": 0.12470703125}` |
| physical and candidate order invariance pass | `True` | `{"candidate_order_shuffled_with_gold_remap": -0.0009765625, "physical_order_shuffled_roles_preserved": 0.0}` |
| no single avenue explains the result | `False` | `[{"base_accuracy": 1.0, "max_single_avenue_accuracy": 0.99609375, "no_single_avenue_explains_result": false, "num_avenues": 4, "single_avenue_accuracy": {"0": 0.896484375, "1": 0.99609375, "2": 0.7939453125, "3": 0.748046875}}, {"base_accuracy": 1.0, "max_single_avenue_accuracy": 0.5341796875, "no_single_avenue_explains_result": true, "num_avenues": 4, "single_avenue_accuracy": {"0": 0.5205078125, "1": 0.107421875, "2": 0.5341796875, "3": 0.2685546875}}, {"base_accuracy": 1.0, "max_single_avenue_accuracy": 0.9853515625, "no_single_avenue_explains_result": false, "num_avenues": 4, "single_avenue_accuracy": {"0": 0.9580078125, "1": 0.5595703125, "2": 0.9853515625, "3": 0.603515625}}, {"base_accuracy": 0.9375, "max_single_avenue_accuracy": 0.927734375, "no_single_avenue_explains_result": false, "num_avenues": 4, "single_avenue_accuracy": {"0": 0.904296875, "1": 0.130859375, "2": 0.927734375, "3": 0.20703125}}, {"base_accuracy": 1.0, "max_single_avenue_accuracy": 0.9560546875, "no_single_avenue_explains_result": true, "num_avenues": 4, "single_avenue_accuracy": {"0": 0.46875, "1": 0.802734375, "2": 0.4873046875, "3": 0.9560546875}}, {"base_accuracy": 0.3046875, "max_single_avenue_accuracy": 0.42578125, "no_single_avenue_explains_result": false, "num_avenues": 4, "single_avenue_accuracy": {"0": 0.3828125, "1": 0.009765625, "2": 0.42578125, "3": 0.3896484375}}, {"base_accuracy": 0.9013671875, "max_single_avenue_accuracy": 0.9013671875, "no_single_avenue_explains_result": false, "num_avenues": 4, "single_avenue_accuracy": {"0": 0.9013671875, "1": 0.2021484375, "2": 0.69921875, "3": 0.1611328125}}, {"base_accuracy": 1.0, "max_single_avenue_accuracy": 0.7978515625, "no_single_avenue_explains_result": true, "num_avenues": 4, "single_avenue_accuracy": {"0": 0.6064453125, "1": 0.1337890625, "2": 0.55078125, "3": 0.7978515625}}, {"base_accuracy": 1.0, "max_single_avenue_accuracy": 1.0, "no_single_avenue_explains_result": false, "num_avenues": 4, "single_avenue_accuracy": {"0": 1.0, "1": 0.5, "2": 0.916015625, "3": 0.46484375}}, {"base_accuracy": 1.0, "max_single_avenue_accuracy": 0.96484375, "no_single_avenue_explains_result": true, "num_avenues": 4, "single_avenue_accuracy": {"0": 0.96484375, "1": 0.2373046875, "2": 0.96484375, "3": 0.9453125}}]` |
| multi-avenue improves over Stage 4 baseline mean or robustness | `True` | `{"stage4_baseline": {"delta": 0.6374, "frozen_mean": 0.1518, "min_seed_accuracy": 0.4277, "name": "candidate_token_direct_lr3e4_clip1", "std_accuracy": 0.2447, "trainable_mean": 0.7892, "weak_seeds_below_060": 4}, "stage5_delta": 0.6490234375, "stage5_mean": 0.91435546875, "stage5_min": 0.3046875, "stage5_std": 0.20585050328517243}` |
| leakage audits pass | `True` | `"split_and_output"` |
| trainable shared model receives gradients and changes | `True` | `"all_final_rows"` |
| frozen comparator remains frozen | `True` | `"all_final_rows"` |
| checkpoints saved | `True` | `["trainable", "frozen", "randomized_labels", "raw_latent", "text_only", "majority_baseline", "candidate_order_baseline", "single_agent_full_context", "single_view_text_role_0", "single_view_text_role_1", "single_view_text_role_2", "single_view_text_role_3", "candidate_pair_compatibility_mlp", "explicit_evidence_oracle"]` |

## Failure Summary

- Final Stage 5 success is not claimed.
- Failed final gates: `no single avenue explains the result`.
- The selected multi-avenue variant often keeps near-full accuracy when only one avenue is visible, so the final result is not clean evidence that multiple avenues jointly drive the gain.

## Conservative Interpretation

- Stage 5 final gates passed: `False`.
- Do not claim improvement from Stage 5 until a single selected variant passes final validation on seeds `[30..39]` with all Stage 4 controls and invariance gates.
