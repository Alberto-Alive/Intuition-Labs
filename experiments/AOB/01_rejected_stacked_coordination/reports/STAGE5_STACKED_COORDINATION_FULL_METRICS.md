# Stage 5 Stacked Candidate-Token-Direct Coordination Full Metrics

## Scope

- Baseline: `candidate_token_direct_lr3e4_clip1`.
- Dataset/control policy: fixed Stage 3.8 schema-aware benchmark; no label, metadata-separation, leakage, or corruption-control changes.
- Comparator: exact same coordinator architecture per variant with shared model frozen.
- Search protocol: cheap `[0,1,2]`, medium `[0,1,2,3,4]`, final fresh `[40..49]`; selection uses dev metrics and controls.
- Final locked variant: `None`.

## Required Questions

- Did stacking improve mean accuracy?: `no`
- Did stacking improve weak seeds?: `not_run`
- Did stacking reduce variance?: `no`
- Did stacking improve hard families?: `see per_family_accuracy in JSON`
- Did stacking increase trainable-frozen delta?: `no`
- Did stacking keep controls near chance?: `no`
- Did stacking preserve invariance?: `see success_criteria`
- Did later blocks fix earlier errors?: `see depth_metrics.later_blocks_transition_counts`
- Did later blocks introduce new errors?: `see depth_metrics.later_blocks_transition_counts`
- Is the improvement worth the compute?: `see compute_metrics and efficiency_normalized_metrics`
- Evidence of iterative refinement rather than just more parameters?: `supported only if early-block transitions show wrong->right exceeding right->wrong with controls passing`

## Variant Summaries

### Cheap

| variant | blocks | seeds | mean acc | min | std | delta | vs 1-block | weak<0.60 | controls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stacked_candidate_token_direct_4blocks_lr3e4_clip1 | 4 | 3 | 0.9857 | 0.9570 | 0.0203 | 0.8477 | 0.0260 | 0 | `True` |
| candidate_token_direct_lr3e4_clip1 | 1 | 3 | 0.9596 | 0.8789 | 0.0571 | 0.8021 | 0.0000 | 0 | `True` |
| stacked_candidate_token_direct_2blocks_lr3e4_clip1 | 2 | 3 | 0.8047 | 0.5156 | 0.2074 | 0.6497 | -0.1549 | 1 | `True` |
| stacked_candidate_token_direct_3blocks_lr3e4_clip1 | 3 | 3 | 0.6888 | 0.0742 | 0.4346 | 0.5378 | -0.2708 | 1 | `True` |

### Medium

| variant | blocks | seeds | mean acc | min | std | delta | vs 1-block | weak<0.60 | controls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| candidate_token_direct_lr3e4_clip1 | 1 | 5 | 0.9313 | 0.8691 | 0.0589 | 0.7941 | 0.0000 | 0 | `True` |
| stacked_candidate_token_direct_4blocks_lr3e4_clip1 | 4 | 5 | 0.7910 | 0.4277 | 0.2450 | 0.6129 | -0.1402 | 2 | `True` |

## Per-Seed Final Rows

| variant | seed | trainable | frozen | delta | text | raw | oracle | weak | peak GB | inf ms/ex |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|

## Controls

| phase | variant | control | mean | pass |
|---|---|---|---:|---|
| cheap | stacked_candidate_token_direct_4blocks_lr3e4_clip1 | candidate_only | 0.1029 | `True` |
| cheap | stacked_candidate_token_direct_4blocks_lr3e4_clip1 | candidate_metadata_only | 0.1237 | `True` |
| cheap | stacked_candidate_token_direct_4blocks_lr3e4_clip1 | view_masked_candidates_visible | 0.1198 | `True` |
| cheap | stacked_candidate_token_direct_4blocks_lr3e4_clip1 | schema_only | 0.1328 | `True` |
| cheap | stacked_candidate_token_direct_4blocks_lr3e4_clip1 | null_evidence_values | 0.1237 | `True` |
| cheap | stacked_candidate_token_direct_4blocks_lr3e4_clip1 | evidence_only_no_candidates | 0.1250 | `True` |
| cheap | stacked_candidate_token_direct_4blocks_lr3e4_clip1 | value_shuffle_within_schema | 0.1497 | `True` |
| cheap | stacked_candidate_token_direct_4blocks_lr3e4_clip1 | cross_example_view_bundle_shuffle | 0.1641 | `True` |
| cheap | stacked_candidate_token_direct_4blocks_lr3e4_clip1 | candidate_evidence_mismatch | 0.1029 | `True` |
| cheap | stacked_candidate_token_direct_4blocks_lr3e4_clip1 | schema_preserved_role_value_shuffle | 0.1445 | `True` |
| cheap | stacked_candidate_token_direct_4blocks_lr3e4_clip1 | randomized_labels | 0.1198 | `True` |
| cheap | stacked_candidate_token_direct_4blocks_lr3e4_clip1 | hidden_states_shuffled_across_examples | 0.1667 | `True` |
| cheap | candidate_token_direct_lr3e4_clip1 | candidate_only | 0.1367 | `True` |
| cheap | candidate_token_direct_lr3e4_clip1 | candidate_metadata_only | 0.1237 | `True` |
| cheap | candidate_token_direct_lr3e4_clip1 | view_masked_candidates_visible | 0.1055 | `True` |
| cheap | candidate_token_direct_lr3e4_clip1 | schema_only | 0.1328 | `True` |
| cheap | candidate_token_direct_lr3e4_clip1 | null_evidence_values | 0.1367 | `True` |
| cheap | candidate_token_direct_lr3e4_clip1 | evidence_only_no_candidates | 0.1250 | `True` |
| cheap | candidate_token_direct_lr3e4_clip1 | value_shuffle_within_schema | 0.1367 | `True` |
| cheap | candidate_token_direct_lr3e4_clip1 | cross_example_view_bundle_shuffle | 0.1706 | `True` |
| cheap | candidate_token_direct_lr3e4_clip1 | candidate_evidence_mismatch | 0.1016 | `True` |
| cheap | candidate_token_direct_lr3e4_clip1 | schema_preserved_role_value_shuffle | 0.1328 | `True` |
| cheap | candidate_token_direct_lr3e4_clip1 | randomized_labels | 0.1042 | `True` |
| cheap | candidate_token_direct_lr3e4_clip1 | hidden_states_shuffled_across_examples | 0.1680 | `True` |
| cheap | stacked_candidate_token_direct_2blocks_lr3e4_clip1 | candidate_only | 0.0781 | `True` |
| cheap | stacked_candidate_token_direct_2blocks_lr3e4_clip1 | candidate_metadata_only | 0.1237 | `True` |
| cheap | stacked_candidate_token_direct_2blocks_lr3e4_clip1 | view_masked_candidates_visible | 0.1237 | `True` |
| cheap | stacked_candidate_token_direct_2blocks_lr3e4_clip1 | schema_only | 0.1536 | `True` |
| cheap | stacked_candidate_token_direct_2blocks_lr3e4_clip1 | null_evidence_values | 0.1549 | `True` |
| cheap | stacked_candidate_token_direct_2blocks_lr3e4_clip1 | evidence_only_no_candidates | 0.1250 | `True` |
| cheap | stacked_candidate_token_direct_2blocks_lr3e4_clip1 | value_shuffle_within_schema | 0.1393 | `True` |
| cheap | stacked_candidate_token_direct_2blocks_lr3e4_clip1 | cross_example_view_bundle_shuffle | 0.1484 | `True` |
| cheap | stacked_candidate_token_direct_2blocks_lr3e4_clip1 | candidate_evidence_mismatch | 0.1068 | `True` |
| cheap | stacked_candidate_token_direct_2blocks_lr3e4_clip1 | schema_preserved_role_value_shuffle | 0.1341 | `True` |
| cheap | stacked_candidate_token_direct_2blocks_lr3e4_clip1 | randomized_labels | 0.1120 | `True` |
| cheap | stacked_candidate_token_direct_2blocks_lr3e4_clip1 | hidden_states_shuffled_across_examples | 0.1354 | `True` |
| cheap | stacked_candidate_token_direct_3blocks_lr3e4_clip1 | candidate_only | 0.1510 | `True` |
| cheap | stacked_candidate_token_direct_3blocks_lr3e4_clip1 | candidate_metadata_only | 0.1237 | `True` |
| cheap | stacked_candidate_token_direct_3blocks_lr3e4_clip1 | view_masked_candidates_visible | 0.1367 | `True` |
| cheap | stacked_candidate_token_direct_3blocks_lr3e4_clip1 | schema_only | 0.1367 | `True` |
| cheap | stacked_candidate_token_direct_3blocks_lr3e4_clip1 | null_evidence_values | 0.1380 | `True` |
| cheap | stacked_candidate_token_direct_3blocks_lr3e4_clip1 | evidence_only_no_candidates | 0.1250 | `True` |
| cheap | stacked_candidate_token_direct_3blocks_lr3e4_clip1 | value_shuffle_within_schema | 0.1250 | `True` |
| cheap | stacked_candidate_token_direct_3blocks_lr3e4_clip1 | cross_example_view_bundle_shuffle | 0.1289 | `True` |
| cheap | stacked_candidate_token_direct_3blocks_lr3e4_clip1 | candidate_evidence_mismatch | 0.1016 | `True` |
| cheap | stacked_candidate_token_direct_3blocks_lr3e4_clip1 | schema_preserved_role_value_shuffle | 0.1224 | `True` |
| cheap | stacked_candidate_token_direct_3blocks_lr3e4_clip1 | randomized_labels | 0.1315 | `True` |
| cheap | stacked_candidate_token_direct_3blocks_lr3e4_clip1 | hidden_states_shuffled_across_examples | 0.1328 | `True` |
| medium | candidate_token_direct_lr3e4_clip1 | candidate_only | 0.1187 | `True` |
| medium | candidate_token_direct_lr3e4_clip1 | candidate_metadata_only | 0.1237 | `True` |
| medium | candidate_token_direct_lr3e4_clip1 | view_masked_candidates_visible | 0.1199 | `True` |
| medium | candidate_token_direct_lr3e4_clip1 | schema_only | 0.1184 | `True` |
| medium | candidate_token_direct_lr3e4_clip1 | null_evidence_values | 0.1176 | `True` |
| medium | candidate_token_direct_lr3e4_clip1 | evidence_only_no_candidates | 0.1250 | `True` |
| medium | candidate_token_direct_lr3e4_clip1 | value_shuffle_within_schema | 0.1195 | `True` |
| medium | candidate_token_direct_lr3e4_clip1 | cross_example_view_bundle_shuffle | 0.1754 | `True` |
| medium | candidate_token_direct_lr3e4_clip1 | candidate_evidence_mismatch | 0.1363 | `True` |
| medium | candidate_token_direct_lr3e4_clip1 | schema_preserved_role_value_shuffle | 0.1371 | `True` |
| medium | candidate_token_direct_lr3e4_clip1 | randomized_labels | 0.1082 | `True` |
| medium | candidate_token_direct_lr3e4_clip1 | hidden_states_shuffled_across_examples | 0.1738 | `True` |
| medium | stacked_candidate_token_direct_4blocks_lr3e4_clip1 | candidate_only | 0.1102 | `True` |
| medium | stacked_candidate_token_direct_4blocks_lr3e4_clip1 | candidate_metadata_only | 0.1237 | `True` |
| medium | stacked_candidate_token_direct_4blocks_lr3e4_clip1 | view_masked_candidates_visible | 0.1203 | `True` |
| medium | stacked_candidate_token_direct_4blocks_lr3e4_clip1 | schema_only | 0.1340 | `True` |
| medium | stacked_candidate_token_direct_4blocks_lr3e4_clip1 | null_evidence_values | 0.1332 | `True` |
| medium | stacked_candidate_token_direct_4blocks_lr3e4_clip1 | evidence_only_no_candidates | 0.1250 | `True` |
| medium | stacked_candidate_token_direct_4blocks_lr3e4_clip1 | value_shuffle_within_schema | 0.1266 | `True` |
| medium | stacked_candidate_token_direct_4blocks_lr3e4_clip1 | cross_example_view_bundle_shuffle | 0.1488 | `True` |
| medium | stacked_candidate_token_direct_4blocks_lr3e4_clip1 | candidate_evidence_mismatch | 0.1301 | `True` |
| medium | stacked_candidate_token_direct_4blocks_lr3e4_clip1 | schema_preserved_role_value_shuffle | 0.1410 | `True` |
| medium | stacked_candidate_token_direct_4blocks_lr3e4_clip1 | randomized_labels | 0.1430 | `True` |
| medium | stacked_candidate_token_direct_4blocks_lr3e4_clip1 | hidden_states_shuffled_across_examples | 0.1484 | `True` |

## Acceptance Gates

| phase | criterion | pass | value |
|---|---|---|---|
| cheap | trainable beats frozen on at least 8/10 final seeds | `True` | `11` |
| cheap | mean trainable-frozen delta >= +0.20 | `True` | `0.7093098958333334` |
| cheap | bootstrap CI lower bound > +0.05 | `True` | `[0.544580078125, 0.8375651041666666]` |
| cheap | Stage 4 controls near chance | `True` | `{"candidate_evidence_mismatch": 0.10319010416666667, "candidate_metadata_only": 0.12369791666666667, "candidate_only": 0.1171875, "cross_example_view_bundle_shuffle": 0.15299479166666666, "evidence_only_no_candidates": 0.125, "hidden_states_shuffled_across_examples": 0.15071614583333334, "null_evidence_values": 0.13834635416666666, "randomized_labels": 0.11686197916666667, "schema_only": 0.13899739583333334, "schema_preserved_role_value_shuffle": 0.13346354166666666, "value_shuffle_within_schema": 0.1376953125, "view_masked_candidates_visible": 0.12141927083333333}` |
| cheap | physical-order and candidate-order invariance pass | `True` | `{"candidate_order_shuffled_with_gold_remap": {"delta_from_trainable": 0.0, "mean_accuracy": 0.8597005208333334}, "physical_order_shuffled_roles_preserved": {"delta_from_trainable": 0.0, "mean_accuracy": 0.8597005208333334}}` |
| cheap | trainable shared model receives gradients and changes | `True` | `"all_completed_rows"` |
| cheap | frozen comparator remains frozen | `True` | `"all_completed_rows"` |
| medium | trainable beats frozen on at least 8/10 final seeds | `True` | `10` |
| medium | mean trainable-frozen delta >= +0.20 | `True` | `0.703515625` |
| medium | bootstrap CI lower bound > +0.05 | `True` | `[0.5610986328125, 0.816796875]` |
| medium | Stage 4 controls near chance | `True` | `{"candidate_evidence_mismatch": 0.133203125, "candidate_metadata_only": 0.12369791666666667, "candidate_only": 0.114453125, "cross_example_view_bundle_shuffle": 0.162109375, "evidence_only_no_candidates": 0.125, "hidden_states_shuffled_across_examples": 0.1611328125, "null_evidence_values": 0.125390625, "randomized_labels": 0.1255859375, "schema_only": 0.126171875, "schema_preserved_role_value_shuffle": 0.1390625, "value_shuffle_within_schema": 0.123046875, "view_masked_candidates_visible": 0.1201171875}` |
| medium | physical-order and candidate-order invariance pass | `True` | `{"candidate_order_shuffled_with_gold_remap": {"delta_from_trainable": 0.0009765625, "mean_accuracy": 0.862109375}, "physical_order_shuffled_roles_preserved": {"delta_from_trainable": 0.0, "mean_accuracy": 0.8611328125}}` |
| medium | trainable shared model receives gradients and changes | `True` | `"all_completed_rows"` |
| medium | frozen comparator remains frozen | `True` | `"all_completed_rows"` |

## Interpretation

- Do not claim success unless the final selected variant passes all Stage 4 controls, invariance, update/frozen audits, and delta gates.
- Failed variants and controls are retained in the tables above and JSON/JSONL artifacts.
