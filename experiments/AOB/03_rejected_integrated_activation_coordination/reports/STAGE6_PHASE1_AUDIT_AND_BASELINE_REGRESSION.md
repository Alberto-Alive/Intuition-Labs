# Stage 6 Phase 1 Audit And Baseline Regression

## Decision

- Strict Phase 2 eligible variants: `[]`
- Phase 2 medium validation was not run by this audit.
- Stage 4 clone remains the mainline.
- Baseline regression clean: `False`
- Direct control-path regression clean (no unexplained evaluator differences): `True`
- Direct exact threshold parity: `False`
- Completed Stage 6 artifact budget matches Stage 4: `False`

## Baseline Regression

- Direct path check: trained `candidate_token_direct_lr3e4_clip1` once per cheap-screen seed `[0,1,2]` at `512/256/256`, then evaluated that same fit through the original Stage 4 controls and the new Stage 6 controls.
- Artifact check: compared the completed Stage 4 cheap-screen artifact against the completed Stage 6 Phase 1 artifact.
- Full per-seed numeric pass/fail details are in `results/stage6_phase1_baseline_regression.json`.

### Direct Control-Path Means

| control | Stage 4 dev | Stage 6 dev | delta | Stage 4 dev pass | Stage 6 dev pass | Stage 4 test | Stage 6 test | delta | Stage 4 test pass | Stage 6 test pass |
|---|---:|---:|---:|---|---|---:|---:|---:|---|---|
| trainable | 0.7930 | 0.7930 | +0.0000 | `n/a` | `n/a` | 0.7799 | 0.7799 | +0.0000 | `n/a` | `n/a` |
| frozen | 0.1484 | 0.1484 | +0.0000 | `n/a` | `n/a` | 0.1576 | 0.1576 | +0.0000 | `n/a` | `n/a` |
| oracle | 1.0000 | 1.0000 | +0.0000 | `n/a` | `n/a` | 1.0000 | 1.0000 | +0.0000 | `n/a` | `n/a` |
| candidate_only | 0.1159 | 0.1159 | +0.0000 | `True` | `True` | 0.1237 | 0.1237 | +0.0000 | `False` | `False` |
| candidate_metadata_only | 0.1250 | 0.1250 | +0.0000 | `True` | `True` | 0.1250 | 0.1250 | +0.0000 | `True` | `True` |
| schema_only | 0.1458 | 0.1458 | +0.0000 | `True` | `True` | 0.1536 | 0.1536 | +0.0000 | `False` | `False` |
| view_masked_candidates_visible | 0.1289 | 0.1289 | +0.0000 | `True` | `True` | 0.1510 | 0.1510 | +0.0000 | `True` | `True` |
| evidence_only_no_candidates | 0.1250 | 0.1250 | +0.0000 | `True` | `True` | 0.1250 | 0.1250 | +0.0000 | `True` | `True` |
| null_evidence_values | 0.1458 | 0.1458 | +0.0000 | `True` | `True` | 0.1576 | 0.1576 | +0.0000 | `False` | `False` |
| value_shuffle_within_schema | 0.1172 | 0.1237 | +0.0065 | `True` | `True` | 0.1341 | 0.1354 | +0.0013 | `True` | `True` |
| cross_example_view_bundle_shuffle | 0.1510 | 0.1406 | -0.0104 | `True` | `True` | 0.1484 | 0.1289 | -0.0195 | `True` | `True` |
| candidate_evidence_mismatch | 0.1120 | 0.1094 | -0.0026 | `True` | `True` | 0.1133 | 0.1042 | -0.0091 | `True` | `True` |
| schema_preserved_role_value_shuffle | 0.1354 | 0.1419 | +0.0065 | `True` | `True` | 0.1302 | 0.1341 | +0.0039 | `True` | `True` |
| randomized_labels | 0.1589 | 0.1589 | +0.0000 | `False` | `False` | 0.1042 | 0.1042 | +0.0000 | `True` | `True` |
| hidden_states_shuffled | 0.1341 | 0.1602 | +0.0260 | `True` | `False` | 0.1419 | 0.1654 | +0.0234 | `True` | `True` |
| physical_order_shuffled_roles_preserved | 0.7930 | 0.7930 | +0.0000 | `True` | `True` | 0.7799 | 0.7799 | +0.0000 | `True` | `True` |
| candidate_order_shuffled_with_gold_remap | 0.7943 | 0.7930 | -0.0013 | `True` | `True` | 0.7760 | 0.7773 | +0.0013 | `True` | `True` |

### Completed Artifact Comparison

- The completed Stage 6 artifact remains invalid for Phase 2 promotion if it was produced with a reduced `12/4` budget while Stage 4 used `50/6`.

| control | Stage 4 dev mean | Stage 6 dev mean | delta | Stage 4 test mean | Stage 6 test mean | delta |
|---|---:|---:|---:|---:|---:|---:|
| trainable | 0.8763 | 0.7357 | -0.1406 | 0.8880 | 0.6979 | -0.1901 |
| frozen | 0.1276 | 0.1680 | +0.0404 | 0.0885 | 0.1523 | +0.0638 |
| oracle | 1.0000 | 1.0000 | +0.0000 | 1.0000 | 1.0000 | +0.0000 |
| candidate_only | 0.1276 | 0.1862 | +0.0586 | 0.1367 | 0.1315 | -0.0052 |
| schema_only | 0.1289 | 0.1615 | +0.0326 | 0.1250 | 0.0977 | -0.0273 |
| view_masked_candidates_visible | 0.1328 | 0.1823 | +0.0495 | 0.1198 | 0.1042 | -0.0156 |
| value_shuffle_within_schema | 0.1328 | 0.1289 | -0.0039 | 0.1432 | 0.1211 | -0.0221 |
| candidate_evidence_mismatch | 0.1328 | 0.1341 | +0.0013 | 0.1159 | 0.1198 | +0.0039 |

Regression reasons:
- direct control-path regression found explainable Stage 6-only threshold flips from evaluator seed/control call-site differences
- Stage 6 baseline mean controls failed
- Stage 6 completed artifact did not use the original Stage 4 50-epoch/6-patience baseline budget
- candidate_only mean above near-chance threshold: 0.1862
- view_masked_candidates_visible mean above near-chance threshold: 0.1823
- no_candidate_position_shortcut_audit mean above near-chance threshold: 0.1914

Direct control-path Stage 6-only failures:
- `[{"control": "hidden_states_shuffled", "reason": "same corruption family; Stage 4 records hidden_states_shuffled_across_examples and Stage 6 records hidden_states_shuffled with its own evaluation seed offset", "seed": 0, "split": "dev", "stage4_accuracy": 0.15625, "stage6_accuracy": 0.18359375}]`

Direct control-path unexplained differences:
- `[]`

## Control Failure Breakdown

| variant | mean train | mean frozen | mean delta | standard every-seed | masks every-seed | leakage | frozen low | meaningful delta | strict eligible | failing controls by seed |
|---|---:|---:|---:|---|---|---|---|---|---|---|
| candidate_token_direct_lr3e4_clip1 | 0.7357 | 0.1680 | 0.5677 | `False` | `False` | `True` | `True` | `True` | `False` | `{"0": ["candidate_only", "schema_only", "null_evidence_values", "hidden_states_shuffled", "no_candidate_position_shortcut_audit"], "1": [], "2": ["candidate_only", "view_masked_candidates_visible", "no_candidate_position_shortcut_audit"]}` |
| integrated_bridge_late1_lr3e4 | 0.1628 | 0.1628 | 0.0000 | `False` | `False` | `True` | `True` | `False` | `False` | `{"0": ["schema_only", "view_masked_candidates_visible", "null_evidence_values", "value_shuffle_within_schema", "cross_example_view_bundle_shuffle", "schema_preserved_role_value_shuffle", "hidden_states_shuffled"], "1": [], "2": ["candidate_only", "no_candidate_position_shortcut_audit", "role_token_ablation", "cross_stream_attention_disabled"]}` |
| integrated_bridge_late2_lr3e4 | 0.2539 | 0.1458 | 0.1081 | `False` | `False` | `True` | `True` | `False` | `False` | `{"0": [], "1": [], "2": ["candidate_only", "no_candidate_position_shortcut_audit", "role_token_ablation", "cross_stream_attention_disabled"]}` |
| integrated_candidate_guided2_lr3e4 | 0.5599 | 0.1211 | 0.4388 | `False` | `True` | `True` | `True` | `True` | `False` | `{"0": ["schema_preserved_role_value_shuffle", "hidden_states_shuffled", "role_token_ablation", "cross_stream_attention_disabled"], "1": [], "2": ["randomized_labels", "hidden_states_shuffled"]}` |
| integrated_early_mixing1_lr3e4 | 0.8060 | 0.1758 | 0.6302 | `False` | `False` | `True` | `True` | `True` | `False` | `{"0": ["hidden_states_shuffled"], "1": [], "2": ["candidate_only", "view_masked_candidates_visible", "cross_example_view_bundle_shuffle", "no_candidate_position_shortcut_audit", "role_token_ablation", "cross_stream_attention_disabled"]}` |
| integrated_late_mixing_only2_lr3e4 | 0.3385 | 0.1510 | 0.1875 | `False` | `True` | `True` | `True` | `False` | `False` | `{"0": ["schema_only", "view_masked_candidates_visible", "null_evidence_values", "randomized_labels"], "1": [], "2": ["randomized_labels"]}` |
| integrated_router_late1_lr3e4 | 0.4258 | 0.1589 | 0.2669 | `False` | `False` | `True` | `True` | `False` | `False` | `{"0": ["candidate_only", "no_candidate_position_shortcut_audit"], "1": [], "2": []}` |
| integrated_router_plus_tokens_late1_lr3e4 | 0.5169 | 0.1784 | 0.3385 | `False` | `False` | `True` | `True` | `True` | `False` | `{"0": ["candidate_only", "randomized_labels", "no_candidate_position_shortcut_audit", "coordination_token_ablation"], "1": [], "2": ["candidate_only", "randomized_labels", "no_candidate_position_shortcut_audit", "coordination_token_ablation", "role_token_ablation", "cross_stream_attention_disabled"]}` |
| integrated_summary_late1_lr3e4 | 0.9297 | 0.1641 | 0.7656 | `False` | `False` | `True` | `True` | `True` | `False` | `{"0": ["view_masked_candidates_visible", "hidden_states_shuffled"], "1": ["randomized_labels"], "2": ["candidate_only", "randomized_labels", "no_candidate_position_shortcut_audit"]}` |

## Per-Control Numeric Breakdown

- Values are dev-split means across cheap-screen seeds `[0,1,2]`; per-seed values are in `results/stage6_phase1_control_breakdown.json`.

### candidate_token_direct_lr3e4_clip1

| control | mean accuracy | max accuracy | pass every seed | failing seeds | rule |
|---|---:|---:|---|---|---|
| candidate_only | 0.1862 | 0.2266 | `False` | `[0, 2]` | <= 0.18 |
| candidate_metadata_only | 0.1250 | 0.1250 | `True` | `[]` | <= 0.18 |
| schema_only | 0.1615 | 0.2031 | `False` | `[0]` | <= 0.18 |
| view_masked_candidates_visible | 0.1823 | 0.2305 | `False` | `[2]` | <= 0.18 |
| evidence_only_no_candidates | 0.1250 | 0.1250 | `True` | `[]` | <= 0.18 |
| null_evidence_values | 0.1615 | 0.2031 | `False` | `[0]` | <= 0.18 |
| value_shuffle_within_schema | 0.1289 | 0.1406 | `True` | `[]` | <= 0.18 |
| cross_example_view_bundle_shuffle | 0.1471 | 0.1680 | `True` | `[]` | <= 0.18 |
| candidate_evidence_mismatch | 0.1341 | 0.1602 | `True` | `[]` | <= 0.18 |
| schema_preserved_role_value_shuffle | 0.1615 | 0.1758 | `True` | `[]` | <= 0.18 |
| randomized_labels | 0.0755 | 0.1289 | `True` | `[]` | <= 0.18 |
| hidden_states_shuffled | 0.1589 | 0.1836 | `False` | `[0]` | <= 0.18 |
| physical_order_shuffled_roles_preserved | 0.7357 | 1.0000 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| candidate_order_shuffled_with_gold_remap | 0.7396 | 1.0000 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| attention_mask_audit | n/a | n/a | `True` | `[]` | mask booleans enforce no gold/candidate-position/initial role mixing |
| role_stream_isolation_audit | n/a | n/a | `True` | `[]` | role tokens cannot initially see other role streams |
| candidate_token_isolation_audit | n/a | n/a | `True` | `[]` | candidate tokens cannot see gold and no candidate-position embedding is used |
| no_gold_token_audit | n/a | n/a | `True` | `[]` | model-facing records contain no oracle-only fields |
| no_candidate_position_shortcut_audit | 0.1914 | 0.2266 | `False` | `[0, 2]` | <= 0.18 |
| packed_sequence_position_shuffle_audit | n/a | n/a | `True` | `[]` | no packed position embedding shortcut |
| role_block_permutation_audit | 0.7357 | 1.0000 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| candidate_block_permutation_with_remapped_labels | 0.7396 | 1.0000 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| coordination_token_ablation | 0.7357 | 1.0000 | `True` | `[]` | not gated when no coordination/router token exists |
| candidate_token_ablation | 0.1250 | 0.1250 | `True` | `[]` | <= 0.18 |
| role_token_ablation | 0.0781 | 0.0977 | `True` | `[]` | <= 0.18 |
| cross_stream_attention_disabled | 0.0781 | 0.0977 | `True` | `[]` | <= 0.18 |

### integrated_bridge_late1_lr3e4

| control | mean accuracy | max accuracy | pass every seed | failing seeds | rule |
|---|---:|---:|---|---|---|
| candidate_only | 0.1354 | 0.1836 | `False` | `[2]` | <= 0.18 |
| candidate_metadata_only | 0.1250 | 0.1250 | `True` | `[]` | <= 0.18 |
| schema_only | 0.1628 | 0.2148 | `False` | `[0]` | <= 0.18 |
| view_masked_candidates_visible | 0.1628 | 0.2148 | `False` | `[0]` | <= 0.18 |
| evidence_only_no_candidates | 0.1250 | 0.1250 | `True` | `[]` | <= 0.18 |
| null_evidence_values | 0.1628 | 0.2148 | `False` | `[0]` | <= 0.18 |
| value_shuffle_within_schema | 0.1628 | 0.2148 | `False` | `[0]` | <= 0.18 |
| cross_example_view_bundle_shuffle | 0.1628 | 0.2148 | `False` | `[0]` | <= 0.18 |
| candidate_evidence_mismatch | 0.1380 | 0.1680 | `True` | `[]` | <= 0.18 |
| schema_preserved_role_value_shuffle | 0.1628 | 0.2148 | `False` | `[0]` | <= 0.18 |
| randomized_labels | 0.0833 | 0.1211 | `True` | `[]` | <= 0.18 |
| hidden_states_shuffled | 0.1628 | 0.2148 | `False` | `[0]` | <= 0.18 |
| physical_order_shuffled_roles_preserved | 0.1628 | 0.2148 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| candidate_order_shuffled_with_gold_remap | 0.1628 | 0.2148 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| attention_mask_audit | n/a | n/a | `True` | `[]` | mask booleans enforce no gold/candidate-position/initial role mixing |
| role_stream_isolation_audit | n/a | n/a | `True` | `[]` | role tokens cannot initially see other role streams |
| candidate_token_isolation_audit | n/a | n/a | `True` | `[]` | candidate tokens cannot see gold and no candidate-position embedding is used |
| no_gold_token_audit | n/a | n/a | `True` | `[]` | model-facing records contain no oracle-only fields |
| no_candidate_position_shortcut_audit | 0.1628 | 0.1836 | `False` | `[2]` | <= 0.18 |
| packed_sequence_position_shuffle_audit | 1.0000 | 1.0000 | `True` | `[]` | no packed position embedding shortcut |
| role_block_permutation_audit | 0.1628 | 0.2148 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| candidate_block_permutation_with_remapped_labels | 0.1628 | 0.2148 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| coordination_token_ablation | 0.1628 | 0.2148 | `True` | `[]` | not gated when no coordination/router token exists |
| candidate_token_ablation | 0.1250 | 0.1250 | `True` | `[]` | <= 0.18 |
| role_token_ablation | 0.1354 | 0.1836 | `False` | `[2]` | <= 0.18 |
| cross_stream_attention_disabled | 0.1354 | 0.1836 | `False` | `[2]` | <= 0.18 |

### integrated_bridge_late2_lr3e4

| control | mean accuracy | max accuracy | pass every seed | failing seeds | rule |
|---|---:|---:|---|---|---|
| candidate_only | 0.1510 | 0.2109 | `False` | `[2]` | <= 0.18 |
| candidate_metadata_only | 0.1250 | 0.1250 | `True` | `[]` | <= 0.18 |
| schema_only | 0.1354 | 0.1641 | `True` | `[]` | <= 0.18 |
| view_masked_candidates_visible | 0.1354 | 0.1641 | `True` | `[]` | <= 0.18 |
| evidence_only_no_candidates | 0.1250 | 0.1250 | `True` | `[]` | <= 0.18 |
| null_evidence_values | 0.1354 | 0.1641 | `True` | `[]` | <= 0.18 |
| value_shuffle_within_schema | 0.1289 | 0.1523 | `True` | `[]` | <= 0.18 |
| cross_example_view_bundle_shuffle | 0.1263 | 0.1523 | `True` | `[]` | <= 0.18 |
| candidate_evidence_mismatch | 0.1458 | 0.1641 | `True` | `[]` | <= 0.18 |
| schema_preserved_role_value_shuffle | 0.1380 | 0.1719 | `True` | `[]` | <= 0.18 |
| randomized_labels | 0.1016 | 0.1328 | `True` | `[]` | <= 0.18 |
| hidden_states_shuffled | 0.1393 | 0.1758 | `True` | `[]` | <= 0.18 |
| physical_order_shuffled_roles_preserved | 0.2539 | 0.5195 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| candidate_order_shuffled_with_gold_remap | 0.2513 | 0.5117 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| attention_mask_audit | n/a | n/a | `True` | `[]` | mask booleans enforce no gold/candidate-position/initial role mixing |
| role_stream_isolation_audit | n/a | n/a | `True` | `[]` | role tokens cannot initially see other role streams |
| candidate_token_isolation_audit | n/a | n/a | `True` | `[]` | candidate tokens cannot see gold and no candidate-position embedding is used |
| no_gold_token_audit | n/a | n/a | `True` | `[]` | model-facing records contain no oracle-only fields |
| no_candidate_position_shortcut_audit | 0.1549 | 0.2109 | `False` | `[2]` | <= 0.18 |
| packed_sequence_position_shuffle_audit | 1.0000 | 1.0000 | `True` | `[]` | no packed position embedding shortcut |
| role_block_permutation_audit | 0.2539 | 0.5195 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| candidate_block_permutation_with_remapped_labels | 0.2513 | 0.5117 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| coordination_token_ablation | 0.2539 | 0.5195 | `True` | `[]` | not gated when no coordination/router token exists |
| candidate_token_ablation | 0.1250 | 0.1250 | `True` | `[]` | <= 0.18 |
| role_token_ablation | 0.1667 | 0.2109 | `False` | `[2]` | <= 0.18 |
| cross_stream_attention_disabled | 0.1667 | 0.2109 | `False` | `[2]` | <= 0.18 |

### integrated_candidate_guided2_lr3e4

| control | mean accuracy | max accuracy | pass every seed | failing seeds | rule |
|---|---:|---:|---|---|---|
| candidate_only | 0.1263 | 0.1641 | `True` | `[]` | <= 0.18 |
| candidate_metadata_only | 0.1250 | 0.1250 | `True` | `[]` | <= 0.18 |
| schema_only | 0.1419 | 0.1758 | `True` | `[]` | <= 0.18 |
| view_masked_candidates_visible | 0.1120 | 0.1797 | `True` | `[]` | <= 0.18 |
| evidence_only_no_candidates | 0.1250 | 0.1250 | `True` | `[]` | <= 0.18 |
| null_evidence_values | 0.1367 | 0.1680 | `True` | `[]` | <= 0.18 |
| value_shuffle_within_schema | 0.1302 | 0.1523 | `True` | `[]` | <= 0.18 |
| cross_example_view_bundle_shuffle | 0.1445 | 0.1797 | `True` | `[]` | <= 0.18 |
| candidate_evidence_mismatch | 0.1562 | 0.1680 | `True` | `[]` | <= 0.18 |
| schema_preserved_role_value_shuffle | 0.1523 | 0.1914 | `False` | `[0]` | <= 0.18 |
| randomized_labels | 0.1354 | 0.1836 | `False` | `[2]` | <= 0.18 |
| hidden_states_shuffled | 0.1641 | 0.2070 | `False` | `[0, 2]` | <= 0.18 |
| physical_order_shuffled_roles_preserved | 0.5599 | 1.0000 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| candidate_order_shuffled_with_gold_remap | 0.5612 | 1.0000 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| attention_mask_audit | n/a | n/a | `True` | `[]` | mask booleans enforce no gold/candidate-position/initial role mixing |
| role_stream_isolation_audit | n/a | n/a | `True` | `[]` | role tokens cannot initially see other role streams |
| candidate_token_isolation_audit | n/a | n/a | `True` | `[]` | candidate tokens cannot see gold and no candidate-position embedding is used |
| no_gold_token_audit | n/a | n/a | `True` | `[]` | model-facing records contain no oracle-only fields |
| no_candidate_position_shortcut_audit | 0.1380 | 0.1641 | `True` | `[]` | <= 0.18 |
| packed_sequence_position_shuffle_audit | 1.0000 | 1.0000 | `True` | `[]` | no packed position embedding shortcut |
| role_block_permutation_audit | 0.5599 | 1.0000 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| candidate_block_permutation_with_remapped_labels | 0.5612 | 1.0000 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| coordination_token_ablation | 0.5599 | 1.0000 | `True` | `[]` | not gated when no coordination/router token exists |
| candidate_token_ablation | 0.1250 | 0.1250 | `True` | `[]` | <= 0.18 |
| role_token_ablation | 0.1432 | 0.2148 | `False` | `[0]` | <= 0.18 |
| cross_stream_attention_disabled | 0.1432 | 0.2148 | `False` | `[0]` | <= 0.18 |

### integrated_early_mixing1_lr3e4

| control | mean accuracy | max accuracy | pass every seed | failing seeds | rule |
|---|---:|---:|---|---|---|
| candidate_only | 0.1680 | 0.1875 | `False` | `[2]` | <= 0.18 |
| candidate_metadata_only | 0.1250 | 0.1250 | `True` | `[]` | <= 0.18 |
| schema_only | 0.1276 | 0.1797 | `True` | `[]` | <= 0.18 |
| view_masked_candidates_visible | 0.1393 | 0.2109 | `False` | `[2]` | <= 0.18 |
| evidence_only_no_candidates | 0.1250 | 0.1250 | `True` | `[]` | <= 0.18 |
| null_evidence_values | 0.1302 | 0.1797 | `True` | `[]` | <= 0.18 |
| value_shuffle_within_schema | 0.1211 | 0.1523 | `True` | `[]` | <= 0.18 |
| cross_example_view_bundle_shuffle | 0.1458 | 0.1953 | `False` | `[2]` | <= 0.18 |
| candidate_evidence_mismatch | 0.1302 | 0.1523 | `True` | `[]` | <= 0.18 |
| schema_preserved_role_value_shuffle | 0.1602 | 0.1758 | `True` | `[]` | <= 0.18 |
| randomized_labels | 0.0781 | 0.1133 | `True` | `[]` | <= 0.18 |
| hidden_states_shuffled | 0.1628 | 0.1875 | `False` | `[0]` | <= 0.18 |
| physical_order_shuffled_roles_preserved | 0.8060 | 1.0000 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| candidate_order_shuffled_with_gold_remap | 0.8021 | 1.0000 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| attention_mask_audit | n/a | n/a | `True` | `[]` | mask booleans enforce no gold/candidate-position/initial role mixing |
| role_stream_isolation_audit | n/a | n/a | `True` | `[]` | role tokens cannot initially see other role streams |
| candidate_token_isolation_audit | n/a | n/a | `True` | `[]` | candidate tokens cannot see gold and no candidate-position embedding is used |
| no_gold_token_audit | n/a | n/a | `True` | `[]` | model-facing records contain no oracle-only fields |
| no_candidate_position_shortcut_audit | 0.1680 | 0.1875 | `False` | `[2]` | <= 0.18 |
| packed_sequence_position_shuffle_audit | 1.0000 | 1.0000 | `True` | `[]` | no packed position embedding shortcut |
| role_block_permutation_audit | 0.8060 | 1.0000 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| candidate_block_permutation_with_remapped_labels | 0.8021 | 1.0000 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| coordination_token_ablation | 0.8060 | 1.0000 | `True` | `[]` | not gated when no coordination/router token exists |
| candidate_token_ablation | 0.1250 | 0.1250 | `True` | `[]` | <= 0.18 |
| role_token_ablation | 0.1276 | 0.1836 | `False` | `[2]` | <= 0.18 |
| cross_stream_attention_disabled | 0.1237 | 0.1836 | `False` | `[2]` | <= 0.18 |

### integrated_late_mixing_only2_lr3e4

| control | mean accuracy | max accuracy | pass every seed | failing seeds | rule |
|---|---:|---:|---|---|---|
| candidate_only | 0.0729 | 0.1211 | `True` | `[]` | <= 0.18 |
| candidate_metadata_only | 0.1250 | 0.1250 | `True` | `[]` | <= 0.18 |
| schema_only | 0.1536 | 0.2031 | `False` | `[0]` | <= 0.18 |
| view_masked_candidates_visible | 0.1549 | 0.2070 | `False` | `[0]` | <= 0.18 |
| evidence_only_no_candidates | 0.1250 | 0.1250 | `True` | `[]` | <= 0.18 |
| null_evidence_values | 0.1536 | 0.2031 | `False` | `[0]` | <= 0.18 |
| value_shuffle_within_schema | 0.1341 | 0.1758 | `True` | `[]` | <= 0.18 |
| cross_example_view_bundle_shuffle | 0.1328 | 0.1758 | `True` | `[]` | <= 0.18 |
| candidate_evidence_mismatch | 0.1432 | 0.1602 | `True` | `[]` | <= 0.18 |
| schema_preserved_role_value_shuffle | 0.1380 | 0.1758 | `True` | `[]` | <= 0.18 |
| randomized_labels | 0.2083 | 0.2422 | `False` | `[0, 2]` | <= 0.18 |
| hidden_states_shuffled | 0.1393 | 0.1758 | `True` | `[]` | <= 0.18 |
| physical_order_shuffled_roles_preserved | 0.3385 | 0.7578 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| candidate_order_shuffled_with_gold_remap | 0.3372 | 0.7539 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| attention_mask_audit | n/a | n/a | `True` | `[]` | mask booleans enforce no gold/candidate-position/initial role mixing |
| role_stream_isolation_audit | n/a | n/a | `True` | `[]` | role tokens cannot initially see other role streams |
| candidate_token_isolation_audit | n/a | n/a | `True` | `[]` | candidate tokens cannot see gold and no candidate-position embedding is used |
| no_gold_token_audit | n/a | n/a | `True` | `[]` | model-facing records contain no oracle-only fields |
| no_candidate_position_shortcut_audit | 0.1250 | 0.1250 | `True` | `[]` | <= 0.18 |
| packed_sequence_position_shuffle_audit | 1.0000 | 1.0000 | `True` | `[]` | no packed position embedding shortcut |
| role_block_permutation_audit | 0.3385 | 0.7578 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| candidate_block_permutation_with_remapped_labels | 0.3372 | 0.7539 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| coordination_token_ablation | 0.3385 | 0.7578 | `True` | `[]` | not gated when no coordination/router token exists |
| candidate_token_ablation | 0.1250 | 0.1250 | `True` | `[]` | <= 0.18 |
| role_token_ablation | 0.1133 | 0.1719 | `True` | `[]` | <= 0.18 |
| cross_stream_attention_disabled | 0.1133 | 0.1719 | `True` | `[]` | <= 0.18 |

### integrated_router_late1_lr3e4

| control | mean accuracy | max accuracy | pass every seed | failing seeds | rule |
|---|---:|---:|---|---|---|
| candidate_only | 0.1536 | 0.2188 | `False` | `[0]` | <= 0.18 |
| candidate_metadata_only | 0.1250 | 0.1250 | `True` | `[]` | <= 0.18 |
| schema_only | 0.1497 | 0.1719 | `True` | `[]` | <= 0.18 |
| view_masked_candidates_visible | 0.1497 | 0.1719 | `True` | `[]` | <= 0.18 |
| evidence_only_no_candidates | 0.1250 | 0.1250 | `True` | `[]` | <= 0.18 |
| null_evidence_values | 0.1497 | 0.1719 | `True` | `[]` | <= 0.18 |
| value_shuffle_within_schema | 0.1432 | 0.1602 | `True` | `[]` | <= 0.18 |
| cross_example_view_bundle_shuffle | 0.1471 | 0.1641 | `True` | `[]` | <= 0.18 |
| candidate_evidence_mismatch | 0.1419 | 0.1719 | `True` | `[]` | <= 0.18 |
| schema_preserved_role_value_shuffle | 0.1393 | 0.1602 | `True` | `[]` | <= 0.18 |
| randomized_labels | 0.1107 | 0.1602 | `True` | `[]` | <= 0.18 |
| hidden_states_shuffled | 0.1484 | 0.1680 | `True` | `[]` | <= 0.18 |
| physical_order_shuffled_roles_preserved | 0.4258 | 1.0000 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| candidate_order_shuffled_with_gold_remap | 0.4258 | 1.0000 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| attention_mask_audit | n/a | n/a | `True` | `[]` | mask booleans enforce no gold/candidate-position/initial role mixing |
| role_stream_isolation_audit | n/a | n/a | `True` | `[]` | role tokens cannot initially see other role streams |
| candidate_token_isolation_audit | n/a | n/a | `True` | `[]` | candidate tokens cannot see gold and no candidate-position embedding is used |
| no_gold_token_audit | n/a | n/a | `True` | `[]` | model-facing records contain no oracle-only fields |
| no_candidate_position_shortcut_audit | 0.1719 | 0.2188 | `False` | `[0]` | <= 0.18 |
| packed_sequence_position_shuffle_audit | 1.0000 | 1.0000 | `True` | `[]` | no packed position embedding shortcut |
| role_block_permutation_audit | 0.4258 | 1.0000 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| candidate_block_permutation_with_remapped_labels | 0.4258 | 1.0000 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| coordination_token_ablation | 0.1211 | 0.1719 | `True` | `[]` | <= 0.18 |
| candidate_token_ablation | 0.1250 | 0.1250 | `True` | `[]` | <= 0.18 |
| role_token_ablation | 0.1224 | 0.1719 | `True` | `[]` | <= 0.18 |
| cross_stream_attention_disabled | 0.1224 | 0.1719 | `True` | `[]` | <= 0.18 |

### integrated_router_plus_tokens_late1_lr3e4

| control | mean accuracy | max accuracy | pass every seed | failing seeds | rule |
|---|---:|---:|---|---|---|
| candidate_only | 0.1576 | 0.1875 | `False` | `[0, 2]` | <= 0.18 |
| candidate_metadata_only | 0.1250 | 0.1250 | `True` | `[]` | <= 0.18 |
| schema_only | 0.1068 | 0.1328 | `True` | `[]` | <= 0.18 |
| view_masked_candidates_visible | 0.0990 | 0.1211 | `True` | `[]` | <= 0.18 |
| evidence_only_no_candidates | 0.1250 | 0.1250 | `True` | `[]` | <= 0.18 |
| null_evidence_values | 0.1068 | 0.1328 | `True` | `[]` | <= 0.18 |
| value_shuffle_within_schema | 0.1172 | 0.1172 | `True` | `[]` | <= 0.18 |
| cross_example_view_bundle_shuffle | 0.1432 | 0.1719 | `True` | `[]` | <= 0.18 |
| candidate_evidence_mismatch | 0.1602 | 0.1758 | `True` | `[]` | <= 0.18 |
| schema_preserved_role_value_shuffle | 0.1276 | 0.1445 | `True` | `[]` | <= 0.18 |
| randomized_labels | 0.1510 | 0.1875 | `False` | `[0, 2]` | <= 0.18 |
| hidden_states_shuffled | 0.1471 | 0.1641 | `True` | `[]` | <= 0.18 |
| physical_order_shuffled_roles_preserved | 0.5169 | 1.0000 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| candidate_order_shuffled_with_gold_remap | 0.5169 | 1.0000 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| attention_mask_audit | n/a | n/a | `True` | `[]` | mask booleans enforce no gold/candidate-position/initial role mixing |
| role_stream_isolation_audit | n/a | n/a | `True` | `[]` | role tokens cannot initially see other role streams |
| candidate_token_isolation_audit | n/a | n/a | `True` | `[]` | candidate tokens cannot see gold and no candidate-position embedding is used |
| no_gold_token_audit | n/a | n/a | `True` | `[]` | model-facing records contain no oracle-only fields |
| no_candidate_position_shortcut_audit | 0.1654 | 0.1875 | `False` | `[0, 2]` | <= 0.18 |
| packed_sequence_position_shuffle_audit | 1.0000 | 1.0000 | `True` | `[]` | no packed position embedding shortcut |
| role_block_permutation_audit | 0.5169 | 1.0000 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| candidate_block_permutation_with_remapped_labels | 0.5169 | 1.0000 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| coordination_token_ablation | 0.4648 | 0.8633 | `False` | `[0, 2]` | <= 0.18 |
| candidate_token_ablation | 0.1250 | 0.1250 | `True` | `[]` | <= 0.18 |
| role_token_ablation | 0.1458 | 0.1875 | `False` | `[2]` | <= 0.18 |
| cross_stream_attention_disabled | 0.1458 | 0.1875 | `False` | `[2]` | <= 0.18 |

### integrated_summary_late1_lr3e4

| control | mean accuracy | max accuracy | pass every seed | failing seeds | rule |
|---|---:|---:|---|---|---|
| candidate_only | 0.1615 | 0.2383 | `False` | `[2]` | <= 0.18 |
| candidate_metadata_only | 0.1250 | 0.1250 | `True` | `[]` | <= 0.18 |
| schema_only | 0.1393 | 0.1719 | `True` | `[]` | <= 0.18 |
| view_masked_candidates_visible | 0.1823 | 0.2344 | `False` | `[0]` | <= 0.18 |
| evidence_only_no_candidates | 0.1250 | 0.1250 | `True` | `[]` | <= 0.18 |
| null_evidence_values | 0.1367 | 0.1680 | `True` | `[]` | <= 0.18 |
| value_shuffle_within_schema | 0.1445 | 0.1719 | `True` | `[]` | <= 0.18 |
| cross_example_view_bundle_shuffle | 0.1497 | 0.1680 | `True` | `[]` | <= 0.18 |
| candidate_evidence_mismatch | 0.1302 | 0.1523 | `True` | `[]` | <= 0.18 |
| schema_preserved_role_value_shuffle | 0.1341 | 0.1484 | `True` | `[]` | <= 0.18 |
| randomized_labels | 0.1536 | 0.2305 | `False` | `[1, 2]` | <= 0.18 |
| hidden_states_shuffled | 0.1680 | 0.1875 | `False` | `[0]` | <= 0.18 |
| physical_order_shuffled_roles_preserved | 0.9297 | 1.0000 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| candidate_order_shuffled_with_gold_remap | 0.9323 | 1.0000 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| attention_mask_audit | n/a | n/a | `True` | `[]` | mask booleans enforce no gold/candidate-position/initial role mixing |
| role_stream_isolation_audit | n/a | n/a | `True` | `[]` | role tokens cannot initially see other role streams |
| candidate_token_isolation_audit | n/a | n/a | `True` | `[]` | candidate tokens cannot see gold and no candidate-position embedding is used |
| no_gold_token_audit | n/a | n/a | `True` | `[]` | model-facing records contain no oracle-only fields |
| no_candidate_position_shortcut_audit | 0.1654 | 0.2383 | `False` | `[2]` | <= 0.18 |
| packed_sequence_position_shuffle_audit | 1.0000 | 1.0000 | `True` | `[]` | no packed position embedding shortcut |
| role_block_permutation_audit | 0.9297 | 1.0000 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| candidate_block_permutation_with_remapped_labels | 0.9323 | 1.0000 | `True` | `[]` | abs(delta_from_trainable) <= 0.08 |
| coordination_token_ablation | 0.9297 | 1.0000 | `True` | `[]` | not gated when no coordination/router token exists |
| candidate_token_ablation | 0.1250 | 0.1250 | `True` | `[]` | <= 0.18 |
| role_token_ablation | 0.1536 | 0.1758 | `True` | `[]` | <= 0.18 |
| cross_stream_attention_disabled | 0.1536 | 0.1758 | `True` | `[]` | <= 0.18 |

## Early-Mixing Reconciliation

- The original Stage 6 Phase 1 table used variant-level mean gates.
- The contradictory prose counted strict per-seed row failures.
- These are different gate groups; the Stage 6 report now labels the table column as `mean-gate pass`, and the audit uses strict every-seed eligibility for Phase 2.
- `integrated_early_mixing1_lr3e4` mean gates passed in the original table, but strict every-seed eligibility is `False`.
- Failing controls by seed: `{"0": ["hidden_states_shuffled"], "1": [], "2": ["candidate_only", "view_masked_candidates_visible", "cross_example_view_bundle_shuffle", "no_candidate_position_shortcut_audit", "role_token_ablation", "cross_stream_attention_disabled"]}`

## Integrated Summary Investigation

- `integrated_summary_late1_lr3e4` is not promoted: strict eligible `False`.
- Failing controls by seed: `{"0": ["view_masked_candidates_visible", "hidden_states_shuffled"], "1": ["randomized_labels"], "2": ["candidate_only", "randomized_labels", "no_candidate_position_shortcut_audit"]}`
- Leakage and mask audits passed, so the observed high accuracy is best classified as control-threshold/shortcut behavior rather than proven gold leakage.

## Phase 2 Rule

- Advance only variants with every-seed standard controls passing, integrated mask/leakage passing, physical and candidate-order invariance passing, low frozen comparator, meaningful trainable-frozen delta, and no contradictory pass/fail statements.
- Result under that rule: `[]`.
- Because no variant passed strict eligibility and the baseline regression is not clean, Phase 2 was not run.

## Stage 6 Mean-Gate Snapshot

| variant | original mean-gate pass | controls pass | masks pass | mean delta |
|---|---|---|---|---:|
| integrated_early_mixing1_lr3e4 | `True` | `True` | `True` | 0.6302 |
| integrated_candidate_guided2_lr3e4 | `True` | `True` | `True` | 0.4388 |
| integrated_router_plus_tokens_late1_lr3e4 | `True` | `True` | `True` | 0.3385 |
| integrated_summary_late1_lr3e4 | `False` | `False` | `True` | 0.7656 |
| candidate_token_direct_lr3e4_clip1 | `False` | `False` | `True` | 0.5677 |
| integrated_router_late1_lr3e4 | `False` | `True` | `True` | 0.2669 |
| integrated_late_mixing_only2_lr3e4 | `False` | `False` | `True` | 0.1875 |
| integrated_bridge_late2_lr3e4 | `False` | `True` | `True` | 0.1081 |
| integrated_bridge_late1_lr3e4 | `False` | `True` | `True` | 0.0000 |
