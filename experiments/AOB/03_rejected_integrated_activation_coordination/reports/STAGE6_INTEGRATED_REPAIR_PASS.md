# Stage 6 Integrated Repair Pass

## Scope

- Bounded cheap-screen repair/salvage pass only.
- Phase 2 was not run.
- Final validation was not run.
- Stage 4 `candidate_token_direct_lr3e4_clip1` remains the mainline.
- No controls were weakened or removed.

## Decision

- Repaired variant found: `False`
- Later Phase 2 candidate: `None`
- Final decision: repaired variant found: no; freeze Stage 6 and keep clone/stacking/multi-avenue as mainline

## Variants

| variant | family | coordinator | layers | repair |
|---|---|---|---:|---|
| candidate_token_direct_lr3e4_clip1 | stage4_clone_anchor | candidate_token_cross_attention | 1 | `{"candidate_order_shuffle_weight": 0.0, "dropout": 0.0, "uniform_conditions": [], "uniform_weight": 0.0}` |
| integrated_candidate_guided2_repair_uniform_lr3e4 | repair_candidate_guided | integrated_candidate_guided | 2 | `{"candidate_order_shuffle_weight": 0.25, "dropout": 0.1, "uniform_conditions": ["candidate_only", "role_token_ablation", "cross_stream_attention_disabled", "hidden_states_shuffled_across_examples", "candidate_token_ablation"], "uniform_weight": 0.2}` |
| integrated_candidate_guided_late2_repair_lr3e4 | repair_candidate_guided_late_only | integrated_candidate_guided_late | 2 | `{"candidate_order_shuffle_weight": 0.25, "dropout": 0.1, "uniform_conditions": ["candidate_only", "role_token_ablation", "cross_stream_attention_disabled", "hidden_states_shuffled_across_examples", "candidate_token_ablation"], "uniform_weight": 0.2}` |
| integrated_early_mixing1_isolated_repair_lr3e4 | repair_early_mixing_isolated | integrated_early_mixing_isolated | 2 | `{"candidate_order_shuffle_weight": 0.25, "dropout": 0.1, "uniform_conditions": ["candidate_only", "role_token_ablation", "cross_stream_attention_disabled", "hidden_states_shuffled_across_examples", "candidate_token_ablation"], "uniform_weight": 0.2}` |

## Strict Eligibility

| variant | seeds | mean train | mean frozen | mean delta | strict every-seed pass | failing controls by seed |
|---|---:|---:|---:|---:|---|---|
| candidate_token_direct_lr3e4_clip1 | 3 | 0.8255 | 0.1641 | 0.6615 | `False` | `{"0": [], "1": ["schema_preserved_role_value_shuffle"], "2": ["candidate_evidence_mismatch", "cross_example_view_bundle_shuffle"]}` |
| integrated_candidate_guided2_repair_uniform_lr3e4 | 3 | 0.4414 | 0.1654 | 0.2760 | `False` | `{"0": ["hidden_states_shuffled", "randomized_labels", "view_masked_candidates_visible"], "1": ["randomized_labels"], "2": ["candidate_only", "cross_example_view_bundle_shuffle", "cross_stream_attention_disabled", "delta_low", "hidden_states_shuffled", "no_candidate_position_shortcut_audit", "null_evidence_values", "randomized_labels", "role_token_ablation", "schema_only", "schema_preserved_role_value_shuffle", "value_shuffle_within_schema", "view_masked_candidates_visible"]}` |
| integrated_candidate_guided_late2_repair_lr3e4 | 3 | 0.4128 | 0.1667 | 0.2461 | `False` | `{"0": ["candidate_only", "hidden_states_shuffled", "no_candidate_position_shortcut_audit", "randomized_labels", "view_masked_candidates_visible"], "1": ["delta_low"], "2": []}` |
| integrated_early_mixing1_isolated_repair_lr3e4 | 3 | 0.3984 | 0.1523 | 0.2461 | `False` | `{"0": ["hidden_states_shuffled"], "1": ["randomized_labels", "view_masked_candidates_visible"], "2": ["delta_low"]}` |

## Control Means

| variant | candidate-only | metadata-only | schema-only | hidden-shuffled | randomized | no-candidate-position | role-ablation | cross-stream-disabled | physical delta ok | candidate-order delta ok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| candidate_token_direct_lr3e4_clip1 | 0.1003 | 0.1250 | 0.0703 | 0.1602 | 0.0391 | 0.1276 | 0.1094 | 0.1094 | `True` | `True` |
| integrated_candidate_guided2_repair_uniform_lr3e4 | 0.1367 | 0.1250 | 0.1432 | 0.1849 | 0.2188 | 0.1549 | 0.1576 | 0.1576 | `True` | `True` |
| integrated_candidate_guided_late2_repair_lr3e4 | 0.1302 | 0.1250 | 0.1393 | 0.1445 | 0.1341 | 0.1484 | 0.1133 | 0.1133 | `True` | `True` |
| integrated_early_mixing1_isolated_repair_lr3e4 | 0.1211 | 0.1250 | 0.1458 | 0.1628 | 0.1406 | 0.1341 | 0.0990 | 0.1289 | `True` | `True` |

## Router Status

- `integrated_router_plus_tokens_late1_lr3e4` was not included. The previous run showed coordination-token ablation failures, and this bounded pass did not add a separate fixed router path.

## Interpretation

- A repaired variant requires every seed to pass Stage 4 controls, integrated mask/leakage audits, ablation controls, physical-order invariance, candidate-order remap invariance, low frozen comparator, and meaningful delta.
- Integrated success is not claimed by this repair pass.
