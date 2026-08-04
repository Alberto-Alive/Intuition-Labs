# Stage 5.1 Multi-Avenue Portfolio Analysis

## Scope

- Checkpoint-only analysis using saved Stage 5 final checkpoints.
- No retraining, architecture change, dataset change, or control change was performed.
- Framing: avenue portfolio benefit, not mandatory joint avenue fusion.

## Portfolio Metrics

| mechanism | mean acc | min seed | weak <0.60 | frozen mean | delta |
|---|---:|---:|---:|---:|---:|
| full_4_avenue | 0.9144 | 0.3047 | 1 | 0.2655 | 0.6488 |
| single_avenue_0 | 0.7604 | 0.3828 | 3 | 0.2278 | 0.5325 |
| single_avenue_1 | 0.3680 | 0.0098 | 8 | 0.1418 | 0.2262 |
| single_avenue_2 | 0.7285 | 0.4258 | 4 | 0.2242 | 0.5043 |
| single_avenue_3 | 0.5542 | 0.1611 | 5 | 0.1695 | 0.3847 |
| random_single_avenue_per_example | 0.6027 | 0.3154 | 5 | 0.1925 | 0.4103 |
| best_fixed_single_avenue | 0.8489 | 0.4258 | 2 | 0.2549 | 0.5940 |
| oracle_best_avenue_per_example | 0.8710 | 0.4658 | 2 | 0.3826 | 0.4884 |
| Stage 4 4x1 baseline | 0.7892 | 0.4277 | 4 | 0.1518 | 0.6374 |

## Controls

| control | mean | pass <= 0.18 |
|---|---:|---|
| candidate_evidence_mismatch | 0.1287 | `True` |
| candidate_metadata_only | 0.1240 | `True` |
| candidate_only | 0.1461 | `True` |
| cross_example_view_bundle_shuffle | 0.1471 | `True` |
| evidence_only_no_candidates | 0.1250 | `True` |
| hidden_states_shuffled_across_examples | 0.1625 | `True` |
| null_evidence_values | 0.1223 | `True` |
| randomized_labels | 0.1170 | `True` |
| schema_only | 0.1335 | `True` |
| schema_preserved_role_value_shuffle | 0.1238 | `True` |
| value_shuffle_within_schema | 0.1312 | `True` |
| view_masked_candidates_visible | 0.1247 | `True` |

## Avenue Dominance

- Seed-level best fixed avenues: `['avenue_0', 'avenue_1', 'avenue_2', 'avenue_3']`
- Dominant avenue counts across families/seeds: `{'avenue_1': 5, 'avenue_3': 13, 'avenue_0': 16, 'avenue_2': 19}`
- Same-compute duplicated single-avenue baseline available: `False`
- Note: No trained duplicated-single-avenue Stage 5 checkpoint was available; no new model was trained for Stage 5.1.

## Attention Usage

- Mean all-candidate attention by avenue: `{'0': 0.32484188894558497, '1': 0.2006946527626043, '2': 0.2668797375514832, '3': 0.20758372074032755}`
- Mean gold-candidate attention by avenue: `{'0': 0.32830291609386725, '1': 0.2096873099631566, '2': 0.26184347125701957, '3': 0.2001663026859565}`

## Per-Family Accuracy

| family | full | best single | best avenue | full - best | random single |
|---|---:|---:|---|---:|---:|
| test_real_import_restore_0 | 1.0000 | 0.8926 | avenue_0 | 0.1074 | 0.8019 |
| test_real_import_restore_1 | 0.7158 | 0.7327 | avenue_0 | -0.0169 | 0.4950 |
| test_real_import_restore_3 | 1.0000 | 0.8173 | avenue_2 | 0.1827 | 0.6590 |
| test_real_import_restore_4 | 1.0000 | 0.8559 | avenue_0 | 0.1441 | 0.6882 |
| test_real_import_restore_2 | 0.9010 | 0.6916 | avenue_2 | 0.2094 | 0.5820 |
| test_real_import_restore_5 | 0.7896 | 0.6425 | avenue_2 | 0.1471 | 0.4635 |
| test_real_import_restore_6 | 0.9188 | 0.7574 | avenue_0 | 0.1614 | 0.5628 |

## Per-Family Dominance

| family | dominant avenue counts |
|---|---|
| test_real_import_restore_0 | `{'avenue_1': 1, 'avenue_3': 1}` |
| test_real_import_restore_1 | `{'avenue_1': 1, 'avenue_2': 1, 'avenue_0': 2}` |
| test_real_import_restore_3 | `{'avenue_1': 1, 'avenue_3': 4, 'avenue_2': 5, 'avenue_0': 4}` |
| test_real_import_restore_4 | `{'avenue_1': 1, 'avenue_2': 1, 'avenue_3': 1, 'avenue_0': 1}` |
| test_real_import_restore_2 | `{'avenue_0': 3, 'avenue_2': 5, 'avenue_1': 1, 'avenue_3': 4}` |
| test_real_import_restore_5 | `{'avenue_0': 2, 'avenue_2': 3, 'avenue_3': 2}` |
| test_real_import_restore_6 | `{'avenue_2': 4, 'avenue_0': 4, 'avenue_3': 1}` |

## Interpretation Gates

- Full beats Stage 4 mean accuracy: `True`.
- Full has fewer weak seeds than Stage 4: `True`.
- Full beats best fixed single avenue on average: `True`.
- Different avenues dominate families/seeds: `True`.
- All Stage 4 controls pass: `True`.
- Supports portfolio benefit: `True`.
- Full substantially beats every single avenue by mean: `True`.
- Best fixed single avenue does not recover most of mean full result: `True`.
- Seeds where one single avenue is within 0.02 of full: `6`.
- Supports joint multi-avenue fusion: `False`.

## Conservative Conclusion

- Stage 5.1 supports a portfolio-benefit interpretation under the requested criteria.
- Do not claim joint multi-avenue fusion: single-avenue ablations recover much of the result.
