# Stage 4 Final Candidate-Token Direct Validation

## Exact Selected Config

- Selected architecture: `candidate_token_direct_lr3e4_clip1`
- Candidate-conditioned readout: candidate queries attend directly over role token states.
- LR: `0.0003`
- Gradient clipping: `1.0`
- Epochs/patience: `50` / `6`
- Frozen comparator: exact same architecture and training budget; only shared-agent/model weights are frozen.
- Seeds: `[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`
- Split sizes: train `2048`, dev `512`, test `1024`.
- CUDA required; bf16; batch size 32 with fallback [32,16,8].

## Lock Statement

- No architecture, dataset, control, or hyperparameter changes were made after medium selection.
- This final run uses the fixed Stage 3.8 schema-aware benchmark and fresh seeds `[10..19]`.
- `role_embedding_shuffle` remains diagnostic-only and non-gating.

## Correction: physical-order invariance evaluator bug

- Legacy buggy physical-order mean: `0.2701`.
- Corrected all-24-permutation mean: `0.7904`.
- Normal mean: `0.7904`.
- Canonicalized mean: `0.7904`.
- Bug: the legacy physical-order evaluator permuted `role_ids` and `clone_activations` but did not permute `token_states` or `token_mask`, which are the tensors consumed by the candidate-token-direct coordinator.
- Correction: all role-axis tensors are now permuted together: pooled/message/evidence readouts, `role_ids`, `token_states`, and `token_mask`.
- No training, architecture, dataset, control, or hyperparameter changes were made. The correction uses the exact saved final Stage 4 checkpoints for seeds `[10..19]`.

## Pre-Run Shortcut Diagnostics

| gate | pass | value | threshold |
|---|---|---:|---:|
| candidate_only | `True` | 0.1251 | 0.1800 |
| candidate_metadata_only | `True` | 0.1255 | 0.1800 |
| view_masked_candidates_visible | `True` | 0.1251 | 0.1800 |
| role_pair_only | `True` | 0.1250 | 0.1800 |
| lexical_overlap | `True` | 0.1250 | 0.1800 |
| static_frequency | `True` | 0.1119 | 0.2000 |
| schema_only_baseline | `True` | 0.1257 | 0.1800 |
| null_evidence_values_baseline | `True` | 0.1229 | 0.1800 |
| evidence_only_no_candidates_baseline | `True` | 0.1267 | 0.1800 |
| all_role_oracle | `True` | 1.0000 | 0.9000 |

## Per-Seed Accuracy

| seed | batch | max CUDA GB | trainable | frozen | delta | text | raw | majority | cand-order | max single-view | full ctx | candidate-pair | oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 32 | 0.126 | 1.0000 | 0.2002 | 0.7998 | 0.1250 | 0.2012 | 0.1250 | 0.1250 | 0.1250 | 0.1895 | 1.0000 | 1.0000 |
| 11 | 32 | 0.126 | 0.9238 | 0.1494 | 0.7744 | 0.1240 | 0.2383 | 0.1250 | 0.1250 | 0.1250 | 0.2061 | 1.0000 | 1.0000 |
| 12 | 32 | 0.126 | 0.5215 | 0.1113 | 0.4102 | 0.1250 | 0.1729 | 0.1250 | 0.1250 | 0.1250 | 0.2188 | 1.0000 | 1.0000 |
| 13 | 32 | 0.126 | 1.0000 | 0.1045 | 0.8955 | 0.1260 | 0.2217 | 0.1250 | 0.1250 | 0.1250 | 0.2061 | 1.0000 | 1.0000 |
| 14 | 32 | 0.126 | 1.0000 | 0.1455 | 0.8545 | 0.1250 | 0.2197 | 0.1250 | 0.1250 | 0.1260 | 0.2334 | 1.0000 | 1.0000 |
| 15 | 32 | 0.126 | 0.5010 | 0.1143 | 0.3867 | 0.1250 | 0.2031 | 0.1250 | 0.1250 | 0.1250 | 0.2002 | 1.0000 | 1.0000 |
| 16 | 32 | 0.126 | 0.9990 | 0.1689 | 0.8301 | 0.1289 | 0.2344 | 0.1250 | 0.1250 | 0.1250 | 0.2373 | 1.0000 | 1.0000 |
| 17 | 32 | 0.126 | 0.5186 | 0.1504 | 0.3682 | 0.1250 | 0.2305 | 0.1250 | 0.1250 | 0.1250 | 0.2031 | 1.0000 | 1.0000 |
| 18 | 32 | 0.126 | 1.0000 | 0.1797 | 0.8203 | 0.1250 | 0.1211 | 0.1250 | 0.1250 | 0.1338 | 0.2051 | 1.0000 | 1.0000 |
| 19 | 32 | 0.126 | 0.4277 | 0.1934 | 0.2344 | 0.1270 | 0.1924 | 0.1250 | 0.1250 | 0.1289 | 0.2070 | 1.0000 | 1.0000 |

## Mean/Std/Bootstrap CI

- Completed seeds: `[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`
- Mean trainable accuracy: `0.7892`
- Mean frozen accuracy: `0.1518`
- Mean trainable-frozen delta: `0.6374`
- Delta std: `0.2406`
- Bootstrap 95% CI for delta: `[0.4818, 0.7836]`

## Schema-Aware Control Table

| control | mean | std | max | pass |
|---|---:|---:|---:|---|
| candidate_only | 0.1284 | 0.0493 | 0.2490 | `True` |
| candidate_metadata_only | 0.1255 | 0.0019 | 0.1299 | `True` |
| view_masked_candidates_visible | 0.1218 | 0.0516 | 0.2188 | `True` |
| schema_only | 0.1275 | 0.0491 | 0.2080 | `True` |
| null_evidence_values | 0.1283 | 0.0451 | 0.2080 | `True` |
| evidence_only_no_candidates | 0.1250 | 0.0000 | 0.1250 | `True` |
| value_shuffle_within_schema | 0.1398 | 0.0169 | 0.1729 | `True` |
| cross_example_view_bundle_shuffle | 0.1534 | 0.0336 | 0.2383 | `True` |
| candidate_evidence_mismatch | 0.1267 | 0.0064 | 0.1396 | `True` |
| schema_preserved_role_value_shuffle | 0.1360 | 0.0202 | 0.1729 | `True` |
| randomized_labels | 0.1454 | 0.0747 | 0.3213 | `True` |
| hidden_states_shuffled_across_examples | 0.1526 | 0.0310 | 0.2080 | `True` |

## Invariance Control Table

| control | mean | delta from trainable | tolerance | pass |
|---|---:|---:|---:|---|
| physical_order_shuffled_roles_preserved | 0.7904 | 0.0013 | 0.0800 | `True` |
| candidate_order_shuffled_with_gold_remap | 0.7192 | -0.0699 | 0.0800 | `True` |

## Role-Embedding Shuffle Diagnostic

| seed | trainable | diagnostic | delta | gating? |
|---:|---:|---:|---:|---|
| 10 | 1.0000 | 0.2959 | -0.7041 | `False` |
| 11 | 0.9238 | 0.1904 | -0.7334 | `False` |
| 12 | 0.5215 | 0.2852 | -0.2363 | `False` |
| 13 | 1.0000 | 0.2695 | -0.7305 | `False` |
| 14 | 1.0000 | 0.2285 | -0.7715 | `False` |
| 15 | 0.5010 | 0.1582 | -0.3428 | `False` |
| 16 | 0.9990 | 0.2227 | -0.7764 | `False` |
| 17 | 0.5186 | 0.2520 | -0.2666 | `False` |
| 18 | 1.0000 | 0.3145 | -0.6855 | `False` |
| 19 | 0.4277 | 0.1689 | -0.2588 | `False` |

## Frozen Comparator Audit

| seed | frozen shared grad | frozen shared delta | shared identity |
|---:|---:|---:|---|
| 10 | 0.0000 | 0.0000 | `True` |
| 11 | 0.0000 | 0.0000 | `True` |
| 12 | 0.0000 | 0.0000 | `True` |
| 13 | 0.0000 | 0.0000 | `True` |
| 14 | 0.0000 | 0.0000 | `True` |
| 15 | 0.0000 | 0.0000 | `True` |
| 16 | 0.0000 | 0.0000 | `True` |
| 17 | 0.0000 | 0.0000 | `True` |
| 18 | 0.0000 | 0.0000 | `True` |
| 19 | 0.0000 | 0.0000 | `True` |

## Trainable Update Audit

| seed | shared grad | shared delta | coordinator grad | coordinator delta | token variance | token cosine |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.2129 | 6.4021 | 0.9258 | 2.6127 | 0.2695 | 0.4361 |
| 11 | 0.1457 | 7.1233 | 0.6646 | 2.0383 | 0.6519 | 0.3407 |
| 12 | 0.0549 | 6.7458 | 0.4823 | 2.5924 | 0.5680 | 0.5855 |
| 13 | 0.0820 | 8.5469 | 0.5239 | 3.0101 | 0.6349 | 0.4406 |
| 14 | 0.1418 | 6.3474 | 0.6514 | 2.6130 | 0.6722 | 0.2786 |
| 15 | 0.0958 | 5.8609 | 0.6407 | 2.4763 | 0.2145 | 0.4921 |
| 16 | 0.2536 | 7.5445 | 1.1219 | 2.7140 | 0.3345 | 0.2941 |
| 17 | 0.0533 | 6.2228 | 0.4509 | 2.3417 | 0.5900 | 0.3547 |
| 18 | 0.1432 | 6.7581 | 0.6586 | 2.4378 | 0.5741 | 0.3628 |
| 19 | 0.1446 | 6.5571 | 0.7489 | 2.3573 | 0.3888 | 0.6134 |

## Per-Family Accuracy

### Seed 10
| family | n | trainable | frozen | delta |
|---|---:|---:|---:|---:|
| test_real_import_restore_1 | 199 | 1.0000 | 0.1055 | 0.8945 |
| test_real_import_restore_2 | 75 | 1.0000 | 0.0000 | 1.0000 |
| test_real_import_restore_3 | 273 | 1.0000 | 0.3700 | 0.6300 |
| test_real_import_restore_6 | 477 | 1.0000 | 0.1740 | 0.8260 |

### Seed 11
| family | n | trainable | frozen | delta |
|---|---:|---:|---:|---:|
| test_real_import_restore_0 | 82 | 0.8293 | 0.4756 | 0.3537 |
| test_real_import_restore_1 | 262 | 0.9962 | 0.0954 | 0.9008 |
| test_real_import_restore_2 | 144 | 0.8472 | 0.1319 | 0.7153 |
| test_real_import_restore_3 | 319 | 0.9122 | 0.1129 | 0.7994 |
| test_real_import_restore_6 | 217 | 0.9401 | 0.1567 | 0.7834 |

### Seed 12
| family | n | trainable | frozen | delta |
|---|---:|---:|---:|---:|
| test_real_import_restore_2 | 108 | 0.4907 | 0.0000 | 0.4907 |
| test_real_import_restore_3 | 416 | 0.7163 | 0.2043 | 0.5120 |
| test_real_import_restore_6 | 500 | 0.3660 | 0.0580 | 0.3080 |

### Seed 13
| family | n | trainable | frozen | delta |
|---|---:|---:|---:|---:|
| test_real_import_restore_0 | 299 | 1.0000 | 0.0067 | 0.9933 |
| test_real_import_restore_2 | 122 | 1.0000 | 0.0164 | 0.9836 |
| test_real_import_restore_3 | 603 | 1.0000 | 0.1708 | 0.8292 |

### Seed 14
| family | n | trainable | frozen | delta |
|---|---:|---:|---:|---:|
| test_real_import_restore_1 | 205 | 1.0000 | 0.0000 | 1.0000 |
| test_real_import_restore_2 | 557 | 1.0000 | 0.1167 | 0.8833 |
| test_real_import_restore_6 | 262 | 1.0000 | 0.3206 | 0.6794 |

### Seed 15
| family | n | trainable | frozen | delta |
|---|---:|---:|---:|---:|
| test_real_import_restore_1 | 192 | 0.5833 | 0.0104 | 0.5729 |
| test_real_import_restore_2 | 213 | 0.5962 | 0.0657 | 0.5305 |
| test_real_import_restore_4 | 229 | 0.2271 | 0.2271 | 0.0000 |
| test_real_import_restore_6 | 390 | 0.5692 | 0.1256 | 0.4436 |

### Seed 16
| family | n | trainable | frozen | delta |
|---|---:|---:|---:|---:|
| test_real_import_restore_0 | 190 | 0.9947 | 0.4895 | 0.5053 |
| test_real_import_restore_2 | 130 | 1.0000 | 0.0000 | 1.0000 |
| test_real_import_restore_3 | 321 | 1.0000 | 0.1308 | 0.8692 |
| test_real_import_restore_6 | 383 | 1.0000 | 0.0992 | 0.9008 |

### Seed 17
| family | n | trainable | frozen | delta |
|---|---:|---:|---:|---:|
| test_real_import_restore_0 | 174 | 0.6954 | 0.2471 | 0.4483 |
| test_real_import_restore_1 | 187 | 0.1872 | 0.0214 | 0.1658 |
| test_real_import_restore_2 | 22 | 1.0000 | 1.0000 | 0.0000 |
| test_real_import_restore_3 | 577 | 0.5841 | 0.1473 | 0.4367 |
| test_real_import_restore_6 | 64 | 0.2500 | 0.0000 | 0.2500 |

### Seed 18
| family | n | trainable | frozen | delta |
|---|---:|---:|---:|---:|
| test_real_import_restore_0 | 452 | 1.0000 | 0.0000 | 1.0000 |
| test_real_import_restore_1 | 350 | 1.0000 | 0.0000 | 1.0000 |
| test_real_import_restore_3 | 38 | 1.0000 | 0.0000 | 1.0000 |
| test_real_import_restore_5 | 184 | 1.0000 | 1.0000 | 0.0000 |

### Seed 19
| family | n | trainable | frozen | delta |
|---|---:|---:|---:|---:|
| test_real_import_restore_1 | 203 | 0.4039 | 0.1626 | 0.2414 |
| test_real_import_restore_2 | 92 | 0.2500 | 0.3370 | -0.0870 |
| test_real_import_restore_3 | 575 | 0.5061 | 0.2330 | 0.2730 |
| test_real_import_restore_4 | 42 | 0.4762 | 0.0000 | 0.4762 |
| test_real_import_restore_5 | 112 | 0.1964 | 0.0000 | 0.1964 |

## Acceptance Gates

| criterion | pass | value |
|---|---|---|
| completed seeds >= 10 | `True` | `10` |
| trainable beats frozen on at least 8/10 seeds | `True` | `10` |
| mean trainable-frozen delta >= +0.20 | `True` | `0.63740234375` |
| bootstrap 95% CI lower bound for delta > +0.05 | `True` | `[0.4818310546875, 0.7836083984374999]` |
| trainable beats text-only, raw-latent, majority, candidate-order, and full-context baselines by mean accuracy | `True` | `{"candidate_order_baseline": 0.125, "majority_baseline": 0.125, "raw_latent": 0.203515625, "single_agent_full_context": 0.21064453125, "text_only": 0.1255859375, "trainable_mean": 0.78916015625}` |
| trainable beats every single-view baseline by mean accuracy | `True` | `{"single_view_text_role_0": 0.12578125, "single_view_text_role_1": 0.124609375, "single_view_text_role_2": 0.1224609375, "single_view_text_role_3": 0.124609375}` |
| schema-aware corruption controls collapse near chance | `True` | `{"candidate_evidence_mismatch": 0.12666015625, "candidate_metadata_only": 0.12548828125, "candidate_only": 0.12841796875, "cross_example_view_bundle_shuffle": 0.15341796875, "evidence_only_no_candidates": 0.125, "hidden_states_shuffled_across_examples": 0.15263671875, "null_evidence_values": 0.1283203125, "randomized_labels": 0.14541015625, "schema_only": 0.1275390625, "schema_preserved_role_value_shuffle": 0.13603515625, "value_shuffle_within_schema": 0.13984375, "view_masked_candidates_visible": 0.12177734375}` |
| invariance controls preserve accuracy within tolerance | `True` | `{"candidate_order_shuffled_with_gold_remap": {"mean_accuracy": 0.71923828125, "mean_delta_from_trainable": -0.06992187500000002}, "physical_order_shuffled_roles_preserved": {"mean_accuracy": 0.7904296875, "mean_delta_from_trainable": 0.0012695312499999556}}` |
| pre-run shortcut diagnostics pass | `True` | `{"all_role_oracle": true, "candidate_metadata_only": true, "candidate_only": true, "evidence_only_no_candidates_baseline": true, "lexical_overlap": true, "null_evidence_values_baseline": true, "positive_control_learnability": true, "role_pair_only": true, "schema_only_baseline": true, "static_frequency": true, "view_masked_candidates_visible": true}` |
| leakage audits pass | `True` | `"split_candidate_patch_hash_and_output"` |
| trainable shared model receives gradients and changes | `True` | `"all_completed_seeds"` |
| frozen comparator remains frozen | `True` | `"all_completed_seeds"` |
| checkpoints saved for every seed and required method | `True` | `["trainable", "frozen", "randomized_labels", "text_only", "raw_latent", "majority_baseline", "candidate_order_baseline", "single_agent_full_context", "single_view_text_role_0", "single_view_text_role_1", "single_view_text_role_2", "single_view_text_role_3", "candidate_pair_compatibility_mlp", "explicit_evidence_oracle"]` |

## Appendix: Legacy Failed Physical-Order Result

- These old failed-control values are retained for transparency. They are not used as the corrected gate result.
- Legacy final single-permutation mean: `0.2386`.
- Legacy all-24 buggy reproduction mean: `0.2701`.

| seed | old final physical | legacy all-24 bug mean | corrected all-24 mean | normal | canonicalized |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.2959 | 0.3489 | 1.0000 | 1.0000 | 1.0000 |
| 11 | 0.1904 | 0.2221 | 0.9238 | 0.9238 | 0.9238 |
| 12 | 0.2852 | 0.3018 | 0.5215 | 0.5215 | 0.5215 |
| 13 | 0.2695 | 0.3128 | 1.0000 | 1.0000 | 1.0000 |
| 14 | 0.2285 | 0.2785 | 1.0000 | 1.0000 | 1.0000 |
| 15 | 0.1582 | 0.1853 | 0.5010 | 0.5010 | 0.5010 |
| 16 | 0.2227 | 0.2758 | 0.9990 | 0.9990 | 0.9990 |
| 17 | 0.2520 | 0.2659 | 0.5186 | 0.5186 | 0.5186 |
| 18 | 0.3145 | 0.3293 | 1.0000 | 1.0000 | 1.0000 |
| 19 | 0.1689 | 0.1809 | 0.4404 | 0.4404 | 0.4404 |

## Failed Seeds/OOMs

- Failed/interrupted: `[]`
- OOM retries: `[]`

## Conservative Interpretation

- Final Stage 4 gates passed: `True`
- Conservative claim: On the clean Stage 3.8 schema-aware real-code import-restoration benchmark, the candidate-token-direct shared-weight latent coordination architecture substantially outperforms an exact frozen same-architecture comparator and text/raw/single-view baselines across fresh seeds, while passing shortcut, corruption, invariance, leakage, and parameter-update audits.
- This does not claim open-ended code repair, SWE-bench success, general coding-agent success, or human-level coding.
