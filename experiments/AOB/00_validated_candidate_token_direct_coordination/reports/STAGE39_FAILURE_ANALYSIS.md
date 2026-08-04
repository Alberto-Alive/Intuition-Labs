# Stage 3.9 Failure Analysis

## Scope

- Source: Stage 3.8 full hard-validation checkpoints and JSON results.
- Training: none.
- Dataset and controls: unchanged Stage 3.8 schema-aware benchmark.
- Relevant seeds analyzed: `[0, 2, 4, 5, 7, 8, 9]`.

## Summary

- Stage 3.8 all-seed mean trainable: `0.7406`
- Stage 3.8 all-seed mean frozen: `0.5886`
- Stage 3.8 all-seed mean delta: `0.1521`
- Weak-seed mean delta: `-0.0088`
- Primary interpretation: `optimizer_instability_with_seed_sensitive_underfitting`
- Classification counts: `{"coordinator_bottleneck_or_candidate_conditioning_limit": 5, "frozen_already_solves_specific_families": 2, "insufficient_candidate_conditioned_readout": 5, "optimizer_instability": 5, "topk_readout_selects_wrong_tokens": 6, "trainable_underfits_specific_families": 5}`

## Per-Seed Diagnosis

| seed | trainable | frozen | delta | frozen families | low train families | active var train/frozen | top-k jaccard | interpretation flags |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 1.0000 | 1.0000 | 0.0000 | 0 | 0 | 0.6317/0.1267 | 0.0081 | topk_readout_selects_wrong_tokens |
| 2 | 0.5410 | 0.5674 | -0.0264 | 2 | 2 | 0.2196/0.0403 | 0.0530 | frozen_already_solves_specific_families, trainable_underfits_specific_families, optimizer_instability, topk_readout_selects_wrong_tokens, coordinator_bottleneck_or_candidate_conditioning_limit, insufficient_candidate_conditioned_readout |
| 4 | 0.4463 | 0.5088 | -0.0625 | 2 | 2 | 0.3844/0.0623 | 0.0575 | frozen_already_solves_specific_families, trainable_underfits_specific_families, optimizer_instability, topk_readout_selects_wrong_tokens, coordinator_bottleneck_or_candidate_conditioning_limit, insufficient_candidate_conditioned_readout |
| 5 | 1.0000 | 0.9980 | 0.0020 | 0 | 0 | 0.6189/0.0762 | 0.0037 | topk_readout_selects_wrong_tokens |
| 7 | 0.4873 | 0.4854 | 0.0020 | 1 | 2 | 0.3592/0.0717 | 0.0463 | trainable_underfits_specific_families, optimizer_instability, topk_readout_selects_wrong_tokens, coordinator_bottleneck_or_candidate_conditioning_limit, insufficient_candidate_conditioned_readout |
| 8 | 0.5557 | 0.5127 | 0.0430 | 0 | 2 | 0.4456/0.0771 | 0.0341 | trainable_underfits_specific_families, optimizer_instability, topk_readout_selects_wrong_tokens, coordinator_bottleneck_or_candidate_conditioning_limit, insufficient_candidate_conditioned_readout |
| 9 | 0.3760 | 0.3955 | -0.0195 | 2 | 3 | 0.4276/0.0988 | 0.0209 | trainable_underfits_specific_families, optimizer_instability, coordinator_bottleneck_or_candidate_conditioning_limit, insufficient_candidate_conditioned_readout |

## Per-Family Accuracy

### Seed 0

| family | n | trainable | frozen | delta | train-only | frozen-only |
|---|---:|---:|---:|---:|---:|---:|
| test_real_import_restore_0 | 46 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| test_real_import_restore_1 | 335 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| test_real_import_restore_2 | 96 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| test_real_import_restore_3 | 38 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| test_real_import_restore_6 | 509 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |

### Seed 2

| family | n | trainable | frozen | delta | train-only | frozen-only |
|---|---:|---:|---:|---:|---:|---:|
| test_real_import_restore_0 | 352 | 0.4574 | 0.4915 | -0.0341 | 0.1534 | 0.1875 |
| test_real_import_restore_6 | 672 | 0.5848 | 0.6071 | -0.0223 | 0.1369 | 0.1592 |

### Seed 4

| family | n | trainable | frozen | delta | train-only | frozen-only |
|---|---:|---:|---:|---:|---:|---:|
| test_real_import_restore_0 | 110 | 0.6727 | 0.3091 | 0.3636 | 0.4182 | 0.0545 |
| test_real_import_restore_2 | 21 | 1.0000 | 0.7619 | 0.2381 | 0.2381 | 0.0000 |
| test_real_import_restore_3 | 497 | 0.4889 | 0.5855 | -0.0966 | 0.1127 | 0.2093 |
| test_real_import_restore_6 | 396 | 0.3030 | 0.4571 | -0.1540 | 0.0758 | 0.2298 |

### Seed 5

| family | n | trainable | frozen | delta | train-only | frozen-only |
|---|---:|---:|---:|---:|---:|---:|
| test_real_import_restore_0 | 393 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| test_real_import_restore_3 | 496 | 1.0000 | 0.9980 | 0.0020 | 0.0020 | 0.0000 |
| test_real_import_restore_4 | 135 | 1.0000 | 0.9926 | 0.0074 | 0.0074 | 0.0000 |

### Seed 7

| family | n | trainable | frozen | delta | train-only | frozen-only |
|---|---:|---:|---:|---:|---:|---:|
| test_real_import_restore_1 | 56 | 0.6786 | 0.5357 | 0.1429 | 0.2679 | 0.1250 |
| test_real_import_restore_2 | 225 | 0.6267 | 0.5333 | 0.0933 | 0.2089 | 0.1156 |
| test_real_import_restore_3 | 489 | 0.4601 | 0.4724 | -0.0123 | 0.1902 | 0.2025 |
| test_real_import_restore_6 | 254 | 0.3740 | 0.4528 | -0.0787 | 0.1260 | 0.2047 |

### Seed 8

| family | n | trainable | frozen | delta | train-only | frozen-only |
|---|---:|---:|---:|---:|---:|---:|
| test_real_import_restore_2 | 681 | 0.5081 | 0.5154 | -0.0073 | 0.0499 | 0.0573 |
| test_real_import_restore_3 | 159 | 0.6855 | 0.4403 | 0.2453 | 0.2453 | 0.0000 |
| test_real_import_restore_4 | 184 | 0.5435 | 0.4837 | 0.0598 | 0.0598 | 0.0000 |

### Seed 9

| family | n | trainable | frozen | delta | train-only | frozen-only |
|---|---:|---:|---:|---:|---:|---:|
| test_real_import_restore_0 | 331 | 0.2961 | 0.3474 | -0.0514 | 0.1631 | 0.2145 |
| test_real_import_restore_2 | 59 | 1.0000 | 0.6271 | 0.3729 | 0.3729 | 0.0000 |
| test_real_import_restore_3 | 127 | 0.4567 | 0.8268 | -0.3701 | 0.0000 | 0.3701 |
| test_real_import_restore_6 | 507 | 0.3353 | 0.2919 | 0.0434 | 0.1578 | 0.1144 |

## Top-K Token Selection

| seed | train field rate | frozen field rate | train evidence rate | frozen evidence rate | top train tokens | top frozen tokens |
|---:|---:|---:|---:|---:|---|---|
| 0 | 0.0090 | 0.0286 | 0.0857 | 0.2344 | :, option, from, ], patch_shape, symbol_placeholder, value, area | patch, program, [, patch_shape, odd, :, line, core |
| 2 | 0.0610 | 0.0935 | 0.1328 | 0.1421 | :, name, a, sanitized_uid, odd, provider_name_shape, even, provider_placeholder | :, provider, patch_shape, candidates, import_slot_shape, patch, a, restore_local_import |
| 4 | 0.1289 | 0.0801 | 0.0771 | 0.1394 | provider_placeholder, :, symbol_surface, from, candidates, or, core, patch | option, operation, provider_placeholder, 0, provider_area, odd, import, c |
| 5 | 0.0222 | 0.1133 | 0.2363 | 0.2678 | :, operation, value, provider_placeholder, even, coordination, class, program | [, odd, patch, -, symbol_surface, experiment, withheld, candidates |
| 7 | 0.0127 | 0.1003 | 0.1057 | 0.1221 | :, exact_symbol_module_path, option, c, symbol, [, patch, b | :, withheld, provider_area, or, value, symbol_placeholder, artifact, class |
| 8 | 0.1821 | 0.0647 | 0.1072 | 0.0415 | withheld, artifact, exact_symbol_module_path, import_slot_shape, :, or, symbol_surface, experiment | [, ], withheld, patch_shape, -, provider_name_shape, a, : |
| 9 | 0.0693 | 0.1982 | 0.3760 | 0.2346 | symbol, value, :, candidates, option, restore_local_import, provider_name_shape, withheld | :, provider, provider_placeholder, symbol_surface, symbol_placeholder, option, provider_name_shape, provider_area |

## Interpretation

- The failing/weak seeds are not explained by dataset shortcuts: Stage 3.8 controls and positive controls passed.
- The dominant failure mode is a robustness problem in the candidate-agnostic top-k role-message path: frozen often extracts enough schema/category signal to match or beat trainable, while trainable is seed-sensitive and underfits several families.
- The next search should prioritize candidate-conditioned readout/scoring and readout/training stability, while preserving the Stage 3.8 schema-aware controls.
