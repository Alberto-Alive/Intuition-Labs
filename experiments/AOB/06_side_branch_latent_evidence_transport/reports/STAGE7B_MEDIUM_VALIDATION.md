# Stage 7B Medium Validation

- Benchmark: `real_import_restore_candidate_balanced_34b` with `balanced_categories_v3` candidates.
- Dataset, labels, candidates, schema, controls, and split protocol are inherited from Stage 4/5.
- Locked Stage 5 trainable mean: `0.9144`.
- Test set accessed: `False`.

## Variant Summary

| variant | family | seeds | steps | workspace | params | trainable | frozen | delta frozen | delta Stage5 | delta widened | hard-family | controls | invariance | weak seeds | failure |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| b7_e_compat_ce_t1_w0p10 | B7_compatibility_ablation | 4 | 1 | 0 | 518466 | 1.0000 | 0.1592 | 0.8408 | 0.0856 | 0.0449 | 1.0000 | `True` | `True` | 0 | none |
| b7_e_compat_ce_t4_w0p10 | B7_compatibility_ablation | 4 | 4 | 0 | 518466 | 1.0000 | 0.1562 | 0.8438 | 0.0856 | 0.0449 | 1.0000 | `True` | `True` | 0 | none |
| e_compat_ce_t2_w0p10 | E_evidence_clustering | 5 | 2 | 0 | 518466 | 1.0000 | 0.1504 | 0.8496 | 0.0856 | 0.0449 | 1.0000 | `True` | `True` | 0 | none |
| b7_e_compat_ce_t2_w0p05 | B7_compatibility_ablation | 4 | 2 | 0 | 518466 | 0.9985 | 0.2578 | 0.7407 | 0.0841 | 0.0435 | 0.9953 | `True` | `True` | 0 | none |
| b7_e_compat_ce_t2_no_aux | B7_compatibility_ablation | 4 | 2 | 0 | 518466 | 0.9673 | 0.2988 | 0.6685 | 0.0529 | 0.0122 | 0.8943 | `True` | `True` | 1 | none |
| b7_workspace_disabled_param_control | B7_workspace_ablation | 4 | 1 | 0 | 649282 | 0.9634 | 0.1787 | 0.7847 | 0.0490 | 0.0083 | 0.8817 | `True` | `True` | 1 | none |
| b7_workspace_only_m16 | B7_workspace_ablation | 4 | 1 | 16 | 678466 | 0.9634 | 0.1387 | 0.8247 | 0.0490 | 0.0083 | 0.8817 | `True` | `True` | 1 | none |
| f_stage5_widened_param_control | F_minimal_control | 5 | 0 | 0 | 649282 | 0.9551 | 0.2477 | 0.7074 | 0.0407 | 0.0000 | 0.8925 | `True` | `True` | 1 | none |
| b7_e_compat_ce_t2_w0p20 | B7_compatibility_ablation | 4 | 2 | 0 | 518466 | 1.0000 | 0.2700 | 0.7300 | 0.0856 | 0.0449 | 1.0000 | `True` | `False` | 0 | invariance_failure |
| b7_workspace_only_m8_evidencepooled | B7_workspace_ablation | 4 | 1 | 8 | 677954 | 1.0000 | 0.2803 | 0.7197 | 0.0856 | 0.0449 | 1.0000 | `True` | `False` | 0 | invariance_failure |
| f_stage5_workspace_only_m8 | F_minimal_control | 4 | 1 | 8 | 677954 | 0.9106 | 0.2598 | 0.6509 | -0.0038 | -0.0444 | 0.8095 | `True` | `True` | 1 | did_not_beat_locked_stage5_reference |
| b7_workspace_only_m4 | B7_workspace_ablation | 4 | 1 | 4 | 677698 | 0.8281 | 0.1245 | 0.7036 | -0.0863 | -0.1270 | 0.6573 | `True` | `True` | 2 | did_not_beat_locked_stage5_reference |

- Primary Stage 7B comparator: `f_stage5_widened_param_control`.
- Selected Stage 7B architecture: `['e_compat_ce_t2_w0p10']`.
- No final claim is made from Stage 7B.
