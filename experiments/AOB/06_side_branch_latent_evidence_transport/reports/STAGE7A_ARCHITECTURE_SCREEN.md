# Stage 7A Architecture Screen

- Benchmark: `real_import_restore_candidate_balanced_34b` with `balanced_categories_v3` candidates.
- Dataset, labels, candidates, schema, controls, and split protocol are inherited from Stage 4/5.
- Locked Stage 5 trainable mean: `0.9144`.
- Test set accessed: `False`.

## Variant Summary

| variant | family | seeds | steps | workspace | params | trainable | frozen | delta frozen | delta Stage5 | controls | invariance | weak seeds | failure |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| e_compat_ce_t2_w0p10 | E_evidence_clustering | 3 | 2 | 0 | 518466 | 1.0000 | 0.1693 | 0.8307 | 0.0856 | `True` | `True` | 0 | none |
| f_stage5_widened_param_control | F_minimal_control | 3 | 0 | 0 | 649282 | 1.0000 | 0.1615 | 0.8385 | 0.0856 | `True` | `True` | 0 | none |
| f_stage5_workspace_only_m8 | F_minimal_control | 3 | 1 | 8 | 677954 | 1.0000 | 0.1706 | 0.8294 | 0.0856 | `True` | `True` | 0 | none |
| f_stage5_candidate_residual_gating | F_minimal_control | 3 | 1 | 0 | 518466 | 0.9870 | 0.1680 | 0.8190 | 0.0726 | `True` | `True` | 0 | none |
| a_candidate_only_t1_shared_gated_pre | A_candidate_only_refinement | 3 | 1 | 0 | 518466 | 0.9792 | 0.1680 | 0.8112 | 0.0648 | `True` | `True` | 0 | none |
| b_bidirectional_t2_stopgrad_diag | B_bidirectional_evidence_transport | 3 | 2 | 0 | 518466 | 0.9701 | 0.1589 | 0.8112 | 0.0557 | `True` | `True` | 1 | none |
| d_energy_compat_t2_w0p10 | D_energy_margin_refinement | 3 | 2 | 0 | 518466 | 0.9505 | 0.1589 | 0.7917 | 0.0361 | `True` | `True` | 1 | none |
| d_energy_margin_t2_w0p1 | D_energy_margin_refinement | 3 | 2 | 0 | 518466 | 0.9505 | 0.1589 | 0.7917 | 0.0361 | `True` | `True` | 1 | none |
| a_candidate_only_t4_shared_gated_pre | A_candidate_only_refinement | 3 | 4 | 0 | 518466 | 1.0000 | 0.1680 | 0.8320 | 0.0856 | `False` | `True` | 0 | control_failure |
| b_bidirectional_t4_shared | B_bidirectional_evidence_transport | 3 | 4 | 0 | 518466 | 0.9792 | 0.1641 | 0.8151 | 0.0648 | `False` | `True` | 0 | control_failure |
| d_energy_margin_t2_w0p05 | D_energy_margin_refinement | 3 | 2 | 0 | 518466 | 0.9505 | 0.1536 | 0.7969 | 0.0361 | `False` | `True` | 1 | control_failure |
| c_workspace_candidate_write_m8_t2 | C_latent_workspace | 3 | 2 | 8 | 677954 | 0.7383 | 0.1484 | 0.5898 | -0.1761 | `False` | `True` | 1 | did_not_beat_locked_stage5_reference |
| c_workspace_learned_m8_t2 | C_latent_workspace | 3 | 2 | 8 | 677954 | 0.7383 | 0.1523 | 0.5859 | -0.1761 | `False` | `True` | 1 | did_not_beat_locked_stage5_reference |
| a_candidate_only_t6_shared_gated_pre | A_candidate_only_refinement | 3 | 6 | 0 | 518466 | 0.7227 | 0.1576 | 0.5651 | -0.1917 | `False` | `True` | 1 | did_not_beat_locked_stage5_reference |
| a_candidate_only_t2_shared_gated_pre | A_candidate_only_refinement | 3 | 2 | 0 | 518466 | 0.7044 | 0.1589 | 0.5456 | -0.2100 | `True` | `True` | 1 | did_not_beat_locked_stage5_reference |
| e_no_aux_compat_t2 | E_evidence_clustering | 3 | 2 | 0 | 518466 | 0.7044 | 0.1589 | 0.5456 | -0.2100 | `True` | `True` | 1 | did_not_beat_locked_stage5_reference |
| e_triplet_t2_w0p10 | E_evidence_clustering | 3 | 2 | 0 | 518466 | 0.7044 | 0.1615 | 0.5430 | -0.2100 | `False` | `True` | 1 | did_not_beat_locked_stage5_reference |
| b_bidirectional_t2_shared | B_bidirectional_evidence_transport | 3 | 2 | 0 | 518466 | 0.7005 | 0.1589 | 0.5417 | -0.2139 | `True` | `True` | 1 | did_not_beat_locked_stage5_reference |
| f_stage5_energy_only_t1_w0p10 | F_minimal_control | 3 | 1 | 0 | 485442 | 0.6263 | 0.1432 | 0.4831 | -0.2881 | `False` | `True` | 2 | did_not_beat_locked_stage5_reference |
| d_energy_margin_t2_w0p2 | D_energy_margin_refinement | 3 | 2 | 0 | 518466 | 0.5794 | 0.1693 | 0.4102 | -0.3350 | `True` | `True` | 2 | did_not_beat_locked_stage5_reference |
| b_bidirectional_t4_unshared | B_bidirectional_evidence_transport | 3 | 4 | 0 | 819906 | 0.5664 | 0.1536 | 0.4128 | -0.3480 | `False` | `True` | 2 | did_not_beat_locked_stage5_reference |
| a_candidate_only_t2_shared_plain_post | A_candidate_only_refinement | 3 | 2 | 0 | 485442 | 0.5404 | 0.1471 | 0.3932 | -0.3740 | `False` | `True` | 2 | did_not_beat_locked_stage5_reference |
| c_workspace_evidencepooled_m8_t2 | C_latent_workspace | 3 | 2 | 8 | 677954 | 0.4557 | 0.1549 | 0.3008 | -0.4587 | `False` | `True` | 2 | did_not_beat_locked_stage5_reference |
| f_stage5_plus_one_refinement | F_minimal_control | 3 | 1 | 0 | 485442 | 0.4362 | 0.1510 | 0.2852 | -0.4782 | `True` | `True` | 2 | did_not_beat_locked_stage5_reference |
| a_candidate_only_t4_unshared_gated_pre | A_candidate_only_refinement | 3 | 4 | 0 | 819906 | 0.4076 | 0.1523 | 0.2552 | -0.5068 | `False` | `True` | 3 | did_not_beat_locked_stage5_reference |
| c_workspace_learned_m16_t2 | C_latent_workspace | 3 | 2 | 16 | 678466 | 0.2943 | 0.1667 | 0.1276 | -0.6201 | `True` | `True` | 3 | did_not_beat_locked_stage5_reference |
| c_workspace_learned_m4_t2 | C_latent_workspace | 3 | 2 | 4 | 677698 | 0.2891 | 0.1732 | 0.1159 | -0.6253 | `True` | `True` | 3 | did_not_beat_locked_stage5_reference |

- Selected variants: `['f_stage5_widened_param_control', 'e_compat_ce_t2_w0p10', 'f_stage5_workspace_only_m8']`.
