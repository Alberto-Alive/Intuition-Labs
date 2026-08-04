# Stage 7C tau-bench Final Validation

## Decision
- Decision: `C. STAGE7C_FINAL_FAILED`
- Conservative final claim allowed: `False`
- Selected Stage 7B variant: `variant_d_4x1_lr3e4_clip1`
- Airline cross-domain validation run: `False`.

## Dataset
- Candidate pool: `results\stage7b0_tau_candidate_pool_balanced.jsonl`
- Split meta: `{"final_task_count": 24, "final_task_ids_source_split": "dev", "fit_selection_split": "train_only_no_final_labels", "no_final_labels_used_for_training_or_early_stopping": true, "source_stage7b_split_file_declared_no_final_heldout_tasks_used": true, "task_overlap_train_final": 0, "train_task_count": 69, "train_task_ids_source_split": "train"}`
- Split summary: `{"final": {"candidates": 192, "generator_source_distribution_audit_only": {"llm_agent:claude-3-7-sonnet-20250219": 48, "llm_agent:gpt-4.1-2025-04-14": 46, "llm_agent:gpt-4.1-mini-2025-04-14": 49, "llm_agent:o4-mini-2025-04-16": 49}, "oracle_empty_tasks": 4, "oracle_pass_at_8": 0.8333333333333334, "success_count_distribution": {"0": 4, "1": 1, "2": 2, "3": 3, "4": 4, "5": 3, "6": 3, "7": 4}, "tasks": 24}, "task_overlap_train_final": 0, "train": {"candidates": 552, "generator_source_distribution_audit_only": {"llm_agent:claude-3-7-sonnet-20250219": 115, "llm_agent:gpt-4.1-2025-04-14": 144, "llm_agent:gpt-4.1-mini-2025-04-14": 159, "llm_agent:o4-mini-2025-04-16": 134}, "oracle_empty_tasks": 10, "oracle_pass_at_8": 0.855072463768116, "success_count_distribution": {"0": 10, "1": 4, "2": 4, "3": 9, "4": 10, "5": 10, "6": 8, "7": 14}, "tasks": 69}}`

## Final Metrics
- Trainable pass@1 mean: `0.4292`
- Frozen pass@1 mean: `0.4833`
- Best non-oracle baseline: `best_generator_on_train_baseline`
- Best non-oracle pass@1 mean: `0.5417`
- Trainable-frozen delta: `-0.0542`
- Trainable-best-baseline delta: `-0.1125`
- Oracle pass@8 mean: `0.8333`
- Trainable selection efficiency: `0.5150`

## Statistics
- Trainable vs best baseline: `{"absolute_pass_at_1_delta": -0.11249999701976776, "baseline": "best_generator_on_train_baseline", "baseline_pass_at_1": 0.5416666865348816, "n_pairs": 240, "paired_bootstrap_95_ci": {"lower": -0.17916665971279144, "mean": -0.11249999701976776, "upper": -0.0416666679084301}, "paired_sign_test": {"losses": 52, "n_untied": 77, "ties": 163, "two_sided_p": 0.0027990538772776244, "wins": 25}, "trainable_pass_at_1": 0.42916667461395264}`
- Trainable vs frozen: `{"absolute_pass_at_1_delta": -0.05416666716337204, "baseline": "frozen_same_architecture_latent_selector", "baseline_pass_at_1": 0.4833333194255829, "n_pairs": 240, "paired_bootstrap_95_ci": {"lower": -0.11666666716337204, "mean": -0.05416666716337204, "upper": 0.012604166870005333}, "paired_sign_test": {"losses": 42, "n_untied": 71, "ties": 169, "two_sided_p": 0.1539127693264563, "wins": 29}, "trainable_pass_at_1": 0.42916667461395264}`

## Controls
`{"artifact": "stage7c_tau_final_control_results", "candidate_order_invariance_pass_count": 10, "control_summary": {"candidate_evidence_mismatch": {"base_minus_control_pass_at_1_mean": 0.012500000000000011, "pass_at_1_mean": 0.4166666666666667, "pass_count": 10}, "candidate_only": {"base_minus_control_pass_at_1_mean": -0.004166666666666652, "pass_at_1_mean": 0.4333333333333334, "pass_count": 10}, "candidate_order_shuffled_with_label_remap": {"base_minus_control_pass_at_1_mean": 0.0, "pass_at_1_mean": 0.4291666666666667, "pass_count": 10}, "cross_task_policy_shuffle": {"base_minus_control_pass_at_1_mean": 0.0, "pass_at_1_mean": 0.4291666666666667, "pass_count": 10}, "cross_task_tool_observation_shuffle": {"base_minus_control_pass_at_1_mean": 0.01666666666666668, "pass_at_1_mean": 0.4125, "pass_count": 10}, "cross_task_user_goal_shuffle": {"base_minus_control_pass_at_1_mean": 0.004166666666666685, "pass_at_1_mean": 0.425, "pass_count": 10}, "evidence_only_no_candidate": {"base_minus_control_pass_at_1_mean": -0.016666666666666663, "pass_at_1_mean": 0.4458333333333333, "pass_count": 10}, "hidden_states_shuffled_across_examples": {"base_minus_control_pass_at_1_mean": -0.016666666666666663, "pass_at_1_mean": 0.4458333333333333, "pass_count": 10}, "physical_order_shuffled_roles_avenues_preserved": {"base_minus_control_pass_at_1_mean": 0.0, "pass_at_1_mean": 0.4291666666666667, "pass_count": 10}, "policy_only": {"base_minus_control_pass_at_1_mean": -0.03749999999999999, "pass_at_1_mean": 0.4666666666666667, "pass_count": 10}, "randomized_labels": {"base_minus_control_pass_at_1_mean": 0.3416666666666667, "pass_at_1_mean": 0.0875, "pass_count": 10}, "schema_template_only": {"base_minus_control_pass_at_1_mean": 0.00416666666666668, "pass_at_1_mean": 0.425, "pass_count": 10}, "tool_observations_only": {"base_minus_control_pass_at_1_mean": 0.04166666666666668, "pass_at_1_mean": 0.3875, "pass_count": 10}, "trajectory_only": {"base_minus_control_pass_at_1_mean": 0.020833333333333343, "pass_at_1_mean": 0.4083333333333333, "pass_count": 10}, "user_goal_only": {"base_minus_control_pass_at_1_mean": 0.0, "pass_at_1_mean": 0.4291666666666667, "pass_count": 10}}, "created_at_utc": "2026-05-17T07:41:15.375937+00:00", "evidence_corruption_max_delta": 0.01666666666666668, "generator_identity_only_diagnostic": {"audit_only_not_selector_visible": true, "best_generator_on_train_counts": {"llm_agent:claude-3-7-sonnet-20250219": 10}, "pass_at_1_mean": 0.5416666666666667, "selection_efficiency_mean": 0.65}, "physical_order_invariance_pass_count": 10, "randomized_labels_control_delta": 0.3416666666666667, "stronger_evidence_use_gates": {"candidate_evidence_mismatch_degrades_substantially": false, "cross_task_goal_shuffle_degrades_substantially": false, "cross_task_policy_shuffle_degrades_substantially": false, "cross_task_tool_shuffle_degrades_substantially": false, "full_model_beats_candidate_only_control": false, "full_model_beats_trajectory_only_control": true}}`

## Gates
`{"candidate_order_invariance_passes": true, "candidate_order_randomized_audit_pass": true, "checkpoints_saved": true, "completed_seeds_gte_10": true, "duplicate_and_near_duplicate_audits_pass": true, "evidence_corruption_control_degrades_by_gte_0_05": false, "final_report_generated": true, "gradient_update_frozen_audits_pass": true, "leakage_audits_pass": true, "oracle_pass_at_8_between_0_25_and_0_85": true, "paired_bootstrap_ci_lower_vs_best_non_oracle_gt_0": false, "paired_bootstrap_ci_lower_vs_frozen_gt_0": false, "physical_order_invariance_passes": true, "randomized_labels_collapse": false, "raw_trajectory_separation_audit_pass": true, "trainable_beats_best_non_oracle_by_gte_0_05": false, "trainable_beats_frozen_on_gte_8_of_10_seeds": false, "trainable_mean_pass_at_1_beats_frozen": false, "trainable_selection_efficiency_beats_best_non_oracle": false}`

## Failure Diagnosis
`{"architecture_mismatch": true, "best_baseline_saturation": true, "candidate_trajectories_too_easy_or_hard": false, "evidence_corruption_not_degrading": true, "generator_source_leakage": false, "oracle_pass_at_8_mean": 0.8333333333333334, "oracle_pass_at_8_too_high_or_low": false, "pool_size_too_small": false, "stronger_evidence_use_gates": {"candidate_evidence_mismatch_degrades_substantially": false, "cross_task_goal_shuffle_degrades_substantially": false, "cross_task_policy_shuffle_degrades_substantially": false, "cross_task_tool_shuffle_degrades_substantially": false, "full_model_beats_candidate_only_control": false, "full_model_beats_trajectory_only_control": true}, "trainable_frozen_gap_absent": true}`

## Claim
`No final claim allowed.`
