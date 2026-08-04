# Stage 7C Failure Forensics

## Conclusion
A. PAUSE_STAGE7. Stage 7C failed as a final benchmark result. The dominant diagnosed categories are: dev_overfitting, best_baseline_saturation, trainable_frozen_gap_absent, evidence_controls_failed, candidate_only_or_trajectory_only_explains_result, insufficient_final_task_count, architecture_mismatch, text_embedding_baselines_stronger_than_expected. No Stage 7 success or final benchmark claim is supported by these artifacts.

No model, architecture, pool, split, hyperparameter, control, trajectory, or label was changed. No final benchmark claim is made.

## 1. Final Failure
- Trainable latent mean pass@1: `0.4292`
- Frozen latent mean pass@1: `0.4833`
- Best non-oracle baseline: `best_generator_on_train_baseline` at `0.5417`
- Oracle pass@8: `0.8333`
- Trainable vs frozen delta: `-0.0542`
- Trainable vs best baseline delta: `-0.1125`
- Seeds won/lost vs frozen: `{"lost": 8, "tied": 0, "won": 2}`
- Seeds won/lost vs best baseline: `{"lost": 10, "tied": 0, "won": 0}`
- CI vs frozen: `{"lower": -0.11666666716337204, "mean": -0.05416666716337204, "upper": 0.012604166870005333}`
- CI vs best baseline: `{"lower": -0.17916665971279144, "mean": -0.11249999701976776, "upper": -0.0416666679084301}`
- Failed gates: `["evidence_corruption_control_degrades_by_gte_0_05", "paired_bootstrap_ci_lower_vs_best_non_oracle_gt_0", "paired_bootstrap_ci_lower_vs_frozen_gt_0", "randomized_labels_collapse", "trainable_beats_best_non_oracle_by_gte_0_05", "trainable_beats_frozen_on_gte_8_of_10_seeds", "trainable_mean_pass_at_1_beats_frozen", "trainable_selection_efficiency_beats_best_non_oracle"]`

## 2. Stage 7B Dev vs Stage 7C Final
- Dev trainable: `0.5833`; final trainable: `0.4292`
- Dev frozen: `0.5417`; final frozen: `0.4833`
- Dev best baseline: `0.5972`; final best baseline: `best_generator_on_train_baseline` at `0.5417`
- Dev oracle pass@8: `0.8333`; final oracle pass@8: `0.8333`
- Evidence corruption degradation dev vs final: `0.1528` vs `0.0167`
- Candidate-only dev/final: `{"base_minus_control_pass_at_1_mean": 0.027777777777777773, "pass_at_1_mean": 0.5555555555555556, "pass_count": 3}` / `{"base_minus_control_pass_at_1_mean": -0.004166666666666652, "pass_at_1_mean": 0.4333333333333334, "pass_count": 10}`
- Trajectory-only dev/final: `{"base_minus_control_pass_at_1_mean": 0.18055555555555558, "pass_at_1_mean": 0.40277777777777773, "pass_count": 3}` / `{"base_minus_control_pass_at_1_mean": 0.020833333333333343, "pass_at_1_mean": 0.4083333333333333, "pass_count": 10}`

## 3. Failure Categories
- `dev_overfitting`: `True` evidence=`{"dev_trainable": 0.5833333333333334, "dev_trainable_minus_best": -0.01388888888888884, "final_trainable": 0.4291666666666667, "final_trainable_minus_best": -0.11250000000000004}`
- `final_pool_distribution_shift`: `False` evidence=`{"dev_final_trainable_gap": 0.15416666666666667, "dev_oracle": 0.8333333333333334, "final_oracle": 0.8333333333333334, "note": "Stage 7C reused the frozen held-out task IDs from the Stage 7B split, so the observed score drop is not attributed to oracle-rate distribution shift."}`
- `oracle_pass_at_8_too_high_or_too_low`: `False` evidence=`{"oracle_pass_at_8": 0.8333333333333334}`
- `best_baseline_saturation`: `True` evidence=`{"best_baseline": "best_generator_on_train_baseline", "best_baseline_pass_at_1": 0.5416666666666667, "oracle_minus_best_baseline": 0.29166666666666663}`
- `trainable_frozen_gap_absent`: `True` evidence=`{"seeds_won_lost_vs_frozen": {"lost": 8, "tied": 0, "won": 2}, "trainable_vs_frozen_delta": -0.054166666666666585}`
- `evidence_controls_failed`: `True` evidence=`{"evidence_answers": {"cross_task_degraded": false, "mismatch_degraded": false}, "evidence_corruption_final": 0.01666666666666668}`
- `candidate_only_or_trajectory_only_explains_result`: `True` evidence=`{"candidate_only_delta": -0.004166666666666652, "trajectory_only_delta": 0.020833333333333343}`
- `generator_identity_source_leakage`: `False` evidence=`{"leakage_gates_pass": true}`
- `hidden_evaluator_leakage`: `False` evidence=`{"leakage_gates_pass": true}`
- `insufficient_final_task_count`: `True` evidence=`{"final_task_count": 24}`
- `optimizer_training_instability`: `False` evidence=`{"activation_requires_grad_before_coordinator": true, "candidate_order_invariance": true, "checkpoints_saved": true, "coordinator_grad_delta_gt_0": true, "frozen_agent_grad_delta_zero": true, "leakage_audits_pass": true, "no_detach_in_main_path": true, "physical_order_invariance": true, "shared_parameter_identity": true, "trainable_grad_norm_gt_0": true, "trainable_parameter_delta_gt_0": true}`
- `architecture_mismatch`: `True` evidence=`{"evidence_corruption_final": 0.01666666666666668, "trainable_vs_frozen_delta": -0.054166666666666585}`
- `text_embedding_baselines_stronger_than_expected`: `True` evidence=`{"embedding_reranker": 0.5, "text_only_multi_agent_reviewer": 0.5041666666666667, "text_only_single_reviewer": 0.48749999999999993}`

## 4. Evidence-Use Analysis
- Full model: `{"conditional_accuracy_mean": 0.5149999999999999, "pass_at_1_mean": 0.4291666666666667, "selection_efficiency_mean": 0.5149999999999999}`
- Controls: `{"candidate_evidence_mismatch": {"base_minus_control_pass_at_1_mean": 0.012500000000000011, "pass_at_1_mean": 0.4166666666666667, "pass_count": 10}, "candidate_only": {"base_minus_control_pass_at_1_mean": -0.004166666666666652, "pass_at_1_mean": 0.4333333333333334, "pass_count": 10}, "cross_task_policy_shuffle": {"base_minus_control_pass_at_1_mean": 0.0, "pass_at_1_mean": 0.4291666666666667, "pass_count": 10}, "cross_task_tool_observation_shuffle": {"base_minus_control_pass_at_1_mean": 0.01666666666666668, "pass_at_1_mean": 0.4125, "pass_count": 10}, "cross_task_user_goal_shuffle": {"base_minus_control_pass_at_1_mean": 0.004166666666666685, "pass_at_1_mean": 0.425, "pass_count": 10}, "hidden_states_shuffled_across_examples": {"base_minus_control_pass_at_1_mean": -0.016666666666666663, "pass_at_1_mean": 0.4458333333333333, "pass_count": 10}, "policy_only": {"base_minus_control_pass_at_1_mean": -0.03749999999999999, "pass_at_1_mean": 0.4666666666666667, "pass_count": 10}, "schema_template_only": {"base_minus_control_pass_at_1_mean": 0.00416666666666668, "pass_at_1_mean": 0.425, "pass_count": 10}, "tool_observations_only": {"base_minus_control_pass_at_1_mean": 0.04166666666666668, "pass_at_1_mean": 0.3875, "pass_count": 10}, "trajectory_only": {"base_minus_control_pass_at_1_mean": 0.020833333333333343, "pass_at_1_mean": 0.4083333333333333, "pass_count": 10}, "user_goal_only": {"base_minus_control_pass_at_1_mean": 0.0, "pass_at_1_mean": 0.4291666666666667, "pass_count": 10}}`
- Did evidence mismatch degrade? `False` delta=`0.0125`
- Did cross-task shuffles degrade? `False` deltas=`{"cross_task_policy_shuffle": 0.0, "cross_task_tool_observation_shuffle": 0.01666666666666668, "cross_task_user_goal_shuffle": 0.004166666666666685}`
- Candidate-only matched/exceeded full model? `True` delta=`-0.0042`
- Trajectory-only matched/exceeded full model? `False` delta=`0.0208`
- Interpretation: The final model does not show reliable task-evidence use: evidence mismatch and cross-task shuffles do not degrade by the required margin, and candidate-only slightly exceeds the full model.

## 5. Baseline Saturation
- `best_generator_on_train_baseline` pass@1=`0.5417` conditional=`0.6500` efficiency=`0.6500` MRR=`0.6316` top-2=`0.6250`
- `best_generator_on_train_dev_baseline` pass@1=`0.5417` conditional=`0.6500` efficiency=`0.6500` MRR=`0.6316` top-2=`0.6250`
- `embedding_reranker` pass@1=`0.5000` conditional=`0.6000` efficiency=`0.6000` MRR=`0.5868` top-2=`0.5417`
- `first_trajectory_order_baseline` pass@1=`0.2917` conditional=`0.3500` efficiency=`0.3500` MRR=`0.5087` top-2=`0.6250`
- `frozen_same_architecture_latent_selector` pass@1=`0.4833` conditional=`0.5800` efficiency=`0.5800` MRR=`0.6167` top-2=`0.6667`
- `llm_judge_baseline` unavailable.
- `random_trajectory` pass@1=`0.5250` conditional=`0.6300` efficiency=`0.6300` MRR=`0.6397` top-2=`0.6750`
- `randomized_labels_trainable_latent_selector` pass@1=`0.3917` conditional=`0.4700` efficiency=`0.4700` MRR=`0.5660` top-2=`0.6458`
- `raw_latent_selector` pass@1=`0.4917` conditional=`0.5900` efficiency=`0.5900` MRR=`0.6136` top-2=`0.6167`
- `static_trajectory_feature_reranker` pass@1=`0.4500` conditional=`0.5400` efficiency=`0.5400` MRR=`0.5924` top-2=`0.6167`
- `text_only_multi_agent_reviewer` pass@1=`0.5042` conditional=`0.6050` efficiency=`0.6050` MRR=`0.6088` top-2=`0.5958`
- `text_only_single_reviewer` pass@1=`0.4875` conditional=`0.5850` efficiency=`0.5850` MRR=`0.5973` top-2=`0.5792`
- Strongest baseline: `{"conditional_accuracy": 0.6500000000000001, "method": "best_generator_on_train_baseline", "mrr": 0.6315972222222221, "pass_at_1": 0.5416666666666667, "selection_efficiency": 0.65, "top_2_accuracy": 0.625, "unavailable": false}`
- Strongest baseline may exploit trajectory/source artifacts: `True`
- Oracle minus strongest baseline: `0.2917`

## 6. Pool Difficulty
- Final task count: `24`
- Oracle-positive/empty: `20` / `4`
- All-success/all-fail tasks: `0` / `4`
- Success-count distribution: `{"0": 4, "1": 1, "2": 2, "3": 3, "4": 4, "5": 3, "6": 3, "7": 4}`
- Oracle-positive success-count distribution: `{"1": 1, "2": 2, "3": 3, "4": 4, "5": 3, "6": 3, "7": 4}`
- Random expected pass@1: `0.4740`
- Per-task-type breakdown: `{"retail_official_existing_trajectory": {"oracle_positive_rate": 0.8333333333333334, "tasks": 24}}`
- Per-generator success rates: `{"llm_agent:claude-3-7-sonnet-20250219": {"candidates": 48, "success_rate": 0.4375}, "llm_agent:gpt-4.1-2025-04-14": {"candidates": 46, "success_rate": 0.6086956521739131}, "llm_agent:gpt-4.1-mini-2025-04-14": {"candidates": 49, "success_rate": 0.3877551020408163}, "llm_agent:o4-mini-2025-04-16": {"candidates": 49, "success_rate": 0.46938775510204084}}`
- Near-duplicate trajectory rate: `0.0000`

## 7. Training/Audit Sanity
- Aggregate checks: `{"activation_requires_grad_before_coordinator": true, "candidate_order_invariance": true, "checkpoints_saved": true, "coordinator_grad_delta_gt_0": true, "frozen_agent_grad_delta_zero": true, "leakage_audits_pass": true, "no_detach_in_main_path": true, "physical_order_invariance": true, "shared_parameter_identity": true, "trainable_grad_norm_gt_0": true, "trainable_parameter_delta_gt_0": true}`
- Audit rows: `76` types=`{"candidate_order_randomized": 1, "controls": 10, "duplicate_trajectory_hash": 1, "frozen_audit": 10, "invariance_audit": 10, "near_duplicate_trajectory": 1, "randomized_labels_audit": 10, "raw_latent_audit": 10, "raw_trajectory_separation": 1, "stage7_output_leakage": 1, "stage7_taubench_split_leakage": 1, "success_gates": 10, "trainable_audit": 10}`

## 8. Decision Recommendation
`A. PAUSE_STAGE7`

The failure should not be softened: the final trainable selector underperformed frozen and non-oracle baselines, evidence controls did not support task-evidence use, and no Stage 7 final claim is supported.
