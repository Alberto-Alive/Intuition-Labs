# Stage 7B.0 tau-bench Candidate Pool Mining

## Decision
- Decision: `A. READY_FOR_STAGE7B_DEV`
- Selector training performed: `False`.
- Final benchmark claim made: `False`.

## Source Inventory
- Source files scanned: `{"existing_stage7_candidate_files": 5, "leaderboard_submission_files": 45, "leaderboard_trajectory_files": 0, "tau2_domain_metadata_files": 18, "tau2_result_files": 26}`
- tau2 repo commit: `0ed8c0ef0a1f6b024a0fb733e186922411874879`
- Raw trajectories loaded: `10832`
- Unique domains: `airline, retail, telecom, telecom-workflow`
- Task counts per domain: `{"airline": 50, "retail": 114, "telecom": 114, "telecom-workflow": 114}`
- Trajectories per task distribution: `{"airline": {"16": 50}, "retail": {"16": 114}, "telecom": {"40": 114}, "telecom-workflow": {"32": 114}}`
- Tasks with >=8 trajectories: `{"airline": 50, "retail": 114, "telecom": 114, "telecom-workflow": 114}`
- Existing-label oracle pass@8 estimate: `{"airline": {"oracle_pass_at_8_estimate": 0.84, "tasks_with_known_labels": 50}, "retail": {"oracle_pass_at_8_estimate": 0.9736842105263158, "tasks_with_known_labels": 114}, "telecom": {"oracle_pass_at_8_estimate": 0.7192982456140351, "tasks_with_known_labels": 114}, "telecom-workflow": {"oracle_pass_at_8_estimate": 0.9649122807017544, "tasks_with_known_labels": 114}}`
- Generator/source distribution: `{"airline|llm_agent:claude-3-7-sonnet-20250219|claude-3-7-sonnet-20250219_airline_default_gpt-4.1-2025-04-14_4trials.json": 200, "airline|llm_agent:gpt-4.1-2025-04-14|gpt-4.1-2025-04-14_airline_default_gpt-4.1-2025-04-14_4trials.json": 200, "airline|llm_agent:gpt-4.1-mini-2025-04-14|gpt-4.1-mini-2025-04-14_airline_base_gpt-4.1-2025-04-14_4trials.json": 200, "airline|llm_agent:o4-mini-2025-04-16|o4-mini-2025-04-16_airline_default_gpt-4.1-2025-04-14_4trials.json": 200, "retail|llm_agent:claude-3-7-sonnet-20250219|claude-3-7-sonnet-20250219_retail_default_gpt-4.1-2025-04-14_4trials.json": 456, "retail|llm_agent:gpt-4.1-2025-04-14|gpt-4.1-2025-04-14_retail_default_gpt-4.1-2025-04-14_4trials.json": 456, "retail|llm_agent:gpt-4.1-mini-2025-04-14|gpt-4.1-mini-2025-04-14_retail_base_gpt-4.1-2025-04-14_4trials.json": 456, "retail|llm_agent:o4-mini-2025-04-16|o4-mini-2025-04-16_retail_default_gpt-4.1-2025-04-14_4trials.json": 456, "telecom-workflow|llm_agent:gpt-4.1-2025-04-14|gpt-4.1-2025-04-14_telecom-workflow_default_gpt-4.1-2025-04-14_4trials.json": 456, "telecom-workflow|llm_agent:o4-mini-2025-04-16|o4-mini-2025-04-16_telecom-workflow_default_gpt-4.1-2025-04-14_4trials.json": 456, "telecom-workflow|llm_agent_gt:gpt-4.1-2025-04-14|gpt-4.1-2025-04-14_telecom-workflow_op_gpt-4.1-2025-04-14_4trials.json": 456, "telecom-workflow|llm_agent_gt:o4-mini-2025-04-16|o4-mini-2025-04-16_telecom-workflow_op_gpt-4.1-2025-04-14_4trials.json": 456, "telecom-workflow|llm_agent_solo:gpt-4.1-2025-04-14|gpt-4.1-2025-04-14_telecom-workflow_no-user_gpt-4.1-2025-04-14_4trials.json": 456, "telecom-workflow|llm_agent_solo:o4-mini-2025-04-16|o4-mini-2025-04-16_telecom-workflow_no-user_gpt-4.1-2025-04-14_4trials.json": 456, "telecom-workflow|llm_agent_solo_gt:gpt-4.1-2025-04-14|gpt-4.1-2025-04-14_telecom-workflow_no-user-op_gpt-4.1-2025-04-14_4trials.json": 456, "telecom-workflow|llm_agent_solo_gt:o4-mini-2025-04-16|o4-mini-2025-04-16_telecom-workflow_no-user-op_gpt-4.1-2025-04-14_4trials.json": 456, "telecom|llm_agent:claude-3-7-sonnet-20250219|claude-3-7-sonnet-20250219_telecom_default_gpt-4.1-2025-04-14_4trials.json": 456, "telecom|llm_agent:gpt-4.1-2025-04-14|gpt-4.1-2025-04-14_telecom_default_gpt-4.1-2025-04-14_4trials.json": 456, "telecom|llm_agent:gpt-4.1-mini-2025-04-14|gpt-4.1-mini-2025-04-14_telecom_base_gpt-4.1-2025-04-14_4trials.json": 456, "telecom|llm_agent:o4-mini-2025-04-16|o4-mini-2025-04-16_telecom_default_gpt-4.1-2025-04-14_4trials.json": 456, "telecom|llm_agent_gt:gpt-4.1-2025-04-14|gpt-4.1-2025-04-14_telecom_op_gpt-4.1-2025-04-14_4trials.json": 456, "telecom|llm_agent_gt:o4-mini-2025-04-16|o4-mini-2025-04-16_telecom_op_gpt-4.1-2025-04-14_4trials.json": 456, "telecom|llm_agent_solo:gpt-4.1-2025-04-14|gpt-4.1-2025-04-14_telecom_no-user_gpt-4.1-2025-04-14_4trials.json": 456, "telecom|llm_agent_solo:o4-mini-2025-04-16|o4-mini-2025-04-16_telecom_no-user_gpt-4.1-2025-04-14_4trials.json": 456, "telecom|llm_agent_solo_gt:gpt-4.1-2025-04-14|gpt-4.1-2025-04-14_telecom_no-user-op_gpt-4.1-2025-04-14_4trials.json": 456, "telecom|llm_agent_solo_gt:o4-mini-2025-04-16|o4-mini-2025-04-16_telecom_no-user-op_gpt-4.1-2025-04-14_4trials.json": 456}`
- Duplicate trajectory rate: `0.000000`
- Near-duplicate trajectory pair rate: `0.000000`

## Official Label Recompute
- Labels available: `True`
- Evaluator: `{"command": ["conda", "run", "-n", "stage7_tau", "tau2", "evaluate-trajs", "results\\stage7b0_tau_eval_input\\retail_source_results.json", "-o", "results\\stage7b0_tau_eval_output"], "input_json": "results\\stage7b0_tau_eval_input\\retail_source_results.json", "labels_available": 1824, "labels_missing": 0, "latency_seconds": 410.5251963000046, "official_evaluator_ran": true, "output_json": "results\\stage7b0_tau_eval_output\\updated_retail_source_results.json", "returncode": 0, "stderr_log": "results\\stage7b0_tau_eval_stderr.log", "stderr_tail": "\u001b[32m2026-05-17 05:08:23.617\u001b[0m | \u001b[33m\u001b[1mWARNING \u001b[0m | \u001b[36mtau2.utils.utils\u001b[0m:\u001b[36m<module>\u001b[0m:\u001b[36m15\u001b[0m - \u001b[33m\u001b[1mNo .env file found\u001b[0m\n\u001b[32m2026-05-17 05:08:23.618\u001b[0m | \u001b[1mINFO    \u001b[0m | \u001b[36mtau2.utils.utils\u001b[0m:\u001b[36m<module>\u001b[0m:\u001b[36m23\u001b[0m - \u001b[1mUsing data directory from environment: W:\\HocusPocus\\`
- Per-domain success stats: `{"retail": {"labels_available": 1824, "success_count": 1324, "success_rate": 0.7258771929824561, "trajectories": 1824}}`

## Pool Summaries
### all_available_pool
- Artifact: `results\stage7b0_tau_candidate_pool_all.jsonl`
- Tasks/candidates: `114` / `912`
- Oracle pass@8: `0.9825`
- Oracle-positive tasks: `112`
- Oracle-empty tasks: `2`
- All-success tasks: `34`
- All-fail tasks: `2`
- Success-count distribution: `{"0": 2, "1": 4, "2": 5, "3": 9, "4": 11, "5": 11, "6": 16, "7": 22, "8": 34}`
- Baselines: first `0.7368`, random `0.7259`, best-generator `0.7456`
- Leakage audit passes: `True`
- Duplicate audit passes: `True`
- Near-duplicate audit passes: `True`
- Readiness gates: `{"all_fail_tasks_retained_but_not_dominant": true, "all_success_tasks_not_dominant": false, "at_least_50_retail_tasks_with_k8_official_labels": true, "duplicate_trajectory_audit_passes_or_reported": true, "first_trajectory_baseline_meaningfully_below_oracle": true, "generator_source_identity_blinded_from_selector_visible_text": true, "near_duplicate_trajectory_audit_passes_or_reported": true, "no_final_claim_made": true, "official_evaluator_recomputed_labels": true, "oracle_pass_at_8_between_0_25_and_0_85": false, "output_evaluator_leakage_audit_passes": true, "preferred_at_least_100_retail_tasks": true, "random_baseline_meaningfully_below_oracle": true}`

### balanced_pool
- Artifact: `results\stage7b0_tau_candidate_pool_balanced.jsonl`
- Tasks/candidates: `93` / `744`
- Oracle pass@8: `0.8495`
- Oracle-positive tasks: `79`
- Oracle-empty tasks: `14`
- All-success tasks: `0`
- All-fail tasks: `14`
- Success-count distribution: `{"0": 14, "1": 5, "2": 6, "3": 12, "4": 14, "5": 13, "6": 11, "7": 18}`
- Baselines: first `0.3978`, random `0.4919`, best-generator `0.5806`
- Leakage audit passes: `True`
- Duplicate audit passes: `True`
- Near-duplicate audit passes: `True`
- Readiness gates: `{"all_fail_tasks_retained_but_not_dominant": true, "all_success_tasks_not_dominant": true, "at_least_50_retail_tasks_with_k8_official_labels": true, "duplicate_trajectory_audit_passes_or_reported": true, "first_trajectory_baseline_meaningfully_below_oracle": true, "generator_source_identity_blinded_from_selector_visible_text": true, "near_duplicate_trajectory_audit_passes_or_reported": true, "no_final_claim_made": true, "official_evaluator_recomputed_labels": true, "oracle_pass_at_8_between_0_25_and_0_85": true, "output_evaluator_leakage_audit_passes": true, "preferred_at_least_100_retail_tasks": false, "random_baseline_meaningfully_below_oracle": true}`

### nontrivial_pool
- Artifact: `results\stage7b0_tau_candidate_pool_nontrivial.jsonl`
- Tasks/candidates: `93` / `744`
- Oracle pass@8: `0.8495`
- Oracle-positive tasks: `79`
- Oracle-empty tasks: `14`
- All-success tasks: `0`
- All-fail tasks: `14`
- Success-count distribution: `{"0": 14, "1": 2, "2": 20, "3": 8, "4": 10, "5": 10, "6": 11, "7": 18}`
- Baselines: first `0.4731`, random `0.4677`, best-generator `0.5054`
- Leakage audit passes: `True`
- Duplicate audit passes: `True`
- Near-duplicate audit passes: `True`
- Readiness gates: `{"all_fail_tasks_retained_but_not_dominant": true, "all_success_tasks_not_dominant": true, "at_least_50_retail_tasks_with_k8_official_labels": true, "duplicate_trajectory_audit_passes_or_reported": true, "first_trajectory_baseline_meaningfully_below_oracle": true, "generator_source_identity_blinded_from_selector_visible_text": true, "near_duplicate_trajectory_audit_passes_or_reported": true, "no_final_claim_made": true, "official_evaluator_recomputed_labels": true, "oracle_pass_at_8_between_0_25_and_0_85": true, "output_evaluator_leakage_audit_passes": true, "preferred_at_least_100_retail_tasks": false, "random_baseline_meaningfully_below_oracle": true}`
