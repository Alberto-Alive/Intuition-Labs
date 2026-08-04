# Stage 7B tau-bench Dev Validation

## Decision
- Decision: `A. READY_FOR_STAGE7C_FINAL`
- Selected config for Stage 7C: `variant_d_4x1_lr3e4_clip1`
- Stage 7C final validation run: `False`.
- Final benchmark claim made: `False`.

## Dataset
- Candidate pool: `results\stage7b0_tau_candidate_pool_balanced.jsonl`
- Pool name: `balanced_pool`
- Domain: `retail`
- K: `8`
- Split summary: `{"dev": {"candidates": 192, "generator_source_distribution_audit_only": {"llm_agent:claude-3-7-sonnet-20250219": 48, "llm_agent:gpt-4.1-2025-04-14": 46, "llm_agent:gpt-4.1-mini-2025-04-14": 49, "llm_agent:o4-mini-2025-04-16": 49}, "oracle_empty_tasks": 4, "oracle_pass_at_8": 0.8333333333333334, "success_count_distribution": {"0": 4, "1": 1, "2": 2, "3": 3, "4": 4, "5": 3, "6": 3, "7": 4}, "tasks": 24}, "task_overlap_train_dev": 0, "train": {"candidates": 552, "generator_source_distribution_audit_only": {"llm_agent:claude-3-7-sonnet-20250219": 115, "llm_agent:gpt-4.1-2025-04-14": 144, "llm_agent:gpt-4.1-mini-2025-04-14": 159, "llm_agent:o4-mini-2025-04-16": 134}, "oracle_empty_tasks": 10, "oracle_pass_at_8": 0.855072463768116, "success_count_distribution": {"0": 10, "1": 4, "2": 4, "3": 9, "4": 10, "5": 10, "6": 8, "7": 14}, "tasks": 69}}`

## Variant Comparison
### variant_a_4x4_lr3e4_clip1
- Trainable pass@1 mean: `0.4722`
- Frozen pass@1 mean: `0.4028`
- Trainable-frozen delta: `0.0694`
- Best non-oracle baseline mean: `0.5972`
- Trainable-best baseline delta: `-0.1250`
- Oracle pass@8 mean: `0.8333`
- Selection efficiency mean: `0.5667`
- Control health mean: `0.7500`

### variant_b_4x4_lr1e4_clip1
- Trainable pass@1 mean: `0.4444`
- Frozen pass@1 mean: `0.3194`
- Trainable-frozen delta: `0.1250`
- Best non-oracle baseline mean: `0.5972`
- Trainable-best baseline delta: `-0.1528`
- Oracle pass@8 mean: `0.8333`
- Selection efficiency mean: `0.5333`
- Control health mean: `0.7500`

### variant_c_4x4_dropout005_lr3e4_clip1
- Trainable pass@1 mean: `0.4722`
- Frozen pass@1 mean: `0.3750`
- Trainable-frozen delta: `0.0972`
- Best non-oracle baseline mean: `0.5972`
- Trainable-best baseline delta: `-0.1250`
- Oracle pass@8 mean: `0.8333`
- Selection efficiency mean: `0.5667`
- Control health mean: `0.7500`

### variant_d_4x1_lr3e4_clip1
- Trainable pass@1 mean: `0.5833`
- Frozen pass@1 mean: `0.5417`
- Trainable-frozen delta: `0.0417`
- Best non-oracle baseline mean: `0.5972`
- Trainable-best baseline delta: `-0.0139`
- Oracle pass@8 mean: `0.8333`
- Selection efficiency mean: `0.7000`
- Control health mean: `0.8333`

## Gates
`{"candidate_order_invariance_passes": true, "completed_seeds_gte_3": true, "duplicate_and_near_duplicate_audits_pass": true, "gradient_update_audits_pass": true, "leakage_audits_pass": true, "mismatch_or_cross_task_shuffle_degrades_by_0_05": true, "no_final_claim_made": true, "one_final_config_selected_for_stage7c": true, "oracle_pass_at_8_between_0_25_and_0_85": true, "physical_order_invariance_passes": true, "randomized_labels_collapse": true, "trainable_beats_frozen_mean_pass_at_1": true, "trainable_beats_random_and_first": true, "trainable_competitive_with_embedding_static_text": true}`

## Failure Diagnosis
`{"architecture_mismatch": false, "candidate_only_shortcut": true, "candidate_pool_too_easy_or_hard": false, "embedding_static_text_saturation": false, "evidence_views_not_useful": false, "insufficient_training_tasks": false, "oracle_pass_at_8_mean": 0.8333333333333334, "possible_generator_identity_leakage": false}`

## Notes
- Labels are official tau2 evaluator success/failure labels from Stage 7B.0.
- Generator/source identity is used only in audit-only baselines and breakdowns.
- Selector-visible fields remain free of official labels, evaluator verdicts, hidden goal state, and generator identity.
- This is dev-only architecture/config selection and signal detection.
