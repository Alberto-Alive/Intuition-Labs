# Stage ARC-HYBRID-1 Empirical Search

## Scope

- Stage: `ARC-HYBRID-1_EMPIRICAL_LATENT_VERIFIER_FOR_REFINEMENT_LOOP`.
- Bounded empirical search over ARC candidate generation, candidate evidence, latent verification, and verifier-guided refinement.
- The latent architecture is a verifier/ranker/controller inside candidate pools; it is not claimed to solve ARC from scratch.
- Candidate source families, generator rank, and oracle correctness are logged externally only and are not encoded into model inputs.
- Final validation was not launched.

## Leaderboard

| rank | variant | status | trainable | frozen | delta | natural solve | recall | controls |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | A_output_grid_only_latent_cross_attention | completed | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `True` |
| 2 | B_execution_mismatch_latent_verifier | completed | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `True` |

## Candidate Generator Audit

| phase | split | examples | recall | diversity | top source families |
|---|---|---:|---:|---:|---|
| phase1_gold_present | train | 75 | 1.0000 | 0.4850 | forced_gold_control:75, geometric_transform:57, geometric_translation:10, output_grid_mutation:182 |
| phase1_gold_present | dev | 8 | 1.0000 | 0.4609 | forced_gold_control:8, geometric_transform:13, geometric_translation:3, output_grid_mutation:10 |
| phase1_gold_present | test | 9 | 1.0000 | 0.5325 | forced_gold_control:9, geometric_transform:13, geometric_translation:2, output_grid_mutation:17 |
| phase2_natural_pool | train | 75 | 0.0000 | 0.4843 | geometric_transform:65, geometric_translation:11, output_grid_mutation:236, procedural_diff_patch:45 |
| phase2_natural_pool | dev | 8 | 0.0000 | 0.4850 | geometric_transform:11, geometric_translation:3, output_grid_mutation:17, procedural_diff_patch:7 |
| phase2_natural_pool | test | 9 | 0.0000 | 0.5114 | geometric_transform:14, geometric_translation:2, output_grid_mutation:21, procedural_diff_patch:7 |

## Controls And Gates

| gate | pass |
|---|---|
| trainable_beats_frozen_on_at_least_4_of_5_seeds | `False` |
| mean_trainable_frozen_delta_at_least_0_15 | `False` |
| bootstrap_ci_lower_bound_gt_0 | `False` |
| candidate_only_near_chance | `True` |
| metadata_only_near_chance | `True` |
| execution_score_heuristic_does_not_explain_result | `False` |
| train_pair_shuffle_degrades | `True` |
| candidate_evidence_mismatch_degrades | `True` |
| trace_mismatch_degrades_if_used | `True` |
| mismatch_map_mismatch_degrades_if_used | `True` |
| candidate_order_remap_invariance_passes | `True` |
| role_order_invariance_passes | `True` |
| hidden_state_shuffle_collapses | `True` |
| frozen_audit_passes | `True` |
| trainable_update_audit_passes | `True` |
| final_validation_not_launched | `True` |

## Failure Taxonomy

```json
{
  "generator_recall_failure": 2,
  "natural_pool_low_recall": 1,
  "verifier_ranking_failure_did_not_beat_frozen": 2
}
```

## Refinement

| variant | strategy | recall before | recall after | solve before | solve after | quality before | quality after |
|---|---|---:|---:|---:|---:|---:|---:|
| A_output_grid_only_latent_cross_attention | random_refinement | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7498 | 0.7498 |
| A_output_grid_only_latent_cross_attention | heuristic_train_fit_refinement | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7498 | 0.7498 |
| A_output_grid_only_latent_cross_attention | frozen_verifier_guided_refinement | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7498 | 0.7498 |
| A_output_grid_only_latent_cross_attention | trainable_latent_verifier_guided_refinement | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7498 | 0.7498 |
| B_execution_mismatch_latent_verifier | random_refinement | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7498 | 0.7498 |
| B_execution_mismatch_latent_verifier | heuristic_train_fit_refinement | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7498 | 0.7498 |
| B_execution_mismatch_latent_verifier | frozen_verifier_guided_refinement | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7498 | 0.7498 |
| B_execution_mismatch_latent_verifier | trainable_latent_verifier_guided_refinement | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7498 | 0.7498 |

## Required Answers

1. Which candidate generator/refiner produced useful candidate pools? `output_grid_mutation`; see generator audit for external source-family counts.
2. What was generator recall? Phase-2 mean recall `0.0000`.
3. Did latent verification beat frozen on gold-present sets? `False (best delta 0.0000)`.
4. Did latent verification beat execution-score heuristics? `False (trainable 0.0000, execution heuristic 0.0000)`.
5. Did latent verification improve natural candidate-pool selection? `False (trainable 0.0000, frozen 0.0000)`.
6. Which evidence views mattered most? `raw diff object color`.
7. Did traces or mismatch maps help? `best evidence variant B_execution_mismatch_latent_verifier delta 0.0000`.
8. Did program evidence help beyond output evidence? `not_established_without_positive controlled delta`.
9. Did pyramidal/compositional verification help after flat verifier worked? `not_run_in_this_bounded_smoke`; this stage keeps the interface open and gates it behind a flat-verifier pass.
10. Did verifier-guided refinement improve solve rate? `False (best random_refinement solve delta 0.0000)`.
11. Were failures mostly generator recall or verifier ranking? `generator_recall_failure`.
12. Which variants failed and why? `{'generator_recall_failure': 2, 'natural_pool_low_recall': 1, 'verifier_ranking_failure_did_not_beat_frozen': 2}`.
13. Is this hybrid stronger than pure ARC latent-rule-clone verification? `not established unless medium gates pass against the frozen and heuristic baselines`.
14. Is it ready for medium validation? `False`.

## Claim Boundary

- Do not claim ARC-AGI-2 solved.
- Do not claim SOTA, general abstraction, or autonomous reasoning.
- Allowed claim only after gates pass: On controlled ARC-AGI-2 candidate/refinement pools, shared-weight latent clone coordination improves candidate verification/ranking over exact frozen, candidate-only, metadata-only, and heuristic baselines while passing leakage, mismatch, shuffle, and invariance controls.
