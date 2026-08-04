# Stage PLAN-1.1 Rollout-Control Repair and Multi-Seed Planning Search

## Scope

- Final validation run: `False`.
- Claim boundary: no autonomous planning, world-model, or general-agent claim.
- Repaired control: `final_state_mismatch` now changes both explicit `outcome_view` final-position tokens and the last `rollout_view` state token.

## Multi-Seed Results

| variant | seeds | trainable | frozen | delta | candidate-only | length-only | unigram | bigram | ablation drop | controls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P1_rollout_enabled | 3 | 0.3177 | 0.1354 | 0.1823 | 0.0677 | 0.0990 | 0.1042 | 0.1458 | 0.2188 | `False` |
| P1_rollout_no_final | 3 | 0.2812 | 0.1198 | 0.1615 | 0.0469 | 0.0990 | 0.1042 | 0.1458 | 0.1823 | `True` |
| simple_cross_attention_rollout_baseline | 3 | 0.3021 | 0.1562 | 0.1458 | 0.0469 | 0.0990 | 0.1042 | 0.1458 | 0.2031 | `True` |
| P1_rollout_plus_cost | 3 | 0.2917 | 0.1875 | 0.1042 | 0.0938 | 0.0990 | 0.1042 | 0.1458 | 0.1927 | `True` |
| P1_stacked_2 | 3 | 0.3438 | 0.2552 | 0.0885 | 0.0625 | 0.0990 | 0.1042 | 0.1458 | 0.2448 | `False` |
| P1_final_only | 3 | 0.0521 | 0.0573 | -0.0052 | 0.0521 | 0.0990 | 0.1042 | 0.1458 | 0.0000 | `False` |
| P1_rollout_plus_risk | 3 | 0.4062 | 0.4479 | -0.0417 | 0.0833 | 0.0990 | 0.1042 | 0.1458 | 0.3073 | `False` |
| P1_pyramidal_rollout_composition | 3 | 0.5938 | 0.6562 | -0.0625 | 0.0990 | 0.0990 | 0.1042 | 0.1458 | 0.4948 | `False` |

## Medium Trigger

| gate | pass |
|---|---|
| trainable_beats_frozen_on_at_least_2_of_3_cheap_seeds | `True` |
| mean_trainable_frozen_delta_at_least_0_15 | `True` |
| top1_clearly_above_random | `True` |
| candidate_only_near_chance | `True` |
| length_only_near_chance | `True` |
| action_stat_near_chance | `True` |
| metadata_source_only_near_chance | `True` |
| state_plan_mismatch_collapses | `True` |
| goal_shuffle_collapses | `True` |
| rollout_mismatch_collapses | `True` |
| final_state_mismatch_collapses_or_removed | `True` |
| candidate_order_remap_invariance_passes | `True` |
| role_order_invariance_passes | `True` |
| meaningful_view_ablation_hurts | `True` |
| no_shortcut_baseline_explains_result | `True` |
| frozen_comparator_remains_frozen | `True` |
| trainable_shared_model_changes | `True` |
| final_validation_not_launched | `True` |
| passes | `True` |
| variant | `P1_rollout_no_final` |
| final_state_evidence_present | `False` |
| recommended_variant | `P1_rollout_no_final` |
| p1_rollout_enabled_passes | `False` |

## Final-State Mismatch Audit

- `normal_evaluation` mean top1: `0.3177` values `[0.3438, 0.3594, 0.25]`.
- `final_state_removed` mean top1: `0.3229` values `[0.3438, 0.375, 0.25]`.
- `final_state_zeroed` mean top1: `0.3177` values `[0.3438, 0.3594, 0.25]`.
- `rollout_kept_correct_final_state_wrong` mean top1: `0.3281` values `[0.3281, 0.3594, 0.2969]`.
- `rollout_wrong_final_state_kept_correct` mean top1: `0.1302` values `[0.1719, 0.0625, 0.1562]`.
- `both_rollout_and_final_state_wrong` mean top1: `0.1458` values `[0.1719, 0.1719, 0.0938]`.
- `repaired_final_state_mismatch` mean top1: `0.3229` values `[0.3281, 0.3438, 0.2969]`.

## Rollout Evidence Audit

- `normal_evaluation` mean top1: `0.3177` values `[0.3438, 0.3594, 0.25]`.
- `rollout_order_shuffle` mean top1: `0.2812` values `[0.2656, 0.3125, 0.2656]`.
- `rollout_action_mismatch` mean top1: `0.1042` values `[0.1719, 0.0469, 0.0938]`.
- `rollout_state_mismatch` mean top1: `0.1042` values `[0.0938, 0.0938, 0.125]`.
- `rollout_prefix_only` mean top1: `0.1875` values `[0.2031, 0.1875, 0.1719]`.
- `rollout_suffix_only` mean top1: `0.1667` values `[0.2188, 0.1094, 0.1719]`.
- `rollout_final_only` mean top1: `0.0260` values `[0.0781, 0.0, 0.0]`.
- `rollout_no_final` mean top1: `0.3073` values `[0.3438, 0.3438, 0.2344]`.

## Required Answers

1. Was `final_state_mismatch` failing because of a model shortcut, unused final-state tokens, or a control bug? The old control was malformed because it left correct endpoint evidence in `rollout_view[-1]`; after repair, explicit final-state evidence still appears mostly unused/redundant because P1 retains performance when final-state tokens are corrupted.
2. Does repaired P1 still beat frozen? P1 mean trainable `0.3177` vs frozen `0.1354`, delta `0.1823`.
3. Does the result survive seeds `[0,1,2]`? Trainable beats frozen on `3` of `3` seeds.
4. Does the model depend on rollout evidence, final-state evidence, or both? Mostly rollout evidence; explicit final-state corruption with rollout kept correct retains performance.
5. Do rollout mismatch and final-state mismatch now collapse? Recommended variant `P1_rollout_no_final` has rollout gate `True` and final-state gate `True`. Original P1 final-state gate is `False`.
6. Does risk/cost cloning help after controls are repaired? Risk delta over P1 trainable `0.0885`; cost delta over P1 trainable `-0.0260`. Risk controls pass `False`, cost controls pass `True`.
7. Does pyramidal planning composition improve over flat rollout querying? Pyramid mean trainable `0.5938` vs P1 `0.3177`, delta `0.2760`.
8. Can no-rollout latent inference learn under easier curricula? Best curriculum `N4_collision_wrong_goal` trainable `0.3125` vs random `0.2500` and frozen `0.2188`.
9. Is PLAN-1 ready for medium validation? `True` for recommended variant `P1_rollout_no_final`; original P1_rollout_enabled remains `False`.
10. Is this pivot empirically stronger than ARC-1.1? Recommended PLAN-1.1 variant `P1_rollout_no_final` passes cheap gates; P1 delta `0.1823` vs ARC-1.1 best delta `0.0000`; the comparison is synthetic-vs-ARC and should not be overclaimed.

## Risk-Clone Diagnosis

- Mean trainable top1: `0.4062`.
- Mean frozen top1: `0.4479`.
- Mean delta: `-0.0417`.
- Diagnosis: Frozen remains competitive/high on at least one seed; risk features behave like an easy random-feature signal and must be treated as suspicious.

## Claim Boundary

No autonomous planning claim. No world-model claim. No general-agent claim. The only allowed claim remains gated: "On controlled synthetic candidate-plan verification, rollout-enabled shared-weight latent coordination beats an exact frozen comparator while using state/goal/constraint/rollout evidence rather than candidate artifacts."
