# PLAN-ARCH-2 Medium Rule-Event Constraint Binding Validation

## Scope
- Final validation was not launched.
- Architecture candidates were locked before medium validation.
- Pyramids, stacking, candidate self-attention, and broad search were not run.

## Answers
1. Does `shared_weight_constraint_clone_with_event_tokens` beat frozen on fresh seeds? `5/5`, mean_delta=0.7078, CI95=[0.646875, 0.775].
2. Does it pass all shortcut, mismatch, shuffle, and leakage controls? `True`.
3. Are event/rule tokens claimable? `True`; tokens are audited as visible constraint-example and rollout-derived, with teacher labels used only as training targets.
4. Does the clone architecture beat or match simple rule-event baselines? `clone=1.0000, best_simple=1.0000`.
5. Which family is hardest after scaling? `color_zone=1.0000`.
6. Does N=16 remain strong? `False` mean_delta=0.7078.
7. Do rule-token and event-token ablations hurt? `rule_hurt_mean=0.8828, event_hurt_mean=0.8984`.
8. Does the result survive entity/color/key/subgoal permutation? `mean_top1=1.0000`.
9. Is this ready for final validation? `False`.
10. Should the winning rule-event binding module be transferred back to ARC-hybrid and real-code candidate verification? `False`.

## Medium Gates
| stage | mean trainable | mean frozen | mean delta | seed wins | CI low | gates |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| N8_medium | 1.0000 | 0.2922 | 0.7078 | 5 | 0.6469 | `True` |
| N16_stress | 1.0000 | 0.2922 | 0.7078 | 5 | 0.6469 | `False` |

## Variant Summary
| variant | trainable mean | frozen mean | mean delta | seed wins |
| --- | ---: | ---: | ---: | ---: |
| shared_weight_constraint_clone_with_event_tokens | 1.0000 | 0.2922 | 0.7078 | 5 |
| minimal_rule_event_cross_attention | 1.0000 | 0.6875 | 0.3125 | 4 |
| P1_rollout_no_final_constraint_with_event_tokens | 1.0000 | 0.6875 | 0.3125 | 4 |
| transition_tuple_verifier_with_aux_heads | 1.0000 | 0.7734 | 0.2266 | 2 |
| rule_event_bilinear_verifier | 1.0000 | 0.7781 | 0.2219 | 2 |

## Claim Boundary
Allowed medium claim only if gates pass: on controlled learned-constraint candidate-plan verification, shared-weight latent clone coordination with rule-event tokens learns to bind constraint evidence to candidate rollout events, outperforming an exact frozen comparator while shortcut, mismatch, shuffle, and leakage controls pass.

Forbidden claims remain: autonomous planning, world modeling, general agentic AI, ARC solving, and SOTA planning.
