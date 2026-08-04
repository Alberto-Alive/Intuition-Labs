# PLAN-ARCH-1.6 Oracle-to-Latent Constraint Binding Bridge

## Scope
- No planning claim.
- No architecture improvement claim unless gates pass.
- Medium validation was not launched.
- Candidate self-attention, pyramids, stacking, and multi-avenue variants were not run.

## Answers
1. At which oracle-ladder level does learning first work? `A_direct_oracle_input`.
2. Does direct oracle input solve as expected? `1.0000`.
3. Does parsed event input make the task learnable? `0.9688`.
4. Can the model infer active rules from constraint examples? `tracked; best_dev_top1=1.0000`.
5. Can the model bind active rules to candidate rollout events? `mismatch=15/15, ablation_both_hurt=15/15`.
6. Do auxiliary heads help? `best_aux=1.0000, no_aux=1.0000, helps=False`.
7. Does pretraining help? `best_pretrain=1.0000, best_nonpretrain=1.0000, helps=False`.
8. Does rule-event cross-attention or bilinear matching help? `cross=1.0000, bilinear=1.0000`.
9. Which constraint family remains hardest? `color_zone=1.0000`.
10. Does any claimable variant beat frozen while controls pass? `True`.
11. Should broad architecture search resume after the binding module works? `True`.

## Oracle Ladder
| level | mean top1 |
| --- | ---: |
| A_direct_oracle_input | 1.0000 |
| B_violation_type_input | 1.0000 |
| C_parsed_event_input | 0.9688 |
| D_structured_constraint_input | 1.0000 |
| E_raw_constraint_examples | 1.0000 |

## Leaderboard
| variant | seed | trainable | frozen | delta | shortcut | controls |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| rule_event_bilinear_verifier | 2 | 1.0000 | 0.1406 | 0.8594 | 0.1406 | `True` |
| transition_tuple_verifier_with_aux_heads | 2 | 1.0000 | 0.1406 | 0.8594 | 0.1406 | `True` |
| shared_weight_constraint_clone_with_event_tokens | 1 | 1.0000 | 0.2500 | 0.7500 | 0.1719 | `True` |
| rule_event_bilinear_verifier | 1 | 1.0000 | 0.2656 | 0.7344 | 0.1719 | `True` |
| transition_tuple_verifier_with_aux_heads | 1 | 1.0000 | 0.2656 | 0.7344 | 0.1719 | `True` |
| shared_weight_constraint_clone_with_event_tokens | 2 | 1.0000 | 0.4219 | 0.5781 | 0.1406 | `True` |
| minimal_rule_event_cross_attention | 2 | 1.0000 | 0.5781 | 0.4219 | 0.1406 | `True` |
| P1_rollout_no_final_constraint_with_event_tokens | 2 | 1.0000 | 0.5781 | 0.4219 | 0.1406 | `True` |
| shared_weight_constraint_clone_with_event_tokens | 0 | 1.0000 | 0.6094 | 0.3906 | 0.1406 | `True` |
| minimal_rule_event_cross_attention | 1 | 1.0000 | 0.7500 | 0.2500 | 0.1719 | `True` |
| P1_rollout_no_final_constraint_with_event_tokens | 1 | 1.0000 | 0.7500 | 0.2500 | 0.1719 | `True` |
| rule_event_bilinear_verifier | 0 | 1.0000 | 0.9219 | 0.0781 | 0.1406 | `True` |
| transition_tuple_verifier_with_aux_heads | 0 | 1.0000 | 0.9219 | 0.0781 | 0.1406 | `True` |
| minimal_rule_event_cross_attention | 0 | 1.0000 | 1.0000 | 0.0000 | 0.1406 | `True` |
| P1_rollout_no_final_constraint_with_event_tokens | 0 | 1.0000 | 1.0000 | 0.0000 | 0.1562 | `True` |

## Variant Summary
| variant | trainable mean | frozen mean | mean delta | seed wins | controls | gate |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| minimal_rule_event_cross_attention | 1.0000 | 0.7760 | 0.2240 | 2 | `True` | `True` |
| rule_event_bilinear_verifier | 1.0000 | 0.4427 | 0.5573 | 3 | `True` | `True` |
| shared_weight_constraint_clone_with_event_tokens | 1.0000 | 0.4271 | 0.5729 | 3 | `True` | `True` |
| transition_tuple_verifier_with_aux_heads | 1.0000 | 0.4427 | 0.5573 | 3 | `True` | `True` |
| P1_rollout_no_final_constraint_with_event_tokens | 1.0000 | 0.7760 | 0.2240 | 2 | `True` | `True` |

## Claim Boundary
This stage is a bridge from symbolic oracle reasoning to latent neural constraint binding. It does not claim planning or an architecture improvement unless gates pass.
