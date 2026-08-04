# PLAN-ARCH-1.5 Learned-Constraint Shortcut Repair

## Scope
- No planning claim.
- No architecture improvement claim.
- Medium validation was not launched.
- Candidate self-attention, pyramids, stacking, and multi-avenue variants were not run.

## Answers
1. What shortcut explained the PLAN-ARCH-1.4 scores? `candidate_only was highest among recorded 1.4 shortcut controls (means={candidate_only:0.5594, state_goal_only:0.5208, constraint_examples_only:0.5479, action_bigram:0.2552}); this points to candidate/source-family construction artifacts, not endpoint solving.`.
2. Does the repaired dataset make endpoint, action, rollout-only, and family-proxy baselines near chance? `True` (shortcut_max_mean=0.1615, random=0.1250).
3. Are constraint examples necessary? `not established (mismatch=15/15, ablation_hurts=0/15, clean_above_chance=0/15)`.
4. Is rollout evidence necessary? `not established (mismatch=15/15, ablation_hurts=0/15, clean_above_chance=0/15)`.
5. Does rollout + constraint evidence outperform either alone? `rollout_only=0.0781, constraint_only=0.1250, rollout_plus_constraint=1.0000, pass=True`.
6. Which constraint family remains learnable after repair? `color_zone=0.182; key_door=0.238; ordered_subgoal=0.238`.
7. Does trainable beat frozen? `transition_tuple_verifier`.
8. Does the clone architecture beat minimal cross-attention? `False`.
9. Are any variants valid enough for medium validation? `False`.
10. Should architecture search resume after shortcut repair? `False`.

## Shortcut Decomposition
| baseline | mean top1 |
| --- | ---: |
| action_bigram | 0.1510 |
| action_unigram | 0.1250 |
| candidate_action_only | 0.1354 |
| candidate_slot_probe | 0.1250 |
| constraint_examples_only | 0.1250 |
| endpoint_only | 0.1250 |
| family_id_proxy_probe | 0.1250 |
| family_specific_heuristics | 1.0000 |
| length_only | 0.1250 |
| random | 0.1250 |
| rollout_only | 0.0781 |
| rollout_plus_constraint_oracle | 1.0000 |
| rollout_statistics_only | 0.0885 |
| terrain_only | 0.1250 |

## Leaderboard
| variant | seed | trainable | frozen | delta | shortcut | controls |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| transition_tuple_verifier | 0 | 0.1250 | 0.0625 | 0.0625 | 0.1406 | `False` |
| candidate_token_direct_transition_tokens | 2 | 0.1719 | 0.1250 | 0.0469 | 0.1875 | `False` |
| P1_rollout_no_final_constraint | 0 | 0.1094 | 0.0625 | 0.0469 | 0.1406 | `False` |
| minimal_transition_cross_attention | 0 | 0.1719 | 0.0000 | 0.0000 | 0.1719 | `False` |
| transition_tuple_verifier | 2 | 0.1406 | 0.0938 | 0.0469 | 0.1875 | `False` |
| P1_rollout_no_final_constraint | 1 | 0.1250 | 0.1250 | 0.0000 | 0.1719 | `False` |
| shared_weight_constraint_clone_verifier | 2 | 0.1250 | 0.1250 | 0.0000 | 0.1719 | `False` |
| minimal_transition_cross_attention | 2 | 0.1250 | 0.0000 | 0.0000 | 0.1719 | `False` |
| minimal_transition_cross_attention | 1 | 0.1094 | 0.0000 | 0.0000 | 0.1719 | `False` |
| candidate_token_direct_transition_tokens | 0 | 0.0938 | 0.1250 | -0.0312 | 0.1406 | `False` |
| shared_weight_constraint_clone_verifier | 0 | 0.0938 | 0.1250 | -0.0312 | 0.1406 | `False` |
| transition_tuple_verifier | 1 | 0.0781 | 0.1406 | -0.0625 | 0.1719 | `False` |
| candidate_token_direct_transition_tokens | 1 | 0.0781 | 0.1094 | -0.0312 | 0.2188 | `False` |
| shared_weight_constraint_clone_verifier | 1 | 0.0938 | 0.1406 | -0.0469 | 0.2188 | `False` |
| P1_rollout_no_final_constraint | 2 | 0.0469 | 0.1562 | -0.1094 | 0.1406 | `False` |

## Variant Summary
| variant | trainable mean | frozen mean | mean delta | seed wins | controls | gate |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| minimal_transition_cross_attention | 0.1354 | 0.0000 | 0.0000 | 0 | `False` | `False` |
| transition_tuple_verifier | 0.1146 | 0.0990 | 0.0156 | 2 | `False` | `False` |
| candidate_token_direct_transition_tokens | 0.1146 | 0.1198 | -0.0052 | 1 | `False` | `False` |
| shared_weight_constraint_clone_verifier | 0.1042 | 0.1302 | -0.0260 | 0 | `False` | `False` |
| P1_rollout_no_final_constraint | 0.0938 | 0.1146 | -0.0208 | 1 | `False` | `False` |

## Claim Boundary
This stage repairs the learned-constraint benchmark and tests whether constraint-conditioned candidate-plan verification is cleanly learnable. It does not claim planning or an architecture improvement.
