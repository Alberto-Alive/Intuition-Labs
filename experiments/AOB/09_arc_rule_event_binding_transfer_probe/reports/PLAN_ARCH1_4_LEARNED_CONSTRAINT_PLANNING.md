# PLAN-ARCH-1.4 Learned Constraint Planning Benchmark

## Scope
- No autonomous planning claim.
- No world-model claim.
- No architecture improvement claim unless gates pass.
- Medium validation was not launched.
- Candidate self-attention, pyramid, stacking, and multi-avenue search were not run.

## Answers
1. Does learned-constraint planning avoid endpoint-heuristic solving? `True` (endpoint=0.1250, random=0.1250).
2. Can trainable latent coordination beat frozen? `P1_rollout_no_final_constraint`.
3. Which constraint families are learnable? `color_zone=0.273; key_door=1.000; ordered_subgoal=0.524`.
4. Does the model use constraint examples? `0/15 pass`.
5. Does it use rollout/transition evidence? `10/15 pass`.
6. Are shortcut baselines near chance? `0/15 pass`.
7. Does minimal transition cross-attention outperform clone architecture? `False`.
8. Is the clone architecture adding value over simple baselines? `best_clone=0.5938, minimal=0.5885, adds_value=True`.
9. Is there a medium-ready learned-constraint variant? `False`.
10. Should architecture search resume after this benchmark is established? `False`.

## Leaderboard
| variant | seed | trainable | frozen | delta | shortcut | controls |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| minimal_transition_cross_attention | 0 | 0.5781 | 0.0000 | 0.0000 | 0.2500 | `False` |
| transition_tuple_verifier | 2 | 0.5938 | 0.4062 | 0.1875 | 0.5781 | `False` |
| candidate_token_direct_transition_tokens | 2 | 0.5938 | 0.5000 | 0.0938 | 0.5781 | `False` |
| shared_weight_constraint_clone_verifier | 2 | 0.5938 | 0.5000 | 0.0938 | 0.5781 | `False` |
| P1_rollout_no_final_constraint | 2 | 0.5938 | 0.5156 | 0.0781 | 0.5781 | `False` |
| minimal_transition_cross_attention | 2 | 0.5938 | 0.0000 | 0.0000 | 0.5781 | `False` |
| transition_tuple_verifier | 1 | 0.5938 | 0.5938 | 0.0000 | 0.5938 | `False` |
| candidate_token_direct_transition_tokens | 1 | 0.5938 | 0.5938 | 0.0000 | 0.5938 | `False` |
| P1_rollout_no_final_constraint | 1 | 0.5938 | 0.5938 | 0.0000 | 0.5938 | `False` |
| shared_weight_constraint_clone_verifier | 1 | 0.5938 | 0.5938 | 0.0000 | 0.5938 | `False` |
| minimal_transition_cross_attention | 1 | 0.5938 | 0.0000 | 0.0000 | 0.5938 | `False` |
| P1_rollout_no_final_constraint | 0 | 0.5938 | 0.5625 | 0.0312 | 0.6562 | `False` |
| transition_tuple_verifier | 0 | 0.5000 | 0.5312 | -0.0312 | 0.6562 | `False` |
| shared_weight_constraint_clone_verifier | 0 | 0.5000 | 0.5312 | -0.0312 | 0.6562 | `False` |
| candidate_token_direct_transition_tokens | 0 | 0.5000 | 0.5938 | -0.0938 | 0.6562 | `False` |

## Variant Summary
| variant | trainable mean | frozen mean | mean delta | seed wins | controls | gate |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| P1_rollout_no_final_constraint | 0.5938 | 0.5573 | 0.0365 | 2 | `False` | `False` |
| minimal_transition_cross_attention | 0.5885 | 0.0000 | 0.0000 | 0 | `False` | `False` |
| transition_tuple_verifier | 0.5625 | 0.5104 | 0.0521 | 1 | `False` | `False` |
| candidate_token_direct_transition_tokens | 0.5625 | 0.5625 | 0.0000 | 1 | `False` | `False` |
| shared_weight_constraint_clone_verifier | 0.5625 | 0.5417 | 0.0208 | 1 | `False` | `False` |

## Claim Boundary
This stage establishes a better candidate-plan verification benchmark. It does not claim autonomous planning, world-modeling, or an architecture improvement.
