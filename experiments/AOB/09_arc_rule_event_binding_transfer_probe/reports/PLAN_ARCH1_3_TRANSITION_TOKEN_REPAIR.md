# PLAN-ARCH-1.3 Transition-Token Verifier Repair

## Scope
- No planning claim.
- No architecture improvement claim.
- Medium validation was not launched.
- Candidate self-attention, pyramids, stacking, and multi-avenue variants were not tested.

## Answers
1. Can deterministic visible heuristics solve each clean ladder level? `L1 final_position_equals_goal=1.000; L2 final_position_equals_goal=1.000; L3 final_position_equals_goal=1.000; L4 final_position_equals_goal=1.000`.
2. Are decisive transition/goal/obstacle tokens visible to the model? `pass=True, audited_examples=12`.
3. Can transition_tuple_verifier micro-overfit? `True`.
4. Do candidate logits and gradients behave correctly? `True`, failures=[].
5. Does a minimal transition model learn? `True`.
6. Does the clone architecture learn once transition representation is fixed? `True`.
7. Does final-state-visible help without oracle flags? `oracle_top1=1.0 diagnostic_only=True; final_visible_best=1.000; no_final_best=0.000`.
8. Can no-final transition inference work after easier levels pass? `ran=False reason=Levels 1-4 did not pass transition-only held-out gates.`.
9. Is the problem representation, architecture, objective, or data difficulty? `representation is learnable, but controls/shortcut/mismatch gates are not clean`.
10. Is any transition verifier ready for medium validation? `False`.

## Micro-Overfit
| gate | N | examples | train acc | target | pass |
| --- | ---: | ---: | ---: | ---: | --- |
| 4_examples_N2 | 2 | 4 | 1.0000 | 0.9500 | `True` |
| 16_examples_N4 | 4 | 16 | 1.0000 | 0.9000 | `True` |
| 64_examples_N8 | 8 | 64 | 1.0000 | 0.8500 | `True` |

## Held-Out Rows
| level | variant | seed | top1 | frozen | delta | shortcut | controls |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | candidate_token_direct_transition_tokens | 0 | 0.9688 | 0.9062 | 0.0625 | 0.2812 | `False` |
| 1 | engineered_feature_diagnostic_mlp | 0 | 1.0000 | 0.0000 | 0.0000 | 0.2812 | `False` |
| 1 | minimal_transition_cross_attention | 0 | 1.0000 | 0.0000 | 0.0000 | 0.2812 | `False` |
| 1 | simple_cross_attention_verifier_baseline | 0 | 1.0000 | 0.9062 | 0.0938 | 0.2812 | `False` |
| 1 | transition_tuple_verifier | 0 | 1.0000 | 0.9375 | 0.0625 | 0.2812 | `False` |
| 2 | engineered_feature_diagnostic_mlp | 0 | 1.0000 | 0.0000 | 0.0000 | 0.1562 | `False` |
| 2 | minimal_transition_cross_attention | 0 | 1.0000 | 0.0000 | 0.0000 | 0.1562 | `True` |
| 2 | transition_tuple_verifier | 0 | 0.9375 | 0.9688 | -0.0312 | 0.1562 | `True` |
| 3 | engineered_feature_diagnostic_mlp | 0 | 1.0000 | 0.0000 | 0.0000 | 0.1875 | `False` |
| 3 | minimal_transition_cross_attention | 0 | 1.0000 | 0.0000 | 0.0000 | 0.1875 | `True` |
| 3 | transition_tuple_verifier | 0 | 0.9688 | 0.9688 | 0.0000 | 0.1875 | `False` |
| 4 | engineered_feature_diagnostic_mlp | 0 | 1.0000 | 0.0000 | 0.0000 | 0.3438 | `False` |
| 4 | minimal_transition_cross_attention | 0 | 1.0000 | 0.0000 | 0.0000 | 0.3438 | `False` |
| 4 | transition_tuple_verifier | 0 | 1.0000 | 0.9688 | 0.0312 | 0.3438 | `False` |

## Medium Trigger
```json
{
  "candidate_order_remap_passes": true,
  "controls_pass": false,
  "hidden_state_shuffle_collapses": true,
  "level_1_clean_heldout_passes": false,
  "level_2_clean_heldout_passes": true,
  "mean_delta": 0.015625,
  "mean_delta_at_least_0_10": false,
  "meaningful_transition_token_ablation_hurts": true,
  "overall": false,
  "role_order_remap_passes": true,
  "shortcut_baselines_near_chance": false,
  "three_seed_requirement_evaluable": false,
  "trainable_beats_frozen_on_at_least_2_of_3_seeds": false,
  "transition_tuple_micro_overfits": true
}
```

## Claim Boundary
This is an implementation and representation repair stage only. The outputs do not claim planning ability or an architecture improvement.
