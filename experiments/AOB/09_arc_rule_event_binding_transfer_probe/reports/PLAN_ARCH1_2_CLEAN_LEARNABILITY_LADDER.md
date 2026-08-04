# PLAN-ARCH-1.2 Clean Learnability Ladder

## Scope
- No planning claim.
- No architecture improvement claim.
- Medium validation was not launched.

## Answers
1. Which clean difficulty level first becomes learnable? `none`.
2. Does final-state evidence help when controls are repaired? `best_with_final=1.0000, best_no_final=0.2812, helps=True`.
3. Can the model infer the final transition without seeing final state? `clean_gate=True, best_top1=0.1875`.
4. Which representation works best: rollout tokens, transition tokens, or hybrid? `transition/hybrid (transition_tuple_verifier top1=0.3125)`.
5. Does candidate self-attention still help on any clean level? `False`.
6. Does trainable beat frozen on clean held-out splits? `True`.
7. Are failures due to over-hard candidate pools or architecture weakness? `scoring/tokenization/mask/gradient issue`.
8. Is there a medium-ready benchmark/variant? `none`.

## First Failure
- `level 1 obvious_clean_success_failure variant=P1_rollout_no_final`

## Leaderboard
| level | variant | seed | trainable | frozen | delta | shortcut | controls |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | oracle_success_flag_sanity | 2 | 1.0000 | 0.8750 | 0.1250 | 0.3125 | `False` |
| 0 | oracle_success_flag_sanity | 0 | 1.0000 | 1.0000 | 0.0000 | 0.2188 | `False` |
| 0 | oracle_success_flag_sanity | 1 | 0.9688 | 0.9375 | 0.0312 | 0.2188 | `False` |
| 1 | transition_tuple_verifier | 0 | 0.2500 | 0.0938 | 0.1562 | 0.2188 | `True` |
| 1 | candidate_token_direct_transition_tokens | 0 | 0.1875 | 0.1250 | 0.0625 | 0.2188 | `True` |
| 1 | P1_rollout_no_final | 0 | 0.1562 | 0.1875 | -0.0312 | 0.2188 | `True` |
| 1 | P1_rollout_with_final_state | 0 | 0.1562 | 0.1875 | -0.0312 | 0.2188 | `True` |
| 1 | simple_cross_attention_verifier_baseline | 0 | 0.1250 | 0.1250 | 0.0000 | 0.2188 | `True` |
| 1 | Q_candidate_self_attention | 1 | 0.0938 | 0.0938 | 0.0000 | 0.2188 | `True` |
| 1 | Q_bidirectional_candidate_evidence | 0 | 0.0625 | 0.1875 | -0.1250 | 0.2188 | `True` |
| 1 | P1_rollout_no_final | 1 | 0.2812 | 0.2812 | 0.0000 | 0.2188 | `False` |
| 1 | P1_rollout_with_final_state | 1 | 0.2812 | 0.2812 | 0.0000 | 0.2188 | `False` |
