# PLAN-ARCH-1.1 Shortcut Regression Repair

## Scope
- No planning claim.
- No architecture improvement claim.
- This stage repairs the benchmark and retests candidate-query mechanisms only.
- Medium validation was not launched.

## Audit Answer
1. Why did shortcut baseline reach 0.6250? The prior PLAN-ARCH-1 shortcut score was the trainable stats_mlp baseline. Its features included rollout-derived collision/progress and start-goal progress, while gold candidates were padded BFS plans and negatives came from visibly different action/progress families.
2. Was this a data-generation artifact, baseline bug, or token leakage? mixed representation leak and baseline bug; the dataset exposed systematic action/progress artifacts and the stats_mlp shortcut consumed rollout/state-goal features instead of candidate-only action statistics.
3. Can the clean PLAN-1.1 baseline be reproduced? Repaired P1 clean controls: `True`. Prior PLAN-1.1 controls are summarized in the audit JSON.
4. Does Q_candidate_self_attention still beat P1 after shortcut repair? `False`.
5. Does candidate self-attention add real evidence use or only exploit candidate-list artifacts? No. candidate_self_attention_only stayed near chance ([0.109375, 0.140625, 0.125]), but candidate_pair_artifact_baseline failed on 2/3 seeds and trainable did not beat frozen reliably.
6. Are any variants medium-ready after repair? `none`.

## Leaderboard
| rank | variant | trainable | frozen | delta | shortcut | controls |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | P1_rollout_no_final | 0.1562 | 0.1719 | -0.0156 | 0.1562 | `True` |
| 2 | Q_bidirectional_candidate_evidence | 0.1250 | 0.0938 | 0.0312 | 0.1562 | `True` |
| 3 | P1_rollout_no_final | 0.1094 | 0.1094 | 0.0000 | 0.1875 | `True` |
| 4 | Q_candidate_self_attention | 0.1094 | 0.1094 | 0.0000 | 0.1875 | `True` |
| 5 | Q_candidate_self_attention_plus_rollout_no_final | 0.1094 | 0.1094 | 0.0000 | 0.1875 | `True` |
| 6 | Q_candidate_self_attention_with_stronger_mismatch_controls | 0.1094 | 0.1094 | 0.0000 | 0.1875 | `True` |
| 7 | Q_bidirectional_candidate_evidence | 0.0938 | 0.1250 | -0.0312 | 0.1875 | `True` |
| 8 | Q_bidirectional_candidate_evidence | 0.0781 | 0.1094 | -0.0312 | 0.1719 | `True` |

## Repair Notes
- Repaired pools use balanced candidate slots and hide source metadata.
- All candidates have the same action length.
- Repaired negatives are no-collision paths whose visible rollout prefix ends one step from the goal; only the final transition determines success.
- The stats shortcut baseline was corrected to use candidate/action statistics only, not rollout progress or collision features.
