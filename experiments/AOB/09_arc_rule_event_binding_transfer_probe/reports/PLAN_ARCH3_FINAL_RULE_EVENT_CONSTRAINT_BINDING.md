# PLAN-ARCH-3 Final Rule-Event Constraint Binding Validation

## Scope
- Architecture, rule/event token semantics, training hyperparameters, and gates were locked before this run.
- Final candidates: shared-weight clone and minimal rule-event cross-attention.
- Candidate self-attention, pyramids, stacking, multi-avenue variants, broad search, and post-hoc tuning were not run.

## Answers
1. Does the clone architecture pass final validation on N=8? `True`.
2. Does it pass final validation on N=16? `False`.
3. Does it beat exact frozen on fresh seeds? N8 `9/10`, N16 `9/10`.
4. Do all shortcut/mismatch/shuffle/leakage controls pass? `False`.
5. Is the result explained by candidate, rollout, endpoint, family, or action shortcuts? `N8 clean; N16 rollout-only shortcut control failed at N16_final:seed72`; max shortcut=0.1758.
6. Does the clone beat or merely match the simple cross-attention baseline? `clone_matches_minimal_top1`.
7. Which variant has the strongest trainable-frozen separation? `minimal_rule_event_cross_attention:N8_final`.
8. Which variant has the best compute/performance tradeoff? `minimal_rule_event_cross_attention:N8_final`.
9. Which constraint family is hardest? `all_tied=1.0000`.
10. Is the result ready to transfer to ARC-hybrid, real-code verification, or program-plan verification? `False`.

## Primary Clone Metrics
| stage | trainable | frozen | delta | wins | CI low | top2 | top3 | MRR | gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| N8_final | 1.0000 | 0.6656 | 0.3344 | 9 | 0.1949 | 1.0000 | 1.0000 | 1.0000 | `True` |
| N16_final | 1.0000 | 0.6340 | 0.3660 | 9 | 0.2141 | 1.0000 | 1.0000 | 1.0000 | `False` |

## Variant Summary
| stage | variant | trainable | frozen | delta | wins | controls | latency ms/ex |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| N8_final | shared_weight_constraint_clone_with_event_tokens | 1.0000 | 0.6656 | 0.3344 | 9 | `True` | 0.5046 |
| N8_final | minimal_rule_event_cross_attention | 1.0000 | 0.5555 | 0.4445 | 9 | `True` | 0.5053 |
| N16_final | shared_weight_constraint_clone_with_event_tokens | 1.0000 | 0.6340 | 0.3660 | 9 | `False` | 3.5259 |
| N16_final | minimal_rule_event_cross_attention | 1.0000 | 0.5949 | 0.4051 | 9 | `False` | 3.4100 |

## Robustness
- Entity/color/key/subgoal permutation mean top1 for clone: 0.9955
- Oracle feature audit pass: `True`

## Claim Boundary
Allowed final claim only if both final gates pass: on controlled synthetic learned-constraint candidate-plan verification, shared-weight latent clone coordination with visible rule/event tokens learns to bind task-specific constraint evidence to candidate rollout events, outperforming an exact frozen same-architecture comparator across fresh seeds while shortcut, mismatch, shuffle, leakage, and invariance controls pass.

Forbidden claims remain: autonomous planning, world modeling, general agentic AI, ARC solving, SOTA planning, and open-ended agent reasoning.
