# PLAN-ARCH-2.1 N16 Stress-Control Repair

## Scope
- Final validation was not launched.
- The locked N=8 architecture and rule/event token semantics were not changed.
- Candidate self-attention, pyramids, stacking, multi-avenue variants, and broad search were not run.

## Answers
1. Which N16 control failed and why? `rollout_only_near_chance`; candidate-pool balancing weakness: generic rollout probes stayed near chance, but the trained blank-rule control learned an event-order prior from the expanded N16 pool on the failed seed.
2. Was the failure caused by candidate-pool imbalance or model behavior? Candidate-pool balancing weakness in the old N16 expansion created a blank-rule event-order prior; the verifier architecture was unchanged.
3. Can N16 shortcuts be brought back near chance? `True`; max generic shortcut=0.1523.
4. Does `shared_weight_constraint_clone_with_event_tokens` still beat frozen on repaired N16? `4/5`, mean_delta=0.2727, CI95=[0.1180, 0.4220].
5. Is final validation ready for N8 only or N8+N16? `B. Final-ready N8 + N16`.
6. Should the rule-event binding module remain locked? `True`.
7. Should this module later transfer to ARC-hybrid and real-code verification? `True`.

## Repaired N16 Gate
- trainable mean: 1.0000
- frozen mean: 0.7273
- mean delta: 0.2727
- seed wins: 4/5
- gates pass: `True`

## Variant Summary
| variant | trainable | frozen | delta | wins | controls | gate |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| minimal_rule_event_cross_attention | 1.0000 | 0.4859 | 0.5141 | 4 | `True` | `True` |
| shared_weight_constraint_clone_with_event_tokens | 1.0000 | 0.7273 | 0.2727 | 4 | `True` | `True` |
| rule_event_bilinear_verifier | 1.0000 | 0.8227 | 0.1773 | 2 | `True` | `False` |

## Claim Boundary
PLAN-ARCH-2.1 only repairs or diagnoses the N16 stress-control issue. It does not claim final success, autonomous planning, world modeling, general agentic AI, ARC solving, or SOTA planning.
