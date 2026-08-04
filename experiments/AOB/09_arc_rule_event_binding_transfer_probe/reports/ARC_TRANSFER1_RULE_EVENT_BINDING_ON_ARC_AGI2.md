# ARC-TRANSFER-1 Rule-Event Binding on Local ARC-AGI-2

## Scope
- Transfer/probe stage only; no ARC-solving claim.
- Final validation was not launched.
- The locked PLAN-ARCH rule-event binding architecture was used with an exact frozen comparator.

## Dataset Verification
- dataset path: `W:\HocusPocus\ARC-AGI-2`
- loaded correctly: `True`
- task counts: `{'evaluation': 120, 'test': 0, 'training': 1000}`
- example task ids: `['00576224', '007bbfb7', '009d5c81']`

## Answers
1. Did the local ARC-AGI-2 dataset load correctly? `True`.
2. What candidate recall did the generator achieve? natural=0.0294, gold-present=1.0000.
3. In gold-present mode, did the verifier beat frozen? Weakly: N8 delta=0.0526, seed wins=1; N16 ran=`False`.
4. Did rule/event tokens help over raw candidate grids? This runner tests rule/event tokens directly; raw-grid-only is represented by candidate-only/metadata controls, not a new architecture.
5. Did shortcut controls pass? `True`; all controls pass=`True`.
6. In natural-pool mode, did the verifier improve selection? delta_over_heuristic=0.0000.
7. Are failures mostly generator recall or verifier ranking? `{'generator_failed': 35, 'solved': 0, 'verifier_selected_wrong_candidate': 1}`.
8. Is this transfer path worth scaling? `False`; not yet; candidate recall and ARC rule/event extraction need repair before scaling.

## Gold-Present Metrics
| N | trainable | frozen | delta |
| ---: | ---: | ---: | ---: |
| 8 | 0.1610 | 0.1084 | 0.0526 |
| 16 | 0.0000 | 0.0000 | 0.0000 |

## Claim Boundary
Do not interpret this as ARC solving. This stage only probes whether the synthetic rule-event binding verifier transfers to ARC candidate ranking when candidates are supplied.
