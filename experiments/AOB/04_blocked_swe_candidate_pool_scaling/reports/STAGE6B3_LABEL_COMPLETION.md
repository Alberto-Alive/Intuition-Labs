# Stage 6B.3 Label Completion

## Scope

- Candidate patches were not changed.
- Official labels are accepted only from SWE-bench harness `resolved` reports.
- Missing labels are quarantined rather than fabricated or treated as negatives.
- No selector training was executed in Stage 6B.3.

## Artifacts

- Complete-label subset: `results/stage6b3_completed_labeled_candidate_pools.jsonl`.
- Quarantine file: `results/stage6b3_quarantined_incomplete_labels.jsonl`.
- Audit: `results/stage6b3_label_completion_audit.json`.

## Label Completion

- Initial missing candidate labels: `5`.
- Final missing candidate labels: `5`.
- Official labels available: `195/200`.
- Resume attempted: `False`.
- Resume namespace: `none`.

## Complete Subset

- Complete tasks: `20`.
- Complete candidates: `160`.
- Oracle pass@8: `0.3500`.
- Oracle-positive tasks: `7`.
- Oracle-empty tasks: `13`.

## Quarantine

- Quarantined tasks: `5`.
- Quarantined candidates: `5`.
- Task ids: `["sphinx-doc__sphinx-9591", "sympy__sympy-15349", "sphinx-doc__sphinx-9711", "sympy__sympy-14248", "sympy__sympy-15345"]`.

## Stage 6B.2 Reporting Inconsistency

- Finding: `historical gate naming/reporting inconsistency fixed`.
- Explanation: The Stage 6B.2 report line `Harness executed: True` was correct: the official harness was invoked. The older quality gate `official_harness_executed_or_reused: False` was a naming bug because it actually required complete label availability. The Stage 6B.2 runner now separates harness invocation from the real failing condition: official labels are still incomplete for all 200 candidates.

## Success

| gate | pass |
|---|---:|
| `all_200_labels_or_strict_complete_subset` | `True` |
| `complete_subset_has_only_fully_labeled_tasks` | `True` |
| `incomplete_tasks_quarantined` | `True` |
| `no_selector_visible_leakage` | `True` |
| `no_patch_alteration` | `True` |
| `no_gold_reference_candidate_leakage` | `True` |
| `report_inconsistency_explained` | `True` |
| `stage6b3_success` | `True` |

No final selector claim is made.
