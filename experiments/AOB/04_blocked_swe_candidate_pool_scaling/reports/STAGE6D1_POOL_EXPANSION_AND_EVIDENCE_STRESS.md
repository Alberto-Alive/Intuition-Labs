# Stage 6D.1 Pool Expansion and Evidence Stress

## Scope

- Selector training was not executed.
- Candidate patches were not altered.
- Gold/reference patches, raw-preserving expansion, and fabricated candidates were not used.
- Official labels and raw harness logs are excluded from selector-visible fields.
- No final SWE-bench improvement claim is made.

## Near-Reference Quarantine

- Near-reference threshold: `0.95`.
- Near-reference candidates: `2`.
- Quarantined full tasks: `2`.
- Quarantined task ids: `django__django-13925, django__django-7530`.

## Pool Status

- Pool all complete K=8 tasks: `20`.
- Pool all oracle-positive tasks: `7`.
- Pool all oracle pass@8: `0.3500`.
- Pool strict complete K=8 tasks: `18`.
- Pool strict oracle-positive tasks: `5`.
- Pool strict oracle pass@8: `0.2778`.

## Official Label Expansion

- New official harness runs executed: `False`.
- Complete tasks after expansion: `20`.
- Minimum target: `50`.
- Preferred target: `100`.
- Compute limit documented: `True`.

## Evidence Stress

- Pool all evidence diagnostic passes: `True`.
- Pool all issue+context+patch minus patch-only: `0.0000`.
- Pool all mismatch degradation: `0.0500`.
- Pool strict evidence diagnostic passes: `True`.
- Pool strict issue+context+patch minus patch-only: `0.0000`.
- Pool strict mismatch degradation: `0.0556`.

## Gates

| Gate | Pass |
|---|---:|
| `near_reference_candidates_quarantined_or_explicitly_audited` | `True` |
| `at_least_50_complete_k8_officially_labeled_tasks_or_compute_limit_documented` | `True` |
| `at_least_15_oracle_positive_tasks` | `False` |
| `oracle_pass_at_8_between_0_20_and_0_80` | `True` |
| `no_exact_reference_hash_hits` | `True` |
| `no_raw_preserving_expansion` | `True` |
| `no_fabricated_candidates` | `True` |
| `output_leakage_audit_passes` | `True` |
| `source_generator_leakage_audit_passes` | `True` |
| `duplicate_patch_hash_audit_passes` | `True` |
| `evidence_use_diagnostic_passes` | `True` |
| `no_selector_final_claim_made` | `True` |

## Blockers

- insufficient official harness throughput / generated K=8 task supply for the 50-task minimum.
- insufficient oracle-positive tasks after near-reference quarantine.
- generator/source skew remains high.
- near-reference contamination was present and required full-task quarantine.
- patch-only shortcut dominance remains on pool_strict.

## Decision

Do not train the selector; Stage 6D.1 gates did not pass.
