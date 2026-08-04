# Stage 6D.3 Candidate Source Expansion

## Scope

- Selector training was not executed.
- Selector architecture was not changed.
- Gold/reference patches, raw-preserving expansion, and fabricated candidates were not used.
- Official labels and raw harness logs are not selector-visible.
- No final SWE-bench improvement claim is made.

## Source Inventory

- Source files scanned: `3`.
- Raw records loaded: `213`.
- Unique source instance IDs: `31`.
- Overlap with SWE-bench Verified: `31`.
- Tasks with >=1 candidate: `31`.
- Tasks with >=4 candidates: `25`.
- Tasks with >=8 candidates: `25`.
- Invalid diff exclusions: `0`.
- Duplicate exclusions: `7`.
- Exact reference exclusions: `0`.
- Near-reference flags: `3`.

## Pool Construction

- Pool all unlabeled K=8 tasks: `25`.
- Pool strict unlabeled K=8 tasks: `23`.
- Strict quarantined task IDs: `django__django-13925, django__django-7530`.

## Self-Generation Plan

- Additional strict K=8 tasks needed for minimum: `27`.
- Additional strict K=8 tasks needed for preferred: `77`.
- Planned task count: `80`.
- Estimated total generation attempts: `2560`.

## Pre-Harness Gates

| Gate | Pass |
|---|---:|
| `at_least_50_k8_generated_candidate_tasks_in_pool_all` | `False` |
| `at_least_50_k8_generated_candidate_tasks_in_pool_strict_or_source_limit_documented` | `True` |
| `exact_reference_patch_hits_zero` | `True` |
| `duplicate_candidate_patch_audit_passes` | `True` |
| `no_raw_preserving_expansion` | `True` |
| `no_fabricated_candidates` | `True` |
| `candidate_order_randomized` | `True` |
| `source_generator_skew_reported` | `True` |
| `near_reference_candidates_flagged_or_quarantined` | `True` |
| `selector_training_not_executed` | `True` |
| `no_final_model_claim_made` | `True` |

## Decision

NEED_SELF_GENERATION
