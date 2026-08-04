# Stage 6D.4 Self-Generation Candidate Expansion

## Scope

- Selector training was not executed.
- Selector architecture was not changed.
- Gold/reference patches, raw-preserving expansion, and fabricated candidates were not used.
- Official SWE-bench harness was not executed.
- No final SWE-bench improvement claim is made.

## Generation

- Tasks selected for generation: `78`.
- Tasks attempted: `0`.
- Tasks reaching K=8 from new generation: `0`.
- Accepted generated candidates total: `0`.
- Attempts per accepted candidate: `0.0000`.
- Invalid diff rate: `0.0000`.
- Duplicate rate: `0.0000`.
- Exact reference hits: `0`.
- Near-reference flags: `0`.

## Pool

- Pool all K=8 tasks: `25`.
- Pool strict K=8 tasks: `23`.
- Remaining tasks needed for 50 strict K=8: `27`.
- Remaining tasks needed for 100 strict K=8: `77`.
- Estimated official harness evaluations required next: `184`.
- Mean pairwise normalized patch edit distance: `0.7394`.

## Gates

| Gate | Pass |
|---|---:|
| `at_least_50_pool_all_k8_tasks` | `False` |
| `at_least_50_pool_strict_k8_tasks` | `False` |
| `preferred_at_least_100_pool_strict_k8_tasks` | `False` |
| `exact_reference_patch_hits_zero` | `True` |
| `duplicate_candidate_patch_audit_passes` | `True` |
| `invalid_diff_rate_reported` | `True` |
| `no_raw_preserving_expansion` | `True` |
| `no_fabricated_candidates` | `True` |
| `candidate_order_randomized` | `True` |
| `source_generator_skew_reported` | `True` |
| `near_reference_candidates_flagged_or_quarantined` | `True` |
| `selector_training_not_executed` | `True` |
| `official_harness_not_executed` | `True` |
| `no_final_model_claim_made` | `True` |

## Decision

SELF_GENERATION_BLOCKED

No configured real generation backend produced accepted candidates. Configure command_line_agent, mini_swe_agent, api_generator, or provide local_jsonl_ingest outputs; dry_run is testing-only.
