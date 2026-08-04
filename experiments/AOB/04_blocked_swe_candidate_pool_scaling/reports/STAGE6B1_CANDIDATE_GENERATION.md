# Stage 6B.1 Candidate Generation

## Scope

- Generates or ingests real generated SWE-bench candidate patches only.
- Does not train the latent selector.
- Does not run the official SWE-bench harness.
- Does not use reference/gold patches as candidates.
- Does not use raw-preserving expansion.
- Does not fabricate missing candidates.
- Stores raw trajectory records separately from selector candidate patch text.
- Generator identity is retained in audit/output metadata, not embedded in selector-visible patch text.
- No final model claim is made.

## Artifacts

- Generated candidates JSONL: `results\stage6b1_generated_candidates.jsonl`.
- Generation audit: `results\stage6b1_generation_audit.json`.
- Report: `reports\STAGE6B1_CANDIDATE_GENERATION.md`.

## Summary

- Requested tasks: `10`.
- Tasks attempted: `2`.
- Candidate rows accepted: `4`.
- Tasks with K unique generated candidates: `0`.
- Candidates per task distribution: `{"1": 1, "3": 1}`.
- Duplicate rate before selection: `0.0000`.
- Exact reference hits excluded: `0`.
- Generator distribution: `{"qwen/qwen3-coder-next": 4}`.
- Mean pairwise normalized patch edit distance: `0.6978`.

## Audits

- Empty patch audit passes: `True`.
- Invalid diff audit passes: `False`.
- Duplicate patch hash audit passes: `True`.
- Exact reference hash audit passes: `True`.
- Near-reference similarity hits: `0`.
- Candidate files saved audit passes: `True`.
- Raw-preserving expansion audit passes: `True`.

## Quality Gates

| gate | pass |
|---|---:|
| `at_least_25_tasks_attempted` | `False` |
| `at_least_20_tasks_with_k8_unique_generated_candidates` | `False` |
| `no_exact_reference_patch_hits` | `True` |
| `duplicate_rate_below_threshold` | `True` |
| `candidate_order_randomizable` | `True` |
| `all_generated_candidate_files_saved` | `True` |
| `no_raw_preserving_expansion` | `True` |
| `no_missing_candidate_fabrication` | `True` |
| `no_final_model_claim_made` | `True` |
| `selector_training_executed` | `False` |
| `candidate_generation_quality_gates_pass` | `False` |

## Current Limitation

Stage 6B.1 quality gates did not pass. Add more generator backends or local generated prediction files, then rerun this script. The script intentionally keeps tasks with fewer than K unique patches incomplete rather than inventing candidates.

## Next Step

When the gates pass, run Stage 6B.0 with `--source-jsonl results\stage6b1_generated_candidates.jsonl` and the official SWE-bench Docker harness.

No final model claim is made.
