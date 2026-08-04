# Stage 6B.1a Generator Reliability Ramp

## Scope

- Generates or ingests generated SWE-bench candidate patches only.
- Does not train the latent selector.
- Does not run the official SWE-bench harness.
- Does not use reference/gold patches as selector candidates.
- Does not use raw-preserving expansion.
- Does not fabricate missing candidates.
- Stores rejected attempts only in the attempt audit.
- Stores raw trajectories separately from selector-visible patch text.
- No final model claim is made.

## Artifacts

- Generated candidates JSONL: `results\stage6b1a_generated_candidates.jsonl`.
- Generation attempts JSONL: `results\stage6b1a_generation_attempts.jsonl`.
- Generation audit: `results\stage6b1a_generation_audit.json`.
- Report: `reports\STAGE6B1A_GENERATOR_RELIABILITY_RAMP.md`.

## Coverage

- Requested tasks: `25`.
- Tasks attempted: `25`.
- Accepted candidate rows: `9`.
- Tasks with >=1 candidate: `4`.
- Tasks with >=4 candidates: `0`.
- Tasks with >=8 candidates: `0`.
- Accepted candidates per task distribution: `{"0": 21, "1": 1, "2": 1, "3": 2}`.

## Rejections

- Terminal status counts: `{"accepted": 9, "invalid_diff": 1, "no_patch_found": 16, "task_skipped": 21}`.
- Rejection reason counts: `{"invalid_diff": 1, "no_patch_found": 16}`.
- Invalid diff rate: `0.0385`.
- Invalid diff reasons: `{"invalid_hunk_line_prefix": 1}`.
- Duplicate rate: `0.0000`.
- Exact reference patch hits: `0`.
- Near-reference similarity flags: `0`.

## Diversity

- Generator contribution counts: `{"qwen/qwen3-coder-next": 9}`.
- Config contribution counts: `{"hf_message_trajectory": 9}`.
- Mean pairwise normalized patch edit distance: `0.2763`.

## Quality Gates

| gate | pass |
|---|---:|
| `at_least_25_tasks_attempted` | `True` |
| `at_least_20_tasks_with_k8_unique_generated_candidates` | `False` |
| `invalid_diff_rate_reported_and_below_50_percent` | `True` |
| `duplicate_rate_below_threshold` | `True` |
| `no_exact_reference_patch_hits` | `True` |
| `candidate_order_randomizable` | `True` |
| `all_generated_candidate_files_saved` | `True` |
| `no_raw_preserving_expansion` | `True` |
| `no_missing_candidate_fabrication` | `True` |
| `selector_training_executed` | `False` |
| `official_harness_evaluation_executed` | `False` |
| `no_final_model_claim_made` | `True` |
| `generator_reliability_quality_gates_pass` | `False` |

## Current Limitation

Stage 6B.1a quality gates did not pass. Add more generator backends or generated prediction artifacts, then rerun this script. The runner intentionally leaves underfilled tasks underfilled rather than inventing candidates.

## Next Step

When the gates pass, run Stage 6B.0 with `--source-jsonl results\stage6b1a_generated_candidates.jsonl` and then the official SWE-bench Docker harness.

No final model claim is made.
