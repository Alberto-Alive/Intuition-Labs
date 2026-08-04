# Stage 6B.1b Candidate Supply Expansion

## Scope

- Generates or ingests generated SWE-bench candidate patches only.
- Does not train the latent selector.
- Does not run the official SWE-bench harness.
- Does not use reference/gold patches as selector candidates.
- Does not use raw-preserving expansion.
- Does not fabricate missing candidates.
- Extracts multiple trajectory diff blocks only when their normalized patch hashes are distinct.
- Stores rejected attempts only in the attempt audit.
- Stores raw trajectories separately from selector-visible patch text.
- No final model claim is made.

## Artifacts

- Generated candidates JSONL: `results/stage6b1b_generated_candidates.jsonl`.
- Generation attempts JSONL: `results/stage6b1b_generation_attempts.jsonl`.
- Generation audit: `results/stage6b1b_generation_audit.json`.
- Report: `reports/STAGE6B1B_CANDIDATE_SUPPLY_EXPANSION.md`.

## Coverage

- Requested tasks: `25`.
- Tasks attempted: `25`.
- Accepted candidate rows: `200`.
- Tasks with >=1 candidate: `25`.
- Tasks with >=4 candidates: `25`.
- Tasks with >=8 candidates: `25`.
- Accepted candidates per task distribution: `{"8": 25}`.

## Source Coverage

- Source files loaded: `3`.
- Total raw records loaded: `755`.
- Unique instance_ids in sources: `600`.
- Overlap with selected benchmark tasks: `25/25`.
- Top missing instance_ids: `[]`.
- Top source files by accepted candidates: `[{"accepted_candidates": 196, "source_name": "hf:togethercomputer/CoderForge-Preview-32B-SWE-Bench-Verified-Evaluation-trajectories:trajectory+messages"}, {"accepted_candidates": 4, "source_name": "hf:hanspeterlyngsoeraaschoujensen/swebench-eval-trajectories:messages"}]`.

## Rejections

- Terminal status counts: `{"accepted": 200, "invalid_diff": 16}`.
- Task skipped cause counts: `{}`.
- Rejection reason counts: `{"invalid_diff": 16}`.
- Invalid diff rate: `0.0741`.
- Invalid diff reasons: `{"invalid_hunk_line_prefix": 2, "missing_file_header": 14}`.
- Duplicate rate: `0.0000`.
- Exact reference patch hits: `0`.
- Near-reference similarity flags: `0`.

## Diversity

- Generator contribution counts: `{"qwen/qwen3-coder-next": 4, "swebench-python-qwen3-coder-32b-sft-mix-25000-no-penalty": 196}`.
- Config contribution counts: `{"hf_message_trajectory": 4, "swebench-python-qwen3-coder-32b-sft-mix-25000-no-penalty": 196}`.
- Mean pairwise normalized patch edit distance: `0.7345`.

## Quality Gates

| gate | pass |
|---|---:|
| `at_least_25_tasks_attempted` | `True` |
| `at_least_20_tasks_with_k8_unique_generated_candidates` | `True` |
| `at_least_160_accepted_candidates_total` | `True` |
| `invalid_diff_rate_reported_and_below_50_percent` | `True` |
| `duplicate_rate_below_threshold` | `True` |
| `no_exact_reference_patch_hits` | `True` |
| `candidate_order_randomizable` | `True` |
| `all_generated_candidate_files_saved` | `True` |
| `no_raw_preserving_expansion` | `True` |
| `no_missing_candidate_fabrication` | `True` |
| `selector_training_not_executed` | `True` |
| `official_harness_not_executed` | `True` |
| `no_final_model_claim_made` | `True` |
| `candidate_supply_expansion_quality_gates_pass` | `True` |

## Next Step

When the gates pass, Stage 6B.0 can consume `--source-jsonl results/stage6b1b_generated_candidates.jsonl` in a later run; the official SWE-bench Docker harness was intentionally not run here.

No final model claim is made.
