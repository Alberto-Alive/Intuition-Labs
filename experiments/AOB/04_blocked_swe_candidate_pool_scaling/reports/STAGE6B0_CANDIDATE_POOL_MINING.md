# Stage 6B.0 Candidate Pool Mining

## Scope

- No final model claim is made from Stage 6B.0.
- The selector is not trained by this mining script.
- Candidate patches are accepted only from generated public prediction or trajectory artifacts.
- Reference/gold patches are excluded by exact normalized patch hash and audited.
- Raw-preserving patch expansion is not used.
- Correctness labels are accepted only from the official Docker SWE-bench harness `resolved` result.
- Raw harness logs are stored under the harness root and are not selector-visible input.
- True generator/source identities are retained only in audit metadata; selector-visible candidate source is blinded.

## Environment

- Created at UTC: `2026-05-16T17:50:31.395267+00:00`.
- Platform: `Windows-11-10.0.26200-SP0`.
- Python: `3.13.2`.
- Docker: `29.0.1 linux/amd64`.
- WSL: `FileNotFoundError: [WinError 2] The system cannot find the file specified`.
- Packages: `{"datasets": "4.6.1", "docker": "not found", "numpy": "2.1.2", "pytest": "9.0.2", "swebench": "not found", "torch": "2.7.0+cu128"}`.
- Official harness command module: `python -m swebench.harness.run_evaluation`.

## Artifacts

- Candidate pool: `results\stage6b0_candidate_pool_mining.jsonl`.
- Candidate-pool audit: `results\stage6b0_candidate_pool_mining_audit.json`.
- Report: `reports\STAGE6B0_CANDIDATE_POOL_MINING.md`.
- Harness root: `results\stage6b0_harness`.
- Official raw harness log copies: `results\stage6b0_harness\official_logs`.

## Source Mining

- Benchmark dataset loaded: `True`.
- Benchmark rows: `500`.
- Generated patch records loaded: `2`.
- Official tasks with at least one generated candidate after filters: `2`.
- Official tasks with at least K=8 unique generated candidates: `0`.
- Raw-preserving expansion used: `False`.
- Exact reference patch exclusions from sources: `0`.
- Duplicate patch exclusions: `0`.
- Invalid diff exclusions: `0`.

## Candidate Pool

- Requested tasks: `25`.
- Tasks built: `0`.
- K=8 validation passes: `False`.
- Duplicate candidate patch hash audit passes: `True`.
- Exact reference candidate audit passes: `True`.
- Candidate order randomized audit passes: `False`.
- Output leakage audit passes: `True`.

## Official Harness Labels

- Candidate result availability: `0/0`.
- Tasks with all official labels: `0`.
- Availability audit passes: `False`.
- Oracle pass@8 on complete labeled tasks: `0.0000`.
- Oracle-positive tasks: `0`.
- Oracle-empty tasks retained for unconditional pass@1: `0`.
- Per-repository oracle pass@8: `{}`.
- Per-generator pass rate: `{}`.

## Baselines

- Random candidate pass@1: `0.0000`.
- First-candidate pass@1: `0.0000`.
- Best-generator-on-dev baseline: `{"all_pass_at_1": 0.0, "by_split_pass_at_1": {}, "dev_candidate_level_rates": {}, "selected_generator": null}`.

## Quality Gates

| gate | pass |
|---|---:|
| `at_least_100_real_tasks_attempted_or_documented_limit` | `True` |
| `at_least_50_tasks_with_full_k8_official_labels` | `False` |
| `oracle_pass_at_8_between_0_20_and_0_80` | `False` |
| `at_least_30_oracle_positive_train_examples` | `False` |
| `at_least_10_oracle_positive_dev_examples` | `False` |
| `first_candidate_baseline_not_equal_oracle` | `False` |
| `best_generator_source_baseline_not_equal_oracle` | `False` |
| `no_gold_reference_patch_leakage` | `True` |
| `all_audits_complete` | `True` |
| `no_raw_preserving_expansion` | `True` |
| `no_final_model_claim_made` | `True` |
| `pool_quality_gates_pass` | `False` |

## Current Limitation

The Stage 6B.0 pool quality gates did not pass. This script does not fabricate missing candidates: tasks with fewer than K=8 unique generated public patches are excluded, and raw-preserving expansion remains disabled. Add more public prediction artifacts through `--source-jsonl`, `--source-json`, or `--source-root`, then rerun with the same seed to resume harness labeling.

No final model claim is made.
