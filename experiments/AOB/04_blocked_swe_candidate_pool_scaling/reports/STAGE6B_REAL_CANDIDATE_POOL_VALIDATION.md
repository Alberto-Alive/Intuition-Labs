# Stage 6B Real Candidate-Pool Validation

## Scope

- No final claim is made from this Stage 6B-real run.
- Candidate patches are generated candidates only: public CoderForge/OpenHands `output_patch` artifacts expanded to K=8 with raw-preserving generated variants.
- Reference/gold patches are not used as candidates; exact reference-patch hash hits are audited separately.
- Correctness labels come only from the official Docker SWE-bench harness `resolved` result when available.
- Raw harness logs are kept under the harness root and are not selector-visible input.

## Environment

- Created at UTC: `2026-05-16T17:18:07.797827+00:00`.
- Platform: `Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.39`.
- Python: `3.12.3`.
- Docker: `29.0.1 linux/amd64`.
- WSL: `Linux Syschestrator 5.15.167.4-microsoft-standard-WSL2 #1 SMP Tue Nov 5 00:21:55 UTC 2024 x86_64 x86_64 x86_64 GNU/Linux`.
- Packages: `{"datasets": "4.8.5", "docker": "7.1.0", "numpy": "2.4.5", "swebench": "4.1.0", "torch": "2.12.0"}`.
- Official harness command module: `python -m swebench.harness.run_evaluation`.
- Official harness namespace/image tags: namespace `swebench`, instance tag `latest`, env tag `latest`.
- Official docs: https://www.swebench.com/SWE-bench/guides/evaluation/

## Commands

- Runner argv: `["/mnt/w/HocusPocus/armyofbots - m4 but picpalac choose/src/experiments/run_stage6b_real_candidate_pool_validation.py", "--n-tasks", "1", "--reuse-harness", "--max-workers", "1", "--timeout", "1800", "--selector-epochs", "1", "--pool-output", "results/stage6b_real_candidate_pools.jsonl", "--pool-audit", "results/stage6b_real_candidate_pool_audit.json", "--selector-results", "results/stage6b_real_selector_results.json", "--selector-audit", "results/stage6b_real_selector_audit.jsonl", "--report", "reports/STAGE6B_REAL_CANDIDATE_POOL_VALIDATION.md"]`.
- Official harness command template: `python -m swebench.harness.run_evaluation --dataset_name {dataset_name} --split {split} --predictions_path {predictions_path} --max_workers {max_workers} --timeout {timeout} --run_id {run_id} --cache_level {cache_level} --clean False --report_dir {report_dir} --namespace {namespace}`.
- WSL setup used for this run: `python3 -m venv .venv_stage6b` then `. .venv_stage6b/bin/activate` then `python -m pip install swebench datasets torch numpy pytest`.

## Artifacts

- Candidate pool: `results/stage6b_real_candidate_pools.jsonl`.
- Candidate-pool audit: `results/stage6b_real_candidate_pool_audit.json`.
- Selector results: `results/stage6b_real_selector_results.json`.
- Selector audit: `results/stage6b_real_selector_audit.jsonl`.
- Harness root: `results/stage6b_real_harness`.
- Official raw harness log copies: `results/stage6b_real_harness/official_logs`.

## Candidate Pool

- Requested tasks: `1`.
- Tasks built: `1`.
- Tasks with all official labels: `1`.
- K=8 validation passes: `True`.
- No exact gold/reference candidate hash hits: `True`.
- Duplicate candidate patch hash audit passes: `True`.
- Candidate order randomized audit passes: `True`.
- Output leakage audit passes: `True`.

## Harness Labels

- Official harness result availability: `8/8`.
- Availability audit passes: `True`.
- Oracle pass@8: `0.0000`.
- Oracle-positive tasks: `0`.
- Oracle-empty tasks retained for unconditional pass@1: `1`.
- Per-repository oracle pass@8: `{"astropy/astropy": 0.0}`.
- Per-generator pass rate: `{"coderforge_qwen3_candidate_generator_0": 0.0, "coderforge_qwen3_candidate_generator_1": 0.0, "coderforge_qwen3_candidate_generator_2": 0.0, "coderforge_qwen3_candidate_generator_3": 0.0}`.

## Baselines And Selector

- First-candidate pass@1: `0.0000`.
- Random-candidate pass@1: `0.0000`.
- Best generator-source pass rate: `0.0000`.
- Selector completed rows: `0`.
- Selector completed seeds: `0`.
- Selector claim status: `no_final_claim`.
- Selector error: `ValueError: Stage 6 latent training requires at least one oracle-positive train example`.

## Success Criteria

| criterion | pass |
|---|---:|
| `at_least_100_real_tasks_complete_or_documented` | `False` |
| `k8_valid_generated_candidates_per_task` | `True` |
| `official_harness_labels_available_for_all_evaluated_candidates` | `True` |
| `oracle_pass_at_8_between_0_20_and_0_80` | `False` |
| `first_candidate_not_trivially_equal_oracle` | `False` |
| `generator_source_not_trivially_equal_oracle` | `False` |
| `all_audits_ran` | `True` |
| `no_gold_reference_patch_leakage` | `True` |
| `no_final_claim_made` | `True` |

## Fewer Than 100 Tasks

Fewer than 100 tasks completed in this run. The run remains a real-harness validation artifact because labels are only accepted from the official SWE-bench harness, but it does not meet the Stage 6B-real completion target. Scaling this to 100 tasks requires running 800 official Docker evaluations for K=8.

No final claim is made.
