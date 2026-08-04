# Stage 6B.2 Official Harness Labeling

## Scope

- Candidate source: `results/stage6b1b_generated_candidates.jsonl`.
- Candidate patches were copied unchanged into slot prediction files.
- Correctness labels are accepted only from the official SWE-bench harness `resolved` field.
- Raw harness logs are stored under the harness root and are not included in selector-visible candidate fields.
- No selector training was executed and no final selector claim is made.

## Artifacts

- Labeled candidate pool: `results/stage6b2_labeled_candidate_pools.jsonl`.
- Harness results: `results/stage6b2_harness_results.jsonl`.
- Official label audit: `results/stage6b2_official_label_audit.json`.
- Harness root: `results/stage6b2_official_harness`.
- Report: `reports/STAGE6B2_OFFICIAL_HARNESS_LABELING.md`.

## Harness

- Official command module: `python -m swebench.harness.run_evaluation`.
- Harness executed: `True`.
- Labels available: `195/200`.
- Tasks with complete labels: `20/25`.

## Label Summary

- Tasks loaded: `25`.
- K=8 preserved: `True`.
- Oracle pass@8: `0.3500`.
- Oracle-positive tasks: `7`.
- Oracle-empty tasks: `13`.
- Per-repository oracle pass@8: `{"astropy/astropy": 0.0, "django/django": 0.5555555555555556, "matplotlib/matplotlib": 0.25, "psf/requests": 0.0, "pydata/xarray": 0.0, "pytest-dev/pytest": 0.0, "sphinx-doc/sphinx": 1.0}`.
- Per-generator pass rate: `{"qwen/qwen3-coder-next": 0.5, "swebench-python-qwen3-coder-32b-sft-mix-25000-no-penalty": 0.07692307692307693}`.

## Baselines

- First-candidate pass@1: `0.1500`.
- Random-candidate pass@1: `0.0500`.
- Best-generator-on-dev pass@1: `0.1500`.
- Best generator on dev: `swebench-python-qwen3-coder-32b-sft-mix-25000-no-penalty`.

## Audits

- Duplicate candidate patch hash audit passes: `True`.
- Exact reference patch hash hits: `0`.
- Candidate order randomized audit passes: `True`.
- Output leakage audit passes: `True`.
- Source/generator leakage audit passes: `True`.
- Raw-preserving expansion used: `False`.
- Fabricated candidates used: `False`.

## Quality Gates

| gate | pass |
|---|---:|
| `all_25_tasks_preserve_k8_candidates` | `True` |
| `official_harness_labels_available_for_all_200_candidates` | `False` |
| `oracle_pass_at_8_between_0_20_and_0_80` | `True` |
| `at_least_5_oracle_positive_tasks` | `True` |
| `first_candidate_baseline_does_not_equal_oracle` | `True` |
| `best_generator_source_baseline_does_not_equal_oracle` | `True` |
| `no_gold_reference_patch_leakage` | `True` |
| `all_audits_complete` | `True` |
| `no_raw_preserving_expansion` | `True` |
| `no_fabricated_candidates` | `True` |
| `candidate_patches_not_altered` | `True` |
| `selector_training_not_executed` | `True` |
| `official_harness_executed_or_reused` | `True` |
| `no_final_selector_claim_made` | `True` |
| `stage6b2_official_label_quality_gates_pass` | `False` |

No final selector claim is made.
