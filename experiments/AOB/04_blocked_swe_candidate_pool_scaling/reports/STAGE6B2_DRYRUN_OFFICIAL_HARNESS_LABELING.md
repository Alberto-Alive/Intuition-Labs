# Stage 6B.2 Official Harness Labeling

## Scope

- Candidate source: `results/stage6b1b_generated_candidates.jsonl`.
- Candidate patches were copied unchanged into slot prediction files.
- Correctness labels are accepted only from the official SWE-bench harness `resolved` field.
- Raw harness logs are stored under the harness root and are not included in selector-visible candidate fields.
- No selector training was executed and no final selector claim is made.

## Artifacts

- Labeled candidate pool: `results/stage6b2_dryrun_labeled_candidate_pools.jsonl`.
- Harness results: `results/stage6b2_dryrun_harness_results.jsonl`.
- Official label audit: `results/stage6b2_dryrun_official_label_audit.json`.
- Harness root: `results/stage6b2_dryrun_official_harness`.
- Report: `reports/STAGE6B2_DRYRUN_OFFICIAL_HARNESS_LABELING.md`.

## Harness

- Official command module: `python -m swebench.harness.run_evaluation`.
- Harness executed: `False`.
- Labels available: `0/200`.
- Tasks with complete labels: `0/25`.

## Label Summary

- Tasks loaded: `25`.
- K=8 preserved: `True`.
- Oracle pass@8: `0.0000`.
- Oracle-positive tasks: `0`.
- Oracle-empty tasks: `0`.
- Per-repository oracle pass@8: `{}`.
- Per-generator pass rate: `{}`.

## Baselines

- First-candidate pass@1: `0.0000`.
- Random-candidate pass@1: `0.0000`.
- Best-generator-on-dev pass@1: `0.0000`.
- Best generator on dev: `None`.

## Audits

- Duplicate candidate patch hash audit passes: `True`.
- Exact reference patch hash hits: `0`.
- Candidate order randomized audit passes: `True`.
- Output leakage audit passes: `True`.
- Source/generator leakage audit passes: `False`.
- Raw-preserving expansion used: `False`.
- Fabricated candidates used: `False`.

## Quality Gates

| gate | pass |
|---|---:|
| `all_25_tasks_preserve_k8_candidates` | `True` |
| `official_harness_labels_available_for_all_200_candidates` | `False` |
| `oracle_pass_at_8_between_0_20_and_0_80` | `False` |
| `at_least_5_oracle_positive_tasks` | `False` |
| `first_candidate_baseline_does_not_equal_oracle` | `False` |
| `best_generator_source_baseline_does_not_equal_oracle` | `False` |
| `no_gold_reference_patch_leakage` | `True` |
| `all_audits_complete` | `True` |
| `no_raw_preserving_expansion` | `True` |
| `no_fabricated_candidates` | `True` |
| `candidate_patches_not_altered` | `True` |
| `selector_training_not_executed` | `True` |
| `official_harness_executed_or_reused` | `False` |
| `no_final_selector_claim_made` | `True` |
| `stage6b2_official_label_quality_gates_pass` | `False` |

No final selector claim is made.
