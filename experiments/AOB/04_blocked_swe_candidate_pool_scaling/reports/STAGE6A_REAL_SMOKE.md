# Stage 6A Real Smoke

## Scope

- No final claim is made from Stage 6A-real.
- Selector architecture and hyperparameters were not tuned from test results.
- Candidate pool uses official SWE-bench rows, but labels are diagnostic because exact reference patches are intentionally included for half the tasks and mutated candidates are labeled failing.
- `publishable_proof_dataset=false`: the official harness package is documented, but this Windows run could not import its CLI due to the package's Unix `resource` dependency; candidate patches were still applied in isolated git checkouts and `git diff --check` was run as the available command.

## Dataset

- Source: `SWE-bench/SWE-bench_Lite`, split `test`.
- Documentation: https://www.swebench.com/SWE-bench/guides/datasets/
- Tasks complete: `20`.
- Candidate pool validates with diagnostic gold allowed: `True`.
- Overall oracle pass@8: `0.5000`.
- Isolated patch applications: `140/160` applied; available test command `git diff --check`.

## Metrics

| method | pass@1 | conditional accuracy | oracle pass@8 | MRR | top-2 |
|---|---:|---:|---:|---:|---:|
| `oracle_pass_at_8` | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.5000 |
| `random_candidate` | 0.2500 | 0.5000 | 0.5000 | 0.2917 | 0.2500 |
| `first_candidate_order_baseline` | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.5000 |
| `best_generator_on_dev_baseline` | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.5000 |
| `trainable_shared_weight_latent_selector` | 0.2500 | 0.5000 | 0.5000 | 0.3333 | 0.2500 |
| `frozen_same_architecture_latent_selector` | 0.2500 | 0.5000 | 0.5000 | 0.3333 | 0.2500 |

## Controls

| control | pass@1 | conditional accuracy |
|---|---:|---:|
| `candidate_evidence_mismatch` | 0.2500 | 0.5000 |
| `candidate_only` | 0.2500 | 0.5000 |
| `candidate_order_shuffled_with_label_remap` | 0.2500 | 0.5000 |
| `context_only` | 0.2500 | 0.5000 |
| `cross_task_view_bundle_shuffle` | 0.2500 | 0.5000 |
| `evidence_only_no_candidates` | 0.0000 | 0.0000 |
| `hidden_states_shuffled_across_examples` | 0.2500 | 0.5000 |
| `issue_only` | 0.2500 | 0.5000 |
| `patch_only` | 0.2500 | 0.5000 |
| `physical_order_shuffled_roles_avenues_preserved` | 0.2500 | 0.5000 |
| `schema_template_only` | 0.2500 | 0.5000 |
| `view_masked_candidates_visible` | 0.2500 | 0.5000 |

## Audits

- Split leakage audit passes: `True`.
- Output leakage audit passes: `True`.
- Duplicate candidate patch hash audit passes: `True`.
- Order invariance passes: `True`.
- Gradient audits pass: `True`.

## Stage 6A-real Success

- Candidate pool validates: `True`.
- Oracle pass@8 between 0.25 and 0.80: `True`.
- At least 20 real tasks complete: `True`.
- Leakage/order/duplicate audits ran: `True`.
- Trainable/frozen/baselines execute without failure: `True`.

No final claim should be made from this Stage 6A-real smoke run.
