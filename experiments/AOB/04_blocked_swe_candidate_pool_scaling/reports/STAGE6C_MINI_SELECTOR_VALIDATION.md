# Stage 6C-mini Selector Validation

## Scope

- Diagnostic selector-only validation on a tiny officially labeled complete-label subset.
- Candidate patches are fixed generated candidates from Stage 6B.1b/6B.3.
- The existing Stage 6 candidate-token-direct architecture is used unchanged.
- No final SWE-bench improvement, publishable result, or end-to-end patch generation claim is made.

## Dataset

- Complete tasks: `20`.
- Oracle-positive tasks: `7`.
- Oracle-empty tasks: `13`.
- Folds completed: `5`.

## Mean Metrics

| method | pass@1 | conditional accuracy | oracle pass@8 | efficiency | MRR | top-2 |
|---|---:|---:|---:|---:|---:|---:|
| `best_generator_on_dev_baseline` | 0.1567 | 0.5000 | 0.3433 | 0.5000 | 0.1907 | 0.1567 |
| `embedding_reranker` | 0.2367 | 0.7000 | 0.3433 | 0.7000 | 0.2800 | 0.3033 |
| `first_candidate_order_baseline` | 0.1567 | 0.5000 | 0.3433 | 0.5000 | 0.1907 | 0.1567 |
| `frozen_same_architecture_latent_selector` | 0.1067 | 0.3000 | 0.3433 | 0.3000 | 0.1850 | 0.2233 |
| `llm_judge_baseline` | 0.1567 | 0.5000 | 0.3433 | 0.5000 | 0.1907 | 0.1567 |
| `oracle_pass_at_8` | 0.3433 | 1.0000 | 0.3433 | 1.0000 | 0.3433 | 0.3433 |
| `random_candidate` | 0.1167 | 0.4000 | 0.3433 | 0.4000 | 0.1806 | 0.1567 |
| `randomized_labels_trainable_latent_selector` | 0.0667 | 0.2000 | 0.3433 | 0.2000 | 0.1078 | 0.0667 |
| `raw_latent_selector` | 0.1067 | 0.3000 | 0.3433 | 0.3000 | 0.1567 | 0.1067 |
| `single_agent_full_context_reviewer` | 0.1867 | 0.5000 | 0.3433 | 0.5000 | 0.2517 | 0.3033 |
| `static_patch_feature_reranker` | 0.0667 | 0.2000 | 0.3433 | 0.2000 | 0.1369 | 0.1167 |
| `text_only_multi_agent_reviewer` | 0.1867 | 0.5000 | 0.3433 | 0.5000 | 0.2517 | 0.3033 |
| `trainable_shared_weight_latent_selector` | 0.1867 | 0.5000 | 0.3433 | 0.5000 | 0.2471 | 0.2933 |
| `visible_test_heuristic` | 0.1567 | 0.5000 | 0.3433 | 0.5000 | 0.1907 | 0.1567 |

## Controls

| control | pass@1 | conditional accuracy | efficiency |
|---|---:|---:|---:|
| `candidate_evidence_mismatch` | 0.1867 | 0.5000 | 0.5000 |
| `candidate_only` | 0.1867 | 0.5000 | 0.5000 |
| `candidate_order_shuffled_with_label_remap` | 0.1867 | 0.5000 | 0.5000 |
| `context_only` | 0.1067 | 0.3000 | 0.3000 |
| `cross_task_view_bundle_shuffle` | 0.1867 | 0.5000 | 0.5000 |
| `evidence_only_no_candidates` | 0.1067 | 0.3000 | 0.3000 |
| `hidden_states_shuffled_across_examples` | 0.1867 | 0.5000 | 0.5000 |
| `issue_only` | 0.1067 | 0.3000 | 0.3000 |
| `patch_only` | 0.1867 | 0.5000 | 0.5000 |
| `physical_order_shuffled_roles_avenues_preserved` | 0.1867 | 0.5000 | 0.5000 |
| `randomized_labels` | 0.0667 | 0.0000 | 0.0000 |
| `schema_template_only` | 0.1967 | 0.6000 | 0.6000 |
| `view_masked_candidates_visible` | 0.1867 | 0.5000 | 0.5000 |

## Paired Test

- Comparison: `trainable_shared_weight_latent_selector_vs_first_candidate_order_baseline`.
- Mean delta: `0.0500`.
- Bootstrap 95% CI: `[-0.1, 0.2]`.
- Paired permutation p-value: `1.0`.

## Diagnostic Gates

| gate | pass |
|---|---:|
| `trainable_latent_executes_on_all_folds` | `True` |
| `trainable_beats_frozen_mean_pass_at_1` | `True` |
| `trainable_beats_best_simple_non_oracle_baseline_mean_pass_at_1` | `True` |
| `trainable_beats_random_mean_pass_at_1` | `True` |
| `randomized_labels_collapse_toward_chance` | `True` |
| `candidate_evidence_mismatch_degrades_substantially` | `False` |
| `candidate_order_invariance_passes` | `True` |
| `physical_order_invariance_passes` | `True` |
| `no_leakage_audits_fail` | `True` |
| `trainable_gradient_audits_pass` | `True` |
| `frozen_gradient_audits_pass` | `True` |
| `shared_identity_and_no_detach_audits_pass` | `True` |

## Diagnosis

- too few oracle-positive train/dev/test examples for stable selector validation.
- complete-label pool is tiny.
- generator/source skew is high.
- candidate/view construction may let patch-only or candidate-only views capture most signal.
- candidate/evidence mismatch did not degrade enough, suggesting weak issue-context use.

## Conclusion

Stage 6C-mini did not show reliable selector signal; larger official labeled candidate pools or stronger candidate/view construction are needed.
