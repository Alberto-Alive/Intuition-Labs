# Stage 6D Pool Scale and View Hardening

## Scope

- Selector training was not executed.
- Candidate patches were not altered.
- Gold/reference patches, raw-preserving expansion, and fabricated candidates were not used.
- Official labels and raw harness logs are excluded from selector-visible views.
- No final SWE-bench improvement claim is made.

## Pool Status

- Complete K=8 officially labeled tasks: `20`.
- Official candidate labels: `160`.
- Oracle-positive tasks: `7`.
- Oracle-empty tasks: `13`.
- Oracle pass@8: `0.3500`.
- Scale compute limit documented: `True`.

## Baselines

- First-candidate pass@1: `0.1500`.
- Random pass@1: `0.1000`.
- Best-generator-on-dev pass@1: `0.1500`.
- Issue+context+patch reranker pass@1: `0.2500`.
- Patch-only reranker pass@1: `0.2500`.
- Candidate/evidence mismatch pass@1: `0.2000`.

## Audit Notes

- Exact reference hash hits: `0`.
- Near-reference similarity hits: `2`.
- Output leakage audit passes: `True`.
- Source/generator leakage audit passes: `True`.
- Duplicate patch hash audit passes: `True`.

## View Diagnostics

- Issue+context minus patch-only: `0.0000`.
- Issue+context minus mismatch: `0.0500`.
- Issue/context signal detected: `True`.

## Gates

| Gate | Pass |
|---|---:|
| `at_least_100_complete_k8_officially_labeled_tasks_or_compute_limit_documented` | `True` |
| `at_least_30_oracle_positive_tasks_total` | `False` |
| `oracle_pass_at_8_between_0_20_and_0_80` | `True` |
| `all_leakage_reference_order_audits_pass` | `False` |
| `source_generator_skew_reported` | `True` |
| `evidence_views_constructed_for_all_examples` | `True` |
| `view_diagnostics_show_issue_context_evidence_signal` | `True` |
| `no_selector_final_claim_made` | `True` |

## Blockers

- official harness throughput / available complete-label pool is below the 100-task target.
- insufficient oracle-positive tasks for Stage 6E train/dev/test targets.
- generator/source skew remains high.
- patch-only shortcut dominance remains a risk.
- near-reference similarity audit flagged 2 generated candidates at threshold 0.95.

## Conclusion

Stage 6D gates did not pass; larger official labeled candidate pools and/or stronger evidence retrieval are needed before Stage 6E.
