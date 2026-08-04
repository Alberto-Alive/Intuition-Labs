# Stage 6D.2 Pool Scaling

## Scope

- Selector training was not executed.
- Selector architecture was not changed.
- Gold/reference patches, raw-preserving expansion, and fabricated candidates were not used.
- Official labels and raw harness logs are not selector-visible.
- No final SWE-bench improvement claim is made.

## Pool Scale

- Pool all complete K=8 tasks: `20`.
- Pool all official labels: `160`.
- Pool all oracle-positive tasks: `7`.
- Pool all oracle pass@8: `0.3500`.
- Pool strict complete K=8 tasks: `18`.
- Pool strict official labels: `144`.
- Pool strict oracle-positive tasks: `5`.
- Pool strict oracle pass@8: `0.2778`.

## Source Coverage

- Candidate source files scanned: `3`.
- Raw records loaded: `213`.
- Unique generated-source instance IDs: `31`.
- Tasks with K=8 unique generated candidates in sources: `25`.
- Extra K=8 generated tasks without complete official labels: `5`.
- New official harness runs executed: `False`.
- Compute limit documented: `True`.

## Pool Baselines

| Pool | First | Random | Best Generator | Embedding | Patch Only | Issue+Context+Patch | Mismatch |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pool_all` | `0.1500` | `0.0500` | `0.1500` | `0.2500` | `0.2500` | `0.2500` | `0.2000` |
| `pool_strict` | `0.1111` | `0.0556` | `0.1111` | `0.1667` | `0.1667` | `0.1667` | `0.1111` |

## Source Skew

- Pool all unique source agents: `2`.
- Pool all high generator skew: `True`.
- Pool strict unique source agents: `2`.
- Pool strict high generator skew: `True`.

## Evidence Stress

- Pool all evidence diagnostic passes: `True`.
- Pool all mismatch degradation: `0.0500`.
- Pool strict evidence diagnostic passes: `True`.
- Pool strict issue+context+patch minus patch-only: `0.0000`.
- Pool strict mismatch degradation: `0.0556`.
- Pool strict cross-task shuffle degradation: `0.0556`.
- Pool strict edited-file removal degradation: `0.0000`.

## Gates

| Gate | Pass |
|---|---:|
| `pool_strict_has_at_least_50_complete_k8_tasks_or_compute_limit_documented` | `True` |
| `pool_strict_has_at_least_15_oracle_positive_tasks` | `False` |
| `oracle_pass_at_8_between_0_20_and_0_80` | `True` |
| `no_exact_reference_hash_hits` | `True` |
| `near_reference_tasks_quarantined` | `True` |
| `no_raw_preserving_expansion` | `True` |
| `no_fabricated_candidates` | `True` |
| `output_leakage_audit_passes` | `True` |
| `source_generator_leakage_audit_passes` | `True` |
| `duplicate_patch_hash_audit_passes` | `True` |
| `evidence_use_diagnostic_passes` | `True` |
| `no_selector_training_executed` | `True` |
| `no_final_model_claim_made` | `True` |

## Blockers

- official harness throughput / complete official labels remain below the 50-task minimum.
- insufficient public generated K=8 patch supply available locally.
- insufficient oracle-positive tasks in pool_strict.
- generator/source skew remains high.
- patch-only shortcut dominance remains.

## Decision

Do not train the selector; Stage 6D.2 gates did not pass.
