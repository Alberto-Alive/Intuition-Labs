# Stage 3.6 Latent Learnability Bridge

## Scope

- Dataset mode: `real_import_restore_candidate_balanced_34b` / `balanced_categories_v3`.
- Locked architecture: `topk_attention_no_head`.
- Benchmark changes: `none`.
- Full 10-seed validation: not run.

## Candidate Pair Compatibility MLP Audit

- Uses only model-facing inputs: `True`
- Candidate-only diagnostic accuracy: `0.1250`
- Candidate-metadata-only diagnostic accuracy: `0.1250`
- View-masked+candidates diagnostic accuracy: `0.1250`

| field name | source | model-facing? | candidate-only predictive? | allowed? |
|---|---|---:|---:|---:|
| view_bits[4] | model-facing private view text parsed for balanced category phrases | `True` | `False` | `True` |
| candidate_bits[4] | model-facing candidate.attributes balanced_categories_v3 | `True` | `False` | `True` |
| equality_bits[4] | derived comparison of model-facing view_bits and candidate_bits | `True` | `False` | `True` |
| mean_equality | derived aggregate of equality_bits | `True` | `False` | `True` |
| valid_parse_flag | derived from whether model-facing category phrases are present | `True` | `False` | `True` |

## Leakage Gates

- Candidate-only: `0.1250`
- Candidate-metadata-only: `0.1250`
- View-masked + candidates-visible: `0.1250`
- Lexical overlap: `0.1250`
- Static frequency: `0.1250`
- All-role oracle: `1.0000`

## 64-Example Overfit Tests

| method | train | dev | test | max train | epoch >=0.50 | epoch >=0.80 | epoch >=0.95 | grad norm | param delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| active_topk_readout_without_cloned_agent_split | 1.0000 | 0.4531 | 0.4219 | 1.0000 | 3 | 8 | 11 | 0.2202 | 13.1433 |
| candidate_pair_compatibility_mlp | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1 | 1 | 1 | 0.0786 | 7.8972 |
| direct_candidate_query_structured_embeddings | 1.0000 | 0.9531 | 0.9062 | 1.0000 | 2 | 4 | 5 | 0.3153 | 5.5144 |
| frozen_locked_latent | 1.0000 | 0.4688 | 0.3281 | 1.0000 | 12 | 22 | 54 | 2.7378 | 10.1006 |
| single_agent_full_context_tiny_transformer | 1.0000 | 0.1719 | 0.2812 | 1.0000 | 12 | 30 | 46 |  |  |
| trainable_locked_latent | 1.0000 | 0.8906 | 0.9375 | 1.0000 | 6 | 21 | 29 | 0.6222 | 13.5375 |

## Message And Readout Diagnostics

- Probe mean train accuracy: `1.0000`
- Probe mean dev accuracy: `1.0000`
- Probe mean test accuracy: `1.0000`
- Active/readout message variance: `0.229084`
- Active/readout mean pairwise cosine: `0.7711`

## Bottleneck Ablations

- Bottleneck ablations were not run because the trainable locked latent model reached the 64-example overfit target.
- Inferred bottleneck: `none`

## Tiny Smoke

- Smoke passed: `False`
- Trainable: `0.4609`
- Frozen: `0.0977`
- Text-only: `0.1250`
- Raw latent: `0.1250`
- Candidate-only: `0.0547`
- View-masked: `0.0664`
- Role-label-shuffled: `0.2539`
- Oracle: `1.0000`

## Summary

- Candidate pair compatibility MLP truly model-facing: `True`
- Locked latent can overfit 64 examples: `True`
- Bottleneck ablations ran: `False`
- Any tiny ablation fixes overfitting: `False`
- Tiny smoke ran: `True`
- Full validation justified: `False`
