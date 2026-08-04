# Stage 4.1 Physical-Order Invariance Audit

## Scope

- Selected final architecture audited: `candidate_token_direct_lr3e4_clip1`.
- Dataset/control policy unchanged: Stage 3.8 schema-aware balanced_categories_v3 with role schemas preserved.
- No full 10-seed validation was run.
- No success claim is made.

## Shortcut Diagnostics

- Status: `True`.
- Source: carried forward from the fixed Stage 4 final benchmark because Stage 4.1 did not change the dataset or controls.

## No-Training Checkpoint Audit

| seed | normal | prior failed physical | fixed builtin physical | all-perm mean | all-perm min | all-perm max | canonical mean | legacy-bug mean | integrity pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 1.0000 | 0.2959 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.3489 | `True` |
| 11 | 0.9238 | 0.1904 | 0.9238 | 0.9238 | 0.9238 | 0.9238 | 0.9238 | 0.2221 | `True` |
| 12 | 0.5215 | 0.2852 | 0.5215 | 0.5215 | 0.5215 | 0.5215 | 0.5215 | 0.3018 | `True` |
| 13 | 1.0000 | 0.2695 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.3128 | `True` |
| 14 | 1.0000 | 0.2285 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.2785 | `True` |
| 15 | 0.5010 | 0.1582 | 0.5010 | 0.5010 | 0.5010 | 0.5010 | 0.5010 | 0.1853 | `True` |
| 16 | 0.9990 | 0.2227 | 0.9990 | 0.9990 | 0.9990 | 0.9990 | 0.9990 | 0.2758 | `True` |
| 17 | 0.5186 | 0.2520 | 0.5186 | 0.5186 | 0.5186 | 0.5186 | 0.5186 | 0.2659 | `True` |
| 18 | 1.0000 | 0.3145 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.3293 | `True` |
| 19 | 0.4404 | 0.1689 | 0.4404 | 0.4404 | 0.4404 | 0.4404 | 0.4404 | 0.1809 | `True` |

## Checkpoint Summary

- Normal mean: `0.7904`
- Correct all-permutation mean: `0.7904`
- Canonicalized all-permutation mean: `0.7904`
- Legacy buggy permutation mean: `0.2701`
- Fixed tensor permutation matches normal within 0.02: `True`
- Canonical sorting matches normal within 0.02: `True`

## Implementation Trace

| component | uses physical order? | uses role id? | permutation-safe? | notes |
|---|---|---|---|---|
| dataset physical_order_shuffled_roles_preserved | `False` | `False` | `True` | Dataset examples are unchanged; the control is a runtime tensor/role-slot perturbation. |
| SharedClonedAgentSystem.collect_clone_representations | `True` | `True` | `True` | Prompts are collected by slot; explicit role_ids can be supplied and are carried separately. |
| legacy physical-order control path | `True` | `True` | `False` | Before Stage 4.1 it permuted clone_activations and role_ids but not token_states/token_mask, which candidate-token direct consumes. |
| Stage 4.1 repaired physical-order control path | `True` | `True` | `True` | All role-axis tensors are gathered together: messages, pooled/msg/evidence, token_states, and token_mask. |
| CandidateTokenCrossAttentionCoordinator | `True` | `True` | `True` | It flattens role-token groups but has no learned slot embedding or role-slot positional encoding; attention is order invariant when role ids and token tensors stay aligned. |
| role embeddings | `False` | `True` | `True` | Embedding lookup depends on role_id, not physical slot. |
| canonicalize_role_order option | `False` | `True` | `True` | Sorts all role-axis tensors by role_id immediately before coordinator/readout. |

## Cheap Validation

| seed | dev trainable | dev frozen | dev delta | dev physical | dev cand-order | dev candidate-only | dev schema-only | dev value-shuffle | dev mismatch | test trainable | test frozen |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.0000 | 0.1797 | 0.8203 | 1.0000 | 1.0000 | 0.1797 | 0.1523 | 0.0859 | 0.1328 | 1.0000 | 0.1211 |
| 1 | 0.8555 | 0.1016 | 0.7539 | 0.8555 | 0.8555 | 0.0352 | 0.1172 | 0.0977 | 0.1211 | 0.8672 | 0.1367 |
| 2 | 0.5195 | 0.1602 | 0.3594 | 0.5195 | 0.5234 | 0.1875 | 0.1797 | 0.1641 | 0.1367 | 0.4922 | 0.1445 |

## Cheap Validation Summary

- Trainable beats frozen on dev seeds: `3/3`
- Mean dev trainable-frozen delta: `0.6445`
- Mean dev physical-order delta from trainable: `0.0000`
- Mean dev candidate-order delta from trainable: `0.0013`
- Controls near chance: `True`
- Cheap validation passed: `True`

## Conservative Interpretation

- Failed final invariance gate cause: `control/tensor permutation bug`.
- Role tensors/ids/masks are permuted correctly after repair: `True`.
- Canonical role sorting fixes checkpoint behavior: `True`.
- Full final validation justified by this Stage 4.1 cheap gate: `True`.
- Stage 4.1 does not claim benchmark success; it only audits and repairs the physical-order invariance failure.
