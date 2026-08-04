# Stage 4 Schema-Aware Architecture Search

## Scope

- Fixed benchmark: Stage 3.8 schema-aware `balanced_categories_v3`.
- Dataset/control changes: none.
- Role-specific schemas preserved as legitimate evidence.
- `role_embedding_shuffle` remains diagnostic-only and non-gating.
- Selection uses dev metrics and dev controls only; test is smoke sanity only.
- Full 10-seed final validation: not run.

## Variant Plan

| variant | family | lr | epochs | clip | description |
|---|---|---:|---:|---:|---|
| candidate_token_direct_lr1e3_clip1 | candidate_conditioned_readout | 0.001 | 50 | 1.00 | Candidate queries attend directly over role token states; pooled clone path only feeds compatibility with existing system. |
| candidate_token_direct_lr3e4_clip1 | candidate_conditioned_readout_training | 0.0003 | 50 | 1.00 | Candidate-token direct readout with lower LR for stability. |
| topk_k4_last2_lr1e3_clip1 | layer_token_readout | 0.001 | 50 | 1.00 | Top-k no-head readout with k=4 over the last two layers. |
| multi_query_topk4_lr1e3_clip1 | multi_query_topk_readout | 0.001 | 50 | 1.00 | Four learned queries, per-query top-k=4, no message head. |
| attention_pool_lr1e3_clip1 | layer_token_readout | 0.001 | 50 | 1.00 | Soft token attention over all tokens instead of hard top-k. |
| bilinear_topk_lr1e3_clip1 | coordinator_variant | 0.001 | 50 | 1.00 | Bilinear candidate-message scoring with top-k no-head readout. |
| locked_topk_lr3e4_clip1 | training_variant | 0.0003 | 50 | 1.00 | Locked top-k no-head architecture with lower LR and gradient clipping. |
| locked_topk_lr3e3_clip1 | training_variant | 0.003 | 50 | 1.00 | Locked top-k no-head architecture with higher LR and gradient clipping. |

## Cheap Screen

| rank | variant | seeds | wins | trainable | frozen | delta | controls | pass |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | candidate_token_direct_lr3e4_clip1 | 3 | 3 | 0.8763 | 0.1276 | 0.7487 | `True` | `True` |
| 2 | bilinear_topk_lr1e3_clip1 | 3 | 3 | 0.9987 | 0.3724 | 0.6263 | `True` | `True` |
| 3 | candidate_token_direct_lr1e3_clip1 | 3 | 3 | 0.7083 | 0.1576 | 0.5508 | `True` | `True` |
| 4 | multi_query_topk4_lr1e3_clip1 | 3 | 3 | 0.4688 | 0.1458 | 0.3229 | `True` | `True` |
| 5 | attention_pool_lr1e3_clip1 | 3 | 3 | 0.8438 | 0.5599 | 0.2839 | `True` | `True` |
| 6 | topk_k4_last2_lr1e3_clip1 | 3 | 2 | 0.8294 | 0.5977 | 0.2318 | `True` | `True` |
| 7 | locked_topk_lr3e4_clip1 | 3 | 2 | 0.8385 | 0.6992 | 0.1393 | `True` | `False` |
| 8 | locked_topk_lr3e3_clip1 | 3 | 2 | 0.8346 | 0.7370 | 0.0977 | `True` | `False` |

## Cheap Per-Seed Rows

| variant | seed | dev train | dev frozen | dev delta | dev cand-only | dev schema | dev masked | dev value-shuffle | dev mismatch | test train | test frozen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate_token_direct_lr1e3_clip1 | 0 | 0.6289 | 0.1992 | 0.4297 | 0.1914 | 0.0781 | 0.1172 | 0.1562 | 0.1094 | 0.5859 | 0.1680 |
| candidate_token_direct_lr1e3_clip1 | 1 | 1.0000 | 0.0977 | 0.9023 | 0.0820 | 0.0000 | 0.0156 | 0.1602 | 0.1484 | 1.0000 | 0.1211 |
| candidate_token_direct_lr1e3_clip1 | 2 | 0.4961 | 0.1758 | 0.3203 | 0.1211 | 0.1172 | 0.0664 | 0.1133 | 0.1055 | 0.3242 | 0.1445 |
| candidate_token_direct_lr3e4_clip1 | 0 | 1.0000 | 0.1680 | 0.8320 | 0.1836 | 0.1992 | 0.1875 | 0.1250 | 0.1289 | 1.0000 | 0.0664 |
| candidate_token_direct_lr3e4_clip1 | 1 | 0.6289 | 0.0547 | 0.5742 | 0.0938 | 0.1016 | 0.1211 | 0.1406 | 0.1367 | 0.6641 | 0.1094 |
| candidate_token_direct_lr3e4_clip1 | 2 | 1.0000 | 0.1602 | 0.8398 | 0.1055 | 0.0859 | 0.0898 | 0.1328 | 0.1328 | 1.0000 | 0.0898 |
| topk_k4_last2_lr1e3_clip1 | 0 | 1.0000 | 0.6758 | 0.3242 | 0.1250 | 0.1914 | 0.1758 | 0.1484 | 0.1289 | 1.0000 | 0.6602 |
| topk_k4_last2_lr1e3_clip1 | 1 | 1.0000 | 0.5352 | 0.4648 | 0.0352 | 0.0664 | 0.0430 | 0.1484 | 0.1406 | 1.0000 | 0.3789 |
| topk_k4_last2_lr1e3_clip1 | 2 | 0.4883 | 0.5820 | -0.0938 | 0.1367 | 0.1367 | 0.1211 | 0.1289 | 0.1172 | 0.3672 | 0.4883 |
| multi_query_topk4_lr1e3_clip1 | 0 | 0.5547 | 0.2109 | 0.3438 | 0.0469 | 0.1094 | 0.1602 | 0.1641 | 0.1055 | 0.5469 | 0.1719 |
| multi_query_topk4_lr1e3_clip1 | 1 | 0.3828 | 0.0703 | 0.3125 | 0.1328 | 0.1484 | 0.1445 | 0.0977 | 0.1641 | 0.5078 | 0.1406 |
| multi_query_topk4_lr1e3_clip1 | 2 | 0.4688 | 0.1562 | 0.3125 | 0.0977 | 0.1250 | 0.1250 | 0.1289 | 0.1172 | 0.4297 | 0.1484 |
| attention_pool_lr1e3_clip1 | 0 | 1.0000 | 0.2070 | 0.7930 | 0.0430 | 0.0469 | 0.0508 | 0.1328 | 0.1328 | 1.0000 | 0.0781 |
| attention_pool_lr1e3_clip1 | 1 | 1.0000 | 0.9844 | 0.0156 | 0.0469 | 0.1094 | 0.0664 | 0.1680 | 0.1367 | 1.0000 | 0.9883 |
| attention_pool_lr1e3_clip1 | 2 | 0.5312 | 0.4883 | 0.0430 | 0.1172 | 0.1211 | 0.1211 | 0.1445 | 0.1406 | 0.4844 | 0.4336 |
| bilinear_topk_lr1e3_clip1 | 0 | 1.0000 | 0.6562 | 0.3438 | 0.0586 | 0.1797 | 0.1484 | 0.1484 | 0.1523 | 0.9922 | 0.7500 |
| bilinear_topk_lr1e3_clip1 | 1 | 1.0000 | 0.1680 | 0.8320 | 0.1562 | 0.0977 | 0.1250 | 0.1172 | 0.1484 | 1.0000 | 0.2383 |
| bilinear_topk_lr1e3_clip1 | 2 | 0.9961 | 0.2930 | 0.7031 | 0.1641 | 0.1250 | 0.1680 | 0.1406 | 0.1172 | 0.9922 | 0.1445 |
| locked_topk_lr3e4_clip1 | 0 | 1.0000 | 0.6172 | 0.3828 | 0.1445 | 0.1953 | 0.1406 | 0.1484 | 0.1406 | 1.0000 | 0.5859 |
| locked_topk_lr3e4_clip1 | 1 | 0.5156 | 0.4805 | 0.0352 | 0.0781 | 0.1055 | 0.1016 | 0.1172 | 0.1406 | 0.5547 | 0.4648 |
| locked_topk_lr3e4_clip1 | 2 | 1.0000 | 1.0000 | 0.0000 | 0.0469 | 0.1680 | 0.1523 | 0.1367 | 0.1328 | 1.0000 | 1.0000 |
| locked_topk_lr3e3_clip1 | 0 | 1.0000 | 0.6016 | 0.3984 | 0.0664 | 0.0117 | 0.0117 | 0.1328 | 0.1133 | 1.0000 | 0.5898 |
| locked_topk_lr3e3_clip1 | 1 | 1.0000 | 0.6094 | 0.3906 | 0.1719 | 0.0469 | 0.0469 | 0.1562 | 0.1680 | 1.0000 | 0.3789 |
| locked_topk_lr3e3_clip1 | 2 | 0.5039 | 1.0000 | -0.4961 | 0.0742 | 0.1211 | 0.1211 | 0.1406 | 0.1211 | 0.2891 | 0.9883 |

## Medium Validation

| variant | seeds | wins | trainable | frozen | delta | controls | pass |
|---|---:|---:|---:|---:|---:|---|---|
| candidate_token_direct_lr3e4_clip1 | 5 | 5 | 0.9727 | 0.1293 | 0.8434 | `True` | `True` |
| candidate_token_direct_lr1e3_clip1 | 5 | 5 | 0.6891 | 0.1598 | 0.5293 | `True` | `True` |
| bilinear_topk_lr1e3_clip1 | 5 | 4 | 0.8949 | 0.3984 | 0.4965 | `True` | `True` |

## Decision

- Cheap-screen passed variants: `['candidate_token_direct_lr3e4_clip1', 'bilinear_topk_lr1e3_clip1', 'candidate_token_direct_lr1e3_clip1', 'multi_query_topk4_lr1e3_clip1', 'attention_pool_lr1e3_clip1', 'topk_k4_last2_lr1e3_clip1']`
- Medium validation ran: `True`
- Medium passed variants: `['candidate_token_direct_lr3e4_clip1', 'candidate_token_direct_lr1e3_clip1', 'bilinear_topk_lr1e3_clip1']`
- Final validation justified: `True`
- No success claim is made from this search screen.
