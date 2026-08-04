# Architecture Search

This report uses the semantic no-literal-cue benchmark. Architecture selection is based on dev metrics and dev controls; test metrics are logged after evaluation and are not used to choose candidates.

## Search Space
- **message_readout**: active_query, attention_pool, topk_attention, multi_query, last1/last2 layer token states, residual active+pooled
- **message_head**: linear, mlp, gated_mlp, residual_mlp, dims 32/64/128, dropout 0/0.05
- **coordinator**: candidate_query_cross_attention, bilinear_candidate, contrastive_candidate, global_then_candidate, two_round_message_passing
- **loss**: main CE only, very small private-cue aux, message variance anti-collapse, norm regularizer support

## Stage1 Candidate Table

| candidate | seed | valid | dev train | dev frozen | dev delta | test train | test frozen | test delta | text | raw | random | hidden shuffle | view masked | view shuffled | role shuffle | collapse cosine | reasons |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| active_no_aux_dim64 | 0 | no | 0.1875 | 0.1042 | 0.0833 | 0.1406 | 0.1719 | -0.0312 | 0.1250 | 0.1719 | 0.0625 | 0.1406 | 0.1406 | 0.1406 | 0.1406 | 1.0000 | active_message_readout collapsed; message_head_output collapsed |
| active_no_aux_dim32 | 0 | no | 0.1458 | 0.1250 | 0.0208 | 0.1562 | 0.1719 | -0.0156 | 0.1250 | 0.1719 | 0.1094 | 0.1562 | 0.1562 | 0.1562 | 0.1562 | 1.0000 | active_message_readout collapsed; message_head_output collapsed |
| active_no_aux_dim128 | 0 | no | 0.1458 | 0.1042 | 0.0417 | 0.1562 | 0.1719 | -0.0156 | 0.1250 | 0.1719 | 0.0625 | 0.1562 | 0.1562 | 0.1562 | 0.1562 | 1.0000 | active_message_readout collapsed; message_head_output collapsed |
| active_last2_layers | 0 | no | 0.1042 | 0.1250 | -0.0208 | 0.1719 | 0.1719 | 0.0000 | 0.1250 | 0.1719 | 0.0625 | 0.1719 | 0.1719 | 0.1719 | 0.1719 | 1.0000 | trainable <= frozen on dev; trainable <= text-only on dev; active_message_readout collapsed; message_head_output collapsed |
| active_multiquery4 | 0 | no | 0.1250 | 0.1875 | -0.0625 | 0.0781 | 0.0938 | -0.0156 | 0.1250 | 0.1719 | 0.1094 | 0.0781 | 0.0781 | 0.0781 | 0.0781 | 1.0000 | trainable <= frozen on dev; trainable <= text-only on dev; active_message_readout collapsed; message_head_output collapsed |
| attention_pool_dim64 | 0 | no | 0.1042 | 0.1250 | -0.0208 | 0.1719 | 0.1719 | 0.0000 | 0.1250 | 0.1719 | 0.0938 | 0.1719 | 0.1719 | 0.1719 | 0.1719 | 0.9985 | trainable <= frozen on dev; trainable <= text-only on dev; active_message_readout collapsed; message_head_output collapsed |
| topk_attention_dim64 | 0 | no | 0.1042 | 0.2083 | -0.1042 | 0.1719 | 0.1719 | 0.0000 | 0.1250 | 0.1719 | 0.1094 | 0.1719 | 0.1719 | 0.1719 | 0.1719 | 0.9861 | trainable <= frozen on dev; trainable <= text-only on dev; message_head_output collapsed |
| residual_active_dim64 | 0 | no | 0.1042 | 0.1458 | -0.0417 | 0.1719 | 0.1562 | 0.0156 | 0.1250 | 0.1719 | 0.1719 | 0.1719 | 0.1719 | 0.1719 | 0.1719 | 0.9876 | trainable <= frozen on dev; trainable <= text-only on dev; active_message_readout collapsed; message_head_output collapsed |
| active_no_head | 0 | no | 0.0833 | 0.1458 | -0.0625 | 0.1406 | 0.1562 | -0.0156 | 0.1250 | 0.1719 | 0.0625 | 0.1406 | 0.1406 | 0.1406 | 0.1406 | 1.0000 | trainable <= frozen on dev; trainable <= text-only on dev; active_message_readout collapsed |
| attention_pool_no_head | 0 | no | 0.1250 | 0.1667 | -0.0417 | 0.1719 | 0.1406 | 0.0312 | 0.1250 | 0.1719 | 0.1094 | 0.1719 | 0.1719 | 0.1562 | 0.1719 | 0.9964 | trainable <= frozen on dev; trainable <= text-only on dev; active_message_readout collapsed |
| topk_attention_no_head | 0 | yes | 0.1667 | 0.0833 | 0.0833 | 0.1406 | 0.1875 | -0.0469 | 0.1250 | 0.1719 | 0.1562 | 0.1562 | 0.1406 | 0.1406 | 0.0781 | 0.9389 | kept |
| residual_active_no_head | 0 | no | 0.1250 | 0.1875 | -0.0625 | 0.2031 | 0.1406 | 0.0625 | 0.1250 | 0.1719 | 0.1094 | 0.1562 | 0.1719 | 0.1875 | 0.1875 | 1.0000 | trainable <= frozen on dev; trainable <= text-only on dev; active_message_readout collapsed |
| linear_head_dim64 | 0 | no | 0.1667 | 0.2083 | -0.0417 | 0.1875 | 0.1719 | 0.0156 | 0.1250 | 0.1719 | 0.0938 | 0.1875 | 0.1875 | 0.1875 | 0.1875 | 1.0000 | trainable <= frozen on dev; active_message_readout collapsed; message_head_output collapsed |
| gated_head_dim64 | 0 | no | 0.2083 | 0.1667 | 0.0417 | 0.1719 | 0.1875 | -0.0156 | 0.1250 | 0.1719 | 0.1094 | 0.1719 | 0.1719 | 0.1719 | 0.1719 | 1.0000 | active_message_readout collapsed; message_head_output collapsed |
| residual_head_dim64 | 0 | no | 0.1250 | 0.1042 | 0.0208 | 0.1719 | 0.1719 | 0.0000 | 0.1250 | 0.1719 | 0.1406 | 0.1719 | 0.1719 | 0.1719 | 0.1719 | 1.0000 | trainable <= text-only on dev; active_message_readout collapsed; message_head_output collapsed |
| dropout005_head | 0 | no | 0.2083 | 0.1042 | 0.1042 | 0.1719 | 0.1250 | 0.0469 | 0.1250 | 0.1719 | 0.1719 | 0.1719 | 0.1719 | 0.1719 | 0.1719 | 1.0000 | active_message_readout collapsed; message_head_output collapsed |
| variance_reg | 0 | no | 0.1667 | 0.1667 | 0.0000 | 0.1406 | 0.1875 | -0.0469 | 0.1250 | 0.1719 | 0.1562 | 0.1406 | 0.1406 | 0.1406 | 0.1406 | 1.0000 | trainable <= frozen on dev; active_message_readout collapsed; message_head_output collapsed |
| tiny_aux005 | 0 | no | 0.2083 | 0.0833 | 0.1250 | 0.1719 | 0.1406 | 0.0312 | 0.1250 | 0.1719 | 0.0625 | 0.1719 | 0.1719 | 0.1719 | 0.1719 | 1.0000 | active_message_readout collapsed; message_head_output collapsed |
| aux005_no_head | 0 | no | 0.1458 | 0.1667 | -0.0208 | 0.1562 | 0.1406 | 0.0156 | 0.1250 | 0.1719 | 0.1406 | 0.1562 | 0.1562 | 0.1562 | 0.1562 | 1.0000 | trainable <= frozen on dev; active_message_readout collapsed |
| aux05_mlp | 0 | no | 0.1250 | 0.1250 | 0.0000 | 0.1719 | 0.1719 | 0.0000 | 0.1250 | 0.1719 | 0.1719 | 0.1719 | 0.1719 | 0.1719 | 0.1719 | 1.0000 | trainable <= frozen on dev; trainable <= text-only on dev; active_message_readout collapsed; message_head_output collapsed |
| aux05_no_head | 0 | no | 0.1042 | 0.1042 | 0.0000 | 0.1719 | 0.1719 | 0.0000 | 0.1250 | 0.1719 | 0.1719 | 0.1719 | 0.1719 | 0.1719 | 0.1719 | 1.0000 | trainable <= frozen on dev; trainable <= text-only on dev; active_message_readout collapsed |
| candidate_query_2layer | 0 | no | 0.1250 | 0.1667 | -0.0417 | 0.1719 | 0.1406 | 0.0312 | 0.1250 | 0.1719 | 0.0938 | 0.1719 | 0.1719 | 0.1719 | 0.1719 | 1.0000 | trainable <= frozen on dev; trainable <= text-only on dev; active_message_readout collapsed; message_head_output collapsed |
| bilinear_candidate | 0 | no | 0.0833 | 0.1042 | -0.0208 | 0.1406 | 0.1250 | 0.0156 | 0.1250 | 0.1719 | 0.1562 | 0.1406 | 0.1406 | 0.1406 | 0.1406 | 1.0000 | trainable <= frozen on dev; trainable <= text-only on dev; active_message_readout collapsed; message_head_output collapsed |
| contrastive_candidate | 0 | no | 0.1458 | 0.0625 | 0.0833 | 0.1562 | 0.1094 | 0.0469 | 0.1250 | 0.1719 | 0.1719 | 0.1562 | 0.1562 | 0.1562 | 0.1562 | 1.0000 | active_message_readout collapsed; message_head_output collapsed |
| global_then_candidate | 0 | no | 0.1667 | 0.1042 | 0.0625 | 0.1406 | 0.1250 | 0.0156 | 0.1250 | 0.1719 | 0.1719 | 0.1406 | 0.1406 | 0.1406 | 0.1406 | 1.0000 | active_message_readout collapsed; message_head_output collapsed |
| two_round_message_passing | 0 | no | 0.1042 | 0.0833 | 0.0208 | 0.1719 | 0.1406 | 0.0312 | 0.1250 | 0.1719 | 0.1719 | 0.1719 | 0.1719 | 0.1719 | 0.1719 | 1.0000 | trainable <= text-only on dev; active_message_readout collapsed; message_head_output collapsed |

Top variants by valid dev delta:
- `topk_attention_no_head`: dev delta 0.0833, test delta -0.0469

Full controls and diagnostics for kept rows:
| candidate | seed | explicit evidence oracle | evidence diag train/frozen | candidate-order baseline | physical-order shuffled | candidate-order shuffled | active variance | active norm | active cosine | private cue mean | combined-message probe | shared grad | shared delta | coordinator grad | coordinator delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| topk_attention_no_head | 0 | 1.0000 | 0.1719/0.1562 | 0.1250 | 0.1406 | 0.1406 | 0.0604 | 11.3693 | 0.9389 | 0.5938 | 0.1250 | 0.0634 | 1.1312 | 0.4524 | 0.4850 |

Per-family accuracy for kept rows:
- `topk_attention_no_head` seed 0: inventory_delta=0.1905, pagination_cursor=0.0952, permission_merge=0.1364

Per-role ablation for kept rows:
- `topk_attention_no_head` seed 0: base 0.1406; role 0: drop 0.0312, role 1: drop 0.0000, role 2: drop 0.0156, role 3: drop 0.0000

## Stage2 Candidate Table

| candidate | seed | valid | dev train | dev frozen | dev delta | test train | test frozen | test delta | text | raw | random | hidden shuffle | view masked | view shuffled | role shuffle | collapse cosine | reasons |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| topk_attention_no_head | 0 | yes | 0.8250 | 0.3563 | 0.4687 | 0.8047 | 0.3125 | 0.4922 | 0.1289 | 0.1328 | 0.1719 | 0.1445 | 0.0977 | 0.1367 | 0.2266 | 0.4958 | kept |
| topk_attention_no_head | 1 | yes | 0.9375 | 0.2562 | 0.6813 | 0.9141 | 0.3008 | 0.6133 | 0.1172 | 0.1172 | 0.0781 | 0.1328 | 0.1172 | 0.0938 | 0.2109 | 0.4688 | kept |
| topk_attention_no_head | 2 | yes | 0.9250 | 0.2500 | 0.6750 | 0.8750 | 0.3672 | 0.5078 | 0.1367 | 0.1250 | 0.1094 | 0.1250 | 0.0781 | 0.1250 | 0.2109 | 0.7190 | kept |

Top variants by valid dev delta:
- `topk_attention_no_head`: dev delta 0.6813, test delta 0.6133
- `topk_attention_no_head`: dev delta 0.6750, test delta 0.5078
- `topk_attention_no_head`: dev delta 0.4687, test delta 0.4922

Full controls and diagnostics for kept rows:
| candidate | seed | explicit evidence oracle | evidence diag train/frozen | candidate-order baseline | physical-order shuffled | candidate-order shuffled | active variance | active norm | active cosine | private cue mean | combined-message probe | shared grad | shared delta | coordinator grad | coordinator delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| topk_attention_no_head | 1 | 1.0000 | 0.4492/0.1094 | 0.1250 | 0.9141 | 0.9141 | 0.5410 | 14.0268 | 0.4688 | 0.9453 | 0.1367 | 1.3481 | 14.0068 | 1.4442 | 3.1140 |
| topk_attention_no_head | 2 | 1.0000 | 0.1914/0.0742 | 0.1250 | 0.8750 | 0.8711 | 0.2901 | 14.1225 | 0.7190 | 0.9189 | 0.1406 | 0.8910 | 12.0098 | 1.3873 | 3.1713 |
| topk_attention_no_head | 0 | 1.0000 | 0.5039/0.1250 | 0.1250 | 0.8047 | 0.7891 | 0.5072 | 13.9409 | 0.4958 | 0.9238 | 0.1367 | 1.2460 | 14.6119 | 1.9258 | 3.0456 |

Per-family accuracy for kept rows:
- `topk_attention_no_head` seed 1: inventory_delta=0.9176, pagination_cursor=0.8588, permission_merge=0.9651
- `topk_attention_no_head` seed 2: inventory_delta=0.9529, pagination_cursor=0.9176, permission_merge=0.7558
- `topk_attention_no_head` seed 0: inventory_delta=0.9059, pagination_cursor=0.8588, permission_merge=0.6512

Per-role ablation for kept rows:
- `topk_attention_no_head` seed 1: base 0.9141; role 0: drop 0.3633, role 1: drop 0.1562, role 2: drop 0.3750, role 3: drop 0.4297
- `topk_attention_no_head` seed 2: base 0.8750; role 0: drop 0.3516, role 1: drop 0.3203, role 2: drop 0.0703, role 3: drop 0.3633
- `topk_attention_no_head` seed 0: base 0.8047; role 0: drop 0.3047, role 1: drop 0.1094, role 2: drop 0.3242, role 3: drop 0.4414

## Robustness Results

Stage 1 is a fast screen. Stage 2 and Stage 3 rows appear here only after the corresponding run is executed. No hard-validation success claim is made from Stage 1 alone.
- Stage 2 `topk_attention_no_head`: mean trainable 0.8646, mean frozen 0.3268, mean delta 0.5378, std delta 0.0538, bootstrap 95% CI [0.4922, 0.6133], seeds 3.
  Per-seed deltas: seed 0=0.4922, seed 1=0.6133, seed 2=0.5078
- Stage 3 hard validation was not run. Required hard-validation budget remains: n_train >= 2048, n_dev >= 512, n_test >= 1024, seeds >= 10.

## Conservative Interpretation

The Stage 2 top-k no-head variant is a promising medium-validation result if present above, because it compares against an exact frozen same-architecture comparator and keeps the locked controls near chance. It is not a hard-validation success claim until Stage 3 is run.

No result should be interpreted from train accuracy. A candidate is rejected when trainable does not beat the exact frozen comparator, when text/raw baselines are not beaten, when locked controls are above threshold, when collapse statistics show near-zero variance with cosine near one, or when leakage checks fail. Do not claim open-ended agent collaboration or general LLM self-organization from this benchmark.
