# E5.1 Faithful State-Attention Findings

## What Changed

- Added direct fixed-state K/V attention variants: current-step tokens can attend over the rolling state matrix as additional K/V memory.
- Added an option to disable pooled `state.mean(...)` shortcuts in attention conditioning and the answer head.
- Added optional slot-specific state updates using slot-to-current cross-attention.
- Added optional stricter routed writes and soft top-k routing variants.

## CUDA Screens

- Phase 1: `R2_slot_dropout_retention`, `G1_state_kv_meanless`, `G2_state_kv_cross_update`, `G3_state_kv_strict_topk`, `G4_state_kv_rank_pressure`.
- Phase 2: `G5_state_kv_mild_rank`, `G6_state_kv_soft_topk`.
- Phase 3: `G1_state_kv_meanless`, `G7_g1_soft_repair`, `H1_learned_geometry_keys`, `H2_geometry_soft_topk`, `H3_geometry_cross_update`.
- Phase 4: `H2_geometry_soft_topk`, `H4_geometry_no_rank`, `H5_geometry_static_heavy`, `H6_geometry_dynamic_heavy`, `H7_geometry_topk4`, `H8_geometry_topk16`, `H9_geometry_temp_anneal`.
- Phase 5: `H5_geometry_static_heavy`, `H10_possibility_geometry_dense`, `H11_possibility_static_heavy`, `H12_possibility_high_capacity`, `H13_possibility_sparse_write`.
- Hardware: NVIDIA GeForce RTX 5070 Ti.
- Budget: 160 train steps, seeds 0/1, eval lengths 32/64/128/256/512, 96 eval examples.

## Main Result

| variant | C | N64 | N128 | N256 | N512 | rank | cosine | route shuffle | slot perm | slot drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R2_slot_dropout_retention | 64 | 0.9115 | 0.7865 | 0.4635 | 0.2344 | 8.08 | 0.994 | -0.0156 | 0.0000 | -0.0104 |
| G1_state_kv_meanless | 128 | 0.9688 | 0.9583 | 0.8438 | 0.7604 | 2.37 | 0.958 | 0.0000 | 0.0000 | 0.2292 |
| G2_state_kv_cross_update | 64 | 0.9375 | 0.7656 | 0.5990 | 0.5052 | 3.94 | 0.589 | 0.0365 | 0.0052 | 0.3854 |
| G3_state_kv_strict_topk | 0 | 0.7396 | 0.7500 | 0.7031 | 0.5573 | 7.84 | 0.201 | 0.2135 | 0.0052 | 0.0365 |
| G4_state_kv_rank_pressure | 64 | 0.9115 | 0.5573 | 0.4479 | 0.2604 | 10.83 | 0.395 | 0.0781 | 0.0052 | 0.2760 |
| G5_state_kv_mild_rank | 64 | 0.9427 | 0.7500 | 0.3750 | 0.3177 | 7.77 | 0.942 | 0.0000 | 0.0000 | 0.0885 |
| G6_state_kv_soft_topk | 64 | 0.9740 | 0.8906 | 0.7031 | 0.5625 | 4.52 | 0.640 | 0.0104 | 0.0000 | 0.1146 |
| G7_g1_soft_repair | 64 | 0.9844 | 0.8438 | 0.5521 | 0.3281 | 4.15 | 0.958 | 0.0104 | 0.0000 | 0.2552 |
| H1_learned_geometry_keys | 64 | 0.9063 | 0.8177 | 0.6563 | 0.5781 | 2.45 | 0.977 | -0.0104 | 0.0000 | 0.3698 |
| H2_geometry_soft_topk | 64 | 0.9531 | 0.8333 | 0.7604 | 0.6719 | 5.09 | 0.807 | 0.0000 | 0.0000 | 0.0260 |
| H3_geometry_cross_update | 64 | 0.9167 | 0.7240 | 0.5469 | 0.5417 | 5.98 | 0.665 | 0.0000 | 0.0000 | 0.3177 |
| H4_geometry_no_rank | 64 | 0.9635 | 0.8177 | 0.6198 | 0.5781 | 3.06 | 0.827 | -0.0052 | 0.0000 | 0.3177 |
| H5_geometry_static_heavy | 128 | 0.9635 | 0.9375 | 0.8646 | 0.7344 | 2.55 | 0.785 | 0.0052 | 0.0000 | 0.0312 |
| H6_geometry_dynamic_heavy | 64 | 0.9583 | 0.8490 | 0.7396 | 0.5938 | 2.73 | 0.801 | -0.0052 | 0.0000 | 0.5156 |
| H7_geometry_topk4 | 64 | 0.9167 | 0.8542 | 0.7188 | 0.5729 | 3.38 | 0.819 | -0.0156 | -0.0052 | 0.2240 |
| H8_geometry_topk16 | 64 | 0.9531 | 0.5521 | 0.3646 | 0.3385 | 2.56 | 0.903 | 0.1406 | 0.0000 | 0.3958 |
| H9_geometry_temp_anneal | 64 | 0.9740 | 0.8385 | 0.6302 | 0.5729 | 2.47 | 0.844 | 0.0000 | 0.0000 | 0.0208 |
| H10_possibility_geometry_dense | 128 | 0.9688 | 0.9688 | 0.8594 | 0.6510 | 3.72 | 0.828 | -0.0052 | 0.0000 | 0.1146 |
| H11_possibility_static_heavy | 64 | 0.9635 | 0.8021 | 0.7135 | 0.5208 | 2.66 | 0.787 | 0.0521 | 0.0000 | 0.3646 |
| H12_possibility_high_capacity | 32 | 0.9115 | 0.7604 | 0.6198 | 0.5312 | 2.58 | 0.828 | 0.0104 | -0.0052 | 0.2292 |
| H13_possibility_sparse_write | 128 | 0.9688 | 0.9167 | 0.7604 | 0.6146 | 2.17 | 0.840 | 0.0000 | 0.0000 | 0.4115 |

## Interpretation

Direct state-KV attention translated the original idea more faithfully than the previous pooled-state conditioning. `G1_state_kv_meanless` was the clearest accuracy win: it reached C128 and mean N512 accuracy 0.7604, far above the R2 control at 0.2344.

The tradeoff is still unresolved. The best accuracy variant still did not learn meaningful slot identity: route shuffle and slot permutation did not hurt, and rank stayed low. Variants that improved matrix organization (`G3`, `G4`) lost accuracy and became seed-sensitive. `G6` improved slot geometry versus G1 while retaining strong N64/N128 accuracy, but it did not match G1's N512 extrapolation.

The learned-geometry idea was partially supported. `H2_geometry_soft_topk` added learned static slot-key geometry plus dynamic rolling-state content and reached N512 mean 0.6719 with better rank/cosine than G1. It did not beat G1's N512 mean, but it was less collapsed and had a stronger N512 minimum than G1 in this two-seed screen. `H1` and `H3` were weaker, and `G7` showed that simply adding soft rank/write pressure to G1 hurts long-context accuracy.

The H2-local screen found a better setting: `H5_geometry_static_heavy`. Increasing the learned static slot-key contribution and reducing dynamic key contribution reached C128, N256 0.8646, and N512 0.7344. It nearly matched G1's N512 mean while improving the N512 minimum and reducing slot cosine from G1's 0.958 to 0.785. Removing rank pressure alone (`H4`) was not enough. Making dynamic keys stronger (`H6`) hurt N512 stability. Broader top-16 routing (`H8`) failed badly; sharper top-4 routing (`H7`) was also weaker. The useful knob appears to be stronger learned key geometry, not more route diffuseness or more dynamic-key dominance.

The possibility-geometry screen implemented a learned basis bank that lets each slot mix many key vectors, with a density loss targeting more than 50% active possibilities. The mechanism worked mechanically: H10-H13 held mean gate density around 0.53-0.55 and active fraction around 0.56-0.61. Empirically, more possibility capacity did not translate into better long-context storage. `H10_possibility_geometry_dense` improved N128 to 0.9688 and increased rank to 3.72, but fell to N512 0.6510, below H5's 0.7344 and G1's 0.7604. Doubling the basis to 128 (`H12`) was worse, so raw geometric capacity appears to overcomplicate the addressing before it improves extrapolation.

The controls show what changed: possibility geometry made the model more dependent on the state matrix, especially under slot dropout, but still did not make absolute slot identity matter. Slot permutation stayed near zero degradation across H10-H13. That means the model is using the rolling state as a bag of useful vectors, not a stable indexed memory with slot-specific addresses.

## Next Probe

Promote `G1_state_kv_meanless`, `H5_geometry_static_heavy`, and only `H10_possibility_geometry_dense` to a longer 320-step, 3-5 seed screen. Treat H10 as an auxiliary direction, not the lead: test smaller basis counts, weaker basis scale, and a delayed density schedule. Avoid high-capacity basis expansion, broad top-k, strict routing, and early rank pressure.
