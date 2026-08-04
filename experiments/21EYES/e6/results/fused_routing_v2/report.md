# Experiment 6 Fused Routing V2 Report

## Run Scope

Single-seed run only (`seed=0`). Treat quality differences as preliminary unless they are large and supported by diagnostics.

## Summary Table

| Variant | Seed | Params | In-dist acc | Long acc | High distractor | Many overwrite | AUC | Top5 recall | Route entropy | Eff tokens | Selected tokens | Beta | GPU MB | train tok/s | eval tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 887552 | 0.355 | 0.309 | 0.105 | 0.293 | NA | NA | NA | 21.397 | NA | NA | 780.1 | 255834.8 | 343421.4 |
| block_route_topk_b16_k4 | 0 | 920576 | 0.375 | 0.240 | 0.102 | 0.348 | 0.530 | 0.046 | 4.288 | 18.439 | 66.000 | NA | 2188.3 | 55926.6 | 16566.6 |
| block_route_topk_b32_k4 | 0 | 920576 | 0.355 | 0.295 | 0.125 | 0.324 | 0.535 | 0.030 | 4.288 | 20.619 | 91.300 | NA | 4162.2 | 45297.4 | 16393.8 |
| coarse_to_fine_block_topk_4 | 0 | 920592 | 0.375 | 0.242 | 0.090 | 0.293 | 0.488 | 0.050 | 2.826 | 20.989 | 66.000 | 0.129 | 1408.1 | 104322.9 | 30170.5 |
| dense_augmented_qk | 0 | 920592 | 0.383 | 0.318 | 0.086 | 0.301 | 0.482 | 0.049 | 2.832 | 23.820 | NA | 0.129 | 878.2 | 74969.3 | 17079.0 |
| param_matched | 0 | 920960 | 0.398 | 0.289 | 0.129 | 0.336 | NA | NA | NA | 26.844 | NA | NA | 780.1 | 340643.2 | 350707.8 |

## Accuracy by Task

- `baseline` seed `0`: {'matched_distractors': 0.2654320987654321, 'multi_hop_lookup': 0.27439024390243905, 'password_overwrite': 0.2740963855421687, 'variable_shadowing': 0.28378378378378377}
- `block_route_topk_b16_k4` seed `0`: {'matched_distractors': 0.25308641975308643, 'multi_hop_lookup': 0.24085365853658536, 'password_overwrite': 0.28012048192771083, 'variable_shadowing': 0.2702702702702703}
- `block_route_topk_b32_k4` seed `0`: {'matched_distractors': 0.25, 'multi_hop_lookup': 0.2652439024390244, 'password_overwrite': 0.2921686746987952, 'variable_shadowing': 0.3108108108108108}
- `coarse_to_fine_block_topk_4` seed `0`: {'matched_distractors': 0.23148148148148148, 'multi_hop_lookup': 0.2774390243902439, 'password_overwrite': 0.2289156626506024, 'variable_shadowing': 0.25675675675675674}
- `dense_augmented_qk` seed `0`: {'matched_distractors': 0.2623456790123457, 'multi_hop_lookup': 0.2804878048780488, 'password_overwrite': 0.286144578313253, 'variable_shadowing': 0.2972972972972973}
- `param_matched` seed `0`: {'matched_distractors': 0.3055555555555556, 'multi_hop_lookup': 0.25609756097560976, 'password_overwrite': 0.28313253012048195, 'variable_shadowing': 0.3108108108108108}

## Accuracy by Sequence Length

- `baseline` seed `0`: {'128': 0.2513020833333333, '256': 0.29296875, '512': 0.32421875}
- `block_route_topk_b16_k4` seed `0`: {'128': 0.2747395833333333, '256': 0.25, '512': 0.23046875}
- `block_route_topk_b32_k4` seed `0`: {'128': 0.2682291666666667, '256': 0.31640625, '512': 0.2734375}
- `coarse_to_fine_block_topk_4` seed `0`: {'128': 0.2526041666666667, '256': 0.2265625, '512': 0.2578125}
- `dense_augmented_qk` seed `0`: {'128': 0.2565104166666667, '256': 0.30078125, '512': 0.3359375}
- `param_matched` seed `0`: {'128': 0.2877604166666667, '256': 0.265625, '512': 0.3125}

## Accuracy by Distractor Count

- `baseline` seed `0`: {'17+': 0.10840108401084012, '5-8': 0.5039370078740157, '9-16': 0.2785388127853881}
- `block_route_topk_b16_k4` seed `0`: {'17+': 0.11382113821138211, '5-8': 0.4448818897637795, '9-16': 0.2724505327245053}
- `block_route_topk_b32_k4` seed `0`: {'17+': 0.14092140921409213, '5-8': 0.484251968503937, '9-16': 0.2770167427701674}
- `coarse_to_fine_block_topk_4` seed `0`: {'17+': 0.0948509485094851, '5-8': 0.43700787401574803, '9-16': 0.2617960426179604}
- `dense_augmented_qk` seed `0`: {'17+': 0.1002710027100271, '5-8': 0.5118110236220472, '9-16': 0.2937595129375951}
- `param_matched` seed `0`: {'17+': 0.13279132791327913, '5-8': 0.5, '9-16': 0.2937595129375951}

## Accuracy by Overwrite Depth

- `baseline` seed `0`: {'1': 0.0851063829787234, '2': 0.20634920634920634, '3': 0.27756653992395436, '4+': 0.30919220055710306}
- `block_route_topk_b16_k4` seed `0`: {'1': 0.10638297872340426, '2': 0.1865079365079365, '3': 0.28517110266159695, '4+': 0.2883008356545961}
- `block_route_topk_b32_k4` seed `0`: {'1': 0.0851063829787234, '2': 0.1746031746031746, '3': 0.3041825095057034, '4+': 0.318941504178273}
- `coarse_to_fine_block_topk_4` seed `0`: {'1': 0.0425531914893617, '2': 0.18253968253968253, '3': 0.2661596958174905, '4+': 0.2785515320334262}
- `dense_augmented_qk` seed `0`: {'1': 0.0425531914893617, '2': 0.21031746031746032, '3': 0.3193916349809886, '4+': 0.30779944289693595}
- `param_matched` seed `0`: {'1': 0.06382978723404255, '2': 0.23015873015873015, '3': 0.3193916349809886, '4+': 0.31197771587743733}

## Routing Diagnostics

- `baseline` seed `0` route_entropy=[None, None, None, None] route_score_mean=[None, None, None, None] route_score_std=[None, None, None, None] useful_rank=[None, None, None, None] distractor_rank=[None, None, None, None] auc=[None, None, None, None] top1=[None, None, None, None] top3=[None, None, None, None] top5=[None, None, None, None] top10=[None, None, None, None] distractor_outranks=[None, None, None, None] beta=[None, None, None, None] alpha=[None, None, None, None] gate_mean=[None, None, None, None]
- `block_route_topk_b16_k4` seed `0` route_entropy=[None, 4.2879724383354185, 4.287706679105758, 4.287429404258728] route_score_mean=[None, -0.002939491411234485, -0.011404983824468218, -0.008797271149524022] route_score_std=[None, 0.019212590903043746, 0.03941535660997033, 0.05297833019867539] useful_rank=[None, 0.4628929836128009, 0.4937915026313931, 0.5249309256592823] distractor_rank=[None, 0.5146096401615068, 0.5129459583986318, 0.5019554907514248] auc=[None, 0.5870303391566267, 0.5270989002965507, 0.4765816291765077] top1=[None, 0.006640625, 0.0046875, 0.007421875] top3=[None, 0.020703125, 0.02109375, 0.03125] top5=[None, 0.032421875, 0.0484375, 0.055859375] top10=[None, 0.082421875, 0.113671875, 0.106640625] distractor_outranks=[None, 0.85546875, 0.8578125, 0.90390625] beta=[None, None, None, None] alpha=[None, None, None, None] gate_mean=[None, None, None, None]
- `block_route_topk_b32_k4` seed `0` route_entropy=[None, 4.28778504729271, 4.287732219696045, 4.287538176774978] route_score_mean=[None, -0.024024224397726356, -0.008451372006675228, -0.00782273473814712] route_score_std=[None, 0.03377829249948263, 0.037786835245788096, 0.04920731205493212] useful_rank=[None, 0.5526915484268102, 0.517488381358271, 0.538517108267115] distractor_rank=[None, 0.6152960856619757, 0.5171679675142513, 0.5312209913274273] auc=[None, 0.6091214429659886, 0.49586182707280385, 0.49940691064402926] top1=[None, 0.000390625, 0.00390625, 0.0078125] top3=[None, 0.005859375, 0.01328125, 0.02734375] top5=[None, 0.016796875, 0.02578125, 0.048828125] top10=[None, 0.037109375, 0.06875, 0.110546875] distractor_outranks=[None, 0.8546875, 0.89453125, 0.87734375] beta=[None, None, None, None] alpha=[None, None, None, None] gate_mean=[None, None, None, None]
- `coarse_to_fine_block_topk_4` seed `0` route_entropy=[None, 2.9535373985767364, 3.524975425004959, 2.0001675873994826] route_score_mean=[None, 11.249249756336212, -0.607162594050169, -4.647402393817901] route_score_std=[None, 14.774444079399109, 3.0439884960651398, 5.275276362895966] useful_rank=[None, 0.2842516810014786, 0.535889340833819, 0.5450327949467464] distractor_rank=[None, 0.2786601248662919, 0.49509470215562035, 0.5751269137021154] auc=[None, 0.48921894333470844, 0.4352474598257686, 0.5387351549245067] top1=[None, 0.00859375, 0.002734375, 0.002734375] top3=[None, 0.034375, 0.01484375, 0.013671875] top5=[None, 0.091796875, 0.0296875, 0.028125] top10=[None, 0.2609375, 0.069140625, 0.064453125] distractor_outranks=[None, 0.9, 0.92109375, 0.8796875] beta=[0.05000000074505806, 0.13761112093925476, 0.16937720775604248, 0.15951281785964966] alpha=[None, None, None, None] gate_mean=[None, None, None, None]
- `dense_augmented_qk` seed `0` route_entropy=[None, 2.9535373985767364, 3.5592909872531893, 1.9833168238401413] route_score_mean=[None, 11.249249756336212, -0.6362715761177242, -4.788328516483307] route_score_std=[None, 14.774444079399109, 2.9608987510204314, 5.350684762001038] useful_rank=[None, 0.2842516810014786, 0.5630861916972207, 0.5476229710649931] distractor_rank=[None, 0.2786601248662919, 0.5102601547841914, 0.5736627382808365] auc=[None, 0.48921894333470844, 0.42204756766441276, 0.5344182885703048] top1=[None, 0.00859375, 0.003125, 0.00234375] top3=[None, 0.034375, 0.0140625, 0.012890625] top5=[None, 0.091796875, 0.028515625, 0.02578125] top10=[None, 0.2609375, 0.069921875, 0.066015625] distractor_outranks=[None, 0.9, 0.91953125, 0.878125] beta=[0.05000000074505806, 0.13761112093925476, 0.16937720775604248, 0.15951281785964966] alpha=[None, None, None, None] gate_mean=[None, None, None, None]
- `param_matched` seed `0` route_entropy=[None, None, None, None] route_score_mean=[None, None, None, None] route_score_std=[None, None, None, None] useful_rank=[None, None, None, None] distractor_rank=[None, None, None, None] auc=[None, None, None, None] top1=[None, None, None, None] top3=[None, None, None, None] top5=[None, None, None, None] top10=[None, None, None, None] distractor_outranks=[None, None, None, None] beta=[None, None, None, None] alpha=[None, None, None, None] gate_mean=[None, None, None, None]

## Attention Diagnostics

- `baseline` seed `0` attention_entropy=[2.4940638542175293, 2.2928290516138077, 2.095096293091774, 1.5306413233280183] effective_tokens=[36.07251057624817, 19.772630310058595, 17.919399452209472, 11.823745608329773] effective_blocks=[3.603385257720947, 3.945603775978088, 3.2152880907058714, 2.424580517411232] selected_tokens=[None, None, None, None] selected_blocks=[None, None, None, None]
- `block_route_topk_b16_k4` seed `0` attention_entropy=[2.5877264976501464, 1.7354628920555115, 2.14896642267704, 1.7531607747077942] effective_tokens=[39.46695256233215, 9.467373728752136, 14.126390671730041, 10.694288575649262] effective_blocks=[3.7859288454055786, 2.386670950055122, 2.6668676495552064, 2.282296320796013] selected_tokens=[None, 66.0, 66.0, 66.0] selected_blocks=[None, 4.59375, 4.59375, 4.59375]
- `block_route_topk_b32_k4` seed `0` attention_entropy=[2.3616475045681, 2.0250921964645388, 2.214988973736763, 1.7432878881692886] effective_tokens=[34.913935613632205, 12.964785015583038, 19.972833395004272, 14.62256691455841] effective_blocks=[1.9927951514720916, 1.9592975854873658, 1.9724252671003342, 1.769139677286148] selected_tokens=[None, 91.3, 91.3, 91.3] selected_blocks=[None, 3.3375, 3.3375, 3.3375]
- `coarse_to_fine_block_topk_4` seed `0` attention_entropy=[2.269451206922531, 2.5667128682136537, 2.1249764263629913, 1.5046195328235625] effective_tokens=[33.65908536911011, 24.110331106185914, 16.505551385879517, 9.67970231771469] effective_blocks=[3.3911077797412874, 3.1608535051345825, 2.465124946832657, 1.918467679619789] selected_tokens=[None, 66.0, 66.0, 66.0] selected_blocks=[None, 4.59375, 4.59375, 4.59375]
- `dense_augmented_qk` seed `0` attention_entropy=[2.269451206922531, 2.573918843269348, 2.127191686630249, 1.554214495420456] effective_tokens=[33.65908536911011, 27.58352997303009, 20.383441042900085, 13.65481663942337] effective_blocks=[3.3911077797412874, 4.2582993626594545, 3.0840353667736053, 2.403660625219345] selected_tokens=[None, None, None, None] selected_blocks=[None, None, None, None]
- `param_matched` seed `0` attention_entropy=[2.3781736373901365, 2.4560973942279816, 2.511817967891693, 2.1233926743268965] effective_tokens=[37.10105986595154, 24.524350333213807, 25.46157901287079, 20.28795721530914] effective_blocks=[3.542124664783478, 4.375616109371185, 3.9956778943538667, 3.447942179441452] selected_tokens=[None, None, None, None] selected_blocks=[None, None, None, None]

## Performance

- `baseline` seed `0`: peak_gpu_memory_mb=780.1, train_tokens_per_sec=255834.8, eval_tokens_per_sec=343421.4, train_seconds=160.1, eval_seconds=0.9, attention_kernel={'cuda_sdp_flags': {'flash_sdp_enabled': True, 'math_sdp_enabled': True, 'mem_efficient_sdp_enabled': True}, 'expected_training_paths': ['sdpa_causal', 'sdpa_causal', 'sdpa_causal', 'sdpa_causal'], 'last_attention_paths': ['manual_dense', 'manual_dense', 'manual_dense', 'manual_dense'], 'note': 'PyTorch does not expose the exact selected SDPA backend here; flags show enabled backend families and paths show whether SDPA was requested.'}
- `block_route_topk_b16_k4` seed `0`: peak_gpu_memory_mb=2188.3, train_tokens_per_sec=55926.6, eval_tokens_per_sec=16566.6, train_seconds=732.4, eval_seconds=17.8, attention_kernel={'cuda_sdp_flags': {'flash_sdp_enabled': True, 'math_sdp_enabled': True, 'mem_efficient_sdp_enabled': True}, 'expected_training_paths': ['sdpa_causal', 'gathered_block_sparse', 'gathered_block_sparse', 'gathered_block_sparse'], 'last_attention_paths': ['manual_dense', 'gathered_block_sparse', 'gathered_block_sparse', 'gathered_block_sparse'], 'note': 'PyTorch does not expose the exact selected SDPA backend here; flags show enabled backend families and paths show whether SDPA was requested.'}
- `block_route_topk_b32_k4` seed `0`: peak_gpu_memory_mb=4162.2, train_tokens_per_sec=45297.4, eval_tokens_per_sec=16393.8, train_seconds=904.2, eval_seconds=18.0, attention_kernel={'cuda_sdp_flags': {'flash_sdp_enabled': True, 'math_sdp_enabled': True, 'mem_efficient_sdp_enabled': True}, 'expected_training_paths': ['sdpa_causal', 'gathered_block_sparse', 'gathered_block_sparse', 'gathered_block_sparse'], 'last_attention_paths': ['manual_dense', 'gathered_block_sparse', 'gathered_block_sparse', 'gathered_block_sparse'], 'note': 'PyTorch does not expose the exact selected SDPA backend here; flags show enabled backend families and paths show whether SDPA was requested.'}
- `coarse_to_fine_block_topk_4` seed `0`: peak_gpu_memory_mb=1408.1, train_tokens_per_sec=104322.9, eval_tokens_per_sec=30170.5, train_seconds=392.6, eval_seconds=9.8, attention_kernel={'cuda_sdp_flags': {'flash_sdp_enabled': True, 'math_sdp_enabled': True, 'mem_efficient_sdp_enabled': True}, 'expected_training_paths': ['sdpa_causal', 'sdpa_dense_augmented_qk_train/gathered_block_sparse_eval', 'sdpa_dense_augmented_qk_train/gathered_block_sparse_eval', 'sdpa_dense_augmented_qk_train/gathered_block_sparse_eval'], 'last_attention_paths': ['manual_dense', 'gathered_block_sparse', 'gathered_block_sparse', 'gathered_block_sparse'], 'note': 'PyTorch does not expose the exact selected SDPA backend here; flags show enabled backend families and paths show whether SDPA was requested.'}
- `dense_augmented_qk` seed `0`: peak_gpu_memory_mb=878.2, train_tokens_per_sec=74969.3, eval_tokens_per_sec=17079.0, train_seconds=546.4, eval_seconds=17.3, attention_kernel={'cuda_sdp_flags': {'flash_sdp_enabled': True, 'math_sdp_enabled': True, 'mem_efficient_sdp_enabled': True}, 'expected_training_paths': ['sdpa_causal', 'sdpa_dense_augmented_qk', 'sdpa_dense_augmented_qk', 'sdpa_dense_augmented_qk'], 'last_attention_paths': ['manual_dense', 'manual_dense_augmented_qk', 'manual_dense_augmented_qk', 'manual_dense_augmented_qk'], 'note': 'PyTorch does not expose the exact selected SDPA backend here; flags show enabled backend families and paths show whether SDPA was requested.'}
- `param_matched` seed `0`: peak_gpu_memory_mb=780.1, train_tokens_per_sec=340643.2, eval_tokens_per_sec=350707.8, train_seconds=120.2, eval_seconds=0.8, attention_kernel={'cuda_sdp_flags': {'flash_sdp_enabled': True, 'math_sdp_enabled': True, 'mem_efficient_sdp_enabled': True}, 'expected_training_paths': ['sdpa_causal', 'sdpa_causal', 'sdpa_causal', 'sdpa_causal'], 'last_attention_paths': ['manual_dense', 'manual_dense', 'manual_dense', 'manual_dense'], 'note': 'PyTorch does not expose the exact selected SDPA backend here; flags show enabled backend families and paths show whether SDPA was requested.'}

## Comparison Against Old Inter-Layer Route Gate

- old `inter_layer_route_gate` seed `0`: in_dist=0.355, high_distractor=0.102, train_tokens_per_sec=112430.0, eval_tokens_per_sec=45435.8, peak_gpu_memory_mb=1186.9

## Conclusion

not supported: route scores did not rank useful evidence above distractors

## Failure Analysis

The dense augmented route channel learned nonzero beta values, but the route scores did not reliably separate useful evidence from distractors. Mean useful-vs-distractor AUC stayed near or below 0.5 for the dense and coarse-to-fine variants, and top-k useful recall remained low. In this implementation, dense augmented training also did not beat the old route-gate training throughput, although it used less peak memory.

The block-routed variants reduced candidate tokens/blocks, but this gathered implementation was slower than dense SDPA because it materializes selected K/V tensors and cannot use a fused sparse attention kernel. The b32 setting selected more tokens than b16 and used more memory, so larger pages did not create a practical speed path here.

The isolated high-distractor gains for block routing were not enough to offset weaker long-context accuracy, low useful evidence recall, and poor throughput.

## Next Recommended Experiment

First add stronger controls for sparse selection: random block selection with the same local window and candidate budget, plus an oracle-eval-only upper bound that is never used for training. Then rerun b16 with a real fused block-sparse kernel or a grouped-query implementation that avoids materializing `[batch, heads, query, candidates, d_head]` K/V tensors. Only after throughput is credible should this be repeated with three seeds and a candidate-budget sweep.

## Plots

- `W:\Intuition-Labs\experiments\21EYES\e6\plots\fused_routing_v2\accuracy_by_variant.png`
- `W:\Intuition-Labs\experiments\21EYES\e6\plots\fused_routing_v2\accuracy_vs_seq_len.png`
- `W:\Intuition-Labs\experiments\21EYES\e6\plots\fused_routing_v2\accuracy_vs_distractors.png`
- `W:\Intuition-Labs\experiments\21EYES\e6\plots\fused_routing_v2\gate_mean_by_layer.png`
- `W:\Intuition-Labs\experiments\21EYES\e6\plots\fused_routing_v2\gate_entropy_by_layer.png`
- `W:\Intuition-Labs\experiments\21EYES\e6\plots\fused_routing_v2\attention_entropy_by_layer.png`
- `W:\Intuition-Labs\experiments\21EYES\e6\plots\fused_routing_v2\alpha_by_layer.png`
- `W:\Intuition-Labs\experiments\21EYES\e6\plots\fused_routing_v2\route_entropy_by_layer.png`
- `W:\Intuition-Labs\experiments\21EYES\e6\plots\fused_routing_v2\beta_by_layer.png`
- `W:\Intuition-Labs\experiments\21EYES\e6\plots\fused_routing_v2\useful_vs_distractor_auc.png`
- `W:\Intuition-Labs\experiments\21EYES\e6\plots\fused_routing_v2\useful_topk_recall.png`
- `W:\Intuition-Labs\experiments\21EYES\e6\plots\fused_routing_v2\throughput_by_variant.png`
- `W:\Intuition-Labs\experiments\21EYES\e6\plots\fused_routing_v2\gpu_memory_by_variant.png`
- `W:\Intuition-Labs\experiments\21EYES\e6\plots\fused_routing_v2\accuracy_vs_effective_tokens.png`
