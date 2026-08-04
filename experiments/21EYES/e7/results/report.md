# Experiment 7 Report

## Run Scope

Single-seed run only (`seed=0`). Treat results as preliminary unless effect sizes are large and controls agree.

## Summary Table

| Variant | Seed | Params | Acc | In-dist | Long mean | High distractor | Many overwrite | Slot div | Slot entropy | Align cos | Innov/stable | Control drop | GPU MB | Train tok/s | Eval tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_transformer | 0 | 887552 | 0.278 | 0.414 | 0.305 | 0.070 | 0.297 | NA | NA | 0.000 | NA | NA | 667.6 | 252208.8 | 264883.1 |
| memory_slot_attention | 0 | 1782016 | 0.217 | 0.242 | 0.273 | 0.078 | 0.219 | 0.009 | 4.155 | 0.000 | 0.000 | -0.010 | 5618.8 | 17784.0 | 12170.3 |
| mirrored_slot_attention | 0 | 1484800 | 0.209 | 0.242 | 0.277 | 0.070 | 0.180 | 0.014 | 4.156 | 0.000 | 0.000 | -0.026 | 5616.4 | 17437.7 | 8768.4 |
| param_matched_baseline | 0 | 1940736 | 0.255 | 0.344 | 0.281 | 0.109 | 0.258 | NA | NA | 0.000 | NA | NA | 686.7 | 224051.1 | 253574.4 |
| predictive_slot_residual | 0 | 2227072 | 0.194 | 0.234 | 0.238 | 0.062 | 0.195 | 0.753 | 3.091 | 1.000 | 0.894 | -0.021 | 8715.3 | 16147.4 | 8777.6 |
| slot_alignment_loss | 0 | 1832320 | 0.184 | 0.219 | 0.227 | 0.055 | 0.195 | 0.013 | 4.155 | 1.000 | 0.000 | 0.018 | 5714.0 | 17413.5 | 9241.9 |

## Accuracy by Task

- `baseline_transformer` seed `0`: {'matched_distractors': 0.25609756097560976, 'multi_hop_lookup': 0.23270440251572327, 'password_overwrite': 0.3063583815028902, 'variable_shadowing': 0.3194444444444444}
- `memory_slot_attention` seed `0`: {'matched_distractors': 0.18292682926829268, 'multi_hop_lookup': 0.24528301886792453, 'password_overwrite': 0.17341040462427745, 'variable_shadowing': 0.2777777777777778}
- `mirrored_slot_attention` seed `0`: {'matched_distractors': 0.1524390243902439, 'multi_hop_lookup': 0.27672955974842767, 'password_overwrite': 0.18497109826589594, 'variable_shadowing': 0.22916666666666666}
- `param_matched_baseline` seed `0`: {'matched_distractors': 0.23780487804878048, 'multi_hop_lookup': 0.2641509433962264, 'password_overwrite': 0.2658959537572254, 'variable_shadowing': 0.25}
- `predictive_slot_residual` seed `0`: {'matched_distractors': 0.17682926829268292, 'multi_hop_lookup': 0.20754716981132076, 'password_overwrite': 0.2023121387283237, 'variable_shadowing': 0.1875}
- `slot_alignment_loss` seed `0`: {'matched_distractors': 0.17073170731707318, 'multi_hop_lookup': 0.1949685534591195, 'password_overwrite': 0.18497109826589594, 'variable_shadowing': 0.1875}

## Accuracy by Sequence Length

- `baseline_transformer` seed `0`: {'128': 0.2604166666666667, '256': 0.28125, '512': 0.328125}
- `memory_slot_attention` seed `0`: {'128': 0.1796875, '256': 0.2734375, '512': 0.2734375}
- `mirrored_slot_attention` seed `0`: {'128': 0.1640625, '256': 0.2890625, '512': 0.265625}
- `param_matched_baseline` seed `0`: {'128': 0.23697916666666666, '256': 0.28125, '512': 0.28125}
- `predictive_slot_residual` seed `0`: {'128': 0.1640625, '256': 0.2109375, '512': 0.265625}
- `slot_alignment_loss` seed `0`: {'128': 0.15625, '256': 0.2421875, '512': 0.2109375}

## Accuracy by Distractor Count

- `baseline_transformer` seed `0`: {'17+': 0.09574468085106383, '5-8': 0.5775862068965517, '9-16': 0.2767857142857143}
- `memory_slot_attention` seed `0`: {'17+': 0.10106382978723404, '5-8': 0.41379310344827586, '9-16': 0.21428571428571427}
- `mirrored_slot_attention` seed `0`: {'17+': 0.09042553191489362, '5-8': 0.3448275862068966, '9-16': 0.22916666666666666}
- `param_matched_baseline` seed `0`: {'17+': 0.11170212765957446, '5-8': 0.5, '9-16': 0.25}
- `predictive_slot_residual` seed `0`: {'17+': 0.07446808510638298, '5-8': 0.35344827586206895, '9-16': 0.20535714285714285}
- `slot_alignment_loss` seed `0`: {'17+': 0.05319148936170213, '5-8': 0.33620689655172414, '9-16': 0.20535714285714285}

## Accuracy by Overwrite Depth

- `baseline_transformer` seed `0`: {'1': 0.07407407407407407, '2': 0.23846153846153847, '3': 0.28125, '4+': 0.30704225352112674}
- `memory_slot_attention` seed `0`: {'1': 0.07407407407407407, '2': 0.23076923076923078, '3': 0.21875, '4+': 0.22253521126760564}
- `mirrored_slot_attention` seed `0`: {'1': 0.0, '2': 0.3, '3': 0.1640625, '4+': 0.2084507042253521}
- `param_matched_baseline` seed `0`: {'1': 0.037037037037037035, '2': 0.26153846153846155, '3': 0.28125, '4+': 0.2591549295774648}
- `predictive_slot_residual` seed `0`: {'1': 0.0, '2': 0.23076923076923078, '3': 0.1953125, '4+': 0.19436619718309858}
- `slot_alignment_loss` seed `0`: {'1': 0.037037037037037035, '2': 0.19230769230769232, '3': 0.15625, '4+': 0.2028169014084507}

## Slot Diagnostics

- `baseline_transformer` seed `0` diversity=[None, None, None, None] read_entropy=[None, None, None, None] utilization=[None, None, None, None] slot_norm=[None, None, None, None] cross_layer_cosine=0.0 predicted_actual=0.0 predicted_actual_by_pair=[] innovation_stable_ratio=[None, None, None, None]
- `memory_slot_attention` seed `0` diversity=[0.0071750511764548715, 0.008784903562627733, 0.010315332026220859, 0.011341203865595163] read_entropy=[4.158070909976959, 4.157385206222534, 4.156028831005097, 4.1467549204826355] utilization=[1.0, 1.0, 1.0, 1.0] slot_norm=[10.52681610584259, 10.688182425498962, 10.532206583023072, 9.9635822057724] cross_layer_cosine=0.9940310046076775 predicted_actual=0.0 predicted_actual_by_pair=[] innovation_stable_ratio=[0.0, 0.0, 0.0, 0.0]
- `mirrored_slot_attention` seed `0` diversity=[0.008926832932047546, 0.012098894105292857, 0.01609368557110429, 0.018878695275634527] read_entropy=[4.1564106225967405, 4.155986166000366, 4.155820906162262, 4.156364548206329] utilization=[1.0, 1.0, 1.0, 1.0] slot_norm=[10.630330920219421, 10.753088641166688, 10.474129176139831, 9.422358393669128] cross_layer_cosine=0.9884643599390983 predicted_actual=0.0 predicted_actual_by_pair=[] innovation_stable_ratio=[0.0, 0.0, 0.0, 0.0]
- `param_matched_baseline` seed `0` diversity=[None, None, None, None] read_entropy=[None, None, None, None] utilization=[None, None, None, None] slot_norm=[None, None, None, None] cross_layer_cosine=0.0 predicted_actual=0.0 predicted_actual_by_pair=[] innovation_stable_ratio=[None, None, None, None]
- `predictive_slot_residual` seed `0` diversity=[0.014104759879410267, 0.9998154625296592, 0.9999449148774147, 0.9997425571084022] read_entropy=[4.156413400173188, 2.58486185669899, 2.7923653423786163, 2.8290489315986633] utilization=[1.0, 1.0, 1.0, 1.0] slot_norm=[9.024332046508789, 9.907270431518555, 10.275012993812561, 10.629613733291626] cross_layer_cosine=-0.008626631181687116 predicted_actual=0.9997642934322357 predicted_actual_by_pair=[0.9997768476605415, 0.9996962532401085, 0.9998196452856064] innovation_stable_ratio=[0.0, 1.0546821147203445, 1.2289909720420837, 1.294025832414627]
- `slot_alignment_loss` seed `0` diversity=[0.009795822808519006, 0.011820256547071039, 0.01371801842469722, 0.015513227344490588] read_entropy=[4.154624629020691, 4.156600248813629, 4.154066681861877, 4.155879127979278] utilization=[1.0, 1.0, 1.0, 1.0] slot_norm=[10.556777691841125, 10.688227438926697, 10.511965155601501, 9.73154046535492] cross_layer_cosine=0.9930772066116333 predicted_actual=0.9999938189983368 predicted_actual_by_pair=[0.9999929934740066, 0.9999945878982544, 0.9999939203262329] innovation_stable_ratio=[0.0, 0.0, 0.0, 0.0]

## Slot Control Results

- `baseline_transformer` seed `0`: not applicable
- `memory_slot_attention` seed `0`: {'base_accuracy': 0.22916666666666666, 'base_loss': 3.553899129231771, 'controls': {'hidden_state_shuffle': {'accuracy': 0.3020833333333333, 'drop': -0.07291666666666666, 'examples': 96, 'loss': 3.537968953450521, 'tokens_per_sec': 10180.750576938653}, 'slot_random': {'accuracy': 0.15625, 'drop': 0.07291666666666666, 'examples': 96, 'loss': 3.83221435546875, 'tokens_per_sec': 10037.306629801922}, 'slot_shuffle': {'accuracy': 0.22916666666666666, 'drop': 0.0, 'examples': 96, 'loss': 3.5538482666015625, 'tokens_per_sec': 9871.255150715078}, 'slot_zero': {'accuracy': 0.2708333333333333, 'drop': -0.04166666666666666, 'examples': 96, 'loss': 3.5137430826822915, 'tokens_per_sec': 9848.466815391654}}, 'examples': 96}
- `mirrored_slot_attention` seed `0`: {'base_accuracy': 0.23958333333333334, 'base_loss': 3.5439351399739585, 'controls': {'hidden_state_shuffle': {'accuracy': 0.3333333333333333, 'drop': -0.09374999999999997, 'examples': 96, 'loss': 3.471277872721354, 'tokens_per_sec': 4985.332258282493}, 'slot_random': {'accuracy': 0.20833333333333334, 'drop': 0.03125, 'examples': 96, 'loss': 3.62432861328125, 'tokens_per_sec': 4854.280718668006}, 'slot_shuffle': {'accuracy': 0.23958333333333334, 'drop': 0.0, 'examples': 96, 'loss': 3.5439961751302085, 'tokens_per_sec': 4974.335094561874}, 'slot_zero': {'accuracy': 0.28125, 'drop': -0.04166666666666666, 'examples': 96, 'loss': 3.3319549560546875, 'tokens_per_sec': 5034.239011576473}}, 'examples': 96}
- `param_matched_baseline` seed `0`: not applicable
- `predictive_slot_residual` seed `0`: {'base_accuracy': 0.1875, 'base_loss': 3.637939453125, 'controls': {'hidden_state_shuffle': {'accuracy': 0.21875, 'drop': -0.03125, 'examples': 96, 'loss': 3.78875732421875, 'tokens_per_sec': 5126.969474114849}, 'slot_random': {'accuracy': 0.15625, 'drop': 0.03125, 'examples': 96, 'loss': 3.8413289388020835, 'tokens_per_sec': 5184.265450592768}, 'slot_shuffle': {'accuracy': 0.1875, 'drop': 0.0, 'examples': 96, 'loss': 3.6378377278645835, 'tokens_per_sec': 5179.137113943744}, 'slot_zero': {'accuracy': 0.2708333333333333, 'drop': -0.08333333333333331, 'examples': 96, 'loss': 3.6870320638020835, 'tokens_per_sec': 5255.283253217195}}, 'examples': 96}
- `slot_alignment_loss` seed `0`: {'base_accuracy': 0.21875, 'base_loss': 3.5821736653645835, 'controls': {'hidden_state_shuffle': {'accuracy': 0.25, 'drop': -0.03125, 'examples': 96, 'loss': 3.4856974283854165, 'tokens_per_sec': 5285.220848540293}, 'slot_random': {'accuracy': 0.17708333333333334, 'drop': 0.04166666666666666, 'examples': 96, 'loss': 3.7925923665364585, 'tokens_per_sec': 5238.070473530059}, 'slot_shuffle': {'accuracy': 0.21875, 'drop': 0.0, 'examples': 96, 'loss': 3.5820515950520835, 'tokens_per_sec': 5265.279594618144}, 'slot_zero': {'accuracy': 0.15625, 'drop': 0.0625, 'examples': 96, 'loss': 3.5886942545572915, 'tokens_per_sec': 5211.251179461838}}, 'examples': 96}

## Evidence Diagnostics

- `baseline_transformer` seed `0` useful_vs_distractor_slot_auc=[None, None, None, None] useful_slot_top1=[None, None, None, None] useful_slot_top3=[None, None, None, None] useful_slot_top5=[None, None, None, None] useful_slot_top10=[None, None, None, None] useful_concentration=[None, None, None, None] distractor_concentration=[None, None, None, None]
- `memory_slot_attention` seed `0` useful_vs_distractor_slot_auc=[0.5350406198122073, 0.5153596227348316, 0.5018185757973697, 0.4910811568319332] useful_slot_top1=[0.00625, 0.003125, 0.1109375, 0.20546875] useful_slot_top3=[0.01953125, 0.02109375, 0.21875, 0.7265625] useful_slot_top5=[0.0375, 0.06328125, 0.31171875, 0.790625] useful_slot_top10=[0.065625, 0.15703125, 0.44375, 0.853125] useful_concentration=[0.02065557833702769, 0.028809998897486366, 0.47493194707494696, 0.24192441846244037] distractor_concentration=[0.02083942631725222, 0.029304305420373565, 0.4674103902245406, 0.24201960184145718]
- `mirrored_slot_attention` seed `0` useful_vs_distractor_slot_auc=[0.4857829116575886, 0.49332594669249374, 0.48476030773017553, 0.48435505937668494] useful_slot_top1=[0.00234375, 0.003125, 0.00390625, 0.0078125] useful_slot_top3=[0.00546875, 0.009375, 0.01640625, 0.03125] useful_slot_top5=[0.01953125, 0.021875, 0.02734375, 0.0484375] useful_slot_top10=[0.03984375, 0.0421875, 0.06640625, 0.125] useful_concentration=[0.7469532581570093, 0.6117076817638007, 0.6999065682379296, 0.7773267221840797] distractor_concentration=[0.7413658090867102, 0.6220874321559677, 0.7140485397656449, 0.7780087234277744]
- `param_matched_baseline` seed `0` useful_vs_distractor_slot_auc=[None, None, None, None] useful_slot_top1=[None, None, None, None] useful_slot_top3=[None, None, None, None] useful_slot_top5=[None, None, None, None] useful_slot_top10=[None, None, None, None] useful_concentration=[None, None, None, None] distractor_concentration=[None, None, None, None]
- `predictive_slot_residual` seed `0` useful_vs_distractor_slot_auc=[0.46074762163625566, 0.5679609066632111, 0.5224191445682663, 0.479908285534475] useful_slot_top1=[0.00234375, 0.0078125, 0.15625, 0.12890625] useful_slot_top3=[0.01796875, 0.04296875, 0.22265625, 0.18828125] useful_slot_top5=[0.11484375, 0.06953125, 0.28984375, 0.4125] useful_slot_top10=[0.43515625, 0.19375, 0.421875, 0.6359375] useful_concentration=[0.036757387468242086, 0.015815995004959404, 0.016129562284913846, 0.01599557507724967] distractor_concentration=[0.037167390354443344, 0.01580379955994431, 0.016098070578300393, 0.016048064749338665]
- `slot_alignment_loss` seed `0` useful_vs_distractor_slot_auc=[0.42982159011298793, 0.5106514091428835, 0.4943094514601398, 0.5069403826375491] useful_slot_top1=[0.00078125, 0.0125, 0.01953125, 0.01328125] useful_slot_top3=[0.009375, 0.04765625, 0.109375, 0.059375] useful_slot_top5=[0.01953125, 0.06796875, 0.1609375, 0.10234375] useful_slot_top10=[0.05390625, 0.15234375, 0.2796875, 0.19609375] useful_concentration=[0.018831761233741418, 0.018989600022905506, 0.018811350845498963, 0.018818474482395688] distractor_concentration=[0.018945210037054495, 0.01898317845480051, 0.018859616969712077, 0.018803654599469154]

## Attention Diagnostics

- `baseline_transformer` seed `0` attention_entropy=[2.358303356170654, 2.4372838139533997, 2.0875704526901244, 1.4853520333766936] effective_tokens=[36.473932552337644, 23.830723094940186, 18.74083003997803, 12.032005262374877] token_to_slot_entropy=[None, None, None, None] slot_to_token_entropy=[None, None, None, None]
- `memory_slot_attention` seed `0` attention_entropy=[3.431696492433548, 1.7553754150867462, 1.5302815347909928, 1.590733140707016] effective_tokens=[64.90235252380371, 9.218820488452911, 8.611258339881896, 9.757172787189484] token_to_slot_entropy=[4.148997378349304, 4.130090951919556, 2.497751259803772, 3.37329677939415] slot_to_token_entropy=[5.267435765266418, 5.267006206512451, 5.2342467427253725, 5.2552859544754025]
- `mirrored_slot_attention` seed `0` attention_entropy=[3.4016697347164153, 1.4844683915376664, 1.5395612746477128, 1.45396568775177] effective_tokens=[61.26250801086426, 7.176165199279785, 8.321416902542115, 7.857835125923157] token_to_slot_entropy=[1.889474979043007, 1.672057281434536, 1.2545477479696274, 0.885199175029993] slot_to_token_entropy=[5.223867380619049, 5.226348102092743, 5.207219684123993, 5.139126396179199]
- `param_matched_baseline` seed `0` attention_entropy=[2.3749765038490294, 2.482317399978638, 2.1017366707324983, 1.8167109966278077] effective_tokens=[35.838886165618895, 23.74032621383667, 19.329955720901488, 15.54442231655121] token_to_slot_entropy=[None, None, None, None] slot_to_token_entropy=[None, None, None, None]
- `predictive_slot_residual` seed `0` attention_entropy=[3.5361702263355257, 1.5562342435121537, 1.7694465905427932, 1.7370720237493515] effective_tokens=[68.61719007492066, 8.009267354011536, 10.994737792015076, 12.778779911994935] token_to_slot_entropy=[4.128118050098419, 4.157911348342895, 4.157755255699158, 4.158035373687744] slot_to_token_entropy=[5.26595675945282, 5.267878615856171, 5.267883253097534, 5.26789767742157]
- `slot_alignment_loss` seed `0` attention_entropy=[3.478604018688202, 2.0875464975833893, 2.1252885192632673, 1.994305792450905] effective_tokens=[65.90157957077027, 16.35395575761795, 19.80191013813019, 17.86060314178467] token_to_slot_entropy=[4.154777228832245, 4.1549549221992494, 4.155838894844055, 4.155869007110596] slot_to_token_entropy=[5.267666900157929, 5.267755234241486, 5.2678020119667055, 5.267816758155822]

## Performance

- `baseline_transformer` seed `0`: params=887552, trainable=887552, peak_gpu_memory_mb=667.6, train_tokens_per_sec=252208.8, eval_tokens_per_sec=264883.1, train_seconds=162.4, eval_seconds=0.6, attention_kernel={'cuda_sdp_flags': {'flash_sdp_enabled': True, 'math_sdp_enabled': True, 'mem_efficient_sdp_enabled': True}, 'expected_training_paths': ['sdpa_causal', 'sdpa_causal', 'sdpa_causal', 'sdpa_causal'], 'last_attention_paths': ['manual_dense', 'manual_dense', 'manual_dense', 'manual_dense'], 'note': 'Slot operations are dense tensor projections over causal hidden states; token self-attention uses SDPA during training.', 'slot_variant': False}
- `memory_slot_attention` seed `0`: params=1782016, trainable=1782016, peak_gpu_memory_mb=5618.8, train_tokens_per_sec=17784.0, eval_tokens_per_sec=12170.3, train_seconds=863.7, eval_seconds=12.1, attention_kernel={'cuda_sdp_flags': {'flash_sdp_enabled': True, 'math_sdp_enabled': True, 'mem_efficient_sdp_enabled': True}, 'expected_training_paths': ['sdpa_causal', 'sdpa_causal', 'sdpa_causal', 'sdpa_causal'], 'last_attention_paths': ['manual_dense', 'manual_dense', 'manual_dense', 'manual_dense'], 'note': 'Slot operations are dense tensor projections over causal hidden states; token self-attention uses SDPA during training.', 'slot_variant': True}
- `mirrored_slot_attention` seed `0`: params=1484800, trainable=1484800, peak_gpu_memory_mb=5616.4, train_tokens_per_sec=17437.7, eval_tokens_per_sec=8768.4, train_seconds=880.9, eval_seconds=16.8, attention_kernel={'cuda_sdp_flags': {'flash_sdp_enabled': True, 'math_sdp_enabled': True, 'mem_efficient_sdp_enabled': True}, 'expected_training_paths': ['sdpa_causal', 'sdpa_causal', 'sdpa_causal', 'sdpa_causal'], 'last_attention_paths': ['manual_dense', 'manual_dense', 'manual_dense', 'manual_dense'], 'note': 'Slot operations are dense tensor projections over causal hidden states; token self-attention uses SDPA during training.', 'slot_variant': True}
- `param_matched_baseline` seed `0`: params=1940736, trainable=1940736, peak_gpu_memory_mb=686.7, train_tokens_per_sec=224051.1, eval_tokens_per_sec=253574.4, train_seconds=182.8, eval_seconds=0.6, attention_kernel={'cuda_sdp_flags': {'flash_sdp_enabled': True, 'math_sdp_enabled': True, 'mem_efficient_sdp_enabled': True}, 'expected_training_paths': ['sdpa_causal', 'sdpa_causal', 'sdpa_causal', 'sdpa_causal'], 'last_attention_paths': ['manual_dense', 'manual_dense', 'manual_dense', 'manual_dense'], 'note': 'Slot operations are dense tensor projections over causal hidden states; token self-attention uses SDPA during training.', 'slot_variant': False}
- `predictive_slot_residual` seed `0`: params=2227072, trainable=2227072, peak_gpu_memory_mb=8715.3, train_tokens_per_sec=16147.4, eval_tokens_per_sec=8777.6, train_seconds=951.2, eval_seconds=16.8, attention_kernel={'cuda_sdp_flags': {'flash_sdp_enabled': True, 'math_sdp_enabled': True, 'mem_efficient_sdp_enabled': True}, 'expected_training_paths': ['sdpa_causal', 'sdpa_causal', 'sdpa_causal', 'sdpa_causal'], 'last_attention_paths': ['manual_dense', 'manual_dense', 'manual_dense', 'manual_dense'], 'note': 'Slot operations are dense tensor projections over causal hidden states; token self-attention uses SDPA during training.', 'slot_variant': True}
- `slot_alignment_loss` seed `0`: params=1832320, trainable=1832320, peak_gpu_memory_mb=5714.0, train_tokens_per_sec=17413.5, eval_tokens_per_sec=9241.9, train_seconds=882.1, eval_seconds=16.0, attention_kernel={'cuda_sdp_flags': {'flash_sdp_enabled': True, 'math_sdp_enabled': True, 'mem_efficient_sdp_enabled': True}, 'expected_training_paths': ['sdpa_causal', 'sdpa_causal', 'sdpa_causal', 'sdpa_causal'], 'last_attention_paths': ['manual_dense', 'manual_dense', 'manual_dense', 'manual_dense'], 'note': 'Slot operations are dense tensor projections over causal hidden states; token self-attention uses SDPA during training.', 'slot_variant': True}

## Conclusion

not supported: the parameter-matched baseline equals or beats the slot variants

## Failure Analysis

Interpret failures against the critical controls: if parameter matching removes gains, if slot controls do not reduce accuracy, if slot diversity is near zero, or if predictive innovation collapses toward zero, the slot-state mediation lemma is not supported by this run.

## Next Recommended Experiment

If E7 is positive, rerun the six-variant matrix with seeds 1 and 2 and sweep `lambda_align` over 0.01, 0.05, and 0.1. If E7 is negative, isolate whether causal per-token slots are too weak by testing chunk-causal prefix slots with the same anti-cheat constraints.

## Plots

- `W:\Intuition-Labs\experiments\21EYES\e7\plots\accuracy_by_variant.png`
- `W:\Intuition-Labs\experiments\21EYES\e7\plots\accuracy_by_task.png`
- `W:\Intuition-Labs\experiments\21EYES\e7\plots\accuracy_vs_seq_len.png`
- `W:\Intuition-Labs\experiments\21EYES\e7\plots\accuracy_vs_distractors.png`
- `W:\Intuition-Labs\experiments\21EYES\e7\plots\accuracy_vs_overwrite_depth.png`
- `W:\Intuition-Labs\experiments\21EYES\e7\plots\slot_diversity_by_layer.png`
- `W:\Intuition-Labs\experiments\21EYES\e7\plots\slot_entropy_by_layer.png`
- `W:\Intuition-Labs\experiments\21EYES\e7\plots\attention_entropy_by_layer.png`
- `W:\Intuition-Labs\experiments\21EYES\e7\plots\innovation_stable_ratio.png`
- `W:\Intuition-Labs\experiments\21EYES\e7\plots\slot_alignment_by_layer.png`
- `W:\Intuition-Labs\experiments\21EYES\e7\plots\slot_control_drops.png`
- `W:\Intuition-Labs\experiments\21EYES\e7\plots\throughput_by_variant.png`
- `W:\Intuition-Labs\experiments\21EYES\e7\plots\gpu_memory_by_variant.png`
