# Experiment 6 Report

## Summary Table

| Variant | Seed | Params | In-dist acc | Long acc | High distractor | Many overwrite | Gate mean | Alpha | GPU MB | ex/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 887552 | 0.379 | 0.303 | 0.094 | 0.309 | NA | NA | 970.7 | 1078.6 |
| inter_layer_route_gate | 0 | 920584 | 0.355 | 0.314 | 0.102 | 0.344 | 0.891 | 0.081 | 1186.9 | 197.2 |
| param_matched | 0 | 920960 | 0.359 | 0.312 | 0.098 | 0.352 | NA | NA | 971.3 | 1048.5 |
| prev_attn_reuse | 0 | 887560 | 0.348 | 0.314 | 0.098 | 0.344 | 0.012 | 0.108 | 1355.6 | 205.8 |
| random_gate | 0 | 920584 | 0.316 | 0.277 | 0.105 | 0.344 | 0.992 | 0.024 | 1186.2 | 201.9 |
| same_layer_u_gate | 0 | 920584 | 0.355 | 0.305 | 0.113 | 0.340 | 0.864 | 0.085 | 1232.8 | 157.9 |

## Accuracy by Task

- `baseline` seed `0`: {'matched_distractors': 0.2623456790123457, 'multi_hop_lookup': 0.27134146341463417, 'password_overwrite': 0.2620481927710843, 'variable_shadowing': 0.31756756756756754}
- `inter_layer_route_gate` seed `0`: {'matched_distractors': 0.25925925925925924, 'multi_hop_lookup': 0.28353658536585363, 'password_overwrite': 0.28313253012048195, 'variable_shadowing': 0.32094594594594594}
- `param_matched` seed `0`: {'matched_distractors': 0.2839506172839506, 'multi_hop_lookup': 0.29878048780487804, 'password_overwrite': 0.28012048192771083, 'variable_shadowing': 0.28378378378378377}
- `prev_attn_reuse` seed `0`: {'matched_distractors': 0.2623456790123457, 'multi_hop_lookup': 0.2774390243902439, 'password_overwrite': 0.2891566265060241, 'variable_shadowing': 0.30743243243243246}
- `random_gate` seed `0`: {'matched_distractors': 0.23148148148148148, 'multi_hop_lookup': 0.2804878048780488, 'password_overwrite': 0.2469879518072289, 'variable_shadowing': 0.30067567567567566}
- `same_layer_u_gate` seed `0`: {'matched_distractors': 0.2839506172839506, 'multi_hop_lookup': 0.2896341463414634, 'password_overwrite': 0.29518072289156627, 'variable_shadowing': 0.2635135135135135}

## Accuracy by Sequence Length

- `baseline` seed `0`: {'128': 0.2604166666666667, '256': 0.3046875, '512': 0.30078125}
- `inter_layer_route_gate` seed `0`: {'128': 0.2669270833333333, '256': 0.3359375, '512': 0.29296875}
- `param_matched` seed `0`: {'128': 0.26953125, '256': 0.3046875, '512': 0.3203125}
- `prev_attn_reuse` seed `0`: {'128': 0.2630208333333333, '256': 0.30078125, '512': 0.328125}
- `random_gate` seed `0`: {'128': 0.2552083333333333, '256': 0.28515625, '512': 0.26953125}
- `same_layer_u_gate` seed `0`: {'128': 0.26953125, '256': 0.32421875, '512': 0.28515625}

## Accuracy by Distractor Count

- `baseline` seed `0`: {'17+': 0.1002710027100271, '5-8': 0.5078740157480315, '9-16': 0.2876712328767123}
- `inter_layer_route_gate` seed `0`: {'17+': 0.12737127371273713, '5-8': 0.515748031496063, '9-16': 0.2861491628614916}
- `param_matched` seed `0`: {'17+': 0.11382113821138211, '5-8': 0.5, '9-16': 0.3013698630136986}
- `prev_attn_reuse` seed `0`: {'17+': 0.1111111111111111, '5-8': 0.5196850393700787, '9-16': 0.289193302891933}
- `random_gate` seed `0`: {'17+': 0.12466124661246612, '5-8': 0.5, '9-16': 0.2511415525114155}
- `same_layer_u_gate` seed `0`: {'17+': 0.13008130081300814, '5-8': 0.5118110236220472, '9-16': 0.2815829528158295}

## Gate Statistics by Layer

- `baseline` seed `0` gate_mean=[None, None, None, None] gate_entropy=[None, None, None, None] alpha=[None, None, None, None] useful_rank=[None, None, None, None] distractor_rank=[None, None, None, None]
- `inter_layer_route_gate` seed `0` gate_mean=[None, 0.9928246676921845, 0.9064056068658829, 0.7743282690644264] gate_entropy=[None, 0.04257023455575108, None, None] alpha=[0.0, -0.00047314740368165076, 0.1526738405227661, 0.17096328735351562] useful_rank=[None, 0.3750264500864432, 0.45310437764019296, 0.4807222830360843] distractor_rank=[None, 0.32656052973907207, 0.38011770062003053, 0.37922333090536997]
- `param_matched` seed `0` gate_mean=[None, None, None, None] gate_entropy=[None, None, None, None] alpha=[None, None, None, None] useful_rank=[None, None, None, None] distractor_rank=[None, None, None, None]
- `prev_attn_reuse` seed `0` gate_mean=[None, 0.011638474545907228, 0.011638472869526595, 0.011638472881168127] gate_entropy=[None, None, None, None] alpha=[0.0, 0.16530345380306244, 0.0684070885181427, 0.19945962727069855] useful_rank=[None, 0.21821480442122265, 0.48234427776624217, 0.5254000868306321] distractor_rank=[None, 0.17396155985479708, 0.4763078175164992, 0.4839711575943511]
- `random_gate` seed `0` gate_mean=[None, 0.9926513090729714, 0.9924838155508041, 0.9923622384667397] gate_entropy=[None, 0.04342478737235069, 0.04423837009817362, 0.04482849482446909] alpha=[0.0, -0.031020699068903923, 0.055589716881513596, 0.07087807357311249] useful_rank=[None, 0.4002433773348457, 0.49633241481205914, 0.47595141016499837] distractor_rank=[None, 0.2857730873962282, 0.44883724100218386, 0.39312898312782635]
- `same_layer_u_gate` seed `0` gate_mean=[0.5788220316171646, 0.997356629371643, 0.9270103096961975, 0.9536812692880631] gate_entropy=[None, None, None, None] alpha=[0.09622503072023392, 0.008340971544384956, 0.12266089767217636, 0.11343234032392502] useful_rank=[0.24551623074785311, 0.28016814956936287, 0.4718763353601389, 0.48609875115653267] distractor_rank=[0.3014313531777589, 0.21398836552689318, 0.37924218515399843, 0.38939296882017516]

## Attention Entropy by Layer

- `baseline` seed `0`: [2.436968070268631, 2.1520296394824983, 1.868164101243019, 1.2239078924059867]
- `inter_layer_route_gate` seed `0`: [2.201302433013916, 2.0819444984197615, 2.060451513528824, 1.5749614477157592]
- `param_matched` seed `0`: [2.3944798707962036, 2.0370492905378343, 2.1439372092485427, 1.5873158127069473]
- `prev_attn_reuse` seed `0`: [2.440450942516327, 2.0599562734365464, 2.3321471571922303, 1.6387330919504166]
- `random_gate` seed `0`: [2.2284926950931547, 2.1573443591594694, 2.0351002663373947, 1.6062659621238708]
- `same_layer_u_gate` seed `0`: [2.19299721121788, 2.215469178557396, 2.140000346302986, 1.7221586167812348]

## GPU Memory and Throughput

- `baseline` seed `0`: peak_gpu_memory_mb=970.7, tokens_per_sec=248499.4, examples_per_sec=1078.6
- `inter_layer_route_gate` seed `0`: peak_gpu_memory_mb=1186.9, tokens_per_sec=45435.8, examples_per_sec=197.2
- `param_matched` seed `0`: peak_gpu_memory_mb=971.3, tokens_per_sec=241574.0, examples_per_sec=1048.5
- `prev_attn_reuse` seed `0`: peak_gpu_memory_mb=1355.6, tokens_per_sec=47421.2, examples_per_sec=205.8
- `random_gate` seed `0`: peak_gpu_memory_mb=1186.2, tokens_per_sec=46519.6, examples_per_sec=201.9
- `same_layer_u_gate` seed `0`: peak_gpu_memory_mb=1232.8, tokens_per_sec=36379.6, examples_per_sec=157.9

## Conclusion

inconclusive: performance or diagnostics do not cleanly satisfy the support criteria

## Failure Analysis

Check for alpha values near zero, gate means near one, random_gate parity, and any collapse on the longer sequence profiles. If those appear, the result should be treated as negative even if in-distribution accuracy improves.

## Next Recommended Experiment

If the route gate is promising, rerun the full matrix with three seeds and add a fixed-parameter-count sweep over `d_route` and `d_model`. If it is negative, isolate whether the bottleneck is the synthetic task, the sigmoid gate saturation, or insufficient training horizon.

## Plots

- `W:\Intuition-Labs\experiments\21EYES\e6\plots\accuracy_by_variant.png`
- `W:\Intuition-Labs\experiments\21EYES\e6\plots\accuracy_vs_seq_len.png`
- `W:\Intuition-Labs\experiments\21EYES\e6\plots\accuracy_vs_distractors.png`
- `W:\Intuition-Labs\experiments\21EYES\e6\plots\gate_mean_by_layer.png`
- `W:\Intuition-Labs\experiments\21EYES\e6\plots\gate_entropy_by_layer.png`
- `W:\Intuition-Labs\experiments\21EYES\e6\plots\attention_entropy_by_layer.png`
- `W:\Intuition-Labs\experiments\21EYES\e6\plots\alpha_by_layer.png`
