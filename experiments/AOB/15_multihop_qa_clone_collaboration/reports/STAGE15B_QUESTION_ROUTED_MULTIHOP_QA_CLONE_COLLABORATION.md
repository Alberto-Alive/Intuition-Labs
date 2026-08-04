# Stage 15B Question-Routed Multihop QA Clone Collaboration (best_from_screen:learned_distinct_multi_support)

## Scale Comparison

N | seq_len | EM | F1 | clone_cosine | meta_weight_balance | joint_beats_single | answer_window_hit
--- | ---: | ---: | ---: | ---: | ---: | --- | ---: 
4 | 128 | 0.0781 | 0.1536 | 0.668218 | 0.000365 | True | 1.0000
8 | 256 | 0.0156 | 0.0511 | 0.628233 | 0.000067 | False | 1.0000
16 | 512 | 0.0000 | 0.0279 | 0.661184 | 0.000058 | False | 1.0000

## Scale N=4
- Context sequence length: `128`
- Window size: `32`
- Exact match: `0.0781`
- F1: `0.1536`
- Clone cosine: `0.668218`
- Meta-attention mean weights: `[0.2506, 0.2498, 0.2497, 0.2499]`
- Meta-attention weight balance std: `0.000365`
- Routing selected-window mass: `[0.25, 0.25, 0.25, 0.25]`
- Routing answer-window hit rate: `1.0000`
- Routing support-window hit rate: `1.0000`
- Per-clone answer coverage: `[0.4219, 0.3281, 0.1719, 0.0781]`
- Joint beats best single clone F1: `True`
- Best single clone F1: `0.1431`
- Per-clone F1: `[0.1431, 0.0526, 0.0349, 0.0563]`

## Scale N=8
- Context sequence length: `256`
- Window size: `32`
- Exact match: `0.0156`
- F1: `0.0511`
- Clone cosine: `0.628233`
- Meta-attention mean weights: `[0.1249, 0.1249, 0.125, 0.125, 0.1251, 0.1251, 0.125, 0.125]`
- Meta-attention weight balance std: `0.000067`
- Routing selected-window mass: `[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]`
- Routing answer-window hit rate: `1.0000`
- Routing support-window hit rate: `1.0000`
- Per-clone answer coverage: `[0.2031, 0.1562, 0.125, 0.125, 0.0625, 0.1406, 0.1094, 0.0781]`
- Joint beats best single clone F1: `False`
- Best single clone F1: `0.0753`
- Per-clone F1: `[0.0294, 0.0753, 0.0463, 0.0482, 0.0177, 0.0575, 0.0385, 0.0441]`

## Scale N=16
- Context sequence length: `512`
- Window size: `32`
- Exact match: `0.0000`
- F1: `0.0279`
- Clone cosine: `0.661184`
- Meta-attention mean weights: `[0.0626, 0.0626, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0624, 0.0625, 0.0624, 0.0625, 0.0624, 0.0624, 0.0625, 0.0625]`
- Meta-attention weight balance std: `0.000058`
- Routing selected-window mass: `[0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625]`
- Routing answer-window hit rate: `1.0000`
- Routing support-window hit rate: `1.0000`
- Per-clone answer coverage: `[0.1562, 0.0781, 0.1094, 0.125, 0.0469, 0.0312, 0.0625, 0.0469, 0.0938, 0.0781, 0.0156, 0.0312, 0.0156, 0.0156, 0.0781, 0.0156]`
- Joint beats best single clone F1: `False`
- Best single clone F1: `0.0455`
- Per-clone F1: `[0.0275, 0.0268, 0.0324, 0.0281, 0.0207, 0.0161, 0.0454, 0.0393, 0.0154, 0.0393, 0.0397, 0.0185, 0.0427, 0.0157, 0.0174, 0.0455]`
