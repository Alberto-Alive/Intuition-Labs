# Stage 15C Curriculum Question-Routed Multihop QA Clone Collaboration (best_from_screen:curriculum_pretrain_distinct_multi_support)

## Scale Comparison

N | seq_len | EM | F1 | clone_cosine | meta_weight_balance | joint_beats_single | answer_window_hit
--- | ---: | ---: | ---: | ---: | ---: | --- | ---: 
4 | 128 | 0.0938 | 0.1604 | 0.838012 | 0.000478 | True | 1.0000
8 | 256 | 0.0000 | 0.0398 | 0.877096 | 0.000425 | False | 1.0000
16 | 512 | 0.0000 | 0.0336 | 0.893539 | 0.000042 | False | 1.0000

## Scale N=4
- Context sequence length: `128`
- Window size: `32`
- Exact match: `0.0938`
- F1: `0.1604`
- Clone cosine: `0.838012`
- Meta-attention mean weights: `[0.2507, 0.2497, 0.2494, 0.2502]`
- Meta-attention weight balance std: `0.000478`
- Routing selected-window mass: `[0.25, 0.25, 0.25, 0.25]`
- Routing answer-window hit rate: `1.0000`
- Routing support-window hit rate: `1.0000`
- Per-clone answer coverage: `[0.4219, 0.3281, 0.1719, 0.0781]`
- Joint beats best single clone F1: `True`
- Best single clone F1: `0.1151`
- Per-clone F1: `[0.1151, 0.0419, 0.0472, 0.0095]`

## Scale N=8
- Context sequence length: `256`
- Window size: `32`
- Exact match: `0.0000`
- F1: `0.0398`
- Clone cosine: `0.877096`
- Meta-attention mean weights: `[0.1241, 0.1247, 0.1251, 0.1248, 0.1252, 0.1254, 0.1255, 0.1253]`
- Meta-attention weight balance std: `0.000425`
- Routing selected-window mass: `[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]`
- Routing answer-window hit rate: `1.0000`
- Routing support-window hit rate: `1.0000`
- Per-clone answer coverage: `[0.2031, 0.1562, 0.125, 0.125, 0.0625, 0.1406, 0.1094, 0.0781]`
- Joint beats best single clone F1: `False`
- Best single clone F1: `0.0438`
- Per-clone F1: `[0.0438, 0.0221, 0.0109, 0.0318, 0.0257, 0.0323, 0.0292, 0.0234]`

## Scale N=16
- Context sequence length: `512`
- Window size: `32`
- Exact match: `0.0000`
- F1: `0.0336`
- Clone cosine: `0.893539`
- Meta-attention mean weights: `[0.0625, 0.0625, 0.0626, 0.0625, 0.0625, 0.0625, 0.0625, 0.0626, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0624]`
- Meta-attention weight balance std: `0.000042`
- Routing selected-window mass: `[0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625]`
- Routing answer-window hit rate: `1.0000`
- Routing support-window hit rate: `1.0000`
- Per-clone answer coverage: `[0.1562, 0.0781, 0.1094, 0.125, 0.0469, 0.0312, 0.0625, 0.0469, 0.0938, 0.0781, 0.0156, 0.0312, 0.0156, 0.0156, 0.0781, 0.0156]`
- Joint beats best single clone F1: `False`
- Best single clone F1: `0.0430`
- Per-clone F1: `[0.0303, 0.0139, 0.0312, 0.016, 0.0296, 0.028, 0.031, 0.0413, 0.0146, 0.043, 0.0289, 0.0081, 0.0288, 0.0069, 0.022, 0.0411]`
