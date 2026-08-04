# Stage 16 Exploratory Combination Search (best_from_screen:scratch_continuous_distinct_multi_support)

## Scale Comparison

N | seq_len | EM | F1 | clone_cosine | meta_weight_balance | joint_beats_single | answer_window_hit
--- | ---: | ---: | ---: | ---: | ---: | --- | ---: 
4 | 128 | 0.0938 | 0.1748 | 0.681778 | 0.000245 | True | 1.0000
8 | 256 | 0.0000 | 0.0329 | 0.633961 | 0.000152 | False | 1.0000
16 | 512 | 0.0156 | 0.0728 | 0.469938 | 0.000018 | True | 1.0000

## Scale N=4
- Context sequence length: `128`
- Window size: `32`
- Exact match: `0.0938`
- F1: `0.1748`
- Clone cosine: `0.681778`
- Meta-attention mean weights: `[0.2502, 0.2503, 0.2498, 0.2497]`
- Meta-attention weight balance std: `0.000245`
- Communication mode: `continuous`
- Message-token mass: `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`
- Routing selected-window mass: `[0.25, 0.25, 0.25, 0.25]`
- Routing answer-window hit rate: `1.0000`
- Routing support-window hit rate: `1.0000`
- Per-clone answer coverage: `[0.4219, 0.3281, 0.1719, 0.0781]`
- Joint beats best single clone F1: `True`
- Best single clone F1: `0.1568`
- Per-clone F1: `[0.1568, 0.0656, 0.0482, 0.034]`

## Scale N=8
- Context sequence length: `256`
- Window size: `32`
- Exact match: `0.0000`
- F1: `0.0329`
- Clone cosine: `0.633961`
- Meta-attention mean weights: `[0.1253, 0.1251, 0.1251, 0.125, 0.1249, 0.1249, 0.1248, 0.1249]`
- Meta-attention weight balance std: `0.000152`
- Communication mode: `continuous`
- Message-token mass: `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`
- Routing selected-window mass: `[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]`
- Routing answer-window hit rate: `1.0000`
- Routing support-window hit rate: `1.0000`
- Per-clone answer coverage: `[0.2031, 0.1562, 0.125, 0.125, 0.0625, 0.1406, 0.1094, 0.0781]`
- Joint beats best single clone F1: `False`
- Best single clone F1: `0.0690`
- Per-clone F1: `[0.0351, 0.0499, 0.0302, 0.052, 0.0546, 0.069, 0.0466, 0.0391]`

## Scale N=16
- Context sequence length: `512`
- Window size: `32`
- Exact match: `0.0156`
- F1: `0.0728`
- Clone cosine: `0.469938`
- Meta-attention mean weights: `[0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625]`
- Meta-attention weight balance std: `0.000018`
- Communication mode: `continuous`
- Message-token mass: `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`
- Routing selected-window mass: `[0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625]`
- Routing answer-window hit rate: `1.0000`
- Routing support-window hit rate: `1.0000`
- Per-clone answer coverage: `[0.1562, 0.0781, 0.1094, 0.125, 0.0469, 0.0312, 0.0625, 0.0469, 0.0938, 0.0781, 0.0156, 0.0312, 0.0156, 0.0156, 0.0781, 0.0156]`
- Joint beats best single clone F1: `True`
- Best single clone F1: `0.0488`
- Per-clone F1: `[0.0488, 0.0183, 0.0383, 0.0199, 0.0361, 0.022, 0.0314, 0.0312, 0.0391, 0.0337, 0.0298, 0.048, 0.0258, 0.0258, 0.0095, 0.0369]`
