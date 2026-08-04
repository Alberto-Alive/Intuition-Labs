# Stage 15 Multihop QA Clone Collaboration

## Scale Comparison

N | seq_len | EM | F1 | clone_cosine | meta_weight_balance | joint_beats_single
--- | ---: | ---: | ---: | ---: | ---: | ---
4 | 128 | 0.0859 | 0.1344 | 0.958575 | 0.001251 | False
8 | 256 | 0.0234 | 0.0895 | 0.735461 | 0.000315 | True
16 | 512 | 0.0078 | 0.0484 | 0.869647 | 0.000133 | False

## Scale N=4
- Context sequence length: `128`
- Window size: `32`
- Exact match: `0.0859`
- F1: `0.1344`
- Clone cosine: `0.958575`
- Meta-attention mean weights: `[0.2511, 0.2504, 0.2479, 0.2506]`
- Meta-attention weight balance std: `0.001251`
- Per-clone answer coverage: `[0.4677, 0.2258, 0.1452, 0.1613]`
- Joint beats best single clone F1: `False`
- Best single clone F1: `0.1344`
- Per-clone F1: `[0.1344, 0.1344, 0.1344, 0.1344]`

## Scale N=8
- Context sequence length: `256`
- Window size: `32`
- Exact match: `0.0234`
- F1: `0.0895`
- Clone cosine: `0.735461`
- Meta-attention mean weights: `[0.1251, 0.1254, 0.1248, 0.1252, 0.1254, 0.1247, 0.1247, 0.1246]`
- Meta-attention weight balance std: `0.000315`
- Per-clone answer coverage: `[0.2705, 0.2131, 0.0984, 0.123, 0.0246, 0.123, 0.0902, 0.0574]`
- Joint beats best single clone F1: `True`
- Best single clone F1: `0.0845`
- Per-clone F1: `[0.0837, 0.0837, 0.0845, 0.0845, 0.0845, 0.0845, 0.0845, 0.0845]`

## Scale N=16
- Context sequence length: `512`
- Window size: `32`
- Exact match: `0.0078`
- F1: `0.0484`
- Clone cosine: `0.869647`
- Meta-attention mean weights: `[0.0626, 0.0626, 0.0623, 0.0625, 0.0626, 0.0622, 0.0624, 0.0623, 0.0625, 0.0625, 0.0627, 0.0625, 0.0627, 0.0624, 0.0625, 0.0627]`
- Meta-attention weight balance std: `0.000133`
- Per-clone answer coverage: `[0.1967, 0.123, 0.082, 0.0656, 0.041, 0.0902, 0.0656, 0.0492, 0.0492, 0.0574, 0.0328, 0.0328, 0.0246, 0.0082, 0.0656, 0.0164]`
- Joint beats best single clone F1: `False`
- Best single clone F1: `0.0484`
- Per-clone F1: `[0.0484, 0.0484, 0.0484, 0.0484, 0.0484, 0.0484, 0.0484, 0.0484, 0.0484, 0.0484, 0.0484, 0.0484, 0.0484, 0.0484, 0.0484, 0.0484]`
