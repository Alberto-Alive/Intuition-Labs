# Stage 13 Joint Clone Next-Token Collaboration

## Setup
- Clones: `4`
- Hidden size / layers: `64` / `3`
- Sequence length: `128`
- Vocab size: `4096`
- Dataset sequences: `{'train_sequences': 512, 'dev_sequences': 128, 'test_sequences': 128}`

## Final Held-Out Metrics
- Joint perplexity: `1.1391`
- Per-clone perplexity: `[1.1391, 1.1391, 1.1391, 1.139]`
- Joint beats best single clone: `False`
- Mean off-diagonal clone cosine: `1.0000`
- Attention entropy per clone: `[1.2512, 1.2515, 1.2512, 1.2515]`
- Pairwise attention KL overall mean: `0.029265`
- Bias-ablation attention shift KL: `0.006180`
- Joint perplexity without learned bias: `1.1393`
- Joint weights: `[0.2542, 0.2477, 0.2491, 0.2491]`

## Bias Norms
- Layer 0: l2=14.8627, mean_abs=0.029024, per_clone_l2=[7.4425, 7.4098, 7.4394, 7.4335]
- Layer 1: l2=17.3356, mean_abs=0.033236, per_clone_l2=[8.6954, 8.6661, 8.6409, 8.6688]
- Layer 2: l2=16.0103, mean_abs=0.030829, per_clone_l2=[7.9985, 7.9892, 8.0076, 8.0252]

## Training Trend
- Initial dev joint perplexity: `32.9941`
- Final dev joint perplexity: `1.1406`
- Initial clone perplexity spread: `0.0112`
- Final clone perplexity spread: `0.0002`
- Initial mean off-diagonal clone cosine: `1.0000`
- Final mean off-diagonal clone cosine: `1.0000`
