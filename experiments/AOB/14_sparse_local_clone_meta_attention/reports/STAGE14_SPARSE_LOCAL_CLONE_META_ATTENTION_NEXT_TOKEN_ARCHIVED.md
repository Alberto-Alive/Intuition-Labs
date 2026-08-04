# Stage 14 Sparse Local Clone Meta-Attention

## Setup
- Clones: `4`
- Hidden size / layers: `64` / `3`
- Sequence length / window size: `64` / `16`
- Vocab size: `4096`
- Dataset sequences: `{'train_sequences': 158282, 'dev_sequences': 39555, 'test_sequences': 39488}`
- Best epoch: `9`

## Final Held-Out Metrics
- Joint perplexity: `1.2756`
- Per-clone perplexity: `[13691.2563, 9315.8272, 6643.3999, 3674.6393]`
- Joint beats best single clone: `True`
- Mean off-diagonal clone cosine: `0.493813`
- Clone cosine dropped vs Stage 13 `0.99997`: `True` (delta `-0.506157`)
- Attention entropy per clone: `[1.4102, 1.4978, 1.1613, 0.7673]`
- Meta-attention mean weights: `[0.0639, 0.0775, 0.1667, 0.692]`
- Meta-attention std weights: `[0.0415, 0.0721, 0.083, 0.1308]`
- Meta-attention mean entropy: `0.8518`
- Meta-attention mean max weight: `0.6932`
- Clone window outside-mass mean / max: `0.000000` / `0.000000`

## Fixed Windows
- Clone 0: `[0:16]`
- Clone 1: `[16:32]`
- Clone 2: `[32:48]`
- Clone 3: `[48:64]`

## Training Trend
- Initial dev joint perplexity: `1.4731`
- Final dev joint perplexity: `1.3729`
- Initial clone perplexity spread: `1737.5501`
- Final clone perplexity spread: `9452.1721`
- Initial mean off-diagonal clone cosine: `0.518484`
- Final mean off-diagonal clone cosine: `0.458751`
