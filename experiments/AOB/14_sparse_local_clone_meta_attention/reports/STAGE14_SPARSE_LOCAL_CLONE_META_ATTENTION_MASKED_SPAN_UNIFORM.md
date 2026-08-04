# Stage 14b Masked-Span Local Clone Meta-Attention

## Setup
- Mask strategy / span length: `uniform` / `4`
- Clones: `4`
- Hidden size / layers: `64` / `3`
- Sequence length / window size: `64` / `16`
- Vocab size: `742`
- Dataset sequences: `{'train_sequences': 19826, 'dev_sequences': 4967, 'test_sequences': 4956}`
- Best epoch: `12`

## Final Held-Out Metrics
- Joint perplexity: `8.5558`
- Per-clone perplexity: `[6846.8539, 1076.3519, 1315.9002, 6306.3206]`
- Joint beats best single clone: `True`
- Mean off-diagonal clone cosine: `0.449747`
- Clone cosine dropped vs Stage 13 `0.99997`: `True` (delta `-0.550223`)
- Attention entropy per clone: `[2.2271, 2.2799, 2.224, 2.3038]`
- Meta-attention mean weights: `[0.2442, 0.262, 0.2732, 0.2206]`
- Meta-attention std weights: `[0.28, 0.2728, 0.2747, 0.2758]`
- Meta-attention mean entropy: `0.8537`
- Meta-attention mean max weight: `0.6789`
- Target window distribution: `[0.2678, 0.2607, 0.2637, 0.2078]`
- Clone window outside-mass mean / max: `0.000000` / `0.000000`

## Fixed Windows
- Clone 0: `[0:16]`
- Clone 1: `[16:32]`
- Clone 2: `[32:48]`
- Clone 3: `[48:64]`

## Training Trend
- Initial dev joint perplexity: `39.1617`
- Final dev joint perplexity: `8.2503`
- Initial clone perplexity spread: `0.4589`
- Final clone perplexity spread: `5070.7679`
- Initial mean off-diagonal clone cosine: `0.986145`
- Final mean off-diagonal clone cosine: `0.448763`
