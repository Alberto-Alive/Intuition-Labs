# Stage 14b Masked-Span Local Clone Meta-Attention

## Setup
- Mask strategy / span length: `center` / `4`
- Clones: `4`
- Hidden size / layers: `64` / `3`
- Sequence length / window size: `64` / `16`
- Vocab size: `742`
- Dataset sequences: `{'train_sequences': 19826, 'dev_sequences': 4967, 'test_sequences': 4956}`
- Best epoch: `12`

## Final Held-Out Metrics
- Joint perplexity: `1.7364`
- Per-clone perplexity: `[20052826.0011, 941616.6057, 5458720.672, 68102330623.2987]`
- Joint beats best single clone: `True`
- Mean off-diagonal clone cosine: `0.343509`
- Clone cosine dropped vs Stage 13 `0.99997`: `True` (delta `-0.656461`)
- Attention entropy per clone: `[1.4523, 1.3393, 1.2138, 1.5495]`
- Meta-attention mean weights: `[0.1585, 0.3254, 0.3568, 0.1593]`
- Meta-attention std weights: `[0.215, 0.3262, 0.3251, 0.2213]`
- Meta-attention mean entropy: `0.7092`
- Meta-attention mean max weight: `0.6803`
- Target window distribution: `[0.0, 1.0, 0.0, 0.0]`
- Clone window outside-mass mean / max: `0.000000` / `0.000000`

## Fixed Windows
- Clone 0: `[0:16]`
- Clone 1: `[16:32]`
- Clone 2: `[32:48]`
- Clone 3: `[48:64]`

## Training Trend
- Initial dev joint perplexity: `8.2210`
- Final dev joint perplexity: `1.7597`
- Initial clone perplexity spread: `122.9715`
- Final clone perplexity spread: `75665010383.0347`
- Initial mean off-diagonal clone cosine: `0.667888`
- Final mean off-diagonal clone cosine: `0.343022`
