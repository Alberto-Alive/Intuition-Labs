# Stage 14b Masked-Span Local Clone Meta-Attention

## Setup
- Mask strategy / span length: `boundary` / `4`
- Clones: `4`
- Hidden size / layers: `64` / `3`
- Sequence length / window size: `64` / `16`
- Vocab size: `742`
- Dataset sequences: `{'train_sequences': 19826, 'dev_sequences': 4967, 'test_sequences': 4956}`
- Best epoch: `12`

## Final Held-Out Metrics
- Joint perplexity: `2.7246`
- Per-clone perplexity: `[904525861.9217, 212471.071, 4512347.5956, 4132952838.1623]`
- Joint beats best single clone: `True`
- Mean off-diagonal clone cosine: `0.313757`
- Clone cosine dropped vs Stage 13 `0.99997`: `True` (delta `-0.686213`)
- Attention entropy per clone: `[1.6103, 1.4362, 1.7233, 0.8422]`
- Meta-attention mean weights: `[0.19, 0.3151, 0.3125, 0.1823]`
- Meta-attention std weights: `[0.2688, 0.3224, 0.3154, 0.2613]`
- Meta-attention mean entropy: `0.7091`
- Meta-attention mean max weight: `0.7111`
- Target window distribution: `[0.3333, 0.3333, 0.3333, 0.0]`
- Clone window outside-mass mean / max: `0.000000` / `0.000000`

## Fixed Windows
- Clone 0: `[0:16]`
- Clone 1: `[16:32]`
- Clone 2: `[32:48]`
- Clone 3: `[48:64]`

## Training Trend
- Initial dev joint perplexity: `30.0349`
- Final dev joint perplexity: `2.7559`
- Initial clone perplexity spread: `16.9790`
- Final clone perplexity spread: `4298212504.0556`
- Initial mean off-diagonal clone cosine: `0.650950`
- Final mean off-diagonal clone cosine: `0.312648`
