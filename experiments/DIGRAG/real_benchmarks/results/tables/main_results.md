# LongMemEval — main results

Decoder `Qwen/Qwen2.5-1.5B-Instruct`, 90/seed × 3 seeds, top-k=6, ctx budget 600 tok, dev-calibrated abstention τ=0.4.

| Metric | Lexical (BM25) | Vector | Hybrid | GrepRAG | DIGRAG-raw | DIGRAG-gated |
|---|---|---|---|---|---|---|
| Final-answer accuracy (↑) | 0.296 ±0.037 | 0.259 ±0.028 | 0.285 ±0.029 | 0.256 ±0.016 | 0.348 ±0.021 | 0.267 ±0.024 |
|   · answerable subset (↑) | 0.271 | 0.231 | 0.279 | 0.231 | 0.343 | 0.219 |
|   · abstention (unanswerable) (↑) | 0.632 | 0.632 | 0.368 | 0.579 | 0.421 | 0.895 |
| Temporal/update accuracy (↑) | 0.210 | 0.261 | 0.227 | 0.210 | 0.319 | 0.252 |
| Answerability accuracy (↑) | 0.752 | 0.711 | 0.752 | 0.637 | 0.896 | 0.470 |
| Supporting-evidence recall (↑) | 0.890 | 0.880 | 0.966 | 0.854 | 0.966 | 0.966 |
| Retrieval recall@5 (↑) | 0.814 | 0.787 | 0.914 | 0.780 | 0.914 | 0.914 |
| Source-span correctness (↑) | 0.010 | 0.011 | 0.014 | 0.018 | 0.735 | 0.875 |
| Unsupported-claim rate (↓) | 0.053 | 0.046 | 0.064 | 0.051 | 0.089 | 0.038 |
| Mean context tokens (↓) | 559.100 | 560.300 | 562.300 | 563.000 | 592.900 | 592.900 |
| Mean latency (ms) (↓) | 125.300 | 125.100 | 125.600 | 125.000 | 142.700 | 142.700 |