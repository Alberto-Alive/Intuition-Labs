# Conflict / freshness stress-test (semi-real, separate)

Injected stale contradictory variants; correct = current value. 70/seed × 3 seeds.

| Metric | Lexical (BM25) | Vector | Hybrid | GrepRAG | DIGRAG-raw | DIGRAG-gated |
|---|---|---|---|---|---|---|
| Final-answer accuracy (↑) | 0.386 ±0.051 | 0.357 ±0.053 | 0.343 ±0.065 | 0.238 ±0.037 | 0.452 ±0.058 | 0.452 ±0.058 |
|   · answerable subset (↑) | 0.386 | 0.357 | 0.343 | 0.238 | 0.452 | 0.452 |
|   · abstention (unanswerable) (↑) | — | — | — | — | — | — |
| Temporal/update accuracy (↑) | 0.386 | 0.357 | 0.343 | 0.238 | 0.452 | 0.452 |
| Answerability accuracy (↑) | 0.429 | 0.424 | 0.409 | 0.295 | 0.576 | 0.576 |
| Supporting-evidence recall (↑) | 0.867 | 0.619 | 0.781 | 0.719 | 0.781 | 0.781 |
| Retrieval recall@5 (↑) | 0.705 | 0.452 | 0.624 | 0.581 | 0.624 | 0.624 |
| Source-span correctness (↑) | 0.422 | 0.303 | 0.454 | 0.339 | 0.529 | 0.529 |
| Unsupported-claim rate (↓) | 0.043 | 0.041 | 0.062 | 0.047 | 0.128 | 0.128 |
| Mean context tokens (↓) | 604.600 | 601.500 | 592.500 | 605.800 | 624.000 | 624.000 |
| Mean latency (ms) (↓) | 588.700 | 587.700 | 579.900 | 587.600 | 649.900 | 649.900 |