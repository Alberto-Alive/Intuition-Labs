# MultiHop-RAG — generalization

Decoder `Qwen/Qwen2.5-1.5B-Instruct`, 150/seed × 3 seeds.

| Metric | Lexical (BM25) | Vector | Hybrid | GrepRAG | DIGRAG-raw | DIGRAG-gated |
|---|---|---|---|---|---|---|
| Final-answer accuracy (↑) | 0.407 ±0.025 | 0.347 ±0.028 | 0.324 ±0.036 | 0.287 ±0.030 | 0.449 ±0.019 | 0.380 ±0.048 |
|   · answerable subset (↑) | 0.348 | 0.302 | 0.282 | 0.237 | 0.441 | 0.315 |
|   · abstention (unanswerable) (↑) | 0.849 | 0.679 | 0.641 | 0.660 | 0.509 | 0.868 |
| Temporal/update accuracy (↑) | 0.191 | 0.154 | 0.136 | 0.136 | 0.446 | 0.382 |
| Answerability accuracy (↑) | 0.451 | 0.398 | 0.369 | 0.340 | 0.558 | 0.447 |
| Supporting-evidence recall (↑) | 0.859 | 0.693 | 0.711 | 0.647 | 0.711 | 0.711 |
| Retrieval recall@5 (↑) | 0.699 | 0.518 | 0.538 | 0.499 | 0.538 | 0.538 |
| Source-span correctness (↑) | 0.810 | 0.664 | 0.689 | 0.652 | 0.647 | 0.697 |
| Unsupported-claim rate (↓) | 0.055 | 0.061 | 0.061 | 0.045 | 0.116 | 0.088 |
| Mean context tokens (↓) | 606.600 | 605.000 | 603.800 | 607.600 | 629.200 | 629.200 |
| Mean latency (ms) (↓) | 735.400 | 732.800 | 731.100 | 733.500 | 817.400 | 817.400 |