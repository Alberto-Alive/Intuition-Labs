# LongMemEval — ablation study

| Metric | Full DIGRAG (gated) | − gate (raw packet) | − typed buckets (context only) | − context (buckets only) | − temporal/freshness | − conflict flags | − constrained decoding | chunk-stuffing (no packet) |
|---|---|---|---|---|---|---|---|---|
| Final-answer accuracy (↑) | 0.327 ±0.000 | 0.391 ±0.000 | 0.282 ±0.000 | 0.127 ±0.000 | 0.327 ±0.000 | 0.336 ±0.000 | 0.364 ±0.000 | 0.318 ±0.000 |
|   · answerable subset (↑) | 0.282 | 0.388 | 0.243 | 0.097 | 0.282 | 0.291 | 0.349 | 0.291 |
|   · abstention (unanswerable) (↑) | 1.000 | 0.429 | 0.857 | 0.571 | 1.000 | 1.000 | 0.571 | 0.714 |
| Temporal/update accuracy (↑) | 0.340 | 0.383 | 0.277 | 0.106 | 0.340 | 0.362 | 0.362 | 0.298 |
| Answerability accuracy (↑) | 0.518 | 0.891 | 0.482 | 0.500 | 0.518 | 0.527 | 0.873 | 0.791 |
| Supporting-evidence recall (↑) | 0.973 | 0.973 | 0.973 | 0.973 | 0.973 | 0.973 | 0.973 | 0.973 |
| Retrieval recall@5 (↑) | 0.932 | 0.932 | 0.932 | 0.932 | 0.932 | 0.932 | 0.932 | 0.932 |
| Source-span correctness (↑) | 0.880 | 0.737 | 0.833 | 0.852 | 0.780 | 0.824 | 0.232 | 0.000 |
| Unsupported-claim rate (↓) | 0.036 | 0.093 | 0.026 | 0.049 | 0.030 | 0.039 | 0.080 | 0.051 |
| Mean context tokens (↓) | 589.400 | 589.400 | 590.300 | 409.300 | 594.000 | 592.800 | 589.400 | 560.700 |
| Mean latency (ms) (↓) | 162.500 | 162.500 | 162.300 | 126.500 | 163.400 | 163.000 | 149.800 | 142.900 |

**Δ final-answer accuracy vs full DIGRAG** (most harmful removal first):

- − context (buckets only): -0.200
- − typed buckets (context only): -0.045
- chunk-stuffing (no packet): -0.009
- − temporal/freshness: +0.000
- − conflict flags: +0.009
- − constrained decoding: +0.036
- − gate (raw packet): +0.064