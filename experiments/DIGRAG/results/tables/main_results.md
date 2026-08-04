# Main results

Dataset: 760 docs, 216 questions. Decoder: shared `Qwen/Qwen2.5-1.5B-Instruct`.

| Metric | Grep | Vector RAG | Hybrid RAG | GrepRAG | DIGIT | DIGIT (no gate) |
|---|---|---|---|---|---|---|
| Final-answer accuracy (↑) | 0.625 | 0.773 | 0.750 | 0.431 | 1.000 | 0.801 |
| Exact-value accuracy (↑) | 0.587 | 0.865 | 0.809 | 0.484 | 1.000 | 0.873 |
| Source-span correctness (↑) | 0.450 | 0.737 | 0.884 | 0.540 | 0.909 | 0.904 |
| Unsupported-claim rate (↓) | 0.006 | 0.008 | 0.002 | 0.034 | 0.000 | 0.000 |
| Conflict-detection accuracy (↑) | 0.861 | 0.898 | 0.917 | 0.861 | 1.000 | 0.875 |
| Missing-evidence recall (↑) | 0.889 | 0.889 | 0.833 | 0.333 | 1.000 | 1.000 |
| Answerability accuracy (↑) | 0.750 | 0.787 | 0.875 | 0.727 | 1.000 | 0.884 |
| Retrieval recall@5 (↑) | 0.955 | 0.856 | 0.990 | 0.783 | 0.955 | 0.955 |
| Mean context tokens (↓) | 392.7 | 481.2 | 504.6 | 383.2 | 515.6 | 515.6 |
| Mean output tokens (↓) | 63.838 | 74.074 | 73.463 | 62.986 | 58.463 | 58.463 |
| Mean latency (ms) (↓) | 248.6 | 307.4 | 354.8 | 241.8 | 320.3 | 320.3 |
| Cost ($/question) (↓) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |