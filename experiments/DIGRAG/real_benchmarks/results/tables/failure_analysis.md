## Failure analysis

### LongMemEval: final-answer accuracy by question type

| Question type | Lexical (BM25) | Vector | Hybrid | GrepRAG | DIGRAG-raw | DIGRAG-gated |
|---|---|---|---|---|---|---|
| knowledge-update | 0.277 | 0.362 | 0.277 | 0.319 | 0.489 | 0.404 |
| multi-session | 0.236 | 0.153 | 0.250 | 0.194 | 0.264 | 0.236 |
| single-session-assistant | 0.343 | 0.314 | 0.314 | 0.400 | 0.429 | 0.143 |
| single-session-preference | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| single-session-user | 0.867 | 0.567 | 0.700 | 0.533 | 0.733 | 0.667 |
| temporal-reasoning | 0.167 | 0.194 | 0.194 | 0.139 | 0.208 | 0.153 |

### MultiHop-RAG: final-answer accuracy by question type

| Question type | Lexical (BM25) | Vector | Hybrid | GrepRAG | DIGRAG-raw | DIGRAG-gated |
|---|---|---|---|---|---|---|
| comparison_query | 0.074 | 0.067 | 0.059 | 0.052 | 0.111 | 0.089 |
| inference_query | 0.704 | 0.618 | 0.586 | 0.474 | 0.730 | 0.467 |
| null_query | 0.849 | 0.679 | 0.641 | 0.660 | 0.509 | 0.868 |
| temporal_query | 0.191 | 0.154 | 0.136 | 0.136 | 0.446 | 0.382 |
