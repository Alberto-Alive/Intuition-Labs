# Final-answer accuracy by case type

| Case type | Grep | Vector RAG | Hybrid RAG | GrepRAG | DIGIT |
|---|---|---|---|---|---|
| alias | 1.000 | 0.889 | 0.889 | 0.389 | 1.000 |
| ambiguous_keyword | 1.000 | 1.000 | 1.000 | 0.944 | 1.000 |
| approved_rejected | 0.000 | 0.778 | 0.667 | 0.056 | 1.000 |
| code_symbol | 0.667 | 0.944 | 1.000 | 0.444 | 1.000 |
| conflict | 1.000 | 0.889 | 0.833 | 0.722 | 1.000 |
| date | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exception | 0.833 | 0.000 | 0.167 | 0.000 | 1.000 |
| multi_doc | 0.278 | 0.778 | 0.056 | 0.333 | 1.000 |
| policy_condition | 0.667 | 0.667 | 0.833 | 0.667 | 1.000 |
| stale_current | 0.000 | 1.000 | 1.000 | 0.167 | 1.000 |
| table | 0.167 | 0.444 | 0.722 | 0.111 | 1.000 |
| unanswerable | 0.889 | 0.889 | 0.833 | 0.333 | 1.000 |