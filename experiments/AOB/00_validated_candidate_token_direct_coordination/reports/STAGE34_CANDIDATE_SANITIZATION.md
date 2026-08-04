# Stage 3.4 Candidate Sanitization

## Dataset Repair

- Dataset mode: `real_import_restore_candidate_sanitized_v3`.
- Oracle-only compatibility fields are stored in `oracle_metadata`; model-facing `metadata`, candidate text, and candidate attributes are sanitized.
- Locked architecture unchanged: `topk_attention_no_head`.

## Dataset-Only Diagnostics

- Candidate-only baseline: `0.1250`
- Candidate-metadata-only baseline: `0.1250`
- Role-pair-only baseline: `0.1250`
- Family-only baseline: `0.1211`
- View-masked + candidates-visible baseline: `0.1250`
- Lexical-overlap baseline: `0.1250`
- Static frequency baseline: `0.1250`
- Max single-view baseline: `0.1250`
- Max pairwise structured oracle: `1.0000`
- All-role structured oracle: `1.0000`
- Dataset gates passed: `True`

## Tiny CUDA Smoke

| seed | trainable | frozen | text | raw | candidate-only | view-masked | role-shuffled | oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1250 | 0.1250 | 0.1250 | 0.1055 | 0.1250 | 0.1250 | 0.1250 | 1.0000 |

## Decision

- Candidate metadata fully removed from model inputs: `True`
- Candidate-only collapsed: `True`
- View-masked collapsed: `True`
- Role-pair-only collapsed: `True`
- Oracle remains high: `True`
- Smoke passed: `True`
- Full validation justified now: `False`
