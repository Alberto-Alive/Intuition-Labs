# Stage 3.5 Model-Facing Learnability Diagnosis

## Stage 3.4 Inspection

- Model-facing inputs learnable: `False`
- Best Stage 3.4 positive-control test accuracy: `0.1484`
- Stage 3.4 over-sanitized: `True`
- Inspection result: `model-facing records appear over-sanitized or indistinguishable after removing slot/uid artifacts`

## Stage 3.4 Positive Controls

| control | train | dev | test |
|---|---:|---:|---:|
| all_view_bow_candidate_scorer | 0.2266 | 0.1406 | 0.1133 |
| all_view_tfidf_logistic_candidate_scorer | 0.2773 | 0.1484 | 0.1406 |
| small_cross_encoder_all_views_candidate | 0.1172 | 0.2031 | 0.0977 |
| candidate_pair_compatibility_mlp | 0.1250 | 0.1250 | 0.1250 |
| single_agent_full_context_tiny_transformer | 0.4102 | 0.0938 | 0.1250 |
| pairwise_view_text_roles_0_1 | 0.1602 | 0.1094 | 0.1172 |
| pairwise_view_text_roles_0_2 | 0.2578 | 0.1250 | 0.0938 |
| pairwise_view_text_roles_0_3 | 0.1875 | 0.1172 | 0.1484 |
| pairwise_view_text_roles_1_2 | 0.3125 | 0.1250 | 0.0977 |
| pairwise_view_text_roles_1_3 | 0.2188 | 0.1406 | 0.1328 |
| pairwise_view_text_roles_2_3 | 0.3281 | 0.1094 | 0.0977 |

## Stage 3.4b Balanced Representation

- Representation: `balanced_categories_v3`
- Best positive-control test accuracy: `1.0000`
- Candidate-only baseline: `0.1250`
- Candidate-metadata-only baseline: `0.1250`
- View-masked + candidates-visible baseline: `0.1250`
- Lexical-overlap baseline: `0.1250`
- Static frequency baseline: `0.1250`
- All-role oracle: `1.0000`

### Stage 3.4b Positive Controls

| control | train | dev | test |
|---|---:|---:|---:|
| all_view_bow_candidate_scorer | 0.1914 | 0.1562 | 0.1484 |
| all_view_tfidf_logistic_candidate_scorer | 0.1875 | 0.1328 | 0.1406 |
| small_cross_encoder_all_views_candidate | 0.4297 | 0.3750 | 0.2656 |
| candidate_pair_compatibility_mlp | 1.0000 | 1.0000 | 1.0000 |
| single_agent_full_context_tiny_transformer | 0.2148 | 0.1953 | 0.1836 |
| pairwise_view_text_roles_0_1 | 0.2148 | 0.1406 | 0.1562 |
| pairwise_view_text_roles_0_2 | 0.1797 | 0.1406 | 0.0664 |
| pairwise_view_text_roles_0_3 | 0.1836 | 0.1406 | 0.0977 |
| pairwise_view_text_roles_1_2 | 0.2422 | 0.1562 | 0.1055 |
| pairwise_view_text_roles_1_3 | 0.2227 | 0.1641 | 0.1250 |
| pairwise_view_text_roles_2_3 | 0.1758 | 0.1328 | 0.1328 |

### Stage 3.4b Tiny Smoke

- Trainable: `0.0820`
- Frozen: `0.2344`
- Text-only: `0.1250`
- Raw latent: `0.1211`
- Candidate-only: `0.1719`
- View-masked: `0.0469`
- Role-shuffled: `0.0625`
- Oracle: `1.0000`

## Decision

- Stage 3.4b fixes learnability while keeping candidate-only/view-masked near chance: `True`
- Full validation justified: `False`
