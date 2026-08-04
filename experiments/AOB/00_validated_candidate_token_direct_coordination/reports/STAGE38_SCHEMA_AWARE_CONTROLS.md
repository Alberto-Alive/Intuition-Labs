# Stage 3.8 Schema-Aware Controls

## Scope

- Full 10-seed validation: not run.
- Locked architecture changes: none.
- Default view format: Stage 3.4b schema-aware `balanced_categories_v3`; role/view schema fields are preserved.

## Role Schema Policy

- Role-specific schemas are treated as legitimate program-analysis evidence: `True`.
- Simple runtime role-id shuffling is reclassified as `role_embedding_shuffle`, diagnostic only.
- Expected behavior: it may not collapse when view text itself exposes role identity through schema.

## Replacement Controls

| control | corruption target |
|---|---|
| value_shuffle_within_schema | Preserve each role schema and candidate list; shuffle evidence values across examples within the same role. |
| cross_example_view_bundle_shuffle | Preserve all role schemas; replace the full multi-view evidence bundle with another example's bundle. |
| candidate_evidence_mismatch | Preserve the view bundle; replace candidates from another example and randomize labels. |
| schema_preserved_role_value_shuffle | Preserve role labels and field names; shuffle content values inside each role. |
| null_evidence_values | Preserve schemas and field names; replace evidence values with `MASK`. |
| schema_only | Preserve schemas and field names; remove evidence values while keeping candidates visible. |
| evidence_only_no_candidates | Preserve evidence views; hide candidate identities and representations. |
| candidate_only | Preserve candidate representations; hide private views. |

## Dataset Shortcut Gates

| diagnostic | accuracy | gate |
|---|---:|---|
| candidate_only | 0.1250 | `0.18` |
| candidate_metadata_only | 0.1250 | `0.18` |
| view_masked_candidates_visible | 0.1250 | `0.18` |
| role_pair_only | 0.1250 | `0.18` |
| lexical_overlap | 0.1250 | `0.18` |
| static_frequency | 0.1328 | `0.20` |
| all_role_oracle | 1.0000 | `0.90` |

## Schema-Only Dataset Baselines

| baseline | train | dev | test |
|---|---:|---:|---:|
| schema_only_baseline | 0.1250 | 0.1250 | 0.1250 |
| null_evidence_values_baseline | 0.1250 | 0.1250 | 0.1250 |
| candidate_only_baseline | 0.1250 | 0.1250 | 0.1250 |
| evidence_only_no_candidates_baseline | 0.1250 | 0.1250 | 0.1250 |

## Positive-Control Bridge

| control | train | dev | test |
|---|---:|---:|---:|
| candidate_pair_compatibility_mlp | 1.0000 | 1.0000 | 1.0000 |
| small_cross_encoder_all_views_candidate | 0.5742 | 0.4062 | 0.2852 |
| all_view_tfidf_logistic_candidate_scorer | 0.2734 | 0.1328 | 0.1172 |
| single_agent_full_context_tiny_transformer | 0.2266 | 0.1484 | 0.1641 |

- Best positive-control test accuracy: `1.0000`
- Positive-control learnability confirmed: `True`

## Tiny Smoke

| metric | accuracy |
|---|---:|
| trainable | 0.3789 |
| frozen | 0.0547 |
| text_only | 0.1250 |
| raw_latent | 0.0938 |
| candidate_only | 0.0781 |
| schema_only | 0.1016 |
| value_shuffle_within_schema | 0.1250 |
| cross_example_view_bundle_shuffle | 0.1523 |
| candidate_evidence_mismatch | 0.1523 |
| schema_preserved_role_value_shuffle | 0.1367 |
| null_evidence_values | 0.1172 |
| evidence_only_no_candidates | 0.1250 |
| oracle | 1.0000 |

## Control Rows

| control | trainable | frozen | text | raw | trainable near chance? |
|---|---:|---:|---:|---:|---|
| none | 0.3789 | 0.0547 | 0.1250 | 0.0938 | `False` |
| candidate_only | 0.0781 | 0.0625 | 0.1250 | 0.1133 | `True` |
| view_masked_candidates_visible | 0.1445 | 0.0273 | 0.1250 | 0.1172 | `True` |
| schema_only | 0.1016 | 0.0273 | 0.1250 | 0.1289 | `True` |
| value_shuffle_within_schema | 0.1250 | 0.0352 | 0.1250 | 0.0938 | `True` |
| cross_example_view_bundle_shuffle | 0.1523 | 0.0312 | 0.1250 | 0.1016 | `True` |
| candidate_evidence_mismatch | 0.1523 | 0.1484 | 0.1250 | 0.1445 | `True` |
| schema_preserved_role_value_shuffle | 0.1367 | 0.0312 | 0.1250 | 0.1211 | `True` |
| null_evidence_values | 0.1172 | 0.0273 | 0.1250 | 0.1289 | `True` |
| evidence_only_no_candidates | 0.1250 | 0.1250 | 0.1250 | 0.1250 | `True` |
| role_embedding_shuffle_diagnostic | 0.2852 | 0.0234 | 0.1250 | 0.0820 | `False` |

## Decision

- Schema-aware corruption controls collapse: `True`
- Shortcut diagnostics pass: `True`
- Full validation justified: `True`
