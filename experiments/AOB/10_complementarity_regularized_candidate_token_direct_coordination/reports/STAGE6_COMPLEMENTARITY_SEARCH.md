# Stage 6 Complementarity Search

## Scope

- Fixed benchmark: Stage 3.8 schema-aware balanced_categories_v3.
- Dataset labels, oracle metadata separation, leakage controls, schema-aware corruption controls, and frozen comparator policy are unchanged.
- Baseline reference: Stage 4 `candidate_token_direct_lr3e4_clip1`, trainable `0.7892`, frozen `0.1518`, delta `+0.6374`.
- Selection uses `full_model_accuracy - best_single_avenue_accuracy` on dev metrics from cheap and medium phases only.
- Final success is not claimed unless the selected 10-seed run passes the complementarity gates and all Stage 4 controls.

## Search Space

| variant | family | d | adv | red | red lambda | coord | dropout | dominance |
|---|---|---:|---:|---|---:|---|---:|---|
| stage5_reference_dropout05 | reference | 0 | 0 | none | 0 | standard_attention | 0.05 | off |
| bottleneck_d8 | message_bottleneck_sweep | 8 | 0 | none | 0 | standard_attention | 0.05 | off |
| bottleneck_d16 | message_bottleneck_sweep | 16 | 0 | none | 0 | standard_attention | 0.05 | off |
| bottleneck_d32 | message_bottleneck_sweep | 32 | 0 | none | 0 | standard_attention | 0.05 | off |
| bottleneck_d64 | message_bottleneck_sweep | 64 | 0 | none | 0 | standard_attention | 0.05 | off |
| adv_lambda_0p01 | single_avenue_adversarial_heads | 32 | 0.01 | none | 0 | standard_attention | 0.05 | off |
| adv_lambda_0p03 | single_avenue_adversarial_heads | 32 | 0.03 | none | 0 | standard_attention | 0.05 | off |
| adv_lambda_0p1 | single_avenue_adversarial_heads | 32 | 0.1 | none | 0 | standard_attention | 0.05 | off |
| adv_lambda_0p3 | single_avenue_adversarial_heads | 32 | 0.3 | none | 0 | standard_attention | 0.05 | off |
| red_cosine_l001_standard | redundancy_dominance_penalty | 32 | 0 | cosine | 0.001 | standard_attention | 0.10 | off |
| red_cosine_l003_standard_dominance_weak | redundancy_dominance_penalty | 32 | 0 | cosine | 0.003 | standard_attention | 0.10 | weak |
| red_orthogonal_l01_gated | redundancy_dominance_penalty | 32 | 0 | orthogonality | 0.01 | gated_attention | 0.10 | weak |
| red_covariance_l003_poe | redundancy_dominance_penalty | 32 | 0 | covariance | 0.003 | product_of_experts | 0.10 | medium |
| combo_d16_adv003_red003_weakdom | combined_top_pick | 16 | 0.03 | cosine | 0.003 | standard_attention | 0.10 | weak |

## Pre-Run Controls

- Source: `computed_for_stage6`
- Shortcut diagnostics pass: `True`

| gate | pass |
|---|---|
| all_role_oracle | `True` |
| candidate_metadata_only | `True` |
| candidate_only | `True` |
| evidence_only_no_candidates_baseline | `True` |
| lexical_overlap | `True` |
| null_evidence_values_baseline | `True` |
| positive_control_learnability | `True` |
| role_pair_only | `True` |
| schema_only_baseline | `True` |
| static_frequency | `True` |
| view_masked_candidates_visible | `True` |

## Phase Summaries

### Cheap

| variant | n | dev train | best single | comp score | dev frozen | dev delta | controls | invariance | selected-valid |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| stage5_reference_dropout05 | 3 | 0.6966 | 0.6237 | 0.0729 | 0.1732 | 0.5234 | `True` | `True` | `True` |
| bottleneck_d8 | 3 | 0.7031 | 0.3906 | 0.3125 | 0.1771 | 0.5260 | `True` | `True` | `False` |
| bottleneck_d16 | 3 | 0.4518 | 0.3607 | 0.0911 | 0.1732 | 0.2786 | `True` | `True` | `False` |
| bottleneck_d32 | 3 | 0.5885 | 0.3359 | 0.2526 | 0.1732 | 0.4154 | `True` | `True` | `False` |
| bottleneck_d64 | 3 | 0.5404 | 0.2643 | 0.2760 | 0.1732 | 0.3672 | `True` | `True` | `False` |
| adv_lambda_0p01 | 3 | 0.5833 | 0.5065 | 0.0768 | 0.1706 | 0.4128 | `True` | `True` | `False` |
| adv_lambda_0p03 | 3 | 0.1536 | 0.1615 | -0.0078 | 0.1680 | -0.0143 | `True` | `True` | `False` |
| adv_lambda_0p1 | 3 | 0.1536 | 0.1628 | -0.0091 | 0.1680 | -0.0143 | `True` | `True` | `False` |
| adv_lambda_0p3 | 3 | 0.1628 | 0.1615 | 0.0013 | 0.1680 | -0.0052 | `True` | `True` | `False` |
| red_cosine_l001_standard | 3 | 0.4310 | 0.3451 | 0.0859 | 0.1680 | 0.2630 | `True` | `True` | `False` |
| red_cosine_l003_standard_dominance_weak | 3 | 0.4284 | 0.3776 | 0.0508 | 0.1680 | 0.2604 | `True` | `True` | `False` |
| red_orthogonal_l01_gated | 3 | 0.5260 | 0.2995 | 0.2266 | 0.1289 | 0.3971 | `True` | `True` | `False` |
| combo_d16_adv003_red003_weakdom | 3 | 0.3099 | 0.1888 | 0.1211 | 0.1732 | 0.1367 | `True` | `True` | `False` |
| red_covariance_l003_poe | 3 | 1.0000 | 0.9531 | 0.0469 | 0.3333 | 0.6667 | `True` | `False` | `False` |

## Selected Variants

- Selected after cheap: `[]`
- Selected after medium: `[]`

## Final Validation

- Final validation has not run; this report does not claim Stage 6 success.

## Final Gates

| criterion | pass | value |
|---|---|---|
| completed seeds >= 10 | `False` | `0` |

## Failure Summary

- Final Stage 6 success is not claimed.
- Failed final gates: `completed seeds >= 10`.

## Conservative Interpretation

- Stage 6 final gates passed: `False`.
- Do not claim Stage 6 success unless the selected final variant passes all complementarity, baseline, control, invariance, and leakage gates.
