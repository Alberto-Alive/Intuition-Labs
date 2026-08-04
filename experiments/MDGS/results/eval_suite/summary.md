# Extrapolation Safety Evaluation

Corpus fingerprint: `8d0bb8702a0871d5`
Seeds: `123, 231, 347, 451, 569`

## Safety Metrics

| Metric | E1 | E2 | Delta (E2-E1) |
| --- | --- | --- | --- |
| `unsafe_success_rate_label` | 0.063 +- 0.000 | 0.112 +- 0.056 | +0.048 |
| `unsafe_success_rate_correctness` | 0.063 +- 0.000 | 0.016 +- 0.002 | -0.047 |
| `success_precision_vs_is_correct` | 0.937 +- 0.000 | 0.984 +- 0.002 | +0.047 |
| `success_auroc` | 0.752 +- 0.005 | 0.722 +- 0.084 | -0.030 |
| `success_auprc` | 0.951 +- 0.006 | 0.952 +- 0.016 | +0.000 |
| `success_brier` | 0.263 +- 0.004 | 0.641 +- 0.025 | +0.378 |
| `success_ece_10bin` | 0.260 +- 0.007 | 0.677 +- 0.022 | +0.417 |
| `success_label_auroc` | 0.960 +- 0.003 | 0.974 +- 0.004 | +0.014 |
| `success_label_auprc` | 0.963 +- 0.009 | 0.937 +- 0.009 | -0.026 |
| `success_label_brier` | 0.047 +- 0.002 | 0.060 +- 0.004 | +0.013 |
| `success_label_ece_10bin` | 0.046 +- 0.007 | 0.054 +- 0.012 | +0.008 |
| `nonfailure_correct_auroc` | 0.761 +- 0.009 | 0.774 +- 0.020 | +0.013 |
| `nonfailure_correct_auprc` | 0.951 +- 0.007 | 0.949 +- 0.010 | -0.001 |
| `nonfailure_correct_brier` | 0.113 +- 0.012 | 0.093 +- 0.001 | -0.020 |
| `nonfailure_correct_ece_10bin` | 0.102 +- 0.025 | 0.080 +- 0.005 | -0.022 |
| `failure_incorrect_auroc` | 0.761 +- 0.009 | 0.774 +- 0.020 | +0.013 |
| `failure_incorrect_auprc` | 0.313 +- 0.018 | 0.426 +- 0.011 | +0.113 |
| `failure_incorrect_brier` | 0.113 +- 0.012 | 0.093 +- 0.001 | -0.020 |
| `failure_incorrect_ece_10bin` | 0.102 +- 0.025 | 0.080 +- 0.005 | -0.022 |

## Label Recovery

| Metric | E1 | E2 | Delta (E2-E1) |
| --- | --- | --- | --- |
| `trajectory_acc` | 0.614 +- 0.112 | 0.658 +- 0.121 | +0.045 |
| `trajectory_macro_f1` | 0.499 +- 0.125 | 0.633 +- 0.127 | +0.134 |
| `pattern_acc` | 0.967 +- 0.004 | 0.967 +- 0.003 | -0.001 |
| `pattern_macro_f1` | 0.951 +- 0.005 | 0.949 +- 0.005 | -0.002 |
| `confidence_acc` | 0.952 +- 0.016 | 0.945 +- 0.017 | -0.007 |
| `confidence_macro_f1` | 0.953 +- 0.016 | 0.946 +- 0.017 | -0.007 |
| `outcome_acc` | 0.843 +- 0.021 | 0.877 +- 0.015 | +0.033 |
| `outcome_macro_f1` | 0.684 +- 0.021 | 0.808 +- 0.034 | +0.123 |
| `failure_recall` | 0.411 +- 0.088 | 0.812 +- 0.069 | +0.401 |
| `failure_likely_recall` | 0.411 +- 0.088 | 0.812 +- 0.069 | +0.401 |
| `joint_acc` | 0.495 +- 0.080 | 0.543 +- 0.111 | +0.048 |

## Monotonicity

| Metric | E1 | E2 | Delta (E2-E1) |
| --- | --- | --- | --- |
| `monotonicity_violation_rate` | 0.450 +- 0.077 | 0.443 +- 0.038 | -0.007 |
| `all_supported_monotonicity_violation_rate` | 0.397 +- 0.075 | 0.339 +- 0.005 | -0.057 |
| `nonfailure_monotonicity_violation_rate` | 0.343 +- 0.079 | 0.247 +- 0.045 | -0.097 |
| `all_supported_nonfailure_monotonicity_violation_rate` | 0.343 +- 0.079 | 0.212 +- 0.060 | -0.132 |

## Baselines

- E1 entropy-threshold outcome accuracy: `0.453`
- E1 logistic-to-labels outcome macro-F1: `0.698`
- E1 MLP-to-labels outcome macro-F1: `0.653`
- E2 entropy-threshold outcome accuracy: `0.261`
- E2 logistic-to-labels outcome macro-F1: `0.617`
- E2 MLP-to-labels outcome macro-F1: `0.818`
- Logistic-to-correctness AUROC: `0.751`
- MLP-to-correctness AUROC: `0.704`

## Label Audit

- E1 P(correct | SUCCESS_LIKELY): `1.000`
- E1 P(correct | UNCERTAIN): `1.000`
- E1 P(correct | FAILURE_LIKELY): `0.000`
- E1 correctness separation (SUCCESS - UNCERTAIN): `0.000`
- E1 max outcome-share swing under +-5% threshold shifts: `0.031`
- E2 P(correct | SUCCESS_LIKELY): `0.998`
- E2 P(correct | UNCERTAIN): `0.900`
- E2 P(correct | FAILURE_LIKELY): `0.420`
- E2 correctness separation (SUCCESS - UNCERTAIN): `0.097`
- E2 max outcome-share swing under +-5% threshold shifts: `0.208`

## Acceptance

- absolute_floor_failure_recall_ok: `True`
- absolute_floor_outcome_macro_f1_ok: `True`
- absolute_floor_outcome_acc_ok: `True`
- absolute_floor_trajectory_acc_ok: `False`
- safety_win: `False`
- non_regression_outcome_macro_f1: `True`
- non_regression_confidence_macro_f1: `True`
- monotonicity_ok: `False`
- nonfailure_monotonicity_ok: `False`
- label_revision_trigger: `True`