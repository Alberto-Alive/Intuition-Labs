# Predeclared Aux Validation Rule

This file is written before the validation sweep starts.

`aux` validates only the training-teacher claim:

> MDGS gives the transformer useful auxiliary uncertainty supervision even
> when MDGS is not fused into the final prediction logits.

The verdict does not include an anchor check. `aux` has no anchor claim.

1. Accuracy preservation: mean ID accuracy is no more than 0.005 below the
   baseline mean.
2. OOD uncertainty advantage: mean OOD AUROC by uncertainty, averaged over all
   OOD modes, is at least 0.05 higher than baseline.
3. Risk/calibration behavior: mean risk-coverage AUC is no more than 0.005
   worse than baseline, ECE is no more than 0.02 worse, and NLL is no more than
   0.03 worse.
4. Seed stability: at least 10 seeds are present, and `aux` has a positive
   paired OOD-uncertainty AUROC advantage over baseline on at least 7 seeds.
5. Aux MDGS ablation drop: the real `aux` validation score is at least 0.01
   higher than the mean score of the broken auxiliary MDGS variants
   (`aux_no_mdgs`, `aux_random_mdgs`, `aux_shuffled_mdgs`,
   `aux_detached_mdgs`, `aux_frozen_mdgs`), and at least 4 of those 5 variants
   score below real `aux`.

Validation score is fixed before the run:

```text
accuracy
- 0.05 * nll
- 0.05 * ece
- 0.05 * brier
- 0.50 * risk_coverage_auc
- 2.00 * confident_wrong_90
+ 0.10 * mean_ood_auroc_by_uncertainty
+ 0.10 * mean_ood_auroc_by_negative_support
+ 0.05 * error_auroc_by_uncertainty
```

NaN error-AUROC values, which happen when a model makes no ID errors, are
treated as neutral 0.5 only for this validation score. The raw NaN is preserved
in the metrics JSON.
