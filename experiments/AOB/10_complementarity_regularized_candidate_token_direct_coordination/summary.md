# Complementarity-Regularized Candidate-Token Direct Coordination

Status: `negative_result`

Track: `mainline`

Source: Stage 4 candidate-token direct coordination plus Stage 5 multi-avenue setup.

## Research Question

Can architectural complementarity regularizers make the four-avenue candidate-token-direct model require multiple avenues, instead of letting one avenue explain most of the result?

## Selection Metric

`full_model_accuracy - best_single_avenue_accuracy`

## Conclusion

No promotion: the full cheap screen selected no non-reference Stage 6 variant, so medium and final validation did not run.

## Run Summary

- Completed rows: `42` cheap, `0` medium, `0` final.
- Selection after cheap: `[]`; the runner stopped before medium with no cheap-stage variant passing selection gates.
- Best non-reference complementarity signal: `bottleneck_d8` with dev train `0.7031`, best single avenue `0.3906`, complementarity score `0.3125`.
- `bottleneck_d8` still failed cheap selection validity with `2` weak dev seeds below `0.60`.
- The Stage 5 reference was cheap selection-valid, but the Stage 6 selector excludes that reference row from promotion.
- The product-of-experts row reached dev train `1.0000`, but best single avenue accuracy was `0.9531` and invariance did not pass.
- Pre-run controls passed; Stage 6 final gates were not reached.

## Main Gate

Stage 6 success requires the final 10-seed selected variant to pass:

- `combined_accuracy - best_single_avenue_accuracy >= 0.20`
- `combined_accuracy >= 0.80`
- `best_single_avenue_accuracy <= 0.55`
- Stage 4 frozen, baseline, control, invariance, and leakage gates.

## Experiment Families

- Message bottleneck sweep: per-avenue bottleneck dimensions `8, 16, 32, 64`.
- Single-avenue adversarial heads: `lambda_adv` values `0.01, 0.03, 0.1, 0.3`.
- Redundancy and dominance penalties: cosine, orthogonality, and covariance-style penalties with standard, gated, and product-of-experts candidate-token coordinators.

## Folder Layout

- `code/`: Stage 6 source copy and runner.
- `configs/`: local Stage 4 base config copy.
- `reports/`: generated Stage 6 report.
- `results/`: generated Stage 6 JSON, audit logs, and checkpoints.
