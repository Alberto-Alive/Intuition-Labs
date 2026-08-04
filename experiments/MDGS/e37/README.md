# Validated Aux MDGS Reference

This folder is a self-contained reference snapshot for the validated aux result:

> MDGS uncertainty helps train the transformer as an auxiliary teacher, even when MDGS is not directly fused into the final prediction path.

## What To Keep

- `code/transformer_uncertainty/models.py`: model definitions, including the validated `AuxMDGSTransformer`.
- `code/transformer_uncertainty/train.py`: loss wiring for `aux` and broken aux controls.
- `code/transformer_uncertainty/validate_aux.py`: dedicated aux validation harness and verdict logic.
- `results/VALIDATION_REPORT.md`: human-readable 10-seed result.
- `results/verdict.json`: machine-readable pass/fail verdict.
- `results/aggregate_metrics.json`: full aggregate metric table.
- `results/all_validation_metrics.json`: per-seed metrics.
- `results/PASS_FAIL_RULE.md`: predeclared validation rule.
- `results/run_config.json`: exact validation configuration.

## Best Architecture

The best target architecture is `aux`.

```text
tokens -> transformer memory
CLS -> class logits
witness queries attend to memory -> MDGS -> auxiliary uncertainty/support/geometry losses
```

The final prediction path is still plain CLS classification:

```text
memory[:, CLS] -> base_head -> logits
```

MDGS does not feed into the final logits. Its job is to teach the shared transformer memory during training.

## Validation Result

Verdict: `VALIDATED`

Main aggregate numbers from 10 CUDA seeds:

| comparison | result |
| --- | ---: |
| aux accuracy minus baseline | +0.0025 |
| aux OOD uncertainty AUC minus baseline | +0.0931 |
| aux risk AUC minus baseline | -0.00003 |
| paired OOD uncertainty AUC positive seeds | 9 / 10 |
| aux validation score | 1.2146 |
| broken aux mean validation score | 1.1556 |
| broken variants below aux | 4 / 5 |

Pass/fail checks:

```json
{
  "accuracy_preserved": true,
  "ood_uncertainty_advantage": true,
  "risk_equal_or_better": true,
  "seed_stability": true,
  "aux_mdgs_ablation_drop": true
}
```

## Why `aux` Is The Reference Variant

`aux` preserved ID accuracy and improved OOD uncertainty. It also beat the mean score of broken auxiliary controls. That supports the claim that the improvement came from meaningful MDGS supervision, not merely from adding extra parameters or arbitrary extra losses.

The clean claim is:

> MDGS works as a training teacher. It gives the transformer uncertainty-aware supervision that improves learned representations and OOD behavior.

In simpler terms:

> The model learned better instincts because it was trained with an uncertainty-aware tutor.

## Why The Other Aux Variants Were Not As Good

`baseline`

The baseline is a normal transformer with uncertainty equal to inverse softmax confidence. It had strong accuracy, but lower OOD uncertainty AUC:

- baseline OOD uncertainty AUC: `0.8386`
- aux OOD uncertainty AUC: `0.9317`

So baseline learned the task, but did not learn as useful an uncertainty signal.

`aux_no_mdgs`

This removes MDGS auxiliary supervision. It remained accurate, but its OOD uncertainty behavior was much closer to baseline:

- aux_no_mdgs OOD uncertainty AUC: `0.8453`
- aux OOD uncertainty AUC: `0.9317`

This is the most direct evidence that the MDGS teacher mattered.

`aux_random_mdgs`

This keeps extra auxiliary machinery but randomizes the MDGS targets. It performed much worse:

- validation score: `1.0874`
- OOD uncertainty AUC: `0.6393`

This shows the gain was not just from having extra loss terms.

`aux_shuffled_mdgs`

This keeps real target values but assigns them to the wrong examples. It also degraded:

- validation score: `1.1275`
- OOD uncertainty AUC: `0.7560`

This shows that the MDGS signal must be example-aligned.

`aux_frozen_mdgs`

This freezes the MDGS extractor. It preserved accuracy but failed badly on OOD uncertainty:

- validation score: `1.1435`
- OOD uncertainty AUC: `0.4986`

This shows that a trainable MDGS teacher is important for useful uncertainty behavior.

`aux_detached_mdgs`

This was the one caveat. It was essentially tied with real aux on validation score:

- aux validation score: `1.2146`
- aux_detached_mdgs validation score: `1.2149`

But aux still had better mean OOD uncertainty AUC:

- aux OOD uncertainty AUC: `0.9317`
- aux_detached_mdgs OOD uncertainty AUC: `0.9243`

Interpretation: the validation supports meaningful MDGS supervision, but it does not prove that gradients from MDGS into the transformer backbone are strictly necessary. Detached MDGS still provides strong training pressure through the auxiliary outputs and remains a close control.

## What This Does Not Prove

This aux result does not prove that MDGS is used directly during prediction. The final logits do not consume MDGS.

Use this claim only:

> Uncertainty helps learning.

Do not use this claim from aux alone:

> Uncertainty participates directly in prediction.

That direct prediction-time claim belongs to `shared` or related shared-attention variants.
