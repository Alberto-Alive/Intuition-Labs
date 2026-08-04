# Aux Validation Report

## Verdict

`VALIDATED`

## Predeclared Rule

See `PASS_FAIL_RULE.md`. Results below were judged against that rule.

## Run Config

```json
{
  "out": "transformer_uncertainty/runs/aux_validation_gpu_20260508",
  "seeds": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9
  ],
  "models": [
    "baseline",
    "aux"
  ],
  "ablations": [
    "aux_no_mdgs",
    "aux_random_mdgs",
    "aux_shuffled_mdgs",
    "aux_detached_mdgs",
    "aux_frozen_mdgs"
  ],
  "references": [],
  "ood_modes": [
    "easy",
    "medium",
    "hard",
    "support_mismatch"
  ],
  "train_ood_mode": "easy",
  "resume": true,
  "device": "cuda",
  "num_threads": 1,
  "epochs": 20,
  "batch_size": 128,
  "lr": 0.002,
  "weight_decay": 0.0001,
  "grad_clip": 5.0,
  "aux_warmup_epochs": 2,
  "n_train": 1200,
  "n_val": 400,
  "n_test": 400,
  "ood_ratio_train": 0.25,
  "seq_len": 18,
  "vocab_size": 64,
  "d_model": 64,
  "nhead": 4,
  "num_layers": 2,
  "num_refine_layers": 1,
  "dim_feedforward": 128,
  "dropout": 0.05,
  "num_views": 6,
  "num_prototypes": 16
}
```

## Aggregate Metrics

| model | validation_score | accuracy | nll | ece | brier | risk_coverage_auc | error_auroc_by_uncertainty | selective_acc_reject10_uncertainty | confident_wrong_90 | ood_mean_auroc_by_uncertainty | ood_mean_auroc_by_negative_support | ood_easy_auroc_by_uncertainty | ood_medium_auroc_by_uncertainty | ood_hard_auroc_by_uncertainty | ood_support_mismatch_auroc_by_uncertainty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1.2020 | 0.9953 | 0.0127 | 0.0053 | 0.0068 | 0.0000 | 0.9946 | 1.0000 | 0.0010 | 0.8386 | 0.8386 | 0.7669 | 0.9174 | 0.9175 | 0.7527 |
| aux | 1.2146 | 0.9978 | 0.0175 | 0.0161 | 0.0032 | 0.0000 | 0.9751 | 1.0000 | 0.0000 | 0.9317 | 0.9106 | 0.9991 | 0.9828 | 0.9621 | 0.7829 |
| aux_no_mdgs | 1.2049 | 0.9973 | 0.0084 | 0.0047 | 0.0038 | 0.0000 | 0.9965 | 1.0000 | 0.0002 | 0.8453 | 0.8453 | 0.7805 | 0.9194 | 0.9117 | 0.7697 |
| aux_random_mdgs | 1.0874 | 0.9918 | 0.0274 | 0.0113 | 0.0127 | 0.0004 | 0.4651 | 0.9939 | 0.0017 | 0.6393 | 0.1470 | 0.6031 | 0.6517 | 0.6691 | 0.6331 |
| aux_shuffled_mdgs | 1.1275 | 0.9938 | 0.0275 | 0.0171 | 0.0098 | 0.0001 | 0.7467 | 0.9981 | 0.0007 | 0.7560 | 0.2753 | 0.7916 | 0.7659 | 0.7229 | 0.7435 |
| aux_detached_mdgs | 1.2149 | 0.9988 | 0.0153 | 0.0132 | 0.0026 | 0.0000 | 0.9806 | 1.0000 | 0.0002 | 0.9243 | 0.9113 | 0.9993 | 0.9657 | 0.9348 | 0.7974 |
| aux_frozen_mdgs | 1.1435 | 0.9970 | 0.0156 | 0.0083 | 0.0052 | 0.0001 | 0.6663 | 0.9989 | 0.0012 | 0.4986 | 0.7065 | 0.5266 | 0.5037 | 0.4928 | 0.4711 |

## Pass/Fail Checks

- `accuracy_preserved`: PASS
- `ood_uncertainty_advantage`: PASS
- `risk_equal_or_better`: PASS
- `seed_stability`: PASS
- `aux_mdgs_ablation_drop`: PASS

```json
{
  "accuracy_diff_aux_minus_baseline": 0.002499997615814209,
  "ood_auc_uncertainty_diff_aux_minus_baseline": 0.09313796875000013,
  "risk_auc_diff_aux_minus_baseline": -2.978180061745661e-05,
  "ece_diff_aux_minus_baseline": 0.010835294751450418,
  "nll_diff_aux_minus_baseline": 0.004727318859659135,
  "paired_ood_auc_uncertainty_diff": {
    "n": 10,
    "mean": 0.09313796875000004,
    "std": 0.0561747667412149,
    "positive": 9,
    "negative": 1,
    "zero": 0
  },
  "aux_validation_score": 1.2146398007827435,
  "broken_aux_ablation_mean_validation_score": 1.1556100904503794,
  "broken_aux_ablation_count_below_aux": 4,
  "aux_no_mdgs_validation_score": 1.2048596274105878,
  "validation_score_diff_aux_minus_aux_no_mdgs": 0.009780173372155643,
  "aux_random_mdgs_validation_score": 1.08736077846055,
  "validation_score_diff_aux_minus_aux_random_mdgs": 0.12727902232219357,
  "aux_shuffled_mdgs_validation_score": 1.12746902487071,
  "validation_score_diff_aux_minus_aux_shuffled_mdgs": 0.08717077591203348,
  "aux_detached_mdgs_validation_score": 1.2148576334130232,
  "validation_score_diff_aux_minus_aux_detached_mdgs": -0.0002178326302797462,
  "aux_frozen_mdgs_validation_score": 1.1435033880970251,
  "validation_score_diff_aux_minus_aux_frozen_mdgs": 0.07113641268571835
}
```

## Paired Aux vs Baseline

| metric | mean paired diff | positive seeds | n |
| --- | ---: | ---: | ---: |
| accuracy | 0.0025 | 6 | 10 |
| risk_coverage_auc | -0.0000 | 2 | 10 |
| ece | 0.0108 | 10 | 10 |
| nll | 0.0047 | 6 | 10 |
| ood_mean_auroc_by_uncertainty | 0.0931 | 9 | 10 |
| validation_score | 0.0126 | 8 | 10 |

## Broken Aux Comparison

| ablation | aux minus ablation score |
| --- | ---: |
| aux_no_mdgs | 0.0098 |
| aux_random_mdgs | 0.1273 |
| aux_shuffled_mdgs | 0.0872 |
| aux_detached_mdgs | -0.0002 |
| aux_frozen_mdgs | 0.0711 |
