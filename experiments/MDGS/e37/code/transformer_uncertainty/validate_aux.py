from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Dict, List

import torch

from .data import OOD_FACTORIES
from .validate_anchor import (
    DEFAULT_OOD_MODES,
    aggregate,
    evaluate_checkpoint,
    make_data_cfg,
    make_model_cfg,
    mean_metric,
    paired_diff,
    safe_mean,
    train_or_load,
)


MAIN_MODELS = ["baseline", "aux"]
AUX_ABLATION_MODELS = [
    "aux_no_mdgs",
    "aux_random_mdgs",
    "aux_shuffled_mdgs",
    "aux_detached_mdgs",
    "aux_frozen_mdgs",
]

PASS_FAIL_RULE = """# Predeclared Aux Validation Rule

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
"""


def decide_verdict(records: List[Dict[str, object]], agg: Dict[str, Dict[str, Dict[str, float]]], seeds: List[int]):
    baseline = "baseline"
    aux = "aux"
    broken = AUX_ABLATION_MODELS

    acc_diff = mean_metric(agg, aux, "accuracy") - mean_metric(agg, baseline, "accuracy")
    ood_diff = mean_metric(agg, aux, "ood_mean_auroc_by_uncertainty") - mean_metric(
        agg, baseline, "ood_mean_auroc_by_uncertainty"
    )
    risk_diff = mean_metric(agg, aux, "risk_coverage_auc") - mean_metric(agg, baseline, "risk_coverage_auc")
    ece_diff = mean_metric(agg, aux, "ece") - mean_metric(agg, baseline, "ece")
    nll_diff = mean_metric(agg, aux, "nll") - mean_metric(agg, baseline, "nll")
    ood_pair = paired_diff(records, aux, baseline, "ood_mean_auroc_by_uncertainty")

    aux_score = mean_metric(agg, aux, "validation_score")
    broken_scores = [mean_metric(agg, name, "validation_score") for name in broken if name in agg]
    broken_mean = safe_mean(broken_scores)
    broken_below = sum(1 for score in broken_scores if math.isfinite(score) and score < aux_score)

    checks = {
        "accuracy_preserved": bool(acc_diff >= -0.005),
        "ood_uncertainty_advantage": bool(ood_diff >= 0.05),
        "risk_equal_or_better": bool(risk_diff <= 0.005 and ece_diff <= 0.02 and nll_diff <= 0.03),
        "seed_stability": bool(len(seeds) >= 10 and ood_pair["n"] >= 10 and ood_pair["positive"] >= 7),
        "aux_mdgs_ablation_drop": bool((aux_score - broken_mean) >= 0.01 and broken_below >= 4),
    }
    verdict = "VALIDATED" if all(checks.values()) else "INVALIDATED"
    details = {
        "accuracy_diff_aux_minus_baseline": acc_diff,
        "ood_auc_uncertainty_diff_aux_minus_baseline": ood_diff,
        "risk_auc_diff_aux_minus_baseline": risk_diff,
        "ece_diff_aux_minus_baseline": ece_diff,
        "nll_diff_aux_minus_baseline": nll_diff,
        "paired_ood_auc_uncertainty_diff": ood_pair,
        "aux_validation_score": aux_score,
        "broken_aux_ablation_mean_validation_score": broken_mean,
        "broken_aux_ablation_count_below_aux": broken_below,
    }
    for name in broken:
        details[f"{name}_validation_score"] = mean_metric(agg, name, "validation_score")
        details[f"validation_score_diff_aux_minus_{name}"] = aux_score - mean_metric(agg, name, "validation_score")
    return {"verdict": verdict, "checks": checks, "details": details}


def write_report(
    out_dir: Path,
    args,
    records: List[Dict[str, object]],
    agg: Dict[str, Dict[str, Dict[str, float]]],
    verdict: Dict[str, object],
) -> None:
    metrics = [
        "validation_score",
        "accuracy",
        "nll",
        "ece",
        "brier",
        "risk_coverage_auc",
        "error_auroc_by_uncertainty",
        "selective_acc_reject10_uncertainty",
        "confident_wrong_90",
        "ood_mean_auroc_by_uncertainty",
        "ood_mean_auroc_by_negative_support",
        "ood_easy_auroc_by_uncertainty",
        "ood_medium_auroc_by_uncertainty",
        "ood_hard_auroc_by_uncertainty",
        "ood_support_mismatch_auroc_by_uncertainty",
    ]
    ordered_variants = [v for v in args.models + args.ablations + args.references if v in agg]
    lines = [
        "| model | " + " | ".join(metrics) + " |",
        "| --- | " + " | ".join(["---:"] * len(metrics)) + " |",
    ]
    for variant in ordered_variants:
        vals = []
        for metric in metrics:
            value = mean_metric(agg, variant, metric)
            vals.append("nan" if not math.isfinite(value) else f"{value:.4f}")
        lines.append(f"| {variant} | " + " | ".join(vals) + " |")

    check_lines = [
        f"- `{name}`: {'PASS' if passed else 'FAIL'}"
        for name, passed in verdict["checks"].items()
    ]

    pair_metrics = [
        "accuracy",
        "risk_coverage_auc",
        "ece",
        "nll",
        "ood_mean_auroc_by_uncertainty",
        "validation_score",
    ]
    pair_lines = ["| metric | mean paired diff | positive seeds | n |", "| --- | ---: | ---: | ---: |"]
    for metric in pair_metrics:
        diff = paired_diff(records, "aux", "baseline", metric)
        pair_lines.append(f"| {metric} | {diff['mean']:.4f} | {diff['positive']} | {diff['n']} |")

    ablation_lines = ["| ablation | aux minus ablation score |", "| --- | ---: |"]
    aux_score = mean_metric(agg, "aux", "validation_score")
    for name in AUX_ABLATION_MODELS:
        if name in agg:
            ablation_score = mean_metric(agg, name, "validation_score")
            ablation_lines.append(f"| {name} | {aux_score - ablation_score:.4f} |")

    text = f"""# Aux Validation Report

## Verdict

`{verdict['verdict']}`

## Predeclared Rule

See `PASS_FAIL_RULE.md`. Results below were judged against that rule.

## Run Config

```json
{json.dumps(vars(args), indent=2)}
```

## Aggregate Metrics

{chr(10).join(lines)}

## Pass/Fail Checks

{chr(10).join(check_lines)}

```json
{json.dumps(verdict['details'], indent=2)}
```

## Paired Aux vs Baseline

{chr(10).join(pair_lines)}

## Broken Aux Comparison

{chr(10).join(ablation_lines)}
"""
    (out_dir / "VALIDATION_REPORT.md").write_text(text, encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="Pressure-test aux MDGS as a training teacher.")
    p.add_argument("--out", type=str, default="transformer_uncertainty/runs/aux_validation")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--models", nargs="+", default=MAIN_MODELS)
    p.add_argument("--ablations", nargs="+", default=AUX_ABLATION_MODELS)
    p.add_argument("--references", nargs="+", default=[])
    p.add_argument("--ood-modes", nargs="+", default=DEFAULT_OOD_MODES, choices=sorted(OOD_FACTORIES))
    p.add_argument("--train-ood-mode", choices=sorted(OOD_FACTORIES), default="easy")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--aux-warmup-epochs", type=int, default=2)

    p.add_argument("--n-train", type=int, default=1200)
    p.add_argument("--n-val", type=int, default=400)
    p.add_argument("--n-test", type=int, default=400)
    p.add_argument("--ood-ratio-train", type=float, default=0.25)
    p.add_argument("--seq-len", type=int, default=18)
    p.add_argument("--vocab-size", type=int, default=64)

    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-refine-layers", type=int, default=1)
    p.add_argument("--dim-feedforward", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--num-views", type=int, default=6)
    p.add_argument("--num-prototypes", type=int, default=16)
    return p.parse_args()


def main(args=None):
    args = parse_args() if args is None else args
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "PASS_FAIL_RULE.md").write_text(PASS_FAIL_RULE, encoding="utf-8")
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    variants = list(dict.fromkeys(args.models + args.ablations + args.references))
    model_cfg = make_model_cfg(args)
    records: List[Dict[str, object]] = []

    for seed in args.seeds:
        data_cfg = make_data_cfg(args, seed)
        for variant in variants:
            run_dir = out_dir / f"{variant}_s{seed}"
            print(json.dumps({"event": "run_start", "variant": variant, "seed": seed, "run_dir": str(run_dir)}))
            train_or_load(variant, seed, run_dir, args, data_cfg, model_cfg)
            metrics = evaluate_checkpoint(variant, run_dir, args, data_cfg, model_cfg, args.ood_modes)
            record = {"variant": variant, "seed": seed, **metrics}
            records.append(record)
            (run_dir / "validation_metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "event": "validated",
                        "variant": variant,
                        "seed": seed,
                        "accuracy": metrics["accuracy"],
                        "risk_auc": metrics["risk_coverage_auc"],
                        "ood_auc_unc_mean": metrics["ood_mean_auroc_by_uncertainty"],
                        "validation_score": metrics["validation_score"],
                    }
                )
            )
            gc.collect()
            if args.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

    agg = aggregate(records)
    verdict = decide_verdict(records, agg, list(args.seeds))
    (out_dir / "all_validation_metrics.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (out_dir / "aggregate_metrics.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    write_report(out_dir, args, records, agg, verdict)
    print(json.dumps({"event": "verdict", **verdict}))
    print(f"Saved validation report to {out_dir / 'VALIDATION_REPORT.md'}")


if __name__ == "__main__":
    main()
