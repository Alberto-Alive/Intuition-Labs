# DIGIT Extrapolation E2

Uncertainty-tower version of the Extrapolation experiment.

This variant keeps the trace-driven Extrapolation setup from `e1`, but changes the
bottleneck design for `confidence` and `outcome`.

## What Changed From E1

`e1` mixed the outcome prediction into the same general bottleneck path that handled
the other primitives. `e2` adds a dedicated uncertainty path that is explicitly fed
uncertainty-oriented signals:

- attention-entropy trajectory summary
- max attention mass
- probe agreement across repeated stochastic passes
- probability margin
- max-softmax confidence
- predictive entropy from the probe probabilities
- variation ratio (`1 - agreement`)
- support / subgroup-size / absolute-margin features

The goal is to make `SUCCESS_LIKELY` / `UNCERTAIN` / `FAILURE_LIKELY` prediction more
sensitive to uncertainty features and less dependent on the shared bottleneck alone.

## Architecture

`e2` still predicts the same four primitives:

- `trajectory_shape`
- `attention_pattern`
- `confidence`
- `outcome`

But the head structure is now split:

- `trajectory_shape` and `attention_pattern` are predicted much like `e1`
- `confidence` is predicted from a dedicated uncertainty tower
- `outcome` is predicted from that uncertainty tower plus the other primitive
  descriptors

This is meant to be a more sensible uncertainty architecture than the original mixed
head.

## Status

This folder starts from the same copied `data_cache/` and `trace_cache/` contents as
`e1` so it can be run immediately, but its model has changed.

The earlier "implemented architecture change, not a completed benchmark" note is now
outdated. Checked-in multi-seed results exist under
[`../results/eval_suite/summary.md`](../results/eval_suite/summary.md) and
[`../results/eval_suite/e2`](../results/eval_suite/e2).

That still means:

- the copied cached trace corpus is still a useful starting point
- the copied `reference_metrics.v3.json` is **not** an `e2` result
- `e2` writes its own reference metrics to `trace_cache/reference_metrics.e2.json`
  after you run the experiment

What is completed today is the external eval suite, not a local `reference_metrics`
writeback inside `e2/trace_cache/`.

## Current Results

Across eval-suite seeds `123, 231, 347, 451, 569`, `e2` currently achieves:

| Metric | Result |
| --- | --- |
| `outcome_acc` | `0.877 +- 0.015` |
| `outcome_macro_f1` | `0.808 +- 0.034` |
| `failure_recall` | `0.812 +- 0.069` |
| `joint_acc` | `0.543 +- 0.111` |
| `unsafe_success_rate_correctness` | `0.016 +- 0.002` |
| `success_precision_vs_is_correct` | `0.984 +- 0.002` |
| `all_supported_nonfailure_monotonicity_violation_rate` | `0.212 +- 0.060` |

Compared with `e1`, the main gains are:

- `outcome_macro_f1`: `0.684 -> 0.808`
- `failure_recall`: `0.411 -> 0.812`
- `unsafe_success_rate_correctness`: `0.063 -> 0.016`
- `all_supported_nonfailure_monotonicity_violation_rate`: `0.343 -> 0.212`

Baseline context from the same eval suite:

- entropy-threshold outcome accuracy is `0.261`
- logistic-to-labels outcome macro-F1 is `0.617`
- MLP-to-labels outcome macro-F1 is `0.818`

Interpretation:

- `e2` achieves what it was designed to improve: it is much better than `e1` at
  recognizing likely-failure cases while preserving solid overall outcome accuracy.
- It clearly beats the entropy-threshold and logistic baselines on outcome labeling
  and comes close to the MLP baseline on outcome macro-F1.
- It is still not a clean safety win. In the checked-in comparison, `success_brier`
  and `success_ece_10bin` get worse, `safety_win` remains `False`, monotonicity is
  still not accepted, and `label_revision_trigger` is `True`.

## Quick Start

```bash
cd experiments/DIGIT/Extrapolation/e2
python3 -m pip install -r requirements.txt

python3 scripts/test_smoke.py
python3 scripts/collect_traces.py
python3 scripts/run_baselines.py
python3 scripts/run_experiment.py
```
