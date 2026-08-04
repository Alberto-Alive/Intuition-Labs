# DIGIT Extrapolation E3

Partial-monotone successor to `e2`.

`e3` keeps the trace-driven Extrapolation setup and the same four primitives, but it
changes how `confidence` and `outcome` are produced.

## What Changed From E2

`e2` used a dedicated uncertainty tower, but the `confidence` and `outcome` heads
were still ordinary MLP classifiers. That meant they could learn useful failure
signals, but they were not mathematically prevented from becoming more optimistic
when the uncertainty evidence got worse.

`e3` replaces that path with a partial-monotone risk formulation:

- `trajectory_shape` and `attention_pattern` stay unconstrained
- `confidence` and `outcome` are derived from scalar risk heads with nonnegative
  weights
- those risk heads only see explicitly ordered badness features

The ordered badness features are:

- `mean_entropy`
- `predictive_entropy`
- `1 - max_attention_mass`
- `1 - agreement`
- `1 - prob_margin`
- `variation_ratio`
- support deficit below the success gate
- group-size deficit below the success gate

So, within that path, worsening uncertainty evidence can only push risk upward.

## Architecture

`e3` still predicts:

- `trajectory_shape`
- `attention_pattern`
- `confidence`
- `outcome`

But the uncertainty interface is now split into:

- free primitive heads for `trajectory_shape` and `attention_pattern`
- a monotone confidence-risk head mapped to `LOW / MEDIUM / HIGH`
- a monotone outcome-risk head mapped to `SUCCESS_LIKELY / UNCERTAIN / FAILURE_LIKELY`

The class probabilities come from ordered thresholds over a scalar risk, not from a
free 3-way classifier.

## Status

This folder starts from the copied `data_cache/` and `trace_cache/` contents from
`e2`, so it can be exercised immediately.

It is an architecture candidate, not a benchmarked accepted replacement yet:

- the copied cached trace corpus is only a shared starting point
- `e3` writes its own reference file to `trace_cache/reference_metrics.e3.json`
- top-level `run_eval_suite.py` still compares only `e1` and `e2`

## Quick Start

```bash
cd experiments/DIGIT/Extrapolation/e3
python3 -m pip install -r requirements.txt

python3 scripts/test_smoke.py
python3 scripts/test_monotonicity.py
python3 scripts/run_experiment.py
```

## What To Check First

Use `scripts/test_monotonicity.py` before a full training run. It loads the cached
trace corpus, applies the same perturbation families used by the eval suite, and
asserts that the untrained `e3` outcome head does not increase success probability
under those worsening moves.
