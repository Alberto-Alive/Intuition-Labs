# DIGIT Extrapolation E1

Frozen copy of the original smoke-test-first entropy-gating variant of DIGIT.

This package reuses the Adult query/data pipeline shape from `PoC`, but swaps the
`answer/support/confidence/risk` bottleneck for an entropy-trajectory bottleneck:

- `trajectory_shape`
- `attention_pattern`
- `confidence`
- `outcome`

## What This Variant Is

This branch asks a different question than the main PoC. The PoC bottleneck learns
discrete answers about the subgroup itself. Extrapolation instead learns discrete
labels about the *model's internal behavior* on a query:

- `trajectory_shape`: whether attention entropy looks decreasing, stable, increasing,
  or volatile across the probe trajectory
- `attention_pattern`: whether the probe looks focused, mixed, or diffuse
- `confidence`: low / medium / high confidence
- `outcome`: `SUCCESS_LIKELY`, `UNCERTAIN`, or `FAILURE_LIKELY`

The pipeline is trace-driven rather than prototype-driven:

- a small trace probe is trained on Adult subgroup queries
- repeated probe passes produce real attention-entropy trajectories, agreement scores,
  probability margins, and attention-mass summaries
- those traces are clustered / thresholded into the four discrete primitives above
- DIGIT is then trained to map a query into that entropy-gating bottleneck and decode
  a short natural-language explanation

In other words, Extrapolation is a research variant for learning when a query looks
answerable, unstable, or likely to fail based on internal trajectory signatures,
rather than directly learning the original PoC's answer/support/risk interface.

## What It Has Demonstrated

What is demonstrated here today is *feasibility of the trace-driven entropy-gating
setup*, not a replacement for the main PoC privacy claim.

- The end-to-end trace pipeline works on real Adult subgroup queries and produces a
  reusable cached corpus in `trace_cache/` with `1313` train, `318` validation, and
  `318` test traces, plus `35` hard-mined training cases.
- The current reference corpus (`trace_cache/reference_metrics.v3.json`,
  fingerprint `de718ff12a86a7e4`) shows that DIGIT can recover these discrete
  primitives with solid test performance:
  - `trajectory_acc = 0.736`, `trajectory_macro_f1 = 0.695`
  - `pattern_acc = 0.959`, `pattern_macro_f1 = 0.939`
  - `confidence_acc = 0.978`, `confidence_macro_f1 = 0.979`
  - `outcome_acc = 0.862`, `outcome_macro_f1 = 0.715`
  - `joint_acc = 0.601`
- On the outcome head, the full DIGIT model clearly beats trivial baselines such as a
  simple entropy-threshold rule (`0.453` accuracy) and the training-set majority class
  (`0.654` accuracy on the cached test split).
- The learned bottleneck is especially strong at recognizing likely-success cases
  (`success_likely_recall = 1.000`) and high-level attention pattern classes.

What it does **not** demonstrate yet:

- it does not yet show better performance than all simple discriminative baselines;
  for example, the cached logistic-regression baseline is still stronger on some heads
  such as outcome accuracy
- it does not yet provide the privacy evidence from the main DIGIT PoC; there are no
  membership- or attribute-inference results in this package
- it should be read as an exploratory trace-gating result: DIGIT can learn a useful
  discrete interface over internal entropy traces, but this branch is not yet a
  validated privacy benchmark

## Current Eval-Suite Results

Checked-in multi-seed comparison results now live in
[`../results/eval_suite/summary.md`](../results/eval_suite/summary.md).

Across seeds `123, 231, 347, 451, 569`, `e1` currently achieves:

| Metric | Result |
| --- | --- |
| `outcome_acc` | `0.843 +- 0.021` |
| `outcome_macro_f1` | `0.684 +- 0.021` |
| `failure_recall` | `0.411 +- 0.088` |
| `joint_acc` | `0.495 +- 0.080` |
| `unsafe_success_rate_correctness` | `0.063 +- 0.000` |
| `success_precision_vs_is_correct` | `0.937 +- 0.000` |
| `all_supported_nonfailure_monotonicity_violation_rate` | `0.343 +- 0.079` |

Useful context from the same eval suite:

- `trajectory_acc = 0.614 +- 0.112`
- `pattern_acc = 0.967 +- 0.004`
- `confidence_acc = 0.952 +- 0.016`
- entropy-threshold outcome accuracy is `0.453`
- logistic-to-labels outcome macro-F1 is `0.698`
- MLP-to-labels outcome macro-F1 is `0.653`

Interpretation:

- `e1` is a workable first trace-gating baseline with strong pattern and confidence
  recovery.
- It beats the entropy-threshold baseline on outcome prediction and slightly beats
  the MLP baseline on outcome macro-F1, but it still trails the logistic baseline on
  that metric.
- The main weakness is failure detection: `failure_recall` is only `0.411`, which is
  exactly the gap that `e2` was built to address.

## Quick Start

```bash
cd experiments/DIGIT/Extrapolation/e1
python3 -m pip install -r requirements.txt

python3 scripts/test_smoke.py
python3 scripts/collect_traces.py
python3 scripts/run_baselines.py
python3 scripts/run_experiment.py
```

## What Changed

The research path now supports a trace-driven revision:

- `collect_traces.py` trains a small attention probe on subgroup queries
- real attention-entropy trajectories are logged from that probe
- Extrapolation training uses recorded traces instead of deterministic prototypes
- `run_experiment.py` trains DIGIT on the collected traces and compares against simple baselines

`test_smoke.py` still exercises the lightweight synthetic path so structural
breakages are caught quickly.
