# DIGIT Extrapolation

Versioned Extrapolation experiments.

## Layout

- `e1/`: frozen copy of the original Extrapolation experiment
- `e2/`: uncertainty-tower variant with a dedicated path for confidence / outcome
- `e3/`: partial-monotone variant that constrains confidence / outcome on ordered uncertainty axes
- `e10/`: current cooperative diffusion-evidence branch and strongest truthful diffusion result so far

## Which Folder To Use

Use `e1` if you want the current baseline exactly as it existed before the split.

Use `e2` if you want the revised architecture that makes `SUCCESS_LIKELY` /
`UNCERTAIN` / `FAILURE_LIKELY` prediction more explicitly depend on uncertainty
signals such as entropy, agreement, margin, max-softmax, and attention statistics.

Use `e3` if you want to test the next architecture step: keep the useful `e2`
uncertainty evidence, but force the `confidence` and `outcome` path to be monotone
with respect to worsening uncertainty signals.

Use `e10` if you want the current truthful diffusion branch. As of April 4, 2026,
the latest reference run in [summary.json](/mnt/w/Intuition-Labs/experiments/DIGIT/Extrapolation/results/e10/seed_123/summary.json)
has `outcome_acc = 0.874`, `outcome_macro_f1 = 0.803`, `confidence_acc = 0.915`,
`failure_recall = 0.941`, `commitment_monotonicity_violation_rate = 0.069`, and
`not_collapsed = true`.

## Quick Start

```bash
cd experiments/DIGIT/Extrapolation/e1
# or:
cd experiments/DIGIT/Extrapolation/e2
# or:
cd experiments/DIGIT/Extrapolation/e3
```

## Evaluation

The safety-first evaluation entrypoints live at the top level:

```bash
cd experiments/DIGIT/Extrapolation
python3 scripts/test_eval_logic.py
python3 scripts/run_eval_suite.py
```

`run_eval_suite.py` runs the controlled `e1` vs `e2` benchmark on the fixed cached
corpus, writes per-run artifacts under `results/`, and emits an aggregated
`results/eval_suite/summary.json` plus `summary.md`.

`e3` is not wired into that top-level suite yet. Test it from inside `e3/` with
`scripts/test_smoke.py`, `scripts/test_monotonicity.py`, and `scripts/run_experiment.py`.

## Uncertainty Acceptance

For this branch, "acceptable uncertainty" is not just higher outcome accuracy. The
uncertainty head should:

- catch likely failures reliably
- avoid claiming success on incorrect cases
- behave monotonically when uncertainty signals worsen
- remain reasonably stable under small threshold shifts

The table below combines the current hard acceptance logic in
`scripts/run_eval_suite.py` with a stricter practical target for a trustworthy
uncertainty interface.

| Dimension | Metric | Suite floor / gate | Practical target | Current `e2` | Read |
| --- | --- | --- | --- | --- | --- |
| Failure detection | `failure_recall` | `>= 0.30` | `>= 0.70` | `0.812` | Pass |
| Outcome quality | `outcome_macro_f1` | `>= 0.62` | `>= 0.75` | `0.808` | Pass |
| Outcome quality | `outcome_acc` | `>= 0.82` | `>= 0.85` | `0.877` | Pass |
| Trace recovery | `trajectory_acc` | `>= 0.75` | `>= 0.75` | `0.658` | Fail |
| Unsafe success | `unsafe_success_rate_correctness` | safety win needs delta `<= -0.05` vs `e1` | `<= 0.02` absolute | `0.016` | Pass |
| Success trustworthiness | `success_precision_vs_is_correct` | tracked, not hard-gated | `>= 0.97` | `0.984` | Pass |
| Success-label calibration | `success_label_brier` | must improve if recall gain is weak | `<= 0.06` | `0.060` | Borderline |
| Success-label calibration | `success_label_ece_10bin` | tracked, not hard-gated | `<= 0.05` | `0.054` | Slight miss |
| Nonfailure calibration | `nonfailure_correct_ece_10bin` | tracked, not hard-gated | `<= 0.08` | `0.080` | Borderline |
| Monotonicity | `monotonicity_violation_rate` | `< 0.05` and better than `e1` | `< 0.05` | `0.443` | Fail |
| Nonfailure monotonicity | `nonfailure_monotonicity_violation_rate` | `< 0.05` and better than `e1` | `< 0.05` | `0.247` | Fail |
| Threshold robustness | `max_outcome_share_swing` under `+-5%` threshold shifts | trigger if `> 0.10` | `<= 0.10`, ideally `<= 0.05` | `0.208` | Fail |
| Semantic separation | `P(correct | SUCCESS) - P(correct | UNCERTAIN)` | trigger if `< 0.10` | `>= 0.10`, ideally `>= 0.15` | `0.097` | Slight miss |

Current bottom line:

- `e2` already clears the failure-detection and unsafe-success bar.
- `e2` still fails the monotonicity and threshold-stability bar.
- `e2` is therefore better than `e1` as a failure detector, but not yet an accepted
  uncertainty interface.
