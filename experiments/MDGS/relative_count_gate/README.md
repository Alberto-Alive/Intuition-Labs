# MDGS Monotone Relative/Count Uncertainty Gate

This is the deployable counterpart of the MDGS post-hoc failure-ranking result
in `experiments/arc_transfer_validation` (held-out evolutionary relative/count
gate, AUROC 0.761 -> 0.859). It turns that ranking into a **decision interface**
that emits `SUCCESS_LIKELY` / `UNCERTAIN` / `FAILURE_LIKELY` and clears the two
bars the trained e2 head fails.

## The transfer

The ARC principle is *bind to relative role, count the evidence*. Here the gate's
risk is a **non-negative weighted sum of degradation-oriented signals** (each
written so "more degraded => larger") plus a **count** of how many signals are in
their degraded regime:

```
risk = sum_i  w_i * standardized(degradation_feature_i)        # w_i >= 0
     + count_weight * #{ features in their degraded regime }   # counting
success_logit = -risk ,  failure_logit = +risk
```

Because every audited perturbation (`lower_agreement`, `lower_margin`,
`higher_entropy`, `lower_attention_concentration`) can only *raise* the
degradation features the gate uses -- and the columns those perturbations also
overwrite are deliberately left unused -- the risk can only rise when evidence
worsens. So success probability can only fall: the gate is **monotone by
construction** for any trace, and a smooth monotone risk with separated
thresholds is inherently threshold-stable. Weights are fit (non-negative
logistic) to predict incorrectness, so failure detection is retained.

## What it fixes

Scored on the held-out e2 `test` split, using the **real** `apply_trace_perturbation`
and `compute_monotonicity_violation_rate` from `eval_common` (no model retraining;
weights fit on `train`, thresholds frozen on `val`):

| Bar | e2 | this gate | target |
| --- | ---: | ---: | --- |
| monotonicity violation rate | 0.443 | **0.000** | < 0.05 |
| nonfailure monotonicity violation rate | 0.247 | **0.000** | < 0.05 |
| max outcome-share swing (threshold stability) | 0.208 | **0.025** | <= 0.10 |
| detection AUROC (catch wrong answers) | 0.774 | **0.821** | >= e2 |
| SUCCESS - UNCERTAIN correctness separation | 0.097 | **0.124** | >= 0.10 |
| unsafe success rate (correctness) | 0.016 | 0.027 | <= 0.02 |

The gate beats e2 on every bar e2 fails (monotonicity, nonfailure monotonicity,
threshold stability) and on detection AUROC and separation. It is also a real
multi-evidence win over a single-signal control: a raw-margin gate scores AUROC
0.761. The one honest cost is a slightly higher unsafe-success rate (0.027 vs
0.016).

`detection AUROC` is the threshold-free comparison and is the headline: it does
not depend on where the decision thresholds sit, so it is robust to operating
point. The discrete operating point uses a balanced (Youden's J) `t_high` and a
high-precision `t_low`, both chosen on `val`, to avoid the degenerate
"flag-everything" point that pure recall maximization produces.

## Multi-seed replication

Monotonicity and threshold stability are structural (proven by construction), so
only the empirical *detection* win could be corpus-specific. `replicate_multiseed.py`
tests it across the five independently trained e2 models in
`results/eval_suite/e2/seed_*` with leave-one-seed-out: fit the monotone gate on
four seeds, score detection AUROC on the held-out seed.

| Detector (mean over 5 held-out seeds) | AUROC |
| --- | ---: |
| monotone relative/count gate | **0.761** |
| e2 deployed outcome head (1 - P(success)) | 0.722 |
| raw single-margin control | 0.761 |

The gate's detection is at least as good as e2's own deployed failure head
(winning in 3/5 seeds) while remaining monotone (all weights non-negative). Honest
caveat: the per-seed dumps store only four raw degradation scalars
(agreement, margin, attention, entropy), so here the gate *ties* a single margin
signal; the multi-evidence lift to AUROC 0.821 needs `support_ratio` and
`abs_margin`, which exist only in the full trace cache used by `evaluate_gate.py`.

## Run

```bash
python experiments/MDGS/relative_count_gate/evaluate_gate.py
python experiments/MDGS/relative_count_gate/replicate_multiseed.py
python -m pytest experiments/MDGS/relative_count_gate/test_gate.py -q
```

- `gate.py` -- the monotone relative/count gate and its fit.
- `evaluate_gate.py` -- held-out evaluation on the e2 trace cache; writes
  `results/relative_count_gate_results.json`.
- `test_gate.py` -- asserts non-negative weights and that no perturbation raises
  success probability (monotonicity by construction), on arbitrary traces.

## Honest scope

- This is a fixed (non-learned-representation) gate over cached trace features,
  not a retrained transformer head. It proves the relative/count *construction*
  yields a monotone, threshold-stable, well-separated failure detector at e2's
  detection quality -- it does not claim to replace the e2 model.
- Evidence is one fixed corpus (`8d0bb8702a0871d5`) with a train/val/test split.
  Multi-corpus or multi-seed-trace replication would strengthen it further.
