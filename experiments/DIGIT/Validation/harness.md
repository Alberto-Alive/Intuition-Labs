# DIGIT Validation — Common Evaluation Harness

> **Status: FROZEN — 2026-03-24**
> This document is locked for the current validation cycle. All runs must conform to the conventions below. Deviations must be noted explicitly in the run's `summary.md`.

---

## 1. Seeds

| Mode | Seeds |
|------|-------|
| Full | 0, 1, 2, 3, 4 |
| Reduced | 0, 1, 2 — only if the run is explicitly marked `seed_mode: reduced` in `config.json` |

Reduced-seed runs must include a justification in `summary.md` and must not be used as primary evidence for any claim.

---

## 1b. Comparative modes

Comparative DIGIT-vs-DP runs must report both modes separately:

| Mode | `min_group_size` | Purpose |
|------|------------------|---------|
| `mechanism` | `1` | Isolate DIGIT discretization vs DP noise without shared interface protection. |
| `system` | `10` | Measure the deployed interface under the production policy. |

Comparative run variants use the naming convention `{mode}__{setting}` (for example `mechanism__default`, `system__default`, `mechanism__a2_s2_c2_r2`).

---

## 2. Baselines

Every run reports results for all of the following systems unless the run spec explicitly excludes one with a reason:

| System key | Description |
|------------|-------------|
| `raw` | Direct access — no privacy protection. Upper bound on utility, lower bound on privacy. |
| `digit` | DIGIT PoC as frozen in this validation cycle. |
| `dp_laplace_e0.1` | DP-Laplace, ε = 0.1 |
| `dp_laplace_e0.5` | DP-Laplace, ε = 0.5 |
| `dp_laplace_e1.0` | DP-Laplace, ε = 1.0 — **primary DP comparison** |
| `dp_laplace_e3.0` | DP-Laplace, ε = 3.0 |

The full DP sweep (`e0.1`, `e0.5`, `e1.0`, `e3.0`) must be reported alongside the primary. If only one DP point is shown in a figure or summary table, it must be `e1.0` and the full sweep must appear in an appendix or supplementary table.

---

## 3. Metrics

Metrics are frozen. Do not compute ad hoc alternatives without adding them to this document first.

### 3a. Privacy attacks (membership inference, attribute inference, subgroup presence, linkage)

| Metric | Role |
|--------|------|
| AUROC | **Primary** |
| Attack advantage `Adv = 2·(AUROC − 0.5)` | Secondary |
| TPR @ 1% FPR | Secondary |
| TPR @ 5% FPR | Secondary |

All four metrics must appear in `metrics.json`. Report mean ± std across seeds.

**Reported AUROC is always the worst-case (highest) across all attack variants run.** For membership inference this means `max(direct_auroc, shadow_auroc)`. Both sub-attack results must also be stored in `metrics.json` and `predictions.parquet` for transparency.

#### Membership inference attack configuration (frozen)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Direct attack | LR + MLP, take stronger | Baseline check; expected weak for group queries |
| Shadow folds K | 32 | Enough in/out observations per record (~16 each) |
| Queries per record per fold M | 30 | Averages over query-level noise |
| Shadow pool fraction | 0.5 (N/2 per fold) | Standard Shokri et al. setting |
| Target records per seed | min(500, N_member/2, N_nonmember/2) | Capped to keep wall time reasonable |
| Meta-classifier | LR (C=0.1) + MLP (64-32), take stronger | Same as direct attack |
| Test split | 40% stratified | Gives reliable AUROC on small samples |

### 3b. Reconstruction / inversion

| Metric | Role |
|--------|------|
| Exact-match rate | Primary |
| Top-k accuracy (k=5) | Secondary |
| Mean attribute-level error / distance | Secondary |

Define the distance function per attribute type in the run spec (`categorical`: 0/1 mismatch, `continuous`: normalized absolute error). Document it in `config.json` under `reconstruction_distance`.

### 3c. Adaptive and composition attacks

| Metric | Role |
|--------|------|
| Leakage-vs-query-budget curve (AUROC at each checkpoint) | Primary |
| Cumulative attack advantage at budget checkpoints | Secondary |

Checkpoints: every 50 queries up to the budget cap, plus the final point. Store the full curve in `events.jsonl` (one entry per checkpoint) and the final-point summary in `metrics.json`.

### 3d. Utility — binary classification

| Metric | Role |
|--------|------|
| AUROC | Primary |
| AUPRC | Primary |
| F1 (threshold = 0.5) | Secondary |
| Accuracy (threshold = 0.5) | Secondary |

### 3e. Utility — multiclass classification

| Metric | Role |
|--------|------|
| Macro-F1 | **Primary** |
| Per-class F1 | Secondary (stored in `metrics.json`, not required in summary tables) |

### 3f. Utility — regression

| Metric | Role |
|--------|------|
| RMSE | Primary |
| Pearson r | Primary |
| Spearman ρ | Secondary |

---

## 4. Per-run output structure

Every run produces exactly these files under its seed directory (see §5 for path convention):

| File | Contents |
|------|----------|
| `config.json` | Full reproducible config: system key, dataset, task, split hash, seed, mode, query budget, DP ε if applicable, hyperparameters, DIGIT commit, harness version. |
| `metrics.json` | All frozen metrics for this run type. Flat key-value, including `*_mean`, `*_std`, `*_ci_low`, `*_ci_high` for aggregated summaries where applicable. |
| `events.jsonl` | One JSON object per event (query, checkpoint, attack step). Required for adaptive/composition runs; optional but encouraged for all others. |
| `predictions.parquet` | Model outputs and ground-truth labels. Use `.csv` only if parquet is unavailable — record the reason in `summary.md`. |
| `summary.md` | Human-readable one-page summary: system, dataset, task, seed, top-line metrics, notable observations, deviations from harness. |
| `stdout.log` | Captured stdout and stderr from the run. |

The harness normalizes PoC outputs into this format. Do not inherit ad hoc output structures from `experiments/DIGIT/PoC/outputs/`.

Comparative privacy runs must also record:
- `mode`
- `attacker_knowledge`
- `pass_fail_threshold`
- `query_budget`
- `cumulative_epsilon` when the system is DP-based

`config.json` must include `"harness_version": "1.1"` and `"seed_mode": "full"` or `"seed_mode": "reduced"`.

---

## 5. Results folder convention

```
Validation/results/
  {phase}/                      # phase1_privacy | phase2_utility | phase3_robustness
    {dataset}/                  # nist_genomics | prism | tcga | mimic_iv_demo
      {task}/                   # e.g. membership_inference | attribute_inference | subtype_prediction
        {system}/               # raw | digit | dp_laplace_e0.1 | dp_laplace_e0.5 | dp_laplace_e1.0 | dp_laplace_e3.0
          {variant}/            # default | ablation_no_bottleneck | ablation_weak_bottleneck | etc.
            seed_0/
            seed_1/
            seed_2/
            seed_3/
            seed_4/
```

`variant` is `default` for standard runs. Ablation and sensitivity variants must be named descriptively and documented in the run spec.

Aggregated results (mean ± std across seeds, cross-system comparison tables) are written to `validation_report.md`. Do not write aggregated results into individual seed directories.

---

## 6. Aggregation rules

- Aggregate over the full seed set (0–4) unless the run is marked `seed_mode: reduced`.
- Report mean ± std for all scalar metrics.
- For curves (leakage-vs-budget), report the mean curve with a shaded std band.
- Flag any seed that failed or was excluded; do not silently drop it.

---

## 7. Harness version

This document is `harness_version: "1.1"`. Version `1.1` adds comparative mode labeling, flat CI fields, and explicit budget/accounting metadata for DIGIT-vs-DP runs. Any future change that affects metric definitions, file structure, or baseline set requires a version bump and a new checklist item before runs begin.
