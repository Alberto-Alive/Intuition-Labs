# DIGIT PoC — UCI Adult Income Experiment

## Purpose

Validate the DIGIT architecture on real data. The feasibility experiment proved the plumbing works on synthetic data. This PoC proves the discrete bottleneck produces useful answers on a real dataset while resisting formal privacy attacks.

## Dataset

**UCI Adult Income** (48,842 records, 14 attributes, binary label: income >50K).

- Train private data: 60% (~27,600 records)
- Validation: 20% (~9,200)
- Test: 20% (~9,200)

Queries target demographic subgroups defined by combinations of occupation, education, sex, race, age bracket, workclass, marital status, hours bracket, etc. Ground-truth primitives are derived from actual income statistics of each subgroup.

## What changed from the feasibility experiment

| Component | Feasibility | PoC |
|-----------|-------------|-----|
| Data | Synthetic random | UCI Adult Income |
| Queries | Random numeric vectors | Structured demographic filters |
| Ground truth | Randomly sampled | Derived from real income rates |
| Encoder | Linear projection per field | Embedding table per categorical field |
| Executor | Indexes by group_id | Filters records by query predicates |
| Baselines | Straw-man rules | + DP Laplace at 6 epsilon values |
| Evaluation | Single seed, no attacks | 5 seeds, MIA, AIA, significance tests |

The bottleneck and decoder are architecturally identical.

## Evaluation protocol

- **5 seeds** [42, 137, 256, 512, 1024] with mean and standard deviation
- **Early stopping** with patience=15 on validation loss
- **Class-weighted loss** to handle label imbalance

### Metrics

- Primitive accuracy (answer, support, confidence, risk, joint)
- Macro F1, precision, recall
- Token-level generation accuracy, BLEU-4
- Leakage score (squared correlation proxy)

### Privacy attacks

1. **Membership Inference Attack (MIA)**: shadow-model approach. Can an attacker determine if a specific record was in the private dataset by observing DIGIT's output? AUC near 0.5 = strong privacy.
2. **Attribute Inference Attack (AIA)**: does DIGIT's output help predict a target individual's income beyond what demographics alone reveal? Advantage near 0 = strong privacy.

### Comparisons

- Baseline A: 3-class rule-based
- Baseline B: full-primitive rule-based
- DP Laplace: at epsilon = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

### Statistical rigor

- Paired bootstrap test (10,000 resamples, 95% CI)
- Paired t-test over seeds
- Wilcoxon signed-rank (nonparametric)
- Bonferroni correction for multiple comparisons

### Ablation

Three bottleneck sizes:
- Small: 3×2×2×2 = 24 combinations (4.58 bits)
- Default: 6×4×3×3 = 216 combinations (7.75 bits)
- Large: 8×6×5×5 = 1200 combinations (10.23 bits)

## How to run

```bash
cd experiments/DIGIT/PoC
pip install -r requirements.txt

# Smoke test
python scripts/test_smoke.py

# Full experiment (5 seeds, baselines, DP, privacy attacks)
python scripts/run_main.py --output-dir outputs

# Ablation study
python scripts/run_ablation.py --output-dir outputs_ablation
```

## Outputs

- `outputs/experiment_results.json` — all metrics
- `outputs/figures/` — training curves, confusion matrix, privacy-utility tradeoff
- `outputs/tables/` — LaTeX tables ready for paper inclusion
- `outputs/seed_*/` — per-seed checkpoints and training history
