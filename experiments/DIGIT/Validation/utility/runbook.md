# Utility Runbook

Run utility experiments in this order:
1. MIMIC-IV demo sanity check
2. NIST Genomics PPFL utility task
3. TCGA omics utility task
4. PRISM drug-response task
5. comparative frontier aggregation (`python3 -m Validation.runners.run_utility_frontier`)

For every run, record:
- DIGIT version / commit
- dataset snapshot and split
- preprocessing version
- model family and hyperparameters
- baseline definitions
- seed
- runtime and hardware

Every utility result must report:
- main metric
- confidence interval across seeds
- retained utility relative to raw baseline
- comparison to at least one DP baseline
- comparative mode label when paired with privacy results
- known failure modes
