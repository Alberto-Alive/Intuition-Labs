# NIST Genomics PPFL Utility Task

## Goal
Show that DIGIT preserves useful predictive signal on a realistic genomics privacy benchmark while reducing leakage.

## Candidate task
Use one benchmark prediction or classification task supplied by the competitor pack or tutorial workflow.
If multiple tasks are available, choose one primary task and one secondary task.

## Split
Use the official split if provided. Otherwise define train / validation / test once and freeze it.

## Baselines
- raw-access model
- DIGIT
- at least one DP baseline

## Metrics
- AUROC
- AUPRC
- calibration / Brier score
- retained utility relative to raw baseline
- utility-privacy frontier against DP baseline

## Acceptance target
DIGIT should preserve nontrivial utility and show a favorable privacy-utility tradeoff versus the matched DP baseline.

## Report
Include main metric, confidence interval across seeds, and one plot showing privacy versus utility.
