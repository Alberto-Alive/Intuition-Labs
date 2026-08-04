# PRISM Drug-Response Utility Task

## Goal
Test whether DIGIT preserves signal that is useful for drug-response prediction or ranking.

## Dataset
PRISM Repurposing primary screen.
Work on a reproducible subset first if local compute is limited.

## Primary task
Predict or rank cell line response for held-out drug-cell line pairs.

## Recommended setup
- split by cell line, drug, or pair depending on the claim
- start with pair split for a basic signal-retention test
- add a harder split later that withholds drugs or cell lines

## Baselines
- raw-access model
- DIGIT
- DP baseline

## Metrics
- Pearson / Spearman correlation for continuous response
- RMSE or MAE
- ranking metric such as NDCG or Recall@k if using retrieval/ranking
- retained utility relative to raw baseline

## Acceptance target
DIGIT should keep a clear, useful fraction of raw predictive performance and compare favorably with the DP baseline at similar privacy strength.

## Report
Include split definition, feature set, query interface used by DIGIT, and error bars across seeds.
