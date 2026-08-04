# Utility Evaluation Plan

This folder defines how DIGIT will be evaluated for scientific usefulness after privacy protection is applied.

The automated runner is governed by [LOCKED_PROTOCOL.md](/mnt/w/Intuition-Labs/experiments/DIGIT/Validation/utility/LOCKED_PROTOCOL.md), which now defines task-specific fair utility benchmarks for `raw`, `digit`, and `dp`.

## Principle
A privacy method is only credible for scientific settings if useful signal survives.

## Required comparisons
- DIGIT vs raw-access baseline
- DIGIT vs one or more DP baselines
- DIGIT ablations with weaker / stronger bottlenecks

## Required reporting
- retained utility relative to raw baseline
- utility delta relative to DP baselines
- confidence intervals across seeds
- failure modes and degraded subgroups

## Automated Runner
- entrypoint: `python3 -m Validation.runners.run_utility_frontier`
- default published mode: `system`
- outputs:
  - `phase2_utility/privacy_utility_frontier_summary.json`
  - `phase2_utility/privacy_utility_frontier_summary.md`

## Current fair automated tasks
- `nist_genomics`: binary `eur_ancestry`
- `mimic_iv_demo`: binary `hospital_expire` sanity check
- `tcga`: multiclass `morphology_group`, removed from the utility query schema
- `prism`: held-out pair regression/ranking on `response`, with response-derived query fields removed
