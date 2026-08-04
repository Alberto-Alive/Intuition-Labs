# Robustness Evaluation Plan

This folder tests whether DIGIT behaves consistently under stress, not just whether it scores well once.

## Required checks
- multiple random seeds
- query-budget sweeps
- bottleneck-size sweeps
- threshold / abstention sensitivity
- dataset shift or split sensitivity
- train / validation / test hygiene checks

## Goal
Show whether DIGIT fails gracefully or catastrophically when assumptions are stressed.
