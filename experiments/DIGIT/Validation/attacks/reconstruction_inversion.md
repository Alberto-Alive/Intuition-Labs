# Reconstruction / Inversion Experiment

## Purpose
Test whether an attacker can reconstruct sensitive record-level or sample-level properties from DIGIT outputs.

## Primary datasets
- NIST Genomics PPFL competitor pack
- TCGA

## Reconstruction targets
Choose targets that matter scientifically and privacy-wise:
- genotype pattern slice
- mutation status vector slice
- biomarker bucket
- small signature profile segment

## Attacker access
The attacker gets DIGIT outputs only, optionally with allowed auxiliary covariates.

## Baselines
- raw non-private system
- DIGIT
- DP baseline

## Procedure
1. Define the reconstruction target before running.
2. Generate outputs under a fixed query budget.
3. Train an inversion model or solve a direct reconstruction problem.
4. Evaluate on held-out targets.
5. Stress-test with repeated adaptive queries.

## Metrics
- exact match rate
- Hamming / reconstruction error
- top-k target recovery rate
- correlation between reconstructed and true signal

## Pass / fail guidance
DIGIT should prevent high-fidelity reconstruction under the allowed budget.
If partial reconstruction is possible, quantify how much information is exposed and through which queries.

## Deliverables
- reconstruction examples
- error distributions
- success vs query budget curves
- risk summary by target type
