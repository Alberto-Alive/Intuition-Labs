# MIMIC-IV Demo Sanity Check

## Goal
Validate that the DIGIT pipeline behaves sensibly on realistic healthcare-style structured data before moving to full-access clinical datasets.

## Important note
MIMIC-IV demo is a sanity-check dataset, not a final clinical validation dataset.

## Candidate tasks
Choose one or two simple tasks:
- length-of-stay bucket prediction
- mortality surrogate if support is sufficient
- phenotype / diagnosis bucket prediction

## Split
Use a patient-level split and freeze it.

## Baselines
- raw-access model
- DIGIT
- DP baseline

## Metrics
- AUROC or balanced accuracy
- calibration
- retained utility relative to raw baseline
- runtime / stability notes

## Acceptance target
DIGIT should run end-to-end cleanly, preserve basic predictive utility, and expose no obvious implementation issues before you invest in full MIMIC-IV access.

## Report
Explicitly label this as a pipeline sanity check, not a headline scientific claim.
