# TCGA Omics Utility Task

## Goal
Test whether DIGIT preserves meaningful omics signal on a realistic cancer dataset.

## Primary task
Subtype classification or biomarker-status prediction on a clearly defined cancer cohort.
Pick one cancer type with enough support for a first pass.

## Secondary task
Risk-group or survival-stratification surrogate if labels and preprocessing are stable enough.

## Split
Use a frozen train / validation / test split by patient.
Do not allow leakage across aliquots or related samples.

## Baselines
- raw-access model
- DIGIT
- DP baseline

## Metrics
- AUROC / macro F1 for classification
- calibration
- subgroup utility gaps if clinically relevant
- retained utility relative to raw baseline

## Acceptance target
DIGIT should keep the scientific signal usable enough to support downstream interpretation, while sharply lowering attack success in the paired privacy experiments.

## Report
Document cohort definition, preprocessing, labels, and all exclusions.
