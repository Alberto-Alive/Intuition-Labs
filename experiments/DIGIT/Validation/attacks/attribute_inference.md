# Attribute Inference Experiment

## Purpose
Test whether an attacker can infer a hidden sensitive attribute of a protected unit from DIGIT outputs plus allowed side information.

## Primary datasets
- NIST Genomics PPFL competitor pack
- MIMIC-IV demo
- TCGA

## Example hidden attributes
Choose one or two per dataset:
- NIST: genotype-derived trait or cohort label
- MIMIC-IV demo: hidden diagnosis bucket or demographic bucket if policy-appropriate
- TCGA: subtype / biomarker bucket held out from attacker view

## Attacker access
The attacker gets:
- DIGIT responses
- non-sensitive covariates allowed by the threat model
- query budget defined in threat_model.md

## Baselines
- raw non-private system
- DIGIT
- at least one DP baseline
- majority-class attacker

## Procedure
1. Select a hidden attribute with enough support.
2. Create attacker features from outputs and allowed metadata.
3. Train attacker models on shadow data.
4. Evaluate on held-out targets.
5. Repeat across seeds and query budgets.

## Metrics
- balanced accuracy
- macro F1
- AUROC if binary
- calibration / Brier score for attacker confidence

## Pass / fail guidance
DIGIT should keep attacker performance close to naive or majority baselines, and materially below raw access.
If an attribute remains inferable, document exactly which interface enables it.

## Deliverables
- table of hidden attributes tested
- attacker performance by method
- query-budget sensitivity plot
- list of leaked attribute patterns if any
