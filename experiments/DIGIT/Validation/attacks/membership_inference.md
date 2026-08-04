# Membership Inference Experiment

## Purpose
Test whether an attacker can tell if a protected unit was part of the source data used to answer or train the DIGIT-backed system.

## Primary datasets
- NIST Genomics PPFL competitor pack
- MIMIC-IV demo
- TCGA

## Protected unit
Define one protected unit per dataset before running:
- NIST: one individual or one sample
- MIMIC-IV demo: one patient stay / patient record
- TCGA: one patient case / sample

## Attacker access
The attacker gets:
- DIGIT outputs for chosen queries
- metadata allowed by the threat model
- no raw records

Run two settings:
1. black-box only
2. black-box + limited side information

## Baselines
- raw non-private system
- DIGIT
- at least one DP baseline matched for utility target

## Procedure
1. Build matched member / non-member examples.
2. Generate outputs under the same query budget for all methods.
3. Train a shadow attacker if applicable.
4. Evaluate on held-out targets.
5. Repeat over at least 5 seeds.

## Metrics
- ROC AUC
- TPR at 1% and 5% FPR
- precision / recall at chosen threshold
- attack advantage over random guessing

## Pass / fail guidance
DIGIT should materially reduce attack success relative to raw access.
A practical target for the first validation pass is:
- AUC close to 0.50 and clearly below raw baseline
- no high-confidence operating point with strong TPR at low FPR
- performance competitive with or better than the chosen DP baseline at matched utility

## Deliverables
- attack curves
- per-dataset summary table
- seeds and confidence intervals
- failure-case examples
