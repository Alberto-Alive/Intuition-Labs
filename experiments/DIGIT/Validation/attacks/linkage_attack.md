# Linkage Attack Experiment

## Purpose
Test whether an attacker can link DIGIT outputs to external auxiliary records belonging to the same person or sample.

## Primary datasets
- MIMIC-IV demo
- TCGA

## Attacker access
The attacker has:
- DIGIT outputs
- an auxiliary table with overlapping but incomplete fields
- a linkage strategy defined in advance

## Baselines
- raw non-private system
- DIGIT
- DP baseline

## Procedure
1. Build or simulate an auxiliary dataset with realistic overlap.
2. Generate DIGIT outputs for a protected set.
3. Attempt one-to-one or top-k matching.
4. Repeat across overlap strengths and noise levels.

## Metrics
- top-1 linkage accuracy
- top-k recall
- precision / recall for matched pairs
- false linkage rate

## Pass / fail guidance
DIGIT should sharply reduce successful linkage relative to raw access, especially in the realistic-overlap setting.

## Deliverables
- linkage setup description
- matching performance table
- sensitivity to auxiliary-data quality
