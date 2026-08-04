# Subgroup Presence Detection Experiment

## Purpose
Test whether an attacker can determine that a sensitive subgroup is present in the protected data at all.

## Primary datasets
- NIST Genomics PPFL competitor pack
- TCGA

## Example subgroup definitions
- rare genotype or mutation pattern
- cancer subtype / biomarker-defined subgroup
- small cohort with a distinctive signature

## Attacker access
The attacker can issue aggregate or targeted queries allowed by DIGIT and observe the responses.

## Baselines
- raw non-private aggregates
- DIGIT
- DP baseline

## Procedure
1. Define subgroup templates before any testing.
2. Construct worlds with and without the subgroup.
3. Issue the same query program in both worlds.
4. Train or score an attacker to distinguish the two worlds.
5. Repeat over subgroup sizes and prevalence levels.

## Metrics
- attack success rate
- AUROC for world discrimination
- minimum detectable subgroup size

## Pass / fail guidance
DIGIT should require a much larger subgroup size or query budget than raw access before reliable detection becomes possible.

## Deliverables
- minimum detectable subgroup table
- attack success vs subgroup size curves
- examples of high-risk query patterns
