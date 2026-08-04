# DIGIT Validation — Context of Use

> **Status: FROZEN — 2026-03-24**
> This document is locked for the current validation cycle. Substantive changes require a new validation cycle and a fresh checklist.

## System under evaluation
DIGIT PoC (`experiments/DIGIT/PoC`) as of the frozen commit on the `origins` branch. The Feasibility and Extrapolation variants are out of scope for this validation cycle.

## Purpose
DIGIT is a constrained-query privacy architecture intended to enable useful scientific analysis on sensitive data while reducing direct leakage of record-level information through a discrete bottleneck.

## Primary intended environments
This validation package targets scientific workflows in drug development, life sciences, and pharma-adjacent research settings.

## Intended uses in this validation phase
- privacy-preserving scientific query answering on sensitive biomedical data
- exploratory analysis support for genomics, clinical, and drug-response datasets
- comparison against raw-access and differential privacy baselines under matched tasks

## Benchmark datasets in scope
- NIST Genomics PPFL competitor pack (privacy stress-testing)
- PRISM drug-response dataset (utility evaluation)
- TCGA (omics realism and subtype prediction)
- MIMIC-IV demo subset (healthcare sanity check)

No other datasets are evaluated in this cycle. Dataset versions and splits are fixed in `Validation/datasets/`.

## Users assumed in scope
- trusted evaluators acting as research scientists
- computational biologists
- privacy / security reviewers
- ML researchers evaluating privacy-utility tradeoffs

Anonymous public users and unrestricted third-party access are out of scope in this cycle.

## Privacy unit
The privacy unit for this validation cycle is one subject record (one person / patient / participant), including linked records belonging to the same subject where applicable.

## Attacker model
The attacker is assumed to have:
- black-box access to DIGIT outputs
- knowledge of the domain, task type, and query format
- possible background knowledge about candidate records or subgroups
- no direct access to raw data, hidden prompts, internal traces, or internal intermediate state

## Attacker goals in scope
- membership inference
- attribute inference
- subgroup presence inference
- reconstruction / inversion
- linkage
- adaptive query exploitation
- composition stress under repeated querying

## Query budget for evaluation
- Maximum **1,000 queries per evaluation run** (single dataset, single adversary)
- Adaptive query attack budget: **500 sequential queries** with full response history
- Composition stress test: **10,000 queries** across a single dataset, logged in full

These bounds apply to privacy attack evaluations. Utility benchmark runs use fixed benchmark-specific protocols defined in `Validation/datasets/` and associated configs.

## Governance assumptions for this validation cycle
The validation assumes:
- bounded query access
- rate-limited interaction
- no raw data export
- no unrestricted free-form querying without governance controls
- fixed logging and reproducibility settings
- hidden internal traces and hidden internal prompts

Open public deployment is out of scope for this cycle.

## Out of scope in this phase
- autonomous clinical decision-making
- regulatory submissions that rely solely on DIGIT outputs
- unrestricted free-form querying without rate limits or governance controls
- claims of equivalence to differential privacy unless formally proven later

## Core trust claim for validation
DIGIT should preserve useful scientific signal for specific benchmark tasks while limiting leakage through a constrained discrete interface, under a declared threat model and a bounded query regime.

## Success criteria
DIGIT validation is considered successful only if all of the following hold:
1. formal assumptions and limits are clearly documented
2. privacy attacks are run against DIGIT and strong baselines
3. utility is retained on relevant scientific tasks
4. failure cases are surfaced, not hidden
5. outputs are reproducible from code and fixed configs

## Reporting stance
All results must be reported as context-of-use specific. This package is meant to help trusted reviewers understand what DIGIT can do, what it cannot do, and under which assumptions each claim holds.