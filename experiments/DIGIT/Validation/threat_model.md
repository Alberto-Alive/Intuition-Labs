# DIGIT Validation — Threat Model

## Privacy unit
The privacy unit for this validation cycle is **one subject record**.

For benchmark-specific execution, this maps to the dataset’s natural protected unit:
- patient for MIMIC-IV demo style tasks
- individual sample / case for TCGA tasks
- individual profile / sample for genomics stress tests
- experimental row / cell-line response record for PRISM, where that row is the protected unit for the task

## Attacker knowledge
Attackers may know:
- the task domain
- the allowed query format
- the DIGIT method class
- benchmark protocols and public evaluation setup

Auxiliary-information attackers may additionally know:
- subgroup prevalence
- public cohort statistics
- partial labels
- known exemplars or candidate records

Unless explicitly stated for a white-box evaluation, attackers do **not** know:
- raw private data
- hidden prompts
- internal traces
- unreleased intermediate state

## Query budget
This threat model uses the frozen budgets defined in `context_of_use.md`:
- Maximum **1,000 queries per evaluation run** (single dataset, single adversary)
- Adaptive query attack budget: **500 sequential queries** with full response history
- Composition stress test: **10,000 queries** across a single dataset, logged in full

Changing these bounds requires a new validation cycle.

## Adversary classes
### A1. External observer
Sees released DIGIT outputs and knows the method class, but does not control internal training.

### A2. Querying adversary
Can submit repeated allowed queries and adapt future queries based on prior answers.

### A3. Auxiliary-information adversary
Has outside knowledge such as subgroup prevalence, public cohort statistics, partial labels, or known exemplars.

### A4. White-box evaluator
Has model and code access for red-team benchmarking.

## Primary attack goals
- membership inference
- attribute inference
- subgroup presence detection
- reconstruction / inversion attempts
- linkage using auxiliary information
- adaptive query steering to reduce uncertainty
- composition attacks across repeated queries

## Trust assumptions
- the DIGIT implementation matches the released code and config
- query budgets, rate limits, and logging are enforced during evaluation
- datasets are split correctly with no leakage from evaluation into model tuning

## Non-goals
This validation package does not assume DIGIT prevents all possible leakage. It is designed to measure, bound, and document leakage under declared conditions.

## Key questions each experiment must answer
1. What is the protected unit?
2. What can the attacker observe?
3. What prior knowledge is allowed?
4. How many queries can be made?
5. What constitutes a successful breach?
6. How does DIGIT compare with raw access and DP baselines?