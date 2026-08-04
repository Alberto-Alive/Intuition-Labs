# Adaptive Repeated-Query Attack

## Purpose
Test whether a smart attacker can reduce uncertainty by steering follow-up queries based on previous DIGIT answers.

## Primary datasets
- all validation datasets

## Attacker access
The attacker can choose the next query after seeing the previous answer, subject to the interface limits defined in threat_model.md.

## Baselines
- DIGIT under standard query budget
- DIGIT under tighter budget / abstention rule
- DP baseline if comparable

## Procedure
1. Define an attacker objective: membership, attribute recovery, subgroup detection, or reconstruction.
2. Let the attacker adaptively choose queries for a fixed budget.
3. Record attack success after each additional query.
4. Repeat over multiple budgets and stopping rules.

## Metrics
- breach rate vs query count
- information gain per query
- first-breach query index
- abstention / rejection rate if supported

## Pass / fail guidance
DIGIT should not collapse under repeated querying. The breach curve should stay flat or degrade slowly, and any high-risk query chains should be documented and blocked in governance rules.

## Deliverables
- attack traces
- budget curves
- list of dangerous query templates
- recommended hard limits for deployment
