# Composition Stress Test

## Purpose
Estimate cumulative privacy risk as the total number of queries, users, or sessions grows.

## Primary datasets
- all validation datasets

## Procedure
1. Define total budget scenarios: small, medium, large.
2. Aggregate outputs across repeated sessions and simulated users.
3. Re-run the strongest attacks from this folder at each budget level.
4. Compare cumulative risk against DIGIT design assumptions.

## Metrics
- strongest attack success vs total budget
- cumulative leakage proxy vs budget
- number of sessions until practical breach

## Pass / fail guidance
DIGIT should have a documented safe operating region. If performance degrades beyond a threshold, record the threshold and turn it into a governance control.

## Deliverables
- budget-risk table
- safe operating region summary
- recommended deployment limits
