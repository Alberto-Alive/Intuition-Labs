# DIGIT Validation — Formal Bounds Workplan

## Goal
State the strongest formal claims that DIGIT can honestly support before empirical testing.

## Required derivations
### 1. Per-query output capacity
Derive the maximum information capacity of the bottleneck per query from the discrete output alphabet actually exposed by the implementation.

### 2. Multi-query composition
Bound worst-case cumulative leakage as a function of:
- number of allowed queries
- whether queries are independent or adaptive
- whether outputs are deterministic or stochastic
- whether repeated querying can collapse uncertainty

### 3. Neighboring-world distinguishability
Define a neighboring-world relation appropriate to each benchmark:
- one patient changed / removed
- one sample changed / removed
- one subgroup prevalence changed
Then quantify how distinguishable those worlds are under DIGIT outputs.

### 4. Abstention and uncertainty behavior
If DIGIT uses confidence, support, or abstention channels, bound how these channels contribute to leakage.

### 5. Failure conditions
Document exact conditions under which the formal argument no longer applies, including:
- unlimited repeated querying
- unlogged side channels
- hidden continuous outputs
- cached intermediate states exposed to the attacker

## Deliverables
- theorem-style statements or proposition-style claims
- explicit assumptions list
- composition discussion
- counterexample section
- mapping from formal assumptions to actual code paths

## Minimum acceptable honesty standard
If a claim cannot be formally defended, label it as an empirical observation rather than a guarantee.
