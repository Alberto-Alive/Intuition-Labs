# RFQ Theory

## Core object

RFQ treats a trained neural network not as one indivisible computation, but as
a population of possible internal computations exposed by structural
intervention.

A perturbation mask `m` produces a subnetwork:

    f_m

The central observation to test is that many different masks can preserve task
performance while differing substantially in structure.

Some are genuinely redundant:

    f_a approximately equals f_b

Others are complementary:

    f_a and f_b solve overlapping but non-identical subsets of the problem.

RFQ seeks the quotient:

    successful subnetworks / functional equivalence

and then a minimal complementary basis over the quotient classes.

## Stage 1 — intervention population

Generate many structural variants:

    M = {m_1, ..., m_N}

Measure behavior, not only scalar accuracy.

A behavior signature can include:

- predictions,
- calibrated probabilities,
- errors by class,
- responses to corruptions,
- responses to transformations,
- internal-change signatures.

## Stage 2 — strategy discovery

Cluster successful masks by structure and behavior.

Within each family, recursively remove modules while preserving a discovery
criterion.

The result is a collection of sparse working cores.

## Stage 3 — functional quotienting

Two strategies belong to the same equivalence class when they are behaviorally
indistinguishable within a specified evidence budget:

    s_i ~ s_j

The operational test in this package uses:

- prediction agreement,
- Jensen-Shannon divergence,
- optionally transformation-response similarity.

The quotient removes duplicate implementations while preserving at least one
representative from each class.

## Stage 4 — complementary basis selection

The quotient may still contain strategies that are not equivalent but add little
marginal value.

Choose a subset B minimizing cost subject to behavior preservation:

    minimize Cost(B)

    subject to Performance(B) >= Performance(full) - epsilon

A richer objective also preserves:

- OOD performance,
- full-model fidelity,
- oracle coverage,
- diversity of useful errors.

## Stage 5 — sparse conditional routing

For each input x, choose one retained strategy:

    r(x) -> s_k

The router should be substantially cheaper than executing the dense model or all
strategies.

## Recursive extension

Each retained strategy can itself be subjected to RFQ:

    strategy
      -> intervention population
      -> quotient
      -> smaller complementary basis

This is the "recursive" part of Recursive Functional Quotienting.

## What would count as a strong result

A publication-grade result should show that RFQ:

1. discovers multiple non-trivial strategies in ordinary trained models,
2. distinguishes true redundancy from complementary behavior,
3. preserves more robustness or capability than matched pruning baselines,
4. lowers active compute through routing,
5. does so under strictly held-out evaluation,
6. remains consistent across seeds and architectures.
