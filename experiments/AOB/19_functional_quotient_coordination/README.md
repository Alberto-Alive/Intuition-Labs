# Stage 19: Functional Quotient Coordination (FQC)

Tests whether AOB can scale many identical shared-weight clones **without
externally forcing them to cover distinct evidence**, by maintaining an
evolving functional map of (1) what work is already accomplished, (2) which
contributions are functionally redundant, and (3) what unresolved residual
remains.

The mechanism adapts the RFQ principle (see `rfq_reference/`, extracted from
`RFQ_Publication_Grade_Research_Package_2026-07-18`) from pruning subnetworks
to coordinating agent contributions:

```text
shared goal
-> clones propose work in parallel (private randomness only; no scheduler,
   no reservations, no identity slots)
-> encode each contribution relative to the goal
-> quotient functionally equivalent/duplicate contributions
-> estimate uncovered residuals
-> redirect redundant clones toward high-value unresolved residuals
-> repeat
```

The optimization target is marginal contribution to unresolved global work,
not representational diversity. FQC never receives the ground-truth
decomposition of task parts.

## Layout

```text
code/fqc_core.py          quotient-class state + residual map (numpy)
code/run_experiment_a.py  Experiment A: controlled mechanism test (Stage 18 world)
code/run_experiment_b.py  Experiment B: decisive HotpotQA test (Stage 16 setup)
tests/                    unit tests for the core and both experiments
results/                  raw CSV/JSON/JSONL results + RESULTS.md
rfq_reference/            vendored copy of the RFQ package essentials
```

Stages 16 and 18 are imported by file path and are not modified.

## Experiment A (controlled)

Stage 18's coordinator is *centralized*: `sequential_select` reserves distinct
work for each agent inside a round. Experiment A removes the scheduler: every
clone proposes in parallel from the same shared quotient state, one public
intent-quotient pass collapses equivalent intents (relative-novelty rule), and
redundant clones are redirected once toward the residual (or stand down when
unresolved work is scarcer than the team). Duplicates are measured against the
ground-truth basis exactly as in Stage 18.

Run: `python code/run_experiment_a.py --seeds 3 --tasks-per-seed 12 --tag full`

## Experiment B (decisive, HotpotQA)

Builds on Stage 16's best variant (`scratch_continuous_distinct_multi_support`)
but removes `learned_distinct`'s forced no-replacement window routing. Clones
pick windows independently (natural overlap); FQC quotients the picks by
position-free window-content similarity and redirects redundant clones by
question-relevance minus coverage. Killer controls: quotient disabled,
reassignment disabled, shuffled contents, random reassignment, lexical-only
similarity, and a duplicated-evidence stress test where index-distinctness no
longer implies content-distinctness.

Run: `python code/run_experiment_b.py --seeds 0,1,2 --scales 4,8,16 --tag medium`

## Verdict

FQC solves the non-duplication problem forced allocation was solving: it
matches the centralized reservation coordinator in the controlled world up to
N=64 (flat duplicate fraction, oracle-bound rounds at zero noise), replaces
forced distinct routing at N=4 QA (full coverage, 3/3 seeds above independent
clones), and beats forced routing when evidence contains duplicates. It does
not fix the Stage 16 N=8/16 QA collapse — no coordination scheme does,
including forced coverage itself — because that failure is a learned-relevance
/dilution failure (only oracle support routing escapes it). Full gate-by-gate
evaluation and Stage 16/18 comparisons: `results/RESULTS.md`.
