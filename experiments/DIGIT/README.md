# DIGIT

DIGIT is a goal-directed interface for interacting with protected scientific datasets.

Instead of asking for raw-data access, a researcher asks whether a dataset can answer a specific question. DIGIT translates that question into structured intent, runs only approved analysis inside the protected environment, and returns bounded outputs rather than raw records.

`Question -> intent -> approved query -> bounded statistics -> coarse answer`

![DIGIT system diagram](public/digit/digit.png)

## Why This Exists

In biomedicine and other sensitive domains, many valuable datasets remain scientifically underused. The data exists, but the path from a research question to a usable answer is blocked by privacy, governance, access, and tooling friction.

DIGIT starts from a different premise: many data bottlenecks are really interface bottlenecks. The goal is not broad raw-data access. The goal is bounded evidence.

## Core Idea

- Researchers ask whether a dataset can answer a question.
- DIGIT encodes that question into structured scientific intent.
- Only predefined, governable query families are allowed to execute.
- Computation happens inside the protected environment.
- Responses are intentionally coarse, such as `yes`, `no`, `maybe`, `blocked`, with support and confidence signals.

This can range from a directly queryable tool to a more autonomous supervisor that helps keep experiments grounded in what the dataset can and cannot answer.

## How DIGIT Works

1. An encoder translates a plain-language question into structured intent: subgroup, outcome, comparison, and evidence target.
2. The intent is routed to an approved query type that can run safely inside the protected environment.
3. An executor computes bounded aggregate statistics over relevant records without exposing raw data.
4. A bottleneck compresses the result into a coarse answer and readable explanation.

## Why The Bottleneck Matters

The bottleneck is the main architectural move.

- Raw records stay inside the protected environment.
- Allowed query families are predefined and governable.
- Outputs are intentionally coarse rather than fully analytical.
- Exposure can be tuned at both the query layer and the answer layer.

The claim is not zero privacy risk. The claim is that remaining exposure can be made bounded, testable, and tunable.

## Current Public Read

The public validation summary below reflects the published table in `experiments/DIGIT/Validation/PUBLIC_RESULTS_TABLE.md` as of March 27, 2026.

| Theme | Current read |
| --- | --- |
| Strongest signal | DIGIT materially reduces membership-inference leakage relative to raw, but the patched DP rerun is better than DIGIT on `nist_genomics / mechanism`. |
| Deployed interface | In `system` mode, membership-inference results are generally near chance and roughly comparable to the best tested DP point. |
| Main weakness | Plain direct-query DIGIT remains weak on differencing attacks. |
| Practical direction | A stability-gated wrapper materially improves the main differencing weakness and brings rich DIGIT close to DP on that benchmark. |

### Headline Numbers

| Attack | Dataset | Mode | Raw AUROC | DIGIT AUROC | Best DP AUROC |
| --- | --- | --- | ---: | ---: | ---: |
| Membership inference | `nist_genomics` | mechanism | `1.000` | `0.592` | `0.534` |
| Membership inference | `nist_genomics` | system | `0.545` | `0.538` | `0.536` |
| Membership inference | `tcga` | mechanism | `0.585` | `0.539` | `0.536` |
| Differencing attack | `nist_genomics` | mechanism | `0.884` | `0.880` | `0.821` |

The `nist_genomics / mechanism / membership inference` DP value above comes from the patched rerun in `experiments/DIGIT/Validation/results/mia_nist_mechanism_dp_patched/phase1_privacy/membership_inference_summary.json`, where the best tested DP point is `dp_laplace_e0p1` at AUROC `0.534`.

The honest current framing is not "DIGIT replaces DP." DIGIT still shows clear value relative to raw access, but the patched membership-inference rerun does not support a clean DIGIT-over-DP claim.

## What The Interface Leaves Behind

If every question must pass through a structured, governed representation, the interaction trace itself becomes useful. DIGIT can accumulate a reusable map of:

- what a dataset can answer
- where evidence repeatedly converges
- which important questions keep failing
- how experiments relate to the bounded evidence available

That map of scientific intuition is not an extra feature. It is a structural consequence of the interface.

## What Would Falsify DIGIT

- If repeated querying can recover membership or sensitive attributes too reliably, the privacy claim fails.
- If bounded outputs destroy too much signal to guide real scientific decisions, the utility claim fails.
- If interaction traces do not accumulate into anything scientifically useful at scale, the infrastructure claim weakens substantially.

## Repository Guide

- [`digit.json`](digit.json): source research-note content that originally described the idea in site/article format.
- [`experiments/DIGIT/PoC`](experiments/DIGIT/PoC): proof-of-concept implementation and benchmarks on a controlled dataset.
- [`experiments/DIGIT/Validation`](experiments/DIGIT/Validation): privacy, utility, robustness, governance, and dataset-validation work.
- [`experiments/DIGIT/Validation/PUBLIC_RESULTS_TABLE.md`](experiments/DIGIT/Validation/PUBLIC_RESULTS_TABLE.md): public-safe summary of current validation claims.
- [`experiments/DIGIT/Extrapolation`](experiments/DIGIT/Extrapolation): extrapolation experiments and supporting traces.
- [`reports/OverviewReport.md`](reports/OverviewReport.md): higher-level adoption and impact framing.

## Status

DIGIT is best read as a proof-of-concept for a new scientific interface layer, not as a finished product.

The main claim is architectural: useful interaction and controlled exposure do not have to be opposites.
