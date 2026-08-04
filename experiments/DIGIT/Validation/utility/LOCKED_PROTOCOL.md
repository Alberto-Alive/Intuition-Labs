# Locked Fair Utility Protocol

This file defines the **fair automated utility runner** used by
`Validation.runners.run_utility_frontier`.

The automated runner is now task-specific, not a generic proxy fallback:
- each dataset has a locked downstream target
- forbidden query fields are removed before utility evaluation
- train / validation / test splits are fixed and reproducible
- the runner refuses datasets with no protocol entry

## Automated Utility Tasks

| Dataset | Task type | Locked target | Query fields removed | Role |
|---|---|---|---|---|
| `nist_genomics` | Binary classification | `eur_ancestry` | none | Headline |
| `mimic_iv_demo` | Binary classification | `hospital_expire` | none | Sanity-check |
| `tcga` | Multiclass classification | `morphology_group` | `morphology_group` | Headline |
| `prism` | Regression / ranking | `response` | `response_bin`, `abs_response_bin`, `sensitivity_flag` | Headline |

## Locked Rules

- The target column must not remain in the utility query schema.
- Target-derived query fields must not remain in the utility query schema.
- The automated runner uses a frozen split rule per dataset.
- `system` mode is the default published utility mode.
- Internal primitive accuracy is diagnostic only; it is not the headline utility claim.

## Task Adapter Notes

- `nist_genomics` and `mimic_iv_demo` use direct binary targets.
- `tcga` uses one-vs-rest binary subproblems internally because the current DIGIT/raw/DP query stack is binary-rate based.
- `prism` uses a threshold ladder of binary subproblems internally and reports regression/ranking metrics on held-out pair targets.

## Consequence

- `run_utility_frontier` now evaluates the locked fair tasks above.
- There is no public fallback to the old generic proxy-label utility benchmark.
- Any future dataset added to the utility runner must first receive a locked protocol entry.
