# DIGIT Validation Checklist

## Phase 0 — framing
- [x] Context of use is frozen and specific enough to evaluate.
- [x] Threat model names the privacy unit, attacker knowledge, attacker goals, and query budget.
- [x] Common evaluation harness is fixed: seeds, baselines, metrics, logging, and report format.

## Phase 1 — privacy evidence
- [ ] Membership inference run completed.
- [ ] Comparative MIA mechanism-mode ablation run completed.
- [ ] Attribute inference run completed.
- [ ] Subgroup presence run completed.
- [ ] Reconstruction / inversion run completed.
- [ ] Linkage attack run completed.
- [ ] Adaptive query attack run completed.
- [ ] Composition stress run completed.
- [ ] DIGIT is compared against raw access and at least one DP baseline.
- [ ] DIGIT-vs-DP results are reported separately for mechanism and system modes.
- [ ] Privacy failures are written up honestly, with examples.

## Phase 2 — scientific utility
- [ ] NIST genomics utility benchmark run completed.
- [ ] PRISM drug-response benchmark run completed.
- [ ] TCGA subtype prediction benchmark run completed.
- [ ] MIMIC-IV demo sanity benchmark run completed.
- [ ] Utility metrics are reported side-by-side for DIGIT, raw, and DP baselines.
- [ ] Matched privacy-utility frontier is reported for DIGIT variants vs DP epsilons.
- [ ] Utility losses are quantified, not hand-waved.

## Phase 3 — robustness and trustworthiness
- [ ] Robustness sweeps run across seeds, budgets, and bottleneck settings.
- [ ] Failure cases are collected and categorized.
- [ ] Governance and release criteria are documented.
- [ ] Revalidation triggers are documented.
- [ ] Reproducibility instructions are complete enough for another person to rerun the core results.

## Phase 4 — final claims
- [ ] Validation report states what DIGIT can do.
- [ ] Validation report states what DIGIT cannot do.
- [ ] Validation report avoids claiming DP-equivalence unless formally proven.
- [ ] Recommended safe deployment conditions are stated.
- [ ] Recommended unsafe or unsupported uses are stated.
