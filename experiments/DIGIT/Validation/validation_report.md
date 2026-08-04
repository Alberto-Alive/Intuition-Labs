# DIGIT Validation Report Template

## 1. Executive summary
Summarize what DIGIT was tested on, what claims were evaluated, and the main privacy-utility findings.

## 2. Context of use
Reference `context_of_use.md` and state the exact use-case for each benchmark.

## 3. Threat model
Reference `threat_model.md` and declare attacker powers, privacy unit, and breach definitions.

## 4. Formal claims and limits
Reference `formal_bounds.md`. Separate proven bounds from empirical observations.

## 5. Benchmark suite
Reference the datasets in `datasets/`:
- NIST Genomics PPFL competitor pack for privacy stress-testing
- PRISM for drug-discovery utility
- TCGA for omics realism
- MIMIC-IV demo for healthcare-style sanity checks

## 6. Baselines
At minimum compare against:
- raw-access baseline
- one or more differential privacy baselines
- ablations of DIGIT that remove or relax the bottleneck

## 7. Privacy attacks
Reference `attacks/`. Report success rates, AUCs, confidence intervals, strongest failure cases, and separate mechanism-mode vs system-mode conclusions.

## 8. Utility results
Reference `utility/`. Report performance retention relative to raw and DP baselines, plus the matched privacy-utility frontier for DIGIT bottleneck variants vs DP epsilons.

## 9. Robustness results
Reference `robustness/`. Report repeated-query behavior, shift sensitivity, and stability across seeds.

## 10. Governance and reproducibility
Reference `governance/`. Document versioning, auditability, configs, and release criteria.

## 11. Honest limitations
State clearly where DIGIT fails, where results are preliminary, and what remains unproven.

## 12. Decision summary
Conclude with one of:
- suitable for research-only use under declared constraints
- promising but not yet trustworthy enough
- unsuitable for the tested context of use
