# Privacy Attacks Plan

This folder contains the attack battery for Validation.

## Required attack families
- membership inference
- attribute inference
- subgroup presence detection
- reconstruction / inversion attempts
- linkage with auxiliary information
- repeated adaptive query attacks
- composition stress tests

Every attack should be run against DIGIT, raw baseline, and at least one DP baseline where feasible.

The comparative attack battery is implemented through the `Validation.runners.run_*` modules, with `python3 -m Validation.runners.run_comparative_suite` as the top-level orchestration entrypoint.

For a single smoke+suite entrypoint from the repo root, use `python3 experiments/DIGIT/Validation/scripts/run_all_comparative.py`.

## Interim results snapshot

Current completed head-to-head evidence is the comparative membership inference benchmark in
`Validation/results/phase1_privacy/membership_inference_summary.json`.

Interpretation note: lower `auroc_mean` means less leakage. The table below is privacy-only;
utility and repeated-query results are still incomplete.

| Dataset | Mode | DIGIT | Best DP | Raw | Current read |
| --- | --- | ---: | ---: | ---: | --- |
| nist_genomics | mechanism | 0.592 | 0.534 (`eps=0.1`, patched rerun) | 1.000 | DIGIT clearly beats raw, but patched DP is better in this setting |
| nist_genomics | system | 0.538 | 0.536 (`eps=0.5`) | 0.545 | DIGIT is essentially tied with DP, slightly better than raw |
| tcga | mechanism | 0.539 | 0.537 (`eps=0.25`) | 0.585 | DIGIT is roughly tied with the best DP setting, better than raw |
| tcga | system | 0.493 | 0.490 (`eps=2.0`) | 0.477 | All systems are near chance; DIGIT is competitive but not best |

Current takeaways:
- DIGIT shows meaningful privacy value relative to raw access and is competitive with DP in some settings, but it does not currently show a clean win over the best tested DP point.
- The patched `nist_genomics` `mechanism` rerun no longer supports the earlier DIGIT-over-DP read for that setting.
- `tcga` in `system` mode is the weakest DIGIT result so far; the gap to the best DP setting is small, but DIGIT does not lead there.
- No overall DIGIT-vs-DP claim should be made until the utility frontier and repeated-query attacks are complete.
