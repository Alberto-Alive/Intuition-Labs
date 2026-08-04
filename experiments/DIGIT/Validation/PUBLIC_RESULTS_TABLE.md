# DIGIT Validation Table

As of `2026-03-27`.

Use this file as the source-of-truth summary for public-facing discussion.

## Read First

- Primary metric below is **privacy attack AUROC**.
- **Lower is better**.
- `0.50` is chance.
- "Best DP" means the lowest-leakage DP-Laplace point among the tested epsilons in that setting.
- The `nist_genomics / mechanism / membership inference` DP row below uses the patched rerun in `results/mia_nist_mechanism_dp_patched`, replacing the older stale `0.923` value.
- Primary rows below are based on **full 5-seed runs** reconstructed from per-seed `metrics.json`.
- Exploratory rows are clearly marked and should not be used as the main claim in a deck.

## Primary Results You Can Show

| Attack | Dataset | Mode | Raw AUROC | DIGIT AUROC | Best DP AUROC | Read |
|---|---|---:|---:|---:|---:|---|
| Membership inference | `nist_genomics` | mechanism | `1.000` | `0.592` | `0.534` | DIGIT is dramatically better than raw, but the patched best-DP rerun is better than DIGIT in this setting. |
| Membership inference | `nist_genomics` | system | `0.545` | `0.538` | `0.536` | All systems are near chance; DIGIT is roughly at parity with DP. |
| Membership inference | `tcga` | mechanism | `0.585` | `0.539` | `0.536` | DIGIT improves over raw and is close to the best DP point. |
| Membership inference | `tcga` | system | `0.477` | `0.493` | `0.490` | All systems are near chance; do not over-interpret small deltas. |
| Differencing attack | `nist_genomics` | mechanism | `0.884` | `0.880` | `0.821` | DIGIT is only slightly better than raw and worse than DP here. This is the main privacy weakness. |
| Differencing attack | `nist_genomics` | system | `0.836` | `0.835` | `0.823` | DIGIT is close to raw and slightly worse than DP. |
| Differencing attack | `tcga` | mechanism | `0.988` | `0.989` | `0.984` | Everyone is highly vulnerable in this setting, including DIGIT and DP. |
| Differencing attack | `tcga` | system | `0.986` | `0.987` | `0.983` | Everyone is highly vulnerable in this setting, including DIGIT and DP. |

## Public-Safe Narrative

| Theme | Supported statement |
|---|---|
| Strongest result | DIGIT shows real privacy value on membership-inference style attacks relative to raw access. |
| Deployed-interface result | In system mode, DIGIT is generally near chance on MIA and roughly comparable to DP. |
| Main weakness | DIGIT is weak on differencing attacks in its plain direct-query form. |
| Honest comparison vs DP | DIGIT is **not** a clean across-the-board winner over DP, and the patched `nist_genomics / mechanism` rerun removes the earlier apparent DIGIT-over-DP win there. |

## Exploratory But Promising

These results are useful in conversation, but they are **not** the primary Public table because they are reduced-seed or wrapper experiments.

| Experiment | Setting | Plain DIGIT | Modified DIGIT | Best DP | Read |
|---|---|---:|---:|---:|---|
| Stability-gated wrapper on rich DIGIT (`a6_s4_c3_r3`) | `nist_genomics / mechanism / differencing` | `0.885` | `0.830` | `0.828` | A simple stability-gated public wrapper closed almost all of the gap to DP without changing the internal rich DIGIT bottleneck. |
| Coarser DIGIT sweep | `nist_genomics / mechanism / differencing` | `a6 = 0.885` | `a2 = 0.838` | `0.828` | Coarser buckets help, but not nearly as much as the wrapper. |

## Do Not Put In The Deck As Primary Evidence

| Result | Why not primary |
|---|---|
| `composition_stress_summary.json` | Final AUROCs are informative, but the threshold/accounting fields are easy to misread and should not be a lead table. |
| Query averaging DIGIT-vs-DP comparison | The current DP path is bugged in evaluation, so that comparison is not Public-safe yet. |
| Reduced 3-seed quick sweeps | Useful for direction-finding, not for headline claims. |

## Suggested Public Framing

Use this exact framing:

1. DIGIT already shows meaningful privacy value relative to raw access on membership-inference attacks.
2. The patched `nist_genomics / mechanism` rerun puts the best tested DP point ahead of DIGIT on that benchmark.
3. DIGIT's main weakness is direct-query differencing.
4. The product opportunity is therefore not "DIGIT alone replaces DP", but "DIGIT plus governed release logic can provide a commercially useful privacy-governed analysis interface."

## Source Files

- Membership summary: [membership_inference_summary.json](/mnt/w/Intuition-Labs/experiments/DIGIT/Validation/results/phase1_privacy/membership_inference_summary.json)
- Differencing per-seed runs: [phase1_privacy](/mnt/w/Intuition-Labs/experiments/DIGIT/Validation/results/phase1_privacy)
- Stability-gate trial: [differencing_attack_summary.json](/mnt/w/Intuition-Labs/experiments/DIGIT/Validation/results/stability_gate_trial/phase1_privacy/differencing_attack_summary.json)
- Composition stress summary: [composition_stress_summary.json](/mnt/w/Intuition-Labs/experiments/DIGIT/Validation/results/phase1_privacy/composition_stress_summary.json)
