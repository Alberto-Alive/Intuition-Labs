# Comparative Attack Plan — DIGIT vs DP-Laplace

> **Purpose:** Determine whether DIGIT's discrete bottleneck provides better, worse, or equivalent
> privacy to DP-Laplace at equivalent utility. The current MIA results cannot answer this because
> the group query interface (`min_group_size=10`) dominates and masks both mechanisms equally.

> **Governing rule:** All DIGIT-vs-DP conclusions will be reported both at matched query policy
> and at matched downstream utility, with uncertainty intervals across seeds.

---

## 0. Evaluation modes

Every attack is run in two modes. Results must be reported separately and labeled clearly.

| Mode | `min_group_size` | Purpose |
|------|-----------------|---------|
| **Mechanism** | 1 (individual queries allowed) | Isolates the privacy mechanism itself — DIGIT discretization vs DP noise — with no shared interface protection. Required to make any DIGIT-vs-DP comparison. |
| **System** | 10 (production policy) | Reflects what users actually face. Shows what the deployed system leaks, not what the mechanism would leak in isolation. |

Without the mechanism mode, critics can say "the policy layer hid the real difference."
Without the system mode, critics can say "you tested an unrealistic configuration."
Both are needed.

---

## 1. The required ablation first

**Run MIA in mechanism mode (`min_group_size=1`) before any other comparison.**

This is the prerequisite for every comparison below. With the production policy on, every
attack produces the same result for DIGIT and DP because the interface dominates.

Expected direction in mechanism mode (not absolute claims — report what you measure):
- RAW is expected to leak substantially more than in system mode
- DP is expected to show partial protection proportional to ε
- DIGIT is expected to show protection from output coarseness alone
- The gap between them is what the comparison is actually measuring

---

## 2. Privacy-utility tradeoff curve

The proper comparison is a curve, not a single number: how much utility do you lose per unit
of privacy gained? DIGIT and DP must be compared at matched utility points, not arbitrary
parameter settings.

**Privacy axis:** MIA AUROC (lower = more private)

**Utility axis — two levels, both required:**
- *Internal*: primitive prediction accuracy (answer, support, confidence, risk) — useful for
  diagnosing the system but not the final claim
- *End-task*: downstream classification AUROC, subgroup retrieval quality, or drug-response
  ranking quality depending on dataset — this is what trust claims must rest on

**Curve construction:**
- DP curve: ε ∈ {0.1, 0.5, 1.0, 3.0} already measured; add more points if the curve is sparse
- DIGIT curve: vary answer classes ∈ {2, 3, 4, 6} and/or support/confidence/risk granularity
- Overlay both curves on the same axes
- If DIGIT's curve sits above DP's at the same privacy level, it delivers more utility per unit
  of privacy — that is the defensible comparative claim

---

## 3. Attacks where DIGIT has a structural advantage over DP

### 3a. Query averaging / noise averaging attack

DP noise can be reduced by averaging. Issue the same query N times, average the N responses —
noise variance drops as 1/N. With enough queries an attacker can recover the true statistic
to arbitrary precision given unlimited budget.

DIGIT's discrete output may resist this: averaging 100 responses of "YES" still gives "YES".
The output coarseness may set a hard ceiling on precision that repeated queries cannot overcome.
This is an expected structural advantage — not a guaranteed one, because a deterministic system
can be probed if the attacker can bracket bucket boundaries.

- Run against DP at each ε: how many repeated queries are needed to recover the true rate
  within ±0.01? Plot queries-to-recovery vs ε.
- Run against DIGIT: measure how close the attacker's estimate can get to the true statistic
  as query count grows. Show whether there is a floor it cannot cross.
- Report query budget explicitly; stay within the harness budget cap.

### 3b. Differencing attack

Issue Q1 = broad query (e.g. all females aged 30–40), then Q2 = same query with one additional
filter that excludes a target person. The difference Q1 − Q2 isolates the target's contribution.

- With DP: each difference leaks O(sensitivity/ε) information per query pair. With enough
  pairs the target's label becomes recoverable. Privacy budget accumulates per pair.
- With DIGIT: the discrete output may not change between Q1 and Q2 if the target's
  contribution does not cross a bucket boundary. Information leaked per pair is bounded
  by the coarseness of the output, not by ε.
- Measure: label inference accuracy vs number of differencing pairs for each system.
- Report the privacy budget consumed by the DP attacker to reach 80% accuracy.

### 3c. Reconstruction via repeated targeted queries

Can an attacker reconstruct the label of a specific individual by issuing many targeted queries
around that person's subgroup?

- With DP: sufficient queries can average away noise, especially for rare-attribute individuals
  who appear in few matching subgroups.
- With DIGIT: output coarseness sets a precision floor — the attacker cannot distinguish
  "rate = 0.31" from "rate = 0.28" if both fall in the same bucket. Reconstruction accuracy
  may plateau below the true label regardless of query count.
- Measure: reconstruction accuracy vs query budget, plotted as a curve per system.

---

## 4. Attacks that apply equally to both

### 4a. Attribute inference

The attacker knows some attributes of a target and uses query responses to infer a hidden
sensitive attribute (e.g., disease status, EUR ancestry, mortality risk).

- Issue queries with known attributes as filters; train a classifier on the responses to
  predict the hidden attribute.
- Compare inference accuracy across systems at matched query budgets.
- Expected: both DP and DIGIT provide some protection; DP protection scales with ε;
  DIGIT protection scales with output coarseness. Neither guarantee is formal — measure both.
- This is the most realistic threat model for medical and genomic use cases.

### 4b. Canary detection

Insert a synthetic record with rare or unique attribute values into the dataset. Issue
highly specific queries targeting that record's attributes.

- With RAW: canary is likely detectable via exact statistics.
- With DP: detection requires queries to average away noise; budget accumulates.
- With DIGIT: high-specificity queries return POLICY_BLOCKED or INSUFFICIENT_EVIDENCE under
  the production policy, potentially preventing detection regardless of budget.
- Measure: queries required to confirm canary presence at 95% confidence for each system,
  in both mechanism and system modes.

### 4c. Linkage attack

The attacker has access to two systems backed by overlapping datasets. Issue queries to both
and try to link records across datasets using correlated responses.

- Tests whether the output format creates a cross-system fingerprint.
- DIGIT's coarser output may reduce linkage precision relative to DP-noised statistics, but
  a correlated output structure could still enable linkage if the primitives are predictable.
- Measure: precision and recall of correct record linkage vs query budget.

### 4d. Composition stress

Issue a large query budget (e.g., 10,000 queries) with incremental refinements and measure
cumulative leakage as a function of query count.

- DP has a formal composition theorem: total leakage is bounded by the sum of per-query ε.
  DIGIT has no equivalent formal bound.
- This test measures DIGIT's empirical composition curve and checks whether it stays flat
  or rises with budget.
- Plot: MIA AUROC vs cumulative query count for each system.
- A flat curve is strong evidence. A rising curve must be reported as a failure.
- Report the query count at which each system's AUROC exceeds 0.55, 0.60, 0.70 as
  budget thresholds.

---

## 5. Adversarial / white-box attacks (hardest)

### 5a. Gradient-based query optimization

Use surrogate gradients (Gumbel-Softmax or REINFORCE) to optimize query inputs toward
outputs that maximally expose a target record.

- DIGIT's non-differentiable discrete output may frustrate gradient-based approaches.
- Measure: best MIA AUROC achieved by an optimized attacker vs a random attacker at
  matched query budgets.

### 5b. Model extraction / surrogate training

Issue diverse queries, collect responses, and train a surrogate model that approximates
the private dataset's statistical distribution.

- Issue 10,000 diverse queries; train a neural network on (query → response).
- Measure surrogate accuracy on held-out queries vs a ground-truth model.
- DIGIT's coarser output should produce less accurate surrogates than DP-noised outputs.

### 5c. Auxiliary knowledge amplification

The attacker has a partial public dataset (e.g., 1000 Genomes reference panel, published
clinical summary statistics) that overlaps with the private dataset. Combine auxiliary
knowledge with query responses to amplify inference beyond what either source allows alone.

- Realistic for genomics (reference panels are public) and clinical data.
- Measure: attribute inference accuracy with vs without auxiliary data, for each system.
- Tests whether DIGIT's output leaks information on top of what is already publicly known.

---

## 6. Statistical reporting requirements

Every attack result must include:

| Field | Requirement |
|-------|-------------|
| Seeds | All 5 seeds (0–4); reduced (0–2) only if marked and justified |
| Summary statistic | Mean ± std across seeds |
| Uncertainty | 95% bootstrap confidence interval across seeds |
| Query budget | Exact number of queries used by the attacker, per seed |
| Attacker knowledge | Explicit statement of what the attacker knows (member list, aux data, etc.) |
| Pass/fail criterion | Pre-registered threshold (e.g., AUROC < 0.60 = pass for MIA) |
| Both modes | Mechanism mode (min_group_size=1) and system mode (min_group_size=10) reported |

No result may be reported as a point estimate without an uncertainty interval.

---

## 7. Priority order

| Priority | Attack | Mode | Why |
|----------|--------|------|-----|
| 1 | MIA ablation | Mechanism | Prerequisite for all comparisons |
| 2 | Query averaging (3a) | Both | DIGIT's clearest structural advantage over DP |
| 3 | Differencing (3b) | Both | Most realistic real-world attack on statistical query systems |
| 4 | Privacy-utility curve (§2) | Both | Required for any defensible "better than DP" claim |
| 5 | Attribute inference (4a) | Both | Most realistic threat for medical/genomic use cases |
| 6 | Composition stress (4d) | System | Required to make any claim about query budget safety |
| 7 | Reconstruction (3c) | Both | Strong evidence if DIGIT resists and DP does not |
| 8 | Canary detection (4b) | Both | Clean, simple, good for public communication |
| 9 | Model extraction (5b) | Mechanism | Hardest to defend against; important for completeness |
| 10 | Gradient optimization (5a) | Mechanism | Academic / adversarial completeness |
