# DIGRAG on Real Benchmarks — Does the Evidence Bottleneck Survive Outside the Synthetic Sandbox?

*Real-benchmark transfer (LongMemEval), mechanism ablation, and generalization
(MultiHop-RAG) for the DIGRAG evidence-bottleneck hypothesis, plus a semi-real
conflict/freshness stress-test. Same decoder, same embedder, same retrieval
budget, same context-token budget across every system.*

> **Framing.** Grep can often find the right evidence, but **retrieval recall is
> not answer correctness.** We test whether a typed evidence bottleneck between
> retrieval and decoding improves grounded QA. This is **evidence for or against
> a hypothesis**, not a claim that "DIGRAG beats RAG." The result is **partial
> support**: the typed packet helps where applicability/recency/multi-evidence
> matter, does nothing for simple lookups, and the deterministic gate is an
> abstention knob (not an accuracy lever).

---

## Result separation (read this first)

| Track | Where | One-line result |
|---|---|---|
| **Synthetic** (sandbox) | [`../README.md`](../README.md) | Schema-aware DIGIT-style packet → ~1.00 on engineered traps (architectural demo). |
| **LongMemEval** (real) | §3 | DIGRAG-raw 0.348 best; gate over-abstains; wins on knowledge-update. |
| **Ablation** (mechanism) | §4 | Packet > same-passage stuffing (+0.073); buckets-only collapses. |
| **MultiHop-RAG** (real, generalization) | §5 | DIGRAG-raw 0.449 best; recall ≠ accuracy; gate best on `null_query`. |
| **Conflict stress** (semi-real, separate) | §6 | DIGRAG 0.452 > lexical 0.386 on stale/contradiction. |

## TL;DR result

| Benchmark (decoder = Qwen2.5-1.5B-Instruct, 3 seeds) | Lexical | Vector | Hybrid | GrepRAG | **DIGRAG-raw** | DIGRAG-gated |
|---|---|---|---|---|---|---|
| **LongMemEval** final-answer acc | 0.296 | 0.259 | 0.285 | 0.256 | **0.348** | 0.267 |
| **MultiHop-RAG** final-answer acc | 0.407 | 0.347 | 0.324 | 0.287 | **0.449** | 0.380 |
| **Conflict stress** (semi-real) | 0.386 | 0.357 | 0.343 | 0.238 | **0.452** | 0.452 |

Seed std is ≈0.02–0.03, so the DIGRAG-raw gains are outside seed noise. The
decoder, embedder, top-k, and **context-token budget (≈600 tok) are identical**
across systems — the only thing that changes is whether the retrieved passages
are stuffed as chunks or compiled into a typed evidence packet.

**Headline mechanism (the cleanest controlled comparison).** Holding the
retrieved passages fixed (the same hybrid top-k), compiling them into a typed
packet beats stuffing them as chunks: **0.391 → 0.318** (LongMemEval ablation,
`B_no_gate` vs `H_stuff_passages`). That +0.073 is the bottleneck effect.

**The thesis figure** (`results/plots/recall_vs_answer.png`): every system sits
far below the recall=accuracy diagonal, and on MultiHop **lexical retrieval has
the *highest* supporting-evidence recall (0.859) yet a *lower* answer accuracy
(0.407) than DIGRAG (recall 0.711, accuracy 0.449).** Recall ≠ correctness.

---

## 1. Setup (fairness first)

- **One decoder for everything:** `Qwen2.5-1.5B-Instruct` (same as the synthetic
  DIGRAG experiment, for continuity), greedy/deterministic decoding.
- **One embedder/reranker:** `all-MiniLM-L6-v2` + `ms-marco-MiniLM-L-6-v2`.
- **Same retrieval budget:** top-k = 6 passages for every system.
- **Same context-token budget:** ≈600 tokens of context payload for every
  system; the DIGRAG packet is trimmed (lowest-overlap units dropped) to fit the
  *same* budget as the chunk systems (measured mean context tokens are within
  ~5%: ~560 vs ~590).
- **Same answer prompt contract:** `ANSWER: …` / `CITES: …` for all systems.
- **No gold leakage:** retrieval and the DIGRAG compiler read only the question,
  an optional question date, and the retrieved passage text/timestamps. Gold
  answers / gold evidence are used **only** by the scorer.
- **Scoring:** deterministic surface judge — normalized exact-match OR
  containment (short gold) OR token-F1 ≥ 0.5. No LLM judge (reproducible). This
  *understates* absolute accuracy but is identical across systems.
- **Seeds:** 3 seeds control the random question subsample; decoding/retrieval
  are deterministic. The abstention threshold τ is **calibrated on a dev split
  (seed 999), never on the test seeds.**

## 2. Systems

`lexical` (BM25), `vector` (dense), `hybrid` (BM25⊕dense RRF + cross-encoder
rerank), `greprag` (LLM writes keyword queries → BM25), `digrag_raw` (hybrid
passages → typed evidence packet, constrained decode, **no gate**),
`digrag_gated` (+ dev-calibrated answerability/abstention gate).

### The generic DIGRAG compiler (`compiler.py`)

Strictly benchmark-agnostic — no trap schemas, no dataset-specific fields. From
the *same* retrieved passages it builds sentence-level **evidence units**, each
tagged with `pid` + timestamp, plus generic typed spans (dates, numbers, named
entities via capitalized-n-gram heuristics, quotes, ids), then:

- **freshness logic:** parse heterogeneous timestamps, mark the `most_recent`
  unit, gated on *generic* temporal markers in the question (`current`, `now`,
  `latest`, `last`, `as of`, …) — not on any benchmark label;
- **conflict flags:** contradiction cues + disagreeing values on
  question-relevant units;
- **answerability:** question↔unit lexical overlap;
- **constrained decoding:** answer only from the packet, cite pids, prefer the
  most-recent unit for "current/latest" questions.

The **gate** (gated variant only) abstains when answerability < τ (τ
dev-calibrated per benchmark).

## 3. Experiment 1 — LongMemEval transfer

500-question long-term-memory chat QA; each question has its own ~500-turn
haystack with per-session dates (temporal-reasoning + knowledge-update types).
Full table: [`results/tables/main_results.md`](results/tables/main_results.md).

- **DIGRAG-raw is best overall (0.348)**, +0.05 over the best baseline (lexical
  0.296), driven by **knowledge-update 0.489 vs hybrid 0.277** and temporal-
  reasoning 0.208 (best). On **simple single-session-user lookups lexical wins
  (0.867)** — the bottleneck does not help easy exact lookups, as predicted.
- **Supporting-evidence recall is ~equal** for hybrid and DIGRAG (0.966); the
  accuracy gap is therefore a *decoding/grounding* effect, not retrieval.
- **The gate hurt overall accuracy here** (τ calibrated to 0.40 → over-abstains
  on answerable questions: answerable acc 0.343 → 0.219), while maximizing
  abstention recall (0.42 → 0.90). The gate is a *precision/abstention* knob.

## 4. Experiment 2 — Mechanism / ablation (LongMemEval)

Full table + deltas: [`results/tables/ablation_results.md`](results/tables/ablation_results.md).
`![ablation](results/plots/ablation.png)`

| Ablation | Final-answer acc | Reading |
|---|---|---|
| Full DIGRAG (gated) | 0.327 | gate present |
| − gate (raw packet) | **0.391** | gate *removal* helps accuracy (+0.064) |
| − constrained decoding | 0.364 | packet content matters more than the instruction |
| − conflict flags | 0.336 | small effect on aggregate |
| − temporal/freshness | 0.327 | no aggregate effect (helps only temporal subset) |
| − typed buckets (context only) | 0.282 | buckets contribute (−0.109 vs raw packet) |
| chunk-stuffing, **same passages, no packet** | 0.318 | **packet > stuffing by +0.073** |
| − context (**buckets only**) | **0.127** | catastrophic — buckets need prose context |

**Which component matters?** The **typed packet as a whole** (buckets *and*
context, with provenance) is the lever. Buckets-only collapses; context-only
drifts; the *combination*, compiled with provenance, beats stuffing the same
passages. **The deterministic gate is not an accuracy component** — it trades
answerable accuracy for abstention recall and only earns its keep on
unanswerable questions (see §5, MultiHop `null_query`: gated 0.868 vs raw 0.509).

## 5. Experiment 3 — Generalization (MultiHop-RAG)

2,556 multi-hop news questions over a shared 609-doc corpus (comparison /
inference / temporal / **null**=unanswerable). Table:
[`results/tables/generalization_results.md`](results/tables/generalization_results.md).

- **DIGRAG-raw generalizes (0.449, best)**, with the largest gains on
  **temporal_query 0.446 vs lexical 0.191** (+0.25) and inference 0.730 (best).
- **Recall ≠ correctness, sharply:** lexical retrieval has the highest support
  recall (0.859) but lower accuracy (0.407) than DIGRAG (recall 0.711, acc
  0.449). Compiling beats retrieving-more.
- **The gate shines exactly where it should:** unanswerable `null_query`
  abstention is best with the gate (0.868) and worst with raw DIGRAG (0.509).
- **Honest downside:** DIGRAG *raises* the unsupported-claim rate (0.116 vs
  ~0.06 for baselines) — surfacing typed candidate entities leads the small
  decoder to stitch some ungrounded tokens. The gate lowers it (0.088).

## 6. Conflict / freshness stress-test (semi-real, **separate**)

Built from MultiHop by injecting one **stale, contradictory** variant of a gold
document (decoy value, older date) per question; correct = the *current* value.
Table: [`results/tables/conflict_stress.md`](results/tables/conflict_stress.md).
DIGRAG (0.452) > lexical (0.386) > hybrid (0.343): the freshness/conflict logic
helps when documents contradict across time. Clearly marked semi-synthetic;
never mixed into the main scores.

## 7. Conclusion — does the evidence support the hypothesis?

**Partially — and informatively.**

- **Supports** the bottleneck hypothesis: a typed evidence packet improves
  final-answer accuracy over chunk-stuffing the *same* retrieved passages,
  **consistently across LongMemEval, MultiHop-RAG, and the conflict stress-test
  and in the controlled ablation**, with gains *not* explained by retrieval
  recall and concentrated in temporal / knowledge-update / multi-evidence
  questions.
- **Does not support** a universal win: gains are modest (~+0.05 absolute at
  this model scale), simple single-session lookups are better served by plain
  lexical search, and the deterministic abstention gate is a calibration knob
  that can *hurt* accuracy while helping abstention.
- **Caveat:** DIGRAG trades a higher unsupported-claim rate for higher
  correctness; the gate reverses that trade.

So: **the evidence bottleneck is a real, transferable effect, but a targeted one
— not a free lunch.**

## 8. Limitations

- Small decoder (1.5B) and a surface-form judge depress *absolute* accuracy
  (~0.25–0.45); a stronger decoder would lift all systems and likely shrink (not
  necessarily close) the gap. All systems share the model, so comparisons hold.
- Subsampled test sets (90–150/seed) for tractable, repeated runs; seed std is
  reported.
- The generic extractors (regex/heuristic NER, date parsing) are noisy on open
  text; a learned extractor would change DIGRAG's ceiling.
- The conflict track is semi-synthetic and reported separately.

## 9. Related work

- **"Is Grep All You Need?" (arXiv:2605.15184)** — lexical grep is a strong
  agentic-search substrate. We corroborate that **lexical recall is high** (and
  often the strongest baseline) yet show recall ≠ answer correctness.
- **"GrepRAG" (arXiv:2601.23254)** — optimizes grep-like retrieval for code
  completion; our `greprag` baseline reproduces the agentic-grep idea and is
  bottlenecked by the small query-writer.
- **DIGIT / synthetic DIGRAG (`../`)** — the typed-bottleneck idea this work
  ports from a synthetic sandbox to real benchmarks.

## 10. Reviewer objections (and our answers)

- **Synthetic bias?** This evaluation is on *real* benchmarks (LongMemEval,
  MultiHop-RAG). The only synthetic part is the clearly-separated conflict track.
- **Schema-aware extraction?** The real compiler uses only generic types
  (entities/dates/numbers/quotes/ids/temporal & conflict markers). No
  benchmark-specific fields; the ablation shows the effect is the *packet*, not
  any hand-coded field.
- **Deterministic-gate advantage?** The gate is reported as an *ablation* and
  separately from `digrag_raw`. It does **not** drive the accuracy gain — it
  often *hurts* accuracy and only helps abstention. The headline system is
  `digrag_raw` (no gate).
- **Oracle leakage?** Systems never read gold. Gold is used only by the scorer
  and (for the conflict track) by the offline corpus-construction step. τ is
  calibrated on a dev split, never test.
- **Benchmark-specific tuning?** Only τ is tuned, on dev. No prompts, rules, or
  fields are tuned per benchmark; the same compiler/prompt run on all.
- **Token-budget fairness?** Enforced: identical ≈600-token context budget;
  measured mean context tokens are within ~5% across systems (the packet is
  trimmed to fit). DIGRAG does **not** win by spending more tokens.
- **Do gains come from retrieval, prompting, or the bottleneck?** Not retrieval
  (recall is equal or *lower* than the best baseline). Not the prompt (the
  `−constrained decoding` ablation barely moves). The controlled
  `packet vs same-passage stuffing` comparison (+0.073) isolates the **evidence
  bottleneck** as the source.

## 11. Reproduce

```bash
pip install -r ../requirements.txt          # plus: huggingface_hub, datasets
python -m real_benchmarks.run_all --seeds 0,1,2          # all 4 blocks, one model load
python -m real_benchmarks.report                          # tables + plots
# single block:
python -m real_benchmarks.run --benchmark longmemeval --systems main --seeds 0,1,2 --limit 90
python -m real_benchmarks.run --benchmark longmemeval --systems ablation --seeds 0 --limit 110
python -m real_benchmarks.run --benchmark multihop --systems main --seeds 0,1,2 --limit 150
```

Artifacts per block in `results/<block>/`: `metrics.json`, `predictions.jsonl`,
`retrieved_evidence.jsonl` (DIGRAG packets), `per_question.jsonl`. Tables in
`results/tables/`, figures in `results/plots/`.

## 12. Layout

```
real_benchmarks/
  adapters/longmemeval.py · multihop.py     normalize to shared schema (gold = eval-only)
  schema.py                                  Passage / Example / SystemOutput
  retrieval.py                               BM25 · dense · hybrid(RRF+rerank) · greprag + budgeted select
  compiler.py                                GENERIC typed evidence packet (no benchmark fields)
  systems.py                                 6 systems + 8 ablations, prompts, parse, gate
  metrics.py                                 deterministic judge + all metrics (seed mean/std)
  conflict_stress.py                         semi-real stale/contradiction track
  run.py / run_all.py                        batched GPU driver (dev-calibrated τ)
  report.py                                  tables + plots + failure analysis
```
