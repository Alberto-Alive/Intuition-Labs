DIGIT End-to-End: How It Actually Works
The Problem
You have private data (think medical records, financial data). Someone asks a question about it. You want to answer in natural language without leaking the underlying records. Differential privacy adds noise — DIGIT instead structurally limits what information can escape.

The Five Components
1. Query Encoder (learned)
query_encoder.py

Input: a numeric vector of 8 query fields (e.g., "age range", "region", "condition" encoded as numbers).

Each field is projected to a 256-d token via a linear layer, positional embeddings are added, then it runs through a 4-layer transformer encoder. The output is mean-pooled into a single 256-d vector z_q.

This is standard — nothing novel here. It just gives you a rich representation of "what was asked."

2. Safe Private Executor (deterministic, NOT learned)
safe_executor.py

This is the privacy firewall. It's not a neural net — it's deterministic code with torch.no_grad(). It takes the query + private dataset and computes only approved aggregates:

support_ratio: what fraction of records match this group (a single float)
variance_bucket: one-hot over 4 buckets (how spread out is the data)
confidence_bucket: one-hot over 3 buckets (based on group size — small group = low confidence)
min_group_size_pass: binary — did the group have at least 5 records?
policy_flags: 3-way one-hot — [too_specific, too_general, normal]
Output dimension: 12 floats total. No raw rows leave this module. If a group has fewer than 5 records, that's flagged and the model is trained to say "insufficient evidence."

3. Primitive Bottleneck Head (learned)
bottleneck.py

This is the key innovation. It takes z_q (256-d) concatenated with executor features (12-d) = 268-d input. Runs through a 2-layer MLP, then four independent linear classification heads produce logits for:

Primitive	Classes	What it means
Answer	6	YES, NO, MAYBE, INSUFFICIENT_EVIDENCE, INCONSISTENT_SIGNAL, POLICY_BLOCKED
Support	4	VERY_LOW, LOW, MEDIUM, HIGH
Confidence	3	LOW, MEDIUM, HIGH
Risk	3	LOW, MEDIUM, HIGH
The logits are then discretized. During training, this uses Gumbel-Softmax (hard=True) — forward pass produces a true one-hot vector, but gradients flow through the soft approximation. Temperature anneals from 2.0 → 0.5 over 10k steps. At inference, it's just argmax.

Why this matters: After discretization, the output is exactly 4 one-hot vectors. That's 6 × 4 × 3 × 3 = 216 possible combinations — 7.75 bits maximum. The decoder physically cannot receive more information about the private data than this.

4. Decoder (learned)
decoder.py

A 4-layer transformer decoder. It cross-attends to exactly 5 memory tokens:

Query token: z_q projected through a linear layer
Answer token: one-hot → linear embedding
Support token: one-hot → linear embedding
Confidence token: one-hot → linear embedding
Risk token: one-hot → linear embedding
Then it autoregressively generates text tokens. Teacher-forced during training, greedy at inference.

The constraint is architectural: the decoder's memory input is built in decoder.py:81-101 from only the primitives + query. There's no skip connection, no side channel to the executor or private data. It's not a loss penalty hoping to prevent leakage — the decoder literally has no tensor path to access anything else.

5. Loss Function (5 terms)
losses.py

Loss	What it does	Weight
Primitive	Cross-entropy: did the bottleneck predict the right answer/support/confidence/risk?	1.0
Generation	Cross-entropy: did the decoder produce the right tokens?	1.0
Policy	Soft penalty: if executor says "too specific," the answer should be POLICY_BLOCKED; if group too small, should be INSUFFICIENT_EVIDENCE	0.5
Leakage	Squared correlation between decoder logits and executor features — a mutual information proxy. Penalizes the model if the decoder's outputs are statistically correlated with private aggregates beyond what the primitives encode	1.0
Abstention	Encourages MAYBE/INSUFFICIENT when support_ratio < 3% — don't be confident about tiny groups	0.3
The leakage loss (losses.py:129-159) is a belt-and-suspenders measure. The architecture already prevents leakage structurally, but this loss catches any residual statistical correlation that might leak through the query encoding path.

Training Dynamics
Optimizer: AdamW, lr=3e-4, weight_decay=0.01, cosine annealing
Gumbel temperature: starts at 2.0 (soft, easy to optimize) → anneals to 0.5 (nearly hard discrete) over 10k steps. This is critical — if you start with hard argmax, the bottleneck heads can't learn because gradients are zero.
Gradient clipping: max norm 1.0
Validation: uses mode="hard" (true argmax) to measure actual discrete performance
What the Baselines Are
Baseline A: Skip the neural net entirely. Executor produces aggregates → hand-coded rules map to YES/NO/MAYBE (3 classes only) → template text. No learning.
Baseline B: Same executor → more sophisticated hand-coded rules map to all 4 primitives (6+4+3+3 classes) → template text. Still no learning.
These exist to show that the learned bottleneck heads classify better than hand-coded thresholds, and the learned decoder generates better text than templates.

Data Flow Summary (what to put on a slide)

         LEARNED              NOT LEARNED           LEARNED (the gate)        LEARNED
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐    ┌──────────────┐
│ Query Encoder   │    │  Safe Executor   │    │  Bottleneck Head    │    │   Decoder    │
│                 │    │                  │    │                     │    │              │
│ 8 fields        │    │ private data +   │    │ z_q + agg_feats    │    │ 5 memory     │
│ → 4L transformer│    │ query → 12 safe  │    │ → MLP → 4 logit    │    │ tokens only  │
│ → mean pool     │    │ aggregate floats │    │   heads → Gumbel   │    │ → 4L xformer │
│ → z_q (256-d)   │    │ (no raw rows)    │    │   → 4 one-hot vecs │    │ → text       │
└────────┬────────┘    └────────┬─────────┘    └──────────┬──────────┘    └──────────────┘
         │                      │                         │
         │              ┌───────┘              ┌──────────┘
         │              │                      │
         └──────────────┼──► CONCAT ───────────┘
                        │      ↓
                        │   268-d input
                        │      ↓
                        │   4 discrete primitives (7.75 bits max)
                        │      ↓
                        └──► decoder ONLY sees these + z_q
The Question You'll Get Asked
"Why not just use differential privacy?"

DP adds calibrated noise to query answers and tracks a cumulative privacy budget — after enough queries, you must stop answering or the budget is spent. DIGIT's bound is per-query and structural: each query leaks at most 7.75 bits regardless of how many queries came before. There's no budget to exhaust. The tradeoff is that DIGIT's answers are coarser (categorical primitives, not numeric), but for "should I worry about X" type questions, that's often enough.

"What stops the query encoder from encoding private data into z_q?"

The encoder never sees private data. It only receives the query fields. z_q is a function of the question alone. The private data enters only through the executor, which produces 12 approved aggregate floats, which then go through the discrete bottleneck. The decoder sees z_q (query info) + primitives (discretized private info). Even if the query correlates with private records, the bottleneck caps what can cross.