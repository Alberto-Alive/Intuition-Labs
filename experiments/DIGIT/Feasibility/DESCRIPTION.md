# DIGIT — Discrete Intuition Gated Encoder-Decoder

## What it is

DIGIT is a transformer encoder-decoder that answers natural-language queries over private data without exposing that data. Between the encoder and decoder sits a **discrete bottleneck** — a hard gate that compresses all private information into exactly four categorical primitives:

| Primitive | Classes |
|-----------|---------|
| Answer | YES, NO, MAYBE, INSUFFICIENT_EVIDENCE, INCONSISTENT_SIGNAL, POLICY_BLOCKED |
| Support | VERY_LOW, LOW, MEDIUM, HIGH |
| Confidence | LOW, MEDIUM, HIGH |
| Risk | LOW, MEDIUM, HIGH |

The decoder only ever sees these primitives and the encoded query. It has no attention path to raw data.

## Architecture

```
Query → [Transformer Encoder] → z_q ─────────────────────┐
                                  │                        │
                                  ▼                        ▼
Private Data → [Safe Executor] → agg → [Bottleneck] → 4 discrete   → [Transformer Decoder] → text
               (deterministic,          (Gumbel-Softmax    primitives
                no gradients)            / argmax)
```

**Encoder.** Each query field is projected to a token and processed by a 4-layer transformer encoder, then mean-pooled to a 256-d vector z_q.

**Safe Executor.** A deterministic, non-learned module that computes aggregate statistics (support ratio, variance bucket, confidence bucket, policy flags) from the private dataset. Enforces a minimum group size of 5. No raw rows leave this module.

**Bottleneck.** Takes z_q + executor features, produces logits for each primitive, then discretizes via Gumbel-Softmax (training) or argmax (inference). This is the hard information gate — at most log2(6x4x3x3) = 7.75 bits per query can cross it.

**Decoder.** A 4-layer transformer decoder that cross-attends to 5 memory tokens: one per primitive embedding plus the projected query. Autoregressively generates bounded intuition text.

## Training

Five loss terms, jointly optimized:

1. **Primitive classification** — cross-entropy on each primitive head
2. **Generation** — cross-entropy on output tokens
3. **Policy violation** — penalizes when primitives contradict executor policy flags (e.g., answering YES when group size is below threshold)
4. **Leakage** — squared correlation between decoder logits and executor features, as a mutual information proxy
5. **Abstention encouragement** — rewards cautious answers (MAYBE, INSUFFICIENT_EVIDENCE) when support is low

Gumbel-Softmax temperature anneals from 2.0 to 0.5 over 10k steps.

## Results (synthetic data, 5.5M params)

| System | Answer Acc | Notes |
|--------|-----------|-------|
| Baseline A (template, 3-class) | 0.197 | Rule-based |
| Baseline B (template, all primitives) | 0.156 | Rule-based |
| DIGIT (learned) | 0.316 | Best val loss 1.659, leakage 0.012 |

The learned model roughly doubles answer accuracy over the baselines while keeping leakage at 0.012 — meaning the decoder extracts negligible private information beyond what the discrete primitives explicitly encode.

## Privacy guarantee

The bottleneck bounds worst-case leakage to 7.75 bits per query regardless of private dataset size. In practice, measured leakage is far lower. This is a structural property of the architecture, not a statistical one — it holds without noise injection or a privacy budget.
