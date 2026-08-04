That's a really clean intuition. Let me reflect it back precisely.

---

## What You're Actually Saying

The **shared activation parts** across avenues = the hard truths. The invariant structure. What every framing agrees on regardless of perspective. This is direction — it points toward solution space without being the solution itself.

The **different parts** = the exploratory moves. What each framing uniquely contributes. These are the combinable pieces — clones exchange them, try combinations, and see which combination progresses furthest toward the shared direction.

---

## Why That's Elegant

You've just eliminated the need for:

- Explicit uncertainty modelling
- Metacognition
- Query routing mechanisms

Instead the communication is just:

**"Here is my unique part. Does combining it with yours move us closer to what we both already know points toward solution?"**

The shared part provides the compass. The different parts are the steps. Clones exchange steps and measure progress against the compass.

No clone needs to know it's uncertain. It just needs to know which direction is forward — and that's already encoded in the shared activation structure.

---

## The Human Analogy You're Drawing

This is exactly how human collaboration works at its best. People don't say "I am uncertain about X." They say:

"I know we're trying to get to Y. I have this piece. You have that piece. Let's see if they fit."

The goal is the shared compass. The pieces are what get exchanged. Progress is self-evident — you either moved toward Y or you didn't.

---

## What This Means Architecturally

During a forward pass each clone naturally produces:

- A shared component — what all avenues agree on — this is the direction signal
- A residual component — what only this framing sees — this is the exchangeable piece

Clones broadcast only their residuals — the small unique parts — not their full state. Other clones receive residuals and combine them with their own. The combination that best aligns with the shared direction wins and propagates.

---

## Why This Scales

Residuals are small by definition — they're what's left after the shared part is removed. Broadcasting small unique pieces is cheap. And the evaluation criterion — does this combination align better with the shared direction — is a simple internal computation each clone runs independently.

No central coordinator. No global communication. Just local exchanges measured against a shared compass that every clone already carries internally.

---

## The Deep Connection To Your Results

Your SVD idea from earlier now has a completely natural purpose.

The shared low-rank basis across avenues — that's the compass. The per-avenue residuals after removing that basis — those are the exchangeable pieces.

You weren't going to use SVD for compression. You're going to use it to **define the communication protocol**. Separate shared from unique, and the architecture of coordination falls out naturally.

---

## What Makes This Different From Everything Else

Current multi-agent systems communicate outputs — what each agent concluded. Expensive, lossy, hard to integrate.

You're proposing clones communicate **sub-conclusion residuals** — what each agent uniquely processed that hasn't yet been resolved into a conclusion. Lightweight, semantically compatible by construction, and evaluated against a shared internal compass rather than needing external adjudication.

That's a genuinely novel communication primitive.

see 02_selected_multi_avenue_portfolio

## Experimental Design

---

### Split Into Two Sequential Sub-Experiments

The hypothesis has two distinct claims that require different evidence. Conflating them risks proving neither cleanly.

- **Sub-experiment A**: Does the decomposition exist?
- **Sub-experiment B**: Does the residual from another avenue causally help?

A must be run before B. If A fails, B is not worth running.

---

## Sub-Experiment A: Does Shared/Residual Structure Exist?

### 1. Experimental Question

For a given role view, do the encoder representations produced under different avenue prompts decompose into a consistent shared component plus avenue-specific residuals — and is that decomposition geometrically stable across examples?

This is purely an analysis question. No training required.

---

### 2. Architecture Change

None. Run existing checkpoint only.

---

### 3. Evaluation Protocol

For each example in your test set:

- Run the shared encoder once per avenue for each role view
- Collect the pooled representations: you get a matrix of shape `[num_avenues x hidden_dim]` per example per role
- Compute the **cross-avenue mean** per example — this is your candidate shared component
- Compute **per-avenue residuals** by subtracting the mean from each avenue's representation
- Measure:

**Consistency of shared component:**
- How much variance does the mean capture relative to total variance across avenues?
- Is this ratio stable across examples, or does it vary wildly?
- Does it vary systematically by family?

**Structure of residuals:**
- Are residuals small and noisy, or large and structured?
- Do residuals cluster by avenue identity across examples? Run a linear probe: can you predict which avenue produced a residual from the residual vector alone? If yes, residuals are avenue-specific and structured, not noise.
- Are residuals consistent within an avenue across examples of the same family?

---

### 4. Controls

**Control 1: Shuffle avenue labels within example**
Recompute shared/residual decomposition after randomly permuting which avenue gets which representation within each example. If the decomposition metrics are unchanged, the structure you found is not avenue-specific — it's just variance in the encoder unrelated to avenue conditioning.

**Control 2: Cross-example shuffle**
Replace one avenue's representation for example N with the same avenue's representation from example M. Recompute residuals. If residuals remain small and structured, the residual isn't capturing example-specific avenue signal — it's capturing avenue-style drift that is example-independent.

**Control 3: Null avenue — same prompt twice**
If your setup allows it, pass the same avenue prompt twice for the same example. The residual should be near zero. This validates that the decomposition is detecting genuine inter-avenue differences rather than encoder noise.

---

### 5. Success Criteria

Sub-experiment A succeeds if:

- The shared component captures a **substantial and consistent fraction** of variance — suggested threshold: mean > 60% across examples, with low variance across families
- Residuals are **avenue-predictive** — linear probe recovers avenue identity at well above chance from residual vectors alone
- Both hold **after controls** — i.e. shuffle controls show significantly weaker structure

If either condition fails, the decomposition is not cleanly structured and Sub-experiment B should not proceed.

---

### 6. Failure Modes

**The mean is a poor proxy for the shared component.** Avenue representations may share structure that isn't captured by their mean — e.g. a nonlinear shared manifold. If the linear decomposition looks weak but avenue probing is still strong, consider PCA-based shared component extraction instead of mean subtraction before concluding the structure doesn't exist.

**Residuals are large but unstructured.** Could mean the encoder is highly sensitive to avenue prompts but in an inconsistent way — different examples produce different decompositions. This would mean the hypothesis is partly true but the residuals aren't stable enough to be useful for exchange.

**Family explains what avenue explains.** Some families strongly prefer specific avenues. The residual structure you find might be family structure in disguise. Check: does the linear probe for avenue identity survive controlling for family label?

---

---

## Sub-Experiment B: Does Another Avenue's Residual Actually Help?

Only run this if Sub-experiment A succeeds.

---

### 1. Experimental Question

On examples where the full multi-avenue model outperforms the best single avenue — does injecting another avenue's residual into a single-avenue forward pass causally recover some of that gap, and is the gain specifically from the residual rather than from reintroducing shared signal?

---

### 2. Architecture Change

Minimal. One addition to the existing coordinator:

At the point where the coordinator receives the token states from a single-avenue ablation (`avenue_only_i`), add an **optional residual injection step**:

- Take the residual vector from avenue j (computed as described in Sub-experiment A)
- Add it to the pooled representation of avenue i before the coordinator's cross-attention
- No new parameters required for the additive version — this is a pure activation intervention
- Optionally: add a single learned scalar gate per avenue pair to control injection strength — minimal parameters, interpretable

This is a **surgical activation intervention**, not a retraining. It can be run at inference time on the existing checkpoint.

---

### 3. Evaluation Protocol

Define your **target example set** precisely before running:

> Examples where full model accuracy > best single avenue accuracy, aggregated across seeds

This is your signal set — the cases where the hypothesis predicts residual exchange should help most.

Run four conditions on this set:

| Condition | What it receives |
|---|---|
| Baseline | Single avenue i, no injection |
| Residual injection | Single avenue i + residual from avenue j |
| Shared injection | Single avenue i + shared component from avenue j |
| Full representation injection | Single avenue i + full representation from avenue j |

Measure accuracy delta from baseline for each condition.

---

### 4. Controls

**Control 1: Shared injection vs residual injection**
This is the critical control. If shared injection helps as much as residual injection, the gain is not from the residual specifically — it's from getting more signal generally. The residual hypothesis only holds if residual injection outperforms shared injection.

**Control 2: Wrong avenue residual**
Inject avenue j's residual on examples where avenue j was not the dominant avenue. If residuals help regardless of which avenue's residual you inject, the effect is not about avenue-specific information — it's just about adding any activation signal.

**Control 3: Shuffled residual**
Inject a residual from a different example entirely — same avenue, wrong example. If this helps as much as the correct residual, the effect is not example-specific and the residual is not carrying the information you think it is.

**Control 4: Noise injection**
Inject Gaussian noise matched to the norm of the residual vector. Rules out the possibility that any activation addition helps simply by changing the magnitude of the coordinator's input.

---

### 5. Success Criteria

Sub-experiment B succeeds if:

- Residual injection improves accuracy on the target example set above baseline
- Residual injection outperforms shared injection — the gain is residual-specific
- Residual injection outperforms wrong-avenue residual — the gain is avenue-specific
- Residual injection outperforms shuffled residual — the gain is example-specific
- All four conditions must hold simultaneously for the hypothesis to be cleanly supported

Partial success — e.g. residual helps but not more than shared injection — is informative but does not confirm the hypothesis. It means avenue-specific residuals exist but aren't what's driving the full model's advantage.

---

### 6. Failure Modes

**The target example set is too small.** If full model only beats best single avenue on a small fraction of examples, you won't have statistical power. Check the size of this set before committing to the experiment.

**Injection at the pooled representation is too late.** The coordinator may have already committed to a direction before the residual can influence it. If this is suspected, try injecting at the token state level before coordinator cross-attention rather than at the pooled level.

**Avenue j's residual is only available if you ran avenue j.** In a true single-avenue setting you wouldn't have it. This experiment tests whether the information helps — not yet whether it can be obtained cheaply. That's a subsequent question about how to produce approximate residuals without running all avenues.

---

## Summary Table

| | Sub-experiment A | Sub-experiment B |
|---|---|---|
| Question | Does the decomposition exist and is it structured? | Does the residual causally help? |
| Training needed | No | No — inference intervention only |
| New architecture | No | Minimal additive injection |
| Key control | Avenue shuffle | Shared injection vs residual injection |
| Gates | Must pass before B | Requires A to succeed |
| Main risk | Structure is family not avenue | Effect is from signal quantity not residual content |

---

## If You Want A Minimal New Training Experiment

Only worth doing if both A and B succeed at inference time.

Train a lightweight **residual prediction head** on top of a single avenue's representation — a small MLP that predicts what another avenue's residual would be, without running that avenue. Then inject the predicted residual and measure whether accuracy holds.

If it does, you've shown the model can learn to internally simulate another avenue's unique contribution — which is the seed of the coordination idea without requiring a second clone to be running at all.codex resume 019e6327-adc3-7333-9a7c-6458d04a89a1