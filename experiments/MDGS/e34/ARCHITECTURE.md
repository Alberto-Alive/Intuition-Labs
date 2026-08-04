# UGLY Architecture Spec

UGLY means **Uncertainty-Guided Latent Yielding**.

It has two systems trained through the same loss path.

## System A: prediction

System A learns:

```text
context -> answer
```

For this toy project:

```text
2D point -> class label
```

In a language model version:

```text
token context -> next token
```

## System B: reliability geometry

System B learns:

```text
latent state -> evidence quality / distance-to-truth context
```

It computes four primitives:

```text
M = margin / direction
D = disagreement between views
G = geometric spread / instability
S = support / knownness
```

These are not provided as labels. They are computed from the model's own latent state and internal views.

## Why primitive tokens?

Each primitive becomes a vector token:

```text
pM, pD, pG, pS
```

These tokens attend to each other:

```text
[pM, pD, pG, pS] -> primitive attention -> uncertainty context u
```

The uncertainty context is fused back into prediction:

```text
h_final = h + gate(h, u) * adapter(u)
```

Then prediction happens from `h_final`.

This makes System B part of the forward computation. Therefore normal prediction loss can train System B naturally through derivatives/backpropagation.

## Why gated fusion?

The model should not be forced to use uncertainty if it is not useful.

The gate starts small, and the trainer warms up its influence:

```text
epoch 1: small influence
epoch N: full influence if learned useful structure
```

## JEPA-style latent grounding

This toy implementation includes an optional lightweight JEPA-style consistency loss:

```text
encoder(x + noise1) -> z_context
EMA_target_encoder(x + noise2) -> z_target
predictor(z_context) ~= stopgrad(z_target)
```

This is not full I-JEPA; it is a small latent consistency objective for this 2D prototype. Its job is to make the latent space stable enough for System B to inspect.

## What to test

The architecture is useful only if UGLY does better than the baseline on reliability/geometry metrics, not merely raw accuracy.

Check:

```text
accuracy
ECE
confident_wrong_90
risk_coverage_auc
OOD AUROC by uncertainty
OOD AUROC by negative support
surfaces for uncertainty/support/commitment
```

## Expected behavior on two moons

```text
inside moons: high support, low uncertainty
boundary: high disagreement/uncertainty
far OOD: low support, high uncertainty
safe regions: high commitment
```
