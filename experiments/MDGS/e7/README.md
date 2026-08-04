# DIGIT Extrapolation E7

`e7` is the next experiment after the diffusion-only `e6` refactor.

The contract stays strict:

- final `outcome_logits` come only from diffusion path voting
- certainty still comes from diffusion-path behaviour
- prototype geometry is auxiliary context for fragility, not a second outcome path

## Core Idea

`e6` proved that removing the clean-latent shortcut makes the architecture
truthful, but it also exposed a gap: diffusion stability alone is not enough.
A perturbed example can collapse into a sharp basin and look "certain" even
when that basin is just a brittle failure/ambiguity attractor.

`e7` adds a diffusion-native prototype bank to describe where reconstructed
diffusion paths land in semantic space:

- success family
- failure family
- boundary family

Those prototype families are computed from reconstructed diffusion-path states,
not from a separate clean-latent classifier.

## E7 Architecture

1. Build the same joint latent from `z_q + executor_features`.
2. Run the same random diffusion paths as `e6`.
3. Decode the same diffusion-path outcome vote.
4. Project each reconstructed path latent into a prototype state space.
5. Compute family support, family margin, family conflict, and anchor switch
   rate across paths.
6. Feed certainty stats + prototype geometry into a `fragility_head`.
7. Define:

```text
certainty_score   = 1 - cert_risk
fragility_score   = 1 - fragility_risk
decisiveness      = 1 - ambiguity_cert
commitment_depth  = certainty_score * fragility_score * decisiveness
```

The final confidence risk is:

```text
confidence_risk = max(cert_risk, fragility_risk)
```

## Training Additions

`e7` keeps all of the `e6` diffusion losses and adds three scaffold losses:

- `prototype_alignment`
  Prototype family distributions are aligned to detached diffusion outcome
  distributions path by path.
- `perturbation_rank`
  Using the same diffusion timestep and same sampled noise, perturbed latents
  should be harder to denoise than base latents when prototype geometry
  actually degrades.
- `perturbation_proximity`
  Base and perturbed latents are softly kept near each other so the encoder
  cannot satisfy ranking by simply throwing perturbed examples somewhere else.

## Important Constraint

This experiment deliberately does **not** restore the old `e4/e5` anchor
decision stack.

Not allowed in `e7`:

- anchor-driven final outcome logits
- clean-latent outcome shortcuts
- hand-written `worsening_*` formulas driving the final decision path

Allowed in `e7`:

- prototype geometry derived from diffusion reconstructions
- fragility estimation from prototype + certainty statistics
- denoiser ranking losses against named perturbations

## Status

This directory is a scaffold for the `e7` idea. It should be runnable as a
separate experiment variant, but it is not benchmarked yet. Expect tuning and
debugging before treating any `e7` run as comparable to `e6`.
