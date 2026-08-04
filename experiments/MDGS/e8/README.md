# DIGIT Extrapolation E8

`e8` is the corrective follow-up to `e7`.

The diffusion contract remains strict:

- final `outcome_logits` come only from diffusion path voting
- certainty is still derived from diffusion-path behaviour
- prototype geometry only modulates fragility / support, never the final outcome path

## Why E8 Exists

`e7` added prototype geometry, but the first run exposed four concrete issues:

- certainty semantics were inverted in downstream use
- fragility was almost constant and barely influenced confidence
- prototype alignment taught geometry to mimic the outcome vote
- stage-1 selection still ignored the monotonicity objective

`e8` keeps the same diffusion-owned outcome path and repairs those problems.

## E8 Architecture

1. Build the joint latent from `z_q + executor_features`.
2. Run random diffusion paths and decode the mean path vote into `outcome_logits`.
3. Compute diffusion certainty stats from reconstruction variance, agreement,
   entropy, denoise energy, and basin stability.
4. Project reconstructed path states into a prototype bank and compute:
   - top support
   - total support
   - family margin
   - family conflict
   - family switch rate
   - true anchor switch rate
   - nearest-anchor distance
5. Predict `certainty_score` from diffusion stats and `fragility_risk` from
   diffusion stats + prototype geometry.
6. Define:

```text
cert_risk           = 1 - certainty_score
fragility_score     = 1 - fragility_risk
effective_certainty = certainty_score * fragility_score
decisiveness_score  = 1 - ambiguity_cert
commitment_depth    = effective_certainty * decisiveness_score
```

Confidence classes are produced from `effective_certainty`, not from a max-risk
shortcut.

## Training Additions

`e8` keeps the `e6` diffusion losses and adds geometry/fragility supervision:

- `prototype_pull`
  Reconstructed path states should stay close to the winning prototype within
  each family.
- `prototype_usage`
  Prototype usage should not collapse to a single anchor.
- `prototype_repulsion`
  Distinct prototype anchors should maintain margin in state space.
- `perturbation_rank`
  Using the same diffusion timestep and sampled noise, perturbed latents should
  be harder to denoise when worsening creates stronger certainty/commitment
  violations.
- `perturbation_proximity`
  Base and perturbed latents are kept near each other to block trivial encoder
  escape routes.
- `fragility_target` / `fragility_rank`
  Fragility is trained directly against detached perturbation sensitivity rather
  than inferred only through downstream commitment penalties.

## Selection And Diagnostics

Stage-1 retention and final checkpoint selection now factor in
`commitment_monotonicity_violation_rate`, not just outcome/failure quality.

The experiment exports extra diagnostics for:

- `effective_certainty`
- `prototype_total_support`
- `prototype_family_switch_rate`
- `prototype_anchor_switch_rate`
- `prototype_nearest_anchor_distance`
- fragility-target and perturbation-rank losses

## E8 Corrections

Compared with `e7b`, this cut also tightens the concrete failure modes that the
first prototype run exposed:

- prototype usage is now support-weighted so family collapse is penalized
- uncertain examples are pushed to show some anchor/family motion across paths
- the guard no longer collapses to near-zero from multiplicative conflict terms
- stage 1 rejects structurally collapsed checkpoints by default
- stage 2 is skipped by default so primitive evaluation stays tied to the
  bottleneck experiment

## Important Constraint

`e8` still does **not** restore the old `e4/e5` anchor decision stack.

Not allowed:

- anchor-driven final outcome logits
- clean-latent outcome shortcuts
- hand-written `worsening_*` formulas in the final decision path

Allowed:

- prototype geometry derived from diffusion reconstructions
- fragility estimation from prototype + certainty statistics
- denoiser ranking losses against named perturbations

## Status

This directory is a runnable scaffold for the `e8` idea. It is intended for
fresh training runs; old `e7` artifacts are not comparable once these semantics
change.
