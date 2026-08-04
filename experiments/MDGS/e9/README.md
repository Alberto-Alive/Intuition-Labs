# DIGIT Extrapolation E9

`e9` is the diffusion-basin follow-up to `e8`.

The diffusion contract remains strict:

- final `outcome_logits` come only from diffusion path voting
- certainty is still derived from diffusion-path behaviour
- prototype geometry only modulates fragility / support, never the final outcome path

## Why E9 Exists

`e8` repaired collapse and calibration, but it still exposed one core limitation:

- diffusion certainty mostly measured basin stability, not monotone trust
- prototype geometry mostly read the diffusion basin after the fact
- worsening perturbations could still harden optimistic boundary examples

`e9` keeps the same diffusion-owned outcome path and uses prototype families to
shape reconstructed diffusion basins during training.

## E9 Architecture

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

`e9` keeps the `e8` stack and adds path-level basin shaping:

- `prototype_pull`
  Reconstructed path states should stay close to the winning prototype within
  each family.
- `prototype_usage`
  Prototype usage should not collapse to a single anchor.
- `prototype_repulsion`
  Distinct prototype anchors should maintain margin in state space.
- `family_supervision`
  Outcome labels supervise prototype-family assignment on both the aggregate
  geometry and each reconstructed diffusion path.
- `path_family_margin`
  Individual reconstructed paths are pushed to prefer the target family by a
  margin, not only on average across the batch.
- `basin_order`
  Under named worsening perturbations, same-`t` / same-`eps` diffusion pairs
  should move downhill in ordered prototype space instead of sharpening toward
  optimistic basins.
- `ordered_geometry`
  Worst-case reconstructed diffusion paths are explicitly ranked so worsening
  must not increase ordered success-vs-failure score, must not increase
  success-family support, and must not become easier to denoise for
  nonfailure examples.
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

## E9 Corrections

Compared with `e8`, this cut directly targets the remaining monotonicity gap:

- prototype families now supervise reconstructed diffusion paths directly
- worsening perturbations now incur a basin-order loss in prototype space
- diffusion remains the final decision path; prototype families shape geometry
  but do not replace the outcome vote

## Important Constraint

`e9` still does **not** restore the old `e4/e5` anchor decision stack.

Not allowed:

- anchor-driven final outcome logits
- clean-latent outcome shortcuts
- hand-written `worsening_*` formulas in the final decision path

Allowed:

- prototype geometry derived from diffusion reconstructions
- fragility estimation from prototype + certainty statistics
- denoiser ranking losses against named perturbations

## Status

This directory is a runnable scaffold for the `e9` idea. It is intended for
fresh training runs; old `e8` artifacts are not directly comparable once these
basin-shaping semantics change.
