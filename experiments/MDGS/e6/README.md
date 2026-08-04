# DIGIT Extrapolation E6

Diffusion-based certainty and outcome model for DIGIT extrapolation.

`e6` is no longer the earlier hybrid design where a clean-latent classifier
could bypass diffusion. The current implementation makes diffusion own the
final outcome prediction and uses diffusion-derived certainty statistics only
for confidence.

---

## Core Idea

**Outcome** = the mean vote of multiple noisy latent reconstructions.

The model builds a joint latent from the query encoding and executor features,
adds diffusion noise, denoises repeatedly, decodes each recovered latent into
an outcome distribution, and averages those distributions. That averaged vote
is the final `outcome_logits`.

**Certainty** = stability of that vote under random noise.

If repeated noisy reconstructions keep returning the same outcome family, the
model is certain. If those reconstructions disagree or wander across basins,
certainty drops.

**Monotonicity** = external perturbation discipline.

`e6` no longer learns private latent "worsening directions". Monotonicity is
trained and audited only through the named trace perturbations used elsewhere
in the experiment stack, so the model is trained and evaluated on the same
notion of worsening.

---

## Loyalty Constraint

The critical design rule:

> `outcome_logits` must come only from diffusion-path aggregation, and
> `confidence_logits` must come only from diffusion-derived certainty stats.

This is enforced by:
1. There is no `direct_outcome_head`.
2. `DiffusionTrustHead` consumes only 5 certainty statistics:
   latent recovery variance, answer agreement, answer entropy,
   denoise energy, and basin stability.
3. The denoiser does not receive the clean latent as an input.

The encoder still gets gradient through the denoising objective and through the
diffusion-owned outcome vote, but not through a separate clean-latent outcome
classifier.

---

## Architecture

```text
Input x
  └─ AdultQueryEncoder ────────────────────────────── z_q [B, 256]
  └─ EntropyTrajectoryExecutor (frozen) ──────────── executor_features [B, 25]

DiffusionBottleneckHead
  ├─ shared_joint MLP (z_q + executor_features) ──── free heads
  │    ├─ trajectory_shape_logits [B, 4]
  │    └─ attention_pattern_logits [B, 3]
  │
  ├─ joint_latent_proj(z_q + executor_features) ──── z0_joint [B, 256]
  │
  ├─ Random diffusion paths (N=6)
  │    z_t = sqrt(ᾱ_t)*z0_joint + sqrt(1-ᾱ_t)*eps
  │    z0_hat = LatentDenoiser(z_t, t)
  │    p_k = path_outcome_decoder(z0_hat)
  │
  ├─ Diffusion ensemble vote
  │    p_mean = mean_k p_k
  │    outcome_logits = log(p_mean)
  │
  └─ DiffusionTrustHead (certainty stats only)
       └─ confidence_logits [B, 3]

IntuitionDecoder ──────────────────────────────────── text generation
```

---

## Changed Files vs E5

| File | Change |
|------|--------|
| `extrapolation/models/bottleneck.py` | Full rewrite: diffusion owns outcome + certainty |
| `extrapolation/models/digit.py` | Wires the diffusion-owned bottleneck without clean outcome shortcuts |
| `extrapolation/losses.py` | Keeps primitive outcome CE on diffusion-owned `outcome_logits`; removes shortcut bootstrap loss |
| `extrapolation/config.py` | Removes stale shortcut/worsening config knobs |
| `scripts/test_smoke.py` | Asserts shortcut absence and diffusion-owned certificates |
| `scripts/test_monotonicity.py` | Tests the confidence head on diffusion certainty stats only |

---

## Key Hyperparameters

| Param | Default | Meaning |
|-------|---------|---------|
| `num_diffusion_steps` | 8 | Forward/reverse steps per path |
| `num_random_paths` | 6 | Paths used for outcome voting and certainty |
| `diffusion_beta_start` | 0.02 | Linear noise schedule start |
| `diffusion_beta_end` | 0.20 | Linear noise schedule end |
| `diffusion_denoiser_hidden` | 256 | LatentDenoiser hidden dim |
| `diffusion_trust_hidden` | 128 | DiffusionTrustHead hidden dim |
| `lambda_diffusion_denoise` | 1.0 | Weight of denoising MSE loss |

---

## Status

Architecture refactored on April 3, 2026 to remove the clean-latent outcome
shortcut and make diffusion own the final outcome vote.

The code now reflects that contract, but the old saved `dev_run*` artifacts
predate this refactor and should not be treated as results for the new design.
Re-run training before evaluating thresholds or comparing against `e5`.
