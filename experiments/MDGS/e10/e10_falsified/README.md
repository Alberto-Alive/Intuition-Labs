# DIGIT Extrapolation E10

`e10` is the cooperative diffusion-evidence redesign.

It keeps diffusion as the source of the internal state dynamics, but stops using
separate post-hoc heads for outcome, certainty, fragility, and guard. Instead,
each reconstructed diffusion path emits a shared evidence state:

- `order`: success vs failure direction
- `boundary`: ambiguity / boundary mass
- `support`: admissibility / on-manifold support

All public quantities are then read from that same aggregated path evidence.

## Core Idea

`e9` improved commitment monotonicity, but it still split the model into:

- diffusion path reconstructions
- a separate certainty readout
- a separate fragility readout
- a separate guard formula
- a separate outcome decoder

That broke the cooperative behavior that existed in `e3`, where worsening one
part of the evidence naturally hurt the others.

`e10` restores that cooperation in a diffusion-native way:

1. Diffusion reconstructs multiple path states.
2. Each path state is decoded into shared evidence.
3. Path evidence is aggregated into certificates and public decisions.
4. Perturbation losses train worsening inputs to move downhill in that same
   ordered evidence space.

## Architecture

For each reconstructed path latent `z_hat_i`, `e10` decodes:

- `order_i`
- `boundary_i`
- `support_i`

From those, each path produces class evidence:

```text
success_i   = support_i * (1 - boundary_i) * success_side(order_i)
failure_i   = support_i * (1 - boundary_i) * failure_side(order_i)
uncertain_i = 1 - support_i * (1 - boundary_i)
```

Across paths, the model aggregates:

- mean path outcome evidence
- path disagreement
- order / boundary / support means and spreads
- denoising energy
- prototype-family geometry over reconstructed path states

Public cooperative quantities are then defined from the same shared state:

- `stability_cert`
- `support_cert`
- `success_guard`
- `success_cert`
- `failure_cert`
- `ambiguity_cert`
- `confidence_logits`
- `outcome_logits`
- `commitment_depth`

`commitment_depth` is intentionally not just certainty. It represents
admissible nonfailure commitment:

```text
success_cert = success_base * success_guard
commitment_depth = success_guard * support_cert * (1 - ambiguity_cert)
```

`success_base` reaches the outcome and confidence heads directly through the
shared evidence vector. `success_guard` remains a separate admissibility
certificate instead of participating in a multi-stage product chain, which
keeps gradients alive while preserving the cooperative design.

## What Stayed Diffusion-Owned

Allowed:

- final decisions depend on reconstructed diffusion paths
- prototype families regularise the diffusion evidence manifold
- monotonicity losses shape where perturbed paths land

Not allowed:

- clean-latent outcome shortcuts
- anchor/prototype families directly replacing the diffusion decision path
- separate post-hoc certainty or fragility heads bypassing the shared evidence

## Training Additions

`e10` keeps denoising and prototype regularisation, and adds a stronger
cooperative geometry contract:

- `family_supervision`
  Prototype families still regularise success / failure / boundary geometry.
- `basin_order`
  Same-`t`, same-noise base vs perturbed pairs must move downhill in ordered
  evidence space.
- `ordered_geometry`
  Worst-case optimistic perturbed paths are penalised directly.
- `perturbation_rank`
  Perturbed latents should be harder to denoise when they become more optimistic.
- `perturbation_proximity`
  Base and perturbed latents stay nearby so the encoder cannot win by jumping
  to unrelated regions.
- `fragility_target` / `fragility_rank`
  Fragility is supervised as a derived risk over support / overlap / optimism
  failures rather than as a separate standalone subsystem.

## Intended Property

The target behavior is:

- deep success basin:
  high certainty, high support, high success
- boundary region:
  higher ambiguity, lower certainty, lower commitment
- deep failure basin:
  high certainty of failure, low commitment

Under worsening perturbations, the reconstructed path cloud should move
downhill:

- lower order
- lower support
- higher denoising difficulty
- lower guard
- lower commitment

That is the mechanism `e10` is trying to learn natively, rather than patching
monotonicity after the fact.
