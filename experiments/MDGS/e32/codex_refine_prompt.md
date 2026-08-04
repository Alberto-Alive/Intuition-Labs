You are helping refine a PyTorch implementation of a disturbance-profile uncertainty architecture.

Goal:
Adapt the provided scaffold to my existing repository so I can train it cleanly.

Core idea:
- The base model produces a latent anchor z and clean logits.
- A bank of disturbance probes creates perturbed latent witnesses.
- Each probe tests a different uncertainty property:
  - Gaussian noise: local robustness.
  - Feature masking: feature reliance / brittleness.
  - Boundary probe: decision-boundary fragility.
  - Optional learned probe: trainable uncertainty-revealing directions.
- A profile measurer turns disturbed predictions into an uncertainty profile.
- A profile aggregator predicts risk / abstention.
- Class direction should remain controlled by clean logits; disturbance profile should mostly control commitment/abstention.

Please do the following:

1. Inspect my repo and identify the existing model entrypoint, training loop, dataset class, and config system.
2. Adapt `LatentClassifier` to wrap my existing model without changing its core prediction behavior.
3. Find the best internal representation to use as `z`. Prefer the latent right before the classifier head.
4. Integrate `DisturbanceProfileModel` into the training loop.
5. Add config flags:
   - enable_disturbance_profile
   - gaussian_probe_k
   - gaussian_probe_sigma
   - feature_mask_probe_k
   - feature_mask_prob
   - boundary_probe_k
   - boundary_probe_epsilon
   - include_learned_probe
   - detach_profile_from_base
   - risk_loss_weight
   - commit_risk_threshold
   - min_clean_margin
6. Add logging for:
   - task loss
   - risk loss
   - total loss
   - accuracy
   - commit rate
   - committed accuracy
   - risk probability mean
   - each profile feature mean
   - per-class commit and recall if labels are multiclass
7. Make a staged training recipe:
   - base-only training
   - frozen-base profile-head training
   - optional end-to-end fine-tuning
8. Add tests or smoke checks for:
   - forward pass shape correctness
   - no NaNs in profile
   - profile feature names match profile dimension
   - learned probe regularizers work
   - model can run with learned probe disabled
9. Keep the implementation simple and debuggable. Do not replace my classifier direction with the profile head. The profile head should predict risk/abstention, not class.
10. Produce a short README section explaining how to run the new training mode.

Important constraints:
- Preserve existing model behavior when `enable_disturbance_profile=false`.
- Avoid higher-order gradients by default in the boundary probe.
- Make learned probes optional and off by default.
- Keep all disturbance magnitudes configurable.
- Make it easy to run ablations where each probe is turned on/off.
