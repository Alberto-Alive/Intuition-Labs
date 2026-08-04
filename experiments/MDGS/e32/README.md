# E32 Disturbance-Profile Fork

This folder is a runnable fork of `e31` with an extra disturbance-profile head attached after the latent `z` anchor.

## What stays the same

- Dataset loading, trace cache handling, and class labels come from `e31`.
- The witness / diffusion / readout path still produces the class logits.
- Existing command-line flags from `e31` still work.

## What is new

- Disturbance probes over the latent anchor `z`.
- A profile head that turns disturbance responses into risk / abstention signals.
- Staged training:
  1. base model training
  2. frozen-base profile/risk training
  3. end-to-end fine-tuning
- Learned probes stay off until the final stage.

## Main training command

Run the fork with the same style of command as `e31`, plus optional disturbance flags:

```bash
python scripts/run_experiment.py \
  --seed 123 \
  --enable-disturbance-profile \
  --enable-boundary-probe \
  --enable-feature-mask-probe \
  --profile-loss-weight 0.05 \
  --risk-loss-weight 1.0
```

Useful optional flags:

- `--detach-profile-from-base`
- `--attach-profile-to-base`
- `--disturbance-num-samples`
- `--disturbance-noise-std`
- `--enable-learned-probe`
- `--disturbance-feature-mask-prob`
- `--disturbance-boundary-epsilon`
- `--risk-target-mode`
- `--commit-risk-threshold`
- `--min-clean-margin`

## Smoke test

```bash
pytest tests/test_e32_smoke.py
```

## Standalone profile demo

`train_disturbance_profile.py` remains a small CSV/tabular example for the generic profile scaffold.
