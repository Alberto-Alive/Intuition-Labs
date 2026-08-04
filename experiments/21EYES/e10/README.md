# E10: Activation-Space Crossing for Cooperative Multi-Avenue Attention

This scout tests whether private avenue streams become useful when their
activations periodically enter a shared feature subspace, mix across avenues,
and return to their private trajectories.

E10 is not an E9 junction-token replay. Crossing modules operate on existing
per-token avenue activations shaped `[B, T, A, D]`; they do not create learned
junction tokens, a separate memory object, or a token bottleneck.

## Run

```bash
conda activate tnt
cd experiments/21EYES/e10
bash run_scout.sh
```

The scout writes per-run metrics below `results/<variant>/<seed>/`, plots below
`plots/`, and the final report at `results/report.md`.

## Variants

- `baseline_transformer`: normal decoder-only transformer.
- `param_matched_baseline`: baseline with residual adapters sized near the
  four-stream crossing variants.
- `single_avenue_reference`: one E8-style avenue.
- `e8_best_reference`: mirrored four-avenue E8 reference.
- `shared_subspace_crossing`: learned shared basis plus learned avenue mixing.
- `gated_subspace_crossing`: token-conditional source gates in shared space.
- `superposition_crossing`: all avenues write one superposed shared channel.
- `rotation_crossing`: orthogonally initialized crossing projections with a
  light optional orthogonality penalty.
- `pairwise_crossing`: staged pair and pair-summary shared-space mixing.
- `crossing_every_2_layers` and `crossing_every_layer`: frequency controls for
  the simple shared operator.
- `scout_best_custom`: follow-up crossing variant chosen from initial scout
  diagnostics.

## Controls

Crossing variants expose `crossing_zero`, `crossing_shuffle`,
`crossing_random`, `crossing_identity`, avenue controls, single-avenue
evaluation, leave-one-avenue-out evaluation, and hidden-state shuffling before
crossing. The synthetic generator returns evidence metadata for evaluation only;
training consumes causal token sequences and answer labels only.
