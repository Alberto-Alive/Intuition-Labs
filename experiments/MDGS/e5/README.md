# DIGIT Extrapolation E5

Directional-monotonicity successor concept to `e4`.

`e4` turned uncertainty into an architectural state. `e5` is the next step:
make monotonicity architectural too.

## Status

This folder is a runnable fork of the current `e4` baseline so `e5` work can
start from a working training/evaluation loop instead of another blank design
brief.

Right now that means:

- the code is runnable
- the runner surface works under `e5/`
- the underlying model still behaves like the copied `e4` baseline until `e5`
  geometry is implemented

In other words: this is the landing zone for the next architecture, not the
finished architecture itself.

## Core Idea

Uncertainty is a state property. Monotonicity is a relation between nearby
states.

So `e5` should not just add more penalty terms. It should represent the geometry
of *worsening moves* explicitly.

The current best candidate is:

- keep the `e4` anchor state space
- attach a local worsening cone to each state
- treat worsening severity as directional, not symmetric
- distinguish ambiguity-like worsening from failure-like worsening

In plain terms:

- `e4` asks: "where am I in trusted reasoning space?"
- `e5` should ask: "if I move in a worse direction from here, what am I allowed
  to become more confident about?"

## Proposed E5 Geometry

The intended geometry is a local cone field over the anchor manifold.

Each state should expose:

- a trust chart of ordered deficits
- a worsening cone: directions that count as epistemically worse
- a severity measure inside that cone
- sector structure that distinguishes:
  - ambiguity / abstention directions
  - failure directions

The main trust coordinates will likely include:

- support deficit
- approach deficit
- leap cost
- conflict
- agreement deficit
- margin deficit
- attention-concentration deficit
- variation-ratio increase

The key design principle is:

- worsening toward generic damage should raise `UNCERTAIN`
- worsening toward a clean failure basin should raise `FAILURE_LIKELY`
- worsening should not increase `SUCCESS_LIKELY`

## Why This Is Different From E4

`e4` gives geometry of states.

`e5` should give geometry of *downward moves through* those states.

That means:

- `e4` models anchor basins and commitment
- `e5` models which local moves are epistemically downhill

This is the difference between:

- a certainty object
- an order object

## Minimal First Implementation

The smallest credible `e5` should probably:

1. keep the current `e4` anchor bottleneck as the state geometry
2. add a trust-chart head that exposes ordered deficit coordinates
3. derive worsening basis directions from actual perturbations such as
   `lower_agreement`, `lower_margin`, `higher_entropy`,
   `lower_attention_concentration`, and `higher_variation_ratio`
4. project local moves into ambiguity-vs-failure sectors
5. make outcome and commitment monotone along worsening-cone radius instead of
   relying only on regularization losses

## Current Runner Surface

Single run:

```bash
python3 experiments/DIGIT/Extrapolation/e5/scripts/run_experiment.py \
  --output-dir experiments/DIGIT/Extrapolation/results/e5/dev_run
```

Six-way inherited stage-1 sweep:

```bash
python3 experiments/DIGIT/Extrapolation/e5/scripts/run_anchor_sweep.py \
  --max-parallel auto \
  --output-root experiments/DIGIT/Extrapolation/results/e5/anchor_sweep
```

The current sweep is inherited from `e4` and should be treated as bootstrap
infrastructure, not as the final `e5` experiment grid.

## Non-Goals

`e5` should not just be:

- more monotonicity penalties on top of `e4`
- threshold tuning without new geometry
- another symmetric distance score
- a second uncertainty tower disguised as "order"

If monotonicity still lives only in audits and losses, then `e5` has not become
architectural enough.
