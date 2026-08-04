# DIGIT Extrapolation E4

Epistemic-anchor successor concept to `e3`.

`e4` is the point where the project stops treating uncertainty as a score that can
be patched onto a prediction head and instead treats certainty as something the
model must earn on the forward path.

## Status

This folder now contains a runnable first implementation:

- `extrapolation/models/bottleneck.py` implements the anchor-based epistemic bottleneck
- `extrapolation/losses.py` supervises anchor proxies and anchor geometry
- `scripts/run_experiment.py` runs staged training with anchor audits and commitment audits
- `scripts/run_anchor_sweep.py` launches the six stage-1 ablations with adaptive concurrency

The working thesis is:

- `e2` made uncertainty more explicit
- `e3` tried to make uncertainty more monotone
- `e4` should make certainty architectural

## Core Idea

The current direction is an anchor-based epistemic bottleneck:

- anchors represent trusted support regions in reasoning space
- anchors have geometry, not just centers
- certainty depends on support, approach quality, leap size, and anchor conflict
- the model is only allowed deeper commitment when those signals justify it

In plain terms: a state should not become highly certain just because it looks
familiar. It should also have arrived there in a supported way.

## Proposed E4 Signals

- `anchor_support`: how near the current state is to trusted anchor regions
- `approach_quality`: whether the state arrived from a familiar direction
- `leap_size`: whether the jump from the previous reasoning state was too large
- `anchor_conflict`: whether multiple incompatible anchors support the state at once
- `commitment_depth`: how far the model is allowed to commit given the above

## Prior Art Vs E4

| Prior line | What already exists | What it does not fully solve | Proposed `e4` move |
| --- | --- | --- | --- |
| Bayesian weight uncertainty | [Bayes by Backprop](https://proceedings.mlr.press/v37/blundell15.html) learns distributions over weights | Global parameter uncertainty is not the same as per-input reasoning certainty | Move certainty to input-conditioned epistemic states instead of weight posteriors alone |
| Training-support in representation space | [Deep k-Nearest Neighbors](https://arxiv.org/abs/1803.04765) and Mahalanobis-style OOD work score support using hidden representations | Mostly scores point similarity to training support, not whether the reasoning path into that region was sound | Keep support distance, but add path-sensitive approach and leap quality |
| Activation surprise | [Surprise Adequacy](https://arxiv.org/abs/1808.08444) compares new activations to training activations | Flags novelty, but does not define an architectural certainty mechanism | Use novelty as one epistemic signal inside the bottleneck, not as a standalone alert |
| Prototype / anchor methods | [Prototypical Networks](https://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning) and [P-ODN](https://arxiv.org/abs/1905.01851) use learned prototypes, radii, or open-set regions | Prototypes are usually static support points, not full certainty objects with approach geometry and commitment effects | Upgrade anchors from points to geometric support regions with direction-aware entry behavior |
| Trajectory-aware prototype work | [ProtoryNet](https://openreview.net/forum?id=KwgQn_Aws3_) and [ProtoCAD](https://openreview.net/forum?id=OcHQVmfLn2c) add temporal or trajectory structure to prototypes | They are not built as epistemic bottlenecks for safe commitment and abstention | Reuse the trajectory intuition, but make it govern certainty formation directly |
| Integrated abstention | [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html) puts reject/abstain on the forward path | Reject option is integrated, but not grounded in anchor support, arrival direction, and commitment depth | Make abstention a consequence of weak support or blocked commitment, not a separate afterthought |

## What Might Be New

The ingredients above already exist in parts. The potentially new synthesis is:

- certainty is computed by a dedicated epistemic bottleneck on the forward path
- anchors are geometric support objects, not only points or radii
- arrival direction and leap size matter, not just endpoint distance
- commitment depth is limited by epistemic support before the final outcome head
- novelty is treated as "be more cautious and verify" rather than "always reject"

In one sentence:

`e4` aims to make certainty the permitted depth of commitment of a reasoning state
relative to trusted anchors and supported transitions.

## Minimal First Implementation

The smallest credible `e4` should probably:

1. choose one epistemic state inside the bottleneck
2. learn anchor centers plus shape/scale
3. score support, approach quality, leap size, and conflict
4. replace the final free confidence / outcome heads with a commitment-depth head
5. map commitment depth to `LOW / MEDIUM / HIGH` and `SUCCESS_LIKELY / UNCERTAIN / FAILURE_LIKELY`

## Current Runner Surface

Single run:

```bash
python3 experiments/DIGIT/Extrapolation/e4/scripts/run_experiment.py \
  --output-dir experiments/DIGIT/Extrapolation/results/e4/dev_run
```

Six-way stage-1 sweep with adaptive GPU scheduling:

```bash
python3 experiments/DIGIT/Extrapolation/e4/scripts/run_anchor_sweep.py \
  --max-parallel auto \
  --output-root experiments/DIGIT/Extrapolation/results/e4/anchor_sweep
```

The sweep runner defaults to six named experiments:

- `anchor_base`
- `anchor_no_approach`
- `anchor_no_leap`
- `anchor_no_boundary_family`
- `anchor_hard_assignment`
- `anchor_no_barrier`

## Non-Goals

`e4` should not just be:

- another uncertainty MLP
- post-hoc confidence calibration
- distance-to-training-data used as a blanket rejection rule
- a pile of penalty terms with no architectural role

If the model can ignore the epistemic module and behave the same way, then `e4`
has not become architectural enough.
