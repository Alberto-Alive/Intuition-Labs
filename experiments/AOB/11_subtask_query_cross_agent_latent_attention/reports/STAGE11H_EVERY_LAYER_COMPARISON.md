# Stage 11H Every-Layer Shared Bus Comparison

## Source Runs

- Results source: `results/stage11h_shared_bus_retest3/results.json`.
- Report source: `reports/STAGE11H_SHARED_BUS_RETEST3.md`.
- Budget: 3 seeds (`121`, `122`, `123`), 16 epochs, 192 train worlds, 64 dev worlds, batch size 96.
- Label/task mode: `history_tiebreak` with private goal/map/history masking.

## Methods Compared

- `private_view_shared_bus`: shared bus inserted at the default configured layers `(1, 2)`.
- `private_view_shared_bus_every`: shared bus inserted after every decoder layer `(0, 1, 2)`.

## Primary Results

| method | rollout success mean | rollout raw | action acc mean | bus write norm | bus read norm |
|---|---:|---|---:|---:|---:|
| `private_view_shared_bus` | 0.3646 +/- 0.0483 | [0.3906, 0.2969, 0.4062] | 0.6510 | 0.9691 | 0.7403 |
| `private_view_shared_bus_every` | 0.2812 +/- 0.0128 | [0.2656, 0.2812, 0.2969] | 0.6473 | 0.8299 | 0.6825 |

Default layer insertion beat every-layer insertion by `+0.0833` mean rollout success. Supervised action accuracy was similar, so the difference is mainly in rollout behavior rather than next-action classification.

## Mechanism Ablations

| ablation | default bus rollout | default delta | every-layer rollout | every-layer delta |
|---|---:|---:|---:|---:|
| `remove_path` | 0.0469 | -0.3177 | 0.0469 | -0.2344 |
| `cross_task_bus_shuffle` | 0.0469 | -0.3177 | 0.0260 | -0.2552 |
| `role_write_zero_planner` | 0.0573 | -0.3073 | 0.0781 | -0.2031 |
| `mask_actor_peer_reads` | 0.3438 | -0.0208 | 0.2604 | -0.0208 |
| `mask_action_bus` | 0.2865 | -0.0781 | 0.3281 | +0.0469 |

Both variants have damaging `remove_path`, `cross_task_bus_shuffle`, and `role_write_zero_planner` ablations. The default bus has larger absolute damage because its unablated rollout is higher. The every-layer variant is therefore not a failure of the mechanism, but it is not the best Stage 11H insertion schedule.

## Interpretation

- The shared-bus result is robust to inserting the bus every layer in the sense that mechanism ablations still damage rollout.
- Default middle-plus-late insertion is better empirically on the 3-seed retest.
- The every-layer `mask_action_bus` ablation improves rollout, suggesting early repeated bus reads can make the final bus/head path less clean or over-coupled.

## Decision

Keep `private_view_shared_bus` with default layers `(1, 2)` as the Stage 11H lead. Do not change the insertion schedule after the seed-stability step unless a new preregistered branch is opened separately.
