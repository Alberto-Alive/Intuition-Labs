# Stage 11H Seed Stability

## Purpose

Test whether the Stage 11H shared-bus signal survives additional random seeds without changing architecture or hyperparameters.

## Fixed Setup

- Methods: `private_view_shared_bus`, `private_view_history_no_path`, `single_actor`.
- Seeds: `121`, `122`, `123`, `124`, `125`, `126`, `127`, `128`.
- Seeds `121`-`123` source: `results/stage11h_shared_bus_retest3/results.json`.
- Seeds `124`-`128` source: `results/stage11h_seed_stability_extra5/results.json`.
- Epochs: `16`.
- Train worlds: `192`.
- Dev worlds: `64`.
- Batch size: `96`.
- Label mode: `history_tiebreak`.
- No architecture changes and no hyperparameter changes were made for this seed-stability run.

## Primary Metric

Rollout success rate on the dev worlds. Std below is population std, matching the experiment harness aggregation. Sample std is included in the raw table for audit.

## Eight-Seed Rollout Results

| method | seeds | rollout raw | mean | population std | sample std |
|---|---:|---|---:|---:|---:|
| `private_view_shared_bus` | 8 | [0.390625, 0.296875, 0.406250, 0.343750, 0.328125, 0.328125, 0.312500, 0.343750] | 0.343750 | 0.034939 | 0.037351 |
| `private_view_history_no_path` | 8 | [0.125000, 0.093750, 0.078125, 0.109375, 0.046875, 0.046875, 0.031250, 0.015625] | 0.068359 | 0.036592 | 0.039118 |
| `single_actor` | 8 | [0.328125, 0.296875, 0.421875, 0.296875, 0.375000, 0.250000, 0.171875, 0.250000] | 0.298828 | 0.073053 | 0.078097 |

## Comparisons

- `private_view_shared_bus` vs `private_view_history_no_path`: `+0.275391` mean rollout success.
- `private_view_shared_bus` vs full-view `single_actor`: `+0.044922` mean rollout success.
- `single_actor` vs `private_view_history_no_path`: `+0.230469` mean rollout success.

## Interpretation

The shared-bus effect is stable across eight seeds relative to the private no-path control. The margin over full-view `single_actor` is positive but much smaller than the margin over the private no-path baseline, so the defensible Stage 11H claim remains strongest for recovering private role information through a structurally load-bearing bus.

The full-view `single_actor` has higher variance than `private_view_shared_bus` in this batch (`0.073053` vs `0.034939` population std). This makes the full-view ceiling comparison encouraging but not final-evaluation-grade by itself.

## Decision

Proceed to pre-registration before any final evaluation. Do not change the Stage 11H architecture, insertion schedule, label mode, training budget, or hyperparameters after this point.
