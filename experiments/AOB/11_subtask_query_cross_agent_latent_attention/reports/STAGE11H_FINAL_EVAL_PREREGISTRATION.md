# Stage 11H Final Evaluation Preregistration

## Status

This file is written before the Stage 11H final evaluation script is run. The final evaluation must be run once, exactly as specified here, and reported as-is regardless of outcome.

This preregistration file must be committed before running the final evaluation.

## Fixed Method Under Evaluation

Primary method:

- `private_view_shared_bus`

Architecture is fixed from the Stage 11H seed-stability run:

- shared cross-role residual bus
- fixed non-overlapping role write subspaces
- default bus insertion layers from `RunConfig.cross_layers`, currently `(1, 2)`
- `private_view=True`
- `history_private_view=True`
- `summary_mode="causal_mean"`
- `label_mode="history_tiebreak"`

No architecture changes, insertion-schedule changes, label-mode changes, hyperparameter changes, or task changes are allowed before this final evaluation.

## Baselines

Run these methods under the same training and evaluation budget:

- `private_view_history_no_path`
- full-view `single_actor`
- full-view `role_clones_no_path`

## Ablations

Run these eval-time ablations on `private_view_shared_bus` only:

- `remove_path`
- `cross_task_bus_shuffle`
- `role_write_zero_planner`
- `ablation_compound_planner_zero_actor_mask`

Report ablation rollout success and delta versus unablated `private_view_shared_bus`.

## Seeds And Budget

Final evaluation seeds:

- `201`
- `202`
- `203`
- `204`
- `205`

Training budget per method/seed:

- epochs: `16`
- train worlds: `192`
- final held-out worlds: `64`
- batch size: `96`
- max steps: current fixed `RunConfig.max_steps`
- model size and optimizer settings: current fixed `RunConfig`

## Held-Out Episode Set

The held-out episode set is the final evaluation world set generated inside `train_and_evaluate` for each final seed using the existing deterministic world-generation protocol:

- split name: `dev`
- world count: `64`
- seed rule: `seed * 1000 + 29`

These final seeds (`201`-`205`) were not used in Stage 11H probe, retest, compound ablation, every-layer comparison, or seed-stability reporting. No result from these final held-out episode sets may be inspected before this preregistration is committed.

## Primary Metric

Primary metric:

- rollout success mean across the 5 final seeds

The final report must include raw per-seed rollout success, mean, and std for each method.

## Secondary Metrics

Secondary metrics:

- supervised action accuracy
- valid action rate
- mean actions
- bus write norm ratio
- bus read norm ratio
- ablation rollout deltas for `private_view_shared_bus`

Secondary metrics are diagnostic and must not replace the primary metric.

## Primary Claim

The primary claim is supported only if `private_view_shared_bus` outperforms both:

- `private_view_history_no_path`
- full-view `single_actor`

on mean rollout success across the 5 final seeds.

Full-view `role_clones_no_path` is included as an additional baseline and must be reported, but the stated primary performance claim is defined against `private_view_history_no_path` and full-view `single_actor`.

## Secondary Mechanism Claim

The secondary mechanism claim is supported only if these ablations on `private_view_shared_bus` are damaging by more than `0.10` absolute rollout success:

- `remove_path`
- `role_write_zero_planner`

`cross_task_bus_shuffle` and `ablation_compound_planner_zero_actor_mask` must also be reported, but the preregistered thresholded mechanism claim is specifically for `remove_path` and `role_write_zero_planner`.

## Honest Scope

The expected performance margin over full-view `single_actor` is modest, approximately `0.04` to `0.05`, based on the 8-seed Stage 11H seed-stability run. The mechanism claim is stronger than the performance-margin claim.

The defensible scope is a Stage 11H final evaluation of a constructed private-role gridworld task with a structurally load-bearing shared bus. This does not by itself establish a broad claim about spontaneous multi-agent specialization or pretrained LLM agents.

## Reporting Rule

Run the final evaluation once after this file is committed. Do not iterate, retune, reseed, change methods, add or remove ablations, or rerun after seeing the result.

Write all results as-is to `reports/STAGE11H_FINAL_EVAL.md`.
