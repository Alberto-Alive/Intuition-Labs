# Stage 11C Best Variant Retest

## Hypothesis Tested

Direct architecture-faithful variants of the subtask-query cross-agent block may rescue the negative Stage 11A result if failure came from layer placement, query source, summary source, or role count.

## Tried Before This Batch

- Single Actor baseline with one Actor stream and no cross-agent path.
- Role Clones, No Latent Cross-Agent Path with Planner/Reader/Critic/Actor streams.
- Capacity Control, Self Read with the cross block restricted to self-state reads.
- Base Proposed Cross-Agent Latent Attention with middle-plus-late insertion, last-token summaries, and subtask-only Q.
- Shared-Query Control replacing role/subtask-specific Q with a learned shared query.
- Eval-time ablations on the base proposed model: remove path, role K/V shuffle, cross-task K/V shuffle, Actor peer-read mask.

## Retested In Stage 11C

- `single_actor`: roles=['actor'], cross_mode=`none`, query=`subtask`, summary=`last_token`, layers=default.
- `role_clones_no_path`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`none`, query=`subtask`, summary=`last_token`, layers=default.
- `capacity_control_self_read`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`self_only`, query=`subtask`, summary=`last_token`, layers=default.
- `cross_agent_latent_attention`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`cross_peer`, query=`subtask`, summary=`last_token`, layers=default.
- `cross_causal_mean_summary`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`cross_peer`, query=`subtask`, summary=`causal_mean`, layers=default.

## Still Not Tried

- A pretrained causal decoder checkpoint.
- Full token-level cross-role attention instead of role summaries.
- Learned or external subtask decomposition beyond fixed role/subtask tokens.
- Learned residual gate schedules or initialized near-zero gates.
- Retrained mechanism controls for every variant.
- More role-count schedules beyond four roles and Actor/Critic two-role.
- Larger model/data/epoch scaling and difficulty-sliced validation.
- Held-out final evaluation.

## Task And Budget

Same executable gridworld harness as Stage 11A. Train worlds `192`, dev worlds `64`, epochs `16`, batch size `96`, seeds `[101, 102, 103]`.
Hardware `cuda`, CUDA `True`, GPU `NVIDIA GeForce RTX 5070 Ti`.

## Results

| method | seeds | params | action acc mean | rollout success mean | valid action mean | entropy | inject norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| `capacity_control_self_read` | 3 | 223940 | 0.6387 +/- 0.0294 | 0.3385 +/- 0.0849 | 0.4677 | 0.0000 | 0.1523 |
| `cross_agent_latent_attention` | 3 | 223940 | 0.6328 +/- 0.0175 | 0.3594 +/- 0.0675 | 0.4838 | 1.0658 | 0.1555 |
| `cross_causal_mean_summary` | 3 | 223940 | 0.6398 +/- 0.0327 | 0.3698 +/- 0.0820 | 0.4680 | 1.0740 | 0.4313 |
| `role_clones_no_path` | 3 | 173636 | 0.6333 +/- 0.0143 | 0.3490 +/- 0.0531 | 0.4419 | 0.0000 | 0.0000 |
| `single_actor` | 3 | 173636 | 0.6328 +/- 0.0137 | 0.3490 +/- 0.0531 | 0.4419 | 0.0000 | 0.0000 |

## Mechanism Ablations On Base Proposed

| ablation | rollout success mean | action acc mean |
|---|---:|---:|
| `cross_task_kv_shuffle` | 0.3490 +/- 0.0737 | 0.6402 +/- 0.0161 |
| `mask_actor_peer_reads` | 0.3490 +/- 0.0629 | 0.6318 +/- 0.0169 |
| `remove_path` | 0.3594 +/- 0.0556 | 0.6343 +/- 0.0163 |
| `role_kv_shuffle` | 0.3594 +/- 0.0675 | 0.6328 +/- 0.0173 |

## Eval-Time Ablations By Cross Method

| method | ablation | rollout success mean | delta vs method |
|---|---|---:|---:|
| `cross_agent_latent_attention` | `cross_task_kv_shuffle` | 0.3490 | -0.0104 |
| `cross_agent_latent_attention` | `mask_actor_peer_reads` | 0.3490 | -0.0104 |
| `cross_agent_latent_attention` | `remove_path` | 0.3594 | +0.0000 |
| `cross_agent_latent_attention` | `role_kv_shuffle` | 0.3594 | +0.0000 |
| `cross_causal_mean_summary` | `cross_task_kv_shuffle` | 0.3333 | -0.0365 |
| `cross_causal_mean_summary` | `mask_actor_peer_reads` | 0.3854 | +0.0156 |
| `cross_causal_mean_summary` | `remove_path` | 0.3542 | -0.0156 |
| `cross_causal_mean_summary` | `role_kv_shuffle` | 0.3854 | +0.0156 |

## Conclusions

- Best method in this screen: `cross_causal_mean_summary` at `0.3698` mean rollout success.
- Base proposed method: `0.3594` mean rollout success.
- Single/no-path control ceiling in this screen: `0.3490` mean rollout success.
- `cross_causal_mean_summary`: rollout `0.3698`, delta vs base `+0.0104`, delta vs best single/no-path `+0.0208`.
- A variant beat the single/no-path controls on this development screen. It still requires retrained mechanism controls and a stronger/pretrained backbone before claim escalation.
- Best-variant eval-time mechanism drops: `[('cross_task_kv_shuffle', -0.03645833333333337)]`.
- Base proposed mechanism ablations still did not materially damage rollout success, so attention patterns remain non-causal evidence.

## Next Decision

Retain only the best Stage 11B variant for a medium dev validation with retrained no-path, shared-query, K/V shuffle, and Actor-mask controls. Do not touch final evaluation yet.
