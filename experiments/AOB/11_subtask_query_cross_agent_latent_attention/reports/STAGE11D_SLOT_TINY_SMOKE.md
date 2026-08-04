# Stage 11D Slot Tiny Smoke

## Hypothesis Tested

Direct architecture-faithful variants of the subtask-query cross-agent block may rescue the negative Stage 11A result if failure came from layer placement, query source, summary source, or role count.

## Tried Before This Batch

- Single Actor baseline with one Actor stream and no cross-agent path.
- Role Clones, No Latent Cross-Agent Path with Planner/Reader/Critic/Actor streams.
- Capacity Control, Self Read with the cross block restricted to self-state reads.
- Base Proposed Cross-Agent Latent Attention with middle-plus-late insertion, last-token summaries, and subtask-only Q.
- Shared-Query Control replacing role/subtask-specific Q with a learned shared query.
- Eval-time ablations on the base proposed model: remove path, role K/V shuffle, cross-task K/V shuffle, Actor peer-read mask.

## Newly Tried In Stage 11B

- `slot_self_read_m2`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`self_only`, query=`subtask`, summary=`causal_mean`, layers=default, slots=2, topk=0, role_dropout=0.0.
- `slot_bottleneck_m2`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`cross_peer`, query=`subtask`, summary=`causal_mean`, layers=default, slots=2, topk=0, role_dropout=0.0.
- `slot_bottleneck_m2_topk2`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`cross_peer`, query=`subtask`, summary=`causal_mean`, layers=default, slots=2, topk=2, role_dropout=0.0.
- `slot_bottleneck_m2_dropout`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`cross_peer`, query=`subtask`, summary=`causal_mean`, layers=default, slots=2, topk=0, role_dropout=0.25.
- `slot_bottleneck_m4_topk2`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`cross_peer`, query=`subtask`, summary=`causal_mean`, layers=default, slots=4, topk=2, role_dropout=0.0.

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

Same executable gridworld harness as Stage 11A. Train worlds `24`, dev worlds `10`, epochs `1`, batch size `24`, seeds `[101]`.
Hardware `cuda`, CUDA `True`, GPU `NVIDIA GeForce RTX 5070 Ti`.

## Results

| method | seeds | params | action acc mean | rollout success mean | valid action mean | entropy | inject norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| `slot_bottleneck_m2` | 1 | 240836 | 0.3200 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.8500 | 1.7917 | 0.1012 |
| `slot_bottleneck_m2_dropout` | 1 | 240836 | 0.3200 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.8500 | 1.7917 | 0.1017 |
| `slot_bottleneck_m2_topk2` | 1 | 240836 | 0.3200 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.8500 | 0.6931 | 0.1014 |
| `slot_bottleneck_m4_topk2` | 1 | 241092 | 0.1900 +/- 0.0000 | 0.1000 +/- 0.0000 | 0.2071 | 0.6931 | 0.0995 |
| `slot_self_read_m2` | 1 | 240836 | 0.3200 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.8500 | 0.6931 | 0.1014 |

## Mechanism Ablations On Base Proposed

No mechanism ablations were logged.

## Eval-Time Ablations By Cross Method

| method | ablation | rollout success mean | delta vs method |
|---|---|---:|---:|
| `slot_bottleneck_m2` | `cross_task_kv_shuffle` | 0.0000 | +0.0000 |
| `slot_bottleneck_m2` | `mask_actor_peer_reads` | 0.0000 | +0.0000 |
| `slot_bottleneck_m2` | `remove_path` | 0.0000 | +0.0000 |
| `slot_bottleneck_m2` | `role_kv_shuffle` | 0.0000 | +0.0000 |
| `slot_bottleneck_m2_dropout` | `cross_task_kv_shuffle` | 0.0000 | +0.0000 |
| `slot_bottleneck_m2_dropout` | `mask_actor_peer_reads` | 0.0000 | +0.0000 |
| `slot_bottleneck_m2_dropout` | `remove_path` | 0.0000 | +0.0000 |
| `slot_bottleneck_m2_dropout` | `role_kv_shuffle` | 0.0000 | +0.0000 |
| `slot_bottleneck_m2_topk2` | `cross_task_kv_shuffle` | 0.0000 | +0.0000 |
| `slot_bottleneck_m2_topk2` | `mask_actor_peer_reads` | 0.0000 | +0.0000 |
| `slot_bottleneck_m2_topk2` | `remove_path` | 0.0000 | +0.0000 |
| `slot_bottleneck_m2_topk2` | `role_kv_shuffle` | 0.0000 | +0.0000 |
| `slot_bottleneck_m4_topk2` | `cross_task_kv_shuffle` | 0.1000 | +0.0000 |
| `slot_bottleneck_m4_topk2` | `mask_actor_peer_reads` | 0.1000 | +0.0000 |
| `slot_bottleneck_m4_topk2` | `remove_path` | 0.1000 | +0.0000 |
| `slot_bottleneck_m4_topk2` | `role_kv_shuffle` | 0.1000 | +0.0000 |

## Conclusions

- Best method in this screen: `slot_bottleneck_m4_topk2` at `0.1000` mean rollout success.
- Base proposed method: `0.0000` mean rollout success.
- Single/no-path control ceiling in this screen: `0.0000` mean rollout success.
- A variant beat the single/no-path controls on this development screen. It still requires retrained mechanism controls and a stronger/pretrained backbone before claim escalation.

## Next Decision

Retain only the best Stage 11B variant for a medium dev validation with retrained no-path, shared-query, K/V shuffle, and Actor-mask controls. Do not touch final evaluation yet.
