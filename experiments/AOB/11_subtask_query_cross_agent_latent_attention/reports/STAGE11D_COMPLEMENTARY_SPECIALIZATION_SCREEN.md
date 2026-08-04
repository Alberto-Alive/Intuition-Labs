# Stage 11D Complementary Specialization Screen

## Hypothesis Tested

Slot-export bottlenecks, sparse top-k peer reads, and role dropout may force more complementary role specialization if the previous failures came from redundant all-role summaries rather than the cross-agent premise itself.

## Tried Before This Batch

- Single Actor baseline with one Actor stream and no cross-agent path.
- Role Clones, No Latent Cross-Agent Path with Planner/Reader/Critic/Actor streams.
- Capacity Control, Self Read with the cross block restricted to self-state reads.
- Base Proposed Cross-Agent Latent Attention with middle-plus-late insertion, last-token summaries, and subtask-only Q.
- Shared-Query Control replacing role/subtask-specific Q with a learned shared query.
- Eval-time ablations on the base proposed model: remove path, role K/V shuffle, cross-task K/V shuffle, Actor peer-read mask.

## Newly Tried Or Retested In Stage 11D

- `single_actor`: roles=['actor'], cross_mode=`none`, query=`subtask`, summary=`last_token`, layers=default, slots=0, topk=0, role_dropout=0.0.
- `role_clones_no_path`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`none`, query=`subtask`, summary=`last_token`, layers=default, slots=0, topk=0, role_dropout=0.0.
- `capacity_control_self_read`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`self_only`, query=`subtask`, summary=`last_token`, layers=default, slots=0, topk=0, role_dropout=0.0.
- `cross_causal_mean_summary`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`cross_peer`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0.
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

Same executable gridworld harness as Stage 11A. Train worlds `192`, dev worlds `64`, epochs `8`, batch size `96`, seeds `[101, 102, 103]`.
Hardware `cuda`, CUDA `True`, GPU `NVIDIA GeForce RTX 5070 Ti`.

## Results

| method | seeds | params | action acc mean | rollout success mean | valid action mean | entropy | inject norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| `capacity_control_self_read` | 3 | 223940 | 0.6737 +/- 0.0158 | 0.2969 +/- 0.0710 | 0.3749 | 0.0000 | 0.0954 |
| `cross_causal_mean_summary` | 3 | 223940 | 0.6566 +/- 0.0151 | 0.3646 +/- 0.1070 | 0.3851 | 1.0739 | 0.4182 |
| `role_clones_no_path` | 3 | 173636 | 0.6785 +/- 0.0089 | 0.3698 +/- 0.0940 | 0.3868 | 0.0000 | 0.0000 |
| `single_actor` | 3 | 173636 | 0.6785 +/- 0.0089 | 0.3698 +/- 0.0940 | 0.3868 | 0.0000 | 0.0000 |
| `slot_bottleneck_m2` | 3 | 240836 | 0.6746 +/- 0.0083 | 0.3021 +/- 0.0147 | 0.3598 | 1.7916 | 0.1899 |
| `slot_bottleneck_m2_dropout` | 3 | 240836 | 0.6823 +/- 0.0108 | 0.3125 +/- 0.0221 | 0.3599 | 1.7912 | 0.1808 |
| `slot_bottleneck_m2_topk2` | 3 | 240836 | 0.6862 +/- 0.0044 | 0.3229 +/- 0.0195 | 0.3549 | 0.6917 | 0.1860 |
| `slot_bottleneck_m4_topk2` | 3 | 241092 | 0.6665 +/- 0.0130 | 0.3021 +/- 0.0849 | 0.3811 | 0.6889 | 0.2625 |
| `slot_self_read_m2` | 3 | 240836 | 0.6762 +/- 0.0092 | 0.2969 +/- 0.0000 | 0.3578 | 0.6865 | 0.1912 |

## Mechanism Ablations On Base Proposed

No mechanism ablations were logged.

## Eval-Time Ablations By Cross Method

| method | ablation | rollout success mean | delta vs method |
|---|---|---:|---:|
| `cross_causal_mean_summary` | `cross_task_kv_shuffle` | 0.3073 | -0.0573 |
| `cross_causal_mean_summary` | `mask_actor_peer_reads` | 0.3906 | +0.0260 |
| `cross_causal_mean_summary` | `remove_path` | 0.3906 | +0.0260 |
| `cross_causal_mean_summary` | `role_kv_shuffle` | 0.3698 | +0.0052 |
| `slot_bottleneck_m2` | `cross_task_kv_shuffle` | 0.3073 | +0.0052 |
| `slot_bottleneck_m2` | `mask_actor_peer_reads` | 0.3125 | +0.0104 |
| `slot_bottleneck_m2` | `remove_path` | 0.3229 | +0.0208 |
| `slot_bottleneck_m2` | `role_kv_shuffle` | 0.3073 | +0.0052 |
| `slot_bottleneck_m2_dropout` | `cross_task_kv_shuffle` | 0.3073 | -0.0052 |
| `slot_bottleneck_m2_dropout` | `mask_actor_peer_reads` | 0.3229 | +0.0104 |
| `slot_bottleneck_m2_dropout` | `remove_path` | 0.3281 | +0.0156 |
| `slot_bottleneck_m2_dropout` | `role_kv_shuffle` | 0.3177 | +0.0052 |
| `slot_bottleneck_m2_topk2` | `cross_task_kv_shuffle` | 0.3125 | -0.0104 |
| `slot_bottleneck_m2_topk2` | `mask_actor_peer_reads` | 0.3229 | +0.0000 |
| `slot_bottleneck_m2_topk2` | `remove_path` | 0.3229 | +0.0000 |
| `slot_bottleneck_m2_topk2` | `role_kv_shuffle` | 0.3281 | +0.0052 |
| `slot_bottleneck_m4_topk2` | `cross_task_kv_shuffle` | 0.3125 | +0.0104 |
| `slot_bottleneck_m4_topk2` | `mask_actor_peer_reads` | 0.3177 | +0.0156 |
| `slot_bottleneck_m4_topk2` | `remove_path` | 0.3177 | +0.0156 |
| `slot_bottleneck_m4_topk2` | `role_kv_shuffle` | 0.2969 | -0.0052 |

## Conclusions

- Best method in this screen: `role_clones_no_path` at `0.3698` mean rollout success.
- Reference cross method `cross_causal_mean_summary`: `0.3646` mean rollout success.
- Single/no-path control ceiling in this screen: `0.3698` mean rollout success.
- `cross_causal_mean_summary`: rollout `0.3646`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.0052`.
- `slot_bottleneck_m2`: rollout `0.3021`, delta vs reference `-0.0625`, delta vs best single/no-path `-0.0677`.
- `slot_bottleneck_m2_dropout`: rollout `0.3125`, delta vs reference `-0.0521`, delta vs best single/no-path `-0.0573`.
- `slot_bottleneck_m2_topk2`: rollout `0.3229`, delta vs reference `-0.0417`, delta vs best single/no-path `-0.0469`.
- `slot_bottleneck_m4_topk2`: rollout `0.3021`, delta vs reference `-0.0625`, delta vs best single/no-path `-0.0677`.
- `slot_self_read_m2`: rollout `0.2969`, delta vs reference `-0.0677`, delta vs best single/no-path `-0.0729`.
- No cross-agent variant beat the single/no-path controls. The current evidence says these architectural tweaks do not rescue the randomly initialized gridworld setup.

## Next Decision

Do not spend final-evaluation compute on this randomized-decoder gridworld line. The next discriminating step is a pretrained causal decoder or a task redesign that demonstrably requires complementary role-state information.
