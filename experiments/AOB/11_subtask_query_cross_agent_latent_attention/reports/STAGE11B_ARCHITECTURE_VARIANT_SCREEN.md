# Stage 11B Architecture Variant Screen

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

- `single_actor`: roles=['actor'], cross_mode=`none`, query=`subtask`, summary=`last_token`, layers=default.
- `role_clones_no_path`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`none`, query=`subtask`, summary=`last_token`, layers=default.
- `capacity_control_self_read`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`self_only`, query=`subtask`, summary=`last_token`, layers=default.
- `cross_agent_latent_attention`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`cross_peer`, query=`subtask`, summary=`last_token`, layers=default.
- `cross_late_only`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`cross_peer`, query=`subtask`, summary=`last_token`, layers=[2].
- `cross_every_layer`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`cross_peer`, query=`subtask`, summary=`last_token`, layers=[0, 1, 2].
- `cross_subtask_state_query`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`cross_peer`, query=`subtask_state`, summary=`last_token`, layers=default.
- `cross_causal_mean_summary`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`cross_peer`, query=`subtask`, summary=`causal_mean`, layers=default.
- `cross_two_role_actor_critic`: roles=['critic', 'actor'], cross_mode=`cross_peer`, query=`subtask`, summary=`last_token`, layers=default.

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
| `cross_agent_latent_attention` | 3 | 223940 | 0.6748 +/- 0.0176 | 0.3125 +/- 0.0710 | 0.3779 | 1.0976 | 0.0931 |
| `cross_causal_mean_summary` | 3 | 223940 | 0.6566 +/- 0.0151 | 0.3646 +/- 0.1070 | 0.3851 | 1.0739 | 0.4182 |
| `cross_every_layer` | 3 | 249092 | 0.6690 +/- 0.0154 | 0.3281 +/- 0.0884 | 0.4090 | 1.0778 | 0.0822 |
| `cross_late_only` | 3 | 198788 | 0.6710 +/- 0.0144 | 0.3281 +/- 0.0893 | 0.3571 | 1.0510 | 0.1010 |
| `cross_subtask_state_query` | 3 | 223940 | 0.6758 +/- 0.0184 | 0.3125 +/- 0.0710 | 0.3778 | 1.0941 | 0.0931 |
| `cross_two_role_actor_critic` | 3 | 223940 | 0.6727 +/- 0.0162 | 0.3073 +/- 0.0725 | 0.3737 | 0.0000 | 0.0924 |
| `role_clones_no_path` | 3 | 173636 | 0.6785 +/- 0.0089 | 0.3698 +/- 0.0940 | 0.3868 | 0.0000 | 0.0000 |
| `single_actor` | 3 | 173636 | 0.6785 +/- 0.0089 | 0.3698 +/- 0.0940 | 0.3868 | 0.0000 | 0.0000 |

## Mechanism Ablations On Base Proposed

| ablation | rollout success mean | action acc mean |
|---|---:|---:|
| `cross_task_kv_shuffle` | 0.3385 +/- 0.0516 | 0.6803 +/- 0.0197 |
| `mask_actor_peer_reads` | 0.3125 +/- 0.0797 | 0.6747 +/- 0.0166 |
| `remove_path` | 0.3229 +/- 0.0703 | 0.6707 +/- 0.0148 |
| `role_kv_shuffle` | 0.3125 +/- 0.0710 | 0.6747 +/- 0.0176 |

## Conclusions

- Best method in this screen: `role_clones_no_path` at `0.3698` mean rollout success.
- Base proposed method: `0.3125` mean rollout success.
- Single/no-path control ceiling in this screen: `0.3698` mean rollout success.
- `cross_causal_mean_summary`: rollout `0.3646`, delta vs base `+0.0521`, delta vs best single/no-path `-0.0052`.
- `cross_every_layer`: rollout `0.3281`, delta vs base `+0.0156`, delta vs best single/no-path `-0.0417`.
- `cross_late_only`: rollout `0.3281`, delta vs base `+0.0156`, delta vs best single/no-path `-0.0417`.
- `cross_subtask_state_query`: rollout `0.3125`, delta vs base `+0.0000`, delta vs best single/no-path `-0.0573`.
- `cross_two_role_actor_critic`: rollout `0.3073`, delta vs base `-0.0052`, delta vs best single/no-path `-0.0625`.
- No cross-agent variant beat the single/no-path controls. The current evidence says these architectural tweaks do not rescue the randomly initialized gridworld setup.
- Base proposed mechanism ablations still did not materially damage rollout success, so attention patterns remain non-causal evidence.

## Next Decision

Do not spend final-evaluation compute on this randomized-decoder gridworld line. The next discriminating step is a pretrained causal decoder or a task redesign that demonstrably requires complementary role-state information.
