# Stage 11E Role-Gated LoRA Screen

## Hypothesis Tested

Role-gated shared-basis LoRA adapters may break transverse linearity across cloned role streams. The critical test is whether LoRA plus cross-agent latent reads beats the matching LoRA no-path control.

## Tried Before This Batch

- Single Actor baseline with one Actor stream and no cross-agent path.
- Role Clones, No Latent Cross-Agent Path with Planner/Reader/Critic/Actor streams.
- Capacity Control, Self Read with the cross block restricted to self-state reads.
- Base Proposed Cross-Agent Latent Attention with middle-plus-late insertion, last-token summaries, and subtask-only Q.
- Shared-Query Control replacing role/subtask-specific Q with a learned shared query.
- Eval-time ablations on the base proposed model: remove path, role K/V shuffle, cross-task K/V shuffle, Actor peer-read mask.
- Stage 11B layer, query, summary, and role-count variants: late-only, every-layer, subtask+state Q, causal-mean summaries, and Actor/Critic two-role.
- Stage 11C longer retest of the causal-mean summary variant against selected controls.
- Stage 11D complementary-specialization variants: slot self-read, slot bottlenecks, sparse top-k slot reads, and source-role dropout.

## Newly Tried Or Retested In Stage 11E

- `single_actor`: roles=['actor'], cross_mode=`none`, query=`subtask`, summary=`last_token`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax.
- `role_clones_no_path`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`none`, query=`subtask`, summary=`last_token`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax.
- `capacity_control_self_read`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`self_only`, query=`subtask`, summary=`last_token`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax.
- `cross_causal_mean_summary`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`cross_peer`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax.
- `role_lora_no_path`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`none`, query=`subtask`, summary=`last_token`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=8, lora_bases=4, lora_alpha=8.0, lora_gate=softmax.
- `role_lora_self_read`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`self_only`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=8, lora_bases=4, lora_alpha=8.0, lora_gate=softmax.
- `role_lora_cross_causal_mean`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`cross_peer`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=8, lora_bases=4, lora_alpha=8.0, lora_gate=softmax.
- `role_lora_cross_slots_m2_topk2`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`cross_peer`, query=`subtask`, summary=`causal_mean`, layers=default, slots=2, topk=2, role_dropout=0.0, lora_rank=8, lora_bases=4, lora_alpha=8.0, lora_gate=softmax.

## Still Not Tried

- A pretrained causal decoder checkpoint.
- Full token-level cross-role attention instead of role summaries.
- Learned or external subtask decomposition beyond fixed role/subtask tokens.
- Learned residual gate schedules or initialized near-zero gates.
- Retrained mechanism controls for every variant.
- Actor-only source routing or fixed sparse role-routing graphs.
- More role-count schedules beyond four roles and Actor/Critic two-role.
- Larger model/data/epoch scaling and difficulty-sliced validation.
- Held-out final evaluation.

## Task And Budget

Same executable gridworld harness as Stage 11A. Train worlds `192`, dev worlds `64`, epochs `8`, batch size `96`, seeds `[101, 102, 103]`.
Hardware `cuda`, CUDA `True`, GPU `NVIDIA GeForce RTX 5070 Ti`.

## Results

| method | seeds | params | action acc mean | rollout success mean | valid action mean | entropy | inject norm | lora norm | lora gate entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `capacity_control_self_read` | 3 | 223940 | 0.6737 +/- 0.0158 | 0.2969 +/- 0.0710 | 0.3749 | 0.0000 | 0.0954 | 0.0000 | 0.0000 |
| `cross_causal_mean_summary` | 3 | 223940 | 0.6566 +/- 0.0151 | 0.3646 +/- 0.1070 | 0.3851 | 1.0739 | 0.4182 | 0.0000 | 0.0000 |
| `role_clones_no_path` | 3 | 173636 | 0.6785 +/- 0.0089 | 0.3698 +/- 0.0940 | 0.3868 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `role_lora_cross_causal_mean` | 3 | 236660 | 0.6684 +/- 0.0206 | 0.3542 +/- 0.1309 | 0.3885 | 1.0819 | 0.3653 | 0.0183 | 1.3859 |
| `role_lora_cross_slots_m2_topk2` | 3 | 253556 | 0.6606 +/- 0.0138 | 0.3281 +/- 0.0765 | 0.3886 | 0.6830 | 0.2294 | 0.0179 | 1.3859 |
| `role_lora_no_path` | 3 | 186356 | 0.6988 +/- 0.0212 | 0.3385 +/- 0.0769 | 0.3589 | 0.0000 | 0.0000 | 0.0179 | 1.3861 |
| `role_lora_self_read` | 3 | 236660 | 0.6628 +/- 0.0221 | 0.3073 +/- 0.1226 | 0.4225 | 0.0000 | 0.3891 | 0.0180 | 1.3861 |
| `single_actor` | 3 | 173636 | 0.6785 +/- 0.0089 | 0.3698 +/- 0.0940 | 0.3868 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Mechanism Ablations On Base Proposed

No mechanism ablations were logged.

## Eval-Time Ablations By Cross Method

| method | ablation | rollout success mean | delta vs method |
|---|---|---:|---:|
| `cross_causal_mean_summary` | `cross_task_kv_shuffle` | 0.3073 | -0.0573 |
| `cross_causal_mean_summary` | `mask_actor_peer_reads` | 0.3906 | +0.0260 |
| `cross_causal_mean_summary` | `remove_path` | 0.3906 | +0.0260 |
| `cross_causal_mean_summary` | `role_kv_shuffle` | 0.3698 | +0.0052 |
| `role_lora_cross_causal_mean` | `cross_task_kv_shuffle` | 0.2969 | -0.0573 |
| `role_lora_cross_causal_mean` | `mask_actor_peer_reads` | 0.3490 | -0.0052 |
| `role_lora_cross_causal_mean` | `remove_path` | 0.3073 | -0.0469 |
| `role_lora_cross_causal_mean` | `role_kv_shuffle` | 0.3438 | -0.0104 |
| `role_lora_cross_slots_m2_topk2` | `cross_task_kv_shuffle` | 0.3229 | -0.0052 |
| `role_lora_cross_slots_m2_topk2` | `mask_actor_peer_reads` | 0.3333 | +0.0052 |
| `role_lora_cross_slots_m2_topk2` | `remove_path` | 0.3229 | -0.0052 |
| `role_lora_cross_slots_m2_topk2` | `role_kv_shuffle` | 0.3281 | +0.0000 |

## Conclusions

- Best method in this screen: `role_clones_no_path` at `0.3698` mean rollout success.
- Reference cross method `cross_causal_mean_summary`: `0.3646` mean rollout success.
- Single/no-path control ceiling in this screen: `0.3698` mean rollout success.
- `cross_causal_mean_summary`: rollout `0.3646`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.0052`.
- `role_lora_cross_causal_mean`: rollout `0.3542`, delta vs reference `-0.0104`, delta vs best single/no-path `-0.0156`.
- `role_lora_cross_slots_m2_topk2`: rollout `0.3281`, delta vs reference `-0.0365`, delta vs best single/no-path `-0.0417`.
- `role_lora_no_path`: rollout `0.3385`, delta vs reference `-0.0260`, delta vs best single/no-path `-0.0312`.
- `role_lora_self_read`: rollout `0.3073`, delta vs reference `-0.0573`, delta vs best single/no-path `-0.0625`.
- LoRA cross-path test `role_lora_cross_causal_mean` vs `role_lora_no_path`: `0.3542` vs `0.3385` (`+0.0156`).
- LoRA cross-path test `role_lora_cross_slots_m2_topk2` vs `role_lora_no_path`: `0.3281` vs `0.3385` (`-0.0104`).
- No LoRA cross-agent variant beat the single/no-path controls. The LoRA branch does not rescue the randomized-decoder gridworld setup.

## Next Decision

Do not escalate the LoRA cross-agent branch on this randomized gridworld. Explicit symmetry breaking can change behavior, but it has not made peer latent reads beat their matching no-path controls and the original single/no-path baselines.
