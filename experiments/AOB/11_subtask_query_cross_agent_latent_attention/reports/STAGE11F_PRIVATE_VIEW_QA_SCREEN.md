# Stage 11F Private-View Q/A Relay Screen

## Hypothesis Tested

Private role views plus a constrained latent question/answer relay may create real inter-agent dependence: the Actor is missing goal and map evidence, specialists hold complementary private state, and the Actor can only recover it through low-bandwidth action-conditioned answers.

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
- Stage 11E transverse-linearity diagnostics: role-gated shared-basis LoRA, fixed-role hard LoRA, no-path/self-read controls, and LoRA plus cross-agent causal-mean/slot reads.

## Newly Tried Or Retested In Stage 11F

- `single_actor`: roles=['actor'], cross_mode=`none`, query=`subtask`, summary=`last_token`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=False, qa_aux=0.0.
- `role_clones_no_path`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`none`, query=`subtask`, summary=`last_token`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=False, qa_aux=0.0.
- `cross_causal_mean_summary`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`cross_peer`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=False, qa_aux=0.0.
- `private_view_no_path`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`none`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.0.
- `private_view_cross_causal_mean`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`cross_peer`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.0.
- `private_view_qa_soft`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`qa_soft`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.0.
- `private_view_qa_hard`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`qa_hard`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.0.
- `private_view_qa_hard_aux`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`qa_hard`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.35.
- `private_view_qa_hard_aux_late`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`qa_hard`, query=`subtask`, summary=`causal_mean`, layers=[2], slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.35.
- `private_view_qa_hard_aux_every`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`qa_hard`, query=`subtask`, summary=`causal_mean`, layers=[0, 1, 2], slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.35.
- `full_view_qa_hard_aux`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`qa_hard`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=False, qa_aux=0.35.

## Still Not Tried

- A pretrained causal decoder checkpoint.
- Full token-level cross-role attention instead of role summaries.
- Learned or external subtask decomposition beyond fixed role/subtask tokens.
- Learned residual gate schedules or initialized near-zero gates.
- Retrained mechanism controls for every variant.
- Actor-only source routing or fixed sparse role-routing graphs.
- Stateful multi-turn latent Q/A with learned query selection rather than fixed action-conditioned questions.
- More role-count schedules beyond four roles and Actor/Critic two-role.
- Larger model/data/epoch scaling and difficulty-sliced validation.
- Held-out final evaluation.

## Task And Budget

Same executable gridworld harness as Stage 11A. Train worlds `192`, dev worlds `64`, epochs `8`, batch size `96`, seeds `[101, 102, 103]`.
Hardware `cuda`, CUDA `True`, GPU `NVIDIA GeForce RTX 5070 Ti`.

## Results

| method | seeds | params | action acc mean | rollout success mean | valid action mean | entropy | inject norm | qa entropy | yes | no | irrelevant |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cross_causal_mean_summary` | 3 | 250188 | 0.6655 +/- 0.0201 | 0.2812 +/- 0.0221 | 0.3797 | 1.0629 | 0.4061 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `full_view_qa_hard_aux` | 3 | 250188 | 0.6966 +/- 0.0276 | 0.4167 +/- 0.0820 | 0.4068 | 0.5137 | 0.2583 | 0.5137 | 0.6095 | 0.3892 | 0.0007 |
| `private_view_cross_causal_mean` | 3 | 250188 | 0.5801 +/- 0.0552 | 0.1823 +/- 0.0589 | 0.3663 | 0.8120 | 0.6399 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `private_view_no_path` | 3 | 173636 | 0.5309 +/- 0.0250 | 0.0938 +/- 0.0460 | 0.3109 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `private_view_qa_hard` | 3 | 250188 | 0.5199 +/- 0.0317 | 0.0990 +/- 0.0195 | 0.3266 | 1.0430 | 0.3907 | 1.0430 | 0.1793 | 0.1794 | 0.3939 |
| `private_view_qa_hard_aux` | 3 | 250188 | 0.5272 +/- 0.0274 | 0.1146 +/- 0.0531 | 0.3467 | 0.5064 | 0.3390 | 0.5064 | 0.6229 | 0.3746 | 0.0013 |
| `private_view_qa_hard_aux_every` | 3 | 288464 | 0.5276 +/- 0.0323 | 0.1042 +/- 0.0390 | 0.3032 | 0.5461 | 0.3247 | 0.5461 | 0.6156 | 0.3797 | 0.0024 |
| `private_view_qa_hard_aux_late` | 3 | 211912 | 0.5396 +/- 0.0366 | 0.1250 +/- 0.0221 | 0.3622 | 0.5131 | 0.3418 | 0.5131 | 0.5927 | 0.4060 | 0.0006 |
| `private_view_qa_soft` | 3 | 250188 | 0.5172 +/- 0.0205 | 0.1042 +/- 0.0074 | 0.3256 | 1.2111 | 0.3042 | 1.2111 | 0.2323 | 0.2583 | 0.2581 |
| `role_clones_no_path` | 3 | 173636 | 0.6785 +/- 0.0089 | 0.3698 +/- 0.0940 | 0.3868 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `single_actor` | 3 | 173636 | 0.6785 +/- 0.0089 | 0.3698 +/- 0.0940 | 0.3868 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Mechanism Ablations On Base Proposed

No mechanism ablations were logged.

## Eval-Time Ablations By Cross Method

| method | ablation | rollout success mean | delta vs method |
|---|---|---:|---:|
| `cross_causal_mean_summary` | `cross_task_kv_shuffle` | 0.2500 | -0.0312 |
| `cross_causal_mean_summary` | `mask_actor_peer_reads` | 0.2812 | +0.0000 |
| `cross_causal_mean_summary` | `remove_path` | 0.2865 | +0.0052 |
| `cross_causal_mean_summary` | `role_kv_shuffle` | 0.2865 | +0.0052 |
| `full_view_qa_hard_aux` | `action_answer_shuffle` | 0.4167 | +0.0000 |
| `full_view_qa_hard_aux` | `cross_task_kv_shuffle` | 0.4062 | -0.0104 |
| `full_view_qa_hard_aux` | `mask_actor_peer_reads` | 0.3906 | -0.0260 |
| `full_view_qa_hard_aux` | `remove_path` | 0.4010 | -0.0156 |
| `full_view_qa_hard_aux` | `role_kv_shuffle` | 0.4115 | -0.0052 |
| `private_view_cross_causal_mean` | `cross_task_kv_shuffle` | 0.1146 | -0.0677 |
| `private_view_cross_causal_mean` | `mask_actor_peer_reads` | 0.0781 | -0.1042 |
| `private_view_cross_causal_mean` | `remove_path` | 0.0990 | -0.0833 |
| `private_view_cross_causal_mean` | `role_kv_shuffle` | 0.1875 | +0.0052 |
| `private_view_qa_hard` | `action_answer_shuffle` | 0.0990 | +0.0000 |
| `private_view_qa_hard` | `cross_task_kv_shuffle` | 0.0990 | +0.0000 |
| `private_view_qa_hard` | `mask_actor_peer_reads` | 0.1198 | +0.0208 |
| `private_view_qa_hard` | `remove_path` | 0.1198 | +0.0208 |
| `private_view_qa_hard` | `role_kv_shuffle` | 0.0990 | +0.0000 |
| `private_view_qa_hard_aux` | `action_answer_shuffle` | 0.1146 | +0.0000 |
| `private_view_qa_hard_aux` | `cross_task_kv_shuffle` | 0.1146 | +0.0000 |
| `private_view_qa_hard_aux` | `mask_actor_peer_reads` | 0.1250 | +0.0104 |
| `private_view_qa_hard_aux` | `remove_path` | 0.1198 | +0.0052 |
| `private_view_qa_hard_aux` | `role_kv_shuffle` | 0.1250 | +0.0104 |
| `private_view_qa_hard_aux_every` | `action_answer_shuffle` | 0.1042 | +0.0000 |
| `private_view_qa_hard_aux_every` | `cross_task_kv_shuffle` | 0.1042 | +0.0000 |
| `private_view_qa_hard_aux_every` | `mask_actor_peer_reads` | 0.0885 | -0.0156 |
| `private_view_qa_hard_aux_every` | `remove_path` | 0.0885 | -0.0156 |
| `private_view_qa_hard_aux_every` | `role_kv_shuffle` | 0.1094 | +0.0052 |
| `private_view_qa_hard_aux_late` | `action_answer_shuffle` | 0.1250 | +0.0000 |
| `private_view_qa_hard_aux_late` | `cross_task_kv_shuffle` | 0.1250 | +0.0000 |
| `private_view_qa_hard_aux_late` | `mask_actor_peer_reads` | 0.1250 | +0.0000 |
| `private_view_qa_hard_aux_late` | `remove_path` | 0.1250 | +0.0000 |
| `private_view_qa_hard_aux_late` | `role_kv_shuffle` | 0.1250 | +0.0000 |
| `private_view_qa_soft` | `action_answer_shuffle` | 0.1042 | +0.0000 |
| `private_view_qa_soft` | `cross_task_kv_shuffle` | 0.1094 | +0.0052 |
| `private_view_qa_soft` | `mask_actor_peer_reads` | 0.1146 | +0.0104 |
| `private_view_qa_soft` | `remove_path` | 0.1250 | +0.0208 |
| `private_view_qa_soft` | `role_kv_shuffle` | 0.1042 | +0.0000 |

## Conclusions

- Best method in this screen: `full_view_qa_hard_aux` at `0.4167` mean rollout success.
- Reference cross method `cross_causal_mean_summary`: `0.2812` mean rollout success.
- Single/no-path control ceiling in this screen: `0.3698` mean rollout success.
- `cross_causal_mean_summary`: rollout `0.2812`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.0885`.
- `full_view_qa_hard_aux`: rollout `0.4167`, delta vs reference `+0.1354`, delta vs best single/no-path `+0.0469`.
- `private_view_cross_causal_mean`: rollout `0.1823`, delta vs reference `-0.0990`, delta vs best single/no-path `-0.1875`.
- `private_view_no_path`: rollout `0.0938`, delta vs reference `-0.1875`, delta vs best single/no-path `-0.2760`.
- `private_view_qa_hard`: rollout `0.0990`, delta vs reference `-0.1823`, delta vs best single/no-path `-0.2708`.
- `private_view_qa_hard_aux`: rollout `0.1146`, delta vs reference `-0.1667`, delta vs best single/no-path `-0.2552`.
- `private_view_qa_hard_aux_every`: rollout `0.1042`, delta vs reference `-0.1771`, delta vs best single/no-path `-0.2656`.
- `private_view_qa_hard_aux_late`: rollout `0.1250`, delta vs reference `-0.1562`, delta vs best single/no-path `-0.2448`.
- `private_view_qa_soft`: rollout `0.1042`, delta vs reference `-0.1771`, delta vs best single/no-path `-0.2656`.
- Stage 11F private-view no-path baseline: `0.0938`. Best private Q/A relay `private_view_qa_hard_aux_late`: `0.1250` (`+0.0312` vs private no-path).
- Private continuous causal-mean peer read: `0.1823` (`+0.0885` vs private no-path).
- The private-view task redesign created a useful-dependence signal relative to the Actor-blind no-path baseline. This supports the information-partition premise, but the relay still needs stronger mechanism ablations and harder task slices before claim escalation.
- Best-variant eval-time mechanism drops: `[('mask_actor_peer_reads', -0.026041666666666685)]`.

## Next Decision

Retain `private_view_qa_hard_aux_late` for a targeted Stage 11F retest with more seeds, difficulty slices, and retrained private-view no-path/continuous-read controls. Do not compare it as a replacement for the original same-view Experiment 11 claim; it is a task-redesign branch.
