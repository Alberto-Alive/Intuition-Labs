# Stage 11F Private-View Q/A Bias Screen

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
- `private_view_no_path`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`none`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.0.
- `private_view_cross_causal_mean`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`cross_peer`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.0.
- `private_view_qa_hard_aux`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`qa_hard`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.35.
- `private_view_qa_soft_bias`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`qa_soft_bias`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.0.
- `private_view_qa_hard_aux_bias`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`qa_hard_bias`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.35.
- `private_view_qa_hard_aux_bias_late`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`qa_hard_bias`, query=`subtask`, summary=`causal_mean`, layers=[2], slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.35.
- `private_view_qa_hard_aux_bias_every`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`qa_hard_bias`, query=`subtask`, summary=`causal_mean`, layers=[0, 1, 2], slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.35.
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
| `full_view_qa_hard_aux` | 3 | 250318 | 0.6420 +/- 0.0381 | 0.3125 +/- 0.0221 | 0.3893 | 0.5133 | 0.2765 | 0.5133 | 0.6079 | 0.3905 | 0.0009 |
| `private_view_cross_causal_mean` | 3 | 250318 | 0.6061 +/- 0.0469 | 0.2188 +/- 0.1217 | 0.3584 | 0.7856 | 0.5865 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `private_view_no_path` | 3 | 173636 | 0.5309 +/- 0.0250 | 0.0938 +/- 0.0460 | 0.3109 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `private_view_qa_hard_aux` | 3 | 250318 | 0.5301 +/- 0.0312 | 0.1302 +/- 0.0448 | 0.3434 | 0.5426 | 0.3116 | 0.5426 | 0.6189 | 0.3779 | 0.0015 |
| `private_view_qa_hard_aux_bias` | 3 | 250318 | 0.5235 +/- 0.0339 | 0.1198 +/- 0.0266 | 0.3508 | 0.5353 | 0.3670 | 0.5353 | 0.6119 | 0.3850 | 0.0013 |
| `private_view_qa_hard_aux_bias_every` | 3 | 288659 | 0.5234 +/- 0.0178 | 0.1302 +/- 0.0410 | 0.3579 | 0.5527 | 0.3540 | 0.5527 | 0.6174 | 0.3794 | 0.0017 |
| `private_view_qa_hard_aux_bias_late` | 3 | 211977 | 0.5377 +/- 0.0314 | 0.1042 +/- 0.0266 | 0.3801 | 0.5107 | 0.3813 | 0.5107 | 0.6157 | 0.3829 | 0.0006 |
| `private_view_qa_soft_bias` | 3 | 250318 | 0.5232 +/- 0.0259 | 0.1094 +/- 0.0338 | 0.3372 | 1.2202 | 0.3675 | 1.2202 | 0.2147 | 0.2248 | 0.3275 |
| `role_clones_no_path` | 3 | 173636 | 0.6785 +/- 0.0089 | 0.3698 +/- 0.0940 | 0.3868 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `single_actor` | 3 | 173636 | 0.6785 +/- 0.0089 | 0.3698 +/- 0.0940 | 0.3868 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Mechanism Ablations On Base Proposed

No mechanism ablations were logged.

## Eval-Time Ablations By Cross Method

| method | ablation | rollout success mean | delta vs method |
|---|---|---:|---:|
| `full_view_qa_hard_aux` | `action_answer_shuffle` | 0.3125 | +0.0000 |
| `full_view_qa_hard_aux` | `cross_task_kv_shuffle` | 0.3073 | -0.0052 |
| `full_view_qa_hard_aux` | `mask_actor_peer_reads` | 0.2917 | -0.0208 |
| `full_view_qa_hard_aux` | `remove_path` | 0.2812 | -0.0312 |
| `full_view_qa_hard_aux` | `role_kv_shuffle` | 0.3125 | +0.0000 |
| `private_view_cross_causal_mean` | `cross_task_kv_shuffle` | 0.0885 | -0.1302 |
| `private_view_cross_causal_mean` | `mask_actor_peer_reads` | 0.0729 | -0.1458 |
| `private_view_cross_causal_mean` | `remove_path` | 0.1042 | -0.1146 |
| `private_view_cross_causal_mean` | `role_kv_shuffle` | 0.2083 | -0.0104 |
| `private_view_qa_hard_aux` | `action_answer_shuffle` | 0.1302 | +0.0000 |
| `private_view_qa_hard_aux` | `cross_task_kv_shuffle` | 0.1198 | -0.0104 |
| `private_view_qa_hard_aux` | `mask_actor_peer_reads` | 0.1094 | -0.0208 |
| `private_view_qa_hard_aux` | `remove_path` | 0.1094 | -0.0208 |
| `private_view_qa_hard_aux` | `role_kv_shuffle` | 0.1094 | -0.0208 |
| `private_view_qa_hard_aux_bias` | `action_answer_shuffle` | 0.1146 | -0.0052 |
| `private_view_qa_hard_aux_bias` | `cross_task_kv_shuffle` | 0.1250 | +0.0052 |
| `private_view_qa_hard_aux_bias` | `mask_actor_peer_reads` | 0.1042 | -0.0156 |
| `private_view_qa_hard_aux_bias` | `remove_path` | 0.0990 | -0.0208 |
| `private_view_qa_hard_aux_bias` | `role_kv_shuffle` | 0.1302 | +0.0104 |
| `private_view_qa_hard_aux_bias_every` | `action_answer_shuffle` | 0.1250 | -0.0052 |
| `private_view_qa_hard_aux_bias_every` | `cross_task_kv_shuffle` | 0.1198 | -0.0104 |
| `private_view_qa_hard_aux_bias_every` | `mask_actor_peer_reads` | 0.1146 | -0.0156 |
| `private_view_qa_hard_aux_bias_every` | `remove_path` | 0.1094 | -0.0208 |
| `private_view_qa_hard_aux_bias_every` | `role_kv_shuffle` | 0.1250 | -0.0052 |
| `private_view_qa_hard_aux_bias_late` | `action_answer_shuffle` | 0.1042 | +0.0000 |
| `private_view_qa_hard_aux_bias_late` | `cross_task_kv_shuffle` | 0.1094 | +0.0052 |
| `private_view_qa_hard_aux_bias_late` | `mask_actor_peer_reads` | 0.0990 | -0.0052 |
| `private_view_qa_hard_aux_bias_late` | `remove_path` | 0.0990 | -0.0052 |
| `private_view_qa_hard_aux_bias_late` | `role_kv_shuffle` | 0.1146 | +0.0104 |
| `private_view_qa_soft_bias` | `action_answer_shuffle` | 0.0833 | -0.0260 |
| `private_view_qa_soft_bias` | `cross_task_kv_shuffle` | 0.0625 | -0.0469 |
| `private_view_qa_soft_bias` | `mask_actor_peer_reads` | 0.0625 | -0.0469 |
| `private_view_qa_soft_bias` | `remove_path` | 0.0625 | -0.0469 |
| `private_view_qa_soft_bias` | `role_kv_shuffle` | 0.1042 | -0.0052 |

## Conclusions

- Best method in this screen: `role_clones_no_path` at `0.3698` mean rollout success.
- Single/no-path control ceiling in this screen: `0.3698` mean rollout success.
- `full_view_qa_hard_aux`: rollout `0.3125`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.0573`.
- `private_view_cross_causal_mean`: rollout `0.2188`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.1510`.
- `private_view_no_path`: rollout `0.0938`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.2760`.
- `private_view_qa_hard_aux`: rollout `0.1302`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.2396`.
- `private_view_qa_hard_aux_bias`: rollout `0.1198`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.2500`.
- `private_view_qa_hard_aux_bias_every`: rollout `0.1302`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.2396`.
- `private_view_qa_hard_aux_bias_late`: rollout `0.1042`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.2656`.
- `private_view_qa_soft_bias`: rollout `0.1094`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.2604`.
- Stage 11F private-view no-path baseline: `0.0938`. Best private Q/A relay `private_view_qa_hard_aux`: `0.1302` (`+0.0365` vs private no-path).
- Private continuous causal-mean peer read: `0.2188` (`+0.1250` vs private no-path).
- The private-view task redesign created a useful-dependence signal relative to the Actor-blind no-path baseline. This supports the information-partition premise, but the relay still needs stronger mechanism ablations and harder task slices before claim escalation.

## Next Decision

Retain `private_view_qa_hard_aux` for a targeted Stage 11F retest with more seeds, difficulty slices, and retrained private-view no-path/continuous-read controls. Do not compare it as a replacement for the original same-view Experiment 11 claim; it is a task-redesign branch.
