# Stage 11F Private-View Q/A Retest 16 Epochs Extra Seeds

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
- `private_view_qa_hard_aux_bias`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`qa_hard_bias`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.35.

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

Same executable gridworld harness as Stage 11A. Train worlds `192`, dev worlds `64`, epochs `16`, batch size `96`, seeds `[104, 105]`.
Hardware `cuda`, CUDA `True`, GPU `NVIDIA GeForce RTX 5070 Ti`.

## Results

| method | seeds | params | action acc mean | rollout success mean | valid action mean | entropy | inject norm | qa entropy | yes | no | irrelevant |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `private_view_cross_causal_mean` | 2 | 250318 | 0.6727 +/- 0.0245 | 0.3281 +/- 0.0625 | 0.4359 | 0.5770 | 0.6250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `private_view_no_path` | 2 | 173636 | 0.5339 +/- 0.0034 | 0.1094 +/- 0.0156 | 0.4107 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `private_view_qa_hard_aux_bias` | 2 | 250318 | 0.6204 +/- 0.0318 | 0.2656 +/- 0.0156 | 0.4708 | 0.3204 | 0.3255 | 0.3204 | 0.5390 | 0.4582 | 0.0018 |
| `role_clones_no_path` | 2 | 173636 | 0.6726 +/- 0.0153 | 0.4141 +/- 0.0391 | 0.4180 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `single_actor` | 2 | 173636 | 0.6726 +/- 0.0153 | 0.4141 +/- 0.0391 | 0.4180 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Mechanism Ablations On Base Proposed

No mechanism ablations were logged.

## Eval-Time Ablations By Cross Method

| method | ablation | rollout success mean | delta vs method |
|---|---|---:|---:|
| `private_view_cross_causal_mean` | `cross_task_kv_shuffle` | 0.0469 | -0.2812 |
| `private_view_cross_causal_mean` | `mask_actor_peer_reads` | 0.0781 | -0.2500 |
| `private_view_cross_causal_mean` | `remove_path` | 0.0625 | -0.2656 |
| `private_view_cross_causal_mean` | `role_kv_shuffle` | 0.1719 | -0.1562 |
| `private_view_qa_hard_aux_bias` | `action_answer_shuffle` | 0.0625 | -0.2031 |
| `private_view_qa_hard_aux_bias` | `cross_task_kv_shuffle` | 0.0859 | -0.1797 |
| `private_view_qa_hard_aux_bias` | `mask_actor_peer_reads` | 0.1016 | -0.1641 |
| `private_view_qa_hard_aux_bias` | `remove_path` | 0.1094 | -0.1562 |
| `private_view_qa_hard_aux_bias` | `role_kv_shuffle` | 0.1953 | -0.0703 |

## Conclusions

- Best method in this screen: `role_clones_no_path` at `0.4141` mean rollout success.
- Single/no-path control ceiling in this screen: `0.4141` mean rollout success.
- `private_view_cross_causal_mean`: rollout `0.3281`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.0859`.
- `private_view_no_path`: rollout `0.1094`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.3047`.
- `private_view_qa_hard_aux_bias`: rollout `0.2656`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.1484`.
- Stage 11F private-view no-path baseline: `0.1094`. Best private Q/A relay `private_view_qa_hard_aux_bias`: `0.2656` (`+0.1562` vs private no-path).
- Private continuous causal-mean peer read: `0.3281` (`+0.2188` vs private no-path).
- The private-view task redesign created a useful-dependence signal relative to the Actor-blind no-path baseline. This supports the information-partition premise, but the relay still needs stronger mechanism ablations and harder task slices before claim escalation.

## Next Decision

Retain `private_view_qa_hard_aux_bias` for a targeted Stage 11F retest with more seeds, difficulty slices, and retrained private-view no-path/continuous-read controls. Do not compare it as a replacement for the original same-view Experiment 11 claim; it is a task-redesign branch.
