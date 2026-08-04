# Stage 11F Private-View Q/A Retest 16 Epochs

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

Same executable gridworld harness as Stage 11A. Train worlds `192`, dev worlds `64`, epochs `16`, batch size `96`, seeds `[101, 102, 103]`.
Hardware `cuda`, CUDA `True`, GPU `NVIDIA GeForce RTX 5070 Ti`.

## Results

| method | seeds | params | action acc mean | rollout success mean | valid action mean | entropy | inject norm | qa entropy | yes | no | irrelevant |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `private_view_cross_causal_mean` | 3 | 250318 | 0.6346 +/- 0.0090 | 0.2500 +/- 0.0383 | 0.3889 | 0.6306 | 0.7351 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `private_view_no_path` | 3 | 173636 | 0.5046 +/- 0.0166 | 0.1146 +/- 0.0195 | 0.4443 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `private_view_qa_hard_aux` | 3 | 250318 | 0.5214 +/- 0.0337 | 0.1146 +/- 0.0295 | 0.4296 | 0.3816 | 0.2748 | 0.3816 | 0.6030 | 0.3957 | 0.0007 |
| `private_view_qa_hard_aux_bias` | 3 | 250318 | 0.6205 +/- 0.0275 | 0.3021 +/- 0.0516 | 0.4628 | 0.3159 | 0.3227 | 0.3159 | 0.6432 | 0.3557 | 0.0006 |
| `role_clones_no_path` | 3 | 173636 | 0.6333 +/- 0.0143 | 0.3490 +/- 0.0531 | 0.4419 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `single_actor` | 3 | 173636 | 0.6328 +/- 0.0137 | 0.3490 +/- 0.0531 | 0.4419 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Mechanism Ablations On Base Proposed

No mechanism ablations were logged.

## Eval-Time Ablations By Cross Method

| method | ablation | rollout success mean | delta vs method |
|---|---|---:|---:|
| `private_view_cross_causal_mean` | `cross_task_kv_shuffle` | 0.0729 | -0.1771 |
| `private_view_cross_causal_mean` | `mask_actor_peer_reads` | 0.0938 | -0.1562 |
| `private_view_cross_causal_mean` | `remove_path` | 0.1198 | -0.1302 |
| `private_view_cross_causal_mean` | `role_kv_shuffle` | 0.2396 | -0.0104 |
| `private_view_qa_hard_aux` | `action_answer_shuffle` | 0.1146 | +0.0000 |
| `private_view_qa_hard_aux` | `cross_task_kv_shuffle` | 0.1042 | -0.0104 |
| `private_view_qa_hard_aux` | `mask_actor_peer_reads` | 0.1302 | +0.0156 |
| `private_view_qa_hard_aux` | `remove_path` | 0.1302 | +0.0156 |
| `private_view_qa_hard_aux` | `role_kv_shuffle` | 0.1302 | +0.0156 |
| `private_view_qa_hard_aux_bias` | `action_answer_shuffle` | 0.0365 | -0.2656 |
| `private_view_qa_hard_aux_bias` | `cross_task_kv_shuffle` | 0.1042 | -0.1979 |
| `private_view_qa_hard_aux_bias` | `mask_actor_peer_reads` | 0.1354 | -0.1667 |
| `private_view_qa_hard_aux_bias` | `remove_path` | 0.1250 | -0.1771 |
| `private_view_qa_hard_aux_bias` | `role_kv_shuffle` | 0.2292 | -0.0729 |

## Conclusions

- Best method in this screen: `role_clones_no_path` at `0.3490` mean rollout success.
- Single/no-path control ceiling in this screen: `0.3490` mean rollout success.
- `private_view_cross_causal_mean`: rollout `0.2500`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.0990`.
- `private_view_no_path`: rollout `0.1146`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.2344`.
- `private_view_qa_hard_aux`: rollout `0.1146`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.2344`.
- `private_view_qa_hard_aux_bias`: rollout `0.3021`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.0469`.
- Stage 11F private-view no-path baseline: `0.1146`. Best private Q/A relay `private_view_qa_hard_aux_bias`: `0.3021` (`+0.1875` vs private no-path).
- Private continuous causal-mean peer read: `0.2500` (`+0.1354` vs private no-path).
- The private-view task redesign created a useful-dependence signal relative to the Actor-blind no-path baseline. This supports the information-partition premise, but the relay still needs stronger mechanism ablations and harder task slices before claim escalation.

## Next Decision

Retain `private_view_qa_hard_aux_bias` for a targeted Stage 11F retest with more seeds, difficulty slices, and retrained private-view no-path/continuous-read controls. Do not compare it as a replacement for the original same-view Experiment 11 claim; it is a task-redesign branch.
