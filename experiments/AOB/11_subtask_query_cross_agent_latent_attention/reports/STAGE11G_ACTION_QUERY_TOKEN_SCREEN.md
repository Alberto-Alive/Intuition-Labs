# Stage 11G Action-Query Token Relay Screen

## Hypothesis Tested

Explicit action-query tokens may make the private-view Q/A relay more GPT-native than the Stage 11F direct action-logit bias: peer answers update the Actor's per-action token states, and the final action is scored from those tokens after causal decoding.

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

## Newly Tried Or Retested In Stage 11G

- `single_actor`: roles=['actor'], cross_mode=`none`, query=`subtask`, summary=`last_token`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=False, qa_aux=0.0, action_query_tokens=False.
- `role_clones_no_path`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`none`, query=`subtask`, summary=`last_token`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=False, qa_aux=0.0, action_query_tokens=False.
- `private_view_no_path`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`none`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.0, action_query_tokens=False.
- `private_view_cross_causal_mean`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`cross_peer`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.0, action_query_tokens=False.
- `private_view_qa_hard_aux_bias`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`qa_hard_bias`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.35, action_query_tokens=False.
- `private_view_action_tokens_no_path`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`none`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.0, action_query_tokens=True.
- `private_view_qa_token_soft`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`qa_soft_token`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.0, action_query_tokens=True.
- `private_view_qa_token_hard`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`qa_hard_token`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.0, action_query_tokens=True.
- `private_view_qa_token_hard_aux`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`qa_hard_token`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.35, action_query_tokens=True.
- `private_view_qa_token_hard_aux_late`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`qa_hard_token`, query=`subtask`, summary=`causal_mean`, layers=[2], slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.35, action_query_tokens=True.
- `private_view_qa_token_hard_aux_every`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`qa_hard_token`, query=`subtask`, summary=`causal_mean`, layers=[0, 1, 2], slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.35, action_query_tokens=True.
- `full_view_qa_token_hard_aux`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`qa_hard_token`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=False, qa_aux=0.35, action_query_tokens=True.

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
| `full_view_qa_token_hard_aux` | 3 | 250383 | 0.6135 +/- 0.0192 | 0.3333 +/- 0.0575 | 0.5083 | 0.5134 | 0.2448 | 0.5134 | 0.6341 | 0.3648 | 0.0006 |
| `private_view_action_tokens_no_path` | 3 | 173701 | 0.5034 +/- 0.0309 | 0.1250 +/- 0.0255 | 0.4514 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `private_view_cross_causal_mean` | 3 | 250383 | 0.6047 +/- 0.0550 | 0.2188 +/- 0.1112 | 0.3701 | 0.9109 | 0.5742 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `private_view_no_path` | 3 | 173701 | 0.5238 +/- 0.0192 | 0.1094 +/- 0.0128 | 0.3416 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `private_view_qa_hard_aux_bias` | 3 | 250383 | 0.5392 +/- 0.0320 | 0.1094 +/- 0.0383 | 0.3596 | 0.5200 | 0.3336 | 0.5200 | 0.6195 | 0.3780 | 0.0011 |
| `private_view_qa_token_hard` | 3 | 250383 | 0.5076 +/- 0.0395 | 0.1458 +/- 0.0575 | 0.4393 | 1.0019 | 0.2729 | 1.0019 | 0.2209 | 0.1431 | 0.3027 |
| `private_view_qa_token_hard_aux` | 3 | 250383 | 0.5083 +/- 0.0307 | 0.1250 +/- 0.0383 | 0.4410 | 0.5206 | 0.2933 | 0.5206 | 0.6435 | 0.3558 | 0.0003 |
| `private_view_qa_token_hard_aux_every` | 3 | 288724 | 0.5329 +/- 0.0240 | 0.1302 +/- 0.0448 | 0.4052 | 0.5173 | 0.2991 | 0.5173 | 0.6744 | 0.3246 | 0.0005 |
| `private_view_qa_token_hard_aux_late` | 3 | 212042 | 0.5345 +/- 0.0205 | 0.1146 +/- 0.0195 | 0.4741 | 0.5210 | 0.2985 | 0.5210 | 0.6602 | 0.3390 | 0.0003 |
| `private_view_qa_token_soft` | 3 | 250383 | 0.5172 +/- 0.0384 | 0.1250 +/- 0.0338 | 0.4439 | 1.2660 | 0.2413 | 1.2660 | 0.2463 | 0.2382 | 0.2784 |
| `role_clones_no_path` | 3 | 173701 | 0.6756 +/- 0.0144 | 0.3542 +/- 0.0483 | 0.3762 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `single_actor` | 3 | 173701 | 0.6756 +/- 0.0144 | 0.3542 +/- 0.0483 | 0.3762 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Mechanism Ablations On Base Proposed

No mechanism ablations were logged.

## Eval-Time Ablations By Cross Method

| method | ablation | rollout success mean | delta vs method |
|---|---|---:|---:|
| `full_view_qa_token_hard_aux` | `action_answer_shuffle` | 0.3438 | +0.0104 |
| `full_view_qa_token_hard_aux` | `cross_task_kv_shuffle` | 0.3438 | +0.0104 |
| `full_view_qa_token_hard_aux` | `mask_actor_peer_reads` | 0.3021 | -0.0312 |
| `full_view_qa_token_hard_aux` | `remove_path` | 0.3021 | -0.0312 |
| `full_view_qa_token_hard_aux` | `role_kv_shuffle` | 0.3385 | +0.0052 |
| `private_view_cross_causal_mean` | `cross_task_kv_shuffle` | 0.1094 | -0.1094 |
| `private_view_cross_causal_mean` | `mask_actor_peer_reads` | 0.1094 | -0.1094 |
| `private_view_cross_causal_mean` | `remove_path` | 0.1198 | -0.0990 |
| `private_view_cross_causal_mean` | `role_kv_shuffle` | 0.1979 | -0.0208 |
| `private_view_qa_hard_aux_bias` | `action_answer_shuffle` | 0.1406 | +0.0312 |
| `private_view_qa_hard_aux_bias` | `cross_task_kv_shuffle` | 0.1146 | +0.0052 |
| `private_view_qa_hard_aux_bias` | `mask_actor_peer_reads` | 0.1146 | +0.0052 |
| `private_view_qa_hard_aux_bias` | `remove_path` | 0.1146 | +0.0052 |
| `private_view_qa_hard_aux_bias` | `role_kv_shuffle` | 0.1146 | +0.0052 |
| `private_view_qa_token_hard` | `action_answer_shuffle` | 0.1615 | +0.0156 |
| `private_view_qa_token_hard` | `cross_task_kv_shuffle` | 0.1458 | +0.0000 |
| `private_view_qa_token_hard` | `mask_actor_peer_reads` | 0.1562 | +0.0104 |
| `private_view_qa_token_hard` | `remove_path` | 0.1510 | +0.0052 |
| `private_view_qa_token_hard` | `role_kv_shuffle` | 0.1458 | +0.0000 |
| `private_view_qa_token_hard_aux` | `action_answer_shuffle` | 0.1250 | +0.0000 |
| `private_view_qa_token_hard_aux` | `cross_task_kv_shuffle` | 0.1250 | +0.0000 |
| `private_view_qa_token_hard_aux` | `mask_actor_peer_reads` | 0.1042 | -0.0208 |
| `private_view_qa_token_hard_aux` | `remove_path` | 0.1042 | -0.0208 |
| `private_view_qa_token_hard_aux` | `role_kv_shuffle` | 0.1198 | -0.0052 |
| `private_view_qa_token_hard_aux_every` | `action_answer_shuffle` | 0.1406 | +0.0104 |
| `private_view_qa_token_hard_aux_every` | `cross_task_kv_shuffle` | 0.1146 | -0.0156 |
| `private_view_qa_token_hard_aux_every` | `mask_actor_peer_reads` | 0.1146 | -0.0156 |
| `private_view_qa_token_hard_aux_every` | `remove_path` | 0.1094 | -0.0208 |
| `private_view_qa_token_hard_aux_every` | `role_kv_shuffle` | 0.1094 | -0.0208 |
| `private_view_qa_token_hard_aux_late` | `action_answer_shuffle` | 0.1094 | -0.0052 |
| `private_view_qa_token_hard_aux_late` | `cross_task_kv_shuffle` | 0.1146 | +0.0000 |
| `private_view_qa_token_hard_aux_late` | `mask_actor_peer_reads` | 0.0990 | -0.0156 |
| `private_view_qa_token_hard_aux_late` | `remove_path` | 0.0990 | -0.0156 |
| `private_view_qa_token_hard_aux_late` | `role_kv_shuffle` | 0.1146 | +0.0000 |
| `private_view_qa_token_soft` | `action_answer_shuffle` | 0.1250 | +0.0000 |
| `private_view_qa_token_soft` | `cross_task_kv_shuffle` | 0.1250 | +0.0000 |
| `private_view_qa_token_soft` | `mask_actor_peer_reads` | 0.1458 | +0.0208 |
| `private_view_qa_token_soft` | `remove_path` | 0.1406 | +0.0156 |
| `private_view_qa_token_soft` | `role_kv_shuffle` | 0.1250 | +0.0000 |

## Conclusions

- Best method in this screen: `role_clones_no_path` at `0.3542` mean rollout success.
- Single/no-path control ceiling in this screen: `0.3542` mean rollout success.
- `full_view_qa_token_hard_aux`: rollout `0.3333`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.0208`.
- `private_view_action_tokens_no_path`: rollout `0.1250`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.2292`.
- `private_view_cross_causal_mean`: rollout `0.2188`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.1354`.
- `private_view_no_path`: rollout `0.1094`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.2448`.
- `private_view_qa_hard_aux_bias`: rollout `0.1094`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.2448`.
- `private_view_qa_token_hard`: rollout `0.1458`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.2083`.
- `private_view_qa_token_hard_aux`: rollout `0.1250`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.2292`.
- `private_view_qa_token_hard_aux_every`: rollout `0.1302`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.2240`.
- `private_view_qa_token_hard_aux_late`: rollout `0.1146`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.2396`.
- `private_view_qa_token_soft`: rollout `0.1250`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.2292`.
- Stage 11G action-token no-path baseline: `0.1250`. Best action-token Q/A method `full_view_qa_token_hard_aux`: `0.3333` (`+0.2083` vs token no-path).
- Stage 11F direct-bias reference in this screen: `0.1094`.
- Explicit action-query tokens carried useful peer-answer information relative to the private action-token no-path control.

## Next Decision

Retain `full_view_qa_token_hard_aux` only if mechanism ablations are damaging and it remains competitive with the Stage 11F direct-bias relay under the same seed budget.
