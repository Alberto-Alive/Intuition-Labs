# Stage 11G Action-Query Token Decision Retest 16 Epochs

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

- `single_actor`: roles=['actor'], cross_mode=`none`, query=`subtask`, summary=`last_token`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=False, qa_aux=0.0, action_query_tokens=False, action_query_hybrid=False, action_query_decision=False.
- `role_clones_no_path`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`none`, query=`subtask`, summary=`last_token`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=False, qa_aux=0.0, action_query_tokens=False, action_query_hybrid=False, action_query_decision=False.
- `private_view_no_path`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`none`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.0, action_query_tokens=False, action_query_hybrid=False, action_query_decision=False.
- `private_view_qa_hard_aux_bias`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`qa_hard_bias`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.35, action_query_tokens=False, action_query_hybrid=False, action_query_decision=False.
- `private_view_action_tokens_decision_no_path`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`none`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.0, action_query_tokens=True, action_query_hybrid=False, action_query_decision=True.
- `private_view_qa_token_hard_aux_decision`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`qa_hard_token`, query=`subtask`, summary=`causal_mean`, layers=[0, 1], slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.35, action_query_tokens=True, action_query_hybrid=False, action_query_decision=True.
- `full_view_qa_token_hard_aux_decision`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`qa_hard_token`, query=`subtask`, summary=`causal_mean`, layers=[0, 1], slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=False, qa_aux=0.35, action_query_tokens=True, action_query_hybrid=False, action_query_decision=True.

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
| `full_view_qa_token_hard_aux_decision` | 3 | 250383 | 0.6416 +/- 0.0046 | 0.3594 +/- 0.0797 | 0.4508 | 0.4710 | 0.8647 | 0.4710 | 0.5764 | 0.4225 | 0.0006 |
| `private_view_action_tokens_decision_no_path` | 3 | 173701 | 0.5116 +/- 0.0219 | 0.1250 +/- 0.0221 | 0.4576 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `private_view_no_path` | 3 | 173701 | 0.5174 +/- 0.0309 | 0.1146 +/- 0.0266 | 0.4288 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `private_view_qa_hard_aux_bias` | 3 | 250383 | 0.5979 +/- 0.0361 | 0.2240 +/- 0.0390 | 0.4267 | 0.3570 | 0.2820 | 0.3570 | 0.6247 | 0.3742 | 0.0006 |
| `private_view_qa_token_hard_aux_decision` | 3 | 250383 | 0.5275 +/- 0.0159 | 0.1510 +/- 0.0074 | 0.4205 | 0.4335 | 0.7712 | 0.4335 | 0.6208 | 0.3781 | 0.0007 |
| `role_clones_no_path` | 3 | 173701 | 0.6389 +/- 0.0149 | 0.3542 +/- 0.0410 | 0.4433 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `single_actor` | 3 | 173701 | 0.6389 +/- 0.0149 | 0.3542 +/- 0.0410 | 0.4433 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Mechanism Ablations On Base Proposed

No mechanism ablations were logged.

## Eval-Time Ablations By Cross Method

| method | ablation | rollout success mean | delta vs method |
|---|---|---:|---:|
| `full_view_qa_token_hard_aux_decision` | `action_answer_shuffle` | 0.3594 | +0.0000 |
| `full_view_qa_token_hard_aux_decision` | `cross_task_kv_shuffle` | 0.3594 | +0.0000 |
| `full_view_qa_token_hard_aux_decision` | `mask_actor_peer_reads` | 0.3438 | -0.0156 |
| `full_view_qa_token_hard_aux_decision` | `remove_path` | 0.3542 | -0.0052 |
| `full_view_qa_token_hard_aux_decision` | `role_kv_shuffle` | 0.3594 | +0.0000 |
| `private_view_qa_hard_aux_bias` | `action_answer_shuffle` | 0.0208 | -0.2031 |
| `private_view_qa_hard_aux_bias` | `cross_task_kv_shuffle` | 0.0833 | -0.1406 |
| `private_view_qa_hard_aux_bias` | `mask_actor_peer_reads` | 0.0625 | -0.1615 |
| `private_view_qa_hard_aux_bias` | `remove_path` | 0.0677 | -0.1562 |
| `private_view_qa_hard_aux_bias` | `role_kv_shuffle` | 0.2083 | -0.0156 |
| `private_view_qa_token_hard_aux_decision` | `action_answer_shuffle` | 0.1302 | -0.0208 |
| `private_view_qa_token_hard_aux_decision` | `cross_task_kv_shuffle` | 0.1406 | -0.0104 |
| `private_view_qa_token_hard_aux_decision` | `mask_actor_peer_reads` | 0.1562 | +0.0052 |
| `private_view_qa_token_hard_aux_decision` | `remove_path` | 0.1615 | +0.0104 |
| `private_view_qa_token_hard_aux_decision` | `role_kv_shuffle` | 0.1510 | +0.0000 |

## Conclusions

- Best method in this screen: `full_view_qa_token_hard_aux_decision` at `0.3594` mean rollout success.
- Single/no-path control ceiling in this screen: `0.3542` mean rollout success.
- `full_view_qa_token_hard_aux_decision`: rollout `0.3594`, delta vs reference `+0.0000`, delta vs best single/no-path `+0.0052`.
- `private_view_action_tokens_decision_no_path`: rollout `0.1250`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.2292`.
- `private_view_no_path`: rollout `0.1146`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.2396`.
- `private_view_qa_hard_aux_bias`: rollout `0.2240`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.1302`.
- `private_view_qa_token_hard_aux_decision`: rollout `0.1510`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.2031`.
- Stage 11G action-token no-path baseline: `0.0000`. Best action-token Q/A method `full_view_qa_token_hard_aux_decision`: `0.3594` (`+0.3594` vs token no-path).
- Stage 11F direct-bias reference in this screen: `0.2240`.
- Explicit action-query tokens carried useful peer-answer information relative to the private action-token no-path control.

## Next Decision

Retain `full_view_qa_token_hard_aux_decision` only if mechanism ablations are damaging and it remains competitive with the Stage 11F direct-bias relay under the same seed budget.
