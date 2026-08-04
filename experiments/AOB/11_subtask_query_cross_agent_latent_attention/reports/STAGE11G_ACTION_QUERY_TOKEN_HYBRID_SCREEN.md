# Stage 11G Action-Query Token Hybrid Screen

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

- `single_actor`: roles=['actor'], cross_mode=`none`, query=`subtask`, summary=`last_token`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=False, qa_aux=0.0, action_query_tokens=False, action_query_hybrid=False.
- `role_clones_no_path`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`none`, query=`subtask`, summary=`last_token`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=False, qa_aux=0.0, action_query_tokens=False, action_query_hybrid=False.
- `private_view_no_path`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`none`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.0, action_query_tokens=False, action_query_hybrid=False.
- `private_view_qa_hard_aux_bias`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`qa_hard_bias`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.35, action_query_tokens=False, action_query_hybrid=False.
- `private_view_action_tokens_hybrid_no_path`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`none`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.0, action_query_tokens=True, action_query_hybrid=True.
- `private_view_qa_token_hard_aux_hybrid`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`qa_hard_token`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.35, action_query_tokens=True, action_query_hybrid=True.
- `private_view_qa_token_hard_aux_hybrid_late`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`qa_hard_token`, query=`subtask`, summary=`causal_mean`, layers=[2], slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.35, action_query_tokens=True, action_query_hybrid=True.
- `private_view_qa_token_hard_aux_hybrid_every`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`qa_hard_token`, query=`subtask`, summary=`causal_mean`, layers=[0, 1, 2], slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.35, action_query_tokens=True, action_query_hybrid=True.
- `full_view_qa_token_hard_aux_hybrid`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`qa_hard_token`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=False, qa_aux=0.35, action_query_tokens=True, action_query_hybrid=True.

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
| `full_view_qa_token_hard_aux_hybrid` | 3 | 250383 | 0.6629 +/- 0.0152 | 0.3698 +/- 0.0266 | 0.4203 | 0.5035 | 0.4476 | 0.5035 | 0.6029 | 0.3962 | 0.0005 |
| `private_view_action_tokens_hybrid_no_path` | 3 | 173701 | 0.5224 +/- 0.0204 | 0.1198 +/- 0.0074 | 0.3823 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `private_view_no_path` | 3 | 173701 | 0.5238 +/- 0.0192 | 0.1094 +/- 0.0128 | 0.3416 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `private_view_qa_hard_aux_bias` | 3 | 250383 | 0.5392 +/- 0.0320 | 0.1094 +/- 0.0383 | 0.3596 | 0.5200 | 0.3336 | 0.5200 | 0.6195 | 0.3780 | 0.0011 |
| `private_view_qa_token_hard_aux_hybrid` | 3 | 250383 | 0.5217 +/- 0.0248 | 0.0990 +/- 0.0074 | 0.3722 | 0.5108 | 0.4830 | 0.5108 | 0.6299 | 0.3690 | 0.0006 |
| `private_view_qa_token_hard_aux_hybrid_every` | 3 | 288724 | 0.5293 +/- 0.0283 | 0.0833 +/- 0.0147 | 0.3291 | 0.5264 | 0.5213 | 0.5264 | 0.6294 | 0.3694 | 0.0007 |
| `private_view_qa_token_hard_aux_hybrid_late` | 3 | 212042 | 0.5356 +/- 0.0322 | 0.0833 +/- 0.0390 | 0.3593 | 0.5062 | 0.6059 | 0.5062 | 0.6347 | 0.3643 | 0.0005 |
| `role_clones_no_path` | 3 | 173701 | 0.6756 +/- 0.0144 | 0.3542 +/- 0.0483 | 0.3762 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `single_actor` | 3 | 173701 | 0.6756 +/- 0.0144 | 0.3542 +/- 0.0483 | 0.3762 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Mechanism Ablations On Base Proposed

No mechanism ablations were logged.

## Eval-Time Ablations By Cross Method

| method | ablation | rollout success mean | delta vs method |
|---|---|---:|---:|
| `full_view_qa_token_hard_aux_hybrid` | `action_answer_shuffle` | 0.3698 | +0.0000 |
| `full_view_qa_token_hard_aux_hybrid` | `cross_task_kv_shuffle` | 0.3542 | -0.0156 |
| `full_view_qa_token_hard_aux_hybrid` | `mask_actor_peer_reads` | 0.3542 | -0.0156 |
| `full_view_qa_token_hard_aux_hybrid` | `remove_path` | 0.3542 | -0.0156 |
| `full_view_qa_token_hard_aux_hybrid` | `role_kv_shuffle` | 0.3698 | +0.0000 |
| `private_view_qa_hard_aux_bias` | `action_answer_shuffle` | 0.1406 | +0.0312 |
| `private_view_qa_hard_aux_bias` | `cross_task_kv_shuffle` | 0.1146 | +0.0052 |
| `private_view_qa_hard_aux_bias` | `mask_actor_peer_reads` | 0.1146 | +0.0052 |
| `private_view_qa_hard_aux_bias` | `remove_path` | 0.1146 | +0.0052 |
| `private_view_qa_hard_aux_bias` | `role_kv_shuffle` | 0.1146 | +0.0052 |
| `private_view_qa_token_hard_aux_hybrid` | `action_answer_shuffle` | 0.0885 | -0.0104 |
| `private_view_qa_token_hard_aux_hybrid` | `cross_task_kv_shuffle` | 0.0990 | +0.0000 |
| `private_view_qa_token_hard_aux_hybrid` | `mask_actor_peer_reads` | 0.0990 | +0.0000 |
| `private_view_qa_token_hard_aux_hybrid` | `remove_path` | 0.0990 | +0.0000 |
| `private_view_qa_token_hard_aux_hybrid` | `role_kv_shuffle` | 0.0990 | +0.0000 |
| `private_view_qa_token_hard_aux_hybrid_every` | `action_answer_shuffle` | 0.0833 | +0.0000 |
| `private_view_qa_token_hard_aux_hybrid_every` | `cross_task_kv_shuffle` | 0.0833 | +0.0000 |
| `private_view_qa_token_hard_aux_hybrid_every` | `mask_actor_peer_reads` | 0.0990 | +0.0156 |
| `private_view_qa_token_hard_aux_hybrid_every` | `remove_path` | 0.0990 | +0.0156 |
| `private_view_qa_token_hard_aux_hybrid_every` | `role_kv_shuffle` | 0.0885 | +0.0052 |
| `private_view_qa_token_hard_aux_hybrid_late` | `action_answer_shuffle` | 0.0938 | +0.0104 |
| `private_view_qa_token_hard_aux_hybrid_late` | `cross_task_kv_shuffle` | 0.0833 | +0.0000 |
| `private_view_qa_token_hard_aux_hybrid_late` | `mask_actor_peer_reads` | 0.1042 | +0.0208 |
| `private_view_qa_token_hard_aux_hybrid_late` | `remove_path` | 0.1042 | +0.0208 |
| `private_view_qa_token_hard_aux_hybrid_late` | `role_kv_shuffle` | 0.0833 | +0.0000 |

## Conclusions

- Best method in this screen: `full_view_qa_token_hard_aux_hybrid` at `0.3698` mean rollout success.
- Single/no-path control ceiling in this screen: `0.3542` mean rollout success.
- `full_view_qa_token_hard_aux_hybrid`: rollout `0.3698`, delta vs reference `+0.0000`, delta vs best single/no-path `+0.0156`.
- `private_view_action_tokens_hybrid_no_path`: rollout `0.1198`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.2344`.
- `private_view_no_path`: rollout `0.1094`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.2448`.
- `private_view_qa_hard_aux_bias`: rollout `0.1094`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.2448`.
- `private_view_qa_token_hard_aux_hybrid`: rollout `0.0990`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.2552`.
- `private_view_qa_token_hard_aux_hybrid_every`: rollout `0.0833`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.2708`.
- `private_view_qa_token_hard_aux_hybrid_late`: rollout `0.0833`, delta vs reference `+0.0000`, delta vs best single/no-path `-0.2708`.
- Stage 11G action-token no-path baseline: `0.0000`. Best action-token Q/A method `full_view_qa_token_hard_aux_hybrid`: `0.3698` (`+0.3698` vs token no-path).
- Stage 11F direct-bias reference in this screen: `0.1094`.
- Explicit action-query tokens carried useful peer-answer information relative to the private action-token no-path control.

## Next Decision

Retain `full_view_qa_token_hard_aux_hybrid` only if mechanism ablations are damaging and it remains competitive with the Stage 11F direct-bias relay under the same seed budget.
