# Stage 11G Action-Query Token Tiny Smoke

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

- `private_view_action_tokens_no_path`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`none`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.0, action_query_tokens=True.
- `private_view_qa_token_hard_aux`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`qa_hard_token`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.35, action_query_tokens=True.

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

Same executable gridworld harness as Stage 11A. Train worlds `24`, dev worlds `10`, epochs `2`, batch size `24`, seeds `[101]`.
Hardware `cuda`, CUDA `True`, GPU `NVIDIA GeForce RTX 5070 Ti`.

## Results

| method | seeds | params | action acc mean | rollout success mean | valid action mean | entropy | inject norm | qa entropy | yes | no | irrelevant |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `private_view_action_tokens_no_path` | 1 | 173701 | 0.1900 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.1000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `private_view_qa_token_hard_aux` | 1 | 250383 | 0.1900 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.1000 | 0.5868 | 0.2703 | 0.5868 | 0.6675 | 0.3319 | 0.0002 |

## Mechanism Ablations On Base Proposed

No mechanism ablations were logged.

## Eval-Time Ablations By Cross Method

| method | ablation | rollout success mean | delta vs method |
|---|---|---:|---:|
| `private_view_qa_token_hard_aux` | `action_answer_shuffle` | 0.0000 | +0.0000 |
| `private_view_qa_token_hard_aux` | `cross_task_kv_shuffle` | 0.0000 | +0.0000 |
| `private_view_qa_token_hard_aux` | `mask_actor_peer_reads` | 0.0000 | +0.0000 |
| `private_view_qa_token_hard_aux` | `remove_path` | 0.0000 | +0.0000 |
| `private_view_qa_token_hard_aux` | `role_kv_shuffle` | 0.0000 | +0.0000 |

## Conclusions

- Best method in this screen: `private_view_action_tokens_no_path` at `0.0000` mean rollout success.
- Single/no-path control ceiling in this screen: `0.0000` mean rollout success.
- `private_view_action_tokens_no_path`: rollout `0.0000`, delta vs reference `+0.0000`, delta vs best single/no-path `+0.0000`.
- `private_view_qa_token_hard_aux`: rollout `0.0000`, delta vs reference `+0.0000`, delta vs best single/no-path `+0.0000`.
- Stage 11G action-token no-path baseline: `0.0000`. Best action-token Q/A method `private_view_qa_token_hard_aux`: `0.0000` (`+0.0000` vs token no-path).
- Explicit action-query tokens did not beat their private no-path control in this screen.

## Next Decision

Do not escalate the action-token relay yet; inspect whether the query tokens are undertrained, too late in the causal stream, or need an explicit per-action auxiliary loss.
