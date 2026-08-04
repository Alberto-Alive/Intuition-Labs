# Stage 11H Shared Bus Retest3

## Hypothesis Tested

A partitioned shared residual bus should make cross-role state structurally load-bearing: role-private goal, obstacle, and history evidence is written into fixed subspaces, every role reads the full bus, and the Actor action head receives the final bus as a required input.

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
- Stage 11F private-view Q/A relay variants with complementary goal/map visibility and auxiliary answer supervision.
- Stage 11G explicit action-query token variants for the private-view Q/A relay.

## Newly Tried Or Retested In Stage 11H

- `single_actor`: roles=['actor'], cross_mode=`none`, query=`subtask`, summary=`last_token`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=False, qa_aux=0.0, history_private=False, action_query_tokens=False, action_query_hybrid=False, action_query_decision=False.
- `role_clones_no_path`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`none`, query=`subtask`, summary=`last_token`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=False, qa_aux=0.0, history_private=False, action_query_tokens=False, action_query_hybrid=False, action_query_decision=False.
- `private_view_history_no_path`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`none`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.0, history_private=True, action_query_tokens=False, action_query_hybrid=False, action_query_decision=False.
- `private_view_history_cross_causal_mean`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`cross_peer`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.0, history_private=True, action_query_tokens=False, action_query_hybrid=False, action_query_decision=False.
- `private_view_shared_bus`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`shared_bus`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.0, history_private=True, action_query_tokens=False, action_query_hybrid=False, action_query_decision=False.
- `private_view_shared_bus_every`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`shared_bus`, query=`subtask`, summary=`causal_mean`, layers=[0, 1, 2], slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.0, history_private=True, action_query_tokens=False, action_query_hybrid=False, action_query_decision=False.

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

Same executable gridworld harness as Stage 11A. Train worlds `192`, dev worlds `64`, epochs `16`, batch size `96`, seeds `[121, 122, 123]`.
Hardware `cuda`, CUDA `True`, GPU `NVIDIA GeForce RTX 5070 Ti`.

## Results

| method | seeds | params | action acc mean | rollout success mean | valid action mean | entropy | inject norm | bus write norm | bus read norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `private_view_history_cross_causal_mean` | 3 | 250383 | 0.4762 +/- 0.0233 | 0.1250 +/- 0.0338 | 0.4518 | 0.4871 | 0.5971 | 0.0000 | 0.0000 |
| `private_view_history_no_path` | 3 | 173701 | 0.3789 +/- 0.0120 | 0.0990 +/- 0.0195 | 0.6938 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `private_view_shared_bus` | 3 | 216457 | 0.6510 +/- 0.0359 | 0.3646 +/- 0.0483 | 0.4209 | 0.0000 | 0.7403 | 0.9691 | 0.7403 |
| `private_view_shared_bus_every` | 3 | 237513 | 0.6473 +/- 0.0240 | 0.2812 +/- 0.0128 | 0.4611 | 0.0000 | 0.6825 | 0.8299 | 0.6825 |
| `role_clones_no_path` | 3 | 173701 | 0.6140 +/- 0.0130 | 0.3490 +/- 0.0531 | 0.4522 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `single_actor` | 3 | 173701 | 0.6140 +/- 0.0130 | 0.3490 +/- 0.0531 | 0.4522 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Mechanism Ablations On Base Proposed

No mechanism ablations were logged.

## Eval-Time Ablations By Cross Method

| method | ablation | rollout success mean | delta vs method |
|---|---|---:|---:|
| `private_view_history_cross_causal_mean` | `cross_task_kv_shuffle` | 0.0625 | -0.0625 |
| `private_view_history_cross_causal_mean` | `mask_actor_peer_reads` | 0.0677 | -0.0573 |
| `private_view_history_cross_causal_mean` | `remove_path` | 0.0625 | -0.0625 |
| `private_view_history_cross_causal_mean` | `role_kv_shuffle` | 0.0521 | -0.0729 |
| `private_view_shared_bus` | `cross_task_bus_shuffle` | 0.0469 | -0.3177 |
| `private_view_shared_bus` | `mask_action_bus` | 0.2865 | -0.0781 |
| `private_view_shared_bus` | `mask_actor_peer_reads` | 0.3438 | -0.0208 |
| `private_view_shared_bus` | `remove_path` | 0.0469 | -0.3177 |
| `private_view_shared_bus` | `role_write_zero_actor` | 0.3542 | -0.0104 |
| `private_view_shared_bus` | `role_write_zero_critic` | 0.3698 | +0.0052 |
| `private_view_shared_bus` | `role_write_zero_planner` | 0.0573 | -0.3073 |
| `private_view_shared_bus` | `role_write_zero_reader` | 0.3490 | -0.0156 |
| `private_view_shared_bus_every` | `cross_task_bus_shuffle` | 0.0260 | -0.2552 |
| `private_view_shared_bus_every` | `mask_action_bus` | 0.3281 | +0.0469 |
| `private_view_shared_bus_every` | `mask_actor_peer_reads` | 0.2604 | -0.0208 |
| `private_view_shared_bus_every` | `remove_path` | 0.0469 | -0.2344 |
| `private_view_shared_bus_every` | `role_write_zero_actor` | 0.2760 | -0.0052 |
| `private_view_shared_bus_every` | `role_write_zero_critic` | 0.2344 | -0.0469 |
| `private_view_shared_bus_every` | `role_write_zero_planner` | 0.0781 | -0.2031 |
| `private_view_shared_bus_every` | `role_write_zero_reader` | 0.2760 | -0.0052 |

## Conclusions

- Best method in this screen: `private_view_shared_bus` at `0.3646` mean rollout success.
- Reference cross method `private_view_shared_bus`: `0.3646` mean rollout success.
- Single/no-path control ceiling in this screen: `0.3490` mean rollout success.
- `private_view_history_cross_causal_mean`: rollout `0.1250`, delta vs reference `-0.2396`, delta vs best single/no-path `-0.2240`.
- `private_view_history_no_path`: rollout `0.0990`, delta vs reference `-0.2656`, delta vs best single/no-path `-0.2500`.
- `private_view_shared_bus`: rollout `0.3646`, delta vs reference `+0.0000`, delta vs best single/no-path `+0.0156`.
- `private_view_shared_bus_every`: rollout `0.2812`, delta vs reference `-0.0833`, delta vs best single/no-path `-0.0677`.
- Stage 11H private history no-path baseline: `0.0990`. Best shared-bus method `private_view_shared_bus`: `0.3646` (`+0.2656` vs private history no-path).
- The shared bus carried useful private role information relative to the Actor-blind private no-path control.
- Best-variant eval-time mechanism drops: `[('cross_task_bus_shuffle', -0.3177083333333333), ('mask_action_bus', -0.078125), ('mask_actor_peer_reads', -0.020833333333333315), ('remove_path', -0.3177083333333333), ('role_write_zero_planner', -0.30729166666666663)]`.

## Next Decision

Retain `private_view_shared_bus` for a targeted Stage 11H retest with more seeds and difficulty slices; its eval-time bus ablations were damaging: `[('cross_task_bus_shuffle', -0.3177083333333333), ('mask_action_bus', -0.078125), ('mask_actor_peer_reads', -0.020833333333333315), ('remove_path', -0.3177083333333333), ('role_write_zero_planner', -0.30729166666666663)]`.
