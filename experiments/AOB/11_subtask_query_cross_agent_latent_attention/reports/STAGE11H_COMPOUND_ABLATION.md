# Stage 11H Compound Ablation

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

- `private_view_shared_bus`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`shared_bus`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.0, history_private=True, action_query_tokens=False, action_query_hybrid=False, action_query_decision=False.

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
| `private_view_shared_bus` | 3 | 216457 | 0.6510 +/- 0.0359 | 0.3646 +/- 0.0483 | 0.4209 | 0.0000 | 0.7403 | 0.9691 | 0.7403 |

## Mechanism Ablations On Base Proposed

No mechanism ablations were logged.

## Eval-Time Ablations By Cross Method

| method | ablation | rollout success mean | delta vs method |
|---|---|---:|---:|
| `private_view_shared_bus` | `ablation_compound_planner_zero_actor_mask` | 0.0781 | -0.2865 |
| `private_view_shared_bus` | `cross_task_bus_shuffle` | 0.0469 | -0.3177 |
| `private_view_shared_bus` | `mask_action_bus` | 0.2865 | -0.0781 |
| `private_view_shared_bus` | `mask_actor_peer_reads` | 0.3438 | -0.0208 |
| `private_view_shared_bus` | `remove_path` | 0.0469 | -0.3177 |
| `private_view_shared_bus` | `role_write_zero_actor` | 0.3542 | -0.0104 |
| `private_view_shared_bus` | `role_write_zero_critic` | 0.3698 | +0.0052 |
| `private_view_shared_bus` | `role_write_zero_planner` | 0.0573 | -0.3073 |
| `private_view_shared_bus` | `role_write_zero_reader` | 0.3490 | -0.0156 |

## Compound Ablation Read

- Full `private_view_shared_bus`: `0.3646` mean rollout success.
- Individual `role_write_zero_planner`: `0.0573` (`-0.3073` vs full).
- Individual `mask_actor_peer_reads`: `0.3438` (`-0.0208` vs full).
- Compound `ablation_compound_planner_zero_actor_mask`: `0.0781` (`-0.2865` vs full).
- The compound ablation is not additive. It is slightly less damaging than zeroing the Planner writer alone (`+0.0208` rollout), so the Actor stream bus read is not an independent necessary path in this trained circuit. The load-bearing path is dominated by Planner writes into the bus and the final action-head bus input; the Actor's recurrent bus read looks redundant or mildly compensatory/noisy once Planner evidence is removed.

## Conclusions

- Best method in this screen: `private_view_shared_bus` at `0.3646` mean rollout success.
- Reference cross method `private_view_shared_bus`: `0.3646` mean rollout success.
- Single/no-path control ceiling in this screen: `0.0000` mean rollout success.
- `private_view_shared_bus`: rollout `0.3646`, delta vs reference `+0.0000`, delta vs best single/no-path `+0.3646`.
- Stage 11H private history no-path baseline: `0.0000`. Best shared-bus method `private_view_shared_bus`: `0.3646` (`+0.3646` vs private history no-path).
- The shared bus carried useful private role information relative to the Actor-blind private no-path control.
- Best-variant eval-time mechanism drops: `[('ablation_compound_planner_zero_actor_mask', -0.2864583333333333), ('cross_task_bus_shuffle', -0.3177083333333333), ('mask_action_bus', -0.078125), ('mask_actor_peer_reads', -0.020833333333333315), ('remove_path', -0.3177083333333333), ('role_write_zero_planner', -0.30729166666666663)]`.

## Next Decision

Retain `private_view_shared_bus` for a targeted Stage 11H retest with more seeds and difficulty slices; its eval-time bus ablations were damaging: `[('ablation_compound_planner_zero_actor_mask', -0.2864583333333333), ('cross_task_bus_shuffle', -0.3177083333333333), ('mask_action_bus', -0.078125), ('mask_actor_peer_reads', -0.020833333333333315), ('remove_path', -0.3177083333333333), ('role_write_zero_planner', -0.30729166666666663)]`.
