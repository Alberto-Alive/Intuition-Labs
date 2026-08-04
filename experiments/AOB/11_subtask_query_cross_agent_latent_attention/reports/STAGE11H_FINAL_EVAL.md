# Stage 11H Final Eval

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
- `private_view_history_no_path`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`none`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=True, qa_aux=0.0, history_private=True, action_query_tokens=False, action_query_hybrid=False, action_query_decision=False.
- `single_actor`: roles=['actor'], cross_mode=`none`, query=`subtask`, summary=`last_token`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=False, qa_aux=0.0, history_private=False, action_query_tokens=False, action_query_hybrid=False, action_query_decision=False.
- `role_clones_no_path`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`none`, query=`subtask`, summary=`last_token`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=0, lora_bases=0, lora_alpha=1.0, lora_gate=softmax, private_view=False, qa_aux=0.0, history_private=False, action_query_tokens=False, action_query_hybrid=False, action_query_decision=False.

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

## Task And Budget

Same executable gridworld harness as Stage 11A. Train worlds `192`, dev worlds `64`, epochs `16`, batch size `96`, seeds `[201, 202, 203, 204, 205]`.
Hardware `cuda`, CUDA `True`, GPU `NVIDIA GeForce RTX 5070 Ti`.

## Results

| method | seeds | params | action acc mean | rollout success mean | valid action mean | entropy | inject norm | bus write norm | bus read norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `private_view_history_no_path` | 5 | 173701 | 0.4150 +/- 0.0177 | 0.0688 +/- 0.0390 | 0.7428 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `private_view_shared_bus` | 5 | 216457 | 0.6314 +/- 0.0245 | 0.2906 +/- 0.0675 | 0.4221 | 0.0000 | 0.7368 | 0.9624 | 0.7368 |
| `role_clones_no_path` | 5 | 173701 | 0.6164 +/- 0.0320 | 0.2969 +/- 0.0328 | 0.4653 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `single_actor` | 5 | 173701 | 0.6161 +/- 0.0316 | 0.2969 +/- 0.0328 | 0.4653 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Mechanism Ablations On Base Proposed

No mechanism ablations were logged.

## Eval-Time Ablations By Cross Method

| method | ablation | rollout success mean | delta vs method |
|---|---|---:|---:|
| `private_view_shared_bus` | `ablation_compound_planner_zero_actor_mask` | 0.0656 | -0.2250 |
| `private_view_shared_bus` | `cross_task_bus_shuffle` | 0.0406 | -0.2500 |
| `private_view_shared_bus` | `mask_action_bus` | 0.2313 | -0.0594 |
| `private_view_shared_bus` | `mask_actor_peer_reads` | 0.3000 | +0.0094 |
| `private_view_shared_bus` | `remove_path` | 0.0344 | -0.2563 |
| `private_view_shared_bus` | `role_write_zero_actor` | 0.3094 | +0.0187 |
| `private_view_shared_bus` | `role_write_zero_critic` | 0.3000 | +0.0094 |
| `private_view_shared_bus` | `role_write_zero_planner` | 0.0625 | -0.2281 |
| `private_view_shared_bus` | `role_write_zero_reader` | 0.2656 | -0.0250 |

## Preregistered Raw Rollout Values

| method | seeds 201-205 rollout | mean | std |
|---|---|---:|---:|
| `private_view_shared_bus` | [0.328125, 0.281250, 0.390625, 0.265625, 0.187500] | 0.2906 | 0.0675 |
| `private_view_history_no_path` | [0.031250, 0.140625, 0.078125, 0.046875, 0.046875] | 0.0688 | 0.0390 |
| `single_actor` | [0.312500, 0.265625, 0.328125, 0.250000, 0.328125] | 0.2969 | 0.0328 |
| `role_clones_no_path` | [0.312500, 0.265625, 0.328125, 0.250000, 0.328125] | 0.2969 | 0.0328 |

## Preregistered Ablation Verdict

Preregistered ablations on `private_view_shared_bus`:

| ablation | rollout mean | delta vs full |
|---|---:|---:|
| `remove_path` | 0.0344 | -0.2563 |
| `cross_task_bus_shuffle` | 0.0406 | -0.2500 |
| `role_write_zero_planner` | 0.0625 | -0.2281 |
| `ablation_compound_planner_zero_actor_mask` | 0.0656 | -0.2250 |

The canonical runner also logged non-preregistered shared-bus diagnostics (`mask_action_bus`, `mask_actor_peer_reads`, and zeroing Actor/Critic/Reader writers). Those diagnostics are reported above for transparency but are not used for preregistered claim decisions.

## Preregistered Claim Verdict

Primary performance claim: **not supported**.

- `private_view_shared_bus` beat `private_view_history_no_path` by `+0.2219` mean rollout success (`0.2906` vs `0.0688`).
- `private_view_shared_bus` did not beat full-view `single_actor`; it trailed by `-0.0063` mean rollout success (`0.2906` vs `0.2969`).
- `private_view_shared_bus` also trailed full-view `role_clones_no_path` by `-0.0063` (`0.2906` vs `0.2969`).

Secondary mechanism claim: **supported**.

- `remove_path` dropped rollout by `0.2563`, exceeding the preregistered `>0.10` threshold.
- `role_write_zero_planner` dropped rollout by `0.2281`, exceeding the preregistered `>0.10` threshold.
- The other preregistered diagnostics were also damaging: `cross_task_bus_shuffle` `-0.2500`, compound Planner-zero/Actor-mask `-0.2250`.

## Conclusions

The final evaluation supports a strong mechanism/dependence claim for the Stage 11H shared bus under the constructed private-role task: the bus path is load-bearing, and the Planner write subspace is causally important.

The final evaluation does not support the preregistered primary performance claim because the shared bus did not outperform the full-view `single_actor` mean rollout success. The robust performance result is recovery versus the private no-path control, not superiority over the full-view single-agent baseline.

Report this outcome as-is. Do not rerun or retune this final evaluation.
