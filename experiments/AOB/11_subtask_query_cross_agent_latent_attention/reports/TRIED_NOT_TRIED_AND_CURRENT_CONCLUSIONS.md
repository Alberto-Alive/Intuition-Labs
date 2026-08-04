# Experiment 11 Tried / Not Tried Inventory

## Tried

- `single_actor`: one Actor stream, no cross-agent path.
- `role_clones_no_path`: Planner/Reader/Critic/Actor role streams, no latent cross-agent route.
- `capacity_control_self_read`: four role streams with the cross block restricted to self-state reads.
- `cross_agent_latent_attention`: four role streams, middle-plus-late cross-agent blocks, last-token peer summaries, subtask-only query.
- `shared_query_control`: same as base proposed, but with a learned shared query instead of role/subtask-specific Q.
- Eval-time ablations on the base proposed method: remove path, role K/V shuffle, cross-task K/V shuffle, Actor peer-read mask.
- `cross_late_only`: late-only cross-agent block insertion.
- `cross_every_layer`: cross-agent block after every decoder layer.
- `cross_subtask_state_query`: Q derived from subtask plus receiving role state.
- `cross_causal_mean_summary`: peer K/V from causal-prefix mean summaries instead of last-token summaries.
- `cross_two_role_actor_critic`: two role streams, Critic and Actor.
- Stage 11C longer retest of `cross_causal_mean_summary` against selected controls.
- Eval-time ablations on `cross_causal_mean_summary`: remove path, role K/V shuffle, cross-task K/V shuffle, Actor peer-read mask.
- `slot_self_read_m2`: two-slot causal-mean exporter with self-only reads as a slot capacity/control path.
- `slot_bottleneck_m2`: two exported slots per role, cross-peer latent reads.
- `slot_bottleneck_m2_topk2`: two exported slots per role with sparse top-k2 peer-slot reads.
- `slot_bottleneck_m2_dropout`: two exported slots per role with source-role dropout during training.
- `slot_bottleneck_m4_topk2`: four exported slots per role with sparse top-k2 peer-slot reads.
- Eval-time ablations on slot bottleneck variants: remove path, role K/V shuffle, cross-task K/V shuffle, Actor peer-read mask.
- `role_lora_no_path`: learned role-gated shared-basis LoRA adapters without a cross-agent path.
- `role_lora_self_read`: learned role-gated shared-basis LoRA adapters with self-read capacity control.
- `role_lora_cross_causal_mean`: learned role-gated shared-basis LoRA adapters plus causal-mean peer reads.
- `role_lora_cross_slots_m2_topk2`: learned role-gated shared-basis LoRA adapters plus slot/top-k peer reads.
- `role_lora_hard_no_path`: fixed-role shared-basis LoRA adapters without a cross-agent path.
- `role_lora_hard_self_read`: fixed-role shared-basis LoRA adapters with self-read capacity control.
- `role_lora_hard_cross_causal_mean`: fixed-role shared-basis LoRA adapters plus causal-mean peer reads.
- Eval-time ablations on LoRA cross variants: remove path, role K/V shuffle, cross-task K/V shuffle, Actor peer-read mask.
- Stage 11F private-view Q/A relay variants with complementary goal/map masking and auxiliary answer supervision.
- Stage 11G explicit action-query token, hybrid-token, and decision-token private-view Q/A variants.
- `private_view_history_no_path`: private goal/map/history masking without a cross-role path.
- `private_view_history_cross_causal_mean`: the old continuous causal-mean peer-read path under the stricter history-private masking task.
- `private_view_shared_bus`: partitioned cross-role residual bus with final Actor action head forced to consume the bus.
- `private_view_shared_bus_every`: shared-bus insertion after every decoder layer.
- Eval-time ablations on shared-bus variants: remove path, mask action bus, cross-task bus shuffle, mask Actor bus read, and zero each role writer.

## Not Tried

- Pretrained causal decoder checkpoint.
- Full token-level cross-role attention instead of role summaries.
- Learned or external subtask decomposition beyond fixed role/subtask tokens.
- Learned residual gate schedules or near-zero initialized gates.
- Retrained mechanism controls for the causal-mean variant.
- Shared-query and no-path controls matched specifically to causal-mean summaries.
- Actor-only source routing or fixed sparse role-routing graphs.
- Larger LoRA rank, alpha, and gate-temperature schedules.
- More role-count schedules beyond four roles and Actor/Critic two-role.
- Larger model/data/epoch scaling and difficulty-sliced validation.
- Held-out final evaluation.

## Current Conclusions

- The implementation path works: causal shared-weight role streams, cross-agent Q/K/V reads, residual injection, rollout evaluation, and logging are functional.
- The original last-token subtask-query design did not beat no-path controls in Stage 11A/11B.
- Late-only insertion, every-layer insertion, subtask+state Q, and two-role Actor/Critic did not rescue the result.
- Causal-prefix mean peer summaries are the only weakly promising variant so far.
- Stage 11C: `cross_causal_mean_summary` reached `0.3698` mean rollout success versus `0.3490` for single/no-path controls, but this is small and high-variance.
- Mechanism evidence is not established: cross-task K/V shuffle hurt the causal-mean variant, but role K/V shuffle and Actor peer-read masking did not.
- Stage 11D: slot bottlenecks, sparse top-k reads, and role dropout did not force useful complementary specialization. The best slot variant, `slot_bottleneck_m2_topk2`, reached `0.3229` mean rollout success versus `0.3698` for single/no-path controls and `0.3646` for `cross_causal_mean_summary`.
- Stage 11D slot variants sometimes improved supervised action accuracy, but rollout and ablations moved in the wrong direction: removing the path or masking Actor peer reads did not damage slot variants, and often improved them.
- Stage 11E learned role-gated LoRA did not strongly break transverse symmetry: gate entropy stayed near `log(4)` and the best LoRA cross route remained below single/no-path controls.
- Stage 11E hard fixed-role LoRA did break transverse symmetry, but it still did not make peer reads beneficial. `role_lora_hard_cross_causal_mean` reached `0.3542` versus `0.3646` for its hard no-path control and `0.3698` for single/no-path controls.
- Stage 11E hard self-read reached `0.3802`, a small side signal for adapter/capacity specialization, but it is not evidence for cross-agent latent collaboration.
- Stage 11F private-view Q/A created real inter-agent reliance relative to private no-path, but the best branch used auxiliary answer supervision and direct action-logit bias.
- Stage 11G action-query tokens did not preserve the Stage 11F mechanism; removing or masking the path was weak or backwards.
- Stage 11H shared bus is now the strongest architecture-faithful development lead. In the 3-seed 16-epoch retest, `private_view_shared_bus` reached `0.3646` rollout success versus `0.0990` for `private_view_history_no_path`, `0.1250` for `private_view_history_cross_causal_mean`, and `0.3490` for full-view single/role-clone controls.
- Stage 11H mechanism evidence is materially stronger for the global bus path and Planner/goal subspace than for every role subspace: remove path, cross-task bus shuffle, and zero Planner writer were strongly damaging; Reader/Critic writer-zero controls were weak or non-damaging in the default bus retest.
- No final Experiment 11 claim is supported yet.

## Next

Retain `private_view_shared_bus` for a larger Stage 11H validation with more seeds, difficulty slices, and a task variant that makes obstacle and history subspaces necessary on most decisions. Do not spend final-evaluation compute until all intended writer ablations damage performance under repeated seeds.
