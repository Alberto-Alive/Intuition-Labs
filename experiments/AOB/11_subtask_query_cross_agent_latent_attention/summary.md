# Subtask-Query Cross-Agent Latent Attention

Status: `stage11g_action_token_negative_stage11f_still_active`

Track: `mainline`

Source: new causal-agent architecture proposal after the controlled candidate-selection line through Experiment 10.

## Research Question

Does subtask-query cross-agent latent attention inside shared-weight GPT-style role streams improve real multi-step agent behavior beyond a single causal agent and role-cloned causal agents without latent cross-agent attention?

## Architecture Claim

The proposed architectural change is narrow:

- Use a pretrained causal decoder transformer as the agent backbone.
- Clone that backbone into shared-weight role streams for the same task history.
- Give each receiving role a subtask-conditioned query `Q`.
- Build keys `K` and values `V` from peer role-stream activations.
- Use cross-agent attention weights from `QK^T` to import weighted peer latent content from `V`.
- Inject the imported latent update back into the receiving stream before it emits the next action or token.

## Claim Boundary

This experiment may claim agentic evidence only if the new latent cross-agent attention mechanism improves multi-step action success over compute-matched baselines under pre-registered controls, held-out final evaluation, and mechanism ablations.

It must not claim:

- general agency from a candidate-ranking benchmark,
- collaboration from attention heatmaps alone,
- complementarity from role labels alone,
- gains against an underpowered single-agent or cloned-role baseline,
- publishable evidence from a single seed, tuned final set, or unreported failed variants.

## Research Assets

- [Experiment 11 research spec](reports/EXPERIMENT11_RESEARCH_SPEC.md)
- [Stage 11A agentic gridworld smoke and cheap screen](reports/STAGE11A_AGENTIC_GRIDWORLD_SMOKE_SCREEN.md)
- [Stage 11B architecture variant screen](reports/STAGE11B_ARCHITECTURE_VARIANT_SCREEN.md)
- [Stage 11C best variant retest](reports/STAGE11C_BEST_VARIANT_RETEST.md)
- [Stage 11D complementary-specialization screen](reports/STAGE11D_COMPLEMENTARY_SPECIALIZATION_SCREEN.md)
- [Stage 11D slot tiny smoke](reports/STAGE11D_SLOT_TINY_SMOKE.md)
- [Stage 11E role-gated LoRA screen](reports/STAGE11E_ROLE_GATED_LORA_SCREEN.md)
- [Stage 11E hard role LoRA screen](reports/STAGE11E_HARD_ROLE_LORA_SCREEN.md)
- [Stage 11F private-view Q/A combined 5-seed read](reports/STAGE11F_PRIVATE_VIEW_QA_COMBINED_5SEED.md)
- [Stage 11G action-query token combined read](reports/STAGE11G_ACTION_QUERY_TOKEN_COMBINED.md)
- [Stage 11H shared-bus 3-seed retest](reports/STAGE11H_SHARED_BUS_RETEST3.md)
- [Tried/not-tried inventory and current conclusions](reports/TRIED_NOT_TRIED_AND_CURRENT_CONCLUSIONS.md)
- Implementation: `code/experiment11_gridworld_latent_attention.py`

## Current Empirical Status

Stage 11A implemented the first minimal architecture-faithful harness in this folder: shared-weight causal role streams, subtask-query cross-agent latent attention over peer role K/V summaries, actor-only action emission, an executable gridworld rollout environment, baselines, eval-time mechanism ablations, and machine-readable logs.

The smoke test passed on CUDA: shapes, gradients through the cross-agent block, rollout, causality audit, and logging were live.

The cheap three-seed development screen was negative for the proposed route. Mean dev rollout success:

- `Single Actor`: `0.3698`
- `Role Clones, No Latent Cross-Agent Path`: `0.3698`
- `Capacity Control, Self Read`: `0.2969`
- `Proposed Cross-Agent Latent Attention`: `0.3125`
- `Shared-Query Control`: `0.3073`

Mechanism smoke ablations on the proposed model did not materially reduce rollout success: role K/V shuffle remained `0.3125`, cross-task K/V shuffle was `0.3385`, Actor peer-read masking remained `0.3125`, and removing the path was `0.3229`.

Stage 11B then tested untried architecture-faithful axes: late-only insertion, every-layer insertion, subtask+state query, causal-mean peer summaries, and a two-role Actor/Critic setup. At the same 8-epoch budget, no cross-agent variant beat the single/no-path controls. The only useful direction was `cross_causal_mean_summary`, which improved over the base proposed route (`0.3646` vs `0.3125`) but remained slightly below no-path (`0.3698`).

Stage 11C retested the best variant with a longer 16-epoch budget against selected controls. Mean dev rollout success:

- `Single Actor`: `0.3490`
- `Role Clones, No Latent Cross-Agent Path`: `0.3490`
- `Capacity Control, Self Read`: `0.3385`
- `Base Proposed, Last-Token Summary`: `0.3594`
- `Causal-Mean Summary Variant`: `0.3698`

This is a weak development signal for causal-mean peer summaries (`+0.0208` over no-path, `+0.0104` over base proposed), not a claim. Eval-time mechanism checks were mixed: cross-task K/V shuffle reduced the causal-mean variant to `0.3333`, but removing the path was only `0.3542`, and role K/V shuffle / Actor peer-read masking increased rollout success to `0.3854`. The behavioral mechanism is therefore not established.

Stage 11D tested direct architectural pressure toward complementary specialization: learned slot exports, sparse top-k peer reads, role dropout, and a slot self-read control. Mean dev rollout success at the same 8-epoch/3-seed cheap-screen budget:

- `Single Actor`: `0.3698`
- `Role Clones, No Latent Cross-Agent Path`: `0.3698`
- `Causal-Mean Summary Variant`: `0.3646`
- `Slot Self-Read M2`: `0.2969`
- `Slot Bottleneck M2`: `0.3021`
- `Slot Bottleneck M2 + Dropout`: `0.3125`
- `Slot Bottleneck M2 + Top-K2`: `0.3229`
- `Slot Bottleneck M4 + Top-K2`: `0.3021`

The best slot variant improved supervised action accuracy (`0.6862`) but hurt rollout success relative to no-path controls. Eval-time removals and Actor peer-read masks did not damage slot variants; removing the path often improved them. This is negative evidence for the hypothesis that slot bottlenecks, top-k reads, or role dropout force useful complementary specialization in the current randomized gridworld harness.

Stage 11E tested the transverse-linearity hypothesis with LoRA-style symmetry breakers. The first branch used learned role gates over shared low-rank bases. It did not actually specialize strongly: LoRA gate entropy stayed near `log(4)` (`1.386`), and adapter deltas were small (`~0.018` norm ratio). `role_lora_cross_causal_mean` reached `0.3542` rollout success versus `0.3385` for `role_lora_no_path`, but still remained below the original single/no-path controls (`0.3698`) and below non-LoRA `cross_causal_mean_summary` (`0.3646`).

The amended hard branch forced each role to use a distinct low-rank basis. This created the intended transverse break: gate entropy was effectively `0.0000`, and `role_lora_hard_cross_causal_mean` had a larger adapter norm (`0.0783`). It still did not make peer reads useful. Mean dev rollout success:

- `Single Actor` / `Role Clones, No Path`: `0.3698`
- `Cross Causal-Mean Summary`: `0.3646`
- `Hard LoRA No Path`: `0.3646`
- `Hard LoRA Self Read`: `0.3802`
- `Hard LoRA Cross Causal-Mean`: `0.3542`

The hard self-read result is a small capacity/specialization side signal, not evidence for cross-agent collaboration. The hard cross path was below its matching hard no-path control (`-0.0104`) and below the original no-path controls. Some eval-time ablations damaged the hard cross method (`remove_path` to `0.3125`, cross-task K/V shuffle to `0.2708`), so the path can affect behavior, but it is not beneficial in this harness.

Current working/not-working conclusions:

- Works as infrastructure: causal role streams, cross-agent Q/K/V block, gradient flow, rollout logging, and ablation plumbing.
- Weakly promising: causal-prefix mean summaries for peer K/V, but only as a development lead and without established mechanism evidence.
- Not rescued at cheap-screen scale: late-only layers, every-layer blocks, subtask+state Q, two-role Actor/Critic, slot bottlenecks, sparse top-k reads, role dropout, learned role-gated LoRA, and hard fixed-role LoRA.
- Not supported: the base last-token subtask-query design, attention-pattern-only collaboration claims, or any final Experiment 11 claim.

It is also not final evidence against the premise because all batches used a small randomly initialized GPT-style causal decoder, not a pretrained causal decoder checkpoint.

Stage 11F implemented the task-redesign branch implied by the negative Stage 11A-E results: private role views plus action-conditioned latent Q/A. Planner, Reader, Critic, and Actor receive different masked current-step inputs, and the Actor must recover missing goal/map evidence through peer answers before acting. This changed the empirical picture. In the combined five-seed 16-epoch read, full-view `single_actor` / `role_clones_no_path` reached `0.3750` rollout success, `private_view_no_path` collapsed to `0.1125`, `private_view_cross_causal_mean` recovered to `0.2812`, and the best constrained relay `private_view_qa_hard_aux_bias` reached `0.2875`. Mechanism ablations on the best Q/A relay were strongly damaging: action-answer shuffle `-0.2406`, cross-task shuffle `-0.1906`, remove path `-0.1687`, and Actor peer-read mask `-0.1656`.

Current Stage 11F interpretation: architectural information partitioning can create real inter-agent reliance in this harness. The robust claim is dependence versus private no-path, not superiority over full-view single-agent baselines or over continuous private peer reads. The best Q/A variant uses auxiliary answer supervision and a direct action-logit bias, so it is a development signal for the architecture pattern rather than evidence for spontaneous emergent specialization.

Stage 11G tested the obvious cleanup: explicit action-query tokens inside the GPT stream. Pure token, hybrid token, and decision-token variants were implemented. This did not rescue the mechanism. The best 16-epoch private decision-token Q/A method reached only `0.1510` rollout success versus `0.1250` for its token no-path control, and eval-time ablations were weak or backwards: removing the path slightly improved rollout, and masking Actor peer reads did not hurt. Full-view decision-token methods could learn (`0.3594`), so the failure is not just token insertion plumbing.

Stage 11H implemented the shared cross-role residual bus branch: each role writes to a fixed non-overlapping bus subspace, every role reads the full bus, and the Actor action head concatenates the final bus so the path is structurally load-bearing. The Stage 11H task also adds history-private masking plus `history_tiebreak` labels, so Planner holds goal evidence, Reader holds obstacle evidence, Critic holds history evidence, and Actor is blind to the private fields. In a 3-seed 16-epoch retest, `private_view_shared_bus` reached `0.3646` rollout success versus `0.0990` for `private_view_history_no_path`, `0.1250` for the old private causal-mean path, and `0.3490` for full-view single/role-clone controls. Eval-time bus ablations were strongly damaging: remove path `-0.3177`, cross-task bus shuffle `-0.3177`, zero Planner writer `-0.3073`, mask action bus `-0.0781`, and mask Actor bus read `-0.0208`.

Current Stage 11H interpretation: the shared bus is the strongest architecture-faithful evidence so far for cross-role dependence in this harness. The robust claim is a development-screen result under constructed private role views and a history-dependent task; Reader and Critic writer-zero ablations were weak or non-damaging for the default bus, so the current mechanism evidence is strongest for Planner/goal information and the global bus path, not yet for every role subspace.

Next smallest discriminating experiment: retain `private_view_shared_bus` for a larger Stage 11H validation with more seeds, difficulty slices, and a task variant that makes obstacle and history subspaces necessary on most decisions. Do not escalate Stage 11G, and do not treat Stage 11H as final until all intended writer ablations damage performance under repeated seeds.

## Folder Layout

- `reports/`: architecture, scope, protocol, claim gates, and publication checklist for the experiment.
- `code/`: runnable gridworld screen and Stage 11A-H architecture variants.
- `results/`: Stage 11A-D smoke, screen, and retest artifacts.
- `summary.md`: concise experiment boundary for the AOB ordered path.
