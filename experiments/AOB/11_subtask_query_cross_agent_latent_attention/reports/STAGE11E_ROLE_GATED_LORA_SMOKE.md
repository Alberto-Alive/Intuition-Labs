# Stage 11E Role-Gated LoRA Smoke

## Hypothesis Tested

Role-gated shared-basis LoRA adapters may break transverse linearity across cloned role streams. The critical test is whether LoRA plus cross-agent latent reads beats the matching LoRA no-path control.

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

## Newly Tried Or Retested In Stage 11E

- `role_lora_no_path`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`none`, query=`subtask`, summary=`last_token`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=8, lora_bases=4, lora_alpha=8.0, lora_gate=softmax.
- `role_lora_cross_causal_mean`: roles=['planner', 'reader', 'critic', 'actor'], cross_mode=`cross_peer`, query=`subtask`, summary=`causal_mean`, layers=default, slots=0, topk=0, role_dropout=0.0, lora_rank=8, lora_bases=4, lora_alpha=8.0, lora_gate=softmax.

## Still Not Tried

- A pretrained causal decoder checkpoint.
- Full token-level cross-role attention instead of role summaries.
- Learned or external subtask decomposition beyond fixed role/subtask tokens.
- Learned residual gate schedules or initialized near-zero gates.
- Retrained mechanism controls for every variant.
- Actor-only source routing or fixed sparse role-routing graphs.
- More role-count schedules beyond four roles and Actor/Critic two-role.
- Larger model/data/epoch scaling and difficulty-sliced validation.
- Held-out final evaluation.

## Task And Budget

Same executable gridworld harness as Stage 11A. Train worlds `24`, dev worlds `10`, epochs `1`, batch size `24`, seeds `[101]`.
Hardware `cuda`, CUDA `True`, GPU `NVIDIA GeForce RTX 5070 Ti`.

## Results

| method | seeds | params | action acc mean | rollout success mean | valid action mean | entropy | inject norm | lora norm | lora gate entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `role_lora_cross_causal_mean` | 1 | 236660 | 0.2000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.5143 | 1.0962 | 0.3537 | 0.0065 | 1.3861 |
| `role_lora_no_path` | 1 | 186356 | 0.2000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.1929 | 0.0000 | 0.0000 | 0.0065 | 1.3862 |

## Mechanism Ablations On Base Proposed

No mechanism ablations were logged.

## Eval-Time Ablations By Cross Method

| method | ablation | rollout success mean | delta vs method |
|---|---|---:|---:|
| `role_lora_cross_causal_mean` | `cross_task_kv_shuffle` | 0.0000 | +0.0000 |
| `role_lora_cross_causal_mean` | `mask_actor_peer_reads` | 0.0000 | +0.0000 |
| `role_lora_cross_causal_mean` | `remove_path` | 0.0000 | +0.0000 |
| `role_lora_cross_causal_mean` | `role_kv_shuffle` | 0.0000 | +0.0000 |

## Conclusions

- Best method in this screen: `role_lora_cross_causal_mean` at `0.0000` mean rollout success.
- Single/no-path control ceiling in this screen: `0.0000` mean rollout success.
- `role_lora_cross_causal_mean`: rollout `0.0000`, delta vs reference `+0.0000`, delta vs best single/no-path `+0.0000`.
- `role_lora_no_path`: rollout `0.0000`, delta vs reference `+0.0000`, delta vs best single/no-path `+0.0000`.
- LoRA cross-path test `role_lora_cross_causal_mean` vs `role_lora_no_path`: `0.0000` vs `0.0000` (`+0.0000`).
- No LoRA cross-agent variant beat the single/no-path controls. The LoRA branch does not rescue the randomized-decoder gridworld setup.

## Next Decision

Do not escalate the LoRA cross-agent branch on this randomized gridworld. Explicit symmetry breaking can change behavior, but it has not made peer latent reads beat their matching no-path controls and the original single/no-path baselines.
