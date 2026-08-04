# Stage 11A Agentic Gridworld Smoke and Cheap Screen

## Hypothesis Tested

A subtask-query cross-agent latent path should beat no-path role clones and a self-read capacity control on a cheap multi-step gridworld screen if the implementation creates useful collaboration.

## Architecture Variant

Tiny GPT-style causal decoder role streams with shared token, position, decoder-block, layer-norm, and action-head weights. The proposed variant inserts subtask-query cross-agent latent attention after the configured decoder layers. Role summaries are last-token hidden states; Q is derived from learned subtask/role embeddings; K/V are projected peer role summaries; the weighted value read is gated and injected residually into the receiving role stream before the actor action head.

Important limitation: this batch used a randomly initialized small causal decoder, not a pretrained checkpoint. It is an implementation and development screen, not final evidence for the locked primary claim.

## Task

Executable gridworld navigation. At each decision step, the agent observes the current position, goal, obstacle set, and action history, emits one of U/D/L/R, receives the updated position, and succeeds only by reaching the goal inside the action budget.

## Budgets

- Train worlds: `192`; dev worlds: `64`; epochs: `8`; batch size: `96`; max steps: `14`.
- Seeds: `[101, 102, 103]`.
- Hardware: `cuda`; CUDA available: `True`; GPU: `NVIDIA GeForce RTX 5070 Ti`.

## Primary Results

| method | seeds | params | action acc mean | rollout success mean | valid action mean | attention entropy | inject norm ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| `capacity_control_self_read` | 3 | 223940 | 0.6737 +/- 0.0158 | 0.2969 +/- 0.0710 | 0.3749 | 0.0000 | 0.0954 |
| `cross_agent_latent_attention` | 3 | 223940 | 0.6748 +/- 0.0176 | 0.3125 +/- 0.0710 | 0.3779 | 1.0976 | 0.0931 |
| `role_clones_no_path` | 3 | 173636 | 0.6785 +/- 0.0089 | 0.3698 +/- 0.0940 | 0.3868 | 0.0000 | 0.0000 |
| `shared_query_control` | 3 | 223940 | 0.6738 +/- 0.0169 | 0.3073 +/- 0.0725 | 0.3743 | 1.0986 | 0.0934 |
| `single_actor` | 3 | 173636 | 0.6785 +/- 0.0089 | 0.3698 +/- 0.0940 | 0.3868 | 0.0000 | 0.0000 |

Decision: this screen is negative for the proposed route. The cross-agent latent attention model did not beat Single Actor or Role Clones, No Path on rollout success (`0.3125` vs `0.3698` mean), and the trained shared-query control was close (`0.3073`). The result does not pass the baseline gate even as a development screen.

## Mechanism Ablations

These are eval-time interventions on the trained proposed model, so they are mechanism smoke tests rather than full retrained controls.

| ablation | seeds | action acc mean | rollout success mean | valid action mean | attention entropy | inject norm ratio |
|---|---:|---:|---:|---:|---:|---:|
| `cross_task_kv_shuffle` | 3 | 0.6803 +/- 0.0197 | 0.3385 +/- 0.0516 | 0.3851 | 1.0975 | 0.0908 |
| `mask_actor_peer_reads` | 3 | 0.6747 +/- 0.0166 | 0.3125 +/- 0.0797 | 0.3758 | 0.8238 | 0.0931 |
| `remove_path` | 3 | 0.6707 +/- 0.0148 | 0.3229 +/- 0.0703 | 0.3727 | 0.0000 | 0.0000 |
| `role_kv_shuffle` | 3 | 0.6747 +/- 0.0176 | 0.3125 +/- 0.0710 | 0.3760 | 1.0978 | 0.0932 |

Mechanism smoke result: none of the eval-time K/V shuffles, Actor peer-read mask, or path removal controls materially reduced rollout success relative to the proposed model.

## Routing Diagnostics

- Proposed actor read mass by source role: `{'actor': 0.0, 'critic': 0.3431474467118581, 'planner': 0.3279147694508235, 'reader': 0.3289378136396408}`.
- Attention diagnostics are used only as support; the behavioral ablations above are the relevant mechanism checks.

## Failed or Invalid Runs

- No runtime failures were recorded by the screen script.
- The batch is claim-limited because it does not use a pretrained decoder and does not touch a final held-out split.

## What This Supports

- The Experiment 11 architecture path is implementable in this repo with causal shared-weight role streams, cross-agent Q/K/V reads, residual injection, gradient flow, rollout evaluation, and machine-readable logging.
- The task and harness are sufficient to expose a negative baseline result quickly: the proposed path did not improve the agentic rollout metric under this tiny randomly initialized setup.

## What This Does Not Support

- No publishable or final Experiment 11 claim is supported.
- No claim about pretrained LLM agents is supported by this batch.
- Eval-time ablations do not replace retrained shared-query and capacity-matched controls for a final claim.

## Next Smallest Discriminating Experiment

Before scaling final evaluation, separate task/model weakness from mechanism weakness. The next smallest run should either use a locally available pretrained causal decoder, or explicitly document that no pretrained checkpoint is available and run a stronger randomized-decoder dev validation with more epochs, difficulty slices, and retrained mechanism controls. Continue only if the proposed path beats the no-path role clone under repeated dev seeds and at least one mechanism ablation damages performance.
