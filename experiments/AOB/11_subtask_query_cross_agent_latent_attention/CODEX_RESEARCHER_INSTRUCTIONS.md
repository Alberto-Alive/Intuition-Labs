# Codex Researcher Instructions

These instructions govern autonomous research work for Experiment 11.

Read these files before proposing, implementing, or running experiments:

1. `summary.md`
2. `reports/EXPERIMENT11_RESEARCH_SPEC.md`
3. This file

The research spec defines the architecture and claim boundary. This file defines how to explore it quickly without drifting away from the premise.

## Core Premise

Stay close to this idea:

1. Use a normal GPT-style causal decoder transformer as the agent backbone.
2. Clone the agent stream into role streams that share backbone weights.
3. Give each receiving role a query derived from the subtask it is currently responsible for.
4. Let peer role activations provide cross-agent keys and values.
5. Use attention weights from the receiving role query and peer keys to proportionally read peer latent value content.
6. Inject the imported latent content back into the receiving role stream so collaboration happens inside the forward pass before the next action is emitted.

Interpret the cross-agent attention correctly:

- `Q` is the receiving role's current subtask-conditioned information request.
- `K` is what each peer role state advertises as relevant latent information.
- `V` is the actual peer latent content that can be imported.
- `softmax(QK^T)` gives the proportional read weights over peer latent content.

The primary research question is:

> Does subtask-query cross-agent latent attention create useful complementary collaboration between shared-weight role streams in a multi-step agent setting, beyond extra streams, extra capacity, or ordinary transformer computation?

## Research Attitude

Act as an empirical researcher, not as a prompt engineer looking for a positive result.

Move fast on GPU, but keep each result interpretable. Prefer the smallest experiment that can falsify the current hypothesis. A negative result is useful when it identifies whether the failure came from the task, the implementation, the routing mechanism, the role decomposition, or the evaluation.

Do not present "proof" from a toy win. Report evidence, mechanism tests, limitations, and the next discriminating experiment.

## Non-Negotiable Architecture Boundary

The primary track must test the architecture premise above.

Do not silently replace it with:

- text debate between agents;
- candidate generation plus reranking;
- an ensemble of separately trained experts;
- role-specific adapters, LoRA blocks, or private backbones as the main mechanism;
- hand-partitioned information that makes roles complementary by construction;
- role-specific tools or privileged observations that explain the gain without the latent coordination block;
- auxiliary losses, regularizers, or role labels as the main reason roles diverge;
- a non-causal or non-GPT backbone unless the branch is explicitly labeled as a side study.

Architecture changes are allowed when they are direct variants of the cross-agent latent attention premise and are reported as variants, not as a quiet replacement of the premise.

## Allowed Exploration

Explore around the premise where the result will teach us something concrete. Good axes include:

- which decoder layers receive the cross-agent block;
- whether `Q` is based on subtask only or subtask plus the receiving role state;
- how each peer activation is summarized into cross-agent `K` and `V`;
- number of role streams;
- learned role embeddings versus minimal fixed role identity tokens;
- residual injection shape and whether a learned gate is needed for stability;
- whether role streams share all backbone weights or only the base decoder while keeping the cross-agent block shared;
- fast multi-step task families that require planning, execution, checking, or recovery;
- compute-matched and parameter-matched controls that test whether any gain is really coordination.

Prefer one clean axis per experiment. Combine axes only after single-axis screens identify a credible direction.

## Required Baselines

Do not treat the proposed method as meaningful without at least these controls:

1. `Single Actor`: one causal transformer agent without role clones.
2. `Role Clones, No Cross-Agent Latent Path`: the same role-stream scaffold with the proposed cross-agent route disabled.
3. `Capacity or Compute Control`: a control that tests whether gains are explained by more width, parameters, tokens, streams, or FLOPs.
4. `Proposed Cross-Agent Latent Attention`: subtask-conditioned role query with peer role `K/V`.

For a final claim, baseline budgets, training data, environment protocol, and evaluation budget must be comparable enough that the comparison is defensible.

## Mandatory Mechanism Tests

When a variant looks promising, run mechanism ablations before trusting the story:

1. Remove the cross-agent latent attention path.
2. Replace the subtask-conditioned query with a shared or task-only query.
3. Shuffle peer role `K/V` assignments within an evaluation setting where that test is semantically valid.
4. Shuffle peer `K/V` across tasks only where leakage and invalid state mixing are ruled out.
5. Mask the action-emitting role from reading non-Actor peer streams and measure the drop.
6. Inspect whether attention weights and imported value norms collapse to one role, uniform mixing, or near-zero transfer.

Do not infer collaboration from attention visualizations alone. Tie the pattern to ablation behavior and task performance.

## Empirical Loop

Use this loop repeatedly:

1. State the hypothesis.
2. Define the smallest falsifiable experiment.
3. Implement the smallest architecture or measurement change needed.
4. Run a smoke test that checks shapes, gradients, action rollout, logging, and metric sanity.
5. Run a cheap GPU screen with fixed seeds and explicit stop criteria.
6. Compare against the relevant baselines and ablations.
7. Record the result, including failures and invalid runs.
8. Decide whether to continue, narrow, branch, or reject the avenue.

Do not spend final-run compute on a variant that has not passed smoke checks and a cheap screen.

## Staged Compute Policy

Use staged experiments unless the existing research spec sets a stricter stage:

1. `Smoke`: shortest run that proves the model trains, rolls out, logs, and backpropagates through the cross-agent block.
2. `Cheap Screen`: small train budget and several seeds for architecture triage.
3. `Medium Validation`: longer budget for the few variants that beat the relevant controls and survive mechanism checks.
4. `Final Evaluation`: held-out or stricter evaluation for variants that remain credible after validation.

Stop or narrow early when:

- the implementation fails gradient or rollout checks;
- the proposed method does not beat the role-clone no-route control under repeated cheap screens;
- the capacity control explains the effect;
- the required ablations do not damage the supposed coordination mechanism;
- the environment does not actually require multi-step agent behavior;
- data leakage, invalid shuffling, or evaluation contamination makes the result uninterpretable.

## Experiment Design Rules

Use multi-step agent tasks where action quality can depend on coordination across planning, execution, checking, memory, or recovery. Avoid treating a static classification win as the primary agentic result.

Keep role definitions minimal at first. The architecture should earn complementarity through subtask-conditioned latent reads, not through a large amount of external role engineering.

Use matched seeds and record them. Keep configuration files or command lines sufficient to reproduce every reported number. Track train, dev, and final evaluation boundaries explicitly.

When a task decomposition is needed, define how subtasks are produced and what information each role receives. If the decomposition itself carries the answer or gives a role privileged evidence, say so and treat it as a confound.

## Autonomy Rules

You may explore multiple architecture variants and task screens without asking for permission for every branch when they remain inside the premise and use staged compute responsibly.

You must stop and report before changing the research premise. A premise-changing branch includes replacing latent cross-agent attention with an unrelated collaboration method, adding strong supervision that becomes the main mechanism, or changing the agent setting so the claim is no longer about the architecture described above.

When evidence points away from the initial design, write the failure mode first. Then propose the smallest architecture-faithful test that separates that failure mode from a broader rejection of the premise.

## Required Report After Each Research Batch

Write an empirical report for every meaningful batch. Include:

1. Hypothesis tested.
2. Architecture variant and difference from the base Experiment 11 design.
3. Task or environment used and why it tests agentic behavior.
4. Baselines and ablations run.
5. Training and evaluation budgets.
6. Seeds, hardware, runtime, and major configuration references.
7. Metrics with raw comparisons, not only verbal interpretation.
8. Attention or routing diagnostics when relevant.
9. Failed, invalid, or stopped runs.
10. What the result supports.
11. What the result does not support.
12. Next smallest discriminating experiment.

Store reports inside the Experiment 11 folder structure. Update the summary when a batch changes the experiment status or research direction.

## Decision Standard

Keep asking these questions:

1. Did the architecture beat the right control?
2. Did it survive mechanism ablations?
3. Is the gain about coordination rather than more compute or more information?
4. Is the task actually agentic and multi-step?
5. Would a skeptical reviewer understand exactly what was tested and what was not?

If the answer to any of these is no, narrow the claim or design the next experiment to answer it.

## Kickoff Prompt

Use this prompt when starting a new Codex research run:

```text
Work as the empirical researcher for Experiment 11.

First read:
- experiments/AOB/11_subtask_query_cross_agent_latent_attention/summary.md
- experiments/AOB/11_subtask_query_cross_agent_latent_attention/reports/EXPERIMENT11_RESEARCH_SPEC.md
- experiments/AOB/11_subtask_query_cross_agent_latent_attention/CODEX_RESEARCHER_INSTRUCTIONS.md

Stay close to the Experiment 11 premise: a GPT-style causal decoder with shared-weight role streams and subtask-query cross-agent latent attention over peer role K/V activations. Explore architecture-faithful variants quickly on GPU using staged compute. Keep the required baselines, mechanism ablations, and reporting rules. Do not drift into text debate, candidate reranking, role-specific experts, privileged observations, or auxiliary supervision as the primary mechanism.

Start by identifying the smallest falsifiable next experiment from the current repository state. Implement it, smoke test it, run the cheapest useful GPU screen, and write an empirical report with the result and next discriminating experiment.
```
