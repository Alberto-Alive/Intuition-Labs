# Experiment 11 Research Spec

## Title

Subtask-Query Cross-Agent Latent Attention for Shared-Weight Causal Agent Streams

## Status

`architecture_and_protocol_locked_before_implementation`

## Motivation

Experiments 0 through 10 establish controlled latent coordination evidence and its limits on candidate-selection settings. Experiment 11 changes the target claim. It asks whether a causal language agent can act better in a multi-step environment when role-cloned streams selectively read peer latent activations during the same forward pass.

The architectural idea is not text debate, voting, post-hoc ranking, or a portfolio over candidates. It is internal latent collaboration:

> A receiving role queries peer role activations with its current subtask as the purpose of the lookup.

## Primary Research Question

Given the same pretrained causal decoder backbone, task history, action interface, training budget, and evaluation episodes, does adding subtask-query cross-agent latent attention to role-cloned streams improve held-out multi-step agent success beyond:

1. a single causal agent,
2. role-cloned shared-weight streams with no cross-agent latent path,
3. a capacity and compute control that spends comparable parameters or compute without peer latent reads?

## Mechanism Hypothesis

Subtask-conditioned `Q` turns cross-agent attention into purposeful peer-state retrieval:

- `Q_i`: what receiving role `i` is responsible for resolving now,
- `K_j`: what peer role state `j` currently advertises as available,
- `V_j`: the peer latent content available to import,
- `softmax(Q_i K_j^T)`: how much each peer content stream contributes to the receiving role update.

The architecture should win mainly on tasks where a role stream benefits from reading latent evidence held in another role stream before the next action is emitted.

## Non-Claims

Do not present any of the following as Experiment 11 evidence:

- a candidate scorer choosing among prebuilt patches as an agentic proof,
- attention weights alone as proof of reasoning, disagreement, or collaboration,
- a gain caused only by more tokens, more backbone parameters, or more rollout attempts,
- a result tuned on the final evaluation split,
- a single cherry-picked task trace,
- a prompt-only multi-agent result as proof of the latent mechanism.

## Primary Architecture

### Backbone

Use a normal pretrained GPT-style causal decoder transformer. Freeze the architecture choice before the main search and use the same backbone family and checkpoint for every baseline and proposed method in the primary comparison.

The backbone must support:

- causal next-token or next-action generation from a task and observation history,
- hidden-state access at selected decoder layers,
- insertion of residual cross-agent latent attention blocks,
- the same tool/action protocol for all compared methods.

### Role Streams

For one episode at decision step `t`, create `R` role streams with shared backbone weights:

```text
role stream i input = final task + trajectory history through step t + role subtask i
```

The primary design uses role/subtask descriptions as part of the architecture input contract. It does not add role-specific adapters, hand-partitioned evidence, role-specific tools, auxiliary role losses, redundancy penalties, or anti-dominance losses in the main method.

The first locked role set should be small and fixed before training. A candidate set is:

| Role | Subtask encoded for `Q` |
|---|---|
| Planner | decide the next useful step toward the final task |
| Reader | identify task-relevant evidence in the current history and context |
| Critic | detect contradictions, invalid actions, and missing checks |
| Actor | produce the next environment action or tool call |

Only the Actor stream emits the primary next action in the main method. Other streams affect it through the latent path.

### Subtask-Query Cross-Agent Attention Block

Let `H_i^l` be the hidden states of role stream `i` after the normal causal self-attention portion of decoder layer `l`.

The first implementation should use role summaries rather than full token-to-token all-role attention:

```text
r_j^l = Summary(H_j^l)
s_i   = EncodeSubtask(subtask_i)

q_i^l = Wq(s_i)
k_j^l = Wk(r_j^l)
v_j^l = Wv(r_j^l)

a_ij^l = softmax_j(q_i^l (k_j^l)^T / sqrt(d))
c_i^l  = sum_j a_ij^l v_j^l
H_i^{l+} = H_i^l + Inject(c_i^l)
```

Interpretation:

- `Q` is the receiving role's subtask lookup.
- `K` is the peer-state relevance index.
- `V` is the peer-state latent content.
- The attention weights are the proportional read from peer states.

`Summary` must be fixed before final validation. Acceptable initial choices are current-step last-token state or a causal prefix summary. The summary must not depend on future generated tokens or unavailable future observations.

### Insertion Point

The first architecture should insert the block at a predeclared layer set, not search over every possible layer on the final data. A default starting comparison is:

```text
cross-agent blocks after middle and late decoder layers
```

An architecture search may compare fixed layer schedules on train/dev only:

- late-only,
- middle-plus-late,
- every selected block group.

The selected schedule is then frozen for final evaluation.

### Causality Rule

At action step `t`, cross-agent attention may read only role activations computed from inputs available at `t`.

Forbidden leakage includes:

- future observations,
- future tool results,
- gold future actions,
- hidden states from a longer gold trajectory prefix than the baseline sees,
- final success labels embedded into subtask text or role metadata.

## Experiment Setup

### Environment Requirement

Use a multi-step agent environment with executable actions and observations. The main result must measure acting in an environment, not only candidate ranking.

The environment must provide:

- deterministic or replayable task instances where possible,
- held-out task splits,
- trace logging of task text, observations, tool calls, actions, and terminal result,
- a clear success metric and failure taxonomy,
- time, action, and token budgets shared across methods.

### Stage Plan

| Stage | Purpose | Data access | Exit condition |
|---|---|---|---|
| 11A smoke | Verify shapes, causal masks, gradients, trace logging, and no future leakage. | tiny debug tasks | implementation audits pass |
| 11B mechanism screen | Compare a small locked set of cross-agent block schedules and summary choices. | train/dev only | one method selected or architecture rejected |
| 11C controlled validation | Run selected method and all primary baselines across enough seeds and held-out dev families. | train/dev only | variant frozen with no final-set tuning |
| 11D final evaluation | Evaluate the frozen method once against locked baselines and ablations. | held-out final only | claim gates reported pass or fail |

## Primary Comparisons

The primary table must include:

| Method | Purpose |
|---|---|
| Single Actor | normal causal agent baseline |
| Role Clones, No Latent Cross-Agent Path | isolates the value of cloning and role-conditioned inputs |
| Role Clones, Capacity Control | spends comparable module capacity without peer-state retrieval |
| Role Clones, Subtask-Query Cross-Agent Latent Attention | proposed method |

Optional secondary baselines may include text-mediated role communication or generic peer attention with a non-subtask query. Secondary baselines must not replace the primary table.

## Mandatory Ablations

The final report must include the proposed method under these interventions:

| Ablation | Question |
|---|---|
| Remove cross-agent blocks | Is the latent path necessary? |
| Replace subtask `Q` with a shared query | Does subtask-specific lookup matter? |
| Shuffle peer `K/V` across roles within an episode | Does correct peer identity/state matter? |
| Shuffle peer `K/V` across tasks where legal for shape only | Does task-matched latent content matter? |
| Mask Actor reads from non-Actor peers | Is peer information used before acting? |
| Match or report parameter and inference-compute differences | Is the result only scale or compute? |

All ablations must preserve the actor's allowed task history and action budget. Any ablation that changes available history is a separate result, not a mechanism control.

## Metrics

### Primary

- held-out task success rate under a fixed action and token budget,
- success rate by task family and difficulty slice,
- per-seed mean and dispersion,
- budget-normalized success when methods have different latent compute.

### Secondary

- valid-action rate,
- terminal failure taxonomy,
- action count and token count,
- latency and compute cost,
- tool-call precision where the environment supports an unambiguous label,
- recovery after environment errors or failed tool calls.

### Mechanism Diagnostics

Diagnostics support interpretation but do not replace behavioral metrics:

- cross-agent attention mass by receiving and source role,
- attention entropy over roles by layer and step,
- norm of injected latent update relative to receiving hidden state,
- performance delta after role-state shuffles and masks,
- gradient flow through `Q`, `K`, `V`, and injection projections.

## Claim Gates

### Gate A: Agentic Task Gate

The main task must require a sequence of actions with environment observations. Candidate ranking alone fails this gate.

### Gate B: Baseline Gate

The proposed method must beat or improve the robustness of both Single Actor and Role Clones, No Latent Cross-Agent Path under the same task split and budget. Report capacity-control results even if they win.

### Gate C: Mechanism Gate

At least one subtask-specific mechanism control must materially reduce performance relative to the proposed method:

- shared-query replacement,
- correct-role `K/V` shuffle,
- peer-content cross-task shuffle,
- non-Actor read mask.

Do not infer mechanism only from learned attention visualizations.

### Gate D: Generalization Gate

The selected method must survive a held-out final split and more than one seed. Report variance and failure slices. Any dev-only win remains a development result.

### Gate E: Reproducibility Gate

The final report must provide:

- exact backbone checkpoint and license status,
- environment/task version and split manifests,
- code revision,
- config files,
- hyperparameter selection path,
- seeds,
- compute hardware, memory, wall time, and budgets,
- failed variants and stopped searches,
- statistical uncertainty or an explicit justified limitation.

## Strict Research Rules

1. Predeclare the primary method, baseline table, primary metric, claim gates, and final split before final evaluation.
2. Tune architectures on train/dev only. Do not rerun or revise the method because of final evaluation outcomes.
3. Keep the backbone checkpoint, context window, tool schema, action budget, decoding policy, and episode stopping rule equal across primary methods unless a difference is explicitly measured and reported.
4. Make the first architectural claim about the latent cross-agent attention mechanism, not about general intelligence, general code repair, or autonomous research ability.
5. Report negative and null results from locked baselines and ablations. Do not omit a control because it weakens the narrative.
6. Use multiple seeds for trainable runs and report per-seed results or sufficient summary statistics to expose instability.
7. Preserve full machine-readable traces for final episodes, metrics, configs, and run audits.
8. Never use attention maps as sole evidence of collaboration. Require behavioral ablation evidence.
9. Never compare to a weaker baseline with less task history, fewer permitted actions, or a worse tool protocol without labeling it as non-comparable.
10. Treat agent-loop leakage as a hard failure: no gold future actions, future observations, or task-solution metadata may enter prompts, hidden-state reads, summaries, or subtask text.

## Publication-Quality Reporting Checklist

Before calling the result publishable, the paper draft and artifact bundle should contain:

- a full architecture diagram and formulas for `Q`, `K`, `V`, attention weights, injection, and causal restrictions,
- train/dev/final split definitions and task-family coverage,
- exact baselines and compute-budget comparison,
- implementation details needed to reproduce the architecture and experiment,
- uncertainty reporting for claims supported by experiments,
- limitations and non-claims,
- ethics, data-license, model-license, and LLM-use disclosures relevant to the chosen environment and backbone,
- code/config/trace artifact plan or an explicit reason a component cannot be released.

These requirements are intentionally aligned with current major-venue checklist expectations around claim scope, limitations, architecture reproducibility, experimental details, statistical uncertainty, compute resources, and LLM-use disclosure.

External reporting reference:

- NeurIPS Paper Checklist Guidelines: https://nips.cc/public/guides/PaperChecklist

## Locked Initial Implementation Boundary

The first implementation should include only:

- one pretrained causal decoder backbone path,
- shared-weight role streams,
- fixed role/subtask text contract,
- subtask encoder for `Q`,
- role-state summary projections for `K` and `V`,
- residual latent injection into selected decoder layers,
- actor-only next-action emission,
- primary baselines, ablations, logging, leakage audits, and multi-seed evaluation harness.

Defer until after the core method is tested:

- role-specific adapters,
- role-specific tool access,
- hand-partitioned information,
- learned role graphs,
- redundancy or dominance regularizers,
- auxiliary role-task losses,
- full token-level all-role peer attention,
- paper claims about emergent disagreement, uncertainty, or deliberation stages without targeted intervention tests.
