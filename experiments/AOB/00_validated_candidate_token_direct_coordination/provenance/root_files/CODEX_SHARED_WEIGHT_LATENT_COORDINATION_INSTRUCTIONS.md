# Codex Instruction File: Real Shared-Weight Latent Coordination Experiment

## 0. Mission

Build a rigorous, automatically scored experiment proving or falsifying this hypothesis:

> A single shared pretrained transformer, cloned into multiple agents over partial task views, can be trained end-to-end through an activation/latent coordinator so that its own trainable weights learn coordination-friendly internal representations.

This must not be another frozen-hidden-state probe. This must be a real shared-weight cloned-agent training experiment.

The experiment is successful only if the trainable shared cloned model plus latent coordinator beats strong frozen-latent, text-only, and coordinator-only baselines under clean controls, and audit logs prove that the shared model’s trainable parameters receive gradient and change because of the final group loss.

---

## 1. Non-Negotiable Invariants

These are hard requirements. If any fail, the experiment is invalid.

### 1.1 Shared trainable agent model

There must be one shared agent model `Mθ` reused for every clone.

The clones must not be separate independently initialized models.

Correct:

```text
same Mθ(view_1) -> h_1
same Mθ(view_2) -> h_2
same Mθ(view_3) -> h_3
same Mθ(view_4) -> h_4
```

Wrong:

```text
M1(view_1), M2(view_2), M3(view_3), M4(view_4)
```

All clone paths must reference the same trainable parameter objects.

### 1.2 Shared model must actually learn

The main experiment must update trainable parameters inside the shared agent model.

Acceptable:

- pretrained transformer with shared LoRA/adapters;
- pretrained transformer with trainable adapter blocks inserted into layers;
- small trainable transformer only as fallback or debug baseline;
- selective fine-tuning of a small subset of model layers if memory permits.

Not acceptable:

- frozen transformer plus trained external coordinator only;
- frozen hidden states plus probe;
- MLP pretending to be the cloned agent;
- text-only agents with no activation training;
- a coordinator-only experiment.

### 1.3 Group-level loss

For each task instance:

1. Create `N` partial views.
2. Run the same `Mθ` on each view.
3. Extract differentiable internal activations from each clone.
4. Feed activations into coordinator `Cφ`.
5. Predict one final group answer.
6. Compute one final group loss.
7. Backpropagate through `Cφ`, through clone activations, and into the shared model trainable parameters `θ`.

The final group answer must be the only authority for the main task objective.

### 1.4 No full evidence leakage

No single clone may receive the full answer evidence in the main condition.

The coordinator may receive only:

- clone activations;
- role/view/slot embeddings;
- allowed metadata explicitly included in all baselines.

The coordinator must not receive the gold answer, gold patch text, gold evidence bits, all files, all views, or direct labels except in explicit oracle baselines.

### 1.5 No test contamination

Train labels may be used only for training.

Dev labels may be used for early stopping and model selection.

Test labels must be used only once after all model/variant selection is complete.

The report must include a split audit.

### 1.6 Proof logs are mandatory

The experiment must log:

- shared parameter identity across clones;
- nonzero gradient norms for shared model trainable parameters;
- nonzero parameter deltas for shared model trainable parameters;
- coordinator gradient norms and parameter deltas;
- activation `requires_grad=True` before the coordinator;
- per-clone gradient contribution;
- no detach between clone activations and loss;
- memory/batch settings;
- exact train/dev/test split IDs.

---

## 2. Target Claim

The strongest acceptable claim is:

> On an automatically scored multi-view benchmark, shared-weight cloned transformer agents trained through a latent activation coordinator learn coordination-friendly internal representations that improve group performance beyond frozen-latent, text-only, and coordinator-only baselines.

Do not claim:

- open-ended 1,000-agent collaboration;
- general LLM self-organization;
- universal superiority over all multi-agent systems;
- real-world autonomous coding ability;
- SOTA without matched baselines.

---

## 3. Recommended Benchmark: Multi-View Code Debugging / Patch Selection

Use a real-ish automatically scored benchmark inspired by issue-to-patch workflows.

The first publishable-level benchmark should be patch-candidate selection or bug localization, not free-form patch generation.

### 3.1 Preferred task format

Each instance contains:

- issue or bug description;
- failing test trace or assertion;
- multiple relevant code snippets/files;
- optional docs/config snippet;
- candidate patches or candidate bug locations;
- one gold label.

Agents receive partial views:

```text
Agent 1: issue + failing test trace
Agent 2: file/snippet A
Agent 3: file/snippet B
Agent 4: file/snippet C/docs/config
```

Coordinator predicts:

- correct patch candidate; or
- correct bug location; or
- correct cause class; or
- correct repair option.

The first version should use classification over candidates. This keeps the task automatically scorable and differentiable.

### 3.2 Dataset source options

Codex should search the repo/environment for available datasets or scripts.

Priority order:

1. Existing local SWE-bench / SWE-bench Lite / SWE-bench Verified artifacts, if present.
2. Existing local bug-fix datasets or issue-patch JSONL/CSV files, if present.
3. Local Git repositories with commit history from which bug-fix pairs can be mined.
4. A constructed multi-file patch-selection dataset from local source files and generated candidate edits.
5. Stage 6A synthetic strict task only as fallback, clearly marked as fallback and not publishable proof.

If no real-ish dataset is available, Codex must produce a dataset-construction script and state exactly what could not be completed.

### 3.3 Patch-candidate construction

For each instance, create:

- one gold patch candidate;
- 3–7 distractor candidates.

Distractors should be hard enough:

- edit nearby files/functions;
- use plausible but incorrect constants/operators;
- patch the wrong file;
- patch the right file incorrectly;
- fix a symptom but not the cause;
- preserve style and syntax where possible.

Avoid trivial distractors such as malformed code, empty patches, or obviously unrelated text.

### 3.4 Multi-view requirement

The task should require combining views.

Add audits:

- single partial-view agent baseline;
- single full-context agent baseline;
- per-view ablations;
- view masking;
- view shuffling;
- role-label shuffling.

If a single partial view solves the task nearly as well as the full system, the benchmark is too weak.

---

## 4. Architecture Requirements

### 4.1 Shared agent model `Mθ`

Preferred:

- local pretrained Hugging Face causal transformer;
- shared LoRA/adapters inserted into attention and/or MLP projections;
- base weights frozen at first;
- adapter weights trainable and shared across all clones.

If memory allows, optionally compare:

- LoRA on attention only;
- LoRA on attention + MLP;
- small trainable projection head from hidden states;
- selective unfreezing of top transformer block;
- fully trainable small transformer fallback.

### 4.2 Activation extraction

Extract hidden states from the shared model for each clone.

Test candidate slices:

- final token, last layer;
- final token, top K layers concatenated/projected;
- prompt mean pooling;
- special coordinator/readout token if added;
- selected layer positions from prior diagnostics.

The activation must remain differentiable in the trainable condition.

No `.detach()` in the main training path.

### 4.3 Coordinator `Cφ`

Codex may choose/compare coordinator families, but must include at least two serious variants if time permits.

Recommended variants:

#### A. Coordinator-token cross-attention

```text
coordinator token = learned query
agent activations + role embeddings = keys/values
coordinator attends to all clone activations
classifier predicts final group answer
```

#### B. Role-aware self-attention over agents

```text
agent activation_i + role embedding_i
self-attention across agent slots
CLS/readout token predicts final group answer
```

#### C. Bidirectional coordinator-agent attention

```text
coordinator reads agents
agents update through coordinator signal
repeat 1–2 layers
coordinator predicts answer
```

Use role/view embeddings in all non-oracle main variants.

### 4.4 Shared model and coordinator training

Main training condition:

```text
Mθ adapters: trainable
Cφ: trainable
loss: final group answer cross entropy
```

Frozen baseline:

```text
Mθ adapters/base: frozen
Cφ: trainable
same activations and architecture where possible
```

The only intended difference between main and frozen baseline should be whether `Mθ` trainable parameters are updated.

---

## 5. Required Baselines

Codex must implement or approximate these baselines.

### 5.1 Frozen shared model + activation coordinator

Same pretrained model, same clone views, same coordinator family, but shared model trainable parameters frozen.

This is the most important baseline.

### 5.2 Trainable shared model + activation coordinator

The proposed method.

Must beat frozen baseline to support the core claim.

### 5.3 Text/output-only multi-agent coordinator

Each clone produces visible text/logit summary or generated answer.

Coordinator sees only text/output features, not hidden activations.

This tests whether hidden activation coordination adds value beyond text-mediated collaboration.

### 5.4 Single-agent full-context baseline

One model receives all views concatenated within context and predicts final answer.

This answers whether clone coordination is useful versus simply using one model with full context.

### 5.5 Single-agent partial-view baseline

One model receives only one view.

This proves the task requires distributed information.

### 5.6 Explicit evidence-sharing oracle

Coordinator receives structured gold evidence or all views/candidates in the most direct form.

This is an upper bound, not a fair deployed model.

### 5.7 Coordinator-only / probe baseline

Frozen hidden activations plus trained probe/coordinator.

This must be labeled diagnostic only.

### 5.8 Optional LatentMAS-style baseline

If feasible, implement a training-free latent collaboration baseline:

- frozen model;
- latent memory/coordinator over hidden states;
- no updates to the shared model;
- same task/views.

This is important if the final paper frames against training-free latent collaboration.

### 5.9 Optional Mixture-of-Agents-style baseline

If feasible, implement text-based multi-agent aggregation:

- agents generate text rationales/answers;
- aggregator receives text outputs;
- final answer selected from candidates;
- no hidden activations.

Keep compute/model budget matched as closely as practical.

---

## 6. Required Controls

Controls must be run for the proposed trainable shared model condition and, where useful, for the frozen baseline.

### 6.1 Randomized labels

Train with labels randomly permuted.

Expected: accuracy near chance.

If accuracy remains high, there is leakage or memorization.

### 6.2 Hidden-state shuffle across examples

Shuffle clone activations across examples before coordinator.

Expected: collapse near chance.

### 6.3 View/evidence masking

Mask one or more partial views.

Expected: performance drops.

### 6.4 View/evidence shuffling

Shuffle views between examples while preserving labels.

Expected: performance drops.

### 6.5 Role-label shuffle

Shuffle role/view embeddings while keeping activations fixed.

Expected: performance drops if role identity matters.

### 6.6 Physical order shuffle with role labels preserved

Shuffle clone order in the batch/list but keep correct role embeddings attached.

Expected: minimal or no change.

### 6.7 Candidate order shuffle

Shuffle patch-candidate order while preserving candidate labels.

Expected: minimal or no change if candidate representation is correct.

### 6.8 Output leakage audit

Check whether visible outputs/text summaries contain answer labels, gold patch IDs, candidate IDs, hidden control tokens, or any private evidence not allowed for the baseline.

### 6.9 Split leakage audit

Check train/dev/test overlap by:

- instance ID;
- repository + issue/commit ID;
- file path + patch hash;
- candidate patch hash;
- near-duplicate issue text if feasible.

---

## 7. Memory and Performance Strategy for 16GB GPU

Prioritize a working rigorous experiment over a huge model.

Recommended starting setup:

```text
model: 125M–350M local pretrained transformer if available
training: LoRA/adapters
agents: 4 clones
sequence length: 256–1024 depending on memory
batch size: small, with gradient accumulation
precision: bf16/fp16 if stable
clone execution: sequential if needed to save peak memory
gradient checkpointing: enabled if memory pressure appears
```

Scaling ladder:

1. small trainable transformer sanity check;
2. pretrained 125M–350M + LoRA, 4 clones;
3. pretrained 1B-ish + LoRA, 4 clones;
4. 8 clones;
5. larger model or QLoRA only after the architecture proves itself.

Do not spend the whole run fighting memory on 7B before the method is validated.

---

## 8. Implementation Plan

### Phase 0: Repository and environment inspection

Codex must first inspect:

- existing experiment framework;
- available GPU and CPU memory;
- installed packages;
- local Hugging Face model cache;
- local datasets;
- existing Stage 6A code;
- current report generation utilities.

Produce an initial implementation plan before editing major files.

### Phase 1: Dataset builder

Implement dataset interface:

```python
class MultiViewTaskExample:
    id: str
    views: list[View]
    candidates: list[Candidate]
    label: int
    metadata: dict
```

Each `View` should include:

- role name;
- text content;
- source path/type;
- allowed visibility.

Each `Candidate` should include:

- candidate text/patch/location;
- candidate ID;
- label index only in train/eval metadata, not visible text.

Split builder must create train/dev/test without leakage.

### Phase 2: Shared cloned agent module

Implement wrapper:

```python
class SharedClonedAgentSystem(nn.Module):
    def __init__(self, shared_agent, coordinator, config): ...
    def forward(self, batch):
        # run same shared_agent over each view/role
        # collect differentiable activations
        # feed coordinator
        # return logits and audit metadata
```

Audit requirements:

- assert all clone paths use same module object;
- store parameter IDs;
- verify activation tensors require grad;
- no detach in main path.

### Phase 3: Shared agent model

Implement `SharedTransformerAgent` with two modes:

1. pretrained local Hugging Face model + LoRA/adapters;
2. small transformer fallback.

The fallback must be clearly labeled as fallback.

For pretrained mode:

- load tokenizer/model from local cache when possible;
- configure hidden state output;
- add LoRA/adapters to target modules;
- freeze base weights if using PEFT;
- ensure adapter params are trainable;
- expose list of trainable parameter names.

### Phase 4: Coordinator models

Implement at least:

- coordinator-token cross-attention;
- role-aware self-attention.

Optional:

- bidirectional coordinator-agent attention;
- mixture/gating coordinator;
- auxiliary projection heads.

Coordinator input:

```text
clone activation + role/view embedding + optional slice embedding
```

Coordinator output:

```text
logits over candidates/classes
```

### Phase 5: Training loop

Implement explicit training loop rather than hiding all logic in Trainer.

The loop must record:

- train loss;
- train accuracy;
- dev accuracy;
- shared model grad norm;
- coordinator grad norm;
- shared model parameter delta;
- coordinator parameter delta;
- memory usage;
- early stopping decision.

Pseudocode:

```python
for batch in train_loader:
    optimizer.zero_grad(set_to_none=True)
    logits, audit = system(batch)
    loss = cross_entropy(logits, labels)
    loss.backward()
    log_grad_norms(system)
    optimizer.step()
    log_param_deltas(system)
```

### Phase 6: Baseline runners

Implement each baseline as a first-class method in the result tables.

Do not hide failed baselines.

Baselines must use the same splits, candidate sets, and evaluation metrics.

### Phase 7: Controls

Implement controls as named conditions, not ad-hoc scripts.

Required condition names:

```text
none
randomized_labels
view_masked
view_shuffled
hidden_states_shuffled_across_examples
role_labels_shuffled
physical_order_shuffled_roles_preserved
candidate_order_shuffled
```

### Phase 8: Reporting

Update `reports/REPORT.md` with a section:

```text
Real Shared-Weight Latent Coordination
```

Include:

- architecture summary;
- dataset summary;
- compute/model configuration;
- main comparison table;
- baselines table;
- controls table;
- audit table;
- learning curves;
- memory/runtime summary;
- failure analysis;
- conservative interpretation.

Save machine-readable results to:

```text
results/real_shared_weight_latent_coordination_results.json
results/real_shared_weight_latent_coordination_audit.jsonl
```

---

## 9. Acceptance Gates

Codex must treat these as gating checks.

### Gate A: Valid training graph

Pass criteria:

- clone activations require grad;
- shared adapter/model grad norm > 0 in trainable condition;
- shared adapter/model parameter delta > 0 after training;
- coordinator grad norm > 0;
- clone paths share parameter IDs.

Fail action:

- stop and fix before proceeding to claims.

### Gate B: Frozen baseline works as a control

Pass criteria:

- frozen baseline has shared model grad/delta equal or near zero;
- frozen baseline coordinator updates;
- frozen baseline is evaluated on same data.

Fail action:

- fix baseline implementation.

### Gate C: Proposed method beats frozen baseline

Pass criteria:

```text
trainable_shared_model + coordinator > frozen_shared_model + coordinator
```

Prefer statistically meaningful margin across seeds.

Fail interpretation:

> The current setup does not show that shared model training improves clone coordination.

### Gate D: Proposed method beats text-only baseline

Pass criteria:

```text
trainable latent coordination > text/output-only coordination
```

Fail interpretation:

> Latent coordination did not beat text-mediated coordination under this setup.

### Gate E: Controls collapse

Pass criteria:

- randomized labels near chance;
- hidden-state shuffle near chance;
- view masking/shuffling drops performance;
- physical order shuffle with roles preserved does not significantly hurt;
- role-label shuffle hurts where roles matter.

Fail action:

- investigate leakage or shortcut.

### Gate F: Dataset is not trivial

Pass criteria:

- single partial-view baseline below full method;
- candidate distractors not trivially separable;
- no split duplicates.

Fail action:

- harden dataset before scaling model.

---

## 10. Recommended Experiment Matrix

Start small and rigorous.

### 10.1 Minimal matrix

```text
Seeds: 3 initially, then 10 if promising
Agents: 4
Model: local pretrained small transformer + shared LoRA/adapters
Coordinator variants: cross-attention, self-attention
Task: multi-view patch-candidate selection
```

Methods:

```text
single_partial_view_agent
single_full_context_agent
text_only_multi_agent_coordinator
frozen_shared_agent_latent_coordinator
trainable_shared_agent_latent_coordinator
explicit_evidence_oracle
```

Controls:

```text
randomized_labels
view_masked
view_shuffled
hidden_states_shuffled_across_examples
role_labels_shuffled
physical_order_shuffled_roles_preserved
candidate_order_shuffled
```

### 10.2 If minimal matrix succeeds

Run:

- 10 seeds;
- 8 agents if task supports it;
- bigger model if memory allows;
- harder candidate distractors;
- another dataset source;
- more realistic code tasks.

### 10.3 If minimal matrix fails

Diagnose:

- Does shared model receive gradients?
- Does shared model overfit a tiny train set?
- Does coordinator solve explicit oracle features?
- Is the dataset too hard for the model?
- Is the dataset too easy for text-only baseline?
- Are activations extracted from the wrong layer/token?
- Is LoRA attached to effective target modules?
- Are candidate labels/candidate order handled correctly?

---

## 11. Forbidden Shortcuts

Do not do any of the following and call it success:

- train only a probe on frozen hidden states;
- train only the coordinator;
- use separate models per clone;
- let each clone see full context;
- let the coordinator see gold evidence or labels;
- choose the best test run after seeing test labels;
- omit frozen baseline;
- omit text-only baseline;
- omit gradient/parameter audit;
- hide failed coordinator variants;
- use a toy bit task as the final “real” proof;
- claim SOTA without matched baselines.

---

## 12. Conservative Interpretation Templates

Use exactly this style in the report.

### If successful

> Evidence supports that a shared pretrained transformer with trainable adapters can learn coordination-friendly internal activations when cloned across partial task views and trained through a latent coordinator under final group loss. The result is limited to this benchmark and does not yet imply open-ended multi-agent self-organization.

### If only coordinator improves

> This remains a coordinator-learning result. The experiment does not show that the shared agent model learned to cooperate, because shared model trainable parameters did not materially improve over the frozen baseline.

### If trainable and frozen are similar

> The current architecture/training setup does not demonstrate shared-weight cloned-agent cooperation. Further work should diagnose model capacity, activation extraction, dataset difficulty, and coordinator design.

### If controls fail

> The result is not valid as evidence of latent coordination because one or more leakage/shortcut controls failed.

---

## 13. Deliverables

Required files or equivalents:

```text
src/experiments/real_shared_weight_latent_coordination.py
src/experiments/run_real_shared_weight_latent_coordination.py
src/coordinators/latent_coordination.py
src/datasets/multiview_code_patch_selection.py
configs/real_shared_weight_latent_coordination_cuda.json
configs/real_shared_weight_latent_coordination_cpu_debug.json
tests/test_real_shared_weight_latent_coordination.py
reports/REPORT.md
results/real_shared_weight_latent_coordination_results.json
results/real_shared_weight_latent_coordination_audit.jsonl
```

Tests must verify:

- cloned agents share parameter IDs;
- trainable shared model receives gradients;
- frozen shared model does not receive gradients;
- trainable shared model parameter delta is nonzero;
- activations require grad;
- no train/test overlap in generated splits;
- controls are callable and produce result records;
- report generation does not crash.

---

## 14. Final Codex Checklist

Before claiming success, answer these in the report:

1. What exact model was used?
2. Which parameters were trainable?
3. Were trainable parameters shared across clones?
4. Did the shared model receive nonzero gradients?
5. Did shared model parameters change?
6. Did trainable shared model beat frozen shared model?
7. Did trainable shared model beat text-only coordination?
8. Did controls collapse as expected?
9. Did physical order shuffling preserve performance when role labels were preserved?
10. Did role-label shuffling hurt?
11. Did any single clone or partial view solve the task alone?
12. Was there output leakage?
13. Was there train/dev/test leakage?
14. How much GPU memory was used?
15. What failed variants were tried?
16. What is the most conservative valid claim?

Only after all checklist items are answered may the report include a positive conclusion.
