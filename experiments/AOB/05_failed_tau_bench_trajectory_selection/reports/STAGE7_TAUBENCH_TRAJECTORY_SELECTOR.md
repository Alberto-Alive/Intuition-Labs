# Stage 7 tau-bench Trajectory Selector

## 1. Claim and non-claims
- Claim under test: selector-only choice among fixed K=8 candidate tau-bench trajectories.
- Non-claims: no trajectory generation superiority and no end-to-end tau-bench SOTA claim.
- No final claim is made until Stage 7C gates pass on held-out retail tasks.

## 2. Benchmark setup
- Domains: retail first, airline only for optional cross-domain validation.
- Labels: official tau-bench final-state success/failure from `tau2 evaluate-trajs` or equivalent official evaluator output.
- Candidate unit: one complete agent trajectory for one official task.

## 3. Candidate trajectory pool construction
- K is fixed at 8 candidates per task.
- Raw trajectories and selector-visible trajectories are stored separately.
- Generator identity is preserved only in audit metadata and is not exposed in selector-visible text.

## 4. Selector architecture
- Shared-weight cloned-agent selector with four roles and four avenues.
- Candidate-token-direct scoring is reused from Stage 5/6.
- Frozen same-architecture, raw-latent, randomized-label, and trainable comparators are implemented.

## 5. Baselines
- Random trajectory.
- First trajectory / candidate-order baseline.
- Best generator on dev.
- Embedding reranker.
- Static trajectory-feature reranker.
- Text-only single reviewer.
- Text-only multi-agent reviewer.
- LLM judge hook when an API/model is configured.
- Raw latent selector.
- Frozen same-architecture latent selector.
- Oracle pass@8.

## 6. Controls and leakage audits
- Randomized labels.
- Candidate order shuffle with label remap.
- Physical role/avenue order shuffle.
- Candidate-only, user-goal-only, policy-only, trajectory-only, tool-observation-only, and evidence-only controls.
- Candidate/evidence mismatch and cross-task goal/policy/tool-observation shuffles.
- Hidden-state shuffle, schema-template-only, and generator-identity-only audit diagnostic.

## 7. Main results
No real Stage 7C benchmark result has been run in this workspace.

## 8. Ablations
The runner records control and ablation metrics per seed under `results/stage7_taubench_selector_results.json`.

## 9. Failure cases
If the trainable selector does not beat the best baseline, diagnose oracle pass@8, pool difficulty, reviewer saturation, evidence-shuffle degradation, generator identity leakage, pool size, and architecture mismatch. Do not tune on final test.

## 10. Cost/token/latency
Per-method estimated selector tokens and latency are recorded in each metric row. Trajectory-generation cost metadata is preserved on candidate records when generated through the tau2 adapter.

## 11. Conservative conclusion
No final benchmark claim is made. Stage 7C success gates have not passed in this workspace.
