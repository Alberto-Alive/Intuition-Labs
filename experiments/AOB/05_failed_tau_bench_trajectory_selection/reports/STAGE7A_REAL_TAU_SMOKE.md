# Stage 7 tau-bench Trajectory Selector

## Stage 7A-real smoke status
- Decision: environment and real-pipeline smoke only; no final benchmark claim.
- Conda env: `stage7_tau`.
- tau2 source: official `sierra-research/tau2-bench` checkout at commit `0ed8c0ef0a1f6b024a0fb733e186922411874879`.
- Candidate source: loaded existing official tau2 retail final-result trajectories, because no API credentials were present for new LLM generation in this workspace.
- Domain/tasks: retail tasks `0, 1, 2, 3, 4`.
- Pool: K=8 candidates per task, 40 total candidates.
- Official evaluator command: `tau2 evaluate-trajs results/stage7a_real_tau_eval_input/results.json -o results/stage7a_real_tau_eval_output`.
- Label source: `results/stage7a_real_tau_eval_output/updated_results.json`.
- Windows note: evaluator required `PYTHONUTF8=1` because tau2 loads Results JSON with Python's default file encoding.
- Smoke limitation: oracle pass@8 is `1.0000` on this tiny loaded subset, so selector metrics are pipeline checks only.

## 1. Claim and non-claims
- Claim under test: selector-only choice among fixed K=8 candidate trajectories.
- Non-claims: no trajectory generation superiority and no end-to-end tau-bench SOTA claim.
- Final claim allowed: `False`.

## 2. Benchmark setup
- Phase: `7A`.
- Domain: `retail`.
- Labels were recomputed with the official tau2 final-state evaluator through `tau2 evaluate-trajs`.

## 3. Candidate trajectory pool construction
- Candidate pool: `results\stage7a_real_tau_labeled_candidates.jsonl`.
- K is fixed at 8. Candidate order is randomized and saved. Generator/source identity is audit metadata only and selector-visible candidate IDs were blinded from source slots.

## 4. Selector architecture
- Selected architecture: `stage5_stage7_4x4_candidate_token_direct`.
- Four roles and four avenues use candidate-token-direct scoring with shared cloned weights.

## 5. Baselines
- Random, first trajectory, best generator on dev, embedding reranker, static feature reranker, text-only reviewers, LLM judge hook, raw latent, frozen same-architecture latent, and oracle pass@8.

## 6. Controls and leakage audits
- Output leakage audit passed: `True`.
- Split leakage audit passed: `True`.
- Invariance audit passed: `True`.

## 7. Main results
- Trainable pass@1: `0.0000`.
- Oracle pass@8: `1.0000`.
- Best baseline: `static_trajectory_feature_reranker`.
- Delta vs best baseline: `-1.0000`.

## 8. Ablations
- Controls include randomized labels, candidate/evidence mismatch, cross-task evidence shuffles, schema-only, and single-evidence views.

## 9. Failure cases
- If trainable does not beat the best baseline, diagnose oracle pass@8, pool difficulty, reviewer saturation, evidence-shuffle degradation, generator identity leakage, pool size, and architecture mismatch.

## 10. Cost/token/latency
- Per-method cost/tokens/latency metadata is stored in `cost_tokens_latency` for each metric row.

## 11. Conservative conclusion
No final benchmark claim is made. Stage 7C success gates have not all passed or have not been run.
