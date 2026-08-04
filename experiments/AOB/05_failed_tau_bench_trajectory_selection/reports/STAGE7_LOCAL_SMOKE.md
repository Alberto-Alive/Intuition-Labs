# Stage 7 tau-bench Trajectory Selector

## 1. Claim and non-claims
- Claim under test: selector-only choice among fixed K=8 candidate trajectories.
- Non-claims: no trajectory generation superiority and no end-to-end tau-bench SOTA claim.
- Final claim allowed: `False`.

## 2. Benchmark setup
- Phase: `smoke`.
- Domain: `retail`.
- Labels are official tau-bench task success/failure from the final evaluator when real tau2 evaluation is used.

## 3. Candidate trajectory pool construction
- Candidate pool: `results\stage7_local_smoke_candidates.jsonl`.
- K is fixed at 8. Candidate order is randomized and saved. Generator identity is audit metadata only.

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
- Oracle pass@8: `0.5000`.
- Best baseline: `text_only_single_reviewer`.
- Delta vs best baseline: `-0.5000`.

## 8. Ablations
- Controls include randomized labels, candidate/evidence mismatch, cross-task evidence shuffles, schema-only, and single-evidence views.

## 9. Failure cases
- If trainable does not beat the best baseline, diagnose oracle pass@8, pool difficulty, reviewer saturation, evidence-shuffle degradation, generator identity leakage, pool size, and architecture mismatch.

## 10. Cost/token/latency
- Per-method cost/tokens/latency metadata is stored in `cost_tokens_latency` for each metric row.

## 11. Conservative conclusion
No final benchmark claim is made. Stage 7C success gates have not all passed or have not been run.
