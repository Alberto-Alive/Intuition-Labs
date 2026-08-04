# Stage 6: Latent Patch Selector

## Scope

- Selector only: fixed candidate patches are supplied; this experiment does not claim patch generation.
- Pool size: K=8 candidate patches per task.
- Primary metric: unconditional pass@1, with conditional selector accuracy reported only where oracle pass@8 > 0.
- Phase: `smoke`.
- Candidate pool: `results\stage6_candidate_pools.jsonl`.

## Dataset

- Summary: `{"candidate_source_agents": ["generator_0", "generator_1", "generator_2", "generator_3"], "dataset_names": ["synthetic_swe_style_stage6_smoke"], "num_avenues": 4, "num_candidates": 8, "num_roles": 4, "oracle_empty_tasks": 3, "oracle_pass_at_8": 0.875, "publishable_proof_dataset": false, "repos": ["synthetic/repo_alpha", "synthetic/repo_beta", "synthetic/repo_delta", "synthetic/repo_gamma"], "selector_only_no_patch_generation_claim": true, "split_sizes": {"dev": 5, "test": 5, "train": 14}}`
- Validation passes: `True`

## Results

| seed | architecture | trainable pass@1 | frozen pass@1 | best non-oracle baseline | oracle pass@8 | order invariant | leakage |
|---:|---|---:|---:|---:|---:|---|---|
| 0 | `stage5_4x4_avenue_candidate_token_direct` | 0.0000 | 0.0000 | 0.2000 | 0.8000 | `True` | `True` |

## Summary

- Completed seeds: `1`
- Mean trainable pass@1: `0.0000`
- Mean best non-oracle baseline pass@1: `0.2000`
- Mean delta vs best baseline: `-0.2000`
- Bootstrap 95% CI: `[-0.2, -0.2]`
- Final gates pass: `False`

## Conclusion

No final Stage 6 claim is made from this run. The report is a pipeline or validation artifact unless the 6C gates pass on held-out tasks.
