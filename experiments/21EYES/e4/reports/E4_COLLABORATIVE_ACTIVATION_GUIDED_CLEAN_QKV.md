# E4 Collaborative Activation-Guided Clean-QKV

- Decision: `E4_COLLAB_NOT_BETTER_THAN_E3_1`
- C=256 reached: False
- Best beat E3.1 parent: False
- Best beat E2 reference: False
- Best collaborative mechanism: Current-to-Cache Query, Cache-to-WDA-KV
- One-way E3.1 remained better: True
- Current-to-cache path mattered: 0.325
- Cache-to-current path mattered: 0.325
- Reciprocal refinement helped: -0.025
- Support/contradiction typing remained necessary: True
- Stale/current-override typing helped: False
- Gate selectivity gap: 0.012
- Current override remained strong: 1.000
- Stage 8C run: no
- 10x attention-capacity claim: no
- Final templates used: no

## Top Configs
1. `e4_collab_clean_qkv_b_candidate_summary_00` (Current-to-Cache Query, Cache-to-WDA-KV, candidate_summary) held-out=0.722, C=128, valid=True
2. `e4_collab_clean_qkv_b_stale_override_typed_cache_05` (Current-to-Cache Query, Cache-to-WDA-KV, stale_override_typed_cache) held-out=0.723, C=128, valid=True
3. `e4_collab_clean_qkv_c_unshared_steps_04` (Reciprocal Refinement Loop, unshared_steps) held-out=0.703, C=64, valid=True
4. `e4_collab_clean_qkv_c_shared_steps_03` (Reciprocal Refinement Loop, shared_steps) held-out=0.702, C=64, valid=True

## Remaining Failure Mode
The strongest remaining failure mode is preserving cache-control sensitivity while scaling beyond the pressure-test schedule; gate selectivity improves less than current-override behavior.

## Next Recommended Experiment
Run a narrow N=256/512 study of the best collaborative mechanism with explicit gate-selectivity objectives and support/contradiction/stale typed-cache ablations, without broad search.
