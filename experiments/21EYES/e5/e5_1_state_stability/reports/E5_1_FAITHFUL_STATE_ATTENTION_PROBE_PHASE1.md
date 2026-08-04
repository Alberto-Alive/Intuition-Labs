# E5.1 Faithful State-Attention Probe

- CUDA: True on NVIDIA GeForce RTX 5070 Ti
- Variants: R2_slot_dropout_retention, G1_state_kv_meanless, G2_state_kv_cross_update, G3_state_kv_strict_topk, G4_state_kv_rank_pressure
- Steps: 160; seeds: [0, 1]; eval examples: 96
- Baseline N64: current-only 0.3576; post-attention residual 0.9826

## Summary

| variant | C | N64 | N128 | N256 | N512 | rank | cosine | state-KV | route shuf | slot perm | slot drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R2_slot_dropout_retention | 64 | 0.9115 | 0.7865 | 0.4635 | 0.2344 | 8.08 | 0.994 | 0 | -0.0156 | 0.0000 | -0.0104 |
| G1_state_kv_meanless | 128 | 0.9688 | 0.9583 | 0.8438 | 0.7604 | 2.37 | 0.958 | 32 | 0.0000 | 0.0000 | 0.2292 |
| G2_state_kv_cross_update | 64 | 0.9375 | 0.7656 | 0.5990 | 0.5052 | 3.94 | 0.589 | 32 | 0.0365 | 0.0052 | 0.3854 |
| G3_state_kv_strict_topk | 0 | 0.7396 | 0.7500 | 0.7031 | 0.5573 | 7.84 | 0.201 | 32 | 0.2135 | 0.0052 | 0.0365 |
| G4_state_kv_rank_pressure | 64 | 0.9115 | 0.5573 | 0.4479 | 0.2604 | 10.83 | 0.395 | 32 | 0.0781 | 0.0052 | 0.2760 |

## Notes

- `state-KV` is the mean number of fixed rolling-state slots appended as K/V at the final step.
- Positive shuffle/drop degradation means the corresponding organization mattered for the answer.
- A variant can use the state strongly while still failing slot organization if route/slot controls do not hurt.
