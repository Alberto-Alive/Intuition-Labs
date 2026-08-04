# E5.1 Faithful State-Attention Probe

- CUDA: True on NVIDIA GeForce RTX 5070 Ti
- Variants: G1_state_kv_meanless, G7_g1_soft_repair, H1_learned_geometry_keys, H2_geometry_soft_topk, H3_geometry_cross_update
- Steps: 160; seeds: [0, 1]; eval examples: 96
- Baseline N64: current-only 0.3576; post-attention residual 0.9826

## Summary

| variant | C | N64 | N128 | N256 | N512 | rank | cosine | state-KV | route shuf | slot perm | slot drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G1_state_kv_meanless | 128 | 0.9688 | 0.9583 | 0.8438 | 0.7604 | 2.37 | 0.958 | 32 | 0.0000 | 0.0000 | 0.2292 |
| G7_g1_soft_repair | 64 | 0.9844 | 0.8438 | 0.5521 | 0.3281 | 4.15 | 0.958 | 32 | 0.0104 | 0.0000 | 0.2552 |
| H1_learned_geometry_keys | 64 | 0.9063 | 0.8177 | 0.6563 | 0.5781 | 2.45 | 0.977 | 32 | -0.0104 | 0.0000 | 0.3698 |
| H2_geometry_soft_topk | 64 | 0.9531 | 0.8333 | 0.7604 | 0.6719 | 5.09 | 0.807 | 32 | 0.0000 | 0.0000 | 0.0260 |
| H3_geometry_cross_update | 64 | 0.9167 | 0.7240 | 0.5469 | 0.5417 | 5.98 | 0.665 | 32 | 0.0000 | 0.0000 | 0.3177 |

## Notes

- `state-KV` is the mean number of fixed rolling-state slots appended as K/V at the final step.
- Positive shuffle/drop degradation means the corresponding organization mattered for the answer.
- A variant can use the state strongly while still failing slot organization if route/slot controls do not hurt.
