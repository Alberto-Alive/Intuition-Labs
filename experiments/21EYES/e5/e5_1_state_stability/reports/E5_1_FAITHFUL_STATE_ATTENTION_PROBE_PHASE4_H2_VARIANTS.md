# E5.1 Faithful State-Attention Probe

- CUDA: True on NVIDIA GeForce RTX 5070 Ti
- Variants: H2_geometry_soft_topk, H4_geometry_no_rank, H5_geometry_static_heavy, H6_geometry_dynamic_heavy, H7_geometry_topk4, H8_geometry_topk16, H9_geometry_temp_anneal
- Steps: 160; seeds: [0, 1]; eval examples: 96
- Baseline N64: current-only 0.3576; post-attention residual 0.9826

## Summary

| variant | C | N64 | N128 | N256 | N512 | rank | cosine | state-KV | route shuf | slot perm | slot drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H2_geometry_soft_topk | 64 | 0.9531 | 0.8333 | 0.7604 | 0.6719 | 5.09 | 0.807 | 32 | -0.0052 | 0.0000 | 0.0260 |
| H4_geometry_no_rank | 64 | 0.9635 | 0.8177 | 0.6198 | 0.5781 | 3.06 | 0.827 | 32 | -0.0052 | 0.0000 | 0.3177 |
| H5_geometry_static_heavy | 128 | 0.9635 | 0.9375 | 0.8646 | 0.7344 | 2.55 | 0.785 | 32 | 0.0052 | 0.0000 | 0.0312 |
| H6_geometry_dynamic_heavy | 64 | 0.9583 | 0.8490 | 0.7396 | 0.5938 | 2.73 | 0.801 | 32 | -0.0052 | 0.0000 | 0.5156 |
| H7_geometry_topk4 | 64 | 0.9167 | 0.8542 | 0.7188 | 0.5729 | 3.38 | 0.819 | 32 | -0.0156 | -0.0052 | 0.2240 |
| H8_geometry_topk16 | 64 | 0.9531 | 0.5521 | 0.3646 | 0.3385 | 2.56 | 0.903 | 32 | 0.1406 | 0.0000 | 0.3958 |
| H9_geometry_temp_anneal | 64 | 0.9740 | 0.8385 | 0.6302 | 0.5729 | 2.47 | 0.844 | 32 | 0.0000 | 0.0000 | 0.0208 |

## Notes

- `state-KV` is the mean number of fixed rolling-state slots appended as K/V at the final step.
- Positive shuffle/drop degradation means the corresponding organization mattered for the answer.
- A variant can use the state strongly while still failing slot organization if route/slot controls do not hurt.
