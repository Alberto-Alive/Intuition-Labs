# E5.1 Faithful State-Attention Probe

- CUDA: True on NVIDIA GeForce RTX 5070 Ti
- Variants: H5_geometry_static_heavy, H10_possibility_geometry_dense, H11_possibility_static_heavy, H12_possibility_high_capacity, H13_possibility_sparse_write
- Steps: 160; seeds: [0, 1]; eval examples: 96
- Baseline N64: current-only 0.3576; post-attention residual 0.9826

## Summary

| variant | C | N64 | N128 | N256 | N512 | rank | cosine | geom mean | geom active | state-KV | route shuf | slot perm | slot drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H5_geometry_static_heavy | 128 | 0.9635 | 0.9375 | 0.8646 | 0.7344 | 2.55 | 0.785 | 0.000 | 0.000 | 32 | 0.0052 | 0.0000 | 0.0312 |
| H10_possibility_geometry_dense | 128 | 0.9688 | 0.9688 | 0.8594 | 0.6510 | 3.72 | 0.828 | 0.546 | 0.611 | 32 | -0.0052 | 0.0000 | 0.1146 |
| H11_possibility_static_heavy | 64 | 0.9635 | 0.8021 | 0.7135 | 0.5208 | 2.66 | 0.787 | 0.536 | 0.589 | 32 | 0.0521 | 0.0000 | 0.3646 |
| H12_possibility_high_capacity | 32 | 0.9115 | 0.7604 | 0.6198 | 0.5312 | 2.58 | 0.828 | 0.545 | 0.595 | 32 | 0.0104 | -0.0052 | 0.2292 |
| H13_possibility_sparse_write | 128 | 0.9688 | 0.9167 | 0.7604 | 0.6146 | 2.17 | 0.840 | 0.542 | 0.597 | 32 | 0.0000 | 0.0000 | 0.4115 |

## Notes

- `state-KV` is the mean number of fixed rolling-state slots appended as K/V at the final step.
- `geom mean` is average sigmoid gate mass over learned possibility vectors; `geom active` is the fraction above 0.5.
- Positive shuffle/drop degradation means the corresponding organization mattered for the answer.
- A variant can use the state strongly while still failing slot organization if route/slot controls do not hurt.
