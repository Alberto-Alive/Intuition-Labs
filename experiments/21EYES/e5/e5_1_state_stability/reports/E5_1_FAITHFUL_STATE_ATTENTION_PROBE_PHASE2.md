# E5.1 Faithful State-Attention Probe

- CUDA: True on NVIDIA GeForce RTX 5070 Ti
- Variants: G5_state_kv_mild_rank, G6_state_kv_soft_topk
- Steps: 160; seeds: [0, 1]; eval examples: 96
- Baseline N64: current-only 0.3576; post-attention residual 0.9826

## Summary

| variant | C | N64 | N128 | N256 | N512 | rank | cosine | state-KV | route shuf | slot perm | slot drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G5_state_kv_mild_rank | 64 | 0.9427 | 0.7500 | 0.3750 | 0.3177 | 7.77 | 0.942 | 32 | 0.0000 | 0.0000 | 0.0885 |
| G6_state_kv_soft_topk | 64 | 0.9740 | 0.8906 | 0.7031 | 0.5625 | 4.52 | 0.640 | 32 | 0.0104 | 0.0000 | 0.1146 |

## Notes

- `state-KV` is the mean number of fixed rolling-state slots appended as K/V at the final step.
- Positive shuffle/drop degradation means the corresponding organization mattered for the answer.
- A variant can use the state strongly while still failing slot organization if route/slot controls do not hurt.
