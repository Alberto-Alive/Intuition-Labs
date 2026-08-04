# E5 POF Rolling Activation-State Clean-QKV

Decision: `E5_POF_FIXED_STATE_SCALING_PROMISING`

## Summary

- CUDA: True on NVIDIA GeForce RTX 5070 Ti
- Teacher pass: True
- Best rolling-state variant: `rolling_state_organized_full_combined`
- Capacity C: 256
- Mean accuracy at N=64: rolling-state 0.9618, current-only 0.3403, post-attention residual 0.3542
- Answer-only N=64: 0.0000; delta-match N=64: 0.0000; full-combined N=64: 0.0000
- State ablation degradation at query: 0.6979
- Fixed state: True with 32 slots x 128 dims
- State-size sweep: rolling_state_organized_full_combined:N64=0.9618/C=256
- State memory bytes: 16384
- Equivalent KV bytes by length: {'32': 163840, '64': 327680, '128': 655360, '256': 1310720}

## Audit

- Current self-attention token count: 5 current-step tokens.
- Old-token KV cache in student: False
- Growing activation cache in student: False
- Mean state update norm: 0.8492138981819153
- Erase/write gate means: 0.5219181925058365 / 0.12451835162937641
- Route entropy/max/group-mass: 0.7286007155974706 / 0.3194868005812168 / 0.8415877372026443

## Required Answers

- Fixed-size state worked: True (requires capacity/control support plus ablation degradation > 0.05)
- Beat current-only: True
- Beat post-attention residual memory: True
- Teacher distillation mattered: False
- Delta matching mattered: False
- State ablation hurt: True
- State remained fixed-size: True
- Strongest failure mode: routed writes became entity-organized, but raw state vectors remain correlated, so representation-level slot separation still needs work.
- Exact next recommended experiment: freeze the organized 32x128 setting and test C512 with a stronger representation-orthogonality penalty, keeping train lengths fixed at 8/16/32 first.
