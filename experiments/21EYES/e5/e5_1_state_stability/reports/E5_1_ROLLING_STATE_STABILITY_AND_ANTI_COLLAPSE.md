# E5.1 Rolling State Stability and Anti-Collapse

Decision: `E5_1_STATE_STABILITY_WEAK_SIGNAL`

## Summary

- CUDA: True on NVIDIA GeForce RTX 5070 Ti
- Replay matched C512 probe failure pattern: True
- Best variant: `R2_slot_dropout_retention`
- Accuracy-only capacity C: 128
- Control-valid stable capacity C by existing rule: 0
- Mean accuracy at N=64: best 0.9250, current-only 0.3576, post-attention residual 0.9826
- Mean/min accuracy by length: {'32': {'accuracies': [0.96875, 0.979167, 0.916667, 0.90625, 0.947917], 'mean': 0.94375, 'min': 0.90625}, '64': {'accuracies': [0.916667, 0.947917, 0.885417, 0.927083, 0.947917], 'mean': 0.925, 'min': 0.885417}, '128': {'accuracies': [0.90625, 0.9375, 0.885417, 0.895833, 0.927083], 'mean': 0.910417, 'min': 0.885417}, '256': {'accuracies': [0.802083, 0.770833, 0.729167, 0.75, 0.875], 'mean': 0.785417, 'min': 0.729167}, '512': {'accuracies': [0.458333, 0.229167, 0.208333, 0.375, 0.427083], 'mean': 0.339583, 'min': 0.208333}}
- Old-token-KV audit passed: True
- Growing-cache audit passed: True
- Controls passed: False
- State ablation degradation: 0.6917
- State shuffle degradation: 0.7375
- Route shuffle degradation: -0.0063

## State Audit

- Fixed state: True with 32 slots x 128 dims
- State memory bytes: 16384
- Equivalent KV bytes by length: {'128': 655360, '256': 1310720, '32': 163840, '512': 2621440, '64': 327680}
- State collapse score: 0.979929838180542
- State effective rank: 4.4908883762359615
- Pairwise slot cosine mean/max: 0.9799298477172852 / 0.9999941921234131
- Route entropy/max/group mass: 0.756090543270111 / 0.2903511279821396 / 0.8723147630691528

## Required Answers

- C128 stable reached: False
- C256 stable recovered: False
- C512 improved over replay: False
- Anti-collapse method that helped most: Slot dropout plus long-delay retention.
- State ablation still hurts: True
- Slot collapse improved: see state effective rank and cosine metrics in `e5_1_state_stability_state_audit.json`.
- Remaining task-family collapses: inspect per-family rows in the database; long histories usually fail first on mixed operations, stale memory, and long-range recall.
- Strongest remaining failure mode: seed-dependent state organization drift under long distractor/update chains.
- Exact next recommended experiment: promote only variants with higher effective rank and meaningful state-shuffle degradation into 320/640-step multi-seed screens; keep train lengths fixed at 8/16/32.
