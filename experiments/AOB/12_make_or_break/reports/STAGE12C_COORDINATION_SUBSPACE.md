# Stage 12c Coordination Subspace

## Source

- Preset: `stage5_final`
- Device: `cuda`
- Cache reused: `YES`
- Example-seed rows: `10240`
- Unique examples: `1024`

## Gain Direction Scan

- Available positive seeds: `[31, 37]`
- Gain-positive counts by seed: `{'30': 0, '31': 704, '32': 0, '33': 0, '34': 0, '35': 0, '36': 0, '37': 309, '38': 0, '39': 0}`
- Best transfer site: `message_roles_flat`
- Best intervention-capable site: `token_summary_best_avenue_flat`

| site | mean test bacc | min test bacc | mean test AUC | difficulty mean test bacc | intervention |
|---|---:|---:|---:|---:|---|
| message_roles_flat | 0.5000 | 0.5000 | 0.5426 | 0.9969 | NO |
| raw_pooled_roles_flat | 0.5000 | 0.5000 | 0.5426 | 0.9969 | NO |
| raw_pooled_avenues_mean_roles_flat | 0.5000 | 0.5000 | 0.5423 | 0.9969 | NO |
| token_summary_avenues_mean_roles_flat | 0.5000 | 0.5000 | 0.5362 | 0.9969 | NO |
| token_summary_best_avenue_flat | 0.5000 | 0.5000 | 0.5215 | 0.9969 | YES |
| msg_best_avenue_flat | 0.5000 | 0.5000 | 0.5163 | 0.9969 | NO |
| raw_pooled_best_avenue_flat | 0.5000 | 0.5000 | 0.4939 | 0.9969 | NO |

## Causal Tests

- Token intervention site: `token_summary_best_avenue_flat`

### Source Seed 31

- Gain-positive examples: `704`
- Direction norm: `3.0017`
- Ablation baseline full accuracy on source positives: `1.0000`

| scale | direction | shuffled | orthogonal | noise |
|---:|---:|---:|---:|---:|
| 0.25 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 0.50 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 1.00 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2.00 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 4.00 | 0.3523 | 1.0000 | 1.0000 | 0.9986 |

- Source best-single substitution baseline accuracy: `0.0000`
| scale | direction | shuffled | orthogonal | noise |
|---:|---:|---:|---:|---:|
| 0.25 | 0.0000 | 0.0071 | 0.0000 | 0.0014 |
| 0.50 | 0.0000 | 0.0099 | 0.0000 | 0.0057 |
| 1.00 | 0.0000 | 0.0114 | 0.0028 | 0.0085 |
| 2.00 | 0.0000 | 0.0114 | 0.0000 | 0.0099 |
| 4.00 | 0.0014 | 0.0114 | 0.0085 | 0.0199 |

- Target seed `35` matched negatives: `203`
  Baseline full accuracy: `0.0000`
| scale | direction | shuffled | orthogonal | noise |
|---:|---:|---:|---:|---:|
| 0.25 | 0.0000 | 0.0099 | 0.0000 | 0.0099 |
| 0.50 | 0.0049 | 0.0197 | 0.0000 | 0.0099 |
| 1.00 | 0.0000 | 0.0197 | 0.0000 | 0.0837 |
| 2.00 | 0.0000 | 0.0148 | 0.0049 | 0.1330 |
| 4.00 | 0.0000 | 0.3990 | 0.0049 | 0.1281 |

### Source Seed 37

- Gain-positive examples: `309`
- Direction norm: `0.9008`
- Ablation baseline full accuracy on source positives: `1.0000`

| scale | direction | shuffled | orthogonal | noise |
|---:|---:|---:|---:|---:|
| 0.25 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 0.50 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 1.00 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2.00 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 4.00 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

- Source best-single substitution baseline accuracy: `0.0000`
| scale | direction | shuffled | orthogonal | noise |
|---:|---:|---:|---:|---:|
| 0.25 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.50 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 1.00 | 0.0000 | 0.0000 | 0.0000 | 0.0032 |
| 2.00 | 0.0000 | 0.0000 | 0.0097 | 0.0809 |
| 4.00 | 0.0000 | 0.0000 | 0.3981 | 0.1392 |

- Target seed `35` matched negatives: `83`
  Baseline full accuracy: `0.0000`
| scale | direction | shuffled | orthogonal | noise |
|---:|---:|---:|---:|---:|
| 0.25 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.50 | 0.0000 | 0.0120 | 0.0000 | 0.0000 |
| 1.00 | 0.0000 | 0.0120 | 0.0000 | 0.0000 |
| 2.00 | 0.0000 | 0.0241 | 0.0000 | 0.0120 |
| 4.00 | 0.0000 | 0.0241 | 0.0000 | 0.0843 |

## Verdict

- Stage 12c supported: `NO`
- Mean token-site transfer balanced accuracy: `0.5000`
- Token-site minus difficulty balanced accuracy: `-0.4969`
- Causal hit: `NO`
- Meaning: The aggressive within-example direction search did not produce a clean causal coordination direction at the current summary site.
- Recommended next step: Go lower-level next: layerwise token-state or attention-pattern search rather than more pooled-vector analysis.
