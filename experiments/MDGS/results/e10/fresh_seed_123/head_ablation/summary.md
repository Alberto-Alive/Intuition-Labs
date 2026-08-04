# E10 Head Ablation

Recommended mode: `mlp`

| mode | pre_f1 | post_f1 | post_fail_rec | post_unsafe | post_succ_prec | mono_v | commit_v | swing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp | 0.838 | 0.816 | 0.800 | 0.000 | 1.000 | 0.169 | 0.106 | 0.035 |
| linear_evidence_only | 0.274 | 0.274 | 0.000 | 0.000 | 0.000 | 0.225 | 0.506 | 0.305 |
| ordered_threshold | 0.241 | 0.168 | 0.000 | 0.114 | 0.886 | 0.504 | 0.103 | 0.777 |
