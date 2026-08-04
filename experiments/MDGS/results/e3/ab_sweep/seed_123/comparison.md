# E3 Outcome-Context A/B Sweep

Recommendation rule: among runs with `outcome_acc >= 0.82` and `trajectory_acc >= 0.75`, minimize `success_cert_violation_rate`; break ties by `outcome_macro_f1`, `failure_recall`, then `joint_acc`.

Recommended experiment: `pattern_ctx_only`

| label | out_f1 | out_acc | traj_f1 | fail_rec | joint | succ_cert_v | ctx_flip | traj_ctx_flip | pattern_ctx_flip | pred_S_over_guard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | 0.802 | 0.877 | 0.789 | 0.824 | 0.701 | 0.000 | 0.274 | 0.195 | 0.217 | 0.493 |
| outcome_ctx_half | 0.833 | 0.899 | 0.787 | 0.824 | 0.717 | 0.000 | 0.255 | 0.129 | 0.116 | 0.662 |
| outcome_ctx_zero | 0.842 | 0.918 | 0.782 | 0.765 | 0.711 | 0.000 | 0.000 | 0.000 | 0.000 | 0.726 |
| traj_ctx_only | 0.821 | 0.887 | 0.789 | 0.824 | 0.704 | 0.000 | 0.274 | 0.274 | 0.000 | 0.569 |
| pattern_ctx_only | 0.859 | 0.918 | 0.777 | 0.882 | 0.711 | 0.000 | 0.195 | 0.000 | 0.195 | 0.733 |
