# Fair Utility Frontier Summary

| Dataset | Mode | Role | Primary metric | System | Mean | Retained vs raw | Delta vs best DP |
|---|---|---|---|---|---:|---:|---:|
| nist_genomics | system | headline | auroc | digit_a2_s2_c2_r2 | 0.631 | 0.979 | -0.042 |
| nist_genomics | system | headline | auroc | digit_a3_s2_c2_r2 | 0.687 | 1.067 | 0.014 |
| nist_genomics | system | headline | auroc | digit_a4_s3_c2_r2 | 0.669 | 1.038 | -0.005 |
| nist_genomics | system | headline | auroc | digit_a6_s4_c3_r3 | 0.658 | 1.021 | -0.015 |
| nist_genomics | system | headline | auroc | dp_laplace_e0p1 | 0.599 | 0.930 | -0.074 |
| nist_genomics | system | headline | auroc | dp_laplace_e0p25 | 0.620 | 0.962 | -0.054 |
| nist_genomics | system | headline | auroc | dp_laplace_e0p5 | 0.645 | 1.002 | -0.028 |
| nist_genomics | system | headline | auroc | dp_laplace_e1p0 | 0.668 | 1.037 | -0.005 |
| nist_genomics | system | headline | auroc | dp_laplace_e2p0 | 0.673 | 1.045 | 0.000 |
| nist_genomics | system | headline | auroc | dp_laplace_e3p0 | 0.673 | 1.045 | -0.000 |
| nist_genomics | system | headline | auroc | raw | 0.644 | 1.000 | -0.029 |

Notes:
- `Retained vs raw` is direction-aware: values closer to `1.0` mean the system preserves raw utility.
- `Delta vs best DP` is direction-aware: positive means the system is better than the best DP baseline for that dataset/mode.