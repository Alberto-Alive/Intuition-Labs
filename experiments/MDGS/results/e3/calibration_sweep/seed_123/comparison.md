# E3 Outcome Calibration Sweep

Recommendation rule: keep runs within a small quality delta of `pattern_guard_x2`, then minimize over-guard on incorrect predicted `SUCCESS`, then overall over-guard, then mean excess, then `success_brier`.

Recommended experiment: `pattern_ctx_06`

| label | out_f1 | out_acc | traj_f1 | fail_rec | joint | pred_S_over_guard | pred_S_wrong_over | pred_S_excess | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pattern_guard_x2 | 0.859 | 0.918 | 0.777 | 0.882 | 0.711 | 0.683 | 0.000 | 0.081 | 0.613 | 0.679 |
| pattern_guard_x25 | 0.859 | 0.918 | 0.780 | 0.882 | 0.714 | 0.683 | 0.000 | 0.075 | 0.613 | 0.679 |
| pattern_ctx_06 | 0.859 | 0.918 | 0.777 | 0.882 | 0.708 | 0.700 | 1.000 | 0.083 | 0.613 | 0.678 |
| pattern_ctx_half | 0.856 | 0.915 | 0.780 | 0.882 | 0.714 | 0.721 | 1.000 | 0.082 | 0.613 | 0.678 |
| pattern_ctx_quarter | 0.859 | 0.918 | 0.782 | 0.882 | 0.717 | 0.742 | 1.000 | 0.079 | 0.614 | 0.679 |
| pattern_guard_x3 | 0.859 | 0.918 | 0.780 | 0.882 | 0.714 | 0.683 | 0.000 | 0.067 | 0.614 | 0.681 |
| pattern_guard_x35 | 0.855 | 0.915 | 0.780 | 0.882 | 0.711 | 0.678 | 0.000 | 0.061 | 0.614 | 0.682 |
| pattern_guard_x4 | 0.852 | 0.912 | 0.780 | 0.882 | 0.708 | 0.672 | 0.000 | 0.056 | 0.615 | 0.684 |
| pattern_guard_x2_margin_003 | 0.855 | 0.915 | 0.777 | 0.882 | 0.708 | 0.695 | 0.000 | 0.080 | 0.613 | 0.679 |
| pattern_guard_x3_margin_003 | 0.855 | 0.915 | 0.780 | 0.882 | 0.711 | 0.695 | 0.000 | 0.064 | 0.614 | 0.682 |
| pattern_guard_x2_smooth_001 | 0.838 | 0.906 | 0.782 | 0.882 | 0.711 | 0.690 | 0.000 | 0.072 | 0.610 | 0.679 |
| pattern_guard_x2_smooth_002 | 0.832 | 0.903 | 0.782 | 0.882 | 0.714 | 0.655 | 0.000 | 0.063 | 0.607 | 0.679 |
| pattern_ctx_06_guard_x3 | 0.859 | 0.918 | 0.780 | 0.882 | 0.714 | 0.683 | 0.000 | 0.068 | 0.613 | 0.681 |
| pattern_ctx_half_guard_x2 | 0.856 | 0.915 | 0.780 | 0.882 | 0.714 | 0.721 | 1.000 | 0.082 | 0.613 | 0.678 |
| pattern_ctx_06_guard_x2_smooth_001 | 0.839 | 0.909 | 0.782 | 0.882 | 0.717 | 0.667 | 0.000 | 0.071 | 0.610 | 0.678 |
