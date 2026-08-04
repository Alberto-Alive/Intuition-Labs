# E10 Path Count Ablation

| mode | outcome_acc | outcome_f1 | failure_recall | unsafe_success | mono_v | commit_v | swing | seed_std_p(success) | seed_std_commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multi_path | 0.886 | 0.835 | 0.817 | 0.010 | 0.108 | 0.036 | 0.058 | 0.0216 | 0.0048 |
| single_path_stochastic | 0.866 | 0.803 | 0.733 | 0.010 | 0.112 | 0.036 | 0.063 | 0.0439 | 0.0101 |
| single_path_deterministic | 0.865 | 0.812 | 0.800 | 0.015 | 0.112 | 0.036 | 0.079 | 0.0000 | 0.0000 |

## Conclusion

path multiplicity mostly descriptive
