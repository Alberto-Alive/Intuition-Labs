# AOB goal-relative task-map experiment

This experiment tests whether a learned relational representation of task parts, expressed relative to shared goal tokens and the arrangement of other task parts, supports scalable non-duplicating agent coordination on unseen task geometries.

It compares the previous oracle and raw-space projection baselines with a static learned map, a pooled-goal map, and a permutation-equivariant relational task map.

Run:

```bash
python test_experiment.py
python run_taskmap_experiment.py
```
