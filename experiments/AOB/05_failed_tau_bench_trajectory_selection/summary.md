# Failed tau-bench Trajectory Selection

Status: `failed_final_validation`

Track: `mainline`

Source: `W:\HocusPocus\armyofbots - m4 but picpalac choose - τ-bench trajectory selection`

## Research Question

Can latent trajectory selection beat frozen and generator baselines on tau-bench candidate trajectories?

## Conclusion

No: Stage 7C failed final validation. Trainable pass@1 was 0.4292, frozen was 0.4833, and the best non-oracle baseline was 0.5417. No tau-bench success claim is allowed.

## Why This Experiment Is Kept

Important failed external-style benchmark pivot with forensics.

## Folder Layout

- `code/`: experiment-specific source files that first appear or change at this point in the ordered path.
- `tests/`: experiment-specific tests.
- `configs/`: configs introduced or changed for this experiment.
- `reports/`: source reports and written conclusions.
- `results/`: top-level result artifacts, excluding recursive checkpoint/vendor/cache trees.
- `provenance/`: original source path and copied-file manifest references.

