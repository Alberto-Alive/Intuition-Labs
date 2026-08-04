# Validated Candidate-Token Direct Coordination

Status: `validated_controlled_foundation`

Track: `mainline`

Source: `W:\HocusPocus\armyofbots - m4`

## Research Question

Can shared-weight latent coordination beat visible-output baselines and an exact frozen comparator on the controlled real-code import-restoration benchmark?

## Conclusion

Yes, within the controlled benchmark only: mean trainable accuracy 0.7892 vs frozen 0.1518, delta 0.6374, with shortcut, leakage, invariance, and update/frozen audits passing.

## Why This Experiment Is Kept

This is the clean foundation that later branches compare against.

## Folder Layout

- `code/`: experiment-specific source files that first appear or change at this point in the ordered path.
- `tests/`: experiment-specific tests.
- `configs/`: configs introduced or changed for this experiment.
- `reports/`: source reports and written conclusions.
- `results/`: top-level result artifacts, excluding recursive checkpoint/vendor/cache trees.
- `provenance/`: original source path and copied-file manifest references.
