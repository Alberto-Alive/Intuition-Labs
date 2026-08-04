"""Run the approved E21 Deliverable 2 diagnostics before primary analysis."""

from __future__ import annotations

import json

from experiments.DIGIT.Extrapolation.e21.extrapolation.experiment import (
    run_phase1_suite,
    run_phase2_diagnostics,
)


def main() -> None:
    phase1_artifacts = run_phase1_suite()
    diagnostics, _baseline_runs, _e21_runs = run_phase2_diagnostics(phase1_artifacts)

    seed0 = next(item for item in diagnostics if item.seed == 0)
    payload = {
        "seed0_retrieval_fraction_by_epoch": seed0.seed0_e21_retrieval_fraction_by_epoch,
        "seed0_epoch1_mean_novel_certainty": seed0.seed0_e21_epoch1_mean_novel_certainty,
        "first_epoch_training_accuracy_ge_0_99": [
            {
                "seed": item.seed,
                "baseline": item.baseline_first_epoch_train_accuracy_ge_0_99,
                "e21": item.e21_first_epoch_train_accuracy_ge_0_99,
            }
            for item in diagnostics
        ],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
