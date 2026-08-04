"""Run the approved E17 Deliverable 1 matched training suite."""

from __future__ import annotations

import json

from experiments.DIGIT.Extrapolation.e17.extrapolation.experiment import (
    build_seed0_trajectory_report,
    run_matched_training_suite,
)


def main() -> None:
    artifacts = run_matched_training_suite((0, 1, 2, 3, 4))
    payload = {
        "per_seed": [
            {
                "seed": item.seed,
                "final_training_loss": item.comparison.final_training_loss,
                "final_training_accuracy": item.comparison.final_training_accuracy,
                "seen_test_accuracy": item.comparison.seen_test_accuracy,
                "unseen_test_accuracy": item.comparison.unseen_test_accuracy,
                "stopping_epoch": item.comparison.stopping_epoch,
                "e17_normalized_area": item.comparison.e17_normalized_area,
                "e16_normalized_area": item.comparison.e16_normalized_area,
                "delta_area": item.comparison.delta_area,
            }
            for item in artifacts
        ],
        "seed0_trajectory": [
            {
                "label": point.label,
                "e16_epoch": point.e16_epoch,
                "e16_training_accuracy": point.e16_training_accuracy,
                "e17_epoch": point.e17_epoch,
                "e17_training_accuracy": point.e17_training_accuracy,
            }
            for point in build_seed0_trajectory_report(artifacts[0])
        ],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
