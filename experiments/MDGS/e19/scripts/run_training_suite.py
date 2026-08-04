"""Run the approved E19 Deliverable 1 training suite."""

from __future__ import annotations

import json

from experiments.DIGIT.Extrapolation.e19.extrapolation.experiment import (
    E19Config,
    evaluate_seed0_layer1_certainty_by_level,
    run_training_suite,
)


def main() -> None:
    artifacts = run_training_suite((0, 1, 2, 3, 4))
    seed0 = artifacts[0]
    payload = {
        "per_seed": [
            {
                "seed": item.result.seed,
                "final_training_loss": item.result.final_training_loss,
                "final_training_accuracy": item.result.final_training_accuracy,
                "stopping_epoch": item.result.stopping_epoch,
                "level_accuracies": {
                    summary.level_name: summary.accuracy for summary in item.result.level_accuracies
                },
            }
            for item in artifacts
        ],
        "seed0_bank_occupancy": seed0.model.layer1_bank.occupancy,
        "seed0_layer1_certainty_by_level": {
            summary.level_name: summary.mean_certainty
            for summary in evaluate_seed0_layer1_certainty_by_level(seed0.model, seed0.dataset.test, E19Config(seed=0))
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
