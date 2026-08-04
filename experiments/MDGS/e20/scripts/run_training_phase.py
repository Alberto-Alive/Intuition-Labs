"""Run the approved E20 Deliverable 1 training phase."""

from __future__ import annotations

import json

from experiments.DIGIT.Extrapolation.e20.extrapolation.experiment import run_training_phase


def main() -> None:
    frozen_artifacts = run_training_phase()
    payload = {
        "per_seed": [
            {
                "seed": item.result.seed,
                "final_training_loss": item.result.final_training_loss,
                "final_training_accuracy": item.result.final_training_accuracy,
                "stopping_epoch": item.result.stopping_epoch,
                "checkpoint0_layer1_novel_certainty": item.result.checkpoint0_layer1_novel_certainty,
                "checkpoint0_layer2_novel_certainty": item.result.checkpoint0_layer2_novel_certainty,
                "layer1_archived_difference": item.result.layer1_archived_difference,
                "layer2_archived_difference": item.result.layer2_archived_difference,
            }
            for item in frozen_artifacts
        ]
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
