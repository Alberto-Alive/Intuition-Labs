"""Run Deliverable 1 training for DIGIT Extrapolation E22."""

from __future__ import annotations

import json

from experiments.DIGIT.Extrapolation.e22.extrapolation.experiment import run_deliverable1_suite


def _level_accuracy_payload(level_accuracies) -> dict[str, float]:
    return {summary.level_name: summary.accuracy for summary in level_accuracies}


def _certainty_payload(certainty_summaries) -> dict[str, float]:
    return {summary.level_name: summary.mean_certainty for summary in certainty_summaries}


def main() -> None:
    baseline_runs, e22_runs = run_deliverable1_suite((0, 1, 2, 3, 4))
    seed0_e22 = next(run for run in e22_runs if run.result.seed == 0)

    payload = {
        "baseline": {
            "per_seed": [
                {
                    "seed": run.result.seed,
                    "parameter_count": run.result.parameter_count,
                    "final_training_loss": run.result.final_training_loss,
                    "final_training_accuracy": run.result.final_training_accuracy,
                    "stopping_epoch": run.result.stopping_epoch,
                    "overall_test_accuracy": run.result.overall_test_accuracy,
                    "level_accuracies": _level_accuracy_payload(run.result.level_accuracies),
                }
                for run in baseline_runs
            ],
        },
        "e22": {
            "per_seed": [
                {
                    "seed": run.result.seed,
                    "parameter_count": run.result.parameter_count,
                    "final_training_loss": run.result.final_training_loss,
                    "final_training_accuracy": run.result.final_training_accuracy,
                    "stopping_epoch": run.result.stopping_epoch,
                    "overall_test_accuracy": run.result.overall_test_accuracy,
                    "level_accuracies": _level_accuracy_payload(run.result.level_accuracies),
                    "layer1_occupancy": run.result.layer1_occupancy,
                    "layer2_occupancy": run.result.layer2_occupancy,
                    "layer1_mean_certainty_by_level": _certainty_payload(run.result.layer1_certainty_by_level),
                    "layer2_mean_certainty_by_level": _certainty_payload(run.result.layer2_certainty_by_level),
                }
                for run in e22_runs
            ],
            "seed0": {
                "layer1_full_occupancy": (
                    {
                        "epoch": seed0_e22.seed0_layer1_full_occupancy.epoch,
                        "step": seed0_e22.seed0_layer1_full_occupancy.step,
                    }
                    if seed0_e22.seed0_layer1_full_occupancy is not None
                    else None
                ),
                "layer2_full_occupancy": (
                    {
                        "epoch": seed0_e22.seed0_layer2_full_occupancy.epoch,
                        "step": seed0_e22.seed0_layer2_full_occupancy.step,
                    }
                    if seed0_e22.seed0_layer2_full_occupancy is not None
                    else None
                ),
                "layer1_mean_certainty_by_level_checkpoints": {
                    checkpoint.label: {
                        "epoch": checkpoint.epoch,
                        **_certainty_payload(checkpoint.certainty_by_level),
                    }
                    for checkpoint in (seed0_e22.seed0_layer1_checkpoints or ())
                },
            },
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
