"""Run the approved E21 Deliverable 1 Phase 1 training suite."""

from __future__ import annotations

import json

from experiments.DIGIT.Extrapolation.e19.extrapolation.data import LEVEL_NAMES
from experiments.DIGIT.Extrapolation.e21.extrapolation.experiment import run_phase1_suite


def main() -> None:
    artifacts = run_phase1_suite()
    payload = {
        "per_seed": [
            {
                "seed": item.result.seed,
                "final_phase1_training_loss": item.result.final_training_loss,
                "final_phase1_training_accuracy": item.result.final_training_accuracy,
                "phase1_stopping_epoch": item.result.stopping_epoch,
                "core_bank_fraction": item.result.core_bank_fraction,
                "familiar_bank_fraction": item.result.familiar_bank_fraction,
                "mean_certainty_by_level": {
                    summary.level_name: summary.mean_certainty for summary in item.result.certainty_by_level
                },
                "novel_below_0_95": next(
                    summary.mean_certainty < 0.95
                    for summary in item.result.certainty_by_level
                    if summary.level_name == LEVEL_NAMES[4]
                ),
                "rare_above_0_95": next(
                    summary.mean_certainty > 0.95
                    for summary in item.result.certainty_by_level
                    if summary.level_name == LEVEL_NAMES[3]
                ),
            }
            for item in artifacts
        ]
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
