"""Run Deliverable 1 training for DIGIT Extrapolation E24."""

from __future__ import annotations

import json

from experiments.DIGIT.Extrapolation.e24.extrapolation.experiment import run_deliverable1_suite


def _per_seed_payload(run) -> dict[str, float | int | None]:
    return {
        "seed": run.result.seed,
        "parameter_count": run.result.parameter_count,
        "final_training_loss": run.result.final_training_loss,
        "final_training_accuracy": run.result.final_training_accuracy,
        "stopping_epoch": run.result.stopping_epoch,
        "withheld_macro_accuracy": run.result.withheld_macro_accuracy,
        "final_g_1_raw": run.result.final_g_1_raw,
        "final_g_2_raw": run.result.final_g_2_raw,
        "final_g_1_sigmoid": run.result.final_g_1_sigmoid,
        "final_g_2_sigmoid": run.result.final_g_2_sigmoid,
    }


def _seed0_payload(run) -> dict[str, dict[str, float | int]]:
    return {
        checkpoint.label: {
            "epoch": checkpoint.epoch,
            "withheld_macro_accuracy": checkpoint.withheld_macro_accuracy,
        }
        for checkpoint in (run.seed0_checkpoint_metrics or ())
    }


def main() -> None:
    baseline_runs, e24_runs, ablation_runs = run_deliverable1_suite((0, 1, 2, 3, 4))

    seed0_baseline = next(run for run in baseline_runs if run.result.seed == 0)
    seed0_e24 = next(run for run in e24_runs if run.result.seed == 0)
    seed0_ablation = next(run for run in ablation_runs if run.result.seed == 0)

    payload = {
        "baseline": {
            "per_seed": [_per_seed_payload(run) for run in baseline_runs],
            "seed0_withheld_accuracy_checkpoints": _seed0_payload(seed0_baseline),
        },
        "e24": {
            "per_seed": [_per_seed_payload(run) for run in e24_runs],
            "seed0_withheld_accuracy_checkpoints": _seed0_payload(seed0_e24),
        },
        "learned_memory_ablation": {
            "per_seed": [_per_seed_payload(run) for run in ablation_runs],
            "seed0_withheld_accuracy_checkpoints": _seed0_payload(seed0_ablation),
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
