"""Run the full E21 five-seed Phase 2 comparison without a pass/fail conclusion."""

from __future__ import annotations

import json

from experiments.DIGIT.Extrapolation.e21.extrapolation.config import E21Config
from experiments.DIGIT.Extrapolation.e21.extrapolation.experiment import (
    run_phase1_suite,
    run_phase2_diagnostics,
)


def _first_epoch_reaching_threshold(values: list[float], threshold: float) -> int:
    """Return the first 1-based epoch whose value reaches the threshold, else 51."""
    for index, value in enumerate(values, start=1):
        if value >= threshold:
            return index
    return 51


def main() -> None:
    config = E21Config()
    phase1_artifacts = run_phase1_suite()
    diagnostics, baseline_runs, e21_runs = run_phase2_diagnostics(phase1_artifacts)

    per_seed = []
    for diagnostic, baseline_run, e21_run in zip(diagnostics, baseline_runs, e21_runs, strict=True):
        baseline_union_curve = [item.rare_union_novel_test_accuracy for item in baseline_run.epoch_results]
        e21_union_curve = [item.rare_union_novel_test_accuracy for item in e21_run.epoch_results]
        baseline_novel_curve = [item.novel_test_accuracy for item in baseline_run.epoch_results]
        e21_novel_curve = [item.novel_test_accuracy for item in e21_run.epoch_results]

        baseline_train99 = diagnostic.baseline_first_epoch_train_accuracy_ge_0_99
        e21_train99 = diagnostic.e21_first_epoch_train_accuracy_ge_0_99
        baseline_union75 = _first_epoch_reaching_threshold(baseline_union_curve, config.phase2_accuracy_threshold)
        e21_union75 = _first_epoch_reaching_threshold(e21_union_curve, config.phase2_accuracy_threshold)
        baseline_novel75 = _first_epoch_reaching_threshold(baseline_novel_curve, config.phase2_accuracy_threshold)
        e21_novel75 = _first_epoch_reaching_threshold(e21_novel_curve, config.phase2_accuracy_threshold)

        per_seed.append(
            {
                "seed": diagnostic.seed,
                "epoch_to_0_99_training_accuracy": {
                    "baseline": baseline_train99,
                    "e21": e21_train99,
                    "delta": baseline_train99 - e21_train99,
                },
                "epoch_to_0_75_rare_union_novel_accuracy": {
                    "baseline": baseline_union75,
                    "e21": e21_union75,
                    "delta": baseline_union75 - e21_union75,
                },
                "epoch_to_0_75_novel_only_accuracy": {
                    "baseline": baseline_novel75,
                    "e21": e21_novel75,
                    "delta": baseline_novel75 - e21_novel75,
                },
                "epoch50_accuracy": {
                    "baseline_rare": baseline_run.epoch_results[-1].rare_test_accuracy,
                    "e21_rare": e21_run.epoch_results[-1].rare_test_accuracy,
                    "baseline_novel": baseline_run.epoch_results[-1].novel_test_accuracy,
                    "e21_novel": e21_run.epoch_results[-1].novel_test_accuracy,
                },
            }
        )

    print(json.dumps({"per_seed": per_seed}, indent=2))


if __name__ == "__main__":
    main()
