"""Deliverable 2 evaluation for DIGIT Extrapolation E24.

Re-runs the full training suite (deterministic, same seeds), then collects
withheld cross-entropy and per-combination accuracy for each arm and seed.
"""

from __future__ import annotations

import json
from statistics import mean, pstdev

from experiments.DIGIT.Extrapolation.e24.extrapolation.experiment import (
    WITHHELD_PART_A_VALUES,
    WITHHELD_PART_B_VALUES,
    WithheldEvaluationSummary,
    evaluate_withheld_summary,
    run_deliverable1_suite,
)


def _collect_summaries(runs: list) -> list[tuple[int, WithheldEvaluationSummary]]:
    result = []
    for run in runs:
        summary = evaluate_withheld_summary(run.model, run.dataset.test, run.config)
        result.append((run.result.seed, summary))
    return result


def _format_arm(seed_summaries: list[tuple[int, WithheldEvaluationSummary]]) -> dict:
    all_keys = [
        (a, b)
        for a in WITHHELD_PART_A_VALUES
        for b in WITHHELD_PART_B_VALUES
    ]

    per_seed = []
    for seed, s in seed_summaries:
        per_seed.append({
            "seed": seed,
            "mean_cross_entropy": s.mean_cross_entropy,
            "macro_accuracy": s.macro_accuracy,
            "per_combination_accuracy": {
                f"{k[0]},{k[1]}": v for k, v in s.per_combination_accuracy.items()
            },
        })

    ces = [s.mean_cross_entropy for _, s in seed_summaries]
    macros = [s.macro_accuracy for _, s in seed_summaries]
    agg_combo = {}
    for k in all_keys:
        vals = [s.per_combination_accuracy[k] for _, s in seed_summaries]
        agg_combo[f"{k[0]},{k[1]}"] = {"mean": mean(vals), "std": pstdev(vals)}

    return {
        "per_seed": per_seed,
        "aggregate": {
            "mean_cross_entropy": {"mean": mean(ces), "std": pstdev(ces)},
            "macro_accuracy": {"mean": mean(macros), "std": pstdev(macros)},
            "per_combination_accuracy": agg_combo,
        },
    }


def main() -> None:
    baseline_runs, e24_runs, ablation_runs = run_deliverable1_suite((0, 1, 2, 3, 4))

    baseline_summaries = _collect_summaries(baseline_runs)
    e24_summaries = _collect_summaries(e24_runs)
    ablation_summaries = _collect_summaries(ablation_runs)

    payload = {
        "baseline": _format_arm(baseline_summaries),
        "e24": _format_arm(e24_summaries),
        "learned_memory_ablation": _format_arm(ablation_summaries),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
