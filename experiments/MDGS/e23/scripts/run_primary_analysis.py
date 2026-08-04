"""Run Deliverable 2 analysis for DIGIT Extrapolation E23."""

from __future__ import annotations

import json

from experiments.DIGIT.Extrapolation.e23.extrapolation.experiment import (
    SEEDS,
    WITHHELD_PART_A_VALUES,
    WITHHELD_PART_B_VALUES,
    aggregate_per_combination_accuracy,
    evaluate_withheld_summary,
    run_deliverable1_suite,
    summarise_metric,
)


def _index_by_seed(runs):
    return {run.result.seed: run for run in runs}


def _mean_std_count_positive(values: list[float]) -> dict[str, float | int]:
    summary = summarise_metric(values)
    summary["count_positive"] = sum(value > 0.0 for value in values)
    return summary


def _combo_table_payload(aggregated: dict[tuple[int, int], dict[str, float]]) -> dict[str, dict[str, dict[str, float]]]:
    table: dict[str, dict[str, dict[str, float]]] = {}
    for part_a in WITHHELD_PART_A_VALUES:
        row: dict[str, dict[str, float]] = {}
        for part_b in WITHHELD_PART_B_VALUES:
            row[str(part_b)] = aggregated[(part_a, part_b)]
        table[str(part_a)] = row
    return table


def _cross_entropy_payload(runs, summaries) -> dict:
    per_seed = [
        {
            "seed": run.result.seed,
            "withheld_mean_cross_entropy": summary.mean_cross_entropy,
        }
        for run, summary in zip(runs, summaries, strict=True)
    ]
    values = [summary.mean_cross_entropy for summary in summaries]
    return {
        "per_seed": per_seed,
        "aggregate": summarise_metric(values),
    }


def _training_fit_payload(runs) -> dict:
    per_seed = [
        {
            "seed": run.result.seed,
            "final_training_loss": run.result.final_training_loss,
            "final_training_accuracy": run.result.final_training_accuracy,
            "stopping_epoch": run.result.stopping_epoch,
        }
        for run in runs
    ]
    return {
        "per_seed": per_seed,
        "aggregate": {
            "final_training_loss": summarise_metric([run.result.final_training_loss for run in runs]),
            "final_training_accuracy": summarise_metric([run.result.final_training_accuracy for run in runs]),
            "stopping_epoch": summarise_metric([float(run.result.stopping_epoch) for run in runs]),
        },
    }


def main() -> None:
    baseline_runs, e23_runs, ablation_runs = run_deliverable1_suite(SEEDS)

    baseline_summaries = [
        evaluate_withheld_summary(run.model, run.dataset.test, run.config)
        for run in baseline_runs
    ]
    e23_summaries = [
        evaluate_withheld_summary(run.model, run.dataset.test, run.config)
        for run in e23_runs
    ]
    ablation_summaries = [
        evaluate_withheld_summary(run.model, run.dataset.test, run.config)
        for run in ablation_runs
    ]

    baseline_by_seed = _index_by_seed(baseline_runs)
    e23_by_seed = _index_by_seed(e23_runs)
    ablation_by_seed = _index_by_seed(ablation_runs)

    delta_e23_vs_baseline: list[float] = []
    delta_e23_vs_ablation: list[float] = []
    per_seed_deltas: list[dict[str, float | int]] = []

    for seed in SEEDS:
        delta_baseline = (
            e23_by_seed[seed].result.withheld_macro_accuracy
            - baseline_by_seed[seed].result.withheld_macro_accuracy
        )
        delta_ablation = (
            e23_by_seed[seed].result.withheld_macro_accuracy
            - ablation_by_seed[seed].result.withheld_macro_accuracy
        )
        delta_e23_vs_baseline.append(delta_baseline)
        delta_e23_vs_ablation.append(delta_ablation)
        per_seed_deltas.append(
            {
                "seed": seed,
                "delta_e23_vs_baseline": delta_baseline,
                "delta_e23_vs_ablation": delta_ablation,
            }
        )

    payload = {
        "deltas": {
            "per_seed": per_seed_deltas,
            "aggregate": {
                "delta_e23_vs_baseline": _mean_std_count_positive(delta_e23_vs_baseline),
                "delta_e23_vs_ablation": _mean_std_count_positive(delta_e23_vs_ablation),
            },
        },
        "withheld_cross_entropy": {
            "baseline": _cross_entropy_payload(baseline_runs, baseline_summaries),
            "e23": _cross_entropy_payload(e23_runs, e23_summaries),
            "learned_memory_ablation": _cross_entropy_payload(ablation_runs, ablation_summaries),
        },
        "training_fit_and_stopping_behaviour": {
            "baseline": _training_fit_payload(baseline_runs),
            "e23": _training_fit_payload(e23_runs),
            "learned_memory_ablation": _training_fit_payload(ablation_runs),
        },
        "per_combination_withheld_accuracy_aggregated": {
            "baseline": _combo_table_payload(aggregate_per_combination_accuracy(baseline_summaries)),
            "e23": _combo_table_payload(aggregate_per_combination_accuracy(e23_summaries)),
            "learned_memory_ablation": _combo_table_payload(aggregate_per_combination_accuracy(ablation_summaries)),
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
