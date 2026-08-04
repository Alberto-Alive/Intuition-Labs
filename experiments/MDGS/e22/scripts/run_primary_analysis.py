"""Run Deliverable 2 for DIGIT Extrapolation E22."""

from __future__ import annotations

import json
import os

import numpy as np

from experiments.DIGIT.Extrapolation.e22.extrapolation.experiment import (
    aggregate_primary_analysis,
    analyze_primary_seed,
    run_e22_training_suite,
)

DELIVERABLE1_PATH = os.environ.get("E22_DELIVERABLE1_PATH", "/tmp/e22-deliverable1-gpu.json")
SEEDS = (0, 1, 2, 3, 4)


def _level_means_payload(level_means: tuple[float, float, float, float]) -> dict[str, float]:
    return {
        "core": level_means[0],
        "familiar": level_means[1],
        "rare": level_means[2],
        "novel": level_means[3],
    }


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.array(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
    }


def _load_deliverable1_payload() -> dict:
    with open(DELIVERABLE1_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _index_by_seed(items: list[dict]) -> dict[int, dict]:
    return {int(item["seed"]): item for item in items}


def _secondary_payload(deliverable1_payload: dict) -> dict:
    baseline_by_seed = _index_by_seed(deliverable1_payload["baseline"]["per_seed"])
    e22_by_seed = _index_by_seed(deliverable1_payload["e22"]["per_seed"])

    training_speed_per_seed = []
    novel_accuracy_per_seed = []

    baseline_train_accuracies: list[float] = []
    e22_train_accuracies: list[float] = []
    baseline_stopping_epochs: list[float] = []
    e22_stopping_epochs: list[float] = []
    baseline_novel_accuracies: list[float] = []
    e22_novel_accuracies: list[float] = []

    e22_at_least_baseline_train_accuracy_count = 0
    e22_same_or_fewer_epochs_count = 0
    e22_exceeds_baseline_novel_accuracy_count = 0

    for seed in SEEDS:
        baseline_item = baseline_by_seed[seed]
        e22_item = e22_by_seed[seed]

        baseline_train_accuracy = float(baseline_item["final_training_accuracy"])
        e22_train_accuracy = float(e22_item["final_training_accuracy"])
        baseline_stopping_epoch = int(baseline_item["stopping_epoch"])
        e22_stopping_epoch = int(e22_item["stopping_epoch"])
        baseline_novel_accuracy = float(baseline_item["level_accuracies"]["novel"])
        e22_novel_accuracy = float(e22_item["level_accuracies"]["novel"])

        baseline_train_accuracies.append(baseline_train_accuracy)
        e22_train_accuracies.append(e22_train_accuracy)
        baseline_stopping_epochs.append(float(baseline_stopping_epoch))
        e22_stopping_epochs.append(float(e22_stopping_epoch))
        baseline_novel_accuracies.append(baseline_novel_accuracy)
        e22_novel_accuracies.append(e22_novel_accuracy)

        if e22_train_accuracy >= baseline_train_accuracy:
            e22_at_least_baseline_train_accuracy_count += 1
        if e22_stopping_epoch <= baseline_stopping_epoch:
            e22_same_or_fewer_epochs_count += 1
        if e22_novel_accuracy > baseline_novel_accuracy:
            e22_exceeds_baseline_novel_accuracy_count += 1

        training_speed_per_seed.append(
            {
                "seed": seed,
                "e22_final_training_accuracy": e22_train_accuracy,
                "baseline_final_training_accuracy": baseline_train_accuracy,
                "e22_stopping_epoch": e22_stopping_epoch,
                "baseline_stopping_epoch": baseline_stopping_epoch,
                "training_accuracy_difference": e22_train_accuracy - baseline_train_accuracy,
                "stopping_epoch_difference": e22_stopping_epoch - baseline_stopping_epoch,
            }
        )
        novel_accuracy_per_seed.append(
            {
                "seed": seed,
                "e22_novel_test_accuracy": e22_novel_accuracy,
                "baseline_novel_test_accuracy": baseline_novel_accuracy,
                "novel_accuracy_difference": e22_novel_accuracy - baseline_novel_accuracy,
            }
        )

    return {
        "training_speed": {
            "per_seed": training_speed_per_seed,
            "aggregate": {
                "e22_final_training_accuracy": _mean_std(e22_train_accuracies),
                "baseline_final_training_accuracy": _mean_std(baseline_train_accuracies),
                "e22_stopping_epoch": _mean_std(e22_stopping_epochs),
                "baseline_stopping_epoch": _mean_std(baseline_stopping_epochs),
                "e22_at_least_baseline_training_accuracy_seed_count": e22_at_least_baseline_train_accuracy_count,
                "e22_same_or_fewer_epochs_seed_count": e22_same_or_fewer_epochs_count,
            },
        },
        "novel_accuracy": {
            "per_seed": novel_accuracy_per_seed,
            "aggregate": {
                "e22_novel_test_accuracy": _mean_std(e22_novel_accuracies),
                "baseline_novel_test_accuracy": _mean_std(baseline_novel_accuracies),
                "e22_exceeds_baseline_novel_accuracy_seed_count": e22_exceeds_baseline_novel_accuracy_count,
            },
        },
    }


def main() -> None:
    deliverable1_payload = _load_deliverable1_payload()
    e22_artifacts = run_e22_training_suite(SEEDS)

    layer1_results = []
    layer2_results = []

    for artifacts_for_seed in e22_artifacts:
        layer1_analysis, layer2_analysis = analyze_primary_seed(artifacts_for_seed)
        layer1_results.append(layer1_analysis)
        layer2_results.append(layer2_analysis)

    layer1_aggregate = aggregate_primary_analysis(tuple(layer1_results))
    layer2_aggregate = aggregate_primary_analysis(tuple(layer2_results))

    layer1_pass = (
        layer1_aggregate.significant_seed_count >= 4
        and layer1_aggregate.strict_mean_ordering_seed_count >= 4
        and layer1_aggregate.mean_spearman_rho < 0.0
    )
    layer2_pass = (
        layer2_aggregate.significant_seed_count >= 4
        and layer2_aggregate.strict_mean_ordering_seed_count >= 4
        and layer2_aggregate.mean_spearman_rho < 0.0
    )

    if layer1_pass:
        primary_decision = "PASS"
        carrying_layer = "layer1"
    elif layer2_pass:
        primary_decision = "PASS"
        carrying_layer = "layer2"
    else:
        primary_decision = "FAIL"
        carrying_layer = None

    payload = {
        "primary_analysis": {
            "layer1": {
                "per_seed": [
                    {
                        "seed": item.seed,
                        "jonckheere_terpstra_statistic": item.jonckheere_terpstra_statistic,
                        "jonckheere_terpstra_p_value": item.jonckheere_terpstra_p_value,
                        "strict_mean_ordering": item.strict_mean_ordering,
                        "spearman_rho": item.spearman_rho,
                        "level_means": _level_means_payload(item.level_means),
                    }
                    for item in layer1_results
                ],
                "aggregate": {
                    "significant_seed_count": layer1_aggregate.significant_seed_count,
                    "strict_mean_ordering_seed_count": layer1_aggregate.strict_mean_ordering_seed_count,
                    "mean_spearman_rho": layer1_aggregate.mean_spearman_rho,
                    "std_spearman_rho": layer1_aggregate.std_spearman_rho,
                },
            },
            "layer2": {
                "per_seed": [
                    {
                        "seed": item.seed,
                        "jonckheere_terpstra_statistic": item.jonckheere_terpstra_statistic,
                        "jonckheere_terpstra_p_value": item.jonckheere_terpstra_p_value,
                        "strict_mean_ordering": item.strict_mean_ordering,
                        "spearman_rho": item.spearman_rho,
                        "level_means": _level_means_payload(item.level_means),
                    }
                    for item in layer2_results
                ],
                "aggregate": {
                    "significant_seed_count": layer2_aggregate.significant_seed_count,
                    "strict_mean_ordering_seed_count": layer2_aggregate.strict_mean_ordering_seed_count,
                    "mean_spearman_rho": layer2_aggregate.mean_spearman_rho,
                    "std_spearman_rho": layer2_aggregate.std_spearman_rho,
                },
            },
        },
        "secondary_analysis": _secondary_payload(deliverable1_payload),
        "primary_decision": {
            "decision": primary_decision,
            "carrying_layer": carrying_layer,
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
