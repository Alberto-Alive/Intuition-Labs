"""Run the approved E19 Deliverable 2 primary analysis."""

from __future__ import annotations

import json

from experiments.DIGIT.Extrapolation.e19.extrapolation.experiment import (
    aggregate_primary_analysis,
    analyze_primary_seed,
    run_training_suite,
)


def main() -> None:
    artifacts = run_training_suite((0, 1, 2, 3, 4))
    layer1_results = []
    layer2_results = []

    for artifacts_for_seed in artifacts:
        layer1_analysis, layer2_analysis = analyze_primary_seed(artifacts_for_seed)
        layer1_results.append(layer1_analysis)
        layer2_results.append(layer2_analysis)

    layer1_aggregate = aggregate_primary_analysis(tuple(layer1_results))
    layer2_aggregate = aggregate_primary_analysis(tuple(layer2_results))

    payload = {
        "layer1": {
            "per_seed": [
                {
                    "seed": item.seed,
                    "jonckheere_terpstra_statistic": item.jonckheere_terpstra_statistic,
                    "jonckheere_terpstra_p_value": item.jonckheere_terpstra_p_value,
                    "strict_mean_ordering": item.strict_mean_ordering,
                    "spearman_rho": item.spearman_rho,
                    "level_means": {
                        "core": item.level_means[0],
                        "familiar": item.level_means[1],
                        "rare": item.level_means[2],
                        "novel": item.level_means[3],
                    },
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
                    "level_means": {
                        "core": item.level_means[0],
                        "familiar": item.level_means[1],
                        "rare": item.level_means[2],
                        "novel": item.level_means[3],
                    },
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
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
