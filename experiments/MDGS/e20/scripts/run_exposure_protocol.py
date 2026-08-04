"""Run the approved E20 Deliverable 2 exposure protocol and analysis."""

from __future__ import annotations

import json

from experiments.DIGIT.Extrapolation.e20.extrapolation.config import E20Config
from experiments.DIGIT.Extrapolation.e20.extrapolation.experiment import (
    aggregate_primary_analysis,
    run_exposure_protocol,
)


def main() -> None:
    config = E20Config()
    seed_outputs = run_exposure_protocol(config=config)

    layer1_aggregate = aggregate_primary_analysis(tuple(item.layer1_analysis for item in seed_outputs))
    layer2_aggregate = aggregate_primary_analysis(tuple(item.layer2_analysis for item in seed_outputs))

    novel_accuracy_by_checkpoint = {
        checkpoint: sum(
            summary.novel_accuracy
            for item in seed_outputs
            for summary in item.checkpoint_summaries
            if summary.checkpoint == checkpoint
        )
        / len(seed_outputs)
        for checkpoint in config.checkpoint_rounds
    }

    payload = {
        "checkpoint_means": {
            "layer1": {
                str(item.seed): {
                    str(summary.checkpoint): summary.layer1_mean_certainty
                    for summary in item.checkpoint_summaries
                }
                for item in seed_outputs
            },
            "layer2": {
                str(item.seed): {
                    str(summary.checkpoint): summary.layer2_mean_certainty
                    for summary in item.checkpoint_summaries
                }
                for item in seed_outputs
            },
        },
        "primary_analysis": {
            "layer1": {
                "per_seed": [
                    {
                        "seed": item.layer1_analysis.seed,
                        "jonckheere_terpstra_statistic": item.layer1_analysis.jonckheere_terpstra_statistic,
                        "jonckheere_terpstra_p_value": item.layer1_analysis.jonckheere_terpstra_p_value,
                        "strict_mean_ordering": item.layer1_analysis.strict_mean_ordering,
                        "spearman_rho": item.layer1_analysis.spearman_rho,
                        "checkpoint_means": {
                            str(checkpoint): mean
                            for checkpoint, mean in zip(
                                config.checkpoint_rounds,
                                item.layer1_analysis.checkpoint_means,
                                strict=True,
                            )
                        },
                    }
                    for item in seed_outputs
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
                        "seed": item.layer2_analysis.seed,
                        "jonckheere_terpstra_statistic": item.layer2_analysis.jonckheere_terpstra_statistic,
                        "jonckheere_terpstra_p_value": item.layer2_analysis.jonckheere_terpstra_p_value,
                        "strict_mean_ordering": item.layer2_analysis.strict_mean_ordering,
                        "spearman_rho": item.layer2_analysis.spearman_rho,
                        "checkpoint_means": {
                            str(checkpoint): mean
                            for checkpoint, mean in zip(
                                config.checkpoint_rounds,
                                item.layer2_analysis.checkpoint_means,
                                strict=True,
                            )
                        },
                    }
                    for item in seed_outputs
                ],
                "aggregate": {
                    "significant_seed_count": layer2_aggregate.significant_seed_count,
                    "strict_mean_ordering_seed_count": layer2_aggregate.strict_mean_ordering_seed_count,
                    "mean_spearman_rho": layer2_aggregate.mean_spearman_rho,
                    "std_spearman_rho": layer2_aggregate.std_spearman_rho,
                },
            },
        },
        "secondary_novel_accuracy": {
            str(checkpoint): novel_accuracy_by_checkpoint[checkpoint]
            for checkpoint in config.checkpoint_rounds
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
