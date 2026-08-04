"""Comparative canary detection runner."""
from __future__ import annotations

import argparse
import logging

from Validation.attacks.repeated_query_attacks import run_canary_detection_attack
from Validation.runners.comparative_common import add_common_args, execute_attack_grid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def _per_seed_runner(*, bundle, system_spec, mode, seed, builder, args):
    return run_canary_detection_attack(
        builder,
        bundle.train,
        seed=seed,
        query_budget=args.query_budget,
        canary_feature_values=bundle.attack_spec.canary_feature_values,
        canary_label=bundle.attack_spec.canary_label,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Comparative canary detection")
    add_common_args(parser, "Comparative canary detection")
    args = parser.parse_args(argv)
    execute_attack_grid(
        args=args,
        task="canary_detection",
        phase="phase1_privacy",
        metric_names=[
            "queries_to_detection_95pct",
            "final_confidence",
            "query_count",
            "cumulative_epsilon",
        ],
        per_seed_runner=_per_seed_runner,
        per_seed_runner_name="run_canary_detection_attack",
    )


if __name__ == "__main__":
    main()
