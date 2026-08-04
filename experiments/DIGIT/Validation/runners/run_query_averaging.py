"""Comparative query averaging runner."""
from __future__ import annotations

import argparse
import logging

from Validation.attacks.repeated_query_attacks import run_query_averaging_attack
from Validation.runners.comparative_common import add_common_args, execute_attack_grid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def _per_seed_runner(*, bundle, system_spec, mode, seed, builder, args):
    return run_query_averaging_attack(
        builder["system"],
        bundle.train,
        seed=seed,
        query_budget=args.query_budget,
        tolerance=args.tolerance,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Comparative query averaging attack")
    add_common_args(parser, "Comparative query averaging")
    parser.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args(argv)
    execute_attack_grid(
        args=args,
        task="query_averaging",
        phase="phase1_privacy",
        metric_names=[
            "queries_to_recovery",
            "final_mae",
            "query_count",
            "cumulative_epsilon",
        ],
        per_seed_runner=_per_seed_runner,
        per_seed_runner_name="run_query_averaging_attack",
    )


if __name__ == "__main__":
    main()
