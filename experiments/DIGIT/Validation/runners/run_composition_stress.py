"""Comparative composition stress runner."""
from __future__ import annotations

import argparse
import logging

from Validation.attacks.repeated_query_attacks import run_composition_stress_attack
from Validation.comparative.specs import AttackMode
from Validation.runners.comparative_common import add_common_args, execute_attack_grid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def _per_seed_runner(*, bundle, system_spec, mode, seed, builder, args):
    return run_composition_stress_attack(
        builder["system"],
        bundle.train,
        bundle.test,
        seed=seed,
        query_budget=args.query_budget,
        num_targets=args.num_targets,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Comparative composition stress")
    add_common_args(parser, "Comparative composition stress")
    parser.add_argument("--num-targets", type=int, default=200)
    args = parser.parse_args(argv)
    execute_attack_grid(
        args=args,
        task="composition_stress",
        phase="phase1_privacy",
        metric_names=[
            "auroc",
            "advantage",
            "budget_threshold_0p55",
            "budget_threshold_0p60",
            "budget_threshold_0p70",
            "query_count",
            "cumulative_epsilon",
        ],
        per_seed_runner=_per_seed_runner,
        per_seed_runner_name="run_composition_stress_attack",
        modes=[AttackMode.SYSTEM],
    )


if __name__ == "__main__":
    main()
