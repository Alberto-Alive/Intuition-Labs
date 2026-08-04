"""Comparative gradient-style query optimization runner."""
from __future__ import annotations

import argparse
import logging

from Validation.attacks.whitebox_attack_eval import run_gradient_query_attack
from Validation.comparative.specs import AttackMode
from Validation.runners.comparative_common import add_common_args, execute_attack_grid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def _per_seed_runner(*, bundle, system_spec, mode, seed, builder, args):
    system_factory = builder["rebuild"] or (lambda ds, _s=builder["system"]: _s)
    return run_gradient_query_attack(
        system_factory,
        bundle.train,
        seed=seed,
        num_targets=args.num_targets,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Comparative white-box query optimization")
    add_common_args(parser, "Comparative white-box query optimization")
    parser.add_argument("--num-targets", type=int, default=100)
    args = parser.parse_args(argv)
    execute_attack_grid(
        args=args,
        task="gradient_optimization",
        phase="phase1_privacy",
        metric_names=[
            "optimized_auroc",
            "random_auroc",
            "auroc_gain",
            "query_count",
            "cumulative_epsilon",
        ],
        per_seed_runner=_per_seed_runner,
        per_seed_runner_name="run_gradient_query_attack",
        modes=[AttackMode.MECHANISM],
    )


if __name__ == "__main__":
    main()
