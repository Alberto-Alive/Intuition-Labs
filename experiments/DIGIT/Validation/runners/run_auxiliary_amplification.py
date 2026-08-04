"""Comparative auxiliary amplification runner."""
from __future__ import annotations

import argparse
import logging

from Validation.attacks.auxiliary_amplification_eval import run_auxiliary_amplification_attack
from Validation.runners.comparative_common import add_common_args, execute_attack_grid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def _per_seed_runner(*, bundle, system_spec, mode, seed, builder, args):
    return run_auxiliary_amplification_attack(
        builder["system"],
        bundle.train,
        bundle.test,
        bundle.val,
        seed=seed,
        num_targets=args.num_targets,
        queries_per_target=args.queries_per_target,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Comparative auxiliary knowledge amplification")
    add_common_args(parser, "Comparative auxiliary amplification")
    parser.add_argument("--num-targets", type=int, default=300)
    parser.add_argument("--queries-per-target", type=int, default=10)
    args = parser.parse_args(argv)
    execute_attack_grid(
        args=args,
        task="auxiliary_knowledge",
        phase="phase1_privacy",
        metric_names=[
            "without_aux_auroc",
            "with_aux_auroc",
            "aux_amplification",
            "query_count",
            "cumulative_epsilon",
        ],
        per_seed_runner=_per_seed_runner,
        per_seed_runner_name="run_auxiliary_amplification_attack",
    )


if __name__ == "__main__":
    main()
