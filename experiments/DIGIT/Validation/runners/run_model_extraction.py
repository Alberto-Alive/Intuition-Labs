"""Comparative model extraction runner."""
from __future__ import annotations

import argparse
import logging

from Validation.attacks.model_extraction_eval import run_model_extraction_attack
from Validation.comparative.specs import AttackMode
from Validation.runners.comparative_common import add_common_args, execute_attack_grid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def _per_seed_runner(*, bundle, system_spec, mode, seed, builder, args):
    return run_model_extraction_attack(
        builder["system"],
        bundle.train,
        seed=seed,
        num_queries=args.query_budget,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Comparative model extraction")
    add_common_args(parser, "Comparative model extraction")
    args = parser.parse_args(argv)
    execute_attack_grid(
        args=args,
        task="model_extraction",
        phase="phase1_privacy",
        metric_names=[
            "surrogate_accuracy",
            "answer_accuracy",
            "support_accuracy",
            "confidence_accuracy",
            "risk_accuracy",
            "query_count",
            "cumulative_epsilon",
        ],
        per_seed_runner=_per_seed_runner,
        per_seed_runner_name="run_model_extraction_attack",
        modes=[AttackMode.MECHANISM],
    )


if __name__ == "__main__":
    main()
