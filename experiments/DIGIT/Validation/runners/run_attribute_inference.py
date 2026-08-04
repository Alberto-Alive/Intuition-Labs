"""Comparative attribute inference runner."""
from __future__ import annotations

import argparse
import logging

from Validation.attacks.attribute_inference_eval import run_attribute_inference_attack
from Validation.runners.comparative_common import add_common_args, execute_attack_grid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def _per_seed_runner(*, bundle, system_spec, mode, seed, builder, args):
    return run_attribute_inference_attack(
        builder["system"],
        bundle.train,
        bundle.test,
        seed=seed,
        num_targets=args.num_targets,
        queries_per_target=args.queries_per_target,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Comparative DIGIT-vs-DP attribute inference")
    add_common_args(parser, "Comparative attribute inference")
    parser.add_argument("--num-targets", type=int, default=500)
    parser.add_argument("--queries-per-target", type=int, default=12)
    args = parser.parse_args(argv)
    execute_attack_grid(
        args=args,
        task="attribute_inference",
        phase="phase1_privacy",
        metric_names=[
            "auroc",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "baseline_auroc",
            "auc_advantage_over_baseline",
            "query_count",
            "cumulative_epsilon",
        ],
        per_seed_runner=_per_seed_runner,
        per_seed_runner_name="run_attribute_inference_attack",
    )


if __name__ == "__main__":
    main()
