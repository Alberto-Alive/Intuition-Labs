"""Comparative membership inference runner."""
from __future__ import annotations

import argparse
import logging

from Validation.attacks.membership_inference import ShadowMIAConfig, run_mia_full
from Validation.runners.comparative_common import add_common_args, execute_attack_grid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def _per_seed_runner(*, bundle, system_spec, mode, seed, builder, args):
    shadow_cfg = ShadowMIAConfig(
        n_shadow=args.shadow_folds,
        n_queries_per_record=args.shadow_queries_per_record,
        shadow_frac=0.5,
        batch_size=128,
    )
    return run_mia_full(
        builder["system"],
        bundle.train,
        bundle.test,
        seed,
        num_samples=args.num_samples,
        shadow_cfg=shadow_cfg,
        system_factory=builder["factory"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Comparative DIGIT-vs-DP membership inference")
    add_common_args(parser, "Comparative MIA")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--shadow-folds", type=int, default=32)
    parser.add_argument("--shadow-queries-per-record", type=int, default=30)
    args = parser.parse_args(argv)
    execute_attack_grid(
        args=args,
        task="membership_inference",
        phase="phase1_privacy",
        metric_names=[
            "auroc",
            "advantage",
            "tpr_at_1pct_fpr",
            "tpr_at_5pct_fpr",
            "query_count",
            "cumulative_epsilon",
        ],
        per_seed_runner=_per_seed_runner,
        per_seed_runner_name="run_mia_full",
    )


if __name__ == "__main__":
    main()
