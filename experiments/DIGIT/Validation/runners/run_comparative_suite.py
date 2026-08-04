"""Top-level comparative suite orchestrator."""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_MODULES = [
    "Validation.runners.run_comparative_mia",
    "Validation.runners.run_query_averaging",
    "Validation.runners.run_differencing_attack",
    "Validation.runners.run_utility_frontier",
    "Validation.runners.run_attribute_inference",
    "Validation.runners.run_composition_stress",
    "Validation.runners.run_reconstruction_attack",
    "Validation.runners.run_canary_detection",
    "Validation.runners.run_linkage_attack",
    "Validation.runners.run_model_extraction",
    "Validation.runners.run_gradient_optimization",
    "Validation.runners.run_auxiliary_amplification",
]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the comparative DIGIT-vs-DP suite in priority order")
    parser.add_argument("--datasets", nargs="+", default=["nist_genomics", "tcga"])
    parser.add_argument("--modes", nargs="+", default=["mechanism", "system"])
    parser.add_argument("--seeds", nargs="+", default=["0", "1", "2", "3", "4"])
    parser.add_argument("--query-budget", type=str, default="1000")
    parser.add_argument("--results-dir", type=str, default="experiments/DIGIT/Validation/results")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--data-cache", type=str, default="data_cache")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args(argv)

    base_args = [
        "--datasets",
        *args.datasets,
        "--modes",
        *args.modes,
        "--seeds",
        *args.seeds,
        "--query-budget",
        args.query_budget,
        "--results-dir",
        args.results_dir,
        "--checkpoint-dir",
        args.checkpoint_dir,
        "--data-cache",
        args.data_cache,
    ]
    if args.train:
        base_args.append("--train")

    for module in _MODULES:
        cmd = [sys.executable, "-m", module, *base_args]
        logger.info("Running %s", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
