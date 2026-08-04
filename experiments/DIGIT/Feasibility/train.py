#!/usr/bin/env python
"""Entry point for running the Intuition-Labs experiment."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Intuition-Labs: Privacy-preserving discrete bottleneck model"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (default: use built-in defaults)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Directory for model checkpoints and results",
    )
    args = parser.parse_args()

    from intuition.train import run_experiment
    run_experiment(config_path=args.config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
