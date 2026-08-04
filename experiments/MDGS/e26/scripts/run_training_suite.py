"""Entry point for the E26 training scaffold."""

from __future__ import annotations

from experiments.DIGIT.Extrapolation.e26.extrapolation.experiment import run_training_suite


def main() -> None:
    run_training_suite()


if __name__ == "__main__":
    main()

