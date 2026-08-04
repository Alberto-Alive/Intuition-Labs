"""Run Deliverable 1 training for DIGIT Extrapolation E16."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from experiments.DIGIT.Extrapolation.e16.extrapolation.experiment import run_training_suite


def main() -> None:
    results = run_training_suite(seeds=[0, 1, 2, 3, 4])
    print(json.dumps([asdict(result) for result in results], indent=2))


if __name__ == "__main__":
    main()
