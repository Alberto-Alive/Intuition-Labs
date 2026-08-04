"""Run Deliverable 3 logging summary for DIGIT Extrapolation E15."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from experiments.DIGIT.Extrapolation.e15.extrapolation.experiment import summarize_logging_across_seeds


def main() -> None:
    summary = summarize_logging_across_seeds(seeds=[0, 1, 2, 3, 4])
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
