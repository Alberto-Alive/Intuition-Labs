"""Run the approved Deliverable 3 frozen-bank suite for DIGIT Extrapolation E14."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extrapolation.experiment import run_frozen_bank_suite


def main() -> None:
    results = run_frozen_bank_suite(seeds=[0, 1, 2, 3, 4])
    print(json.dumps([asdict(result) for result in results], indent=2))


if __name__ == "__main__":
    main()
