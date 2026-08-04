"""Run Deliverable 5 for DIGIT Extrapolation E14."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extrapolation.analysis import run_deliverable5


def main() -> None:
    result = run_deliverable5(seeds=[0, 1, 2, 3, 4])
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
