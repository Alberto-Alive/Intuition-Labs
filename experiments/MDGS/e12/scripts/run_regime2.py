from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extrapolation.config import E12Config
from extrapolation.experiment import run_regime2_experiment


def main() -> None:
    output_dir = ROOT / "results"
    results = run_regime2_experiment(output_dir=output_dir, config=E12Config())
    print(json.dumps(results["neutral_verification"], indent=2))
    print(json.dumps(results["regression"], indent=2))
    print(json.dumps(results["regression_full"], indent=2))


if __name__ == "__main__":
    main()
