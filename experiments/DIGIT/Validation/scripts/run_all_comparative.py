"""Single entrypoint for comparative smoke checks and full suite execution.

Run from the repo root with:
    python3 experiments/DIGIT/Validation/scripts/run_all_comparative.py
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_SCRIPT_PATH = Path(__file__).resolve()
_DIGIT_ROOT = _SCRIPT_PATH.parents[2]
_REPO_ROOT = _SCRIPT_PATH.parents[4]


def _run(cmd: list[str], cwd: Path) -> None:
    logger.info("Running %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the comparative Validation smoke checks and full DIGIT-vs-DP suite."
    )
    parser.add_argument("--skip-smoke", action="store_true", help="Skip the synthetic smoke/regression checks.")
    parser.add_argument("--skip-suite", action="store_true", help="Skip the full comparative suite.")
    parser.add_argument("--datasets", nargs="+", default=["nist_genomics", "tcga"])
    parser.add_argument("--modes", nargs="+", default=["mechanism", "system"])
    parser.add_argument("--seeds", nargs="+", default=["0", "1", "2", "3", "4"])
    parser.add_argument("--query-budget", default="1000")
    parser.add_argument("--results-dir", default="Validation/results")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--data-cache", default="data_cache")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args(argv)

    if not args.skip_smoke:
        smoke_cmd = [sys.executable, str(_SCRIPT_PATH.parent / "test_comparative_smoke.py")]
        _run(smoke_cmd, cwd=_REPO_ROOT)

    if not args.skip_suite:
        suite_cmd = [
            sys.executable,
            "-m",
            "Validation.runners.run_comparative_suite",
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
            suite_cmd.append("--train")
        _run(suite_cmd, cwd=_DIGIT_ROOT)


if __name__ == "__main__":
    main()
