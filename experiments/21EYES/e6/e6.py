from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

CONFIGS = [
    "configs/baseline.yaml",
    "configs/param_matched.yaml",
    "configs/same_layer_u_gate.yaml",
    "configs/inter_layer_route_gate.yaml",
    "configs/prev_attn_reuse.yaml",
    "configs/random_gate.yaml",
]


def run(args: list[str]) -> None:
    command = [sys.executable, *args]
    print(f"\n$ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    for config in CONFIGS:
        run(["train.py", "--config", config])

    run(["eval.py", "--report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
