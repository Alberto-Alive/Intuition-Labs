#!/usr/bin/env bash
set -euo pipefail
python -m src.train --model ugly --epochs 50 --device cuda --out runs/ugly_quick
python -m src.train --model baseline --epochs 50 --device cuda --out runs/baseline_quick
