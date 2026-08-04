#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -n "${PYTHON:-}" ]]; then
  py="$PYTHON"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
  if command -v cygpath >/dev/null 2>&1; then
    conda_prefix="$(cygpath -u "$CONDA_PREFIX")"
    conda_python="$conda_prefix/python.exe"
  else
    conda_python="$CONDA_PREFIX/bin/python"
  fi

  if [[ -x "$conda_python" ]]; then
    py="$conda_python"
  fi
fi

if [[ -z "${py:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    py="python"
  elif command -v python3 >/dev/null 2>&1; then
    py="python3"
  elif command -v python.exe >/dev/null 2>&1; then
    py="python.exe"
  else
    echo "Could not find Python. Activate conda env tnt or set PYTHON=/path/to/python." >&2
    exit 127
  fi
fi

"$py" train.py --config configs/baseline_transformer.yaml
"$py" train.py --config configs/param_matched_baseline.yaml
"$py" train.py --config configs/single_avenue.yaml
"$py" train.py --config configs/four_avenue_sum.yaml
"$py" train.py --config configs/four_avenue_gated.yaml
"$py" train.py --config configs/four_avenue_dropout05.yaml
"$py" train.py --config configs/four_avenue_leave_one_out.yaml
"$py" train.py --config configs/mirrored_four_avenue.yaml

"$py" eval.py --report
