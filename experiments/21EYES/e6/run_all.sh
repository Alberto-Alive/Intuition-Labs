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

"$py" train.py --config configs/baseline.yaml
"$py" train.py --config configs/param_matched.yaml
"$py" train.py --config configs/same_layer_u_gate.yaml
"$py" train.py --config configs/inter_layer_route_gate.yaml
"$py" train.py --config configs/prev_attn_reuse.yaml
"$py" train.py --config configs/random_gate.yaml

"$py" eval.py --report
