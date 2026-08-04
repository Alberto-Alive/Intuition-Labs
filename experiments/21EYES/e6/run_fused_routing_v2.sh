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

out_root="results/fused_routing_v2"

"$py" train.py --config configs/fused_routing_v2/baseline.yaml --output-root "$out_root"
"$py" train.py --config configs/fused_routing_v2/param_matched.yaml --output-root "$out_root"
"$py" train.py --config configs/fused_routing_v2/dense_augmented_qk.yaml --output-root "$out_root"
"$py" train.py --config configs/fused_routing_v2/block_route_topk_b16_k4.yaml --output-root "$out_root"
"$py" train.py --config configs/fused_routing_v2/block_route_topk_b32_k4.yaml --output-root "$out_root"
"$py" train.py --config configs/fused_routing_v2/coarse_to_fine_block_topk_4.yaml --output-root "$out_root"

"$py" eval.py --report \
  --results-dir "$out_root" \
  --plots-dir plots/fused_routing_v2 \
  --report-path results/fused_routing_v2/report.md \
  --title "Experiment 6 Fused Routing V2 Report" \
  --include-old-comparison
