# E9 - Cooperative Multi-Avenue / Junction Attention Scout

This scout tests whether four causal avenue streams become useful when they periodically cross through learned bottleneck junction tokens. Junction variants use causal prefix junctions: for each position, learned junction tokens attend only to avenue states at that position or earlier, then each avenue reads back from the local junction bottleneck.

Training uses only causal token sequences and query-position next-token loss. Diagnostic metadata is used only during evaluation.

Variants:

- `baseline_transformer`
- `param_matched_baseline`
- `single_avenue_reference`
- `e8_best_reference`
- `junction_middle_bottleneck`
- `junction_every_layer_bottleneck`
- `junction_every_2_layers_bottleneck`
- `forced_junction_output`
- `scout_best_custom`

Run:

```bash
conda activate tnt
cd experiments/21EYES/e9
bash run_scout.sh
```

Outputs:

- `results/<variant>/<seed>/metrics.jsonl`
- `results/<variant>/<seed>/final_metrics.json`
- `results/<variant>/<seed>/config.yaml`
- `results/<variant>/<seed>/model_summary.txt`
- `results/report.md`
- `plots/*.png`
