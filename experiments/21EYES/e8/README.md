# E8 - Multi-Avenue Inter-Layer Attention Scout

This scout tests whether named learned interpretive pathways are more useful than E7 anonymous slots on the E6 synthetic retrieval/overwrite/multi-hop tasks.

Training uses only causal token sequences and query-position next-token loss. Diagnostic metadata is used only during evaluation.

Variants:

- `baseline_transformer`
- `param_matched_baseline`
- `single_avenue`
- `four_avenue_sum`
- `four_avenue_gated`
- `four_avenue_dropout05`
- `four_avenue_leave_one_out`
- `mirrored_four_avenue`

Run:

```bash
conda activate tnt
cd experiments/21EYES/e8
bash run_scout.sh
```

Outputs:

- `results/<variant>/<seed>/metrics.jsonl`
- `results/<variant>/<seed>/final_metrics.json`
- `results/<variant>/<seed>/config.yaml`
- `results/<variant>/<seed>/model_summary.txt`
- `results/report.md`
- `plots/*.png`
