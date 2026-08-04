# E7 - Slot-State Mediation / Predictive Inter-Layer Slot Alignment

This experiment tests whether attention benefits from a persistent structured slot state rather than direct token-level routing.

The implementation reuses the E6 synthetic tasks and anti-cheat metadata discipline:

- Training input is only the causal token sequence.
- The main loss is query-position next-token cross entropy.
- Diagnostic metadata is used only during evaluation.
- Slot controls are eval-only.

Variants:

- `baseline_transformer`: decoder-only causal transformer.
- `param_matched_baseline`: baseline plus per-layer residual adapters sized to roughly match slot parameters.
- `memory_slot_attention`: per-layer vectorized causal memory slots with token readback.
- `slot_alignment_loss`: memory slots plus cross-layer stop-gradient slot prediction loss.
- `predictive_slot_residual`: next-layer slots use predicted stable slot state plus learned innovation.
- `mirrored_slot_attention`: memory slots with shared slot projection subspace across layers.

Run:

```bash
conda activate tnt
cd experiments/21EYES/e7
bash run_all.sh
```

Outputs:

- `results/<variant>/<seed>/metrics.jsonl`
- `results/<variant>/<seed>/final_metrics.json`
- `results/<variant>/<seed>/config.yaml`
- `results/<variant>/<seed>/model_summary.txt`
- `results/report.md`
- `plots/*.png`
