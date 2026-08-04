# Experiment 6 Fused Routing V2

This extension tests faster forms of activation-conditioned inter-layer attention routing. It writes all new outputs under:

- `results/fused_routing_v2/`
- `plots/fused_routing_v2/`

The previous E6 result folders are not overwritten.

## Variants

- `baseline`: normal causal transformer.
- `param_matched`: non-routing capacity control.
- `dense_augmented_qk`: concatenates learned previous-layer route descriptors onto Q/K and uses one augmented attention score. Beta is learned per head and initialized near zero.
- `block_route_topk_b16_k4`: scores causal blocks from previous-layer activations, then runs exact gathered QK/V attention over local blocks plus four routed 16-token blocks.
- `block_route_topk_b32_k4`: same, with 32-token blocks.
- `coarse_to_fine_block_topk_4`: trains with dense augmented Q/K and evaluates with block-routed exact QK/V attention.

Layer 0 uses normal causal attention because no previous-layer route source exists.

## Run

```bash
conda activate tnt
cd experiments/21EYES/e6
bash run_fused_routing_v2.sh
```

The default configs run seed 0 for 5000 steps. Treat single-seed conclusions as preliminary.

## Outputs

Each run writes:

- `results/fused_routing_v2/<variant>/<seed>/metrics.jsonl`
- `results/fused_routing_v2/<variant>/<seed>/final_metrics.json`
- `results/fused_routing_v2/<variant>/<seed>/config.yaml`
- `results/fused_routing_v2/<variant>/<seed>/model_summary.txt`

The report writer creates:

- `results/fused_routing_v2/report.md`
- `plots/fused_routing_v2/*.png`

## Anti-Cheat Boundary

The model receives only token ids, causal masks, current hidden states, previous-layer hidden states, and learned parameters. Diagnostic relevant and distractor positions are used only inside evaluation metrics.
