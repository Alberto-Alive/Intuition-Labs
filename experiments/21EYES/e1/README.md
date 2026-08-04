# Weight Distribution Attention Transformer POF

This directory implements the `2ndpof` specification as a runnable PyTorch
proof-of-feasibility.

Run a fast smoke check:

```powershell
python -m transformer_uncertainty.run_wda_pof --quick --device cpu
```

Run the requested default-sized proof:

```powershell
python -m transformer_uncertainty.run_wda_pof
```

Artifacts are written to:

```text
transformer_uncertainty/runs/weight_distribution_attention_pof_<date>
```

Run the stricter capacity/compute validation requested in `2i`:

```powershell
python -m transformer_uncertainty.run_wda_capacity_compute_validation
```

The default validation performs a small validation-only WDA hyperparameter
sweep before reporting test metrics. Disable it with `--no-sweep` for faster
debug runs.

Run a fast smoke check for that validation path:

```powershell
python -m transformer_uncertainty.run_wda_capacity_compute_validation --quick --device cpu
```

Capacity/compute artifacts are written to:

```text
transformer_uncertainty/runs/wda_capacity_compute_validation_<date>
```

Run the clean Distribution Attention Transformer proof requested in `3i`:

```powershell
python -m transformer_uncertainty.run_distribution_attention_pof
```

Run a fast smoke check for the DAT path:

```powershell
python -m transformer_uncertainty.run_distribution_attention_pof --quick --device cpu
```

DAT artifacts are written to:

```text
transformer_uncertainty/runs/distribution_attention_pof_<date>
```

Run the T-realized self-attention proof requested in `4i`:

```powershell
python -m transformer_uncertainty.run_t_realized_attention_pof
```

Run a fast smoke check for that path:

```powershell
python -m transformer_uncertainty.run_t_realized_attention_pof --quick --device cpu
```

T-realized attention artifacts are written to:

```text
transformer_uncertainty/runs/t_realized_attention_pof_<date>
```

Run the stacked T-realized attention block validation requested in `5i`:

```powershell
python -m transformer_uncertainty.run_t_realized_blocks_validation
```

Run a fast smoke check for that path:

```powershell
python -m transformer_uncertainty.run_t_realized_blocks_validation --quick --device cpu
```

T-realized block validation artifacts are written to:

```text
transformer_uncertainty/runs/t_realized_blocks_validation_<date>
```
