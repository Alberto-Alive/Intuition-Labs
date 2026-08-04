# Run Commands

These commands reproduce the aux validation setup from the project root.

## Full Validation

```powershell
python -m transformer_uncertainty.validate_aux `
  --out transformer_uncertainty/runs/aux_validation_gpu_20260508 `
  --device cuda `
  --seeds 0 1 2 3 4 5 6 7 8 9 `
  --epochs 20 `
  --models baseline aux `
  --ablations aux_no_mdgs aux_random_mdgs aux_shuffled_mdgs aux_detached_mdgs aux_frozen_mdgs
```

## CPU Smoke Test

```powershell
python -m transformer_uncertainty.validate_aux `
  --out transformer_uncertainty/runs/aux_validation_smoke `
  --seeds 0 `
  --epochs 1 `
  --n-train 128 `
  --n-val 64 `
  --n-test 64 `
  --batch-size 64 `
  --device cpu `
  --models baseline aux `
  --ablations aux_no_mdgs aux_random_mdgs aux_shuffled_mdgs aux_detached_mdgs aux_frozen_mdgs `
  --ood-modes easy hard `
  --num-threads 1
```

## Important Files In The Code Snapshot

- `models.py`
  - `AuxMDGSTransformer`: validated architecture.
  - `MDGSExtractor`: witness attention and MDGS primitive extraction.
  - `build_model`: registers `aux` and broken aux variants.

- `train.py`
  - `loss_for_batch`: combines prediction loss with MDGS auxiliary losses.
  - `_aux_target`: random/shuffled controls.
  - `NO_AUX_LOSS_VARIANTS`: implements `aux_no_mdgs`.

- `validate_aux.py`
  - `PASS_FAIL_RULE`: aux-only validation rule.
  - `decide_verdict`: no anchor-specific check.
  - `write_report`: aggregate report and broken-control comparison.

## Reference Interpretation

Use `aux` when the claim is:

```text
MDGS improves representation learning as an auxiliary uncertainty teacher.
```

Use `shared` or `shared_anchor` only when the claim is:

```text
MDGS participates directly in prediction-time attention.
```
