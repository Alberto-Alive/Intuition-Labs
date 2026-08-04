# E3 Clean-QKV Attention Discovery

- Decision: `E3_CLEAN_QKV_REPAIRED_COMPETITIVE`
- Evaluated rows: 200
- Stage 8C run: no
- 10x attention-capacity claim: no
- Final templates used: no

## Top 3
1. `e3_clean_qkv_e_wda_realized_weights_09` (WDA-Realized Attention Weights) held-out 0.681, C=64, valid=True
2. `e3_clean_qkv_e_wda_realized_weights_08` (WDA-Realized Attention Weights) held-out 0.684, C=64, valid=True
3. `e3_clean_qkv_e_wda_realized_weights_10` (WDA-Realized Attention Weights) held-out 0.672, C=64, valid=True

## Notes
Current self-attention was constrained to current query/candidate/current tokens. Evidence/history tokens were compressed into activation-cache features and only used through attention bias, Q/K/V modulation, WDA realization, T-realization, layer shaping, or gate-selective shaping.
Reference E2 and Stage 8 artifacts were read-only comparisons. No E2 or Stage 8 files were modified.

## Failure Accounting
- Variants with hard disqualifiers recorded: 1
- Clean-QKV-valid variants recorded: 136
