# E3.1 WDA Clean-QKV Hardening

- Decision: `E3_1_WDA_CLEAN_QKV_C128_REACHED`
- C=128 reached: True
- Best beat E3 parents: True
- Best beat E2 reference: False
- Winning WDA target: KV
- Support/contradiction cache mattered: True
- Gate selectivity improved: 0.060
- Current-override behavior improved: 1.000
- Stage 8C run: no
- 10x attention-capacity claim: no
- Final templates used: no

## Top Configs
1. `e3_1_wda_clean_qkv_e_sc_wda_kv_01` (WDA Support/Contradiction Clean-QKV, sc_wda_kv) held-out=0.758, C=128, valid=True
2. `e3_1_wda_clean_qkv_e_sc_wda_v_00` (WDA Support/Contradiction Clean-QKV, sc_wda_v) held-out=0.755, C=128, valid=True
3. `e3_1_wda_clean_qkv_e_sc_wda_kv_01_e31_mut0` (WDA Support/Contradiction Clean-QKV, sc_wda_kv) held-out=0.755, C=128, valid=True
4. `e3_1_wda_clean_qkv_g_wda_v_gate_calibration_00_e31_mut0` (Gate-Selective WDA Clean-QKV, wda_v_gate_calibration) held-out=0.755, C=128, valid=True

## Remaining Failure Mode
The remaining limit is scaling beyond N=64/128 under held-out templates without losing cache-control sensitivity; higher-capacity activation-cache typing and WDA coordinate regularization remain the next pressure points.

## Next Recommended Experiment
Run a narrow support/contradiction typed-cache plus WDA-bias+V/KV comparison at N=128,256 with stronger coordinate entropy/diversity sweeps and no broad architecture search.
