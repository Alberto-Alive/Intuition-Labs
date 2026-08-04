# E2 Attention Synthesis Campaign

Decision: `E2_DISTRIBUTIONAL_ATTENTION_PROMISING`

Stage 8C was not run. No 10x attention-capacity claim is made.

## Top Configurations
1. `e2_d_cleanqkv_v1_replay`
   - family: Clean-QKV Activation Cache v2
   - held-out-template dev: 1.0000
   - capacity C: 8
   - controls pass: True
2. `e2_c_memory_slot_attention_00`
   - family: Memory Slot Attention
   - held-out-template dev: 1.0000
   - capacity C: 8
   - controls pass: True
3. `e2_f_dat_distributional_memory_slots_00`
   - family: DAT-style Distributional Memory Slots
   - held-out-template dev: 1.0000
   - capacity C: 8
   - controls pass: True

## Controls
Promoted and finalist variants used randomized labels, candidate/evidence mismatch, cross-task evidence shuffles, comparator baselines, hidden-state shuffles, memory controls, and mechanism-specific WDA/DAT/T-realized controls where applicable.

## Artifacts
- `results\e2_attention_synthesis_database.jsonl`
- `results\e2_attention_synthesis_round0_sanity.json`
- `results\e2_attention_synthesis_round1_screen.json`
- `results\e2_attention_synthesis_round2_promotions.json`
- `results\e2_attention_synthesis_round3_mutations.json`
- `results\e2_attention_synthesis_round4_finalists.json`
- `results\e2_attention_synthesis_freeze_recommendation.json`
