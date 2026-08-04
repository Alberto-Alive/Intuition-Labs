# E2 Attention Synthesis Campaign

Decision: `E2_NEW_ARCHITECTURE_BEATS_CHAMPION`

Stage 8C was not run. No 10x attention-capacity claim is made.

## Top Configurations
1. `e2_c_memory_slot_attention_00`
   - family: Memory Slot Attention
   - held-out-template dev: 0.9563
   - capacity C: 256
   - controls pass: True
2. `stage8b3_j_explicit_view_objective_assignment_03`
   - family: Stage 8 Champion Replay
   - held-out-template dev: 0.8594
   - capacity C: 128
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
