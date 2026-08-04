# Stage ARC-1 Latent Rule-Clone ARC-AGI-2 Verifier/Ranker

## Scope

- Task: select the correct output grid from correct-size candidates with exactly one gold candidate.
- This is not a full ARC solver, does not predict output size, and does not generate grids.
- Architecture anchor: shared-weight rule clones plus candidate-token direct cross-attention, with an exact frozen same-architecture comparator.
- Final validation run: `False`.

## Phase 0 Data Smoke

- Seed `0` examples: `{'train': 35, 'dev': 12, 'test': 12}`.
- Candidate generator recall: `1.0`.
- Leakage audit pass: `True`.
- Candidate metadata-only dev accuracy: `0.0833`.

## Results

| variant | seeds | trainable | frozen | delta | top2 | top3 | MRR | controls |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A_basic_latent_rule_clones | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.5833 | 0.3014 | `True` |
| B_learned_rule_slot_clones | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.5833 | 0.3014 | `False` |
| C_view_biased_rule_clones | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.5833 | 0.3014 | `True` |

## Controls

| gate | pass |
|---|---|
| trainable_beats_frozen_on_at_least_8_of_10_final_seeds | `False` |
| mean_trainable_frozen_delta_at_least_0_20 | `False` |
| bootstrap_95ci_lower_gt_0_05 | `False` |
| candidate_only_near_chance | `True` |
| candidate_metadata_source_only_near_chance | `True` |
| train_pair_shuffle_collapses | `True` |
| candidate_evidence_mismatch_collapses | `True` |
| candidate_order_remap_invariance_passes | `True` |
| role_order_invariance_passes | `True` |
| frozen_comparator_remains_frozen | `True` |
| trainable_shared_model_receives_gradients_and_changes | `True` |
| final_validation_required_for_primary_success | `True` |

## Required Questions

1. Can latent rule clones infer enough from ARC examples to rank candidate outputs?
   - Current smoke answer: best variant `A_basic_latent_rule_clones` has mean trainable-frozen delta `0.0000`. Treat this as implementation smoke unless a full Phase 2/3 run is launched.
2. Does trainable shared-weight coordination beat exact frozen same-architecture?
   - Smoke result: trainable `0.0000`, frozen `0.0000`; +0.20 mean-delta gate `False`; frozen audit gate `True`.
3. Are results above candidate-only and heuristic baselines?
   - Smoke answer: `False`. Best trainable `0.0000` versus random `0.1250`, candidate-stat `0.0000`, and changed-cell heuristic `0.0833`.
4. Do controls show the model is using examples rather than candidate artifacts?
   - Control status is `{'trainable_beats_frozen_on_at_least_8_of_10_final_seeds': False, 'mean_trainable_frozen_delta_at_least_0_20': False, 'bootstrap_95ci_lower_gt_0_05': False, 'candidate_only_near_chance': True, 'candidate_metadata_source_only_near_chance': True, 'train_pair_shuffle_collapses': True, 'candidate_evidence_mismatch_collapses': True, 'candidate_order_remap_invariance_passes': True, 'role_order_invariance_passes': True, 'frozen_comparator_remains_frozen': True, 'trainable_shared_model_receives_gradients_and_changes': True, 'final_validation_required_for_primary_success': True}`. Failed gates block any claim.
5. Do different rule clones specialize?
   - Attention mass and view ablation summaries are saved; specialization is diagnostic-only in this smoke run.
6. Does stacking help ARC verification?
   - The implementation supports `coordination_blocks`; the default Stage ARC-1 smoke only runs A/B/C with one block.
7. Does multi-avenue rule cloning help?
   - Multi-avenue cloning is represented by configurable repeated/expanded view lists, but no medium/final avenue search is claimed here.
8. Are failures mostly due to hard negatives, weak rule inference, or clone collapse?
   - Error cases and hard-negative confusion are logged. Full diagnosis requires a larger Phase 2/3 run.
9. Is this worth extending into candidate generation/refinement later?
   - Only if Phase 2/3 gates pass: trainable beats frozen, candidate-only remains near chance, mismatch/shuffle controls collapse, and clone ablations show multi-view contribution.

## Claim Boundary

No claim is made that this solves ARC-AGI-2. The allowed claim requires a controlled final run that passes the listed gates.
