# Stage ARC-1.3 Pyramidal Latent View Composition

## Scope

- Task: controlled ARC candidate-output verification with known output size and exactly one gold candidate.
- This is a bounded empirical search over hierarchical latent composition, not final validation.
- Phase 2 was not launched unless every clean gate passed.
- Parent composers receive encoded child latent states only; no gold labels, source metadata, or candidate source IDs are provided.

## Leaderboard

| rank | variant | trainable | frozen | delta | candidate-only | controls | ablation drop | layout | composer | query |
|---:|---|---:|---:|---:|---:|---|---:|---|---|---|
| 1 | P0_flat_baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `False` | 0.0000 | flat | none | leaf_only |
| 2 | P1_pairwise_pooled_composed_only | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `False` | 0.0000 | pairwise | pooled | composed_only |
| 3 | P7_residual_pyramid | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `False` | 0.0000 | residual | token | all_levels |
| 4 | P2_pairwise_token_composed_only | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `False` | 0.0000 |  |  |  |
| 5 | P3_binary_tree_root_intermediate | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `False` | 0.0000 |  |  |  |
| 6 | P4_all_pairs_shared_composer | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `False` | 0.0000 |  |  |  |
| 7 | P5_recursive_weight_tied_composer | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `False` | 0.0000 |  |  |  |
| 8 | P6_candidate_guided_composition | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `False` | 0.0000 |  |  |  |

## Phase 2 Gates

| gate | pass |
|---|---|
| candidate_only_near_chance | `True` |
| candidate_order_remap_invariance_passes | `True` |
| composition_ablation_hurts | `False` |
| frozen_and_trainable_audits_pass | `True` |
| heldout_top1_above_random_N8 | `False` |
| mean_trainable_frozen_delta_gt_0_10 | `False` |
| metadata_only_near_chance | `False` |
| mismatch_collapses | `True` |
| no_candidate_artifact_control_failure | `False` |
| pyramid_order_invariance_passes | `True` |
| role_order_invariance_passes | `True` |
| train_pair_shuffle_collapses | `True` |
| trainable_beats_frozen_on_at_least_2_of_3_seeds | `False` |

## Micro-Overfit Gates

| variant | status | first failed gate |
|---|---|---|
| P0_flat_baseline | `completed` |  |
| P1_pairwise_pooled_composed_only | `completed` |  |
| P2_pairwise_token_composed_only | `failed_overfit_gate` | 12_examples_N4 |
| P3_binary_tree_root_intermediate | `failed_overfit_gate` | 35_examples_N8 |
| P4_all_pairs_shared_composer | `failed_overfit_gate` | 12_examples_N4 |
| P5_recursive_weight_tied_composer | `failed_overfit_gate` | 35_examples_N8 |
| P6_candidate_guided_composition | `failed_overfit_gate` | 12_examples_N4 |
| P7_residual_pyramid | `completed` |  |

## Controls And Ablations

| variant | candidate-only | metadata-only | mismatch | train shuffle | comp shuffle | child mismatch | order delta | role delta | pyramid delta | overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P0_flat_baseline | 0.0000 | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `False` |
| P1_pairwise_pooled_composed_only | 0.0000 | 0.2500 | 0.1250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `False` |
| P7_residual_pyramid | 0.0000 | 0.2500 | 0.1250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `False` |

## Required Answers

1. Does pyramidal composition improve over flat leaf-view querying? Best pyramidal/top variant `P0_flat_baseline` reached `0.0000` top1 versus P0 `0.0000` top1; P0 delta was `0.0000`.
2. Which compositions are useful? The strongest measured ablation drop for the best variant was `0.0000` top1; detailed pair ablations are in `results/arc1_3_pyramidal_ablation.json`.
3. Does token-level composition beat pooled composition? Not established: `P1_pairwise_pooled_composed_only` status is `completed`, while `P2_pairwise_token_composed_only` status is `failed_overfit_gate` with first failed gate `12_examples_N4`.
4. Does candidate-guided composition help or leak? Not established: `P6_candidate_guided_composition` status is `failed_overfit_gate` with first failed gate `12_examples_N4`. The implementation does not use candidate index/source embeddings.
5. Does root-level composition add value beyond pairwise composition? Not established: `P3_binary_tree_root_intermediate` status is `failed_overfit_gate` and `P5_recursive_weight_tied_composer` status is `failed_overfit_gate`.
6. Do composition ablations show parent clones are actually used? This is true only when composition/root/pair ablations produce a meaningful positive drop; see the ablation output.
7. Does the pyramid improve clean held-out accuracy above random? Random is `0.1250`; best clean top1 is `0.0000`.
8. Does trainable beat exact frozen same-architecture? Best trainable `0.0000` versus frozen `0.0000`; frozen/trainable audit gate is `True`.
9. Do controls remain clean? Overall gate for the best variant is `False`.
10. Is pyramidal composition worth moving to Phase 2? `False`.

## Decision

- Phase 2 triggered: `False`.
- No ARC-AGI-2 solving, full ARC generalization, output-size prediction, or grid-generation claim is made.
