# Stage ARC-1.1 Empirical Search

## Scope

- Bounded empirical search over representation, rule-clone views, candidate interaction, objectives, negative difficulty, and simple baselines.
- No final validation was launched.
- Labels, candidate order remapping, source metadata hiding, leakage audits, controls, and frozen-comparator protocol remain fixed.

## Leaderboard

| rank | variant | status | trainable | frozen | delta | candidate-only | controls | family |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | clone_hybrid_all_views_ce | completed | 0.0833 | 0.0000 | 0.0833 | 0.0833 | `True` | view_set |
| 2 | clone_object_bbox_ce | completed | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `True` | representation |
| 3 | clone_changed_cell_ce | completed | 0.0833 | 0.1667 | -0.0833 | 0.1667 | `True` | negative_curriculum |
| 4 | clone_easy_random_ce | completed | 0.7500 | 0.6667 | 0.0833 | 0.5000 | `False` | negative_curriculum |
| 5 | stats_mlp_hybrid_features | failed_overfit_gate | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `False` |  |
| 6 | clone_raw_cell_ce | failed_overfit_gate | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `False` |  |
| 7 | clone_raw_diff_ce | failed_overfit_gate | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `False` |  |
| 8 | clone_color_hist_ce | failed_overfit_gate | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `False` |  |
| 9 | clone_learned_slots_hybrid_ce | failed_overfit_gate | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `False` |  |
| 10 | clone_repeated_avenues_hybrid_ce | failed_overfit_gate | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `False` |  |
| 11 | clone_candidate_guided_hybrid_ce | failed_overfit_gate | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `False` |  |
| 12 | clone_bidirectional_hybrid_ce | failed_overfit_gate | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `False` |  |
| 13 | clone_stack2_hybrid_ce | failed_overfit_gate | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `False` |  |
| 14 | clone_stack3_hybrid_ce | failed_overfit_gate | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `False` |  |
| 15 | clone_hybrid_pairwise | failed_overfit_gate | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `False` |  |
| 16 | clone_hybrid_margin | failed_overfit_gate | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `False` |  |
| 17 | clone_palette_curriculum_ce | failed_overfit_gate | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `False` |  |
| 18 | clone_adversarial_hard_ce | failed_overfit_gate | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `False` |  |

## Failure Taxonomy

```json
{
  "control_failure": 1,
  "did_not_beat_frozen": 2,
  "failed_overfit::12_examples_N4": 1,
  "failed_overfit::35_examples_N8": 1,
  "failed_overfit::4_examples_N2": 12,
  "heldout_not_above_random": 3
}
```

## Overfit Results

| variant | gate | train acc | threshold | pass |
|---|---|---:|---:|---|
| stats_mlp_hybrid_features | 4_examples_N2 | 0.7500 | 0.9500 | `False` |
| clone_raw_cell_ce | 4_examples_N2 | 0.7500 | 0.9500 | `False` |
| clone_raw_diff_ce | 4_examples_N2 | 0.7500 | 0.9500 | `False` |
| clone_object_bbox_ce | 4_examples_N2 | 1.0000 | 0.9500 | `True` |
| clone_object_bbox_ce | 12_examples_N4 | 1.0000 | 0.9000 | `True` |
| clone_object_bbox_ce | 35_examples_N8 | 0.8000 | 0.8000 | `True` |
| clone_color_hist_ce | 4_examples_N2 | 1.0000 | 0.9500 | `True` |
| clone_color_hist_ce | 12_examples_N4 | 1.0000 | 0.9000 | `True` |
| clone_color_hist_ce | 35_examples_N8 | 0.4571 | 0.8000 | `False` |
| clone_hybrid_all_views_ce | 4_examples_N2 | 1.0000 | 0.9500 | `True` |
| clone_hybrid_all_views_ce | 12_examples_N4 | 1.0000 | 0.9000 | `True` |
| clone_hybrid_all_views_ce | 35_examples_N8 | 0.8857 | 0.8000 | `True` |
| clone_learned_slots_hybrid_ce | 4_examples_N2 | 0.7500 | 0.9500 | `False` |
| clone_repeated_avenues_hybrid_ce | 4_examples_N2 | 0.7500 | 0.9500 | `False` |
| clone_candidate_guided_hybrid_ce | 4_examples_N2 | 0.7500 | 0.9500 | `False` |
| clone_bidirectional_hybrid_ce | 4_examples_N2 | 0.7500 | 0.9500 | `False` |
| clone_stack2_hybrid_ce | 4_examples_N2 | 0.7500 | 0.9500 | `False` |
| clone_stack3_hybrid_ce | 4_examples_N2 | 0.7500 | 0.9500 | `False` |
| clone_hybrid_pairwise | 4_examples_N2 | 0.7500 | 0.9500 | `False` |
| clone_hybrid_margin | 4_examples_N2 | 0.7500 | 0.9500 | `False` |
| clone_palette_curriculum_ce | 4_examples_N2 | 1.0000 | 0.9500 | `True` |
| clone_palette_curriculum_ce | 12_examples_N4 | 0.8333 | 0.9000 | `False` |
| clone_easy_random_ce | 4_examples_N2 | 1.0000 | 0.9500 | `True` |
| clone_easy_random_ce | 12_examples_N4 | 1.0000 | 0.9000 | `True` |
| clone_easy_random_ce | 35_examples_N8 | 0.9714 | 0.8000 | `True` |
| clone_changed_cell_ce | 4_examples_N2 | 1.0000 | 0.9500 | `True` |
| clone_changed_cell_ce | 12_examples_N4 | 1.0000 | 0.9000 | `True` |
| clone_changed_cell_ce | 35_examples_N8 | 0.8857 | 0.8000 | `True` |
| clone_adversarial_hard_ce | 4_examples_N2 | 0.7500 | 0.9500 | `False` |

## Control Pass/Fail

| variant | candidate-only | metadata-only | train shuffle | mismatch | order delta | role delta | overall |
|---|---:|---:|---:|---:|---:|---:|---|
| clone_object_bbox_ce | 0.0000 | 0.0833 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `True` |
| clone_hybrid_all_views_ce | 0.0833 | 0.0833 | 0.0833 | 0.0000 | 0.0000 | 0.0000 | `True` |
| clone_easy_random_ce | 0.5000 | 0.1667 | 0.7500 | 0.5833 | 0.0000 | 0.0000 | `False` |
| clone_changed_cell_ce | 0.1667 | 0.0000 | 0.0833 | 0.1667 | 0.0000 | 0.0000 | `True` |

## Representation Comparison

| group | variants | mean top1 |
|---|---:|---:|
| hybrid_cell_object_diff | 2 | 0.4167 |
| hybrid_all | 1 | 0.0833 |
| object | 1 | 0.0000 |

## Objective Comparison

| group | variants | mean top1 |
|---|---:|---:|
| cross_entropy | 4 | 0.2292 |

## Negative Difficulty Comparison

| group | variants | mean top1 |
|---|---:|---:|
| easy_random | 1 | 0.7500 |
| changed_cell_count_matched | 1 | 0.0833 |
| mixed_hard | 2 | 0.0417 |

## Compute Table

| variant | params | train seconds | candidate count |
|---|---:|---:|---:|
| clone_hybrid_all_views_ce | 44897 | 22.98 | 8 |
| clone_object_bbox_ce | 44897 | 20.01 | 8 |
| clone_changed_cell_ce | 44897 | 11.77 | 8 |
| clone_easy_random_ce | 44897 | 33.24 | 8 |

## Required Answers

1. Which representation made ARC candidate verification learnable? `micro_overfit=hybrid_cell_object_diff`; `clean_heldout=none`; `uncontrolled_best=hybrid_cell_object_diff`.
2. Which rule-clone view set worked best? `raw diff object color geometry` among control-clean completed variants.
3. Did candidate-guided processing help? `not established`.
4. Did stacking help? `not established`.
5. Did curriculum negatives help? `best clone_easy_random_ce top1=0.7500 delta=0.0833`.
6. Did pairwise/ranking losses help? `not established`.
7. Did any variant beat frozen and candidate-only baselines? `clean_valid=False`; `uncontrolled=True`.
8. Did any variant keep controls clean? `True`.
9. Is the ARC latent-rule-clone path worth Phase 2? `False` by the configured medium trigger.
10. What is the smallest working architecture? `none`.

## Decision

- Medium validation triggered: `False`.
- Medium-ready variants: `[]`.
- Final validation remains blocked until the Stage ARC-1.1 medium trigger passes.
