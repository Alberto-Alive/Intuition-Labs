# Stage 3.7 Role-Control Diagnosis

## Scope

- Full 10-seed validation: not run.
- Locked architecture changes: none.
- Checkpoint source: no Stage 3.6 tiny-smoke checkpoint artifact was available, so this reran only the one-seed tiny smoke.

## Main Finding

The existing control corrupts only learned runtime role ids. In Stage 3.4b private view text contains role-specific field names such as symbol_surface/provider_area/provider_name_shape/import_slot_shape, so role identity survives the role-id shuffle.

Classification:
- control_weakness: `True`
- role_format_leakage: `True`
- role_pair_shortcut: `False`
- true_role_invariant_learning: `False`
- role_labels_not_needed_by_original_checkpoint: `False`
- role_embeddings_zeroed_above_chance: `True`
- null_role_labels_above_chance: `False`

## Part 1: role_labels_shuffled Implementation

1. Shuffles role embeddings only: `True`
2. Shuffles role text labels: `False`
3. Shuffles private views: `False`
4. Shuffles role/view pairing: `yes for runtime role_id embeddings versus clone activations; no for dataclass view.role/text pairing and no physical view reordering`
5. Shuffle scope: `per-example row-wise permutation inside each forward batch; predict_latent_system reuses the same seed for each batch`
6. Same permutation for train/dev/test: `not inherently; permutations are deterministic from the seed passed to forward. Existing dev/test controls use split-specific seeds; Stage 3.6 tiny smoke evaluated test only.`
7. Corruption time: `evaluation/control time only for role_labels_shuffled; normal trainable/frozen fitting uses condition='none'`

## Original Tiny Smoke

| metric | accuracy |
|---|---:|
| trainable | 0.4414 |
| frozen | 0.0430 |
| text_only | 0.1250 |
| raw_latent | 0.1250 |
| candidate_only | 0.2461 |
| view_masked | 0.1172 |
| role_labels_shuffled | 0.2656 |
| oracle | 1.0000 |

## Part 2: Role Ablations

| variant | trainable | frozen | text | raw | trainable near chance? |
|---|---:|---:|---:|---:|---|
| baseline_none | 0.4414 | 0.0430 | 0.1250 | 0.1250 | `False` |
| A_role_embeddings_zeroed | 0.2891 | 0.0586 | 0.1250 | 0.1211 | `False` |
| B_all_role_labels_same_null_role | 0.0664 | 0.1641 | 0.1250 | 0.1250 | `True` |
| C_role_labels_randomly_permuted_per_example | 0.2500 | 0.0234 | 0.1250 | 0.1250 | `False` |
| D_role_labels_randomly_permuted_globally | 0.4258 | 0.0234 | 0.1250 | 0.1250 | `False` |
| E_private_views_permuted_without_updating_role_labels | 0.2773 | 0.0195 | 0.1250 | 0.1250 | `False` |
| F_private_views_permuted_with_role_labels_preserved | 0.1211 | 0.0234 | 0.1250 | 0.1250 | `True` |
| G_role_embeddings_removed_physical_order_preserved | 0.2891 | 0.0586 | 0.1250 | 0.1211 | `False` |
| H_physical_view_order_randomized_correct_role_labels_preserved | 0.4414 | 0.0430 | 0.1250 | 0.1250 | `False` |
| I_view_format_normalized_original_checkpoint_eval | 0.0938 | 0.0117 | 0.1250 | 0.1250 | `True` |

## Part 3: View-Format Leakage Audit

| diagnostic | train | dev | test |
|---|---:|---:|---:|
| role_id_from_view_text | 1.0000 | 1.0000 | 1.0000 |
| role_id_from_category_fields | 1.0000 | 1.0000 | 1.0000 |
| role_id_from_token_template | 1.0000 | 1.0000 | 1.0000 |
| role_id_from_length_statistics | 0.8760 | 0.8730 | 0.8535 |

- Role id predictable from original view format: `True`

## Part 4: Role-Pair Shortcut Audit

| baseline | accuracy | exceeds gate? |
|---|---:|---|
| role_pair_only | 0.1250 | `False` |
| view_type_pair_only | 0.1250 | `False` |
| category_pair_without_candidate | 0.0938 | `False` |
| candidate_category_without_role | 0.1250 | `False` |
| role_pair_plus_candidate_category | 0.1250 | `False` |

## Part 5: Fix Decision

- Normalized-view-format variant implemented: `True`
- Original max role-id-from-view diagnostic: `1.0000`
- Normalized max role-id-from-view diagnostic: `0.2500`
- Role-pair shortcut detected: `False`

## Part 6: Tiny Smoke After Fix

| metric | accuracy |
|---|---:|
| trainable | 0.0312 |
| frozen | 0.1641 |
| text_only | 0.1250 |
| raw_latent | 0.1172 |
| candidate_only | 0.0469 |
| view_masked | 0.0312 |
| role_labels_shuffled | 0.0312 |
| oracle | 1.0000 |

## Decision

- Cheap fix implemented: `True`
- Full validation justified: `False`
