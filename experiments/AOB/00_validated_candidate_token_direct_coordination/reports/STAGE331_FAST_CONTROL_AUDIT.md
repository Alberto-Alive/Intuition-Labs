# Stage 3.3.1 Fast Control-Harness Audit

## Scope

- Full training jobs run: `0`.
- Method: dataset-only diagnostics plus checkpoint-only evaluation of saved Stage 3.3 trainable/frozen checkpoints.
- This audit does not claim Stage 3.3 success.

## Main Finding

Stage 3.3 controls fail primarily because the v2 candidate attribute codebook is not translation-invariant: candidate metadata alone can infer the hidden evidence tuple. Role-label shuffle also preserves candidate feature columns and the decisive pair structure, so it is not a destructive corruption for this dataset.

Classification:
- candidate_metadata_shortcut: `True`
- family_template_leakage: `False`
- randomized_label_implementation_bug: `False`
- role_pair_structural_shortcut: `True`
- role_shuffle_implementation_bug: `False`
- true_model_effect: `False`
- view_masking_bug: `False`

## Dataset-Only Signals

- Candidate codebook oracle mean accuracy: `1.0000`
- Ordered candidate-bits lookup mean accuracy: `0.1271`
- View-masked candidate codebook oracle mean accuracy: `1.0000`
- Test randomized-label vs original-label mean accuracy: `0.1278`
- Max test randomized-label vs original-label accuracy: `0.1426`
- Max trivial train-random to test-random accuracy: `0.1484`
- Max trivial train-random to test-original accuracy: `0.2500`
- Train/test normalized template duplicate rate: `1.0000`
- Train/test ordered candidate-bits duplicate rate: `0.0024`

## Control Implementation

- `role_labels_shuffled` leaves examples unchanged in `apply_example_control`; role ids are shuffled only inside `SharedClonedAgentSystem.forward`.
- It shuffles role embeddings only. It does not shuffle role text, role/view pairing, physical order, or candidate feature columns.
- `view_masked` masks private view text but leaves candidate text and candidate attributes available to the candidate-query coordinator.
- Pairwise oracle after role-label apply control: `{'0,1': 1.0, '2,3': 1.0}`

## Existing Failed Controls

| control | mean | max | strict pass |
|---|---:|---:|---|
| hidden_states_shuffled_across_examples | 0.2209 | 0.2383 | False |
| randomized_labels | 0.1750 | 0.4121 | False |
| role_labels_shuffled | 0.3704 | 0.4551 | False |
| view_masked | 0.1793 | 0.3105 | False |
| view_shuffled | 0.2070 | 0.2490 | False |

## Checkpoint-Only Ablations

| variant | trainable mean | frozen mean | trainable near chance | frozen near chance |
|---|---:|---:|---|---|
| all_role_labels_same_zero | 0.2975 | 0.2035 | False | False |
| candidate_metadata_removed | 0.1250 | 0.1250 | True | True |
| candidate_only | 0.1983 | 0.1398 | False | True |
| family_id_removed | 0.9999 | 0.7261 | False | False |
| hidden_states_shuffled_across_examples | 0.2301 | 0.2250 | False | False |
| none | 0.9999 | 0.7261 | False | False |
| role_embeddings_zeroed | 0.2737 | 0.2172 | False | False |
| role_labels_random_independent | 0.3158 | 0.2353 | False | False |
| role_labels_random_permutation | 0.3649 | 0.2828 | False | False |
| role_pair_structure_randomized | 0.3089 | 0.2492 | False | False |
| view_only_without_candidate_metadata | 0.1250 | 0.1250 | True | True |
| view_shuffled | 0.2122 | 0.2016 | False | False |
| views_masked | 0.1930 | 0.2055 | False | False |

## Per-Seed Checkpoint Accuracies

| variant | trainable by seed | frozen by seed |
|---|---|---|
| all_role_labels_same_zero | 0:0.3877, 1:0.2734, 2:0.1533, 3:0.1475, 4:0.3594, 5:0.3291, 6:0.4277, 7:0.4434, 8:0.2979, 9:0.1553 | 0:0.1230, 1:0.1104, 2:0.3770, 3:0.2383, 4:0.0508, 5:0.1855, 6:0.2734, 7:0.2969, 8:0.0762, 9:0.3037 |
| candidate_metadata_removed | 0:0.1250, 1:0.1250, 2:0.1250, 3:0.1250, 4:0.1250, 5:0.1250, 6:0.1250, 7:0.1250, 8:0.1250, 9:0.1250 | 0:0.1250, 1:0.1250, 2:0.1250, 3:0.1250, 4:0.1250, 5:0.1250, 6:0.1250, 7:0.1250, 8:0.1250, 9:0.1250 |
| candidate_only | 0:0.3105, 1:0.3301, 2:0.0762, 3:0.0762, 4:0.1279, 5:0.2051, 6:0.1934, 7:0.2676, 8:0.1562, 9:0.2402 | 0:0.0498, 1:0.1104, 2:0.2100, 3:0.1055, 4:0.1699, 5:0.0664, 6:0.2734, 7:0.2129, 8:0.0811, 9:0.1191 |
| family_id_removed | 0:1.0000, 1:1.0000, 2:1.0000, 3:1.0000, 4:0.9990, 5:1.0000, 6:1.0000, 7:1.0000, 8:1.0000, 9:1.0000 | 0:0.0576, 1:0.2402, 2:0.9902, 3:1.0000, 4:0.9990, 5:0.9990, 6:0.8252, 7:1.0000, 8:1.0000, 9:0.1494 |
| hidden_states_shuffled_across_examples | 0:0.2383, 1:0.2168, 2:0.2812, 3:0.2207, 4:0.2354, 5:0.2119, 6:0.2129, 7:0.2236, 8:0.2344, 9:0.2256 | 0:0.0566, 1:0.2393, 2:0.2637, 3:0.2676, 4:0.2354, 5:0.2451, 6:0.2129, 7:0.2393, 8:0.3320, 9:0.1582 |
| none | 0:1.0000, 1:1.0000, 2:1.0000, 3:1.0000, 4:0.9990, 5:1.0000, 6:1.0000, 7:1.0000, 8:1.0000, 9:1.0000 | 0:0.0576, 1:0.2402, 2:0.9902, 3:1.0000, 4:0.9990, 5:0.9990, 6:0.8252, 7:1.0000, 8:1.0000, 9:0.1494 |
| role_embeddings_zeroed | 0:0.4287, 1:0.3584, 2:0.4092, 3:0.0000, 4:0.2666, 5:0.2949, 6:0.3770, 7:0.2676, 8:0.1797, 9:0.1553 | 0:0.0088, 1:0.1104, 2:0.3398, 3:0.2432, 4:0.2959, 5:0.2539, 6:0.2734, 7:0.2734, 8:0.1562, 9:0.2168 |
| role_labels_random_independent | 0:0.3652, 1:0.2715, 2:0.3037, 3:0.2480, 4:0.3662, 5:0.3857, 6:0.3779, 7:0.3516, 8:0.2588, 9:0.2295 | 0:0.0625, 1:0.2422, 2:0.2549, 3:0.2939, 4:0.3174, 5:0.2324, 6:0.2520, 7:0.2979, 8:0.2471, 9:0.1523 |
| role_labels_random_permutation | 0:0.3818, 1:0.3574, 2:0.2812, 3:0.3467, 4:0.4717, 5:0.3750, 6:0.3867, 7:0.4131, 8:0.3154, 9:0.3203 | 0:0.0615, 1:0.2402, 2:0.1973, 3:0.3496, 4:0.4326, 5:0.3711, 6:0.3447, 7:0.3506, 8:0.3359, 9:0.1445 |
| role_pair_structure_randomized | 0:0.3506, 1:0.2939, 2:0.2354, 3:0.3213, 4:0.2520, 5:0.3887, 6:0.4111, 7:0.2754, 8:0.2754, 9:0.2852 | 0:0.1318, 1:0.1104, 2:0.1885, 3:0.3330, 4:0.3809, 5:0.3643, 6:0.3096, 7:0.3115, 8:0.2402, 9:0.1221 |
| view_only_without_candidate_metadata | 0:0.1250, 1:0.1250, 2:0.1250, 3:0.1250, 4:0.1250, 5:0.1250, 6:0.1250, 7:0.1250, 8:0.1250, 9:0.1250 | 0:0.1250, 1:0.1250, 2:0.1250, 3:0.1250, 4:0.1250, 5:0.1250, 6:0.1250, 7:0.1250, 8:0.1250, 9:0.1250 |
| view_shuffled | 0:0.1836, 1:0.2363, 2:0.1914, 3:0.1953, 4:0.2588, 5:0.2178, 6:0.1738, 7:0.2412, 8:0.1758, 9:0.2480 | 0:0.0615, 1:0.2402, 2:0.2461, 3:0.2393, 4:0.2295, 5:0.1738, 6:0.2012, 7:0.2139, 8:0.2627, 9:0.1475 |
| views_masked | 0:0.3105, 1:0.1621, 2:0.0781, 3:0.2891, 4:0.2910, 5:0.2764, 6:0.0801, 7:0.2666, 8:0.0801, 9:0.0957 | 0:0.0566, 1:0.2402, 2:0.0049, 3:0.2676, 4:0.2637, 5:0.3252, 6:0.2773, 7:0.2773, 8:0.1904, 9:0.1514 |

## Interpretation

- Randomized labels are balanced; no dataset-only randomized-label implementation bug was found.
- The role shuffle implementation is doing what it says mechanically, but it is not destructive enough for this pair-codebook dataset because candidate feature columns and pair structure survive.
- View masking is text-correct, but it is not candidate-metadata masking.
- The decisive failure mode is the Stage 3.3b candidate metadata/codebook shortcut, not a demonstrated true model effect.
