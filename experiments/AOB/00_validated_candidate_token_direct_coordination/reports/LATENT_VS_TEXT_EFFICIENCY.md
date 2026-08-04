# Latent vs Text Efficiency

## Scope

- Benchmark: `real_shared_weight_latent_coordination` Stage 3 semantic no-literal-cue test split.
- Architecture: `topk_attention_no_head`; no architecture changes were made.
- Latent methods count generated text tokens as zero. Dense latent communication is reported separately as floats/vectors, not text tokens.
- Text-only baseline uses textual summary/answer messages from agents plus candidate text at the coordinator.
- Accuracy, latency, memory, and token counts are measured from the exact saved Stage 3 checkpoints for trainable, frozen, text-only, and raw-latent methods.
- No refit-based latency or memory measurements are used in the main proof; refit drift is not applicable.
- Latent coordination is not claimed to be faster than text-only coordination unless the measured latency table shows it.
- CUDA device: `NVIDIA GeForce RTX 5070 Ti`
- Stage 3 controls still pass: `True`

## Efficiency Table

| method | checkpoint accuracy | input tok/ex | output comm tok/ex | latent floats/ex | total text tokens | total latent floats | latency ms/ex | GPU GB peak | acc/1k tok | acc/sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| trainable_topk_attention_no_head | 0.8829 | 2376.0 | 0.0 | 256.0 | 24329840 | 2621440 | 1.801 | 0.0885 | 0.372 | 490.28 |
| frozen_topk_attention_no_head | 0.4252 | 2376.0 | 0.0 | 256.0 | 24329840 | 2621440 | 1.634 | 0.0885 | 0.179 | 260.26 |
| text_only_multi_agent_baseline | 0.1227 | 2933.6 | 56.0 | 0.0 | 30613520 | 0 | 1.434 | 0.0144 | 0.041 | 85.52 |
| raw_latent_baseline | 0.1239 | 2376.0 | 0.0 | 256.0 | 24329840 | 2621440 | 1.595 | 0.0885 | 0.052 | 77.71 |
| explicit_evidence_oracle | 1.0000 | 93.0 | 0.0 | 0.0 | 952320 | 0 | 0.000 | 0.0137 | 10.753 | 7283072.49 |

## Stage 3 Control Gate

| criterion | pass | value |
|---|---|---|
| corruption controls and non-latent baselines remain near chance | True | `{'candidate_order_baseline': {'max_accuracy': 0.125, 'mean_accuracy': 0.125}, 'hidden_states_shuffled_across_examples': {'max_accuracy': 0.140625, 'mean_accuracy': 0.130859375}, 'majority_baseline': {'max_accuracy': 0.125, 'mean_accuracy': 0.125}, 'randomized_labels': {'max_accuracy': 0.1318359375, 'mean_accuracy': 0.1232421875}, 'raw_latent': {'max_accuracy': 0.1318359375, 'mean_accuracy': 0.12353515625}, 'role_labels_shuffled': {'max_accuracy': 0.2724609375, 'mean_accuracy': 0.1927734375}, 'text_only': {'max_accuracy': 0.1318359375, 'mean_accuracy': 0.12265625}, 'view_masked': {'max_accuracy': 0.1435546875, 'mean_accuracy': 0.1248046875}, 'view_shuffled': {'max_accuracy': 0.1455078125, 'mean_accuracy': 0.12333984375}}` |
| invariance controls preserve accuracy | True | `{'candidate_order_shuffled': {'mean_accuracy': 0.88447265625, 'mean_delta_from_trainable': 0.0009765625}, 'physical_order_shuffled_roles_preserved': {'mean_accuracy': 0.88349609375, 'mean_delta_from_trainable': 0.0}}` |
| role-label shuffle does not explain the result | True | `{'max_accuracy': 0.2724609375, 'mean_accuracy': 0.1927734375, 'mean_delta_from_trainable': -0.6907226562500001}` |
| leakage audits pass | True | `split_and_output` |

## Claim Gate

| condition | pass |
|---|---|
| trainable_latent_accuracy_high | True |
| text_only_near_chance_or_lower | True |
| latent_uses_fewer_output_communication_tokens | True |
| stage3_controls_still_pass | True |

## Per-Seed Accuracy Check

| seed | trainable | frozen | text-only | raw-latent | oracle |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.9189 | 0.8486 | 0.1201 | 0.1240 | 1.0000 |
| 1 | 0.7891 | 0.4766 | 0.1172 | 0.1338 | 1.0000 |
| 2 | 0.9629 | 0.6738 | 0.1104 | 0.1123 | 1.0000 |
| 3 | 0.9746 | 0.2588 | 0.1270 | 0.1279 | 1.0000 |
| 4 | 0.8672 | 0.4004 | 0.1318 | 0.1201 | 1.0000 |
| 5 | 0.9902 | 0.1719 | 0.1143 | 0.1279 | 1.0000 |
| 6 | 0.9365 | 0.3281 | 0.1260 | 0.1250 | 1.0000 |
| 7 | 0.9863 | 0.5488 | 0.1191 | 0.1182 | 1.0000 |
| 8 | 0.9170 | 0.3438 | 0.1289 | 0.1230 | 1.0000 |
| 9 | 0.4863 | 0.2012 | 0.1318 | 0.1270 | 1.0000 |

## Interpretation

The measured latent path is not faster than the text-only baseline in this run, so no speed advantage is claimed.
latent coordination is much more accurate than text-only coordination while using zero generated communication tokens on this controlled benchmark. This does not establish real-world coding success.
