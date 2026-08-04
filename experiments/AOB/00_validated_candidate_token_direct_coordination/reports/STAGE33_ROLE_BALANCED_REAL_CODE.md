# Stage 3.3 Role-Balanced Real-Code Import-Restoration Validation

## Scope/Config

- Architecture: `topk_attention_no_head`
- Locked architecture unchanged: topk_attention_no_head, shared-weight cloned agent, active/top-k token readout, no message head, no private-cue auxiliary loss, candidate-query coordinator.
- Comparator: exact frozen same-architecture comparator.
- Dataset mode: `real_import_restore_role_balanced_v2` / `import_restore_role_balanced_v2_stage33b`.
- Split sizes: train `2048`, dev `512`, test `1024`; seeds `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`.
- Mixed precision: `bf16`.

## Stage 3.2 Role Diagnosis

- Failed Stage 3.2 seeds: `[0, 1]`
- Masked-critical roles by seed: `{'0': [0], '1': [1]}`
- Single-role-sufficient roles by seed: `{'0': [3], '1': []}`
- Diagnosis result: no exact text leakage; failures were model/protocol role dependence plus family concentration. Full examples are in `results/stage33_role_diagnosis.json`.

## Pre-Training Dataset Diagnostics

- Dataset validity passed: `True`
- Candidate lexical-overlap baseline: `0.1250`
- Static import-frequency baseline: `0.1370`
- Max single-view text baseline: `0.1309`
- Single-role structured oracle means: `{'0': 0.3341796875, '1': 0.334765625, '2': 0.25009765625, '3': 0.24912109375}`
- Pairwise-role structured oracle means: `{'0,1': 1.0, '0,2': 0.5009765625, '0,3': 0.49970703125, '1,2': 0.50009765625, '1,3': 0.49912109375, '2,3': 1.0}`
- All-role structured oracle: `1.0000`

## Per-Seed Accuracy

| seed | batch | trainable | frozen | delta | text | raw | majority | cand-order | max single-view | full ctx | random | hidden shuffle | view masked | view shuffled | role shuffle | physical order | cand-order shuffle | all-role oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 32 | 1.0000 | 0.0615 | 0.9385 | 0.1318 | 0.1270 | 0.1250 | 0.1250 | 0.1318 | 0.1250 | 0.0000 | 0.2383 | 0.3105 | 0.2100 | 0.4082 | 1.0000 | 1.0000 | 1.0000 |
| 1 | 32 | 1.0000 | 0.2109 | 0.7891 | 0.1221 | 0.1250 | 0.1250 | 0.1250 | 0.1289 | 0.1250 | 0.2021 | 0.2051 | 0.0850 | 0.1973 | 0.3896 | 1.0000 | 1.0000 | 1.0000 |
| 2 | 32 | 1.0000 | 0.9727 | 0.0273 | 0.1152 | 0.1250 | 0.1250 | 0.1250 | 0.1260 | 0.1250 | 0.1777 | 0.1992 | 0.0000 | 0.1748 | 0.3174 | 1.0000 | 1.0000 | 1.0000 |
| 3 | 32 | 1.0000 | 1.0000 | 0.0000 | 0.1230 | 0.1211 | 0.1250 | 0.1250 | 0.1309 | 0.1250 | 0.1270 | 0.2207 | 0.2891 | 0.1865 | 0.3486 | 1.0000 | 1.0000 | 1.0000 |
| 4 | 32 | 0.9990 | 1.0000 | -0.0010 | 0.1230 | 0.1250 | 0.1250 | 0.1250 | 0.1260 | 0.1250 | 0.1396 | 0.2354 | 0.2900 | 0.2383 | 0.4551 | 0.9990 | 0.9990 | 1.0000 |
| 5 | 32 | 1.0000 | 0.9980 | 0.0020 | 0.1289 | 0.1113 | 0.1250 | 0.1250 | 0.1289 | 0.1250 | 0.1377 | 0.2119 | 0.2764 | 0.2275 | 0.3896 | 1.0000 | 1.0000 | 1.0000 |
| 6 | 32 | 1.0000 | 0.8398 | 0.1602 | 0.1230 | 0.1191 | 0.1250 | 0.1250 | 0.1338 | 0.1250 | 0.2920 | 0.2148 | 0.0791 | 0.1768 | 0.3760 | 1.0000 | 1.0000 | 1.0000 |
| 7 | 32 | 1.0000 | 1.0000 | 0.0000 | 0.1250 | 0.1064 | 0.1250 | 0.1250 | 0.1367 | 0.1250 | 0.1455 | 0.2227 | 0.2666 | 0.2402 | 0.4004 | 1.0000 | 0.9990 | 1.0000 |
| 8 | 32 | 1.0000 | 1.0000 | 0.0000 | 0.1123 | 0.1162 | 0.1250 | 0.1250 | 0.1250 | 0.1250 | 0.4121 | 0.2344 | 0.0801 | 0.1699 | 0.2764 | 1.0000 | 1.0000 | 1.0000 |
| 9 | 32 | 1.0000 | 0.1641 | 0.8359 | 0.1250 | 0.1250 | 0.1250 | 0.1250 | 0.1289 | 0.1250 | 0.1162 | 0.2266 | 0.1162 | 0.2490 | 0.3428 | 1.0000 | 1.0000 | 1.0000 |

## Mean/CI

- Mean trainable accuracy: `0.9999`
- Mean frozen accuracy: `0.7247`
- Mean delta: `0.2752`
- Bootstrap 95% CI: `[0.0786, 0.5279]`

## Per-Role Ablation

| seed | base | role 0 masked | role 1 masked | role 2 masked | role 3 masked | one-role failure |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 1.0000 | 0.5488 | 0.3105 | 0.6680 | 0.8047 | False |
| 1 | 1.0000 | 0.8838 | 0.8008 | 0.5566 | 0.5098 | False |
| 2 | 1.0000 | 0.7041 | 0.6387 | 0.9150 | 0.8164 | False |
| 3 | 1.0000 | 0.5000 | 0.4414 | 0.8496 | 0.6641 | False |
| 4 | 0.9990 | 0.5088 | 0.5186 | 0.9990 | 0.9980 | False |
| 5 | 1.0000 | 1.0000 | 1.0000 | 0.5059 | 0.5352 | False |
| 6 | 1.0000 | 0.7139 | 0.5576 | 0.5205 | 0.4756 | False |
| 7 | 1.0000 | 0.5088 | 0.5430 | 1.0000 | 1.0000 | False |
| 8 | 1.0000 | 0.6191 | 0.8965 | 0.6211 | 0.7861 | False |
| 9 | 1.0000 | 0.4238 | 0.5674 | 0.8564 | 0.9395 | False |

## Gate Table

| criterion | pass | value |
|---|---|---|
| pre-training role-balanced dataset diagnostics passed | True | `[{"gate": "exact gold symbol leakage == 0.0000", "pass": true, "value": 0.0}, {"gate": "exact gold module path leakage == 0.0000", "pass": true, "value": 0.0}, {"gate": "exact gold full import leakage == 0.0000", "pass": true, "value": 0.0}, {"gate": "exact candidate patch text leakage == 0.0000", "pass": true, "value": 0.0}, {"gate": "exact candidate symbol/module/path leakage == 0.0000", "pass": true, "value": 0.0}, {"gate": "candidate lexical-overlap baseline <= 0.2250 mean", "pass": true, "value": 0.125}, {"gate": "static import-frequency baseline <= 0.3000 mean", "pass": true, "value": 0.13701171875}, {"gate": "single-role lexical-overlap baselines <= 0.2250 mean", "pass": true, "value": {"0": 0.125, "1": 0.125, "2": 0.125, "3": 0.125}}, {"gate": "single-role candidate-frequency baselines <= 0.3000 mean", "pass": true, "value": {"0": 0.25146484375, "1": 0.28203125, "2": 0.2262695312` |
| completed Stage 3.3 seeds >= 10 | True | `10` |
| trainable beats frozen on at least 8/10 seeds | False | `6` |
| mean trainable-frozen delta >= +0.20 | True | `0.2751953125` |
| bootstrap 95% CI lower bound for delta > +0.05 | True | `[0.07862792968750003, 0.5279492187499999]` |
| trainable beats text-only, raw-latent, majority, candidate-order, and full-context baselines by mean accuracy | True | `{"candidate_order_mean": 0.125, "full_context_mean": 0.125, "majority_mean": 0.125, "raw_latent_mean": 0.1201171875, "text_only_mean": 0.12294921875, "trainable_mean": 0.99990234375}` |
| trainable beats every single-view baseline by mean accuracy | True | `{"single_view_text_role_0": 0.12158203125, "single_view_text_role_1": 0.1259765625, "single_view_text_role_2": 0.12666015625, "single_view_text_role_3": 0.1251953125}` |
| corruption controls plus majority/candidate-order remain near chance | False | `{"candidate_order_baseline": {"max_accuracy": 0.125, "mean_accuracy": 0.125}, "hidden_states_shuffled_across_examples": {"max_accuracy": 0.23828125, "mean_accuracy": 0.2208984375}, "majority_baseline": {"max_accuracy": 0.125, "mean_accuracy": 0.125}, "randomized_labels": {"max_accuracy": 0.412109375, "mean_accuracy": 0.175}, "role_labels_shuffled": {"max_accuracy": 0.455078125, "mean_accuracy": 0.37041015625}, "view_masked": {"max_accuracy": 0.310546875, "mean_accuracy": 0.179296875}, "view_shuffled": {"max_accuracy": 0.2490234375, "mean_accuracy": 0.20703125}}` |
| invariance controls preserve accuracy | True | `{"candidate_order_shuffled": {"mean_accuracy": 0.9998046875, "mean_delta_from_trainable": -9.76562500000222e-05}, "physical_order_shuffled_roles_preserved": {"mean_accuracy": 0.99990234375, "mean_delta_from_trainable": 0.0}}` |
| role-label shuffle drops substantially or remains near chance | True | `{"max_accuracy": 0.455078125, "mean_accuracy": 0.37041015625, "mean_delta_from_trainable": -0.6294921874999999}` |
| candidate lexical-overlap baseline <= 0.225 | True | `0.125` |
| static import-frequency baseline <= 0.300 | True | `0.13701171875` |
| no single-role structured oracle > 0.35 mean accuracy | True | `{"0": 0.3341796875, "1": 0.334765625, "2": 0.25009765625, "3": 0.24912109375}` |
| at least one pairwise-role structured oracle > 0.70 mean accuracy | True | `{"0,1": 1.0, "0,2": 0.5009765625, "0,3": 0.49970703125, "1,2": 0.50009765625, "1,3": 0.49912109375, "2,3": 1.0}` |
| all-role structured oracle >= 0.90 | True | `1.0` |
| no single role explains trainable result in more than 1/10 seeds | True | `0` |
| leakage audits pass | True | `split_candidate_patch_hash_and_output` |
| shared trainable model receives gradients and changes | True | `all_completed_seeds` |
| frozen comparator remains frozen | True | `all_completed_seeds` |

## Conservative Interpretation

Stage 3.3 role-balanced criteria pass: `False`.
Do not claim success. Failed gates remain visible in the table above.
