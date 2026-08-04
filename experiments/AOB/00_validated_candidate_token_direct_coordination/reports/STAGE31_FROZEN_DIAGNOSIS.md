# Stage 3.1 Frozen Comparator Diagnosis

Architecture changes: none. The locked `topk_attention_no_head` checkpoints from Stage 3 hard validation were reloaded.

Stage 3.1 acceptance status: **not passed**. The diagnosis found dataset leakage sufficient to explain the high frozen score, and the hardening variants below were stress tests of existing checkpoints rather than retrained acceptance runs.

## Summary

| metric | value |
|---|---:|
| mean trainable accuracy | 0.9228 |
| mean frozen accuracy | 0.7404 |
| mean trainable-frozen delta | 0.1823 |
| candidate lexical-overlap baseline | 0.8234 |
| static import-frequency baseline | 0.5219 |
| exact gold symbol/module/path appears in any private view | 1.0000 |
| exact gold symbol appears in private views | 1.0000 |
| exact gold module appears in private views | 1.0000 |
| exact gold target file appears in private views | 1.0000 |
| gold import line exact in private views | 0.0000 |
| gold candidate near-verbatim in private views | 0.0233 |

## Acceptance Check

| criterion | status | value |
|---|---|---:|
| trainable >= 0.80 | pass | 0.9228 |
| frozen <= 0.55 | fail | 0.7404 |
| mean delta >= +0.25 | fail | 0.1823 |
| exact-symbol leakage audit | fail | 1.0000 |
| architecture unchanged | pass | none |
| hardened variants | diagnostic only | retraining not run |

## Per-Seed Diagnostics

| seed | trainable | frozen | delta | lexical overlap | static frequency | any exact gold value | gold import line exact | frozen correct/trainable wrong | trainable correct/frozen wrong |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.0000 | 0.8564 | 0.1436 | 0.7949 | 0.7158 | 1.0000 | 0.0000 | 0 | 147 |
| 1 | 0.9570 | 0.4648 | 0.4922 | 0.8047 | 0.3154 | 1.0000 | 0.0000 | 1 | 505 |
| 2 | 1.0000 | 1.0000 | 0.0000 | 0.7627 | 0.4395 | 1.0000 | 0.0000 | 0 | 0 |
| 3 | 0.8584 | 0.7080 | 0.1504 | 0.9746 | 0.7256 | 1.0000 | 0.0000 | 79 | 233 |
| 4 | 1.0000 | 0.6494 | 0.3506 | 0.6602 | 0.6475 | 1.0000 | 0.0000 | 0 | 359 |
| 5 | 0.8857 | 0.7197 | 0.1660 | 0.9990 | 0.5000 | 1.0000 | 0.0000 | 35 | 205 |
| 6 | 1.0000 | 0.7295 | 0.2705 | 1.0000 | 0.4922 | 1.0000 | 0.0000 | 0 | 277 |
| 7 | 0.7070 | 0.8887 | -0.1816 | 0.8135 | 0.4375 | 1.0000 | 0.0000 | 218 | 32 |
| 8 | 0.8193 | 0.7598 | 0.0596 | 0.6611 | 0.4453 | 1.0000 | 0.0000 | 141 | 202 |
| 9 | 1.0000 | 0.6279 | 0.3721 | 0.7637 | 0.5000 | 1.0000 | 0.0000 | 0 | 381 |

## Per-Family Frozen Accuracy

| seed | family | n | frozen accuracy |
|---:|---|---:|---:|
| 0 | test_real_import_restore_1 | 752 | 0.8630 |
| 0 | test_real_import_restore_2 | 44 | 1.0000 |
| 0 | test_real_import_restore_3 | 208 | 0.7885 |
| 0 | test_real_import_restore_4 | 20 | 1.0000 |
| 1 | test_real_import_restore_0 | 231 | 0.1818 |
| 1 | test_real_import_restore_2 | 25 | 0.0000 |
| 1 | test_real_import_restore_3 | 95 | 0.3579 |
| 1 | test_real_import_restore_5 | 673 | 0.5944 |
| 2 | test_real_import_restore_0 | 597 | 1.0000 |
| 2 | test_real_import_restore_3 | 228 | 1.0000 |
| 2 | test_real_import_restore_4 | 173 | 1.0000 |
| 2 | test_real_import_restore_6 | 26 | 1.0000 |
| 3 | test_real_import_restore_0 | 6 | 1.0000 |
| 3 | test_real_import_restore_1 | 263 | 0.6654 |
| 3 | test_real_import_restore_2 | 78 | 1.0000 |
| 3 | test_real_import_restore_3 | 677 | 0.6883 |
| 4 | test_real_import_restore_0 | 102 | 0.9706 |
| 4 | test_real_import_restore_1 | 31 | 1.0000 |
| 4 | test_real_import_restore_5 | 348 | 0.4914 |
| 4 | test_real_import_restore_6 | 543 | 0.6703 |
| 5 | test_real_import_restore_0 | 174 | 0.9943 |
| 5 | test_real_import_restore_3 | 614 | 0.5375 |
| 5 | test_real_import_restore_4 | 236 | 0.9915 |
| 6 | test_real_import_restore_1 | 360 | 0.8389 |
| 6 | test_real_import_restore_4 | 57 | 1.0000 |
| 6 | test_real_import_restore_6 | 607 | 0.6392 |
| 7 | test_real_import_restore_1 | 92 | 0.9891 |
| 7 | test_real_import_restore_2 | 11 | 0.0909 |
| 7 | test_real_import_restore_3 | 404 | 0.9307 |
| 7 | test_real_import_restore_5 | 191 | 0.8534 |
| 7 | test_real_import_restore_6 | 326 | 0.8558 |
| 8 | test_real_import_restore_2 | 82 | 0.7439 |
| 8 | test_real_import_restore_3 | 21 | 1.0000 |
| 8 | test_real_import_restore_4 | 219 | 0.7397 |
| 8 | test_real_import_restore_5 | 702 | 0.7607 |
| 9 | test_real_import_restore_2 | 38 | 0.9737 |
| 9 | test_real_import_restore_3 | 56 | 1.0000 |
| 9 | test_real_import_restore_4 | 199 | 0.5025 |
| 9 | test_real_import_restore_5 | 731 | 0.6156 |

## Harder Variant Stress Tests

These rows are checkpoint stress tests without retraining. They identify which hardening transformations remove the frozen shortcut; they are not Stage 3.1 acceptance runs.

| variant | trainable | frozen | delta | lexical overlap | exact gold value | near-verbatim candidate |
|---|---:|---:|---:|---:|---:|---:|
| symbol_redacted_views | 0.6219 | 0.4752 | 0.1467 | 0.6933 | 1.0000 | 0.0102 |
| candidate_name_redacted_views | 0.6099 | 0.4833 | 0.1266 | 0.7799 | 1.0000 | 0.0102 |
| module_path_redacted_views | 0.4038 | 0.2664 | 0.1374 | 0.8067 | 1.0000 | 0.0000 |
| cross_file_evidence_only | 0.2411 | 0.2204 | 0.0207 | 0.4941 | 0.0000 | 0.0000 |
| two_hop_evidence | 0.1833 | 0.1982 | -0.0149 | 0.4976 | 0.0233 | 0.0000 |
| same_package_distractors | 0.9245 | 0.7423 | 0.1822 | 0.8225 | 1.0000 | 0.0093 |

## Attention Inspection

Top-k rows list the selected prompt tokens from the active readout scorer. `private_view` means the token occurs before the candidate block in the clone prompt.

### Seed 2 Example test-00000

- label: 0; frozen prediction: 0; trainable prediction: 0
- gold patch values: `['output_oracle_features', 'src.telemetry.features', 'direct_from_import', 'src/coordinators/probes.py']`

| role | model | selected tokens |
|---|---|---|
| 0 symbol_kind | frozen | `32:observed:private_view:0.18`, `27:function:private_view:0.17`, `57:(:private_view:0.15`, `21:kind:private_view:0.11`, `99:block:candidate_block:0.10`, `58:batch:private_view:0.10`, `20:export:private_view:0.10`, `44:112:private_view:0.09` |
| 0 symbol_kind | trainable | `53:lambda:private_view:0.19`, `34:lines:private_view:0.13`, `122:/:candidate_block:0.12`, `39:/:private_view:0.12`, `92:/:candidate_block:0.12`, `94:/:candidate_block:0.11`, `23:the:private_view:0.11`, `96:.:candidate_block:0.10` |
| 1 provider_area | frozen | `48:(:private_view:0.21`, `124:import:candidate_block:0.15`, `30:provider:private_view:0.13`, `51:batch:private_view:0.12`, `114:test_shared_weight_cloned_agents:candidate_block:0.11`, `44:0066:private_view:0.10`, `17:::private_view:0.09`, `32:::private_view:0.09` |
| 1 provider_area | trainable | `34:/:private_view:0.82`, `33:src:private_view:0.06`, `74:/:candidate_block:0.03`, `72:/:candidate_block:0.02`, `113:/:candidate_block:0.02`, `59:int:private_view:0.02`, `23:provider:private_view:0.02`, `49:0067:private_view:0.02` |
| 2 provider_name_parity | frozen | `41:candidates:candidate_block:0.17`, `21:even:private_view:0.15`, `51:/:candidate_block:0.14`, `58:py:candidate_block:0.13`, `93:test_shared_weight_cloned_agents:candidate_block:0.12`, `32:import:private_view:0.10`, `39:name:private_view:0.09`, `102:py:candidate_block:0.09` |
| 2 provider_name_parity | trainable | `53:/:candidate_block:0.51`, `55:/:candidate_block:0.26`, `99:/:candidate_block:0.04`, `64:/:candidate_block:0.04`, `19:parity:private_view:0.04`, `60:/:candidate_block:0.04`, `51:/:candidate_block:0.04`, `125:/:candidate_block:0.03` |
| 3 import_location | frozen | `83:(:private_view:0.18`, `21:restored:private_view:0.15`, `51:import:private_view:0.13`, `30:py:private_view:0.12`, `102:/:candidate_block:0.12`, `27:/:private_view:0.11`, `20:the:private_view:0.10`, `32:block:private_view:0.10` |
| 3 import_location | trainable | `34:parity:private_view:0.31`, `63:0008:private_view:0.16`, `18:file:private_view:0.11`, `49:.:private_view:0.09`, `72:module:private_view:0.09`, `109:/:candidate_block:0.08`, `33:line:private_view:0.08`, `25:/:private_view:0.07` |

### Seed 2 Example test-00001

- label: 1; frozen prediction: 1; trainable prediction: 1
- gold patch values: `['build_strict_coordination_splits', 'src.datasets.strict_coordination', 'direct_from_import', 'tests/test_shared_weight_cloned_agents.py']`

| role | model | selected tokens |
|---|---|---|
| 0 symbol_kind | frozen | `27:function:private_view:0.17`, `32:observed:private_view:0.17`, `93:(:private_view:0.13`, `64:(:private_view:0.12`, `77:(:private_view:0.11`, `99:candidates:candidate_block:0.10`, `58:4:private_view:0.10`, `51:n_dev:private_view:0.10` |
| 0 symbol_kind | trainable | `55:16:private_view:0.18`, `116:/:candidate_block:0.14`, `99:candidates:candidate_block:0.13`, `34:lines:private_view:0.13`, `52:16:private_view:0.11`, `23:the:private_view:0.11`, `113:.:candidate_block:0.10`, `39:.:private_view:0.10` |
| 1 provider_area | frozen | `48:(:private_view:0.21`, `44:0038:private_view:0.14`, `51:config:private_view:0.14`, `30:provider:private_view:0.13`, `117:test_shared_weight_cloned_agents:candidate_block:0.10`, `57:seed:private_view:0.10`, `17:::private_view:0.09`, `58:::private_view:0.09` |
| 1 provider_area | trainable | `34:/:private_view:0.74`, `79:/:candidate_block:0.08`, `33:src:private_view:0.06`, `53:strictcoordinationconfig:private_view:0.04`, `74:/:candidate_block:0.03`, `72:/:candidate_block:0.02`, `59:int:private_view:0.02`, `116:/:candidate_block:0.02` |
| 2 provider_name_parity | frozen | `21:odd:private_view:0.27`, `41:candidates:candidate_block:0.13`, `51:/:candidate_block:0.11`, `91:py:candidate_block:0.11`, `63:py:candidate_block:0.11`, `99:import:candidate_block:0.10`, `124:test_shared_weight_cloned_agents:candidate_block:0.09`, `66:around:candidate_block:0.09` |
| 2 provider_name_parity | trainable | `53:/:candidate_block:0.51`, `121:/:candidate_block:0.24`, `93:/:candidate_block:0.05`, `58:/:candidate_block:0.04`, `60:/:candidate_block:0.04`, `51:/:candidate_block:0.04`, `88:/:candidate_block:0.04`, `95:/:candidate_block:0.03` |
| 3 import_location | frozen | `21:restored:private_view:0.16`, `76:(:private_view:0.15`, `32:parity:private_view:0.14`, `41:the:private_view:0.12`, `20:the:private_view:0.11`, `30:block:private_view:0.11`, `108:py:candidate_block:0.11`, `109:import:candidate_block:0.11` |
| 3 import_location | trainable | `34:even:private_view:0.86`, `45:/:private_view:0.04`, `103:/:candidate_block:0.02`, `33:::private_view:0.02`, `96:/:candidate_block:0.02`, `18:file:private_view:0.02`, `39:file:private_view:0.01`, `25:/:private_view:0.01` |

### Seed 2 Example test-00002

- label: 2; frozen prediction: 2; trainable prediction: 2
- gold patch values: `['SimulatedAgentConfig', 'src.agents.simulated', 'direct_from_import', 'tests/test_controls.py']`

| role | model | selected tokens |
|---|---|---|
| 0 symbol_kind | frozen | `41:swarm:private_view:0.15`, `124:import:candidate_block:0.13`, `21:kind:private_view:0.13`, `58:0013:private_view:0.12`, `47:(:private_view:0.12`, `20:export:private_view:0.12`, `51:hidden_dim:private_view:0.12`, `66:candidates:candidate_block:0.11` |
| 0 symbol_kind | trainable | `33:/:private_view:0.97`, `34:test_controls:private_view:0.02`, `78:/:candidate_block:0.00`, `52:8:private_view:0.00`, `113:/:candidate_block:0.00`, `54:num_layers:private_view:0.00`, `18:simulatedagentconfig:private_view:0.00`, `116:py:candidate_block:0.00` |
| 1 provider_area | frozen | `51:class:private_view:0.18`, `44:dataclass:private_view:0.15`, `45:(:private_view:0.13`, `30:provider:private_view:0.13`, `124:py:candidate_block:0.12`, `91:block:candidate_block:0.10`, `112:/:candidate_block:0.09`, `17:::private_view:0.09` |
| 1 provider_area | trainable | `34:/:private_view:0.78`, `79:/:candidate_block:0.07`, `33:src:private_view:0.06`, `121:/:candidate_block:0.03`, `52:simulatedagentconfig:private_view:0.02`, `23:provider:private_view:0.02`, `27:core:private_view:0.01`, `112:/:candidate_block:0.01` |
| 2 provider_name_parity | frozen | `21:odd:private_view:0.27`, `41:candidates:candidate_block:0.14`, `51:/:candidate_block:0.11`, `91:py:candidate_block:0.11`, `63:py:candidate_block:0.11`, `99:import:candidate_block:0.10`, `66:around:candidate_block:0.08`, `93:/:candidate_block:0.08` |
| 2 provider_name_parity | trainable | `53:/:candidate_block:0.54`, `121:/:candidate_block:0.22`, `93:/:candidate_block:0.05`, `58:/:candidate_block:0.04`, `60:/:candidate_block:0.04`, `51:/:candidate_block:0.04`, `95:/:candidate_block:0.03`, `88:/:candidate_block:0.03` |
| 3 import_location | frozen | `21:restored:private_view:0.16`, `32:parity:private_view:0.14`, `72:name:private_view:0.13`, `55:0001:private_view:0.13`, `41:the:private_view:0.12`, `20:the:private_view:0.12`, `30:block:private_view:0.11`, `79:0006:private_view:0.10` |
| 3 import_location | trainable | `45:/:private_view:0.25`, `34:odd:private_view:0.19`, `94:/:candidate_block:0.15`, `33:::private_view:0.10`, `18:file:private_view:0.09`, `25:/:private_view:0.08`, `39:file:private_view:0.07`, `92:/:candidate_block:0.06` |

### Seed 2 Example test-00003

- label: 3; frozen prediction: 3; trainable prediction: 3
- gold patch values: `['StrictCoordinationConfig', 'src.datasets.strict_coordination', 'direct_from_import', 'tests/test_shared_weight_cloned_agents.py']`

| role | model | selected tokens |
|---|---|---|
| 0 symbol_kind | frozen | `50:(:private_view:0.14`, `67:(:private_view:0.13`, `57:n_test:private_view:0.13`, `41:def:private_view:0.13`, `21:kind:private_view:0.12`, `109:import:candidate_block:0.12`, `124:py:candidate_block:0.11`, `20:export:private_view:0.11` |
| 0 symbol_kind | trainable | `33:/:private_view:0.92`, `39:0016:private_view:0.06`, `34:test_shared_weight_cloned_agents:private_view:0.01`, `91:/:candidate_block:0.00`, `121:/:candidate_block:0.00`, `55:16:private_view:0.00`, `93:/:candidate_block:0.00`, `74:candidates:candidate_block:0.00` |
| 1 provider_area | frozen | `51:class:private_view:0.18`, `44:dataclass:private_view:0.16`, `45:(:private_view:0.13`, `30:provider:private_view:0.13`, `124:py:candidate_block:0.13`, `91:block:candidate_block:0.10`, `112:/:candidate_block:0.09`, `17:::private_view:0.09` |
| 1 provider_area | trainable | `34:/:private_view:0.76`, `79:/:candidate_block:0.08`, `33:src:private_view:0.07`, `121:/:candidate_block:0.03`, `23:provider:private_view:0.02`, `112:/:candidate_block:0.01`, `52:strictcoordinationconfig:private_view:0.01`, `27:core:private_view:0.01` |
| 2 provider_name_parity | frozen | `21:odd:private_view:0.27`, `41:candidates:candidate_block:0.14`, `51:/:candidate_block:0.11`, `91:py:candidate_block:0.11`, `63:py:candidate_block:0.10`, `99:import:candidate_block:0.10`, `66:around:candidate_block:0.09`, `93:/:candidate_block:0.08` |
| 2 provider_name_parity | trainable | `53:/:candidate_block:0.50`, `121:/:candidate_block:0.25`, `93:/:candidate_block:0.06`, `58:/:candidate_block:0.04`, `60:/:candidate_block:0.04`, `51:/:candidate_block:0.04`, `88:/:candidate_block:0.04`, `95:/:candidate_block:0.03` |
| 3 import_location | frozen | `21:restored:private_view:0.16`, `32:parity:private_view:0.13`, `51:import:private_view:0.13`, `78:(:private_view:0.13`, `41:the:private_view:0.12`, `83:0007:private_view:0.12`, `20:the:private_view:0.11`, `63:0004:private_view:0.11` |
| 3 import_location | trainable | `34:even:private_view:0.89`, `45:/:private_view:0.03`, `33:::private_view:0.02`, `18:file:private_view:0.02`, `39:file:private_view:0.01`, `49:.:private_view:0.01`, `55:4:private_view:0.01`, `25:/:private_view:0.01` |

### Seed 7 Example test-00000

- label: 0; frozen prediction: 0; trainable prediction: 1
- gold patch values: `['ablate_telemetry_channel', 'src.telemetry.ablations', 'direct_from_import', 'src/experiments/run_synthetic.py']`

| role | model | selected tokens |
|---|---|---|
| 0 symbol_kind | frozen | `41:.:private_view:0.20`, `50:::private_view:0.14`, `44:332:private_view:0.12`, `68:::private_view:0.11`, `102:,:private_view:0.11`, `52:{:private_view:0.11`, `85:::private_view:0.11`, `80:,:private_view:0.10` |
| 0 symbol_kind | trainable | `33:use:private_view:0.44`, `36:src:private_view:0.14`, `57:ablate_telemetry_channel:private_view:0.09`, `47:,:private_view:0.08`, `35:in:private_view:0.07`, `103:channel_name:private_view:0.06`, `84:0521:private_view:0.06`, `69:dev:private_view:0.06` |
| 1 provider_area | frozen | `41:::private_view:0.33`, `62:if:private_view:0.14`, `50:::private_view:0.13`, `49:batch:private_view:0.09`, `51:attemptbatch:private_view:0.08`, `87:::candidate_block:0.08`, `3:artifact:private_view:0.07`, `8:artifact:private_view:0.07` |
| 1 provider_area | trainable | `33:src:private_view:0.30`, `57:-:private_view:0.15`, `18:src:private_view:0.14`, `58:attemptbatch:private_view:0.09`, `65:in:private_view:0.09`, `10:the:private_view:0.08`, `68:telemetry_channels:private_view:0.08`, `120:src:candidate_block:0.07` |
| 2 provider_name_parity | frozen | `41:candidates:candidate_block:0.20`, `50:a:candidate_block:0.15`, `21:odd:private_view:0.12`, `117:[:candidate_block:0.12`, `118:c:candidate_block:0.11`, `89:a:candidate_block:0.10`, `8:artifact:private_view:0.10`, `3:artifact:private_view:0.09` |
| 2 provider_name_parity | trainable | `57:.:candidate_block:0.28`, `33:statement:private_view:0.25`, `18:length:private_view:0.13`, `91:tests:candidate_block:0.08`, `47:-:candidate_block:0.08`, `36:from:private_view:0.06`, `110:src:candidate_block:0.06`, `10:the:private_view:0.06` |
| 3 import_location | frozen | `41:file:private_view:0.17`, `85:import:private_view:0.14`, `73:import:private_view:0.12`, `21:restored:private_view:0.12`, `117:import:candidate_block:0.12`, `88:::private_view:0.11`, `8:artifact:private_view:0.11`, `3:artifact:private_view:0.11` |
| 3 import_location | trainable | `57:0036:private_view:0.28`, `18:file:private_view:0.23`, `36:even:private_view:0.20`, `33:line:private_view:0.06`, `81:0040:private_view:0.06`, `118:block:candidate_block:0.06`, `110:src:candidate_block:0.06`, `85:import:private_view:0.05` |

### Seed 7 Example test-00023

- label: 7; frozen prediction: 6; trainable prediction: 7
- gold patch values: `['activation_features', 'src.telemetry.features', 'direct_from_import', 'tests/test_coordinators_smoke.py']`

| role | model | selected tokens |
|---|---|---|
| 0 symbol_kind | frozen | `41:::private_view:0.30`, `87:a:candidate_block:0.13`, `85:::candidate_block:0.11`, `21:kind:private_view:0.10`, `50:]:private_view:0.10`, `8:artifact:private_view:0.09`, `29:or:private_view:0.09`, `55:,:private_view:0.09` |
| 0 symbol_kind | trainable | `33:use:private_view:0.28`, `57:.:private_view:0.22`, `36:tests:private_view:0.16`, `18:activation_features:private_view:0.15`, `116:evaluation:candidate_block:0.05`, `47:batches:private_view:0.05`, `35:in:private_view:0.05`, `102:tests:candidate_block:0.05` |
| 1 provider_area | frozen | `41:::private_view:0.35`, `50:::private_view:0.13`, `53:attemptbatch:private_view:0.10`, `106:[:candidate_block:0.09`, `86:/:candidate_block:0.09`, `118:/:candidate_block:0.08`, `3:artifact:private_view:0.08`, `117:experiments:candidate_block:0.08` |
| 1 provider_area | trainable | `57:layers:private_view:0.32`, `33:src:private_view:0.21`, `47:activation_features:private_view:0.12`, `18:src:private_view:0.11`, `53:attemptbatch:private_view:0.08`, `10:the:private_view:0.06`, `115:src:candidate_block:0.05`, `87:tests:candidate_block:0.05` |
| 2 provider_name_parity | frozen | `41:candidates:candidate_block:0.18`, `21:even:private_view:0.18`, `50:a:candidate_block:0.14`, `117:[:candidate_block:0.11`, `118:c:candidate_block:0.10`, `85:a:candidate_block:0.10`, `86:/:candidate_block:0.10`, `8:artifact:private_view:0.09` |
| 2 provider_name_parity | trainable | `33:statement:private_view:0.32`, `18:length:private_view:0.17`, `57:b:candidate_block:0.11`, `47:-:candidate_block:0.10`, `36:from:private_view:0.09`, `110:src:candidate_block:0.08`, `10:the:private_view:0.07`, `52:tests:candidate_block:0.07` |
| 3 import_location | frozen | `83:none:private_view:0.14`, `73:::private_view:0.14`, `78:def:private_view:0.12`, `21:restored:private_view:0.12`, `87:::candidate_block:0.12`, `41:the:private_view:0.12`, `70:import:private_view:0.12`, `89:a:candidate_block:0.12` |
| 3 import_location | trainable | `18:file:private_view:0.25`, `57:0017:private_view:0.22`, `33:::private_view:0.12`, `36:non:private_view:0.11`, `118:evaluation:candidate_block:0.10`, `35:nearby:private_view:0.07`, `116:src:candidate_block:0.07`, `97:tests:candidate_block:0.06` |

### Seed 7 Example test-00002

- label: 2; frozen prediction: 2; trainable prediction: 2
- gold patch values: `['SimulatedAgentConfig', 'src.agents.simulated', 'direct_from_import', 'tests/test_coordinators_smoke.py']`

| role | model | selected tokens |
|---|---|---|
| 0 symbol_kind | frozen | `41:0025:private_view:0.19`, `44:simulatedagentswarm:private_view:0.13`, `50:n_agents:private_view:0.12`, `49:(:private_view:0.12`, `70:simulatedagentswarm:private_view:0.12`, `39:,:private_view:0.11`, `68:::private_view:0.11`, `73:::private_view:0.11` |
| 0 symbol_kind | trainable | `57:2:private_view:0.32`, `18:simulatedagentconfig:private_view:0.17`, `112:tests:candidate_block:0.10`, `47:::private_view:0.09`, `67:0071:private_view:0.08`, `50:n_agents:private_view:0.08`, `105:tests:candidate_block:0.08`, `36:py:private_view:0.07` |
| 1 provider_area | frozen | `41:::private_view:0.34`, `50:::private_view:0.13`, `102:import:candidate_block:0.10`, `68:::candidate_block:0.10`, `55:::private_view:0.09`, `85:tests:candidate_block:0.08`, `86:/:candidate_block:0.08`, `53:::private_view:0.08` |
| 1 provider_area | trainable | `33:src:private_view:0.23`, `47:true:private_view:0.18`, `57:::private_view:0.17`, `120:tests:candidate_block:0.12`, `18:src:private_view:0.10`, `85:tests:candidate_block:0.07`, `10:the:private_view:0.06`, `118:b:candidate_block:0.06` |
| 2 provider_name_parity | frozen | `41:candidates:candidate_block:0.20`, `50:a:candidate_block:0.15`, `21:odd:private_view:0.13`, `85:a:candidate_block:0.11`, `86:/:candidate_block:0.11`, `87:tests:candidate_block:0.11`, `8:artifact:private_view:0.10`, `3:artifact:private_view:0.10` |
| 2 provider_name_parity | trainable | `33:statement:private_view:0.30`, `18:length:private_view:0.17`, `57:b:candidate_block:0.11`, `47:-:candidate_block:0.11`, `122:tests:candidate_block:0.09`, `36:from:private_view:0.08`, `10:the:private_view:0.08`, `52:tests:candidate_block:0.07` |
| 3 import_location | frozen | `86:,:private_view:0.18`, `89:::candidate_block:0.13`, `21:restored:private_view:0.12`, `41:the:private_view:0.12`, `85:activationpcamlpcoordinator:private_view:0.11`, `8:artifact:private_view:0.11`, `7:location:private_view:0.11`, `3:artifact:private_view:0.11` |
| 3 import_location | trainable | `18:file:private_view:0.25`, `57:0001:private_view:0.22`, `33:::private_view:0.13`, `36:non:private_view:0.11`, `35:nearby:private_view:0.07`, `118:src:candidate_block:0.07`, `99:tests:candidate_block:0.07`, `106:tests:candidate_block:0.07` |

### Seed 7 Example test-00001

- label: 1; frozen prediction: 0; trainable prediction: 0
- gold patch values: `['ActivationClusterRouter', 'src.coordinators.activation', 'direct_from_import', 'tests/test_coordinators_smoke.py']`

| role | model | selected tokens |
|---|---|---|
| 0 symbol_kind | frozen | `41:):private_view:0.23`, `18:activationclusterrouter:private_view:0.14`, `49:,:private_view:0.12`, `85:import:candidate_block:0.12`, `21:kind:private_view:0.10`, `55:::private_view:0.10`, `80:experiments:candidate_block:0.09`, `8:artifact:private_view:0.09` |
| 0 symbol_kind | trainable | `57:patch:candidate_block:0.34`, `18:activationclusterrouter:private_view:0.16`, `47:clusters:private_view:0.11`, `115:tests:candidate_block:0.09`, `36:py:private_view:0.08`, `108:tests:candidate_block:0.07`, `32:tests:private_view:0.07`, `85:import:candidate_block:0.07` |
| 1 provider_area | frozen | `50:::private_view:0.19`, `85:import:candidate_block:0.14`, `41:py:private_view:0.14`, `3:artifact:private_view:0.11`, `115:tests:candidate_block:0.11`, `8:artifact:private_view:0.11`, `120:import:candidate_block:0.10`, `29:or:private_view:0.10` |
| 1 provider_area | trainable | `33:file:private_view:0.52`, `57:patch:candidate_block:0.18`, `18:src:private_view:0.07`, `47:::private_view:0.06`, `115:tests:candidate_block:0.05`, `108:tests:candidate_block:0.04`, `10:the:private_view:0.04`, `35:src:private_view:0.04` |
| 2 provider_name_parity | frozen | `41:activation:private_view:0.20`, `21:even:private_view:0.20`, `85:activationclusterrouter:candidate_block:0.13`, `8:artifact:private_view:0.10`, `3:artifact:private_view:0.09`, `93:a:candidate_block:0.09`, `86:[:candidate_block:0.09`, `102:tests:candidate_block:0.09` |
| 2 provider_name_parity | trainable | `33:statement:private_view:0.31`, `18:length:private_view:0.19`, `47:[:candidate_block:0.09`, `118:activation:candidate_block:0.09`, `36:from:private_view:0.09`, `102:tests:candidate_block:0.08`, `10:the:private_view:0.07`, `95:tests:candidate_block:0.07` |
| 3 import_location | frozen | `86:,:private_view:0.17`, `85:activationpoolingmlpcoordinator:private_view:0.13`, `89:::candidate_block:0.13`, `21:restored:private_view:0.12`, `73:import:private_view:0.12`, `41:the:private_view:0.11`, `8:artifact:private_view:0.11`, `78:,:private_view:0.11` |
| 3 import_location | trainable | `18:file:private_view:0.29`, `33:::private_view:0.18`, `36:non:private_view:0.13`, `57:0002:private_view:0.10`, `35:nearby:private_view:0.08`, `85:activationpoolingmlpcoordinator:private_view:0.08`, `24:tests:private_view:0.07`, `86:,:private_view:0.07` |

## Interpretation

The frozen comparator is not solving open-ended code repair. It is exploiting exact lexical evidence made available inside the private artifacts, especially the reference token, provider module/path, target file, and unredacted import-block text. The candidate lexical-overlap baseline measures this shortcut directly.

Most conservative valid claim: the current dataset is invalid for the intended latent-coordination claim until exact symbol/module/path leakage and candidate-string leakage are removed and the hardening variants are retrained under the locked architecture.
