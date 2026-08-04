# Stage 8A Baseline Capacity

Decision: STAGE8A_BASELINE_ESTABLISHED

Explicit statement: NO_10X_CLAIM. Stage 8A establishes baseline capacity only and does not run architecture search.

## Capacity By Baseline

| Baseline | Kind | Capacity C | Capacity/Compute |
|---|---:|---:|---:|
| standard_monolithic_transformer_full_context | monolithic_transformer | 0 | 0.00000000 |
| random_candidate_baseline | random_candidate | 0 | 0.00000000 |
| candidate_only_baseline | candidate_only | 0 | 0.00000000 |
| evidence_only_baseline | evidence_only | 0 | 0.00000000 |
| query_only_baseline | query_only | 0 | 0.00000000 |
| retrieval_topk_baseline | retrieval_topk | 0 | 0.00000000 |
| same_parameter_count_monolithic_transformer | monolithic_transformer | 0 | 0.00000000 |
| same_compute_budget_monolithic_transformer | monolithic_transformer | 0 | 0.00000000 |
| initial_latent_baseline_diagnostic | trainable_latent | 0 | 0.00000000 |

## Matched Baseline Summary

- Best matched monolithic baseline capacity: C=0 (standard_monolithic_transformer_full_context)
- Best non-oracle baseline capacity: C=0 (standard_monolithic_transformer_full_context)
- Chance level: 0.125

## Accuracy Vs N

- standard_monolithic_transformer_full_context: {'1024': 0.09722222222222222, '128': 0.11458333333333334, '16': 0.1701388888888889, '256': 0.12152777777777778, '32': 0.13194444444444445, '512': 0.15625, '64': 0.1423611111111111, '8': 0.14583333333333334}
- random_candidate_baseline: {'1024': 0.13541666666666669, '128': 0.1423611111111111, '16': 0.10763888888888888, '256': 0.12152777777777778, '32': 0.09722222222222222, '512': 0.1076388888888889, '64': 0.125, '8': 0.1388888888888889}
- candidate_only_baseline: {'1024': 0.1388888888888889, '128': 0.1527777777777778, '16': 0.11805555555555555, '256': 0.11805555555555555, '32': 0.14583333333333334, '512': 0.10069444444444445, '64': 0.13541666666666669, '8': 0.13541666666666666}
- evidence_only_baseline: {'1024': 0.1388888888888889, '128': 0.1423611111111111, '16': 0.10416666666666667, '256': 0.13541666666666666, '32': 0.1284722222222222, '512': 0.125, '64': 0.12152777777777778, '8': 0.11805555555555555}
- query_only_baseline: {'1024': 0.06944444444444445, '128': 0.10763888888888888, '16': 0.0763888888888889, '256': 0.14583333333333334, '32': 0.15625, '512': 0.11111111111111112, '64': 0.11458333333333334, '8': 0.10069444444444445}
- retrieval_topk_baseline: {'1024': 0.1423611111111111, '128': 0.2604166666666667, '16': 0.4131944444444445, '256': 0.14583333333333334, '32': 0.3055555555555556, '512': 0.1701388888888889, '64': 0.2673611111111111, '8': 0.5486111111111112}
- same_parameter_count_monolithic_transformer: {'1024': 0.09722222222222222, '128': 0.11458333333333334, '16': 0.1701388888888889, '256': 0.12152777777777778, '32': 0.13194444444444445, '512': 0.15625, '64': 0.1423611111111111, '8': 0.14583333333333334}
- same_compute_budget_monolithic_transformer: {'1024': 0.09375, '128': 0.10416666666666666, '16': 0.1527777777777778, '256': 0.11458333333333333, '32': 0.13194444444444445, '512': 0.1423611111111111, '64': 0.1423611111111111, '8': 0.13194444444444445}
- initial_latent_baseline_diagnostic: {'1024': 0.1076388888888889, '128': 0.12152777777777778, '16': 0.1701388888888889, '256': 0.13194444444444445, '32': 0.125, '512': 0.1423611111111111, '64': 0.13194444444444445, '8': 0.09722222222222222}

## Compute Vs N

- standard_monolithic_transformer_full_context: {'8': 2949120.0, '16': 6291456.0, '32': 16515072.0, '64': 51118080.0, '128': 176947200.0, '256': 655097856.0, '512': 2517368832.0, '1024': 9865789440.0}
- random_candidate_baseline: {'8': 9216.0, '16': 18432.0, '32': 36864.0, '64': 73728.0, '128': 147456.0, '256': 294912.0, '512': 589824.0, '1024': 1179648.0}
- candidate_only_baseline: {'8': 9216.0, '16': 18432.0, '32': 36864.0, '64': 73728.0, '128': 147456.0, '256': 294912.0, '512': 589824.0, '1024': 1179648.0}
- evidence_only_baseline: {'8': 9216.0, '16': 18432.0, '32': 36864.0, '64': 73728.0, '128': 147456.0, '256': 294912.0, '512': 589824.0, '1024': 1179648.0}
- query_only_baseline: {'8': 9216.0, '16': 9216.0, '32': 9216.0, '64': 9216.0, '128': 9216.0, '256': 9216.0, '512': 9216.0, '1024': 9216.0}
- retrieval_topk_baseline: {'8': 9216.0, '16': 18432.0, '32': 36864.0, '64': 73728.0, '128': 147456.0, '256': 294912.0, '512': 589824.0, '1024': 1179648.0}
- same_parameter_count_monolithic_transformer: {'8': 2949120.0, '16': 6291456.0, '32': 16515072.0, '64': 51118080.0, '128': 176947200.0, '256': 655097856.0, '512': 2517368832.0, '1024': 9865789440.0}
- same_compute_budget_monolithic_transformer: {'8': 1474560.0, '16': 3145728.0, '32': 8257536.0, '64': 25559040.0, '128': 88473600.0, '256': 327548928.0, '512': 1258684416.0, '1024': 4932894720.0}
- initial_latent_baseline_diagnostic: {'8': 70656.0, '16': 71680.0, '32': 73728.0, '64': 77824.0, '128': 86016.0, '256': 102400.0, '512': 135168.0, '1024': 200704.0}

## Capacity By Task Family

### standard_monolithic_transformer_full_context
- compositional_role_evidence: C=0
- conflict_resolution: C=0
- constraint_satisfaction: C=0
- multi_hop_binding: C=0
- needle_binding: C=0
- sparse_relevant_evidence: C=0
### random_candidate_baseline
- compositional_role_evidence: C=0
- conflict_resolution: C=0
- constraint_satisfaction: C=0
- multi_hop_binding: C=0
- needle_binding: C=0
- sparse_relevant_evidence: C=0
### candidate_only_baseline
- compositional_role_evidence: C=0
- conflict_resolution: C=0
- constraint_satisfaction: C=0
- multi_hop_binding: C=0
- needle_binding: C=0
- sparse_relevant_evidence: C=0
### evidence_only_baseline
- compositional_role_evidence: C=0
- conflict_resolution: C=0
- constraint_satisfaction: C=0
- multi_hop_binding: C=0
- needle_binding: C=0
- sparse_relevant_evidence: C=0
### query_only_baseline
- compositional_role_evidence: C=0
- conflict_resolution: C=0
- constraint_satisfaction: C=0
- multi_hop_binding: C=0
- needle_binding: C=0
- sparse_relevant_evidence: C=0
### retrieval_topk_baseline
- compositional_role_evidence: C=0
- conflict_resolution: C=0
- constraint_satisfaction: C=0
- multi_hop_binding: C=0
- needle_binding: C=0
- sparse_relevant_evidence: C=0
### same_parameter_count_monolithic_transformer
- compositional_role_evidence: C=0
- conflict_resolution: C=0
- constraint_satisfaction: C=0
- multi_hop_binding: C=0
- needle_binding: C=0
- sparse_relevant_evidence: C=0
### same_compute_budget_monolithic_transformer
- compositional_role_evidence: C=0
- conflict_resolution: C=0
- constraint_satisfaction: C=0
- multi_hop_binding: C=0
- needle_binding: C=0
- sparse_relevant_evidence: C=0
### initial_latent_baseline_diagnostic
- compositional_role_evidence: C=0
- conflict_resolution: C=0
- constraint_satisfaction: C=0
- multi_hop_binding: C=0
- needle_binding: C=0
- sparse_relevant_evidence: C=0

## Controls And Shortcut Audits

- standard_monolithic_transformer_full_context: {'cross_task_evidence_shuffle': '0/24 pass', 'cross_task_query_shuffle': '0/24 pass', 'distractor_only': '0/24 pass', 'evidence_candidate_mismatch': '0/24 pass', 'randomized_candidate_order_with_label_remap': '24/24 pass', 'randomized_evidence_block_order': '24/24 pass', 'randomized_labels': '1/24 pass', 'schema_template_only': '2/24 pass'}
- random_candidate_baseline: {'cross_task_evidence_shuffle': '0/24 pass', 'cross_task_query_shuffle': '0/24 pass', 'distractor_only': '0/24 pass', 'evidence_candidate_mismatch': '2/24 pass', 'randomized_candidate_order_with_label_remap': '18/24 pass', 'randomized_evidence_block_order': '24/24 pass', 'randomized_labels': '0/24 pass', 'schema_template_only': '0/24 pass'}
- candidate_only_baseline: {'cross_task_evidence_shuffle': '0/24 pass', 'cross_task_query_shuffle': '0/24 pass', 'distractor_only': '0/24 pass', 'evidence_candidate_mismatch': '2/24 pass', 'randomized_candidate_order_with_label_remap': '15/24 pass', 'randomized_evidence_block_order': '24/24 pass', 'randomized_labels': '0/24 pass', 'schema_template_only': '0/24 pass'}
- evidence_only_baseline: {'cross_task_evidence_shuffle': '0/24 pass', 'cross_task_query_shuffle': '0/24 pass', 'distractor_only': '0/24 pass', 'evidence_candidate_mismatch': '2/24 pass', 'randomized_candidate_order_with_label_remap': '16/24 pass', 'randomized_evidence_block_order': '24/24 pass', 'randomized_labels': '0/24 pass', 'schema_template_only': '0/24 pass'}
- query_only_baseline: {'cross_task_evidence_shuffle': '0/24 pass', 'cross_task_query_shuffle': '1/24 pass', 'distractor_only': '0/24 pass', 'evidence_candidate_mismatch': '0/24 pass', 'randomized_candidate_order_with_label_remap': '14/24 pass', 'randomized_evidence_block_order': '24/24 pass', 'randomized_labels': '0/24 pass', 'schema_template_only': '0/24 pass'}
- retrieval_topk_baseline: {'cross_task_evidence_shuffle': '15/24 pass', 'cross_task_query_shuffle': '0/24 pass', 'distractor_only': '14/24 pass', 'evidence_candidate_mismatch': '13/24 pass', 'randomized_candidate_order_with_label_remap': '22/24 pass', 'randomized_evidence_block_order': '24/24 pass', 'randomized_labels': '16/24 pass', 'schema_template_only': '13/24 pass'}
- same_parameter_count_monolithic_transformer: {'cross_task_evidence_shuffle': '0/24 pass', 'cross_task_query_shuffle': '0/24 pass', 'distractor_only': '0/24 pass', 'evidence_candidate_mismatch': '0/24 pass', 'randomized_candidate_order_with_label_remap': '24/24 pass', 'randomized_evidence_block_order': '24/24 pass', 'randomized_labels': '1/24 pass', 'schema_template_only': '2/24 pass'}
- same_compute_budget_monolithic_transformer: {'cross_task_evidence_shuffle': '0/24 pass', 'cross_task_query_shuffle': '0/24 pass', 'distractor_only': '0/24 pass', 'evidence_candidate_mismatch': '0/24 pass', 'randomized_candidate_order_with_label_remap': '24/24 pass', 'randomized_evidence_block_order': '24/24 pass', 'randomized_labels': '1/24 pass', 'schema_template_only': '1/24 pass'}
- initial_latent_baseline_diagnostic: {'cross_task_evidence_shuffle': '0/24 pass', 'cross_task_query_shuffle': '0/24 pass', 'distractor_only': '0/24 pass', 'evidence_candidate_mismatch': '0/24 pass', 'randomized_candidate_order_with_label_remap': '24/24 pass', 'randomized_evidence_block_order': '24/24 pass', 'randomized_labels': '0/24 pass', 'schema_template_only': '0/24 pass'}

- Controls audit JSONL: `results\stage8_stage8a_controls_audit.jsonl`
- Shortcut audit JSON: `results\stage8_shortcut_audit.json`
- Capacity curves CSV: `results\stage8_capacity_curves.csv`
- Compute audit JSON: `results\stage8_compute_audit.json`

Failed baselines, zero-capacity baselines, and failed controls are retained in the JSON artifacts.
