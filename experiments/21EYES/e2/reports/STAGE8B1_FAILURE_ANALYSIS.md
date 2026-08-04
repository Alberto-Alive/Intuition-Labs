# Stage 8B.1 Failure Analysis

## Executive Summary

- Decision: STAGE8B1_VIEW_COLLAPSE_BOTTLENECK
- Recommendation: Add stronger diversity, load balancing, dropout, or architectural separation.
- NO_STAGE8C_RUN: true
- NO_10X_CLAIM: true

## Best Latent Result

- Best latent accuracy by N: {'16': 0.21875, '32': 0.19791666666666666, '64': 0.19791666666666666, '8': 0.17708333333333334, '128': 0.14930555555555555}
- Capacity ratio: {'best_config': 'latent_r4_a2_candidate_task_mixture', 'capacity': 0, 'ratio': 0.0}

## Best Monolithic Baseline Result

- Best monolithic accuracy by N: {'1024': 0.09722222222222222, '128': 0.11458333333333334, '16': 0.1701388888888889, '256': 0.12152777777777778, '32': 0.13194444444444445, '512': 0.15625, '64': 0.1423611111111111, '8': 0.14583333333333334}
- Monolithic analysis: {'capacity': 0, 'accuracy_by_n': {'1024': 0.09722222222222222, '128': 0.11458333333333334, '16': 0.1701388888888889, '256': 0.12152777777777778, '32': 0.13194444444444445, '512': 0.15625, '64': 0.1423611111111111, '8': 0.14583333333333334}, 'failure_point': 'N=8; no monolithic baseline reached accuracy >= 0.85', 'parameter_count': 24640, 'compute': {'attention_ops': 9764601856.0, 'encoder_passes': 1, 'estimated_forward_compute': 9865789440.0, 'latent_views': 1, 'model_kind': 'monolithic_transformer', 'parameter_count_estimate': 24640, 'projection_ops': 101187584.0}, 'implementation_note': 'The current harness labels this as monolithic_transformer, but the implemented selector uses hashed full-evidence featurization with hidden compression rather than a token-level transformer.', 'more_effective_compute_than_latent': 'often yes by estimated forward compute at large N'}

## Capacity Ratio Achieved

{'best_config': 'latent_r4_a2_candidate_task_mixture', 'capacity': 0, 'ratio': 0.0}

## Where Latent Variants Helped

- Delta by N: {'8': 0.03125, '16': 0.048611111111111105, '32': 0.06597222222222221, '64': 0.05555555555555555, '128': 0.03472222222222221, '256': -0.12152777777777778, '512': -0.15625, '1024': -0.09722222222222222}

## Where Latent Variants Failed

- All searched latent variants had effective capacity C=0 under the Stage 8 capacity definition.
- Accuracies mostly remained near chance, so degradation controls often could not pass meaningfully.

## Control Failures

{'avenue_permutation': {'pass_count': 85, 'total': 85, 'mean_accuracy': 0.1256127450980392, 'mean_degradation': -0.0009803921568627447}, 'candidate_only': {'pass_count': 0, 'total': 85, 'mean_accuracy': 0.12181372549019608, 'mean_degradation': 0.0028186274509803926}, 'cross_task_evidence_shuffle': {'pass_count': 0, 'total': 285, 'mean_accuracy': 0.13095760233918127, 'mean_degradation': -0.0014985380116959066}, 'cross_task_query_shuffle': {'pass_count': 0, 'total': 85, 'mean_accuracy': 0.11801470588235294, 'mean_degradation': 0.006617647058823531}, 'distractor_only': {'pass_count': 0, 'total': 85, 'mean_accuracy': 0.12426470588235294, 'mean_degradation': 0.00036764705882352957}, 'evidence_candidate_mismatch': {'pass_count': 2, 'total': 285, 'mean_accuracy': 0.13450292397660818, 'mean_degradation': -0.005043859649122807}, 'evidence_only': {'pass_count': 1, 'total': 85, 'mean_accuracy': 0.11605392156862746, 'mean_degradation': 0.008578431372549019}, 'hidden_state_shuffle': {'pass_count': 0, 'total': 85, 'mean_accuracy': 0.12450980392156863, 'mean_degradation': 0.00012254901960784465}, 'query_only': {'pass_count': 0, 'total': 85, 'mean_accuracy': 0.11666666666666667, 'mean_degradation': 0.007965686274509803}, 'randomized_candidate_order_with_label_remap': {'pass_count': 279, 'total': 285, 'mean_accuracy': 0.12909356725146198, 'mean_degradation': 0.00036549707602339174}, 'randomized_evidence_block_order': {'pass_count': 277, 'total': 285, 'mean_accuracy': 0.1295687134502924, 'mean_degradation': -0.00010964912280701711}, 'randomized_labels': {'pass_count': 4, 'total': 285, 'mean_accuracy': 0.11663011695906432, 'mean_degradation': 0.012828947368421053}, 'role_permutation': {'pass_count': 82, 'total': 85, 'mean_accuracy': 0.12512254901960784, 'mean_degradation': -0.0004901960784313723}, 'schema_template_only': {'pass_count': 0, 'total': 85, 'mean_accuracy': 0.12622549019607843, 'mean_degradation': -0.00159313725490196}}

## Shortcut Audit

{'passes': True, 'shortcut_failures': [], 'candidate_index_predictiveness_max': 0.20833333333333334, 'block_position_predictiveness': {'first_relevant_position_mean_min': 0.4800139033715676, 'first_relevant_position_mean_max': 0.5340547515359463}, 'entity_namespace_predictiveness_max': 0.20833333333333334, 'template_id_predictiveness_max': 0.4375, 'evidence_length_predictiveness_max_spread': 3266.397727272728, 'lexical_artifact_predictiveness': 'no explicit lexical artifact channel detected by stored audits', 'candidate_text_artifact_examples': [], 'task_family_artifact_predictiveness_max': 0.4375}

## View Collapse Audit

{'rows': [{'config_id': '21c8eecac53bed19', 'architecture_name': 'latent_r4_a8_301861', 'roles': 4, 'avenues': 8, 'role_usage_entropy_mean': 0.6947953911497237, 'avenue_usage_entropy_mean': 0.8386970537154976, 'routing_entropy_mean': 2.3208382800221443, 'average_pairwise_hidden_state_similarity': 0.86211804146578, 'attention_concentration_over_views': 0.49881106737789305, 'likely_view_collapse': True}, {'config_id': '0cb82e32eb9c2186', 'architecture_name': 'latent_r16_a8_536110', 'roles': 16, 'avenues': 8, 'role_usage_entropy_mean': 1.4375340990832952, 'avenue_usage_entropy_mean': 1.2379283473347324, 'routing_entropy_mean': 4.850504320114851, 'average_pairwise_hidden_state_similarity': 0.08721685280827333, 'attention_concentration_over_views': 0.48151917103593656, 'likely_view_collapse': False}, {'config_id': 'a3bc1478f61a75a2', 'architecture_name': 'latent_single_role_ablation', 'roles': 1, 'avenues': 8, 'role_usage_entropy_mean': 0.0, 'avenue_usage_entropy_mean': 0.7045207218107051, 'routing_entropy_mean': 2.0505656702443957, 'average_pairwise_hidden_state_similarity': 0.577894234098494, 'attention_concentration_over_views': 1.0, 'likely_view_collapse': False}, {'config_id': 'f1f155da91666af3', 'architecture_name': 'latent_r8_a8_197133', 'roles': 8, 'avenues': 8, 'role_usage_entropy_mean': 0.818852827796165, 'avenue_usage_entropy_mean': 0.8198211384683536, 'routing_entropy_mean': 4.13004861921072, 'average_pairwise_hidden_state_similarity': 0.23564079468471594, 'attention_concentration_over_views': 0.6062150287068562, 'likely_view_collapse': False}, {'config_id': '58596cf7d3a0653a', 'architecture_name': 'latent_r2_a1_978147', 'roles': 2, 'avenues': 1, 'role_usage_entropy_mean': 0.36257644005631373, 'avenue_usage_entropy_mean': 0.0, 'routing_entropy_mean': 0.4838776921387762, 'average_pairwise_hidden_state_similarity': 0.9641642663627863, 'attention_concentration_over_views': 0.47691276798758164, 'likely_view_collapse': True}, {'config_id': '5f2766f2586e6f5b', 'architecture_name': 'latent_r4_a2_857859', 'roles': 4, 'avenues': 2, 'role_usage_entropy_mean': 0.6852010548113148, 'avenue_usage_entropy_mean': 0.11830214917083828, 'routing_entropy_mean': 1.9839966159313918, 'average_pairwise_hidden_state_similarity': 0.577894234098494, 'attention_concentration_over_views': 0.5057319181059147, 'likely_view_collapse': False}, {'config_id': '4dea81436aaff38c', 'architecture_name': 'latent_r2_a1_069577', 'roles': 2, 'avenues': 1, 'role_usage_entropy_mean': 0.2466484343156881, 'avenue_usage_entropy_mean': 0.0, 'routing_entropy_mean': 0.6811461336910725, 'average_pairwise_hidden_state_similarity': 0.9792981930077076, 'attention_concentration_over_views': 0.6441615269697296, 'likely_view_collapse': True}, {'config_id': '6f022397bce7cf54', 'architecture_name': 'latent_r4_a1_784310', 'roles': 4, 'avenues': 1, 'role_usage_entropy_mean': 0.17534898576838737, 'avenue_usage_entropy_mean': 0.0, 'routing_entropy_mean': 0.6792300203815103, 'average_pairwise_hidden_state_similarity': 0.9606513228888313, 'attention_concentration_over_views': 0.8735124439035191, 'likely_view_collapse': True}, {'config_id': '4389d52efa488343', 'architecture_name': 'latent_r2_a4_579363', 'roles': 2, 'avenues': 4, 'role_usage_entropy_mean': 0.03476519961368528, 'avenue_usage_entropy_mean': 0.4168550881645567, 'routing_entropy_mean': 1.3606195915490389, 'average_pairwise_hidden_state_similarity': 0.8807554460862386, 'attention_concentration_over_views': 0.9498444189218214, 'likely_view_collapse': True}, {'config_id': '65676bae1f959e9e', 'architecture_name': 'latent_r4_a1_658127', 'roles': 4, 'avenues': 1, 'role_usage_entropy_mean': 0.42256481280220476, 'avenue_usage_entropy_mean': 0.0, 'routing_entropy_mean': 1.3557478237897158, 'average_pairwise_hidden_state_similarity': 0.9606513228888313, 'attention_concentration_over_views': 0.695183920058043, 'likely_view_collapse': True}], 'summary': {'num_configs': 10, 'collapse_count': 6, 'mean_pairwise_similarity': 0.7086284708390153}}

## Routing Bottleneck Analysis

{'best_accuracy': 0.17708333333333334, 'mean_accuracy': 0.1388888888888889, 'best_train_accuracy_sample': 0.8020833333333334, 'rows': 6, 'best_by_n': {'8': 0.17708333333333334, '16': 0.16666666666666666, '64': 0.13541666666666666}}

## Chunking Bottleneck Analysis

{'best_accuracy': 0.19791666666666666, 'mean_accuracy': 0.15277777777777776, 'best_train_accuracy_sample': 0.84375, 'rows': 6, 'best_by_n': {'8': 0.19791666666666666, '16': 0.15625, '64': 0.16666666666666666}}

## Compute Accounting

{'latent_parameter_count_mean': 23557.508196721312, 'latent_estimated_compute_mean': 325023.47540983604, 'latent_estimated_compute_max': 1187840.0, 'wall_clock_time_mean': 61.98573924918071, 'training_loss_delta_mean': -0.4258200522279873, 'underfit_or_overfit': 'underfit likely: dev accuracy remains near chance despite decreasing training losses; train accuracy is measured in targeted reruns.', 'raw_compute_audit_rows': 61}

## Coordinator Analysis

{'avenue_first_role_aggregation': {'mean_max_accuracy': 0.1527777777777778, 'count': 4}, 'gated_latent_view_selection': {'mean_max_accuracy': 0.16145833333333334, 'count': 8}, 'hierarchical_coordinator': {'mean_max_accuracy': 0.16111111111111112, 'count': 5}, 'mixture_of_views': {'mean_max_accuracy': 0.15729166666666666, 'count': 10}, 'multi_hop_cross_attention': {'mean_max_accuracy': 0.1484375, 'count': 8}, 'recurrent_refinement_4': {'mean_max_accuracy': 0.1527777777777778, 'count': 3}, 'recurrent_refinement_8': {'mean_max_accuracy': 0.19270833333333331, 'count': 2}, 'role_first_avenue_aggregation': {'mean_max_accuracy': 0.15625, 'count': 6}, 'single_cross_attention_layer': {'mean_max_accuracy': 0.1579861111111111, 'count': 6}, 'top_k_latent_view_selection': {'mean_max_accuracy': 0.15123456790123457, 'count': 9}}

## Recommended Next Action

Add stronger diversity, load balancing, dropout, or architectural separation.

## Artifacts

- Failure analysis JSON: `results\stage8b1_failure_analysis.json`
- Diagnostic reruns JSONL: `results\stage8b1_diagnostic_reruns.jsonl`
- View collapse audit: `results\stage8b1_view_collapse_audit.json`
- Shortcut deep audit: `results\stage8b1_shortcut_deep_audit.json`
- Oracle routing: `results\stage8b1_oracle_routing_results.json`
- Oracle chunk: `results\stage8b1_oracle_chunk_results.json`
- Compute match: `results\stage8b1_compute_match_results.json`
