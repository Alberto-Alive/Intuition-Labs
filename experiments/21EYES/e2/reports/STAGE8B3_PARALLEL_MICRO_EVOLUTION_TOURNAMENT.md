# Stage 8B.3 Parallel Latent Attention Micro-Evolution Tournament

## 1. Executive summary

- Decision: STAGE8B3_TOP3_SELECTED
- Stage 8C was not run.
- No 10x attention-capacity claim is made.
- Top-3 selected: 3
- Recommendation: Proceed to Stage 8B.4 larger-scale validation of the frozen top-3 configs only: stage8b3_f_memory_slot_attention_02, stage8b3_j_explicit_view_objective_assignment_02, stage8b3_j_explicit_view_objective_assignment_03. Do not run Stage 8C yet.

## 2. Why Stage 8B failed

Stage 8B and Stage 8B.1 showed zero latent and monolithic capacity, near-chance latent accuracy, weak degradation under evidence controls, and substantial role/avenue/view collapse. Stage 8B.3 therefore treats the problem as architecture discovery rather than final validation.

## 3. Tournament design

- Round 0 sanity: {'passes': True, 'oracle_evidence_accuracy_N8': 1.0, 'oracle_evidence_threshold': 0.9, 'evidence_feature_accuracy_N8': 1.0, 'evidence_feature_accuracy_by_task_family': {'sparse_relevant_evidence': 1.0, 'conflict_resolution': 1.0, 'compositional_role_evidence': 1.0, 'multi_hop_binding': 1.0, 'needle_binding': 1.0, 'constraint_satisfaction': 1.0}, 'easy_family_learned': True, 'decision_if_failed': 'STAGE8B3_BENCHMARK_OR_REPRESENTATION_BROKEN'}
- Budget: {'round1_variants': 60, 'round2_promoted': 20, 'round3_mutations_per_parent': 3, 'round4_finalists': 8, 'train_examples': 96, 'eval_examples': 96, 'round1_n': (2, 4, 8), 'round2_n': (4, 8, 16), 'round3_n': (8, 16), 'round4_n': (8, 16, 32), 'round1_seeds': (0,), 'round2_seeds': (0, 1, 2), 'round3_seeds': (0, 1, 2), 'round4_seeds': (0, 1, 2, 3, 4), 'workers': 1, 'include_n64': False}
- Same-template dev uses dev namespace with train templates. Held-out-template dev uses dev namespace with dev templates. Final templates are not used.

## 4. Architecture families tested

- A. Candidate-to-View Routing: 6 initial variants
- B. Support vs Contradiction Attention: 6 initial variants
- C. Hierarchical Coarse-to-Fine Attention: 6 initial variants
- D. Evidence-Location Supervision: 6 initial variants
- E. Anti-Collapse View Specialization: 6 initial variants
- F. Memory Slot Attention: 6 initial variants
- G. Recurrent Latent Search: 6 initial variants
- H. Cross-Candidate Comparison: 6 initial variants
- I. Retrieval-Then-Latent-Reasoning Hybrid: 6 initial variants
- J. Explicit View Objective Assignment: 6 initial variants

## 5. Round 1 killed variants

- Evaluated: 60
- Killed: 0
- Survivors: 60

## 6. Round 2 promoted variants

- Evaluated: 20
- Survivors: 20

## 7. Round 3 mutations and improvements

- Parents: 20
- Mutations: 60
- Kept mutations: 0

## 8. Round 4 finalists

- Evaluated finalists: 8
- stage8b3_g_recurrent_latent_search_05: score=0.9566, C=32, heldout=1.0
- stage8b3_f_memory_slot_attention_03: score=0.9564, C=32, heldout=1.0
- stage8b3_j_explicit_view_objective_assignment_04: score=0.9564, C=32, heldout=1.0
- stage8b3_g_recurrent_latent_search_01: score=0.9563, C=32, heldout=1.0
- stage8b3_j_explicit_view_objective_assignment_03: score=0.9563, C=32, heldout=1.0
- stage8b3_f_memory_slot_attention_02: score=0.9560, C=32, heldout=1.0
- stage8b3_g_recurrent_latent_search_02: score=0.9560, C=32, heldout=1.0
- stage8b3_j_explicit_view_objective_assignment_02: score=0.9560, C=32, heldout=1.0

## 9. Top 3 selected architectures

1. stage8b3_f_memory_slot_attention_02 (Memory Slot Attention), score=0.9560, C=32
2. stage8b3_j_explicit_view_objective_assignment_02 (Explicit View Objective Assignment), score=0.9560, C=32
3. stage8b3_j_explicit_view_objective_assignment_03 (Explicit View Objective Assignment), score=0.9457, C=32

## 10. Evidence-use controls

- stage8b3_f_memory_slot_attention_02: mismatch degradation=0.879, cross-task evidence degradation=0.877, randomized-label accuracy=0.124
- stage8b3_j_explicit_view_objective_assignment_02: mismatch degradation=0.879, cross-task evidence degradation=0.877, randomized-label accuracy=0.124
- stage8b3_j_explicit_view_objective_assignment_03: mismatch degradation=0.879, cross-task evidence degradation=0.877, randomized-label accuracy=0.124

## 11. View diversity / collapse analysis

- stage8b3_f_memory_slot_attention_02: pairwise similarity=0.007, role entropy=0.546, routing entropy=0.331
- stage8b3_j_explicit_view_objective_assignment_02: pairwise similarity=0.007, role entropy=0.546, routing entropy=0.331
- stage8b3_j_explicit_view_objective_assignment_03: pairwise similarity=0.006, role entropy=0.445, routing entropy=0.331

## 12. Compute accounting

- Compute audit: `results\stage8b3_compute_audit.json`
- Capacity curves: `results\stage8b3_capacity_curves.csv`

## 13. Baseline comparison

{
  "n_values": [
    8,
    16,
    32
  ],
  "round": "round4",
  "rows": [
    {
      "accuracy_by_N": {
        "16": 0.09999999999999999,
        "32": 0.10833333333333334,
        "8": 0.13541666666666666
      },
      "architecture_name": "round4_random_candidate",
      "held_out_template_dev_accuracy": 0.11458333333333333,
      "model_kind": "random_candidate",
      "parameter_count": 0
    },
    {
      "accuracy_by_N": {
        "16": 0.10208333333333333,
        "32": 0.12083333333333333,
        "8": 0.12708333333333333
      },
      "architecture_name": "round4_candidate_only",
      "held_out_template_dev_accuracy": 0.11666666666666667,
      "model_kind": "candidate_only",
      "parameter_count": 0
    },
    {
      "accuracy_by_N": {
        "16": 0.12291666666666666,
        "32": 0.11666666666666667,
        "8": 0.15208333333333332
      },
      "architecture_name": "round4_query_only",
      "held_out_template_dev_accuracy": 0.13055555555555556,
      "model_kind": "query_only",
      "parameter_count": 0
    },
    {
      "accuracy_by_N": {
        "16": 0.11875,
        "32": 0.14791666666666667,
        "8": 0.10208333333333333
      },
      "architecture_name": "round4_evidence_only",
      "held_out_template_dev_accuracy": 0.12291666666666666,
      "model_kind": "evidence_only",
      "parameter_count": 0
    },
    {
      "accuracy_by_N": {
        "16": 0.4,
        "32": 0.30625,
        "8": 0.5520833333333334
      },
      "architecture_name": "round4_retrieval_topk",
      "held_out_template_dev_accuracy": 0.41944444444444445,
      "model_kind": "retrieval_topk",
      "parameter_count": 0
    },
    {
      "accuracy_by_N": {
        "16": 0.12708333333333333,
        "32": 0.13125,
        "8": 0.16666666666666666
      },
      "architecture_name": "round4_true_monolithic_transformer",
      "held_out_template_dev_accuracy": 0.14166666666666666,
      "model_kind": "monolithic_transformer",
      "parameter_count": 12320
    },
    {
      "accuracy_by_N": {
        "16": 0.36041666666666666,
        "32": 0.27708333333333335,
        "8": 0.42291666666666666
      },
      "architecture_name": "round4_legacy_hashed_feature_baseline",
      "held_out_template_dev_accuracy": 0.35347222222222224,
      "model_kind": "retrieval_topk",
      "parameter_count": 0
    },
    {
      "accuracy_by_N": {
        "16": 0.36875,
        "32": 0.26666666666666666,
        "8": 0.49374999999999997
      },
      "architecture_name": "round4_raw_latent_comparator",
      "held_out_template_dev_accuracy": 0.3763888888888889,
      "model_kind": "raw_latent_selector",
      "parameter_count": 0
    }
  ],
  "seeds": [
    0,
    1,
    2,
    3,
    4
  ]
}

## 14. Failure modes

- Failed variants remain in the tournament database with failure reasons.
- Variants are disqualified when randomized labels do not collapse, evidence controls fail, candidate/query-only comparators explain performance, trainable does not beat frozen, or view collapse remains severe.

## 15. Recommendation for Stage 8B.4

Proceed to Stage 8B.4 larger-scale validation of the frozen top-3 configs only: stage8b3_f_memory_slot_attention_02, stage8b3_j_explicit_view_objective_assignment_02, stage8b3_j_explicit_view_objective_assignment_03. Do not run Stage 8C yet.

## Artifacts

- database: `results\stage8b3_tournament_database.jsonl`
- round1: `results\stage8b3_round1_screen.json`
- round2: `results\stage8b3_round2_promotions.json`
- round3: `results\stage8b3_round3_mutations.json`
- round4: `results\stage8b3_round4_finalists.json`
- top3_configs: `results\stage8b3_top3_configs.yaml`
- freeze_manifest: `results\stage8b3_top3_freeze_manifest.json`
- controls_audit: `results\stage8b3_controls_audit.jsonl`
- view_diversity_audit: `results\stage8b3_view_diversity_audit.json`
- compute_audit: `results\stage8b3_compute_audit.json`
- capacity_curves: `results\stage8b3_capacity_curves.csv`
- report: `reports\STAGE8B3_PARALLEL_MICRO_EVOLUTION_TOURNAMENT.md`
