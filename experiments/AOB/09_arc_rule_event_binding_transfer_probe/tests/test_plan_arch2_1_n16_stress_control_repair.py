from __future__ import annotations

from src.experiments.run_plan_arch1_4_learned_constraint_planning import ConstraintDatasetConfig
from src.experiments.run_plan_arch1_6_oracle_to_latent_constraint_binding import oracle_teacher
from src.experiments.run_plan_arch2_1_n16_stress_control_repair import (
    _final_readiness_decision,
    _n16_balance_audit,
    build_repaired_n16_splits,
)
from src.experiments.run_plan_arch2_medium_rule_event_constraint_binding import _oracle_feature_audit


def test_repaired_n16_has_one_valid_candidate_and_balanced_slots() -> None:
    splits = build_repaired_n16_splits(
        ConstraintDatasetConfig(train_examples=32, dev_examples=16, test_examples=128, num_candidates=16, plan_length=48),
        seed=14,
    )
    labels = []
    target_sources = []
    for example in splits["test"]:
        labels.append(example.label)
        target_sources.append(example.metadata["target_source_candidate_index"])
        assert len(example.candidates) == 16
        assert sum(bool(value) for value in example.satisfies_constraint) == 1
        assert example.satisfies_constraint[example.label] is True
        assert example.metadata["all_candidates_same_endpoint"] is True
        assert example.metadata["all_candidates_reach_goal"] is True
        assert example.metadata["source_candidate_indices"].count(example.metadata["target_source_candidate_index"]) == 1
        assert tuple(oracle_teacher(example, example.label)["actual_order"]) == tuple(example.metadata["target_order"])
    assert sorted(set(labels)) == list(range(16))
    assert sorted(set(target_sources)) == list(range(8))


def test_repaired_n16_balance_audit_and_oracle_feature_audit_pass() -> None:
    splits = build_repaired_n16_splits(
        ConstraintDatasetConfig(train_examples=16, dev_examples=8, test_examples=16, num_candidates=16, plan_length=48),
        seed=10,
    )
    balance = _n16_balance_audit(splits)
    assert balance["test"]["singleton_gold_source_fraction"] == 1.0
    assert balance["test"]["all_candidates_same_endpoint"] == 1.0
    audit = _oracle_feature_audit("N16_repaired", 10, splits)
    assert audit["claimable"] is True
    assert audit["forbidden_inputs_absent"] is True


def test_final_readiness_decision_options() -> None:
    assert _final_readiness_decision(True, True).startswith("B.")
    assert _final_readiness_decision(True, False).startswith("A.")
    assert _final_readiness_decision(False, False).startswith("C.")
