from __future__ import annotations

import numpy as np

from src.experiments.run_plan_arch1_4_learned_constraint_planning import _labels, _top1
from src.experiments.run_plan_arch1_5_learned_constraint_shortcut_repair import ConstraintDatasetConfig, build_repaired_constraint_splits
from src.experiments.run_plan_arch1_6_oracle_to_latent_constraint_binding import (
    BLANK_ID,
    _oracle_valid_logits,
    _rule_order_from_examples,
    _violation_type_logits,
    bridge_batch,
    oracle_teacher,
)


def test_oracle_teacher_matches_repaired_dataset_labels() -> None:
    splits = build_repaired_constraint_splits(
        ConstraintDatasetConfig(train_examples=12, dev_examples=4, test_examples=12, num_candidates=8, plan_length=48),
        seed=0,
    )
    for example in splits["test"]:
        teacher_valid = [bool(oracle_teacher(example, idx)["candidate_valid_under_constraint"]) for idx in range(len(example.candidates))]
        assert teacher_valid == list(example.satisfies_constraint)
        assert teacher_valid[example.label] is True
        assert sum(teacher_valid) == 1
        assert _rule_order_from_examples(example) == list(example.metadata["target_order"])


def test_oracle_ladder_diagnostic_inputs_solve_directly() -> None:
    splits = build_repaired_constraint_splits(
        ConstraintDatasetConfig(train_examples=12, dev_examples=4, test_examples=24, num_candidates=8, plan_length=48),
        seed=1,
    )
    labels = _labels(splits["test"])
    assert _top1(_oracle_valid_logits(splits["test"]), labels) == 1.0
    assert _top1(_violation_type_logits(splits["test"]), labels) == 1.0


def test_bridge_batch_can_blank_rule_or_event_without_labels_as_inputs() -> None:
    splits = build_repaired_constraint_splits(
        ConstraintDatasetConfig(train_examples=8, dev_examples=4, test_examples=8, num_candidates=8, plan_length=48),
        seed=2,
    )
    examples = splits["test"]
    normal = bridge_batch(examples, "cpu")
    blank_rule = bridge_batch(examples, "cpu", rule_mode="blank")
    blank_event = bridge_batch(examples, "cpu", event_mode="blank")
    assert normal["rule_order"].shape == (8, 4)
    assert normal["event_order"].shape == (8, 8, 4)
    assert np.all(blank_rule["rule_order"].numpy() == BLANK_ID)
    assert np.all(blank_event["event_order"].numpy() == BLANK_ID)
    assert "labels" in normal
