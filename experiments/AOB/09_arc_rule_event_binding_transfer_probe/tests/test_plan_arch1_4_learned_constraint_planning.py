from __future__ import annotations

import numpy as np

from src.experiments.run_plan_arch1_4_learned_constraint_planning import (
    ConstraintDatasetConfig,
    _action_counts,
    _final_position_predictions,
    _labels,
    build_constraint_splits,
)


def test_constraint_examples_have_one_valid_candidate_and_balanced_endpoints() -> None:
    splits = build_constraint_splits(
        ConstraintDatasetConfig(train_examples=9, dev_examples=3, test_examples=9, num_candidates=8, plan_length=32),
        seed=0,
    )
    examples = splits["test"]
    assert {example.family for example in examples} == {"color_zone", "ordered_subgoal", "key_door"}
    for example in examples:
        assert len(example.candidates) == 8
        assert sum(bool(value) for value in example.satisfies_constraint) == 1
        assert example.satisfies_constraint[example.label] is True
        assert example.metadata["all_candidates_same_length"] is True
        assert example.metadata["all_candidates_same_endpoint"] is True
        assert example.metadata["num_candidates_reaching_goal"] == 8


def test_micro_candidate_counts_keep_one_valid_candidate() -> None:
    for num_candidates in (2, 4):
        splits = build_constraint_splits(
            ConstraintDatasetConfig(
                train_examples=12,
                dev_examples=4,
                test_examples=4,
                num_candidates=num_candidates,
                plan_length=32,
            ),
            seed=1000 + num_candidates,
        )
        for examples in splits.values():
            for example in examples:
                assert len(example.candidates) == num_candidates
                assert sum(bool(value) for value in example.satisfies_constraint) == 1
                assert example.satisfies_constraint[example.label] is True


def test_endpoint_heuristic_is_near_chance_on_constraint_tasks() -> None:
    splits = build_constraint_splits(
        ConstraintDatasetConfig(train_examples=16, dev_examples=8, test_examples=16, num_candidates=8, plan_length=32),
        seed=1,
    )
    examples = splits["test"]
    endpoint_acc = float(np.mean(_final_position_predictions(examples) == _labels(examples)))
    assert endpoint_acc <= 0.25


def test_action_unigram_counts_are_balanced_within_examples() -> None:
    splits = build_constraint_splits(
        ConstraintDatasetConfig(train_examples=6, dev_examples=3, test_examples=6, num_candidates=8, plan_length=32),
        seed=2,
    )
    for example in splits["test"]:
        counts = {tuple(_action_counts(candidate)) for candidate in example.candidates}
        assert len(counts) == 1
