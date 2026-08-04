from __future__ import annotations

import numpy as np

from src.experiments.run_plan_arch1_5_learned_constraint_shortcut_repair import (
    ConstraintDatasetConfig,
    _action_counts,
    _counterfactual_candidate_audit,
    _labels,
    _shortcut_decomposition_row,
    build_repaired_constraint_splits,
)


def test_repaired_examples_are_counterfactual_and_balanced() -> None:
    splits = build_repaired_constraint_splits(
        ConstraintDatasetConfig(train_examples=24, dev_examples=8, test_examples=24, num_candidates=8, plan_length=48),
        seed=0,
    )
    examples = [example for rows in splits.values() for example in rows]
    assert {example.family for example in examples} == {"color_zone", "ordered_subgoal", "key_door"}
    for example in examples:
        assert len(example.candidates) == 8
        assert sum(bool(value) for value in example.satisfies_constraint) == 1
        assert example.satisfies_constraint[example.label] is True
        assert example.metadata["paired_counterfactual"] is True
        assert example.metadata["all_candidates_same_length"] is True
        assert example.metadata["all_candidates_same_endpoint"] is True
        assert example.metadata["all_candidates_reach_goal"] is True
        assert len({tuple(_action_counts(candidate)) for candidate in example.candidates}) == 1


def test_candidate_slots_are_balanced_on_repaired_test_split() -> None:
    splits = build_repaired_constraint_splits(
        ConstraintDatasetConfig(train_examples=24, dev_examples=8, test_examples=64, num_candidates=8, plan_length=48),
        seed=1,
    )
    labels = _labels(splits["test"])
    counts = np.bincount(labels, minlength=8)
    assert counts.min() == counts.max()


def test_counterfactual_audit_passes_core_balance_checks() -> None:
    splits = build_repaired_constraint_splits(
        ConstraintDatasetConfig(train_examples=24, dev_examples=8, test_examples=24, num_candidates=8, plan_length=48),
        seed=2,
    )
    audit = _counterfactual_candidate_audit(splits, seed=2)
    assert audit["one_valid_candidate"] == 1.0
    assert audit["all_candidates_same_endpoint"] == 1.0
    assert audit["all_candidates_reach_goal"] == 1.0
    assert audit["all_candidates_same_length"] == 1.0
    assert audit["action_unigram_balanced"] == 1.0
    assert audit["special_cell_count_balanced"] == 1.0


def test_two_key_shortcut_decomposition_is_clean_on_small_split() -> None:
    splits = build_repaired_constraint_splits(
        ConstraintDatasetConfig(train_examples=48, dev_examples=24, test_examples=48, num_candidates=8, plan_length=48),
        seed=3,
    )
    row = _shortcut_decomposition_row(splits, seed=3)
    chance = row["chance"]
    baselines = row["baselines"]
    assert baselines["rollout_only"] <= chance + 0.15
    assert baselines["constraint_examples_only"] <= chance + 0.15
    assert baselines["rollout_plus_constraint_oracle"] == 1.0
