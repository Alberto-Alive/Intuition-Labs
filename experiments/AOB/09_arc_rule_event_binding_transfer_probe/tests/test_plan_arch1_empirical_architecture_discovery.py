from __future__ import annotations

import numpy as np

from src.experiments.run_plan_arch1_empirical_architecture_discovery import (
    PlanDatasetConfig,
    apply_plan_control,
    build_plan_splits,
    run_plan_arch1_search,
)


def test_plan_gridworld_candidate_sets_have_one_success_and_hidden_metadata() -> None:
    splits = build_plan_splits(
        PlanDatasetConfig(train_examples=4, dev_examples=2, test_examples=2, num_candidates=4, plan_length=10, obstacle_count=4),
        seed=0,
    )
    examples = [example for rows in splits.values() for example in rows]
    assert examples
    for example in examples:
        assert len(example.candidates) == 4
        assert sum(bool(value) for value in example.goal_reached) == 1
        assert example.goal_reached[example.label] is True
        assert example.metadata["model_visible_candidate_sources"] is False
        assert example.metadata["model_visible_gold_label"] is False
        assert example.metadata["model_visible_simulator_success_flag"] is False


def test_plan_candidate_order_shuffle_remaps_label() -> None:
    splits = build_plan_splits(
        PlanDatasetConfig(train_examples=4, dev_examples=2, test_examples=2, num_candidates=4, plan_length=10, obstacle_count=4),
        seed=1,
    )
    example = splits["train"][0]
    shuffled = apply_plan_control([example], "candidate_order_shuffle_with_gold_remap", seed=2)[0]
    assert shuffled.goal_reached[shuffled.label] is True
    assert len(shuffled.candidates) == len(example.candidates)
    assert sorted(map(tuple, shuffled.candidates)) == sorted(map(tuple, example.candidates))


def test_plan_arch1_search_smoke_blocks_final_validation() -> None:
    config = {
        "device": "cpu",
        "cheap_seeds": [0],
        "max_variants": 1,
        "run_overfit_gates": False,
        "run_medium": False,
        "dataset": {
            "grid_size": 7,
            "num_candidates": 4,
            "plan_length": 10,
            "obstacle_count": 4,
            "train_examples": 8,
            "dev_examples": 4,
            "test_examples": 4,
        },
        "cheap_dataset": {
            "train_examples": 8,
            "dev_examples": 4,
            "test_examples": 4,
        },
        "training": {
            "epochs": 1,
            "batch_size": 4,
            "lr": 0.001,
            "patience": 1,
        },
        "feature_training": {
            "epochs": 1,
            "batch_size": 4,
            "lr": 0.003,
            "patience": 1,
            "hidden_dim": 16,
        },
    }
    result = run_plan_arch1_search(config, max_variants=1)
    assert result["metadata"]["final_validation_launched"] is False
    assert result["summary"]["final_validation_launched"] is False
    assert result["rows"]
    assert result["controls"]
    row = next(row for row in result["rows"] if row["status"] == "completed")
    assert row["frozen_audit"]["same_architecture_comparator"] is True
    assert row["frozen_audit"]["frozen_shared_model_zero_delta"] is True
    assert np.isfinite(row["trainable"]["top1"])
