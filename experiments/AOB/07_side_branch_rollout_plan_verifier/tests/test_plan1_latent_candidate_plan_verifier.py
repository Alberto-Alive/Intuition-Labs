from __future__ import annotations

import pytest

from src.datasets.plan_gridworld import (
    PlanGridworldDatasetConfig,
    apply_plan_control,
    build_plan_gridworld_splits,
    leakage_audit_rows,
)


def test_plan1_gridworld_generator_has_single_balanced_hidden_gold() -> None:
    config = PlanGridworldDatasetConfig(
        num_candidates=4,
        train_examples=6,
        dev_examples=3,
        test_examples=3,
        grid_size=6,
        obstacle_density=0.12,
        plan_horizon=12,
        min_goal_distance=3,
        level=2,
    )
    splits = build_plan_gridworld_splits(config, seed=123)
    examples = [example for rows in splits.values() for example in rows]
    assert examples
    assert all(row["pass"] for row in leakage_audit_rows(examples))
    for example in examples:
        assert all(tag == "candidate_plan" for tag in example.public_source_tags)
        assert len({len(candidate.actions) for candidate in example.candidates}) == 1
        scores = [
            (1, -candidate.first_goal_step) if candidate.valid else (0, -10_000 + int(candidate.near_miss) - int(candidate.collision))
            for candidate in example.candidates
        ]
        assert scores.count(max(scores)) == 1
        assert scores.index(max(scores)) == example.label


def test_plan1_candidate_order_control_remaps_gold() -> None:
    config = PlanGridworldDatasetConfig(num_candidates=4, train_examples=2, dev_examples=0, test_examples=0, plan_horizon=12, min_goal_distance=3)
    example = build_plan_gridworld_splits(config, seed=7)["train"][0]
    shuffled = apply_plan_control([example], "candidate_order_shuffle_with_gold_remap", seed=99)[0]
    assert shuffled.candidates[shuffled.label].actions == example.candidates[example.label].actions
    assert sorted(candidate.actions for candidate in shuffled.candidates) == sorted(candidate.actions for candidate in example.candidates)


def test_plan1_frozen_shared_audit_on_tiny_fit() -> None:
    pytest.importorskip("torch")
    from src.experiments.run_stage_plan1_latent_candidate_plan_verifier import (
        PlanModelConfig,
        PlanTrainingConfig,
        fit_plan_verifier,
    )

    config = PlanGridworldDatasetConfig(
        num_candidates=2,
        train_examples=4,
        dev_examples=4,
        test_examples=0,
        grid_size=5,
        obstacle_density=0.08,
        plan_horizon=10,
        min_goal_distance=2,
        level=1,
    )
    splits = build_plan_gridworld_splits(config, seed=5)
    model_config = PlanModelConfig(
        view_names=("state_view", "goal_view", "obstacle_constraint_view", "candidate_action_view", "rollout_view"),
        model_dim=24,
        num_heads=4,
        ff_dim=48,
        max_view_tokens=24,
        max_candidate_tokens=12,
    )
    training = PlanTrainingConfig(epochs=1, batch_size=2, lr=0.001, patience=1)
    trainable = fit_plan_verifier(
        splits["train"],
        splits["dev"],
        model_config,
        training,
        seed=11,
        device="cpu",
        trainable_shared=True,
        method="test_trainable",
    )
    frozen = fit_plan_verifier(
        splits["train"],
        splits["dev"],
        model_config,
        training,
        seed=11,
        device="cpu",
        trainable_shared=False,
        method="test_frozen",
    )
    assert trainable.audit["trainable_shared_model_changed"]
    assert frozen.audit["frozen_shared_model_zero_grad"]
    assert frozen.audit["frozen_shared_model_zero_delta"]
    assert frozen.audit["same_architecture_comparator"]
