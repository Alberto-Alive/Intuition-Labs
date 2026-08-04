from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from src.experiments.run_plan_arch1_3_transition_token_repair import (
    MinimalTransitionCrossAttention,
    _heuristic_sanity_row,
    _level_has_final_state,
    _token_visibility_audit_rows,
    _transition_variant_for_level,
)
from src.experiments.run_plan_arch1_empirical_architecture_discovery import (
    PlanDatasetConfig,
    _labels,
    _variant_plan,
    build_plan_splits,
    collate_plan_batch,
)


def test_plan_arch1_3_visible_final_heuristic_solves_level1() -> None:
    config = PlanDatasetConfig(
        grid_size=8,
        num_candidates=4,
        plan_length=8,
        obstacle_count=0,
        train_examples=4,
        dev_examples=2,
        test_examples=8,
        max_generation_attempts=800,
        shortcut_pool_attempts=500,
        learnability_level=1,
    )
    examples = build_plan_splits(config, seed=3)["test"]
    row = _heuristic_sanity_row(1, "level1", 3, examples, final_state_visible=True)
    assert row["heuristics"]["final_position_equals_goal"]["top1"] == 1.0
    assert row["best_top1"] == 1.0


def test_plan_arch1_3_token_visibility_audit_passes_on_level1() -> None:
    config = PlanDatasetConfig(
        grid_size=8,
        num_candidates=4,
        plan_length=8,
        obstacle_count=0,
        train_examples=4,
        dev_examples=2,
        test_examples=4,
        max_generation_attempts=800,
        shortcut_pool_attempts=500,
        learnability_level=1,
    )
    examples = build_plan_splits(config, seed=4)["test"][:2]
    variants = {variant.name: variant for variant in _variant_plan(None)}
    variant = _transition_variant_for_level(variants["transition_tuple_verifier"], 1)
    rows = _token_visibility_audit_rows(1, "level1", 4, examples, variant, logits=None)
    assert rows
    assert all(row["assertions"]["nonzero_transition_tokens"] for row in rows)
    assert all(row["assertions"]["transition_tokens_differ_across_candidates"] for row in rows)
    assert all(row["assertions"]["candidate_order_remap_changes_labels_correctly"] for row in rows)
    assert all(row["assertions"]["role_order_remap_permuted_consistently"] for row in rows)


def test_plan_arch1_3_minimal_transition_model_forward_shape() -> None:
    config = PlanDatasetConfig(
        grid_size=8,
        num_candidates=4,
        plan_length=8,
        obstacle_count=0,
        train_examples=4,
        dev_examples=2,
        test_examples=2,
        max_generation_attempts=800,
        shortcut_pool_attempts=500,
        learnability_level=1,
    )
    examples = build_plan_splits(config, seed=5)["train"][:2]
    variants = {variant.name: variant for variant in _variant_plan(None)}
    variant = _transition_variant_for_level(variants["transition_tuple_verifier"], 1)
    assert _level_has_final_state(1)
    batch = collate_plan_batch(examples, variant, "cpu")
    model = MinimalTransitionCrossAttention(model_dim=32, num_heads=2, ff_dim=64)
    out = model(batch)
    assert tuple(out["logits"].shape) == (2, 4)
    assert torch.isfinite(out["logits"]).all()
    assert np.array_equal(_labels(examples), batch["labels"].cpu().numpy())
