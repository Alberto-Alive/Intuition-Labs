from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.datasets.arc_agi2_verification import ArcVerificationDatasetConfig, apply_arc_control, build_arc_verification_splits
from src.experiments.run_stage_arc1_latent_rule_clones import (
    ArcModelConfig,
    ArcRuleCloneVerifier,
    collate_arc_batch,
    fit_arc_verifier,
    predict_logits,
    ArcTrainingConfig,
)


def _write_task(path: Path, name: str, color: int) -> None:
    task = {
        "train": [
            {"input": [[0, color], [0, 0]], "output": [[0, color], [color, 0]]},
            {"input": [[color, 0], [0, 0]], "output": [[color, 0], [0, color]]},
        ],
        "test": [
            {"input": [[0, 0], [color, 0]], "output": [[0, color], [color, 0]]},
        ],
    }
    path.joinpath(f"{name}.json").write_text(json.dumps(task), encoding="utf-8")


def _dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "arc"
    train = root / "data" / "training"
    train.mkdir(parents=True)
    for index, color in enumerate([1, 2, 3, 4, 5, 6]):
        _write_task(train, f"task_{index}", color)
    return root


def test_arc_candidate_sets_have_one_gold_and_fixed_size(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path)
    splits = build_arc_verification_splits(
        ArcVerificationDatasetConfig(dataset_root=str(root), num_candidates=8, train_tasks=3, dev_tasks=1, test_tasks=1),
        seed=0,
    )
    examples = [example for rows in splits.values() for example in rows]
    assert examples
    for example in examples:
        gold = np.asarray(example.gold_output)
        assert sum(np.array_equal(np.asarray(candidate), gold) for candidate in example.candidates) == 1
        assert all(np.asarray(candidate).shape == gold.shape for candidate in example.candidates)
        assert example.negative_types[example.label] == "gold"


def test_arc_controls_remap_candidate_order(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path)
    splits = build_arc_verification_splits(
        ArcVerificationDatasetConfig(dataset_root=str(root), num_candidates=8, train_tasks=3, dev_tasks=1, test_tasks=1),
        seed=1,
    )
    example = splits["train"][0]
    shuffled = apply_arc_control([example], "candidate_order_shuffle_with_gold_remap", seed=2)[0]
    assert np.array_equal(np.asarray(shuffled.candidates[shuffled.label]), np.asarray(example.gold_output))
    assert len(shuffled.candidates) == len(example.candidates)


def test_arc_model_forward_and_physical_order_shuffle(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path)
    splits = build_arc_verification_splits(
        ArcVerificationDatasetConfig(dataset_root=str(root), num_candidates=8, train_tasks=3, dev_tasks=1, test_tasks=1),
        seed=2,
    )
    config = ArcModelConfig(model_dim=16, num_heads=2, ff_dim=32, max_rule_tokens=32, max_candidate_tokens=32, view_names=("raw", "diff"))
    model = ArcRuleCloneVerifier(config)
    batch = collate_arc_batch(splits["train"][:2], config, "cpu")
    out = model(batch)
    assert out["logits"].shape == (2, 8)
    logits = predict_logits(model, splits["test"], config, batch_size=2, device="cpu", condition="physical_role_order_shuffle")
    assert logits.shape == (len(splits["test"]), 8)


def test_frozen_arc_shared_model_has_zero_delta(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path)
    splits = build_arc_verification_splits(
        ArcVerificationDatasetConfig(dataset_root=str(root), num_candidates=8, train_tasks=3, dev_tasks=1, test_tasks=1),
        seed=3,
    )
    model_config = ArcModelConfig(model_dim=16, num_heads=2, ff_dim=32, max_rule_tokens=32, max_candidate_tokens=32, view_names=("raw", "diff"))
    training = ArcTrainingConfig(epochs=1, batch_size=2, lr=0.001, patience=1)
    result = fit_arc_verifier(
        splits["train"],
        splits["dev"],
        model_config,
        training,
        seed=10,
        device="cpu",
        trainable_shared=False,
        method="frozen_test",
    )
    assert result.audit["frozen_shared_model_zero_grad"]
    assert result.audit["frozen_shared_model_zero_delta"]
    assert result.audit["coordinator_parameter_delta"] > 0.0

