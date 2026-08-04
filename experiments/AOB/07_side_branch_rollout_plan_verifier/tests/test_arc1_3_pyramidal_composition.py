from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.datasets.arc_agi2_verification import ArcVerificationDatasetConfig, build_arc_verification_splits
from src.experiments.run_stage_arc1_3_pyramidal_composition import (
    ArcPyramidalRuleCloneVerifier,
    PyramidalArcModelConfig,
    fit_pyramid_verifier,
    predict_pyramid_logits,
)
from src.experiments.run_stage_arc1_latent_rule_clones import ArcTrainingConfig, collate_arc_batch


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
    for index, color in enumerate([1, 2, 3, 4, 5, 6, 7, 8]):
        _write_task(train, f"task_{index}", color)
    return root


def _splits(tmp_path: Path):
    root = _dataset_root(tmp_path)
    return build_arc_verification_splits(
        ArcVerificationDatasetConfig(dataset_root=str(root), num_candidates=4, train_tasks=4, dev_tasks=2, test_tasks=2),
        seed=0,
    )


def _config(**kwargs) -> PyramidalArcModelConfig:
    values = {
        "model_dim": 16,
        "num_heads": 2,
        "ff_dim": 32,
        "max_rule_tokens": 24,
        "max_candidate_tokens": 24,
        "max_composed_tokens": 12,
        "view_names": ("raw", "diff", "object", "color", "geometry", "symmetry"),
        "representation_mode": "hybrid_all",
        "pyramid_layout": "pairwise",
        "composer_kind": "token",
        "query_mode": "leaf_plus_composed",
    }
    values.update(kwargs)
    return PyramidalArcModelConfig(**values)


def test_pyramidal_forward_variants_and_attention(tmp_path: Path) -> None:
    splits = _splits(tmp_path)
    configs = [
        _config(pyramid_layout="flat", composer_kind="none", query_mode="leaf_only"),
        _config(pyramid_layout="pairwise", composer_kind="pooled", query_mode="composed_only"),
        _config(pyramid_layout="pairwise", composer_kind="token", query_mode="leaf_plus_composed"),
        _config(pyramid_layout="binary_tree", composer_kind="token", query_mode="root_plus_intermediate"),
        _config(pyramid_layout="pairwise", composer_kind="token", query_mode="composed_only", candidate_guided_composition=True),
        _config(pyramid_layout="residual", composer_kind="token", query_mode="all_levels", residual_pyramid=True),
    ]
    for config in configs:
        model = ArcPyramidalRuleCloneVerifier(config)
        batch = collate_arc_batch(splits["train"][:2], config, "cpu")
        batch["active_view_names"] = config.view_names
        out = model(batch, return_attention=True)
        assert out["logits"].shape == (2, 4)
        assert "latent_state_diagnostics" in out
        logits, attention = predict_pyramid_logits(model, splits["dev"], config, batch_size=2, device="cpu", return_attention=True)
        assert logits.shape == (len(splits["dev"]), 4)
        assert isinstance(attention, list)


def test_pyramidal_frozen_comparator_freezes_composers(tmp_path: Path) -> None:
    splits = _splits(tmp_path)
    config = _config(pyramid_layout="pairwise", composer_kind="pooled", query_mode="leaf_plus_composed")
    training = ArcTrainingConfig(epochs=1, batch_size=2, lr=0.001, patience=1)
    result = fit_pyramid_verifier(
        splits["train"],
        splits["dev"],
        config,
        training,
        seed=11,
        device="cpu",
        trainable_shared=False,
        method="frozen_pyramid_test",
    )
    assert result.audit["frozen_shared_model_zero_grad"]
    assert result.audit["frozen_shared_model_zero_delta"]
    assert result.audit["composition_modules_in_frozen_shared_set"]
    assert result.audit["coordinator_parameter_delta"] > 0.0


def test_pyramid_permutation_and_candidate_order_shapes(tmp_path: Path) -> None:
    splits = _splits(tmp_path)
    config = _config(pyramid_layout="binary_tree", composer_kind="token", query_mode="root_plus_intermediate")
    model = ArcPyramidalRuleCloneVerifier(config)
    base = predict_pyramid_logits(model, splits["dev"], config, batch_size=2, device="cpu")
    permuted = predict_pyramid_logits(model, splits["dev"], config, batch_size=2, device="cpu", condition="pyramid_level_permutation", seed=7)
    role = predict_pyramid_logits(model, splits["dev"], config, batch_size=2, device="cpu", condition="physical_role_order_shuffle", seed=8)
    assert base.shape == permuted.shape == role.shape == (len(splits["dev"]), 4)
    assert np.isfinite(base).all()
