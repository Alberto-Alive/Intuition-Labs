from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.datasets.arc_agi2_verification import ArcVerificationDatasetConfig
from src.experiments.run_stage_arc_hybrid1_empirical_search import (
    CandidateGeneratorConfig,
    apply_hybrid_control,
    build_hybrid_candidate_splits,
    run_arc_hybrid1_search,
    _hybrid_leakage_audit_rows,
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
    for index, color in enumerate([1, 2, 3, 4, 5, 6, 7, 8]):
        _write_task(train, f"task_{index}", color)
    return root


def test_hybrid_gold_present_pool_has_external_only_source_metadata(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path)
    splits = build_hybrid_candidate_splits(
        ArcVerificationDatasetConfig(dataset_root=str(root), num_candidates=4, train_tasks=3, dev_tasks=1, test_tasks=1),
        CandidateGeneratorConfig(num_candidates=4, natural_candidate_count=4, mutation_rounds=1),
        seed=0,
        force_gold=True,
        phase="phase1_gold_present",
    )
    examples = [example for rows in splits.values() for example in rows]
    assert examples
    for example in examples:
        gold = np.asarray(example.gold_output)
        assert sum(np.array_equal(np.asarray(candidate), gold) for candidate in example.candidates) == 1
        assert example.metadata["forced_gold_inserted"] is True
        assert example.metadata["model_visible_candidate_sources"] is False
        assert example.metadata["model_visible_generator_rank"] is False
        assert all(tag == "candidate_grid" for tag in example.public_source_tags)
        assert len(example.metadata["candidate_source_families_external"]) == len(example.candidates)
    audit = _hybrid_leakage_audit_rows(examples, expect_exactly_one_gold=True)
    assert all(row["pass"] for row in audit)


def test_hybrid_candidate_order_shuffle_remaps_label_and_metadata(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path)
    splits = build_hybrid_candidate_splits(
        ArcVerificationDatasetConfig(dataset_root=str(root), num_candidates=4, train_tasks=3, dev_tasks=1, test_tasks=1),
        CandidateGeneratorConfig(num_candidates=4, natural_candidate_count=4, mutation_rounds=1),
        seed=1,
        force_gold=True,
        phase="phase1_gold_present",
    )
    example = splits["train"][0]
    shuffled = apply_hybrid_control([example], "candidate_order_shuffle_with_gold_remap", seed=2)[0]
    assert np.array_equal(np.asarray(shuffled.candidates[shuffled.label]), np.asarray(example.gold_output))
    assert shuffled.metadata["offline_candidate_test_correct"][shuffled.label] is True
    assert len(shuffled.metadata["candidate_train_execution_scores"]) == len(shuffled.candidates)


def test_hybrid_search_smoke_does_not_launch_final_validation(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path)
    config = {
        "device": "cpu",
        "seeds": [0],
        "max_variants": 1,
        "run_overfit_gates": False,
        "run_phase2_natural": True,
        "run_phase3_refinement": False,
        "run_simple_stats_mlp_baseline": False,
        "dataset": {
            "dataset_root": str(root),
            "num_candidates": 4,
            "train_tasks": 4,
            "dev_tasks": 2,
            "test_tasks": 2,
            "max_train_pairs": 3,
        },
        "candidate_generator": {
            "num_candidates": 4,
            "natural_candidate_count": 4,
            "mutation_rounds": 1,
            "max_programs": 8,
        },
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "lr": 0.001,
            "patience": 1,
            "gradient_clip_norm": 1.0,
        },
    }
    result = run_arc_hybrid1_search(config)
    assert result["metadata"]["final_validation_launched"] is False
    assert result["summary"]["final_validation_launched"] is False
    assert result["rows"]
    assert result["candidate_generator_audit"]
    assert result["leakage_audit"]
