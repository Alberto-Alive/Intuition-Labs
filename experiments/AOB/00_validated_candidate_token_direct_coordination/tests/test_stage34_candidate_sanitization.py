import json
from pathlib import Path

from src.datasets.multiview_code_patch_selection import (
    ORACLE_ONLY_METADATA_KEYS,
    SANITIZED_CANDIDATE_REPRESENTATION,
    SANITIZED_DATASET_SOURCE,
    MultiViewCodePatchDatasetConfig,
    assert_no_oracle_only_keys_in_model_record,
    build_multiview_code_patch_splits,
    example_oracle_metadata,
    model_record_from_example,
)
from src.experiments.run_stage34_candidate_sanitization_diagnostics import run_diagnostics


def _sanitized_config(n_train: int = 64, n_dev: int = 32, n_test: int = 64) -> MultiViewCodePatchDatasetConfig:
    return MultiViewCodePatchDatasetConfig(
        n_train=n_train,
        n_dev=n_dev,
        n_test=n_test,
        num_candidates=8,
        n_views=4,
        max_files=12,
        snippet_radius=1,
        dataset_source=SANITIZED_DATASET_SOURCE,
        generator_version="real_import_restore_candidate_sanitized_v3_unit",
        candidate_representation=SANITIZED_CANDIDATE_REPRESENTATION,
    )


def test_sanitized_model_record_excludes_oracle_only_fields_and_exact_values() -> None:
    splits = build_multiview_code_patch_splits(_sanitized_config(n_train=8, n_dev=4, n_test=4), seed=42, repo_root=Path("."))
    example = splits["train"][0]

    assert example.metadata["dataset_source"] == SANITIZED_DATASET_SOURCE
    assert example.oracle_metadata is not None
    assert set(example.metadata).isdisjoint(ORACLE_ONLY_METADATA_KEYS)

    model_record = model_record_from_example(example)
    assert_no_oracle_only_keys_in_model_record(model_record)
    serialized = json.dumps(model_record, sort_keys=True).lower()
    oracle = example_oracle_metadata(example)

    for values in oracle["candidate_patch_values"]:
        symbol, module, _style, target = [str(value).lower() for value in values[:4]]
        assert symbol not in serialized
        assert module not in serialized
        assert target not in serialized
    assert str(oracle["gold_import_statement"]).lower() not in serialized

    for candidate in example.candidates:
        text = candidate.text.lower()
        assert "provider_redacted" in text
        assert "name_redacted" in text
        assert candidate.attributes == ("model_safe_opaque",) * 4


def test_stage34_candidate_only_diagnostics_collapse_while_oracle_stays_high() -> None:
    config = {
        "stage": {
            "name": "stage34_unit",
            "n_train": 64,
            "n_dev": 32,
            "n_test": 64,
            "seeds": [0],
            "epochs": 1,
            "patience": 1,
            "hidden_dim": 16,
            "tiny_layers": 1,
            "tiny_ff_dim": 32,
            "batch_size": 16,
            "lr": 0.002,
            "gradient_accumulation_steps": 1,
            "mixed_precision": "none",
        },
        "dataset_config": {
            "source_roots": ["src", "tests"],
            "max_files": 12,
            "snippet_radius": 1,
            "dataset_source": SANITIZED_DATASET_SOURCE,
            "generator_version": "real_import_restore_candidate_sanitized_v3_unit",
            "candidate_representation": SANITIZED_CANDIDATE_REPRESENTATION,
        },
        "dataset_diagnostics": {
            "baseline_epochs": 1,
            "baseline_patience": 1,
            "baseline_feature_dim": 64,
            "single_view_epochs": 1,
            "single_view_patience": 1,
            "single_view_feature_dim": 64,
        },
    }
    result = run_diagnostics(config)
    summary = result["summary"]

    assert summary["mean_candidate_only_accuracy"] <= 0.18
    assert summary["mean_candidate_metadata_only_accuracy"] <= 0.18
    assert summary["mean_role_pair_only_accuracy"] <= 0.18
    assert summary["mean_view_masked_candidates_visible_accuracy"] <= 0.18
    assert summary["max_single_view_accuracy"] <= 0.18
    assert summary["max_pairwise_structured_oracle_accuracy"] >= 0.70
    assert summary["mean_all_role_structured_oracle_accuracy"] >= 0.90
    assert summary["leakage_passes"] is True
