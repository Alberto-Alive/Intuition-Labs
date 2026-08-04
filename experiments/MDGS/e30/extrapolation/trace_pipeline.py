"""Local trace-cache helpers and simple baselines for E30."""

from __future__ import annotations

import json
import random
import warnings
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data.trace_dataset import load_trace_records, trace_inputs_from_record


def set_global_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _feature_arrays(records: Sequence[Dict[str, Any]]):
    summary = np.array(
        [
            [
                float(record["trace_summary"]["mean_entropy"]),
                float(record["trace_summary"]["slope"]),
                float(record["trace_summary"]["std_entropy"]),
                float(record["trace_summary"]["entropy_range"]),
                float(record["max_attention_mass"]),
                float(record["agreement"]),
                float(record["prob_margin"]),
            ]
            for record in records
        ],
        dtype=np.float32,
    )
    full = np.array(
        [
            trace_inputs_from_record(record)
            + [
                float(record["trace_summary"]["mean_entropy"]),
                float(record["trace_summary"]["slope"]),
                float(record["trace_summary"]["std_entropy"]),
                float(record["trace_summary"]["entropy_range"]),
            ]
            for record in records
        ],
        dtype=np.float32,
    )
    targets = {
        "trajectory_shape": np.array([record["trajectory_shape"] for record in records]),
        "attention_pattern": np.array([record["attention_pattern"] for record in records]),
        "confidence": np.array([record["confidence"] for record in records]),
        "outcome": np.array([record["outcome"] for record in records]),
    }
    return summary, full, targets


def evaluate_trace_baselines(train_records: list[Dict[str, Any]], test_records: list[Dict[str, Any]], config=None) -> Dict[str, Dict]:
    if config is None:
        raise ValueError("config is required for baseline evaluation")

    x_train_summary, x_train_full, y_train = _feature_arrays(train_records)
    x_test_summary, x_test_full, y_test = _feature_arrays(test_records)

    results: Dict[str, Dict] = {}

    train_entropy = x_train_summary[:, 0]
    class_centroids = {
        cls: train_entropy[y_train["outcome"] == cls].mean()
        for cls in sorted(np.unique(y_train["outcome"]))
    }
    threshold_preds = [
        min(class_centroids, key=lambda cls: abs(value - class_centroids[cls]))
        for value in x_test_summary[:, 0]
    ]
    results["entropy_threshold"] = {
        "outcome_accuracy": float(accuracy_score(y_test["outcome"], threshold_preds)),
        "outcome_macro_f1": float(
            f1_score(y_test["outcome"], threshold_preds, average="macro", zero_division=0)
        ),
    }

    logistic = {}
    mlp = {}
    for target_name in ["trajectory_shape", "attention_pattern", "confidence", "outcome"]:
        lr_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=42),
        )
        lr_model.fit(x_train_summary, y_train[target_name])
        lr_pred = lr_model.predict(x_test_summary)
        logistic[target_name] = {
            "accuracy": float(accuracy_score(y_test[target_name], lr_pred)),
            "macro_f1": float(f1_score(y_test[target_name], lr_pred, average="macro", zero_division=0)),
        }

        mlp_model = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=config.baseline_mlp_max_iter,
                early_stopping=config.baseline_mlp_early_stopping,
                random_state=42,
            ),
        )
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            mlp_model.fit(x_train_full, y_train[target_name])
        mlp_pred = mlp_model.predict(x_test_full)
        mlp_inner = mlp_model.named_steps["mlpclassifier"]
        converged = not any(
            issubclass(warning.category, ConvergenceWarning)
            for warning in caught_warnings
        )
        mlp[target_name] = {
            "accuracy": float(accuracy_score(y_test[target_name], mlp_pred)),
            "macro_f1": float(f1_score(y_test[target_name], mlp_pred, average="macro", zero_division=0)),
            "converged": bool(converged),
            "n_iter": int(mlp_inner.n_iter_),
        }

    results["logistic_regression"] = logistic
    results["mlp"] = mlp
    results["mlp_converged"] = bool(mlp["outcome"]["converged"])
    return results


def ensure_trace_corpus(
    config,
    root_dir: str | Path,
    device: torch.device,
    rebuild: bool = False,
    strict_existing: bool = False,
) -> Dict[str, Any]:
    del config, device, strict_existing
    root_dir = Path(root_dir)
    expected_files = {
        split: root_dir / f"{split}.jsonl"
        for split in ("train", "val", "test")
    }

    if rebuild:
        raise NotImplementedError(
            "E30 is self-contained but does not rebuild traces from raw data."
        )

    missing = [str(path) for path in expected_files.values() if not path.exists()]
    if missing:
        fallback_root = Path(__file__).resolve().parents[2] / "e29" / "trace_cache"
        fallback_files = {
            split: fallback_root / f"{split}.jsonl"
            for split in ("train", "val", "test")
        }
        fallback_missing = [str(path) for path in fallback_files.values() if not path.exists()]
        if fallback_missing:
            raise FileNotFoundError(
                f"Missing E30 trace cache files: {missing}. Fallback e29 cache is also incomplete: {fallback_missing}."
            )
        root_dir = fallback_root
        expected_files = fallback_files

    metadata_path = root_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as handle:
            metadata = json.load(handle)

    counts = {
        split: len(load_trace_records(path))
        for split, path in expected_files.items()
    }
    return {
        "trace_dir": str(root_dir),
        "metadata_path": str(metadata_path) if metadata_path.exists() else None,
        "counts": counts,
        "label_version": metadata.get("label_version"),
        "evidence_target_version": metadata.get("evidence_target_version"),
        "corpus_fingerprint": metadata.get("corpus_fingerprint"),
    }
