from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence

import numpy as np

from src.coordinators.mlp import MLPTrainingConfig
from src.datasets.multiview_code_patch_selection import (
    MultiViewCodePatchDatasetConfig,
    MultiViewTaskExample,
    build_multiview_code_patch_splits,
    candidate_attribute_bits,
    example_oracle_metadata,
    format_clone_prompt,
)
from src.experiments.real_shared_weight_latent_coordination import BagOfWordsDiagnosticBaseline, _text_tokens
from src.experiments.run_stage31_frozen_diagnosis import (
    _candidate_lexical_overlap_predictions,
    _candidate_values,
    _stable_argmax,
    _static_import_frequency_predictions,
)
from src.experiments.run_stage3_gpu_hard_validation import _stage_from_config


DEFAULT_CONFIG = "configs/stage33_role_balanced_real_code_cuda.json"
DEFAULT_OUTPUT = "results/stage33_role_balanced_dataset_diagnostics.json"
NEAR_CHANCE_THRESHOLD = 0.2250
FREQUENCY_THRESHOLD = 0.3000
SINGLE_ROLE_STRUCTURED_THRESHOLD = 0.3500
PAIRWISE_STRUCTURED_TARGET = 0.7000
ALL_ROLE_STRUCTURED_TARGET = 0.9000


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 3.3 role-balanced dataset diagnostics.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_path = Path(str(args.output or config.get("dataset_diagnostics_output_path", DEFAULT_OUTPUT)))
    result = run_diagnostics(config, config_path=config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if not result["summary"]["pretraining_dataset_validity_passed"]:
        raise SystemExit(2)


def run_diagnostics(config: Dict[str, object], config_path: Path | None = None) -> Dict[str, object]:
    stage = _stage_from_config(config)
    rows = []
    for seed in [int(value) for value in stage.seeds]:
        dataset_config = stage33_dataset_config(config)
        splits = build_multiview_code_patch_splits(dataset_config, seed=seed, repo_root=Path("."))
        rows.append(_diagnostics_for_seed(seed, splits, config))
    return {
        "metadata": {
            "stage": "stage3.3_role_balanced_dataset_diagnostics",
            "config_path": str(config_path) if config_path else None,
            "dataset_source": "real_import_restore_role_balanced_v2",
            "architecture_changes": "none",
            "scope": "dataset-only shortcut diagnostics before Stage 3.3 training",
            "near_chance_threshold": NEAR_CHANCE_THRESHOLD,
            "single_role_structured_threshold": SINGLE_ROLE_STRUCTURED_THRESHOLD,
            "pairwise_structured_target": PAIRWISE_STRUCTURED_TARGET,
            "all_role_structured_target": ALL_ROLE_STRUCTURED_TARGET,
        },
        "dataset_config": asdict(stage33_dataset_config(config)),
        "seed_rows": rows,
        "summary": _summary(rows),
    }


def stage33_dataset_config(config: Dict[str, object]) -> MultiViewCodePatchDatasetConfig:
    stage = _stage_from_config(config)
    dataset = dict(config.get("dataset_config", {}))
    return MultiViewCodePatchDatasetConfig(
        n_train=stage.n_train,
        n_dev=stage.n_dev,
        n_test=stage.n_test,
        num_candidates=8,
        n_views=4,
        source_roots=tuple(dataset.get("source_roots", ("src", "tests"))),
        max_files=int(dataset.get("max_files", 40)),
        snippet_radius=int(dataset.get("snippet_radius", 2)),
        include_private_signal_tokens=False,
        dataset_source=str(dataset.get("dataset_source", "real_import_restore_role_balanced_v2")),
        generator_version=str(dataset.get("generator_version", f"import_restore_role_balanced_v2_{stage.name}")),
        candidate_representation=str(dataset.get("candidate_representation", "hardened_redacted_v2")),
    )


def _diagnostics_for_seed(
    seed: int,
    splits: Dict[str, Sequence[MultiViewTaskExample]],
    config: Dict[str, object],
) -> Dict[str, object]:
    all_examples = [example for rows in splits.values() for example in rows]
    train = list(splits["train"])
    test = list(splits["test"])
    y_test = np.asarray([example.label for example in test], dtype=np.int64)
    train_labels = np.asarray([example.label for example in train], dtype=np.int64)

    lexical = _candidate_lexical_overlap_predictions(test)
    frequency = _static_import_frequency_predictions(train, test)
    majority = int(np.argmax(np.bincount(train_labels, minlength=8)))
    candidate_order = np.zeros(len(test), dtype=np.int64)
    single_role_lexical = {
        str(role_id): _accuracy(_single_role_lexical_overlap_predictions(test, role_id), y_test)
        for role_id in range(4)
    }
    single_role_frequency = {
        str(role_id): _accuracy(_single_role_candidate_frequency_predictions(train, test, role_id), y_test)
        for role_id in range(4)
    }
    single_role_structured = {
        str(role_id): _accuracy(_structured_oracle_predictions(test, [role_id]), y_test)
        for role_id in range(4)
    }
    pairwise_structured = {
        f"{left},{right}": _accuracy(_structured_oracle_predictions(test, [left, right]), y_test)
        for left in range(4)
        for right in range(left + 1, 4)
    }
    all_role_structured = _accuracy(_structured_oracle_predictions(test, [0, 1, 2, 3]), y_test)
    single_view = _single_view_baselines(seed, splits, config)
    leakage = _leakage_diagnostics(all_examples)

    baselines = {
        "candidate_lexical_overlap_accuracy": _accuracy(lexical, y_test),
        "static_import_frequency_accuracy": _accuracy(frequency, y_test),
        "candidate_order_accuracy": _accuracy(candidate_order, y_test),
        "majority_accuracy": float(np.mean(np.full(len(y_test), majority, dtype=np.int64) == y_test)),
        "single_view_text": single_view,
        "single_role_lexical_overlap": single_role_lexical,
        "single_role_candidate_frequency": single_role_frequency,
        "single_role_structured_oracle": single_role_structured,
        "pairwise_role_structured_oracle": pairwise_structured,
        "all_role_structured_oracle_accuracy": all_role_structured,
    }
    return {
        "seed": int(seed),
        "split_sizes": {split: len(rows) for split, rows in splits.items()},
        "label_counts": {
            split: {str(index): int([example.label for example in rows].count(index)) for index in range(8)}
            for split, rows in splits.items()
        },
        "leakage": leakage,
        "baselines": baselines,
        "gate_results": _gate_results(leakage, baselines),
    }


def _single_role_lexical_overlap_predictions(examples: Sequence[MultiViewTaskExample], role_id: int) -> np.ndarray:
    predictions = []
    for example in examples:
        view_text = example.views[role_id].text.lower()
        view_tokens = Counter(_text_tokens(view_text))
        scores = []
        for index, _candidate in enumerate(example.candidates):
            values = _candidate_values(example, index)
            symbol, module, _style, target_file = values
            exact_score = 0.0
            for value, weight in ((symbol, 4.0), (module, 4.0), (target_file, 2.0)):
                if value and str(value).lower() in view_text:
                    exact_score += weight
            candidate_tokens = [token for value in values for token in _text_tokens(str(value)) if token not in {"from", "import", "py"}]
            scores.append(exact_score + float(sum(view_tokens.get(token, 0) for token in candidate_tokens)))
        predictions.append(_stable_argmax(scores))
    return np.asarray(predictions, dtype=np.int64)


def _single_role_candidate_frequency_predictions(
    train_examples: Sequence[MultiViewTaskExample],
    test_examples: Sequence[MultiViewTaskExample],
    role_id: int,
) -> np.ndarray:
    by_role_pair = Counter()
    by_role_module = Counter()
    by_role_symbol = Counter()
    for example in train_examples:
        evidence = tuple(int(value) for value in example_oracle_metadata(example).get("evidence_bits", [0, 0, 0, 0]))
        role_value = evidence[int(role_id)]
        symbol, module, _style, _target = _candidate_values(example, int(example.label))
        by_role_pair[(role_value, module, symbol)] += 1
        by_role_module[(role_value, module)] += 1
        by_role_symbol[(role_value, symbol)] += 1

    predictions = []
    for example in test_examples:
        evidence = tuple(int(value) for value in example_oracle_metadata(example).get("evidence_bits", [0, 0, 0, 0]))
        role_value = evidence[int(role_id)]
        scores = []
        for index, _candidate in enumerate(example.candidates):
            symbol, module, _style, _target = _candidate_values(example, index)
            scores.append(
                100.0 * by_role_pair[(role_value, module, symbol)]
                + 10.0 * by_role_module[(role_value, module)]
                + by_role_symbol[(role_value, symbol)]
            )
        predictions.append(_stable_argmax(scores))
    return np.asarray(predictions, dtype=np.int64)


def _structured_oracle_predictions(examples: Sequence[MultiViewTaskExample], roles: Sequence[int]) -> np.ndarray:
    predictions: List[int] = []
    selected = [int(role) for role in roles]
    for example in examples:
        evidence = [int(value) for value in example_oracle_metadata(example).get("evidence_bits", [0, 0, 0, 0])]
        scores = []
        for candidate in example.candidates:
            bits = candidate_attribute_bits(example, candidate)
            scores.append(sum(int(bits[role] == evidence[role]) for role in selected))
        predictions.append(int(np.argmax(np.asarray(scores, dtype=np.float32))))
    return np.asarray(predictions, dtype=np.int64)


def _leakage_diagnostics(examples: Sequence[MultiViewTaskExample]) -> Dict[str, float]:
    counts = Counter()
    total = len(examples)
    for example in examples:
        view_text = "\n".join(view.text for view in example.views).lower()
        gold_symbol, gold_module, _style, gold_target = _candidate_values(example, int(example.label))
        gold_import = str(example_oracle_metadata(example).get("gold_import_statement", ""))
        if _contains_exact(view_text, gold_symbol):
            counts["exact_gold_symbol"] += 1
        if _contains_exact(view_text, gold_module):
            counts["exact_gold_module_path"] += 1
        if _contains_exact(view_text, gold_target):
            counts["exact_gold_target_path"] += 1
        if gold_import and gold_import.lower() in view_text:
            counts["exact_gold_full_import_statement"] += 1
        if any(candidate.text and candidate.text.lower() in view_text for candidate in example.candidates):
            counts["exact_candidate_patch_text"] += 1
        candidate_values = [_candidate_values(example, index) for index in range(len(example.candidates))]
        if any(_contains_exact(view_text, value) for values in candidate_values for value in (values[0], values[1], values[3])):
            counts["exact_candidate_symbol_module_or_path"] += 1
    return {
        "exact_gold_symbol_rate": counts["exact_gold_symbol"] / max(1, total),
        "exact_gold_module_path_rate": counts["exact_gold_module_path"] / max(1, total),
        "exact_gold_target_path_rate": counts["exact_gold_target_path"] / max(1, total),
        "exact_gold_full_import_statement_rate": counts["exact_gold_full_import_statement"] / max(1, total),
        "exact_candidate_patch_text_rate": counts["exact_candidate_patch_text"] / max(1, total),
        "exact_candidate_symbol_module_or_path_rate": counts["exact_candidate_symbol_module_or_path"] / max(1, total),
    }


def _contains_exact(text: str, value: str) -> bool:
    value = str(value or "").lower()
    if not value:
        return False
    if re.fullmatch(r"[a-z_][a-z0-9_]*", value):
        return re.search(rf"\b{re.escape(value)}\b", text) is not None
    return value in text


def _single_view_baselines(
    seed: int,
    splits: Dict[str, Sequence[MultiViewTaskExample]],
    config: Dict[str, object],
) -> Dict[str, float]:
    stage = _stage_from_config(config)
    training = MLPTrainingConfig(
        epochs=int(config.get("dataset_diagnostics", {}).get("single_view_epochs", 4)),
        batch_size=int(stage.batch_size),
        lr=float(config.get("dataset_diagnostics", {}).get("single_view_lr", 0.002)),
        weight_decay=0.0001,
        patience=int(config.get("dataset_diagnostics", {}).get("single_view_patience", 2)),
        hidden_dims=(32,),
    )
    out: Dict[str, float] = {}
    y = np.asarray([example.label for example in splits["test"]], dtype=np.int64)
    for role_index in range(4):
        baseline = BagOfWordsDiagnosticBaseline(
            method=f"single_view_text_role_{role_index}",
            num_classes=8,
            training=training,
            seed=seed + 30_000 + role_index,
            device="cuda" if bool(config.get("dataset_diagnostics", {}).get("use_cuda_for_single_view", False)) else "cpu",
            text_builder=lambda example, role_index=role_index: format_clone_prompt(example, role_index),
            feature_dim=int(config.get("dataset_diagnostics", {}).get("single_view_feature_dim", 128)),
        )
        baseline.fit(splits["train"], splits["dev"])
        out[f"single_view_text_role_{role_index}"] = _accuracy(baseline.predict(splits["test"]), y)
    return out


def _gate_results(leakage: Dict[str, float], baselines: Dict[str, object]) -> List[Dict[str, object]]:
    single_role_structured = dict(baselines["single_role_structured_oracle"])
    pairwise_structured = dict(baselines["pairwise_role_structured_oracle"])
    single_role_lexical = dict(baselines["single_role_lexical_overlap"])
    single_role_frequency = dict(baselines["single_role_candidate_frequency"])
    single_view = dict(baselines["single_view_text"])
    return [
        {"gate": "exact gold symbol leakage == 0.0000", "pass": leakage["exact_gold_symbol_rate"] == 0.0, "value": leakage["exact_gold_symbol_rate"]},
        {"gate": "exact gold module path leakage == 0.0000", "pass": leakage["exact_gold_module_path_rate"] == 0.0, "value": leakage["exact_gold_module_path_rate"]},
        {"gate": "exact gold full import leakage == 0.0000", "pass": leakage["exact_gold_full_import_statement_rate"] == 0.0, "value": leakage["exact_gold_full_import_statement_rate"]},
        {"gate": "exact candidate patch text leakage == 0.0000", "pass": leakage["exact_candidate_patch_text_rate"] == 0.0, "value": leakage["exact_candidate_patch_text_rate"]},
        {"gate": "candidate lexical-overlap baseline <= 0.2250", "pass": float(baselines["candidate_lexical_overlap_accuracy"]) <= NEAR_CHANCE_THRESHOLD, "value": baselines["candidate_lexical_overlap_accuracy"]},
        {"gate": "static import-frequency baseline <= 0.3000", "pass": float(baselines["static_import_frequency_accuracy"]) <= FREQUENCY_THRESHOLD, "value": baselines["static_import_frequency_accuracy"]},
        {"gate": "single-role lexical-overlap baselines <= 0.2250", "pass": max(single_role_lexical.values() or [1.0]) <= NEAR_CHANCE_THRESHOLD, "value": single_role_lexical},
        {"gate": "single-role candidate-frequency baselines <= 0.3000", "pass": max(single_role_frequency.values() or [1.0]) <= FREQUENCY_THRESHOLD, "value": single_role_frequency},
        {"gate": "candidate-order baseline near chance <= 0.2250", "pass": float(baselines["candidate_order_accuracy"]) <= NEAR_CHANCE_THRESHOLD, "value": baselines["candidate_order_accuracy"]},
        {"gate": "majority baseline near chance <= 0.2250", "pass": float(baselines["majority_accuracy"]) <= NEAR_CHANCE_THRESHOLD, "value": baselines["majority_accuracy"]},
        {"gate": "single-view text baselines <= 0.2250", "pass": max(single_view.values() or [1.0]) <= NEAR_CHANCE_THRESHOLD, "value": single_view},
        {"gate": "single-role structured oracle <= 0.3500", "pass": max(single_role_structured.values() or [1.0]) <= SINGLE_ROLE_STRUCTURED_THRESHOLD, "value": single_role_structured},
        {"gate": "at least one pairwise-role structured oracle >= 0.7000", "pass": max(pairwise_structured.values() or [0.0]) >= PAIRWISE_STRUCTURED_TARGET, "value": pairwise_structured},
        {"gate": "all-role structured oracle >= 0.9000", "pass": float(baselines["all_role_structured_oracle_accuracy"]) >= ALL_ROLE_STRUCTURED_TARGET, "value": baselines["all_role_structured_oracle_accuracy"]},
    ]


def _summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    mean_single_structured = _mean_nested(rows, "single_role_structured_oracle")
    mean_pairwise_structured = _mean_nested(rows, "pairwise_role_structured_oracle")
    mean_single_lexical = _mean_nested(rows, "single_role_lexical_overlap")
    mean_single_frequency = _mean_nested(rows, "single_role_candidate_frequency")
    max_single_view = max([value for row in rows for value in row["baselines"]["single_view_text"].values()] or [0.0])
    max_leakage = {
        key: max(float(row["leakage"].get(key, 0.0)) for row in rows) if rows else 1.0
        for key in (
            "exact_gold_symbol_rate",
            "exact_gold_module_path_rate",
            "exact_gold_full_import_statement_rate",
            "exact_candidate_patch_text_rate",
            "exact_candidate_symbol_module_or_path_rate",
        )
    }
    summary_values = {
        "mean_candidate_lexical_overlap_accuracy": _mean_baseline(rows, "candidate_lexical_overlap_accuracy"),
        "mean_static_import_frequency_accuracy": _mean_baseline(rows, "static_import_frequency_accuracy"),
        "mean_candidate_order_accuracy": _mean_baseline(rows, "candidate_order_accuracy"),
        "mean_majority_accuracy": _mean_baseline(rows, "majority_accuracy"),
        "mean_single_role_lexical_overlap_accuracy": mean_single_lexical,
        "mean_single_role_candidate_frequency_accuracy": mean_single_frequency,
        "mean_single_role_structured_oracle_accuracy": mean_single_structured,
        "mean_pairwise_role_structured_oracle_accuracy": mean_pairwise_structured,
        "mean_all_role_structured_oracle_accuracy": _mean_baseline(rows, "all_role_structured_oracle_accuracy"),
        "max_single_view_accuracy": max_single_view,
        "max_mean_single_role_structured_oracle_accuracy": max(mean_single_structured.values() or [0.0]),
        "max_mean_pairwise_role_structured_oracle_accuracy": max(mean_pairwise_structured.values() or [0.0]),
    }
    gate_table = [
        {"gate": "exact gold symbol leakage == 0.0000", "pass": max_leakage["exact_gold_symbol_rate"] == 0.0, "value": max_leakage["exact_gold_symbol_rate"]},
        {"gate": "exact gold module path leakage == 0.0000", "pass": max_leakage["exact_gold_module_path_rate"] == 0.0, "value": max_leakage["exact_gold_module_path_rate"]},
        {"gate": "exact gold full import leakage == 0.0000", "pass": max_leakage["exact_gold_full_import_statement_rate"] == 0.0, "value": max_leakage["exact_gold_full_import_statement_rate"]},
        {"gate": "exact candidate patch text leakage == 0.0000", "pass": max_leakage["exact_candidate_patch_text_rate"] == 0.0, "value": max_leakage["exact_candidate_patch_text_rate"]},
        {"gate": "exact candidate symbol/module/path leakage == 0.0000", "pass": max_leakage["exact_candidate_symbol_module_or_path_rate"] == 0.0, "value": max_leakage["exact_candidate_symbol_module_or_path_rate"]},
        {"gate": "candidate lexical-overlap baseline <= 0.2250 mean", "pass": summary_values["mean_candidate_lexical_overlap_accuracy"] <= NEAR_CHANCE_THRESHOLD, "value": summary_values["mean_candidate_lexical_overlap_accuracy"]},
        {"gate": "static import-frequency baseline <= 0.3000 mean", "pass": summary_values["mean_static_import_frequency_accuracy"] <= FREQUENCY_THRESHOLD, "value": summary_values["mean_static_import_frequency_accuracy"]},
        {"gate": "single-role lexical-overlap baselines <= 0.2250 mean", "pass": max(mean_single_lexical.values() or [1.0]) <= NEAR_CHANCE_THRESHOLD, "value": mean_single_lexical},
        {"gate": "single-role candidate-frequency baselines <= 0.3000 mean", "pass": max(mean_single_frequency.values() or [1.0]) <= FREQUENCY_THRESHOLD, "value": mean_single_frequency},
        {"gate": "candidate-order baseline near chance <= 0.2250 mean", "pass": summary_values["mean_candidate_order_accuracy"] <= NEAR_CHANCE_THRESHOLD, "value": summary_values["mean_candidate_order_accuracy"]},
        {"gate": "majority baseline near chance <= 0.2250 mean", "pass": summary_values["mean_majority_accuracy"] <= NEAR_CHANCE_THRESHOLD, "value": summary_values["mean_majority_accuracy"]},
        {"gate": "single-view text baselines <= 0.2250 max", "pass": max_single_view <= NEAR_CHANCE_THRESHOLD, "value": max_single_view},
        {"gate": "no single-role structured oracle > 0.3500 mean", "pass": summary_values["max_mean_single_role_structured_oracle_accuracy"] <= SINGLE_ROLE_STRUCTURED_THRESHOLD, "value": mean_single_structured},
        {"gate": "at least one pairwise-role structured oracle >= 0.7000 mean", "pass": summary_values["max_mean_pairwise_role_structured_oracle_accuracy"] >= PAIRWISE_STRUCTURED_TARGET, "value": mean_pairwise_structured},
        {"gate": "all-role structured oracle >= 0.9000 mean", "pass": summary_values["mean_all_role_structured_oracle_accuracy"] >= ALL_ROLE_STRUCTURED_TARGET, "value": summary_values["mean_all_role_structured_oracle_accuracy"]},
    ]
    return {
        "pretraining_dataset_validity_passed": all(bool(gate["pass"]) for gate in gate_table),
        "gate_table": gate_table,
        "mean_candidate_lexical_overlap_accuracy": _mean_baseline(rows, "candidate_lexical_overlap_accuracy"),
        "mean_static_import_frequency_accuracy": _mean_baseline(rows, "static_import_frequency_accuracy"),
        "mean_candidate_order_accuracy": _mean_baseline(rows, "candidate_order_accuracy"),
        "mean_majority_accuracy": _mean_baseline(rows, "majority_accuracy"),
        "mean_single_role_lexical_overlap_accuracy": mean_single_lexical,
        "mean_single_role_candidate_frequency_accuracy": mean_single_frequency,
        "mean_single_role_structured_oracle_accuracy": mean_single_structured,
        "mean_pairwise_role_structured_oracle_accuracy": mean_pairwise_structured,
        "mean_all_role_structured_oracle_accuracy": _mean_baseline(rows, "all_role_structured_oracle_accuracy"),
        "max_single_view_accuracy": max_single_view,
        "max_mean_single_role_structured_oracle_accuracy": summary_values["max_mean_single_role_structured_oracle_accuracy"],
        "max_mean_pairwise_role_structured_oracle_accuracy": summary_values["max_mean_pairwise_role_structured_oracle_accuracy"],
    }


def _mean_nested(rows: Sequence[Dict[str, object]], name: str) -> Dict[str, float]:
    keys = sorted({key for row in rows for key in dict(row["baselines"][name]).keys()})
    return {
        key: float(mean(float(row["baselines"][name][key]) for row in rows))
        for key in keys
    }


def _mean_baseline(rows: Sequence[Dict[str, object]], name: str) -> float:
    values = [float(row["baselines"][name]) for row in rows]
    return float(mean(values)) if values else 0.0


def _accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(predictions.astype(np.int64) == labels.astype(np.int64))) if len(labels) else 0.0


if __name__ == "__main__":
    main()
