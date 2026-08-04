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
    format_clone_prompt,
)
from src.experiments.real_shared_weight_latent_coordination import BagOfWordsDiagnosticBaseline
from src.experiments.run_stage31_frozen_diagnosis import (
    _candidate_lexical_overlap_predictions,
    _candidate_values,
    _static_import_frequency_predictions,
)
from src.experiments.run_stage3_gpu_hard_validation import _stage_from_config


DEFAULT_CONFIG = "configs/stage32_hardened_real_code_cuda_smoke.json"
DEFAULT_OUTPUT = "results/stage32_hardened_dataset_diagnostics.json"
NEAR_CHANCE_THRESHOLD = 0.2250


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 3.2 hardened dataset-only shortcut diagnostics.")
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
    seeds = [int(value) for value in stage.seeds]
    rows = []
    for seed in seeds:
        dataset_config = stage32_dataset_config(config)
        splits = build_multiview_code_patch_splits(dataset_config, seed=seed, repo_root=Path("."))
        rows.append(_diagnostics_for_seed(seed, splits, config))
    return {
        "metadata": {
            "stage": "stage3.2_hardened_dataset_diagnostics",
            "config_path": str(config_path) if config_path else None,
            "dataset_source": "real_import_restore_hardened",
            "architecture_changes": "none",
            "scope": "dataset-only shortcut diagnostics before Stage 3.2 training",
            "near_chance_threshold": NEAR_CHANCE_THRESHOLD,
        },
        "dataset_config": asdict(stage32_dataset_config(config)),
        "seed_rows": rows,
        "summary": _summary(rows),
    }


def stage32_dataset_config(config: Dict[str, object]) -> MultiViewCodePatchDatasetConfig:
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
        dataset_source="real_import_restore_hardened",
        generator_version=str(dataset.get("generator_version", f"import_restore_redacted_v1_{stage.name}")),
        candidate_representation=str(dataset.get("candidate_representation", "hardened_redacted_v1")),
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
    oracle = _explicit_structured_oracle_predictions(test)
    single_view = _single_view_baselines(seed, splits, config)
    leakage = _leakage_diagnostics(all_examples)

    return {
        "seed": int(seed),
        "split_sizes": {split: len(rows) for split, rows in splits.items()},
        "label_counts": {
            split: {str(index): int([example.label for example in rows].count(index)) for index in range(8)}
            for split, rows in splits.items()
        },
        "leakage": leakage,
        "baselines": {
            "candidate_lexical_overlap_accuracy": _accuracy(lexical, y_test),
            "static_import_frequency_accuracy": _accuracy(frequency, y_test),
            "candidate_order_accuracy": _accuracy(candidate_order, y_test),
            "majority_accuracy": float(np.mean(np.full(len(y_test), majority, dtype=np.int64) == y_test)),
            "single_view_text": single_view,
            "explicit_structured_oracle_accuracy": _accuracy(oracle, y_test),
        },
        "gate_results": _gate_results(leakage, _accuracy(lexical, y_test), _accuracy(frequency, y_test), _accuracy(candidate_order, y_test), float(np.mean(np.full(len(y_test), majority, dtype=np.int64) == y_test)), single_view, _accuracy(oracle, y_test)),
    }


def _leakage_diagnostics(examples: Sequence[MultiViewTaskExample]) -> Dict[str, float]:
    counts = Counter()
    total = len(examples)
    for example in examples:
        view_text = "\n".join(view.text for view in example.views).lower()
        gold_symbol, gold_module, _style, gold_target = _candidate_values(example, int(example.label))
        gold_import = str(example.metadata.get("gold_import_statement", ""))
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


def _explicit_structured_oracle_predictions(examples: Sequence[MultiViewTaskExample]) -> np.ndarray:
    predictions: List[int] = []
    for example in examples:
        evidence = [int(value) for value in example.metadata.get("evidence_bits", [0, 0, 0, 0])]
        scores = []
        for candidate in example.candidates:
            bits = candidate_attribute_bits(example, candidate)
            scores.append(sum(int(a == b) for a, b in zip(bits, evidence)))
        predictions.append(int(np.argmax(np.asarray(scores, dtype=np.float32))))
    return np.asarray(predictions, dtype=np.int64)


def _gate_results(
    leakage: Dict[str, float],
    lexical: float,
    frequency: float,
    candidate_order: float,
    majority: float,
    single_view: Dict[str, float],
    oracle: float,
) -> List[Dict[str, object]]:
    return [
        {"gate": "exact gold symbol leakage == 0.0000", "pass": leakage["exact_gold_symbol_rate"] == 0.0, "value": leakage["exact_gold_symbol_rate"]},
        {"gate": "exact gold module path leakage == 0.0000", "pass": leakage["exact_gold_module_path_rate"] == 0.0, "value": leakage["exact_gold_module_path_rate"]},
        {"gate": "exact gold full import leakage == 0.0000", "pass": leakage["exact_gold_full_import_statement_rate"] == 0.0, "value": leakage["exact_gold_full_import_statement_rate"]},
        {"gate": "exact candidate patch text leakage == 0.0000", "pass": leakage["exact_candidate_patch_text_rate"] == 0.0, "value": leakage["exact_candidate_patch_text_rate"]},
        {"gate": "candidate lexical-overlap baseline <= 0.2250", "pass": lexical <= NEAR_CHANCE_THRESHOLD, "value": lexical},
        {"gate": "static import-frequency baseline <= 0.3000", "pass": frequency <= 0.3000, "value": frequency},
        {"gate": "candidate-order baseline near chance <= 0.2250", "pass": candidate_order <= NEAR_CHANCE_THRESHOLD, "value": candidate_order},
        {"gate": "majority baseline near chance <= 0.2250", "pass": majority <= NEAR_CHANCE_THRESHOLD, "value": majority},
        {"gate": "single-view text baselines <= 0.2250", "pass": max(single_view.values() or [1.0]) <= NEAR_CHANCE_THRESHOLD, "value": single_view},
        {"gate": "explicit structured oracle >= 0.9000", "pass": oracle >= 0.9000, "value": oracle},
    ]


def _summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    gates: Dict[str, List[bool]] = {}
    gate_values: Dict[str, List[object]] = {}
    for row in rows:
        for gate in row["gate_results"]:
            gates.setdefault(str(gate["gate"]), []).append(bool(gate["pass"]))
            gate_values.setdefault(str(gate["gate"]), []).append(gate["value"])
    return {
        "pretraining_dataset_validity_passed": all(all(values) for values in gates.values()) if gates else False,
        "gate_table": [
            {
                "gate": gate,
                "pass": all(gates[gate]),
                "values": gate_values.get(gate, []),
            }
            for gate in sorted(gates)
        ],
        "mean_candidate_lexical_overlap_accuracy": _mean_baseline(rows, "candidate_lexical_overlap_accuracy"),
        "mean_static_import_frequency_accuracy": _mean_baseline(rows, "static_import_frequency_accuracy"),
        "mean_candidate_order_accuracy": _mean_baseline(rows, "candidate_order_accuracy"),
        "mean_majority_accuracy": _mean_baseline(rows, "majority_accuracy"),
        "mean_explicit_structured_oracle_accuracy": _mean_baseline(rows, "explicit_structured_oracle_accuracy"),
        "max_single_view_accuracy": max(
            [value for row in rows for value in row["baselines"]["single_view_text"].values()] or [0.0]
        ),
    }


def _mean_baseline(rows: Sequence[Dict[str, object]], name: str) -> float:
    values = [float(row["baselines"][name]) for row in rows]
    return float(mean(values)) if values else 0.0


def _accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(predictions.astype(np.int64) == labels.astype(np.int64))) if len(labels) else 0.0


if __name__ == "__main__":
    main()
