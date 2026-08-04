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
    SANITIZED_CANDIDATE_REPRESENTATION,
    SANITIZED_DATASET_SOURCE,
    MultiViewCodePatchDatasetConfig,
    MultiViewTaskExample,
    apply_example_control,
    build_multiview_code_patch_splits,
    candidate_attribute_bits,
    example_oracle_metadata,
    format_candidate_block,
    format_clone_prompt,
    model_record_from_example,
)
from src.experiments.real_shared_weight_latent_coordination import BagOfWordsDiagnosticBaseline, _text_tokens
from src.experiments.run_stage31_frozen_diagnosis import (
    _candidate_lexical_overlap_predictions,
    _candidate_values,
    _stable_argmax,
    _static_import_frequency_predictions,
)
from src.experiments.run_stage3_gpu_hard_validation import _stage_from_config


DEFAULT_CONFIG = "configs/stage34_candidate_sanitization_cuda_smoke.json"
DEFAULT_OUTPUT = "results/stage34_candidate_sanitization_diagnostics.json"
STRICT_MAX = 0.1800
PREFERRED_MAX = 0.1500
STATIC_FREQUENCY_MAX = 0.2000
PAIRWISE_STRUCTURED_TARGET = 0.7000
ALL_ROLE_STRUCTURED_TARGET = 0.9000


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 3.4 candidate-sanitization dataset diagnostics.")
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
        dataset_config = stage34_dataset_config(config)
        splits = build_multiview_code_patch_splits(dataset_config, seed=seed, repo_root=Path("."))
        rows.append(_diagnostics_for_seed(seed, splits, config))
    return {
        "metadata": {
            "stage": "stage3.4_candidate_sanitization_diagnostics",
            "config_path": str(config_path) if config_path else None,
            "dataset_source": SANITIZED_DATASET_SOURCE,
            "architecture_changes": "none",
            "scope": "dataset-only shortcut diagnostics before any Stage 3.4 training",
            "preferred_max": PREFERRED_MAX,
            "strict_max": STRICT_MAX,
            "static_frequency_max": STATIC_FREQUENCY_MAX,
            "pairwise_structured_target": PAIRWISE_STRUCTURED_TARGET,
            "all_role_structured_target": ALL_ROLE_STRUCTURED_TARGET,
        },
        "dataset_config": asdict(stage34_dataset_config(config)),
        "seed_rows": rows,
        "summary": _summary(rows),
    }


def stage34_dataset_config(config: Dict[str, object]) -> MultiViewCodePatchDatasetConfig:
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
        dataset_source=str(dataset.get("dataset_source", SANITIZED_DATASET_SOURCE)),
        generator_version=str(dataset.get("generator_version", "real_import_restore_candidate_sanitized_v3_stage34")),
        candidate_representation=str(dataset.get("candidate_representation", SANITIZED_CANDIDATE_REPRESENTATION)),
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

    candidate_only = _bow_accuracy(seed, splits, config, "candidate_only", lambda example: format_candidate_block(example))
    metadata_only = _bow_accuracy(seed + 101, splits, config, "candidate_metadata_only", _candidate_metadata_only_text)
    role_pair_only = _bow_accuracy(seed + 202, splits, config, "role_pair_only", _role_pair_only_text)
    masked = {
        split: apply_example_control(rows, "view_masked", seed=seed + 33_000)
        for split, rows in splits.items()
    }
    view_masked_candidates = _bow_accuracy(
        seed + 303,
        masked,
        config,
        "view_masked_candidates_visible",
        lambda example: "\n".join(view.text for view in example.views) + "\n" + format_candidate_block(example),
    )
    lexical = _candidate_lexical_overlap_predictions(test)
    frequency = _static_import_frequency_predictions(train, test)
    single_view = _single_view_baselines(seed, splits, config)
    family_only = _accuracy(_family_only_predictions(train, test), y_test)
    role_pair_structured = {
        f"{left},{right}": _accuracy(_structured_oracle_predictions(test, [left, right]), y_test)
        for left in range(4)
        for right in range(left + 1, 4)
    }
    all_role_structured = _accuracy(_structured_oracle_predictions(test, [0, 1, 2, 3]), y_test)
    leakage = _model_record_leakage_diagnostics(all_examples)

    baselines = {
        "candidate_only_accuracy": candidate_only,
        "candidate_metadata_only_accuracy": metadata_only,
        "role_pair_only_accuracy": role_pair_only,
        "family_only_accuracy": family_only,
        "view_masked_candidates_visible_accuracy": view_masked_candidates,
        "lexical_overlap_accuracy": _accuracy(lexical, y_test),
        "static_frequency_accuracy": _accuracy(frequency, y_test),
        "single_view_text": single_view,
        "pairwise_structured_oracle": role_pair_structured,
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


def _bow_accuracy(seed: int, splits: Dict[str, Sequence[MultiViewTaskExample]], config: Dict[str, object], method: str, text_builder) -> float:
    stage = _stage_from_config(config)
    diag = dict(config.get("dataset_diagnostics", {}))
    training = MLPTrainingConfig(
        epochs=int(diag.get("baseline_epochs", diag.get("single_view_epochs", 4))),
        batch_size=int(stage.batch_size),
        lr=float(diag.get("baseline_lr", diag.get("single_view_lr", 0.002))),
        weight_decay=0.0001,
        patience=int(diag.get("baseline_patience", diag.get("single_view_patience", 2))),
        hidden_dims=(32,),
    )
    baseline = BagOfWordsDiagnosticBaseline(
        method=method,
        num_classes=8,
        training=training,
        seed=seed + 70_000,
        device="cpu",
        text_builder=text_builder,
        feature_dim=int(diag.get("baseline_feature_dim", diag.get("single_view_feature_dim", 128))),
    )
    baseline.fit(splits["train"], splits["dev"])
    labels = np.asarray([example.label for example in splits["test"]], dtype=np.int64)
    return _accuracy(baseline.predict(splits["test"]), labels)


def _candidate_metadata_only_text(example: MultiViewTaskExample) -> str:
    record = model_record_from_example(example)
    rows = []
    for candidate in record["candidates"]:  # type: ignore[index]
        item = dict(candidate)
        item.pop("text", None)
        item.pop("candidate_id", None)
        item.pop("patch_hash", None)
        rows.append(json.dumps(item, sort_keys=True))
    return "\n".join(rows) + "\n" + json.dumps(record["metadata"], sort_keys=True)  # type: ignore[index]


def _role_pair_only_text(example: MultiViewTaskExample) -> str:
    record = model_record_from_example(example)
    metadata = dict(record["metadata"])  # type: ignore[arg-type]
    hardening = dict(metadata.get("hardening", {})) if isinstance(metadata.get("hardening"), dict) else {}
    return json.dumps(
        {
            "roles": [view.role for view in example.views],
            "role_pair_ids": hardening.get("role_pair_ids", "removed"),
            "pairwise_codebook": hardening.get("pairwise_codebook", "removed"),
        },
        sort_keys=True,
    )


def _family_only_predictions(
    train_examples: Sequence[MultiViewTaskExample],
    test_examples: Sequence[MultiViewTaskExample],
) -> np.ndarray:
    counts = Counter(int(example.label) for example in train_examples)
    by_family: Dict[str, Counter] = {}
    for example in train_examples:
        family = _normalized_family(example)
        by_family.setdefault(family, Counter())[int(example.label)] += 1
    majority = _stable_argmax([counts[index] for index in range(8)])
    predictions = []
    for example in test_examples:
        family_counts = by_family.get(_normalized_family(example))
        if not family_counts:
            predictions.append(majority)
        else:
            predictions.append(_stable_argmax([family_counts[index] for index in range(8)]))
    return np.asarray(predictions, dtype=np.int64)


def _normalized_family(example: MultiViewTaskExample) -> str:
    family = str(example_oracle_metadata(example).get("problem_family", "unknown"))
    return re.sub(r"^(train|dev|test)_", "", family)


def _single_view_baselines(
    seed: int,
    splits: Dict[str, Sequence[MultiViewTaskExample]],
    config: Dict[str, object],
) -> Dict[str, float]:
    return {
        str(role_id): _bow_accuracy(
            seed + 1_000 + role_id,
            splits,
            config,
            f"single_view_text_role_{role_id}",
            lambda example, role_id=role_id: format_clone_prompt(example, role_id),
        )
        for role_id in range(4)
    }


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


def _model_record_leakage_diagnostics(examples: Sequence[MultiViewTaskExample]) -> Dict[str, object]:
    counts = Counter()
    assertion_failures = 0
    total = len(examples)
    for example in examples:
        try:
            model_record = model_record_from_example(example)
        except AssertionError:
            assertion_failures += 1
            continue
        text = json.dumps(model_record, sort_keys=True).lower()
        oracle = example_oracle_metadata(example)
        candidate_values = oracle.get("candidate_patch_values", [])
        gold_values = candidate_values[int(example.label)] if isinstance(candidate_values, list) and int(example.label) < len(candidate_values) else []
        if isinstance(gold_values, list) and len(gold_values) >= 4:
            symbol, module, _style, target = [str(value) for value in gold_values[:4]]
            if _contains_exact(text, symbol):
                counts["exact_gold_symbol"] += 1
            if _contains_exact(text, module):
                counts["exact_gold_module_path"] += 1
            if _contains_exact(text, target):
                counts["exact_gold_path"] += 1
        gold_import = str(oracle.get("gold_import_statement", ""))
        if gold_import and gold_import.lower() in text:
            counts["exact_gold_import_statement"] += 1
        if isinstance(candidate_values, list):
            for values in candidate_values:
                if not isinstance(values, list) or len(values) < 4:
                    continue
                symbol, module, _style, target = [str(value) for value in values[:4]]
                if any(_contains_exact(text, value) for value in (symbol, module, target)):
                    counts["exact_candidate_symbol_module_or_path"] += 1
                    break
    return {
        "model_record_oracle_key_assertion_failures": assertion_failures,
        "exact_gold_symbol_rate": counts["exact_gold_symbol"] / max(1, total),
        "exact_gold_module_path_rate": counts["exact_gold_module_path"] / max(1, total),
        "exact_gold_path_rate": counts["exact_gold_path"] / max(1, total),
        "exact_gold_import_statement_rate": counts["exact_gold_import_statement"] / max(1, total),
        "exact_candidate_symbol_module_or_path_rate": counts["exact_candidate_symbol_module_or_path"] / max(1, total),
        "passes": assertion_failures == 0 and not counts,
    }


def _contains_exact(text: str, value: str) -> bool:
    value = str(value or "").lower()
    if not value or value in {"from", "import", "py"}:
        return False
    if re.fullmatch(r"[a-z_][a-z0-9_]*", value):
        return re.search(rf"\b{re.escape(value)}\b", text) is not None
    return value in text


def _gate_results(leakage: Dict[str, object], baselines: Dict[str, object]) -> List[Dict[str, object]]:
    single_view = dict(baselines["single_view_text"])
    pairwise = dict(baselines["pairwise_structured_oracle"])
    return [
        {"gate": "exact symbol/module/path/candidate leakage remains 0", "pass": bool(leakage.get("passes", False)), "value": leakage},
        {"gate": "candidate-only baseline <= 0.18", "pass": float(baselines["candidate_only_accuracy"]) <= STRICT_MAX, "preferred": float(baselines["candidate_only_accuracy"]) <= PREFERRED_MAX, "value": baselines["candidate_only_accuracy"]},
        {"gate": "candidate-metadata-only baseline <= 0.18", "pass": float(baselines["candidate_metadata_only_accuracy"]) <= STRICT_MAX, "preferred": float(baselines["candidate_metadata_only_accuracy"]) <= PREFERRED_MAX, "value": baselines["candidate_metadata_only_accuracy"]},
        {"gate": "role-pair-only baseline <= 0.18", "pass": float(baselines["role_pair_only_accuracy"]) <= STRICT_MAX, "preferred": float(baselines["role_pair_only_accuracy"]) <= PREFERRED_MAX, "value": baselines["role_pair_only_accuracy"]},
        {"gate": "family-only baseline <= 0.18", "pass": float(baselines["family_only_accuracy"]) <= STRICT_MAX, "preferred": float(baselines["family_only_accuracy"]) <= PREFERRED_MAX, "value": baselines["family_only_accuracy"]},
        {"gate": "view-masked + candidates-visible baseline <= 0.18", "pass": float(baselines["view_masked_candidates_visible_accuracy"]) <= STRICT_MAX, "value": baselines["view_masked_candidates_visible_accuracy"]},
        {"gate": "lexical-overlap baseline <= 0.18", "pass": float(baselines["lexical_overlap_accuracy"]) <= STRICT_MAX, "value": baselines["lexical_overlap_accuracy"]},
        {"gate": "static frequency baseline <= 0.20", "pass": float(baselines["static_frequency_accuracy"]) <= STATIC_FREQUENCY_MAX, "value": baselines["static_frequency_accuracy"]},
        {"gate": "single-view baselines <= 0.18", "pass": max(single_view.values() or [1.0]) <= STRICT_MAX, "value": single_view},
        {"gate": "at least one pairwise-role oracle >= 0.70", "pass": max(pairwise.values() or [0.0]) >= PAIRWISE_STRUCTURED_TARGET, "value": pairwise},
        {"gate": "all-role structured oracle >= 0.90", "pass": float(baselines["all_role_structured_oracle_accuracy"]) >= ALL_ROLE_STRUCTURED_TARGET, "value": baselines["all_role_structured_oracle_accuracy"]},
    ]


def _summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    mean_single_view = _mean_nested(rows, "single_view_text")
    mean_pairwise = _mean_nested(rows, "pairwise_structured_oracle")
    values = {
        "mean_candidate_only_accuracy": _mean_baseline(rows, "candidate_only_accuracy"),
        "mean_candidate_metadata_only_accuracy": _mean_baseline(rows, "candidate_metadata_only_accuracy"),
        "mean_role_pair_only_accuracy": _mean_baseline(rows, "role_pair_only_accuracy"),
        "mean_family_only_accuracy": _mean_baseline(rows, "family_only_accuracy"),
        "mean_view_masked_candidates_visible_accuracy": _mean_baseline(rows, "view_masked_candidates_visible_accuracy"),
        "mean_lexical_overlap_accuracy": _mean_baseline(rows, "lexical_overlap_accuracy"),
        "mean_static_frequency_accuracy": _mean_baseline(rows, "static_frequency_accuracy"),
        "mean_single_view_text": mean_single_view,
        "max_single_view_accuracy": max(mean_single_view.values() or [0.0]),
        "mean_pairwise_structured_oracle": mean_pairwise,
        "max_pairwise_structured_oracle_accuracy": max(mean_pairwise.values() or [0.0]),
        "mean_all_role_structured_oracle_accuracy": _mean_baseline(rows, "all_role_structured_oracle_accuracy"),
        "leakage_passes": all(bool(row["leakage"].get("passes", False)) for row in rows),
    }
    gate_table = [
        {"gate": "exact symbol/module/path/candidate leakage remains 0", "pass": values["leakage_passes"], "value": [row["leakage"] for row in rows]},
        {"gate": "candidate-only baseline <= 0.18", "pass": values["mean_candidate_only_accuracy"] <= STRICT_MAX, "preferred": values["mean_candidate_only_accuracy"] <= PREFERRED_MAX, "value": values["mean_candidate_only_accuracy"]},
        {"gate": "candidate-metadata-only baseline <= 0.18", "pass": values["mean_candidate_metadata_only_accuracy"] <= STRICT_MAX, "preferred": values["mean_candidate_metadata_only_accuracy"] <= PREFERRED_MAX, "value": values["mean_candidate_metadata_only_accuracy"]},
        {"gate": "role-pair-only baseline <= 0.18", "pass": values["mean_role_pair_only_accuracy"] <= STRICT_MAX, "preferred": values["mean_role_pair_only_accuracy"] <= PREFERRED_MAX, "value": values["mean_role_pair_only_accuracy"]},
        {"gate": "family-only baseline <= 0.18", "pass": values["mean_family_only_accuracy"] <= STRICT_MAX, "preferred": values["mean_family_only_accuracy"] <= PREFERRED_MAX, "value": values["mean_family_only_accuracy"]},
        {"gate": "view-masked + candidates-visible <= 0.18", "pass": values["mean_view_masked_candidates_visible_accuracy"] <= STRICT_MAX, "value": values["mean_view_masked_candidates_visible_accuracy"]},
        {"gate": "lexical-overlap baseline <= 0.18", "pass": values["mean_lexical_overlap_accuracy"] <= STRICT_MAX, "value": values["mean_lexical_overlap_accuracy"]},
        {"gate": "static frequency baseline <= 0.20", "pass": values["mean_static_frequency_accuracy"] <= STATIC_FREQUENCY_MAX, "value": values["mean_static_frequency_accuracy"]},
        {"gate": "single-view baselines <= 0.18", "pass": values["max_single_view_accuracy"] <= STRICT_MAX, "value": mean_single_view},
        {"gate": "at least one pairwise-role oracle >= 0.70", "pass": values["max_pairwise_structured_oracle_accuracy"] >= PAIRWISE_STRUCTURED_TARGET, "value": mean_pairwise},
        {"gate": "all-role structured oracle >= 0.90", "pass": values["mean_all_role_structured_oracle_accuracy"] >= ALL_ROLE_STRUCTURED_TARGET, "value": values["mean_all_role_structured_oracle_accuracy"]},
    ]
    return {
        **values,
        "gate_table": gate_table,
        "pretraining_dataset_validity_passed": all(bool(gate["pass"]) for gate in gate_table),
    }


def _mean_nested(rows: Sequence[Dict[str, object]], name: str) -> Dict[str, float]:
    keys = sorted({key for row in rows for key in dict(row["baselines"][name]).keys()})
    return {key: float(mean(float(row["baselines"][name][key]) for row in rows)) for key in keys}


def _mean_baseline(rows: Sequence[Dict[str, object]], name: str) -> float:
    values = [float(row["baselines"][name]) for row in rows]
    return float(mean(values)) if values else 0.0


def _accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(predictions.astype(np.int64) == labels.astype(np.int64))) if len(labels) else 0.0


if __name__ == "__main__":
    main()
