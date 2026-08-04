from __future__ import annotations

import argparse
import inspect
import json
import re
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from src.coordinators.mlp import MLPTrainingConfig
from src.datasets.multiview_code_patch_selection import (
    ATTRIBUTE_VALUE_PAIRS,
    BALANCED_34B_CANDIDATE_REPRESENTATION,
    BALANCED_34B_DATASET_SOURCE,
    Candidate,
    MultiViewTaskExample,
    View,
    apply_example_control,
    build_multiview_code_patch_splits,
    example_oracle_metadata,
    format_candidate_block,
)
from src.experiments.architecture_search import (
    PROPOSED_RAW,
    _agent_config,
    _candidate_specs,
    _coordinator_config,
    _training_config,
)
from src.experiments.real_shared_weight_latent_coordination import (
    BagOfWordsDiagnosticBaseline,
    FitResult,
    SharedClonedAgentSystem,
    TextOutputOnlyCoordinator,
    fit_latent_system,
    predict_latent_system,
    _candidate_feature_tensor,
    _text_tokens,
)
from src.experiments.run_stage3_gpu_hard_validation import ARCHITECTURE, _clear_cuda, _configure_cuda, _stage_from_config
from src.experiments.run_stage34_candidate_sanitization_diagnostics import (
    _accuracy,
    _structured_oracle_predictions,
    run_diagnostics,
    stage34_dataset_config,
)
from src.experiments.run_stage35_model_facing_learnability import (
    CompatibilityFeatureScorer,
    FullContextTinyTransformerControl,
    TfidfCandidateScorer,
    TinyCrossEncoderCandidateScorer,
    _stage34b_config,
)


DEFAULT_CONFIG = "configs/stage34_candidate_sanitization_cuda_smoke.json"
DEFAULT_OUTPUT = "results/stage38_schema_aware_controls.json"
DEFAULT_REPORT = "reports/STAGE38_SCHEMA_AWARE_CONTROLS.md"
NEAR_CHANCE_8WAY_MAX = 0.18
STATIC_FREQUENCY_MAX = 0.20
POSITIVE_CONTROL_TARGET = 0.50


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 3.8 schema-aware control redesign.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--device", default=None, help="Optional device override. Defaults to the config device.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.device is not None:
        config["device"] = str(args.device)
        if str(args.device) == "cpu":
            config["require_cuda"] = False
    result = run_stage38(config, config_path=config_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(result), encoding="utf-8")


def run_stage38(config: Dict[str, object], config_path: Path | None = None) -> Dict[str, object]:
    smoke_config = _stage38_smoke_config(config)
    device, hardware = _configure_cuda(smoke_config)
    stage = _stage_from_config(smoke_config)
    dataset_config = stage34_dataset_config(smoke_config)
    splits = build_multiview_code_patch_splits(dataset_config, seed=0, repo_root=Path("."))

    print("stage38: running schema-aware dataset shortcut diagnostics")
    dataset_diagnostics = run_diagnostics(smoke_config, config_path=config_path)
    schema_dataset_baselines = _schema_dataset_baselines(splits, smoke_config, device=device, seed=0)
    role_embedding_shuffle = _role_embedding_shuffle_reclassification()

    print("stage38: running model-facing positive controls")
    positive_controls = _positive_control_bridge(splits, smoke_config)

    print("stage38: fitting one-seed locked tiny-smoke models")
    models = _fit_tiny_models(stage, splits, device=device, seed=0)
    tiny_smoke = _tiny_smoke_schema_controls(models, splits["test"], seed=0)

    summary = _summary(
        dataset_diagnostics=dataset_diagnostics,
        schema_dataset_baselines=schema_dataset_baselines,
        positive_controls=positive_controls,
        tiny_smoke=tiny_smoke,
    )

    del models
    _clear_cuda()
    return {
        "metadata": {
            "stage": "stage3.8_schema_aware_controls",
            "created_at_utc": _now(),
            "config_path": str(config_path) if config_path else None,
            "dataset_source": BALANCED_34B_DATASET_SOURCE,
            "candidate_representation": BALANCED_34B_CANDIDATE_REPRESENTATION,
            "architecture": ARCHITECTURE,
            "architecture_changes": "none",
            "schema_policy": "role-specific schema fields are preserved by default and treated as legitimate program-analysis evidence",
            "scope": "one-seed tiny smoke plus dataset-only and cheap positive-control diagnostics; no full 10-seed validation",
            "full_10_seed_validation_run": False,
            "device": device,
            "hardware": hardware,
            "stage_config": asdict(stage),
        },
        "role_embedding_shuffle_reclassification": role_embedding_shuffle,
        "dataset_diagnostics": dataset_diagnostics,
        "schema_dataset_baselines": schema_dataset_baselines,
        "positive_control_bridge": positive_controls,
        "tiny_smoke": tiny_smoke,
        "summary": summary,
    }


def _stage38_smoke_config(config: Dict[str, object]) -> Dict[str, object]:
    out = _stage34b_config(config)
    out["stage"] = {
        **dict(out["stage"]),
        "name": "stage38_schema_aware_smoke",
        "n_train": 256,
        "n_dev": 128,
        "n_test": 256,
        "seeds": [0],
        "epochs": int(dict(out["stage"]).get("epochs", 3)),
        "patience": int(dict(out["stage"]).get("patience", 2)),
        "mixed_precision": dict(out["stage"]).get("mixed_precision", "bf16"),
    }
    out["dataset_config"] = {
        **dict(out.get("dataset_config", {})),
        "dataset_source": BALANCED_34B_DATASET_SOURCE,
        "generator_version": "real_import_restore_candidate_balanced_34b_stage38_schema_aware",
        "candidate_representation": BALANCED_34B_CANDIDATE_REPRESENTATION,
    }
    return out


def _role_embedding_shuffle_reclassification() -> Dict[str, object]:
    from src.experiments.real_shared_weight_latent_coordination import SharedClonedAgentSystem

    forward_source = inspect.getsource(SharedClonedAgentSystem.forward)
    return {
        "old_name": "role_labels_shuffled",
        "new_name": "role_embedding_shuffle",
        "primary_gate": False,
        "diagnostic_only": True,
        "expected_behavior": "may not collapse when private view schema text exposes role identity",
        "explanation": (
            "Role schema is legitimate evidence for schema-aware program-analysis agents. "
            "Shuffling runtime role embeddings alone is not a valid corruption when role identity is recoverable from the view schema."
        ),
        "implementation": {
            "shuffles_runtime_role_ids": "condition == \"role_labels_shuffled\"" in forward_source,
            "does_not_change_view_text": True,
            "does_not_change_candidate_set": True,
        },
    }


def _positive_control_bridge(
    splits: Dict[str, Sequence[MultiViewTaskExample]],
    config: Dict[str, object],
) -> Dict[str, object]:
    training = MLPTrainingConfig(epochs=5, batch_size=32, lr=0.002, weight_decay=0.0001, patience=3, hidden_dims=(64,))
    controls = {
        "candidate_pair_compatibility_mlp": CompatibilityFeatureScorer(training=training, seed=14),
        "small_cross_encoder_all_views_candidate": TinyCrossEncoderCandidateScorer(
            text_builder=lambda example, index: _all_views_text(example) + "\n" + example.candidates[index].text,
            seed=13,
            epochs=4,
        ),
        "all_view_tfidf_logistic_candidate_scorer": TfidfCandidateScorer(
            text_builder=lambda example, index: _all_views_text(example) + "\n" + example.candidates[index].text,
            max_features=512,
            training=MLPTrainingConfig(epochs=8, batch_size=32, lr=0.01, weight_decay=0.0001, patience=3, hidden_dims=()),
            seed=12,
        ),
        "single_agent_full_context_tiny_transformer": FullContextTinyTransformerControl(config=config, seed=15),
    }
    rows = {}
    for name, control in controls.items():
        control.fit(splits["train"], splits["dev"])
        rows[name] = {
            split: _accuracy(control.predict(examples), _labels(examples))
            for split, examples in splits.items()
        }
    best = max(float(row["test"]) for row in rows.values())
    return {
        "rows": rows,
        "summary": {
            "best_test_accuracy": best,
            "at_least_one_model_facing_positive_control_ge_0_50": bool(best >= POSITIVE_CONTROL_TARGET),
            "target": POSITIVE_CONTROL_TARGET,
        },
    }


def _schema_dataset_baselines(
    splits: Dict[str, Sequence[MultiViewTaskExample]],
    config: Dict[str, object],
    device: str,
    seed: int,
) -> Dict[str, object]:
    stage = _stage_from_config(config)
    training = MLPTrainingConfig(epochs=4, batch_size=stage.batch_size, lr=0.002, weight_decay=0.0001, patience=2, hidden_dims=(32,))
    variants = {
        "schema_only_baseline": {
            split: _schema_only(rows)
            for split, rows in splits.items()
        },
        "null_evidence_values_baseline": {
            split: _null_evidence_values(rows)
            for split, rows in splits.items()
        },
        "candidate_only_baseline": {
            split: _candidate_only(rows)
            for split, rows in splits.items()
        },
        "evidence_only_no_candidates_baseline": {
            split: _evidence_only_no_candidates(rows)
            for split, rows in splits.items()
        },
    }
    rows = {}
    for index, (name, variant_splits) in enumerate(variants.items()):
        baseline = BagOfWordsDiagnosticBaseline(
            method=f"stage38_{name}",
            num_classes=8,
            training=training,
            seed=seed + 91_000 + index,
            device=device,
            text_builder=lambda example: _all_views_text(example) + "\n" + format_candidate_block(example),
            feature_dim=128,
        )
        baseline.fit(variant_splits["train"], variant_splits["dev"])
        rows[name] = {
            split: _accuracy(baseline.predict(examples), _labels(examples))
            for split, examples in variant_splits.items()
        }
    return {
        "rows": rows,
        "summary": {
            "candidate_only_near_chance": rows["candidate_only_baseline"]["test"] <= NEAR_CHANCE_8WAY_MAX,
            "schema_only_near_chance": rows["schema_only_baseline"]["test"] <= NEAR_CHANCE_8WAY_MAX,
            "null_evidence_values_near_chance": rows["null_evidence_values_baseline"]["test"] <= NEAR_CHANCE_8WAY_MAX,
            "evidence_only_no_candidates_near_chance": rows["evidence_only_no_candidates_baseline"]["test"] <= NEAR_CHANCE_8WAY_MAX,
        },
    }


def _fit_tiny_models(stage, splits: Dict[str, Sequence[MultiViewTaskExample]], device: str, seed: int) -> Dict[str, object]:
    candidate = next(candidate for candidate in _candidate_specs() if candidate.name == ARCHITECTURE)
    agent_config = _agent_config(stage)
    training = _training_config(stage)
    print("stage38: fitting trainable locked latent")
    trainable = fit_latent_system(
        train_examples=splits["train"],
        dev_examples=splits["dev"],
        agent_config=agent_config,
        coordinator_config=candidate.coordinator_config,
        training_config=training,
        num_classes=8,
        seed=seed + 101,
        device=device,
        trainable_agent=True,
        method=f"stage38_trainable__{candidate.name}",
        message_config=candidate.message_config,
    )
    print("stage38: fitting frozen locked latent")
    frozen = fit_latent_system(
        train_examples=splits["train"],
        dev_examples=splits["dev"],
        agent_config=agent_config,
        coordinator_config=candidate.coordinator_config,
        training_config=training,
        num_classes=8,
        seed=seed + 201,
        device=device,
        trainable_agent=False,
        method=f"stage38_frozen__{candidate.name}",
        message_config=candidate.message_config,
    )
    print("stage38: fitting raw latent baseline")
    raw = fit_latent_system(
        train_examples=splits["train"],
        dev_examples=splits["dev"],
        agent_config=agent_config,
        coordinator_config=replace(_coordinator_config(stage, family="cross_attention", num_layers=1, dropout=0.0), input_dim=agent_config.hidden_dim),
        training_config=training,
        num_classes=8,
        seed=seed + 11_000,
        device=device,
        trainable_agent=True,
        method=PROPOSED_RAW,
    )
    print("stage38: fitting text-only baseline")
    text = TextOutputOnlyCoordinator(
        num_classes=8,
        training=MLPTrainingConfig(
            epochs=max(3, int(stage.epochs)),
            batch_size=int(stage.batch_size),
            lr=float(stage.lr),
            weight_decay=0.0001,
            patience=max(2, int(stage.patience)),
            hidden_dims=(32,),
        ),
        seed=seed + 12_000,
        device=device,
        feature_dim=128,
    )
    text.fit(splits["train"], splits["dev"])
    return {"trainable": trainable, "frozen": frozen, "raw_latent": raw, "text_only": text}


def _tiny_smoke_schema_controls(
    models: Dict[str, object],
    examples: Sequence[MultiViewTaskExample],
    seed: int,
) -> Dict[str, object]:
    variants = {
        "none": list(examples),
        "candidate_only": _candidate_only(examples),
        "view_masked_candidates_visible": apply_example_control(examples, "view_masked", seed=seed + 11_000),
        "schema_only": _schema_only(examples),
        "value_shuffle_within_schema": _value_shuffle_within_schema(examples, seed + 12_000),
        "cross_example_view_bundle_shuffle": _cross_example_view_bundle_shuffle(examples, seed + 13_000),
        "candidate_evidence_mismatch": _candidate_evidence_mismatch(examples, seed + 14_000),
        "schema_preserved_role_value_shuffle": _schema_preserved_role_value_shuffle(examples, seed + 15_000),
        "null_evidence_values": _null_evidence_values(examples),
        "evidence_only_no_candidates": _evidence_only_no_candidates(examples),
        "role_embedding_shuffle_diagnostic": list(examples),
    }
    rows = []
    for name, rows_in in variants.items():
        condition = "role_labels_shuffled" if name == "role_embedding_shuffle_diagnostic" else "none"
        zero_roles = name == "candidate_only"
        rows.append(_eval_tiny_variant(name, models, rows_in, condition=condition, seed=seed, zero_roles=zero_roles))
    by_name = {row["control"]: row for row in rows}
    clean = by_name["none"]["accuracy"]
    primary = {
        "trainable": clean["trainable"],
        "frozen": clean["frozen"],
        "text_only": clean["text_only"],
        "raw_latent": clean["raw_latent"],
        "candidate_only": by_name["candidate_only"]["accuracy"]["trainable"],
        "schema_only": by_name["schema_only"]["accuracy"]["trainable"],
        "value_shuffle_within_schema": by_name["value_shuffle_within_schema"]["accuracy"]["trainable"],
        "cross_example_view_bundle_shuffle": by_name["cross_example_view_bundle_shuffle"]["accuracy"]["trainable"],
        "candidate_evidence_mismatch": by_name["candidate_evidence_mismatch"]["accuracy"]["trainable"],
        "schema_preserved_role_value_shuffle": by_name["schema_preserved_role_value_shuffle"]["accuracy"]["trainable"],
        "null_evidence_values": by_name["null_evidence_values"]["accuracy"]["trainable"],
        "evidence_only_no_candidates": by_name["evidence_only_no_candidates"]["accuracy"]["trainable"],
        "oracle": _accuracy(_structured_oracle_predictions(examples, [0, 1, 2, 3]), _labels(examples)),
    }
    return {
        "seed": int(seed),
        "split": "test",
        "primary_accuracy": primary,
        "control_rows": rows,
        "summary": {
            "trainable_above_chance": primary["trainable"] > NEAR_CHANCE_8WAY_MAX,
            "trainable_beats_frozen": primary["trainable"] > primary["frozen"],
            "trainable_beats_text_raw": primary["trainable"] > max(primary["text_only"], primary["raw_latent"]),
            "candidate_only_collapsed": primary["candidate_only"] <= NEAR_CHANCE_8WAY_MAX,
            "schema_only_collapsed": primary["schema_only"] <= NEAR_CHANCE_8WAY_MAX,
            "value_shuffle_collapsed": primary["value_shuffle_within_schema"] <= NEAR_CHANCE_8WAY_MAX,
            "cross_example_bundle_shuffle_collapsed": primary["cross_example_view_bundle_shuffle"] <= NEAR_CHANCE_8WAY_MAX,
            "candidate_evidence_mismatch_collapsed": primary["candidate_evidence_mismatch"] <= NEAR_CHANCE_8WAY_MAX,
            "schema_preserved_role_value_shuffle_collapsed": primary["schema_preserved_role_value_shuffle"] <= NEAR_CHANCE_8WAY_MAX,
            "null_evidence_values_collapsed": primary["null_evidence_values"] <= NEAR_CHANCE_8WAY_MAX,
            "evidence_only_no_candidates_collapsed": primary["evidence_only_no_candidates"] <= NEAR_CHANCE_8WAY_MAX,
            "oracle_high": primary["oracle"] >= 0.90,
        },
    }


def _eval_tiny_variant(
    name: str,
    models: Dict[str, object],
    examples: Sequence[MultiViewTaskExample],
    condition: str,
    seed: int,
    zero_roles: bool = False,
) -> Dict[str, object]:
    labels = _labels(examples)
    accuracies = {}
    for model_name in ("trainable", "frozen", "raw_latent"):
        result = models[model_name]
        if zero_roles:
            with _zero_role_embeddings(result.system):
                pred = predict_latent_system(result, examples, condition, seed)
        else:
            pred = predict_latent_system(result, examples, condition, seed)
        accuracies[model_name] = _accuracy(pred, labels)
    accuracies["text_only"] = _accuracy(models["text_only"].predict(examples), labels)
    return {
        "control": name,
        "condition": condition,
        "zero_role_embeddings": bool(zero_roles),
        "accuracy": accuracies,
        "trainable_near_chance": bool(accuracies["trainable"] <= NEAR_CHANCE_8WAY_MAX),
    }


@contextmanager
def _zero_role_embeddings(system: SharedClonedAgentSystem):
    saved = []
    try:
        with torch.no_grad():
            for module in (system.active_message_readout, system.coordinator):
                embedding = getattr(module, "role_embedding", None) if module is not None else None
                if embedding is not None:
                    saved.append((embedding, embedding.weight.detach().clone()))
                    embedding.weight.zero_()
        yield
    finally:
        with torch.no_grad():
            for embedding, weight in saved:
                embedding.weight.copy_(weight)


def _value_shuffle_within_schema(examples: Sequence[MultiViewTaskExample], seed: int) -> List[MultiViewTaskExample]:
    rng = np.random.default_rng(seed + 101_000)
    out_views = [list(example.views) for example in examples]
    n_roles = len(examples[0].views)
    for role_index in range(n_roles):
        perm = _non_identity_permutation(len(examples), rng)
        for row_index, donor_index in enumerate(perm):
            original = examples[row_index].views[role_index]
            donor = examples[int(donor_index)].views[role_index]
            value = _extract_evidence_value(donor.text, role_index)
            out_views[row_index][role_index] = replace(
                original,
                text=_replace_evidence_value(original.text, role_index, value),
                source_type=f"{original.source_type}:stage38_value_shuffle_within_schema",
            )
    return [replace(example, views=tuple(out_views[index])) for index, example in enumerate(examples)]


def _schema_preserved_role_value_shuffle(examples: Sequence[MultiViewTaskExample], seed: int) -> List[MultiViewTaskExample]:
    rng = np.random.default_rng(seed + 102_000)
    out_views = [list(example.views) for example in examples]
    n_roles = len(examples[0].views)
    for role_index in range(n_roles):
        values = [_extract_evidence_value(example.views[role_index].text, role_index) for example in examples]
        perm = _non_identity_permutation(len(values), rng)
        for row_index, donor_index in enumerate(perm):
            original = examples[row_index].views[role_index]
            out_views[row_index][role_index] = replace(
                original,
                text=_replace_evidence_value(original.text, role_index, values[int(donor_index)]),
                source_type=f"{original.source_type}:stage38_schema_preserved_value_shuffle",
            )
    return [replace(example, views=tuple(out_views[index])) for index, example in enumerate(examples)]


def _cross_example_view_bundle_shuffle(examples: Sequence[MultiViewTaskExample], seed: int) -> List[MultiViewTaskExample]:
    rng = np.random.default_rng(seed + 103_000)
    perm = _non_identity_permutation(len(examples), rng)
    out = []
    for row_index, donor_index in enumerate(perm):
        donor_views = tuple(
            replace(view, source_type=f"{view.source_type}:stage38_bundle_donor")
            for view in examples[int(donor_index)].views
        )
        out.append(replace(examples[row_index], views=donor_views))
    return out


def _candidate_evidence_mismatch(examples: Sequence[MultiViewTaskExample], seed: int) -> List[MultiViewTaskExample]:
    rng = np.random.default_rng(seed + 104_000)
    perm = _non_identity_permutation(len(examples), rng)
    labels = _balanced_random_labels(len(examples), 8, seed + 104_500)
    out = []
    for row_index, donor_index in enumerate(perm):
        donor = examples[int(donor_index)]
        out.append(
            replace(
                examples[row_index],
                candidates=tuple(
                    replace(candidate, candidate_id=f"{examples[row_index].id}-mismatch-{slot}")
                    for slot, candidate in enumerate(donor.candidates)
                ),
                label=int(labels[row_index]),
                metadata={**examples[row_index].metadata, "stage38_candidate_evidence_mismatch": True},
            )
        )
    return out


def _null_evidence_values(examples: Sequence[MultiViewTaskExample]) -> List[MultiViewTaskExample]:
    out = []
    for example in examples:
        views = tuple(
            replace(
                view,
                text=_replace_evidence_value(view.text, role_index, "MASK"),
                source_type=f"{view.source_type}:stage38_null_evidence_value",
            )
            for role_index, view in enumerate(example.views)
        )
        out.append(replace(example, views=views))
    return out


def _schema_only(examples: Sequence[MultiViewTaskExample]) -> List[MultiViewTaskExample]:
    out = []
    for example in examples:
        views = []
        for role_index, view in enumerate(example.views):
            field = _evidence_field_name(role_index, view.text)
            views.append(
                replace(
                    view,
                    text=(
                        "Redacted compatibility artifact:\n"
                        f"{field}: VALUE_REMOVED\n"
                        "exact_symbol_module_path: VALUE_REMOVED"
                    ),
                    source_type=f"{view.source_type}:stage38_schema_only",
                )
            )
        out.append(replace(example, views=tuple(views)))
    return out


def _candidate_only(examples: Sequence[MultiViewTaskExample]) -> List[MultiViewTaskExample]:
    out = []
    for example in examples:
        views = tuple(
            replace(
                view,
                text=(
                    "Program-analysis artifact hidden for candidate-only control.\n"
                    "No schema field or evidence value is available."
                ),
                source_type=f"{view.source_type}:stage38_candidate_only_view_hidden",
            )
            for view in example.views
        )
        out.append(replace(example, views=views))
    return out


def _evidence_only_no_candidates(examples: Sequence[MultiViewTaskExample]) -> List[MultiViewTaskExample]:
    out = []
    for example in examples:
        candidates = tuple(
            Candidate(
                candidate_id=f"{example.id}-candidate-hidden-{index}",
                text="Patch candidate hidden for evidence-only control.",
                source_path="REDACTED_CANDIDATE",
                patch_hash=f"stage38-evidence-only-{example.id}-{index}",
                attributes=("", "", "", ""),
            )
            for index, _candidate in enumerate(example.candidates)
        )
        out.append(replace(example, candidates=candidates))
    return out


def _extract_evidence_value(text: str, role_index: int) -> str:
    field = _evidence_field_name(role_index, text)
    for line in str(text).splitlines():
        if line.strip().lower().startswith(f"{field.lower()}:"):
            return line.split(":", 1)[1].strip()
    for line in str(text).splitlines():
        if ":" in line and "withheld" not in line.lower() and "artifact" not in line.lower():
            return line.split(":", 1)[1].strip()
    return "MASK"


def _replace_evidence_value(text: str, role_index: int, value: str) -> str:
    field = _evidence_field_name(role_index, text)
    rows = []
    replaced = False
    for line in str(text).splitlines():
        if line.strip().lower().startswith(f"{field.lower()}:"):
            rows.append(f"{field}: {value}")
            replaced = True
        else:
            rows.append(line)
    if not replaced:
        rows.insert(1, f"{field}: {value}")
    return "\n".join(rows)


def _evidence_field_name(role_index: int, text: str) -> str:
    for line in str(text).splitlines():
        if ":" not in line:
            continue
        field = line.split(":", 1)[0].strip()
        if field and field.lower() not in {"redacted compatibility artifact", "exact_symbol_module_path"}:
            return field
    fallback = ("symbol_surface", "provider_area", "provider_name_shape", "import_slot_shape")
    return fallback[int(role_index)] if int(role_index) < len(fallback) else f"role_{role_index}_value"


def _all_views_text(example: MultiViewTaskExample) -> str:
    return "\n".join(view.text for view in example.views)


def _labels(examples: Sequence[MultiViewTaskExample]) -> np.ndarray:
    return np.asarray([example.label for example in examples], dtype=np.int64)


def _balanced_random_labels(n: int, num_classes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 105_000)
    labels = np.tile(np.arange(num_classes, dtype=np.int64), int(np.ceil(n / num_classes)))[:n].copy()
    rng.shuffle(labels)
    return labels


def _non_identity_permutation(n: int, rng: np.random.Generator) -> np.ndarray:
    perm = rng.permutation(n)
    if n > 1 and np.array_equal(perm, np.arange(n)):
        perm = np.roll(perm, 1)
    return perm.astype(np.int64, copy=False)


def _summary(
    dataset_diagnostics: Dict[str, object],
    schema_dataset_baselines: Dict[str, object],
    positive_controls: Dict[str, object],
    tiny_smoke: Dict[str, object],
) -> Dict[str, object]:
    diag = dataset_diagnostics.get("summary", {})
    tiny = tiny_smoke["primary_accuracy"]
    tiny_summary = tiny_smoke["summary"]
    schema_base = schema_dataset_baselines["summary"]
    positive_summary = positive_controls["summary"]
    shortcut_passes = bool(
        float(diag.get("mean_candidate_only_accuracy", 1.0)) <= NEAR_CHANCE_8WAY_MAX
        and float(diag.get("mean_candidate_metadata_only_accuracy", 1.0)) <= NEAR_CHANCE_8WAY_MAX
        and float(diag.get("mean_view_masked_candidates_visible_accuracy", 1.0)) <= NEAR_CHANCE_8WAY_MAX
        and float(diag.get("mean_role_pair_only_accuracy", 1.0)) <= NEAR_CHANCE_8WAY_MAX
        and float(diag.get("mean_lexical_overlap_accuracy", 1.0)) <= NEAR_CHANCE_8WAY_MAX
        and float(diag.get("mean_static_frequency_accuracy", 1.0)) <= STATIC_FREQUENCY_MAX
        and bool(schema_base.get("schema_only_near_chance", False))
    )
    corruption_controls_collapse = bool(
        tiny_summary["candidate_only_collapsed"]
        and tiny_summary["schema_only_collapsed"]
        and tiny_summary["value_shuffle_collapsed"]
        and tiny_summary["cross_example_bundle_shuffle_collapsed"]
        and tiny_summary["candidate_evidence_mismatch_collapsed"]
        and tiny_summary["schema_preserved_role_value_shuffle_collapsed"]
        and tiny_summary["null_evidence_values_collapsed"]
        and tiny_summary["evidence_only_no_candidates_collapsed"]
    )
    full_validation_justified = bool(
        tiny_summary["trainable_above_chance"]
        and tiny_summary["trainable_beats_frozen"]
        and tiny_summary["trainable_beats_text_raw"]
        and corruption_controls_collapse
        and bool(positive_summary["at_least_one_model_facing_positive_control_ge_0_50"])
        and shortcut_passes
        and tiny_summary["oracle_high"]
    )
    return {
        "role_specific_schemas_treated_as_legitimate_evidence": True,
        "role_embedding_shuffle_primary_gate": False,
        "replacement_primary_controls": [
            "value_shuffle_within_schema",
            "cross_example_view_bundle_shuffle",
            "candidate_evidence_mismatch",
            "schema_preserved_role_value_shuffle",
            "null_evidence_values",
            "schema_only",
            "evidence_only_no_candidates",
            "candidate_only",
        ],
        "positive_control_learnability_confirmed": bool(positive_summary["at_least_one_model_facing_positive_control_ge_0_50"]),
        "best_positive_control_test_accuracy": float(positive_summary["best_test_accuracy"]),
        "shortcut_diagnostics_pass": shortcut_passes,
        "schema_aware_corruption_controls_collapse": corruption_controls_collapse,
        "tiny_smoke_trainable_accuracy": float(tiny["trainable"]),
        "tiny_smoke_frozen_accuracy": float(tiny["frozen"]),
        "full_validation_justified": full_validation_justified,
        "validation_decision": "justified" if full_validation_justified else "not justified",
    }


def _render_report(result: Dict[str, object]) -> str:
    summary = result["summary"]
    diag = result["dataset_diagnostics"]["summary"]
    schema_base = result["schema_dataset_baselines"]["rows"]
    positive = result["positive_control_bridge"]["rows"]
    tiny = result["tiny_smoke"]
    lines = [
        "# Stage 3.8 Schema-Aware Controls",
        "",
        "## Scope",
        "",
        "- Full 10-seed validation: not run.",
        "- Locked architecture changes: none.",
        "- Default view format: Stage 3.4b schema-aware `balanced_categories_v3`; role/view schema fields are preserved.",
        "",
        "## Role Schema Policy",
        "",
        "- Role-specific schemas are treated as legitimate program-analysis evidence: `True`.",
        "- Simple runtime role-id shuffling is reclassified as `role_embedding_shuffle`, diagnostic only.",
        "- Expected behavior: it may not collapse when view text itself exposes role identity through schema.",
        "",
        "## Replacement Controls",
        "",
        "| control | corruption target |",
        "|---|---|",
        "| value_shuffle_within_schema | Preserve each role schema and candidate list; shuffle evidence values across examples within the same role. |",
        "| cross_example_view_bundle_shuffle | Preserve all role schemas; replace the full multi-view evidence bundle with another example's bundle. |",
        "| candidate_evidence_mismatch | Preserve the view bundle; replace candidates from another example and randomize labels. |",
        "| schema_preserved_role_value_shuffle | Preserve role labels and field names; shuffle content values inside each role. |",
        "| null_evidence_values | Preserve schemas and field names; replace evidence values with `MASK`. |",
        "| schema_only | Preserve schemas and field names; remove evidence values while keeping candidates visible. |",
        "| evidence_only_no_candidates | Preserve evidence views; hide candidate identities and representations. |",
        "| candidate_only | Preserve candidate representations; hide private views. |",
        "",
        "## Dataset Shortcut Gates",
        "",
        "| diagnostic | accuracy | gate |",
        "|---|---:|---|",
    ]
    gate_rows = [
        ("candidate_only", diag.get("mean_candidate_only_accuracy"), NEAR_CHANCE_8WAY_MAX),
        ("candidate_metadata_only", diag.get("mean_candidate_metadata_only_accuracy"), NEAR_CHANCE_8WAY_MAX),
        ("view_masked_candidates_visible", diag.get("mean_view_masked_candidates_visible_accuracy"), NEAR_CHANCE_8WAY_MAX),
        ("role_pair_only", diag.get("mean_role_pair_only_accuracy"), NEAR_CHANCE_8WAY_MAX),
        ("lexical_overlap", diag.get("mean_lexical_overlap_accuracy"), NEAR_CHANCE_8WAY_MAX),
        ("static_frequency", diag.get("mean_static_frequency_accuracy"), STATIC_FREQUENCY_MAX),
        ("all_role_oracle", diag.get("mean_all_role_structured_oracle_accuracy"), 0.90),
    ]
    for name, value, gate in gate_rows:
        lines.append(f"| {name} | {float(value):.4f} | `{gate:.2f}` |")
    lines.extend(
        [
            "",
            "## Schema-Only Dataset Baselines",
            "",
            "| baseline | train | dev | test |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, row in schema_base.items():
        lines.append(f"| {name} | {float(row['train']):.4f} | {float(row['dev']):.4f} | {float(row['test']):.4f} |")
    lines.extend(
        [
            "",
            "## Positive-Control Bridge",
            "",
            "| control | train | dev | test |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, row in positive.items():
        lines.append(f"| {name} | {float(row['train']):.4f} | {float(row['dev']):.4f} | {float(row['test']):.4f} |")
    lines.extend(
        [
            "",
            f"- Best positive-control test accuracy: `{float(summary['best_positive_control_test_accuracy']):.4f}`",
            f"- Positive-control learnability confirmed: `{bool(summary['positive_control_learnability_confirmed'])}`",
            "",
            "## Tiny Smoke",
            "",
            "| metric | accuracy |",
            "|---|---:|",
        ]
    )
    for name, value in tiny["primary_accuracy"].items():
        lines.append(f"| {name} | {float(value):.4f} |")
    lines.extend(
        [
            "",
            "## Control Rows",
            "",
            "| control | trainable | frozen | text | raw | trainable near chance? |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in tiny["control_rows"]:
        acc = row["accuracy"]
        lines.append(
            "| {name} | {trainable:.4f} | {frozen:.4f} | {text:.4f} | {raw:.4f} | `{near}` |".format(
                name=row["control"],
                trainable=float(acc["trainable"]),
                frozen=float(acc["frozen"]),
                text=float(acc["text_only"]),
                raw=float(acc["raw_latent"]),
                near=bool(row["trainable_near_chance"]),
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Schema-aware corruption controls collapse: `{bool(summary['schema_aware_corruption_controls_collapse'])}`",
            f"- Shortcut diagnostics pass: `{bool(summary['shortcut_diagnostics_pass'])}`",
            f"- Full validation justified: `{bool(summary['full_validation_justified'])}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
