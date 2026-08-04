from __future__ import annotations

import argparse
import inspect
import json
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from src.coordinators.mlp import MLPClassifier, MLPTrainingConfig
from src.datasets.multiview_code_patch_selection import (
    ATTRIBUTE_VALUE_PAIRS,
    BALANCED_34B_CANDIDATE_REPRESENTATION,
    BALANCED_34B_DATASET_SOURCE,
    ROLE_NAMES,
    MultiViewTaskExample,
    View,
    apply_example_control,
    build_multiview_code_patch_splits,
    example_oracle_metadata,
)
from src.experiments.architecture_search import (
    PROPOSED_RAW,
    _agent_config,
    _candidate_specs,
    _coordinator_config,
    _training_config,
)
from src.experiments.real_shared_weight_latent_coordination import (
    FitResult,
    SharedClonedAgentSystem,
    TextOutputOnlyCoordinator,
    fit_latent_system,
    predict_latent_system,
    _candidate_feature_tensor,
    _text_tokens,
)
from src.experiments.run_stage3_gpu_hard_validation import ARCHITECTURE, _clear_cuda, _configure_cuda, _stage_from_config
from src.experiments.run_stage34_candidate_sanitization_diagnostics import _accuracy, _structured_oracle_predictions, stage34_dataset_config
from src.experiments.run_stage35_model_facing_learnability import _stage34b_config


DEFAULT_CONFIG = "configs/stage34_candidate_sanitization_cuda_smoke.json"
DEFAULT_OUTPUT = "results/stage37_role_control_diagnosis.json"
DEFAULT_REPORT = "reports/STAGE37_ROLE_CONTROL_DIAGNOSIS.md"
NEAR_CHANCE_8WAY_MAX = 0.18
ROLE_ID_LEAK_HIGH = 0.80


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 3.7 role-control diagnosis.")
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
    result = run_stage37(config, config_path=config_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(result), encoding="utf-8")


def run_stage37(config: Dict[str, object], config_path: Path | None = None) -> Dict[str, object]:
    smoke_config = _stage37_smoke_config(config)
    device, hardware = _configure_cuda(smoke_config)
    stage = _stage_from_config(smoke_config)
    dataset_config = stage34_dataset_config(smoke_config)
    splits = build_multiview_code_patch_splits(dataset_config, seed=0, repo_root=Path("."))

    implementation = _role_shuffle_implementation_audit()
    leakage = _view_format_leakage_audit(splits, device=device, seed=10_000)
    shortcuts = _role_pair_shortcut_audit(splits)

    print("stage37: fitting original one-seed tiny-smoke models")
    original_models = _fit_tiny_models(stage, splits, device=device, seed=0, label="original")
    original_tiny = _tiny_smoke_metrics(original_models, splits["test"], seed=0)
    ablations = _role_ablation_suite(original_models, splits["test"], seed=0)

    role_id_leaks = _role_id_leaks_from_view_format(leakage)
    normalized_splits = {split: _normalize_view_format(rows) for split, rows in splits.items()}
    normalized_leakage = _view_format_leakage_audit(normalized_splits, device=device, seed=20_000)
    normalized_shortcuts = _role_pair_shortcut_audit(normalized_splits)
    normalized_checkpoint_eval = _evaluate_normalized_view_format(original_models, normalized_splits["test"], seed=0)

    fixed_tiny = None
    fixed_ablations = None
    if role_id_leaks:
        print("stage37: fitting normalized-view one-seed tiny-smoke models")
        fixed_models = _fit_tiny_models(stage, normalized_splits, device=device, seed=0, label="normalized_view_format")
        fixed_tiny = _tiny_smoke_metrics(fixed_models, normalized_splits["test"], seed=0)
        fixed_ablations = _role_ablation_suite(fixed_models, normalized_splits["test"], seed=0)
        del fixed_models
        _clear_cuda()

    summary = _summary(
        implementation=implementation,
        leakage=leakage,
        shortcuts=shortcuts,
        original_tiny=original_tiny,
        ablations=ablations,
        normalized_leakage=normalized_leakage,
        fixed_tiny=fixed_tiny,
    )

    del original_models
    _clear_cuda()
    return {
        "metadata": {
            "stage": "stage3.7_role_control_diagnosis",
            "created_at_utc": _now(),
            "config_path": str(config_path) if config_path else None,
            "dataset_source": BALANCED_34B_DATASET_SOURCE,
            "candidate_representation": BALANCED_34B_CANDIDATE_REPRESENTATION,
            "architecture": ARCHITECTURE,
            "architecture_changes": "none",
            "scope": "one-seed tiny-smoke fitting plus cheap role-control and dataset-only diagnostics; no full 10-seed validation",
            "checkpoint_source": "no Stage 3.6 tiny-smoke checkpoint artifact was available in results; reran only the one-seed tiny smoke",
            "device": device,
            "hardware": hardware,
            "stage_config": asdict(stage),
            "full_10_seed_validation_run": False,
        },
        "role_shuffle_implementation": implementation,
        "original_tiny_smoke": original_tiny,
        "role_ablation_evaluations": ablations,
        "view_format_leakage_audit": leakage,
        "role_pair_shortcut_audit": shortcuts,
        "normalized_view_format_variant": {
            "implemented": bool(role_id_leaks),
            "implementation": "Stage 3.7 transient data transform: private view text uses identical field names and generic FEATURE_ALPHA/BETA values for every role; locked architecture unchanged.",
            "checkpoint_only_eval_original_models": normalized_checkpoint_eval,
            "view_format_leakage_audit": normalized_leakage,
            "role_pair_shortcut_audit": normalized_shortcuts,
            "tiny_smoke_after_fix": fixed_tiny,
            "role_ablation_evaluations_after_fix": fixed_ablations,
        },
        "summary": summary,
    }


def _stage37_smoke_config(config: Dict[str, object]) -> Dict[str, object]:
    out = _stage34b_config(config)
    out["stage"] = {
        **dict(out["stage"]),
        "name": "stage37_smoke",
        "n_train": 256,
        "n_dev": 128,
        "n_test": 256,
        "seeds": [0],
        "epochs": int(dict(out["stage"]).get("epochs", 3)),
        "patience": int(dict(out["stage"]).get("patience", 2)),
    }
    out["dataset_config"] = {
        **dict(out.get("dataset_config", {})),
        "dataset_source": BALANCED_34B_DATASET_SOURCE,
        "generator_version": "real_import_restore_candidate_balanced_34b_stage37",
        "candidate_representation": BALANCED_34B_CANDIDATE_REPRESENTATION,
    }
    return out


def _role_shuffle_implementation_audit() -> Dict[str, object]:
    from src.experiments.real_shared_weight_latent_coordination import SharedClonedAgentSystem

    apply_source = inspect.getsource(apply_example_control)
    forward_source = inspect.getsource(SharedClonedAgentSystem.forward)
    collect_source = inspect.getsource(SharedClonedAgentSystem.collect_clone_representations)
    return {
        "answers": {
            "1_shuffles_role_embeddings_only": True,
            "2_shuffles_role_text_labels": False,
            "3_shuffles_private_views": False,
            "4_shuffles_role_view_pairing": (
                "yes for runtime role_id embeddings versus clone activations; no for dataclass view.role/text pairing and no physical view reordering"
            ),
            "5_shuffle_scope": "per-example row-wise permutation inside each forward batch; predict_latent_system reuses the same seed for each batch",
            "6_same_permutation_train_dev_test": (
                "not inherently; permutations are deterministic from the seed passed to forward. Existing dev/test controls use split-specific seeds; Stage 3.6 tiny smoke evaluated test only."
            ),
            "7_corrupted_train_or_test_time": "evaluation/control time only for role_labels_shuffled; normal trainable/frozen fitting uses condition='none'",
        },
        "mechanics": {
            "apply_example_control_returns_examples_unchanged": "role_labels_shuffled" in apply_source and "return list(examples)" in apply_source,
            "forward_shuffles_role_ids": "condition == \"role_labels_shuffled\"" in forward_source and "role_ids = base_role_ids.gather" in forward_source,
            "collect_uses_format_clone_prompt_without_view_role_label": "format_clone_prompt(example, role_index)" in collect_source,
            "role_text_fields_mutated": False,
            "private_view_text_mutated": False,
            "candidate_features_mutated": False,
        },
        "source_excerpts": {
            "apply_example_control": _source_excerpt(apply_source, ["role_labels_shuffled", "return list(examples)"]),
            "SharedClonedAgentSystem.forward": _source_excerpt(forward_source, ["role_labels_shuffled", "role_ids", "_role_permutation"]),
        },
    }


def _fit_tiny_models(stage, splits: Dict[str, Sequence[MultiViewTaskExample]], device: str, seed: int, label: str) -> Dict[str, object]:
    candidate = next(candidate for candidate in _candidate_specs() if candidate.name == ARCHITECTURE)
    agent_config = _agent_config(stage)
    training = _training_config(stage)
    print(f"stage37 {label}: fitting trainable locked latent")
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
        method=f"stage37_{label}_trainable__{candidate.name}",
        message_config=candidate.message_config,
    )
    print(f"stage37 {label}: fitting frozen locked latent")
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
        method=f"stage37_{label}_frozen__{candidate.name}",
        message_config=candidate.message_config,
    )
    print(f"stage37 {label}: fitting raw latent baseline")
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
    print(f"stage37 {label}: fitting text-only baseline")
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
    return {
        "trainable": trainable,
        "frozen": frozen,
        "raw_latent": raw,
        "text_only": text,
    }


def _tiny_smoke_metrics(models: Dict[str, object], examples: Sequence[MultiViewTaskExample], seed: int) -> Dict[str, object]:
    rows = list(examples)
    labels = _labels(rows)
    trainable = models["trainable"]
    frozen = models["frozen"]
    raw = models["raw_latent"]
    text = models["text_only"]
    masked = apply_example_control(rows, "view_masked", seed=seed + 40_000)
    role_shuffled = apply_example_control(rows, "role_labels_shuffled", seed=seed + 50_000)
    with _zero_role_embeddings(trainable.system):
        candidate_only = predict_latent_system(trainable, masked, "none", seed=seed + 60_000)
    metrics = {
        "trainable": _accuracy(predict_latent_system(trainable, rows, "none", seed), labels),
        "frozen": _accuracy(predict_latent_system(frozen, rows, "none", seed), labels),
        "text_only": _accuracy(text.predict(rows), labels),
        "raw_latent": _accuracy(predict_latent_system(raw, rows, "none", seed), labels),
        "candidate_only": _accuracy(candidate_only, labels),
        "view_masked": _accuracy(predict_latent_system(trainable, masked, "view_masked", seed), labels),
        "role_labels_shuffled": _accuracy(predict_latent_system(trainable, role_shuffled, "role_labels_shuffled", seed), labels),
        "oracle": _accuracy(_structured_oracle_predictions(rows, [0, 1, 2, 3]), labels),
    }
    return {
        "seed": int(seed),
        "split": "test",
        "accuracy": metrics,
        "passes_preferred_role_shuffle_gate": bool(metrics["role_labels_shuffled"] <= 0.18),
        "passes_max_role_shuffle_gate": bool(metrics["role_labels_shuffled"] <= 0.20),
        "full_validation_justified": False,
    }


def _role_ablation_suite(models: Dict[str, object], examples: Sequence[MultiViewTaskExample], seed: int) -> Dict[str, object]:
    labels = _labels(examples)
    n = len(examples)
    n_roles = len(examples[0].views)
    rng = np.random.default_rng(seed + 77_700)
    per_example_perm = np.vstack([_non_identity_permutation(n_roles, rng) for _ in range(n)]).astype(np.int64)
    global_perm = np.tile(_non_identity_permutation(n_roles, rng).reshape(1, n_roles), (n, 1)).astype(np.int64)
    all_zero = np.zeros((n, n_roles), dtype=np.int64)
    views_permuted_no_roles = _permute_views_within_example(examples, seed + 81_000)
    views_permuted_across_examples = _permute_views_across_examples_by_role(examples, seed + 82_000)
    order_randomized, order_randomized_role_ids = _randomize_physical_order_with_role_ids(examples, seed + 83_000)
    normalized = _normalize_view_format(examples)

    variants = [
        _variant("baseline_none", "Reference, no role corruption.", examples, labels, seed, mode="none"),
        _variant("A_role_embeddings_zeroed", "Both active-readout and coordinator role embedding tables zeroed.", examples, labels, seed, mode="zero_embeddings"),
        _variant("B_all_role_labels_same_null_role", "All runtime role_ids replaced with role 0.", examples, labels, seed, role_ids=all_zero),
        _variant("C_role_labels_randomly_permuted_per_example", "Independent non-identity role_id permutation for each example.", examples, labels, seed, role_ids=per_example_perm),
        _variant("D_role_labels_randomly_permuted_globally", "One non-identity role_id permutation shared by every example.", examples, labels, seed, role_ids=global_perm),
        _variant("E_private_views_permuted_without_updating_role_labels", "View texts are moved across physical role slots; runtime role_ids remain slot ids.", views_permuted_no_roles, labels, seed),
        _variant("F_private_views_permuted_with_role_labels_preserved", "View texts are permuted across examples within each role slot.", views_permuted_across_examples, labels, seed),
        _variant("G_role_embeddings_removed_physical_order_preserved", "Same operational intervention as A; physical tensor order is unchanged.", examples, labels, seed, mode="zero_embeddings"),
        _variant("H_physical_view_order_randomized_correct_role_labels_preserved", "Views are reordered within each example and explicit role_ids follow the moved view.", order_randomized, labels, seed, role_ids=order_randomized_role_ids),
        _variant("I_view_format_normalized_original_checkpoint_eval", "Original fitted models evaluated on normalized private-view text.", normalized, labels, seed),
    ]
    rows = [_evaluate_variant(models, variant) for variant in variants]
    return {
        "split": "test",
        "chance": 0.125,
        "near_chance_max": NEAR_CHANCE_8WAY_MAX,
        "rows": rows,
        "summary": _ablation_summary(rows),
    }


def _variant(
    name: str,
    description: str,
    examples: Sequence[MultiViewTaskExample],
    labels: np.ndarray,
    seed: int,
    mode: str = "custom",
    role_ids: np.ndarray | None = None,
) -> Dict[str, object]:
    return {
        "variant": name,
        "description": description,
        "examples": list(examples),
        "labels": labels,
        "seed": int(seed),
        "mode": mode,
        "role_ids": role_ids,
    }


def _evaluate_variant(models: Dict[str, object], variant: Dict[str, object]) -> Dict[str, object]:
    examples = list(variant["examples"])
    labels = np.asarray(variant["labels"], dtype=np.int64)
    seed = int(variant["seed"])
    mode = str(variant.get("mode", "custom"))
    role_ids = variant.get("role_ids")

    accuracies = {}
    for model_name in ("trainable", "frozen", "raw_latent"):
        result = models[model_name]
        if mode == "none":
            pred = predict_latent_system(result, examples, "none", seed)
        elif mode == "zero_embeddings":
            pred = _predict_with_zero_role_embeddings(result, examples, seed)
        elif isinstance(role_ids, np.ndarray):
            pred = _predict_with_role_ids(result.system, examples, role_ids, seed)
        else:
            pred = predict_latent_system(result, examples, "none", seed)
        accuracies[model_name] = _accuracy(pred, labels)
    accuracies["text_only"] = _accuracy(models["text_only"].predict(examples), labels)
    return {
        "variant": str(variant["variant"]),
        "description": str(variant["description"]),
        "accuracy": accuracies,
        "trainable_collapses_near_chance": bool(accuracies["trainable"] <= NEAR_CHANCE_8WAY_MAX),
        "frozen_collapses_near_chance": bool(accuracies["frozen"] <= NEAR_CHANCE_8WAY_MAX),
    }


def _predict_with_zero_role_embeddings(result: FitResult, examples: Sequence[MultiViewTaskExample], seed: int) -> np.ndarray:
    with _zero_role_embeddings(result.system):
        return predict_latent_system(result, examples, "none", seed)


def _predict_with_role_ids(
    system: SharedClonedAgentSystem,
    examples: Sequence[MultiViewTaskExample],
    role_ids_matrix: np.ndarray,
    seed: int,
) -> np.ndarray:
    del seed
    system.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(examples), 128):
            batch = list(examples[start : start + 128])
            device = next(system.parameters()).device
            role_ids = torch.as_tensor(role_ids_matrix[start : start + len(batch)], dtype=torch.long, device=device)
            readouts = system.collect_clone_representations(batch, condition="none", role_ids=role_ids)
            clone_activations = readouts["message"]
            if bool(getattr(system.coordinator, "requires_candidate_features", False)):
                candidate_features = _candidate_feature_tensor(batch, clone_activations.device)
                logits = system.coordinator(clone_activations, role_ids, candidate_features)
            else:
                logits = system.coordinator(clone_activations, role_ids)
            preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
    return np.concatenate(preds).astype(np.int64)


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


def _view_format_leakage_audit(splits: Dict[str, Sequence[MultiViewTaskExample]], device: str, seed: int) -> Dict[str, object]:
    train = _view_samples(splits["train"])
    dev = _view_samples(splits["dev"])
    test = _view_samples(splits["test"])
    diagnostics = {
        "role_id_from_view_text": _fit_role_classifier(
            _hashed_text_features([row["view_text"] for row in train], 128),
            _sample_labels(train),
            _hashed_text_features([row["view_text"] for row in dev], 128),
            _sample_labels(dev),
            _hashed_text_features([row["view_text"] for row in test], 128),
            _sample_labels(test),
            device=device,
            seed=seed + 1,
        ),
        "role_id_from_category_fields": _fit_role_classifier(
            _hashed_text_features([_category_field_text(row["view_text"]) for row in train], 64),
            _sample_labels(train),
            _hashed_text_features([_category_field_text(row["view_text"]) for row in dev], 64),
            _sample_labels(dev),
            _hashed_text_features([_category_field_text(row["view_text"]) for row in test], 64),
            _sample_labels(test),
            device=device,
            seed=seed + 2,
        ),
        "role_id_from_token_template": _fit_role_classifier(
            _hashed_text_features([_template_only_text(row["view_text"]) for row in train], 64),
            _sample_labels(train),
            _hashed_text_features([_template_only_text(row["view_text"]) for row in dev], 64),
            _sample_labels(dev),
            _hashed_text_features([_template_only_text(row["view_text"]) for row in test], 64),
            _sample_labels(test),
            device=device,
            seed=seed + 3,
        ),
        "role_id_from_length_statistics": _fit_role_classifier(
            _length_stats([row["view_text"] for row in train]),
            _sample_labels(train),
            _length_stats([row["view_text"] for row in dev]),
            _sample_labels(dev),
            _length_stats([row["view_text"] for row in test]),
            _sample_labels(test),
            device=device,
            seed=seed + 4,
            hidden_dims=(),
            epochs=30,
            lr=0.02,
        ),
    }
    return {
        "chance": 0.25,
        "samples": {"train": len(train), "dev": len(dev), "test": len(test)},
        "diagnostics": diagnostics,
        "summary": {
            "max_test_accuracy": max(float(row["test"]) for row in diagnostics.values()),
            "role_id_predictable_from_view_text": any(float(row["test"]) >= ROLE_ID_LEAK_HIGH for row in diagnostics.values()),
            "high_accuracy_threshold": ROLE_ID_LEAK_HIGH,
        },
        "sample_view_text_by_role": {
            str(role): next((row["view_text"] for row in test if int(row["role_id"]) == role), "")
            for role in range(4)
        },
    }


def _role_pair_shortcut_audit(splits: Dict[str, Sequence[MultiViewTaskExample]]) -> Dict[str, object]:
    train = list(splits["train"])
    test = list(splits["test"])
    y_train = _labels(train)
    y_test = _labels(test)
    baselines = {
        "role_pair_only": _lookup_accuracy(train, y_train, test, y_test, lambda ex: _role_pair_feature(ex)),
        "view_type_pair_only": _lookup_accuracy(train, y_train, test, y_test, lambda ex: _view_type_pair_feature(ex)),
        "category_pair_without_candidate": _lookup_accuracy(train, y_train, test, y_test, lambda ex: _category_pair_feature(ex)),
        "candidate_category_without_role": _lookup_accuracy(train, y_train, test, y_test, lambda ex: _candidate_category_feature(ex)),
        "role_pair_plus_candidate_category": _lookup_accuracy(train, y_train, test, y_test, lambda ex: (_role_pair_feature(ex), _candidate_category_feature(ex))),
    }
    return {
        "chance": 0.125,
        "near_chance_max": NEAR_CHANCE_8WAY_MAX,
        "baselines": {
            name: {
                "accuracy": float(value),
                "exceeds_chance_gate": bool(float(value) > NEAR_CHANCE_8WAY_MAX),
            }
            for name, value in baselines.items()
        },
        "summary": {
            "max_accuracy": max(float(value) for value in baselines.values()),
            "any_exceeds_chance_gate": any(float(value) > NEAR_CHANCE_8WAY_MAX for value in baselines.values()),
        },
    }


def _evaluate_normalized_view_format(models: Dict[str, object], examples: Sequence[MultiViewTaskExample], seed: int) -> Dict[str, object]:
    labels = _labels(examples)
    rows = {}
    for model_name in ("trainable", "frozen", "raw_latent"):
        rows[model_name] = _accuracy(predict_latent_system(models[model_name], examples, "none", seed), labels)
    rows["text_only"] = _accuracy(models["text_only"].predict(examples), labels)
    return {
        "split": "test",
        "accuracy": rows,
        "interpretation": "distribution-shift check only; these models were fit on original role-leaky view templates",
    }


def _normalize_view_format(examples: Sequence[MultiViewTaskExample]) -> List[MultiViewTaskExample]:
    out = []
    for example in examples:
        evidence = list(example_oracle_metadata(example).get("evidence_bits", [0, 0, 0, 0]))
        views = []
        for role_index, view in enumerate(example.views):
            bit = _view_bit(view.text, role_index)
            if bit < 0 and role_index < len(evidence):
                bit = int(evidence[role_index])
            value = "FEATURE_BETA" if int(bit) else "FEATURE_ALPHA"
            views.append(
                replace(
                    view,
                    text=(
                        "Redacted compatibility artifact:\n"
                        f"observed_value: {value}\n"
                        "exact_symbol_module_path: withheld"
                    ),
                    source_path=f"REDACTED_VIEW_SLOT_{role_index}",
                    source_type="redacted_artifact_normalized:stage37",
                )
            )
        out.append(replace(example, views=tuple(views)))
    return out


def _permute_views_within_example(examples: Sequence[MultiViewTaskExample], seed: int) -> List[MultiViewTaskExample]:
    rng = np.random.default_rng(seed + 91_000)
    out = []
    for example in examples:
        perm = _non_identity_permutation(len(example.views), rng)
        views = []
        for slot, donor_index in enumerate(perm):
            donor = example.views[int(donor_index)]
            views.append(
                replace(
                    donor,
                    role=example.views[slot].role,
                    source_type=f"{donor.source_type}:stage37_view_permuted_without_role_update",
                )
            )
        out.append(replace(example, views=tuple(views)))
    return out


def _permute_views_across_examples_by_role(examples: Sequence[MultiViewTaskExample], seed: int) -> List[MultiViewTaskExample]:
    rng = np.random.default_rng(seed + 92_000)
    out_views = [list(example.views) for example in examples]
    for role_index in range(len(examples[0].views)):
        perm = _non_identity_permutation(len(examples), rng)
        for row_index, donor_index in enumerate(perm):
            donor = examples[int(donor_index)].views[role_index]
            original = examples[row_index].views[role_index]
            out_views[row_index][role_index] = replace(
                original,
                text=donor.text,
                source_path=donor.source_path,
                source_type=f"{donor.source_type}:stage37_cross_example_same_role_donor",
            )
    return [replace(example, views=tuple(out_views[index])) for index, example in enumerate(examples)]


def _randomize_physical_order_with_role_ids(
    examples: Sequence[MultiViewTaskExample],
    seed: int,
) -> Tuple[List[MultiViewTaskExample], np.ndarray]:
    rng = np.random.default_rng(seed + 93_000)
    out = []
    role_rows = []
    for example in examples:
        perm = _non_identity_permutation(len(example.views), rng)
        moved = tuple(example.views[int(index)] for index in perm)
        role_rows.append([_role_id_for_view(view) for view in moved])
        out.append(replace(example, views=moved))
    return out, np.asarray(role_rows, dtype=np.int64)


def _role_id_for_view(view: View) -> int:
    try:
        return list(ROLE_NAMES).index(str(view.role))
    except ValueError:
        return 0


def _view_samples(examples: Sequence[MultiViewTaskExample]) -> List[Dict[str, object]]:
    rows = []
    for example in examples:
        for role_id, view in enumerate(example.views):
            rows.append(
                {
                    "example_id": example.id,
                    "role_id": int(role_id),
                    "role_name": view.role,
                    "view_text": view.text,
                    "source_type": view.source_type,
                }
            )
    return rows


def _sample_labels(samples: Sequence[Dict[str, object]]) -> np.ndarray:
    return np.asarray([int(row["role_id"]) for row in samples], dtype=np.int64)


def _fit_role_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    device: str,
    seed: int,
    hidden_dims: Iterable[int] = (32,),
    epochs: int = 20,
    lr: float = 0.01,
) -> Dict[str, object]:
    torch.manual_seed(seed + 101)
    model = MLPClassifier(x_train.shape[1], tuple(hidden_dims), 4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    tx = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    ty = torch.as_tensor(y_train, dtype=torch.long, device=device)
    dx = torch.as_tensor(x_dev, dtype=torch.float32, device=device)
    dy = torch.as_tensor(y_dev, dtype=torch.long, device=device)
    vx = torch.as_tensor(x_test, dtype=torch.float32, device=device)
    vy = torch.as_tensor(y_test, dtype=torch.long, device=device)
    rng = np.random.default_rng(seed + 202)
    best_state = None
    best_dev = -1.0
    stale = 0
    history = []
    for epoch in range(int(epochs)):
        model.train()
        for batch in _batches(rng.permutation(len(y_train)), 64):
            idx = torch.as_tensor(batch, dtype=torch.long, device=device)
            loss = F.cross_entropy(model(tx.index_select(0, idx)), ty.index_select(0, idx))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        train_acc = _tensor_acc(model, tx, ty)
        dev_acc = _tensor_acc(model, dx, dy)
        history.append({"epoch": epoch + 1, "train": train_acc, "dev": dev_acc})
        if dev_acc > best_dev:
            best_dev = dev_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 4:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    return {
        "train": _tensor_acc(model, tx, ty),
        "dev": _tensor_acc(model, dx, dy),
        "test": _tensor_acc(model, vx, vy),
        "epochs_run": len(history),
        "history": history,
    }


def _tensor_acc(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return float((torch.argmax(model(x), dim=1) == y).float().mean().detach().cpu().item())


def _hashed_text_features(texts: Sequence[str], feature_dim: int) -> np.ndarray:
    features = np.zeros((len(texts), feature_dim), dtype=np.float32)
    for row_id, text in enumerate(texts):
        for token in _text_tokens(text):
            raw = __import__("hashlib").sha256(token.encode("utf-8")).hexdigest()
            index = int(raw[:8], 16) % feature_dim
            sign = 1.0 if int(raw[8:10], 16) % 2 == 0 else -1.0
            features[row_id, index] += sign
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1.0)


def _category_field_text(text: str) -> str:
    fields = []
    for line in str(text).splitlines():
        if ":" in line:
            fields.append(line.split(":", 1)[0].strip().lower())
    return " ".join(fields)


def _template_only_text(text: str) -> str:
    rows = []
    for line in str(text).splitlines():
        if ":" in line:
            rows.append(line.split(":", 1)[0].strip().lower() + ": <value>")
        else:
            rows.append(line.strip().lower())
    return "\n".join(rows)


def _length_stats(texts: Sequence[str]) -> np.ndarray:
    rows = []
    for text in texts:
        tokens = _text_tokens(text)
        lines = str(text).splitlines()
        lengths = [len(token) for token in tokens] or [0]
        rows.append(
            [
                float(len(text)),
                float(len(tokens)),
                float(len(lines)),
                float(str(text).count(":")),
                float(str(text).count("_")),
                float(mean(lengths)),
                float(max(lengths)),
                float(sum(char.isdigit() for char in str(text))),
            ]
        )
    arr = np.asarray(rows, dtype=np.float32)
    mean_vec = arr.mean(axis=0, keepdims=True)
    std_vec = arr.std(axis=0, keepdims=True)
    return (arr - mean_vec) / np.maximum(std_vec, 1.0)


def _lookup_accuracy(
    train: Sequence[MultiViewTaskExample],
    y_train: np.ndarray,
    test: Sequence[MultiViewTaskExample],
    y_test: np.ndarray,
    builder: Callable[[MultiViewTaskExample], object],
) -> float:
    predictions = _lookup_predict([builder(example) for example in train], y_train, [builder(example) for example in test], num_classes=8)
    return _accuracy(predictions, y_test)


def _lookup_predict(train_features: Sequence[object], train_labels: np.ndarray, test_features: Sequence[object], num_classes: int) -> np.ndarray:
    table: Dict[str, Counter] = defaultdict(Counter)
    for feature, label in zip(train_features, train_labels):
        table[_stable_feature(feature)][int(label)] += 1
    majority = int(np.argmax(np.bincount(train_labels, minlength=num_classes)))
    out = []
    for feature in test_features:
        counts = table.get(_stable_feature(feature))
        if not counts:
            out.append(majority)
        else:
            out.append(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0])
    return np.asarray(out, dtype=np.int64)


def _role_pair_feature(example: MultiViewTaskExample) -> object:
    roles = [view.role for view in example.views]
    return ((roles[0], roles[1]), (roles[2], roles[3]))


def _view_type_pair_feature(example: MultiViewTaskExample) -> object:
    types = [view.source_type for view in example.views]
    return ((types[0], types[1]), (types[2], types[3]))


def _category_pair_feature(example: MultiViewTaskExample) -> object:
    bits = [_view_bit(view.text, role_id) for role_id, view in enumerate(example.views)]
    return ((bits[0], bits[1]), (bits[2], bits[3]))


def _candidate_category_feature(example: MultiViewTaskExample) -> object:
    return tuple(tuple(_candidate_bits_from_attributes(candidate.attributes)) for candidate in example.candidates)


def _candidate_bits_from_attributes(attributes: Sequence[str]) -> Tuple[int, int, int, int]:
    bits = []
    for index, pair in enumerate(ATTRIBUTE_VALUE_PAIRS):
        value = str(attributes[index]) if index < len(attributes) else ""
        bits.append(1 if value == str(pair[1]) else 0)
    return tuple(bits[:4])  # type: ignore[return-value]


def _view_bit(text: str, role_id: int) -> int:
    lowered = str(text).lower()
    if "feature_beta" in lowered:
        return 1
    if "feature_alpha" in lowered:
        return 0
    if role_id >= len(ATTRIBUTE_VALUE_PAIRS):
        return -1
    low, high = ATTRIBUTE_VALUE_PAIRS[role_id]
    high_text = str(high).replace("_", " ").lower()
    low_text = str(low).replace("_", " ").lower()
    if high_text in lowered:
        return 1
    if low_text in lowered:
        return 0
    return -1


def _labels(examples: Sequence[MultiViewTaskExample]) -> np.ndarray:
    return np.asarray([example.label for example in examples], dtype=np.int64)


def _stable_feature(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _batches(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), int(batch_size)):
        yield indices[start : start + int(batch_size)]


def _non_identity_permutation(n: int, rng: np.random.Generator) -> np.ndarray:
    perm = rng.permutation(n)
    if n > 1 and np.array_equal(perm, np.arange(n)):
        perm = np.roll(perm, 1)
    return perm.astype(np.int64, copy=False)


def _source_excerpt(source: str, needles: Sequence[str], context: int = 4) -> str:
    lines = source.splitlines()
    selected = set()
    for index, line in enumerate(lines):
        if any(needle in line for needle in needles):
            for row in range(max(0, index - context), min(len(lines), index + context + 1)):
                selected.add(row)
    return "\n".join(f"{index + 1}: {lines[index]}" for index in sorted(selected))


def _role_id_leaks_from_view_format(leakage: Dict[str, object]) -> bool:
    summary = leakage.get("summary", {})
    return bool(summary.get("role_id_predictable_from_view_text", False))


def _ablation_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "trainable_near_chance_variants": [
            row["variant"] for row in rows if bool(row.get("trainable_collapses_near_chance", False))
        ],
        "trainable_above_chance_variants": [
            row["variant"] for row in rows if not bool(row.get("trainable_collapses_near_chance", False))
        ],
    }


def _summary(
    implementation: Dict[str, object],
    leakage: Dict[str, object],
    shortcuts: Dict[str, object],
    original_tiny: Dict[str, object],
    ablations: Dict[str, object],
    normalized_leakage: Dict[str, object],
    fixed_tiny: Dict[str, object] | None,
) -> Dict[str, object]:
    original_role = float(original_tiny.get("accuracy", {}).get("role_labels_shuffled", 1.0))
    fixed_acc = fixed_tiny.get("accuracy", {}) if isinstance(fixed_tiny, dict) else {}
    fixed_role = float(fixed_acc.get("role_labels_shuffled", 1.0)) if fixed_acc else None
    role_leak = bool(leakage.get("summary", {}).get("role_id_predictable_from_view_text", False))
    shortcut_leak = bool(shortcuts.get("summary", {}).get("any_exceeds_chance_gate", False))
    null_role_row = _row_by_name(ablations.get("rows", []), "B_all_role_labels_same_null_role")
    zero_role_row = _row_by_name(ablations.get("rows", []), "A_role_embeddings_zeroed")
    null_role_high = bool(null_role_row and float(null_role_row["accuracy"]["trainable"]) > NEAR_CHANCE_8WAY_MAX)
    zero_role_high = bool(zero_role_row and float(zero_role_row["accuracy"]["trainable"]) > NEAR_CHANCE_8WAY_MAX)
    role_labels_not_needed = bool(null_role_high and zero_role_high)
    full_validation_justified = bool(
        fixed_acc
        and float(fixed_acc.get("trainable", 0.0)) >= 0.35
        and float(fixed_acc.get("candidate_only", 1.0)) <= NEAR_CHANCE_8WAY_MAX
        and float(fixed_acc.get("view_masked", 1.0)) <= NEAR_CHANCE_8WAY_MAX
        and float(fixed_acc.get("role_labels_shuffled", 1.0)) <= 0.20
        and float(fixed_acc.get("oracle", 0.0)) >= 0.90
    )
    # Stage 3.7 is diagnostic. Keep full validation blocked until this one-seed result is reviewed.
    full_validation_justified = False
    return {
        "why_role_shuffled_stayed_high": (
            "The existing control corrupts only learned runtime role ids. In Stage 3.4b private view text contains role-specific field names "
            "such as symbol_surface/provider_area/provider_name_shape/import_slot_shape, so role identity survives the role-id shuffle."
        ),
        "classification": {
            "control_weakness": True,
            "role_format_leakage": role_leak,
            "role_pair_shortcut": shortcut_leak,
            "true_role_invariant_learning": bool(not role_leak and not shortcut_leak and role_labels_not_needed),
            "role_labels_not_needed_by_original_checkpoint": role_labels_not_needed,
            "role_embeddings_zeroed_above_chance": zero_role_high,
            "null_role_labels_above_chance": null_role_high,
        },
        "supporting_metrics": {
            "original_role_labels_shuffled_accuracy": original_role,
            "original_candidate_only_accuracy": float(original_tiny.get("accuracy", {}).get("candidate_only", 1.0)),
            "original_role_id_from_view_text_max_test_accuracy": float(leakage.get("summary", {}).get("max_test_accuracy", 0.0)),
            "original_role_pair_shortcut_max_accuracy": float(shortcuts.get("summary", {}).get("max_accuracy", 0.0)),
            "normalized_role_id_from_view_text_max_test_accuracy": float(normalized_leakage.get("summary", {}).get("max_test_accuracy", 0.0)),
            "fixed_role_labels_shuffled_accuracy": fixed_role,
        },
        "cheap_fix_implemented": bool(role_leak),
        "full_validation_justified": full_validation_justified,
        "validation_decision": "not justified",
        "implementation_answers": implementation.get("answers", {}),
    }


def _row_by_name(rows: Sequence[Dict[str, object]], name: str) -> Dict[str, object] | None:
    for row in rows:
        if row.get("variant") == name:
            return row
    return None


def _render_report(result: Dict[str, object]) -> str:
    summary = result["summary"]
    impl = result["role_shuffle_implementation"]["answers"]
    original = result["original_tiny_smoke"]["accuracy"]
    ablations = result["role_ablation_evaluations"]["rows"]
    leakage = result["view_format_leakage_audit"]
    shortcuts = result["role_pair_shortcut_audit"]
    normalized = result["normalized_view_format_variant"]
    fixed = normalized.get("tiny_smoke_after_fix") or {}
    fixed_acc = fixed.get("accuracy", {}) if isinstance(fixed, dict) else {}

    lines = [
        "# Stage 3.7 Role-Control Diagnosis",
        "",
        "## Scope",
        "",
        "- Full 10-seed validation: not run.",
        "- Locked architecture changes: none.",
        "- Checkpoint source: no Stage 3.6 tiny-smoke checkpoint artifact was available, so this reran only the one-seed tiny smoke.",
        "",
        "## Main Finding",
        "",
        str(summary["why_role_shuffled_stayed_high"]),
        "",
        "Classification:",
    ]
    for key, value in summary["classification"].items():
        lines.append(f"- {key}: `{bool(value)}`")
    lines.extend(
        [
            "",
            "## Part 1: role_labels_shuffled Implementation",
            "",
            f"1. Shuffles role embeddings only: `{impl['1_shuffles_role_embeddings_only']}`",
            f"2. Shuffles role text labels: `{impl['2_shuffles_role_text_labels']}`",
            f"3. Shuffles private views: `{impl['3_shuffles_private_views']}`",
            f"4. Shuffles role/view pairing: `{impl['4_shuffles_role_view_pairing']}`",
            f"5. Shuffle scope: `{impl['5_shuffle_scope']}`",
            f"6. Same permutation for train/dev/test: `{impl['6_same_permutation_train_dev_test']}`",
            f"7. Corruption time: `{impl['7_corrupted_train_or_test_time']}`",
            "",
            "## Original Tiny Smoke",
            "",
            "| metric | accuracy |",
            "|---|---:|",
        ]
    )
    for name in ("trainable", "frozen", "text_only", "raw_latent", "candidate_only", "view_masked", "role_labels_shuffled", "oracle"):
        lines.append(f"| {name} | {float(original.get(name, 0.0)):.4f} |")
    lines.extend(
        [
            "",
            "## Part 2: Role Ablations",
            "",
            "| variant | trainable | frozen | text | raw | trainable near chance? |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in ablations:
        acc = row["accuracy"]
        lines.append(
            "| {variant} | {trainable:.4f} | {frozen:.4f} | {text:.4f} | {raw:.4f} | `{near}` |".format(
                variant=row["variant"],
                trainable=float(acc.get("trainable", 0.0)),
                frozen=float(acc.get("frozen", 0.0)),
                text=float(acc.get("text_only", 0.0)),
                raw=float(acc.get("raw_latent", 0.0)),
                near=bool(row.get("trainable_collapses_near_chance", False)),
            )
        )
    lines.extend(
        [
            "",
            "## Part 3: View-Format Leakage Audit",
            "",
            "| diagnostic | train | dev | test |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, row in leakage["diagnostics"].items():
        lines.append(f"| {name} | {float(row['train']):.4f} | {float(row['dev']):.4f} | {float(row['test']):.4f} |")
    lines.extend(
        [
            "",
            f"- Role id predictable from original view format: `{bool(leakage['summary']['role_id_predictable_from_view_text'])}`",
            "",
            "## Part 4: Role-Pair Shortcut Audit",
            "",
            "| baseline | accuracy | exceeds gate? |",
            "|---|---:|---|",
        ]
    )
    for name, row in shortcuts["baselines"].items():
        lines.append(f"| {name} | {float(row['accuracy']):.4f} | `{bool(row['exceeds_chance_gate'])}` |")
    lines.extend(
        [
            "",
            "## Part 5: Fix Decision",
            "",
            f"- Normalized-view-format variant implemented: `{bool(normalized.get('implemented', False))}`",
            f"- Original max role-id-from-view diagnostic: `{float(leakage['summary']['max_test_accuracy']):.4f}`",
            f"- Normalized max role-id-from-view diagnostic: `{float(normalized['view_format_leakage_audit']['summary']['max_test_accuracy']):.4f}`",
            "- Role-pair shortcut detected: `{}`".format(bool(shortcuts["summary"]["any_exceeds_chance_gate"])),
            "",
            "## Part 6: Tiny Smoke After Fix",
            "",
        ]
    )
    if fixed_acc:
        lines.extend(["| metric | accuracy |", "|---|---:|"])
        for name in ("trainable", "frozen", "text_only", "raw_latent", "candidate_only", "view_masked", "role_labels_shuffled", "oracle"):
            lines.append(f"| {name} | {float(fixed_acc.get(name, 0.0)):.4f} |")
    else:
        lines.append("- Not run because no view-format leakage fix was triggered.")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Cheap fix implemented: `{bool(summary['cheap_fix_implemented'])}`",
            f"- Full validation justified: `{bool(summary['full_validation_justified'])}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
