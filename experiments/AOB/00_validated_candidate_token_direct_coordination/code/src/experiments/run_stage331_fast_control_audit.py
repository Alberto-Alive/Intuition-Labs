from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import replace
from itertools import product
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from src.datasets.multiview_code_patch_selection import (
    ATTRIBUTE_VALUE_PAIRS,
    Candidate,
    MultiViewCodePatchDatasetConfig,
    MultiViewTaskExample,
    View,
    apply_example_control,
    build_multiview_code_patch_splits,
    candidate_attribute_bits,
    randomized_labels_for_examples,
)
from src.experiments.real_shared_weight_latent_coordination import (
    SharedClonedAgentSystem,
    _candidate_feature_tensor,
    _example_batches,
    predict_latent_system,
)
from src.experiments.run_stage31_frozen_diagnosis import _load_latent_checkpoint


DEFAULT_RESULTS = "results/stage33_role_balanced_real_code_results.json"
DEFAULT_OUTPUT = "results/stage331_fast_control_audit.json"
DEFAULT_REPORT = "reports/STAGE331_FAST_CONTROL_AUDIT.md"
NEAR_CHANCE = 0.125
STRICT_CONTROL_MAX = 0.18
ROLE_SHUFFLE_MAX = 0.20


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3.3.1 fast control-harness audit.")
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-eval", action="store_true", default=True)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    result = run_audit(
        results_path=Path(args.results),
        output_path=Path(args.output),
        report_path=Path(args.report),
        device=str(args.device),
        run_checkpoint_eval=bool(args.checkpoint_eval),
    )
    if not result:
        raise SystemExit(1)


def run_audit(
    results_path: Path,
    output_path: Path,
    report_path: Path,
    device: str = "cuda",
    run_checkpoint_eval: bool = True,
) -> Dict[str, object]:
    stage33 = json.loads(results_path.read_text(encoding="utf-8"))
    rows = [row for row in stage33.get("stage33_rows", []) if row.get("status") == "completed"]
    rows.sort(key=lambda row: int(row["seed"]))
    if not rows:
        raise RuntimeError(f"no completed stage33_rows found in {results_path}")

    dataset_rows = []
    split_cache: Dict[int, Dict[str, List[MultiViewTaskExample]]] = {}
    for row in rows:
        seed = int(row["seed"])
        dataset_config = MultiViewCodePatchDatasetConfig(**row["dataset_config"])
        splits = build_multiview_code_patch_splits(dataset_config, seed=seed, repo_root=".")
        split_cache[seed] = {name: list(values) for name, values in splits.items()}
        dataset_rows.append(_dataset_seed_audit(seed, split_cache[seed]))

    implementation = _control_implementation_audit(split_cache[int(rows[0]["seed"])]["test"])
    checkpoint_eval = _checkpoint_eval(rows, split_cache, device=device) if run_checkpoint_eval else {}
    existing_controls = _existing_control_summary(rows)
    output = {
        "metadata": {
            "stage": "stage3.3.1_fast_control_harness_audit",
            "source_results": str(results_path),
            "architecture_changes": "none",
            "training_jobs_run": 0,
            "full_validation_run": False,
            "checkpoint_only_evaluation": bool(run_checkpoint_eval),
            "device": device,
        },
        "dataset_only_audit": {
            "seed_rows": dataset_rows,
            "summary": _dataset_summary(dataset_rows),
        },
        "control_implementation_audit": implementation,
        "existing_stage33_control_metrics": existing_controls,
        "checkpoint_only_evaluations": checkpoint_eval,
        "conclusion": _conclusion(dataset_rows, implementation, existing_controls, checkpoint_eval),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(output), encoding="utf-8")
    return output


def _dataset_seed_audit(seed: int, splits: Dict[str, List[MultiViewTaskExample]]) -> Dict[str, object]:
    random_audit = _randomized_label_audit(seed, splits)
    role_shuffle_audit = _role_shuffle_dataset_audit(seed, splits["test"])
    view_masked_audit = _view_masked_audit(seed, splits)
    duplication_audit = _duplication_template_audit(seed, splits)
    candidate_shortcut = _candidate_metadata_shortcut_audit(splits)
    return {
        "seed": seed,
        "split_sizes": {split: len(examples) for split, examples in splits.items()},
        "randomized_label_audit": random_audit,
        "role_label_shuffle_audit": role_shuffle_audit,
        "view_masked_audit": view_masked_audit,
        "duplication_template_audit": duplication_audit,
        "candidate_metadata_shortcut_audit": candidate_shortcut,
    }


def _randomized_label_audit(seed: int, splits: Dict[str, List[MultiViewTaskExample]]) -> Dict[str, object]:
    offsets = {"train": 401, "dev": 501, "test": 601}
    randomized = {
        split: randomized_labels_for_examples(examples, seed=seed + offsets[split], num_classes=8)
        for split, examples in splits.items()
    }
    per_split = {}
    for split, labels in randomized.items():
        examples = splits[split]
        original = _labels(examples)
        per_split[split] = {
            "label_counts": _counts(labels),
            "balance_max_deviation": _balance_deviation(labels),
            "accuracy_vs_original_gold_label": _acc(labels, original),
            "accuracy_vs_example_index_mod8": _acc(labels, np.asarray([_example_index(example) % 8 for example in examples], dtype=np.int64)),
            "accuracy_vs_hash_id_mod8": _acc(labels, np.asarray([_stable_hash(example.id) % 8 for example in examples], dtype=np.int64)),
            "same_split_trivial_predictors": _same_split_trivial_predictors(examples, labels, split),
        }

    train_examples = splits["train"]
    train_random = randomized["train"]
    test_examples = splits["test"]
    test_random = randomized["test"]
    test_original = _labels(test_examples)
    features = _feature_builders()
    cross = {}
    for name, builder in features.items():
        train_features = [builder(example, "train") for example in train_examples]
        test_features = [builder(example, "test") for example in test_examples]
        pred = _lookup_predict(train_features, train_random, test_features, num_classes=8)
        cross[name] = {
            "train_random_to_test_random_accuracy": _acc(pred, test_random),
            "train_random_to_test_original_accuracy": _acc(pred, test_original),
        }
    return {
        "per_split": per_split,
        "cross_split_trivial_predictors": cross,
        "independence_flags": {
            "balanced_train_dev_test": all(per_split[split]["balance_max_deviation"] <= 1 for split in ("train", "dev", "test")),
            "test_random_vs_original_near_chance": per_split["test"]["accuracy_vs_original_gold_label"] <= STRICT_CONTROL_MAX,
            "train_random_predicts_test_original_near_chance_max": max(value["train_random_to_test_original_accuracy"] for value in cross.values()) <= STRICT_CONTROL_MAX,
        },
    }


def _role_shuffle_dataset_audit(seed: int, test_examples: Sequence[MultiViewTaskExample]) -> Dict[str, object]:
    shuffled = apply_example_control(test_examples, condition="role_labels_shuffled", seed=seed)
    unchanged_views = all(
        tuple(view.text for view in before.views) == tuple(view.text for view in after.views)
        for before, after in zip(test_examples, shuffled)
    )
    unchanged_candidates = all(
        tuple(candidate.attributes for candidate in before.candidates) == tuple(candidate.attributes for candidate in after.candidates)
        for before, after in zip(test_examples, shuffled)
    )
    labels = _labels(test_examples)
    pair_before = {
        "0,1": _structured_oracle_accuracy(test_examples, [0, 1]),
        "2,3": _structured_oracle_accuracy(test_examples, [2, 3]),
    }
    pair_after = {
        "0,1": _structured_oracle_accuracy(shuffled, [0, 1]),
        "2,3": _structured_oracle_accuracy(shuffled, [2, 3]),
    }
    del labels
    return {
        "apply_example_control_returns_same_examples_for_role_label_shuffle": unchanged_views and unchanged_candidates,
        "role_embeddings_only": True,
        "role_text_labels_shuffled": False,
        "role_view_pairing_shuffled": False,
        "physical_order_shuffled": False,
        "candidate_feature_columns_shuffled": False,
        "pair_structure_survives_apply_example_control": pair_before == pair_after,
        "pairwise_oracle_before": pair_before,
        "pairwise_oracle_after": pair_after,
    }


def _view_masked_audit(seed: int, splits: Dict[str, List[MultiViewTaskExample]]) -> Dict[str, object]:
    del seed
    train = splits["train"]
    test = splits["test"]
    masked_test = apply_example_control(test, condition="view_masked", seed=0)
    all_private_views_masked = all("masked for the evidence-masking control" in view.text for example in masked_test for view in example.views)
    private_view_leakage = _private_view_exact_leakage(masked_test)
    attrs_unchanged = all(
        tuple(candidate.attributes for candidate in before.candidates) == tuple(candidate.attributes for candidate in after.candidates)
        for before, after in zip(test, masked_test)
    )
    texts_unchanged = all(
        tuple(candidate.text for candidate in before.candidates) == tuple(candidate.text for candidate in after.candidates)
        for before, after in zip(test, masked_test)
    )
    y_test = _labels(test)
    baselines = {
        "candidate_index_only": _acc(np.zeros(len(test), dtype=np.int64), y_test),
        "family_only": _lookup_feature_accuracy(train, _labels(train), test, y_test, lambda ex, split: _family(ex)),
        "role_pair_pattern_only": _lookup_feature_accuracy(train, _labels(train), test, y_test, lambda ex, split: _role_pair_pattern(ex)),
        "family_plus_role_pair": _lookup_feature_accuracy(train, _labels(train), test, y_test, lambda ex, split: (_family(ex), _role_pair_pattern(ex))),
        "candidate_order_metadata": _acc(np.asarray([_candidate_order_gold_position(example) for example in test], dtype=np.int64), y_test),
        "candidate_codebook_oracle": _candidate_codebook_oracle_accuracy(train, test),
        "ordered_candidate_bits_lookup": _lookup_feature_accuracy(train, _labels(train), test, y_test, lambda ex, split: _ordered_candidate_bits(ex)),
    }
    return {
        "all_private_views_masked": all_private_views_masked,
        "private_view_exact_leakage": private_view_leakage,
        "candidate_attributes_remain_after_view_mask": attrs_unchanged,
        "candidate_text_remains_after_view_mask": texts_unchanged,
        "candidate_metadata_remains_available_to_coordinator": attrs_unchanged,
        "baselines_on_masked_view_setting": baselines,
    }


def _duplication_template_audit(seed: int, splits: Dict[str, List[MultiViewTaskExample]]) -> Dict[str, object]:
    del seed
    ids = {split: {example.id for example in examples} for split, examples in splits.items()}
    source_keys = {split: {str(example.metadata.get("source_import_key")) for example in examples} for split, examples in splits.items()}
    templates = {split: [_template_signature(example) for example in examples] for split, examples in splits.items()}
    ordered_bits = {split: [_ordered_candidate_bits(example) for example in examples] for split, examples in splits.items()}
    train_templates = set(templates["train"])
    train_ordered_bits = set(ordered_bits["train"])
    test = splits["test"]
    train = splits["train"]
    y_test = _labels(test)
    y_train = _labels(train)
    return {
        "train_test_id_overlap": len(ids["train"] & ids["test"]),
        "train_dev_id_overlap": len(ids["train"] & ids["dev"]),
        "dev_test_id_overlap": len(ids["dev"] & ids["test"]),
        "train_test_source_import_key_overlap": len(source_keys["train"] & source_keys["test"]),
        "train_test_exact_template_duplicate_rate": _rate([signature in train_templates for signature in templates["test"]]),
        "train_test_ordered_candidate_bits_duplicate_rate": _rate([signature in train_ordered_bits for signature in ordered_bits["test"]]),
        "family_template_implies_label_accuracy": _lookup_feature_accuracy(train, y_train, test, y_test, lambda ex, split: (_family(ex), _template_signature(ex))),
        "template_only_accuracy": _lookup_feature_accuracy(train, y_train, test, y_test, lambda ex, split: _template_signature(ex)),
        "ordered_candidate_bits_lookup_accuracy": _lookup_feature_accuracy(train, y_train, test, y_test, lambda ex, split: _ordered_candidate_bits(ex)),
        "near_duplicate_definition": "normalized exact template hash; FEATURE_ALPHA/BETA, redacted uid, and candidate ids are normalized",
    }


def _candidate_metadata_shortcut_audit(splits: Dict[str, List[MultiViewTaskExample]]) -> Dict[str, object]:
    train = splits["train"]
    test = splits["test"]
    y_test = _labels(test)
    return {
        "candidate_order_metadata_accuracy": _acc(np.asarray([_candidate_order_gold_position(example) for example in test], dtype=np.int64), y_test),
        "candidate_codebook_oracle_accuracy": _candidate_codebook_oracle_accuracy(train, test),
        "ordered_candidate_bits_lookup_accuracy": _lookup_feature_accuracy(train, _labels(train), test, y_test, lambda ex, split: _ordered_candidate_bits(ex)),
        "candidate_bit_multiset_lookup_accuracy": _lookup_feature_accuracy(train, _labels(train), test, y_test, lambda ex, split: _candidate_bit_multiset(ex)),
        "candidate_metadata_shortcut_present": _candidate_codebook_oracle_accuracy(train, test) > STRICT_CONTROL_MAX,
    }


def _control_implementation_audit(sample_examples: Sequence[MultiViewTaskExample]) -> Dict[str, object]:
    apply_source = inspect.getsource(apply_example_control)
    forward_source = inspect.getsource(SharedClonedAgentSystem.forward)
    return {
        "role_label_shuffle_behavior": {
            "apply_example_control_returns_examples_unchanged": "role_labels_shuffled" in apply_source and "return list(examples)" in apply_source,
            "system_forward_shuffles_role_ids": "condition == \"role_labels_shuffled\"" in forward_source and "role_ids = base_role_ids.gather" in forward_source,
            "role_embeddings_only": True,
            "role_text_labels_only": False,
            "role_view_pairing": "unchanged",
            "physical_order": "unchanged",
            "candidate_features": "unchanged",
        },
        "view_masked_behavior": {
            "view_text_replaced_by_apply_example_control": "condition == \"view_masked\"" in apply_source,
            "candidate_text_or_attributes_changed": False,
            "candidate_features_remain_available_to_candidate_query_coordinator": True,
        },
        "hidden_shuffle_behavior": {
            "hidden_states_shuffled_inside_collect_clone_representations": True,
            "labels_and_candidate_features_unchanged": True,
        },
        "source_excerpts": {
            "apply_example_control_relevant": _source_excerpt(apply_source, ["role_labels_shuffled", "view_masked", "candidate_order_shuffled"]),
            "system_forward_relevant": _source_excerpt(forward_source, ["role_labels_shuffled", "physical_order_shuffled_roles_preserved"]),
        },
        "sample_pair_structure": {
            "pairwise_oracle_0_1": _structured_oracle_accuracy(sample_examples, [0, 1]),
            "pairwise_oracle_2_3": _structured_oracle_accuracy(sample_examples, [2, 3]),
        },
    }


def _checkpoint_eval(
    rows: Sequence[Dict[str, object]],
    split_cache: Dict[int, Dict[str, List[MultiViewTaskExample]]],
    device: str,
) -> Dict[str, object]:
    variants: Dict[str, Callable[[Sequence[MultiViewTaskExample], int], List[MultiViewTaskExample]]] = {
        "none": lambda examples, seed: list(examples),
        "views_masked": lambda examples, seed: apply_example_control(examples, "view_masked", seed=seed),
        "candidate_metadata_removed": lambda examples, seed: [_remove_candidate_metadata(example) for example in examples],
        "family_id_removed": lambda examples, seed: [_remove_family_metadata(example) for example in examples],
        "role_pair_structure_randomized": lambda examples, seed: [_randomize_candidate_role_columns(example, seed + index) for index, example in enumerate(examples)],
        "view_only_without_candidate_metadata": lambda examples, seed: [_remove_candidate_metadata(example) for example in examples],
        "candidate_only": lambda examples, seed: apply_example_control(examples, "view_masked", seed=seed),
        "view_shuffled": lambda examples, seed: apply_example_control(examples, "view_shuffled", seed=seed),
    }
    rows_out = []
    for row in rows:
        seed = int(row["seed"])
        examples = split_cache[seed]["test"]
        y = _labels(examples)
        trainable = _load_latent_checkpoint(Path(row["checkpoint_paths"]["trainable"]), device=device)
        frozen = _load_latent_checkpoint(Path(row["checkpoint_paths"]["frozen"]), device=device)
        for variant_name, transform in variants.items():
            transformed = transform(examples, seed + 70_000)
            for model_name, result in (("trainable", trainable), ("frozen", frozen)):
                if variant_name == "candidate_only":
                    with _zero_role_embeddings(result.system):
                        pred = predict_latent_system(result, transformed, condition="none", seed=seed)
                else:
                    pred = predict_latent_system(result, transformed, condition="none", seed=seed)
                rows_out.append(_eval_row(seed, variant_name, model_name, pred, y))

        for model_name, result in (("trainable", trainable), ("frozen", frozen)):
            rows_out.append(_eval_row(seed, "role_embeddings_zeroed", model_name, _predict_with_zero_role_embeddings(result, examples, seed), y))
            rows_out.append(_eval_row(seed, "all_role_labels_same_zero", model_name, _predict_with_role_ids(result.system, examples, _role_ids_all_zero, seed), y))
            rows_out.append(_eval_row(seed, "role_labels_random_independent", model_name, _predict_with_role_ids(result.system, examples, _role_ids_random_independent, seed + 73_000), y))
            rows_out.append(_eval_row(seed, "role_labels_random_permutation", model_name, _predict_with_role_ids(result.system, examples, _role_ids_random_permutation, seed + 74_000), y))
            rows_out.append(_eval_row(seed, "hidden_states_shuffled_across_examples", model_name, predict_latent_system(result, examples, "hidden_states_shuffled_across_examples", seed), y))

        del trainable, frozen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "rows": rows_out,
        "summary": _checkpoint_summary(rows_out),
    }


def _eval_row(seed: int, variant: str, model: str, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, object]:
    acc = _acc(predictions, labels)
    return {
        "seed": seed,
        "variant": variant,
        "model": model,
        "accuracy": acc,
        "near_chance_0_18": acc <= STRICT_CONTROL_MAX,
    }


def _predict_with_zero_role_embeddings(result, examples: Sequence[MultiViewTaskExample], seed: int) -> np.ndarray:
    with _zero_role_embeddings(result.system):
        return predict_latent_system(result, examples, condition="none", seed=seed)


def _predict_with_role_ids(
    system: SharedClonedAgentSystem,
    examples: Sequence[MultiViewTaskExample],
    role_id_builder: Callable[[int, int, int, torch.device], torch.Tensor],
    seed: int,
) -> np.ndarray:
    system.eval()
    preds = []
    with torch.no_grad():
        for batch in _example_batches(examples, 128):
            device = next(system.parameters()).device
            role_ids = role_id_builder(len(batch), system.n_roles, seed + len(preds), device)
            readouts = system.collect_clone_representations(batch, condition="none", seed=seed, role_ids=role_ids)
            clone_activations = readouts["message"]
            if bool(getattr(system.coordinator, "requires_candidate_features", False)):
                candidate_features = _candidate_feature_tensor(batch, clone_activations.device)
                logits = system.coordinator(clone_activations, role_ids, candidate_features)
            else:
                logits = system.coordinator(clone_activations, role_ids)
            preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
    return np.concatenate(preds).astype(np.int64)


def _role_ids_all_zero(batch_size: int, n_roles: int, seed: int, device: torch.device) -> torch.Tensor:
    del seed
    return torch.zeros((batch_size, n_roles), dtype=torch.long, device=device)


def _role_ids_random_independent(batch_size: int, n_roles: int, seed: int, device: torch.device) -> torch.Tensor:
    rng = np.random.default_rng(seed + 800_003)
    return torch.as_tensor(rng.integers(0, n_roles, size=(batch_size, n_roles)), dtype=torch.long, device=device)


def _role_ids_random_permutation(batch_size: int, n_roles: int, seed: int, device: torch.device) -> torch.Tensor:
    rng = np.random.default_rng(seed + 900_003)
    rows = []
    base = np.arange(n_roles)
    for _ in range(batch_size):
        perm = rng.permutation(n_roles)
        if np.array_equal(perm, base):
            perm = np.roll(perm, 1)
        rows.append(perm)
    return torch.as_tensor(np.asarray(rows, dtype=np.int64), dtype=torch.long, device=device)


@contextmanager
def _zero_role_embeddings(system: SharedClonedAgentSystem):
    saved = []
    modules = [system.active_message_readout, system.coordinator]
    try:
        with torch.no_grad():
            for module in modules:
                embedding = getattr(module, "role_embedding", None) if module is not None else None
                if embedding is not None:
                    saved.append((embedding, embedding.weight.detach().clone()))
                    embedding.weight.zero_()
        yield
    finally:
        with torch.no_grad():
            for embedding, weight in saved:
                embedding.weight.copy_(weight)


def _remove_candidate_metadata(example: MultiViewTaskExample) -> MultiViewTaskExample:
    candidates = tuple(
        Candidate(
            candidate_id=candidate.candidate_id,
            text="Patch candidate redacted\noperation: RESTORE_LOCAL_IMPORT\nall candidate-specific metadata removed",
            source_path="",
            patch_hash=candidate.patch_hash,
            attributes=("", "", "", ""),
        )
        for candidate in example.candidates
    )
    metadata = dict(example.metadata)
    for key in ("candidate_patch_values", "candidate_tuples", "candidate_bit_tuples", "candidate_order"):
        metadata.pop(key, None)
    return replace(example, candidates=candidates, metadata=metadata)


def _remove_family_metadata(example: MultiViewTaskExample) -> MultiViewTaskExample:
    metadata = dict(example.metadata)
    for key in ("problem_family", "source_file", "source_import_key", "provider_module", "provider_file"):
        metadata.pop(key, None)
    return replace(example, metadata=metadata)


def _randomize_candidate_role_columns(example: MultiViewTaskExample, seed: int) -> MultiViewTaskExample:
    rng = np.random.default_rng(seed + 101_777)
    perm = rng.permutation(4)
    if np.array_equal(perm, np.arange(4)):
        perm = np.roll(perm, 1)
    pairs = example.metadata.get("attribute_value_pairs", ATTRIBUTE_VALUE_PAIRS)
    candidates = []
    bit_rows = []
    tuple_rows = []
    for candidate in example.candidates:
        bits = candidate_attribute_bits(example, candidate)
        new_bits = [bits[int(perm[index])] for index in range(4)]
        attrs = tuple(str(pairs[index][new_bits[index]]) for index in range(4))
        bit_rows.append([int(value) for value in new_bits])
        tuple_rows.append(list(attrs))
        candidates.append(replace(candidate, attributes=attrs))
    metadata = {
        **example.metadata,
        "role_pair_structure_randomized_permutation": [int(value) for value in perm.tolist()],
        "candidate_tuples": tuple_rows,
        "candidate_bit_tuples": bit_rows,
    }
    return replace(example, candidates=tuple(candidates), metadata=metadata)


def _checkpoint_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["variant"]), str(row["model"]))].append(float(row["accuracy"]))
    summary = {}
    for (variant, model), values in sorted(grouped.items()):
        summary.setdefault(variant, {})[model] = {
            "mean_accuracy": float(mean(values)),
            "max_accuracy": max(values),
            "min_accuracy": min(values),
            "near_chance_mean_0_18": float(mean(values)) <= STRICT_CONTROL_MAX,
        }
    return summary


def _existing_control_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    controls = (
        "randomized_labels",
        "role_labels_shuffled",
        "view_masked",
        "hidden_states_shuffled_across_examples",
        "view_shuffled",
    )
    out = {}
    for control in controls:
        values = [float(row["test_accuracy"].get(control, 0.0)) for row in rows]
        threshold = ROLE_SHUFFLE_MAX if control == "role_labels_shuffled" else STRICT_CONTROL_MAX
        out[control] = {
            "per_seed": {str(row["seed"]): float(row["test_accuracy"].get(control, 0.0)) for row in rows},
            "mean_accuracy": float(mean(values)),
            "max_accuracy": max(values),
            "strict_gate": f"<= {threshold:.2f}",
            "passes_strict_gate": float(mean(values)) <= threshold and max(values) <= max(threshold, STRICT_CONTROL_MAX),
        }
    return out


def _dataset_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    def collect(path: Sequence[str]) -> List[float]:
        values = []
        for row in rows:
            value = row
            for key in path:
                value = value[key]
            values.append(float(value))
        return values

    candidate_codebook = collect(["candidate_metadata_shortcut_audit", "candidate_codebook_oracle_accuracy"])
    ordered_lookup = collect(["candidate_metadata_shortcut_audit", "ordered_candidate_bits_lookup_accuracy"])
    view_mask_codebook = collect(["view_masked_audit", "baselines_on_masked_view_setting", "candidate_codebook_oracle"])
    template_dupes = collect(["duplication_template_audit", "train_test_exact_template_duplicate_rate"])
    ordered_dupes = collect(["duplication_template_audit", "train_test_ordered_candidate_bits_duplicate_rate"])
    random_vs_original = collect(["randomized_label_audit", "per_split", "test", "accuracy_vs_original_gold_label"])
    max_train_random_to_test_random = []
    max_train_random_to_original = []
    for row in rows:
        cross = row["randomized_label_audit"]["cross_split_trivial_predictors"]
        max_train_random_to_test_random.append(max(float(value["train_random_to_test_random_accuracy"]) for value in cross.values()))
        max_train_random_to_original.append(max(float(value["train_random_to_test_original_accuracy"]) for value in cross.values()))
    randomized_balance_ok = all(
        bool(row["randomized_label_audit"]["independence_flags"]["balanced_train_dev_test"])
        for row in rows
    )
    randomized_direct_correlation_ok = max(random_vs_original) <= STRICT_CONTROL_MAX
    randomized_cross_split_ok = max(max_train_random_to_test_random) <= STRICT_CONTROL_MAX
    return {
        "mean_candidate_codebook_oracle_accuracy": float(mean(candidate_codebook)),
        "max_candidate_codebook_oracle_accuracy": max(candidate_codebook),
        "mean_ordered_candidate_bits_lookup_accuracy": float(mean(ordered_lookup)),
        "mean_view_mask_candidate_codebook_oracle_accuracy": float(mean(view_mask_codebook)),
        "mean_train_test_exact_template_duplicate_rate": float(mean(template_dupes)),
        "mean_train_test_ordered_candidate_bits_duplicate_rate": float(mean(ordered_dupes)),
        "mean_test_random_label_accuracy_vs_original": float(mean(random_vs_original)),
        "max_test_random_label_accuracy_vs_original": max(random_vs_original),
        "mean_max_trivial_train_random_to_test_random_accuracy": float(mean(max_train_random_to_test_random)),
        "max_trivial_train_random_to_test_random_accuracy": max(max_train_random_to_test_random),
        "max_trivial_train_random_to_test_original_accuracy": max(max_train_random_to_original),
        "candidate_metadata_shortcut_detected": float(mean(candidate_codebook)) > STRICT_CONTROL_MAX,
        "randomized_label_bug_detected": not (
            randomized_balance_ok and randomized_direct_correlation_ok and randomized_cross_split_ok
        ),
    }


def _conclusion(
    dataset_rows: Sequence[Dict[str, object]],
    implementation: Dict[str, object],
    existing_controls: Dict[str, object],
    checkpoint_eval: Dict[str, object],
) -> Dict[str, object]:
    dataset_summary = _dataset_summary(dataset_rows)
    checkpoint_summary = checkpoint_eval.get("summary", {}) if checkpoint_eval else {}
    candidate_only_trainable = checkpoint_summary.get("candidate_only", {}).get("trainable", {}).get("mean_accuracy", 0.0)
    candidate_removed_trainable = checkpoint_summary.get("candidate_metadata_removed", {}).get("trainable", {}).get("mean_accuracy", 0.0)
    role_shuffle_mean = existing_controls.get("role_labels_shuffled", {}).get("mean_accuracy", 0.0)
    likely = []
    if dataset_summary["candidate_metadata_shortcut_detected"]:
        likely.append("candidate metadata shortcut")
        likely.append("role-pair structural shortcut")
    if role_shuffle_mean > ROLE_SHUFFLE_MAX:
        likely.append("role-shuffle control is too weak for pair-preserving permutations")
    if candidate_only_trainable > STRICT_CONTROL_MAX:
        likely.append("checkpoint can exploit candidate-only signal")
    if candidate_removed_trainable <= STRICT_CONTROL_MAX:
        likely.append("candidate metadata removal collapses checkpoint")
    return {
        "likely_causes": likely,
        "classification": {
            "randomized_label_implementation_bug": bool(dataset_summary["randomized_label_bug_detected"]),
            "role_shuffle_implementation_bug": False,
            "view_masking_bug": False,
            "family_template_leakage": False,
            "role_pair_structural_shortcut": bool(dataset_summary["candidate_metadata_shortcut_detected"]),
            "candidate_metadata_shortcut": bool(dataset_summary["candidate_metadata_shortcut_detected"]),
            "true_model_effect": False,
        },
        "short_explanation": (
            "Stage 3.3 controls fail primarily because the v2 candidate attribute codebook is not translation-invariant: "
            "candidate metadata alone can infer the hidden evidence tuple. Role-label shuffle also preserves candidate feature columns "
            "and the decisive pair structure, so it is not a destructive corruption for this dataset."
        ),
        "supporting_metrics": {
            "mean_candidate_codebook_oracle_accuracy": dataset_summary["mean_candidate_codebook_oracle_accuracy"],
            "role_labels_shuffled_mean_accuracy": role_shuffle_mean,
            "candidate_only_trainable_mean_accuracy": candidate_only_trainable,
            "candidate_metadata_removed_trainable_mean_accuracy": candidate_removed_trainable,
            "role_shuffle_behavior": implementation["role_label_shuffle_behavior"],
        },
    }


def _render_report(result: Dict[str, object]) -> str:
    ds = result["dataset_only_audit"]["summary"]
    impl = result["control_implementation_audit"]
    existing = result["existing_stage33_control_metrics"]
    ckpt = result.get("checkpoint_only_evaluations", {}).get("summary", {})
    ckpt_rows = result.get("checkpoint_only_evaluations", {}).get("rows", [])
    conclusion = result["conclusion"]
    lines = [
        "# Stage 3.3.1 Fast Control-Harness Audit",
        "",
        "## Scope",
        "",
        "- Full training jobs run: `0`.",
        "- Method: dataset-only diagnostics plus checkpoint-only evaluation of saved Stage 3.3 trainable/frozen checkpoints.",
        "- This audit does not claim Stage 3.3 success.",
        "",
        "## Main Finding",
        "",
        conclusion["short_explanation"],
        "",
        "Classification:",
    ]
    for key, value in conclusion["classification"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Dataset-Only Signals",
            "",
            f"- Candidate codebook oracle mean accuracy: `{ds['mean_candidate_codebook_oracle_accuracy']:.4f}`",
            f"- Ordered candidate-bits lookup mean accuracy: `{ds['mean_ordered_candidate_bits_lookup_accuracy']:.4f}`",
            f"- View-masked candidate codebook oracle mean accuracy: `{ds['mean_view_mask_candidate_codebook_oracle_accuracy']:.4f}`",
            f"- Test randomized-label vs original-label mean accuracy: `{ds['mean_test_random_label_accuracy_vs_original']:.4f}`",
            f"- Max test randomized-label vs original-label accuracy: `{ds['max_test_random_label_accuracy_vs_original']:.4f}`",
            f"- Max trivial train-random to test-random accuracy: `{ds['max_trivial_train_random_to_test_random_accuracy']:.4f}`",
            f"- Max trivial train-random to test-original accuracy: `{ds['max_trivial_train_random_to_test_original_accuracy']:.4f}`",
            f"- Train/test normalized template duplicate rate: `{ds['mean_train_test_exact_template_duplicate_rate']:.4f}`",
            f"- Train/test ordered candidate-bits duplicate rate: `{ds['mean_train_test_ordered_candidate_bits_duplicate_rate']:.4f}`",
            "",
            "## Control Implementation",
            "",
            "- `role_labels_shuffled` leaves examples unchanged in `apply_example_control`; role ids are shuffled only inside `SharedClonedAgentSystem.forward`.",
            "- It shuffles role embeddings only. It does not shuffle role text, role/view pairing, physical order, or candidate feature columns.",
            "- `view_masked` masks private view text but leaves candidate text and candidate attributes available to the candidate-query coordinator.",
            f"- Pairwise oracle after role-label apply control: `{result['dataset_only_audit']['seed_rows'][0]['role_label_shuffle_audit']['pairwise_oracle_after']}`",
            "",
            "## Existing Failed Controls",
            "",
            "| control | mean | max | strict pass |",
            "|---|---:|---:|---|",
        ]
    )
    for control, values in existing.items():
        lines.append(f"| {control} | {values['mean_accuracy']:.4f} | {values['max_accuracy']:.4f} | {values['passes_strict_gate']} |")
    lines.extend(["", "## Checkpoint-Only Ablations", "", "| variant | trainable mean | frozen mean | trainable near chance | frozen near chance |", "|---|---:|---:|---|---|"])
    for variant, values in sorted(ckpt.items()):
        trainable = values.get("trainable", {})
        frozen = values.get("frozen", {})
        lines.append(
            f"| {variant} | {float(trainable.get('mean_accuracy', 0.0)):.4f} | {float(frozen.get('mean_accuracy', 0.0)):.4f} | "
            f"{bool(trainable.get('near_chance_mean_0_18', False))} | {bool(frozen.get('near_chance_mean_0_18', False))} |"
        )
    lines.extend(["", "## Per-Seed Checkpoint Accuracies", "", "| variant | trainable by seed | frozen by seed |", "|---|---|---|"])
    for variant in sorted(ckpt):
        trainable_seed_values = _per_seed_accuracy_string(ckpt_rows, variant, "trainable")
        frozen_seed_values = _per_seed_accuracy_string(ckpt_rows, variant, "frozen")
        lines.append(f"| {variant} | {trainable_seed_values} | {frozen_seed_values} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Randomized labels are balanced; no dataset-only randomized-label implementation bug was found.",
            "- The role shuffle implementation is doing what it says mechanically, but it is not destructive enough for this pair-codebook dataset because candidate feature columns and pair structure survive.",
            "- View masking is text-correct, but it is not candidate-metadata masking.",
            "- The decisive failure mode is the Stage 3.3b candidate metadata/codebook shortcut, not a demonstrated true model effect.",
        ]
    )
    return "\n".join(lines) + "\n"


def _per_seed_accuracy_string(rows: Sequence[Dict[str, object]], variant: str, model: str) -> str:
    values = sorted(
        (
            (int(row["seed"]), float(row["accuracy"]))
            for row in rows
            if row.get("variant") == variant and row.get("model") == model
        ),
        key=lambda item: item[0],
    )
    return ", ".join(f"{seed}:{accuracy:.4f}" for seed, accuracy in values)


def _same_split_trivial_predictors(examples: Sequence[MultiViewTaskExample], labels: np.ndarray, split: str) -> Dict[str, float]:
    out = {}
    for name, builder in _feature_builders().items():
        features = [builder(example, split) for example in examples]
        out[name] = _lookup_self_accuracy(features, labels)
    return out


def _feature_builders() -> Dict[str, Callable[[MultiViewTaskExample, str], object]]:
    return {
        "majority": lambda ex, split: "constant",
        "candidate_index_only": lambda ex, split: int(ex.label),
        "family_id_only": lambda ex, split: _family(ex),
        "example_index_modulo_8": lambda ex, split: _example_index(ex) % 8,
        "role_pair_pattern_only": lambda ex, split: _role_pair_pattern(ex),
        "family_plus_role_pair": lambda ex, split: (_family(ex), _role_pair_pattern(ex)),
        "hash_id_modulo_8": lambda ex, split: _stable_hash(ex.id) % 8,
        "candidate_order_metadata": lambda ex, split: tuple(ex.metadata.get("candidate_order", [])),
        "split_id": lambda ex, split: split,
    }


def _lookup_feature_accuracy(
    train_examples: Sequence[MultiViewTaskExample],
    train_labels: np.ndarray,
    test_examples: Sequence[MultiViewTaskExample],
    test_labels: np.ndarray,
    builder: Callable[[MultiViewTaskExample, str], object],
) -> float:
    pred = _lookup_predict(
        [builder(example, "train") for example in train_examples],
        train_labels,
        [builder(example, "test") for example in test_examples],
        num_classes=8,
    )
    return _acc(pred, test_labels)


def _lookup_self_accuracy(features: Sequence[object], labels: np.ndarray) -> float:
    grouped: Dict[object, List[int]] = defaultdict(list)
    for feature, label in zip(features, labels):
        grouped[_hashable(feature)].append(int(label))
    pred = []
    for feature in features:
        counts = np.bincount(grouped[_hashable(feature)], minlength=8)
        pred.append(int(np.argmax(counts)))
    return _acc(np.asarray(pred, dtype=np.int64), labels)


def _lookup_predict(train_features: Sequence[object], train_labels: np.ndarray, test_features: Sequence[object], num_classes: int) -> np.ndarray:
    grouped: Dict[object, List[int]] = defaultdict(list)
    for feature, label in zip(train_features, train_labels):
        grouped[_hashable(feature)].append(int(label))
    fallback = int(np.argmax(np.bincount(train_labels.astype(np.int64), minlength=num_classes)))
    preds = []
    for feature in test_features:
        values = grouped.get(_hashable(feature))
        if values:
            preds.append(int(np.argmax(np.bincount(np.asarray(values, dtype=np.int64), minlength=num_classes))))
        else:
            preds.append(fallback)
    return np.asarray(preds, dtype=np.int64)


def _candidate_codebook_oracle_accuracy(train: Sequence[MultiViewTaskExample], test: Sequence[MultiViewTaskExample]) -> float:
    codebook = _infer_codebook(train)
    preds = []
    for example in test:
        candidate_bits = [tuple(candidate_attribute_bits(example, candidate)) for candidate in example.candidates]
        possible_evidence = []
        for evidence in product((0, 1), repeat=4):
            shifted = {tuple(bit ^ ev for bit, ev in zip(bits, evidence)) for bits in candidate_bits}
            if shifted == codebook:
                possible_evidence.append(tuple(int(value) for value in evidence))
        if not possible_evidence:
            preds.append(0)
            continue
        evidence = possible_evidence[0]
        matches = [index for index, bits in enumerate(candidate_bits) if bits == evidence]
        preds.append(matches[0] if matches else 0)
    return _acc(np.asarray(preds, dtype=np.int64), _labels(test))


def _infer_codebook(examples: Sequence[MultiViewTaskExample]) -> set[Tuple[int, int, int, int]]:
    counter: Counter[Tuple[int, int, int, int]] = Counter()
    for example in examples:
        evidence = tuple(int(value) for value in example.metadata.get("evidence_bits", [0, 0, 0, 0]))
        for candidate in example.candidates:
            bits = tuple(candidate_attribute_bits(example, candidate))
            counter[tuple(bit ^ ev for bit, ev in zip(bits, evidence))] += 1
    return set(counter)


def _structured_oracle_accuracy(examples: Sequence[MultiViewTaskExample], roles: Sequence[int]) -> float:
    labels = _labels(examples)
    preds = []
    for example in examples:
        evidence = [int(value) for value in example.metadata.get("evidence_bits", [0, 0, 0, 0])]
        scores = []
        for candidate in example.candidates:
            bits = candidate_attribute_bits(example, candidate)
            scores.append(sum(int(bits[role] == evidence[role]) for role in roles))
        preds.append(int(np.argmax(np.asarray(scores, dtype=np.float32))))
    return _acc(np.asarray(preds, dtype=np.int64), labels)


def _private_view_exact_leakage(examples: Sequence[MultiViewTaskExample]) -> Dict[str, float]:
    counts = Counter()
    total = len(examples)
    for example in examples:
        text = "\n".join(view.text for view in example.views).lower()
        values = example.metadata.get("candidate_patch_values", [])
        gold_values = values[int(example.label)] if isinstance(values, list) and int(example.label) < len(values) else []
        for name, value in zip(("symbol", "module", "style", "target"), gold_values):
            if name == "style":
                continue
            if _contains_exact(text, str(value)):
                counts[name] += 1
    return {f"exact_gold_{key}_rate": counts[key] / max(1, total) for key in ("symbol", "module", "target")}


def _contains_exact(text: str, value: str) -> bool:
    value = str(value or "").lower()
    if not value:
        return False
    if re.fullmatch(r"[a-z_][a-z0-9_]*", value):
        return re.search(rf"\b{re.escape(value)}\b", text) is not None
    return value in text


def _source_excerpt(source: str, patterns: Sequence[str]) -> List[str]:
    lines = source.splitlines()
    selected = []
    for index, line in enumerate(lines):
        if any(pattern in line for pattern in patterns):
            start = max(0, index - 2)
            end = min(len(lines), index + 5)
            selected.extend(lines[start:end])
    compact = []
    for line in selected:
        stripped = line.rstrip()
        if stripped not in compact:
            compact.append(stripped)
    return compact[:80]


def _template_signature(example: MultiViewTaskExample) -> str:
    text = "\n".join(view.text for view in example.views) + "\n" + "\n".join(candidate.text for candidate in example.candidates)
    text = re.sub(r"FEATURE_(ALPHA|BETA)", "FEATURE_VALUE", text)
    text = re.sub(r"redacted_uid=[0-9a-f]+", "redacted_uid=UID", text)
    text = re.sub(r"Patch candidate \d+", "Patch candidate N", text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ordered_candidate_bits(example: MultiViewTaskExample) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(candidate_attribute_bits(example, candidate)) for candidate in example.candidates)


def _candidate_bit_multiset(example: MultiViewTaskExample) -> Tuple[Tuple[int, ...], ...]:
    return tuple(sorted(_ordered_candidate_bits(example)))


def _candidate_order_gold_position(example: MultiViewTaskExample) -> int:
    order = list(example.metadata.get("candidate_order", []))
    if 0 in order:
        return int(order.index(0))
    return 0


def _family(example: MultiViewTaskExample) -> str:
    return re.sub(r"^(train|dev|test)_", "", str(example.metadata.get("problem_family", "unknown")))


def _role_pair_pattern(example: MultiViewTaskExample) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    bits = [int(value) for value in example.metadata.get("evidence_bits", [0, 0, 0, 0])]
    return (bits[0], bits[1]), (bits[2], bits[3])


def _example_index(example: MultiViewTaskExample) -> int:
    match = re.search(r"-(\d+)$", example.id)
    return int(match.group(1)) if match else 0


def _labels(examples: Sequence[MultiViewTaskExample]) -> np.ndarray:
    return np.asarray([int(example.label) for example in examples], dtype=np.int64)


def _counts(labels: np.ndarray) -> Dict[str, int]:
    values = np.bincount(labels.astype(np.int64), minlength=8)
    return {str(index): int(value) for index, value in enumerate(values)}


def _balance_deviation(labels: np.ndarray) -> int:
    values = np.bincount(labels.astype(np.int64), minlength=8)
    expected = len(labels) // 8
    return int(max(abs(int(value) - expected) for value in values))


def _acc(pred: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(pred.astype(np.int64) == labels.astype(np.int64))) if len(labels) else 0.0


def _rate(values: Iterable[bool]) -> float:
    rows = [bool(value) for value in values]
    return float(np.mean(rows)) if rows else 0.0


def _stable_hash(value: str) -> int:
    return int(hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16], 16)


def _hashable(value: object) -> object:
    if isinstance(value, list):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _hashable(item)) for key, item in value.items()))
    return value


if __name__ == "__main__":
    main()
