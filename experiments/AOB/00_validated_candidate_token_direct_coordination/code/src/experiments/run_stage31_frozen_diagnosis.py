from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from src.coordinators.latent_coordination import (
    LatentCoordinatorConfig,
    make_candidate_query_coordinator,
    make_latent_coordinator,
)
from src.datasets.multiview_code_patch_selection import (
    Candidate,
    MultiViewCodePatchDatasetConfig,
    MultiViewTaskExample,
    View,
    build_multiview_code_patch_splits,
    example_oracle_metadata,
    format_clone_prompt,
)
from src.experiments.real_shared_weight_latent_coordination import (
    FitResult,
    MessageChannelConfig,
    SharedClonedAgentSystem,
    SharedTransformerAgent,
    SharedTransformerAgentConfig,
    predict_latent_system,
    _text_tokens,
)


RESULTS_PATH = Path("results/stage3_gpu_hard_validation_results.json")
OUTPUT_PATH = Path("results/stage31_frozen_diagnosis_results.json")
REPORT_PATH = Path("reports/STAGE31_FROZEN_DIAGNOSIS.md")
VARIANTS = (
    "symbol_redacted_views",
    "candidate_name_redacted_views",
    "module_path_redacted_views",
    "cross_file_evidence_only",
    "two_hop_evidence",
    "same_package_distractors",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3.1 frozen-comparator diagnosis.")
    parser.add_argument("--results", default=str(RESULTS_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--report", default=str(REPORT_PATH))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention-seeds", nargs="*", type=int, default=[2, 7])
    parser.add_argument("--max-attention-examples", type=int, default=4)
    args = parser.parse_args()

    device = _resolve_device(args.device)
    stage3 = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rows = sorted(stage3.get("stage3_rows", []), key=lambda row: int(row["seed"]))
    if not rows:
        raise RuntimeError(f"no stage3_rows found in {args.results}")

    seed_rows = []
    all_family_rows = []
    variant_rows = []
    attention_rows = []
    for row in rows:
        seed = int(row["seed"])
        dataset_config = MultiViewCodePatchDatasetConfig(**row["dataset_config"])
        splits = build_multiview_code_patch_splits(dataset_config, seed=seed, repo_root=".")
        test_examples = splits["test"]
        train_examples = splits["train"]
        frozen = _load_latent_checkpoint(Path(row["checkpoint_paths"]["frozen"]), device=device)
        trainable = _load_latent_checkpoint(Path(row["checkpoint_paths"]["trainable"]), device=device)

        frozen_predictions = predict_latent_system(frozen, test_examples, condition="none", seed=seed + 2_000_000)
        trainable_predictions = predict_latent_system(trainable, test_examples, condition="none", seed=seed + 2_000_000)
        labels = _labels(test_examples)

        family_rows = _per_family_rows(seed, test_examples, frozen_predictions, labels)
        all_family_rows.extend(family_rows)
        lexical_predictions = _candidate_lexical_overlap_predictions(test_examples)
        frequency_predictions = _static_import_frequency_predictions(train_examples, test_examples)
        exact_audit = _exact_symbol_leakage_audit(test_examples)
        candidate_audit = _candidate_string_leakage_audit(test_examples)
        seed_rows.append(
            {
                "seed": seed,
                "trainable_accuracy": _accuracy(trainable_predictions, labels),
                "frozen_accuracy": _accuracy(frozen_predictions, labels),
                "delta": _accuracy(trainable_predictions, labels) - _accuracy(frozen_predictions, labels),
                "candidate_lexical_overlap_accuracy": _accuracy(lexical_predictions, labels),
                "static_import_frequency_accuracy": _accuracy(frequency_predictions, labels),
                "exact_symbol_leakage": exact_audit,
                "candidate_string_leakage": candidate_audit,
                "frozen_correct_trainable_wrong": int(np.sum((frozen_predictions == labels) & (trainable_predictions != labels))),
                "trainable_correct_frozen_wrong": int(np.sum((trainable_predictions == labels) & (frozen_predictions != labels))),
                "both_wrong": int(np.sum((trainable_predictions != labels) & (frozen_predictions != labels))),
                "family_rows": family_rows,
            }
        )

        for variant in VARIANTS:
            variant_test = [_apply_variant(example, variant) for example in test_examples]
            variant_labels = _labels(variant_test)
            variant_frozen = predict_latent_system(frozen, variant_test, condition="none", seed=seed + 2_000_000)
            variant_trainable = predict_latent_system(trainable, variant_test, condition="none", seed=seed + 2_000_000)
            variant_lexical = _candidate_lexical_overlap_predictions(variant_test)
            variant_exact = _exact_symbol_leakage_audit(variant_test)
            variant_candidate = _candidate_string_leakage_audit(variant_test)
            variant_rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "probe_type": "checkpoint_stress_test_without_retraining",
                    "trainable_accuracy": _accuracy(variant_trainable, variant_labels),
                    "frozen_accuracy": _accuracy(variant_frozen, variant_labels),
                    "delta": _accuracy(variant_trainable, variant_labels) - _accuracy(variant_frozen, variant_labels),
                    "candidate_lexical_overlap_accuracy": _accuracy(variant_lexical, variant_labels),
                    "exact_symbol_leakage_any_rate": variant_exact["any_gold_value_rate"],
                    "candidate_string_near_verbatim_rate": variant_candidate["near_verbatim_candidate_rate"],
                }
            )

        if seed in set(args.attention_seeds):
            attention_rows.extend(
                _attention_inspection_rows(
                    seed=seed,
                    examples=test_examples,
                    frozen=frozen,
                    trainable=trainable,
                    frozen_predictions=frozen_predictions,
                    trainable_predictions=trainable_predictions,
                    max_examples=int(args.max_attention_examples),
                )
            )

        del frozen, trainable
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = {
        "metadata": {
            "stage": "stage3.1_frozen_comparator_diagnosis",
            "source_results": str(args.results),
            "device": device,
            "architecture_changes": "none",
            "locked_architecture": "topk_attention_no_head",
            "variant_probe_scope": "Existing checkpoints evaluated on transformed test examples; variants are not retrained hard-validation runs.",
        },
        "seed_rows": seed_rows,
        "per_family_frozen_accuracy": all_family_rows,
        "summary": _summary(seed_rows, variant_rows, all_family_rows),
        "variant_probe_rows": variant_rows,
        "attention_inspection": attention_rows,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(results), encoding="utf-8")


def _resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _load_latent_checkpoint(path: Path, device: str) -> FitResult:
    checkpoint = torch.load(path, map_location=device)
    agent_config = SharedTransformerAgentConfig(**_tupleize_config(checkpoint["agent_config"]))
    message_config = MessageChannelConfig(**_tupleize_config(checkpoint["message_config"]))
    coordinator_config = LatentCoordinatorConfig(**_tupleize_config(checkpoint["coordinator_config"]))

    agent = SharedTransformerAgent(agent_config).to(device)
    agent.configure_trainable(bool(checkpoint["trainable_agent"]))
    coordinator_input_dim = int(message_config.message_dim) if message_config.use_message_head else int(agent.output_dim)
    effective_config = replace(coordinator_config, input_dim=coordinator_input_dim)
    candidate_query_families = {
        "candidate_query_cross_attention",
        "candidate_token_cross_attention",
        "bilinear_candidate",
        "contrastive_candidate",
        "global_then_candidate",
        "two_round_message_passing",
    }
    if message_config.coordinator_family in candidate_query_families:
        family = "cross_attention" if message_config.coordinator_family == "candidate_query_cross_attention" else message_config.coordinator_family
        coordinator = make_candidate_query_coordinator(
            n_roles=int(checkpoint["n_roles"]),
            candidate_feature_dim=4,
            config=replace(effective_config, family=family),
        ).to(device)
    elif message_config.coordinator_family == "latent":
        coordinator = make_latent_coordinator(
            n_roles=int(checkpoint["n_roles"]),
            num_classes=int(checkpoint["num_classes"]),
            config=effective_config,
        ).to(device)
    else:
        raise ValueError(f"unknown coordinator family in checkpoint: {message_config.coordinator_family}")

    system = SharedClonedAgentSystem(
        agent,
        coordinator,
        n_roles=int(checkpoint["n_roles"]),
        visible_explicit_evidence=bool(checkpoint.get("visible_explicit_evidence", False)),
        message_config=message_config,
    ).to(device)
    system.load_state_dict(checkpoint["system_state_dict"])
    system.eval()
    return FitResult(
        method=str(checkpoint["method"]),
        condition=str(checkpoint["condition"]),
        agent=agent,
        coordinator=coordinator,
        system=system,
        trainable_agent=bool(checkpoint["trainable_agent"]),
        param_count=int(checkpoint["param_count"]),
        audit=dict(checkpoint.get("audit", {})),
        history=list(checkpoint.get("history", [])),
    )


def _tupleize_config(values: Dict[str, object]) -> Dict[str, object]:
    tuple_fields = {
        "lora_target_modules",
        "active_message_layers",
    }
    return {key: tuple(value) if key in tuple_fields and isinstance(value, list) else value for key, value in values.items()}


def _labels(examples: Sequence[MultiViewTaskExample]) -> np.ndarray:
    return np.asarray([int(example.label) for example in examples], dtype=np.int64)


def _accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(predictions.astype(np.int64) == labels.astype(np.int64))) if len(labels) else 0.0


def _per_family_rows(seed: int, examples: Sequence[MultiViewTaskExample], predictions: np.ndarray, labels: np.ndarray) -> List[Dict[str, object]]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        grouped[str(example_oracle_metadata(example).get("problem_family", "unknown"))].append(index)
    rows = []
    for family, indices in sorted(grouped.items()):
        idx = np.asarray(indices, dtype=np.int64)
        rows.append(
            {
                "seed": seed,
                "problem_family": family,
                "n": int(len(indices)),
                "frozen_accuracy": _accuracy(predictions[idx], labels[idx]),
            }
        )
    return rows


def _candidate_lexical_overlap_predictions(examples: Sequence[MultiViewTaskExample]) -> np.ndarray:
    predictions = []
    for example in examples:
        view_text = "\n".join(view.text for view in example.views).lower()
        view_tokens = Counter(_text_tokens(view_text))
        scores = []
        for index, candidate in enumerate(example.candidates):
            values = _candidate_values(example, index)
            symbol, module, _style, target_file = values
            exact_score = 0.0
            for value, weight in ((symbol, 4.0), (module, 4.0), (target_file, 2.0)):
                if value and str(value).lower() in view_text:
                    exact_score += weight
            candidate_tokens = _candidate_identity_tokens(values)
            overlap = sum(view_tokens.get(token, 0) for token in candidate_tokens)
            scores.append(exact_score + float(overlap))
        predictions.append(_stable_argmax(scores))
    return np.asarray(predictions, dtype=np.int64)


def _candidate_identity_tokens(values: Tuple[str, str, str, str]) -> List[str]:
    tokens: List[str] = []
    for value in values:
        tokens.extend(_text_tokens(str(value)))
    return [token for token in tokens if token not in {"from", "import", "py"}]


def _static_import_frequency_predictions(
    train_examples: Sequence[MultiViewTaskExample],
    test_examples: Sequence[MultiViewTaskExample],
) -> np.ndarray:
    repo_pair = Counter()
    repo_module = Counter()
    repo_symbol = Counter()
    family_pair = Counter()
    for example in train_examples:
        family = _normalized_family(example)
        symbol, module, _style, _target = _candidate_values(example, int(example.label))
        repo_pair[(module, symbol)] += 1
        repo_module[module] += 1
        repo_symbol[symbol] += 1
        family_pair[(family, module, symbol)] += 1

    predictions = []
    for example in test_examples:
        family = _normalized_family(example)
        scores = []
        for index, _candidate in enumerate(example.candidates):
            symbol, module, _style, _target = _candidate_values(example, index)
            scores.append(
                1000.0 * family_pair[(family, module, symbol)]
                + 100.0 * repo_pair[(module, symbol)]
                + 10.0 * repo_module[module]
                + repo_symbol[symbol]
            )
        predictions.append(_stable_argmax(scores))
    return np.asarray(predictions, dtype=np.int64)


def _normalized_family(example: MultiViewTaskExample) -> str:
    family = str(example_oracle_metadata(example).get("problem_family", "unknown"))
    return re.sub(r"^(train|dev|test)_", "", family)


def _stable_argmax(scores: Sequence[float]) -> int:
    best_score = max(float(score) for score in scores)
    for index, score in enumerate(scores):
        if float(score) == best_score:
            return int(index)
    return 0


def _candidate_values(example: MultiViewTaskExample, index: int) -> Tuple[str, str, str, str]:
    values = example_oracle_metadata(example).get("candidate_patch_values")
    if isinstance(values, list) and index < len(values) and isinstance(values[index], list) and len(values[index]) >= 4:
        return tuple(str(value) for value in values[index][:4])  # type: ignore[return-value]
    return _parse_candidate_values(example.candidates[index])


def _parse_candidate_values(candidate: Candidate) -> Tuple[str, str, str, str]:
    module = ""
    symbol = ""
    target = str(candidate.source_path)
    match = re.search(r"from\s+([A-Za-z_][A-Za-z0-9_.]*)\s+import\s+([A-Za-z_][A-Za-z0-9_]*)", candidate.text)
    if match:
        module, symbol = match.group(1), match.group(2)
    target_match = re.search(r"diff --git a/(.*?) b/", candidate.text)
    if target_match:
        target = target_match.group(1)
    return symbol, module, "direct_from_import", target


def _exact_symbol_leakage_audit(examples: Sequence[MultiViewTaskExample]) -> Dict[str, object]:
    counts = Counter()
    role_counts = Counter()
    total = len(examples)
    examples_with_any = 0
    examples_with_all = 0
    for example in examples:
        gold = _candidate_values(example, int(example.label))
        keys = {
            "symbol": gold[0],
            "module": gold[1],
            "target_file": gold[3],
        }
        view_texts = [view.text.lower() for view in example.views]
        found_any = False
        found_all = True
        for key, value in keys.items():
            value_lower = str(value).lower()
            hit_roles = [role_index for role_index, text in enumerate(view_texts) if value_lower and value_lower in text]
            if hit_roles:
                counts[key] += 1
                found_any = True
                for role_index in hit_roles:
                    role_counts[f"role_{role_index}_{key}"] += 1
            else:
                found_all = False
        if found_any:
            examples_with_any += 1
        if found_all:
            examples_with_all += 1
    return {
        "examples": int(total),
        "any_gold_value_rate": examples_with_any / max(1, total),
        "all_gold_value_rate": examples_with_all / max(1, total),
        "symbol_rate": counts["symbol"] / max(1, total),
        "module_rate": counts["module"] / max(1, total),
        "target_file_rate": counts["target_file"] / max(1, total),
        "role_value_rates": {key: value / max(1, total) for key, value in sorted(role_counts.items())},
    }


def _candidate_string_leakage_audit(examples: Sequence[MultiViewTaskExample]) -> Dict[str, object]:
    total = 0
    exact_candidate = 0
    near_verbatim_candidate = 0
    exact_import_line = 0
    gold_exact_import_line_examples = 0
    gold_near_examples = 0
    max_ratios = []
    for example in examples:
        view_text = "\n".join(view.text for view in example.views).lower()
        gold_near = False
        gold_line_hit = False
        for index, candidate in enumerate(example.candidates):
            total += 1
            candidate_text = candidate.text.lower()
            import_line = _candidate_import_line(candidate).lower()
            if candidate_text and candidate_text in view_text:
                exact_candidate += 1
            if import_line and import_line in view_text:
                exact_import_line += 1
                if index == int(example.label):
                    gold_line_hit = True
            ratios = [SequenceMatcher(None, candidate_text, view.text.lower()).ratio() for view in example.views]
            max_ratio = max(ratios) if ratios else 0.0
            max_ratios.append(max_ratio)
            if max_ratio >= 0.55 or (import_line and _near_line_hit(import_line, view_text)):
                near_verbatim_candidate += 1
                if index == int(example.label):
                    gold_near = True
        if gold_line_hit:
            gold_exact_import_line_examples += 1
        if gold_near:
            gold_near_examples += 1
    return {
        "candidate_instances": int(total),
        "exact_full_candidate_rate": exact_candidate / max(1, total),
        "exact_import_line_rate": exact_import_line / max(1, total),
        "near_verbatim_candidate_rate": near_verbatim_candidate / max(1, total),
        "gold_exact_import_line_example_rate": gold_exact_import_line_examples / max(1, len(examples)),
        "gold_near_verbatim_example_rate": gold_near_examples / max(1, len(examples)),
        "mean_max_candidate_view_similarity": float(mean(max_ratios)) if max_ratios else 0.0,
    }


def _candidate_import_line(candidate: Candidate) -> str:
    for line in candidate.text.splitlines():
        stripped = line.strip()
        if stripped.startswith("+ "):
            return stripped[2:].strip()
    return ""


def _near_line_hit(import_line: str, view_text: str) -> bool:
    for line in view_text.splitlines():
        if SequenceMatcher(None, import_line, line.strip()).ratio() >= 0.85:
            return True
    return False


def _apply_variant(example: MultiViewTaskExample, variant: str) -> MultiViewTaskExample:
    if variant == "same_package_distractors":
        return _same_package_distractor_example(example)

    values_by_candidate = [_candidate_values(example, index) for index in range(len(example.candidates))]
    gold = values_by_candidate[int(example.label)]
    symbols = sorted({values[0] for values in values_by_candidate if values[0]}, key=len, reverse=True)
    modules = sorted({values[1] for values in values_by_candidate if values[1]}, key=len, reverse=True)
    paths = sorted({values[3] for values in values_by_candidate if values[3]}, key=len, reverse=True)

    new_views = []
    for role_index, view in enumerate(example.views):
        text = view.text
        if variant == "symbol_redacted_views":
            text = _replace_exact_identifier(text, gold[0], "[SYMBOL_REDACTED]")
        elif variant == "candidate_name_redacted_views":
            for symbol in symbols:
                text = _replace_exact_identifier(text, symbol, "[CANDIDATE_NAME_REDACTED]")
        elif variant == "module_path_redacted_views":
            for module in modules:
                text = text.replace(module, "[MODULE_PATH_REDACTED]")
            for path in paths:
                text = text.replace(path, "[FILE_PATH_REDACTED]")
        elif variant == "cross_file_evidence_only":
            if role_index in {0, 3}:
                text = (
                    f"{view.role} artifact masked for cross-file-evidence-only variant.\n"
                    "Target-file usage and import-block text withheld."
                )
            else:
                text = _strip_direct_import_strings(text, symbols, modules, paths)
        elif variant == "two_hop_evidence":
            text = _two_hop_text(role_index, text, symbols, modules, paths)
        else:
            raise ValueError(f"unknown variant: {variant}")
        new_views.append(replace(view, text=text))
    return replace(example, views=tuple(new_views), metadata={**example.metadata, "stage31_variant": variant})


def _replace_exact_identifier(text: str, value: str, replacement: str) -> str:
    if not value:
        return text
    return re.sub(rf"\b{re.escape(value)}\b", replacement, text)


def _strip_direct_import_strings(text: str, symbols: Sequence[str], modules: Sequence[str], paths: Sequence[str]) -> str:
    for symbol in symbols:
        text = _replace_exact_identifier(text, symbol, "[NAME_REDACTED]")
    for module in modules:
        text = text.replace(module, "[MODULE_REDACTED]")
    for path in paths:
        text = text.replace(path, "[PATH_REDACTED]")
    text = "\n".join(line for line in text.splitlines() if "import " not in line.lower())
    return text


def _two_hop_text(role_index: int, text: str, symbols: Sequence[str], modules: Sequence[str], paths: Sequence[str]) -> str:
    text = _strip_direct_import_strings(text, symbols, modules, paths)
    if role_index == 0:
        return "Usage-site artifact with exact binding redacted:\n" + text
    if role_index == 1:
        return "Provider-source artifact with exact module/name redacted:\n" + text
    if role_index == 2:
        return "\n".join(line for line in text.splitlines() if "resolved provider module" not in line.lower())
    if role_index == 3:
        return "\n".join(line for line in text.splitlines() if "traceback file" not in line.lower() and "import block" not in line.lower())
    return text


def _same_package_distractor_example(example: MultiViewTaskExample) -> MultiViewTaskExample:
    gold_symbol, gold_module, _style, _target = _candidate_values(example, int(example.label))
    parts = gold_module.split(".")
    package_prefix = ".".join(parts[:-1]) if len(parts) > 1 else gold_module
    new_candidates = []
    new_patch_values = []
    for index, candidate in enumerate(example.candidates):
        symbol, module, style, target = _candidate_values(example, index)
        if index != int(example.label) and package_prefix:
            tail = module.split(".")[-1] if module else f"distractor_{index}"
            module = f"{package_prefix}.{tail}"
        text = re.sub(r"from\s+([A-Za-z_][A-Za-z0-9_.]*)\s+import\s+", f"from {module} import ", candidate.text)
        new_candidates.append(replace(candidate, text=text, patch_hash=_sha256(text)))
        new_patch_values.append([symbol if index != int(example.label) else gold_symbol, module, style, target])
    return replace(
        example,
        candidates=tuple(new_candidates),
        metadata={**example.metadata, "candidate_patch_values": new_patch_values, "stage31_variant": "same_package_distractors"},
    )


def _sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _attention_inspection_rows(
    seed: int,
    examples: Sequence[MultiViewTaskExample],
    frozen: FitResult,
    trainable: FitResult,
    frozen_predictions: np.ndarray,
    trainable_predictions: np.ndarray,
    max_examples: int,
) -> List[Dict[str, object]]:
    labels = _labels(examples)
    candidate_indices: List[int] = []
    categories = [
        np.where((frozen_predictions == labels) & (trainable_predictions != labels))[0],
        np.where((trainable_predictions == labels) & (frozen_predictions != labels))[0],
        np.where((trainable_predictions == labels) & (frozen_predictions == labels))[0],
        np.where((trainable_predictions != labels) & (frozen_predictions != labels))[0],
    ]
    for indices in categories:
        for value in indices[: max(1, max_examples // len(categories))]:
            candidate_indices.append(int(value))
    if len(candidate_indices) < max_examples:
        for value in range(len(examples)):
            if value not in candidate_indices:
                candidate_indices.append(value)
            if len(candidate_indices) >= max_examples:
                break

    rows = []
    for example_index in candidate_indices[:max_examples]:
        example = examples[example_index]
        row = {
            "seed": seed,
            "example_id": example.id,
            "example_index": int(example_index),
            "label": int(example.label),
            "frozen_prediction": int(frozen_predictions[example_index]),
            "trainable_prediction": int(trainable_predictions[example_index]),
            "gold_patch_values": list(_candidate_values(example, int(example.label))),
            "roles": [],
        }
        for role_index, view in enumerate(example.views):
            row["roles"].append(
                {
                    "role_index": role_index,
                    "role": view.role,
                    "frozen_topk": _topk_selected_tokens(frozen.system, example, role_index),
                    "trainable_topk": _topk_selected_tokens(trainable.system, example, role_index),
                }
            )
        rows.append(row)
    return rows


def _topk_selected_tokens(system: SharedClonedAgentSystem, example: MultiViewTaskExample, role_index: int, k: int = 8) -> List[Dict[str, object]]:
    if system.active_message_readout is None:
        return []
    system.eval()
    prompt = format_clone_prompt(example, role_index)
    tokens = _text_tokens(prompt)
    max_length = int(system.shared_agent.max_length)
    with torch.no_grad():
        readouts = system.shared_agent.forward_texts_with_readouts(
            [prompt],
            use_msg_token=system.message_config.use_msg_token,
            msg_position=system.message_config.msg_position,
            selected_layer_ids=system.message_config.active_message_layers,
        )
        token_states = readouts["token_states"]
        token_mask = readouts["token_mask"].bool()
        role_ids = torch.as_tensor([role_index], dtype=torch.long, device=token_states.device)
        role = system.active_message_readout.role_embedding(role_ids)
        scores = system.active_message_readout.pool_scorer(token_states + role.unsqueeze(1)).squeeze(0).squeeze(-1)
        scores = scores.masked_fill(~token_mask.squeeze(0), -1e9)
        top_values, top_indices = torch.topk(scores, k=min(k, int(token_mask.sum().item())), dim=-1)
        weights = torch.softmax(top_values, dim=-1)
    out = []
    for score, weight, index in zip(top_values.detach().cpu().tolist(), weights.detach().cpu().tolist(), top_indices.detach().cpu().tolist()):
        base_pos = int(index) % max_length
        layer_pos = int(index) // max_length
        token = tokens[base_pos] if base_pos < len(tokens) else "[PAD]"
        out.append(
            {
                "position": int(base_pos),
                "layer_slot": int(layer_pos),
                "token": token,
                "score": float(score),
                "topk_weight": float(weight),
                "source_region": _prompt_region(prompt, base_pos),
            }
        )
    return out


def _prompt_region(prompt: str, token_position: int) -> str:
    tokens = _text_tokens(prompt)
    candidate_start = None
    for index, token in enumerate(tokens):
        if token == "patch" and index + 1 < len(tokens) and tokens[index + 1] == "candidates":
            candidate_start = index
            break
    if candidate_start is not None and token_position >= candidate_start:
        return "candidate_block"
    return "private_view"


def _summary(
    seed_rows: Sequence[Dict[str, object]],
    variant_rows: Sequence[Dict[str, object]],
    family_rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    variant_summary = []
    for variant in VARIANTS:
        rows = [row for row in variant_rows if row["variant"] == variant]
        if not rows:
            continue
        variant_summary.append(
            {
                "variant": variant,
                "probe_type": rows[0]["probe_type"],
                "mean_trainable_accuracy": float(mean(float(row["trainable_accuracy"]) for row in rows)),
                "mean_frozen_accuracy": float(mean(float(row["frozen_accuracy"]) for row in rows)),
                "mean_delta": float(mean(float(row["delta"]) for row in rows)),
                "mean_candidate_lexical_overlap_accuracy": float(mean(float(row["candidate_lexical_overlap_accuracy"]) for row in rows)),
                "mean_exact_symbol_leakage_any_rate": float(mean(float(row["exact_symbol_leakage_any_rate"]) for row in rows)),
                "mean_candidate_string_near_verbatim_rate": float(mean(float(row["candidate_string_near_verbatim_rate"]) for row in rows)),
            }
        )
    family_summary = []
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in family_rows:
        grouped[str(row["problem_family"])].append(float(row["frozen_accuracy"]))
    for family, values in sorted(grouped.items()):
        family_summary.append({"problem_family": family, "mean_frozen_accuracy": float(mean(values)), "n_seed_rows": len(values)})
    return {
        "mean_trainable_accuracy": float(mean(float(row["trainable_accuracy"]) for row in seed_rows)),
        "mean_frozen_accuracy": float(mean(float(row["frozen_accuracy"]) for row in seed_rows)),
        "mean_delta": float(mean(float(row["delta"]) for row in seed_rows)),
        "mean_candidate_lexical_overlap_accuracy": float(mean(float(row["candidate_lexical_overlap_accuracy"]) for row in seed_rows)),
        "mean_static_import_frequency_accuracy": float(mean(float(row["static_import_frequency_accuracy"]) for row in seed_rows)),
        "mean_exact_symbol_leakage_any_rate": float(mean(float(row["exact_symbol_leakage"]["any_gold_value_rate"]) for row in seed_rows)),
        "mean_exact_symbol_rate": float(mean(float(row["exact_symbol_leakage"]["symbol_rate"]) for row in seed_rows)),
        "mean_exact_module_rate": float(mean(float(row["exact_symbol_leakage"]["module_rate"]) for row in seed_rows)),
        "mean_exact_target_file_rate": float(mean(float(row["exact_symbol_leakage"]["target_file_rate"]) for row in seed_rows)),
        "mean_gold_exact_import_line_example_rate": float(mean(float(row["candidate_string_leakage"]["gold_exact_import_line_example_rate"]) for row in seed_rows)),
        "mean_gold_near_verbatim_example_rate": float(mean(float(row["candidate_string_leakage"]["gold_near_verbatim_example_rate"]) for row in seed_rows)),
        "variant_probe_summary": variant_summary,
        "family_summary": family_summary,
    }


def _render_report(results: Dict[str, object]) -> str:
    summary = results["summary"]
    seed_rows = results["seed_rows"]
    lines = [
        "# Stage 3.1 Frozen Comparator Diagnosis",
        "",
        "Architecture changes: none. The locked `topk_attention_no_head` checkpoints from Stage 3 hard validation were reloaded.",
        "",
        "Stage 3.1 acceptance status: **not passed**. The diagnosis found dataset leakage sufficient to explain the high frozen score, and the hardening variants below were stress tests of existing checkpoints rather than retrained acceptance runs.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| mean trainable accuracy | {summary['mean_trainable_accuracy']:.4f} |",
        f"| mean frozen accuracy | {summary['mean_frozen_accuracy']:.4f} |",
        f"| mean trainable-frozen delta | {summary['mean_delta']:.4f} |",
        f"| candidate lexical-overlap baseline | {summary['mean_candidate_lexical_overlap_accuracy']:.4f} |",
        f"| static import-frequency baseline | {summary['mean_static_import_frequency_accuracy']:.4f} |",
        f"| exact gold symbol/module/path appears in any private view | {summary['mean_exact_symbol_leakage_any_rate']:.4f} |",
        f"| exact gold symbol appears in private views | {summary['mean_exact_symbol_rate']:.4f} |",
        f"| exact gold module appears in private views | {summary['mean_exact_module_rate']:.4f} |",
        f"| exact gold target file appears in private views | {summary['mean_exact_target_file_rate']:.4f} |",
        f"| gold import line exact in private views | {summary['mean_gold_exact_import_line_example_rate']:.4f} |",
        f"| gold candidate near-verbatim in private views | {summary['mean_gold_near_verbatim_example_rate']:.4f} |",
        "",
        "## Acceptance Check",
        "",
        "| criterion | status | value |",
        "|---|---|---:|",
        f"| trainable >= 0.80 | {'pass' if summary['mean_trainable_accuracy'] >= 0.80 else 'fail'} | {summary['mean_trainable_accuracy']:.4f} |",
        f"| frozen <= 0.55 | {'pass' if summary['mean_frozen_accuracy'] <= 0.55 else 'fail'} | {summary['mean_frozen_accuracy']:.4f} |",
        f"| mean delta >= +0.25 | {'pass' if summary['mean_delta'] >= 0.25 else 'fail'} | {summary['mean_delta']:.4f} |",
        f"| exact-symbol leakage audit | {'pass' if summary['mean_exact_symbol_leakage_any_rate'] == 0.0 else 'fail'} | {summary['mean_exact_symbol_leakage_any_rate']:.4f} |",
        "| architecture unchanged | pass | none |",
        "| hardened variants | diagnostic only | retraining not run |",
        "",
        "## Per-Seed Diagnostics",
        "",
        "| seed | trainable | frozen | delta | lexical overlap | static frequency | any exact gold value | gold import line exact | frozen correct/trainable wrong | trainable correct/frozen wrong |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in seed_rows:
        lines.append(
            f"| {row['seed']} | {row['trainable_accuracy']:.4f} | {row['frozen_accuracy']:.4f} | {row['delta']:.4f} | "
            f"{row['candidate_lexical_overlap_accuracy']:.4f} | {row['static_import_frequency_accuracy']:.4f} | "
            f"{row['exact_symbol_leakage']['any_gold_value_rate']:.4f} | "
            f"{row['candidate_string_leakage']['gold_exact_import_line_example_rate']:.4f} | "
            f"{row['frozen_correct_trainable_wrong']} | {row['trainable_correct_frozen_wrong']} |"
        )

    lines.extend(
        [
            "",
            "## Per-Family Frozen Accuracy",
            "",
            "| seed | family | n | frozen accuracy |",
            "|---:|---|---:|---:|",
        ]
    )
    for row in results["per_family_frozen_accuracy"]:
        lines.append(f"| {row['seed']} | {row['problem_family']} | {row['n']} | {row['frozen_accuracy']:.4f} |")

    lines.extend(
        [
            "",
            "## Harder Variant Stress Tests",
            "",
            "These rows are checkpoint stress tests without retraining. They identify which hardening transformations remove the frozen shortcut; they are not Stage 3.1 acceptance runs.",
            "",
            "| variant | trainable | frozen | delta | lexical overlap | exact gold value | near-verbatim candidate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["variant_probe_summary"]:
        lines.append(
            f"| {row['variant']} | {row['mean_trainable_accuracy']:.4f} | {row['mean_frozen_accuracy']:.4f} | "
            f"{row['mean_delta']:.4f} | {row['mean_candidate_lexical_overlap_accuracy']:.4f} | "
            f"{row['mean_exact_symbol_leakage_any_rate']:.4f} | {row['mean_candidate_string_near_verbatim_rate']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Attention Inspection",
            "",
            "Top-k rows list the selected prompt tokens from the active readout scorer. `private_view` means the token occurs before the candidate block in the clone prompt.",
        ]
    )
    for row in results["attention_inspection"]:
        lines.extend(
            [
                "",
                f"### Seed {row['seed']} Example {row['example_id']}",
                "",
                f"- label: {row['label']}; frozen prediction: {row['frozen_prediction']}; trainable prediction: {row['trainable_prediction']}",
                f"- gold patch values: `{row['gold_patch_values']}`",
                "",
                "| role | model | selected tokens |",
                "|---|---|---|",
            ]
        )
        for role in row["roles"]:
            frozen_tokens = _format_attention_tokens(role["frozen_topk"])
            trainable_tokens = _format_attention_tokens(role["trainable_topk"])
            lines.append(f"| {role['role_index']} {role['role']} | frozen | {frozen_tokens} |")
            lines.append(f"| {role['role_index']} {role['role']} | trainable | {trainable_tokens} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The frozen comparator is not solving open-ended code repair. It is exploiting exact lexical evidence made available inside the private artifacts, especially the reference token, provider module/path, target file, and unredacted import-block text. The candidate lexical-overlap baseline measures this shortcut directly.",
            "",
            "Most conservative valid claim: the current dataset is invalid for the intended latent-coordination claim until exact symbol/module/path leakage and candidate-string leakage are removed and the hardening variants are retrained under the locked architecture.",
        ]
    )
    return "\n".join(lines) + "\n"


def _format_attention_tokens(tokens: Sequence[Dict[str, object]]) -> str:
    formatted = []
    for token in tokens:
        formatted.append(f"{token['position']}:{token['token']}:{token['source_region']}:{token['topk_weight']:.2f}")
    return "`" + "`, `".join(formatted) + "`"


if __name__ == "__main__":
    main()
