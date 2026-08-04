from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence

import numpy as np
import torch

from src.datasets.multiview_code_patch_selection import (
    MultiViewCodePatchDatasetConfig,
    MultiViewTaskExample,
    View,
    build_multiview_code_patch_splits,
    candidate_attribute_bits,
)
from src.experiments.real_shared_weight_latent_coordination import predict_latent_system
from src.experiments.run_stage31_frozen_diagnosis import _load_latent_checkpoint, _topk_selected_tokens


DEFAULT_RESULTS = "results/stage32_hardened_real_code_results.json"
DEFAULT_OUTPUT = "results/stage33_role_diagnosis.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3.3 role-dependence diagnosis for Stage 3.2 failures.")
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples-per-role", type=int, default=20)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    result = run_diagnosis(
        results_path=Path(args.results),
        output_path=Path(args.output),
        device=str(args.device),
        max_samples_per_role=int(args.max_samples_per_role),
    )
    if not result.get("failed_seed_diagnoses"):
        raise SystemExit(1)


def run_diagnosis(
    results_path: Path,
    output_path: Path,
    device: str = "cuda",
    max_samples_per_role: int = 20,
) -> Dict[str, object]:
    stage32 = json.loads(results_path.read_text(encoding="utf-8"))
    rows = sorted(stage32.get("stage32_rows", []), key=lambda row: int(row["seed"]))
    failed_rows = [row for row in rows if bool(row.get("per_role_ablation", {}).get("one_role_only_failure", False))]

    diagnoses = []
    for row in failed_rows:
        diagnoses.append(_diagnose_seed(row, device=device, max_samples_per_role=max_samples_per_role))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output = {
        "metadata": {
            "stage": "stage3.3_role_dependence_diagnosis",
            "source_results": str(results_path),
            "architecture_changes": "none",
            "locked_architecture": "topk_attention_no_head",
            "scope": "evaluation-time diagnosis of Stage 3.2 failed role-dependence seeds",
        },
        "failed_seed_count": len(diagnoses),
        "failed_seed_diagnoses": diagnoses,
        "summary": _summary(diagnoses),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    return output


def _diagnose_seed(row: Dict[str, object], device: str, max_samples_per_role: int) -> Dict[str, object]:
    seed = int(row["seed"])
    dataset_config = MultiViewCodePatchDatasetConfig(**row["dataset_config"])
    splits = build_multiview_code_patch_splits(dataset_config, seed=seed, repo_root=".")
    test_examples = list(splits["test"])
    labels = _labels(test_examples)
    trainable = _load_latent_checkpoint(Path(row["checkpoint_paths"]["trainable"]), device=device)
    frozen = _load_latent_checkpoint(Path(row["checkpoint_paths"]["frozen"]), device=device)

    base_trainable = predict_latent_system(trainable, test_examples, condition="none", seed=seed)
    base_frozen = predict_latent_system(frozen, test_examples, condition="none", seed=seed)
    per_role_masked = dict(row.get("per_role_ablation", {}).get("accuracy_when_role_masked", {}))
    masked_critical_roles = _masked_critical_roles(row)

    only_role_accuracy: Dict[str, float] = {}
    only_role_frozen_accuracy: Dict[str, float] = {}
    only_role_predictions: Dict[int, np.ndarray] = {}
    only_role_frozen_predictions: Dict[int, np.ndarray] = {}
    for role_id in range(len(test_examples[0].views)):
        variant = _only_single_role(test_examples, role_id)
        preds = predict_latent_system(trainable, variant, condition="none", seed=seed + 100 + role_id)
        frozen_preds = predict_latent_system(frozen, variant, condition="none", seed=seed + 200 + role_id)
        only_role_predictions[role_id] = preds
        only_role_frozen_predictions[role_id] = frozen_preds
        only_role_accuracy[str(role_id)] = _accuracy(preds, labels)
        only_role_frozen_accuracy[str(role_id)] = _accuracy(frozen_preds, labels)

    role_structured_oracle = {
        str(role_id): _accuracy(_structured_oracle_predictions(test_examples, [role_id]), labels)
        for role_id in range(len(test_examples[0].views))
    }
    pair_structured_oracle = {
        f"{left},{right}": _accuracy(_structured_oracle_predictions(test_examples, [left, right]), labels)
        for left in range(len(test_examples[0].views))
        for right in range(left + 1, len(test_examples[0].views))
    }
    all_role_structured_oracle = _accuracy(_structured_oracle_predictions(test_examples, [0, 1, 2, 3]), labels)

    base_accuracy = _accuracy(base_trainable, labels)
    single_role_sufficient_roles = [
        role_id
        for role_id in range(len(test_examples[0].views))
        if only_role_accuracy[str(role_id)] >= max(0.225, base_accuracy - 0.05)
    ]
    explaining_roles = sorted(set(masked_critical_roles) | set(single_role_sufficient_roles))

    samples = []
    for role_id in explaining_roles:
        samples.extend(
            _samples_for_role(
                seed=seed,
                role_id=role_id,
                examples=test_examples,
                labels=labels,
                trainable=trainable,
                frozen=frozen,
                base_trainable=base_trainable,
                base_frozen=base_frozen,
                only_role_trainable=only_role_predictions[role_id],
                only_role_frozen=only_role_frozen_predictions[role_id],
                max_samples=max_samples_per_role,
            )
        )

    single_view_text = {
        str(role_id): float(row.get("test_accuracy", {}).get(f"single_view_text_role_{role_id}", 0.0))
        for role_id in range(len(test_examples[0].views))
    }
    family_breakdown = _family_breakdown(
        test_examples,
        labels,
        base_trainable,
        base_frozen,
        only_role_predictions,
        explaining_roles,
    )

    return {
        "seed": seed,
        "base_trainable_accuracy": _accuracy(base_trainable, labels),
        "base_frozen_accuracy": _accuracy(base_frozen, labels),
        "stage32_test_accuracy": row.get("test_accuracy", {}),
        "accuracy_with_each_role_masked": per_role_masked,
        "accuracy_with_only_each_single_role_available": only_role_accuracy,
        "frozen_accuracy_with_only_each_single_role_available": only_role_frozen_accuracy,
        "accuracy_drop_when_role_masked": row.get("per_role_ablation", {}).get("accuracy_drop", {}),
        "masked_critical_roles": masked_critical_roles,
        "single_role_sufficient_roles": single_role_sufficient_roles,
        "explaining_roles": explaining_roles,
        "single_view_text_accuracy": single_view_text,
        "single_role_structured_oracle_accuracy": role_structured_oracle,
        "pairwise_role_structured_oracle_accuracy": pair_structured_oracle,
        "all_role_structured_oracle_accuracy": all_role_structured_oracle,
        "per_family_breakdown": family_breakdown,
        "role_failure_interpretation": _interpret_failure(
            row=row,
            masked_critical_roles=masked_critical_roles,
            single_role_sufficient_roles=single_role_sufficient_roles,
            explaining_roles=explaining_roles,
            only_role_accuracy=only_role_accuracy,
            single_view_text=single_view_text,
            role_structured_oracle=role_structured_oracle,
            family_breakdown=family_breakdown,
        ),
        "single_role_sufficient_samples": samples,
    }


def _masked_critical_roles(row: Dict[str, object]) -> List[int]:
    ablation = row.get("per_role_ablation", {})
    base = float(ablation.get("base_accuracy", row.get("test_accuracy", {}).get("trainable", 0.0)))
    drops = {int(role): float(drop) for role, drop in dict(ablation.get("accuracy_drop", {})).items()}
    threshold = max(0.10, base - 0.225)
    out = [role for role, drop in sorted(drops.items()) if drop > threshold]
    if out:
        return out
    if drops:
        return [max(drops, key=drops.get)]
    return []


def _only_single_role(examples: Sequence[MultiViewTaskExample], role_id: int) -> List[MultiViewTaskExample]:
    out = []
    for example in examples:
        views = []
        for index, view in enumerate(example.views):
            if index == role_id:
                views.append(view)
            else:
                views.append(
                    View(
                        role=view.role,
                        text="This role is masked for single-role sufficiency diagnosis. No private structural evidence is available.",
                        source_path=view.source_path,
                        source_type=f"{view.source_type}:single_role_masked",
                        allowed_visibility=view.allowed_visibility,
                    )
                )
        out.append(replace(example, views=tuple(views)))
    return out


def _structured_oracle_predictions(examples: Sequence[MultiViewTaskExample], roles: Sequence[int]) -> np.ndarray:
    predictions: List[int] = []
    selected = [int(role) for role in roles]
    for example in examples:
        evidence = [int(value) for value in example.metadata.get("evidence_bits", [0, 0, 0, 0])]
        scores = []
        for candidate in example.candidates:
            bits = candidate_attribute_bits(example, candidate)
            scores.append(sum(int(bits[role] == evidence[role]) for role in selected))
        predictions.append(int(np.argmax(np.asarray(scores, dtype=np.float32))))
    return np.asarray(predictions, dtype=np.int64)


def _samples_for_role(
    seed: int,
    role_id: int,
    examples: Sequence[MultiViewTaskExample],
    labels: np.ndarray,
    trainable,
    frozen,
    base_trainable: np.ndarray,
    base_frozen: np.ndarray,
    only_role_trainable: np.ndarray,
    only_role_frozen: np.ndarray,
    max_samples: int,
) -> List[Dict[str, object]]:
    sufficient = np.where(only_role_trainable == labels)[0]
    rows = []
    for example_index in sufficient[:max_samples]:
        example = examples[int(example_index)]
        chosen = int(only_role_trainable[int(example_index)])
        gold = int(example.label)
        rows.append(
            {
                "seed": seed,
                "suspect_role": int(role_id),
                "example_index": int(example_index),
                "example_id": example.id,
                "problem_family": str(example.metadata.get("problem_family")),
                "candidate_list_after_redaction": [
                    {
                        "index": index,
                        "candidate_id": candidate.candidate_id,
                        "text": candidate.text,
                        "attributes": list(candidate.attributes),
                    }
                    for index, candidate in enumerate(example.candidates)
                ],
                "private_views_after_redaction": [
                    {
                        "role_index": index,
                        "role": view.role,
                        "source_type": view.source_type,
                        "text": view.text,
                    }
                    for index, view in enumerate(example.views)
                ],
                "topk_selected_tokens_by_role": [
                    {
                        "role_index": index,
                        "role": view.role,
                        "trainable_topk": _topk_selected_tokens(trainable.system, example, index),
                        "frozen_topk": _topk_selected_tokens(frozen.system, example, index),
                    }
                    for index, view in enumerate(example.views)
                ],
                "chosen_candidate_index": chosen,
                "chosen_candidate": example.candidates[chosen].text,
                "gold_candidate_index": gold,
                "gold_candidate": example.candidates[gold].text,
                "base_trainable_prediction": int(base_trainable[int(example_index)]),
                "base_trainable_succeeds": bool(base_trainable[int(example_index)] == labels[int(example_index)]),
                "suspect_role_only_succeeds": True,
                "frozen_base_prediction": int(base_frozen[int(example_index)]),
                "frozen_base_succeeds": bool(base_frozen[int(example_index)] == labels[int(example_index)]),
                "frozen_suspect_role_only_prediction": int(only_role_frozen[int(example_index)]),
                "frozen_suspect_role_only_succeeds": bool(only_role_frozen[int(example_index)] == labels[int(example_index)]),
            }
        )
    return rows


def _family_breakdown(
    examples: Sequence[MultiViewTaskExample],
    labels: np.ndarray,
    base_trainable: np.ndarray,
    base_frozen: np.ndarray,
    only_role_predictions: Dict[int, np.ndarray],
    suspect_roles: Sequence[int],
) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[int]] = {}
    for index, example in enumerate(examples):
        grouped.setdefault(str(example.metadata.get("problem_family")), []).append(index)
    out: Dict[str, Dict[str, float]] = {}
    for family, indices in sorted(grouped.items()):
        idx = np.asarray(indices, dtype=np.int64)
        row = {
            "n": float(len(indices)),
            "trainable_accuracy": _accuracy(base_trainable[idx], labels[idx]),
            "frozen_accuracy": _accuracy(base_frozen[idx], labels[idx]),
        }
        for role_id in suspect_roles:
            row[f"only_role_{role_id}_accuracy"] = _accuracy(only_role_predictions[int(role_id)][idx], labels[idx])
        out[family] = row
    return out


def _interpret_failure(
    row: Dict[str, object],
    masked_critical_roles: Sequence[int],
    single_role_sufficient_roles: Sequence[int],
    explaining_roles: Sequence[int],
    only_role_accuracy: Dict[str, float],
    single_view_text: Dict[str, float],
    role_structured_oracle: Dict[str, float],
    family_breakdown: Dict[str, Dict[str, float]],
) -> Dict[str, object]:
    interpretations = []
    exact_leakage = bool(row.get("output_leakage_audit_passes", True)) and bool(row.get("split_leakage_audit_passes", True))
    if exact_leakage:
        interpretations.append("exact symbol/module/path/candidate textual leakage was not observed in Stage 3.2 audits")
    if single_role_sufficient_roles:
        interpretations.append(f"single-role-only sufficiency is observed for roles: {[int(role) for role in single_role_sufficient_roles]}")
    if masked_critical_roles:
        interpretations.append(f"masked-role criticality is observed for roles: {[int(role) for role in masked_critical_roles]}")
    for role in explaining_roles:
        key = str(role)
        if single_view_text.get(key, 1.0) <= 0.225 and only_role_accuracy.get(key, 0.0) > 0.225:
            interpretations.append(f"role {role} is not explained by the text-only single-view baseline; the failure is latent/model-protocol dependence")
        if role_structured_oracle.get(key, 0.0) > 0.35:
            interpretations.append(f"role {role} has a dataset-level structured shortcut above 0.35")
    concentrated = [
        family
        for family, values in family_breakdown.items()
        if float(values.get("trainable_accuracy", 0.0)) >= 0.70 or any(key.startswith("only_role_") and float(value) >= 0.70 for key, value in values.items())
    ]
    if concentrated:
        interpretations.append(f"accuracy is concentrated in dataset families: {concentrated}")
    if not interpretations:
        interpretations.append("no single textual or family artifact fully explains the failure")
    return {
        "masked_critical_roles": [int(role) for role in masked_critical_roles],
        "single_role_sufficient_roles": [int(role) for role in single_role_sufficient_roles],
        "explaining_roles": [int(role) for role in explaining_roles],
        "classification": _classify_failure(explaining_roles, only_role_accuracy, single_view_text, role_structured_oracle, concentrated),
        "notes": interpretations,
    }


def _classify_failure(
    suspect_roles: Sequence[int],
    only_role_accuracy: Dict[str, float],
    single_view_text: Dict[str, float],
    role_structured_oracle: Dict[str, float],
    concentrated: Sequence[str],
) -> str:
    if any(single_view_text.get(str(role), 0.0) > 0.225 for role in suspect_roles):
        return "single-role textual leakage"
    if any(role_structured_oracle.get(str(role), 0.0) > 0.35 for role in suspect_roles):
        return "candidate construction artifact"
    if concentrated:
        return "dataset family imbalance with single-role latent shortcut"
    if any(only_role_accuracy.get(str(role), 0.0) > 0.225 for role in suspect_roles):
        return "single-role latent shortcut"
    return "masked-role sensitivity without single-role sufficiency"


def _summary(diagnoses: Sequence[Dict[str, object]]) -> Dict[str, object]:
    counts = Counter(diag.get("role_failure_interpretation", {}).get("classification", "unknown") for diag in diagnoses)
    return {
        "failed_seeds": [int(diag["seed"]) for diag in diagnoses],
        "classification_counts": dict(counts),
        "mean_base_trainable_accuracy": _mean([float(diag["base_trainable_accuracy"]) for diag in diagnoses]),
        "mean_base_frozen_accuracy": _mean([float(diag["base_frozen_accuracy"]) for diag in diagnoses]),
        "masked_critical_roles_by_seed": {str(diag["seed"]): diag.get("masked_critical_roles", []) for diag in diagnoses},
        "single_role_sufficient_roles_by_seed": {str(diag["seed"]): diag.get("single_role_sufficient_roles", []) for diag in diagnoses},
        "explaining_roles_by_seed": {str(diag["seed"]): diag.get("explaining_roles", []) for diag in diagnoses},
    }


def _labels(examples: Sequence[MultiViewTaskExample]) -> np.ndarray:
    return np.asarray([example.label for example in examples], dtype=np.int64)


def _accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(predictions.astype(np.int64) == labels.astype(np.int64))) if len(labels) else 0.0


def _mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


if __name__ == "__main__":
    main()
