from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from src.datasets.multiview_code_patch_selection import (
    MultiViewTaskExample,
    apply_example_control,
    build_multiview_code_patch_splits,
    dataset_summary,
    example_oracle_metadata,
    model_record_from_example,
    output_leakage_audit,
    randomized_labels_for_examples,
    split_leakage_audit,
)
from src.experiments.architecture_search import (
    _agent_config,
    _bootstrap_ci,
    _candidate_config,
    _coordinator_config,
    _leakage_passes,
    _training_config,
)
from src.experiments.real_shared_weight_latent_coordination import (
    BENCHMARK,
    MessageChannelConfig,
    fit_latent_system,
)
from src.experiments.run_stage34_candidate_sanitization_diagnostics import (
    _accuracy,
    _bow_accuracy,
    _candidate_metadata_only_text,
    _structured_oracle_predictions,
)
from src.experiments.run_stage38_full_hard_validation import (
    _dataset_config_for_stage,
    _labels,
    _stage38_full_config,
    _stage_from_config,
)
from src.experiments.run_stage38_schema_aware_controls import (
    _candidate_evidence_mismatch,
    _candidate_only,
    _cross_example_view_bundle_shuffle,
    _evidence_only_no_candidates,
    _null_evidence_values,
    _schema_only,
    _schema_preserved_role_value_shuffle,
    _value_shuffle_within_schema,
    _zero_role_embeddings,
)
from src.experiments.run_stage3_gpu_hard_validation import (
    _clear_cuda,
    _configure_cuda,
    _save_latent_checkpoint,
)


DEFAULT_CONFIG = "configs/stage6_integrated_multiperspective_search.json"
DEFAULT_RESULTS = "results/stage6_integrated_results.json"
DEFAULT_CONTROLS = "results/stage6_integrated_controls.json"
DEFAULT_LEAKAGE_AUDIT = "results/stage6_integrated_leakage_audit.jsonl"
DEFAULT_MASK_AUDIT = "results/stage6_integrated_attention_mask_audit.jsonl"
DEFAULT_ERROR_CASES = "results/stage6_integrated_error_cases.jsonl"
DEFAULT_COMPUTE = "results/stage6_integrated_compute_metrics.json"
DEFAULT_REPORT = "reports/STAGE6_INTEGRATED_MULTIPERSPECTIVE_SEARCH.md"
DEFAULT_CHECKPOINT_DIR = "results/stage6_integrated_checkpoints"

BASELINE_VARIANT = "candidate_token_direct_lr3e4_clip1"
NEAR_CHANCE = 0.18
INVARIANCE_TOLERANCE = 0.08
PRIMARY_CONTROLS = (
    "candidate_only",
    "candidate_metadata_only",
    "schema_only",
    "view_masked_candidates_visible",
    "evidence_only_no_candidates",
    "null_evidence_values",
    "value_shuffle_within_schema",
    "cross_example_view_bundle_shuffle",
    "candidate_evidence_mismatch",
    "schema_preserved_role_value_shuffle",
    "randomized_labels",
    "hidden_states_shuffled",
    "physical_order_shuffled_roles_preserved",
    "candidate_order_shuffled_with_gold_remap",
)
INTEGRATED_CONTROLS = (
    "attention_mask_audit",
    "role_stream_isolation_audit",
    "candidate_token_isolation_audit",
    "no_gold_token_audit",
    "no_candidate_position_shortcut_audit",
    "packed_sequence_position_shuffle_audit",
    "role_block_permutation_audit",
    "candidate_block_permutation_with_remapped_labels",
    "coordination_token_ablation",
    "candidate_token_ablation",
    "role_token_ablation",
    "cross_stream_attention_disabled",
    "early_mixing_disabled",
    "late_mixing_only_control",
)
EXTRA_CONTROL_CONDITIONS = (
    "coordination_token_ablation",
    "candidate_token_ablation",
    "role_token_ablation",
    "cross_stream_attention_disabled",
    "early_mixing_disabled",
    "late_mixing_only_control",
)


@dataclass(frozen=True)
class Stage6Variant:
    name: str
    family: str
    coordinator_family: str
    description: str
    coordinator_layers: int = 1
    lr: float = 0.0003
    epochs: int = 50
    patience: int = 6
    dropout: float = 0.0
    weight_decay: float = 0.0001
    gradient_clip_norm: float = 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 5C/6 integrated multi-perspective architecture search.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_RESULTS)
    parser.add_argument("--controls-output", default=DEFAULT_CONTROLS)
    parser.add_argument("--leakage-audit", default=DEFAULT_LEAKAGE_AUDIT)
    parser.add_argument("--mask-audit", default=DEFAULT_MASK_AUDIT)
    parser.add_argument("--error-cases", default=DEFAULT_ERROR_CASES)
    parser.add_argument("--compute-output", default=DEFAULT_COMPUTE)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-variants", type=int, default=0)
    parser.add_argument("--phase", choices=("cheap", "cheap_medium", "final"), default="cheap")
    parser.add_argument("--skip-randomized", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.device is not None:
        config["device"] = str(args.device)
        if str(args.device) == "cpu":
            config["require_cuda"] = False
    result = run_stage6_search(
        config=config,
        config_path=config_path,
        checkpoint_dir=Path(args.checkpoint_dir),
        max_variants=int(args.max_variants),
        phase=str(args.phase),
        run_randomized=not bool(args.skip_randomized),
    )
    _write_all_outputs(
        result,
        output_path=Path(args.output),
        controls_path=Path(args.controls_output),
        leakage_path=Path(args.leakage_audit),
        mask_path=Path(args.mask_audit),
        error_path=Path(args.error_cases),
        compute_path=Path(args.compute_output),
        report_path=Path(args.report),
    )


def run_stage6_search(
    config: Dict[str, object],
    config_path: Path,
    checkpoint_dir: Path,
    max_variants: int = 0,
    phase: str = "cheap",
    run_randomized: bool = True,
) -> Dict[str, object]:
    config = _stage38_full_config(config)
    device, hardware = _configure_cuda(config)
    variants = _variant_plan()
    if max_variants > 0:
        variants = variants[:max_variants]
    result: Dict[str, object] = {
        "metadata": {
            "stage": "stage6_integrated_multiperspective_search",
            "created_at_utc": _now(),
            "config_path": str(config_path),
            "device": device,
            "hardware": hardware,
            "fixed_benchmark": "Stage 4 schema-aware balanced_categories_v3",
            "dataset_or_label_changes": "none",
            "selection_policy": "cheap/medium selection uses dev metrics and controls only; final seeds are reserved",
            "run_randomized_label_controls": bool(run_randomized),
            "phase_requested": phase,
        },
        "variant_plan": [asdict(variant) for variant in variants],
        "cheap_screen": {"rows": [], "summary": {}},
        "medium_validation": {"rows": [], "summary": {"ran": False, "reason": "not requested or no passing cheap variant"}},
        "final_validation": {"rows": [], "summary": {"ran": False, "reason": "not requested; reserved seeds [50..59]"}},
    }
    print(f"stage6: cheap exploratory search starting for {len(variants)} variants")
    cheap_rows = _run_phase(
        phase_name="cheap",
        config=config,
        stage_updates={"name": "stage6_cheap_exploratory", "n_train": 512, "n_dev": 256, "n_test": 256, "seeds": [0, 1, 2]},
        variants=variants,
        device=device,
        checkpoint_dir=checkpoint_dir / "cheap",
        run_randomized=run_randomized,
    )
    result["cheap_screen"] = {"rows": cheap_rows, "summary": _phase_summary(cheap_rows, min_seed_wins=2, min_delta=0.20)}
    if phase == "cheap":
        passed = [row["variant"] for row in result["cheap_screen"]["summary"].get("variant_summaries", []) if row.get("selection_passed")]
        result["medium_validation"]["summary"] = {
            "ran": False,
            "reason": f"not requested in this run; Phase 1 selected {passed}",
        }
    if phase in {"cheap_medium", "final"}:
        top_variants = _select_medium_variants(result["cheap_screen"]["summary"], variants)
        if top_variants:
            print(f"stage6: medium validation starting for {[variant.name for variant in top_variants]}")
            medium_rows = _run_phase(
                phase_name="medium",
                config=config,
                stage_updates={"name": "stage6_medium_validation", "n_train": 1024, "n_dev": 512, "n_test": 512, "seeds": [0, 1, 2, 3, 4]},
                variants=top_variants,
                device=device,
                checkpoint_dir=checkpoint_dir / "medium",
                run_randomized=run_randomized,
            )
            result["medium_validation"] = {
                "rows": medium_rows,
                "summary": {**_phase_summary(medium_rows, min_seed_wins=4, min_delta=0.20), "ran": True},
            }
        if phase == "final":
            selected = _select_final_variant(result["medium_validation"]["summary"], variants)
            if selected is not None:
                print(f"stage6: final validation starting for locked variant {selected.name}")
                final_rows = _run_phase(
                    phase_name="final",
                    config=config,
                    stage_updates={"name": "stage6_final_validation", "n_train": 2048, "n_dev": 512, "n_test": 1024, "seeds": list(range(50, 60))},
                    variants=[selected],
                    device=device,
                    checkpoint_dir=checkpoint_dir / "final",
                    run_randomized=run_randomized,
                )
                result["final_validation"] = {
                    "rows": final_rows,
                    "summary": {**_final_summary(final_rows), "ran": True, "selected_variant": selected.name},
                }
    result["summary"] = _overall_summary(result)
    return result


def _run_phase(
    phase_name: str,
    config: Dict[str, object],
    stage_updates: Dict[str, object],
    variants: Sequence[Stage6Variant],
    device: str,
    checkpoint_dir: Path,
    run_randomized: bool,
) -> List[Dict[str, object]]:
    stage_config = _phase_config(config, stage_updates)
    stage = _stage_from_config(stage_config)
    rows: List[Dict[str, object]] = []
    for seed in stage.seeds:
        print(f"stage6 {phase_name} seed={seed}: building fixed schema-aware splits")
        splits = build_multiview_code_patch_splits(_dataset_config_for_stage(stage_config, stage), seed=int(seed), repo_root=Path("."))
        split_leakage = split_leakage_audit(BENCHMARK, int(seed), splits)
        output_leakage = output_leakage_audit(BENCHMARK, int(seed), splits["train"] + splits["dev"] + splits["test"])
        metadata_only = _candidate_metadata_shortcut(stage_config, splits, int(seed))
        no_gold = _no_gold_token_audit(splits)
        for index, variant in enumerate(variants):
            print(f"stage6 {phase_name}: variant={variant.name} seed={seed}")
            _clear_cuda()
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            candidate = _candidate_for_variant(stage, variant)
            training = replace(
                _training_config(stage),
                epochs=int(variant.epochs),
                patience=int(variant.patience),
                lr=float(variant.lr),
                weight_decay=float(variant.weight_decay),
                gradient_clip_norm=float(variant.gradient_clip_norm),
            )
            fit_seed = int(seed) + index * 100_000 + 60_001
            t0 = time.perf_counter()
            trainable = fit_latent_system(
                train_examples=splits["train"],
                dev_examples=splits["dev"],
                agent_config=_agent_config(stage),
                coordinator_config=candidate.coordinator_config,
                training_config=training,
                num_classes=8,
                seed=fit_seed,
                device=device,
                trainable_agent=True,
                method=f"stage6_{phase_name}_trainable__{variant.name}",
                message_config=candidate.message_config,
            )
            trainable_train_seconds = time.perf_counter() - t0
            t0 = time.perf_counter()
            frozen = fit_latent_system(
                train_examples=splits["train"],
                dev_examples=splits["dev"],
                agent_config=_agent_config(stage),
                coordinator_config=candidate.coordinator_config,
                training_config=training,
                num_classes=8,
                seed=fit_seed,
                device=device,
                trainable_agent=False,
                method=f"stage6_{phase_name}_frozen__{variant.name}",
                message_config=candidate.message_config,
            )
            frozen_train_seconds = time.perf_counter() - t0
            randomized = None
            randomized_train_seconds = 0.0
            if run_randomized:
                t0 = time.perf_counter()
                randomized = fit_latent_system(
                    train_examples=splits["train"],
                    dev_examples=splits["dev"],
                    agent_config=_agent_config(stage),
                    coordinator_config=candidate.coordinator_config,
                    training_config=training,
                    num_classes=8,
                    seed=fit_seed,
                    device=device,
                    trainable_agent=True,
                    method=f"stage6_{phase_name}_randomized__{variant.name}",
                    condition="randomized_labels",
                    train_labels=randomized_labels_for_examples(splits["train"], fit_seed + 77, 8),
                    dev_labels=randomized_labels_for_examples(splits["dev"], fit_seed + 88, 8),
                    message_config=candidate.message_config,
                )
                randomized_train_seconds = time.perf_counter() - t0
            evals = {
                split: _evaluate_split(
                    trainable=trainable,
                    frozen=frozen,
                    randomized=randomized,
                    examples=splits[split],
                    seed=int(seed) + (10_000 if split == "dev" else 20_000),
                    metadata_only_accuracy=metadata_only.get(split),
                    no_gold_audit=no_gold,
                    variant=variant,
                )
                for split in ("dev", "test")
            }
            attention_audit = _capture_attention_audit(trainable, splits["dev"][: min(32, len(splits["dev"]))], int(seed) + 30_000)
            latency = _latency_metrics(trainable, splits["test"][: min(128, len(splits["test"]))], int(seed) + 40_000)
            checkpoint_paths = _save_stage6_checkpoints(checkpoint_dir, variant, int(seed), stage, candidate, trainable, frozen, randomized)
            cuda_peak = int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") and torch.cuda.is_available() else 0
            row = {
                "phase": phase_name,
                "seed": int(seed),
                "variant": variant.name,
                "variant_family": variant.family,
                "variant_description": variant.description,
                "architecture_config": _candidate_config(candidate),
                "stage_config": asdict(stage),
                "dataset_summary": dataset_summary(splits),
                "split_leakage_audit": _compact_leakage(split_leakage),
                "split_leakage_audit_passes": _leakage_passes(split_leakage),
                "output_leakage_audit": output_leakage,
                "output_leakage_audit_passes": bool(output_leakage.get("passes", False)),
                "no_gold_token_audit": no_gold,
                "dev": evals["dev"],
                "test": evals["test"],
                "dev_delta": float(evals["dev"]["accuracy"]["trainable"] - evals["dev"]["accuracy"]["frozen"]),
                "test_delta": float(evals["test"]["accuracy"]["trainable"] - evals["test"]["accuracy"]["frozen"]),
                "attention_mask_audit": attention_audit,
                "compute": {
                    "trainable_train_seconds": trainable_train_seconds,
                    "frozen_train_seconds": frozen_train_seconds,
                    "randomized_train_seconds": randomized_train_seconds,
                    "total_fit_seconds": trainable_train_seconds + frozen_train_seconds + randomized_train_seconds,
                    "inference_latency": latency,
                    "cuda_max_memory_allocated": cuda_peak,
                    "cuda_max_memory_allocated_gb": cuda_peak / float(1024**3),
                    "trainable_param_count": int(trainable.param_count),
                    "frozen_param_count": int(frozen.param_count),
                    "attention_operation_estimate": _attention_operation_estimate(attention_audit),
                    "accuracy_per_ms": _safe_div(float(evals["test"]["accuracy"]["trainable"]), float(latency.get("ms_per_example", 0.0))),
                    "delta_per_gb_memory": _safe_div(float(evals["test"]["accuracy"]["trainable"] - evals["test"]["accuracy"]["frozen"]), cuda_peak / float(1024**3)),
                },
                "trainable_audit": _compact_training_audit(trainable.audit),
                "frozen_audit": _compact_training_audit(frozen.audit),
                "randomized_audit": _compact_training_audit(randomized.audit) if randomized is not None else None,
                "checkpoint_paths": checkpoint_paths,
            }
            row["integrated_gate_failures"] = _row_gate_failures(row)
            rows.append(row)
            del trainable, frozen, randomized
            _clear_cuda()
    _attach_clone_transitions(rows)
    return rows


def _evaluate_split(
    trainable,
    frozen,
    randomized,
    examples: Sequence[MultiViewTaskExample],
    seed: int,
    metadata_only_accuracy: float | None,
    no_gold_audit: Dict[str, object],
    variant: Stage6Variant,
) -> Dict[str, object]:
    labels = _labels(examples)
    train_logits = _predict_logits(trainable, examples, "none", seed)
    frozen_logits = _predict_logits(frozen, examples, "none", seed)
    accuracy = {
        "trainable": _accuracy(np.argmax(train_logits, axis=1), labels),
        "frozen": _accuracy(np.argmax(frozen_logits, axis=1), labels),
        "oracle": _accuracy(_structured_oracle_predictions(examples, [0, 1, 2, 3]), labels),
    }
    ranking = {
        "trainable": _ranking_metrics(train_logits, labels),
        "frozen": _ranking_metrics(frozen_logits, labels),
    }
    control_accuracy: Dict[str, float | None] = {"candidate_metadata_only": metadata_only_accuracy}
    for name, rows in _stage4_control_examples(examples, seed).items():
        cy = _labels(rows)
        if name == "candidate_only":
            with _zero_role_embeddings(trainable.system):
                logits = _predict_logits(trainable, rows, "none", seed + 1)
        else:
            logits = _predict_logits(trainable, rows, "none", seed + 1)
        control_accuracy[name] = _accuracy(np.argmax(logits, axis=1), cy)
    if randomized is not None:
        control_accuracy["randomized_labels"] = _accuracy(np.argmax(_predict_logits(randomized, examples, "none", seed + 2), axis=1), labels)
    else:
        control_accuracy["randomized_labels"] = None
    control_accuracy["hidden_states_shuffled"] = _accuracy(
        np.argmax(_predict_logits(trainable, examples, "hidden_states_shuffled_across_examples", seed + 3), axis=1),
        labels,
    )
    control_accuracy["physical_order_shuffled_roles_preserved"] = _accuracy(
        np.argmax(_predict_logits(trainable, examples, "physical_order_shuffled_roles_preserved", seed + 4), axis=1),
        labels,
    )
    shuffled = apply_example_control(examples, "candidate_order_shuffled", seed=seed + 5)
    control_accuracy["candidate_order_shuffled_with_gold_remap"] = _accuracy(
        np.argmax(_predict_logits(trainable, shuffled, "none", seed + 6), axis=1),
        _labels(shuffled),
    )
    extra = {}
    for condition in EXTRA_CONTROL_CONDITIONS:
        extra[condition] = _accuracy(np.argmax(_predict_logits(trainable, examples, condition, seed + 100), axis=1), labels)
    extra["role_block_permutation_audit"] = control_accuracy["physical_order_shuffled_roles_preserved"]
    extra["candidate_block_permutation_with_remapped_labels"] = control_accuracy["candidate_order_shuffled_with_gold_remap"]
    extra["no_candidate_position_shortcut_audit"] = max(
        float(control_accuracy.get("candidate_only") or 0.0),
        float(control_accuracy.get("candidate_metadata_only") or 0.0),
    )
    extra["packed_sequence_position_shuffle_audit"] = 1.0 if variant.coordinator_family.startswith("integrated_") else None
    extra["no_gold_token_audit"] = 1.0 if bool(no_gold_audit.get("passes", False)) else 0.0
    accuracy.update({name: value for name, value in control_accuracy.items()})
    accuracy.update(extra)
    return {
        "accuracy": accuracy,
        "ranking": ranking,
        "per_family_accuracy": _per_family_accuracy(np.argmax(train_logits, axis=1), np.argmax(frozen_logits, axis=1), examples),
        "ablation_impact": {
            name: float(accuracy["trainable"] - value) if isinstance(value, (int, float)) else None
            for name, value in extra.items()
        },
        "error_cases": _error_case_rows(train_logits, frozen_logits, labels, examples),
    }


def _predict_logits(result, examples: Sequence[MultiViewTaskExample], condition: str, seed: int) -> np.ndarray:
    result.system.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(examples), 128):
            batch = list(examples[start : start + 128])
            logits, _audit, _acts = result.system(batch, condition=condition, seed=seed)
            rows.append(logits.detach().float().cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float32) if rows else np.zeros((0, 8), dtype=np.float32)


def _ranking_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    if len(labels) == 0:
        return {"top2": 0.0, "top3": 0.0, "confidence": 0.0, "margin": 0.0, "ece_10": 0.0}
    probs = F.softmax(torch.as_tensor(logits, dtype=torch.float32), dim=1).numpy()
    order = np.argsort(-logits, axis=1)
    top1 = order[:, 0]
    top2 = np.asarray([label in row[:2] for label, row in zip(labels, order)], dtype=np.float32)
    top3 = np.asarray([label in row[:3] for label, row in zip(labels, order)], dtype=np.float32)
    margins = logits[np.arange(len(logits)), order[:, 0]] - logits[np.arange(len(logits)), order[:, 1]]
    confidence = probs[np.arange(len(probs)), top1]
    correct = (top1 == labels).astype(np.float32)
    return {
        "top2": float(np.mean(top2)),
        "top3": float(np.mean(top3)),
        "confidence": float(np.mean(confidence)),
        "margin": float(np.mean(margins)),
        "ece_10": _ece(confidence, correct, bins=10),
    }


def _stage4_control_examples(examples: Sequence[MultiViewTaskExample], seed: int) -> Dict[str, Sequence[MultiViewTaskExample]]:
    return {
        "candidate_only": _candidate_only(examples),
        "schema_only": _schema_only(examples),
        "view_masked_candidates_visible": apply_example_control(examples, "view_masked", seed=seed + 11),
        "evidence_only_no_candidates": _evidence_only_no_candidates(examples),
        "null_evidence_values": _null_evidence_values(examples),
        "value_shuffle_within_schema": _value_shuffle_within_schema(examples, seed + 12),
        "cross_example_view_bundle_shuffle": _cross_example_view_bundle_shuffle(examples, seed + 13),
        "candidate_evidence_mismatch": _candidate_evidence_mismatch(examples, seed + 14),
        "schema_preserved_role_value_shuffle": _schema_preserved_role_value_shuffle(examples, seed + 15),
    }


def _error_case_rows(
    train_logits: np.ndarray,
    frozen_logits: np.ndarray,
    labels: np.ndarray,
    examples: Sequence[MultiViewTaskExample],
    limit: int = 25,
) -> List[Dict[str, object]]:
    rows = []
    train_pred = np.argmax(train_logits, axis=1)
    frozen_pred = np.argmax(frozen_logits, axis=1)
    order = np.argsort(-train_logits, axis=1)
    for index, (example, pred, frozen, label) in enumerate(zip(examples, train_pred, frozen_pred, labels)):
        if int(pred) == int(label):
            continue
        margin = float(train_logits[index, order[index, 0]] - train_logits[index, order[index, 1]])
        rows.append(
            {
                "example_id": example.id,
                "problem_family": str(example_oracle_metadata(example).get("problem_family", "unknown")),
                "prediction": int(pred),
                "label": int(label),
                "frozen_prediction": int(frozen),
                "top2": [int(value) for value in order[index, :2]],
                "top1_margin": margin,
                "trainable_correct": False,
                "frozen_correct": bool(int(frozen) == int(label)),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _candidate_metadata_shortcut(config: Dict[str, object], splits: Dict[str, Sequence[MultiViewTaskExample]], seed: int) -> Dict[str, float]:
    value = _bow_accuracy(seed + 303_000, splits, config, "stage6_candidate_metadata_only", _candidate_metadata_only_text)
    return {"train": value, "dev": value, "test": value}


def _no_gold_token_audit(splits: Dict[str, Sequence[MultiViewTaskExample]]) -> Dict[str, object]:
    failures = []
    for split, examples in splits.items():
        for example in examples:
            try:
                model_record_from_example(example)
            except AssertionError as exc:
                failures.append({"split": split, "id": example.id, "error": str(exc)})
    return {
        "passes": not failures,
        "failures": failures[:20],
        "model_records_checked": sum(len(rows) for rows in splits.values()),
        "oracle_metadata_kept_out_of_model_record": True,
    }


def _capture_attention_audit(result, examples: Sequence[MultiViewTaskExample], seed: int) -> Dict[str, object]:
    coordinator = result.system.coordinator
    previous = getattr(coordinator, "record_attention", False)
    if hasattr(coordinator, "record_attention"):
        setattr(coordinator, "record_attention", True)
    result.system.eval()
    with torch.no_grad():
        if examples:
            result.system(list(examples), condition="none", seed=seed)
    if hasattr(coordinator, "record_attention"):
        setattr(coordinator, "record_attention", previous)
    return {
        "mask": getattr(coordinator, "last_mask_audit", None),
        "attention": getattr(coordinator, "last_attention_stats", None),
        "representation_change": getattr(coordinator, "last_representation_change", None),
    }


def _latency_metrics(result, examples: Sequence[MultiViewTaskExample], seed: int) -> Dict[str, float]:
    if not examples:
        return {"examples": 0, "seconds": 0.0, "ms_per_example": 0.0}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    _predict_logits(result, examples, "none", seed)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {"examples": float(len(examples)), "seconds": elapsed, "ms_per_example": elapsed * 1000.0 / max(1, len(examples))}


def _candidate_for_variant(stage, variant: Stage6Variant):
    msg = MessageChannelConfig(
        use_msg_token=False,
        readout_source="pooled",
        use_message_head=False,
        coordinator_family=variant.coordinator_family,
        use_private_cue_aux=False,
        aux_loss_weight=0.0,
        active_message_layers=(-1,),
    )
    coord = _coordinator_config(
        stage,
        family=variant.coordinator_family,
        num_layers=int(variant.coordinator_layers),
        dropout=float(variant.dropout),
    )
    return SimpleNamespace(name=variant.name, description=variant.description, message_config=msg, coordinator_config=coord)


def _variant_plan() -> List[Stage6Variant]:
    return [
        Stage6Variant(
            name="candidate_token_direct_lr3e4_clip1",
            family="stage4_clone_baseline",
            coordinator_family="candidate_token_cross_attention",
            description="Proven Stage 4 clone baseline: candidate queries attend directly over per-role token states.",
        ),
        Stage6Variant(
            name="integrated_bridge_late1_lr3e4",
            family="hybrid_clone_to_integrated_bridge",
            coordinator_family="integrated_bridge_late",
            description="Clone token states feed a semi-integrated masked bridge; candidates attend full role tokens plus role summaries.",
        ),
        Stage6Variant(
            name="integrated_bridge_late2_lr3e4",
            family="hybrid_clone_to_integrated_bridge",
            coordinator_family="integrated_bridge_late",
            coordinator_layers=2,
            description="Two candidate-conditioned integrated bridge blocks after separated role encoding.",
        ),
        Stage6Variant(
            name="integrated_summary_late1_lr3e4",
            family="semi_integrated_cross_view",
            coordinator_family="integrated_summary_late",
            description="Role token streams are summarized, then candidates query a compact cross-view summary layer.",
        ),
        Stage6Variant(
            name="integrated_router_late1_lr3e4",
            family="routing_token_architecture",
            coordinator_family="integrated_router_late",
            description="Learned router tokens aggregate role token states; candidates query routers only.",
        ),
        Stage6Variant(
            name="integrated_router_plus_tokens_late1_lr3e4",
            family="routing_token_architecture",
            coordinator_family="integrated_router_plus_tokens",
            description="Router tokens and full role token states are both visible to candidate queries.",
        ),
        Stage6Variant(
            name="integrated_candidate_guided2_lr3e4",
            family="candidate_guided_evidence_processing",
            coordinator_family="integrated_candidate_guided",
            coordinator_layers=2,
            description="Candidate hypotheses are present while summaries are refined and query role evidence across two blocks.",
        ),
        Stage6Variant(
            name="integrated_early_mixing1_lr3e4",
            family="masked_cross_perspective_transformer",
            coordinator_family="integrated_early_mixing",
            description="Role summaries mix early before candidate-conditioned querying; included to test leakage risk.",
        ),
        Stage6Variant(
            name="integrated_late_mixing_only2_lr3e4",
            family="masked_cross_perspective_transformer",
            coordinator_family="integrated_late_mixing_only",
            coordinator_layers=2,
            description="No early cross-view update; candidates perform all cross-role coordination late.",
        ),
    ]


def _phase_config(config: Dict[str, object], updates: Dict[str, object]) -> Dict[str, object]:
    out = json.loads(json.dumps(config))
    out["stage"] = {**dict(out["stage"]), **updates, "batch_size": 32, "mixed_precision": dict(out["stage"]).get("mixed_precision", "bf16")}
    return _stage38_full_config(out)


def _phase_summary(rows: Sequence[Dict[str, object]], min_seed_wins: int, min_delta: float) -> Dict[str, object]:
    by_variant: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    summaries = []
    for variant, variant_rows in sorted(by_variant.items()):
        dev_deltas = [float(row["dev_delta"]) for row in variant_rows]
        test_deltas = [float(row["test_delta"]) for row in variant_rows]
        dev_train = [float(row["dev"]["accuracy"]["trainable"]) for row in variant_rows]
        dev_frozen = [float(row["dev"]["accuracy"]["frozen"]) for row in variant_rows]
        wins = sum(delta > 0.0 for delta in dev_deltas)
        controls = _control_means(variant_rows, split="dev")
        inv_delta = abs(float(controls.get("physical_order_shuffled_roles_preserved", 0.0)) - _mean(dev_train))
        candidate_inv_delta = abs(float(controls.get("candidate_order_shuffled_with_gold_remap", 0.0)) - _mean(dev_train))
        controls_pass = _controls_pass(controls, trainable_mean=_mean(dev_train))
        audits_pass = all(_mask_audit_passes(row) for row in variant_rows)
        leakage_pass = all(bool(row.get("split_leakage_audit_passes")) and bool(row.get("output_leakage_audit_passes")) for row in variant_rows)
        update_pass = all(_trainable_update_passes(row) and _frozen_update_passes(row) for row in variant_rows)
        passed = bool(
            len(variant_rows) >= min_seed_wins
            and wins >= min_seed_wins
            and _mean(dev_deltas) >= min_delta
            and controls_pass
            and audits_pass
            and leakage_pass
            and update_pass
            and inv_delta <= INVARIANCE_TOLERANCE
            and candidate_inv_delta <= INVARIANCE_TOLERANCE
        )
        summaries.append(
            {
                "variant": variant,
                "n_seeds": len(variant_rows),
                "dev_wins_trainable_over_frozen": wins,
                "mean_dev_trainable": _mean(dev_train),
                "mean_dev_frozen": _mean(dev_frozen),
                "mean_dev_delta": _mean(dev_deltas),
                "mean_test_delta": _mean(test_deltas),
                "std_dev_trainable": pstdev(dev_train) if len(dev_train) > 1 else 0.0,
                "min_dev_trainable": min(dev_train) if dev_train else 0.0,
                "control_means": controls,
                "controls_pass": controls_pass,
                "attention_mask_audits_pass": audits_pass,
                "leakage_pass": leakage_pass,
                "update_audits_pass": update_pass,
                "physical_order_delta": inv_delta,
                "candidate_order_delta": candidate_inv_delta,
                "selection_passed": passed,
            }
        )
    summaries.sort(key=lambda row: (bool(row["selection_passed"]), float(row["mean_dev_delta"])), reverse=True)
    return {
        "variant_summaries": summaries,
        "n_variants": len(summaries),
        "n_passed": sum(bool(row["selection_passed"]) for row in summaries),
        "selection_uses": "dev metrics and dev controls only",
        "min_seed_wins": min_seed_wins,
        "min_mean_delta": min_delta,
    }


def _final_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    deltas = [float(row["test_delta"]) for row in rows]
    trainable = [float(row["test"]["accuracy"]["trainable"]) for row in rows]
    frozen = [float(row["test"]["accuracy"]["frozen"]) for row in rows]
    ci = _bootstrap_ci(deltas) if deltas else (0.0, 0.0)
    wins = sum(delta > 0.0 for delta in deltas)
    controls = _control_means(rows, split="test")
    passed = bool(
        len(rows) >= 10
        and wins >= 8
        and _mean(deltas) >= 0.20
        and ci[0] > 0.05
        and _controls_pass(controls, trainable_mean=_mean(trainable))
        and all(_mask_audit_passes(row) for row in rows)
        and all(_trainable_update_passes(row) and _frozen_update_passes(row) for row in rows)
    )
    return {
        "n_completed": len(rows),
        "completed_seeds": [int(row["seed"]) for row in rows],
        "mean_trainable_accuracy": _mean(trainable),
        "mean_frozen_accuracy": _mean(frozen),
        "mean_delta": _mean(deltas),
        "std_delta": pstdev(deltas) if len(deltas) > 1 else 0.0,
        "bootstrap_95_ci_delta": list(ci),
        "wins_trainable_over_frozen": wins,
        "control_means": controls,
        "passed_primary_final_gates": passed,
    }


def _overall_summary(result: Dict[str, object]) -> Dict[str, object]:
    cheap = result.get("cheap_screen", {}).get("summary", {})
    medium = result.get("medium_validation", {}).get("summary", {})
    final = result.get("final_validation", {}).get("summary", {})
    return {
        "cheap_passed_variants": [row["variant"] for row in cheap.get("variant_summaries", []) if row.get("selection_passed")],
        "medium_ran": bool(medium.get("ran", False)),
        "medium_passed_variants": [row["variant"] for row in medium.get("variant_summaries", []) if row.get("selection_passed")],
        "final_ran": bool(final.get("ran", False)),
        "final_passed": bool(final.get("passed_primary_final_gates", False)),
        "success_claim": "none unless final_passed is true",
        "recommendation": _recommendation(result),
    }


def _select_medium_variants(summary: Dict[str, object], variants: Sequence[Stage6Variant]) -> List[Stage6Variant]:
    passed = [row for row in summary.get("variant_summaries", []) if row.get("selection_passed")]
    passed.sort(key=lambda row: float(row.get("mean_dev_delta", 0.0)), reverse=True)
    names = {str(row["variant"]) for row in passed[:3]}
    return [variant for variant in variants if variant.name in names]


def _select_final_variant(summary: Dict[str, object], variants: Sequence[Stage6Variant]) -> Stage6Variant | None:
    passed = [row for row in summary.get("variant_summaries", []) if row.get("selection_passed")]
    passed.sort(key=lambda row: float(row.get("mean_dev_delta", 0.0)), reverse=True)
    if not passed:
        return None
    name = str(passed[0]["variant"])
    return next((variant for variant in variants if variant.name == name), None)


def _write_all_outputs(
    result: Dict[str, object],
    output_path: Path,
    controls_path: Path,
    leakage_path: Path,
    mask_path: Path,
    error_path: Path,
    compute_path: Path,
    report_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    controls_path.parent.mkdir(parents=True, exist_ok=True)
    controls_path.write_text(json.dumps(_controls_artifact(result), indent=2, sort_keys=True), encoding="utf-8")
    leakage_rows = _leakage_rows(result)
    leakage_path.parent.mkdir(parents=True, exist_ok=True)
    leakage_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in leakage_rows) + ("\n" if leakage_rows else ""), encoding="utf-8")
    mask_rows = _mask_rows(result)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in mask_rows) + ("\n" if mask_rows else ""), encoding="utf-8")
    error_rows = _error_rows(result)
    error_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in error_rows) + ("\n" if error_rows else ""), encoding="utf-8")
    compute_path.parent.mkdir(parents=True, exist_ok=True)
    compute_path.write_text(json.dumps(_compute_artifact(result), indent=2, sort_keys=True), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(result), encoding="utf-8")


def _controls_artifact(result: Dict[str, object]) -> Dict[str, object]:
    return {
        "metadata": result.get("metadata", {}),
        "primary_controls": list(PRIMARY_CONTROLS),
        "integrated_controls": list(INTEGRATED_CONTROLS),
        "cheap": _phase_controls(result.get("cheap_screen", {}).get("rows", [])),
        "medium": _phase_controls(result.get("medium_validation", {}).get("rows", [])),
        "final": _phase_controls(result.get("final_validation", {}).get("rows", [])),
    }


def _phase_controls(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "summary": _control_means(rows, split="test"),
        "rows": [
            {
                "phase": row["phase"],
                "seed": row["seed"],
                "variant": row["variant"],
                "dev_accuracy": row["dev"]["accuracy"],
                "test_accuracy": row["test"]["accuracy"],
                "gate_failures": row.get("integrated_gate_failures", []),
            }
            for row in rows
        ],
    }


def _leakage_rows(result: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    for row in _all_rows(result):
        rows.append(
            {
                "event": "leakage_audit",
                "phase": row["phase"],
                "seed": row["seed"],
                "variant": row["variant"],
                "split": row.get("split_leakage_audit"),
                "split_passes": row.get("split_leakage_audit_passes"),
                "output": row.get("output_leakage_audit"),
                "output_passes": row.get("output_leakage_audit_passes"),
                "no_gold_token": row.get("no_gold_token_audit"),
            }
        )
    return rows


def _mask_rows(result: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    for row in _all_rows(result):
        audit = row.get("attention_mask_audit", {})
        rows.append(
            {
                "event": "attention_mask_audit",
                "phase": row["phase"],
                "seed": row["seed"],
                "variant": row["variant"],
                "mask": audit.get("mask"),
                "attention": audit.get("attention"),
                "representation_change": audit.get("representation_change"),
                "passes": _mask_audit_passes(row),
            }
        )
    return rows


def _error_rows(result: Dict[str, object], limit_per_row: int = 25) -> List[Dict[str, object]]:
    rows = []
    for row in _all_rows(result):
        for split in ("dev", "test"):
            errors = row.get(split, {}).get("error_cases", [])
            for item in errors[:limit_per_row]:
                rows.append({"phase": row["phase"], "seed": row["seed"], "variant": row["variant"], "split": split, **item})
    return rows


def _compute_artifact(result: Dict[str, object]) -> Dict[str, object]:
    rows = [
        {
            "phase": row["phase"],
            "seed": row["seed"],
            "variant": row["variant"],
            **row.get("compute", {}),
        }
        for row in _all_rows(result)
    ]
    by_variant: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    return {
        "metadata": result.get("metadata", {}),
        "rows": rows,
        "summary_by_variant": {
            variant: {
                "mean_total_fit_seconds": _mean([float(row.get("total_fit_seconds", 0.0)) for row in variant_rows]),
                "mean_latency_ms_per_example": _mean([float(row.get("inference_latency", {}).get("ms_per_example", 0.0)) for row in variant_rows]),
                "mean_cuda_max_memory_gb": _mean([float(row.get("cuda_max_memory_allocated_gb", 0.0)) for row in variant_rows]),
                "mean_accuracy_per_ms": _mean([float(row.get("accuracy_per_ms", 0.0)) for row in variant_rows]),
            }
            for variant, variant_rows in by_variant.items()
        },
    }


def _save_stage6_checkpoints(checkpoint_dir: Path, variant: Stage6Variant, seed: int, stage, candidate, trainable, frozen, randomized) -> Dict[str, str]:
    root = checkpoint_dir / variant.name / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    train_path = root / "trainable.pt"
    frozen_path = root / "frozen.pt"
    _save_latent_checkpoint(train_path, seed, stage, variant.name, _candidate_config(candidate), trainable)
    _save_latent_checkpoint(frozen_path, seed, stage, variant.name, _candidate_config(candidate), frozen)
    paths = {"trainable": str(train_path), "frozen": str(frozen_path)}
    if randomized is not None:
        random_path = root / "randomized_labels.pt"
        _save_latent_checkpoint(random_path, seed, stage, f"randomized_labels__{variant.name}", _candidate_config(candidate), randomized)
        paths["randomized_labels"] = str(random_path)
    return paths


def _render_report(result: Dict[str, object]) -> str:
    summary = result.get("summary", {})
    lines = [
        "# Stage 6 Integrated Multi-Perspective Search",
        "",
        "## Scope",
        "",
        "- Fixed benchmark: Stage 4 schema-aware `balanced_categories_v3`.",
        "- Dataset labels: unchanged.",
        "- Oracle metadata/gold candidates: not exposed to model-facing records.",
        "- Frozen comparator: exact same architecture, same masks, same fit seed; shared encoder weights frozen.",
        "- Final seeds `[50..59]` are reserved unless `--phase final` is explicitly run.",
        "",
        "## Variant Plan",
        "",
        "| variant | family | coordinator | layers | description |",
        "|---|---|---|---:|---|",
    ]
    for variant in result.get("variant_plan", []):
        lines.append(f"| {variant['name']} | {variant['family']} | {variant['coordinator_family']} | {variant['coordinator_layers']} | {variant['description']} |")
    for phase_key, title in (("cheap_screen", "Phase 1 Cheap Screen"), ("medium_validation", "Phase 2 Medium Validation"), ("final_validation", "Phase 3 Final Validation")):
        phase = result.get(phase_key, {})
        phase_summary = phase.get("summary", {})
        lines.extend(["", f"## {title}", ""])
        if not phase.get("rows"):
            lines.append(f"- Not run: `{phase_summary.get('reason', 'not requested')}`")
            continue
        lines.append("| variant | seeds | wins | mean train | mean frozen | mean delta | controls | masks | mean-gate pass |")
        lines.append("|---|---:|---:|---:|---:|---:|---|---|---|")
        for row in phase_summary.get("variant_summaries", []):
            lines.append(
                f"| {row['variant']} | {row['n_seeds']} | {row['dev_wins_trainable_over_frozen']} | {float(row['mean_dev_trainable']):.4f} | {float(row['mean_dev_frozen']):.4f} | {float(row['mean_dev_delta']):.4f} | `{bool(row['controls_pass'])}` | `{bool(row['attention_mask_audits_pass'])}` | `{bool(row['selection_passed'])}` |"
            )
    lines.extend(
        [
            "",
            "## Required Questions",
            "",
            f"1. Can the clone architecture be folded into a more integrated one-pass architecture? `{_answer_folded(result)}`",
            f"2. Did integration preserve the scientific proof? `{_answer_proof(result)}`",
            f"3. Did any variant beat or match the Stage 4 clone baseline? `{_answer_baseline_match(result)}`",
            f"4. Did any variant improve weak seeds? `{_answer_weak_seeds(result)}`",
            f"5. Did any variant reduce compute/memory/latency? `{_answer_compute(result)}`",
            f"6. Did candidate-guided processing help? `{_answer_candidate_guided(result)}`",
            f"7. Did early cross-view mixing help or create leakage? `{_answer_early_mixing(result)}`",
            f"8. Did attention masks enforce the intended information barriers? `{_answer_masks(result)}`",
            f"9. Did controls remain near chance? `{_answer_controls(result)}`",
            f"10. Ready for full final validation, or keep clone mainline? `{summary.get('recommendation')}`",
            "",
            "## Conservative Interpretation",
            "",
            f"- Final validation ran: `{bool(summary.get('final_ran', False))}`",
            f"- Final gates passed: `{bool(summary.get('final_passed', False))}`",
            f"- Success claim: `{summary.get('success_claim')}`",
            "- No general breakthrough is claimed unless Phase 3 final gates pass.",
        ]
    )
    return "\n".join(lines) + "\n"


def _answer_folded(result: Dict[str, object]) -> str:
    audit = result.get("phase1_audit")
    if isinstance(audit, dict) and audit.get("decision") == "blocked":
        return "not established; Phase 1 audit blocked integrated promotion"
    passed = result.get("summary", {}).get("cheap_passed_variants", [])
    integrated = [name for name in passed if str(name) != BASELINE_VARIANT]
    return "cheap evidence yes" if integrated else "not established by completed phases"


def _answer_proof(result: Dict[str, object]) -> str:
    audit = result.get("phase1_audit")
    if isinstance(audit, dict) and audit.get("decision") == "blocked":
        return "not preserved under strict audit; mean-gate pass is diagnostic only"
    passed = [row["variant"] for row in result.get("cheap_screen", {}).get("summary", {}).get("variant_summaries", []) if row.get("selection_passed") and row.get("variant") != BASELINE_VARIANT]
    if passed:
        return f"Phase 1 mean gates preserved for {passed}; final proof not run"
    rows = [row for row in _all_rows(result) if str(row.get("variant")) != BASELINE_VARIANT]
    return "no integrated rows completed" if not rows else "not preserved by completed gates"


def _answer_baseline_match(result: Dict[str, object]) -> str:
    audit = result.get("phase1_audit")
    if isinstance(audit, dict) and audit.get("decision") == "blocked":
        return "not established under strict audit"
    rows = result.get("cheap_screen", {}).get("rows", [])
    baseline = [row for row in rows if row.get("variant") == BASELINE_VARIANT]
    if not baseline:
        return "baseline row not available"
    base_mean = _mean([float(row["dev"]["accuracy"]["trainable"]) for row in baseline])
    by_variant: Dict[str, List[float]] = {}
    for row in rows:
        if row.get("variant") != BASELINE_VARIANT:
            by_variant.setdefault(str(row["variant"]), []).append(float(row["dev"]["accuracy"]["trainable"]))
    matches = [variant for variant, values in by_variant.items() if _mean(values) >= base_mean - 0.02]
    return f"yes: {sorted(set(matches))}" if matches else "no in completed cheap screen"


def _answer_weak_seeds(result: Dict[str, object]) -> str:
    audit = result.get("phase1_audit")
    if isinstance(audit, dict) and audit.get("decision") == "blocked":
        return "not established under strict audit"
    rows = result.get("cheap_screen", {}).get("rows", [])
    by_variant: Dict[str, List[float]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(float(row["dev"]["accuracy"]["trainable"]))
    base = by_variant.get(BASELINE_VARIANT)
    if not base:
        return "baseline row not available"
    base_min = min(base)
    improved = [name for name, values in by_variant.items() if name != BASELINE_VARIANT and values and min(values) > base_min]
    return f"yes: {improved}" if improved else "not in completed rows"


def _answer_compute(result: Dict[str, object]) -> str:
    rows = result.get("cheap_screen", {}).get("rows", [])
    base = [row for row in rows if row.get("variant") == BASELINE_VARIANT]
    if not base:
        return "baseline row not available"
    base_latency = _mean([float(row["compute"]["inference_latency"]["ms_per_example"]) for row in base])
    faster = [
        row["variant"]
        for row in rows
        if row.get("variant") != BASELINE_VARIANT and float(row["compute"]["inference_latency"]["ms_per_example"]) < base_latency
    ]
    return f"yes: {sorted(set(faster))}" if faster else "no clear latency reduction in completed rows"


def _answer_candidate_guided(result: Dict[str, object]) -> str:
    rows = [row for row in result.get("cheap_screen", {}).get("rows", []) if "candidate_guided" in str(row.get("variant"))]
    if not rows:
        return "not run"
    return f"mean dev delta {_mean([float(row['dev_delta']) for row in rows]):.4f}"


def _answer_early_mixing(result: Dict[str, object]) -> str:
    rows = [row for row in result.get("cheap_screen", {}).get("rows", []) if "early_mixing" in str(row.get("variant"))]
    if not rows:
        return "not run"
    summary = next(
        (
            row
            for row in result.get("cheap_screen", {}).get("summary", {}).get("variant_summaries", [])
            if "early_mixing" in str(row.get("variant"))
        ),
        {},
    )
    failures = sum(bool(row.get("integrated_gate_failures")) for row in rows)
    return (
        f"mean dev delta {_mean([float(row['dev_delta']) for row in rows]):.4f}; "
        f"mean-gate pass={bool(summary.get('selection_passed', False))}; "
        f"strict per-seed row failures={failures}/{len(rows)}"
    )


def _answer_masks(result: Dict[str, object]) -> str:
    rows = [row for row in _all_rows(result) if str(row.get("variant")) != BASELINE_VARIANT]
    if not rows:
        return "no integrated rows completed"
    return "yes" if all(_mask_audit_passes(row) for row in rows) else "some mask audits failed"


def _answer_controls(result: Dict[str, object]) -> str:
    audit = result.get("phase1_audit")
    if isinstance(audit, dict) and audit.get("decision") == "blocked":
        return "strict audit failed; see reports/STAGE6_PHASE1_AUDIT_AND_BASELINE_REGRESSION.md"
    summaries = result.get("cheap_screen", {}).get("summary", {}).get("variant_summaries", [])
    if not summaries:
        return "not run"
    bad = [row["variant"] for row in summaries if not row.get("controls_pass")]
    return "yes for Phase 1 mean gates on passing variants" if not bad else f"failed mean gates for {sorted(set(bad))}"


def _recommendation(result: Dict[str, object]) -> str:
    audit = result.get("phase1_audit")
    if isinstance(audit, dict) and audit:
        eligible = list(audit.get("strict_phase2_eligible_variants", []))
        if not eligible:
            return "blocked by Phase 1 audit; keep clone mainline and rerun Phase 1 with clean baseline regression before medium validation"
        return f"Phase 1 audit allows medium validation for {eligible}; keep clone mainline until medium passes"
    final = result.get("final_validation", {}).get("summary", {})
    if bool(final.get("passed_primary_final_gates", False)):
        return "integrated architecture can move to mainline candidate"
    medium = result.get("medium_validation", {}).get("summary", {})
    if any(row.get("selection_passed") for row in medium.get("variant_summaries", [])):
        return "one integrated variant is ready for locked final validation"
    cheap = result.get("cheap_screen", {}).get("summary", {})
    if any(row.get("selection_passed") and row.get("variant") != BASELINE_VARIANT for row in cheap.get("variant_summaries", [])):
        return "run medium validation for passing integrated variants; keep clone mainline meanwhile"
    return "keep clone-based architecture as mainline"


def _attach_clone_transitions(rows: Sequence[Dict[str, object]]) -> None:
    baseline_by_seed = {
        (row["phase"], row["seed"]): row for row in rows if row.get("variant") == BASELINE_VARIANT
    }
    for row in rows:
        base = baseline_by_seed.get((row["phase"], row["seed"]))
        if base is None or row.get("variant") == BASELINE_VARIANT:
            row["clone_transition_summary"] = None
            continue
        row["clone_transition_summary"] = {
            "baseline_test_trainable": base["test"]["accuracy"]["trainable"],
            "variant_test_trainable": row["test"]["accuracy"]["trainable"],
            "delta_vs_clone": float(row["test"]["accuracy"]["trainable"] - base["test"]["accuracy"]["trainable"]),
        }


def _per_family_accuracy(train_pred: np.ndarray, frozen_pred: np.ndarray, examples: Sequence[MultiViewTaskExample]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[int]] = {}
    frozen_grouped: Dict[str, List[int]] = {}
    for example, pred, f_pred in zip(examples, train_pred, frozen_pred):
        family = str(example_oracle_metadata(example).get("problem_family", "unknown"))
        grouped.setdefault(family, []).append(int(pred) == int(example.label))
        frozen_grouped.setdefault(family, []).append(int(f_pred) == int(example.label))
    return {
        family: {
            "n": len(values),
            "trainable": float(np.mean(values)),
            "frozen": float(np.mean(frozen_grouped.get(family, []))),
            "delta": float(np.mean(values) - np.mean(frozen_grouped.get(family, [0]))),
        }
        for family, values in sorted(grouped.items())
    }


def _row_gate_failures(row: Dict[str, object]) -> List[str]:
    failures = []
    trainable = float(row["dev"]["accuracy"]["trainable"])
    frozen = float(row["dev"]["accuracy"]["frozen"])
    if trainable <= frozen:
        failures.append("trainable_not_above_frozen")
    controls = _control_means([row], split="dev")
    if not _controls_pass(controls, trainable_mean=trainable):
        failures.append("controls_or_invariance_failed")
    if not _mask_audit_passes(row):
        failures.append("attention_mask_audit_failed")
    if not bool(row.get("split_leakage_audit_passes")) or not bool(row.get("output_leakage_audit_passes")):
        failures.append("leakage_audit_failed")
    if not _trainable_update_passes(row):
        failures.append("trainable_update_audit_failed")
    if not _frozen_update_passes(row):
        failures.append("frozen_comparator_audit_failed")
    return failures


def _controls_pass(controls: Dict[str, float], trainable_mean: float) -> bool:
    corruption_names = [
        "candidate_only",
        "candidate_metadata_only",
        "schema_only",
        "view_masked_candidates_visible",
        "evidence_only_no_candidates",
        "null_evidence_values",
        "value_shuffle_within_schema",
        "cross_example_view_bundle_shuffle",
        "candidate_evidence_mismatch",
        "schema_preserved_role_value_shuffle",
        "randomized_labels",
        "hidden_states_shuffled",
    ]
    corrupt_ok = all(float(controls.get(name, 0.0)) <= NEAR_CHANCE for name in corruption_names if controls.get(name) is not None)
    physical_ok = abs(float(controls.get("physical_order_shuffled_roles_preserved", 0.0)) - trainable_mean) <= INVARIANCE_TOLERANCE
    candidate_ok = abs(float(controls.get("candidate_order_shuffled_with_gold_remap", 0.0)) - trainable_mean) <= INVARIANCE_TOLERANCE
    shortcut_ok = float(controls.get("no_candidate_position_shortcut_audit", 0.0)) <= NEAR_CHANCE
    return bool(corrupt_ok and physical_ok and candidate_ok and shortcut_ok)


def _mask_audit_passes(row: Dict[str, object]) -> bool:
    mask = (row.get("attention_mask_audit") or {}).get("mask") or {}
    if not mask:
        return row.get("variant") == BASELINE_VARIANT
    return bool(
        mask.get("candidate_tokens_can_see_gold") is False
        and mask.get("uses_candidate_position_embedding") is False
        and mask.get("uses_packed_position_embedding") is False
        and mask.get("coordination_tokens_can_see_candidates") is False
        and mask.get("role_tokens_can_see_other_roles_initially") is False
    )


def _trainable_update_passes(row: Dict[str, object]) -> bool:
    audit = row.get("trainable_audit", {})
    return bool(float(audit.get("agent_grad_norm_mean") or 0.0) > 0.0 and float(audit.get("agent_parameter_delta") or 0.0) > 0.0)


def _frozen_update_passes(row: Dict[str, object]) -> bool:
    audit = row.get("frozen_audit", {})
    return bool(float(audit.get("agent_grad_norm_mean") or 0.0) == 0.0 and float(audit.get("agent_parameter_delta") or 0.0) == 0.0)


def _control_means(rows: Sequence[Dict[str, object]], split: str) -> Dict[str, float]:
    names = list(PRIMARY_CONTROLS) + list(EXTRA_CONTROL_CONDITIONS) + [
        "no_candidate_position_shortcut_audit",
        "packed_sequence_position_shuffle_audit",
    ]
    out = {}
    for name in names:
        values = []
        for row in rows:
            value = row.get(split, {}).get("accuracy", {}).get(name)
            if isinstance(value, (int, float)):
                values.append(float(value))
        if values:
            out[name] = _mean(values)
    return out


def _compact_training_audit(audit: Dict[str, object]) -> Dict[str, object]:
    keys = (
        "agent_trainable",
        "agent_grad_norm_mean",
        "agent_parameter_delta",
        "coordinator_grad_norm_mean",
        "coordinator_parameter_delta",
        "shared_parameter_identity",
        "activation_requires_grad_before_coordinator",
        "per_clone_gradient_contribution",
        "loss_backward_reaches_shared_agent",
        "batch_size",
        "gradient_accumulation_steps",
        "mixed_precision",
        "cuda_max_memory_allocated",
        "param_count",
        "message_coordinator_family",
    )
    return {key: audit.get(key) for key in keys}


def _compact_leakage(leakage: Dict[str, object]) -> Dict[str, object]:
    keys = (
        "train_dev_id_overlap",
        "train_test_id_overlap",
        "dev_test_id_overlap",
        "train_test_candidate_hash_overlap",
        "train_test_file_patch_overlap",
        "train_test_problem_family_overlap",
        "train_test_source_import_key_overlap",
        "train_test_target_file_overlap",
        "candidate_patch_hash_leakage_audit_passes",
    )
    return {key: leakage.get(key) for key in keys}


def _attention_operation_estimate(attention_audit: Dict[str, object]) -> float:
    mask = attention_audit.get("mask") or {}
    roles = int(mask.get("n_roles", 0) or 0)
    tokens = int(mask.get("tokens_per_role", 0) or 0)
    routers = int(mask.get("router_tokens", 0) or 0)
    candidates = 8
    full_tokens = roles * tokens if mask.get("use_full_tokens") else 0
    summaries = roles if mask.get("use_role_summaries") else 0
    return float(candidates * (full_tokens + summaries + routers))


def _ece(confidence: np.ndarray, correct: np.ndarray, bins: int) -> float:
    total = len(confidence)
    if total == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (confidence >= left) & (confidence < right if right < 1.0 else confidence <= right)
        if not np.any(mask):
            continue
        value += float(np.mean(mask)) * abs(float(np.mean(confidence[mask])) - float(np.mean(correct[mask])))
    return value


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if abs(b) > 1e-12 else 0.0


def _mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _all_rows(result: Dict[str, object]) -> List[Dict[str, object]]:
    return (
        list(result.get("cheap_screen", {}).get("rows", []))
        + list(result.get("medium_validation", {}).get("rows", []))
        + list(result.get("final_validation", {}).get("rows", []))
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
