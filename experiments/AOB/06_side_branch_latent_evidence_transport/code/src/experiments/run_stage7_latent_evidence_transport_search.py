from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import pstdev
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

from src.datasets.multiview_code_patch_selection import (
    BALANCED_34B_CANDIDATE_REPRESENTATION,
    BALANCED_34B_DATASET_SOURCE,
    MultiViewTaskExample,
    apply_example_control,
    build_multiview_code_patch_splits,
    dataset_summary,
    output_leakage_audit,
    randomized_labels_for_examples,
    split_leakage_audit,
)
from src.experiments.architecture_search import (
    StageConfig,
    _agent_config,
    _audit_subset,
    _bootstrap_ci,
    _candidate_config,
    _fit_baselines,
    _training_config,
)
from src.experiments.real_shared_weight_latent_coordination import (
    BENCHMARK,
    fit_latent_system,
    predict_context_baseline,
    predict_latent_system,
)
from src.experiments.run_stage3_gpu_hard_validation import (
    _audit_float,
    _clear_cuda,
    _compact_leakage,
    _configure_cuda,
    _failure_row,
    _is_oom,
    _load_result,
    _save_bow_checkpoint,
    _save_context_checkpoint,
    _save_fixed_checkpoint,
    _save_latent_checkpoint,
    _save_oracle_checkpoint,
    _save_text_checkpoint,
)
from src.experiments.run_stage34_candidate_sanitization_diagnostics import _accuracy, _structured_oracle_predictions
from src.experiments.run_stage35_model_facing_learnability import CompatibilityFeatureScorer
from src.experiments.run_stage38_full_hard_validation import (
    _dataset_config_for_stage,
    _labels,
    _positive_training,
    _pre_run_checks,
    _stage38_full_config,
    _stage_from_config,
)
from src.experiments.run_stage38_schema_aware_controls import _zero_role_embeddings
from src.experiments.run_stage4_final_candidate_token_direct_validation import (
    INVARIANCE_CONTROLS,
    LATENT_CONTROLS,
    PRIMARY_CONTROLS,
    _control_examples,
)
from src.experiments.stage7_variant_registry import (
    Stage7VariantSpec,
    stage7_ablation_plan,
    stage7b_medium_variant_plan,
    stage7_variant_by_name,
    stage7_variant_plan,
)


DEFAULT_CONFIG = "configs/stage4_final_candidate_token_direct_validation.json"
STAGE7A_OUTPUT = Path("results/stage7a_architecture_screen_results.json")
STAGE7A_AUDIT = Path("results/stage7a_architecture_screen_audit.jsonl")
STAGE7A_REPORT = Path("reports/STAGE7A_ARCHITECTURE_SCREEN.md")
STAGE7B_OUTPUT = Path("results/stage7b_medium_validation_results.json")
STAGE7B_AUDIT = Path("results/stage7b_medium_validation_audit.jsonl")
STAGE7B_REPORT = Path("reports/STAGE7B_MEDIUM_VALIDATION.md")
STAGE7C_OUTPUT = Path("results/stage7c_final_latent_evidence_transport_results.json")
STAGE7C_AUDIT = Path("results/stage7c_final_latent_evidence_transport_audit.jsonl")
STAGE7C_REPORT = Path("reports/STAGE7C_FINAL_LATENT_EVIDENCE_TRANSPORT.md")
DEFAULT_CHECKPOINT_DIR = Path("results/stage7_latent_evidence_transport_checkpoints")
LOCKED_STAGE5_BASELINE = {
    "name": "stage5_4role_x4avenue_candidate_token_direct",
    "trainable_mean_accuracy": 0.9144,
    "frozen_mean_accuracy": 0.2653,
    "delta": 0.6490,
    "source": "locked user-provided Stage 5 baseline",
}
LOCKED_STAGE4_BASELINE = {
    "name": "stage4_candidate_token_direct_lr3e4_clip1",
    "trainable_mean_accuracy": 0.7892,
    "frozen_mean_accuracy": 0.1518,
    "delta": 0.6374,
    "source": "locked user-provided Stage 4 baseline",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 7 latent evidence transport architecture search.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("stage7a", "stage7b", "stage7c", "all"), default="stage7a")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-variants", type=int, default=0)
    parser.add_argument("--selected-variant", default=None)
    parser.add_argument("--stage5-results", default=None)
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.device is not None:
        config["device"] = str(args.device)
        if str(args.device) == "cpu":
            config["require_cuda"] = False
    run_stage7(
        config=config,
        config_path=config_path,
        phase=str(args.phase),
        max_variants=int(args.max_variants),
        selected_variant_name=args.selected_variant,
        stage5_results_path=Path(args.stage5_results) if args.stage5_results else None,
        checkpoint_dir=Path(args.checkpoint_dir),
        skip_preflight=bool(args.skip_preflight),
    )


def run_stage7(
    config: Dict[str, object],
    config_path: Path,
    phase: str,
    max_variants: int = 0,
    selected_variant_name: str | None = None,
    stage5_results_path: Path | None = None,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    skip_preflight: bool = False,
) -> Dict[str, object]:
    config = _stage7_base_config(config)
    device, hardware = _configure_cuda(config)
    stage5_baseline = _load_stage5_baseline(stage5_results_path)
    outputs: Dict[str, object] = {}
    if phase in {"stage7a", "all"}:
        variants = stage7_variant_plan(max_variants=max_variants)
        outputs["stage7a"] = run_stage7a(
            config,
            config_path,
            variants,
            device,
            hardware,
            stage5_baseline,
            checkpoint_dir,
        )
    if phase in {"stage7b", "all"}:
        variants = _stage7b_variants_from_stage7a(outputs.get("stage7a"), selected_variant_name, max_variants)
        outputs["stage7b"] = run_stage7b(
            config,
            config_path,
            variants,
            device,
            hardware,
            stage5_baseline,
            checkpoint_dir,
        )
    if phase in {"stage7c", "all"}:
        selected = _selected_stage7c_variant(outputs.get("stage7b"), selected_variant_name)
        outputs["stage7c"] = run_stage7c(
            config,
            config_path,
            selected,
            device,
            hardware,
            stage5_baseline,
            checkpoint_dir,
            skip_preflight=skip_preflight,
        )
    return outputs.get(phase) if phase != "all" else outputs


def run_stage7a(
    config: Dict[str, object],
    config_path: Path,
    variants: Sequence[Stage7VariantSpec],
    device: str,
    hardware: Dict[str, object],
    stage5_baseline: Dict[str, object],
    checkpoint_dir: Path,
) -> Dict[str, object]:
    stage = _phase_stage(config, "stage7a")
    result = {
        "metadata": _metadata("stage7a_architecture_screen", config_path, hardware, device),
        "config": _phase_config_dict(config, stage),
        "locked_baselines": _locked_baselines(stage5_baseline),
        "variant_plan": [variant.to_dict() for variant in variants],
        "stage7a_rows": [],
        "failed_or_interrupted_variants": [],
        "selection_policy": "dev-only architecture screen; held-out test split is not built or evaluated",
    }
    print(f"stage7a: dev-only architecture screen starting for {len(variants)} variants")
    rows, failures = _run_variant_phase(
        phase="stage7a",
        config=config,
        stage=stage,
        variants=variants,
        device=device,
        checkpoint_dir=checkpoint_dir / "stage7a",
        save_checkpoints=False,
        include_baselines=True,
        include_randomized=True,
    )
    result["stage7a_rows"] = rows
    result["failed_or_interrupted_variants"] = failures
    _write_stage_outputs(result, STAGE7A_OUTPUT, STAGE7A_AUDIT, STAGE7A_REPORT, "stage7a")
    return result


def run_stage7b(
    config: Dict[str, object],
    config_path: Path,
    variants: Sequence[Stage7VariantSpec],
    device: str,
    hardware: Dict[str, object],
    stage5_baseline: Dict[str, object],
    checkpoint_dir: Path,
) -> Dict[str, object]:
    stage = _phase_stage(config, "stage7b")
    result = _load_result(STAGE7B_OUTPUT)
    result.setdefault("metadata", _metadata("stage7b_medium_validation", config_path, hardware, device))
    result["metadata"].update(_metadata("stage7b_medium_validation", config_path, hardware, device))
    result["config"] = _phase_config_dict(config, stage)
    result["locked_baselines"] = _locked_baselines(stage5_baseline)
    result["variant_plan"] = [variant.to_dict() for variant in variants]
    result.setdefault("stage7b_rows", [])
    result["stage7b_rows"] = [
        row
        for row in result.get("stage7b_rows", [])
        if row.get("status") == "completed" and row.get("variant") in {variant.name for variant in variants}
    ]
    result["failed_or_interrupted_variants"] = []
    result["selection_policy"] = "dev-only medium validation of top Stage 7A variants plus requested ablations; no held-out test split is built or evaluated"
    if not variants:
        result["failed_or_interrupted_variants"].append({"status": "not_started", "reason": "no Stage 7A variant passed selection gates"})
        _write_stage_outputs(result, STAGE7B_OUTPUT, STAGE7B_AUDIT, STAGE7B_REPORT, "stage7b")
        return result
    _write_stage_outputs(result, STAGE7B_OUTPUT, STAGE7B_AUDIT, STAGE7B_REPORT, "stage7b")
    print(f"stage7b: medium dev validation starting for variants {[variant.name for variant in variants]}")
    rows, failures = _run_variant_phase(
        phase="stage7b",
        config=config,
        stage=stage,
        variants=variants,
        device=device,
        checkpoint_dir=checkpoint_dir / "stage7b",
        save_checkpoints=True,
        include_baselines=True,
        include_randomized=True,
        existing_rows=result["stage7b_rows"],
        row_callback=lambda current_rows, current_failures: _write_stage_outputs(
            {**result, "stage7b_rows": current_rows, "failed_or_interrupted_variants": current_failures},
            STAGE7B_OUTPUT,
            STAGE7B_AUDIT,
            STAGE7B_REPORT,
            "stage7b",
        ),
    )
    result["stage7b_rows"] = rows
    result["failed_or_interrupted_variants"] = failures
    _write_stage_outputs(result, STAGE7B_OUTPUT, STAGE7B_AUDIT, STAGE7B_REPORT, "stage7b")
    return result


def run_stage7c(
    config: Dict[str, object],
    config_path: Path,
    selected: Stage7VariantSpec | None,
    device: str,
    hardware: Dict[str, object],
    stage5_baseline: Dict[str, object],
    checkpoint_dir: Path,
    skip_preflight: bool = False,
) -> Dict[str, object]:
    stage = _phase_stage(config, "stage7c")
    result = _load_result(STAGE7C_OUTPUT)
    result.setdefault("metadata", _metadata("stage7c_final_latent_evidence_transport", config_path, hardware, device))
    result["metadata"].update(_metadata("stage7c_final_latent_evidence_transport", config_path, hardware, device))
    result["config"] = _phase_config_dict(config, stage)
    result["locked_baselines"] = _locked_baselines(stage5_baseline)
    result.setdefault("stage7c_rows", [])
    result.setdefault("stage7c_ablations", [])
    result.setdefault("failed_or_interrupted_seeds", [])
    result.setdefault("oom_retries", [])
    if selected is None:
        result["selected_architecture"] = None
        result["failed_or_interrupted_seeds"].append({"status": "not_started", "reason": "no Stage 7B selected architecture"})
        _write_stage_outputs(result, STAGE7C_OUTPUT, STAGE7C_AUDIT, STAGE7C_REPORT, "stage7c")
        return result
    result["selected_architecture"] = selected.to_dict()
    result["selection_policy"] = "single architecture selected from Stage 7B dev-only validation before test access"
    if not skip_preflight:
        print("stage7c: running final pre-run shortcut diagnostics")
        result["pre_run"] = _pre_run_checks(_phase_config_dict(config, stage), config_path, device)
        _write_stage_outputs(result, STAGE7C_OUTPUT, STAGE7C_AUDIT, STAGE7C_REPORT, "stage7c")
        if bool(config.get("stop_on_preflight_failure", True)) and not bool(result["pre_run"]["summary"]["passes"]):
            print("stage7c: pre-run checks failed; final validation was not started")
            return result
    completed = {
        int(row["seed"])
        for row in result.get("stage7c_rows", [])
        if row.get("status") == "completed" and row.get("variant") == selected.name and _row_has_stage7_checkpoints(row, result["config"])
    }
    fallbacks = [int(value) for value in config.get("gpu_safety", {}).get("batch_size_fallbacks", [32, 16, 8])]
    effective_batch = int(config.get("gpu_safety", {}).get("effective_batch_size", fallbacks[0]))
    for seed in [int(value) for value in stage.seeds]:
        if seed in completed:
            print(f"stage7c seed={seed}: existing completed row found; skipping")
            continue
        result["stage7c_rows"] = [row for row in result.get("stage7c_rows", []) if int(row.get("seed", -1)) != seed]
        for batch_size in fallbacks:
            accumulation = max(1, int(math.ceil(effective_batch / float(batch_size))))
            seed_stage = replace(stage, seeds=(seed,), batch_size=batch_size, gradient_accumulation_steps=accumulation)
            try:
                row = _run_seed_variant(
                    phase="stage7c",
                    config=config,
                    stage=seed_stage,
                    variant=selected,
                    seed=seed,
                    device=device,
                    checkpoint_dir=checkpoint_dir / "stage7c",
                    save_checkpoints=True,
                    include_baselines=True,
                    include_randomized=True,
                    include_test=True,
                )
                result["stage7c_rows"].append(row)
                _write_stage_outputs(result, STAGE7C_OUTPUT, STAGE7C_AUDIT, STAGE7C_REPORT, "stage7c")
                print(
                    "stage7c seed={seed}: completed trainable={trainable:.4f} frozen={frozen:.4f} delta={delta:.4f}".format(
                        seed=seed,
                        trainable=float(row["test_accuracy"]["trainable"]),
                        frozen=float(row["test_accuracy"]["frozen"]),
                        delta=float(row["test_delta"]),
                    )
                )
                break
            except RuntimeError as exc:
                if _is_oom(exc) and batch_size != fallbacks[-1]:
                    retry = _failure_row(seed, batch_size, accumulation, "oom_retry", exc)
                    retry["variant"] = selected.name
                    result["oom_retries"].append(retry)
                    _clear_cuda()
                    _write_stage_outputs(result, STAGE7C_OUTPUT, STAGE7C_AUDIT, STAGE7C_REPORT, "stage7c")
                    print(f"stage7c seed={seed}: CUDA OOM at batch_size={batch_size}; retrying smaller batch")
                    continue
                failure = _failure_row(seed, batch_size, accumulation, "failed", exc)
                failure["variant"] = selected.name
                result["failed_or_interrupted_seeds"].append(failure)
                _clear_cuda()
                _write_stage_outputs(result, STAGE7C_OUTPUT, STAGE7C_AUDIT, STAGE7C_REPORT, "stage7c")
                break
            except Exception as exc:
                result["failed_or_interrupted_seeds"].append(_exception_row(seed, selected.name, batch_size, accumulation, exc))
                _clear_cuda()
                _write_stage_outputs(result, STAGE7C_OUTPUT, STAGE7C_AUDIT, STAGE7C_REPORT, "stage7c")
                break
            finally:
                _clear_cuda()
    if bool(config.get("stage7c", {}).get("run_ablations", False)) and not result.get("stage7c_ablations"):
        result["stage7c_ablations"] = _run_stage7c_ablations(config, stage, selected, device, checkpoint_dir)
    _write_stage_outputs(result, STAGE7C_OUTPUT, STAGE7C_AUDIT, STAGE7C_REPORT, "stage7c")
    return result


def _run_variant_phase(
    phase: str,
    config: Dict[str, object],
    stage: StageConfig,
    variants: Sequence[Stage7VariantSpec],
    device: str,
    checkpoint_dir: Path,
    save_checkpoints: bool,
    include_baselines: bool,
    include_randomized: bool,
    existing_rows: Sequence[Dict[str, object]] | None = None,
    row_callback=None,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = list(existing_rows or [])
    failures: List[Dict[str, object]] = []
    completed = {(str(row.get("variant")), int(row.get("seed", -1))) for row in rows if row.get("status") == "completed"}
    for seed in [int(value) for value in stage.seeds]:
        print(f"{phase} seed={seed}: building dev-only schema-aware splits")
        splits = build_multiview_code_patch_splits(_dataset_config_for_stage(_phase_config_dict(config, stage), stage), seed=seed, repo_root=Path("."))
        baselines = _fit_baselines(stage, splits, seed=seed, device=device) if include_baselines else {}
        positive = _fit_positive_control(config, splits, seed)
        for variant in variants:
            if (variant.name, seed) in completed:
                print(f"{phase} variant={variant.name} seed={seed}: existing completed row found; skipping")
                continue
            try:
                row = _run_seed_variant(
                    phase=phase,
                    config=config,
                    stage=stage,
                    variant=variant,
                    seed=seed,
                    device=device,
                    checkpoint_dir=checkpoint_dir,
                    save_checkpoints=save_checkpoints,
                    include_baselines=include_baselines,
                    include_randomized=include_randomized,
                    include_test=False,
                    prebuilt_splits=splits,
                    prebuilt_baselines=baselines,
                    prebuilt_positive=positive,
                )
                rows.append(row)
                completed.add((variant.name, seed))
                print(
                    "{phase} variant={variant} seed={seed}: dev trainable={trainable:.4f} frozen={frozen:.4f}".format(
                        phase=phase,
                        variant=variant.name,
                        seed=seed,
                        trainable=float(row["dev_accuracy"]["trainable"]),
                        frozen=float(row["dev_accuracy"]["frozen"]),
                    )
                )
                if row_callback is not None:
                    row_callback(rows, failures)
            except RuntimeError as exc:
                status = "oom_failed" if _is_oom(exc) else "failed"
                failures.append(_variant_failure_row(phase, variant.name, seed, status, exc))
                _clear_cuda()
                if row_callback is not None:
                    row_callback(rows, failures)
            except Exception as exc:
                failures.append(_variant_failure_row(phase, variant.name, seed, "failed", exc))
                _clear_cuda()
                if row_callback is not None:
                    row_callback(rows, failures)
            finally:
                _clear_cuda()
    return rows, failures


def _run_seed_variant(
    phase: str,
    config: Dict[str, object],
    stage: StageConfig,
    variant: Stage7VariantSpec,
    seed: int,
    device: str,
    checkpoint_dir: Path,
    save_checkpoints: bool,
    include_baselines: bool,
    include_randomized: bool,
    include_test: bool,
    prebuilt_splits: Dict[str, Sequence[MultiViewTaskExample]] | None = None,
    prebuilt_baselines: Dict[str, object] | None = None,
    prebuilt_positive: CompatibilityFeatureScorer | None = None,
) -> Dict[str, object]:
    _clear_cuda()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    phase_config = _phase_config_dict(config, stage)
    splits = prebuilt_splits or build_multiview_code_patch_splits(_dataset_config_for_stage(phase_config, stage), seed=seed, repo_root=Path("."))
    split_leakage = split_leakage_audit(BENCHMARK, seed, splits)
    output_examples = list(splits["train"]) + list(splits["dev"]) + (list(splits["test"]) if include_test else [])
    output_leakage = output_leakage_audit(BENCHMARK, seed, output_examples)
    baselines = prebuilt_baselines or (_fit_baselines(stage, splits, seed=seed, device=device) if include_baselines else {})
    positive = prebuilt_positive or _fit_positive_control(config, splits, seed)
    candidate = variant.to_candidate(stage)
    training = replace(
        _training_config(stage),
        epochs=int(variant.epochs),
        patience=int(variant.patience),
        lr=float(variant.lr),
        weight_decay=float(variant.weight_decay),
        gradient_clip_norm=float(variant.gradient_clip_norm),
    )
    fit_start = time.perf_counter()
    fit = _fit_stage7_methods(candidate, variant, training, stage, splits, seed, device, include_randomized)
    fit_seconds = time.perf_counter() - fit_start
    eval_start = time.perf_counter()
    metrics = _stage7_metrics(fit, baselines, positive, splits, seed, include_test=include_test)
    eval_seconds = time.perf_counter() - eval_start
    checkpoint_paths = _save_stage7_checkpoints(seed, stage, variant, candidate, fit, baselines, positive, checkpoint_dir, save_checkpoints)
    cuda_peak = int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") and torch.cuda.is_available() else 0
    row = {
        "stage": phase,
        "variant": variant.name,
        "variant_family": variant.family,
        "description": variant.description,
        "seed": int(seed),
        "status": "completed",
        "completed_at_utc": _now(),
        "stage_config": asdict(stage),
        "variant_config": variant.to_dict(),
        "architecture_config": _candidate_config(candidate),
        "dataset_summary": dataset_summary(splits),
        "split_leakage_audit": _compact_leakage(split_leakage),
        "output_leakage_audit": output_leakage,
        "split_leakage_audit_passes": _split_leakage_passes(split_leakage, include_test=include_test),
        "output_leakage_audit_passes": bool(output_leakage.get("passes", False)),
        "dev_accuracy": metrics["dev"],
        "dev_delta": float(metrics["dev"]["trainable"] - metrics["dev"]["frozen"]),
        "test_accuracy": metrics.get("test", {}),
        "test_delta": float(metrics.get("test", {}).get("trainable", 0.0) - metrics.get("test", {}).get("frozen", 0.0)) if include_test else None,
        "trainable_audit": _audit_subset(fit["trainable"].audit),
        "frozen_audit": _audit_subset(fit["frozen"].audit),
        "randomized_audit": _audit_subset(fit["randomized"].audit) if "randomized" in fit else {},
        "transport_audit": fit["trainable"].audit.get("latent_evidence_transport_config", {}),
        "param_count": int(fit["trainable"].param_count),
        "frozen_param_count": int(fit["frozen"].param_count),
        "fit_latency_sec": float(fit_seconds),
        "eval_latency_sec": float(eval_seconds),
        "cuda_max_memory_allocated": cuda_peak,
        "cuda_max_memory_allocated_gb": cuda_peak / float(1024**3),
        "checkpoint_paths": checkpoint_paths,
        "per_family_accuracy": _per_family_compare(fit["trainable"], fit["frozen"], splits["test" if include_test else "dev"], seed),
        "compute_overhead": {
            "refinement_steps": int(variant.refinement_steps),
            "workspace_slots": int(variant.workspace_slots),
            "fit_latency_sec": float(fit_seconds),
            "eval_latency_sec": float(eval_seconds),
            "cuda_max_memory_allocated_gb": cuda_peak / float(1024**3),
            "param_count": int(fit["trainable"].param_count),
        },
    }
    row["seed_criteria"] = _seed_criteria(row, config, include_test=include_test)
    return row


def _fit_stage7_methods(
    candidate,
    variant: Stage7VariantSpec,
    training,
    stage: StageConfig,
    splits: Dict[str, Sequence[MultiViewTaskExample]],
    seed: int,
    device: str,
    include_randomized: bool,
) -> Dict[str, object]:
    agent_config = _agent_config(stage)
    trainable = fit_latent_system(
        train_examples=splits["train"],
        dev_examples=splits["dev"],
        agent_config=agent_config,
        coordinator_config=candidate.coordinator_config,
        training_config=training,
        num_classes=8,
        seed=seed + 10_101,
        device=device,
        trainable_agent=True,
        method=f"stage7_trainable__{variant.name}",
        message_config=candidate.message_config,
    )
    frozen = fit_latent_system(
        train_examples=splits["train"],
        dev_examples=splits["dev"],
        agent_config=agent_config,
        coordinator_config=candidate.coordinator_config,
        training_config=training,
        num_classes=8,
        seed=seed + 20_201,
        device=device,
        trainable_agent=False,
        method=f"stage7_frozen__{variant.name}",
        message_config=candidate.message_config,
    )
    out = {"trainable": trainable, "frozen": frozen}
    if include_randomized:
        randomized = fit_latent_system(
            train_examples=splits["train"],
            dev_examples=splits["dev"],
            agent_config=agent_config,
            coordinator_config=candidate.coordinator_config,
            training_config=training,
            num_classes=8,
            seed=seed + 30_301,
            device=device,
            trainable_agent=True,
            method=f"stage7_randomized_labels__{variant.name}",
            condition="randomized_labels",
            train_labels=randomized_labels_for_examples(splits["train"], seed + 40_401, 8),
            dev_labels=randomized_labels_for_examples(splits["dev"], seed + 50_501, 8),
            message_config=candidate.message_config,
        )
        out["randomized"] = randomized
    return out


def _stage7_metrics(
    fit: Dict[str, object],
    baselines: Dict[str, object],
    positive: CompatibilityFeatureScorer,
    splits: Dict[str, Sequence[MultiViewTaskExample]],
    seed: int,
    include_test: bool,
) -> Dict[str, Dict[str, float]]:
    out = {}
    for split_name in ("dev", "test") if include_test else ("dev",):
        examples = list(splits[split_name])
        split_seed = seed + (1_000 if split_name == "dev" else 2_000)
        y = _labels(examples)
        values = {
            "trainable": _accuracy(predict_latent_system(fit["trainable"], examples, "none", split_seed), y),
            "frozen": _accuracy(predict_latent_system(fit["frozen"], examples, "none", split_seed), y),
            "candidate_pair_compatibility_mlp": _accuracy(positive.predict(examples), y),
            "all_role_oracle": _accuracy(_structured_oracle_predictions(examples, [0, 1, 2, 3]), y),
            "explicit_evidence_oracle": _accuracy(_structured_oracle_predictions(examples, [0, 1, 2, 3]), y),
            "hidden_states_shuffled_across_examples": _accuracy(
                predict_latent_system(fit["trainable"], examples, "hidden_states_shuffled_across_examples", split_seed), y
            ),
            "physical_order_shuffled_roles_avenues_preserved": _accuracy(
                predict_latent_system(fit["trainable"], examples, "physical_order_shuffled_roles_avenues_preserved", split_seed), y
            ),
            "physical_order_shuffled_roles_preserved": _accuracy(
                predict_latent_system(fit["trainable"], examples, "physical_order_shuffled_roles_preserved", split_seed), y
            ),
        }
        if "randomized" in fit:
            values["randomized_labels"] = _accuracy(predict_latent_system(fit["randomized"], examples, "none", split_seed), y)
        if baselines:
            values.update(
                {
                    "text_only": _accuracy(baselines["text"].predict(examples), y),
                    "raw_latent": _accuracy(predict_latent_system(baselines["raw"], examples, "none", split_seed), y),
                    "majority_baseline": float(np.mean(np.full(len(y), int(baselines["majority_prediction"]), dtype=np.int64) == y)),
                    "candidate_order_baseline": float(np.mean(np.full(len(y), int(baselines["candidate_order_prediction"]), dtype=np.int64) == y)),
                    "single_agent_full_context": _accuracy(predict_context_baseline(baselines["full_context"], examples), y),
                }
            )
            for method, baseline in dict(baselines.get("single_view", {})).items():
                values[str(method)] = _accuracy(baseline.predict(examples), y)
        for name, rows in _control_examples(examples, split_seed).items():
            cy = _labels(rows)
            if name == "candidate_only":
                with _zero_role_embeddings(fit["trainable"].system):
                    pred = predict_latent_system(fit["trainable"], rows, "none", split_seed)
            else:
                pred = predict_latent_system(fit["trainable"], rows, "none", split_seed)
            values[name] = _accuracy(pred, cy)
        shuffled = apply_example_control(examples, "candidate_order_shuffled", seed=split_seed + 77_000)
        values["candidate_order_shuffled_with_label_remap"] = _accuracy(
            predict_latent_system(fit["trainable"], shuffled, "none", split_seed), _labels(shuffled)
        )
        values["candidate_order_shuffled_with_gold_remap"] = values["candidate_order_shuffled_with_label_remap"]
        values["candidate_order_shuffled"] = values["candidate_order_shuffled_with_label_remap"]
        values["view_masked"] = values["view_masked_candidates_visible"]
        values["role_labels_shuffled"] = values["physical_order_shuffled_roles_preserved"]
        out[split_name] = values
    return out


def _fit_positive_control(config: Dict[str, object], splits: Dict[str, Sequence[MultiViewTaskExample]], seed: int) -> CompatibilityFeatureScorer:
    scorer = CompatibilityFeatureScorer(training=_positive_training(config), seed=seed + 14_000)
    scorer.fit(splits["train"], splits["dev"])
    return scorer


def _save_stage7_checkpoints(
    seed: int,
    stage: StageConfig,
    variant: Stage7VariantSpec,
    candidate,
    fit: Dict[str, object],
    baselines: Dict[str, object],
    positive: CompatibilityFeatureScorer,
    checkpoint_dir: Path,
    save_checkpoints: bool,
) -> Dict[str, str]:
    if not save_checkpoints:
        return {}
    root = checkpoint_dir / variant.name / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "trainable": root / "trainable.pt",
        "frozen": root / "frozen.pt",
        "randomized_labels": root / "randomized_labels.pt",
        "text_only": root / "text_only_multi_agent_baseline.pt",
        "raw_latent": root / "raw_latent_baseline.pt",
        "majority_baseline": root / "majority_baseline.pt",
        "candidate_order_baseline": root / "candidate_order_baseline.pt",
        "single_agent_full_context": root / "single_agent_full_context.pt",
        "single_view_text_role_0": root / "single_view_text_role_0.pt",
        "single_view_text_role_1": root / "single_view_text_role_1.pt",
        "single_view_text_role_2": root / "single_view_text_role_2.pt",
        "single_view_text_role_3": root / "single_view_text_role_3.pt",
        "candidate_pair_compatibility_mlp": root / "candidate_pair_compatibility_mlp.pt",
        "explicit_evidence_oracle": root / "explicit_evidence_oracle.pt",
    }
    _save_latent_checkpoint(paths["trainable"], seed, stage, candidate.name, _candidate_config(candidate), fit["trainable"])
    _save_latent_checkpoint(paths["frozen"], seed, stage, candidate.name, _candidate_config(candidate), fit["frozen"])
    if "randomized" in fit:
        _save_latent_checkpoint(paths["randomized_labels"], seed, stage, candidate.name, _candidate_config(candidate), fit["randomized"])
    if baselines:
        _save_text_checkpoint(paths["text_only"], seed, stage, baselines["text"])
        _save_latent_checkpoint(paths["raw_latent"], seed, stage, "raw_latent_baseline", {}, baselines["raw"])
        _save_fixed_checkpoint(paths["majority_baseline"], seed, stage, "majority_baseline", int(baselines["majority_prediction"]))
        _save_fixed_checkpoint(paths["candidate_order_baseline"], seed, stage, "candidate_order_baseline", int(baselines["candidate_order_prediction"]))
        _save_context_checkpoint(paths["single_agent_full_context"], seed, stage, baselines["full_context"])
        for method, baseline in dict(baselines.get("single_view", {})).items():
            if method in paths:
                _save_bow_checkpoint(paths[method], seed, stage, baseline)
    _save_stage7_positive_checkpoint(paths["candidate_pair_compatibility_mlp"], seed, stage, positive)
    _save_oracle_checkpoint(paths["explicit_evidence_oracle"], seed, stage)
    return {name: str(path) for name, path in paths.items() if path.exists()}


def _save_stage7_positive_checkpoint(path: Path, seed: int, stage: StageConfig, scorer: CompatibilityFeatureScorer) -> None:
    payload = {
        "seed": int(seed),
        "stage": asdict(stage),
        "method": "candidate_pair_compatibility_mlp",
        "param_count": int(getattr(scorer, "param_count", 0)),
        "history": getattr(scorer, "history", []),
    }
    torch.save(payload, path)


def _write_stage_outputs(result: Dict[str, object], output_path: Path, audit_path: Path, report_path: Path, phase: str) -> None:
    if phase == "stage7c":
        _attach_pre_run_candidate_metadata(result)
    result["summary"] = _summary(result, phase)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(output_path)
    audit_rows = _audit_rows(result, phase)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in audit_rows) + ("\n" if audit_rows else ""), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(result, phase), encoding="utf-8")


def _summary(result: Dict[str, object], phase: str) -> Dict[str, object]:
    row_key = {"stage7a": "stage7a_rows", "stage7b": "stage7b_rows", "stage7c": "stage7c_rows"}[phase]
    rows = [row for row in result.get(row_key, []) if row.get("status") == "completed"]
    baseline = result.get("locked_baselines", {}).get("stage5", LOCKED_STAGE5_BASELINE)
    if phase in {"stage7a", "stage7b"}:
        summaries = _variant_summaries(rows, result.get("config", {}), baseline, split="dev")
        if phase == "stage7b":
            summaries, selected_names = _stage7b_select_against_widened(summaries)
        else:
            selected = [item for item in summaries if item["selection_eligible"]]
            selected.sort(key=lambda item: (float(item["mean_trainable_accuracy"]), float(item["mean_delta_vs_frozen"])), reverse=True)
            selected_names = [item["variant"] for item in selected[:3]]
        return {
            "n_completed_rows": len(rows),
            "n_failed": len(result.get("failed_or_interrupted_variants", [])),
            "variant_summaries": summaries,
            "selected_top_variants": selected_names,
            "primary_stage7b_comparator": "f_stage5_widened_param_control" if phase == "stage7b" else "locked_stage5_mean",
            "stage5_reference_trainable_mean": float(baseline.get("trainable_mean_accuracy", 0.9144)),
            "test_set_accessed": False,
        }
    return _stage7c_summary(rows, result, baseline)


def _variant_summaries(rows: Sequence[Dict[str, object]], config: Dict[str, object], baseline: Dict[str, object], split: str) -> List[Dict[str, object]]:
    near = float(config.get("near_chance_max", 0.18))
    inv_tol = float(config.get("invariance_tolerance", 0.08))
    baseline_key = "dev_mean_accuracy" if split == "dev" and "dev_mean_accuracy" in baseline else "trainable_mean_accuracy"
    stage5_mean = float(baseline.get(baseline_key, 0.9144))
    by_variant: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    out = []
    for variant, values in sorted(by_variant.items()):
        acc = [float(row[f"{split}_accuracy"]["trainable"]) for row in values]
        frozen = [float(row[f"{split}_accuracy"]["frozen"]) for row in values]
        deltas = [a - f for a, f in zip(acc, frozen)]
        control_means = {
            name: _mean([float(row[f"{split}_accuracy"].get(name, 1.0)) for row in values])
            for name in LATENT_CONTROLS + ("randomized_labels", "hidden_states_shuffled_across_examples")
        }
        invariance = {
            name: _mean([float(row[f"{split}_accuracy"].get(name, 0.0)) for row in values])
            for name in ("physical_order_shuffled_roles_avenues_preserved", "candidate_order_shuffled_with_label_remap")
        }
        per_family = _aggregate_per_family(values)
        hard_family_accuracy = min((float(item.get("trainable", 0.0)) for item in per_family.values()), default=0.0)
        train_mean = _mean(acc)
        controls_pass = all(value <= near for value in control_means.values())
        invariance_pass = all(abs(value - train_mean) <= inv_tol for value in invariance.values())
        gradient_pass = all(
            float(row.get("trainable_audit", {}).get("agent_grad_norm_mean") or 0.0) > 0.0
            and float(row.get("trainable_audit", {}).get("agent_parameter_delta") or 0.0) > 0.0
            and float(row.get("trainable_audit", {}).get("coordinator_grad_norm_mean") or 0.0) > 0.0
            for row in values
        )
        frozen_pass = all(
            _audit_float(row.get("frozen_audit", {}), "agent_grad_norm_mean", 1.0) == 0.0
            and _audit_float(row.get("frozen_audit", {}), "agent_parameter_delta", 1.0) == 0.0
            for row in values
        )
        beats_stage5 = train_mean > stage5_mean
        frozen_far_below = _mean(deltas) >= 0.20
        selection_eligible = bool(beats_stage5 and frozen_far_below and controls_pass and invariance_pass and gradient_pass and frozen_pass)
        out.append(
            {
                "variant": variant,
                "family": values[0].get("variant_family"),
                "n_seeds": len(values),
                "mean_trainable_accuracy": train_mean,
                "mean_frozen_accuracy": _mean(frozen),
                "mean_delta_vs_frozen": _mean(deltas),
                "delta_vs_locked_stage5": train_mean - stage5_mean,
                "weak_seed_count": sum(value < stage5_mean for value in acc),
                "controls": control_means,
                "controls_pass": controls_pass,
                "invariance": invariance,
                "invariance_pass": invariance_pass,
                "gradient_update_audits_pass": gradient_pass,
                "frozen_update_audits_pass": frozen_pass,
                "selection_eligible": selection_eligible,
                "failure_reason": "none" if selection_eligible else _failure_reason(beats_stage5, frozen_far_below, controls_pass, invariance_pass, gradient_pass, frozen_pass),
                "parameter_count": int(values[0].get("param_count", 0)),
                "compute_overhead": _mean_compute(values),
                "per_family_accuracy": per_family,
                "hard_family_accuracy": hard_family_accuracy,
            }
        )
    out.sort(key=lambda item: (bool(item["selection_eligible"]), float(item["mean_trainable_accuracy"])), reverse=True)
    return out


def _stage7b_select_against_widened(summaries: List[Dict[str, object]]) -> tuple[List[Dict[str, object]], List[str]]:
    widened = next((row for row in summaries if row.get("variant") == "f_stage5_widened_param_control"), None)
    if widened is None:
        for row in summaries:
            row["delta_vs_widened_param_control"] = None
            row["beats_widened_param_control"] = False
            row["stage7b_selection_eligible"] = False
        return summaries, []
    widened_mean = float(widened.get("mean_trainable_accuracy", 0.0))
    widened_weak = int(widened.get("weak_seed_count", 999))
    widened_hard = float(widened.get("hard_family_accuracy", 0.0))
    widened_compute = float(widened.get("compute_overhead", {}).get("fit_latency_sec", 0.0)) + float(
        widened.get("compute_overhead", {}).get("eval_latency_sec", 0.0)
    )
    true_candidates = []
    for row in summaries:
        variant = str(row.get("variant"))
        delta = float(row.get("mean_trainable_accuracy", 0.0)) - widened_mean
        compute = float(row.get("compute_overhead", {}).get("fit_latency_sec", 0.0)) + float(row.get("compute_overhead", {}).get("eval_latency_sec", 0.0))
        row["delta_vs_widened_param_control"] = delta
        row["beats_widened_param_control"] = delta > 0.0
        row["ties_widened_param_control"] = abs(delta) <= 0.005
        row["better_weak_seed_count_than_widened"] = int(row.get("weak_seed_count", 999)) < widened_weak
        row["better_hard_family_than_widened"] = float(row.get("hard_family_accuracy", 0.0)) > widened_hard
        row["lower_compute_than_widened"] = compute < widened_compute
        true_variant = variant not in {"f_stage5_widened_param_control", "b7_workspace_disabled_param_control"}
        meaningful_tie = bool(
            row["ties_widened_param_control"]
            and (
                row["better_weak_seed_count_than_widened"]
                or row["better_hard_family_than_widened"]
                or row["lower_compute_than_widened"]
            )
        )
        eligible = bool(
            true_variant
            and bool(row.get("controls_pass", False))
            and bool(row.get("invariance_pass", False))
            and bool(row.get("gradient_update_audits_pass", False))
            and bool(row.get("frozen_update_audits_pass", False))
            and float(row.get("mean_delta_vs_frozen", 0.0)) >= 0.20
            and (delta > 0.0 or meaningful_tie)
        )
        row["stage7b_selection_eligible"] = eligible
        if eligible:
            true_candidates.append(row)
    true_candidates.sort(
        key=lambda row: (
            float(row.get("delta_vs_widened_param_control", -999.0)),
            -int(row.get("weak_seed_count", 999)),
            float(row.get("hard_family_accuracy", 0.0)),
            -float(row.get("compute_overhead", {}).get("fit_latency_sec", 0.0)),
        ),
        reverse=True,
    )
    if true_candidates:
        return summaries, [str(true_candidates[0]["variant"])]
    widened["stage7b_selection_eligible"] = True
    widened["failure_reason"] = "honest_capacity_improved_baseline_selected_no_true_transport_improved"
    return summaries, ["f_stage5_widened_param_control"]


def _aggregate_per_family(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, Dict[str, List[float]]] = {}
    counts: Dict[str, int] = {}
    for row in rows:
        for family, values in dict(row.get("per_family_accuracy", {})).items():
            bucket = grouped.setdefault(str(family), {"trainable": [], "frozen": [], "delta": []})
            bucket["trainable"].append(float(values.get("trainable", 0.0)))
            bucket["frozen"].append(float(values.get("frozen", 0.0)))
            bucket["delta"].append(float(values.get("delta", 0.0)))
            counts[str(family)] = counts.get(str(family), 0) + int(values.get("n", 0))
    return {
        family: {
            "n": int(counts.get(family, 0)),
            "trainable": _mean(values["trainable"]),
            "frozen": _mean(values["frozen"]),
            "delta": _mean(values["delta"]),
        }
        for family, values in grouped.items()
    }


def _stage7c_summary(rows: Sequence[Dict[str, object]], result: Dict[str, object], baseline: Dict[str, object]) -> Dict[str, object]:
    test_rows = [row for row in rows if row.get("test_accuracy")]
    if not test_rows:
        return {"n_completed": 0, "passed_stage7_success_gates": False, "success_criteria": [{"criterion": "completed final seeds", "pass": False, "value": 0}]}
    stage5_mean = float(baseline.get("trainable_mean_accuracy", 0.9144))
    stage5_seed_acc = _stage5_seed_accuracy_map(baseline)
    trainable = [float(row["test_accuracy"]["trainable"]) for row in test_rows]
    frozen = [float(row["test_accuracy"]["frozen"]) for row in test_rows]
    deltas_frozen = [a - f for a, f in zip(trainable, frozen)]
    paired = [float(row["test_accuracy"]["trainable"]) - stage5_seed_acc[int(row["seed"])] for row in test_rows if int(row["seed"]) in stage5_seed_acc]
    paired_ci = _bootstrap_ci(paired) if paired else (None, None)
    stage7_beats_stage5_seed_wins = sum(value > 0.0 for value in paired) if paired else None
    config = result.get("config", {})
    near = float(config.get("near_chance_max", 0.18))
    inv_tol = float(config.get("invariance_tolerance", 0.08))
    train_mean = _mean(trainable)
    controls = {name: _mean([float(row["test_accuracy"].get(name, 1.0)) for row in test_rows]) for name in PRIMARY_CONTROLS}
    invariance = {
        name: _mean([float(row["test_accuracy"].get(name, 0.0)) for row in test_rows])
        for name in ("physical_order_shuffled_roles_avenues_preserved", "candidate_order_shuffled_with_label_remap")
    }
    criteria = [
        {"criterion": "completed final seeds >= 10", "pass": len(test_rows) >= 10 and not result.get("failed_or_interrupted_seeds"), "value": len(test_rows)},
        {"criterion": "Stage 7 mean >= locked Stage 5 mean + 0.02", "pass": train_mean >= stage5_mean + 0.02, "value": {"stage7": train_mean, "stage5": stage5_mean}},
        {
            "criterion": "Stage 7 beats Stage 5 on >= 7/10 final seeds",
            "pass": stage7_beats_stage5_seed_wins is not None and stage7_beats_stage5_seed_wins >= 7,
            "value": stage7_beats_stage5_seed_wins if stage7_beats_stage5_seed_wins is not None else "requires locked Stage 5 per-seed results",
        },
        {
            "criterion": "paired bootstrap 95% CI lower bound for Stage7 - Stage5 > 0",
            "pass": paired_ci[0] is not None and float(paired_ci[0]) > 0.0,
            "value": paired_ci if paired else "requires locked Stage 5 per-seed results",
        },
        {"criterion": "Stage 7 beats exact frozen comparator on >= 9/10 seeds", "pass": sum(delta > 0.0 for delta in deltas_frozen) >= 9, "value": sum(delta > 0.0 for delta in deltas_frozen)},
        {"criterion": "all Stage 4/5 corruption controls pass", "pass": all(value <= near for value in controls.values()), "value": controls},
        {"criterion": "physical role/avenue and candidate-order invariance pass", "pass": all(abs(value - train_mean) <= inv_tol for value in invariance.values()), "value": invariance},
        {
            "criterion": "randomized labels and hidden-state shuffle collapse",
            "pass": controls.get("randomized_labels", 1.0) <= near and controls.get("hidden_states_shuffled_across_examples", 1.0) <= near,
            "value": {
                "randomized_labels": controls.get("randomized_labels"),
                "hidden_states_shuffled_across_examples": controls.get("hidden_states_shuffled_across_examples"),
            },
        },
        {
            "criterion": "trainable shared model gradient/update audit passes",
            "pass": all(float(row.get("trainable_audit", {}).get("agent_grad_norm_mean") or 0.0) > 0.0 and float(row.get("trainable_audit", {}).get("agent_parameter_delta") or 0.0) > 0.0 for row in test_rows),
            "value": "all_completed_seeds",
        },
        {
            "criterion": "frozen shared model gradient/update audit passes",
            "pass": all(_audit_float(row.get("frozen_audit", {}), "agent_grad_norm_mean", 1.0) == 0.0 and _audit_float(row.get("frozen_audit", {}), "agent_parameter_delta", 1.0) == 0.0 for row in test_rows),
            "value": "all_completed_seeds",
        },
    ]
    return {
        "n_completed": len(test_rows),
        "completed_seeds": [int(row["seed"]) for row in test_rows],
        "mean_trainable_accuracy": train_mean,
        "mean_frozen_accuracy": _mean(frozen),
        "mean_delta_vs_frozen": _mean(deltas_frozen),
        "std_delta_vs_frozen": pstdev(deltas_frozen) if len(deltas_frozen) > 1 else 0.0,
        "delta_vs_locked_stage5_mean": train_mean - stage5_mean,
        "stage7_beats_stage5_seed_wins": stage7_beats_stage5_seed_wins,
        "stage7_minus_stage5_bootstrap_95_ci": paired_ci,
        "passed_stage7_success_gates": all(bool(item["pass"]) for item in criteria),
        "success_criteria": criteria,
        "compute_overhead": _mean_compute(test_rows),
    }


def _render_report(result: Dict[str, object], phase: str) -> str:
    title = {
        "stage7a": "Stage 7A Architecture Screen",
        "stage7b": "Stage 7B Medium Validation",
        "stage7c": "Stage 7C Final Latent Evidence Transport",
    }[phase]
    summary = result.get("summary", {})
    lines = [
        f"# {title}",
        "",
        f"- Benchmark: `{BALANCED_34B_DATASET_SOURCE}` with `{BALANCED_34B_CANDIDATE_REPRESENTATION}` candidates.",
        "- Dataset, labels, candidates, schema, controls, and split protocol are inherited from Stage 4/5.",
        f"- Locked Stage 5 trainable mean: `{float(result.get('locked_baselines', {}).get('stage5', LOCKED_STAGE5_BASELINE).get('trainable_mean_accuracy', 0.9144)):.4f}`.",
        f"- Test set accessed: `{bool(summary.get('test_set_accessed', phase == 'stage7c'))}`.",
        "",
    ]
    if phase in {"stage7a", "stage7b"}:
        lines.extend([
            "## Variant Summary",
            "",
            "| variant | family | seeds | steps | workspace | params | trainable | frozen | delta frozen | delta Stage5 | delta widened | hard-family | controls | invariance | weak seeds | failure |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|",
        ])
        plan = {row["name"]: row for row in result.get("variant_plan", [])}
        for row in summary.get("variant_summaries", []):
            spec = plan.get(row["variant"], {})
            lines.append(
                "| {variant} | {family} | {seeds} | {steps} | {workspace} | {params} | {train:.4f} | {frozen:.4f} | {delta:.4f} | {stage5:.4f} | {widened} | {hard:.4f} | `{controls}` | `{inv}` | {weak} | {failure} |".format(
                    variant=row["variant"],
                    family=row.get("family", ""),
                    seeds=int(row.get("n_seeds", 0)),
                    steps=int(spec.get("refinement_steps", 0)),
                    workspace=int(spec.get("workspace_slots", 0)),
                    params=int(row.get("parameter_count", 0)),
                    train=float(row.get("mean_trainable_accuracy", 0.0)),
                    frozen=float(row.get("mean_frozen_accuracy", 0.0)),
                    delta=float(row.get("mean_delta_vs_frozen", 0.0)),
                    stage5=float(row.get("delta_vs_locked_stage5", 0.0)),
                    widened=(
                        f"{float(row['delta_vs_widened_param_control']):.4f}"
                        if row.get("delta_vs_widened_param_control") is not None
                        else "n/a"
                    ),
                    hard=float(row.get("hard_family_accuracy", 0.0)),
                    controls=bool(row.get("controls_pass", False)),
                    inv=bool(row.get("invariance_pass", False)),
                    weak=int(row.get("weak_seed_count", 0)),
                    failure=row.get("failure_reason", "none"),
                )
            )
        if phase == "stage7b":
            lines.extend(
                [
                    "",
                    f"- Primary Stage 7B comparator: `{summary.get('primary_stage7b_comparator')}`.",
                    f"- Selected Stage 7B architecture: `{summary.get('selected_top_variants', [])}`.",
                    "- No final claim is made from Stage 7B.",
                ]
            )
        else:
            lines.extend(["", f"- Selected variants: `{summary.get('selected_top_variants', [])}`."])
    else:
        lines.extend([
            "## Final Summary",
            "",
            f"- Completed seeds: `{summary.get('completed_seeds', [])}`",
            f"- Mean trainable accuracy: `{float(summary.get('mean_trainable_accuracy', 0.0)):.4f}`",
            f"- Mean frozen accuracy: `{float(summary.get('mean_frozen_accuracy', 0.0)):.4f}`",
            f"- Delta vs locked Stage 5 mean: `{float(summary.get('delta_vs_locked_stage5_mean', 0.0)):.4f}`",
            f"- Passed Stage 7 gates: `{bool(summary.get('passed_stage7_success_gates', False))}`",
            "",
            "## Success Gates",
            "",
            "| criterion | pass | value |",
            "|---|---|---|",
        ])
        for item in summary.get("success_criteria", []):
            lines.append(f"| {item.get('criterion')} | `{bool(item.get('pass', False))}` | `{item.get('value')}` |")
        lines.extend([
            "",
            "## Per-Seed Results",
            "",
            "| seed | trainable | frozen | delta | randomized | hidden-shuffle | physical-order | candidate-remap | max CUDA GB |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in result.get("stage7c_rows", []):
            if row.get("status") != "completed":
                continue
            test = row.get("test_accuracy", {})
            lines.append(
                "| {seed} | {train:.4f} | {frozen:.4f} | {delta:.4f} | {rand:.4f} | {hidden:.4f} | {phys:.4f} | {cand:.4f} | {mem:.3f} |".format(
                    seed=int(row["seed"]),
                    train=float(test.get("trainable", 0.0)),
                    frozen=float(test.get("frozen", 0.0)),
                    delta=float(row.get("test_delta", 0.0)),
                    rand=float(test.get("randomized_labels", 0.0)),
                    hidden=float(test.get("hidden_states_shuffled_across_examples", 0.0)),
                    phys=float(test.get("physical_order_shuffled_roles_avenues_preserved", 0.0)),
                    cand=float(test.get("candidate_order_shuffled_with_label_remap", 0.0)),
                    mem=float(row.get("cuda_max_memory_allocated_gb", 0.0)),
                )
            )
        lines.extend([
            "",
            "## Conclusion",
            "",
        ])
        if bool(summary.get("passed_stage7_success_gates", False)):
            lines.append(
                "Stage 7 shows that iterative latent evidence transport improves over one-shot candidate-token-direct latent coordination on the clean schema-aware real-code benchmark. Because the comparison uses the locked Stage 5 baseline, exact frozen same-architecture comparators, and the full shortcut/corruption/invariance audit suite, the result supports the claim that learned latent state dynamics can improve shared-weight cloned-agent coordination beyond static latent readout."
            )
        else:
            lines.append(
                "Stage 7 did not beat the locked Stage 5 candidate-token-direct multi-avenue baseline under strict controls. This suggests that the current one-shot candidate-token-direct architecture already captures most of the useful coordination signal on this benchmark, or that latent evidence transport requires a different task/scale to show value."
            )
    failures = result.get("failed_or_interrupted_variants", []) or result.get("failed_or_interrupted_seeds", [])
    if failures:
        lines.extend(["", "## Failures", "", "```json", json.dumps(failures, indent=2, sort_keys=True), "```"])
    return "\n".join(lines) + "\n"


def _audit_rows(result: Dict[str, object], phase: str) -> List[Dict[str, object]]:
    row_key = {"stage7a": "stage7a_rows", "stage7b": "stage7b_rows", "stage7c": "stage7c_rows"}[phase]
    rows: List[Dict[str, object]] = []
    for row in result.get(row_key, []):
        if row.get("status") != "completed":
            continue
        rows.append(
            {
                "stage": phase,
                "event": "stage7_variant_seed_result",
                "variant": row.get("variant"),
                "seed": row.get("seed"),
                "dev_accuracy": row.get("dev_accuracy"),
                "test_accuracy": row.get("test_accuracy"),
                "trainable_audit": row.get("trainable_audit"),
                "frozen_audit": row.get("frozen_audit"),
                "transport_audit": row.get("transport_audit"),
                "seed_criteria": row.get("seed_criteria"),
                "split_leakage_audit_passes": row.get("split_leakage_audit_passes"),
                "output_leakage_audit_passes": row.get("output_leakage_audit_passes"),
            }
        )
    rows.extend(result.get("failed_or_interrupted_variants", []))
    rows.extend(result.get("failed_or_interrupted_seeds", []))
    return rows


def _stage7b_variants_from_stage7a(stage7a_result: object, selected_variant_name: str | None, max_variants: int) -> List[Stage7VariantSpec]:
    if selected_variant_name:
        return [stage7_variant_by_name(selected_variant_name)]
    requested = stage7b_medium_variant_plan()
    if max_variants > 0:
        return requested[:max_variants]
    if isinstance(stage7a_result, dict):
        names = [str(value) for value in stage7a_result.get("summary", {}).get("selected_top_variants", [])][:3]
    else:
        previous = _load_result(STAGE7A_OUTPUT)
        names = [str(value) for value in previous.get("summary", {}).get("selected_top_variants", [])][:3]
    required = {"f_stage5_widened_param_control", "e_compat_ce_t2_w0p10", "f_stage5_workspace_only_m8"}
    if required.issubset(set(names)):
        return requested
    return [stage7_variant_by_name(name) for name in names]


def _selected_stage7c_variant(stage7b_result: object, selected_variant_name: str | None) -> Stage7VariantSpec | None:
    if selected_variant_name:
        return stage7_variant_by_name(selected_variant_name)
    if isinstance(stage7b_result, dict):
        names = [str(value) for value in stage7b_result.get("summary", {}).get("selected_top_variants", [])]
    else:
        previous = _load_result(STAGE7B_OUTPUT)
        names = [str(value) for value in previous.get("summary", {}).get("selected_top_variants", [])]
    return stage7_variant_by_name(names[0]) if names else None


def _run_stage7c_ablations(
    config: Dict[str, object],
    stage: StageConfig,
    selected: Stage7VariantSpec,
    device: str,
    checkpoint_dir: Path,
) -> List[Dict[str, object]]:
    ablation_seeds = [int(value) for value in config.get("stage7c", {}).get("ablation_seeds", [40, 41, 42])]
    ablation_stage = replace(stage, n_test=0, seeds=tuple(ablation_seeds))
    rows, failures = _run_variant_phase(
        phase="stage7c_ablation_dev",
        config=config,
        stage=ablation_stage,
        variants=stage7_ablation_plan(selected),
        device=device,
        checkpoint_dir=checkpoint_dir / "stage7c_ablations",
        save_checkpoints=False,
        include_baselines=False,
        include_randomized=False,
    )
    return rows + failures


def _phase_stage(config: Dict[str, object], phase: str) -> StageConfig:
    base = dict(config.get("stage", {}))
    if phase == "stage7a":
        base.update({"name": "stage7a_architecture_screen", "n_train": 512, "n_dev": 256, "n_test": 0, "seeds": [0, 1, 2]})
    elif phase == "stage7b":
        base.update({"name": "stage7b_medium_validation", "n_train": 2048, "n_dev": 512, "n_test": 0, "seeds": [0, 1, 2, 3, 4]})
    elif phase == "stage7c":
        base.update({"name": "stage7c_final_latent_evidence_transport", "n_train": 2048, "n_dev": 512, "n_test": 1024, "seeds": [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]})
    else:
        raise ValueError(f"unknown Stage 7 phase: {phase}")
    base.setdefault("epochs", 50)
    base.setdefault("patience", 6)
    base.setdefault("lr", 0.0003)
    base.setdefault("mixed_precision", "bf16")
    return _stage_from_config({**config, "stage": base})


def _phase_config_dict(config: Dict[str, object], stage: StageConfig) -> Dict[str, object]:
    out = _stage7_base_config(config)
    out["stage"] = asdict(stage)
    return out


def _stage7_base_config(config: Dict[str, object]) -> Dict[str, object]:
    out = _stage38_full_config(config)
    out["architecture"] = "stage7_latent_evidence_transport"
    out["architecture_changes"] = "dev_search_then_locked_final"
    inherited_dataset = {**dict(out.get("dataset_config", {})), **dict(config.get("dataset_config", {}))}
    out["dataset_config"] = {
        **inherited_dataset,
        "dataset_source": BALANCED_34B_DATASET_SOURCE,
        "candidate_representation": BALANCED_34B_CANDIDATE_REPRESENTATION,
    }
    return out


def _metadata(stage_name: str, config_path: Path, hardware: Dict[str, object], device: str) -> Dict[str, object]:
    return {
        "benchmark": BENCHMARK,
        "task": stage_name,
        "architecture_family": "latent_evidence_transport",
        "base_architecture": "stage5_4role_x4avenue_candidate_token_direct",
        "dataset": BALANCED_34B_DATASET_SOURCE,
        "candidate_representation": BALANCED_34B_CANDIDATE_REPRESENTATION,
        "schema_policy": "role-specific schema fields preserved as legitimate program-analysis evidence",
        "config_path": str(config_path),
        "device": device,
        "created_or_updated_at_utc": _now(),
        "hardware": hardware,
        "test_set_tuning_forbidden": True,
    }


def _locked_baselines(stage5_baseline: Dict[str, object]) -> Dict[str, object]:
    return {"stage4": LOCKED_STAGE4_BASELINE, "stage5": stage5_baseline}


def _load_stage5_baseline(path: Path | None) -> Dict[str, object]:
    baseline = dict(LOCKED_STAGE5_BASELINE)
    if path is None or not path.exists():
        baseline["per_seed_accuracy_available"] = False
        return baseline
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("stage5_final_rows") or data.get("stage7c_rows") or data.get("rows") or []
    per_seed = {}
    per_seed_dev = {}
    for row in rows:
        if row.get("status") != "completed":
            continue
        test = row.get("test_accuracy", {})
        if "trainable" in test:
            per_seed[int(row["seed"])] = float(test["trainable"])
        dev = row.get("dev_accuracy", {})
        if "trainable" in dev:
            per_seed_dev[int(row["seed"])] = float(dev["trainable"])
    if per_seed:
        baseline["per_seed_test_accuracy"] = per_seed
        baseline["per_seed_accuracy_available"] = True
        baseline["trainable_mean_accuracy"] = _mean(per_seed.values())
        baseline["source"] = str(path)
    else:
        baseline["per_seed_accuracy_available"] = False
        baseline["source"] = str(path)
    if per_seed_dev:
        baseline["dev_mean_accuracy"] = _mean(per_seed_dev.values())
    return baseline


def _stage5_seed_accuracy_map(baseline: Dict[str, object]) -> Dict[int, float]:
    raw = baseline.get("per_seed_test_accuracy", {})
    if not isinstance(raw, dict):
        return {}
    return {int(seed): float(value) for seed, value in raw.items()}


def _attach_pre_run_candidate_metadata(result: Dict[str, object]) -> None:
    seed_values = {}
    for row in result.get("pre_run", {}).get("shortcut_diagnostics", {}).get("seed_rows", []):
        baselines = row.get("baselines", {})
        seed_values[int(row.get("seed", -1))] = float(baselines.get("candidate_metadata_only_accuracy", 1.0))
    for row in result.get("stage7c_rows", []):
        value = seed_values.get(int(row.get("seed", -1)))
        if value is not None:
            row.setdefault("test_accuracy", {})["candidate_metadata_only"] = value


def _per_family_compare(trainable, frozen, examples: Sequence[MultiViewTaskExample], seed: int) -> Dict[str, Dict[str, float]]:
    train_pred = predict_latent_system(trainable, examples, "none", seed + 60_000)
    frozen_pred = predict_latent_system(frozen, examples, "none", seed + 61_000)
    labels = _labels(examples)
    groups: Dict[str, List[int]] = {}
    for index, example in enumerate(examples):
        family = str((example.oracle_metadata or example.metadata).get("problem_family", "unknown"))
        groups.setdefault(family, []).append(index)
    out = {}
    for family, indices in groups.items():
        idx = np.asarray(indices, dtype=np.int64)
        train_acc = float(np.mean(train_pred[idx] == labels[idx]))
        frozen_acc = float(np.mean(frozen_pred[idx] == labels[idx]))
        out[family] = {"n": int(len(indices)), "trainable": train_acc, "frozen": frozen_acc, "delta": train_acc - frozen_acc}
    return out


def _seed_criteria(row: Dict[str, object], config: Dict[str, object], include_test: bool) -> Dict[str, object]:
    split = "test_accuracy" if include_test else "dev_accuracy"
    acc = row[split]
    near = float(config.get("near_chance_max", 0.18))
    inv_tol = float(config.get("invariance_tolerance", 0.08))
    return {
        "trainable_beats_frozen": float(acc.get("trainable", 0.0)) > float(acc.get("frozen", 0.0)),
        "controls_near_chance": {name: float(acc.get(name, 1.0)) <= near for name in LATENT_CONTROLS + ("randomized_labels", "hidden_states_shuffled_across_examples") if name in acc},
        "invariance_controls_preserve_accuracy": {
            name: abs(float(acc.get(name, 0.0)) - float(acc.get("trainable", 0.0))) <= inv_tol
            for name in ("physical_order_shuffled_roles_avenues_preserved", "candidate_order_shuffled_with_label_remap")
        },
        "trainable_agent_changed": float(row.get("trainable_audit", {}).get("agent_parameter_delta") or 0.0) > 0.0,
        "frozen_agent_unchanged": _audit_float(row.get("frozen_audit", {}), "agent_parameter_delta", 1.0) == 0.0,
    }


def _row_has_stage7_checkpoints(row: Dict[str, object], config: Dict[str, object]) -> bool:
    if not bool(config.get("save_checkpoints", True)):
        return True
    paths = row.get("checkpoint_paths", {})
    return isinstance(paths, dict) and paths.get("trainable") and paths.get("frozen") and Path(str(paths["trainable"])).exists() and Path(str(paths["frozen"])).exists()


def _split_leakage_passes(leakage: Dict[str, object], include_test: bool) -> bool:
    if include_test:
        return bool(
            int(leakage.get("train_test_id_overlap", 1)) == 0
            and int(leakage.get("train_test_candidate_hash_overlap", 1)) == 0
            and int(leakage.get("train_test_file_patch_overlap", 1)) == 0
            and int(leakage.get("train_test_problem_family_overlap", 1)) == 0
            and int(leakage.get("train_test_source_import_key_overlap", 1)) == 0
            and int(leakage.get("train_test_target_file_overlap", 1)) == 0
            and bool(leakage.get("candidate_patch_hash_leakage_audit_passes", False))
        )
    return bool(
        int(leakage.get("train_dev_id_overlap", 1)) == 0
        and int(leakage.get("train_dev_candidate_hash_overlap", 1)) == 0
        and int(leakage.get("train_dev_source_import_key_overlap", 1)) == 0
        and int(leakage.get("train_dev_target_file_overlap", 1)) == 0
    )


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return float(sum(values) / len(values)) if values else 0.0


def _mean_compute(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    return {
        "fit_latency_sec": _mean(row.get("fit_latency_sec", 0.0) for row in rows),
        "eval_latency_sec": _mean(row.get("eval_latency_sec", 0.0) for row in rows),
        "cuda_max_memory_allocated_gb": _mean(row.get("cuda_max_memory_allocated_gb", 0.0) for row in rows),
        "param_count": _mean(row.get("param_count", 0.0) for row in rows),
        "refinement_steps": _mean(row.get("compute_overhead", {}).get("refinement_steps", 0.0) for row in rows),
        "workspace_slots": _mean(row.get("compute_overhead", {}).get("workspace_slots", 0.0) for row in rows),
    }


def _failure_reason(
    beats_stage5: bool,
    frozen_far_below: bool,
    controls_pass: bool,
    invariance_pass: bool,
    gradient_pass: bool,
    frozen_pass: bool,
) -> str:
    if not beats_stage5:
        return "did_not_beat_locked_stage5_reference"
    if not frozen_far_below:
        return "frozen_comparator_not_far_below_trainable"
    if not controls_pass:
        return "control_failure"
    if not invariance_pass:
        return "invariance_failure"
    if not gradient_pass:
        return "trainable_gradient_update_audit_failure"
    if not frozen_pass:
        return "frozen_shared_weight_update_audit_failure"
    return "none"


def _variant_failure_row(phase: str, variant: str, seed: int, status: str, exc: BaseException) -> Dict[str, object]:
    return {
        "stage": phase,
        "variant": variant,
        "seed": int(seed),
        "status": status,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "time_utc": _now(),
    }


def _exception_row(seed: int, variant: str, batch_size: int, accumulation: int, exc: BaseException) -> Dict[str, object]:
    return {
        "seed": int(seed),
        "variant": variant,
        "batch_size": int(batch_size),
        "gradient_accumulation_steps": int(accumulation),
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "time_utc": _now(),
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
