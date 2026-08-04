from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from types import SimpleNamespace
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
    example_oracle_metadata,
    format_clone_prompt,
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
    _candidate_specs,
    _coordinator_config,
    _fit_baselines,
    _leakage_passes,
    _per_role_ablation,
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
from src.experiments.run_stage4_final_candidate_token_direct_validation import (
    INVARIANCE_CONTROLS,
    PRIMARY_CONTROLS,
)


DEFAULT_CONFIG = "configs/stage5_stacked_coordination.json"
SELECTED_BASELINE = "candidate_token_direct_lr3e4_clip1"
RESULTS_PATH = "results/stage5_stacked_coordination_results.json"
AUDIT_PATH = "results/stage5_stacked_coordination_audit.jsonl"
ERROR_CASES_PATH = "results/stage5_stacked_coordination_error_cases.jsonl"
ATTENTION_SUMMARIES_PATH = "results/stage5_stacked_coordination_attention_summaries.jsonl"
COMPUTE_METRICS_PATH = "results/stage5_stacked_coordination_compute_metrics.json"
REPORT_PATH = "reports/STAGE5_STACKED_COORDINATION_FULL_METRICS.md"
NEAR_CHANCE_DEFAULT = 0.18


@dataclass(frozen=True)
class Stage5Variant:
    name: str
    block_count: int
    description: str
    shared_block_weights: bool = False
    use_residual_update: bool = True
    use_layer_norm: bool = True
    use_feedforward: bool = True
    candidate_dropout: float = 0.0
    block_dropout: float = 0.0
    lr: float = 0.0003
    epochs: int = 50
    patience: int = 6
    weight_decay: float = 0.0001
    gradient_clip_norm: float = 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 5 stacked candidate-token-direct coordination.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("cheap", "medium", "final", "all"), default="all")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-variants", type=int, default=0, help="Optional prefix limit for smoke probes. 0 means all.")
    parser.add_argument("--max-seeds", type=int, default=0, help="Optional prefix seed limit for smoke probes. 0 means all protocol seeds.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.device is not None:
        config["device"] = str(args.device)
        if str(args.device) == "cpu":
            config["require_cuda"] = False
    run_stage5(
        config=config,
        config_path=config_path,
        requested_phase=str(args.phase),
        max_variants=int(args.max_variants),
        max_seeds=int(args.max_seeds),
    )


def run_stage5(
    config: Dict[str, object],
    config_path: Path,
    requested_phase: str = "all",
    max_variants: int = 0,
    max_seeds: int = 0,
) -> Dict[str, object]:
    config = _stage5_config(config)
    device, hardware = _configure_cuda(config)
    output_path = Path(str(config["output_path"]))
    audit_path = Path(str(config["audit_log_path"]))
    report_path = Path(str(config["report_path"]))
    error_path = Path(str(config["error_cases_path"]))
    attention_path = Path(str(config["attention_summaries_path"]))
    compute_path = Path(str(config["compute_metrics_path"]))

    result = _load_result(output_path)
    result.setdefault("metadata", _metadata(config, config_path, hardware, device))
    result["metadata"].update(_metadata(config, config_path, hardware, device))
    result["config"] = config
    result.setdefault("pre_run", {})
    result.setdefault("phases", {})
    result.setdefault("failed_or_interrupted_seeds", [])
    result.setdefault("oom_retries", [])

    if bool(config.get("skip_pre_run_checks", False)) and not result.get("pre_run"):
        result["pre_run"] = {
            "summary": {
                "passes": True,
                "skipped": True,
                "reason": "explicit skip_pre_run_checks for local smoke only",
            }
        }
    if not result.get("pre_run"):
        print("stage5: running Stage 3.8 pre-run shortcut diagnostics")
        result["pre_run"] = _pre_run_checks(config, config_path, device)
        _write_outputs(result, output_path, audit_path, report_path, error_path, attention_path, compute_path)
        if bool(config.get("stop_on_preflight_failure", True)) and not bool(result["pre_run"].get("summary", {}).get("passes", False)):
            print("stage5: pre-run checks failed; experiment was not started")
            return result

    variants = _variant_plan(config)
    if max_variants > 0:
        variants = variants[:max_variants]

    phases_to_run = _requested_phases(requested_phase)
    if "cheap" in phases_to_run:
        cheap_stage = _phase_stage(config, "cheap", max_seeds=max_seeds)
        result["phases"]["cheap"] = _run_phase(
            phase="cheap",
            config=config,
            stage=cheap_stage,
            variants=variants,
            device=device,
            hardware=hardware,
            result=result,
        )
        _write_outputs(result, output_path, audit_path, report_path, error_path, attention_path, compute_path)

    if "medium" in phases_to_run:
        top = _select_top_variants(result, "cheap", limit=2)
        medium_variants = [variant for variant in variants if variant.name in top]
        if not medium_variants:
            result["phases"]["medium"] = {"rows": [], "summary": {"ran": False, "reason": "no cheap-screen variant passed"}}
        else:
            medium_stage = _phase_stage(config, "medium", max_seeds=max_seeds)
            result["phases"]["medium"] = _run_phase(
                phase="medium",
                config=config,
                stage=medium_stage,
                variants=medium_variants,
                device=device,
                hardware=hardware,
                result=result,
            )
        _write_outputs(result, output_path, audit_path, report_path, error_path, attention_path, compute_path)

    if "final" in phases_to_run:
        selected = _select_top_variants(result, "medium", limit=1)
        final_variants = [variant for variant in variants if variant.name in selected]
        if not final_variants:
            result["phases"]["final"] = {"rows": [], "summary": {"ran": False, "reason": "no medium variant passed"}}
        else:
            final_stage = _phase_stage(config, "final", max_seeds=max_seeds)
            result["metadata"]["final_variant_locked_before_final_seeds"] = final_variants[0].name
            result["phases"]["final"] = _run_phase(
                phase="final",
                config=config,
                stage=final_stage,
                variants=final_variants,
                device=device,
                hardware=hardware,
                result=result,
            )
        _write_outputs(result, output_path, audit_path, report_path, error_path, attention_path, compute_path)

    _write_outputs(result, output_path, audit_path, report_path, error_path, attention_path, compute_path)
    return result


def _run_phase(
    phase: str,
    config: Dict[str, object],
    stage: StageConfig,
    variants: Sequence[Stage5Variant],
    device: str,
    hardware: Dict[str, object],
    result: Dict[str, object],
) -> Dict[str, object]:
    existing = result.get("phases", {}).get(phase, {})
    rows = [row for row in existing.get("rows", []) if row.get("status") == "completed"]
    completed = {(str(row.get("variant")), int(row.get("seed"))) for row in rows}
    fallbacks = [int(value) for value in config.get("gpu_safety", {}).get("batch_size_fallbacks", [32, 16, 8])]
    effective_batch = int(config.get("gpu_safety", {}).get("effective_batch_size", fallbacks[0]))
    for variant in variants:
        for seed_value in stage.seeds:
            seed = int(seed_value)
            if (variant.name, seed) in completed:
                print(f"stage5 {phase}: existing row variant={variant.name} seed={seed}; skipping")
                continue
            rows = [row for row in rows if not (row.get("variant") == variant.name and int(row.get("seed", -1)) == seed)]
            print(f"stage5 {phase}: variant={variant.name} blocks={variant.block_count} seed={seed}")
            seed_completed = False
            for batch_size in fallbacks:
                accumulation = max(1, int(math.ceil(effective_batch / float(batch_size))))
                run_stage = replace(stage, batch_size=batch_size, gradient_accumulation_steps=accumulation)
                try:
                    row = _run_variant_seed(phase, run_stage, variant, seed, device, hardware, config)
                    rows.append(row)
                    seed_completed = True
                    completed.add((variant.name, seed))
                    break
                except RuntimeError as exc:
                    if _is_oom(exc) and batch_size != fallbacks[-1]:
                        retry = _failure_row(seed, batch_size, accumulation, "oom_retry", exc)
                        retry["phase"] = phase
                        retry["variant"] = variant.name
                        result["oom_retries"].append(retry)
                        _clear_cuda()
                        print(f"stage5 {phase}: OOM variant={variant.name} seed={seed} batch={batch_size}; retrying")
                        continue
                    failure = _failure_row(seed, batch_size, accumulation, "failed", exc)
                    failure["phase"] = phase
                    failure["variant"] = variant.name
                    result["failed_or_interrupted_seeds"].append(failure)
                    _clear_cuda()
                    break
                except Exception as exc:
                    result["failed_or_interrupted_seeds"].append(
                        {
                            "phase": phase,
                            "variant": variant.name,
                            "seed": seed,
                            "batch_size": batch_size,
                            "gradient_accumulation_steps": accumulation,
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                            "time_utc": _now(),
                        }
                    )
                    _clear_cuda()
                    break
                finally:
                    _clear_cuda()
            if not seed_completed and bool(config.get("stop_on_failed_seed", False)):
                break
    return {"rows": rows, "summary": _phase_summary(rows, config)}


def _run_variant_seed(
    phase: str,
    stage: StageConfig,
    variant: Stage5Variant,
    seed: int,
    device: str,
    hardware: Dict[str, object],
    config: Dict[str, object],
) -> Dict[str, object]:
    _clear_cuda()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    dataset_config = _dataset_config_for_stage(config, stage)
    dataset_start = time.perf_counter()
    splits = build_multiview_code_patch_splits(dataset_config, seed=seed, repo_root=Path("."))
    dataset_seconds = time.perf_counter() - dataset_start
    split_leakage = split_leakage_audit(BENCHMARK, seed, splits)
    output_leakage = output_leakage_audit(BENCHMARK, seed, splits["train"] + splits["dev"] + splits["test"])

    baseline_start = time.perf_counter()
    baselines = _fit_baselines(stage, splits, seed=seed, device=device)
    baseline_seconds = time.perf_counter() - baseline_start

    candidate = _candidate_for_variant(stage, variant)
    fit_start = time.perf_counter()
    fit = _fit_variant_methods(candidate, stage, variant, splits, seed, device, config, phase)
    training_seconds = time.perf_counter() - fit_start

    metrics = _metrics(fit, baselines, splits, seed, variant, config)
    row = {
        "stage": "stage5_stacked_coordination",
        "phase": phase,
        "variant": variant.name,
        "block_count": int(variant.block_count),
        "description": variant.description,
        "architecture_config": _candidate_config(candidate),
        "variant_config": asdict(variant),
        "seed": int(seed),
        "status": "completed",
        "completed_at_utc": _now(),
        "device": device,
        "cuda_device_name": hardware.get("cuda_device_name"),
        "cuda_total_memory": hardware.get("cuda_total_memory"),
        "cuda_max_memory_allocated": int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") and torch.cuda.is_available() else 0,
        "cuda_max_memory_allocated_gb": _cuda_gb(int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") and torch.cuda.is_available() else 0),
        "stage_config": asdict(stage),
        "dataset_config": asdict(dataset_config),
        "dataset_summary": dataset_summary(splits),
        "split_leakage_audit": _compact_leakage(split_leakage),
        "split_leakage_audit_passes": _leakage_passes(split_leakage),
        "output_leakage_audit": output_leakage,
        "output_leakage_audit_passes": bool(output_leakage.get("passes", False)),
        "dev_accuracy": metrics["dev"]["accuracy"],
        "test_accuracy": metrics["test"]["accuracy"],
        "dev_delta": metrics["dev"]["accuracy"]["trainable"] - metrics["dev"]["accuracy"]["frozen"],
        "test_delta": metrics["test"]["accuracy"]["trainable"] - metrics["test"]["accuracy"]["frozen"],
        "calibration": {"dev": metrics["dev"]["calibration"], "test": metrics["test"]["calibration"]},
        "depth_metrics": {"dev": metrics["dev"]["depth"], "test": metrics["test"]["depth"]},
        "stability_metrics": _stability_metrics(fit, metrics),
        "ablation_metrics": metrics["test"]["ablations"],
        "attention_summary": metrics["test"]["attention_summary"],
        "representation_summary": metrics["test"]["representation_summary"],
        "per_family_accuracy": _per_family_compare(fit["trainable"], fit["frozen"], splits["test"], seed),
        "per_role_ablation": _per_role_ablation(fit["trainable"], splits["test"], seed),
        "trainable_audit": _audit_subset(fit["trainable"].audit),
        "frozen_audit": _audit_subset(fit["frozen"].audit),
        "randomized_audit": _audit_subset(fit["randomized"].audit),
        "timing": {
            "dataset_build_seconds": dataset_seconds,
            "baseline_fit_seconds": baseline_seconds,
            "variant_fit_seconds": training_seconds,
            "trainable_fit_seconds": fit["timing"]["trainable"],
            "frozen_fit_seconds": fit["timing"]["frozen"],
            "randomized_fit_seconds": fit["timing"]["randomized"],
            "positive_control_fit_seconds": fit["timing"]["candidate_pair_compatibility_mlp"],
        },
        "compute_metrics": _row_compute_metrics(stage, variant, fit, metrics, training_seconds),
        "diagnostic_predictions": metrics["test"]["diagnostic_predictions"],
        "test_case_records": _case_records(splits["test"]),
    }
    row["success_criteria"] = _seed_criteria(row, config)
    row["failure_analysis"] = _seed_failure_analysis(row, config)
    return row


def _fit_variant_methods(
    candidate,
    stage: StageConfig,
    variant: Stage5Variant,
    splits: Dict[str, Sequence[MultiViewTaskExample]],
    seed: int,
    device: str,
    config: Dict[str, object],
    phase: str,
) -> Dict[str, object]:
    del phase
    agent_config = _agent_config(stage)
    training = replace(
        _training_config(stage),
        epochs=int(variant.epochs),
        patience=int(variant.patience),
        lr=float(variant.lr),
        weight_decay=float(variant.weight_decay),
        gradient_clip_norm=float(variant.gradient_clip_norm),
    )
    timing = {}
    start = time.perf_counter()
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
        method=f"stage5_trainable__{variant.name}",
        message_config=candidate.message_config,
    )
    timing["trainable"] = time.perf_counter() - start
    start = time.perf_counter()
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
        method=f"stage5_frozen__{variant.name}",
        message_config=candidate.message_config,
    )
    timing["frozen"] = time.perf_counter() - start
    start = time.perf_counter()
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
        method=f"stage5_randomized_labels__{variant.name}",
        condition="randomized_labels",
        train_labels=randomized_labels_for_examples(splits["train"], seed + 40_401, 8),
        dev_labels=randomized_labels_for_examples(splits["dev"], seed + 50_501, 8),
        message_config=candidate.message_config,
    )
    timing["randomized"] = time.perf_counter() - start
    start = time.perf_counter()
    positive = CompatibilityFeatureScorer(training=_positive_training(config), seed=seed + 14_000)
    positive.fit(splits["train"], splits["dev"])
    timing["candidate_pair_compatibility_mlp"] = time.perf_counter() - start
    return {
        "trainable": trainable,
        "frozen": frozen,
        "randomized": randomized,
        "candidate_pair_compatibility_mlp": positive,
        "timing": timing,
    }


def _metrics(
    fit: Dict[str, object],
    baselines: Dict[str, object],
    splits: Dict[str, Sequence[MultiViewTaskExample]],
    seed: int,
    variant: Stage5Variant,
    config: Dict[str, object],
) -> Dict[str, Dict[str, object]]:
    out = {}
    for split_name in ("dev", "test"):
        examples = list(splits[split_name])
        split_seed = seed + (1_000 if split_name == "dev" else 2_000)
        y = _labels(examples)
        train = _prediction_details(fit["trainable"], examples, "none", split_seed, capture=True)
        frozen = _prediction_details(fit["frozen"], examples, "none", split_seed, capture=False)
        randomized = _prediction_details(fit["randomized"], examples, "none", split_seed, capture=False)
        values = {
            "trainable": _accuracy(train["predictions"], y),
            "frozen": _accuracy(frozen["predictions"], y),
            "text_only": _accuracy(baselines["text"].predict(examples), y),
            "raw_latent": _accuracy(predict_latent_system(baselines["raw"], examples, "none", split_seed), y),
            "majority_baseline": float(np.mean(np.full(len(y), int(baselines["majority_prediction"]), dtype=np.int64) == y)),
            "candidate_order_baseline": float(np.mean(np.full(len(y), int(baselines["candidate_order_prediction"]), dtype=np.int64) == y)),
            "single_agent_full_context": _accuracy(predict_context_baseline(baselines["full_context"], examples), y),
            "candidate_pair_compatibility_mlp": _accuracy(fit["candidate_pair_compatibility_mlp"].predict(examples), y),
            "all_role_oracle": _accuracy(_structured_oracle_predictions(examples, [0, 1, 2, 3]), y),
            "explicit_evidence_oracle": _accuracy(_structured_oracle_predictions(examples, [0, 1, 2, 3]), y),
            "randomized_labels": _accuracy(randomized["predictions"], y),
            "hidden_states_shuffled_across_examples": _accuracy(
                predict_latent_system(fit["trainable"], examples, "hidden_states_shuffled_across_examples", split_seed), y
            ),
            "role_token_states_shuffled_across_examples": _accuracy(
                predict_latent_system(fit["trainable"], examples, "role_token_states_shuffled_across_examples", split_seed), y
            ),
            "physical_order_shuffled_roles_preserved": _accuracy(
                predict_latent_system(fit["trainable"], examples, "physical_order_shuffled_roles_preserved", split_seed), y
            ),
            "role_embedding_shuffle_diagnostic": _accuracy(
                predict_latent_system(fit["trainable"], examples, "role_labels_shuffled", split_seed), y
            ),
        }
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
        values["candidate_order_shuffled_with_gold_remap"] = _accuracy(
            predict_latent_system(fit["trainable"], shuffled, "none", split_seed), _labels(shuffled)
        )
        values["candidate_order_shuffled"] = values["candidate_order_shuffled_with_gold_remap"]
        values["view_masked"] = values["view_masked_candidates_visible"]
        values["role_labels_shuffled"] = values["role_embedding_shuffle_diagnostic"]
        out[split_name] = {
            "accuracy": values,
            "calibration": _calibration_metrics(train["logits"], y),
            "depth": _depth_metrics(train["early_logits"], y),
            "ablations": _ablation_metrics(fit["trainable"], examples, y, split_seed, variant),
            "attention_summary": train["attention_summary"],
            "representation_summary": train["representation_summary"],
            "timing": train["timing"],
            "diagnostic_predictions": {
                "gold": y.astype(int).tolist(),
                "trainable_pred": train["predictions"].astype(int).tolist(),
                "frozen_pred": frozen["predictions"].astype(int).tolist(),
                "trainable_probabilities": np.round(_softmax_np(train["logits"]), 6).tolist(),
                "early_predictions": train["early_predictions"].astype(int).tolist() if train["early_predictions"].size else [],
            },
        }
    return out


def _prediction_details(result, examples: Sequence[MultiViewTaskExample], condition: str, seed: int, capture: bool) -> Dict[str, object]:
    result.system.eval()
    logits_parts = []
    early_parts = []
    attention_accumulator = _AttentionAccumulator()
    rep_changes: List[float] = []
    rep_cosines: List[float] = []
    batch_latencies = []
    memory_samples = []
    coordinator = result.coordinator
    old_capture = getattr(coordinator, "capture_diagnostics", False)
    if hasattr(coordinator, "capture_diagnostics"):
        coordinator.capture_diagnostics = bool(capture)
    try:
        with torch.no_grad():
            for batch in _example_batches(examples, 128):
                start = time.perf_counter()
                logits, _audit, _acts = result.system(batch, condition=condition, seed=seed)
                if logits.device.type == "cuda":
                    torch.cuda.synchronize(logits.device)
                elapsed = time.perf_counter() - start
                batch_latencies.append(elapsed)
                logits_parts.append(logits.detach().float().cpu())
                if logits.device.type == "cuda":
                    memory_samples.append(float(torch.cuda.memory_allocated(logits.device)))
                diag = getattr(coordinator, "last_diagnostics", {}) if capture else {}
                early = diag.get("early_logits") if isinstance(diag, dict) else None
                if isinstance(early, torch.Tensor) and early.numel() > 0:
                    early_parts.append(early)
                attention = diag.get("attention_weights") if isinstance(diag, dict) else None
                if isinstance(attention, torch.Tensor) and attention.numel() > 0:
                    attention_accumulator.update(attention, diag)
                rep_changes.extend(float(value) for value in diag.get("representation_change_l2", []) if isinstance(diag, dict))
                rep_cosines.extend(float(value) for value in diag.get("representation_cosine", []) if isinstance(diag, dict))
    finally:
        if hasattr(coordinator, "capture_diagnostics"):
            coordinator.capture_diagnostics = old_capture
    logits_np = torch.cat(logits_parts, dim=0).numpy() if logits_parts else np.zeros((0, 0), dtype=np.float32)
    early_np = _concat_early_logits(early_parts)
    predictions = logits_np.argmax(axis=1).astype(np.int64) if logits_np.size else np.zeros(0, dtype=np.int64)
    early_predictions = early_np.argmax(axis=2).T.astype(np.int64) if early_np.size else np.zeros((0, 0), dtype=np.int64)
    n_examples = max(1, len(examples))
    return {
        "logits": logits_np,
        "predictions": predictions,
        "early_logits": early_np,
        "early_predictions": early_predictions,
        "attention_summary": attention_accumulator.summary(),
        "representation_summary": {
            "representation_change_l2_mean": _mean(rep_changes),
            "representation_change_l2_std": _std(rep_changes),
            "representation_cosine_mean": _mean(rep_cosines),
            "representation_cosine_std": _std(rep_cosines),
        },
        "timing": {
            "wall_clock_inference_seconds": float(sum(batch_latencies)),
            "inference_time_per_example_ms": float(sum(batch_latencies) * 1000.0 / n_examples),
            "latency_per_batch_ms_mean": float(_mean([value * 1000.0 for value in batch_latencies])),
            "latency_per_batch_ms_max": float(max([value * 1000.0 for value in batch_latencies], default=0.0)),
            "average_cuda_memory_gb": _cuda_gb(_mean(memory_samples)),
        },
    }


class _AttentionAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.role_mass_sum: torch.Tensor | None = None
        self.candidate_role_mass_sum: torch.Tensor | None = None
        self.entropy_sum: torch.Tensor | None = None
        self.flat_mass_sum: torch.Tensor | None = None
        self.roles = 0
        self.tokens = 0
        self.block_pair_l1: List[float] = []

    def update(self, attention: torch.Tensor, diag: Dict[str, object]) -> None:
        # attention: blocks, batch, candidates, roles*tokens
        blocks, batch, candidates, flat_tokens = attention.shape
        roles = int(diag.get("roles", 1))
        tokens = int(diag.get("tokens_per_role", max(1, flat_tokens // max(1, roles))))
        self.roles = roles
        self.tokens = tokens
        reshaped = attention.reshape(blocks, batch, candidates, roles, tokens)
        role_mass = reshaped.sum(dim=-1)
        role_mass_sum = role_mass.sum(dim=(1, 2))
        candidate_role_mass_sum = role_mass.sum(dim=1)
        entropy = -(attention.clamp_min(1e-9) * attention.clamp_min(1e-9).log()).sum(dim=-1).sum(dim=(1, 2))
        flat_mass_sum = attention.sum(dim=(1, 2))
        if self.role_mass_sum is None:
            self.role_mass_sum = role_mass_sum
            self.candidate_role_mass_sum = candidate_role_mass_sum
            self.entropy_sum = entropy
            self.flat_mass_sum = flat_mass_sum
        else:
            self.role_mass_sum += role_mass_sum
            self.candidate_role_mass_sum += candidate_role_mass_sum
            self.entropy_sum += entropy
            self.flat_mass_sum += flat_mass_sum
        self.count += int(batch * candidates)
        if blocks > 1:
            flat = attention.reshape(blocks, batch * candidates, flat_tokens)
            for block_index in range(blocks - 1):
                self.block_pair_l1.append(float((flat[block_index + 1] - flat[block_index]).abs().mean().item()))

    def summary(self) -> Dict[str, object]:
        if self.count <= 0 or self.role_mass_sum is None or self.entropy_sum is None:
            return {
                "attention_available": False,
                "attention_mass_per_role": [],
                "attention_mass_per_candidate_role": [],
                "candidate_to_token_attention_entropy": [],
                "attention_diversity_across_blocks_l1": 0.0,
            }
        role_mass = (self.role_mass_sum / float(self.count)).numpy()
        candidate_role_mass = (self.candidate_role_mass_sum / max(1.0, float(self.count) / max(1, self.candidate_role_mass_sum.shape[1]))).numpy()
        entropy = (self.entropy_sum / float(self.count)).numpy()
        top_tokens = []
        if self.flat_mass_sum is not None:
            flat_mean = (self.flat_mass_sum / float(self.count)).numpy()
            for block_index, row in enumerate(flat_mean):
                top = np.argsort(-row)[:8]
                top_tokens.append(
                    {
                        "block": int(block_index + 1),
                        "tokens": [
                            {
                                "role": int(index // max(1, self.tokens)),
                                "token_index": int(index % max(1, self.tokens)),
                                "attention_mass": float(row[int(index)]),
                            }
                            for index in top
                        ],
                    }
                )
        return {
            "attention_available": True,
            "attention_mass_per_role": np.round(role_mass, 6).tolist(),
            "attention_mass_per_candidate_role": np.round(candidate_role_mass, 6).tolist(),
            "attention_mass_per_block": np.round(role_mass.sum(axis=1), 6).tolist(),
            "candidate_to_token_attention_entropy": np.round(entropy, 6).tolist(),
            "attention_diversity_across_blocks_l1": float(_mean(self.block_pair_l1)),
            "attention_diversity_across_roles_l1": float(np.mean(np.abs(role_mass - role_mass.mean(axis=1, keepdims=True)))) if role_mass.size else 0.0,
            "top_attended_tokens_per_block": top_tokens,
        }


def _ablation_metrics(result, examples: Sequence[MultiViewTaskExample], labels: np.ndarray, seed: int, variant: Stage5Variant) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    blocks = int(variant.block_count)
    if blocks > 1:
        for name, disabled in _disabled_block_sets(blocks).items():
            with _coordinator_attr(result.coordinator, "disabled_blocks", set(disabled)):
                pred = predict_latent_system(result, examples, "none", seed)
            metrics[f"remove_{name}"] = _accuracy(pred, labels)
    with _coordinator_attr(result.coordinator, "zero_candidate_features", True):
        metrics["zero_candidate_representations"] = _accuracy(predict_latent_system(result, examples, "none", seed), labels)
    with _coordinator_attr(result.coordinator, "zero_token_states", True):
        metrics["zero_role_token_states"] = _accuracy(predict_latent_system(result, examples, "none", seed), labels)
    metrics["shuffle_role_token_states_across_examples"] = _accuracy(
        predict_latent_system(result, examples, "role_token_states_shuffled_across_examples", seed), labels
    )
    metrics["shuffle_hidden_states_across_examples"] = _accuracy(
        predict_latent_system(result, examples, "hidden_states_shuffled_across_examples", seed), labels
    )
    metrics["freeze_shared_model_only_exact_frozen_comparator"] = "reported_as_frozen_accuracy"
    metrics["train_coordinator_only"] = "equivalent_to_exact_frozen_comparator"
    metrics["train_shared_model_only"] = "not_run_extra_fit"
    metrics["freeze_coordinator_only"] = "not_run_extra_fit"
    metrics["avenue_block_ablations"] = "not_applicable_single_avenue_stage5"
    return metrics


def _disabled_block_sets(blocks: int) -> Dict[str, Iterable[int]]:
    values: Dict[str, Iterable[int]] = {"block_1_only": [0], "final_block": [blocks - 1]}
    if blocks > 2:
        values["middle_block"] = [blocks // 2]
    return values


@contextmanager
def _coordinator_attr(coordinator, name: str, value):
    old = getattr(coordinator, name, None)
    setattr(coordinator, name, value)
    try:
        yield
    finally:
        setattr(coordinator, name, old)


def _calibration_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, object]:
    if logits.size == 0:
        return {}
    probs = _softmax_np(logits)
    pred = probs.argmax(axis=1)
    correct = pred == labels
    confidence = probs.max(axis=1)
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2] if probs.shape[1] > 1 else confidence
    gold_probs = probs[np.arange(len(labels)), labels]
    ranks = 1 + np.sum(probs > gold_probs[:, None], axis=1)
    entropy = -(probs.clip(1e-9) * np.log(probs.clip(1e-9))).sum(axis=1)
    bins = []
    ece = 0.0
    for low in np.linspace(0.0, 0.9, 10):
        high = low + 0.1
        mask = (confidence >= low) & (confidence < high if high < 1.0 else confidence <= high)
        if not np.any(mask):
            bins.append({"low": float(low), "high": float(high), "n": 0, "accuracy": 0.0, "confidence": 0.0})
            continue
        acc = float(correct[mask].mean())
        conf = float(confidence[mask].mean())
        ece += float(mask.mean()) * abs(acc - conf)
        bins.append({"low": float(low), "high": float(high), "n": int(mask.sum()), "accuracy": acc, "confidence": conf})
    return {
        "mean_confidence_correct": float(confidence[correct].mean()) if np.any(correct) else 0.0,
        "mean_confidence_incorrect": float(confidence[~correct].mean()) if np.any(~correct) else 0.0,
        "accuracy_vs_confidence_bins": bins,
        "expected_calibration_error": float(ece),
        "top1_top2_logit_margin_mean": float(margin.mean()),
        "gold_candidate_rank_mean": float(ranks.mean()),
        "top2_accuracy": float(np.mean(ranks <= 2)),
        "top3_accuracy": float(np.mean(ranks <= 3)),
        "candidate_distribution_entropy_mean": float(entropy.mean()),
        "confidence_correct_minus_incorrect": (
            float(confidence[correct].mean() - confidence[~correct].mean()) if np.any(correct) and np.any(~correct) else 0.0
        ),
    }


def _depth_metrics(early_logits: np.ndarray, labels: np.ndarray) -> Dict[str, object]:
    if early_logits.size == 0:
        return {"early_logits_available": False}
    # early_logits shape: blocks, examples, candidates
    pred_by_block = early_logits.argmax(axis=2)
    correct = pred_by_block == labels[None, :]
    block_acc = {f"block_{idx + 1}": float(correct[idx].mean()) for idx in range(correct.shape[0])}
    transitions = {}
    for idx in range(correct.shape[0] - 1):
        a = correct[idx]
        b = correct[idx + 1]
        transitions[f"block_{idx + 1}_to_{idx + 2}"] = _transition_counts(a, b)
    if correct.shape[0] > 1:
        transitions["block_1_to_final"] = _transition_counts(correct[0], correct[-1])
    return {
        "early_logits_available": True,
        "accuracy_after_intermediate_block": block_acc,
        "later_blocks_transition_counts": transitions,
        "later_blocks_correct_earlier_missed": sum(value["wrong_to_right"] for value in transitions.values()),
        "later_blocks_break_earlier_correct": sum(value["right_to_wrong"] for value in transitions.values()),
    }


def _transition_counts(before: np.ndarray, after: np.ndarray) -> Dict[str, int]:
    return {
        "wrong_to_right": int((~before & after).sum()),
        "right_to_wrong": int((before & ~after).sum()),
        "wrong_to_wrong": int((~before & ~after).sum()),
        "right_to_right": int((before & after).sum()),
    }


def _stability_metrics(fit: Dict[str, object], metrics: Dict[str, object]) -> Dict[str, object]:
    train_history = list(getattr(fit["trainable"], "history", []))
    dev_acc = [float(row.get("dev_acc", 0.0)) for row in train_history]
    losses = [float(row.get("train_loss", 0.0)) for row in train_history]
    best_epoch = int(max(range(len(dev_acc)), key=lambda idx: dev_acc[idx]) + 1) if dev_acc else 0
    values = {
        "train_dev_gap_at_best": 0.0,
        "epoch_of_best_dev_accuracy": best_epoch,
        "early_stopping_epoch": len(train_history),
        "number_of_epochs_trained": len(train_history),
        "loss_curve_summary": _curve_summary(losses),
        "dev_accuracy_curve_summary": _curve_summary(dev_acc),
        "gradient_norm_statistics": {
            "agent_grad_norm_mean": fit["trainable"].audit.get("agent_grad_norm_mean"),
            "agent_grad_norm_std": fit["trainable"].audit.get("agent_grad_norm_std"),
            "coordinator_grad_norm_mean": fit["trainable"].audit.get("coordinator_grad_norm_mean"),
            "coordinator_grad_norm_std": fit["trainable"].audit.get("coordinator_grad_norm_std"),
        },
        "training_diverged_or_collapsed": bool(any(not np.isfinite(value) for value in losses) or (dev_acc and max(dev_acc) <= 0.18)),
        "nan_inf_checks_pass": bool(all(np.isfinite(value) for value in losses + dev_acc)),
    }
    if train_history and best_epoch > 0:
        best = train_history[best_epoch - 1]
        values["train_dev_gap_at_best"] = float(best.get("train_acc", 0.0) - best.get("dev_acc", 0.0))
    return values


def _curve_summary(values: Sequence[float]) -> Dict[str, float]:
    return {
        "first": float(values[0]) if values else 0.0,
        "last": float(values[-1]) if values else 0.0,
        "best": float(max(values)) if values else 0.0,
        "min": float(min(values)) if values else 0.0,
        "mean": _mean(values),
        "std": _std(values),
    }


def _row_compute_metrics(stage: StageConfig, variant: Stage5Variant, fit: Dict[str, object], metrics: Dict[str, object], training_seconds: float) -> Dict[str, object]:
    test = metrics["test"]
    infer = test["timing"]
    peak_gb = _cuda_gb(fit["trainable"].audit.get("cuda_max_memory_allocated", 0))
    acc = float(test["accuracy"]["trainable"])
    frozen = float(test["accuracy"]["frozen"])
    per_ms = float(infer.get("inference_time_per_example_ms", 0.0))
    coord_params = int(sum(parameter.numel() for parameter in fit["trainable"].coordinator.parameters()))
    shared_params = int(fit["trainable"].audit.get("shared_parameter_count") or 0)
    attention_ops = int(8 * 4 * 128 * int(variant.block_count))
    return {
        "wall_clock_training_time_seconds": float(training_seconds),
        "wall_clock_inference_time_per_example_ms": per_ms,
        "latency_per_batch_ms": infer.get("latency_per_batch_ms_mean", 0.0),
        "peak_cuda_memory_gb": peak_gb,
        "average_cuda_memory_gb": infer.get("average_cuda_memory_gb", 0.0),
        "batch_size_used": int(stage.batch_size),
        "fallback_batch_size_if_oom": int(stage.batch_size),
        "number_of_clone_passes_inference_test": int(stage.n_test * 4),
        "number_of_coordinator_blocks": int(variant.block_count),
        "approximate_attention_operations_per_example": attention_ops,
        "trainable_parameter_count": int(fit["trainable"].param_count),
        "frozen_parameter_count": int(fit["frozen"].param_count),
        "coordinator_parameter_count": coord_params,
        "shared_model_parameter_count": shared_params,
        "accuracy_per_second_training": acc / max(training_seconds, 1e-9),
        "accuracy_per_peak_cuda_gb": acc / max(peak_gb, 1e-9),
        "delta_over_frozen_per_extra_coordinator_block": (acc - frozen) / max(1, int(variant.block_count) - 1),
        "marginal_accuracy_gain_per_added_block": "computed_in_phase_summary",
        "accuracy_per_inference_millisecond": acc / max(per_ms, 1e-9),
        "trainable_frozen_delta_per_inference_millisecond": (acc - frozen) / max(per_ms, 1e-9),
    }


def _candidate_for_variant(stage: StageConfig, variant: Stage5Variant):
    locked = next(candidate for candidate in _candidate_specs() if candidate.name == "topk_attention_no_head")
    msg = replace(
        locked.message_config,
        readout_source="pooled",
        use_message_head=False,
        use_private_cue_aux=False,
        aux_loss_weight=0.0,
        coordinator_family="stacked_candidate_token_cross_attention",
    )
    coord = replace(
        _coordinator_config(stage, family="stacked_candidate_token_cross_attention", num_layers=int(variant.block_count), dropout=0.0),
        family="stacked_candidate_token_cross_attention",
        shared_block_weights=bool(variant.shared_block_weights),
        use_residual_update=bool(variant.use_residual_update),
        use_layer_norm=bool(variant.use_layer_norm),
        use_feedforward=bool(variant.use_feedforward),
        candidate_dropout=float(variant.candidate_dropout),
        block_dropout=float(variant.block_dropout),
    )
    return SimpleNamespace(name=variant.name, description=variant.description, message_config=msg, coordinator_config=coord)


def _variant_plan(config: Dict[str, object]) -> List[Stage5Variant]:
    variants = [
        Stage5Variant(
            name=SELECTED_BASELINE,
            block_count=1,
            description="Existing Stage 4 candidate-token-direct one-block baseline; LR 3e-4; clip 1.0.",
        ),
        Stage5Variant(
            name="stacked_candidate_token_direct_2blocks_lr3e4_clip1",
            block_count=2,
            description="Two unshared candidate-token-direct coordination blocks with residual, LayerNorm, and feedforward.",
        ),
        Stage5Variant(
            name="stacked_candidate_token_direct_3blocks_lr3e4_clip1",
            block_count=3,
            description="Three unshared candidate-token-direct coordination blocks with residual, LayerNorm, and feedforward.",
        ),
        Stage5Variant(
            name="stacked_candidate_token_direct_4blocks_lr3e4_clip1",
            block_count=4,
            description="Four unshared candidate-token-direct coordination blocks with residual, LayerNorm, and feedforward.",
        ),
    ]
    if bool(config.get("include_optional_bounded_variants", False)):
        variants.extend(
            [
                Stage5Variant(
                    name="stacked_candidate_token_direct_4blocks_shared_lr3e4_clip1",
                    block_count=4,
                    shared_block_weights=True,
                    description="Four candidate-token-direct blocks with shared block weights.",
                ),
                Stage5Variant(
                    name="stacked_candidate_token_direct_4blocks_noff_lr3e4_clip1",
                    block_count=4,
                    use_feedforward=False,
                    description="Four candidate-token-direct blocks without the block feedforward MLP.",
                ),
                Stage5Variant(
                    name="stacked_candidate_token_direct_4blocks_candidate_dropout05_lr3e4_clip1",
                    block_count=4,
                    candidate_dropout=0.05,
                    description="Four candidate-token-direct blocks with bounded candidate dropout.",
                ),
            ]
        )
    return variants


def _control_examples(examples: Sequence[MultiViewTaskExample], seed: int) -> Dict[str, Sequence[MultiViewTaskExample]]:
    return {
        "candidate_only": _candidate_only(examples),
        "view_masked_candidates_visible": apply_example_control(examples, "view_masked", seed=seed + 11_000),
        "schema_only": _schema_only(examples),
        "null_evidence_values": _null_evidence_values(examples),
        "evidence_only_no_candidates": _evidence_only_no_candidates(examples),
        "value_shuffle_within_schema": _value_shuffle_within_schema(examples, seed + 12_000),
        "cross_example_view_bundle_shuffle": _cross_example_view_bundle_shuffle(examples, seed + 13_000),
        "candidate_evidence_mismatch": _candidate_evidence_mismatch(examples, seed + 14_000),
        "schema_preserved_role_value_shuffle": _schema_preserved_role_value_shuffle(examples, seed + 15_000),
    }


def _per_family_compare(trainable, frozen, examples: Sequence[MultiViewTaskExample], seed: int) -> Dict[str, Dict[str, float]]:
    train_pred = predict_latent_system(trainable, examples, "none", seed + 61_000)
    frozen_pred = predict_latent_system(frozen, examples, "none", seed + 61_000)
    labels = _labels(examples)
    grouped: Dict[str, List[int]] = {}
    for index, example in enumerate(examples):
        grouped.setdefault(str(example_oracle_metadata(example).get("problem_family")), []).append(index)
    out = {}
    for family, indices in sorted(grouped.items()):
        idx = np.asarray(indices, dtype=np.int64)
        train_acc = float(np.mean(train_pred[idx] == labels[idx]))
        frozen_acc = float(np.mean(frozen_pred[idx] == labels[idx]))
        out[family] = {"n": int(len(indices)), "trainable": train_acc, "frozen": frozen_acc, "delta": train_acc - frozen_acc}
    return out


def _phase_summary(rows: Sequence[Dict[str, object]], config: Dict[str, object]) -> Dict[str, object]:
    completed = [row for row in rows if row.get("status") == "completed"]
    by_variant: Dict[str, List[Dict[str, object]]] = {}
    for row in completed:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    variant_summaries = []
    baseline_mean = None
    baseline_by_seed = {}
    for row in completed:
        if int(row.get("block_count", 0)) == 1:
            baseline_by_seed[int(row["seed"])] = float(row["test_accuracy"]["trainable"])
    if baseline_by_seed:
        baseline_mean = _mean(list(baseline_by_seed.values()))
    near = float(config.get("near_chance_max", NEAR_CHANCE_DEFAULT))
    for variant, variant_rows in sorted(by_variant.items()):
        train = [float(row["test_accuracy"]["trainable"]) for row in variant_rows]
        frozen = [float(row["test_accuracy"]["frozen"]) for row in variant_rows]
        deltas = [float(row["test_delta"]) for row in variant_rows]
        controls = {name: _mean([float(row["test_accuracy"].get(name, 1.0)) for row in variant_rows]) for name in PRIMARY_CONTROLS}
        blocks = int(variant_rows[0].get("block_count", 0)) if variant_rows else 0
        matching_baseline = [float(row["test_accuracy"]["trainable"]) - baseline_by_seed[int(row["seed"])] for row in variant_rows if int(row["seed"]) in baseline_by_seed and blocks > 1]
        variant_summaries.append(
            {
                "variant": variant,
                "block_count": blocks,
                "n_seeds": len(variant_rows),
                "mean_accuracy": _mean(train),
                "std_accuracy": _std(train),
                "min_seed_accuracy": min(train) if train else 0.0,
                "max_seed_accuracy": max(train) if train else 0.0,
                "median_seed_accuracy": float(median(train)) if train else 0.0,
                "worst_seed_accuracy": min(train) if train else 0.0,
                "mean_frozen_accuracy": _mean(frozen),
                "mean_trainable_frozen_delta": _mean(deltas),
                "bootstrap_95_ci_delta": list(_bootstrap_ci(deltas)) if deltas else [0.0, 0.0],
                "trainable_beats_frozen_seed_count": sum(delta > 0.0 for delta in deltas),
                "weak_seed_count_below_0_60": sum(value < 0.60 for value in train),
                "improvement_over_1_block_baseline": _mean(matching_baseline) if matching_baseline else ((float(_mean(train) - baseline_mean) if baseline_mean is not None and blocks > 1 else 0.0)),
                "controls_pass": all(value <= near for value in controls.values()),
                "control_means": controls,
                "dev_selection_score": _mean([float(row["dev_delta"]) for row in variant_rows]),
            }
        )
    variant_summaries.sort(key=lambda row: (float(row["dev_selection_score"]), float(row["mean_accuracy"])), reverse=True)
    _attach_marginal_gain(variant_summaries)
    return {
        "ran": bool(rows),
        "n_completed": len(completed),
        "variant_summaries": variant_summaries,
        "depth_scaling": _depth_scaling_summary(variant_summaries),
        "success_criteria": _overall_criteria(completed, config),
        "selection_uses": "dev metrics and controls only; final test seeds are fresh and fixed",
    }


def _attach_marginal_gain(summaries: List[Dict[str, object]]) -> None:
    by_blocks = {int(row["block_count"]): row for row in summaries}
    for blocks, row in by_blocks.items():
        previous = by_blocks.get(blocks - 1)
        row["marginal_gain_from_adding_block"] = (
            float(row["mean_accuracy"]) - float(previous["mean_accuracy"]) if previous is not None else 0.0
        )


def _depth_scaling_summary(summaries: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    for row in sorted(summaries, key=lambda item: int(item["block_count"])):
        out.append(
            {
                "block_count": int(row["block_count"]),
                "mean_accuracy": float(row["mean_accuracy"]),
                "min_seed_accuracy": float(row["min_seed_accuracy"]),
                "std_across_seeds": float(row["std_accuracy"]),
                "trainable_frozen_delta": float(row["mean_trainable_frozen_delta"]),
                "improvement_over_1_block_baseline": float(row["improvement_over_1_block_baseline"]),
                "marginal_gain_from_adding_block": float(row.get("marginal_gain_from_adding_block", 0.0)),
            }
        )
    return out


def _overall_criteria(rows: Sequence[Dict[str, object]], config: Dict[str, object]) -> List[Dict[str, object]]:
    if not rows:
        return [{"criterion": "completed seeds", "pass": False, "value": 0}]
    near = float(config.get("near_chance_max", NEAR_CHANCE_DEFAULT))
    inv_tol = float(config.get("invariance_tolerance", 0.08))
    trainable = [float(row["test_accuracy"]["trainable"]) for row in rows]
    deltas = [float(row["test_delta"]) for row in rows]
    ci_low, ci_high = _bootstrap_ci(deltas) if deltas else (0.0, 0.0)
    trainable_mean = _mean(trainable)
    control_means = {name: _mean([float(row["test_accuracy"].get(name, 1.0)) for row in rows]) for name in PRIMARY_CONTROLS}
    invariance = {name: _mean([float(row["test_accuracy"].get(name, 0.0)) for row in rows]) for name in INVARIANCE_CONTROLS}
    return [
        {"criterion": "trainable beats frozen on at least 8/10 final seeds", "pass": len(rows) >= 10 and sum(delta > 0 for delta in deltas) >= 8, "value": sum(delta > 0 for delta in deltas)},
        {"criterion": "mean trainable-frozen delta >= +0.20", "pass": _mean(deltas) >= 0.20, "value": _mean(deltas)},
        {"criterion": "bootstrap CI lower bound > +0.05", "pass": ci_low > 0.05, "value": [ci_low, ci_high]},
        {"criterion": "Stage 4 controls near chance", "pass": all(value <= near for value in control_means.values()), "value": control_means},
        {
            "criterion": "physical-order and candidate-order invariance pass",
            "pass": all(abs(value - trainable_mean) <= inv_tol for value in invariance.values()),
            "value": {key: {"mean_accuracy": value, "delta_from_trainable": value - trainable_mean} for key, value in invariance.items()},
        },
        {
            "criterion": "trainable shared model receives gradients and changes",
            "pass": all(float(row.get("trainable_audit", {}).get("agent_grad_norm_mean") or 0.0) > 0.0 and float(row.get("trainable_audit", {}).get("agent_parameter_delta") or 0.0) > 0.0 for row in rows),
            "value": "all_completed_rows",
        },
        {
            "criterion": "frozen comparator remains frozen",
            "pass": all(_audit_float(row.get("frozen_audit", {}), "agent_grad_norm_mean", 1.0) == 0.0 and _audit_float(row.get("frozen_audit", {}), "agent_parameter_delta", 1.0) == 0.0 for row in rows),
            "value": "all_completed_rows",
        },
    ]


def _seed_criteria(row: Dict[str, object], config: Dict[str, object]) -> Dict[str, object]:
    near = float(config.get("near_chance_max", NEAR_CHANCE_DEFAULT))
    inv_tol = float(config.get("invariance_tolerance", 0.08))
    trainable = float(row["test_accuracy"]["trainable"])
    return {
        "trainable_beats_frozen": float(row["test_delta"]) > 0.0,
        "primary_controls_near_chance": {name: float(row["test_accuracy"].get(name, 1.0)) <= near for name in PRIMARY_CONTROLS},
        "invariance_controls_preserve_accuracy": {
            name: abs(float(row["test_accuracy"].get(name, 0.0)) - trainable) <= inv_tol for name in INVARIANCE_CONTROLS
        },
        "frozen_agent_unchanged": _audit_float(row.get("frozen_audit", {}), "agent_parameter_delta", 1.0) == 0.0,
        "trainable_agent_changed": float(row.get("trainable_audit", {}).get("agent_parameter_delta") or 0.0) > 0.0,
    }


def _seed_failure_analysis(row: Dict[str, object], config: Dict[str, object]) -> Dict[str, object]:
    near = float(config.get("near_chance_max", NEAR_CHANCE_DEFAULT))
    controls_bad = [name for name in PRIMARY_CONTROLS if float(row["test_accuracy"].get(name, 0.0)) > near]
    return {
        "failed_seed": bool(float(row["test_delta"]) <= 0.0 or controls_bad),
        "controls_above_threshold": controls_bad,
        "likely_failure_type": "control_failure" if controls_bad else "frozen_strength_or_seed_instability" if float(row["test_delta"]) <= 0.0 else "none",
    }


def _select_top_variants(result: Dict[str, object], phase: str, limit: int) -> List[str]:
    phase_data = result.get("phases", {}).get(phase, {})
    summaries = [row for row in phase_data.get("summary", {}).get("variant_summaries", []) if row.get("controls_pass", False)]
    summaries.sort(key=lambda row: (float(row.get("dev_selection_score", 0.0)), float(row.get("mean_accuracy", 0.0))), reverse=True)
    return [str(row["variant"]) for row in summaries[:limit]]


def _case_records(examples: Sequence[MultiViewTaskExample]) -> List[Dict[str, object]]:
    rows = []
    for index, example in enumerate(examples):
        oracle = example_oracle_metadata(example)
        rows.append(
            {
                "index": index,
                "family": str(oracle.get("problem_family")),
                "gold_candidate": int(example.label),
                "candidate_metadata_sanitized_view": [
                    {
                        "slot": candidate_index,
                        "candidate_id": candidate.candidate_id,
                        "attributes": list(candidate.attributes),
                        "text_excerpt": candidate.text[:240],
                    }
                    for candidate_index, candidate in enumerate(example.candidates)
                ],
            }
        )
    return rows


def _error_cases(result: Dict[str, object], max_per_bucket: int = 12) -> List[Dict[str, object]]:
    rows = [row for phase in result.get("phases", {}).values() for row in phase.get("rows", []) if row.get("status") == "completed"]
    by_phase_seed: Dict[tuple[str, int], List[Dict[str, object]]] = {}
    for row in rows:
        by_phase_seed.setdefault((str(row["phase"]), int(row["seed"])), []).append(row)
    cases: List[Dict[str, object]] = []
    for (_phase, _seed), group in by_phase_seed.items():
        baseline = next((row for row in group if int(row.get("block_count", 0)) == 1), None)
        if baseline is not None:
            base_pred = np.asarray(baseline["diagnostic_predictions"]["trainable_pred"], dtype=np.int64)
            gold = np.asarray(baseline["diagnostic_predictions"]["gold"], dtype=np.int64)
            for row in group:
                if int(row.get("block_count", 0)) <= 1:
                    continue
                pred = np.asarray(row["diagnostic_predictions"]["trainable_pred"], dtype=np.int64)
                _append_bucket_cases(cases, "1-block wrong, stacked right", row, (~(base_pred == gold)) & (pred == gold), max_per_bucket)
                _append_bucket_cases(cases, "1-block right, stacked wrong", row, (base_pred == gold) & (~(pred == gold)), max_per_bucket)
        if len(group) > 1:
            gold = np.asarray(group[0]["diagnostic_predictions"]["gold"], dtype=np.int64)
            all_correct = np.ones_like(gold, dtype=bool)
            all_wrong = np.ones_like(gold, dtype=bool)
            for row in group:
                pred = np.asarray(row["diagnostic_predictions"]["trainable_pred"], dtype=np.int64)
                all_correct &= pred == gold
                all_wrong &= pred != gold
            _append_bucket_cases(cases, "all variants right", group[0], all_correct, max_per_bucket)
            _append_bucket_cases(cases, "all variants wrong", group[0], all_wrong, max_per_bucket)
        for row in group:
            pred = np.asarray(row["diagnostic_predictions"]["trainable_pred"], dtype=np.int64)
            frozen = np.asarray(row["diagnostic_predictions"]["frozen_pred"], dtype=np.int64)
            gold = np.asarray(row["diagnostic_predictions"]["gold"], dtype=np.int64)
            _append_bucket_cases(cases, "frozen right, trainable wrong", row, (frozen == gold) & (pred != gold), max_per_bucket)
            _append_bucket_cases(cases, "trainable right, frozen wrong", row, (pred == gold) & (frozen != gold), max_per_bucket)
    return cases


def _append_bucket_cases(cases: List[Dict[str, object]], bucket: str, row: Dict[str, object], mask: np.ndarray, max_per_bucket: int) -> None:
    existing = sum(1 for item in cases if item.get("bucket") == bucket and item.get("phase") == row.get("phase") and item.get("variant") == row.get("variant"))
    if existing >= max_per_bucket:
        return
    indices = np.flatnonzero(mask)[: max(0, max_per_bucket - existing)]
    probs = row["diagnostic_predictions"].get("trainable_probabilities", [])
    pred = row["diagnostic_predictions"].get("trainable_pred", [])
    gold = row["diagnostic_predictions"].get("gold", [])
    records = row.get("test_case_records", [])
    for index in indices:
        p = np.asarray(probs[int(index)], dtype=np.float64) if probs else np.zeros(0)
        sorted_probs = np.sort(p) if p.size else np.zeros(0)
        cases.append(
            {
                "bucket": bucket,
                "phase": row.get("phase"),
                "variant": row.get("variant"),
                "seed": row.get("seed"),
                "family": records[int(index)].get("family") if int(index) < len(records) else None,
                "example_index": int(index),
                "gold_candidate": int(gold[int(index)]) if gold else None,
                "predicted_candidate": int(pred[int(index)]) if pred else None,
                "candidate_probabilities": p.round(6).tolist(),
                "top2_margin": float(sorted_probs[-1] - sorted_probs[-2]) if sorted_probs.size >= 2 else 0.0,
                "role_attention_summary": row.get("attention_summary", {}),
                "top_attended_tokens_per_block": row.get("attention_summary", {}).get("top_attended_tokens_per_block", []),
                "candidate_metadata_sanitized_view": records[int(index)].get("candidate_metadata_sanitized_view") if int(index) < len(records) else [],
                "control_status": row.get("success_criteria", {}),
            }
        )


def _attention_rows(result: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    for phase_name, phase in result.get("phases", {}).items():
        for row in phase.get("rows", []):
            if row.get("status") != "completed":
                continue
            rows.append(
                {
                    "event": "attention_summary",
                    "phase": phase_name,
                    "variant": row.get("variant"),
                    "seed": row.get("seed"),
                    "block_count": row.get("block_count"),
                    "attention_summary": row.get("attention_summary", {}),
                    "representation_summary": row.get("representation_summary", {}),
                    "time_utc": row.get("completed_at_utc", _now()),
                }
            )
    return rows


def _compute_summary(result: Dict[str, object]) -> Dict[str, object]:
    rows = [row for phase in result.get("phases", {}).values() for row in phase.get("rows", []) if row.get("status") == "completed"]
    by_variant: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    summary = {}
    for variant, values in by_variant.items():
        compute = [row.get("compute_metrics", {}) for row in values]
        summary[variant] = {
            "block_count": int(values[0].get("block_count", 0)),
            "mean_training_seconds": _mean([float(item.get("wall_clock_training_time_seconds", 0.0)) for item in compute]),
            "mean_inference_ms_per_example": _mean([float(item.get("wall_clock_inference_time_per_example_ms", 0.0)) for item in compute]),
            "mean_peak_cuda_memory_gb": _mean([float(item.get("peak_cuda_memory_gb", 0.0)) for item in compute]),
            "mean_accuracy_per_inference_millisecond": _mean([float(item.get("accuracy_per_inference_millisecond", 0.0)) for item in compute]),
            "mean_delta_per_inference_millisecond": _mean([float(item.get("trainable_frozen_delta_per_inference_millisecond", 0.0)) for item in compute]),
            "coordinator_parameter_count": int(compute[0].get("coordinator_parameter_count", 0)) if compute else 0,
        }
    return {
        "created_at_utc": _now(),
        "variant_compute_summary": summary,
        "efficiency_normalized_metrics": _efficiency_normalized(result),
    }


def _efficiency_normalized(result: Dict[str, object]) -> List[Dict[str, object]]:
    rows = [row for phase in result.get("phases", {}).values() for row in phase.get("rows", []) if row.get("status") == "completed"]
    baseline = [row for row in rows if int(row.get("block_count", 0)) == 1]
    base_acc = _mean([float(row["test_accuracy"]["trainable"]) for row in baseline])
    base_latency = _mean([float(row["compute_metrics"].get("wall_clock_inference_time_per_example_ms", 0.0)) for row in baseline])
    base_memory = _mean([float(row["compute_metrics"].get("peak_cuda_memory_gb", 0.0)) for row in baseline])
    out = []
    for row in rows:
        compute = row.get("compute_metrics", {})
        acc = float(row["test_accuracy"]["trainable"])
        latency = float(compute.get("wall_clock_inference_time_per_example_ms", 0.0))
        memory = float(compute.get("peak_cuda_memory_gb", 0.0))
        added_blocks = max(1, int(row.get("block_count", 1)) - 1)
        out.append(
            {
                "phase": row.get("phase"),
                "variant": row.get("variant"),
                "seed": row.get("seed"),
                "accuracy_per_inference_millisecond": acc / max(latency, 1e-9),
                "delta_per_inference_millisecond": float(row["test_delta"]) / max(latency, 1e-9),
                "accuracy_per_peak_cuda_gb": acc / max(memory, 1e-9),
                "improvement_over_1_block_per_added_block": (acc - base_acc) / added_blocks,
                "improvement_over_1_block_per_added_latency_ms": (acc - base_acc) / max(latency - base_latency, 1e-9),
                "improvement_over_1_block_per_added_memory_gb": (acc - base_acc) / max(memory - base_memory, 1e-9),
            }
        )
    return out


def _write_outputs(
    result: Dict[str, object],
    output_path: Path,
    audit_path: Path,
    report_path: Path,
    error_path: Path,
    attention_path: Path,
    compute_path: Path,
) -> None:
    _attach_pre_run_candidate_metadata(result)
    for phase in result.get("phases", {}).values():
        for row in phase.get("rows", []):
            if row.get("status") == "completed":
                row["success_criteria"] = _seed_criteria(row, result.get("config", {}))
                row["failure_analysis"] = _seed_failure_analysis(row, result.get("config", {}))
        if "rows" in phase:
            phase["summary"] = _phase_summary(phase.get("rows", []), result.get("config", {}))
    result["summary"] = _summary(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(output_path)
    audit_rows = _audit_rows(result)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in audit_rows) + ("\n" if audit_rows else ""), encoding="utf-8")
    errors = _error_cases(result, max_per_bucket=int(result.get("config", {}).get("error_cases_per_bucket", 12)))
    error_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in errors) + ("\n" if errors else ""), encoding="utf-8")
    attention_rows = _attention_rows(result)
    attention_path.parent.mkdir(parents=True, exist_ok=True)
    attention_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in attention_rows) + ("\n" if attention_rows else ""), encoding="utf-8")
    compute = _compute_summary(result)
    compute_path.parent.mkdir(parents=True, exist_ok=True)
    compute_path.write_text(json.dumps(compute, indent=2, sort_keys=True), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(result), encoding="utf-8")


def _attach_pre_run_candidate_metadata(result: Dict[str, object]) -> None:
    seed_values = {}
    rows = result.get("pre_run", {}).get("shortcut_diagnostics", {}).get("seed_rows", [])
    for row in rows:
        baselines = row.get("baselines", {})
        if "candidate_metadata_only_accuracy" in baselines:
            seed_values[int(row.get("seed", -1))] = float(baselines.get("candidate_metadata_only_accuracy", 1.0))
    fallback = _mean(list(seed_values.values())) if seed_values else 0.0
    for phase in result.get("phases", {}).values():
        for row in phase.get("rows", []):
            if row.get("status") != "completed":
                continue
            value = seed_values.get(int(row.get("seed", -1)), fallback)
            row.setdefault("test_accuracy", {})["candidate_metadata_only"] = value
            row.setdefault("dev_accuracy", {})["candidate_metadata_only"] = value


def _audit_rows(result: Dict[str, object]) -> List[Dict[str, object]]:
    rows = [{"event": "pre_run_summary", "time_utc": _now(), **dict(result.get("pre_run", {}).get("summary", {}))}]
    for phase_name, phase in result.get("phases", {}).items():
        rows.append({"event": "phase_summary", "phase": phase_name, "summary": phase.get("summary", {}), "time_utc": _now()})
        for row in phase.get("rows", []):
            rows.append(
                {
                    "event": "variant_seed_completed",
                    "phase": phase_name,
                    "variant": row.get("variant"),
                    "block_count": row.get("block_count"),
                    "seed": row.get("seed"),
                    "test_accuracy": row.get("test_accuracy"),
                    "test_delta": row.get("test_delta"),
                    "success_criteria": row.get("success_criteria"),
                    "failure_analysis": row.get("failure_analysis"),
                    "trainable_audit": row.get("trainable_audit"),
                    "frozen_audit": row.get("frozen_audit"),
                    "compute_metrics": row.get("compute_metrics"),
                    "time_utc": row.get("completed_at_utc", _now()),
                }
            )
    rows.extend(result.get("oom_retries", []))
    rows.extend(result.get("failed_or_interrupted_seeds", []))
    return rows


def _summary(result: Dict[str, object]) -> Dict[str, object]:
    phase_summaries = {phase: data.get("summary", {}) for phase, data in result.get("phases", {}).items()}
    final_summary = phase_summaries.get("final", {})
    medium_summary = phase_summaries.get("medium", {})
    selected = result.get("metadata", {}).get("final_variant_locked_before_final_seeds")
    return {
        "phase_summaries": phase_summaries,
        "selected_final_variant": selected,
        "final_success": bool(final_summary.get("success_criteria")) and all(bool(item.get("pass")) for item in final_summary.get("success_criteria", [])),
        "medium_passed_variants": [row.get("variant") for row in medium_summary.get("variant_summaries", []) if row.get("controls_pass")],
        "no_success_claim_if_controls_fail": True,
    }


def _render_report(result: Dict[str, object]) -> str:
    config = result.get("config", {})
    lines = [
        "# Stage 5 Stacked Candidate-Token-Direct Coordination Full Metrics",
        "",
        "## Scope",
        "",
        f"- Baseline: `{SELECTED_BASELINE}`.",
        "- Dataset/control policy: fixed Stage 3.8 schema-aware benchmark; no label, metadata-separation, leakage, or corruption-control changes.",
        "- Comparator: exact same coordinator architecture per variant with shared model frozen.",
        "- Search protocol: cheap `[0,1,2]`, medium `[0,1,2,3,4]`, final fresh `[40..49]`; selection uses dev metrics and controls.",
        f"- Final locked variant: `{result.get('metadata', {}).get('final_variant_locked_before_final_seeds')}`.",
        "",
        "## Required Questions",
        "",
    ]
    qa = _report_answers(result)
    for question, answer in qa.items():
        lines.append(f"- {question}: `{answer}`")
    lines.extend(["", "## Variant Summaries", ""])
    for phase_name, phase in result.get("phases", {}).items():
        lines.extend([f"### {phase_name.title()}", "", "| variant | blocks | seeds | mean acc | min | std | delta | vs 1-block | weak<0.60 | controls |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"])
        for row in phase.get("summary", {}).get("variant_summaries", []):
            lines.append(
                f"| {row['variant']} | {int(row['block_count'])} | {int(row['n_seeds'])} | {float(row['mean_accuracy']):.4f} | {float(row['min_seed_accuracy']):.4f} | {float(row['std_accuracy']):.4f} | {float(row['mean_trainable_frozen_delta']):.4f} | {float(row['improvement_over_1_block_baseline']):.4f} | {int(row['weak_seed_count_below_0_60'])} | `{bool(row['controls_pass'])}` |"
            )
        lines.append("")
    lines.extend(["## Per-Seed Final Rows", ""])
    lines.append("| variant | seed | trainable | frozen | delta | text | raw | oracle | weak | peak GB | inf ms/ex |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|")
    for row in result.get("phases", {}).get("final", {}).get("rows", []):
        test = row.get("test_accuracy", {})
        compute = row.get("compute_metrics", {})
        lines.append(
            "| {variant} | {seed} | {trainable:.4f} | {frozen:.4f} | {delta:.4f} | {text:.4f} | {raw:.4f} | {oracle:.4f} | `{weak}` | {gb:.3f} | {ms:.3f} |".format(
                variant=row.get("variant"),
                seed=int(row.get("seed", 0)),
                trainable=float(test.get("trainable", 0.0)),
                frozen=float(test.get("frozen", 0.0)),
                delta=float(row.get("test_delta", 0.0)),
                text=float(test.get("text_only", 0.0)),
                raw=float(test.get("raw_latent", 0.0)),
                oracle=float(test.get("all_role_oracle", 0.0)),
                weak=bool(float(test.get("trainable", 0.0)) < 0.60),
                gb=float(compute.get("peak_cuda_memory_gb", 0.0)),
                ms=float(compute.get("wall_clock_inference_time_per_example_ms", 0.0)),
            )
        )
    lines.extend(["", "## Controls", ""])
    lines.append("| phase | variant | control | mean | pass |")
    lines.append("|---|---|---|---:|---|")
    for phase_name, phase in result.get("phases", {}).items():
        for summary in phase.get("summary", {}).get("variant_summaries", []):
            for control, value in summary.get("control_means", {}).items():
                lines.append(f"| {phase_name} | {summary['variant']} | {control} | {float(value):.4f} | `{float(value) <= float(config.get('near_chance_max', NEAR_CHANCE_DEFAULT))}` |")
    lines.extend(["", "## Acceptance Gates", ""])
    lines.append("| phase | criterion | pass | value |")
    lines.append("|---|---|---|---|")
    for phase_name, phase in result.get("phases", {}).items():
        for criterion in phase.get("summary", {}).get("success_criteria", []):
            lines.append(f"| {phase_name} | {criterion['criterion']} | `{bool(criterion['pass'])}` | `{json.dumps(criterion['value'], sort_keys=True)}` |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Do not claim success unless the final selected variant passes all Stage 4 controls, invariance, update/frozen audits, and delta gates.",
            "- Failed variants and controls are retained in the tables above and JSON/JSONL artifacts.",
        ]
    )
    return "\n".join(lines) + "\n"


def _report_answers(result: Dict[str, object]) -> Dict[str, object]:
    final = result.get("phases", {}).get("final", {}).get("summary", {})
    summaries = final.get("variant_summaries", [])
    selected = summaries[0] if summaries else {}
    baseline = next((row for row in summaries if int(row.get("block_count", -1)) == 1), {})
    controls_pass = bool(selected.get("controls_pass", False))
    return {
        "Did stacking improve mean accuracy?": _yes_no(float(selected.get("improvement_over_1_block_baseline", 0.0)) > 0.0),
        "Did stacking improve weak seeds?": "see weak_seed_count_below_0_60" if selected else "not_run",
        "Did stacking reduce variance?": _yes_no(bool(baseline) and float(selected.get("std_accuracy", 0.0)) < float(baseline.get("std_accuracy", 0.0))),
        "Did stacking improve hard families?": "see per_family_accuracy in JSON",
        "Did stacking increase trainable-frozen delta?": _yes_no(bool(baseline) and float(selected.get("mean_trainable_frozen_delta", 0.0)) > float(baseline.get("mean_trainable_frozen_delta", 0.0))),
        "Did stacking keep controls near chance?": _yes_no(controls_pass),
        "Did stacking preserve invariance?": "see success_criteria",
        "Did later blocks fix earlier errors?": "see depth_metrics.later_blocks_transition_counts",
        "Did later blocks introduce new errors?": "see depth_metrics.later_blocks_transition_counts",
        "Is the improvement worth the compute?": "see compute_metrics and efficiency_normalized_metrics",
        "Evidence of iterative refinement rather than just more parameters?": "supported only if early-block transitions show wrong->right exceeding right->wrong with controls passing",
    }


def _stage5_config(config: Dict[str, object]) -> Dict[str, object]:
    out = _stage38_full_config(config)
    out["output_path"] = str(out.get("output_path", RESULTS_PATH))
    out["audit_log_path"] = str(out.get("audit_log_path", AUDIT_PATH))
    out["error_cases_path"] = str(out.get("error_cases_path", ERROR_CASES_PATH))
    out["attention_summaries_path"] = str(out.get("attention_summaries_path", ATTENTION_SUMMARIES_PATH))
    out["compute_metrics_path"] = str(out.get("compute_metrics_path", COMPUTE_METRICS_PATH))
    out["report_path"] = str(out.get("report_path", REPORT_PATH))
    out["architecture"] = "stage5_stacked_candidate_token_direct"
    out["architecture_changes"] = "bounded_depth_scaling_only_before_final_selection"
    out["stage"] = {
        **dict(out["stage"]),
        "name": "stage5_stacked_coordination",
        "n_train": 512,
        "n_dev": 256,
        "n_test": 256,
        "seeds": [0, 1, 2],
        "epochs": int(dict(out["stage"]).get("epochs", 50)),
        "patience": int(dict(out["stage"]).get("patience", 6)),
        "batch_size": int(dict(out["stage"]).get("batch_size", 32)),
        "lr": float(dict(out["stage"]).get("lr", 0.0003)),
        "mixed_precision": dict(out["stage"]).get("mixed_precision", "bf16"),
    }
    out["protocol"] = dict(out.get("protocol", {})) or {
        "cheap": {"n_train": 512, "n_dev": 256, "n_test": 256, "seeds": [0, 1, 2]},
        "medium": {"n_train": 1024, "n_dev": 512, "n_test": 512, "seeds": [0, 1, 2, 3, 4]},
        "final": {"n_train": 2048, "n_dev": 512, "n_test": 1024, "seeds": [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]},
    }
    out["dataset_config"] = {
        **dict(out.get("dataset_config", {})),
        "dataset_source": BALANCED_34B_DATASET_SOURCE,
        "generator_version": "real_import_restore_candidate_balanced_34b_stage5_stacked_coordination",
        "candidate_representation": BALANCED_34B_CANDIDATE_REPRESENTATION,
    }
    out.setdefault("near_chance_max", NEAR_CHANCE_DEFAULT)
    out.setdefault("invariance_tolerance", 0.08)
    out.setdefault("error_cases_per_bucket", 12)
    out.setdefault("include_optional_bounded_variants", False)
    return out


def _phase_stage(config: Dict[str, object], phase: str, max_seeds: int = 0) -> StageConfig:
    protocol = dict(config.get("protocol", {})).get(phase)
    if not isinstance(protocol, dict):
        raise ValueError(f"missing stage5 protocol phase: {phase}")
    out = json.loads(json.dumps(config))
    seeds = list(protocol["seeds"])
    if max_seeds > 0:
        seeds = seeds[:max_seeds]
    out["stage"] = {
        **dict(out["stage"]),
        "name": f"stage5_{phase}",
        "n_train": int(protocol["n_train"]),
        "n_dev": int(protocol["n_dev"]),
        "n_test": int(protocol["n_test"]),
        "seeds": seeds,
        "epochs": int(dict(out["stage"]).get("epochs", 50)),
        "patience": int(dict(out["stage"]).get("patience", 6)),
        "batch_size": int(dict(out["stage"]).get("batch_size", 32)),
        "lr": float(dict(out["stage"]).get("lr", 0.0003)),
    }
    return _stage_from_config(out)


def _requested_phases(phase: str) -> List[str]:
    if phase == "all":
        return ["cheap", "medium", "final"]
    if phase == "cheap":
        return ["cheap"]
    if phase == "medium":
        return ["medium"]
    return ["final"]


def _metadata(config: Dict[str, object], config_path: Path, hardware: Dict[str, object], device: str) -> Dict[str, object]:
    return {
        "benchmark": BENCHMARK,
        "task": "stage5_stacked_candidate_token_direct_coordination",
        "baseline": SELECTED_BASELINE,
        "dataset": BALANCED_34B_DATASET_SOURCE,
        "candidate_representation": BALANCED_34B_CANDIDATE_REPRESENTATION,
        "schema_policy": "role-specific schema fields preserved as legitimate program-analysis evidence",
        "config_path": str(config_path),
        "device": device,
        "hardware": hardware,
        "created_or_updated_at_utc": _now(),
        "success_claim": "not evaluated until final selected variant passes all gates",
    }


def _load_result(path: Path) -> Dict[str, object]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64, copy=False)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def _concat_early_logits(parts: Sequence[torch.Tensor]) -> np.ndarray:
    if not parts:
        return np.zeros((0, 0, 0), dtype=np.float32)
    blocks = int(parts[0].shape[0])
    per_block = []
    for block_index in range(blocks):
        per_block.append(torch.cat([part[block_index] for part in parts], dim=0))
    return torch.stack(per_block, dim=0).numpy()


def _example_batches(examples: Sequence[MultiViewTaskExample], batch_size: int) -> Iterable[List[MultiViewTaskExample]]:
    for start in range(0, len(examples), batch_size):
        yield list(examples[start : start + batch_size])


def _mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0


def _cuda_gb(value: float | int | None) -> float:
    return float(value or 0.0) / float(1024**3)


def _yes_no(value: bool) -> str:
    return "yes" if bool(value) else "no"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
