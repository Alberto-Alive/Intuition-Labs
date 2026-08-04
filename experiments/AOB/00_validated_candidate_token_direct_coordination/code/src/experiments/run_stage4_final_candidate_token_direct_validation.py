from __future__ import annotations

import argparse
import json
import math
import traceback
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.nn import functional as F

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
    _candidate_specs,
    _coordinator_config,
    _fit_baselines,
    _leakage_passes,
    _per_role_ablation,
    _training_config,
)
from src.experiments.real_shared_weight_latent_coordination import (
    BENCHMARK,
    MessageChannelConfig,
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
    _save_candidate_pair_checkpoint,
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


DEFAULT_CONFIG = "configs/stage4_final_candidate_token_direct_validation.json"
SELECTED_ARCHITECTURE = "candidate_token_direct_lr3e4_clip1"
PRIMARY_CONTROLS = (
    "candidate_only",
    "candidate_metadata_only",
    "view_masked_candidates_visible",
    "schema_only",
    "null_evidence_values",
    "evidence_only_no_candidates",
    "value_shuffle_within_schema",
    "cross_example_view_bundle_shuffle",
    "candidate_evidence_mismatch",
    "schema_preserved_role_value_shuffle",
    "randomized_labels",
    "hidden_states_shuffled_across_examples",
)
LATENT_CONTROLS = (
    "candidate_only",
    "view_masked_candidates_visible",
    "schema_only",
    "null_evidence_values",
    "evidence_only_no_candidates",
    "value_shuffle_within_schema",
    "cross_example_view_bundle_shuffle",
    "candidate_evidence_mismatch",
    "schema_preserved_role_value_shuffle",
)
INVARIANCE_CONTROLS = (
    "physical_order_shuffled_roles_preserved",
    "candidate_order_shuffled_with_gold_remap",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final Stage 4 validation for candidate_token_direct_lr3e4_clip1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_final_validation(config, config_path=config_path)


def run_final_validation(config: Dict[str, object], config_path: Path) -> Dict[str, object]:
    config = _final_config(config)
    device, hardware = _configure_cuda(config)
    output_path = Path(str(config["output_path"]))
    audit_path = Path(str(config["audit_log_path"]))
    report_path = Path(str(config["report_path"]))
    result = _load_result(output_path)
    result.setdefault("metadata", _metadata(config, config_path, hardware, device))
    result["metadata"].update(_metadata(config, config_path, hardware, device))
    result["config"] = config
    result.setdefault("stage4_final_rows", [])
    result.setdefault("failed_or_interrupted_seeds", [])
    result.setdefault("oom_retries", [])

    print("stage4 final: running pre-run shortcut diagnostics")
    result["pre_run"] = _pre_run_checks(config, config_path, device)
    _write_outputs(result, output_path, audit_path, report_path)
    if bool(config.get("stop_on_preflight_failure", True)) and not bool(result["pre_run"]["summary"]["passes"]):
        print("stage4 final: pre-run checks failed; validation was not started")
        return result

    result["stage4_final_rows"] = [
        row
        for row in result["stage4_final_rows"]
        if row.get("status") == "completed" and row.get("candidate") == SELECTED_ARCHITECTURE and _row_has_required_checkpoints(row, config)
    ]
    completed = {int(row["seed"]) for row in result["stage4_final_rows"] if row.get("status") == "completed"}
    stage_base = _stage_from_config(config)
    seeds = [int(value) for value in config["stage"]["seeds"]]
    fallbacks = [int(value) for value in config.get("gpu_safety", {}).get("batch_size_fallbacks", [32, 16, 8])]
    effective_batch = int(config.get("gpu_safety", {}).get("effective_batch_size", fallbacks[0]))

    for seed in seeds:
        if seed in completed:
            print(f"stage4 final seed={seed}: existing completed row found; skipping")
            continue
        result["stage4_final_rows"] = [row for row in result["stage4_final_rows"] if int(row.get("seed", -1)) != seed]
        print(f"stage4 final seed={seed}: starting fixed selected-architecture validation")
        seed_completed = False
        for batch_size in fallbacks:
            accumulation = max(1, int(math.ceil(effective_batch / float(batch_size))))
            stage = replace(stage_base, seeds=(seed,), batch_size=batch_size, gradient_accumulation_steps=accumulation)
            try:
                row = _run_seed(stage, seed, device, hardware, config)
                result["stage4_final_rows"].append(row)
                seed_completed = True
                _write_outputs(result, output_path, audit_path, report_path)
                print(
                    "stage4 final seed={seed}: completed trainable={trainable:.4f} frozen={frozen:.4f} "
                    "delta={delta:.4f} max_cuda_gb={memory:.3f}".format(
                        seed=seed,
                        trainable=float(row["test_accuracy"]["trainable"]),
                        frozen=float(row["test_accuracy"]["frozen"]),
                        delta=float(row["test_delta"]),
                        memory=float(row.get("cuda_max_memory_allocated_gb", 0.0)),
                    )
                )
                break
            except RuntimeError as exc:
                if _is_oom(exc) and batch_size != fallbacks[-1]:
                    retry = _failure_row(seed, batch_size, accumulation, "oom_retry", exc)
                    result["oom_retries"].append(retry)
                    _clear_cuda()
                    _write_outputs(result, output_path, audit_path, report_path)
                    print(f"stage4 final seed={seed}: CUDA OOM at batch_size={batch_size}; retrying smaller batch")
                    continue
                failure = _failure_row(seed, batch_size, accumulation, "failed", exc)
                result["failed_or_interrupted_seeds"].append(failure)
                _clear_cuda()
                _write_outputs(result, output_path, audit_path, report_path)
                print(f"stage4 final seed={seed}: failed; see report/results for traceback")
                break
            except Exception as exc:
                failure = {
                    "seed": int(seed),
                    "batch_size": int(batch_size),
                    "gradient_accumulation_steps": int(accumulation),
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "time_utc": _now(),
                }
                result["failed_or_interrupted_seeds"].append(failure)
                _clear_cuda()
                _write_outputs(result, output_path, audit_path, report_path)
                print(f"stage4 final seed={seed}: failed; see report/results for traceback")
                break
            finally:
                _clear_cuda()
        if not seed_completed and bool(config.get("stop_on_failed_seed", False)):
            break
    _write_outputs(result, output_path, audit_path, report_path)
    return result


def _run_seed(stage: StageConfig, seed: int, device: str, hardware: Dict[str, object], config: Dict[str, object]) -> Dict[str, object]:
    _clear_cuda()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    dataset_config = _dataset_config_for_stage(config, stage)
    print(f"stage4 final seed={seed}: building fixed schema-aware splits")
    splits = build_multiview_code_patch_splits(dataset_config, seed=seed, repo_root=Path("."))
    print(f"stage4 final seed={seed}: fitting baselines")
    baselines = _fit_baselines(stage, splits, seed=seed, device=device)
    split_leakage = split_leakage_audit(BENCHMARK, seed, splits)
    output_leakage = output_leakage_audit(BENCHMARK, seed, splits["train"] + splits["dev"] + splits["test"])
    candidate = _selected_candidate(stage)
    print(f"stage4 final seed={seed}: fitting selected {SELECTED_ARCHITECTURE}")
    fit = _fit_selected_methods(candidate, stage, splits, seed, device, config)
    metrics = _metrics(fit, baselines, splits, seed)
    checkpoint_paths = _save_checkpoints(seed, stage, candidate, fit, baselines, config)
    cuda_peak = int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") and torch.cuda.is_available() else 0
    row = {
        "stage": "stage4_final_candidate_token_direct_validation",
        "seed": int(seed),
        "candidate": SELECTED_ARCHITECTURE,
        "description": candidate.description,
        "architecture_config": _candidate_config(candidate),
        "dev_accuracy": metrics["dev"],
        "test_accuracy": metrics["test"],
        "dev_delta": metrics["dev"]["trainable"] - metrics["dev"]["frozen"],
        "test_delta": metrics["test"]["trainable"] - metrics["test"]["frozen"],
        "status": "completed",
        "completed_at_utc": _now(),
        "device": device,
        "cuda_device_name": hardware.get("cuda_device_name"),
        "cuda_total_memory": hardware.get("cuda_total_memory"),
        "cuda_max_memory_allocated": cuda_peak,
        "cuda_max_memory_allocated_gb": cuda_peak / float(1024**3),
        "stage_config": asdict(stage),
        "dataset_config": asdict(dataset_config),
        "dataset_summary": dataset_summary(splits),
        "split_leakage_audit": _compact_leakage(split_leakage),
        "split_leakage_audit_passes": _leakage_passes(split_leakage),
        "output_leakage_audit": output_leakage,
        "output_leakage_audit_passes": bool(output_leakage.get("passes", False)),
        "architecture_locked_after_medium_selection": True,
        "dataset_or_control_changes_after_medium_selection": "none",
        "schema_policy": "role-specific schema fields preserved as legitimate program-analysis evidence",
        "role_embedding_shuffle_primary_gate": False,
        "mixed_precision": stage.mixed_precision,
        "batch_size": stage.batch_size,
        "gradient_accumulation_steps": stage.gradient_accumulation_steps,
        "checkpoint_paths": checkpoint_paths,
        "trainable_audit": _audit_subset(fit["trainable"].audit),
        "frozen_audit": _audit_subset(fit["frozen"].audit),
        "randomized_audit": _audit_subset(fit["randomized"].audit),
        "readout_collapse": {
            "trainable": _readout_stats(fit["trainable"], splits["test"]),
            "frozen": _readout_stats(fit["frozen"], splits["test"]),
        },
        "per_family_accuracy": _per_family_compare(fit["trainable"], fit["frozen"], splits["test"], seed),
        "per_role_ablation": _per_role_ablation(fit["trainable"], splits["test"], seed),
    }
    row["success_criteria"] = _seed_criteria(row, config)
    row["failure_analysis"] = _seed_failure_analysis(row)
    return row


def _fit_selected_methods(candidate, stage: StageConfig, splits: Dict[str, Sequence[MultiViewTaskExample]], seed: int, device: str, config: Dict[str, object]) -> Dict[str, object]:
    agent_config = _agent_config(stage)
    training = replace(
        _training_config(stage),
        epochs=50,
        patience=6,
        lr=0.0003,
        weight_decay=0.0001,
        gradient_clip_norm=1.0,
    )
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
        method=f"stage4_final_trainable__{SELECTED_ARCHITECTURE}",
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
        method=f"stage4_final_frozen__{SELECTED_ARCHITECTURE}",
        message_config=candidate.message_config,
    )
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
        method=f"stage4_final_randomized_labels__{SELECTED_ARCHITECTURE}",
        condition="randomized_labels",
        train_labels=randomized_labels_for_examples(splits["train"], seed + 40_401, 8),
        dev_labels=randomized_labels_for_examples(splits["dev"], seed + 50_501, 8),
        message_config=candidate.message_config,
    )
    positive = CompatibilityFeatureScorer(training=_positive_training(config), seed=seed + 14_000)
    positive.fit(splits["train"], splits["dev"])
    return {"trainable": trainable, "frozen": frozen, "randomized": randomized, "candidate_pair_compatibility_mlp": positive}


def _metrics(fit: Dict[str, object], baselines: Dict[str, object], splits: Dict[str, Sequence[MultiViewTaskExample]], seed: int) -> Dict[str, Dict[str, float]]:
    out = {}
    for split_name in ("dev", "test"):
        examples = list(splits[split_name])
        split_seed = seed + (1_000 if split_name == "dev" else 2_000)
        y = _labels(examples)
        values = {
            "trainable": _accuracy(predict_latent_system(fit["trainable"], examples, "none", split_seed), y),
            "frozen": _accuracy(predict_latent_system(fit["frozen"], examples, "none", split_seed), y),
            "text_only": _accuracy(baselines["text"].predict(examples), y),
            "raw_latent": _accuracy(predict_latent_system(baselines["raw"], examples, "none", split_seed), y),
            "majority_baseline": float(np.mean(np.full(len(y), int(baselines["majority_prediction"]), dtype=np.int64) == y)),
            "candidate_order_baseline": float(np.mean(np.full(len(y), int(baselines["candidate_order_prediction"]), dtype=np.int64) == y)),
            "single_agent_full_context": _accuracy(predict_context_baseline(baselines["full_context"], examples), y),
            "candidate_pair_compatibility_mlp": _accuracy(fit["candidate_pair_compatibility_mlp"].predict(examples), y),
            "all_role_oracle": _accuracy(_structured_oracle_predictions(examples, [0, 1, 2, 3]), y),
            "explicit_evidence_oracle": _accuracy(_structured_oracle_predictions(examples, [0, 1, 2, 3]), y),
            "randomized_labels": _accuracy(predict_latent_system(fit["randomized"], examples, "none", split_seed), y),
            "hidden_states_shuffled_across_examples": _accuracy(
                predict_latent_system(fit["trainable"], examples, "hidden_states_shuffled_across_examples", split_seed), y
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
        out[split_name] = values
    return out


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


def _save_checkpoints(seed: int, stage: StageConfig, candidate, fit: Dict[str, object], baselines: Dict[str, object], config: Dict[str, object]) -> Dict[str, str]:
    if not bool(config.get("save_checkpoints", True)):
        return {}
    root = Path(str(config["checkpoint_dir"])) / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "trainable": root / "trainable_candidate_token_direct_lr3e4_clip1.pt",
        "frozen": root / "frozen_candidate_token_direct_lr3e4_clip1.pt",
        "randomized_labels": root / "randomized_labels_candidate_token_direct_lr3e4_clip1.pt",
        "raw_latent": root / "raw_latent_baseline.pt",
        "text_only": root / "text_only_multi_agent_baseline.pt",
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
    _save_latent_checkpoint(paths["randomized_labels"], seed, stage, f"randomized_labels__{candidate.name}", _candidate_config(candidate), fit["randomized"])
    _save_latent_checkpoint(
        paths["raw_latent"],
        seed,
        stage,
        "raw_latent_baseline",
        {
            "description": "Raw pooled latent baseline.",
            "message_config": asdict(baselines["raw"].system.message_config),
            "coordinator_config": asdict(_coordinator_config(stage, family="cross_attention", num_layers=1, dropout=0.0)),
        },
        baselines["raw"],
    )
    _save_text_checkpoint(paths["text_only"], seed, stage, baselines["text"])
    _save_fixed_checkpoint(paths["majority_baseline"], seed, stage, "majority_baseline", int(baselines["majority_prediction"]))
    _save_fixed_checkpoint(paths["candidate_order_baseline"], seed, stage, "candidate_order_baseline", int(baselines["candidate_order_prediction"]))
    _save_context_checkpoint(paths["single_agent_full_context"], seed, stage, baselines["full_context"])
    for method, baseline in dict(baselines.get("single_view", {})).items():
        if method in paths:
            _save_bow_checkpoint(paths[method], seed, stage, baseline)
    _save_candidate_pair_checkpoint(paths["candidate_pair_compatibility_mlp"], seed, stage, fit["candidate_pair_compatibility_mlp"])
    _save_oracle_checkpoint(paths["explicit_evidence_oracle"], seed, stage)
    return {name: str(path) for name, path in paths.items()}


def _overall_criteria(result: Dict[str, object]) -> List[Dict[str, object]]:
    rows = [row for row in result.get("stage4_final_rows", []) if row.get("status") == "completed"]
    if not rows:
        return [{"criterion": "completed seeds >= 10", "pass": False, "value": 0}]
    config = result.get("config", {})
    near = float(config.get("near_chance_max", 0.18))
    inv_tol = float(config.get("invariance_tolerance", 0.08))
    _attach_pre_run_candidate_metadata(rows, result)
    deltas = [float(row["test_delta"]) for row in rows]
    ci_low, ci_high = _bootstrap_ci(deltas)
    trainable_mean = _mean([float(row["test_accuracy"]["trainable"]) for row in rows])
    wins = sum(delta > 0.0 for delta in deltas)
    baseline_names = ("text_only", "raw_latent", "majority_baseline", "candidate_order_baseline", "single_agent_full_context")
    baseline_means = {name: _mean([float(row["test_accuracy"].get(name, 0.0)) for row in rows]) for name in baseline_names}
    single_view_means = {
        f"single_view_text_role_{index}": _mean([float(row["test_accuracy"].get(f"single_view_text_role_{index}", 0.0)) for row in rows])
        for index in range(4)
    }
    control_means = {name: _mean([float(row["test_accuracy"].get(name, 1.0)) for row in rows]) for name in PRIMARY_CONTROLS}
    invariance = {name: _mean([float(row["test_accuracy"].get(name, 0.0)) for row in rows]) for name in INVARIANCE_CONTROLS}
    return [
        {"criterion": "completed seeds >= 10", "pass": len(rows) >= 10 and not result.get("failed_or_interrupted_seeds"), "value": len(rows)},
        {"criterion": "trainable beats frozen on at least 8/10 seeds", "pass": len(rows) >= 10 and wins >= 8, "value": wins},
        {"criterion": "mean trainable-frozen delta >= +0.20", "pass": _mean(deltas) >= 0.20, "value": _mean(deltas)},
        {"criterion": "bootstrap 95% CI lower bound for delta > +0.05", "pass": ci_low > 0.05, "value": [ci_low, ci_high]},
        {
            "criterion": "trainable beats text-only, raw-latent, majority, candidate-order, and full-context baselines by mean accuracy",
            "pass": all(trainable_mean > value for value in baseline_means.values()),
            "value": {"trainable_mean": trainable_mean, **baseline_means},
        },
        {
            "criterion": "trainable beats every single-view baseline by mean accuracy",
            "pass": all(trainable_mean > value for value in single_view_means.values()),
            "value": single_view_means,
        },
        {"criterion": "schema-aware corruption controls collapse near chance", "pass": all(value <= near for value in control_means.values()), "value": control_means},
        {
            "criterion": "invariance controls preserve accuracy within tolerance",
            "pass": all(abs(value - trainable_mean) <= inv_tol for value in invariance.values()),
            "value": {name: {"mean_accuracy": value, "mean_delta_from_trainable": value - trainable_mean} for name, value in invariance.items()},
        },
        {"criterion": "pre-run shortcut diagnostics pass", "pass": bool(result.get("pre_run", {}).get("summary", {}).get("passes", False)), "value": result.get("pre_run", {}).get("summary", {}).get("gates", {})},
        {
            "criterion": "leakage audits pass",
            "pass": all(
                bool(row.get("split_leakage_audit_passes"))
                and bool(row.get("output_leakage_audit_passes"))
                and bool(row.get("split_leakage_audit", {}).get("candidate_patch_hash_leakage_audit_passes", False))
                for row in rows
            ),
            "value": "split_candidate_patch_hash_and_output",
        },
        {
            "criterion": "trainable shared model receives gradients and changes",
            "pass": all(
                float(row.get("trainable_audit", {}).get("agent_grad_norm_mean") or 0.0) > 0.0
                and float(row.get("trainable_audit", {}).get("agent_parameter_delta") or 0.0) > 0.0
                for row in rows
            ),
            "value": "all_completed_seeds",
        },
        {
            "criterion": "frozen comparator remains frozen",
            "pass": all(
                _audit_float(row.get("frozen_audit", {}), "agent_grad_norm_mean", 1.0) == 0.0
                and _audit_float(row.get("frozen_audit", {}), "agent_parameter_delta", 1.0) == 0.0
                for row in rows
            ),
            "value": "all_completed_seeds",
        },
        {
            "criterion": "checkpoints saved for every seed and required method",
            "pass": all(
                all(row.get("checkpoint_paths", {}).get(name) and Path(str(row.get("checkpoint_paths", {}).get(name))).exists() for name in _required_checkpoint_names())
                for row in rows
            ),
            "value": list(_required_checkpoint_names()),
        },
    ]


def _seed_criteria(row: Dict[str, object], config: Dict[str, object]) -> Dict[str, object]:
    test = row["test_accuracy"]
    near = float(config.get("near_chance_max", 0.18))
    inv_tol = float(config.get("invariance_tolerance", 0.08))
    return {
        "trainable_beats_frozen": float(test["trainable"]) > float(test["frozen"]),
        "primary_controls_near_chance_without_candidate_metadata": {name: float(test.get(name, 1.0)) <= near for name in PRIMARY_CONTROLS if name != "candidate_metadata_only"},
        "invariance_controls_preserve_accuracy": {
            name: abs(float(test.get(name, 0.0)) - float(test["trainable"])) <= inv_tol for name in INVARIANCE_CONTROLS
        },
        "role_embedding_shuffle_primary_gate": False,
        "trainable_agent_changed": float(row.get("trainable_audit", {}).get("agent_parameter_delta") or 0.0) > 0.0,
        "frozen_agent_unchanged": _audit_float(row.get("frozen_audit", {}), "agent_parameter_delta", 1.0) == 0.0,
    }


def _summary(result: Dict[str, object]) -> Dict[str, object]:
    rows = [row for row in result.get("stage4_final_rows", []) if row.get("status") == "completed"]
    _attach_pre_run_candidate_metadata(rows, result)
    deltas = [float(row["test_delta"]) for row in rows]
    trainable = [float(row["test_accuracy"]["trainable"]) for row in rows]
    frozen = [float(row["test_accuracy"]["frozen"]) for row in rows]
    ci_low, ci_high = _bootstrap_ci(deltas) if deltas else (0.0, 0.0)
    criteria = _overall_criteria(result)
    return {
        "completed_seeds": [int(row["seed"]) for row in rows],
        "n_completed": len(rows),
        "n_failed_or_interrupted": len(result.get("failed_or_interrupted_seeds", [])),
        "mean_trainable_accuracy": _mean(trainable),
        "mean_frozen_accuracy": _mean(frozen),
        "mean_delta": _mean(deltas),
        "std_delta": pstdev(deltas) if len(deltas) > 1 else 0.0,
        "bootstrap_95_ci_delta": [ci_low, ci_high],
        "passed_stage4_final_validation_gates": all(bool(item["pass"]) for item in criteria) if criteria else False,
        "success_criteria": criteria,
    }


def _write_outputs(result: Dict[str, object], output_path: Path, audit_path: Path, report_path: Path) -> None:
    for row in result.get("stage4_final_rows", []):
        if row.get("status") == "completed":
            row["success_criteria"] = _seed_criteria(row, result.get("config", {}))
    result["summary"] = _summary(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(output_path)
    audit_rows = _audit_rows(result)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in audit_rows) + ("\n" if audit_rows else ""), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(result), encoding="utf-8")


def _render_report(result: Dict[str, object]) -> str:
    rows = [row for row in result.get("stage4_final_rows", []) if row.get("status") == "completed"]
    _attach_pre_run_candidate_metadata(rows, result)
    config = result.get("config", {})
    stage = config.get("stage", {})
    selected = config.get("selected_variant", {})
    summary = result.get("summary", {})
    pre = result.get("pre_run", {})
    lines = [
        "# Stage 4 Final Candidate-Token Direct Validation",
        "",
        "## Exact Selected Config",
        "",
        f"- Selected architecture: `{SELECTED_ARCHITECTURE}`",
        "- Candidate-conditioned readout: candidate queries attend directly over role token states.",
        f"- LR: `{selected.get('lr')}`",
        f"- Gradient clipping: `{selected.get('gradient_clip_norm')}`",
        f"- Epochs/patience: `{selected.get('epochs')}` / `{selected.get('patience')}`",
        "- Frozen comparator: exact same architecture and training budget; only shared-agent/model weights are frozen.",
        f"- Seeds: `{stage.get('seeds')}`",
        f"- Split sizes: train `{stage.get('n_train')}`, dev `{stage.get('n_dev')}`, test `{stage.get('n_test')}`.",
        "- CUDA required; bf16; batch size 32 with fallback [32,16,8].",
        "",
        "## Lock Statement",
        "",
        "- No architecture, dataset, control, or hyperparameter changes were made after medium selection.",
        "- This final run uses the fixed Stage 3.8 schema-aware benchmark and fresh seeds `[10..19]`.",
        "- `role_embedding_shuffle` remains diagnostic-only and non-gating.",
        "",
        *_correction_report_lines(result),
        "",
        "## Pre-Run Shortcut Diagnostics",
        "",
        "| gate | pass | value | threshold |",
        "|---|---|---:|---:|",
    ]
    diag = pre.get("shortcut_diagnostics", {}).get("summary", {})
    schema = pre.get("schema_dataset_baselines", {}).get("summary", {})
    near = float(config.get("near_chance_max", 0.18))
    static_max = float(config.get("static_frequency_max", 0.20))
    pre_rows = [
        ("candidate_only", diag.get("mean_candidate_only_accuracy", 0.0), near, "max"),
        ("candidate_metadata_only", diag.get("mean_candidate_metadata_only_accuracy", 0.0), near, "max"),
        ("view_masked_candidates_visible", diag.get("mean_view_masked_candidates_visible_accuracy", 0.0), near, "max"),
        ("role_pair_only", diag.get("mean_role_pair_only_accuracy", 0.0), near, "max"),
        ("lexical_overlap", diag.get("mean_lexical_overlap_accuracy", 0.0), near, "max"),
        ("static_frequency", diag.get("mean_static_frequency_accuracy", 0.0), static_max, "max"),
        ("schema_only_baseline", schema.get("schema_only_baseline", {}).get("mean_test_accuracy", 0.0), near, "max"),
        ("null_evidence_values_baseline", schema.get("null_evidence_values_baseline", {}).get("mean_test_accuracy", 0.0), near, "max"),
        ("evidence_only_no_candidates_baseline", schema.get("evidence_only_no_candidates_baseline", {}).get("mean_test_accuracy", 0.0), near, "max"),
        ("all_role_oracle", diag.get("mean_all_role_structured_oracle_accuracy", 0.0), float(config.get("oracle_min", 0.90)), "min"),
    ]
    for name, value, threshold, kind in pre_rows:
        passed = value >= threshold if kind == "min" else value <= threshold
        lines.append(f"| {name} | `{bool(passed)}` | {float(value):.4f} | {float(threshold):.4f} |")
    lines.extend(["", "## Per-Seed Accuracy", "", "| seed | batch | max CUDA GB | trainable | frozen | delta | text | raw | majority | cand-order | max single-view | full ctx | candidate-pair | oracle |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        test = row["test_accuracy"]
        max_single = max(float(test.get(f"single_view_text_role_{index}", 0.0)) for index in range(4))
        lines.append(
            "| {seed} | {batch} | {mem:.3f} | {train:.4f} | {frozen:.4f} | {delta:.4f} | {text:.4f} | {raw:.4f} | {maj:.4f} | {corder:.4f} | {single:.4f} | {full:.4f} | {pair:.4f} | {oracle:.4f} |".format(
                seed=row["seed"],
                batch=row.get("batch_size"),
                mem=float(row.get("cuda_max_memory_allocated_gb", 0.0)),
                train=float(test["trainable"]),
                frozen=float(test["frozen"]),
                delta=float(row["test_delta"]),
                text=float(test["text_only"]),
                raw=float(test["raw_latent"]),
                maj=float(test["majority_baseline"]),
                corder=float(test["candidate_order_baseline"]),
                single=max_single,
                full=float(test["single_agent_full_context"]),
                pair=float(test["candidate_pair_compatibility_mlp"]),
                oracle=float(test["all_role_oracle"]),
            )
        )
    ci = summary.get("bootstrap_95_ci_delta", [0.0, 0.0])
    lines.extend(
        [
            "",
            "## Mean/Std/Bootstrap CI",
            "",
            f"- Completed seeds: `{summary.get('completed_seeds', [])}`",
            f"- Mean trainable accuracy: `{float(summary.get('mean_trainable_accuracy', 0.0)):.4f}`",
            f"- Mean frozen accuracy: `{float(summary.get('mean_frozen_accuracy', 0.0)):.4f}`",
            f"- Mean trainable-frozen delta: `{float(summary.get('mean_delta', 0.0)):.4f}`",
            f"- Delta std: `{float(summary.get('std_delta', 0.0)):.4f}`",
            f"- Bootstrap 95% CI for delta: `[{float(ci[0]):.4f}, {float(ci[1]):.4f}]`",
            "",
            "## Schema-Aware Control Table",
            "",
            "| control | mean | std | max | pass |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for name in PRIMARY_CONTROLS:
        values = [float(row["test_accuracy"].get(name, 0.0)) for row in rows]
        lines.append(f"| {name} | {_mean(values):.4f} | {_std(values):.4f} | {(max(values) if values else 0.0):.4f} | `{bool(_mean(values) <= near)}` |")
    trainable_mean = _mean([float(row["test_accuracy"]["trainable"]) for row in rows])
    inv_tol = float(config.get("invariance_tolerance", 0.08))
    lines.extend(["", "## Invariance Control Table", "", "| control | mean | delta from trainable | tolerance | pass |", "|---|---:|---:|---:|---|"])
    for name in INVARIANCE_CONTROLS:
        values = [float(row["test_accuracy"].get(name, 0.0)) for row in rows]
        delta = _mean(values) - trainable_mean
        lines.append(f"| {name} | {_mean(values):.4f} | {delta:.4f} | {inv_tol:.4f} | `{bool(abs(delta) <= inv_tol)}` |")
    lines.extend(["", "## Role-Embedding Shuffle Diagnostic", "", "| seed | trainable | diagnostic | delta | gating? |", "|---:|---:|---:|---:|---|"])
    for row in rows:
        test = row["test_accuracy"]
        diagnostic = float(test.get("role_embedding_shuffle_diagnostic", 0.0))
        trainable = float(test["trainable"])
        lines.append(f"| {row['seed']} | {trainable:.4f} | {diagnostic:.4f} | {diagnostic - trainable:.4f} | `False` |")
    lines.extend(["", "## Frozen Comparator Audit", "", "| seed | frozen shared grad | frozen shared delta | shared identity |", "|---:|---:|---:|---|"])
    for row in rows:
        audit = row.get("frozen_audit", {})
        lines.append(f"| {row['seed']} | {float(audit.get('agent_grad_norm_mean') or 0.0):.4f} | {float(audit.get('agent_parameter_delta') or 0.0):.4f} | `{bool(audit.get('shared_parameter_identity'))}` |")
    lines.extend(["", "## Trainable Update Audit", "", "| seed | shared grad | shared delta | coordinator grad | coordinator delta | token variance | token cosine |", "|---:|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        audit = row.get("trainable_audit", {})
        readout = row.get("readout_collapse", {}).get("trainable", {})
        lines.append(
            f"| {row['seed']} | {float(audit.get('agent_grad_norm_mean') or 0.0):.4f} | {float(audit.get('agent_parameter_delta') or 0.0):.4f} | {float(audit.get('coordinator_grad_norm_mean') or 0.0):.4f} | {float(audit.get('coordinator_parameter_delta') or 0.0):.4f} | {float(readout.get('token_variance_mean', 0.0)):.4f} | {float(readout.get('token_mean_pairwise_cosine', 0.0)):.4f} |"
        )
    lines.extend(["", "## Per-Family Accuracy", ""])
    for row in rows:
        lines.append(f"### Seed {row['seed']}")
        lines.append("| family | n | trainable | frozen | delta |")
        lines.append("|---|---:|---:|---:|---:|")
        for family, fam in row.get("per_family_accuracy", {}).items():
            lines.append(f"| {family} | {int(fam['n'])} | {float(fam['trainable']):.4f} | {float(fam['frozen']):.4f} | {float(fam['delta']):.4f} |")
        lines.append("")
    lines.extend(["## Acceptance Gates", "", "| criterion | pass | value |", "|---|---|---|"])
    for item in summary.get("success_criteria", []):
        lines.append(f"| {item['criterion']} | `{bool(item['pass'])}` | `{json.dumps(item['value'], sort_keys=True)}` |")
    lines.extend(_legacy_physical_order_appendix_lines(result))
    lines.extend(
        [
            "",
            "## Failed Seeds/OOMs",
            "",
            f"- Failed/interrupted: `{result.get('failed_or_interrupted_seeds', [])}`",
            f"- OOM retries: `{result.get('oom_retries', [])}`",
            "",
            "## Conservative Interpretation",
            "",
            f"- Final Stage 4 gates passed: `{bool(summary.get('passed_stage4_final_validation_gates', False))}`",
            *_conservative_claim_lines(summary),
        ]
    )
    return "\n".join(lines) + "\n"


def _correction_report_lines(result: Dict[str, object]) -> List[str]:
    correction = result.get("physical_order_invariance_correction")
    if not isinstance(correction, dict) or not correction:
        return []
    return [
        "## Correction: physical-order invariance evaluator bug",
        "",
        f"- Legacy buggy physical-order mean: `{float(correction.get('legacy_buggy_physical_order_mean', 0.0)):.4f}`.",
        f"- Corrected all-24-permutation mean: `{float(correction.get('corrected_all_24_permutation_mean', 0.0)):.4f}`.",
        f"- Normal mean: `{float(correction.get('normal_mean', 0.0)):.4f}`.",
        f"- Canonicalized mean: `{float(correction.get('canonicalized_mean', 0.0)):.4f}`.",
        "- Bug: the legacy physical-order evaluator permuted `role_ids` and `clone_activations` but did not permute `token_states` or `token_mask`, which are the tensors consumed by the candidate-token-direct coordinator.",
        "- Correction: all role-axis tensors are now permuted together: pooled/message/evidence readouts, `role_ids`, `token_states`, and `token_mask`.",
        "- No training, architecture, dataset, control, or hyperparameter changes were made. The correction uses the exact saved final Stage 4 checkpoints for seeds `[10..19]`.",
    ]


def _legacy_physical_order_appendix_lines(result: Dict[str, object]) -> List[str]:
    correction = result.get("physical_order_invariance_correction")
    if not isinstance(correction, dict) or not correction:
        return []
    rows = list(correction.get("seed_rows", []))
    lines = [
        "",
        "## Appendix: Legacy Failed Physical-Order Result",
        "",
        "- These old failed-control values are retained for transparency. They are not used as the corrected gate result.",
        f"- Legacy final single-permutation mean: `{float(correction.get('legacy_final_single_permutation_mean', 0.0)):.4f}`.",
        f"- Legacy all-24 buggy reproduction mean: `{float(correction.get('legacy_buggy_physical_order_mean', 0.0)):.4f}`.",
        "",
        "| seed | old final physical | legacy all-24 bug mean | corrected all-24 mean | normal | canonicalized |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {seed} | {old:.4f} | {legacy:.4f} | {corrected:.4f} | {normal:.4f} | {canonical:.4f} |".format(
                seed=int(row.get("seed", 0)),
                old=float(row.get("legacy_final_single_permutation_accuracy", 0.0)),
                legacy=float(row.get("legacy_buggy_all_24_mean", 0.0)),
                corrected=float(row.get("corrected_all_24_permutation_accuracy", 0.0)),
                normal=float(row.get("normal_accuracy", 0.0)),
                canonical=float(row.get("canonicalized_accuracy", 0.0)),
            )
        )
    return lines


def _conservative_claim_lines(summary: Dict[str, object]) -> List[str]:
    if bool(summary.get("passed_stage4_final_validation_gates", False)):
        return [
            "- Conservative claim: On the clean Stage 3.8 schema-aware real-code import-restoration benchmark, the candidate-token-direct shared-weight latent coordination architecture substantially outperforms an exact frozen same-architecture comparator and text/raw/single-view baselines across fresh seeds, while passing shortcut, corruption, invariance, leakage, and parameter-update audits.",
            "- This does not claim open-ended code repair, SWE-bench success, general coding-agent success, or human-level coding.",
        ]
    return [
        "- If any gate fails, no tuning is performed in this run and no success claim is made.",
        "- Passing this benchmark would support only the bounded schema-aware import-restoration claim, not open-ended code repair or general coding-agent success.",
    ]


def _selected_candidate(stage: StageConfig):
    locked = next(candidate for candidate in _candidate_specs() if candidate.name == "topk_attention_no_head")
    msg = replace(
        locked.message_config,
        readout_source="pooled",
        use_message_head=False,
        use_private_cue_aux=False,
        aux_loss_weight=0.0,
        coordinator_family="candidate_token_cross_attention",
    )
    coord = replace(
        _coordinator_config(stage, family="candidate_token_cross_attention", num_layers=1, dropout=0.0),
        family="candidate_token_cross_attention",
    )
    return SimpleNamespace(
        name=SELECTED_ARCHITECTURE,
        description="Candidate queries attend directly over role token states; LR 3e-4; gradient clipping 1.0.",
        message_config=msg,
        coordinator_config=coord,
    )


def _readout_stats(result, examples: Sequence[MultiViewTaskExample]) -> Dict[str, float]:
    pooled = []
    tokens = []
    with torch.no_grad():
        for start in range(0, len(examples), 128):
            batch = list(examples[start : start + 128])
            readouts = result.system.collect_clone_representations(batch, condition="none", seed=71_000)
            pooled.append(readouts["message"].detach().float().cpu())
            tokens.append(readouts["token_states"].detach().float().cpu())
    pooled_t = torch.cat(pooled, dim=0)
    token_t = torch.cat(tokens, dim=0)
    return {
        "pooled_variance_mean": _variance_mean(pooled_t),
        "pooled_mean_pairwise_cosine": _mean_pairwise_cosine(pooled_t),
        "pooled_norm_mean": float(pooled_t.norm(dim=-1).mean().item()),
        "token_variance_mean": _variance_mean(token_t),
        "token_mean_pairwise_cosine": _mean_pairwise_cosine(token_t.mean(dim=2)),
        "token_norm_mean": float(token_t.norm(dim=-1).mean().item()),
    }


def _per_family_compare(trainable, frozen, examples: Sequence[MultiViewTaskExample], seed: int) -> Dict[str, Dict[str, float]]:
    train_pred = predict_latent_system(trainable, examples, "none", seed + 61_000)
    frozen_pred = predict_latent_system(frozen, examples, "none", seed + 61_000)
    labels = _labels(examples)
    grouped: Dict[str, List[int]] = {}
    from src.datasets.multiview_code_patch_selection import example_oracle_metadata

    for index, example in enumerate(examples):
        grouped.setdefault(str(example_oracle_metadata(example).get("problem_family")), []).append(index)
    out = {}
    for family, indices in sorted(grouped.items()):
        idx = np.asarray(indices, dtype=np.int64)
        train_acc = float(np.mean(train_pred[idx] == labels[idx]))
        frozen_acc = float(np.mean(frozen_pred[idx] == labels[idx]))
        out[family] = {"n": int(len(indices)), "trainable": train_acc, "frozen": frozen_acc, "delta": train_acc - frozen_acc}
    return out


def _seed_failure_analysis(row: Dict[str, object]) -> Dict[str, object]:
    test = row["test_accuracy"]
    delta = float(row["test_delta"])
    controls_bad = [name for name in PRIMARY_CONTROLS if float(test.get(name, 0.0)) > 0.18]
    return {
        "failed_seed": bool(delta <= 0.0 or controls_bad),
        "likely_failure_type": (
            "control_failure" if controls_bad else "frozen_strength_or_seed_instability" if delta <= 0.0 else "none"
        ),
        "controls_above_threshold": controls_bad,
    }


def _attach_pre_run_candidate_metadata(rows: Sequence[Dict[str, object]], result: Dict[str, object]) -> None:
    seed_values = {}
    for row in result.get("pre_run", {}).get("shortcut_diagnostics", {}).get("seed_rows", []):
        baselines = row.get("baselines", {})
        seed_values[int(row.get("seed", -1))] = float(baselines.get("candidate_metadata_only_accuracy", 1.0))
    for row in rows:
        value = seed_values.get(int(row.get("seed", -1)))
        if value is not None:
            row.setdefault("test_accuracy", {})["candidate_metadata_only"] = value


def _audit_rows(result: Dict[str, object]) -> List[Dict[str, object]]:
    rows = [{"event": "pre_run_summary", "time_utc": _now(), **dict(result.get("pre_run", {}).get("summary", {}))}]
    correction = result.get("physical_order_invariance_correction")
    if isinstance(correction, dict) and correction:
        rows.append(
            {
                "event": "physical_order_invariance_evaluator_correction",
                "stage": "stage4_final_candidate_token_direct_validation",
                "time_utc": correction.get("applied_at_utc", _now()),
                "legacy_buggy_physical_order_mean": correction.get("legacy_buggy_physical_order_mean"),
                "corrected_all_24_permutation_mean": correction.get("corrected_all_24_permutation_mean"),
                "normal_mean": correction.get("normal_mean"),
                "canonicalized_mean": correction.get("canonicalized_mean"),
                "legacy_final_single_permutation_mean": correction.get("legacy_final_single_permutation_mean"),
                "no_training_architecture_dataset_or_hyperparameter_changes": correction.get("no_training_architecture_dataset_or_hyperparameter_changes"),
                "explanation": correction.get("explanation"),
            }
        )
    for row in result.get("stage4_final_rows", []):
        rows.append(
            {
                "event": "seed_completed",
                "stage": "stage4_final_candidate_token_direct_validation",
                "seed": int(row["seed"]),
                "test_accuracy": row.get("test_accuracy", {}),
                "physical_order_invariance_correction": row.get("physical_order_invariance_correction", {}),
                "test_delta": row.get("test_delta"),
                "split_leakage_audit_passes": row.get("split_leakage_audit_passes"),
                "output_leakage_audit_passes": row.get("output_leakage_audit_passes"),
                "trainable_audit": row.get("trainable_audit", {}),
                "frozen_audit": row.get("frozen_audit", {}),
                "readout_collapse": row.get("readout_collapse", {}),
                "per_family_accuracy": row.get("per_family_accuracy", {}),
                "failure_analysis": row.get("failure_analysis", {}),
                "checkpoint_paths": row.get("checkpoint_paths", {}),
                "time_utc": row.get("completed_at_utc", _now()),
            }
        )
    rows.extend(result.get("oom_retries", []))
    rows.extend(result.get("failed_or_interrupted_seeds", []))
    return rows


def _row_has_required_checkpoints(row: Dict[str, object], config: Dict[str, object]) -> bool:
    if not bool(config.get("save_checkpoints", True)):
        return True
    summary = row.get("dataset_summary", {})
    if summary.get("dataset_source") != BALANCED_34B_DATASET_SOURCE:
        return False
    if summary.get("candidate_representation") != BALANCED_34B_CANDIDATE_REPRESENTATION:
        return False
    paths = row.get("checkpoint_paths", {})
    return isinstance(paths, dict) and all(paths.get(name) and Path(str(paths[name])).exists() for name in _required_checkpoint_names())


def _required_checkpoint_names() -> tuple[str, ...]:
    return (
        "trainable",
        "frozen",
        "randomized_labels",
        "text_only",
        "raw_latent",
        "majority_baseline",
        "candidate_order_baseline",
        "single_agent_full_context",
        "single_view_text_role_0",
        "single_view_text_role_1",
        "single_view_text_role_2",
        "single_view_text_role_3",
        "candidate_pair_compatibility_mlp",
        "explicit_evidence_oracle",
    )


def _final_config(config: Dict[str, object]) -> Dict[str, object]:
    out = _stage38_full_config(config)
    out["architecture"] = SELECTED_ARCHITECTURE
    out["architecture_changes"] = "forbidden_after_medium_selection"
    out["stage"] = {
        **dict(out["stage"]),
        "name": "stage4_final_candidate_token_direct_validation",
        "n_train": 2048,
        "n_dev": 512,
        "n_test": 1024,
        "seeds": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        "epochs": 50,
        "patience": 6,
        "batch_size": 32,
        "lr": 0.0003,
        "mixed_precision": "bf16",
    }
    out["dataset_config"] = {
        **dict(out.get("dataset_config", {})),
        "dataset_source": BALANCED_34B_DATASET_SOURCE,
        "generator_version": "real_import_restore_candidate_balanced_34b_stage4_final_candidate_token_direct",
        "candidate_representation": BALANCED_34B_CANDIDATE_REPRESENTATION,
    }
    return out


def _metadata(config: Dict[str, object], config_path: Path, hardware: Dict[str, object], device: str) -> Dict[str, object]:
    return {
        "benchmark": BENCHMARK,
        "task": "stage4_final_candidate_token_direct_validation",
        "architecture": SELECTED_ARCHITECTURE,
        "architecture_selection": "selected by Stage 4 medium validation before final run",
        "architecture_changes_after_medium_selection": "none",
        "dataset": BALANCED_34B_DATASET_SOURCE,
        "candidate_representation": BALANCED_34B_CANDIDATE_REPRESENTATION,
        "schema_policy": "role-specific schema fields preserved as legitimate program-analysis evidence",
        "role_embedding_shuffle_primary_gate": False,
        "config_path": str(config_path),
        "device": device,
        "created_or_updated_at_utc": _now(),
        "hardware": hardware,
        "success_claim": "not evaluated until all final Stage 4 gates pass",
    }


def _variance_mean(tensor: torch.Tensor) -> float:
    flat = tensor.reshape(tensor.shape[0], -1)
    return float(flat.var(dim=0, unbiased=False).mean().item())


def _mean_pairwise_cosine(tensor: torch.Tensor) -> float:
    flat = tensor.reshape(tensor.shape[0], -1)
    if flat.shape[0] < 2:
        return 1.0
    flat = F.normalize(flat, dim=-1)
    sim = flat @ flat.t()
    mask = ~torch.eye(sim.shape[0], dtype=torch.bool)
    return float(sim[mask].mean().item())


def _mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
