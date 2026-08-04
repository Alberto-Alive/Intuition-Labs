from __future__ import annotations

import argparse
import json
import math
import traceback
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import pstdev
from typing import Dict, List, Sequence

import numpy as np
import torch

from src.coordinators.mlp import MLPTrainingConfig
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
    _per_family_accuracy,
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
    ARCHITECTURE,
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
from src.experiments.run_stage34_candidate_sanitization_diagnostics import (
    _accuracy,
    _structured_oracle_predictions,
    run_diagnostics,
    stage34_dataset_config,
)
from src.experiments.run_stage35_model_facing_learnability import CompatibilityFeatureScorer
from src.experiments.run_stage38_schema_aware_controls import (
    _candidate_evidence_mismatch,
    _candidate_only,
    _cross_example_view_bundle_shuffle,
    _evidence_only_no_candidates,
    _null_evidence_values,
    _schema_dataset_baselines,
    _schema_only,
    _schema_preserved_role_value_shuffle,
    _value_shuffle_within_schema,
    _zero_role_embeddings,
)


DEFAULT_CONFIG = "configs/stage38_full_hard_validation.json"
PRIMARY_CORRUPTION_CONTROLS = (
    "candidate_only",
    "view_masked_candidates_visible",
    "schema_only",
    "value_shuffle_within_schema",
    "cross_example_view_bundle_shuffle",
    "candidate_evidence_mismatch",
    "schema_preserved_role_value_shuffle",
    "null_evidence_values",
    "evidence_only_no_candidates",
    "randomized_labels",
    "hidden_states_shuffled_across_examples",
)
SCHEMA_AWARE_CORRUPTION_CONTROLS = (
    "candidate_only",
    "view_masked_candidates_visible",
    "schema_only",
    "value_shuffle_within_schema",
    "cross_example_view_bundle_shuffle",
    "candidate_evidence_mismatch",
    "schema_preserved_role_value_shuffle",
    "null_evidence_values",
    "evidence_only_no_candidates",
)
INVARIANCE_CONTROLS = (
    "physical_order_shuffled_roles_preserved",
    "candidate_order_shuffled_with_gold_remap",
)
DIAGNOSTIC_ONLY_CONTROLS = ("role_embedding_shuffle_diagnostic",)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 3.8 schema-aware full hard validation.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_stage38_full(config, config_path=config_path)


def run_stage38_full(config: Dict[str, object], config_path: Path) -> Dict[str, object]:
    config = _stage38_full_config(config)
    device, hardware = _configure_cuda(config)
    output_path = Path(str(config["output_path"]))
    audit_path = Path(str(config["audit_log_path"]))
    report_path = Path(str(config["report_path"]))
    result = _load_result(output_path)
    result.setdefault("metadata", _metadata(config, config_path, hardware, device))
    result["metadata"].update(_metadata(config, config_path, hardware, device))
    result["config"] = config
    result.setdefault("stage38_rows", [])
    result.setdefault("failed_or_interrupted_seeds", [])
    result.setdefault("oom_retries", [])

    print("stage38 full: running pre-run shortcut diagnostics")
    result["pre_run"] = _pre_run_checks(config, config_path, device)
    _write_outputs(result, output_path, audit_path, report_path)
    if bool(config.get("stop_on_preflight_failure", True)) and not bool(result["pre_run"]["summary"]["passes"]):
        print("stage38 full: pre-run checks failed; full seed validation was not started")
        return result

    result["stage38_rows"] = [
        row
        for row in result["stage38_rows"]
        if row.get("status") == "completed" and row.get("candidate") == ARCHITECTURE and _row_has_required_checkpoints(row, config)
    ]
    completed = {
        int(row["seed"])
        for row in result["stage38_rows"]
        if row.get("status") == "completed" and row.get("candidate") == ARCHITECTURE and _row_has_required_checkpoints(row, config)
    }
    stage_base = _stage_from_config(config)
    seeds = [int(value) for value in config["stage"]["seeds"]]
    candidate = next(candidate for candidate in _candidate_specs() if candidate.name == ARCHITECTURE)
    fallbacks = [int(value) for value in config.get("gpu_safety", {}).get("batch_size_fallbacks", [32, 16, 8])]
    effective_batch = int(config.get("gpu_safety", {}).get("effective_batch_size", fallbacks[0]))

    for seed in seeds:
        if seed in completed:
            print(f"stage38 seed={seed}: existing completed row found; skipping")
            continue
        result["stage38_rows"] = [row for row in result["stage38_rows"] if int(row.get("seed", -1)) != seed]
        print(f"stage38 seed={seed}: starting schema-aware hard validation")
        seed_completed = False
        for batch_size in fallbacks:
            accumulation = max(1, int(math.ceil(effective_batch / float(batch_size))))
            stage = replace(stage_base, seeds=(seed,), batch_size=batch_size, gradient_accumulation_steps=accumulation)
            try:
                row = _run_seed(stage, seed, candidate, device, hardware, config)
                result["stage38_rows"].append(row)
                completed.add(seed)
                seed_completed = True
                _write_outputs(result, output_path, audit_path, report_path)
                print(
                    "stage38 seed={seed}: completed trainable={trainable:.4f} frozen={frozen:.4f} "
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
                    print(f"stage38 seed={seed}: CUDA OOM at batch_size={batch_size}; retrying smaller batch")
                    continue
                failure = _failure_row(seed, batch_size, accumulation, "failed", exc)
                result["failed_or_interrupted_seeds"].append(failure)
                _clear_cuda()
                _write_outputs(result, output_path, audit_path, report_path)
                print(f"stage38 seed={seed}: failed; see report/results for traceback")
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
                print(f"stage38 seed={seed}: failed; see report/results for traceback")
                break
            finally:
                _clear_cuda()
        if not seed_completed and bool(config.get("stop_on_failed_seed", False)):
            break

    _write_outputs(result, output_path, audit_path, report_path)
    return result


def _run_seed(stage: StageConfig, seed: int, candidate, device: str, hardware: Dict[str, object], config: Dict[str, object]) -> Dict[str, object]:
    _clear_cuda()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    dataset_config = _dataset_config_for_stage(config, stage)
    print(f"stage38 seed={seed}: building schema-aware balanced_categories_v3 splits")
    splits = build_multiview_code_patch_splits(dataset_config, seed=seed, repo_root=Path("."))
    print(f"stage38 seed={seed}: fitting baselines")
    baselines = _fit_baselines(stage, splits, seed=seed, device=device)
    split_leakage = split_leakage_audit(BENCHMARK, seed, splits)
    output_leakage = output_leakage_audit(BENCHMARK, seed, splits["train"] + splits["dev"] + splits["test"])
    print(f"stage38 seed={seed}: fitting locked {ARCHITECTURE}")
    fit = _fit_stage38_methods(candidate, stage, splits, seed, device, config)
    metrics = _stage38_metrics(fit, baselines, splits, seed)
    checkpoint_paths = _save_stage38_checkpoints(seed, stage, candidate, fit, baselines, config)
    cuda_peak = int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") and torch.cuda.is_available() else 0
    row = {
        "stage": "stage38_full_hard_validation",
        "seed": int(seed),
        "candidate": candidate.name,
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
        "architecture_locked": True,
        "architecture_changes_forbidden": True,
        "schema_policy": "role-specific schema fields preserved as legitimate program-analysis evidence",
        "role_embedding_shuffle_primary_gate": False,
        "mixed_precision": stage.mixed_precision,
        "batch_size": stage.batch_size,
        "gradient_accumulation_steps": stage.gradient_accumulation_steps,
        "checkpoint_paths": checkpoint_paths,
        "trainable_audit": _audit_subset(fit["trainable"].audit),
        "frozen_audit": _audit_subset(fit["frozen"].audit),
        "randomized_audit": _audit_subset(fit["randomized"].audit),
        "per_family_accuracy": _per_family_accuracy(fit["trainable"], splits["test"], seed),
        "per_role_ablation": _per_role_ablation(fit["trainable"], splits["test"], seed),
    }
    row["success_criteria"] = _seed_criteria(row, config)
    return row


def _fit_stage38_methods(candidate, stage: StageConfig, splits: Dict[str, Sequence[MultiViewTaskExample]], seed: int, device: str, config: Dict[str, object]) -> Dict[str, object]:
    agent_config = _agent_config(stage)
    training = _training_config(stage)
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
    randomized = fit_latent_system(
        train_examples=splits["train"],
        dev_examples=splits["dev"],
        agent_config=agent_config,
        coordinator_config=candidate.coordinator_config,
        training_config=training,
        num_classes=8,
        seed=seed + 301,
        device=device,
        trainable_agent=True,
        method=f"stage38_randomized_labels__{candidate.name}",
        condition="randomized_labels",
        train_labels=randomized_labels_for_examples(splits["train"], seed + 401, 8),
        dev_labels=randomized_labels_for_examples(splits["dev"], seed + 501, 8),
        message_config=candidate.message_config,
    )
    positive = CompatibilityFeatureScorer(training=_positive_training(config), seed=seed + 14_000)
    positive.fit(splits["train"], splits["dev"])
    return {"trainable": trainable, "frozen": frozen, "randomized": randomized, "candidate_pair_compatibility_mlp": positive}


def _stage38_metrics(fit: Dict[str, object], baselines: Dict[str, object], splits: Dict[str, Sequence[MultiViewTaskExample]], seed: int) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
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
        controlled = _control_examples(examples, split_seed)
        for name, rows in controlled.items():
            cy = _labels(rows)
            values[name] = _predict_control_accuracy(fit["trainable"], rows, cy, name, split_seed)
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
        "value_shuffle_within_schema": _value_shuffle_within_schema(examples, seed + 12_000),
        "cross_example_view_bundle_shuffle": _cross_example_view_bundle_shuffle(examples, seed + 13_000),
        "candidate_evidence_mismatch": _candidate_evidence_mismatch(examples, seed + 14_000),
        "schema_preserved_role_value_shuffle": _schema_preserved_role_value_shuffle(examples, seed + 15_000),
        "null_evidence_values": _null_evidence_values(examples),
        "evidence_only_no_candidates": _evidence_only_no_candidates(examples),
    }


def _predict_control_accuracy(result, examples: Sequence[MultiViewTaskExample], labels: np.ndarray, control: str, seed: int) -> float:
    if control == "candidate_only":
        with _zero_role_embeddings(result.system):
            return _accuracy(predict_latent_system(result, examples, "none", seed), labels)
    return _accuracy(predict_latent_system(result, examples, "none", seed), labels)


def _save_stage38_checkpoints(seed: int, stage: StageConfig, candidate, fit: Dict[str, object], baselines: Dict[str, object], config: Dict[str, object]) -> Dict[str, str]:
    if not bool(config.get("save_checkpoints", True)):
        return {}
    root = _checkpoint_root(config) / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "trainable": root / "trainable_topk_attention_no_head.pt",
        "frozen": root / "frozen_topk_attention_no_head.pt",
        "randomized_labels": root / "randomized_labels_trainable_topk_attention_no_head.pt",
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


def _save_candidate_pair_checkpoint(path: Path, seed: int, stage: StageConfig, scorer: CompatibilityFeatureScorer) -> None:
    if scorer.model is None:
        raise RuntimeError("cannot save candidate_pair_compatibility_mlp before fit")
    payload = {
        "checkpoint_type": "candidate_pair_compatibility_mlp",
        "benchmark": BENCHMARK,
        "seed": int(seed),
        "stage_config": asdict(stage),
        "method": "candidate_pair_compatibility_mlp",
        "training": asdict(scorer.training),
        "input_dim": int(next(scorer.model.parameters()).shape[-1]),
        "hidden_dims": list(scorer.training.hidden_dims),
        "param_count": int(sum(parameter.numel() for parameter in scorer.model.parameters())),
        "model_state_dict": {key: value.detach().cpu() for key, value in scorer.model.state_dict().items()},
    }
    torch.save(payload, path)


def _pre_run_checks(config: Dict[str, object], config_path: Path, device: str) -> Dict[str, object]:
    diagnostics = run_diagnostics(config, config_path=config_path)
    schema_rows = []
    for seed in [int(value) for value in config["stage"]["seeds"]]:
        splits = build_multiview_code_patch_splits(_dataset_config_for_stage(config, _stage_from_config(config)), seed=seed, repo_root=Path("."))
        schema_rows.append({"seed": seed, "rows": _schema_dataset_baselines(splits, config, device=device, seed=seed)["rows"]})
    schema_summary = _schema_pre_run_summary(schema_rows)
    positive = _pre_run_positive_control(config)
    summary = _pre_run_summary(diagnostics, schema_summary, positive, config)
    return {
        "shortcut_diagnostics": diagnostics,
        "schema_dataset_baselines": {"seed_rows": schema_rows, "summary": schema_summary},
        "positive_control": positive,
        "summary": summary,
    }


def _pre_run_positive_control(config: Dict[str, object]) -> Dict[str, object]:
    stage = _stage_from_config(config)
    splits = build_multiview_code_patch_splits(_dataset_config_for_stage(config, stage), seed=0, repo_root=Path("."))
    scorer = CompatibilityFeatureScorer(training=_positive_training(config), seed=14_000)
    scorer.fit(splits["train"], splits["dev"])
    rows = {
        split: _accuracy(scorer.predict(examples), _labels(examples))
        for split, examples in splits.items()
    }
    return {
        "method": "candidate_pair_compatibility_mlp",
        "seed": 0,
        "accuracy": rows,
        "learnability_confirmed": bool(float(rows["test"]) >= float(config.get("positive_control_min", 0.50))),
        "target": float(config.get("positive_control_min", 0.50)),
    }


def _schema_pre_run_summary(seed_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    names = (
        "schema_only_baseline",
        "null_evidence_values_baseline",
        "candidate_only_baseline",
        "evidence_only_no_candidates_baseline",
    )
    out = {}
    for name in names:
        test_values = [float(row["rows"][name]["test"]) for row in seed_rows]
        out[name] = {"mean_test_accuracy": _mean(test_values), "max_test_accuracy": max(test_values) if test_values else 0.0}
    return out


def _pre_run_summary(diagnostics: Dict[str, object], schema_summary: Dict[str, object], positive: Dict[str, object], config: Dict[str, object]) -> Dict[str, object]:
    diag = diagnostics.get("summary", {})
    near = float(config.get("near_chance_max", 0.18))
    static_max = float(config.get("static_frequency_max", 0.20))
    oracle_min = float(config.get("oracle_min", 0.90))
    gates = {
        "candidate_only": float(diag.get("mean_candidate_only_accuracy", 1.0)) <= near,
        "candidate_metadata_only": float(diag.get("mean_candidate_metadata_only_accuracy", 1.0)) <= near,
        "view_masked_candidates_visible": float(diag.get("mean_view_masked_candidates_visible_accuracy", 1.0)) <= near,
        "role_pair_only": float(diag.get("mean_role_pair_only_accuracy", 1.0)) <= near,
        "lexical_overlap": float(diag.get("mean_lexical_overlap_accuracy", 1.0)) <= near,
        "static_frequency": float(diag.get("mean_static_frequency_accuracy", 1.0)) <= static_max,
        "schema_only_baseline": float(schema_summary["schema_only_baseline"]["mean_test_accuracy"]) <= near,
        "null_evidence_values_baseline": float(schema_summary["null_evidence_values_baseline"]["mean_test_accuracy"]) <= near,
        "evidence_only_no_candidates_baseline": float(schema_summary["evidence_only_no_candidates_baseline"]["mean_test_accuracy"]) <= near,
        "all_role_oracle": float(diag.get("mean_all_role_structured_oracle_accuracy", 0.0)) >= oracle_min,
        "positive_control_learnability": bool(positive.get("learnability_confirmed", False)),
    }
    return {
        "gates": gates,
        "passes": all(gates.values()),
        "near_chance_max": near,
        "static_frequency_max": static_max,
        "oracle_min": oracle_min,
    }


def _overall_criteria(result: Dict[str, object]) -> List[Dict[str, object]]:
    rows = [row for row in result.get("stage38_rows", []) if row.get("status") == "completed"]
    if not rows:
        return [{"criterion": "completed seeds >= 10", "pass": False, "value": 0}]
    config = result.get("config", {})
    near = float(config.get("near_chance_max", 0.18))
    inv_tol = float(config.get("invariance_tolerance", 0.08))
    deltas = [float(row["test_delta"]) for row in rows]
    ci_low, ci_high = _bootstrap_ci(deltas)
    trainable_values = [float(row["test_accuracy"]["trainable"]) for row in rows]
    trainable_mean = _mean(trainable_values)
    frozen_wins = sum(float(row["test_delta"]) > 0.0 for row in rows)
    single_view_names = tuple(f"single_view_text_role_{index}" for index in range(4))
    baseline_names = ("text_only", "raw_latent", "majority_baseline", "candidate_order_baseline", "single_agent_full_context")
    baseline_means = {name: _mean([float(row["test_accuracy"].get(name, 0.0)) for row in rows]) for name in baseline_names}
    single_view_means = {name: _mean([float(row["test_accuracy"].get(name, 0.0)) for row in rows]) for name in single_view_names}
    control_means = {name: _mean([float(row["test_accuracy"].get(name, 1.0)) for row in rows]) for name in PRIMARY_CORRUPTION_CONTROLS}
    invariance = {name: _mean([float(row["test_accuracy"].get(name, 0.0)) for row in rows]) for name in INVARIANCE_CONTROLS}
    required_checkpoints = _required_checkpoint_names()
    return [
        {"criterion": "completed seeds >= 10", "pass": len(rows) >= 10 and not result.get("failed_or_interrupted_seeds"), "value": len(rows)},
        {"criterion": "trainable beats frozen on at least 8/10 seeds", "pass": len(rows) >= 10 and frozen_wins >= 8, "value": frozen_wins},
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
        {
            "criterion": "primary corruption controls collapse near chance",
            "pass": all(value <= near for value in control_means.values()),
            "value": control_means,
        },
        {
            "criterion": "invariance controls preserve accuracy",
            "pass": all(abs(value - trainable_mean) <= inv_tol for value in invariance.values()),
            "value": {
                name: {"mean_accuracy": value, "mean_delta_from_trainable": value - trainable_mean}
                for name, value in invariance.items()
            },
        },
        {
            "criterion": "pre-run shortcut diagnostics and positive controls pass",
            "pass": bool(result.get("pre_run", {}).get("summary", {}).get("passes", False)),
            "value": result.get("pre_run", {}).get("summary", {}).get("gates", {}),
        },
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
            "criterion": "shared trainable model receives gradients and changes",
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
                all(row.get("checkpoint_paths", {}).get(name) and Path(str(row.get("checkpoint_paths", {}).get(name))).exists() for name in required_checkpoints)
                for row in rows
            ),
            "value": list(required_checkpoints),
        },
    ]


def _seed_criteria(row: Dict[str, object], config: Dict[str, object]) -> Dict[str, object]:
    test = row["test_accuracy"]
    near = float(config.get("near_chance_max", 0.18))
    inv_tol = float(config.get("invariance_tolerance", 0.08))
    single_view = {f"single_view_text_role_{index}": float(test.get(f"single_view_text_role_{index}", 0.0)) for index in range(4)}
    return {
        "trainable_beats_frozen": float(test["trainable"]) > float(test["frozen"]),
        "trainable_beats_text_only": float(test["trainable"]) > float(test["text_only"]),
        "trainable_beats_raw_latent": float(test["trainable"]) > float(test["raw_latent"]),
        "trainable_beats_majority": float(test["trainable"]) > float(test["majority_baseline"]),
        "trainable_beats_candidate_order": float(test["trainable"]) > float(test["candidate_order_baseline"]),
        "trainable_beats_full_context": float(test["trainable"]) > float(test["single_agent_full_context"]),
        "trainable_beats_every_single_view": all(float(test["trainable"]) > value for value in single_view.values()),
        "single_view_accuracy": single_view,
        "schema_aware_corruption_controls_near_chance": {name: float(test.get(name, 1.0)) <= near for name in PRIMARY_CORRUPTION_CONTROLS},
        "invariance_controls_preserve_accuracy": {
            name: abs(float(test.get(name, 0.0)) - float(test["trainable"])) <= inv_tol for name in INVARIANCE_CONTROLS
        },
        "role_embedding_shuffle_diagnostic_accuracy": float(test.get("role_embedding_shuffle_diagnostic", 0.0)),
        "role_embedding_shuffle_primary_gate": False,
        "leakage_audits_pass": bool(row.get("split_leakage_audit_passes"))
        and bool(row.get("output_leakage_audit_passes"))
        and bool(row.get("split_leakage_audit", {}).get("candidate_patch_hash_leakage_audit_passes", False)),
        "trainable_agent_changed": float(row.get("trainable_audit", {}).get("agent_parameter_delta") or 0.0) > 0.0,
        "frozen_agent_unchanged": _audit_float(row.get("frozen_audit", {}), "agent_parameter_delta", 1.0) == 0.0,
    }


def _summary(result: Dict[str, object]) -> Dict[str, object]:
    rows = [row for row in result.get("stage38_rows", []) if row.get("status") == "completed"]
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
        "passed_stage38_full_hard_validation_gates": all(bool(item["pass"]) for item in criteria) if criteria else False,
        "success_criteria": criteria,
    }


def _write_outputs(result: Dict[str, object], output_path: Path, audit_path: Path, report_path: Path) -> None:
    for row in result.get("stage38_rows", []):
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


def _audit_rows(result: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    rows.append({"event": "pre_run_summary", "time_utc": _now(), **dict(result.get("pre_run", {}).get("summary", {}))})
    for row in result.get("stage38_rows", []):
        if row.get("status") != "completed":
            continue
        rows.append(
            {
                "event": "seed_completed",
                "stage": "stage38_full_hard_validation",
                "seed": int(row["seed"]),
                "time_utc": row.get("completed_at_utc"),
                "test_accuracy": row.get("test_accuracy", {}),
                "test_delta": row.get("test_delta"),
                "split_leakage_audit_passes": row.get("split_leakage_audit_passes"),
                "output_leakage_audit_passes": row.get("output_leakage_audit_passes"),
                "trainable_audit": row.get("trainable_audit", {}),
                "frozen_audit": row.get("frozen_audit", {}),
                "checkpoint_paths": row.get("checkpoint_paths", {}),
                "success_criteria": row.get("success_criteria", {}),
            }
        )
    rows.extend(result.get("oom_retries", []))
    rows.extend(result.get("failed_or_interrupted_seeds", []))
    return rows


def _render_report(result: Dict[str, object]) -> str:
    rows = [row for row in result.get("stage38_rows", []) if row.get("status") == "completed"]
    config = result.get("config", {})
    stage = config.get("stage", {})
    metadata = result.get("metadata", {})
    summary = result.get("summary", {})
    pre = result.get("pre_run", {})
    lines = [
        "# Stage 3.8 Full Hard Validation",
        "",
        "## Scope/Config",
        "",
        f"- Architecture: `{ARCHITECTURE}`",
        "- Locked architecture changes: none",
        "- Locked setup: shared-weight cloned agent; active/top-k token readout; no message head; no private-cue auxiliary loss; candidate-query coordinator; exact frozen same-architecture comparator.",
        f"- Dataset/control representation: `{BALANCED_34B_CANDIDATE_REPRESENTATION}` from `{BALANCED_34B_DATASET_SOURCE}`.",
        "- Normalized-view-format variant: not used.",
        f"- Device requested/used: `{metadata.get('device', config.get('device'))}`",
        f"- CUDA device: `{metadata.get('hardware', {}).get('cuda_device_name', 'n/a')}`",
        f"- Mixed precision: `{stage.get('mixed_precision', 'none')}`",
        f"- Split sizes: train `{stage.get('n_train')}`, dev `{stage.get('n_dev')}`, test `{stage.get('n_test')}`; candidates `8`.",
        f"- Seeds requested: `{stage.get('seeds')}`",
        f"- Batch size fallback: `{config.get('gpu_safety', {}).get('batch_size_fallbacks', [32, 16, 8])}`",
        "",
        "## Schema-Aware Role Policy",
        "",
        "- Role-specific schema fields are preserved and treated as legitimate program-analysis evidence.",
        "- `role_embedding_shuffle_diagnostic` is diagnostic only and is not a primary corruption gate.",
        "- Primary gates are compatibility-breaking controls that preserve schema where appropriate while breaking evidence/candidate compatibility.",
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
        ("candidate_only", diag.get("mean_candidate_only_accuracy", 0.0), near),
        ("candidate_metadata_only", diag.get("mean_candidate_metadata_only_accuracy", 0.0), near),
        ("view_masked_candidates_visible", diag.get("mean_view_masked_candidates_visible_accuracy", 0.0), near),
        ("role_pair_only", diag.get("mean_role_pair_only_accuracy", 0.0), near),
        ("lexical_overlap", diag.get("mean_lexical_overlap_accuracy", 0.0), near),
        ("static_frequency", diag.get("mean_static_frequency_accuracy", 0.0), static_max),
        ("schema_only_baseline", schema.get("schema_only_baseline", {}).get("mean_test_accuracy", 0.0), near),
        ("null_evidence_values_baseline", schema.get("null_evidence_values_baseline", {}).get("mean_test_accuracy", 0.0), near),
        ("evidence_only_no_candidates_baseline", schema.get("evidence_only_no_candidates_baseline", {}).get("mean_test_accuracy", 0.0), near),
        ("all_role_oracle", diag.get("mean_all_role_structured_oracle_accuracy", 0.0), float(config.get("oracle_min", 0.90))),
    ]
    for name, value, threshold in pre_rows:
        pass_value = value >= threshold if name == "all_role_oracle" else value <= threshold
        lines.append(f"| {name} | `{bool(pass_value)}` | {float(value):.4f} | {float(threshold):.4f} |")
    positive = pre.get("positive_control", {})
    lines.extend(
        [
            "",
            "## Positive-Control Learnability",
            "",
            f"- Method: `{positive.get('method', 'candidate_pair_compatibility_mlp')}`",
            f"- Seed 0 test accuracy: `{float(positive.get('accuracy', {}).get('test', 0.0)):.4f}`",
            f"- Learnability confirmed: `{bool(positive.get('learnability_confirmed', False))}`",
            "",
            "## Per-Seed Accuracy",
            "",
            "| seed | batch | accum | max CUDA GB | trainable | frozen | delta | text | raw | majority | cand-order | max single-view | full ctx | candidate-pair | oracle |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        test = row["test_accuracy"]
        max_single = max(float(test.get(f"single_view_text_role_{index}", 0.0)) for index in range(4))
        lines.append(
            "| {seed} | {batch} | {accum} | {mem:.3f} | {train:.4f} | {frozen:.4f} | {delta:.4f} | {text:.4f} | {raw:.4f} | {maj:.4f} | {corder:.4f} | {single:.4f} | {full:.4f} | {pair:.4f} | {oracle:.4f} |".format(
                seed=row["seed"],
                batch=row.get("batch_size"),
                accum=row.get("gradient_accumulation_steps"),
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
            "## Baselines Table",
            "",
            "| method | mean | std |",
            "|---|---:|---:|",
        ]
    )
    for name in ("trainable", "frozen", "text_only", "raw_latent", "majority_baseline", "candidate_order_baseline", "single_agent_full_context", "candidate_pair_compatibility_mlp", "all_role_oracle"):
        values = [float(row["test_accuracy"].get(name, 0.0)) for row in rows]
        lines.append(f"| {name} | {_mean(values):.4f} | {_std(values):.4f} |")
    for index in range(4):
        name = f"single_view_text_role_{index}"
        values = [float(row["test_accuracy"].get(name, 0.0)) for row in rows]
        lines.append(f"| {name} | {_mean(values):.4f} | {_std(values):.4f} |")
    lines.extend(["", "## Schema-Aware Control Table", "", "| control | mean | std | max | pass |", "|---|---:|---:|---:|---|"])
    for name in PRIMARY_CORRUPTION_CONTROLS:
        values = [float(row["test_accuracy"].get(name, 0.0)) for row in rows]
        lines.append(f"| {name} | {_mean(values):.4f} | {_std(values):.4f} | {(max(values) if values else 0.0):.4f} | `{bool(_mean(values) <= near)}` |")
    trainable_mean = _mean([float(row["test_accuracy"]["trainable"]) for row in rows])
    inv_tol = float(config.get("invariance_tolerance", 0.08))
    lines.extend(["", "## Invariance Control Table", "", "| control | mean | delta from trainable | tolerance | pass |", "|---|---:|---:|---:|---|"])
    for name in INVARIANCE_CONTROLS:
        values = [float(row["test_accuracy"].get(name, 0.0)) for row in rows]
        delta = _mean(values) - trainable_mean
        lines.append(f"| {name} | {_mean(values):.4f} | {delta:.4f} | {inv_tol:.4f} | `{bool(abs(delta) <= inv_tol)}` |")
    lines.extend(["", "## Role-Embedding Shuffle Diagnostic", "", "| seed | trainable | role_embedding_shuffle_diagnostic | delta | gating? |", "|---:|---:|---:|---:|---|"])
    for row in rows:
        test = row["test_accuracy"]
        diagnostic = float(test.get("role_embedding_shuffle_diagnostic", 0.0))
        trainable = float(test["trainable"])
        lines.append(f"| {row['seed']} | {trainable:.4f} | {diagnostic:.4f} | {diagnostic - trainable:.4f} | `False` |")
    lines.extend(["", "## Audit Summary", "", "| seed | split leak | output leak | train grad | train delta | frozen grad | frozen delta | checkpoints |", "|---:|---|---|---:|---:|---:|---:|---|"])
    for row in rows:
        train_audit = row.get("trainable_audit", {})
        frozen_audit = row.get("frozen_audit", {})
        ckpt_ok = all(row.get("checkpoint_paths", {}).get(name) and Path(str(row.get("checkpoint_paths", {}).get(name))).exists() for name in _required_checkpoint_names())
        lines.append(
            "| {seed} | {split} | {output} | {tg:.4f} | {td:.4f} | {fg:.4f} | {fd:.4f} | {ckpt} |".format(
                seed=row["seed"],
                split="pass" if row.get("split_leakage_audit_passes") else "fail",
                output="pass" if row.get("output_leakage_audit_passes") else "fail",
                tg=float(train_audit.get("agent_grad_norm_mean") or 0.0),
                td=float(train_audit.get("agent_parameter_delta") or 0.0),
                fg=float(frozen_audit.get("agent_grad_norm_mean") or 0.0),
                fd=float(frozen_audit.get("agent_parameter_delta") or 0.0),
                ckpt="pass" if ckpt_ok else "fail",
            )
        )
    lines.extend(["", "## Acceptance Gates", "", "| criterion | pass | value |", "|---|---|---|"])
    for item in summary.get("success_criteria", []):
        lines.append(f"| {item['criterion']} | `{bool(item['pass'])}` | `{json.dumps(item['value'], sort_keys=True)}` |")
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
            f"- Full Stage 3.8 gates passed: `{bool(summary.get('passed_stage38_full_hard_validation_gates', False))}`",
            "- Do not interpret `role_embedding_shuffle_diagnostic` as a primary corruption failure; schema text legitimately preserves role identity.",
            "- This report only supports a Stage 3.8 benchmark conclusion if every gate above passes, checkpoints exist, and leakage/audit checks pass.",
        ]
    )
    return "\n".join(lines) + "\n"


def _stage38_full_config(config: Dict[str, object]) -> Dict[str, object]:
    out = json.loads(json.dumps(config))
    out["dataset"] = BALANCED_34B_DATASET_SOURCE
    out["architecture"] = ARCHITECTURE
    out["architecture_changes"] = "forbidden"
    out["dataset_config"] = {
        **dict(out.get("dataset_config", {})),
        "dataset_source": BALANCED_34B_DATASET_SOURCE,
        "generator_version": "real_import_restore_candidate_balanced_34b_stage38_full",
        "candidate_representation": BALANCED_34B_CANDIDATE_REPRESENTATION,
    }
    return out


def _stage_from_config(config: Dict[str, object]) -> StageConfig:
    stage = dict(config["stage"])
    return StageConfig(
        name=str(stage.get("name", "stage38_full_hard_validation")),
        n_train=int(stage["n_train"]),
        n_dev=int(stage["n_dev"]),
        n_test=int(stage["n_test"]),
        seeds=tuple(int(value) for value in stage["seeds"]),
        epochs=int(stage["epochs"]),
        patience=int(stage["patience"]),
        hidden_dim=int(stage["hidden_dim"]),
        tiny_layers=int(stage["tiny_layers"]),
        tiny_ff_dim=int(stage["tiny_ff_dim"]),
        batch_size=int(stage.get("batch_size", 32)),
        lr=float(stage.get("lr", 0.002)),
        max_candidates=1,
        gradient_accumulation_steps=int(stage.get("gradient_accumulation_steps", 1)),
        mixed_precision=str(stage.get("mixed_precision", "bf16")),
    )


def _dataset_config_for_stage(config: Dict[str, object], stage: StageConfig):
    stage_dict = asdict(stage)
    stage_dict["seeds"] = list(stage.seeds)
    return stage34_dataset_config({**config, "stage": stage_dict})


def _positive_training(config: Dict[str, object]) -> MLPTrainingConfig:
    cfg = dict(config.get("positive_control", {}))
    return MLPTrainingConfig(
        epochs=int(cfg.get("epochs", 8)),
        batch_size=int(cfg.get("batch_size", 32)),
        lr=float(cfg.get("lr", 0.002)),
        weight_decay=float(cfg.get("weight_decay", 0.0001)),
        patience=int(cfg.get("patience", 3)),
        hidden_dims=tuple(int(value) for value in cfg.get("hidden_dims", [64])),
    )


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


def _checkpoint_root(config: Dict[str, object]) -> Path:
    return Path(str(config.get("checkpoint_dir", "results/stage38_full_hard_validation_checkpoints")))


def _metadata(config: Dict[str, object], config_path: Path, hardware: Dict[str, object], device: str) -> Dict[str, object]:
    return {
        "benchmark": BENCHMARK,
        "task": "stage38_schema_aware_full_hard_validation",
        "architecture": ARCHITECTURE,
        "architecture_selection": "locked before Stage 3.8 full validation",
        "architecture_changes": "forbidden",
        "dataset": BALANCED_34B_DATASET_SOURCE,
        "candidate_representation": BALANCED_34B_CANDIDATE_REPRESENTATION,
        "schema_policy": "role-specific schema fields preserved as legitimate program-analysis evidence",
        "role_embedding_shuffle_primary_gate": False,
        "config_path": str(config_path),
        "device": device,
        "created_or_updated_at_utc": _now(),
        "hardware": hardware,
        "success_claim": "not evaluated until all Stage 3.8 full hard-validation gates pass",
    }


def _labels(examples: Sequence[MultiViewTaskExample]) -> np.ndarray:
    return np.asarray([example.label for example in examples], dtype=np.int64)


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
