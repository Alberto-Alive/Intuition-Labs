from __future__ import annotations

import argparse
import json
import math
import traceback
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Sequence

import numpy as np
import torch

from src.datasets.multiview_code_patch_selection import (
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
    _candidate_specs,
    _candidate_config,
    _candidate_metrics,
    _collapse_summary,
    _combined_probe_summary,
    _coordinator_config,
    _dataset_config,
    _fit_baselines,
    _leakage_passes,
    _per_family_accuracy,
    _per_role_ablation,
    _private_cue_summary,
    _rejection_reasons,
    _training_config,
)
from src.experiments.real_shared_weight_latent_coordination import (
    BENCHMARK,
    TextOutputOnlyCoordinator,
    fit_latent_system,
    _message_mechanism_diagnostic_rows,
)


DEFAULT_CONFIG = "configs/stage3_gpu_hard_validation.json"
ARCHITECTURE = "topk_attention_no_head"
CORRUPTION_NEAR_CHANCE_GATES = (
    "randomized_labels",
    "hidden_states_shuffled_across_examples",
    "view_masked",
    "view_shuffled",
    "role_labels_shuffled",
    "candidate_order_baseline",
    "majority_baseline",
)
INVARIANCE_PRESERVE_ACCURACY_GATES = (
    "physical_order_shuffled_roles_preserved",
    "candidate_order_shuffled",
)
CONTROL_NAMES = (
    "randomized_labels",
    "hidden_states_shuffled_across_examples",
    "view_masked",
    "view_shuffled",
    "role_labels_shuffled",
    "physical_order_shuffled_roles_preserved",
    "candidate_order_shuffled",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable CUDA Stage 3 hard validation for topk_attention_no_head.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_stage3(config, config_path=config_path)


def run_stage3(config: Dict[str, object], config_path: Path) -> Dict[str, object]:
    device, hardware = _configure_cuda(config)
    output_path = Path(str(config["output_path"]))
    audit_path = Path(str(config["audit_log_path"]))
    report_path = Path(str(config["report_path"]))
    result = _load_result(output_path)
    result.setdefault("metadata", _metadata(config, config_path, hardware, device))
    result["metadata"].update(_metadata(config, config_path, hardware, device))
    result["config"] = config
    result.setdefault("stage3_rows", [])
    result.setdefault("failed_or_interrupted_seeds", [])
    result.setdefault("oom_retries", [])
    result["stage3_rows"] = [
        row
        for row in result["stage3_rows"]
        if row.get("status") == "completed" and row.get("candidate") == ARCHITECTURE and _row_has_required_checkpoints(row, config)
    ]

    completed = {
        int(row["seed"])
        for row in result["stage3_rows"]
        if row.get("status") == "completed" and row.get("candidate") == ARCHITECTURE and _row_has_required_checkpoints(row, config)
    }
    stage_base = _stage_from_config(config)
    seeds = [int(value) for value in config["stage"]["seeds"]]
    candidate = next(candidate for candidate in _candidate_specs() if candidate.name == ARCHITECTURE)
    fallbacks = [int(value) for value in config.get("gpu_safety", {}).get("batch_size_fallbacks", [32, 16, 8])]
    effective_batch = int(config.get("gpu_safety", {}).get("effective_batch_size", fallbacks[0]))

    for seed in seeds:
        if seed in completed:
            print(f"stage3 seed={seed}: existing completed row found; skipping")
            continue
        result["stage3_rows"] = [row for row in result["stage3_rows"] if int(row.get("seed", -1)) != seed]
        print(f"stage3 seed={seed}: starting CUDA hard validation")
        seed_completed = False
        for batch_size in fallbacks:
            accumulation = max(1, int(math.ceil(effective_batch / float(batch_size))))
            stage = replace(
                stage_base,
                seeds=(seed,),
                batch_size=batch_size,
                gradient_accumulation_steps=accumulation,
            )
            try:
                row, audit_rows = _run_seed(stage, seed, candidate, device, hardware, config)
                result["stage3_rows"].append(row)
                completed.add(seed)
                seed_completed = True
                _write_outputs(result, output_path, audit_path, report_path)
                print(
                    "stage3 seed={seed}: completed trainable={trainable:.4f} frozen={frozen:.4f} "
                    "delta={delta:.4f} max_cuda_gb={memory:.3f}".format(
                        seed=seed,
                        trainable=float(row["test_accuracy"]["trainable"]),
                        frozen=float(row["test_accuracy"]["frozen"]),
                        delta=float(row["test_delta"]),
                        memory=float(row.get("cuda_max_memory_allocated_gb", 0.0)),
                    )
                )
                del audit_rows
                break
            except RuntimeError as exc:
                if _is_oom(exc) and batch_size != fallbacks[-1]:
                    retry = _failure_row(seed, batch_size, accumulation, "oom_retry", exc)
                    result["oom_retries"].append(retry)
                    _clear_cuda()
                    _write_outputs(result, output_path, audit_path, report_path)
                    print(f"stage3 seed={seed}: CUDA OOM at batch_size={batch_size}; retrying smaller batch")
                    continue
                failure = _failure_row(seed, batch_size, accumulation, "failed", exc)
                result["failed_or_interrupted_seeds"].append(failure)
                _clear_cuda()
                _write_outputs(result, output_path, audit_path, report_path)
                print(f"stage3 seed={seed}: failed; see results for traceback")
                break
            finally:
                _clear_cuda()
        if not seed_completed and bool(config.get("stop_on_failed_seed", False)):
            break

    _write_outputs(result, output_path, audit_path, report_path)
    return result


def _run_seed(
    stage: StageConfig,
    seed: int,
    candidate,
    device: str,
    hardware: Dict[str, object],
    config: Dict[str, object],
) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    _clear_cuda()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    print(f"stage3 seed={seed}: building real program-analysis splits")
    splits = build_multiview_code_patch_splits(_dataset_config(stage), seed=seed, repo_root=Path("."))
    print(f"stage3 seed={seed}: fitting baselines")
    baselines = _fit_baselines(stage, splits, seed=seed, device=device)
    split_leakage = split_leakage_audit(BENCHMARK, seed, splits)
    output_leakage = output_leakage_audit(BENCHMARK, seed, splits["train"] + splits["dev"] + splits["test"])
    print(f"stage3 seed={seed}: fitting locked {ARCHITECTURE}")
    row, checkpoint_paths = _evaluate_candidate_with_checkpoints(
        candidate=candidate,
        stage=stage,
        splits=splits,
        seed=seed,
        device=device,
        baselines=baselines,
        leakage=split_leakage,
        config=config,
    )
    cuda_peak = int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") and torch.cuda.is_available() else 0
    row.update(
        {
            "status": "completed",
            "completed_at_utc": _now(),
            "device": device,
            "cuda_device_name": hardware.get("cuda_device_name"),
            "cuda_total_memory": hardware.get("cuda_total_memory"),
            "cuda_max_memory_allocated": cuda_peak,
            "cuda_max_memory_allocated_gb": cuda_peak / float(1024**3),
            "stage_config": asdict(stage),
            "dataset_config": asdict(_dataset_config(stage)),
            "dataset_summary": dataset_summary(splits),
            "split_leakage_audit": _compact_leakage(split_leakage),
            "split_leakage_audit_passes": _leakage_passes(split_leakage),
            "output_leakage_audit": output_leakage,
            "output_leakage_audit_passes": bool(output_leakage.get("passes", False)),
            "architecture_locked": True,
            "architecture_changes_forbidden": True,
            "mixed_precision": stage.mixed_precision,
            "batch_size": stage.batch_size,
            "gradient_accumulation_steps": stage.gradient_accumulation_steps,
            "checkpoint_paths": checkpoint_paths,
        }
    )
    row["success_criteria"] = _seed_criteria(row, float(config.get("control_chance_tolerance", 0.10)))
    audit_rows = _audit_rows_for_seed(row, split_leakage, output_leakage)
    return row, audit_rows


def _evaluate_candidate_with_checkpoints(
    candidate,
    stage: StageConfig,
    splits: Dict[str, Sequence[object]],
    seed: int,
    device: str,
    baselines: Dict[str, object],
    leakage: Dict[str, object],
    config: Dict[str, object],
) -> tuple[Dict[str, object], Dict[str, str]]:
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
        method=f"search_trainable__{candidate.name}",
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
        method=f"search_frozen__{candidate.name}",
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
        method=f"search_randomized__{candidate.name}",
        condition="randomized_labels",
        train_labels=randomized_labels_for_examples(splits["train"], seed + 401, 8),
        dev_labels=randomized_labels_for_examples(splits["dev"], seed + 501, 8),
        message_config=candidate.message_config,
    )
    metrics = _candidate_metrics(trainable, frozen, randomized, None, None, baselines, splits, seed)
    diagnostics = _message_mechanism_diagnostic_rows(trainable, splits, seed=seed, num_classes=8, device=device)
    collapse = _collapse_summary(diagnostics)
    private_cue = _private_cue_summary(diagnostics)
    combined_probe = _combined_probe_summary(diagnostics)
    per_family = _per_family_accuracy(trainable, splits["test"], seed)
    per_role = _per_role_ablation(trainable, splits["test"], seed)
    rejection_reasons = _rejection_reasons(metrics, collapse, per_role, leakage, chance=0.125)
    checkpoint_paths = _save_stage3_checkpoints(
        seed=seed,
        stage=stage,
        candidate=candidate,
        trainable=trainable,
        frozen=frozen,
        baselines=baselines,
        config=config,
    )
    row = {
        "stage": stage.name,
        "seed": seed,
        "candidate": candidate.name,
        "description": candidate.description,
        "architecture_config": _candidate_config(candidate),
        "dev_accuracy": metrics["dev"],
        "test_accuracy": metrics["test"],
        "dev_delta": metrics["dev"]["trainable"] - metrics["dev"]["frozen"],
        "test_delta": metrics["test"]["trainable"] - metrics["test"]["frozen"],
        "selection_valid": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
        "trainable_audit": _audit_subset(trainable.audit),
        "frozen_audit": _audit_subset(frozen.audit),
        "message_collapse": collapse,
        "private_cue_probes": private_cue,
        "combined_message_final_label_probe": combined_probe,
        "per_family_accuracy": per_family,
        "per_role_ablation": per_role,
        "leakage_audit_passes": _leakage_passes(leakage),
    }
    return row, checkpoint_paths


def _metadata(config: Dict[str, object], config_path: Path, hardware: Dict[str, object], device: str) -> Dict[str, object]:
    return {
        "benchmark": BENCHMARK,
        "task": "stage3_gpu_hard_validation",
        "architecture": ARCHITECTURE,
        "architecture_selection": "frozen before Stage 3",
        "architecture_changes": "forbidden",
        "dataset": "real_program_analysis_import_restoration",
        "config_path": str(config_path),
        "device": device,
        "created_or_updated_at_utc": _now(),
        "hardware": hardware,
        "success_claim": "not evaluated until all locked Stage 3 criteria pass",
    }


def _checkpoint_root(config: Dict[str, object]) -> Path:
    return Path(str(config.get("checkpoint_dir", "results/stage3_gpu_hard_validation_checkpoints")))


def _row_has_required_checkpoints(row: Dict[str, object], config: Dict[str, object]) -> bool:
    if not bool(config.get("save_checkpoints", True)):
        return True
    if row.get("dataset_summary", {}).get("dataset_source") != "real_program_analysis_import_restoration":
        return False
    if not str(row.get("dataset_summary", {}).get("generator_version", "")).startswith("real_import_restore_stable_categories_balanced"):
        return False
    paths = row.get("checkpoint_paths", {})
    required = (
        "trainable",
        "frozen",
        "text_only",
        "raw_latent",
        "majority_baseline",
        "candidate_order_baseline",
        "single_agent_full_context",
        "single_view_text_role_0",
        "single_view_text_role_1",
        "single_view_text_role_2",
        "single_view_text_role_3",
        "explicit_evidence_oracle",
    )
    return isinstance(paths, dict) and all(paths.get(name) and Path(str(paths[name])).exists() for name in required)


def _save_stage3_checkpoints(
    seed: int,
    stage: StageConfig,
    candidate,
    trainable,
    frozen,
    baselines: Dict[str, object],
    config: Dict[str, object],
) -> Dict[str, str]:
    if not bool(config.get("save_checkpoints", True)):
        return {}
    root = _checkpoint_root(config) / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "trainable": root / "trainable_topk_attention_no_head.pt",
        "frozen": root / "frozen_topk_attention_no_head.pt",
        "raw_latent": root / "raw_latent_baseline.pt",
        "text_only": root / "text_only_multi_agent_baseline.pt",
        "majority_baseline": root / "majority_baseline.pt",
        "candidate_order_baseline": root / "candidate_order_baseline.pt",
        "single_agent_full_context": root / "single_agent_full_context.pt",
        "single_view_text_role_0": root / "single_view_text_role_0.pt",
        "single_view_text_role_1": root / "single_view_text_role_1.pt",
        "single_view_text_role_2": root / "single_view_text_role_2.pt",
        "single_view_text_role_3": root / "single_view_text_role_3.pt",
        "explicit_evidence_oracle": root / "explicit_evidence_oracle.pt",
    }
    raw = baselines["raw"]
    text = baselines["text"]
    _save_latent_checkpoint(
        paths["trainable"],
        seed=seed,
        stage=stage,
        candidate_name=candidate.name,
        architecture_config=_candidate_config(candidate),
        result=trainable,
    )
    _save_latent_checkpoint(
        paths["frozen"],
        seed=seed,
        stage=stage,
        candidate_name=candidate.name,
        architecture_config=_candidate_config(candidate),
        result=frozen,
    )
    _save_latent_checkpoint(
        paths["raw_latent"],
        seed=seed,
        stage=stage,
        candidate_name="raw_latent_baseline",
        architecture_config={
            "description": "Raw pooled latent baseline.",
            "message_config": asdict(raw.system.message_config),
            "coordinator_config": asdict(_coordinator_config(stage, family="cross_attention", num_layers=1, dropout=0.0)),
        },
        result=raw,
    )
    _save_text_checkpoint(paths["text_only"], seed=seed, stage=stage, text=text)
    _save_fixed_checkpoint(
        paths["majority_baseline"],
        seed=seed,
        stage=stage,
        method="majority_baseline",
        prediction=int(baselines["majority_prediction"]),
    )
    _save_fixed_checkpoint(
        paths["candidate_order_baseline"],
        seed=seed,
        stage=stage,
        method="candidate_order_baseline",
        prediction=int(baselines["candidate_order_prediction"]),
    )
    _save_context_checkpoint(paths["single_agent_full_context"], seed=seed, stage=stage, result=baselines["full_context"])
    for method, baseline in dict(baselines.get("single_view", {})).items():
        if method in paths:
            _save_bow_checkpoint(paths[method], seed=seed, stage=stage, baseline=baseline)
    _save_oracle_checkpoint(paths["explicit_evidence_oracle"], seed=seed, stage=stage)
    return {name: str(path) for name, path in paths.items()}


def _save_latent_checkpoint(
    path: Path,
    seed: int,
    stage: StageConfig,
    candidate_name: str,
    architecture_config: Dict[str, object],
    result,
) -> None:
    payload = {
        "checkpoint_type": "latent_system",
        "benchmark": BENCHMARK,
        "architecture": candidate_name,
        "seed": int(seed),
        "stage_config": asdict(stage),
        "architecture_config": architecture_config,
        "agent_config": asdict(result.agent.config),
        "coordinator_config": architecture_config.get("coordinator_config", {}),
        "message_config": asdict(result.system.message_config),
        "trainable_agent": bool(result.trainable_agent),
        "method": result.method,
        "condition": result.condition,
        "num_classes": 8,
        "n_roles": int(result.system.n_roles),
        "visible_explicit_evidence": bool(result.system.visible_explicit_evidence),
        "param_count": int(result.param_count),
        "audit": result.audit,
        "history": result.history,
        "system_state_dict": {key: value.detach().cpu() for key, value in result.system.state_dict().items()},
    }
    torch.save(payload, path)


def _save_text_checkpoint(path: Path, seed: int, stage: StageConfig, text: TextOutputOnlyCoordinator) -> None:
    if text.model is None:
        raise RuntimeError("cannot save text-only checkpoint before the model is fit")
    payload = {
        "checkpoint_type": "text_only_multi_agent_baseline",
        "benchmark": BENCHMARK,
        "seed": int(seed),
        "stage_config": asdict(stage),
        "num_classes": int(text.num_classes),
        "training": asdict(text.training),
        "feature_dim": int(text.feature_dim),
        "param_count": int(text.param_count),
        "history": text.history,
        "model_state_dict": {key: value.detach().cpu() for key, value in text.model.state_dict().items()},
    }
    torch.save(payload, path)


def _save_bow_checkpoint(path: Path, seed: int, stage: StageConfig, baseline) -> None:
    if baseline.model is None:
        raise RuntimeError(f"cannot save {baseline.method} before the model is fit")
    payload = {
        "checkpoint_type": "bag_of_words_single_view_baseline",
        "benchmark": BENCHMARK,
        "seed": int(seed),
        "stage_config": asdict(stage),
        "method": str(baseline.method),
        "num_classes": int(baseline.num_classes),
        "training": asdict(baseline.training),
        "feature_dim": int(baseline.feature_dim),
        "param_count": int(baseline.param_count),
        "history": baseline.history,
        "model_state_dict": {key: value.detach().cpu() for key, value in baseline.model.state_dict().items()},
    }
    torch.save(payload, path)


def _save_context_checkpoint(path: Path, seed: int, stage: StageConfig, result) -> None:
    payload = {
        "checkpoint_type": "single_agent_full_context_baseline",
        "benchmark": BENCHMARK,
        "seed": int(seed),
        "stage_config": asdict(stage),
        "method": result.method,
        "prompt_mode": result.prompt_mode,
        "param_count": int(result.param_count),
        "history": result.history,
        "agent_config": asdict(result.agent.config),
        "agent_state_dict": {key: value.detach().cpu() for key, value in result.agent.state_dict().items()},
        "head_state_dict": {key: value.detach().cpu() for key, value in result.head.state_dict().items()},
    }
    torch.save(payload, path)


def _save_fixed_checkpoint(path: Path, seed: int, stage: StageConfig, method: str, prediction: int) -> None:
    payload = {
        "checkpoint_type": "fixed_prediction_baseline",
        "benchmark": BENCHMARK,
        "seed": int(seed),
        "stage_config": asdict(stage),
        "method": method,
        "prediction": int(prediction),
        "param_count": 0,
    }
    torch.save(payload, path)


def _save_oracle_checkpoint(path: Path, seed: int, stage: StageConfig) -> None:
    payload = {
        "checkpoint_type": "explicit_structured_evidence_oracle",
        "benchmark": BENCHMARK,
        "seed": int(seed),
        "stage_config": asdict(stage),
        "method": "explicit_evidence_oracle",
        "uses_structured_gold_evidence": True,
        "oracle_upper_bound_only": True,
        "param_count": 0,
    }
    torch.save(payload, path)


def _stage_from_config(config: Dict[str, object]) -> StageConfig:
    stage = dict(config["stage"])
    return StageConfig(
        name="stage3",
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
        mixed_precision=str(stage.get("mixed_precision", "none")),
    )


def _configure_cuda(config: Dict[str, object]) -> tuple[str, Dict[str, object]]:
    requested = str(config.get("device", "cuda"))
    require_cuda = bool(config.get("require_cuda", True))
    hardware: Dict[str, object] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "requested_device": requested,
    }
    if requested.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(0)
        props = torch.cuda.get_device_properties(0)
        budget_gb = float(config.get("cuda_memory_budget_gb", 16.0))
        fraction = min(1.0, (budget_gb * (1024**3)) / float(props.total_memory))
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        hardware.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_total_memory": int(props.total_memory),
                "cuda_memory_budget_gb": budget_gb,
                "cuda_memory_fraction": fraction,
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
        return "cuda", hardware
    if require_cuda:
        raise RuntimeError(f"CUDA was requested but is not available; requested_device={requested}")
    hardware["fallback_reason"] = "cuda unavailable or not requested"
    return "cpu", hardware


def _load_result(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_outputs(result: Dict[str, object], output_path: Path, audit_path: Path, report_path: Path) -> None:
    tolerance = float(result.get("config", {}).get("control_chance_tolerance", 0.10))
    for row in result.get("stage3_rows", []):
        if row.get("status") == "completed":
            row["success_criteria"] = _seed_criteria(row, tolerance)
    result["summary"] = _summary(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(output_path)

    audit_rows: List[Dict[str, object]] = []
    for row in result.get("stage3_rows", []):
        audit_rows.extend(_audit_rows_for_seed(row, row.get("split_leakage_audit", {}), row.get("output_leakage_audit", {})))
    audit_rows.extend(result.get("oom_retries", []))
    audit_rows.extend(result.get("failed_or_interrupted_seeds", []))
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in audit_rows) + ("\n" if audit_rows else ""), encoding="utf-8")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(result), encoding="utf-8")


def _summary(result: Dict[str, object]) -> Dict[str, object]:
    rows = [row for row in result.get("stage3_rows", []) if row.get("status") == "completed"]
    deltas = [float(row["test_delta"]) for row in rows]
    trainable = [float(row["test_accuracy"]["trainable"]) for row in rows]
    frozen = [float(row["test_accuracy"]["frozen"]) for row in rows]
    ci_low, ci_high = _bootstrap_ci(deltas) if deltas else (0.0, 0.0)
    criteria = _overall_criteria(rows, result.get("failed_or_interrupted_seeds", []))
    return {
        "completed_seeds": [int(row["seed"]) for row in rows],
        "n_completed": len(rows),
        "n_failed_or_interrupted": len(result.get("failed_or_interrupted_seeds", [])),
        "mean_trainable_accuracy": _mean(trainable),
        "mean_frozen_accuracy": _mean(frozen),
        "mean_delta": _mean(deltas),
        "std_delta": pstdev(deltas) if len(deltas) > 1 else 0.0,
        "bootstrap_95_ci_delta": [ci_low, ci_high],
        "passed_stage3_locked_success_criteria": all(bool(item["pass"]) for item in criteria) if criteria else False,
        "success_criteria": criteria,
    }


def _overall_criteria(rows: Sequence[Dict[str, object]], failures: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    if not rows:
        return [{"criterion": "completed Stage 3 seeds >= 10", "pass": False, "value": 0}]
    deltas = [float(row["test_delta"]) for row in rows]
    ci_low, ci_high = _bootstrap_ci(deltas)
    chance = 1.0 / 8.0
    threshold = chance + 0.10
    corruption_values = {name: [float(row["test_accuracy"].get(name, 1.0)) for row in rows] for name in CORRUPTION_NEAR_CHANCE_GATES}
    invariance_values = {name: [float(row["test_accuracy"].get(name, 0.0)) for row in rows] for name in INVARIANCE_PRESERVE_ACCURACY_GATES}
    trainable_values = [float(row["test_accuracy"]["trainable"]) for row in rows]
    trainable_mean = _mean(trainable_values)
    invariance_tolerance = 0.05
    frozen_wins = sum(delta > 0.0 for delta in deltas)
    single_view_names = tuple(f"single_view_text_role_{index}" for index in range(4))
    single_view_values = {name: [float(row["test_accuracy"].get(name, 0.0)) for row in rows] for name in single_view_names}
    role_shuffle_values = corruption_values.get("role_labels_shuffled", [1.0])
    required_checkpoint_names = (
        "trainable",
        "frozen",
        "text_only",
        "raw_latent",
        "majority_baseline",
        "candidate_order_baseline",
        "single_agent_full_context",
        "single_view_text_role_0",
        "single_view_text_role_1",
        "single_view_text_role_2",
        "single_view_text_role_3",
        "explicit_evidence_oracle",
    )
    return [
        {"criterion": "completed Stage 3 seeds >= 10", "pass": len(rows) >= 10 and not failures, "value": len(rows)},
        {"criterion": "trainable beats frozen on at least 8/10 seeds", "pass": len(rows) >= 10 and frozen_wins >= 8, "value": frozen_wins},
        {"criterion": "mean trainable-frozen delta >= +0.20", "pass": _mean(deltas) >= 0.20, "value": _mean(deltas)},
        {"criterion": "bootstrap 95% CI lower bound for delta > +0.05", "pass": ci_low > 0.05, "value": [ci_low, ci_high]},
        {
            "criterion": "trainable beats text-only, raw-latent, majority, and candidate-order baselines by mean accuracy",
            "pass": bool(
                trainable_mean > _mean([float(row["test_accuracy"]["text_only"]) for row in rows])
                and trainable_mean > _mean([float(row["test_accuracy"]["raw_latent"]) for row in rows])
                and trainable_mean > _mean([float(row["test_accuracy"]["majority_baseline"]) for row in rows])
                and trainable_mean > _mean([float(row["test_accuracy"]["candidate_order_baseline"]) for row in rows])
            ),
            "value": {
                "trainable_mean": trainable_mean,
                "text_only_mean": _mean([float(row["test_accuracy"]["text_only"]) for row in rows]),
                "raw_latent_mean": _mean([float(row["test_accuracy"]["raw_latent"]) for row in rows]),
                "majority_mean": _mean([float(row["test_accuracy"]["majority_baseline"]) for row in rows]),
                "candidate_order_mean": _mean([float(row["test_accuracy"]["candidate_order_baseline"]) for row in rows]),
            },
        },
        {
            "criterion": "trainable beats every single-view baseline by mean accuracy",
            "pass": all(trainable_mean > _mean(values) for values in single_view_values.values()),
            "value": {name: _mean(values) for name, values in single_view_values.items()},
        },
        {
            "criterion": "corruption controls plus majority/candidate-order remain near chance",
            "pass": all(_mean(values) <= threshold for values in corruption_values.values()),
            "value": {
                name: {
                    "mean_accuracy": _mean(values),
                    "max_accuracy": max(values or [0.0]),
                }
                for name, values in corruption_values.items()
            },
        },
        {
            "criterion": "invariance controls preserve accuracy",
            "pass": all(_mean(values) >= trainable_mean - invariance_tolerance for values in invariance_values.values()),
            "value": {
                name: {
                    "mean_accuracy": _mean(values),
                    "mean_delta_from_trainable": _mean(values) - trainable_mean,
                }
                for name, values in invariance_values.items()
            },
        },
        {
            "criterion": "role-label shuffle drops substantially or remains near chance",
            "pass": _mean(role_shuffle_values) <= threshold or _mean(role_shuffle_values) <= trainable_mean - 0.15,
            "value": {
                "mean_accuracy": _mean(role_shuffle_values),
                "max_accuracy": max(role_shuffle_values),
                "mean_delta_from_trainable": _mean(role_shuffle_values) - trainable_mean,
            },
        },
        {
            "criterion": "no single role explains the result",
            "pass": all(not bool(row.get("per_role_ablation", {}).get("one_role_only_failure", True)) for row in rows),
            "value": sum(bool(row.get("per_role_ablation", {}).get("one_role_only_failure", True)) for row in rows),
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
            "criterion": "checkpoints saved for every seed and required method",
            "pass": all(
                all(row.get("checkpoint_paths", {}).get(name) and Path(str(row.get("checkpoint_paths", {}).get(name))).exists() for name in required_checkpoint_names)
                for row in rows
            ),
            "value": list(required_checkpoint_names),
        },
        {
            "criterion": "shared trainable model receives gradients and changes",
            "pass": all(float(row.get("trainable_audit", {}).get("agent_grad_norm_mean") or 0.0) > 0.0 and float(row.get("trainable_audit", {}).get("agent_parameter_delta") or 0.0) > 0.0 for row in rows),
            "value": "all_completed_seeds",
        },
        {
            "criterion": "frozen comparator remains frozen",
            "pass": all(_audit_float(row.get("frozen_audit", {}), "agent_grad_norm_mean", 1.0) == 0.0 and _audit_float(row.get("frozen_audit", {}), "agent_parameter_delta", 1.0) == 0.0 for row in rows),
            "value": "all_completed_seeds",
        },
    ]


def _seed_criteria(row: Dict[str, object], tolerance: float) -> Dict[str, object]:
    chance = 1.0 / 8.0
    threshold = chance + tolerance
    test = row["test_accuracy"]
    single_view = {f"single_view_text_role_{index}": float(test.get(f"single_view_text_role_{index}", 0.0)) for index in range(4)}
    return {
        "trainable_beats_frozen": float(test["trainable"]) > float(test["frozen"]),
        "trainable_beats_text_only": float(test["trainable"]) > float(test["text_only"]),
        "trainable_beats_raw_latent": float(test["trainable"]) > float(test["raw_latent"]),
        "trainable_beats_majority": float(test["trainable"]) > float(test["majority_baseline"]),
        "trainable_beats_candidate_order": float(test["trainable"]) > float(test["candidate_order_baseline"]),
        "trainable_beats_every_single_view": all(float(test["trainable"]) > value for value in single_view.values()),
        "single_view_accuracy": single_view,
        "corruption_controls_near_chance": {name: float(test.get(name, 1.0)) <= threshold for name in CORRUPTION_NEAR_CHANCE_GATES},
        "invariance_controls_preserve_accuracy": {
            name: abs(float(test.get(name, 0.0)) - float(test["trainable"])) <= 0.05
            for name in INVARIANCE_PRESERVE_ACCURACY_GATES
        },
        "role_label_shuffle_drops_or_near_chance": float(test.get("role_labels_shuffled", 1.0)) <= threshold
        or float(test.get("role_labels_shuffled", 1.0)) <= float(test["trainable"]) - 0.15,
        "leakage_audits_pass": bool(row.get("split_leakage_audit_passes"))
        and bool(row.get("output_leakage_audit_passes"))
        and bool(row.get("split_leakage_audit", {}).get("candidate_patch_hash_leakage_audit_passes", False)),
        "trainable_agent_changed": float(row.get("trainable_audit", {}).get("agent_parameter_delta") or 0.0) > 0.0,
        "frozen_agent_unchanged": _audit_float(row.get("frozen_audit", {}), "agent_parameter_delta", 1.0) == 0.0,
    }


def _render_report(result: Dict[str, object]) -> str:
    rows = [row for row in result.get("stage3_rows", []) if row.get("status") == "completed"]
    summary = result.get("summary", {})
    metadata = result.get("metadata", {})
    config = result.get("config", {})
    stage = config.get("stage", {})
    lines = [
        "# Stage 3 GPU Hard Validation",
        "",
        "## Scope",
        "",
        f"- Architecture: `{ARCHITECTURE}`",
        "- Architecture changes: forbidden",
        "- Dataset: real program-analysis import-restoration benchmark",
        "- Comparator: exact frozen same-architecture comparator required",
        f"- Device requested/used: `{metadata.get('device', config.get('device'))}`",
        f"- CUDA device: `{metadata.get('hardware', {}).get('cuda_device_name', 'n/a')}`",
        f"- CUDA total memory: `{metadata.get('hardware', {}).get('cuda_total_memory', 0)}` bytes",
        f"- Mixed precision: `{stage.get('mixed_precision', 'none')}`",
        f"- Batch size: `{stage.get('batch_size', 'n/a')}` with fallback `{config.get('gpu_safety', {}).get('batch_size_fallbacks', [32, 16, 8])}`; effective batch target `{config.get('gpu_safety', {}).get('effective_batch_size', 'n/a')}`",
        f"- Split sizes: train `{stage.get('n_train')}`, dev `{stage.get('n_dev')}`, test `{stage.get('n_test')}`",
        f"- Seeds requested: `{stage.get('seeds')}`",
        "",
        "## Completion",
        "",
        f"- Completed seeds: `{summary.get('completed_seeds', [])}`",
        f"- Failed or interrupted seeds: `{result.get('failed_or_interrupted_seeds', [])}`",
        f"- OOM retries: `{result.get('oom_retries', [])}`",
        "",
        "## Per-Seed Accuracy",
        "",
        "| seed | batch | accum | max CUDA GB | trainable | frozen | delta | text | raw | majority | cand-order | max single-view | full ctx | random | hidden shuffle | view masked | view shuffled | role shuffle | physical order | cand-order shuffle | oracle |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        test = row["test_accuracy"]
        max_single = max(float(test.get(f"single_view_text_role_{index}", 0.0)) for index in range(4))
        lines.append(
            "| {seed} | {batch} | {accum} | {mem:.3f} | {train:.4f} | {frozen:.4f} | {delta:.4f} | {text:.4f} | {raw:.4f} | {maj:.4f} | {corder:.4f} | {single:.4f} | {full:.4f} | {rand:.4f} | {hidden:.4f} | {masked:.4f} | {view:.4f} | {role:.4f} | {physical:.4f} | {candidate:.4f} | {oracle:.4f} |".format(
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
                full=float(test.get("single_agent_full_context", 0.0)),
                rand=float(test["randomized_labels"]),
                hidden=float(test["hidden_states_shuffled_across_examples"]),
                masked=float(test["view_masked"]),
                view=float(test["view_shuffled"]),
                role=float(test["role_labels_shuffled"]),
                physical=float(test["physical_order_shuffled_roles_preserved"]),
                candidate=float(test["candidate_order_shuffled"]),
                oracle=float(test["explicit_evidence_oracle"]),
            )
        )
    lines.extend(
        [
            "",
            "## Single-View Baselines",
            "",
            "| seed | role 0 | role 1 | role 2 | role 3 | max single-view | trainable |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        test = row["test_accuracy"]
        single = [float(test.get(f"single_view_text_role_{index}", 0.0)) for index in range(4)]
        lines.append(
            "| {seed} | {r0:.4f} | {r1:.4f} | {r2:.4f} | {r3:.4f} | {maxv:.4f} | {train:.4f} |".format(
                seed=row["seed"],
                r0=single[0],
                r1=single[1],
                r2=single[2],
                r3=single[3],
                maxv=max(single),
                train=float(test["trainable"]),
            )
        )
    ci = summary.get("bootstrap_95_ci_delta", [0.0, 0.0])
    lines.extend(
        [
            "",
            "## Delta Summary",
            "",
            f"- Mean trainable accuracy: `{float(summary.get('mean_trainable_accuracy', 0.0)):.4f}`",
            f"- Mean frozen accuracy: `{float(summary.get('mean_frozen_accuracy', 0.0)):.4f}`",
            f"- Mean delta: `{float(summary.get('mean_delta', 0.0)):.4f}`",
            f"- Delta std: `{float(summary.get('std_delta', 0.0)):.4f}`",
            f"- Bootstrap 95% CI: `[{float(ci[0]):.4f}, {float(ci[1]):.4f}]`",
            "",
            "## Corrected Gate Table",
            "",
            "| gate | expected behavior | mean test accuracy | max test accuracy | mean delta from trainable | pass |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    threshold = 1.0 / 8.0 + float(config.get("control_chance_tolerance", 0.10))
    trainable_mean = _mean([float(row["test_accuracy"]["trainable"]) for row in rows])
    for name in CORRUPTION_NEAR_CHANCE_GATES:
        values = [float(row["test_accuracy"].get(name, 0.0)) for row in rows]
        lines.append(
            f"| {name} | mean near chance <= {threshold:.4f} | {_mean(values):.4f} | {(max(values) if values else 0.0):.4f} | {_mean(values) - trainable_mean:.4f} | {bool(_mean(values) <= threshold)} |"
        )
    for name in INVARIANCE_PRESERVE_ACCURACY_GATES:
        values = [float(row["test_accuracy"].get(name, 0.0)) for row in rows]
        lines.append(
            f"| {name} | preserve accuracy | {_mean(values):.4f} | {(max(values) if values else 0.0):.4f} | {_mean(values) - trainable_mean:.4f} | {bool(_mean(values) >= trainable_mean - 0.05)} |"
        )
    lines.extend(
        [
            "",
            "## Audit Summary",
            "",
            "| seed | split leak | output leak | shared id | train grad | train delta | frozen grad | frozen delta | active variance | active cosine | one-role failure |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        collapse = row.get("message_collapse", {}).get("active_message_readout", {})
        train_audit = row.get("trainable_audit", {})
        frozen_audit = row.get("frozen_audit", {})
        lines.append(
            "| {seed} | {split} | {output} | {shared} | {tg:.4f} | {td:.4f} | {fg:.4f} | {fd:.4f} | {var:.4f} | {cos:.4f} | {role} |".format(
                seed=row["seed"],
                split="pass" if row.get("split_leakage_audit_passes") else "fail",
                output="pass" if row.get("output_leakage_audit_passes") else "fail",
                shared="pass" if train_audit.get("shared_parameter_identity") and frozen_audit.get("shared_parameter_identity") else "fail",
                tg=float(train_audit.get("agent_grad_norm_mean") or 0.0),
                td=float(train_audit.get("agent_parameter_delta") or 0.0),
                fg=float(frozen_audit.get("agent_grad_norm_mean") or 0.0),
                fd=float(frozen_audit.get("agent_parameter_delta") or 0.0),
                var=float(collapse.get("variance_mean", 0.0)),
                cos=float(collapse.get("mean_cosine_similarity", 0.0)),
                role="fail" if row.get("per_role_ablation", {}).get("one_role_only_failure") else "pass",
            )
        )
    lines.extend(
        [
            "",
            "## Per-Role Ablation",
            "",
            "| seed | base | role 0 masked | role 1 masked | role 2 masked | role 3 masked | one-role failure |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        ablation = row.get("per_role_ablation", {})
        masked = ablation.get("accuracy_when_role_masked", {})
        lines.append(
            "| {seed} | {base:.4f} | {r0:.4f} | {r1:.4f} | {r2:.4f} | {r3:.4f} | {fail} |".format(
                seed=row["seed"],
                base=float(ablation.get("base_accuracy", 0.0)),
                r0=float(masked.get("0", 0.0)),
                r1=float(masked.get("1", 0.0)),
                r2=float(masked.get("2", 0.0)),
                r3=float(masked.get("3", 0.0)),
                fail=bool(ablation.get("one_role_only_failure", False)),
            )
        )
    families = sorted({family for row in rows for family in row.get("per_family_accuracy", {}).keys()})
    lines.extend(["", "## Per-Family Accuracy", ""])
    if families:
        lines.append("| seed | " + " | ".join(families) + " |")
        lines.append("|---:|" + "|".join("---:" for _ in families) + "|")
        for row in rows:
            per_family = row.get("per_family_accuracy", {})
            lines.append("| {seed} | {values} |".format(seed=row["seed"], values=" | ".join(f"{float(per_family.get(family, 0.0)):.4f}" for family in families)))
    else:
        lines.append("- No per-family rows are available.")
    lines.extend(
        [
            "",
            "## Locked Criteria",
            "",
            "| criterion | pass | value |",
            "|---|---|---|",
        ]
    )
    for item in summary.get("success_criteria", []):
        lines.append(f"| {item.get('criterion')} | {bool(item.get('pass'))} | `{item.get('value')}` |")
    passed = bool(summary.get("passed_stage3_locked_success_criteria", False))
    lines.extend(
        [
            "",
            "## Conservative Interpretation",
            "",
            f"Stage 3 locked success criteria pass: `{passed}`.",
        ]
    )
    if passed:
        lines.append("The locked Stage 3 criteria passed for this benchmark run. The most conservative valid claim is limited to this real program-analysis import-restoration patch-selection benchmark; it is not evidence of open-ended code repair success.")
    else:
        lines.append("Do not claim Stage 3 success. Treat this as an incomplete or failed hard-validation result until every locked criterion above passes with no hidden failed seeds.")
    return "\n".join(lines) + "\n"


def _audit_rows_for_seed(row: Dict[str, object], split_leakage: Dict[str, object], output_leakage: Dict[str, object]) -> List[Dict[str, object]]:
    seed = int(row["seed"])
    rows = [
        {"event": "stage3_seed_result", "seed": seed, "candidate": row.get("candidate"), "status": row.get("status"), "test_accuracy": row.get("test_accuracy"), "dev_accuracy": row.get("dev_accuracy")},
        {"event": "stage3_split_leakage_audit", "seed": seed, **split_leakage},
        {"event": "stage3_output_leakage_audit", "seed": seed, **output_leakage},
        {"event": "stage3_trainable_training_audit", "seed": seed, **row.get("trainable_audit", {})},
        {"event": "stage3_frozen_training_audit", "seed": seed, **row.get("frozen_audit", {})},
        {"event": "stage3_message_collapse", "seed": seed, **row.get("message_collapse", {})},
        {"event": "stage3_per_role_ablation", "seed": seed, **row.get("per_role_ablation", {})},
        {"event": "stage3_per_family_accuracy", "seed": seed, "per_family_accuracy": row.get("per_family_accuracy", {})},
        {"event": "stage3_cuda_memory", "seed": seed, "cuda_max_memory_allocated": row.get("cuda_max_memory_allocated"), "cuda_max_memory_allocated_gb": row.get("cuda_max_memory_allocated_gb"), "cuda_device_name": row.get("cuda_device_name")},
    ]
    return rows


def _compact_leakage(leakage: Dict[str, object]) -> Dict[str, object]:
    return {key: value for key, value in leakage.items() if not key.endswith("_ids")}


def _failure_row(seed: int, batch_size: int, accumulation: int, status: str, exc: BaseException) -> Dict[str, object]:
    return {
        "event": "stage3_seed_failure",
        "seed": int(seed),
        "status": status,
        "batch_size": int(batch_size),
        "gradient_accumulation_steps": int(accumulation),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "recorded_at_utc": _now(),
    }


def _is_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in message or "cuda oom" in message


def _clear_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _mean(values: Sequence[float]) -> float:
    return mean(values) if values else 0.0


def _audit_float(audit: Dict[str, object], key: str, default: float) -> float:
    value = audit.get(key)
    if value is None:
        return float(default)
    return float(value)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    main()
