from __future__ import annotations

import argparse
import json
import math
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
    split_leakage_audit,
)
from src.experiments.architecture_search import _bootstrap_ci, _candidate_specs, _fit_baselines
from src.experiments.real_shared_weight_latent_coordination import BENCHMARK
from src.experiments.run_stage3_gpu_hard_validation import (
    ARCHITECTURE,
    CORRUPTION_NEAR_CHANCE_GATES,
    INVARIANCE_PRESERVE_ACCURACY_GATES,
    _audit_float,
    _clear_cuda,
    _compact_leakage,
    _configure_cuda,
    _evaluate_candidate_with_checkpoints,
    _failure_row,
    _is_oom,
    _leakage_passes,
    _load_result,
    _seed_criteria,
    _stage_from_config,
)
from src.experiments.run_stage33_role_balanced_dataset_diagnostics import (
    stage33_dataset_config,
    _accuracy,
    _structured_oracle_predictions,
)


DEFAULT_CONFIG = "configs/stage33_role_balanced_real_code_cuda.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 3.3 role-balanced hardened real-code validation.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_stage33(config, config_path=config_path)


def run_stage33(config: Dict[str, object], config_path: Path) -> Dict[str, object]:
    device, hardware = _configure_cuda(config)
    output_path = Path(str(config["output_path"]))
    audit_path = Path(str(config["audit_log_path"]))
    report_path = Path(str(config["report_path"]))
    result = _load_result(output_path)
    result.setdefault("metadata", _metadata(config, config_path, hardware, device))
    result["metadata"].update(_metadata(config, config_path, hardware, device))
    result["config"] = config
    result["dataset_diagnostics"] = _load_dataset_diagnostics(config)
    result["role_diagnosis"] = _load_role_diagnosis()
    result.setdefault("stage33_rows", [])
    result.setdefault("failed_or_interrupted_seeds", [])
    result.setdefault("oom_retries", [])
    result["stage33_rows"] = [
        row
        for row in result["stage33_rows"]
        if row.get("status") == "completed" and row.get("candidate") == ARCHITECTURE and _row_has_required_checkpoints(row, config)
    ]

    completed = {
        int(row["seed"])
        for row in result["stage33_rows"]
        if row.get("status") == "completed" and row.get("candidate") == ARCHITECTURE and _row_has_required_checkpoints(row, config)
    }
    stage_base = _stage_from_config(config)
    seeds = [int(value) for value in config["stage"]["seeds"]]
    candidate = next(candidate for candidate in _candidate_specs() if candidate.name == ARCHITECTURE)
    fallbacks = [int(value) for value in config.get("gpu_safety", {}).get("batch_size_fallbacks", [32, 16, 8])]
    effective_batch = int(config.get("gpu_safety", {}).get("effective_batch_size", fallbacks[0]))

    for seed in seeds:
        if seed in completed:
            print(f"stage33 seed={seed}: existing completed row found; skipping")
            continue
        result["stage33_rows"] = [row for row in result["stage33_rows"] if int(row.get("seed", -1)) != seed]
        print(f"stage33 seed={seed}: starting role-balanced CUDA validation")
        seed_completed = False
        for batch_size in fallbacks:
            accumulation = max(1, int(math.ceil(effective_batch / float(batch_size))))
            stage = replace(stage_base, seeds=(seed,), batch_size=batch_size, gradient_accumulation_steps=accumulation)
            try:
                row = _run_seed(stage, seed, candidate, device, hardware, config)
                result["stage33_rows"].append(row)
                completed.add(seed)
                seed_completed = True
                _write_outputs(result, output_path, audit_path, report_path)
                print(
                    "stage33 seed={seed}: completed trainable={trainable:.4f} frozen={frozen:.4f} "
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
                    print(f"stage33 seed={seed}: CUDA OOM at batch_size={batch_size}; retrying smaller batch")
                    continue
                failure = _failure_row(seed, batch_size, accumulation, "failed", exc)
                result["failed_or_interrupted_seeds"].append(failure)
                _clear_cuda()
                _write_outputs(result, output_path, audit_path, report_path)
                print(f"stage33 seed={seed}: failed; see results for traceback")
                break
            finally:
                _clear_cuda()
        if not seed_completed and bool(config.get("stop_on_failed_seed", False)):
            break

    _write_outputs(result, output_path, audit_path, report_path)
    return result


def _run_seed(stage, seed: int, candidate, device: str, hardware: Dict[str, object], config: Dict[str, object]) -> Dict[str, object]:
    _clear_cuda()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    dataset_config = stage33_dataset_config({**config, "stage": {**config["stage"], "batch_size": stage.batch_size}})
    print(f"stage33 seed={seed}: building role-balanced real-code splits")
    splits = build_multiview_code_patch_splits(dataset_config, seed=seed, repo_root=Path("."))
    print(f"stage33 seed={seed}: fitting baselines")
    baselines = _fit_baselines(stage, splits, seed=seed, device=device)
    split_leakage = split_leakage_audit(BENCHMARK, seed, splits)
    output_leakage = output_leakage_audit(BENCHMARK, seed, splits["train"] + splits["dev"] + splits["test"])
    print(f"stage33 seed={seed}: fitting locked {ARCHITECTURE}")
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
    _add_structured_oracle_metrics(row, splits)
    cuda_peak = int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") and torch.cuda.is_available() else 0
    row.update(
        {
            "status": "completed",
            "completed_at_utc": _now(),
            "stage": "stage33_role_balanced_real_code",
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
            "mixed_precision": stage.mixed_precision,
            "batch_size": stage.batch_size,
            "gradient_accumulation_steps": stage.gradient_accumulation_steps,
            "checkpoint_paths": checkpoint_paths,
            "hardened_dataset_mode": "real_import_restore_role_balanced_v2",
        }
    )
    row["success_criteria"] = _seed_criteria(row, float(config.get("control_chance_tolerance", 0.10)))
    return row


def _add_structured_oracle_metrics(row: Dict[str, object], splits: Dict[str, Sequence[object]]) -> None:
    for split in ("dev", "test"):
        examples = list(splits[split])
        labels = np.asarray([example.label for example in examples], dtype=np.int64)
        values = row.setdefault(f"{split}_accuracy", {})
        for role_id in range(4):
            values[f"single_role_structured_oracle_{role_id}"] = _accuracy(_structured_oracle_predictions(examples, [role_id]), labels)
        for left in range(4):
            for right in range(left + 1, 4):
                values[f"pairwise_role_structured_oracle_{left}_{right}"] = _accuracy(_structured_oracle_predictions(examples, [left, right]), labels)
        values["all_role_structured_oracle"] = _accuracy(_structured_oracle_predictions(examples, [0, 1, 2, 3]), labels)


def _metadata(config: Dict[str, object], config_path: Path, hardware: Dict[str, object], device: str) -> Dict[str, object]:
    return {
        "benchmark": BENCHMARK,
        "task": "stage33_role_balanced_real_code_import_restoration",
        "architecture": ARCHITECTURE,
        "architecture_selection": "locked before Stage 3.3",
        "architecture_changes": "forbidden",
        "dataset": "real_import_restore_role_balanced_v2",
        "config_path": str(config_path),
        "device": device,
        "created_or_updated_at_utc": _now(),
        "hardware": hardware,
        "success_claim": "not evaluated until all Stage 3.3 gates pass",
    }


def _row_has_required_checkpoints(row: Dict[str, object], config: Dict[str, object]) -> bool:
    if not bool(config.get("save_checkpoints", True)):
        return True
    if row.get("dataset_summary", {}).get("dataset_source") != "real_import_restore_role_balanced_v2":
        return False
    expected_version = str(dict(config.get("dataset_config", {})).get("generator_version", "import_restore_role_balanced_v2_stage33b"))
    if str(row.get("dataset_summary", {}).get("generator_version", "")) != expected_version:
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


def _write_outputs(result: Dict[str, object], output_path: Path, audit_path: Path, report_path: Path) -> None:
    tolerance = float(result.get("config", {}).get("control_chance_tolerance", 0.10))
    for row in result.get("stage33_rows", []):
        if row.get("status") == "completed":
            row["success_criteria"] = _seed_criteria(row, tolerance)
    result["dataset_diagnostics"] = _load_dataset_diagnostics(result.get("config", {}))
    result["role_diagnosis"] = _load_role_diagnosis()
    result["summary"] = _summary(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(output_path)

    audit_rows: List[Dict[str, object]] = []
    for row in result.get("stage33_rows", []):
        audit_rows.extend(_audit_rows_for_seed(row, row.get("split_leakage_audit", {}), row.get("output_leakage_audit", {})))
    audit_rows.extend(result.get("oom_retries", []))
    audit_rows.extend(result.get("failed_or_interrupted_seeds", []))
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in audit_rows) + ("\n" if audit_rows else ""), encoding="utf-8")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(result), encoding="utf-8")


def _summary(result: Dict[str, object]) -> Dict[str, object]:
    rows = [row for row in result.get("stage33_rows", []) if row.get("status") == "completed"]
    rows.sort(key=lambda row: int(row["seed"]))
    deltas = [float(row["test_delta"]) for row in rows]
    ci_low, ci_high = _bootstrap_ci(deltas) if deltas else (0.0, 0.0)
    trainable_mean = _mean([float(row["test_accuracy"]["trainable"]) for row in rows])
    frozen_mean = _mean([float(row["test_accuracy"]["frozen"]) for row in rows])
    threshold = 1.0 / 8.0 + float(result.get("config", {}).get("control_chance_tolerance", 0.10))
    single_view_values = {
        f"single_view_text_role_{role_id}": [float(row["test_accuracy"].get(f"single_view_text_role_{role_id}", 0.0)) for row in rows]
        for role_id in range(4)
    }
    corruption_values = {
        name: [float(row["test_accuracy"].get(name, 0.0)) for row in rows]
        for name in CORRUPTION_NEAR_CHANCE_GATES
    }
    invariance_values = {
        name: [float(row["test_accuracy"].get(name, 0.0)) for row in rows]
        for name in INVARIANCE_PRESERVE_ACCURACY_GATES
    }
    role_shuffle_values = [float(row["test_accuracy"].get("role_labels_shuffled", 0.0)) for row in rows]
    diagnostics_summary = result.get("dataset_diagnostics", {}).get("summary", {})
    max_single_structured = float(diagnostics_summary.get("max_mean_single_role_structured_oracle_accuracy", 1.0))
    max_pairwise_structured = float(diagnostics_summary.get("max_mean_pairwise_role_structured_oracle_accuracy", 0.0))
    all_role_structured = float(diagnostics_summary.get("mean_all_role_structured_oracle_accuracy", 0.0))
    role_failures = sum(bool(row.get("per_role_ablation", {}).get("one_role_only_failure", True)) for row in rows)

    success_criteria = [
        {"criterion": "pre-training role-balanced dataset diagnostics passed", "pass": bool(diagnostics_summary.get("pretraining_dataset_validity_passed", False)), "value": diagnostics_summary.get("gate_table", [])},
        {"criterion": "completed Stage 3.3 seeds >= 10", "pass": len(rows) >= 10, "value": len(rows)},
        {"criterion": "trainable beats frozen on at least 8/10 seeds", "pass": len(rows) >= 10 and sum(delta > 0.0 for delta in deltas) >= 8, "value": sum(delta > 0.0 for delta in deltas)},
        {"criterion": "mean trainable-frozen delta >= +0.20", "pass": _mean(deltas) >= 0.20, "value": _mean(deltas)},
        {"criterion": "bootstrap 95% CI lower bound for delta > +0.05", "pass": ci_low > 0.05, "value": [ci_low, ci_high]},
        {
            "criterion": "trainable beats text-only, raw-latent, majority, candidate-order, and full-context baselines by mean accuracy",
            "pass": bool(
                trainable_mean > _mean([float(row["test_accuracy"]["text_only"]) for row in rows])
                and trainable_mean > _mean([float(row["test_accuracy"]["raw_latent"]) for row in rows])
                and trainable_mean > _mean([float(row["test_accuracy"]["majority_baseline"]) for row in rows])
                and trainable_mean > _mean([float(row["test_accuracy"]["candidate_order_baseline"]) for row in rows])
                and trainable_mean > _mean([float(row["test_accuracy"].get("single_agent_full_context", 0.0)) for row in rows])
            ),
            "value": {
                "trainable_mean": trainable_mean,
                "text_only_mean": _mean([float(row["test_accuracy"]["text_only"]) for row in rows]),
                "raw_latent_mean": _mean([float(row["test_accuracy"]["raw_latent"]) for row in rows]),
                "majority_mean": _mean([float(row["test_accuracy"]["majority_baseline"]) for row in rows]),
                "candidate_order_mean": _mean([float(row["test_accuracy"]["candidate_order_baseline"]) for row in rows]),
                "full_context_mean": _mean([float(row["test_accuracy"].get("single_agent_full_context", 0.0)) for row in rows]),
            },
        },
        {"criterion": "trainable beats every single-view baseline by mean accuracy", "pass": all(trainable_mean > _mean(values) for values in single_view_values.values()), "value": {name: _mean(values) for name, values in single_view_values.items()}},
        {"criterion": "corruption controls plus majority/candidate-order remain near chance", "pass": all(_mean(values) <= threshold for values in corruption_values.values()), "value": {name: {"mean_accuracy": _mean(values), "max_accuracy": max(values or [0.0])} for name, values in corruption_values.items()}},
        {"criterion": "invariance controls preserve accuracy", "pass": all(_mean(values) >= trainable_mean - 0.05 for values in invariance_values.values()), "value": {name: {"mean_accuracy": _mean(values), "mean_delta_from_trainable": _mean(values) - trainable_mean} for name, values in invariance_values.items()}},
        {"criterion": "role-label shuffle drops substantially or remains near chance", "pass": _mean(role_shuffle_values) <= threshold or _mean(role_shuffle_values) <= trainable_mean - 0.15, "value": {"mean_accuracy": _mean(role_shuffle_values), "max_accuracy": max(role_shuffle_values or [0.0]), "mean_delta_from_trainable": _mean(role_shuffle_values) - trainable_mean}},
        {"criterion": "candidate lexical-overlap baseline <= 0.225", "pass": float(diagnostics_summary.get("mean_candidate_lexical_overlap_accuracy", 1.0)) <= 0.225, "value": diagnostics_summary.get("mean_candidate_lexical_overlap_accuracy")},
        {"criterion": "static import-frequency baseline <= 0.300", "pass": float(diagnostics_summary.get("mean_static_import_frequency_accuracy", 1.0)) <= 0.300, "value": diagnostics_summary.get("mean_static_import_frequency_accuracy")},
        {"criterion": "no single-role structured oracle > 0.35 mean accuracy", "pass": max_single_structured <= 0.35, "value": diagnostics_summary.get("mean_single_role_structured_oracle_accuracy")},
        {"criterion": "at least one pairwise-role structured oracle > 0.70 mean accuracy", "pass": max_pairwise_structured > 0.70, "value": diagnostics_summary.get("mean_pairwise_role_structured_oracle_accuracy")},
        {"criterion": "all-role structured oracle >= 0.90", "pass": all_role_structured >= 0.90, "value": all_role_structured},
        {"criterion": "no single role explains trainable result in more than 1/10 seeds", "pass": role_failures <= 1 and len(rows) >= 10, "value": role_failures},
        {"criterion": "leakage audits pass", "pass": all(bool(row.get("split_leakage_audit_passes")) and bool(row.get("output_leakage_audit_passes")) and bool(row.get("split_leakage_audit", {}).get("candidate_patch_hash_leakage_audit_passes", False)) for row in rows), "value": "split_candidate_patch_hash_and_output"},
        {"criterion": "shared trainable model receives gradients and changes", "pass": all(float(row.get("trainable_audit", {}).get("agent_grad_norm_mean") or 0.0) > 0.0 and float(row.get("trainable_audit", {}).get("agent_parameter_delta") or 0.0) > 0.0 for row in rows), "value": "all_completed_seeds"},
        {"criterion": "frozen comparator remains frozen", "pass": all(_audit_float(row.get("frozen_audit", {}), "agent_grad_norm_mean", 1.0) == 0.0 and _audit_float(row.get("frozen_audit", {}), "agent_parameter_delta", 1.0) == 0.0 for row in rows), "value": "all_completed_seeds"},
    ]
    return {
        "completed_seeds": [int(row["seed"]) for row in rows],
        "n_completed": len(rows),
        "n_failed_or_interrupted": len(result.get("failed_or_interrupted_seeds", [])),
        "mean_trainable_accuracy": trainable_mean,
        "mean_frozen_accuracy": frozen_mean,
        "mean_delta": _mean(deltas),
        "std_delta": float(pstdev(deltas)) if len(deltas) > 1 else 0.0,
        "bootstrap_95_ci_delta": [ci_low, ci_high],
        "success_criteria": success_criteria,
        "passed_stage33_role_balanced_success_criteria": bool(success_criteria and all(bool(item["pass"]) for item in success_criteria)),
    }


def _render_report(result: Dict[str, object]) -> str:
    rows = [row for row in result.get("stage33_rows", []) if row.get("status") == "completed"]
    rows.sort(key=lambda row: int(row["seed"]))
    summary = result.get("summary", {})
    diagnostics = result.get("dataset_diagnostics", {}).get("summary", {})
    diagnosis = result.get("role_diagnosis", {}).get("summary", {})
    config = result.get("config", {})
    stage = config.get("stage", {})
    dataset_config = config.get("dataset_config", {})
    lines = [
        "# Stage 3.3 Role-Balanced Real-Code Import-Restoration Validation",
        "",
        "## Scope/Config",
        "",
        f"- Architecture: `{ARCHITECTURE}`",
        "- Locked architecture unchanged: topk_attention_no_head, shared-weight cloned agent, active/top-k token readout, no message head, no private-cue auxiliary loss, candidate-query coordinator.",
        "- Comparator: exact frozen same-architecture comparator.",
        f"- Dataset mode: `real_import_restore_role_balanced_v2` / `{dataset_config.get('generator_version', 'import_restore_role_balanced_v2_stage33b')}`.",
        f"- Split sizes: train `{stage.get('n_train')}`, dev `{stage.get('n_dev')}`, test `{stage.get('n_test')}`; seeds `{stage.get('seeds')}`.",
        f"- Mixed precision: `{stage.get('mixed_precision')}`.",
        "",
        "## Stage 3.2 Role Diagnosis",
        "",
        f"- Failed Stage 3.2 seeds: `{diagnosis.get('failed_seeds', [])}`",
        f"- Masked-critical roles by seed: `{diagnosis.get('masked_critical_roles_by_seed', {})}`",
        f"- Single-role-sufficient roles by seed: `{diagnosis.get('single_role_sufficient_roles_by_seed', {})}`",
        "- Diagnosis result: no exact text leakage; failures were model/protocol role dependence plus family concentration. Full examples are in `results/stage33_role_diagnosis.json`.",
        "",
        "## Pre-Training Dataset Diagnostics",
        "",
        f"- Dataset validity passed: `{bool(diagnostics.get('pretraining_dataset_validity_passed', False))}`",
        f"- Candidate lexical-overlap baseline: `{float(diagnostics.get('mean_candidate_lexical_overlap_accuracy', 0.0)):.4f}`",
        f"- Static import-frequency baseline: `{float(diagnostics.get('mean_static_import_frequency_accuracy', 0.0)):.4f}`",
        f"- Max single-view text baseline: `{float(diagnostics.get('max_single_view_accuracy', 0.0)):.4f}`",
        f"- Single-role structured oracle means: `{diagnostics.get('mean_single_role_structured_oracle_accuracy', {})}`",
        f"- Pairwise-role structured oracle means: `{diagnostics.get('mean_pairwise_role_structured_oracle_accuracy', {})}`",
        f"- All-role structured oracle: `{float(diagnostics.get('mean_all_role_structured_oracle_accuracy', 0.0)):.4f}`",
        "",
        "## Per-Seed Accuracy",
        "",
        "| seed | batch | trainable | frozen | delta | text | raw | majority | cand-order | max single-view | full ctx | random | hidden shuffle | view masked | view shuffled | role shuffle | physical order | cand-order shuffle | all-role oracle |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        test = row["test_accuracy"]
        max_single = max(float(test.get(f"single_view_text_role_{index}", 0.0)) for index in range(4))
        lines.append(
            "| {seed} | {batch} | {train:.4f} | {frozen:.4f} | {delta:.4f} | {text:.4f} | {raw:.4f} | {maj:.4f} | {corder:.4f} | {single:.4f} | {full:.4f} | {rand:.4f} | {hidden:.4f} | {masked:.4f} | {view:.4f} | {role:.4f} | {physical:.4f} | {candidate:.4f} | {oracle:.4f} |".format(
                seed=row["seed"],
                batch=row.get("batch_size"),
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
                oracle=float(test.get("all_role_structured_oracle", test.get("explicit_evidence_oracle", 0.0))),
            )
        )
    ci = summary.get("bootstrap_95_ci_delta", [0.0, 0.0])
    lines.extend(
        [
            "",
            "## Mean/CI",
            "",
            f"- Mean trainable accuracy: `{float(summary.get('mean_trainable_accuracy', 0.0)):.4f}`",
            f"- Mean frozen accuracy: `{float(summary.get('mean_frozen_accuracy', 0.0)):.4f}`",
            f"- Mean delta: `{float(summary.get('mean_delta', 0.0)):.4f}`",
            f"- Bootstrap 95% CI: `[{float(ci[0]):.4f}, {float(ci[1]):.4f}]`",
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
        lines.append(f"| {row['seed']} | {float(ablation.get('base_accuracy', 0.0)):.4f} | {float(masked.get('0', 0.0)):.4f} | {float(masked.get('1', 0.0)):.4f} | {float(masked.get('2', 0.0)):.4f} | {float(masked.get('3', 0.0)):.4f} | {bool(ablation.get('one_role_only_failure', False))} |")
    lines.extend(["", "## Gate Table", "", "| criterion | pass | value |", "|---|---|---|"])
    for item in summary.get("success_criteria", []):
        value = item.get("value")
        if isinstance(value, (dict, list)):
            value = json.dumps(value, sort_keys=True)[:900]
        lines.append(f"| {item.get('criterion')} | {bool(item.get('pass'))} | `{value}` |")
    passed = bool(summary.get("passed_stage33_role_balanced_success_criteria", False))
    lines.extend(["", "## Conservative Interpretation", "", f"Stage 3.3 role-balanced criteria pass: `{passed}`."])
    if passed:
        lines.append("Allowed claim: the locked architecture passed this role-balanced import-restoration patch-selection benchmark only. This is not open-ended code repair evidence.")
    else:
        lines.append("Do not claim success. Failed gates remain visible in the table above.")
    return "\n".join(lines) + "\n"


def _audit_rows_for_seed(row: Dict[str, object], split_leakage: Dict[str, object], output_leakage: Dict[str, object]) -> List[Dict[str, object]]:
    seed = int(row["seed"])
    return [
        {"event": "stage33_seed_result", "seed": seed, "candidate": row.get("candidate"), "status": row.get("status"), "test_accuracy": row.get("test_accuracy"), "dev_accuracy": row.get("dev_accuracy")},
        {"event": "stage33_split_leakage_audit", "seed": seed, **split_leakage},
        {"event": "stage33_output_leakage_audit", "seed": seed, **output_leakage},
        {"event": "stage33_trainable_training_audit", "seed": seed, **row.get("trainable_audit", {})},
        {"event": "stage33_frozen_training_audit", "seed": seed, **row.get("frozen_audit", {})},
        {"event": "stage33_message_collapse", "seed": seed, **row.get("message_collapse", {})},
        {"event": "stage33_per_role_ablation", "seed": seed, **row.get("per_role_ablation", {})},
        {"event": "stage33_per_family_accuracy", "seed": seed, "per_family_accuracy": row.get("per_family_accuracy", {})},
        {"event": "stage33_cuda_memory", "seed": seed, "cuda_max_memory_allocated": row.get("cuda_max_memory_allocated"), "cuda_max_memory_allocated_gb": row.get("cuda_max_memory_allocated_gb"), "cuda_device_name": row.get("cuda_device_name")},
    ]


def _load_dataset_diagnostics(config: Dict[str, object]) -> Dict[str, object]:
    path = Path(str(config.get("dataset_diagnostics_output_path", "results/stage33_role_balanced_dataset_diagnostics.json")))
    if not path.exists():
        return {"available": False, "path": str(path)}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["available"] = True
        data["path"] = str(path)
        return data
    except Exception as exc:
        return {"available": False, "path": str(path), "error": str(exc)}


def _load_role_diagnosis() -> Dict[str, object]:
    path = Path("results/stage33_role_diagnosis.json")
    if not path.exists():
        return {"available": False, "path": str(path)}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["available"] = True
        data["path"] = str(path)
        return data
    except Exception as exc:
        return {"available": False, "path": str(path), "error": str(exc)}


def _mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    main()
