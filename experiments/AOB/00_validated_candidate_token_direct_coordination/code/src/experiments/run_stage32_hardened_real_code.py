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
    randomized_labels_for_examples,
    split_leakage_audit,
)
from src.experiments.architecture_search import (
    _bootstrap_ci,
    _candidate_specs,
    _fit_baselines,
)
from src.experiments.real_shared_weight_latent_coordination import BENCHMARK
from src.experiments.run_stage3_gpu_hard_validation import (
    ARCHITECTURE,
    CORRUPTION_NEAR_CHANCE_GATES,
    INVARIANCE_PRESERVE_ACCURACY_GATES,
    _clear_cuda,
    _compact_leakage,
    _configure_cuda,
    _audit_float,
    _evaluate_candidate_with_checkpoints,
    _failure_row,
    _is_oom,
    _load_result,
    _mean,
    _seed_criteria,
    _stage_from_config,
)
from src.experiments.run_stage32_hardened_dataset_diagnostics import (
    stage32_dataset_config,
)


DEFAULT_CONFIG = "configs/stage32_hardened_real_code_cuda.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 3.2 hardened real-code import-restoration validation.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_stage32(config, config_path=config_path)


def run_stage32(config: Dict[str, object], config_path: Path) -> Dict[str, object]:
    device, hardware = _configure_cuda(config)
    output_path = Path(str(config["output_path"]))
    audit_path = Path(str(config["audit_log_path"]))
    report_path = Path(str(config["report_path"]))
    result = _load_result(output_path)
    result.setdefault("metadata", _metadata(config, config_path, hardware, device))
    result["metadata"].update(_metadata(config, config_path, hardware, device))
    result["config"] = config
    result["dataset_diagnostics"] = _load_dataset_diagnostics(config)
    result.setdefault("stage32_rows", [])
    result.setdefault("failed_or_interrupted_seeds", [])
    result.setdefault("oom_retries", [])
    result["stage32_rows"] = [
        row
        for row in result["stage32_rows"]
        if row.get("status") == "completed" and row.get("candidate") == ARCHITECTURE and _row_has_required_checkpoints(row, config)
    ]

    completed = {
        int(row["seed"])
        for row in result["stage32_rows"]
        if row.get("status") == "completed" and row.get("candidate") == ARCHITECTURE and _row_has_required_checkpoints(row, config)
    }
    stage_base = _stage_from_config(config)
    seeds = [int(value) for value in config["stage"]["seeds"]]
    candidate = next(candidate for candidate in _candidate_specs() if candidate.name == ARCHITECTURE)
    fallbacks = [int(value) for value in config.get("gpu_safety", {}).get("batch_size_fallbacks", [32, 16, 8])]
    effective_batch = int(config.get("gpu_safety", {}).get("effective_batch_size", fallbacks[0]))

    for seed in seeds:
        if seed in completed:
            print(f"stage32 seed={seed}: existing completed row found; skipping")
            continue
        result["stage32_rows"] = [row for row in result["stage32_rows"] if int(row.get("seed", -1)) != seed]
        print(f"stage32 seed={seed}: starting hardened CUDA validation")
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
                row = _run_seed(stage, seed, candidate, device, hardware, config)
                result["stage32_rows"].append(row)
                completed.add(seed)
                seed_completed = True
                _write_outputs(result, output_path, audit_path, report_path)
                print(
                    "stage32 seed={seed}: completed trainable={trainable:.4f} frozen={frozen:.4f} "
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
                    print(f"stage32 seed={seed}: CUDA OOM at batch_size={batch_size}; retrying smaller batch")
                    continue
                failure = _failure_row(seed, batch_size, accumulation, "failed", exc)
                result["failed_or_interrupted_seeds"].append(failure)
                _clear_cuda()
                _write_outputs(result, output_path, audit_path, report_path)
                print(f"stage32 seed={seed}: failed; see results for traceback")
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
    dataset_config = stage32_dataset_config({**config, "stage": {**config["stage"], "batch_size": stage.batch_size}})
    print(f"stage32 seed={seed}: building hardened real-code splits")
    splits = build_multiview_code_patch_splits(dataset_config, seed=seed, repo_root=Path("."))
    print(f"stage32 seed={seed}: fitting baselines")
    baselines = _fit_baselines(stage, splits, seed=seed, device=device)
    split_leakage = split_leakage_audit(BENCHMARK, seed, splits)
    output_leakage = output_leakage_audit(BENCHMARK, seed, splits["train"] + splits["dev"] + splits["test"])
    print(f"stage32 seed={seed}: fitting locked {ARCHITECTURE}")
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
            "stage": "stage32_hardened_real_code",
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
            "hardened_dataset_mode": "real_import_restore_hardened",
        }
    )
    row["success_criteria"] = _seed_criteria(row, float(config.get("control_chance_tolerance", 0.10)))
    return row


def _metadata(config: Dict[str, object], config_path: Path, hardware: Dict[str, object], device: str) -> Dict[str, object]:
    return {
        "benchmark": BENCHMARK,
        "task": "stage32_hardened_real_code_import_restoration",
        "architecture": ARCHITECTURE,
        "architecture_selection": "locked before Stage 3.2",
        "architecture_changes": "forbidden",
        "dataset": "real_import_restore_hardened",
        "config_path": str(config_path),
        "device": device,
        "created_or_updated_at_utc": _now(),
        "hardware": hardware,
        "success_claim": "not evaluated until all Stage 3.2 gates pass",
    }


def _load_dataset_diagnostics(config: Dict[str, object]) -> Dict[str, object]:
    path = Path(str(config.get("dataset_diagnostics_output_path", "results/stage32_hardened_dataset_diagnostics.json")))
    if not path.exists():
        return {"available": False, "path": str(path)}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["available"] = True
        data["path"] = str(path)
        return data
    except Exception as exc:
        return {"available": False, "path": str(path), "error": str(exc)}


def _row_has_required_checkpoints(row: Dict[str, object], config: Dict[str, object]) -> bool:
    if not bool(config.get("save_checkpoints", True)):
        return True
    if row.get("dataset_summary", {}).get("dataset_source") != "real_import_restore_hardened":
        return False
    if not str(row.get("dataset_summary", {}).get("generator_version", "")).startswith("import_restore_redacted_v1"):
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
    for row in result.get("stage32_rows", []):
        if row.get("status") == "completed":
            row["success_criteria"] = _seed_criteria(row, tolerance)
    result["dataset_diagnostics"] = _load_dataset_diagnostics(result.get("config", {}))
    result["summary"] = _summary(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(output_path)

    audit_rows: List[Dict[str, object]] = []
    for row in result.get("stage32_rows", []):
        audit_rows.extend(_audit_rows_for_seed(row, row.get("split_leakage_audit", {}), row.get("output_leakage_audit", {})))
    audit_rows.extend(result.get("oom_retries", []))
    audit_rows.extend(result.get("failed_or_interrupted_seeds", []))
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in audit_rows) + ("\n" if audit_rows else ""), encoding="utf-8")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(result), encoding="utf-8")


def _summary(result: Dict[str, object]) -> Dict[str, object]:
    rows = [row for row in result.get("stage32_rows", []) if row.get("status") == "completed"]
    deltas = [float(row["test_delta"]) for row in rows]
    trainable = [float(row["test_accuracy"]["trainable"]) for row in rows]
    frozen = [float(row["test_accuracy"]["frozen"]) for row in rows]
    ci_low, ci_high = _bootstrap_ci(deltas) if deltas else (0.0, 0.0)
    criteria = _overall_criteria(rows, result.get("failed_or_interrupted_seeds", []), result.get("dataset_diagnostics", {}))
    return {
        "completed_seeds": [int(row["seed"]) for row in rows],
        "n_completed": len(rows),
        "n_failed_or_interrupted": len(result.get("failed_or_interrupted_seeds", [])),
        "mean_trainable_accuracy": _mean(trainable),
        "mean_frozen_accuracy": _mean(frozen),
        "mean_delta": _mean(deltas),
        "std_delta": pstdev(deltas) if len(deltas) > 1 else 0.0,
        "bootstrap_95_ci_delta": [ci_low, ci_high],
        "passed_stage32_hardened_success_criteria": all(bool(item["pass"]) for item in criteria) if criteria else False,
        "success_criteria": criteria,
    }


def _overall_criteria(
    rows: Sequence[Dict[str, object]],
    failures: Sequence[Dict[str, object]],
    diagnostics: Dict[str, object],
) -> List[Dict[str, object]]:
    if not rows:
        return [{"criterion": "completed Stage 3.2 seeds >= 10", "pass": False, "value": 0}]
    deltas = [float(row["test_delta"]) for row in rows]
    ci_low, ci_high = _bootstrap_ci(deltas)
    chance = 1.0 / 8.0
    threshold = chance + 0.10
    corruption_values = {name: [float(row["test_accuracy"].get(name, 1.0)) for row in rows] for name in CORRUPTION_NEAR_CHANCE_GATES}
    invariance_values = {name: [float(row["test_accuracy"].get(name, 0.0)) for row in rows] for name in INVARIANCE_PRESERVE_ACCURACY_GATES}
    trainable_values = [float(row["test_accuracy"]["trainable"]) for row in rows]
    trainable_mean = _mean(trainable_values)
    single_view_names = tuple(f"single_view_text_role_{index}" for index in range(4))
    single_view_values = {name: [float(row["test_accuracy"].get(name, 0.0)) for row in rows] for name in single_view_names}
    role_shuffle_values = corruption_values.get("role_labels_shuffled", [1.0])
    dataset_valid = bool(diagnostics.get("summary", {}).get("pretraining_dataset_validity_passed", False))
    return [
        {"criterion": "pre-training shortcut diagnostics passed", "pass": dataset_valid, "value": diagnostics.get("summary", {}).get("gate_table", "missing")},
        {"criterion": "completed Stage 3.2 seeds >= 10", "pass": len(rows) >= 10 and not failures, "value": len(rows)},
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
        {
            "criterion": "trainable beats every single-view baseline by mean accuracy",
            "pass": all(trainable_mean > _mean(values) for values in single_view_values.values()),
            "value": {name: _mean(values) for name, values in single_view_values.items()},
        },
        {
            "criterion": "corruption controls plus majority/candidate-order remain near chance",
            "pass": all(_mean(values) <= threshold for values in corruption_values.values()),
            "value": {name: {"mean_accuracy": _mean(values), "max_accuracy": max(values or [0.0])} for name, values in corruption_values.items()},
        },
        {
            "criterion": "invariance controls preserve accuracy",
            "pass": all(_mean(values) >= trainable_mean - 0.05 for values in invariance_values.values()),
            "value": {name: {"mean_accuracy": _mean(values), "mean_delta_from_trainable": _mean(values) - trainable_mean} for name, values in invariance_values.items()},
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
            "criterion": "explicit structured oracle remains high",
            "pass": _mean([float(row["test_accuracy"].get("explicit_evidence_oracle", 0.0)) for row in rows]) >= 0.90,
            "value": _mean([float(row["test_accuracy"].get("explicit_evidence_oracle", 0.0)) for row in rows]),
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


def _render_report(result: Dict[str, object]) -> str:
    rows = [row for row in result.get("stage32_rows", []) if row.get("status") == "completed"]
    summary = result.get("summary", {})
    diagnostics = result.get("dataset_diagnostics", {})
    metadata = result.get("metadata", {})
    config = result.get("config", {})
    stage = config.get("stage", {})
    lines = [
        "# Stage 3.2 Hardened Real-Code Import-Restoration Validation",
        "",
        "## Scope/Config",
        "",
        f"- Architecture: `{ARCHITECTURE}`",
        "- Locked architecture: topk_attention_no_head, shared-weight cloned agent, active/top-k token readout, no message head, no private-cue auxiliary loss, candidate-query coordinator.",
        "- Comparator: exact frozen same-architecture comparator.",
        f"- Dataset mode: `real_import_restore_hardened` / `import_restore_redacted_v1`",
        f"- Device requested/used: `{metadata.get('device', config.get('device'))}`",
        f"- CUDA device: `{metadata.get('hardware', {}).get('cuda_device_name', 'n/a')}`",
        f"- Mixed precision: `{stage.get('mixed_precision', 'none')}`",
        f"- Split sizes: train `{stage.get('n_train')}`, dev `{stage.get('n_dev')}`, test `{stage.get('n_test')}`",
        f"- Seeds requested: `{stage.get('seeds')}`",
        "",
        "## Dataset Hardening",
        "",
        "- Private views replace exact symbol, module, import statement, target path, and candidate text with non-identifying typed placeholders.",
        "- Candidate text is redacted with stable non-semantic IDs; exact candidate identity is carried only through structured candidate-query features and audit metadata.",
        "- Distractors are sampled from real local import pairs by same package family, symbol type, module category, provider-name parity, usage pattern, and import-slot parity where possible.",
        "- Primary benchmark is not cross-file-only or two-hop-only.",
        "",
        "## Pre-Training Shortcut Diagnostics",
        "",
    ]
    diag_summary = diagnostics.get("summary", {})
    if diag_summary:
        lines.append(f"- Dataset validity passed: `{bool(diag_summary.get('pretraining_dataset_validity_passed', False))}`")
        lines.append(f"- Candidate lexical-overlap baseline: `{float(diag_summary.get('mean_candidate_lexical_overlap_accuracy', 0.0)):.4f}`")
        lines.append(f"- Static import-frequency baseline: `{float(diag_summary.get('mean_static_import_frequency_accuracy', 0.0)):.4f}`")
        lines.append(f"- Majority baseline: `{float(diag_summary.get('mean_majority_accuracy', 0.0)):.4f}`")
        lines.append(f"- Candidate-order baseline: `{float(diag_summary.get('mean_candidate_order_accuracy', 0.0)):.4f}`")
        lines.append(f"- Max single-view baseline: `{float(diag_summary.get('max_single_view_accuracy', 0.0)):.4f}`")
        lines.append(f"- Explicit structured oracle: `{float(diag_summary.get('mean_explicit_structured_oracle_accuracy', 0.0)):.4f}`")
    else:
        lines.append("- Dataset diagnostics were not found; Stage 3.2 is invalid until they pass.")
    lines.extend(
        [
            "",
            "## Completion",
            "",
            f"- Completed seeds: `{summary.get('completed_seeds', [])}`",
            f"- Failed or interrupted seeds: `{result.get('failed_or_interrupted_seeds', [])}`",
            f"- OOM retries: `{result.get('oom_retries', [])}`",
            "",
            "## Per-Seed Accuracy",
            "",
            "| seed | batch | max CUDA GB | trainable | frozen | delta | text | raw | majority | cand-order | max single-view | full ctx | random | hidden shuffle | view masked | view shuffled | role shuffle | physical order | cand-order shuffle | oracle |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        test = row["test_accuracy"]
        max_single = max(float(test.get(f"single_view_text_role_{index}", 0.0)) for index in range(4))
        lines.append(
            "| {seed} | {batch} | {mem:.3f} | {train:.4f} | {frozen:.4f} | {delta:.4f} | {text:.4f} | {raw:.4f} | {maj:.4f} | {corder:.4f} | {single:.4f} | {full:.4f} | {rand:.4f} | {hidden:.4f} | {masked:.4f} | {view:.4f} | {role:.4f} | {physical:.4f} | {candidate:.4f} | {oracle:.4f} |".format(
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
    ci = summary.get("bootstrap_95_ci_delta", [0.0, 0.0])
    lines.extend(
        [
            "",
            "## Mean/CI",
            "",
            f"- Mean trainable accuracy: `{float(summary.get('mean_trainable_accuracy', 0.0)):.4f}`",
            f"- Mean frozen accuracy: `{float(summary.get('mean_frozen_accuracy', 0.0)):.4f}`",
            f"- Mean delta: `{float(summary.get('mean_delta', 0.0)):.4f}`",
            f"- Delta std: `{float(summary.get('std_delta', 0.0)):.4f}`",
            f"- Bootstrap 95% CI: `[{float(ci[0]):.4f}, {float(ci[1]):.4f}]`",
            "",
            "## Corruption Controls",
            "",
            "| gate | mean accuracy | max accuracy | pass |",
            "|---|---:|---:|---|",
        ]
    )
    threshold = 1.0 / 8.0 + float(config.get("control_chance_tolerance", 0.10))
    for name in CORRUPTION_NEAR_CHANCE_GATES:
        values = [float(row["test_accuracy"].get(name, 0.0)) for row in rows]
        lines.append(f"| {name} | {_mean(values):.4f} | {(max(values) if values else 0.0):.4f} | {bool(_mean(values) <= threshold)} |")
    lines.extend(["", "## Invariance Controls", "", "| gate | mean accuracy | mean delta from trainable | pass |", "|---|---:|---:|---|"])
    trainable_mean = _mean([float(row["test_accuracy"]["trainable"]) for row in rows])
    for name in INVARIANCE_PRESERVE_ACCURACY_GATES:
        values = [float(row["test_accuracy"].get(name, 0.0)) for row in rows]
        lines.append(f"| {name} | {_mean(values):.4f} | {_mean(values) - trainable_mean:.4f} | {bool(_mean(values) >= trainable_mean - 0.05)} |")
    lines.extend(["", "## Single-View Baselines", "", "| seed | role 0 | role 1 | role 2 | role 3 |", "|---:|---:|---:|---:|---:|"])
    for row in rows:
        test = row["test_accuracy"]
        lines.append(f"| {row['seed']} | {float(test.get('single_view_text_role_0', 0.0)):.4f} | {float(test.get('single_view_text_role_1', 0.0)):.4f} | {float(test.get('single_view_text_role_2', 0.0)):.4f} | {float(test.get('single_view_text_role_3', 0.0)):.4f} |")
    lines.extend(["", "## Per-Role Ablation", "", "| seed | base | role 0 masked | role 1 masked | role 2 masked | role 3 masked | one-role failure |", "|---:|---:|---:|---:|---:|---:|---|"])
    for row in rows:
        ablation = row.get("per_role_ablation", {})
        masked = ablation.get("accuracy_when_role_masked", {})
        lines.append(f"| {row['seed']} | {float(ablation.get('base_accuracy', 0.0)):.4f} | {float(masked.get('0', 0.0)):.4f} | {float(masked.get('1', 0.0)):.4f} | {float(masked.get('2', 0.0)):.4f} | {float(masked.get('3', 0.0)):.4f} | {bool(ablation.get('one_role_only_failure', False))} |")
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
    lines.extend(["", "## Audit Summary", "", "| seed | split leak | output leak | shared id | train grad | train delta | frozen grad | frozen delta | active variance | active cosine |", "|---:|---|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        collapse = row.get("message_collapse", {}).get("active_message_readout", {})
        train_audit = row.get("trainable_audit", {})
        frozen_audit = row.get("frozen_audit", {})
        lines.append("| {seed} | {split} | {output} | {shared} | {tg:.4f} | {td:.4f} | {fg:.4f} | {fd:.4f} | {var:.4f} | {cos:.4f} |".format(seed=row["seed"], split="pass" if row.get("split_leakage_audit_passes") else "fail", output="pass" if row.get("output_leakage_audit_passes") else "fail", shared="pass" if train_audit.get("shared_parameter_identity") and frozen_audit.get("shared_parameter_identity") else "fail", tg=float(train_audit.get("agent_grad_norm_mean") or 0.0), td=float(train_audit.get("agent_parameter_delta") or 0.0), fg=float(frozen_audit.get("agent_grad_norm_mean") or 0.0), fd=float(frozen_audit.get("agent_parameter_delta") or 0.0), var=float(collapse.get("variance_mean", 0.0)), cos=float(collapse.get("mean_cosine_similarity", 0.0))))
    lines.extend(["", "## Gate Table", "", "| criterion | pass | value |", "|---|---|---|"])
    for item in summary.get("success_criteria", []):
        lines.append(f"| {item.get('criterion')} | {bool(item.get('pass'))} | `{item.get('value')}` |")
    passed = bool(summary.get("passed_stage32_hardened_success_criteria", False))
    lines.extend(["", "## Conservative Interpretation", "", f"Stage 3.2 hardened criteria pass: `{passed}`."])
    if passed:
        lines.append("Conservative allowed claim: the locked shared-weight cloned-agent latent coordinator passed this hardened import-restoration patch-selection benchmark only. This is not open-ended code repair or SWE-bench evidence.")
    else:
        lines.append("Do not claim real-code latent coordination success. Treat the result as invalid or incomplete until shortcut diagnostics, training deltas, controls, invariances, and audits all pass.")
    return "\n".join(lines) + "\n"


def _audit_rows_for_seed(row: Dict[str, object], split_leakage: Dict[str, object], output_leakage: Dict[str, object]) -> List[Dict[str, object]]:
    seed = int(row["seed"])
    return [
        {"event": "stage32_seed_result", "seed": seed, "candidate": row.get("candidate"), "status": row.get("status"), "test_accuracy": row.get("test_accuracy"), "dev_accuracy": row.get("dev_accuracy")},
        {"event": "stage32_split_leakage_audit", "seed": seed, **split_leakage},
        {"event": "stage32_output_leakage_audit", "seed": seed, **output_leakage},
        {"event": "stage32_trainable_training_audit", "seed": seed, **row.get("trainable_audit", {})},
        {"event": "stage32_frozen_training_audit", "seed": seed, **row.get("frozen_audit", {})},
        {"event": "stage32_message_collapse", "seed": seed, **row.get("message_collapse", {})},
        {"event": "stage32_per_role_ablation", "seed": seed, **row.get("per_role_ablation", {})},
        {"event": "stage32_per_family_accuracy", "seed": seed, "per_family_accuracy": row.get("per_family_accuracy", {})},
        {"event": "stage32_cuda_memory", "seed": seed, "cuda_max_memory_allocated": row.get("cuda_max_memory_allocated"), "cuda_max_memory_allocated_gb": row.get("cuda_max_memory_allocated_gb"), "cuda_device_name": row.get("cuda_device_name")},
    ]


def _leakage_passes(leakage: Dict[str, object]) -> bool:
    return bool(
        int(leakage.get("train_test_id_overlap", 1)) == 0
        and int(leakage.get("train_test_candidate_hash_overlap", 1)) == 0
        and int(leakage.get("train_test_file_patch_overlap", 1)) == 0
        and int(leakage.get("train_test_problem_family_overlap", 1)) == 0
        and int(leakage.get("train_test_source_import_key_overlap", 1)) == 0
        and int(leakage.get("train_test_target_file_overlap", 1)) == 0
        and bool(leakage.get("candidate_patch_hash_leakage_audit_passes", False))
    )


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    main()
