from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

from src.datasets.swe_patch_selection_dataset import STAGE6_NUM_CANDIDATES, stable_patch_hash


BENCHMARK = "stage6b1_candidate_generation"
DEFAULT_DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
DEFAULT_OUTPUT_PATH = Path("results/stage6b1_generated_candidates.jsonl")
DEFAULT_AUDIT_PATH = Path("results/stage6b1_generation_audit.json")
DEFAULT_REPORT_PATH = Path("reports/STAGE6B1_CANDIDATE_GENERATION.md")
DEFAULT_TRAJECTORY_ROOT = Path("results/stage6b1_trajectories")
DEFAULT_CANDIDATE_PATCH_ROOT = Path("results/stage6b1_candidate_patches")
DEFAULT_PROMPT_ROOT = Path("results/stage6b1_prompts")
DEFAULT_DUPLICATE_RATE_THRESHOLD = 0.20
DEFAULT_NEAR_REFERENCE_THRESHOLD = 0.95
DEFAULT_CODERFORGE_DATASET = "togethercomputer/CoderForge-Preview-32B-SWE-Bench-Verified-Evaluation-trajectories"
DEFAULT_HANSPETER_DATASET = "hanspeterlyngsoeraaschoujensen/swebench-eval-trajectories"
DEFAULT_ANTIEVAL_DATASET = "antieval/swebench-trajectories"
LOCAL_PREDICTION_PATTERNS = ("all_preds.jsonl", "preds.json", "predictions.jsonl", "predictions.json", "*.pred", "*.jsonl", "*.json")


@dataclass(frozen=True)
class BenchmarkTask:
    instance_id: str
    record: Dict[str, object]

    @property
    def reference_patch(self) -> str:
        return str(self.record.get("patch", "") or "")

    @property
    def reference_hash(self) -> str:
        return stable_patch_hash(self.reference_patch) if self.reference_patch.strip() else ""


@dataclass(frozen=True)
class CandidateAttempt:
    instance_id: str
    model_name_or_path: str
    model_patch: str
    generator_name: str
    seed: int | None
    temperature: float | None
    config_name: str
    trajectory_path: str
    source_name: str
    source_row_index: int | None

    @property
    def patch_hash(self) -> str:
        return stable_patch_hash(self.model_patch)


@dataclass(frozen=True)
class CandidateOutput:
    instance_id: str
    candidate_id: str
    model_name_or_path: str
    model_patch: str
    generator_name: str
    seed: int | None
    temperature: float | None
    trajectory_path: str
    source_visible_to_selector: bool = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6B.1 real generated candidate generation for SWE-bench tasks.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default="test")
    parser.add_argument("--benchmark-jsonl", default=None, help="Local benchmark records with instance_id and optional patch fields.")
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--task-ids-file", default=None)
    parser.add_argument("--n-tasks", type=int, default=10)
    parser.add_argument("--k", type=int, default=STAGE6_NUM_CANDIDATES)
    parser.add_argument("--seed", type=int, default=606_281)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--trajectory-root", default=str(DEFAULT_TRAJECTORY_ROOT))
    parser.add_argument("--candidate-patch-root", default=str(DEFAULT_CANDIDATE_PATCH_ROOT))
    parser.add_argument("--prompt-root", default=str(DEFAULT_PROMPT_ROOT))
    parser.add_argument("--source-jsonl", action="append", default=[])
    parser.add_argument("--source-json", action="append", default=[])
    parser.add_argument("--source-root", action="append", default=[])
    parser.add_argument("--include-default-hf-public", action="store_true")
    parser.add_argument("--enable-mini-swe-agent", action="store_true")
    parser.add_argument("--mini-swe-agent-command-template", action="append", default=[])
    parser.add_argument("--mini-swe-agent-config", action="append", default=[])
    parser.add_argument("--api-generator-command-template", action="append", default=[])
    parser.add_argument("--api-generator-model", action="append", default=[])
    parser.add_argument("--api-generator-temperature", action="append", default=[])
    parser.add_argument("--api-generator-seed", action="append", default=[])
    parser.add_argument("--duplicate-rate-threshold", type=float, default=DEFAULT_DUPLICATE_RATE_THRESHOLD)
    parser.add_argument("--near-reference-threshold", type=float, default=DEFAULT_NEAR_REFERENCE_THRESHOLD)
    args = parser.parse_args()

    result = run_stage6b1_candidate_generation(
        dataset_name=str(args.dataset_name),
        split=str(args.split),
        benchmark_jsonl=None if args.benchmark_jsonl is None else Path(args.benchmark_jsonl),
        task_ids=tuple(str(value) for value in args.task_id),
        task_ids_file=None if args.task_ids_file is None else Path(args.task_ids_file),
        n_tasks=int(args.n_tasks),
        k=int(args.k),
        seed=int(args.seed),
        output_path=Path(args.output),
        audit_path=Path(args.audit),
        report_path=Path(args.report),
        trajectory_root=Path(args.trajectory_root),
        candidate_patch_root=Path(args.candidate_patch_root),
        prompt_root=Path(args.prompt_root),
        source_jsonl=tuple(str(value) for value in args.source_jsonl),
        source_json=tuple(str(value) for value in args.source_json),
        source_roots=tuple(str(value) for value in args.source_root),
        include_default_hf_public=bool(args.include_default_hf_public),
        enable_mini_swe_agent=bool(args.enable_mini_swe_agent),
        mini_swe_agent_command_templates=tuple(str(value) for value in args.mini_swe_agent_command_template),
        mini_swe_agent_configs=tuple(str(value) for value in args.mini_swe_agent_config),
        api_generator_command_templates=tuple(str(value) for value in args.api_generator_command_template),
        api_generator_models=tuple(str(value) for value in args.api_generator_model),
        api_generator_temperatures=tuple(str(value) for value in args.api_generator_temperature),
        api_generator_seeds=tuple(str(value) for value in args.api_generator_seed),
        duplicate_rate_threshold=float(args.duplicate_rate_threshold),
        near_reference_threshold=float(args.near_reference_threshold),
    )
    audit = result["audit"]
    print(
        "stage6b1: wrote {output}, {audit_path}, {report}; rows={rows}; tasks_k8={tasks_k8}; gates_pass={gates}".format(
            output=args.output,
            audit_path=args.audit,
            report=args.report,
            rows=audit.get("accepted_candidate_rows", 0),
            tasks_k8=audit.get("tasks_with_at_least_k_unique_generated_candidates", 0),
            gates=audit.get("quality_gates", {}).get("candidate_generation_quality_gates_pass", False),
        )
    )


def run_stage6b1_candidate_generation(
    dataset_name: str,
    split: str,
    benchmark_jsonl: Path | None,
    task_ids: Sequence[str],
    task_ids_file: Path | None,
    n_tasks: int,
    k: int,
    seed: int,
    output_path: Path,
    audit_path: Path,
    report_path: Path,
    trajectory_root: Path,
    candidate_patch_root: Path,
    prompt_root: Path,
    source_jsonl: Sequence[str],
    source_json: Sequence[str],
    source_roots: Sequence[str],
    include_default_hf_public: bool,
    enable_mini_swe_agent: bool,
    mini_swe_agent_command_templates: Sequence[str],
    mini_swe_agent_configs: Sequence[str],
    api_generator_command_templates: Sequence[str],
    api_generator_models: Sequence[str],
    api_generator_temperatures: Sequence[str],
    api_generator_seeds: Sequence[str],
    duplicate_rate_threshold: float,
    near_reference_threshold: float,
) -> Dict[str, object]:
    created_at = _now()
    task_records, benchmark_audit = load_benchmark_tasks(
        dataset_name=dataset_name,
        split=split,
        benchmark_jsonl=benchmark_jsonl,
        requested_task_ids=task_ids,
        task_ids_file=task_ids_file,
        n_tasks=n_tasks,
        seed=seed,
    )
    task_ids_selected = tuple(task_records)
    source_audit: Dict[str, object] = {"sources": [], "errors": []}
    attempts: List[CandidateAttempt] = []
    attempts.extend(
        load_local_prediction_sources(
            task_ids=task_ids_selected,
            trajectory_root=trajectory_root,
            source_jsonl=source_jsonl,
            source_json=source_json,
            source_roots=source_roots,
            source_audit=source_audit,
        )
    )
    if include_default_hf_public:
        attempts.extend(load_default_hf_public_sources(task_ids_selected, trajectory_root, source_audit))
    attempts.extend(
        run_mini_swe_agent_backends(
            tasks=task_records,
            trajectory_root=trajectory_root,
            prompt_root=prompt_root,
            enable=enable_mini_swe_agent,
            command_templates=mini_swe_agent_command_templates,
            config_names=mini_swe_agent_configs,
            seed=seed,
            source_audit=source_audit,
        )
    )
    attempts.extend(
        run_api_generator_wrappers(
            tasks=task_records,
            trajectory_root=trajectory_root,
            prompt_root=prompt_root,
            command_templates=api_generator_command_templates,
            models=api_generator_models,
            temperatures=api_generator_temperatures,
            seeds=api_generator_seeds,
            default_seed=seed,
            source_audit=source_audit,
        )
    )
    accepted, build_audit = select_generated_candidates(
        attempts=attempts,
        task_records=task_records,
        k=k,
        seed=seed,
        candidate_patch_root=candidate_patch_root,
        duplicate_rate_threshold=duplicate_rate_threshold,
        near_reference_threshold=near_reference_threshold,
    )
    write_stage6b1_jsonl(output_path, accepted)
    environment = collect_environment(dataset_name, split)
    audit = build_generation_audit(
        created_at=created_at,
        dataset_name=dataset_name,
        split=split,
        requested_tasks=n_tasks,
        k=k,
        seed=seed,
        benchmark_audit=benchmark_audit,
        source_audit=source_audit,
        build_audit=build_audit,
        accepted=accepted,
        output_path=output_path,
        trajectory_root=trajectory_root,
        candidate_patch_root=candidate_patch_root,
        environment=environment,
        duplicate_rate_threshold=duplicate_rate_threshold,
    )
    _write_json(audit_path, audit)
    report = render_report(audit=audit, output_path=output_path, audit_path=audit_path, report_path=report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return {"audit": audit, "rows": accepted}


def load_benchmark_tasks(
    dataset_name: str,
    split: str,
    benchmark_jsonl: Path | None,
    requested_task_ids: Sequence[str],
    task_ids_file: Path | None,
    n_tasks: int,
    seed: int,
) -> tuple[Dict[str, BenchmarkTask], Dict[str, object]]:
    explicit_ids = list(dict.fromkeys(str(value).strip() for value in requested_task_ids if str(value).strip()))
    if task_ids_file is not None:
        explicit_ids.extend(_read_task_ids(task_ids_file))
        explicit_ids = list(dict.fromkeys(explicit_ids))
    audit = {
        "dataset_name": dataset_name,
        "split": split,
        "benchmark_jsonl": None if benchmark_jsonl is None else str(benchmark_jsonl),
        "loaded": False,
        "rows": 0,
        "requested_task_ids": explicit_ids,
        "selected_task_ids": [],
        "missing_requested_task_ids": [],
        "error": None,
    }
    rows: Dict[str, Dict[str, object]] = {}
    try:
        if benchmark_jsonl is not None:
            rows = _load_benchmark_jsonl(benchmark_jsonl)
        else:
            from datasets import load_dataset

            dataset = load_dataset(dataset_name, split=split)
            rows = {str(row["instance_id"]): dict(row) for row in dataset}
        audit.update({"loaded": True, "rows": len(rows)})
    except Exception as exc:
        audit["error"] = f"{type(exc).__name__}: {exc}"
        rows = {}
    if explicit_ids:
        selected = [instance_id for instance_id in explicit_ids if instance_id in rows]
        missing = [instance_id for instance_id in explicit_ids if instance_id not in rows]
        selected = selected[: max(0, int(n_tasks))]
        audit["missing_requested_task_ids"] = missing
    else:
        rng = random.Random(seed + 6_281_001)
        selected = sorted(rows)
        rng.shuffle(selected)
        selected = selected[: max(0, int(n_tasks))]
    audit["selected_task_ids"] = selected
    return {instance_id: BenchmarkTask(instance_id=instance_id, record=rows[instance_id]) for instance_id in selected}, audit


def load_local_prediction_sources(
    task_ids: Sequence[str],
    trajectory_root: Path,
    source_jsonl: Sequence[str],
    source_json: Sequence[str],
    source_roots: Sequence[str],
    source_audit: Dict[str, object],
) -> List[CandidateAttempt]:
    rows: List[CandidateAttempt] = []
    allowed_ids = set(task_ids)
    for path in source_jsonl:
        rows.extend(_load_prediction_file(Path(path), allowed_ids, trajectory_root, source_audit, forced_kind="jsonl"))
    for path in source_json:
        rows.extend(_load_prediction_file(Path(path), allowed_ids, trajectory_root, source_audit, forced_kind="json"))
    for root in source_roots:
        rows.extend(_load_prediction_root(Path(root), allowed_ids, trajectory_root, source_audit))
    return rows


def load_default_hf_public_sources(
    task_ids: Sequence[str],
    trajectory_root: Path,
    source_audit: Dict[str, object],
) -> List[CandidateAttempt]:
    allowed_ids = set(task_ids)
    rows: List[CandidateAttempt] = []
    rows.extend(_load_hf_coderforge(allowed_ids, trajectory_root, source_audit))
    rows.extend(_load_hf_message_trajectory_dataset(DEFAULT_HANSPETER_DATASET, allowed_ids, trajectory_root, source_audit))
    rows.extend(_load_hf_message_trajectory_dataset(DEFAULT_ANTIEVAL_DATASET, allowed_ids, trajectory_root, source_audit))
    return rows


def run_mini_swe_agent_backends(
    tasks: Dict[str, BenchmarkTask],
    trajectory_root: Path,
    prompt_root: Path,
    enable: bool,
    command_templates: Sequence[str],
    config_names: Sequence[str],
    seed: int,
    source_audit: Dict[str, object],
) -> List[CandidateAttempt]:
    installed = _mini_swe_agent_available()
    summary = {
        "source_name": "mini-swe-agent",
        "enabled": bool(enable),
        "installed": bool(installed),
        "command_templates": len(command_templates),
        "configs": list(config_names),
        "records": 0,
        "error": None,
    }
    rows: List[CandidateAttempt] = []
    if not enable:
        source_audit.setdefault("sources", []).append(summary)
        return rows
    if not installed:
        summary["error"] = "mini-SWE-agent is not installed or no executable was found"
        source_audit.setdefault("errors", []).append(summary)
        source_audit.setdefault("sources", []).append(summary)
        return rows
    if not command_templates:
        summary["error"] = "mini-SWE-agent installed, but no --mini-swe-agent-command-template was supplied"
        source_audit.setdefault("errors", []).append(summary)
        source_audit.setdefault("sources", []).append(summary)
        return rows
    configs = list(config_names) or ["default"]
    for task_index, task in enumerate(tasks.values()):
        for template_index, template in enumerate(command_templates):
            config_name = configs[template_index % len(configs)]
            run_seed = seed + 6_281_101 + task_index * 100 + template_index
            prompt_path = write_prompt(prompt_root, task, config_name, run_seed)
            output_path = trajectory_root / "mini_swe_agent_outputs" / _safe_name(task.instance_id) / f"{_safe_name(config_name)}_{run_seed}.json"
            trajectory_path = trajectory_root / "mini_swe_agent" / _safe_name(task.instance_id) / f"{_safe_name(config_name)}_{run_seed}.json"
            command = _format_command_template(
                template,
                task=task,
                prompt_path=prompt_path,
                output_path=output_path,
                seed=run_seed,
                temperature=0.0,
                model="mini-swe-agent",
                config_name=config_name,
            )
            rows.extend(
                _run_external_generator_command(
                    command=command,
                    output_path=output_path,
                    trajectory_path=trajectory_path,
                    task=task,
                    generator_name="mini-swe-agent",
                    model_name="mini-swe-agent",
                    seed=run_seed,
                    temperature=0.0,
                    config_name=config_name,
                    source_name="mini-swe-agent",
                    source_row_index=template_index,
                )
            )
    summary["records"] = len(rows)
    source_audit.setdefault("sources", []).append(summary)
    return rows


def run_api_generator_wrappers(
    tasks: Dict[str, BenchmarkTask],
    trajectory_root: Path,
    prompt_root: Path,
    command_templates: Sequence[str],
    models: Sequence[str],
    temperatures: Sequence[str],
    seeds: Sequence[str],
    default_seed: int,
    source_audit: Dict[str, object],
) -> List[CandidateAttempt]:
    summary = {
        "source_name": "api-generator-wrapper",
        "command_templates": len(command_templates),
        "models": list(models),
        "temperatures": list(temperatures),
        "seeds": list(seeds),
        "records": 0,
        "error": None,
    }
    rows: List[CandidateAttempt] = []
    if not command_templates:
        source_audit.setdefault("sources", []).append(summary)
        return rows
    model_values = list(models) or ["api-generator"]
    temperature_values = [_parse_float(value, 0.0) for value in (list(temperatures) or ["0.0"])]
    seed_values = [_parse_int(value, default_seed) for value in (list(seeds) or [str(default_seed)])]
    specs = []
    for template_index, template in enumerate(command_templates):
        for model in model_values:
            for temperature in temperature_values:
                for seed_value in seed_values:
                    specs.append((template_index, template, model, float(temperature), int(seed_value)))
    for task_index, task in enumerate(tasks.values()):
        for spec_index, (template_index, template, model, temperature, seed_value) in enumerate(specs):
            run_seed = int(seed_value) + 6_281_201 + task_index * 1000 + spec_index
            config_name = f"api_template_{template_index}_temp_{temperature:g}"
            prompt_path = write_prompt(prompt_root, task, config_name, run_seed)
            output_path = trajectory_root / "api_generator_outputs" / _safe_name(task.instance_id) / f"{_safe_name(model)}_{run_seed}.json"
            trajectory_path = trajectory_root / "api_generator" / _safe_name(task.instance_id) / f"{_safe_name(model)}_{run_seed}.json"
            command = _format_command_template(
                template,
                task=task,
                prompt_path=prompt_path,
                output_path=output_path,
                seed=run_seed,
                temperature=temperature,
                model=model,
                config_name=config_name,
            )
            rows.extend(
                _run_external_generator_command(
                    command=command,
                    output_path=output_path,
                    trajectory_path=trajectory_path,
                    task=task,
                    generator_name="api-generator-wrapper",
                    model_name=model,
                    seed=run_seed,
                    temperature=temperature,
                    config_name=config_name,
                    source_name="api-generator-wrapper",
                    source_row_index=spec_index,
                )
            )
    summary["records"] = len(rows)
    source_audit.setdefault("sources", []).append(summary)
    return rows


def select_generated_candidates(
    attempts: Sequence[CandidateAttempt],
    task_records: Dict[str, BenchmarkTask],
    k: int,
    seed: int,
    candidate_patch_root: Path,
    duplicate_rate_threshold: float,
    near_reference_threshold: float,
) -> tuple[List[CandidateOutput], Dict[str, object]]:
    grouped: Dict[str, List[CandidateAttempt]] = defaultdict(list)
    empty_rows = []
    invalid_rows = []
    exact_reference_hits = []
    duplicate_rows = []
    near_reference_hits = []
    all_attempt_hashes: Dict[str, List[str]] = defaultdict(list)
    seen_by_task: Dict[str, set[str]] = defaultdict(set)
    for attempt in attempts:
        if attempt.instance_id not in task_records:
            continue
        patch = _clean_patch(attempt.model_patch)
        patch_hash = stable_patch_hash(patch)
        all_attempt_hashes[attempt.instance_id].append(patch_hash)
        if not patch.strip():
            empty_rows.append(_attempt_audit_row(attempt, patch_hash))
            continue
        if not _looks_like_unified_diff(patch):
            invalid_rows.append(_attempt_audit_row(attempt, patch_hash))
            continue
        reference_hash = task_records[attempt.instance_id].reference_hash
        if reference_hash and patch_hash == reference_hash:
            exact_reference_hits.append(_attempt_audit_row(attempt, patch_hash))
            continue
        reference_patch = task_records[attempt.instance_id].reference_patch
        similarity = _patch_similarity(patch, reference_patch)
        if reference_patch.strip() and similarity >= near_reference_threshold:
            near_reference_hits.append({**_attempt_audit_row(attempt, patch_hash), "similarity": similarity})
        if patch_hash in seen_by_task[attempt.instance_id]:
            duplicate_rows.append(_attempt_audit_row(attempt, patch_hash))
            continue
        seen_by_task[attempt.instance_id].add(patch_hash)
        grouped[attempt.instance_id].append(
            CandidateAttempt(
                instance_id=attempt.instance_id,
                model_name_or_path=attempt.model_name_or_path,
                model_patch=_ensure_trailing_newline(patch),
                generator_name=attempt.generator_name,
                seed=attempt.seed,
                temperature=attempt.temperature,
                config_name=attempt.config_name,
                trajectory_path=attempt.trajectory_path,
                source_name=attempt.source_name,
                source_row_index=attempt.source_row_index,
            )
        )
    rng = random.Random(seed + 6_281_301)
    accepted: List[CandidateOutput] = []
    selected_source_rows = []
    tasks_with_k = []
    for instance_id in task_records:
        rows = grouped.get(instance_id, [])
        selected = _select_diverse_k(rows, k=k, seed=seed + len(accepted))
        if len(selected) >= k:
            tasks_with_k.append(instance_id)
        order = list(range(len(selected)))
        rng.shuffle(order)
        for out_index, row_index in enumerate(order):
            attempt = selected[row_index]
            candidate_id = f"{instance_id}-stage6b1-candidate-{out_index:02d}-{attempt.patch_hash[:12]}"
            _write_candidate_patch_file(candidate_patch_root, instance_id, candidate_id, attempt.model_patch)
            accepted.append(
                CandidateOutput(
                    instance_id=instance_id,
                    candidate_id=candidate_id,
                    model_name_or_path=attempt.model_name_or_path,
                    model_patch=attempt.model_patch,
                    generator_name=attempt.generator_name,
                    seed=attempt.seed,
                    temperature=attempt.temperature,
                    trajectory_path=attempt.trajectory_path,
                    source_visible_to_selector=False,
                )
            )
            selected_source_rows.append(
                {
                    "instance_id": instance_id,
                    "candidate_id": candidate_id,
                    "patch_hash": attempt.patch_hash,
                    "generator_name": attempt.generator_name,
                    "model_name_or_path": attempt.model_name_or_path,
                    "seed": attempt.seed,
                    "temperature": attempt.temperature,
                    "config_name": attempt.config_name,
                    "trajectory_path": attempt.trajectory_path,
                    "source_name": attempt.source_name,
                    "source_row_index": attempt.source_row_index,
                    "pre_shuffle_index": row_index,
                    "output_index": out_index,
                }
            )
    duplicate_attempts = sum(max(0, len(values) - len(set(values))) for values in all_attempt_hashes.values())
    total_attempts = sum(len(values) for values in all_attempt_hashes.values())
    duplicate_rate = float(duplicate_attempts / total_attempts) if total_attempts else 0.0
    counts = Counter(row.instance_id for row in accepted)
    unique_counts = {instance_id: len({row.patch_hash for row in rows}) for instance_id, rows in grouped.items()}
    build_audit = {
        "attempt_rows_loaded": len(attempts),
        "attempt_rows_for_selected_tasks": total_attempts,
        "accepted_candidate_rows": len(accepted),
        "tasks_attempted": len(task_records),
        "tasks_with_any_generated_candidate": len(grouped),
        "tasks_with_at_least_k_unique_generated_candidates": len(tasks_with_k),
        "tasks_with_at_least_k_unique_generated_candidate_ids": tasks_with_k,
        "candidates_per_task_distribution": {str(key): int(value) for key, value in sorted(Counter(counts.values()).items())},
        "unique_candidates_per_task": {key: int(value) for key, value in sorted(unique_counts.items())},
        "duplicate_patch_hash_audit": {
            "duplicates_excluded": duplicate_rows[:200],
            "duplicates_excluded_total": len(duplicate_rows),
            "duplicate_attempts_before_selection": duplicate_attempts,
            "total_attempts_before_selection": total_attempts,
            "duplicate_rate_before_selection": duplicate_rate,
            "threshold": float(duplicate_rate_threshold),
            "passes": duplicate_rate <= float(duplicate_rate_threshold),
        },
        "empty_patch_audit": {"rows": empty_rows[:200], "total": len(empty_rows), "passes": not empty_rows},
        "invalid_diff_audit": {"rows": invalid_rows[:200], "total": len(invalid_rows), "passes": not invalid_rows},
        "exact_reference_patch_hash_audit": {
            "accepted_hits": [],
            "source_hits_excluded": exact_reference_hits[:200],
            "source_hits_excluded_total": len(exact_reference_hits),
            "passes": True,
        },
        "near_reference_similarity_audit": {
            "threshold": float(near_reference_threshold),
            "hits": near_reference_hits[:200],
            "hits_total": len(near_reference_hits),
            "passes": True,
        },
        "generator_source_distribution": {
            "by_generator_name": dict(Counter(row.generator_name for row in accepted)),
            "by_model_name_or_path": dict(Counter(row.model_name_or_path for row in accepted)),
        },
        "candidate_diversity_by_normalized_patch_edit_distance": _candidate_diversity_audit(accepted),
        "selected_source_rows": selected_source_rows,
        "raw_preserving_expansion_used": False,
        "missing_candidate_fabrication_used": False,
        "candidate_order_randomizable": True,
        "source_visible_to_selector_all_false": all(row.source_visible_to_selector is False for row in accepted),
        "generated_candidate_files_saved": True,
    }
    return accepted, build_audit


def build_generation_audit(
    created_at: str,
    dataset_name: str,
    split: str,
    requested_tasks: int,
    k: int,
    seed: int,
    benchmark_audit: Dict[str, object],
    source_audit: Dict[str, object],
    build_audit: Dict[str, object],
    accepted: Sequence[CandidateOutput],
    output_path: Path,
    trajectory_root: Path,
    candidate_patch_root: Path,
    environment: Dict[str, object],
    duplicate_rate_threshold: float,
) -> Dict[str, object]:
    counts = Counter(row.instance_id for row in accepted)
    trajectory_rows = []
    missing_trajectories = []
    for row in accepted:
        exists = bool(row.trajectory_path and Path(row.trajectory_path).exists())
        trajectory_rows.append({"candidate_id": row.candidate_id, "trajectory_path": row.trajectory_path, "exists": exists})
        if not exists:
            missing_trajectories.append({"candidate_id": row.candidate_id, "trajectory_path": row.trajectory_path})
    required_keys = {
        "instance_id",
        "candidate_id",
        "model_name_or_path",
        "model_patch",
        "generator_name",
        "seed",
        "temperature",
        "trajectory_path",
        "source_visible_to_selector",
    }
    schema_rows = []
    for row in accepted:
        data = asdict(row)
        schema_rows.append({"candidate_id": row.candidate_id, "missing": sorted(required_keys - set(data))})
    files_saved = {
        "jsonl_output_exists": output_path.exists(),
        "jsonl_output_rows": len(_read_jsonl(output_path)) if output_path.exists() else 0,
        "trajectory_root": str(trajectory_root),
        "candidate_patch_root": str(candidate_patch_root),
        "missing_trajectories": missing_trajectories,
        "passes": output_path.exists() and not missing_trajectories,
    }
    tasks_with_k = int(build_audit.get("tasks_with_at_least_k_unique_generated_candidates", 0))
    duplicate_audit = dict(build_audit.get("duplicate_patch_hash_audit", {}))
    exact_reference = dict(build_audit.get("exact_reference_patch_hash_audit", {}))
    quality_gates = {
        "at_least_25_tasks_attempted": int(build_audit.get("tasks_attempted", 0)) >= 25,
        "at_least_20_tasks_with_k8_unique_generated_candidates": tasks_with_k >= 20 if int(k) == STAGE6_NUM_CANDIDATES else tasks_with_k >= 20,
        "no_exact_reference_patch_hits": bool(exact_reference.get("passes", False)),
        "duplicate_rate_below_threshold": float(duplicate_audit.get("duplicate_rate_before_selection", 1.0)) <= float(duplicate_rate_threshold),
        "candidate_order_randomizable": bool(build_audit.get("candidate_order_randomizable", False)),
        "all_generated_candidate_files_saved": bool(files_saved.get("passes", False)),
        "no_raw_preserving_expansion": not bool(build_audit.get("raw_preserving_expansion_used", False)),
        "no_missing_candidate_fabrication": not bool(build_audit.get("missing_candidate_fabrication_used", False)),
        "no_final_model_claim_made": True,
        "selector_training_executed": False,
    }
    quality_gates["candidate_generation_quality_gates_pass"] = all(
        bool(value) for key, value in quality_gates.items() if key != "selector_training_executed"
    ) and quality_gates["selector_training_executed"] is False
    return {
        "benchmark": BENCHMARK,
        "created_at_utc": created_at,
        "dataset_name": dataset_name,
        "split": split,
        "requested_tasks": int(requested_tasks),
        "candidate_count_k": int(k),
        "seed": int(seed),
        "selector_training_executed": False,
        "official_harness_evaluation_executed": False,
        "no_final_model_claim_made": True,
        "output_path": str(output_path),
        "accepted_candidate_rows": len(accepted),
        "tasks_attempted": int(build_audit.get("tasks_attempted", 0)),
        "tasks_with_at_least_k_unique_generated_candidates": tasks_with_k,
        "candidates_per_task_distribution": build_audit.get("candidates_per_task_distribution", {}),
        "benchmark_dataset_audit": benchmark_audit,
        "source_generation_audit": source_audit,
        "generation_build_audit": build_audit,
        "duplicate_patch_hash_audit": duplicate_audit,
        "empty_patch_audit": build_audit.get("empty_patch_audit", {}),
        "invalid_diff_audit": build_audit.get("invalid_diff_audit", {}),
        "exact_reference_patch_hash_audit": exact_reference,
        "near_reference_similarity_audit": build_audit.get("near_reference_similarity_audit", {}),
        "generator_source_distribution": build_audit.get("generator_source_distribution", {}),
        "candidate_diversity_by_normalized_patch_edit_distance": build_audit.get("candidate_diversity_by_normalized_patch_edit_distance", {}),
        "candidate_schema_audit": {"required_keys": sorted(required_keys), "rows": schema_rows, "passes": not any(row["missing"] for row in schema_rows)},
        "candidate_file_save_audit": files_saved,
        "trajectory_path_audit": {"rows": trajectory_rows[:200], "missing_total": len(missing_trajectories), "passes": not missing_trajectories},
        "raw_preserving_expansion_audit": {
            "raw_preserving_expansion_used": bool(build_audit.get("raw_preserving_expansion_used", False)),
            "missing_candidate_fabrication_used": bool(build_audit.get("missing_candidate_fabrication_used", False)),
            "passes": not bool(build_audit.get("raw_preserving_expansion_used", False))
            and not bool(build_audit.get("missing_candidate_fabrication_used", False)),
        },
        "quality_gates": quality_gates,
        "environment": environment,
    }


def render_report(audit: Dict[str, object], output_path: Path, audit_path: Path, report_path: Path) -> str:
    gates = audit.get("quality_gates", {})
    duplicate = audit.get("duplicate_patch_hash_audit", {})
    diversity = audit.get("candidate_diversity_by_normalized_patch_edit_distance", {})
    source_distribution = audit.get("generator_source_distribution", {})
    lines = [
        "# Stage 6B.1 Candidate Generation",
        "",
        "## Scope",
        "",
        "- Generates or ingests real generated SWE-bench candidate patches only.",
        "- Does not train the latent selector.",
        "- Does not run the official SWE-bench harness.",
        "- Does not use reference/gold patches as candidates.",
        "- Does not use raw-preserving expansion.",
        "- Does not fabricate missing candidates.",
        "- Stores raw trajectory records separately from selector candidate patch text.",
        "- Generator identity is retained in audit/output metadata, not embedded in selector-visible patch text.",
        "- No final model claim is made.",
        "",
        "## Artifacts",
        "",
        f"- Generated candidates JSONL: `{output_path}`.",
        f"- Generation audit: `{audit_path}`.",
        f"- Report: `{report_path}`.",
        "",
        "## Summary",
        "",
        f"- Requested tasks: `{audit.get('requested_tasks')}`.",
        f"- Tasks attempted: `{audit.get('tasks_attempted')}`.",
        f"- Candidate rows accepted: `{audit.get('accepted_candidate_rows')}`.",
        f"- Tasks with K unique generated candidates: `{audit.get('tasks_with_at_least_k_unique_generated_candidates')}`.",
        f"- Candidates per task distribution: `{json.dumps(audit.get('candidates_per_task_distribution', {}), sort_keys=True)}`.",
        f"- Duplicate rate before selection: `{float(duplicate.get('duplicate_rate_before_selection', 0.0)):.4f}`.",
        f"- Exact reference hits excluded: `{audit.get('exact_reference_patch_hash_audit', {}).get('source_hits_excluded_total', 0)}`.",
        f"- Generator distribution: `{json.dumps(source_distribution.get('by_generator_name', {}), sort_keys=True)}`.",
        f"- Mean pairwise normalized patch edit distance: `{float(diversity.get('mean_pairwise_distance', 0.0)):.4f}`.",
        "",
        "## Audits",
        "",
        f"- Empty patch audit passes: `{audit.get('empty_patch_audit', {}).get('passes')}`.",
        f"- Invalid diff audit passes: `{audit.get('invalid_diff_audit', {}).get('passes')}`.",
        f"- Duplicate patch hash audit passes: `{duplicate.get('passes')}`.",
        f"- Exact reference hash audit passes: `{audit.get('exact_reference_patch_hash_audit', {}).get('passes')}`.",
        f"- Near-reference similarity hits: `{audit.get('near_reference_similarity_audit', {}).get('hits_total')}`.",
        f"- Candidate files saved audit passes: `{audit.get('candidate_file_save_audit', {}).get('passes')}`.",
        f"- Raw-preserving expansion audit passes: `{audit.get('raw_preserving_expansion_audit', {}).get('passes')}`.",
        "",
        "## Quality Gates",
        "",
        "| gate | pass |",
        "|---|---:|",
    ]
    for key, value in gates.items():
        lines.append(f"| `{key}` | `{bool(value)}` |")
    if not bool(gates.get("candidate_generation_quality_gates_pass", False)):
        lines.extend(
            [
                "",
                "## Current Limitation",
                "",
                "Stage 6B.1 quality gates did not pass. Add more generator backends or local generated prediction files, then rerun this script. The script intentionally keeps tasks with fewer than K unique patches incomplete rather than inventing candidates.",
            ]
        )
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            f"When the gates pass, run Stage 6B.0 with `--source-jsonl {output_path}` and the official SWE-bench Docker harness.",
            "",
            "No final model claim is made.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_prediction_root(
    root: Path,
    task_ids: set[str],
    trajectory_root: Path,
    source_audit: Dict[str, object],
) -> List[CandidateAttempt]:
    rows: List[CandidateAttempt] = []
    if not root.exists():
        source_audit.setdefault("sources", []).append({"source_name": f"root:{root}", "loaded": False, "records": 0, "error": "path does not exist"})
        return rows
    paths: List[Path] = []
    for pattern in LOCAL_PREDICTION_PATTERNS:
        paths.extend(root.rglob(pattern))
    seen = set()
    for path in sorted(paths):
        if path in seen or path.is_dir() or path.stat().st_size == 0:
            continue
        seen.add(path)
        if ".git" in path.parts or path.stat().st_size > 200_000_000:
            continue
        rows.extend(_load_prediction_file(path, task_ids, trajectory_root, source_audit, forced_kind=None))
    return rows


def _load_prediction_file(
    path: Path,
    task_ids: set[str],
    trajectory_root: Path,
    source_audit: Dict[str, object],
    forced_kind: str | None,
) -> List[CandidateAttempt]:
    source_name = f"file:{path.as_posix()}"
    summary = {"source_name": source_name, "loaded": False, "records": 0, "candidate_attempts": 0, "error": None}
    rows: List[CandidateAttempt] = []
    try:
        parsed = list(_iter_prediction_rows(path, forced_kind))
        counters: Dict[tuple[str, str], int] = defaultdict(int)
        for row_index, row in enumerate(parsed):
            for instance_id, patch, generator, model, seed, temperature, config_name in _prediction_row_to_attempt_values(row, path, task_ids):
                sample = counters[(instance_id, generator)]
                counters[(instance_id, generator)] += 1
                trajectory_path = _write_raw_trajectory(
                    trajectory_root,
                    source_group="local_prediction",
                    instance_id=instance_id,
                    generator_name=generator,
                    row_index=row_index,
                    sample_index=sample,
                    payload={
                        "source_file": path.as_posix(),
                        "source_row_index": row_index,
                        "raw_row": row,
                    },
                )
                rows.append(
                    CandidateAttempt(
                        instance_id=instance_id,
                        model_name_or_path=model,
                        model_patch=patch,
                        generator_name=generator,
                        seed=seed,
                        temperature=temperature,
                        config_name=config_name,
                        trajectory_path=str(trajectory_path),
                        source_name=source_name,
                        source_row_index=row_index,
                    )
                )
        summary.update({"loaded": True, "records": len(parsed), "candidate_attempts": len(rows)})
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        source_audit.setdefault("errors", []).append(summary)
    source_audit.setdefault("sources", []).append(summary)
    return rows


def _prediction_row_to_attempt_values(
    row: Dict[str, object],
    path: Path,
    task_ids: set[str],
) -> Iterator[tuple[str, str, str, str, int | None, float | None, str]]:
    instance_id = str(row.get("instance_id") or row.get("id") or row.get("task_id") or "")
    if not instance_id and path.stem in task_ids:
        instance_id = path.stem
    if instance_id not in task_ids:
        return
    generator = _clean_generator_name(row.get("generator_name") or row.get("generator") or row.get("agent") or row.get("model_name_or_path") or path.parent.name)
    model = str(row.get("model_name_or_path") or row.get("model") or generator or "unknown_model")
    seed = _optional_int(row.get("seed"))
    temperature = _optional_float(row.get("temperature"))
    config_name = str(row.get("config_name") or row.get("config") or row.get("prompt_name") or "local_prediction")
    patch_values: List[object] = []
    for key in ("model_patch", "output_patch", "generated_patch", "prediction", "diff", "patch"):
        value = row.get(key)
        if isinstance(value, dict):
            for nested in ("model_patch", "output_patch", "generated_patch", "prediction", "diff", "patch"):
                if value.get(nested):
                    patch_values.append(value.get(nested))
        elif value:
            patch_values.append(value)
    if row.get("messages"):
        patch_values.extend(_extract_diff_blocks_from_messages(row.get("messages")))
    for value in patch_values:
        patch = _clean_patch(str(value or ""))
        yield (instance_id, patch, generator, model, seed, temperature, config_name)


def _iter_prediction_rows(path: Path, forced_kind: str | None) -> Iterator[Dict[str, object]]:
    kind = forced_kind or ("jsonl" if path.suffix.lower() in {".jsonl", ".pred"} else "json")
    if kind == "jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    yield parsed
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                yield row
    elif isinstance(data, dict):
        for key in ("predictions", "instances", "rows", "data"):
            if isinstance(data.get(key), list):
                for row in data[key]:
                    if isinstance(row, dict):
                        yield row
                return
        for key, value in data.items():
            if isinstance(value, dict):
                yield {"instance_id": key, **value}
            elif isinstance(value, str):
                yield {"instance_id": key, "model_patch": value}


def _load_hf_coderforge(
    task_ids: set[str],
    trajectory_root: Path,
    source_audit: Dict[str, object],
) -> List[CandidateAttempt]:
    source_name = f"hf:{DEFAULT_CODERFORGE_DATASET}:trajectory"
    summary = {"source_name": source_name, "loaded": False, "records": 0, "candidate_attempts": 0, "error": None}
    rows: List[CandidateAttempt] = []
    try:
        from datasets import load_dataset

        dataset = load_dataset(DEFAULT_CODERFORGE_DATASET, "trajectory", split="train")
        counters: Dict[tuple[str, str], int] = defaultdict(int)
        for row_index, row in enumerate(dataset):
            ds = _json_obj(row.get("ds", {}))
            instance_id = str(ds.get("instance_id") or _instance_id_from_trajectory(str(row.get("trajectory_id", ""))))
            if instance_id not in task_ids:
                continue
            patch = _clean_patch(str(row.get("output_patch", "") or ""))
            generator = _clean_generator_name(row.get("exp_name") or row.get("model") or "CoderForge-Preview-32B")
            sample = counters[(instance_id, generator)]
            counters[(instance_id, generator)] += 1
            trajectory_path = _write_raw_trajectory(
                trajectory_root,
                source_group="hf_coderforge",
                instance_id=instance_id,
                generator_name=generator,
                row_index=row_index,
                sample_index=sample,
                payload={"source_dataset": DEFAULT_CODERFORGE_DATASET, "raw_row": dict(row)},
            )
            rows.append(
                CandidateAttempt(
                    instance_id=instance_id,
                    model_name_or_path=str(row.get("model") or row.get("exp_name") or generator),
                    model_patch=patch,
                    generator_name=generator,
                    seed=_optional_int(row.get("seed")),
                    temperature=_optional_float(row.get("temperature")),
                    config_name=str(row.get("exp_name") or "hf_coderforge"),
                    trajectory_path=str(trajectory_path),
                    source_name=source_name,
                    source_row_index=row_index,
                )
            )
        summary.update({"loaded": True, "records": len(dataset), "candidate_attempts": len(rows)})
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        source_audit.setdefault("errors", []).append(summary)
    source_audit.setdefault("sources", []).append(summary)
    return rows


def _load_hf_message_trajectory_dataset(
    dataset_name: str,
    task_ids: set[str],
    trajectory_root: Path,
    source_audit: Dict[str, object],
) -> List[CandidateAttempt]:
    source_name = f"hf:{dataset_name}"
    summary = {"source_name": source_name, "loaded": False, "records": 0, "candidate_attempts": 0, "error": None}
    rows: List[CandidateAttempt] = []
    try:
        from datasets import load_dataset

        dataset = load_dataset(dataset_name, split="train")
        counters: Dict[tuple[str, str], int] = defaultdict(int)
        for row_index, row in enumerate(dataset):
            instance_id = str(row.get("instance_id") or row.get("id") or "")
            if instance_id not in task_ids:
                continue
            patches = _extract_diff_blocks_from_messages(row.get("messages", ""))
            generator = _clean_generator_name(row.get("model") or row.get("agent") or dataset_name.rsplit("/", 1)[-1])
            for patch in patches:
                sample = counters[(instance_id, generator)]
                counters[(instance_id, generator)] += 1
                trajectory_path = _write_raw_trajectory(
                    trajectory_root,
                    source_group="hf_messages",
                    instance_id=instance_id,
                    generator_name=generator,
                    row_index=row_index,
                    sample_index=sample,
                    payload={"source_dataset": dataset_name, "raw_row": dict(row)},
                )
                rows.append(
                    CandidateAttempt(
                        instance_id=instance_id,
                        model_name_or_path=str(row.get("model") or generator),
                        model_patch=patch,
                        generator_name=generator,
                        seed=_optional_int(row.get("seed")),
                        temperature=_optional_float(row.get("temperature")),
                        config_name="hf_message_trajectory",
                        trajectory_path=str(trajectory_path),
                        source_name=source_name,
                        source_row_index=row_index,
                    )
                )
        summary.update({"loaded": True, "records": len(dataset), "candidate_attempts": len(rows)})
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        source_audit.setdefault("errors", []).append(summary)
    source_audit.setdefault("sources", []).append(summary)
    return rows


def _run_external_generator_command(
    command: str,
    output_path: Path,
    trajectory_path: Path,
    task: BenchmarkTask,
    generator_name: str,
    model_name: str,
    seed: int,
    temperature: float,
    config_name: str,
    source_name: str,
    source_row_index: int,
) -> List[CandidateAttempt]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    started = _now()
    proc = subprocess.run(command, shell=True, text=True, capture_output=True, timeout=None)
    payload: Dict[str, object] = {
        "source_name": source_name,
        "command": command,
        "started_at_utc": started,
        "finished_at_utc": _now(),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "output_path": str(output_path),
        "instance_id": task.instance_id,
        "generator_name": generator_name,
        "model_name_or_path": model_name,
        "seed": seed,
        "temperature": temperature,
        "config_name": config_name,
    }
    output_data = None
    if output_path.exists() and output_path.stat().st_size > 0:
        try:
            output_data = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception:
            output_data = output_path.read_text(encoding="utf-8")
    payload["generator_output"] = output_data
    _write_json(trajectory_path, payload)
    if proc.returncode != 0:
        return []
    patches = _patches_from_generator_output(output_data)
    if not patches:
        patches = _extract_diff_blocks_from_messages(proc.stdout)
    rows = []
    for index, patch in enumerate(patches):
        rows.append(
            CandidateAttempt(
                instance_id=task.instance_id,
                model_name_or_path=model_name,
                model_patch=patch,
                generator_name=generator_name,
                seed=seed,
                temperature=temperature,
                config_name=config_name,
                trajectory_path=str(trajectory_path),
                source_name=source_name,
                source_row_index=source_row_index + index,
            )
        )
    return rows


def _patches_from_generator_output(value: object) -> List[str]:
    if isinstance(value, dict):
        patches = []
        for key in ("model_patch", "output_patch", "generated_patch", "patch", "diff", "prediction"):
            if value.get(key):
                patches.append(_clean_patch(str(value.get(key))))
        if value.get("messages"):
            patches.extend(_extract_diff_blocks_from_messages(value.get("messages")))
        if isinstance(value.get("patches"), list):
            patches.extend(_clean_patch(str(item)) for item in value["patches"])
        return [patch for patch in patches if patch.strip()]
    if isinstance(value, list):
        patches = []
        for item in value:
            patches.extend(_patches_from_generator_output(item))
        return patches
    if isinstance(value, str):
        patch = _clean_patch(value)
        if patch.strip():
            return [patch]
        return _extract_diff_blocks_from_messages(value)
    return []


def write_prompt(prompt_root: Path, task: BenchmarkTask, config_name: str, seed: int) -> Path:
    prompt_root.mkdir(parents=True, exist_ok=True)
    record = task.record
    prompt = (
        "You are generating a candidate patch for a SWE-bench task.\n"
        "Return only a unified git diff. Do not include test results or reference patches.\n\n"
        f"Instance ID: {task.instance_id}\n"
        f"Repository: {record.get('repo', '')}\n"
        f"Base commit: {record.get('base_commit', '')}\n"
        f"Config: {config_name}\n"
        f"Seed: {seed}\n\n"
        f"Problem statement:\n{record.get('problem_statement', '')}\n\n"
        f"Hints:\n{record.get('hints_text', '')}\n\n"
        f"FAIL_TO_PASS:\n{record.get('FAIL_TO_PASS', '')}\n"
    )
    path = prompt_root / _safe_name(task.instance_id) / f"{_safe_name(config_name)}_{seed}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt, encoding="utf-8")
    return path


def write_stage6b1_jsonl(path: Path, rows: Sequence[CandidateOutput]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(asdict(row), sort_keys=True) for row in rows), encoding="utf-8")


def collect_environment(dataset_name: str, split: str) -> Dict[str, object]:
    packages = {}
    for package in ("datasets", "numpy", "swebench", "mini-swe-agent", "pytest"):
        packages[package] = _pip_version(package)
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "dataset_name": dataset_name,
        "split": split,
        "packages": packages,
        "mini_swe_agent_available": _mini_swe_agent_available(),
        "runner_argv": sys.argv,
    }


def _candidate_diversity_audit(rows: Sequence[CandidateOutput]) -> Dict[str, object]:
    grouped: Dict[str, List[CandidateOutput]] = defaultdict(list)
    for row in rows:
        grouped[row.instance_id].append(row)
    per_task = []
    all_distances = []
    for instance_id, task_rows in sorted(grouped.items()):
        distances = []
        for left_index in range(len(task_rows)):
            for right_index in range(left_index + 1, len(task_rows)):
                distance = 1.0 - _patch_similarity(task_rows[left_index].model_patch, task_rows[right_index].model_patch)
                distances.append(distance)
                all_distances.append(distance)
        per_task.append(
            {
                "instance_id": instance_id,
                "candidate_count": len(task_rows),
                "pair_count": len(distances),
                "mean_pairwise_distance": float(sum(distances) / len(distances)) if distances else 0.0,
                "min_pairwise_distance": float(min(distances)) if distances else 0.0,
            }
        )
    return {
        "method": "1 - SequenceMatcher ratio on normalized patch text",
        "tasks_checked": len(grouped),
        "mean_pairwise_distance": float(sum(all_distances) / len(all_distances)) if all_distances else 0.0,
        "min_pairwise_distance": float(min(all_distances)) if all_distances else 0.0,
        "per_task": per_task[:200],
    }


def _select_diverse_k(records: Sequence[CandidateAttempt], k: int, seed: int) -> List[CandidateAttempt]:
    rng = random.Random(seed + 6_281_401)
    grouped: Dict[str, List[CandidateAttempt]] = defaultdict(list)
    for row in records:
        grouped[f"{row.source_name}|{row.generator_name}|{row.model_name_or_path}|{row.config_name}"].append(row)
    for rows in grouped.values():
        rng.shuffle(rows)
    keys = list(grouped)
    rng.shuffle(keys)
    selected: List[CandidateAttempt] = []
    while len(selected) < k and keys:
        next_keys = []
        for key in keys:
            if grouped[key] and len(selected) < k:
                selected.append(grouped[key].pop(0))
            if grouped[key]:
                next_keys.append(key)
        keys = next_keys
    if len(selected) < k:
        remaining = [row for rows in grouped.values() for row in rows]
        rng.shuffle(remaining)
        selected.extend(remaining[: k - len(selected)])
    return selected[:k]


def _write_candidate_patch_file(root: Path, instance_id: str, candidate_id: str, patch: str) -> Path:
    path = root / _safe_name(instance_id) / f"{_safe_name(candidate_id)}.diff"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_ensure_trailing_newline(patch), encoding="utf-8")
    return path


def _write_raw_trajectory(
    root: Path,
    source_group: str,
    instance_id: str,
    generator_name: str,
    row_index: int,
    sample_index: int,
    payload: Dict[str, object],
) -> Path:
    digest = hashlib.sha256(json.dumps(_json_safe(payload), sort_keys=True).encode("utf-8")).hexdigest()[:12]
    path = root / _safe_name(source_group) / _safe_name(instance_id) / f"{row_index:06d}_{sample_index:02d}_{_safe_name(generator_name)}_{digest}.json"
    _write_json(path, payload)
    return path


def _attempt_audit_row(attempt: CandidateAttempt, patch_hash: str) -> Dict[str, object]:
    return {
        "instance_id": attempt.instance_id,
        "generator_name": attempt.generator_name,
        "model_name_or_path": attempt.model_name_or_path,
        "seed": attempt.seed,
        "temperature": attempt.temperature,
        "config_name": attempt.config_name,
        "patch_hash": patch_hash,
        "trajectory_path": attempt.trajectory_path,
        "source_name": attempt.source_name,
        "source_row_index": attempt.source_row_index,
    }


def _extract_diff_blocks_from_messages(messages: object) -> List[str]:
    texts: List[str] = []
    if isinstance(messages, str):
        parsed_messages = _json_message_payload(messages)
        if parsed_messages is not None:
            return _extract_diff_blocks_from_messages(parsed_messages)
        texts = [messages]
    elif isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                content = message.get("content") or message.get("text") or message.get("message") or ""
                if isinstance(content, list):
                    texts.extend(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
                else:
                    texts.append(str(content))
            else:
                texts.append(str(message))
    else:
        texts = [str(messages)]
    texts = [_normalize_message_text(text) for text in texts]
    patches: List[str] = []
    for text in texts:
        for fenced in re.findall(r"```(?:diff|patch)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
            patch = _clean_patch(fenced)
            if patch.strip():
                patches.append(patch)
        lines = text.splitlines()
        starts = [index for index, line in enumerate(lines) if line.startswith("diff --git ")]
        for start in starts:
            block = []
            for line in lines[start:]:
                if line.startswith("```"):
                    break
                block.append(line)
            patch = _clean_patch("\n".join(block))
            if patch.strip():
                patches.append(patch)
    deduped = []
    seen = set()
    for patch in patches:
        patch_hash = stable_patch_hash(patch)
        if patch_hash not in seen:
            seen.add(patch_hash)
            deduped.append(_ensure_trailing_newline(patch))
    return deduped


def _json_message_payload(text: str) -> object | None:
    stripped = str(text or "").strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        parsed = json.loads(stripped)
    except Exception:
        return None
    return parsed if isinstance(parsed, (list, dict)) else None


def _normalize_message_text(text: str) -> str:
    value = str(text or "")
    if "\\n" in value and value.count("\\n") > value.count("\n"):
        try:
            return bytes(value, "utf-8").decode("unicode_escape")
        except Exception:
            return value.replace("\\n", "\n")
    return value


def _load_benchmark_jsonl(path: Path) -> Dict[str, Dict[str, object]]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict) and row.get("instance_id"):
            rows[str(row["instance_id"])] = dict(row)
    return rows


def _read_task_ids(path: Path) -> List[str]:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            values.append(stripped)
    return values


def _format_command_template(
    template: str,
    task: BenchmarkTask,
    prompt_path: Path,
    output_path: Path,
    seed: int,
    temperature: float,
    model: str,
    config_name: str,
) -> str:
    return str(template).format(
        instance_id=task.instance_id,
        repo=task.record.get("repo", ""),
        base_commit=task.record.get("base_commit", ""),
        prompt_path=str(prompt_path),
        output_path=str(output_path),
        seed=int(seed),
        temperature=float(temperature),
        model=model,
        config_name=config_name,
    )


def _mini_swe_agent_available() -> bool:
    if shutil.which("mini-swe-agent") or shutil.which("mini"):
        return True
    try:
        proc = subprocess.run([sys.executable, "-m", "pip", "show", "mini-swe-agent"], text=True, capture_output=True, timeout=30)
        return proc.returncode == 0
    except Exception:
        return False


def _pip_version(package: str) -> str:
    try:
        proc = subprocess.run([sys.executable, "-m", "pip", "show", package], text=True, capture_output=True, timeout=60)
        for line in (proc.stdout or "").splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
        return "not found"
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _patch_similarity(left: str, right: str) -> float:
    left_norm = "\n".join(line.rstrip() for line in str(left).strip().splitlines())
    right_norm = "\n".join(line.rstrip() for line in str(right).strip().splitlines())
    if not left_norm or not right_norm:
        return 0.0
    max_len = 40_000
    return float(SequenceMatcher(None, left_norm[:max_len], right_norm[:max_len]).ratio())


def _clean_patch(value: str) -> str:
    text = str(value or "").replace("\r\n", "\n")
    fenced = re.search(r"```(?:diff|patch)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced and "diff --git " in fenced.group(1):
        text = fenced.group(1)
    if "diff --git " in text:
        text = text[text.find("diff --git ") :]
    return text.strip() + ("\n" if text.strip() else "")


def _looks_like_unified_diff(value: str) -> bool:
    text = str(value or "")
    return (
        "diff --git " in text
        and ("--- " in text or "+++ " in text)
        and bool(re.search(r"^@@(?:\s|$)", text, flags=re.MULTILINE))
    )


def _ensure_trailing_newline(text: str) -> str:
    value = str(text)
    return value if value.endswith("\n") else value + "\n"


def _clean_generator_name(value: object) -> str:
    raw = str(value or "unknown_generator")
    cleaned = re.sub(r"[^A-Za-z0-9_.:/@+-]+", "_", raw).strip("_")
    return cleaned[:160] or "unknown_generator"


def _safe_name(value: object) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))
    return safe[:160] or "value"


def _instance_id_from_trajectory(trajectory_id: str) -> str:
    return str(trajectory_id).rsplit("_run", 1)[0]


def _json_obj(value: object) -> Dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _json_safe(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _parse_int(value: object, default: int) -> int:
    parsed = _optional_int(value)
    return default if parsed is None else parsed


def _parse_float(value: object, default: float) -> float:
    parsed = _optional_float(value)
    return default if parsed is None else parsed


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def _write_json(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(row), indent=2, sort_keys=True), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
