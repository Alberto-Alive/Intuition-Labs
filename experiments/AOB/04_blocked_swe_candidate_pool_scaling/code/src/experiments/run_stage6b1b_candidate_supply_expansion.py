from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

from src.datasets.swe_patch_selection_dataset import STAGE6_NUM_CANDIDATES, stable_patch_hash
from src.experiments.run_stage6b1_candidate_generation import (
    BENCHMARK as STAGE6B1_BENCHMARK,
    DEFAULT_ANTIEVAL_DATASET,
    DEFAULT_CODERFORGE_DATASET,
    DEFAULT_DATASET_NAME,
    DEFAULT_HANSPETER_DATASET,
    LOCAL_PREDICTION_PATTERNS,
    BenchmarkTask,
    CandidateOutput,
    _candidate_diversity_audit,
    _clean_generator_name,
    _clean_patch,
    _ensure_trailing_newline,
    _format_command_template,
    _iter_prediction_rows,
    _json_obj,
    _json_safe,
    _mini_swe_agent_available,
    _now,
    _optional_float,
    _optional_int,
    _parse_float,
    _parse_int,
    _patch_similarity,
    _read_jsonl,
    _safe_name,
    _instance_id_from_trajectory,
    _write_candidate_patch_file,
    _write_json,
    _write_raw_trajectory,
    collect_environment,
    load_benchmark_tasks,
    write_stage6b1_jsonl,
)


BENCHMARK = "stage6b1b_candidate_supply_expansion"
DEFAULT_OUTPUT_PATH = Path("results/stage6b1b_generated_candidates.jsonl")
DEFAULT_ATTEMPTS_PATH = Path("results/stage6b1b_generation_attempts.jsonl")
DEFAULT_AUDIT_PATH = Path("results/stage6b1b_generation_audit.json")
DEFAULT_REPORT_PATH = Path("reports/STAGE6B1B_CANDIDATE_SUPPLY_EXPANSION.md")
DEFAULT_TRAJECTORY_ROOT = Path("results/stage6b1b_trajectories")
DEFAULT_CANDIDATE_PATCH_ROOT = Path("results/stage6b1b_candidate_patches")
DEFAULT_PROMPT_ROOT = Path("results/stage6b1b_prompts")
DEFAULT_MAX_ATTEMPTS_PER_TASK = 32
DEFAULT_DUPLICATE_RATE_THRESHOLD = 0.25
DEFAULT_NEAR_REFERENCE_THRESHOLD = 0.98
DEFAULT_PROMPT_VARIANTS = ("minimal_fix", "test_failure_fix", "localized_patch", "conservative_patch")
DEFAULT_TEMPERATURES = ("0.0", "0.2", "0.7", "1.0")

TERMINAL_STATUSES = {
    "accepted",
    "task_skipped_no_source_artifact",
    "task_skipped_no_matching_instance_id",
    "task_skipped_source_has_no_patch_field",
    "task_skipped_source_has_no_trajectory",
    "task_skipped_repo_not_supported",
    "task_skipped_generation_backend_unavailable",
    "task_skipped_budget_limit",
    "task_skipped_unknown",
    "repo_setup_failed",
    "generator_call_failed",
    "timeout",
    "empty_response",
    "no_patch_found",
    "invalid_diff",
    "duplicate_patch",
    "exact_reference_hash",
    "near_reference_similarity_flag",
    "patch_apply_check_failed",
    "unknown_error",
}

TASK_SKIPPED_STATUSES = {
    status for status in TERMINAL_STATUSES if status.startswith("task_skipped_")
}


@dataclass(frozen=True)
class RawGenerationAttempt:
    instance_id: str
    attempt_id: str
    model_name_or_path: str
    model_patch: str
    generator_name: str
    seed: int | None
    temperature: float | None
    config_name: str
    prompt_variant: str
    trajectory_path: str
    source_name: str
    source_row_index: int | None
    raw_response: str
    forced_terminal_status: str | None = None
    failure_message: str | None = None


@dataclass(frozen=True)
class DiffValidation:
    valid: bool
    reason: str | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6B.1b candidate supply expansion for generated SWE-bench candidate patches.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default="test")
    parser.add_argument("--benchmark-jsonl", default=None)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--task-ids-file", default=None)
    parser.add_argument("--n-tasks", type=int, default=25)
    parser.add_argument("--k", type=int, default=STAGE6_NUM_CANDIDATES)
    parser.add_argument("--seed", type=int, default=606_282)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--attempts-output", default=str(DEFAULT_ATTEMPTS_PATH))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--trajectory-root", default=str(DEFAULT_TRAJECTORY_ROOT))
    parser.add_argument("--candidate-patch-root", default=str(DEFAULT_CANDIDATE_PATCH_ROOT))
    parser.add_argument("--prompt-root", default=str(DEFAULT_PROMPT_ROOT))
    parser.add_argument("--source-jsonl", action="append", default=[])
    parser.add_argument("--source-json", action="append", default=[])
    parser.add_argument("--source-root", action="append", default=[])
    parser.add_argument("--previous-stage6b1-jsonl", action="append", default=[])
    parser.add_argument("--include-default-hf-public", action="store_true")
    parser.add_argument("--enable-mini-swe-agent", action="store_true")
    parser.add_argument("--mini-swe-agent-command-template", action="append", default=[])
    parser.add_argument("--mini-swe-agent-config", action="append", default=[])
    parser.add_argument("--api-generator-command-template", action="append", default=[])
    parser.add_argument("--api-generator-model", action="append", default=[])
    parser.add_argument("--local-generator-command-template", action="append", default=[])
    parser.add_argument("--local-generator-model", action="append", default=[])
    parser.add_argument("--generation-seed", action="append", default=[])
    parser.add_argument("--temperature", action="append", default=[])
    parser.add_argument("--prompt-variant", action="append", default=[])
    parser.add_argument("--max-attempts-per-task", type=int, default=DEFAULT_MAX_ATTEMPTS_PER_TASK)
    parser.add_argument("--generator-timeout-seconds", type=int, default=900)
    parser.add_argument("--apply-check", action="store_true")
    parser.add_argument("--require-apply-check", action="store_true")
    parser.add_argument("--repo-cache-root", default=None)
    parser.add_argument("--apply-check-timeout-seconds", type=int, default=60)
    parser.add_argument("--duplicate-rate-threshold", type=float, default=DEFAULT_DUPLICATE_RATE_THRESHOLD)
    parser.add_argument("--near-reference-threshold", type=float, default=DEFAULT_NEAR_REFERENCE_THRESHOLD)
    args = parser.parse_args()

    result = run_stage6b1b_candidate_supply_expansion(
        dataset_name=str(args.dataset_name),
        split=str(args.split),
        benchmark_jsonl=None if args.benchmark_jsonl is None else Path(args.benchmark_jsonl),
        task_ids=tuple(str(value) for value in args.task_id),
        task_ids_file=None if args.task_ids_file is None else Path(args.task_ids_file),
        n_tasks=int(args.n_tasks),
        k=int(args.k),
        seed=int(args.seed),
        output_path=Path(args.output),
        attempts_path=Path(args.attempts_output),
        audit_path=Path(args.audit),
        report_path=Path(args.report),
        trajectory_root=Path(args.trajectory_root),
        candidate_patch_root=Path(args.candidate_patch_root),
        prompt_root=Path(args.prompt_root),
        source_jsonl=tuple(str(value) for value in args.source_jsonl),
        source_json=tuple(str(value) for value in args.source_json),
        source_roots=tuple(str(value) for value in args.source_root),
        previous_stage6b1_jsonl=tuple(str(value) for value in args.previous_stage6b1_jsonl),
        include_default_hf_public=bool(args.include_default_hf_public),
        enable_mini_swe_agent=bool(args.enable_mini_swe_agent),
        mini_swe_agent_command_templates=tuple(str(value) for value in args.mini_swe_agent_command_template),
        mini_swe_agent_configs=tuple(str(value) for value in args.mini_swe_agent_config),
        api_generator_command_templates=tuple(str(value) for value in args.api_generator_command_template),
        api_generator_models=tuple(str(value) for value in args.api_generator_model),
        local_generator_command_templates=tuple(str(value) for value in args.local_generator_command_template),
        local_generator_models=tuple(str(value) for value in args.local_generator_model),
        generation_seeds=tuple(str(value) for value in args.generation_seed),
        temperatures=tuple(str(value) for value in args.temperature),
        prompt_variants=tuple(str(value) for value in args.prompt_variant),
        max_attempts_per_task=int(args.max_attempts_per_task),
        generator_timeout_seconds=int(args.generator_timeout_seconds),
        apply_check=bool(args.apply_check),
        require_apply_check=bool(args.require_apply_check),
        repo_cache_root=None if args.repo_cache_root is None else Path(args.repo_cache_root),
        apply_check_timeout_seconds=int(args.apply_check_timeout_seconds),
        duplicate_rate_threshold=float(args.duplicate_rate_threshold),
        near_reference_threshold=float(args.near_reference_threshold),
    )
    audit = result["audit"]
    print(
        "stage6b1b: wrote {output}, {attempts}, {audit_path}, {report}; rows={rows}; tasks_k8={tasks_k8}; gates_pass={gates}".format(
            output=args.output,
            attempts=args.attempts_output,
            audit_path=args.audit,
            report=args.report,
            rows=audit.get("accepted_candidate_rows", 0),
            tasks_k8=audit.get("tasks_with_at_least_8_candidates", 0),
            gates=audit.get("quality_gates", {}).get("candidate_supply_expansion_quality_gates_pass", False),
        )
    )


def run_stage6b1b_candidate_supply_expansion(
    dataset_name: str,
    split: str,
    benchmark_jsonl: Path | None,
    task_ids: Sequence[str],
    task_ids_file: Path | None,
    n_tasks: int,
    k: int,
    seed: int,
    output_path: Path,
    attempts_path: Path,
    audit_path: Path,
    report_path: Path,
    trajectory_root: Path,
    candidate_patch_root: Path,
    prompt_root: Path,
    source_jsonl: Sequence[str],
    source_json: Sequence[str],
    source_roots: Sequence[str],
    previous_stage6b1_jsonl: Sequence[str],
    include_default_hf_public: bool,
    enable_mini_swe_agent: bool,
    mini_swe_agent_command_templates: Sequence[str],
    mini_swe_agent_configs: Sequence[str],
    api_generator_command_templates: Sequence[str],
    api_generator_models: Sequence[str],
    local_generator_command_templates: Sequence[str],
    local_generator_models: Sequence[str],
    generation_seeds: Sequence[str],
    temperatures: Sequence[str],
    prompt_variants: Sequence[str],
    max_attempts_per_task: int,
    generator_timeout_seconds: int,
    apply_check: bool,
    require_apply_check: bool,
    repo_cache_root: Path | None,
    apply_check_timeout_seconds: int,
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
    selected_task_ids = tuple(task_records)
    source_audit: Dict[str, object] = {"sources": [], "errors": []}
    raw_attempts: List[RawGenerationAttempt] = []
    raw_attempts.extend(
        load_prediction_attempt_sources(
            task_ids=selected_task_ids,
            trajectory_root=trajectory_root,
            source_jsonl=source_jsonl,
            source_json=source_json,
            source_roots=source_roots,
            previous_stage6b1_jsonl=previous_stage6b1_jsonl,
            source_audit=source_audit,
        )
    )
    if include_default_hf_public:
        raw_attempts.extend(load_hf_public_attempt_sources(selected_task_ids, trajectory_root, source_audit))
    generation_seed_values = _generation_seed_values(generation_seeds, seed)
    temperature_values = _temperature_values(temperatures)
    prompt_variant_values = tuple(str(value) for value in (prompt_variants or DEFAULT_PROMPT_VARIANTS))
    raw_attempts.extend(
        run_command_generation_backends(
            tasks=task_records,
            trajectory_root=trajectory_root,
            prompt_root=prompt_root,
            enable_mini_swe_agent=enable_mini_swe_agent,
            mini_swe_agent_command_templates=mini_swe_agent_command_templates,
            mini_swe_agent_configs=mini_swe_agent_configs,
            api_generator_command_templates=api_generator_command_templates,
            api_generator_models=api_generator_models,
            local_generator_command_templates=local_generator_command_templates,
            local_generator_models=local_generator_models,
            generation_seeds=generation_seed_values,
            temperatures=temperature_values,
            prompt_variants=prompt_variant_values,
            max_attempts_per_task=max_attempts_per_task,
            timeout_seconds=generator_timeout_seconds,
            source_audit=source_audit,
        )
    )
    accepted, attempt_rows, build_audit = select_stage6b1b_candidates(
        raw_attempts=raw_attempts,
        task_records=task_records,
        source_audit=source_audit,
        k=k,
        seed=seed,
        max_attempts_per_task=max_attempts_per_task,
        candidate_patch_root=candidate_patch_root,
        duplicate_rate_threshold=duplicate_rate_threshold,
        near_reference_threshold=near_reference_threshold,
        apply_check=apply_check,
        require_apply_check=require_apply_check,
        repo_cache_root=repo_cache_root,
        apply_check_timeout_seconds=apply_check_timeout_seconds,
    )
    write_stage6b1_jsonl(output_path, accepted)
    write_jsonl(attempts_path, attempt_rows)
    environment = collect_environment(dataset_name, split)
    audit = build_stage6b1b_audit(
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
        raw_attempts=raw_attempts,
        attempt_rows=attempt_rows,
        output_path=output_path,
        attempts_path=attempts_path,
        trajectory_root=trajectory_root,
        candidate_patch_root=candidate_patch_root,
        environment=environment,
        duplicate_rate_threshold=duplicate_rate_threshold,
    )
    _write_json(audit_path, audit)
    report = render_stage6b1b_report(audit=audit, output_path=output_path, attempts_path=attempts_path, audit_path=audit_path, report_path=report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return {"audit": audit, "rows": accepted, "attempts": attempt_rows}


def load_prediction_attempt_sources(
    task_ids: Sequence[str],
    trajectory_root: Path,
    source_jsonl: Sequence[str],
    source_json: Sequence[str],
    source_roots: Sequence[str],
    previous_stage6b1_jsonl: Sequence[str],
    source_audit: Dict[str, object],
) -> List[RawGenerationAttempt]:
    rows: List[RawGenerationAttempt] = []
    allowed_ids = set(task_ids)
    for path in source_jsonl:
        rows.extend(_load_prediction_attempt_file(Path(path), allowed_ids, trajectory_root, source_audit, forced_kind="jsonl", source_group="local_prediction"))
    for path in source_json:
        rows.extend(_load_prediction_attempt_file(Path(path), allowed_ids, trajectory_root, source_audit, forced_kind="json", source_group="local_prediction"))
    for path in previous_stage6b1_jsonl:
        rows.extend(_load_prediction_attempt_file(Path(path), allowed_ids, trajectory_root, source_audit, forced_kind="jsonl", source_group="previous_stage6b1"))
    for root in source_roots:
        rows.extend(_load_prediction_attempt_root(Path(root), allowed_ids, trajectory_root, source_audit))
    return rows


def load_hf_public_attempt_sources(
    task_ids: Sequence[str],
    trajectory_root: Path,
    source_audit: Dict[str, object],
) -> List[RawGenerationAttempt]:
    rows: List[RawGenerationAttempt] = []
    rows.extend(_load_hf_coderforge_expanded(task_ids, trajectory_root, source_audit))
    rows.extend(_load_hf_message_dataset_expanded(DEFAULT_HANSPETER_DATASET, task_ids, trajectory_root, source_audit))
    rows.extend(_load_hf_message_dataset_expanded(DEFAULT_ANTIEVAL_DATASET, task_ids, trajectory_root, source_audit))
    source_audit.setdefault("sources", []).append(
        {
            "source_name": "hf-public-stage6b1b-expanded",
            "datasets": [DEFAULT_CODERFORGE_DATASET, DEFAULT_HANSPETER_DATASET, DEFAULT_ANTIEVAL_DATASET],
            "candidate_attempts": len(rows),
            "multi_diff_trajectory_expansion": "distinct generated patch hashes only",
        }
    )
    return rows


def _load_hf_coderforge_expanded(
    task_ids: Sequence[str],
    trajectory_root: Path,
    source_audit: Dict[str, object],
) -> List[RawGenerationAttempt]:
    allowed_ids = set(task_ids)
    source_name = f"hf:{DEFAULT_CODERFORGE_DATASET}:trajectory+messages"
    summary: Dict[str, object] = {
        "source_name": source_name,
        "loaded": False,
        "records": 0,
        "candidate_attempts": 0,
        "terminal_no_patch_rows": 0,
        "unique_instance_ids": [],
        "matching_records": 0,
        "rows_with_patch_field": 0,
        "rows_with_messages": 0,
        "rows_with_trajectory": 0,
        "error": None,
    }
    rows: List[RawGenerationAttempt] = []
    try:
        from datasets import load_dataset

        dataset = load_dataset(DEFAULT_CODERFORGE_DATASET, "trajectory", split="train")
        counters: Dict[tuple[str, str], int] = defaultdict(int)
        unique_ids = set()
        for row_index, row in enumerate(dataset):
            ds = _json_obj(row.get("ds", {}))
            instance_id = str(ds.get("instance_id") or _instance_id_from_trajectory(str(row.get("trajectory_id", ""))) or "")
            if instance_id:
                unique_ids.add(instance_id)
            if row.get("output_patch"):
                summary["rows_with_patch_field"] = int(summary["rows_with_patch_field"]) + 1
            if row.get("messages"):
                summary["rows_with_messages"] = int(summary["rows_with_messages"]) + 1
            if row.get("messages") or row.get("agent_log") or row.get("trajectory"):
                summary["rows_with_trajectory"] = int(summary["rows_with_trajectory"]) + 1
            if instance_id not in allowed_ids:
                continue
            summary["matching_records"] = int(summary["matching_records"]) + 1
            generator = _clean_generator_name(row.get("exp_name") or row.get("model") or "CoderForge-Preview-32B")
            model = str(row.get("model") or row.get("exp_name") or generator)
            patch_values: List[str] = []
            patch_values.extend(_patch_candidates_from_value(row.get("output_patch"), explicit_patch_field=True))
            patch_values.extend(_patch_candidates_from_value(row.get("messages"), explicit_patch_field=False))
            patch_values.extend(_patch_candidates_from_value(row.get("agent_log"), explicit_patch_field=False))
            patch_values.extend(_patch_candidates_from_value(row.get("trajectory"), explicit_patch_field=False))
            patches = _dedupe_patch_texts(patch_values)
            if not patches:
                patches = [""]
                summary["terminal_no_patch_rows"] = int(summary["terminal_no_patch_rows"]) + 1
            for patch_index, patch in enumerate(patches):
                sample = counters[(instance_id, generator)]
                counters[(instance_id, generator)] += 1
                trajectory_path = _write_raw_trajectory(
                    trajectory_root,
                    source_group="hf_coderforge_expanded",
                    instance_id=instance_id,
                    generator_name=generator,
                    row_index=row_index,
                    sample_index=sample,
                    payload={
                        "source_dataset": DEFAULT_CODERFORGE_DATASET,
                        "source_row_index": row_index,
                        "patch_index": patch_index,
                        "raw_row": _redact_source_row(dict(row)),
                    },
                )
                rows.append(
                    RawGenerationAttempt(
                        instance_id=instance_id,
                        attempt_id=f"hf-coderforge-expanded-{row_index:08d}-{patch_index:02d}",
                        model_name_or_path=model,
                        model_patch=patch,
                        generator_name=generator,
                        seed=_optional_int(row.get("seed")),
                        temperature=_optional_float(row.get("temperature")),
                        config_name=str(row.get("exp_name") or "hf_coderforge_expanded"),
                        prompt_variant="hf_coderforge_expanded",
                        trajectory_path=str(trajectory_path),
                        source_name=source_name,
                        source_row_index=row_index,
                        raw_response=json.dumps(_json_safe({"output_patch": row.get("output_patch"), "messages": row.get("messages")}), sort_keys=True),
                        forced_terminal_status=None if patch else "no_patch_found",
                    )
                )
        summary.update({"loaded": True, "records": len(dataset), "candidate_attempts": len(rows), "unique_instance_ids": sorted(unique_ids)})
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        source_audit.setdefault("errors", []).append(summary)
    source_audit.setdefault("sources", []).append(summary)
    return rows


def _load_hf_message_dataset_expanded(
    dataset_name: str,
    task_ids: Sequence[str],
    trajectory_root: Path,
    source_audit: Dict[str, object],
) -> List[RawGenerationAttempt]:
    allowed_ids = set(task_ids)
    source_name = f"hf:{dataset_name}:messages"
    summary: Dict[str, object] = {
        "source_name": source_name,
        "loaded": False,
        "records": 0,
        "candidate_attempts": 0,
        "terminal_no_patch_rows": 0,
        "unique_instance_ids": [],
        "matching_records": 0,
        "rows_with_patch_field": 0,
        "rows_with_messages": 0,
        "rows_with_trajectory": 0,
        "error": None,
    }
    rows: List[RawGenerationAttempt] = []
    try:
        from datasets import load_dataset

        dataset = load_dataset(dataset_name, split="train")
        counters: Dict[tuple[str, str], int] = defaultdict(int)
        unique_ids = set()
        for row_index, row in enumerate(dataset):
            instance_id = str(row.get("instance_id") or row.get("id") or "")
            if instance_id:
                unique_ids.add(instance_id)
            if any(row.get(key) for key in ("model_patch", "output_patch", "patch", "prediction", "diff", "response")):
                summary["rows_with_patch_field"] = int(summary["rows_with_patch_field"]) + 1
            if row.get("messages"):
                summary["rows_with_messages"] = int(summary["rows_with_messages"]) + 1
            if row.get("messages") or row.get("trajectory"):
                summary["rows_with_trajectory"] = int(summary["rows_with_trajectory"]) + 1
            if instance_id not in allowed_ids:
                continue
            summary["matching_records"] = int(summary["matching_records"]) + 1
            generator = _clean_generator_name(row.get("model") or row.get("agent") or dataset_name.rsplit("/", 1)[-1])
            model = str(row.get("model") or generator)
            patch_values = []
            for key in ("model_patch", "output_patch", "patch", "prediction", "diff", "response", "messages", "trajectory"):
                patch_values.extend(_patch_candidates_from_value(row.get(key), explicit_patch_field=key not in {"messages", "trajectory", "response"}))
            patches = _dedupe_patch_texts(patch_values)
            if not patches:
                patches = [""]
                summary["terminal_no_patch_rows"] = int(summary["terminal_no_patch_rows"]) + 1
            for patch_index, patch in enumerate(patches):
                sample = counters[(instance_id, generator)]
                counters[(instance_id, generator)] += 1
                trajectory_path = _write_raw_trajectory(
                    trajectory_root,
                    source_group="hf_messages_expanded",
                    instance_id=instance_id,
                    generator_name=generator,
                    row_index=row_index,
                    sample_index=sample,
                    payload={"source_dataset": dataset_name, "source_row_index": row_index, "patch_index": patch_index, "raw_row": _redact_source_row(dict(row))},
                )
                rows.append(
                    RawGenerationAttempt(
                        instance_id=instance_id,
                        attempt_id=f"hf-messages-expanded-{_safe_name(dataset_name)}-{row_index:08d}-{patch_index:02d}",
                        model_name_or_path=model,
                        model_patch=patch,
                        generator_name=generator,
                        seed=_optional_int(row.get("seed")),
                        temperature=_optional_float(row.get("temperature")),
                        config_name=str(row.get("config_name") or "hf_message_trajectory"),
                        prompt_variant=str(row.get("prompt_variant") or "hf_message_trajectory"),
                        trajectory_path=str(trajectory_path),
                        source_name=source_name,
                        source_row_index=row_index,
                        raw_response=json.dumps(_json_safe(row), sort_keys=True),
                        forced_terminal_status=None if patch else "no_patch_found",
                    )
                )
        summary.update({"loaded": True, "records": len(dataset), "candidate_attempts": len(rows), "unique_instance_ids": sorted(unique_ids)})
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        source_audit.setdefault("errors", []).append(summary)
    source_audit.setdefault("sources", []).append(summary)
    return rows


def _load_prediction_attempt_root(
    root: Path,
    task_ids: set[str],
    trajectory_root: Path,
    source_audit: Dict[str, object],
) -> List[RawGenerationAttempt]:
    rows: List[RawGenerationAttempt] = []
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
        rows.extend(_load_prediction_attempt_file(path, task_ids, trajectory_root, source_audit, forced_kind=None, source_group="local_prediction_root"))
    return rows


def _load_prediction_attempt_file(
    path: Path,
    task_ids: set[str],
    trajectory_root: Path,
    source_audit: Dict[str, object],
    forced_kind: str | None,
    source_group: str,
) -> List[RawGenerationAttempt]:
    source_name = f"file:{path.as_posix()}"
    summary: Dict[str, object] = {
        "source_name": source_name,
        "loaded": False,
        "records": 0,
        "candidate_attempts": 0,
        "terminal_no_patch_rows": 0,
        "unique_instance_ids": [],
        "matching_records": 0,
        "rows_with_patch_field": 0,
        "rows_with_messages": 0,
        "rows_with_trajectory": 0,
        "error": None,
    }
    rows: List[RawGenerationAttempt] = []
    try:
        parsed = list(_iter_prediction_rows(path, forced_kind))
        counters: Dict[tuple[str, str], int] = defaultdict(int)
        unique_ids = set()
        for row_index, row in enumerate(parsed):
            instance_id = _prediction_instance_id(row, path, task_ids)
            raw_instance_id = _prediction_instance_id(row, path, set())
            if raw_instance_id:
                unique_ids.add(raw_instance_id)
            if _row_has_patch_field(row):
                summary["rows_with_patch_field"] = int(summary["rows_with_patch_field"]) + 1
            if row.get("messages"):
                summary["rows_with_messages"] = int(summary["rows_with_messages"]) + 1
            if row.get("trajectory") or row.get("messages"):
                summary["rows_with_trajectory"] = int(summary["rows_with_trajectory"]) + 1
            if instance_id not in task_ids:
                continue
            summary["matching_records"] = int(summary["matching_records"]) + 1
            generator = _clean_generator_name(row.get("generator_name") or row.get("generator") or row.get("agent") or row.get("model_name_or_path") or path.parent.name)
            model = str(row.get("model_name_or_path") or row.get("model") or generator or "unknown_model")
            seed = _optional_int(row.get("seed"))
            temperature = _optional_float(row.get("temperature"))
            config_name = str(row.get("config_name") or row.get("config") or row.get("prompt_name") or row.get("candidate_id") or "local_prediction")
            prompt_variant = str(row.get("prompt_variant") or row.get("prompt_name") or config_name)
            patches, terminal_if_missing = _patch_candidates_from_prediction_row(row)
            if not patches:
                patches = [""]
                summary["terminal_no_patch_rows"] = int(summary["terminal_no_patch_rows"]) + 1
            for patch_index, patch in enumerate(patches):
                sample = counters[(instance_id, generator)]
                counters[(instance_id, generator)] += 1
                trajectory_path = _write_raw_trajectory(
                    trajectory_root,
                    source_group=source_group,
                    instance_id=instance_id,
                    generator_name=generator,
                    row_index=row_index,
                    sample_index=sample,
                    payload={
                        "source_file": path.as_posix(),
                        "source_row_index": row_index,
                        "patch_index": patch_index,
                        "raw_row": _redact_source_row(row),
                    },
                )
                rows.append(
                    RawGenerationAttempt(
                        instance_id=instance_id,
                        attempt_id=f"{source_group}-{_safe_name(instance_id)}-{row_index:06d}-{patch_index:02d}",
                        model_name_or_path=model,
                        model_patch=patch,
                        generator_name=generator,
                        seed=seed,
                        temperature=temperature,
                        config_name=config_name,
                        prompt_variant=prompt_variant,
                        trajectory_path=str(trajectory_path),
                        source_name=source_name,
                        source_row_index=row_index,
                        raw_response=json.dumps(_json_safe(row), sort_keys=True),
                        forced_terminal_status=terminal_if_missing if not patch else None,
                    )
                )
        summary.update({"loaded": True, "records": len(parsed), "candidate_attempts": len(rows), "unique_instance_ids": sorted(unique_ids)})
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        source_audit.setdefault("errors", []).append(summary)
    source_audit.setdefault("sources", []).append(summary)
    return rows


def _prediction_instance_id(row: Dict[str, object], path: Path, task_ids: set[str]) -> str:
    instance_id = str(row.get("instance_id") or row.get("id") or row.get("task_id") or row.get("issue_id") or "")
    if not instance_id and path.stem in task_ids:
        instance_id = path.stem
    return instance_id


def _patch_candidates_from_prediction_row(row: Dict[str, object]) -> tuple[List[str], str]:
    patches: List[str] = []
    explicit_patch_field_present = False
    for key in ("model_patch", "output_patch", "generated_patch", "prediction", "diff", "patch", "response"):
        if key not in row:
            continue
        explicit_patch_field_present = True
        patches.extend(_patch_candidates_from_value(row.get(key), explicit_patch_field=True))
    if row.get("messages"):
        patches.extend(_patch_candidates_from_value(row.get("messages"), explicit_patch_field=False))
    if row.get("trajectory"):
        patches.extend(_patch_candidates_from_value(row.get("trajectory"), explicit_patch_field=False))
    deduped = _dedupe_patch_texts(patches)
    if deduped:
        return deduped, ""
    return [], "empty_response" if explicit_patch_field_present else "no_patch_found"


def _row_has_patch_field(row: Dict[str, object]) -> bool:
    for key in ("model_patch", "output_patch", "generated_patch", "prediction", "diff", "patch", "response"):
        if key not in row:
            continue
        value = row.get(key)
        if value is not None and value != "":
            return True
    return False


def _patch_candidates_from_value(value: object, explicit_patch_field: bool) -> List[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        patches: List[str] = []
        for key in ("model_patch", "output_patch", "generated_patch", "prediction", "diff", "patch", "response", "messages", "trajectory", "content", "text"):
            if key in value:
                patches.extend(_patch_candidates_from_value(value.get(key), explicit_patch_field=explicit_patch_field and key not in {"messages", "trajectory", "content", "text", "response"}))
        return patches
    if isinstance(value, list):
        patches = []
        for item in value:
            patches.extend(_patch_candidates_from_value(item, explicit_patch_field=explicit_patch_field))
        return patches
    text = _normalize_text(str(value or ""))
    extracted = _extract_patch_candidates_from_text(text)
    if extracted:
        return extracted
    if explicit_patch_field and text.strip():
        return [_clean_patch(text)]
    return []


def _extract_patch_candidates_from_text(text: str) -> List[str]:
    parsed = _json_message_payload(text)
    if parsed is not None:
        return _patch_candidates_from_value(parsed, explicit_patch_field=False)
    patches: List[str] = []
    for fenced in re.findall(r"```(?:diff|patch)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
        patch = _clean_patch(fenced)
        if _text_contains_unified_diff(patch):
            patches.append(patch)
    lines = text.splitlines()
    starts = [index for index, line in enumerate(lines) if line.startswith("diff --git ")]
    for start_index, start in enumerate(starts):
        end = starts[start_index + 1] if start_index + 1 < len(starts) else len(lines)
        block = []
        for line in lines[start:end]:
            if line.startswith("```"):
                break
            block.append(line)
        patch = _clean_patch("\n".join(block))
        if _text_contains_unified_diff(patch):
            patches.append(patch)
    if not patches:
        unified_start = _first_unified_header_index(text)
        if unified_start >= 0:
            patches.append(_clean_patch(text[unified_start:]))
    return _dedupe_patch_texts(patches)


def run_command_generation_backends(
    tasks: Dict[str, BenchmarkTask],
    trajectory_root: Path,
    prompt_root: Path,
    enable_mini_swe_agent: bool,
    mini_swe_agent_command_templates: Sequence[str],
    mini_swe_agent_configs: Sequence[str],
    api_generator_command_templates: Sequence[str],
    api_generator_models: Sequence[str],
    local_generator_command_templates: Sequence[str],
    local_generator_models: Sequence[str],
    generation_seeds: Sequence[int],
    temperatures: Sequence[float],
    prompt_variants: Sequence[str],
    max_attempts_per_task: int,
    timeout_seconds: int,
    source_audit: Dict[str, object],
) -> List[RawGenerationAttempt]:
    rows: List[RawGenerationAttempt] = []
    rows.extend(
        _run_template_backend(
            backend_name="local-generator-wrapper",
            tasks=tasks,
            command_templates=local_generator_command_templates,
            model_values=tuple(local_generator_models) or ("local-generator",),
            config_values=("default",),
            trajectory_root=trajectory_root,
            prompt_root=prompt_root,
            generation_seeds=generation_seeds,
            temperatures=temperatures,
            prompt_variants=prompt_variants,
            max_attempts_per_task=max_attempts_per_task,
            timeout_seconds=timeout_seconds,
            source_audit=source_audit,
        )
    )
    rows.extend(
        _run_template_backend(
            backend_name="api-generator-wrapper",
            tasks=tasks,
            command_templates=api_generator_command_templates,
            model_values=tuple(api_generator_models) or ("api-generator",),
            config_values=("default",),
            trajectory_root=trajectory_root,
            prompt_root=prompt_root,
            generation_seeds=generation_seeds,
            temperatures=temperatures,
            prompt_variants=prompt_variants,
            max_attempts_per_task=max_attempts_per_task,
            timeout_seconds=timeout_seconds,
            source_audit=source_audit,
        )
    )
    mini_installed = _mini_swe_agent_available()
    if enable_mini_swe_agent and not mini_installed:
        source_audit.setdefault("errors", []).append(
            {
                "source_name": "mini-swe-agent",
                "enabled": True,
                "installed": False,
                "error": "mini-SWE-agent is not installed or no executable was found",
            }
        )
    if enable_mini_swe_agent and mini_installed:
        rows.extend(
            _run_template_backend(
                backend_name="mini-swe-agent",
                tasks=tasks,
                command_templates=mini_swe_agent_command_templates,
                model_values=("mini-swe-agent",),
                config_values=tuple(mini_swe_agent_configs) or ("default",),
                trajectory_root=trajectory_root,
                prompt_root=prompt_root,
                generation_seeds=generation_seeds,
                temperatures=(0.0,),
                prompt_variants=prompt_variants,
                max_attempts_per_task=max_attempts_per_task,
                timeout_seconds=timeout_seconds,
                source_audit=source_audit,
            )
        )
    elif not enable_mini_swe_agent:
        source_audit.setdefault("sources", []).append({"source_name": "mini-swe-agent", "enabled": False, "records": 0})
    return rows


def _run_template_backend(
    backend_name: str,
    tasks: Dict[str, BenchmarkTask],
    command_templates: Sequence[str],
    model_values: Sequence[str],
    config_values: Sequence[str],
    trajectory_root: Path,
    prompt_root: Path,
    generation_seeds: Sequence[int],
    temperatures: Sequence[float],
    prompt_variants: Sequence[str],
    max_attempts_per_task: int,
    timeout_seconds: int,
    source_audit: Dict[str, object],
) -> List[RawGenerationAttempt]:
    summary = {"source_name": backend_name, "command_templates": len(command_templates), "records": 0, "error": None}
    rows: List[RawGenerationAttempt] = []
    if not command_templates:
        source_audit.setdefault("sources", []).append(summary)
        return rows
    specs = []
    for template_index, template in enumerate(command_templates):
        for model in model_values:
            for config_name in config_values:
                for seed in generation_seeds:
                    for temperature in temperatures:
                        for prompt_variant in prompt_variants:
                            specs.append((template_index, template, model, config_name, int(seed), float(temperature), prompt_variant))
    for task_index, task in enumerate(tasks.values()):
        for attempt_index, (template_index, template, model, config_name, seed_value, temperature, prompt_variant) in enumerate(specs[: max(0, int(max_attempts_per_task))]):
            run_seed = seed_value + 6_282_001 + task_index * 10_000 + attempt_index
            full_config_name = f"{config_name}|{prompt_variant}|template_{template_index}|temp_{temperature:g}"
            prompt_path = write_variant_prompt(prompt_root, task, full_config_name, run_seed, prompt_variant)
            output_path = trajectory_root / f"{_safe_name(backend_name)}_outputs" / _safe_name(task.instance_id) / f"{_safe_name(model)}_{run_seed}.json"
            trajectory_path = trajectory_root / _safe_name(backend_name) / _safe_name(task.instance_id) / f"{_safe_name(model)}_{run_seed}.json"
            command = _format_command_template(
                template,
                task=task,
                prompt_path=prompt_path,
                output_path=output_path,
                seed=run_seed,
                temperature=temperature,
                model=model,
                config_name=full_config_name,
            )
            rows.extend(
                _run_generator_command_attempt(
                    command=command,
                    output_path=output_path,
                    trajectory_path=trajectory_path,
                    task=task,
                    generator_name=backend_name,
                    model_name=model,
                    seed=run_seed,
                    temperature=temperature,
                    config_name=full_config_name,
                    prompt_variant=prompt_variant,
                    source_name=backend_name,
                    source_row_index=attempt_index,
                    timeout_seconds=timeout_seconds,
                )
            )
    summary["records"] = len(rows)
    source_audit.setdefault("sources", []).append(summary)
    return rows


def _run_generator_command_attempt(
    command: str,
    output_path: Path,
    trajectory_path: Path,
    task: BenchmarkTask,
    generator_name: str,
    model_name: str,
    seed: int,
    temperature: float,
    config_name: str,
    prompt_variant: str,
    source_name: str,
    source_row_index: int,
    timeout_seconds: int,
) -> List[RawGenerationAttempt]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    started = _now()
    payload: Dict[str, object] = {
        "source_name": source_name,
        "command": command,
        "started_at_utc": started,
        "output_path": str(output_path),
        "instance_id": task.instance_id,
        "generator_name": generator_name,
        "model_name_or_path": model_name,
        "seed": seed,
        "temperature": temperature,
        "config_name": config_name,
        "prompt_variant": prompt_variant,
    }
    forced_status: str | None = None
    failure_message: str | None = None
    stdout = ""
    stderr = ""
    output_data: object = None
    try:
        proc = subprocess.run(command, shell=True, text=True, capture_output=True, timeout=timeout_seconds)
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        payload.update({"returncode": proc.returncode, "stdout": stdout, "stderr": stderr})
        if output_path.exists() and output_path.stat().st_size > 0:
            output_text = output_path.read_text(encoding="utf-8")
            try:
                output_data = json.loads(output_text)
            except Exception:
                output_data = output_text
        payload["generator_output"] = output_data
        if proc.returncode != 0:
            forced_status = "generator_call_failed"
            failure_message = stderr.strip()[:4000] or stdout.strip()[:4000] or f"returncode={proc.returncode}"
    except subprocess.TimeoutExpired as exc:
        forced_status = "timeout"
        stdout = str(exc.stdout or "")
        stderr = str(exc.stderr or "")
        failure_message = f"timeout after {timeout_seconds} seconds"
        payload.update({"returncode": None, "stdout": stdout, "stderr": stderr, "timeout_seconds": timeout_seconds})
    except Exception as exc:
        forced_status = "unknown_error"
        failure_message = f"{type(exc).__name__}: {exc}"
        payload.update({"returncode": None, "stdout": stdout, "stderr": stderr, "error": failure_message})
    payload["finished_at_utc"] = _now()
    _write_json(trajectory_path, payload)
    if forced_status is not None:
        return [
            RawGenerationAttempt(
                instance_id=task.instance_id,
                attempt_id=f"{_safe_name(source_name)}-{_safe_name(task.instance_id)}-{source_row_index:06d}",
                model_name_or_path=model_name,
                model_patch="",
                generator_name=generator_name,
                seed=seed,
                temperature=temperature,
                config_name=config_name,
                prompt_variant=prompt_variant,
                trajectory_path=str(trajectory_path),
                source_name=source_name,
                source_row_index=source_row_index,
                raw_response=stdout,
                forced_terminal_status=forced_status,
                failure_message=failure_message,
            )
        ]
    patches = _patch_candidates_from_value(output_data, explicit_patch_field=True)
    if not patches and stdout.strip():
        patches = _extract_patch_candidates_from_text(stdout)
    if not patches:
        status = "empty_response" if not stdout.strip() and output_data is None else "no_patch_found"
        return [
            RawGenerationAttempt(
                instance_id=task.instance_id,
                attempt_id=f"{_safe_name(source_name)}-{_safe_name(task.instance_id)}-{source_row_index:06d}",
                model_name_or_path=model_name,
                model_patch="",
                generator_name=generator_name,
                seed=seed,
                temperature=temperature,
                config_name=config_name,
                prompt_variant=prompt_variant,
                trajectory_path=str(trajectory_path),
                source_name=source_name,
                source_row_index=source_row_index,
                raw_response=stdout,
                forced_terminal_status=status,
            )
        ]
    return [
        RawGenerationAttempt(
            instance_id=task.instance_id,
            attempt_id=f"{_safe_name(source_name)}-{_safe_name(task.instance_id)}-{source_row_index:06d}-{patch_index:02d}",
            model_name_or_path=model_name,
            model_patch=patch,
            generator_name=generator_name,
            seed=seed,
            temperature=temperature,
            config_name=config_name,
            prompt_variant=prompt_variant,
            trajectory_path=str(trajectory_path),
            source_name=source_name,
            source_row_index=source_row_index,
            raw_response=stdout,
        )
        for patch_index, patch in enumerate(patches)
    ]


def select_stage6b1b_candidates(
    raw_attempts: Sequence[RawGenerationAttempt],
    task_records: Dict[str, BenchmarkTask],
    source_audit: Dict[str, object],
    k: int,
    seed: int,
    max_attempts_per_task: int,
    candidate_patch_root: Path,
    duplicate_rate_threshold: float,
    near_reference_threshold: float,
    apply_check: bool,
    require_apply_check: bool,
    repo_cache_root: Path | None,
    apply_check_timeout_seconds: int,
) -> tuple[List[CandidateOutput], List[Dict[str, object]], Dict[str, object]]:
    grouped_attempts: Dict[str, List[RawGenerationAttempt]] = defaultdict(list)
    for attempt in raw_attempts:
        if attempt.instance_id in task_records:
            grouped_attempts[attempt.instance_id].append(attempt)

    rng = random.Random(seed + 6_282_101)
    accepted_intermediate: Dict[str, List[tuple[RawGenerationAttempt, str, str, Dict[str, object]]]] = defaultdict(list)
    seen_hashes_by_task: Dict[str, set[str]] = defaultdict(set)
    attempt_rows: List[Dict[str, object]] = []

    for instance_id, task in task_records.items():
        task_attempts = grouped_attempts.get(instance_id, [])
        if not task_attempts:
            attempt_rows.append(_task_skipped_row(task, max_attempts_per_task, source_audit))
            continue
        attempts_used = 0
        for attempt in task_attempts:
            if attempts_used >= max(0, int(max_attempts_per_task)) or len(accepted_intermediate[instance_id]) >= int(k):
                break
            attempts_used += 1
            audit_row = _base_attempt_row(attempt, generation_attempt_index=attempts_used - 1)
            patch = _clean_patch(attempt.model_patch)
            patch_hash = stable_patch_hash(patch) if patch.strip() else ""
            audit_row["patch_hash"] = patch_hash
            audit_row["patch_length"] = len(patch)
            audit_row["patch_apply_check"] = {"status": "not_requested", "required": bool(require_apply_check)}
            if attempt.forced_terminal_status:
                audit_row["terminal_status"] = _coerce_terminal_status(attempt.forced_terminal_status)
                audit_row["failure_message"] = attempt.failure_message
                attempt_rows.append(audit_row)
                continue
            if not attempt.raw_response.strip() and not patch.strip():
                audit_row["terminal_status"] = "empty_response"
                attempt_rows.append(audit_row)
                continue
            if not patch.strip():
                audit_row["terminal_status"] = "no_patch_found"
                attempt_rows.append(audit_row)
                continue
            validation = validate_unified_diff(patch)
            if not validation.valid:
                audit_row["terminal_status"] = "invalid_diff"
                audit_row["invalid_diff_reason"] = validation.reason
                attempt_rows.append(audit_row)
                continue
            reference_hash = task.reference_hash
            if reference_hash and patch_hash == reference_hash:
                audit_row["terminal_status"] = "exact_reference_hash"
                attempt_rows.append(audit_row)
                continue
            similarity = _patch_similarity(patch, task.reference_patch)
            audit_row["near_reference_similarity"] = similarity
            if task.reference_patch.strip() and similarity >= near_reference_threshold:
                audit_row["terminal_status"] = "near_reference_similarity_flag"
                attempt_rows.append(audit_row)
                continue
            if patch_hash in seen_hashes_by_task[instance_id]:
                audit_row["terminal_status"] = "duplicate_patch"
                attempt_rows.append(audit_row)
                continue
            apply_status = run_patch_apply_check(
                patch=patch,
                task=task,
                apply_check=apply_check,
                require_apply_check=require_apply_check,
                repo_cache_root=repo_cache_root,
                timeout_seconds=apply_check_timeout_seconds,
            )
            audit_row["patch_apply_check"] = apply_status
            if require_apply_check and apply_status.get("status") == "repo_setup_failed":
                audit_row["terminal_status"] = "repo_setup_failed"
                attempt_rows.append(audit_row)
                continue
            if require_apply_check and apply_status.get("status") == "failed":
                audit_row["terminal_status"] = "patch_apply_check_failed"
                attempt_rows.append(audit_row)
                continue
            seen_hashes_by_task[instance_id].add(patch_hash)
            audit_row["terminal_status"] = "accepted"
            attempt_rows.append(audit_row)
            accepted_intermediate[instance_id].append((attempt, _ensure_trailing_newline(patch), patch_hash, audit_row))

    accepted: List[CandidateOutput] = []
    for instance_id in task_records:
        selected = list(accepted_intermediate.get(instance_id, []))
        rng.shuffle(selected)
        for out_index, (attempt, patch, patch_hash, audit_row) in enumerate(selected):
            candidate_id = f"{instance_id}-stage6b1b-candidate-{out_index:02d}-{patch_hash[:12]}"
            patch_path = _write_candidate_patch_file(candidate_patch_root, instance_id, candidate_id, patch)
            audit_row["candidate_id"] = candidate_id
            audit_row["candidate_patch_path"] = str(patch_path)
            accepted.append(
                CandidateOutput(
                    instance_id=instance_id,
                    candidate_id=candidate_id,
                    model_name_or_path=attempt.model_name_or_path,
                    model_patch=patch,
                    generator_name=attempt.generator_name,
                    seed=attempt.seed,
                    temperature=attempt.temperature,
                    trajectory_path=attempt.trajectory_path,
                    source_visible_to_selector=False,
                )
            )

    build_audit = _build_selection_audit(
        accepted=accepted,
        attempt_rows=attempt_rows,
        task_records=task_records,
        k=k,
        max_attempts_per_task=max_attempts_per_task,
        duplicate_rate_threshold=duplicate_rate_threshold,
        near_reference_threshold=near_reference_threshold,
    )
    return accepted, attempt_rows, build_audit


def validate_unified_diff(patch: str) -> DiffValidation:
    text = str(patch or "").replace("\r\n", "\n")
    if not text.strip():
        return DiffValidation(False, "empty_patch")
    lines = text.splitlines()
    if not any(line.startswith("--- ") for line in lines) or not any(line.startswith("+++ ") for line in lines):
        return DiffValidation(False, "missing_file_header")
    hunk_header = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@(?: .*)?$")
    hunk_indexes = [index for index, line in enumerate(lines) if line.startswith("@@")]
    if not hunk_indexes:
        return DiffValidation(False, "missing_hunk_header")
    if any(not hunk_header.match(lines[index]) for index in hunk_indexes):
        return DiffValidation(False, "malformed_hunk_header")
    changed_lines = 0
    in_hunk = False
    for line in lines:
        if line.startswith("diff --git "):
            in_hunk = False
            continue
        if line.startswith("@@"):
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith(("--- ", "+++ ")):
            return DiffValidation(False, "file_header_inside_hunk")
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            changed_lines += 1
        if line and not line.startswith((" ", "+", "-", "\\")):
            return DiffValidation(False, "invalid_hunk_line_prefix")
    if changed_lines == 0:
        return DiffValidation(False, "no_changed_lines")
    return DiffValidation(True)


def run_patch_apply_check(
    patch: str,
    task: BenchmarkTask,
    apply_check: bool,
    require_apply_check: bool,
    repo_cache_root: Path | None,
    timeout_seconds: int,
) -> Dict[str, object]:
    if not apply_check:
        return {"status": "not_requested", "required": bool(require_apply_check)}
    if repo_cache_root is None:
        return {"status": "repo_not_available", "required": bool(require_apply_check), "repo_cache_root": None}
    repo_path = _find_cached_repo(repo_cache_root, task)
    if repo_path is None:
        return {"status": "repo_not_available", "required": bool(require_apply_check), "repo_cache_root": str(repo_cache_root)}
    with tempfile.TemporaryDirectory(prefix="stage6b1b_apply_") as temp_dir:
        worktree = Path(temp_dir) / "repo"
        try:
            shutil.copytree(repo_path, worktree, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv", "node_modules"))
        except Exception as exc:
            return {"status": "repo_setup_failed", "required": bool(require_apply_check), "repo_path": str(repo_path), "error": f"{type(exc).__name__}: {exc}"}
        try:
            proc = subprocess.run(
                ["git", "apply", "--check", "-"],
                input=patch,
                text=True,
                capture_output=True,
                cwd=worktree,
                timeout=timeout_seconds,
            )
        except Exception as exc:
            return {"status": "repo_setup_failed", "required": bool(require_apply_check), "repo_path": str(repo_path), "error": f"{type(exc).__name__}: {exc}"}
    return {
        "status": "passed" if proc.returncode == 0 else "failed",
        "required": bool(require_apply_check),
        "repo_path": str(repo_path),
        "returncode": proc.returncode,
        "stderr": (proc.stderr or "")[:4000],
    }


def build_source_coverage_report(
    source_audit: Dict[str, object],
    raw_attempts: Sequence[RawGenerationAttempt],
    accepted: Sequence[CandidateOutput],
    attempt_rows: Sequence[Dict[str, object]],
    selected_task_ids: Sequence[str],
    k: int,
) -> Dict[str, object]:
    selected = tuple(str(value) for value in selected_task_ids)
    selected_set = set(selected)
    sources = [source for source in source_audit.get("sources", []) if isinstance(source, dict)]
    loaded_sources = [source for source in sources if bool(source.get("loaded", False))]
    unique_ids = set()
    total_raw_records = 0
    for source in loaded_sources:
        total_raw_records += int(source.get("records", 0) or 0)
        for instance_id in source.get("unique_instance_ids", []) or []:
            if instance_id:
                unique_ids.add(str(instance_id))
    for attempt in raw_attempts:
        if attempt.instance_id:
            unique_ids.add(attempt.instance_id)
    before_counts = Counter(
        attempt.instance_id
        for attempt in raw_attempts
        if attempt.instance_id in selected_set and str(attempt.model_patch or "").strip()
    )
    after_counts = Counter(row.instance_id for row in accepted)
    overlap = sorted(selected_set & unique_ids)
    missing = [
        {
            "instance_id": instance_id,
            "raw_candidates_before_filtering": int(before_counts.get(instance_id, 0)),
            "accepted_candidates_after_filtering": int(after_counts.get(instance_id, 0)),
            "missing_to_k": max(0, int(k) - int(after_counts.get(instance_id, 0))),
            "skip_status": _attempt_skip_status_for_instance(attempt_rows, instance_id),
        }
        for instance_id in selected
        if int(after_counts.get(instance_id, 0)) < int(k)
    ]
    top_sources = Counter(
        str(row.get("source_name"))
        for row in attempt_rows
        if row.get("terminal_status") == "accepted" and row.get("source_name")
    )
    return {
        "source_files_loaded": len(loaded_sources),
        "source_artifacts_loaded": len(loaded_sources),
        "total_raw_records_loaded": total_raw_records,
        "unique_instance_ids_in_sources": len(unique_ids),
        "overlap_with_selected_benchmark_tasks": {
            "count": len(overlap),
            "selected_task_count": len(selected),
            "instance_ids": overlap[:200],
        },
        "candidates_per_overlapped_instance_before_filtering": {
            instance_id: int(before_counts.get(instance_id, 0)) for instance_id in overlap
        },
        "candidates_per_overlapped_instance_after_filtering": {
            instance_id: int(after_counts.get(instance_id, 0)) for instance_id in overlap
        },
        "top_missing_instance_ids": missing[:50],
        "top_source_files_by_accepted_candidates": [
            {"source_name": source_name, "accepted_candidates": int(count)}
            for source_name, count in top_sources.most_common(50)
        ],
    }


def _attempt_skip_status_for_instance(attempt_rows: Sequence[Dict[str, object]], instance_id: str) -> str | None:
    for row in attempt_rows:
        if row.get("instance_id") == instance_id and str(row.get("terminal_status")) in TASK_SKIPPED_STATUSES:
            return str(row.get("terminal_status"))
    return None


def build_stage6b1b_audit(
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
    raw_attempts: Sequence[RawGenerationAttempt],
    attempt_rows: Sequence[Dict[str, object]],
    output_path: Path,
    attempts_path: Path,
    trajectory_root: Path,
    candidate_patch_root: Path,
    environment: Dict[str, object],
    duplicate_rate_threshold: float,
) -> Dict[str, object]:
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
    trajectory_rows = []
    missing_trajectories = []
    for row in accepted:
        exists = bool(row.trajectory_path and Path(row.trajectory_path).exists())
        trajectory_rows.append({"candidate_id": row.candidate_id, "trajectory_path": row.trajectory_path, "exists": exists})
        if not exists:
            missing_trajectories.append({"candidate_id": row.candidate_id, "trajectory_path": row.trajectory_path})
    missing_patch_files = [
        {"candidate_id": row.get("candidate_id"), "candidate_patch_path": row.get("candidate_patch_path")}
        for row in attempt_rows
        if row.get("terminal_status") == "accepted" and not Path(str(row.get("candidate_patch_path", ""))).exists()
    ]
    exact_reference_count = int(build_audit.get("exact_reference_patch_hash_audit", {}).get("hits_total", 0))
    duplicate_rate = float(build_audit.get("duplicate_patch_hash_audit", {}).get("duplicate_rate", 1.0))
    invalid_rate = float(build_audit.get("invalid_diff_audit", {}).get("invalid_diff_rate", 1.0))
    tasks_with_k = int(build_audit.get("tasks_with_at_least_8_candidates", 0))
    files_saved_pass = output_path.exists() and attempts_path.exists() and not missing_trajectories and not missing_patch_files
    source_coverage_report = build_source_coverage_report(
        source_audit=source_audit,
        raw_attempts=raw_attempts,
        accepted=accepted,
        attempt_rows=attempt_rows,
        selected_task_ids=tuple(benchmark_audit.get("selected_task_ids", [])),
        k=k,
    )
    quality_gates = {
        "at_least_25_tasks_attempted": int(build_audit.get("tasks_attempted", 0)) >= 25,
        "at_least_20_tasks_with_k8_unique_generated_candidates": tasks_with_k >= 20 if int(k) == STAGE6_NUM_CANDIDATES else tasks_with_k >= 20,
        "at_least_160_accepted_candidates_total": len(accepted) >= 160,
        "invalid_diff_rate_reported_and_below_50_percent": "invalid_diff_rate" in build_audit.get("invalid_diff_audit", {}) and invalid_rate < 0.50,
        "duplicate_rate_below_threshold": duplicate_rate <= float(duplicate_rate_threshold),
        "no_exact_reference_patch_hits": exact_reference_count == 0,
        "candidate_order_randomizable": bool(build_audit.get("candidate_order_randomizable", False)),
        "all_generated_candidate_files_saved": files_saved_pass,
        "no_raw_preserving_expansion": not bool(build_audit.get("raw_preserving_expansion_used", False)),
        "no_missing_candidate_fabrication": not bool(build_audit.get("missing_candidate_fabrication_used", False)),
        "selector_training_not_executed": True,
        "official_harness_not_executed": True,
        "no_final_model_claim_made": True,
    }
    quality_gates["candidate_supply_expansion_quality_gates_pass"] = all(bool(value) for value in quality_gates.values())
    return {
        "benchmark": BENCHMARK,
        "extends_benchmark": STAGE6B1_BENCHMARK,
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
        "attempts_path": str(attempts_path),
        "accepted_candidate_rows": len(accepted),
        "tasks_attempted": int(build_audit.get("tasks_attempted", 0)),
        "tasks_with_at_least_1_candidate": int(build_audit.get("tasks_with_at_least_1_candidate", 0)),
        "tasks_with_at_least_4_candidates": int(build_audit.get("tasks_with_at_least_4_candidates", 0)),
        "tasks_with_at_least_8_candidates": tasks_with_k,
        "accepted_candidates_per_task_distribution": build_audit.get("accepted_candidates_per_task_distribution", {}),
        "benchmark_dataset_audit": benchmark_audit,
        "source_generation_audit": source_audit,
        "source_coverage_report": source_coverage_report,
        "generation_build_audit": build_audit,
        "failure_taxonomy": {
            "terminal_statuses": sorted(TERMINAL_STATUSES),
            "task_skipped_statuses": sorted(TASK_SKIPPED_STATUSES),
            "status_counts": build_audit.get("terminal_status_counts", {}),
            "task_skipped_cause_counts": build_audit.get("task_skipped_cause_counts", {}),
        },
        "rejection_reason_counts": build_audit.get("rejection_reason_counts", {}),
        "invalid_diff_audit": build_audit.get("invalid_diff_audit", {}),
        "duplicate_patch_hash_audit": build_audit.get("duplicate_patch_hash_audit", {}),
        "empty_patch_audit": build_audit.get("empty_patch_audit", {}),
        "exact_reference_patch_hash_audit": build_audit.get("exact_reference_patch_hash_audit", {}),
        "near_reference_similarity_audit": build_audit.get("near_reference_similarity_audit", {}),
        "patch_apply_check_audit": build_audit.get("patch_apply_check_audit", {}),
        "generator_source_distribution": build_audit.get("generator_source_distribution", {}),
        "candidate_diversity_by_normalized_patch_edit_distance": build_audit.get("candidate_diversity_by_normalized_patch_edit_distance", {}),
        "candidate_schema_audit": {"required_keys": sorted(required_keys), "rows": schema_rows, "passes": not any(row["missing"] for row in schema_rows)},
        "candidate_file_save_audit": {
            "jsonl_output_exists": output_path.exists(),
            "attempts_output_exists": attempts_path.exists(),
            "jsonl_output_rows": len(_read_jsonl(output_path)) if output_path.exists() else 0,
            "attempts_output_rows": len(_read_jsonl(attempts_path)) if attempts_path.exists() else 0,
            "trajectory_root": str(trajectory_root),
            "candidate_patch_root": str(candidate_patch_root),
            "missing_trajectories": missing_trajectories,
            "missing_candidate_patch_files": missing_patch_files,
            "passes": files_saved_pass,
        },
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


def render_stage6b1b_report(audit: Dict[str, object], output_path: Path, attempts_path: Path, audit_path: Path, report_path: Path) -> str:
    gates = audit.get("quality_gates", {})
    invalid = audit.get("invalid_diff_audit", {})
    duplicate = audit.get("duplicate_patch_hash_audit", {})
    diversity = audit.get("candidate_diversity_by_normalized_patch_edit_distance", {})
    source_distribution = audit.get("generator_source_distribution", {})
    source_coverage = audit.get("source_coverage_report", {})
    lines = [
        "# Stage 6B.1b Candidate Supply Expansion",
        "",
        "## Scope",
        "",
        "- Generates or ingests generated SWE-bench candidate patches only.",
        "- Does not train the latent selector.",
        "- Does not run the official SWE-bench harness.",
        "- Does not use reference/gold patches as selector candidates.",
        "- Does not use raw-preserving expansion.",
        "- Does not fabricate missing candidates.",
        "- Extracts multiple trajectory diff blocks only when their normalized patch hashes are distinct.",
        "- Stores rejected attempts only in the attempt audit.",
        "- Stores raw trajectories separately from selector-visible patch text.",
        "- No final model claim is made.",
        "",
        "## Artifacts",
        "",
        f"- Generated candidates JSONL: `{output_path}`.",
        f"- Generation attempts JSONL: `{attempts_path}`.",
        f"- Generation audit: `{audit_path}`.",
        f"- Report: `{report_path}`.",
        "",
        "## Coverage",
        "",
        f"- Requested tasks: `{audit.get('requested_tasks')}`.",
        f"- Tasks attempted: `{audit.get('tasks_attempted')}`.",
        f"- Accepted candidate rows: `{audit.get('accepted_candidate_rows')}`.",
        f"- Tasks with >=1 candidate: `{audit.get('tasks_with_at_least_1_candidate')}`.",
        f"- Tasks with >=4 candidates: `{audit.get('tasks_with_at_least_4_candidates')}`.",
        f"- Tasks with >=8 candidates: `{audit.get('tasks_with_at_least_8_candidates')}`.",
        f"- Accepted candidates per task distribution: `{json.dumps(audit.get('accepted_candidates_per_task_distribution', {}), sort_keys=True)}`.",
        "",
        "## Source Coverage",
        "",
        f"- Source files loaded: `{source_coverage.get('source_files_loaded')}`.",
        f"- Total raw records loaded: `{source_coverage.get('total_raw_records_loaded')}`.",
        f"- Unique instance_ids in sources: `{source_coverage.get('unique_instance_ids_in_sources')}`.",
        f"- Overlap with selected benchmark tasks: `{source_coverage.get('overlap_with_selected_benchmark_tasks', {}).get('count')}/{source_coverage.get('overlap_with_selected_benchmark_tasks', {}).get('selected_task_count')}`.",
        f"- Top missing instance_ids: `{json.dumps(source_coverage.get('top_missing_instance_ids', [])[:10], sort_keys=True)}`.",
        f"- Top source files by accepted candidates: `{json.dumps(source_coverage.get('top_source_files_by_accepted_candidates', [])[:10], sort_keys=True)}`.",
        "",
        "## Rejections",
        "",
        f"- Terminal status counts: `{json.dumps(audit.get('failure_taxonomy', {}).get('status_counts', {}), sort_keys=True)}`.",
        f"- Task skipped cause counts: `{json.dumps(audit.get('failure_taxonomy', {}).get('task_skipped_cause_counts', {}), sort_keys=True)}`.",
        f"- Rejection reason counts: `{json.dumps(audit.get('rejection_reason_counts', {}), sort_keys=True)}`.",
        f"- Invalid diff rate: `{float(invalid.get('invalid_diff_rate', 0.0)):.4f}`.",
        f"- Invalid diff reasons: `{json.dumps(invalid.get('reason_counts', {}), sort_keys=True)}`.",
        f"- Duplicate rate: `{float(duplicate.get('duplicate_rate', 0.0)):.4f}`.",
        f"- Exact reference patch hits: `{audit.get('exact_reference_patch_hash_audit', {}).get('hits_total', 0)}`.",
        f"- Near-reference similarity flags: `{audit.get('near_reference_similarity_audit', {}).get('flags_total', 0)}`.",
        "",
        "## Diversity",
        "",
        f"- Generator contribution counts: `{json.dumps(source_distribution.get('accepted_by_generator_name', {}), sort_keys=True)}`.",
        f"- Config contribution counts: `{json.dumps(source_distribution.get('accepted_by_config_name', {}), sort_keys=True)}`.",
        f"- Mean pairwise normalized patch edit distance: `{float(diversity.get('mean_pairwise_distance', 0.0)):.4f}`.",
        "",
        "## Quality Gates",
        "",
        "| gate | pass |",
        "|---|---:|",
    ]
    for key, value in gates.items():
        lines.append(f"| `{key}` | `{bool(value)}` |")
    if not bool(gates.get("candidate_supply_expansion_quality_gates_pass", False)):
        lines.extend(
            [
                "",
                "## Current Limitation",
                "",
                "Stage 6B.1b quality gates did not pass. Add more generated prediction artifacts or configure a generation backend, then rerun this script. The runner intentionally leaves underfilled tasks underfilled rather than inventing candidates.",
            ]
        )
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            f"When the gates pass, Stage 6B.0 can consume `--source-jsonl {output_path}` in a later run; the official SWE-bench Docker harness was intentionally not run here.",
            "",
            "No final model claim is made.",
            "",
        ]
    )
    return "\n".join(lines)


def write_variant_prompt(prompt_root: Path, task: BenchmarkTask, config_name: str, seed: int, prompt_variant: str) -> Path:
    prompt_root.mkdir(parents=True, exist_ok=True)
    record = task.record
    variant_instruction = {
        "minimal_fix": "Make the smallest source change that plausibly fixes the issue.",
        "test_failure_fix": "Focus on the failing behavior described by FAIL_TO_PASS and the problem statement.",
        "localized_patch": "Prefer a localized patch in the files most directly implicated by the issue.",
        "conservative_patch": "Avoid broad rewrites and preserve public behavior outside the described bug.",
    }.get(str(prompt_variant), "Generate one plausible source patch for this task.")
    prompt = (
        "You are generating one candidate patch for a SWE-bench task.\n"
        "Return only a unified git diff. Do not include test results, prose, or reference patches.\n"
        f"Variant: {prompt_variant}\n"
        f"Variant instruction: {variant_instruction}\n\n"
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


def _build_selection_audit(
    accepted: Sequence[CandidateOutput],
    attempt_rows: Sequence[Dict[str, object]],
    task_records: Dict[str, BenchmarkTask],
    k: int,
    max_attempts_per_task: int,
    duplicate_rate_threshold: float,
    near_reference_threshold: float,
) -> Dict[str, object]:
    counts = Counter(row.instance_id for row in accepted)
    terminal_counts = Counter(str(row.get("terminal_status", "unknown_error")) for row in attempt_rows)
    rejection_counts = Counter(
        str(row.get("terminal_status"))
        for row in attempt_rows
        if row.get("terminal_status") != "accepted" and str(row.get("terminal_status")) not in TASK_SKIPPED_STATUSES
    )
    invalid_rows = [row for row in attempt_rows if row.get("terminal_status") == "invalid_diff"]
    duplicate_rows = [row for row in attempt_rows if row.get("terminal_status") == "duplicate_patch"]
    exact_reference_rows = [row for row in attempt_rows if row.get("terminal_status") == "exact_reference_hash"]
    near_reference_rows = [row for row in attempt_rows if row.get("terminal_status") == "near_reference_similarity_flag"]
    empty_rows = [row for row in attempt_rows if row.get("terminal_status") == "empty_response"]
    evaluated_attempts = [row for row in attempt_rows if str(row.get("terminal_status")) not in TASK_SKIPPED_STATUSES]
    patch_bearing_attempts = [
        row
        for row in attempt_rows
        if row.get("patch_hash")
        and row.get("terminal_status")
        in {"accepted", "invalid_diff", "duplicate_patch", "exact_reference_hash", "near_reference_similarity_flag", "patch_apply_check_failed"}
    ]
    duplicate_rate = float(len(duplicate_rows) / len(patch_bearing_attempts)) if patch_bearing_attempts else 0.0
    invalid_rate = float(len(invalid_rows) / len(evaluated_attempts)) if evaluated_attempts else 0.0
    task_ids_at_least_1 = [instance_id for instance_id, count in counts.items() if count >= 1]
    task_ids_at_least_4 = [instance_id for instance_id, count in counts.items() if count >= 4]
    task_ids_at_least_8 = [instance_id for instance_id, count in counts.items() if count >= int(k)]
    apply_status_counts = Counter(str((row.get("patch_apply_check") or {}).get("status", "not_recorded")) for row in attempt_rows)
    accepted_attempt_rows = [row for row in attempt_rows if row.get("terminal_status") == "accepted"]
    per_task_counts = {instance_id: int(counts.get(instance_id, 0)) for instance_id in task_records}
    return {
        "tasks_attempted": len(task_records),
        "max_attempts_per_task": int(max_attempts_per_task),
        "accepted_candidate_rows": len(accepted),
        "accepted_candidates_per_task": per_task_counts,
        "accepted_candidates_per_task_distribution": {str(key): int(value) for key, value in sorted(Counter(per_task_counts.values()).items())},
        "tasks_with_at_least_1_candidate": len(task_ids_at_least_1),
        "tasks_with_at_least_1_candidate_ids": task_ids_at_least_1,
        "tasks_with_at_least_4_candidates": len(task_ids_at_least_4),
        "tasks_with_at_least_4_candidate_ids": task_ids_at_least_4,
        "tasks_with_at_least_8_candidates": len(task_ids_at_least_8),
        "tasks_with_at_least_8_candidate_ids": task_ids_at_least_8,
        "terminal_status_counts": dict(sorted(terminal_counts.items())),
        "task_skipped_cause_counts": dict(sorted(Counter(str(row.get("terminal_status")) for row in attempt_rows if str(row.get("terminal_status")) in TASK_SKIPPED_STATUSES).items())),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "invalid_diff_audit": {
            "rows": invalid_rows[:200],
            "total": len(invalid_rows),
            "reason_counts": dict(sorted(Counter(str(row.get("invalid_diff_reason", "unknown")) for row in invalid_rows).items())),
            "invalid_diff_rate": invalid_rate,
            "passes": invalid_rate < 0.50,
        },
        "empty_patch_audit": {"rows": empty_rows[:200], "total": len(empty_rows), "passes": True},
        "duplicate_patch_hash_audit": {
            "duplicates_excluded": duplicate_rows[:200],
            "duplicates_excluded_total": len(duplicate_rows),
            "patch_bearing_attempts": len(patch_bearing_attempts),
            "duplicate_rate": duplicate_rate,
            "threshold": float(duplicate_rate_threshold),
            "passes": duplicate_rate <= float(duplicate_rate_threshold),
        },
        "exact_reference_patch_hash_audit": {
            "hits": exact_reference_rows[:200],
            "hits_total": len(exact_reference_rows),
            "passes": len(exact_reference_rows) == 0,
        },
        "near_reference_similarity_audit": {
            "threshold": float(near_reference_threshold),
            "flags": near_reference_rows[:200],
            "flags_total": len(near_reference_rows),
            "passes": True,
        },
        "patch_apply_check_audit": {
            "enabled_rows": sum(1 for row in attempt_rows if (row.get("patch_apply_check") or {}).get("status") not in {None, "not_requested"}),
            "status_counts": dict(sorted(apply_status_counts.items())),
            "required_failure_rows": [
                row
                for row in attempt_rows
                if row.get("terminal_status") in {"repo_setup_failed", "patch_apply_check_failed"}
            ][:200],
        },
        "generator_source_distribution": {
            "accepted_by_generator_name": dict(Counter(str(row.get("generator_name")) for row in accepted_attempt_rows)),
            "accepted_by_model_name_or_path": dict(Counter(str(row.get("model_name_or_path")) for row in accepted_attempt_rows)),
            "accepted_by_config_name": dict(Counter(str(row.get("config_name")) for row in accepted_attempt_rows)),
            "accepted_by_prompt_variant": dict(Counter(str(row.get("prompt_variant")) for row in accepted_attempt_rows)),
        },
        "candidate_diversity_by_normalized_patch_edit_distance": _candidate_diversity_audit(accepted),
        "raw_preserving_expansion_used": False,
        "missing_candidate_fabrication_used": False,
        "candidate_order_randomizable": True,
        "source_visible_to_selector_all_false": all(row.source_visible_to_selector is False for row in accepted),
        "generated_candidate_files_saved": True,
    }


def _base_attempt_row(attempt: RawGenerationAttempt, generation_attempt_index: int) -> Dict[str, object]:
    return {
        "instance_id": attempt.instance_id,
        "attempt_id": attempt.attempt_id,
        "generation_attempt_index": int(generation_attempt_index),
        "terminal_status": "unknown_error",
        "candidate_id": None,
        "model_name_or_path": attempt.model_name_or_path,
        "generator_name": attempt.generator_name,
        "seed": attempt.seed,
        "temperature": attempt.temperature,
        "config_name": attempt.config_name,
        "prompt_variant": attempt.prompt_variant,
        "trajectory_path": attempt.trajectory_path,
        "source_name": attempt.source_name,
        "source_row_index": attempt.source_row_index,
        "source_visible_to_selector": False,
        "patch_hash": "",
        "patch_length": 0,
        "candidate_patch_path": None,
        "invalid_diff_reason": None,
        "near_reference_similarity": None,
        "patch_apply_check": None,
        "failure_message": None,
    }


def _task_skipped_row(task: BenchmarkTask, max_attempts_per_task: int, source_audit: Dict[str, object]) -> Dict[str, object]:
    status = _task_skip_status(task.instance_id, source_audit)
    return {
        "instance_id": task.instance_id,
        "attempt_id": f"task-skipped-{_safe_name(task.instance_id)}",
        "generation_attempt_index": None,
        "terminal_status": status,
        "task_skipped_cause": status,
        "candidate_id": None,
        "model_name_or_path": None,
        "generator_name": None,
        "seed": None,
        "temperature": None,
        "config_name": None,
        "prompt_variant": None,
        "trajectory_path": None,
        "source_name": None,
        "source_row_index": None,
        "source_visible_to_selector": False,
        "patch_hash": "",
        "patch_length": 0,
        "candidate_patch_path": None,
        "invalid_diff_reason": None,
        "near_reference_similarity": None,
        "patch_apply_check": {"status": "not_requested", "required": False},
        "failure_message": f"no generation attempts available within max_attempts_per_task={max_attempts_per_task}; cause={status}",
    }


def _task_skip_status(instance_id: str, source_audit: Dict[str, object]) -> str:
    sources = [source for source in source_audit.get("sources", []) if isinstance(source, dict)]
    loaded_sources = [source for source in sources if bool(source.get("loaded", False)) and int(source.get("records", 0) or 0) > 0]
    if not loaded_sources:
        if _generation_backend_unavailable(source_audit):
            return "task_skipped_generation_backend_unavailable"
        return "task_skipped_no_source_artifact"
    instance_rows = _source_rows_for_instance(source_audit, instance_id)
    if not instance_rows:
        return "task_skipped_no_matching_instance_id"
    if any(int(row.get("rows_with_patch_field", 0) or 0) > 0 for row in instance_rows):
        return "task_skipped_unknown"
    if any(int(row.get("rows_with_messages", 0) or 0) > 0 or int(row.get("rows_with_trajectory", 0) or 0) > 0 for row in instance_rows):
        return "task_skipped_source_has_no_patch_field"
    return "task_skipped_source_has_no_trajectory"


def _source_rows_for_instance(source_audit: Dict[str, object], instance_id: str) -> List[Dict[str, object]]:
    rows = []
    for source in source_audit.get("sources", []):
        if not isinstance(source, dict):
            continue
        ids = {str(value) for value in source.get("unique_instance_ids", []) or []}
        if instance_id in ids:
            rows.append(source)
    return rows


def _generation_backend_unavailable(source_audit: Dict[str, object]) -> bool:
    for row in source_audit.get("errors", []):
        if isinstance(row, dict) and str(row.get("source_name", "")).lower() in {"mini-swe-agent", "api-generator-wrapper", "local-generator-wrapper"}:
            return True
    return False


def _coerce_terminal_status(value: str) -> str:
    status = str(value or "unknown_error")
    return status if status in TERMINAL_STATUSES else "unknown_error"


def _find_cached_repo(repo_cache_root: Path, task: BenchmarkTask) -> Path | None:
    candidates = [
        repo_cache_root / task.instance_id,
        repo_cache_root / _safe_name(task.instance_id),
    ]
    repo = str(task.record.get("repo", "") or "")
    if repo:
        candidates.append(repo_cache_root / _safe_name(repo))
        candidates.append(repo_cache_root / repo.replace("/", "__"))
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    return None


def _generation_seed_values(values: Sequence[str], default_seed: int) -> tuple[int, ...]:
    if values:
        parsed = tuple(_parse_int(value, default_seed) for value in values)
    else:
        parsed = (default_seed, default_seed + 1, default_seed + 2, default_seed + 3, default_seed + 4, default_seed + 5, default_seed + 6, default_seed + 7)
    return parsed


def _temperature_values(values: Sequence[str]) -> tuple[float, ...]:
    return tuple(_parse_float(value, 0.0) for value in (values or DEFAULT_TEMPERATURES))


def _normalize_text(text: str) -> str:
    value = _sanitize_unicode(str(text or "")).replace("\r\n", "\n")
    if "\\n" in value and value.count("\\n") > value.count("\n"):
        try:
            return _sanitize_unicode(bytes(value, "utf-8", errors="replace").decode("unicode_escape", errors="replace"))
        except Exception:
            return value.replace("\\n", "\n")
    return value


def _sanitize_unicode(text: str) -> str:
    return str(text or "").encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def _json_message_payload(text: str) -> object | None:
    stripped = _sanitize_unicode(str(text or "")).strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        parsed = json.loads(stripped)
    except Exception:
        return None
    return parsed if isinstance(parsed, (list, dict)) else None


def _redact_source_row(value: object, parent_key: str = "") -> object:
    if isinstance(value, dict):
        redacted: Dict[str, object] = {}
        for key, item in value.items():
            key_text = str(key)
            lowered = key_text.lower()
            parent = str(parent_key).lower()
            if lowered in {"gold_patch", "reference_patch", "test_patch", "reward", "resolved", "test_output"}:
                redacted[key_text] = "[REDACTED_STAGE6B1B_AUDIT_ONLY]"
            elif lowered == "patch" and parent in {"ds", "benchmark", "record", "instance"}:
                redacted[key_text] = "[REDACTED_STAGE6B1B_REFERENCE_PATCH]"
            else:
                redacted[key_text] = _redact_source_row(item, key_text)
        return redacted
    if isinstance(value, list):
        return [_redact_source_row(item, parent_key) for item in value]
    return value


def _text_contains_unified_diff(text: str) -> bool:
    value = str(text or "")
    return "diff --git " in value or _first_unified_header_index(value) >= 0


def _first_unified_header_index(text: str) -> int:
    value = str(text or "")
    pattern = re.compile(r"(?m)^---\s+\S+.*\n\+\+\+\s+\S+.*\n@@\s+-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s+@@")
    match = pattern.search(value)
    return -1 if match is None else match.start()


def _dedupe_patch_texts(values: Iterable[str]) -> List[str]:
    rows = []
    seen = set()
    for value in values:
        safe_value = _sanitize_unicode(str(value or ""))
        patch = _ensure_trailing_newline(_clean_patch(safe_value)) if safe_value.strip() else ""
        key = stable_patch_hash(patch) if patch.strip() else ""
        if patch.strip() and key not in seen:
            seen.add(key)
            rows.append(patch)
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(_json_safe(row), sort_keys=True) for row in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
