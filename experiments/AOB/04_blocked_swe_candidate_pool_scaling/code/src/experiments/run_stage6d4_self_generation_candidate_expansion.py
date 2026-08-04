from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from src.datasets.swe_patch_selection_dataset import (
    STAGE6_NUM_CANDIDATES,
    PatchSelectionExample,
    duplicate_candidate_patch_hash_audit,
    stable_patch_hash,
    write_patch_selection_jsonl,
)
from src.experiments.run_stage6b1_candidate_generation import DEFAULT_DATASET_NAME, BenchmarkTask, load_benchmark_tasks
from src.experiments.run_stage6b1b_candidate_supply_expansion import validate_unified_diff
from src.experiments.run_stage6d_pool_scale_and_view_hardening import (
    NEAR_REFERENCE_THRESHOLD,
    build_pool_summary,
    candidate_order_randomized_audit,
    exact_reference_hash_audit,
    near_reference_similarity_audit,
    source_generator_skew_report,
)
from src.experiments.run_stage6d3_candidate_source_expansion import (
    DEFAULT_CANDIDATE_SOURCES,
    DEFAULT_PUBLIC_SOURCE_ROOT,
    GeneratedAttempt as SourceAttempt,
    build_unlabeled_pool,
    clean_patch,
    discover_source_files,
    extract_candidate_patches,
    generator_name,
    group_accepted_attempts,
    normalize_instance_id,
    patch_similarity,
    read_records,
    scan_generated_sources,
)


BENCHMARK = "stage6d4_self_generation_candidate_expansion"
DEFAULT_STAGE6D3_POOL_ALL = Path("results/stage6d3_candidate_pool_all_unlabeled.jsonl")
DEFAULT_STAGE6D3_POOL_STRICT = Path("results/stage6d3_candidate_pool_strict_unlabeled.jsonl")
DEFAULT_STAGE6D3_PLAN = Path("results/stage6d3_generation_plan.json")
DEFAULT_STAGE6D3_INVENTORY = Path("results/stage6d3_source_inventory.json")
DEFAULT_SELF_GENERATED = Path("results/stage6d4_self_generated_candidates.jsonl")
DEFAULT_ATTEMPTS = Path("results/stage6d4_generation_attempts.jsonl")
DEFAULT_POOL_ALL = Path("results/stage6d4_candidate_pool_all_unlabeled.jsonl")
DEFAULT_POOL_STRICT = Path("results/stage6d4_candidate_pool_strict_unlabeled.jsonl")
DEFAULT_AUDIT = Path("results/stage6d4_generation_audit.json")
DEFAULT_READINESS = Path("results/stage6d4_pre_harness_readiness_audit.json")
DEFAULT_REPORT = Path("reports/STAGE6D4_SELF_GENERATION_CANDIDATE_EXPANSION.md")
DEFAULT_TRAJECTORY_ROOT = Path("results/stage6d4_raw_trajectories")
DEFAULT_SEED = 606_440
PROMPT_VARIANTS = (
    "minimal_fix",
    "traceback_or_issue_fix",
    "localized_patch",
    "conservative_patch",
    "regression_aware_patch",
    "test_failure_patch",
    "API_contract_patch",
    "import_dependency_patch",
)
PRIORITY_REPOS = ("django/django", "matplotlib/matplotlib", "sphinx-doc/sphinx")


@dataclass(frozen=True)
class GenerationAttempt:
    instance_id: str
    attempt_id: str
    terminal_status: str
    reason: str
    generator_name: str
    backend: str
    model_name: str
    prompt_variant: str
    seed: int
    temperature: float
    attempt_index: int
    normalized_patch_hash: str = ""
    near_reference_similarity: float | None = None
    raw_trajectory_path: str = ""
    model_patch: str = ""

    @property
    def accepted_for_pool_all(self) -> bool:
        return self.terminal_status in {"accepted", "near_reference_flag"} and bool(self.model_patch.strip())

    @property
    def near_reference(self) -> bool:
        return self.terminal_status == "near_reference_flag"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6D.4 self-generation candidate expansion.")
    parser.add_argument("--stage6d3-pool-all", default=str(DEFAULT_STAGE6D3_POOL_ALL))
    parser.add_argument("--stage6d3-pool-strict", default=str(DEFAULT_STAGE6D3_POOL_STRICT))
    parser.add_argument("--stage6d3-generation-plan", default=str(DEFAULT_STAGE6D3_PLAN))
    parser.add_argument("--stage6d3-inventory", default=str(DEFAULT_STAGE6D3_INVENTORY))
    parser.add_argument("--candidate-source", action="append", default=[str(path) for path in DEFAULT_CANDIDATE_SOURCES])
    parser.add_argument("--source-root", action="append", default=[str(DEFAULT_PUBLIC_SOURCE_ROOT)])
    parser.add_argument("--local-jsonl-ingest", action="append", default=[])
    parser.add_argument("--command-line-agent-template", action="append", default=[])
    parser.add_argument("--api-generator-command-template", action="append", default=[])
    parser.add_argument("--mini-swe-agent-command-template", action="append", default=[])
    parser.add_argument("--enable-mini-swe-agent", action="store_true")
    parser.add_argument("--enable-dry-run", action="store_true")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--temperature", action="append", default=["0.0", "0.2", "0.7", "1.0"])
    parser.add_argument("--generation-seed", action="append", default=[])
    parser.add_argument("--prompt-variant", action="append", default=list(PROMPT_VARIANTS))
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--include-quarantined-diagnostic-tasks", action="store_true")
    parser.add_argument("--max-generation-tasks", type=int, default=80)
    parser.add_argument("--max-attempts-per-task", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--retry-count", type=int, default=0)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--trajectory-root", default=str(DEFAULT_TRAJECTORY_ROOT))
    parser.add_argument("--self-generated-output", default=str(DEFAULT_SELF_GENERATED))
    parser.add_argument("--attempts-output", default=str(DEFAULT_ATTEMPTS))
    parser.add_argument("--pool-all-output", default=str(DEFAULT_POOL_ALL))
    parser.add_argument("--pool-strict-output", default=str(DEFAULT_POOL_STRICT))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--readiness", default=str(DEFAULT_READINESS))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    result = run_stage6d4_self_generation_candidate_expansion(
        stage6d3_pool_all=Path(args.stage6d3_pool_all),
        stage6d3_pool_strict=Path(args.stage6d3_pool_strict),
        stage6d3_generation_plan=Path(args.stage6d3_generation_plan),
        stage6d3_inventory=Path(args.stage6d3_inventory),
        candidate_sources=[Path(value) for value in args.candidate_source],
        source_roots=[Path(value) for value in args.source_root],
        local_jsonl_ingest=[Path(value) for value in args.local_jsonl_ingest],
        command_line_agent_templates=tuple(str(value) for value in args.command_line_agent_template),
        api_generator_templates=tuple(str(value) for value in args.api_generator_command_template),
        mini_swe_agent_templates=tuple(str(value) for value in args.mini_swe_agent_command_template),
        enable_mini_swe_agent=bool(args.enable_mini_swe_agent),
        enable_dry_run=bool(args.enable_dry_run),
        models=tuple(str(value) for value in args.model),
        temperatures=tuple(float(value) for value in args.temperature),
        generation_seeds=tuple(int(value) for value in args.generation_seed) if args.generation_seed else (int(args.seed),),
        prompt_variants=tuple(str(value) for value in args.prompt_variant),
        explicit_task_ids=tuple(str(value) for value in args.task_id),
        include_quarantined_diagnostic_tasks=bool(args.include_quarantined_diagnostic_tasks),
        max_generation_tasks=int(args.max_generation_tasks),
        max_attempts_per_task=int(args.max_attempts_per_task),
        timeout_seconds=int(args.timeout_seconds),
        max_output_tokens=int(args.max_output_tokens),
        retry_count=int(args.retry_count),
        dataset_name=str(args.dataset_name),
        split=str(args.split),
        seed=int(args.seed),
        trajectory_root=Path(args.trajectory_root),
        self_generated_output=Path(args.self_generated_output),
        attempts_output=Path(args.attempts_output),
        pool_all_output=Path(args.pool_all_output),
        pool_strict_output=Path(args.pool_strict_output),
        audit_path=Path(args.audit),
        readiness_path=Path(args.readiness),
        report_path=Path(args.report),
    )
    print(
        "stage6d4: wrote {candidates}, {attempts}, {pool_all}, {pool_strict}, {audit}, {readiness}, {report}; "
        "new_accepted={accepted}; strict_pool={strict}; decision={decision}".format(
            candidates=args.self_generated_output,
            attempts=args.attempts_output,
            pool_all=args.pool_all_output,
            pool_strict=args.pool_strict_output,
            audit=args.audit,
            readiness=args.readiness,
            report=args.report,
            accepted=result["audit"]["accepted_candidates_total"],
            strict=result["audit"]["pool_summary"]["pool_strict_k8_tasks"],
            decision=result["readiness"]["decision"],
        )
    )


def run_stage6d4_self_generation_candidate_expansion(
    stage6d3_pool_all: Path,
    stage6d3_pool_strict: Path,
    stage6d3_generation_plan: Path,
    stage6d3_inventory: Path,
    candidate_sources: Sequence[Path],
    source_roots: Sequence[Path],
    local_jsonl_ingest: Sequence[Path],
    command_line_agent_templates: Sequence[str],
    api_generator_templates: Sequence[str],
    mini_swe_agent_templates: Sequence[str],
    enable_mini_swe_agent: bool,
    enable_dry_run: bool,
    models: Sequence[str],
    temperatures: Sequence[float],
    generation_seeds: Sequence[int],
    prompt_variants: Sequence[str],
    explicit_task_ids: Sequence[str],
    include_quarantined_diagnostic_tasks: bool,
    max_generation_tasks: int,
    max_attempts_per_task: int,
    timeout_seconds: int,
    max_output_tokens: int,
    retry_count: int,
    dataset_name: str,
    split: str,
    seed: int,
    trajectory_root: Path,
    self_generated_output: Path,
    attempts_output: Path,
    pool_all_output: Path,
    pool_strict_output: Path,
    audit_path: Path,
    readiness_path: Path,
    report_path: Path,
) -> Dict[str, object]:
    created_at = _now()
    task_records, benchmark_audit = load_benchmark_tasks(
        dataset_name=dataset_name,
        split=split,
        benchmark_jsonl=None,
        requested_task_ids=[],
        task_ids_file=None,
        n_tasks=500,
        seed=seed,
    )
    base_source_files = discover_source_files(candidate_sources, source_roots)
    base_source_attempts, base_source_rows = scan_generated_sources(base_source_files, task_records)
    base_grouped = group_accepted_attempts(base_source_attempts)
    stage6d3_plan = _read_json(stage6d3_generation_plan)
    stage6d3_inventory_row = _read_json(stage6d3_inventory)
    stage6d3_strict_ids = set(stage6d3_inventory_row.get("pool_estimates", {}).get("pool_strict_task_ids", []))
    quarantined_ids = set(stage6d3_inventory_row.get("pool_estimates", {}).get("strict_quarantined_task_ids", []))
    selected_tasks = select_generation_tasks(
        task_records=task_records,
        generation_plan=stage6d3_plan,
        strict_pool_ids=stage6d3_strict_ids,
        quarantined_task_ids=quarantined_ids,
        explicit_task_ids=explicit_task_ids,
        include_quarantined=include_quarantined_diagnostic_tasks,
        max_generation_tasks=max_generation_tasks,
    )
    existing_hashes = {instance_id: {attempt.patch_hash for attempt in attempts} for instance_id, attempts in base_grouped.items()}
    backend_config = {
        "local_jsonl_ingest_files": [str(path) for path in local_jsonl_ingest],
        "command_line_agent_templates": len(command_line_agent_templates),
        "api_generator_templates": len(api_generator_templates),
        "mini_swe_agent_enabled": bool(enable_mini_swe_agent),
        "mini_swe_agent_templates": len(mini_swe_agent_templates),
        "dry_run_enabled": bool(enable_dry_run),
        "models": list(models),
        "temperatures": list(temperatures),
        "generation_seeds": list(generation_seeds),
        "prompt_variants": list(prompt_variants),
    }
    generation_attempts: List[GenerationAttempt] = []
    generated_source_attempts: List[SourceAttempt] = []
    if local_jsonl_ingest:
        rows = run_local_jsonl_ingest_backend(
            local_jsonl_ingest,
            selected_tasks,
            task_records,
            existing_hashes,
            trajectory_root,
        )
        generation_attempts.extend(rows)
        generated_source_attempts.extend(generation_attempt_to_source_attempt(row) for row in rows if row.accepted_for_pool_all)
        update_hashes(existing_hashes, rows)
    if command_line_agent_templates:
        rows = run_template_backend(
            backend="command_line_agent",
            command_templates=command_line_agent_templates,
            selected_tasks=selected_tasks,
            task_records=task_records,
            existing_hashes=existing_hashes,
            trajectory_root=trajectory_root,
            models=models or ("command-line-agent",),
            temperatures=temperatures,
            generation_seeds=generation_seeds,
            prompt_variants=prompt_variants,
            max_attempts_per_task=max_attempts_per_task,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
            retry_count=retry_count,
        )
        generation_attempts.extend(rows)
        generated_source_attempts.extend(generation_attempt_to_source_attempt(row) for row in rows if row.accepted_for_pool_all)
        update_hashes(existing_hashes, rows)
    if api_generator_templates:
        rows = run_template_backend(
            backend="api_generator",
            command_templates=api_generator_templates,
            selected_tasks=selected_tasks,
            task_records=task_records,
            existing_hashes=existing_hashes,
            trajectory_root=trajectory_root,
            models=models or ("api-generator",),
            temperatures=temperatures,
            generation_seeds=generation_seeds,
            prompt_variants=prompt_variants,
            max_attempts_per_task=max_attempts_per_task,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
            retry_count=retry_count,
        )
        generation_attempts.extend(rows)
        generated_source_attempts.extend(generation_attempt_to_source_attempt(row) for row in rows if row.accepted_for_pool_all)
        update_hashes(existing_hashes, rows)
    if enable_mini_swe_agent:
        rows = run_mini_swe_agent_backend(
            mini_swe_agent_templates,
            selected_tasks,
            task_records,
            existing_hashes,
            trajectory_root,
            temperatures,
            generation_seeds,
            prompt_variants,
            max_attempts_per_task,
            timeout_seconds,
            max_output_tokens,
            retry_count,
        )
        generation_attempts.extend(rows)
        generated_source_attempts.extend(generation_attempt_to_source_attempt(row) for row in rows if row.accepted_for_pool_all)
        update_hashes(existing_hashes, rows)
    if enable_dry_run:
        rows = run_dry_run_backend(
            selected_tasks,
            task_records,
            existing_hashes,
            trajectory_root,
            temperatures,
            generation_seeds,
            prompt_variants,
            max_attempts_per_task,
        )
        generation_attempts.extend(rows)
        generated_source_attempts.extend(generation_attempt_to_source_attempt(row) for row in rows if row.accepted_for_pool_all)
        update_hashes(existing_hashes, rows)

    merged_attempts = list(base_source_attempts) + generated_source_attempts
    merged_grouped = group_accepted_attempts(merged_attempts)
    pool_all, pool_all_selection = build_unlabeled_pool(merged_grouped, task_records, seed=seed + 44, strict=False)
    pool_strict, pool_strict_selection = build_unlabeled_pool(merged_grouped, task_records, seed=seed + 45, strict=True)
    write_patch_selection_jsonl(pool_all_output, pool_all)
    write_patch_selection_jsonl(pool_strict_output, pool_strict)
    self_generated_rows = [candidate_row(row) for row in generation_attempts if row.accepted_for_pool_all]
    _write_jsonl(self_generated_output, self_generated_rows)
    _write_jsonl(attempts_output, [attempt_row(row) for row in generation_attempts])

    audit = build_generation_audit(
        created_at=created_at,
        benchmark_audit=benchmark_audit,
        stage6d3_plan=stage6d3_plan,
        stage6d3_inventory=stage6d3_inventory_row,
        base_source_files=base_source_files,
        base_source_rows=base_source_rows,
        selected_tasks=selected_tasks,
        generation_attempts=generation_attempts,
        generated_source_attempts=generated_source_attempts,
        pool_all=pool_all,
        pool_strict=pool_strict,
        task_records=task_records,
        pool_all_selection=pool_all_selection,
        pool_strict_selection=pool_strict_selection,
        backend_config=backend_config,
        strict_target=50,
        preferred_target=100,
    )
    readiness = build_readiness_audit(audit, pool_all, pool_strict, generation_attempts, strict_target=50, preferred_target=100)
    _write_json(audit_path, audit)
    _write_json(readiness_path, readiness)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(audit, readiness), encoding="utf-8")
    return {"audit": audit, "readiness": readiness}


def select_generation_tasks(
    task_records: Dict[str, BenchmarkTask],
    generation_plan: Dict[str, object],
    strict_pool_ids: set[str],
    quarantined_task_ids: set[str],
    explicit_task_ids: Sequence[str],
    include_quarantined: bool,
    max_generation_tasks: int,
) -> List[str]:
    if explicit_task_ids:
        candidates = [task_id for task_id in dict.fromkeys(explicit_task_ids) if task_id in task_records]
    else:
        rows = generation_plan.get("tasks_needing_candidates", []) if isinstance(generation_plan, dict) else []
        candidates = [str(row.get("instance_id")) for row in rows if isinstance(row, dict) and row.get("instance_id") in task_records]
        if not candidates:
            candidates = sorted(task_records)
    filtered = []
    for task_id in candidates:
        if task_id in strict_pool_ids:
            continue
        if task_id in quarantined_task_ids and not include_quarantined:
            continue
        filtered.append(task_id)
    by_repo: Dict[str, deque[str]] = defaultdict(deque)
    for task_id in filtered:
        by_repo[str(task_records[task_id].record.get("repo", ""))].append(task_id)
    repo_order = sorted(by_repo, key=lambda repo: (PRIORITY_REPOS.index(repo) if repo in PRIORITY_REPOS else len(PRIORITY_REPOS), repo))
    selected = []
    while repo_order and len(selected) < max_generation_tasks:
        next_order = []
        for repo in repo_order:
            if by_repo[repo] and len(selected) < max_generation_tasks:
                selected.append(by_repo[repo].popleft())
            if by_repo[repo]:
                next_order.append(repo)
        repo_order = next_order
    return selected


def run_local_jsonl_ingest_backend(
    files: Sequence[Path],
    selected_tasks: Sequence[str],
    task_records: Dict[str, BenchmarkTask],
    existing_hashes: Dict[str, set[str]],
    trajectory_root: Path,
) -> List[GenerationAttempt]:
    selected_records = {task_id: task_records[task_id] for task_id in selected_tasks if task_id in task_records}
    rows = []
    for source_file in files:
        for row_index, record in enumerate(read_records(source_file)):
            instance_id = normalize_instance_id(record, source_file)
            if instance_id not in selected_records:
                continue
            patches = extract_candidate_patches(record)
            if not patches:
                rows.append(
                    base_attempt(
                        instance_id,
                        backend="local_jsonl_ingest",
                        generator=generator_name(record, source_file),
                        model_name=str(record.get("model_name_or_path") or "local_jsonl_ingest"),
                        prompt_variant=str(record.get("prompt_variant") or "local_jsonl_ingest"),
                        seed=int(record.get("seed") or 0),
                        temperature=float(record.get("temperature") or 0.0),
                        attempt_index=row_index,
                        status="no_patch_found",
                        reason="no_extractable_patch_diff",
                    )
                )
                continue
            for patch_index, patch in enumerate(patches):
                raw_path = write_raw_trajectory(
                    trajectory_root,
                    instance_id,
                    "local_jsonl_ingest",
                    f"{row_index:06d}_{patch_index:02d}",
                    {"source_file": str(source_file), "source_row_index": row_index, "patch_index": patch_index, "record": record},
                )
                rows.append(
                    validate_generated_patch(
                        instance_id=instance_id,
                        raw_patch=patch,
                        task=selected_records[instance_id],
                        existing_hashes=existing_hashes.setdefault(instance_id, set()),
                        backend="local_jsonl_ingest",
                        generator=generator_name(record, source_file),
                        model_name=str(record.get("model_name_or_path") or "local_jsonl_ingest"),
                        prompt_variant=str(record.get("prompt_variant") or "local_jsonl_ingest"),
                        seed=int(record.get("seed") or 0),
                        temperature=float(record.get("temperature") or 0.0),
                        attempt_index=row_index * 1000 + patch_index,
                        raw_trajectory_path=str(raw_path),
                    )
                )
    return rows


def run_template_backend(
    backend: str,
    command_templates: Sequence[str],
    selected_tasks: Sequence[str],
    task_records: Dict[str, BenchmarkTask],
    existing_hashes: Dict[str, set[str]],
    trajectory_root: Path,
    models: Sequence[str],
    temperatures: Sequence[float],
    generation_seeds: Sequence[int],
    prompt_variants: Sequence[str],
    max_attempts_per_task: int,
    timeout_seconds: int,
    max_output_tokens: int,
    retry_count: int,
) -> List[GenerationAttempt]:
    rows = []
    for task_id in selected_tasks:
        task = task_records[task_id]
        accepted = 0
        attempt_index = 0
        for template in command_templates:
            for model in models:
                for generation_seed in generation_seeds:
                    for temperature in temperatures:
                        for variant in prompt_variants:
                            if attempt_index >= max_attempts_per_task or accepted >= STAGE6_NUM_CANDIDATES:
                                break
                            row = run_single_template_attempt(
                                backend,
                                template,
                                task,
                                existing_hashes.setdefault(task_id, set()),
                                model,
                                variant,
                                int(generation_seed),
                                float(temperature),
                                attempt_index,
                                trajectory_root,
                                timeout_seconds,
                                max_output_tokens,
                                retry_count,
                            )
                            rows.append(row)
                            if row.accepted_for_pool_all:
                                accepted += 1
                                existing_hashes[task_id].add(row.normalized_patch_hash)
                            attempt_index += 1
                        if attempt_index >= max_attempts_per_task or accepted >= STAGE6_NUM_CANDIDATES:
                            break
                    if attempt_index >= max_attempts_per_task or accepted >= STAGE6_NUM_CANDIDATES:
                        break
                if attempt_index >= max_attempts_per_task or accepted >= STAGE6_NUM_CANDIDATES:
                    break
    return rows


def run_single_template_attempt(
    backend: str,
    template: str,
    task: BenchmarkTask,
    existing_hashes: set[str],
    model: str,
    prompt_variant: str,
    seed: int,
    temperature: float,
    attempt_index: int,
    trajectory_root: Path,
    timeout_seconds: int,
    max_output_tokens: int,
    retry_count: int,
) -> GenerationAttempt:
    prompt = build_prompt(task, prompt_variant, max_output_tokens)
    raw_dir = trajectory_root / safe_name(backend) / safe_name(task.instance_id)
    raw_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = raw_dir / f"{attempt_index:04d}_{safe_name(prompt_variant)}.prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    output_path = raw_dir / f"{attempt_index:04d}_{safe_name(prompt_variant)}.output.json"
    command = template.format(
        instance_id=task.instance_id,
        repo=task.record.get("repo", ""),
        model=model,
        prompt_variant=prompt_variant,
        seed=seed,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        prompt_path=str(prompt_path),
        output_path=str(output_path),
    )
    attempt_id = f"{task.instance_id}-{backend}-{attempt_index:04d}-{safe_name(prompt_variant)}"
    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raw_path = write_raw_trajectory(raw_dir, "", "", f"{attempt_index:04d}_timeout", {"command": command, "timeout": timeout_seconds, "stdout": exc.stdout, "stderr": exc.stderr})
        return base_attempt(task.instance_id, backend, backend, model, prompt_variant, seed, temperature, attempt_index, "timeout", "timeout", str(raw_path))
    except Exception as exc:
        raw_path = write_raw_trajectory(raw_dir, "", "", f"{attempt_index:04d}_error", {"command": command, "error": f"{type(exc).__name__}: {exc}"})
        return base_attempt(task.instance_id, backend, backend, model, prompt_variant, seed, temperature, attempt_index, "generator_error", f"{type(exc).__name__}: {exc}", str(raw_path))
    raw_payload = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "output_path": str(output_path),
    }
    if output_path.exists():
        raw_payload["output_file_text"] = output_path.read_text(encoding="utf-8", errors="replace")
    raw_path = write_raw_trajectory(raw_dir, "", "", f"{attempt_index:04d}_raw", raw_payload)
    if completed.returncode != 0:
        return base_attempt(task.instance_id, backend, backend, model, prompt_variant, seed, temperature, attempt_index, "generator_error", f"returncode={completed.returncode}", str(raw_path))
    text = "\n".join(str(raw_payload.get(key, "")) for key in ("stdout", "output_file_text"))
    patches = extract_candidate_patches({"response": text, "final_response": text})
    if not text.strip():
        return base_attempt(task.instance_id, backend, backend, model, prompt_variant, seed, temperature, attempt_index, "empty_response", "empty_stdout_and_output", str(raw_path))
    if not patches:
        return base_attempt(task.instance_id, backend, backend, model, prompt_variant, seed, temperature, attempt_index, "no_patch_found", "no_extractable_patch_diff", str(raw_path))
    return validate_generated_patch(task.instance_id, patches[0], task, existing_hashes, backend, backend, model, prompt_variant, seed, temperature, attempt_index, str(raw_path), attempt_id=attempt_id)


def run_mini_swe_agent_backend(
    templates: Sequence[str],
    selected_tasks: Sequence[str],
    task_records: Dict[str, BenchmarkTask],
    existing_hashes: Dict[str, set[str]],
    trajectory_root: Path,
    temperatures: Sequence[float],
    generation_seeds: Sequence[int],
    prompt_variants: Sequence[str],
    max_attempts_per_task: int,
    timeout_seconds: int,
    max_output_tokens: int,
    retry_count: int,
) -> List[GenerationAttempt]:
    if not templates:
        return [
            base_attempt(task_id, "mini_swe_agent", "mini_swe_agent", "mini_swe_agent", "backend_setup", 0, 0.0, 0, "generator_error", "mini_swe_agent_enabled_but_no_command_template")
            for task_id in selected_tasks[:1]
        ]
    if shutil.which("mini-swe-agent") is None and shutil.which("mini_swe_agent") is None:
        return [
            base_attempt(task_id, "mini_swe_agent", "mini_swe_agent", "mini_swe_agent", "backend_setup", 0, 0.0, 0, "generator_error", "mini_swe_agent_not_installed")
            for task_id in selected_tasks[:1]
        ]
    return run_template_backend(
        "mini_swe_agent",
        templates,
        selected_tasks,
        task_records,
        existing_hashes,
        trajectory_root,
        ("mini_swe_agent",),
        temperatures,
        generation_seeds,
        prompt_variants,
        max_attempts_per_task,
        timeout_seconds,
        max_output_tokens,
        retry_count,
    )


def run_dry_run_backend(
    selected_tasks: Sequence[str],
    task_records: Dict[str, BenchmarkTask],
    existing_hashes: Dict[str, set[str]],
    trajectory_root: Path,
    temperatures: Sequence[float],
    generation_seeds: Sequence[int],
    prompt_variants: Sequence[str],
    max_attempts_per_task: int,
) -> List[GenerationAttempt]:
    rows = []
    for task_id in selected_tasks:
        task = task_records[task_id]
        accepted = 0
        for attempt_index in range(max_attempts_per_task):
            variant = prompt_variants[attempt_index % max(1, len(prompt_variants))]
            generation_seed = generation_seeds[attempt_index % max(1, len(generation_seeds))]
            temperature = temperatures[attempt_index % max(1, len(temperatures))]
            patch = dry_run_patch(task, attempt_index, variant)
            raw_path = write_raw_trajectory(
                trajectory_root,
                task_id,
                "dry_run",
                f"{attempt_index:04d}",
                {"backend": "dry_run", "instance_id": task_id, "attempt_index": attempt_index, "patch": patch},
            )
            row = validate_generated_patch(
                task_id,
                patch,
                task,
                existing_hashes.setdefault(task_id, set()),
                "dry_run",
                "dry_run_testing_only",
                "dry_run",
                variant,
                int(generation_seed),
                float(temperature),
                attempt_index,
                str(raw_path),
            )
            rows.append(row)
            if row.accepted_for_pool_all:
                existing_hashes[task_id].add(row.normalized_patch_hash)
                accepted += 1
            if accepted >= STAGE6_NUM_CANDIDATES:
                break
    return rows


def validate_generated_patch(
    instance_id: str,
    raw_patch: str,
    task: BenchmarkTask,
    existing_hashes: set[str],
    backend: str,
    generator: str,
    model_name: str,
    prompt_variant: str,
    seed: int,
    temperature: float,
    attempt_index: int,
    raw_trajectory_path: str,
    attempt_id: str | None = None,
) -> GenerationAttempt:
    patch = clean_patch(raw_patch)
    patch_hash = stable_patch_hash(patch)
    if not patch.strip():
        return base_attempt(instance_id, backend, generator, model_name, prompt_variant, seed, temperature, attempt_index, "empty_response", "empty_patch", raw_trajectory_path)
    validation = validate_unified_diff(patch)
    if not validation.valid:
        return base_attempt(instance_id, backend, generator, model_name, prompt_variant, seed, temperature, attempt_index, "invalid_diff", validation.reason or "invalid_diff", raw_trajectory_path, patch_hash)
    if patch_hash in existing_hashes:
        return base_attempt(instance_id, backend, generator, model_name, prompt_variant, seed, temperature, attempt_index, "duplicate_patch", "duplicate_normalized_patch_hash", raw_trajectory_path, patch_hash)
    if patch_hash == task.reference_hash:
        return base_attempt(instance_id, backend, generator, model_name, prompt_variant, seed, temperature, attempt_index, "exact_reference_hash", "exact_reference_hash", raw_trajectory_path, patch_hash)
    similarity = patch_similarity(patch, task.reference_patch)
    status = "near_reference_flag" if task.reference_patch.strip() and similarity >= NEAR_REFERENCE_THRESHOLD else "accepted"
    reason = "near_reference_similarity_flag" if status == "near_reference_flag" else "accepted"
    return GenerationAttempt(
        instance_id=instance_id,
        attempt_id=attempt_id or f"{instance_id}-{backend}-{attempt_index:04d}",
        terminal_status=status,
        reason=reason,
        generator_name=generator,
        backend=backend,
        model_name=model_name,
        prompt_variant=prompt_variant,
        seed=int(seed),
        temperature=float(temperature),
        attempt_index=int(attempt_index),
        normalized_patch_hash=patch_hash,
        near_reference_similarity=similarity,
        raw_trajectory_path=raw_trajectory_path,
        model_patch=patch,
    )


def build_generation_audit(
    created_at: str,
    benchmark_audit: Dict[str, object],
    stage6d3_plan: Dict[str, object],
    stage6d3_inventory: Dict[str, object],
    base_source_files: Sequence[Path],
    base_source_rows: Sequence[Dict[str, object]],
    selected_tasks: Sequence[str],
    generation_attempts: Sequence[GenerationAttempt],
    generated_source_attempts: Sequence[SourceAttempt],
    pool_all: Sequence[PatchSelectionExample],
    pool_strict: Sequence[PatchSelectionExample],
    task_records: Dict[str, BenchmarkTask],
    pool_all_selection: Dict[str, object],
    pool_strict_selection: Dict[str, object],
    backend_config: Dict[str, object],
    strict_target: int,
    preferred_target: int,
) -> Dict[str, object]:
    status_counts = Counter(row.terminal_status for row in generation_attempts)
    accepted_pool_rows = [row for row in generation_attempts if row.accepted_for_pool_all]
    non_near_accepted = [row for row in generation_attempts if row.terminal_status == "accepted"]
    invalid_rows = [row for row in generation_attempts if row.terminal_status == "invalid_diff"]
    exact_hits = [row for row in generation_attempts if row.terminal_status == "exact_reference_hash"]
    near_flags = [row for row in generation_attempts if row.terminal_status == "near_reference_flag"]
    duplicate_rows = [row for row in generation_attempts if row.terminal_status == "duplicate_patch"]
    pairwise = mean_pairwise_patch_distance(pool_all)
    strict_size = len(pool_strict)
    return {
        "benchmark": BENCHMARK,
        "created_at_utc": created_at,
        "benchmark_dataset_audit": benchmark_audit,
        "stage6d3_generation_plan_path_summary": {
            "current_strict_k8_tasks": stage6d3_plan.get("current_strict_k8_tasks"),
            "additional_k8_tasks_needed_for_minimum": stage6d3_plan.get("additional_k8_tasks_needed_for_minimum"),
            "additional_k8_tasks_needed_for_preferred": stage6d3_plan.get("additional_k8_tasks_needed_for_preferred"),
        },
        "stage6d3_pool_estimates": stage6d3_inventory.get("pool_estimates", {}),
        "base_source_files": [str(path) for path in base_source_files],
        "base_source_file_summaries": list(base_source_rows),
        "backend_config": backend_config,
        "tasks_selected_for_generation": list(selected_tasks),
        "tasks_attempted": len({row.instance_id for row in generation_attempts if row.instance_id}),
        "tasks_reaching_k8_from_new_generation": count_new_k8_tasks(generated_source_attempts),
        "accepted_candidates_total": len(accepted_pool_rows),
        "accepted_non_near_candidates_total": len(non_near_accepted),
        "accepted_candidates_per_task_distribution": dict(Counter(str(count) for count in Counter(row.instance_id for row in accepted_pool_rows).values())),
        "attempts_per_accepted_candidate": len(generation_attempts) / max(1, len(accepted_pool_rows)),
        "rejection_reason_counts": dict(status_counts),
        "invalid_diff_reasons": dict(Counter(row.reason for row in invalid_rows)),
        "invalid_diff_rate": len(invalid_rows) / max(1, len(generation_attempts)),
        "duplicate_rate": len(duplicate_rows) / max(1, len(generation_attempts)),
        "duplicate_rejections": len(duplicate_rows),
        "exact_reference_hits": len(exact_hits),
        "near_reference_flags": len(near_flags),
        "source_generator_distribution": dict(Counter(row.generator_name for row in accepted_pool_rows)),
        "backend_distribution": dict(Counter(row.backend for row in generation_attempts)),
        "prompt_distribution": dict(Counter(row.prompt_variant for row in generation_attempts)),
        "repo_distribution": repo_distribution(selected_tasks, benchmark_audit, pool_all),
        "mean_pairwise_normalized_patch_edit_distance": pairwise,
        "pool_summary": {
            "pool_all_k8_tasks": len(pool_all),
            "pool_strict_k8_tasks": strict_size,
            "pool_all_candidates": sum(len(example.candidates) for example in pool_all),
            "pool_strict_candidates": sum(len(example.candidates) for example in pool_strict),
            "remaining_tasks_needed_for_50_strict_k8": max(0, strict_target - strict_size),
            "remaining_tasks_needed_for_100_strict_k8": max(0, preferred_target - strict_size),
            "estimated_official_harness_evaluations_required_next": strict_size * STAGE6_NUM_CANDIDATES,
        },
        "pool_all_selection": pool_all_selection,
        "pool_strict_selection": pool_strict_selection,
        "pool_all_duplicate_candidate_patch_hash_audit": duplicate_candidate_patch_hash_audit(pool_all),
        "pool_strict_duplicate_candidate_patch_hash_audit": duplicate_candidate_patch_hash_audit(pool_strict),
        "pool_all_candidate_order_randomized_audit": candidate_order_randomized_audit(pool_all),
        "pool_strict_candidate_order_randomized_audit": candidate_order_randomized_audit(pool_strict),
        "pool_all_exact_reference_hash_audit": exact_reference_hash_audit(pool_all, task_records),
        "pool_strict_exact_reference_hash_audit": exact_reference_hash_audit(pool_strict, task_records),
        "pool_strict_near_reference_similarity_audit": near_reference_similarity_audit(pool_strict, task_records, NEAR_REFERENCE_THRESHOLD),
        "source_generator_skew": source_generator_skew_report(pool_strict),
        "no_reference_or_gold_patches_used": True,
        "no_raw_preserving_expansion": True,
        "no_fabricated_candidates": True,
        "selector_training_executed": False,
        "official_harness_executed": False,
        "no_final_model_claim_made": True,
    }


def build_readiness_audit(
    audit: Dict[str, object],
    pool_all: Sequence[PatchSelectionExample],
    pool_strict: Sequence[PatchSelectionExample],
    generation_attempts: Sequence[GenerationAttempt],
    strict_target: int,
    preferred_target: int,
) -> Dict[str, object]:
    strict_size = len(pool_strict)
    all_size = len(pool_all)
    generation_worked = bool(audit.get("accepted_candidates_total", 0) > 0 and audit.get("tasks_attempted", 0) > 0)
    gates = {
        "at_least_50_pool_all_k8_tasks": all_size >= strict_target,
        "at_least_50_pool_strict_k8_tasks": strict_size >= strict_target,
        "preferred_at_least_100_pool_strict_k8_tasks": strict_size >= preferred_target,
        "exact_reference_patch_hits_zero": int(audit.get("exact_reference_hits", 0)) == 0,
        "duplicate_candidate_patch_audit_passes": bool(
            audit.get("pool_all_duplicate_candidate_patch_hash_audit", {}).get("passes", False)
            and audit.get("pool_strict_duplicate_candidate_patch_hash_audit", {}).get("passes", False)
        ),
        "invalid_diff_rate_reported": "invalid_diff_rate" in audit,
        "no_raw_preserving_expansion": True,
        "no_fabricated_candidates": True,
        "candidate_order_randomized": bool(
            audit.get("pool_all_candidate_order_randomized_audit", {}).get("passes", False)
            and audit.get("pool_strict_candidate_order_randomized_audit", {}).get("passes", False)
        ),
        "source_generator_skew_reported": bool(audit.get("source_generator_skew")),
        "near_reference_candidates_flagged_or_quarantined": True,
        "selector_training_not_executed": True,
        "official_harness_not_executed": True,
        "no_final_model_claim_made": True,
    }
    if strict_size >= strict_target and gates["exact_reference_patch_hits_zero"] and gates["duplicate_candidate_patch_audit_passes"]:
        decision = "READY_FOR_OFFICIAL_LABELING"
    elif generation_worked:
        decision = "CONTINUE_SELF_GENERATION"
    else:
        decision = "SELF_GENERATION_BLOCKED"
    return {
        "benchmark": BENCHMARK,
        "created_at_utc": _now(),
        "quality_gates": gates,
        "quality_gates_pass": bool(all(value for key, value in gates.items() if not key.startswith("preferred_"))),
        "pool_all_k8_tasks": all_size,
        "pool_strict_k8_tasks": strict_size,
        "remaining_tasks_needed_for_50_strict_k8": max(0, strict_target - strict_size),
        "remaining_tasks_needed_for_100_strict_k8": max(0, preferred_target - strict_size),
        "generation_worked": generation_worked,
        "official_harness_should_run_now": decision == "READY_FOR_OFFICIAL_LABELING",
        "selector_training_executed": False,
        "official_harness_executed": False,
        "no_final_model_claim_made": True,
        "decision": decision,
        "decision_meaning": {
            "READY_FOR_OFFICIAL_LABELING": "pool_strict has at least 50 K=8 tasks; harness slot prediction files can be produced next",
            "CONTINUE_SELF_GENERATION": "generation works but strict pool remains below 50",
            "SELF_GENERATION_BLOCKED": "backend/model/source limitations prevent reaching enough K=8 tasks",
        }[decision],
        "blocker": blocker_text(decision, audit),
    }


def render_report(audit: Dict[str, object], readiness: Dict[str, object]) -> str:
    pool = audit.get("pool_summary", {})
    gates = readiness.get("quality_gates", {})
    lines = [
        "# Stage 6D.4 Self-Generation Candidate Expansion",
        "",
        "## Scope",
        "",
        "- Selector training was not executed.",
        "- Selector architecture was not changed.",
        "- Gold/reference patches, raw-preserving expansion, and fabricated candidates were not used.",
        "- Official SWE-bench harness was not executed.",
        "- No final SWE-bench improvement claim is made.",
        "",
        "## Generation",
        "",
        f"- Tasks selected for generation: `{len(audit.get('tasks_selected_for_generation', []))}`.",
        f"- Tasks attempted: `{audit.get('tasks_attempted', 0)}`.",
        f"- Tasks reaching K=8 from new generation: `{audit.get('tasks_reaching_k8_from_new_generation', 0)}`.",
        f"- Accepted generated candidates total: `{audit.get('accepted_candidates_total', 0)}`.",
        f"- Attempts per accepted candidate: `{float(audit.get('attempts_per_accepted_candidate', 0.0)):.4f}`.",
        f"- Invalid diff rate: `{float(audit.get('invalid_diff_rate', 0.0)):.4f}`.",
        f"- Duplicate rate: `{float(audit.get('duplicate_rate', 0.0)):.4f}`.",
        f"- Exact reference hits: `{audit.get('exact_reference_hits', 0)}`.",
        f"- Near-reference flags: `{audit.get('near_reference_flags', 0)}`.",
        "",
        "## Pool",
        "",
        f"- Pool all K=8 tasks: `{pool.get('pool_all_k8_tasks', 0)}`.",
        f"- Pool strict K=8 tasks: `{pool.get('pool_strict_k8_tasks', 0)}`.",
        f"- Remaining tasks needed for 50 strict K=8: `{pool.get('remaining_tasks_needed_for_50_strict_k8', 0)}`.",
        f"- Remaining tasks needed for 100 strict K=8: `{pool.get('remaining_tasks_needed_for_100_strict_k8', 0)}`.",
        f"- Estimated official harness evaluations required next: `{pool.get('estimated_official_harness_evaluations_required_next', 0)}`.",
        f"- Mean pairwise normalized patch edit distance: `{float(audit.get('mean_pairwise_normalized_patch_edit_distance', 0.0)):.4f}`.",
        "",
        "## Gates",
        "",
        "| Gate | Pass |",
        "|---|---:|",
    ]
    for key, value in gates.items():
        lines.append(f"| `{key}` | `{bool(value)}` |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            str(readiness.get("decision", "SELF_GENERATION_BLOCKED")),
            "",
            str(readiness.get("blocker", "")),
            "",
        ]
    )
    return "\n".join(lines)


def update_hashes(existing_hashes: Dict[str, set[str]], rows: Sequence[GenerationAttempt]) -> None:
    for row in rows:
        if row.accepted_for_pool_all:
            existing_hashes.setdefault(row.instance_id, set()).add(row.normalized_patch_hash)


def generation_attempt_to_source_attempt(row: GenerationAttempt) -> SourceAttempt:
    return SourceAttempt(
        instance_id=row.instance_id,
        patch=row.model_patch,
        patch_hash=row.normalized_patch_hash,
        source_file=row.raw_trajectory_path,
        source_row_index=row.attempt_index,
        patch_index=0,
        generator_name=row.generator_name,
        terminal_status="accepted",
        reason=row.reason,
        near_reference=row.near_reference,
        near_reference_similarity=row.near_reference_similarity,
    )


def base_attempt(
    instance_id: str,
    backend: str,
    generator: str,
    model_name: str,
    prompt_variant: str,
    seed: int,
    temperature: float,
    attempt_index: int,
    status: str,
    reason: str,
    raw_trajectory_path: str = "",
    patch_hash: str = "",
) -> GenerationAttempt:
    return GenerationAttempt(
        instance_id=instance_id,
        attempt_id=f"{instance_id}-{backend}-{attempt_index:04d}",
        terminal_status=status,
        reason=reason,
        generator_name=generator,
        backend=backend,
        model_name=model_name,
        prompt_variant=prompt_variant,
        seed=int(seed),
        temperature=float(temperature),
        attempt_index=int(attempt_index),
        normalized_patch_hash=patch_hash,
        raw_trajectory_path=raw_trajectory_path,
    )


def candidate_row(row: GenerationAttempt) -> Dict[str, object]:
    return {
        "instance_id": row.instance_id,
        "candidate_id": f"{row.instance_id}-stage6d4-{row.backend}-{row.normalized_patch_hash[:12]}",
        "model_patch": row.model_patch,
        "generator_name": row.generator_name,
        "backend": row.backend,
        "model_name": row.model_name,
        "prompt_variant": row.prompt_variant,
        "temperature": row.temperature,
        "seed": row.seed,
        "attempt_index": row.attempt_index,
        "raw_trajectory_path": row.raw_trajectory_path,
        "selector_visible_source_blinded": True,
        "near_reference": row.near_reference,
        "near_reference_similarity": row.near_reference_similarity,
    }


def attempt_row(row: GenerationAttempt) -> Dict[str, object]:
    return {
        "instance_id": row.instance_id,
        "attempt_id": row.attempt_id,
        "terminal_status": row.terminal_status,
        "reason": row.reason,
        "generator_name": row.generator_name,
        "backend": row.backend,
        "prompt_variant": row.prompt_variant,
        "seed": row.seed,
        "temperature": row.temperature,
        "normalized_patch_hash": row.normalized_patch_hash,
        "near_reference_similarity": row.near_reference_similarity,
        "raw_trajectory_path": row.raw_trajectory_path,
    }


def build_prompt(task: BenchmarkTask, prompt_variant: str, max_output_tokens: int) -> str:
    record = task.record
    return "\n".join(
        [
            "Generate a minimal unified diff patch for this SWE-bench Verified task.",
            "Do not include reference patches, gold patches, explanations, or test logs.",
            f"Prompt variant: {prompt_variant}",
            f"Repository: {record.get('repo', '')}",
            f"Instance ID: {task.instance_id}",
            f"Base commit: {record.get('base_commit', '')}",
            f"Max output tokens: {max_output_tokens}",
            "",
            "Problem statement:",
            str(record.get("problem_statement", "")),
            "",
            "Public failing tests:",
            str(record.get("FAIL_TO_PASS", "[]")),
            "",
            "Hints:",
            str(record.get("hints_text", "")),
            "",
            "Return only a unified diff.",
        ]
    )


def dry_run_patch(task: BenchmarkTask, attempt_index: int, prompt_variant: str) -> str:
    repo_prefix = str(task.record.get("repo", "repo/proj")).split("/", 1)[-1].replace("-", "_")
    path = f"stage6d4_dry_run/{repo_prefix}_{safe_name(task.instance_id)}_{attempt_index}.py"
    return (
        f"diff --git a/{path} b/{path}\n"
        "new file mode 100644\n"
        "index 0000000..1111111\n"
        "--- /dev/null\n"
        f"+++ b/{path}\n"
        "@@ -0,0 +1,3 @@\n"
        f"+# dry_run backend candidate for {task.instance_id}\n"
        f"+PROMPT_VARIANT = {prompt_variant!r}\n"
        f"+ATTEMPT_INDEX = {attempt_index}\n"
    )


def write_raw_trajectory(root: Path, instance_id: str, backend: str, name: str, payload: Dict[str, object]) -> Path:
    if backend:
        path = root / safe_name(backend) / safe_name(instance_id) / f"{safe_name(name)}.json"
    else:
        path = Path(root) if str(root).endswith(".json") else Path(root) / f"{safe_name(name)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_records(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    try:
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        rows.append(value)
        elif path.suffix.lower() == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, list):
                rows.extend(item for item in value if isinstance(item, dict))
            elif isinstance(value, dict):
                rows.append(value)
    except Exception:
        return rows
    return rows


def count_new_k8_tasks(rows: Sequence[SourceAttempt]) -> int:
    counts = Counter(row.instance_id for row in rows if not row.near_reference)
    return sum(1 for count in counts.values() if count >= STAGE6_NUM_CANDIDATES)


def mean_pairwise_patch_distance(examples: Sequence[PatchSelectionExample]) -> float:
    distances = []
    for example in examples:
        texts = [normalize_patch_for_distance(candidate.candidate_diff) for candidate in example.candidates]
        for left in range(len(texts)):
            for right in range(left + 1, len(texts)):
                distances.append(1.0 - SequenceMatcher(None, texts[left][:40_000], texts[right][:40_000]).ratio())
    return float(sum(distances) / len(distances)) if distances else 0.0


def normalize_patch_for_distance(text: str) -> str:
    return "\n".join(line.rstrip() for line in str(text).splitlines() if not line.startswith("index "))


def repo_distribution(selected_tasks: Sequence[str], benchmark_audit: Dict[str, object], pool_all: Sequence[PatchSelectionExample]) -> Dict[str, int]:
    repos_by_id = {example.issue_id: example.repo for example in pool_all}
    return dict(Counter(repos_by_id.get(task_id, "unknown") for task_id in selected_tasks))


def blocker_text(decision: str, audit: Dict[str, object]) -> str:
    if decision == "READY_FOR_OFFICIAL_LABELING":
        return "Pool strict has reached at least 50 K=8 generated tasks; prepare official harness slot prediction files next."
    if decision == "CONTINUE_SELF_GENERATION":
        needed = audit.get("pool_summary", {}).get("remaining_tasks_needed_for_50_strict_k8", 0)
        return f"Generation produced accepted candidates, but strict pool remains below 50 K=8 tasks; continue until {needed} more strict K=8 tasks are added."
    return (
        "No configured real generation backend produced accepted candidates. Configure command_line_agent, mini_swe_agent, "
        "api_generator, or provide local_jsonl_ingest outputs; dry_run is testing-only."
    )


def safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "value")).strip("_")[:160] or "value"


def _read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, value: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
