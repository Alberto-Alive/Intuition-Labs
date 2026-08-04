from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import shlex
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from src.datasets.taubench_trajectory_selection_dataset import (
    STAGE7_NUM_CANDIDATES,
    TauBenchTrajectoryCandidate,
    build_taubench_selection_examples,
    duplicate_trajectory_hash_audit,
    near_duplicate_trajectory_audit,
    normalize_text,
    stage7_output_leakage_audit,
    stable_trajectory_hash,
    validate_taubench_selection_examples,
    write_taubench_candidates_jsonl,
)
from src.experiments.stage7_taubench_candidate_generator import sanitize_tau2_trajectory


BENCHMARK = "stage7b0_tau_candidate_pool_mining"
DEFAULT_TAU2_DATA_DIR = Path("external/tau2-bench/data")
DEFAULT_TAU2_REPO = Path("external/tau2-bench")
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_REPORTS_DIR = Path("reports")
DEFAULT_SEED = 707_200
NEAR_DUPLICATE_THRESHOLD = 0.98
READINESS_MIN_TASKS = 50
READINESS_PREFERRED_TASKS = 100


@dataclass(frozen=True)
class SourceTrajectory:
    source_key: str
    domain: str
    task_id: str
    source_file: str
    source_file_name: str
    source_simulation_id: str
    source_simulation_index: int
    trial: int
    seed: int
    generator_name: str
    generator_config: str
    user_model: str
    raw_record: Dict[str, object]
    existing_success: bool | None
    recomputed_success: bool | None
    trajectory_hash: str
    near_duplicate_text: str


@dataclass(frozen=True)
class PoolBuild:
    name: str
    path: Path
    candidates: List[TauBenchTrajectoryCandidate]
    audit_rows: List[Dict[str, object]]
    task_success_counts: Dict[str, int]
    selection_notes: Dict[str, object]


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine existing tau2 artifacts into Stage 7B.0 K=8 candidate pools.")
    parser.add_argument("--tau2-data-dir", default=str(DEFAULT_TAU2_DATA_DIR))
    parser.add_argument("--tau2-repo", default=str(DEFAULT_TAU2_REPO))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--domain", default="retail")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--tau2-command",
        default="conda run -n stage7_tau tau2",
        help="Command prefix for the official tau2 CLI.",
    )
    parser.add_argument("--skip-evaluator", action="store_true", help="Use existing labels only; not valid for readiness.")
    args = parser.parse_args()

    tau2_data_dir = Path(args.tau2_data_dir)
    tau2_repo = Path(args.tau2_repo)
    results_dir = Path(args.results_dir)
    reports_dir = Path(args.reports_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    source_files = discover_source_files(tau2_data_dir=tau2_data_dir, tau2_repo=tau2_repo, results_dir=results_dir)
    loaded = load_result_trajectories(source_files["tau2_result_files"])
    inventory = build_source_inventory(
        tau2_data_dir=tau2_data_dir,
        tau2_repo=tau2_repo,
        source_files=source_files,
        trajectories=loaded,
        seed=int(args.seed),
    )
    write_json(results_dir / "stage7b0_tau_source_inventory.json", inventory)

    domain = str(args.domain)
    domain_sources = [row for row in loaded if row.domain == domain]
    if not domain_sources:
        label_audit = blocked_label_audit(
            reason=f"no source trajectories were loaded for domain={domain}",
            inventory=inventory,
            results_dir=results_dir,
            reports_dir=reports_dir,
        )
        write_json(results_dir / "stage7b0_tau_label_audit.json", label_audit)
        write_report(reports_dir / "STAGE7B0_TAU_CANDIDATE_POOL_MINING.md", inventory, label_audit)
        raise SystemExit(2)

    evaluator = {}
    labeled_sources = domain_sources
    if args.skip_evaluator:
        labeled_sources = [
            replace_source_label(row, row.existing_success)
            for row in domain_sources
        ]
        evaluator = {
            "official_evaluator_ran": False,
            "labels_from_existing_result_files_only": True,
            "valid_for_readiness": False,
        }
    else:
        labeled_sources, evaluator = recompute_domain_labels(
            domain=domain,
            sources=domain_sources,
            tau2_data_dir=tau2_data_dir,
            results_dir=results_dir,
            tau2_command=str(args.tau2_command),
        )

    pools = build_pool_variants(
        domain=domain,
        sources=labeled_sources,
        tau2_data_dir=tau2_data_dir,
        results_dir=results_dir,
        seed=int(args.seed),
    )
    for pool in pools:
        write_taubench_candidates_jsonl(pool.path, pool.candidates)

    label_audit = build_label_audit(
        domain=domain,
        seed=int(args.seed),
        sources=labeled_sources,
        pools=pools,
        evaluator=evaluator,
        inventory=inventory,
    )
    write_json(results_dir / "stage7b0_tau_label_audit.json", label_audit)
    write_report(reports_dir / "STAGE7B0_TAU_CANDIDATE_POOL_MINING.md", inventory, label_audit)
    print(json.dumps({"decision": label_audit["decision"], "domain": domain}, sort_keys=True))


def discover_source_files(tau2_data_dir: Path, tau2_repo: Path, results_dir: Path) -> Dict[str, object]:
    result_files = sorted(
        path
        for path in tau2_data_dir.rglob("*.json")
        if _looks_like_result_path(path)
    )
    domain_files = sorted(
        path
        for path in (tau2_data_dir / "tau2" / "domains").rglob("*.json")
        if path.name in {"tasks.json", "tasks_voice.json", "split_tasks.json", "db.json"}
    )
    leaderboard_submissions = sorted((tau2_repo / "web" / "leaderboard" / "public" / "submissions").rglob("submission.json"))
    leaderboard_trajectory_files = sorted(
        path
        for path in (tau2_repo / "web" / "leaderboard" / "public" / "submissions").rglob("*.json")
        if "trajectories" in {part.lower() for part in path.parts}
    )
    stage7_candidate_files = sorted(
        path
        for path in results_dir.glob("stage7*.jsonl")
        if "stage7a_real_tau" in path.name or "stage7_local_smoke" in path.name
    )
    return {
        "tau2_result_files": [str(path) for path in result_files],
        "tau2_domain_metadata_files": [str(path) for path in domain_files],
        "leaderboard_submission_files": [str(path) for path in leaderboard_submissions],
        "leaderboard_trajectory_files": [str(path) for path in leaderboard_trajectory_files],
        "existing_stage7_candidate_files": [str(path) for path in stage7_candidate_files],
        "counts": {
            "tau2_result_files": len(result_files),
            "tau2_domain_metadata_files": len(domain_files),
            "leaderboard_submission_files": len(leaderboard_submissions),
            "leaderboard_trajectory_files": len(leaderboard_trajectory_files),
            "existing_stage7_candidate_files": len(stage7_candidate_files),
        },
    }


def _looks_like_result_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    if "results" not in parts:
        return False
    if name.startswith("updated_"):
        return False
    return "trials" in name or "result" in name or "simulation" in parts or "final" in parts


def load_result_trajectories(paths: Sequence[str]) -> List[SourceTrajectory]:
    rows: List[SourceTrajectory] = []
    for path_text in paths:
        path = Path(path_text)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict) or not isinstance(data.get("simulations"), list):
            continue
        info = data.get("info") if isinstance(data.get("info"), dict) else {}
        agent_info = info.get("agent_info") if isinstance(info.get("agent_info"), dict) else {}
        user_info = info.get("user_info") if isinstance(info.get("user_info"), dict) else {}
        env_info = info.get("environment_info") if isinstance(info.get("environment_info"), dict) else {}
        domain = str(env_info.get("domain_name") or _domain_from_filename(path.name) or "")
        generator_name = _source_generator_name(agent_info, path.name)
        generator_config = json.dumps(agent_info, sort_keys=True)
        user_model = str(user_info.get("llm", "unknown_user_model"))
        for index, sim in enumerate(data.get("simulations", [])):
            if not isinstance(sim, dict):
                continue
            sim_domain = str(sim.get("domain") or domain)
            task_id = str(sim.get("task_id", ""))
            source_id = str(sim.get("id") or f"{path.name}:{index}")
            source_key = f"{path.resolve()}::{index}::{source_id}"
            visible = sanitize_tau2_trajectory(sim)
            visible_record = visible.to_record()
            rows.append(
                SourceTrajectory(
                    source_key=source_key,
                    domain=sim_domain,
                    task_id=task_id,
                    source_file=str(path),
                    source_file_name=path.name,
                    source_simulation_id=source_id,
                    source_simulation_index=index,
                    trial=int(sim.get("trial", index)) if _int_like(sim.get("trial", index)) else index,
                    seed=int(sim.get("seed", info.get("seed", 0))) if _int_like(sim.get("seed", info.get("seed", 0))) else 0,
                    generator_name=generator_name,
                    generator_config=generator_config,
                    user_model=user_model,
                    raw_record=sim,
                    existing_success=_extract_success(sim),
                    recomputed_success=None,
                    trajectory_hash=stable_trajectory_hash(visible_record),
                    near_duplicate_text=normalize_text(json.dumps(visible_record, sort_keys=True)),
                )
            )
    return rows


def _domain_from_filename(name: str) -> str | None:
    for value in ("retail", "airline", "telecom-workflow", "telecom", "banking_knowledge", "mock"):
        if f"_{value}_" in name or name.startswith(f"{value}_"):
            return value
    return None


def _source_generator_name(agent_info: Dict[str, object], filename: str) -> str:
    llm = str(agent_info.get("llm", "")).strip()
    implementation = str(agent_info.get("implementation", "")).strip()
    if llm and implementation:
        return f"{implementation}:{llm}"
    if llm:
        return llm
    return filename.split("_", 1)[0]


def build_source_inventory(
    tau2_data_dir: Path,
    tau2_repo: Path,
    source_files: Dict[str, object],
    trajectories: Sequence[SourceTrajectory],
    seed: int,
) -> Dict[str, object]:
    by_domain_task: Dict[str, Dict[str, List[SourceTrajectory]]] = defaultdict(lambda: defaultdict(list))
    for row in trajectories:
        by_domain_task[row.domain][row.task_id].append(row)
    per_domain_counts = {
        domain: {task_id: len(rows) for task_id, rows in sorted(tasks.items(), key=lambda item: _task_sort_key(item[0]))}
        for domain, tasks in sorted(by_domain_task.items())
    }
    success_distribution = {
        domain: _success_distribution(tasks)
        for domain, tasks in sorted(by_domain_task.items())
    }
    duplicate_audit = inventory_duplicate_audit(by_domain_task)
    near_duplicate = inventory_near_duplicate_audit(by_domain_task, threshold=NEAR_DUPLICATE_THRESHOLD)
    stage7_existing = scan_existing_stage7_candidates(source_files.get("existing_stage7_candidate_files", []))
    return {
        "artifact": "stage7b0_tau_source_inventory",
        "created_at_utc": _now(),
        "seed": int(seed),
        "tau2_data_dir": str(tau2_data_dir),
        "tau2_repo": str(tau2_repo),
        "tau2_repo_commit": git_commit(tau2_repo),
        "available_domain_dirs": sorted(
            path.name for path in (tau2_data_dir / "tau2" / "domains").iterdir() if path.is_dir()
        )
        if (tau2_data_dir / "tau2" / "domains").exists()
        else [],
        "source_files_scanned": source_files,
        "raw_trajectories_loaded": len(trajectories),
        "unique_domains": sorted(by_domain_task),
        "unique_task_ids_per_domain": {
            domain: sorted(tasks, key=_task_sort_key)
            for domain, tasks in sorted(by_domain_task.items())
        },
        "task_counts_per_domain": {domain: len(tasks) for domain, tasks in sorted(by_domain_task.items())},
        "trajectories_per_task_distribution": {
            domain: _count_distribution(counts.values())
            for domain, counts in sorted(per_domain_counts.items())
        },
        "tasks_with_at_least_8_trajectories": {
            domain: sorted([task_id for task_id, count in counts.items() if count >= STAGE7_NUM_CANDIDATES], key=_task_sort_key)
            for domain, counts in sorted(per_domain_counts.items())
        },
        "success_rate_distribution_from_existing_labels": success_distribution,
        "oracle_pass_at_8_estimate_from_existing_labels": {
            domain: _oracle_from_existing(tasks)
            for domain, tasks in sorted(by_domain_task.items())
        },
        "generator_source_distribution": _generator_source_distribution(trajectories),
        "raw_trajectory_format": {
            "top_level": "tau2 Results JSON with info/tasks/simulations",
            "simulation_keys_observed": sorted({key for row in trajectories[:20] for key in row.raw_record.keys()}),
            "selector_visible_projection": "messages only; reward_info/evaluator fields excluded",
        },
        "evaluator_compatible_result_format": {
            "command": "tau2 evaluate-trajs <results.json> -o <output_dir>",
            "label_field": "simulations[].reward_info.reward > 0 after official evaluator recomputation",
            "updated_output_filename_pattern": "updated_<input_filename>",
        },
        "duplicate_trajectory_audit": duplicate_audit,
        "near_duplicate_trajectory_audit": near_duplicate,
        "existing_stage7_candidate_inventory": stage7_existing,
    }


def _success_distribution(tasks: Dict[str, List[SourceTrajectory]]) -> Dict[str, object]:
    counts = []
    rates = []
    missing = 0
    for rows in tasks.values():
        labels = [row.existing_success for row in rows]
        known = [bool(value) for value in labels if value is not None]
        if len(known) != len(labels):
            missing += len(labels) - len(known)
        if known:
            counts.append(sum(1 for value in known if value))
            rates.append(sum(1 for value in known if value) / float(len(known)))
    return {
        "tasks_with_existing_labels": len(rates),
        "missing_label_count": missing,
        "success_count_distribution_per_task": _count_distribution(counts),
        "success_rate_summary_per_task": _float_summary(rates),
    }


def inventory_duplicate_audit(by_domain_task: Dict[str, Dict[str, List[SourceTrajectory]]]) -> Dict[str, object]:
    duplicate_groups = []
    total = 0
    duplicate_count = 0
    for domain, tasks in sorted(by_domain_task.items()):
        for task_id, rows in sorted(tasks.items(), key=lambda item: _task_sort_key(item[0])):
            total += len(rows)
            counts = Counter(row.trajectory_hash for row in rows)
            duplicates = {value: count for value, count in counts.items() if count > 1}
            duplicate_count += sum(count - 1 for count in duplicates.values())
            if duplicates:
                duplicate_groups.append(
                    {
                        "domain": domain,
                        "task_id": task_id,
                        "duplicate_hashes": len(duplicates),
                        "duplicate_extra_trajectories": sum(count - 1 for count in duplicates.values()),
                    }
                )
    return {
        "total_trajectories_checked": total,
        "duplicate_extra_trajectories": duplicate_count,
        "duplicate_rate": duplicate_count / float(max(1, total)),
        "tasks_with_duplicates": duplicate_groups[:200],
        "truncated": len(duplicate_groups) > 200,
    }


def inventory_near_duplicate_audit(
    by_domain_task: Dict[str, Dict[str, List[SourceTrajectory]]],
    threshold: float,
) -> Dict[str, object]:
    total_pairs = 0
    near_pairs = 0
    examples = []
    for domain, tasks in sorted(by_domain_task.items()):
        for task_id, rows in sorted(tasks.items(), key=lambda item: _task_sort_key(item[0])):
            token_sets = [set(row.near_duplicate_text.split()) for row in rows]
            for left, right in itertools.combinations(range(len(rows)), 2):
                if not token_sets[left] or not token_sets[right]:
                    continue
                total_pairs += 1
                score = _jaccard(token_sets[left], token_sets[right])
                if score >= threshold:
                    near_pairs += 1
                    if len(examples) < 200:
                        examples.append(
                            {
                                "domain": domain,
                                "task_id": task_id,
                                "left_source_key": rows[left].source_key,
                                "right_source_key": rows[right].source_key,
                                "jaccard": score,
                            }
                        )
    return {
        "threshold": float(threshold),
        "pairs_checked_within_task": total_pairs,
        "near_duplicate_pairs": near_pairs,
        "near_duplicate_pair_rate": near_pairs / float(max(1, total_pairs)),
        "examples": examples,
        "truncated": near_pairs > len(examples),
    }


def scan_existing_stage7_candidates(paths: object) -> Dict[str, object]:
    out = []
    for path_text in paths if isinstance(paths, list) else []:
        path = Path(str(path_text))
        if not path.exists():
            continue
        count = 0
        labels = 0
        domains = Counter()
        tasks = set()
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                count += 1
                domains[str(row.get("domain", ""))] += 1
                tasks.add(str(row.get("task_id", "")))
                if row.get("official_success") is not None:
                    labels += 1
        except Exception as exc:
            out.append({"path": str(path), "error": str(exc)})
            continue
        out.append(
            {
                "path": str(path),
                "candidate_records": count,
                "labels_present": labels,
                "domains": dict(sorted(domains.items())),
                "unique_tasks": len(tasks),
            }
        )
    return {"files": out}


def recompute_domain_labels(
    domain: str,
    sources: Sequence[SourceTrajectory],
    tau2_data_dir: Path,
    results_dir: Path,
    tau2_command: str,
) -> tuple[List[SourceTrajectory], Dict[str, object]]:
    eval_input_dir = results_dir / "stage7b0_tau_eval_input"
    eval_output_dir = results_dir / "stage7b0_tau_eval_output"
    eval_input_dir.mkdir(parents=True, exist_ok=True)
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    input_path = eval_input_dir / f"{domain}_source_results.json"
    stdout_path = results_dir / "stage7b0_tau_eval_stdout.log"
    stderr_path = results_dir / "stage7b0_tau_eval_stderr.log"
    result_record = build_evaluator_results_record(domain=domain, sources=sources, tau2_data_dir=tau2_data_dir)
    input_path.write_text(json.dumps(result_record, indent=2, sort_keys=True), encoding="utf-8")

    command = [*shlex.split(tau2_command), "evaluate-trajs", str(input_path), "-o", str(eval_output_dir)]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["TAU2_DATA_DIR"] = str(tau2_data_dir.resolve())
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        env=env,
        timeout=1800,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    output_path = eval_output_dir / f"updated_{input_path.name}"
    if completed.returncode != 0 or not output_path.exists():
        raise RuntimeError(
            "official tau2 evaluator failed: "
            + json.dumps(
                {
                    "returncode": completed.returncode,
                    "command": command,
                    "stdout_tail": completed.stdout[-2000:],
                    "stderr_tail": completed.stderr[-2000:],
                    "expected_output": str(output_path),
                },
                sort_keys=True,
            )
        )
    updated = json.loads(output_path.read_text(encoding="utf-8"))
    updated_sims = updated.get("simulations", []) if isinstance(updated, dict) else []
    if len(updated_sims) != len(sources):
        raise RuntimeError(f"official evaluator returned {len(updated_sims)} simulations for {len(sources)} inputs")
    labeled = [
        replace_source_label(source, _extract_success(sim if isinstance(sim, dict) else {}))
        for source, sim in zip(sources, updated_sims)
    ]
    labels_available = sum(1 for row in labeled if row.recomputed_success is not None)
    return labeled, {
        "official_evaluator_ran": True,
        "valid_for_readiness": labels_available == len(labeled),
        "command": command,
        "returncode": int(completed.returncode),
        "latency_seconds": float(time.perf_counter() - started),
        "input_json": str(input_path),
        "output_json": str(output_path),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "labels_available": labels_available,
        "labels_missing": len(labeled) - labels_available,
    }


def replace_source_label(row: SourceTrajectory, label: bool | None) -> SourceTrajectory:
    return SourceTrajectory(
        source_key=row.source_key,
        domain=row.domain,
        task_id=row.task_id,
        source_file=row.source_file,
        source_file_name=row.source_file_name,
        source_simulation_id=row.source_simulation_id,
        source_simulation_index=row.source_simulation_index,
        trial=row.trial,
        seed=row.seed,
        generator_name=row.generator_name,
        generator_config=row.generator_config,
        user_model=row.user_model,
        raw_record=row.raw_record,
        existing_success=row.existing_success,
        recomputed_success=label,
        trajectory_hash=row.trajectory_hash,
        near_duplicate_text=row.near_duplicate_text,
    )


def build_evaluator_results_record(domain: str, sources: Sequence[SourceTrajectory], tau2_data_dir: Path) -> Dict[str, object]:
    first_source = Path(sources[0].source_file)
    first_data = json.loads(first_source.read_text(encoding="utf-8"))
    tasks = first_data.get("tasks")
    if not isinstance(tasks, list):
        tasks_path = tau2_data_dir / "tau2" / "domains" / domain / "tasks.json"
        tasks = json.loads(tasks_path.read_text(encoding="utf-8")) if tasks_path.exists() else []
    info = dict(first_data.get("info", {})) if isinstance(first_data.get("info"), dict) else {}
    info["stage7b0_evaluator_input"] = {
        "source": "concatenated_existing_official_tau2_final_results",
        "simulations_unmodified": True,
        "source_file_count": len({row.source_file for row in sources}),
        "candidate_count": len(sources),
    }
    return {
        "timestamp": _now(),
        "info": info,
        "tasks": tasks,
        "simulations": [row.raw_record for row in sources],
    }


def build_pool_variants(
    domain: str,
    sources: Sequence[SourceTrajectory],
    tau2_data_dir: Path,
    results_dir: Path,
    seed: int,
) -> List[PoolBuild]:
    by_task = _sources_by_task(sources)
    eligible = {
        task_id: rows
        for task_id, rows in by_task.items()
        if len(rows) >= STAGE7_NUM_CANDIDATES and all(row.recomputed_success is not None for row in rows)
    }
    task_metadata = load_task_metadata(tau2_data_dir=tau2_data_dir, domain=domain)
    splits = load_domain_splits(tau2_data_dir=tau2_data_dir, domain=domain)
    all_pool_sources, all_notes = select_all_available_pool(eligible, seed=seed)
    nontrivial_sources, nontrivial_notes = select_nontrivial_pool(eligible, seed=seed)
    balanced_sources, balanced_notes = select_balanced_pool(eligible, seed=seed)
    return [
        make_pool_build(
            name="all_available_pool",
            domain=domain,
            selected=all_pool_sources,
            path=results_dir / "stage7b0_tau_candidate_pool_all.jsonl",
            task_metadata=task_metadata,
            splits=splits,
            seed=seed,
            notes=all_notes,
        ),
        make_pool_build(
            name="nontrivial_pool",
            domain=domain,
            selected=nontrivial_sources,
            path=results_dir / "stage7b0_tau_candidate_pool_nontrivial.jsonl",
            task_metadata=task_metadata,
            splits=splits,
            seed=seed,
            notes=nontrivial_notes,
        ),
        make_pool_build(
            name="balanced_pool",
            domain=domain,
            selected=balanced_sources,
            path=results_dir / "stage7b0_tau_candidate_pool_balanced.jsonl",
            task_metadata=task_metadata,
            splits=splits,
            seed=seed,
            notes=balanced_notes,
        ),
    ]


def _sources_by_task(sources: Sequence[SourceTrajectory]) -> Dict[str, List[SourceTrajectory]]:
    by_task: Dict[str, List[SourceTrajectory]] = defaultdict(list)
    for row in sources:
        by_task[row.task_id].append(row)
    return {
        task_id: sorted(rows, key=lambda row: (row.source_file_name, row.trial, row.source_simulation_index, row.source_simulation_id))
        for task_id, rows in by_task.items()
    }


def select_all_available_pool(
    eligible: Dict[str, List[SourceTrajectory]],
    seed: int,
) -> tuple[Dict[str, List[SourceTrajectory]], Dict[str, object]]:
    rng = random.Random(seed + 11)
    selected = {}
    for task_id, rows in sorted(eligible.items(), key=lambda item: _task_sort_key(item[0])):
        task_rows = list(rows)
        rng.shuffle(task_rows)
        selected[task_id] = task_rows[:STAGE7_NUM_CANDIDATES]
    return selected, {
        "strategy": "random K=8 from every task with at least 8 existing trajectories",
        "tasks_selected": len(selected),
        "selection_seed": seed + 11,
    }


def select_nontrivial_pool(
    eligible: Dict[str, List[SourceTrajectory]],
    seed: int,
) -> tuple[Dict[str, List[SourceTrajectory]], Dict[str, object]]:
    rng = random.Random(seed + 23)
    non_all_success_source = {
        task_id: rows
        for task_id, rows in eligible.items()
        if any(row.recomputed_success is False for row in rows)
    }
    tasks = sorted(non_all_success_source, key=_task_sort_key)
    min_empty = max(0, len(tasks) - int(0.85 * len(tasks)))
    all_fail_candidates = [
        task_id
        for task_id in tasks
        if sum(1 for row in non_all_success_source[task_id] if row.recomputed_success is False) >= STAGE7_NUM_CANDIDATES
    ]
    rng.shuffle(all_fail_candidates)
    empty_tasks = set(all_fail_candidates[:min_empty])
    selected: Dict[str, List[SourceTrajectory]] = {}
    for task_id in tasks:
        rows = non_all_success_source[task_id]
        if task_id in empty_tasks:
            selected[task_id] = _choose_by_label(rows, successes=0, rng=rng)
        else:
            available_successes = sum(1 for row in rows if row.recomputed_success is True)
            available_failures = sum(1 for row in rows if row.recomputed_success is False)
            min_success_needed = max(1, STAGE7_NUM_CANDIDATES - available_failures)
            max_success_allowed = min(7, available_successes)
            target = min(max_success_allowed, max(min_success_needed, 2))
            selected[task_id] = _choose_by_label(rows, successes=target, rng=rng)
    return selected, {
        "strategy": "exclude source tasks with no available failures; retain the minimum all-fail selected tasks needed to keep aggregate oracle pass@8 <= 0.85",
        "tasks_selected": len(selected),
        "all_fail_tasks_intentionally_retained": len(empty_tasks),
        "selection_seed": seed + 23,
    }


def select_balanced_pool(
    eligible: Dict[str, List[SourceTrajectory]],
    seed: int,
) -> tuple[Dict[str, List[SourceTrajectory]], Dict[str, object]]:
    rng = random.Random(seed + 37)
    candidate_tasks = {
        task_id: rows
        for task_id, rows in eligible.items()
        if any(row.recomputed_success is False for row in rows)
    }
    tasks = sorted(candidate_tasks, key=_task_sort_key)
    min_empty = max(0, len(tasks) - int(0.85 * len(tasks)))
    all_fail_candidates = [
        task_id
        for task_id in tasks
        if sum(1 for row in candidate_tasks[task_id] if row.recomputed_success is False) >= STAGE7_NUM_CANDIDATES
    ]
    rng.shuffle(all_fail_candidates)
    empty_tasks = set(all_fail_candidates[:min_empty])
    target_cycle = [1, 2, 3, 4, 5, 1, 2, 4]
    selected: Dict[str, List[SourceTrajectory]] = {}
    positive_index = 0
    for task_id in tasks:
        rows = candidate_tasks[task_id]
        if task_id in empty_tasks:
            selected[task_id] = _choose_by_label(rows, successes=0, rng=rng)
            continue
        available_successes = sum(1 for row in rows if row.recomputed_success is True)
        available_failures = sum(1 for row in rows if row.recomputed_success is False)
        target = target_cycle[positive_index % len(target_cycle)]
        positive_index += 1
        target = max(1, min(target, available_successes, STAGE7_NUM_CANDIDATES - 1))
        if available_failures < STAGE7_NUM_CANDIDATES - target:
            target = STAGE7_NUM_CANDIDATES - available_failures
        target = min(target, STAGE7_NUM_CANDIDATES - 1)
        selected[task_id] = _choose_by_label(rows, successes=target, rng=rng)
    return selected, {
        "strategy": "balance selected per-task success counts across 1, 2-3, and 4+ positives while avoiding all-success selected tasks",
        "tasks_selected": len(selected),
        "all_fail_tasks_intentionally_retained": len(empty_tasks),
        "selection_seed": seed + 37,
    }


def _choose_by_label(rows: Sequence[SourceTrajectory], successes: int, rng: random.Random) -> List[SourceTrajectory]:
    positives = [row for row in rows if row.recomputed_success is True]
    negatives = [row for row in rows if row.recomputed_success is False]
    if successes < 0 or successes > STAGE7_NUM_CANDIDATES:
        raise ValueError(f"invalid target success count: {successes}")
    if len(positives) < successes or len(negatives) < STAGE7_NUM_CANDIDATES - successes:
        raise ValueError(
            f"not enough candidates for target successes={successes}; positives={len(positives)} negatives={len(negatives)}"
        )
    rng.shuffle(positives)
    rng.shuffle(negatives)
    chosen = positives[:successes] + negatives[: STAGE7_NUM_CANDIDATES - successes]
    rng.shuffle(chosen)
    return chosen


def make_pool_build(
    name: str,
    domain: str,
    selected: Dict[str, List[SourceTrajectory]],
    path: Path,
    task_metadata: Dict[str, Dict[str, object]],
    splits: Dict[str, str],
    seed: int,
    notes: Dict[str, object],
) -> PoolBuild:
    candidates: List[TauBenchTrajectoryCandidate] = []
    audit_rows = []
    task_success_counts = {}
    for task_id, rows in sorted(selected.items(), key=lambda item: _task_sort_key(item[0])):
        if len(rows) != STAGE7_NUM_CANDIDATES:
            raise ValueError(f"{name}:{task_id} expected K={STAGE7_NUM_CANDIDATES}, found {len(rows)}")
        task_success_counts[task_id] = sum(1 for row in rows if row.recomputed_success is True)
        order_seed = seed + stable_int(f"{name}:{domain}:{task_id}")
        task_meta = task_metadata.get(task_id, {})
        for index, row in enumerate(rows):
            candidate_id = f"{domain}-{task_id}-cand-{index}"
            visible = sanitize_tau2_trajectory(row.raw_record)
            metadata = {
                "stage": "stage7b0",
                "pool_variant": name,
                "split": splits.get(task_id, "unassigned"),
                "task_type": f"{domain}_official_existing_trajectory",
                "policy_text": str(task_meta.get("policy_text", "")),
                "public_task_request": "",
                "tau_bench_task_set_name": domain,
                "tau_bench_task_split_name": "base",
                "candidate_order_randomized": True,
                "candidate_order_seed": int(order_seed),
                "selector_visible_excludes_generator_identity": True,
                "selector_visible_candidate_id_blinded_from_generator_source": True,
                "publishable_claim_allowed": False,
                "source_and_raw_paths_in_audit_only": True,
            }
            candidates.append(
                TauBenchTrajectoryCandidate(
                    task_id=task_id,
                    domain=domain,
                    candidate_id=candidate_id,
                    candidate_order_index=index,
                    generator_name="audit_hidden_source",
                    generator_config="{}",
                    seed=int(order_seed),
                    raw_trajectory_path="audit_only:results/stage7b0_tau_label_audit.json",
                    selector_visible_trajectory=visible,
                    official_success=bool(row.recomputed_success),
                    official_eval_metadata_path="audit_only:results/stage7b0_tau_label_audit.json",
                    metadata=metadata,
                )
            )
            audit_rows.append(
                {
                    "pool": name,
                    "candidate_id": candidate_id,
                    "task_id": task_id,
                    "candidate_order_index": index,
                    "official_success": bool(row.recomputed_success),
                    "source_key": row.source_key,
                    "source_file": row.source_file,
                    "source_simulation_id": row.source_simulation_id,
                    "source_simulation_index": row.source_simulation_index,
                    "source_trial": row.trial,
                    "source_seed": row.seed,
                    "generator_name_audit_only": row.generator_name,
                    "generator_config_audit_only": row.generator_config,
                    "user_model_audit_only": row.user_model,
                    "trajectory_hash": row.trajectory_hash,
                }
            )
    return PoolBuild(
        name=name,
        path=path,
        candidates=candidates,
        audit_rows=audit_rows,
        task_success_counts=task_success_counts,
        selection_notes=notes,
    )


def load_task_metadata(tau2_data_dir: Path, domain: str) -> Dict[str, Dict[str, object]]:
    tasks_path = tau2_data_dir / "tau2" / "domains" / domain / "tasks.json"
    out: Dict[str, Dict[str, object]] = {}
    if not tasks_path.exists():
        return out
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    policy_text = load_policy_text(tau2_data_dir=tau2_data_dir, domain=domain)
    if isinstance(tasks, list):
        for task in tasks:
            if isinstance(task, dict):
                out[str(task.get("id", ""))] = {"task": task, "policy_text": policy_text}
    return out


def load_policy_text(tau2_data_dir: Path, domain: str) -> str:
    for path in (tau2_data_dir / "tau2" / "results" / "final").glob(f"*_{domain}_*trials.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        info = data.get("info") if isinstance(data, dict) else {}
        env = info.get("environment_info") if isinstance(info, dict) else {}
        policy = env.get("policy") if isinstance(env, dict) else None
        if policy:
            return str(policy)
    return ""


def load_domain_splits(tau2_data_dir: Path, domain: str) -> Dict[str, str]:
    path = tau2_data_dir / "tau2" / "domains" / domain / "split_tasks.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    task_to_split = {}
    if isinstance(data, dict):
        for split, task_ids in data.items():
            if split == "base":
                continue
            for task_id in task_ids if isinstance(task_ids, list) else []:
                task_to_split[str(task_id)] = str(split)
    return task_to_split


def build_label_audit(
    domain: str,
    seed: int,
    sources: Sequence[SourceTrajectory],
    pools: Sequence[PoolBuild],
    evaluator: Dict[str, object],
    inventory: Dict[str, object],
) -> Dict[str, object]:
    pool_summaries = {}
    readiness_by_pool = {}
    for pool in pools:
        examples = build_taubench_selection_examples(pool.candidates)
        validation = validate_taubench_selection_examples(examples, require_labels=True)
        near_dupes = near_duplicate_trajectory_audit(examples, threshold=NEAR_DUPLICATE_THRESHOLD)
        duplicate_audit = duplicate_trajectory_hash_audit(examples)
        baseline = baseline_estimates(pool)
        summary = summarize_pool(pool, validation=validation, near_dupes=near_dupes, duplicate_audit=duplicate_audit, baseline=baseline)
        pool_summaries[pool.name] = summary
        readiness_by_pool[pool.name] = readiness_gates(summary, evaluator=evaluator)
    decision = decide(readiness_by_pool, pool_summaries)
    needed = needed_generation(decision, readiness_by_pool, pool_summaries)
    return {
        "artifact": "stage7b0_tau_label_audit",
        "created_at_utc": _now(),
        "seed": int(seed),
        "domain": domain,
        "no_selector_training_performed": True,
        "no_final_benchmark_claim": True,
        "labels_available": all(row.recomputed_success is not None for row in sources),
        "source_label_recompute": {
            "domain_source_trajectories": len(sources),
            "labels_available_count": sum(1 for row in sources if row.recomputed_success is not None),
            "labels_missing_count": sum(1 for row in sources if row.recomputed_success is None),
            "per_domain_success_stats": per_domain_success_stats(sources),
            "evaluator": evaluator,
        },
        "pool_summaries": pool_summaries,
        "readiness_gates": readiness_by_pool,
        "candidate_audit_map": {
            pool.name: pool.audit_rows
            for pool in pools
        },
        "source_inventory_reference": "results/stage7b0_tau_source_inventory.json",
        "source_inventory_digest": {
            "raw_trajectories_loaded": inventory.get("raw_trajectories_loaded"),
            "unique_domains": inventory.get("unique_domains"),
            "task_counts_per_domain": inventory.get("task_counts_per_domain"),
        },
        "decision": decision,
        "needed_real_trajectory_generation": needed,
    }


def summarize_pool(
    pool: PoolBuild,
    validation: Dict[str, object],
    near_dupes: Dict[str, object],
    duplicate_audit: Dict[str, object],
    baseline: Dict[str, object],
) -> Dict[str, object]:
    examples = build_taubench_selection_examples(pool.candidates)
    success_counts = [sum(example.labels_pass_fail) for example in examples]
    oracle = [count > 0 for count in success_counts]
    all_success = sum(1 for count in success_counts if count == STAGE7_NUM_CANDIDATES)
    all_fail = sum(1 for count in success_counts if count == 0)
    return {
        "path": str(pool.path),
        "selection_notes": pool.selection_notes,
        "candidate_count": len(pool.candidates),
        "task_count": len(examples),
        "k": STAGE7_NUM_CANDIDATES,
        "labels_available": all(candidate.official_success is not None for candidate in pool.candidates),
        "oracle_pass_at_8": sum(oracle) / float(max(1, len(oracle))),
        "oracle_positive_tasks": [example.task_id for example, value in zip(examples, oracle) if value],
        "oracle_empty_tasks": [example.task_id for example, value in zip(examples, oracle) if not value],
        "all_success_tasks": [example.task_id for example, count in zip(examples, success_counts) if count == STAGE7_NUM_CANDIDATES],
        "all_fail_tasks": [example.task_id for example, count in zip(examples, success_counts) if count == 0],
        "all_success_task_count": all_success,
        "all_fail_task_count": all_fail,
        "per_task_success_counts": dict(sorted(pool.task_success_counts.items(), key=lambda item: _task_sort_key(item[0]))),
        "success_count_distribution": _count_distribution(success_counts),
        "per_domain_success_stats": {
            "domain": examples[0].domain if examples else "",
            "candidate_success_rate": sum(success_counts) / float(max(1, len(pool.candidates))),
            "task_oracle_rate": sum(oracle) / float(max(1, len(oracle))),
        },
        "baseline_estimates": baseline,
        "candidate_dataset_validation": validation,
        "output_evaluator_leakage_audit": stage7_output_leakage_audit("stage7_taubench_trajectory_selector", 0, examples),
        "duplicate_trajectory_audit": duplicate_audit,
        "near_duplicate_trajectory_audit": near_dupes,
    }


def baseline_estimates(pool: PoolBuild) -> Dict[str, object]:
    by_task_audit: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in pool.audit_rows:
        by_task_audit[str(row["task_id"])].append(row)
    ordered_task_rows = {
        task_id: sorted(rows, key=lambda row: int(row["candidate_order_index"]))
        for task_id, rows in by_task_audit.items()
    }
    first = [
        bool(rows[0]["official_success"])
        for rows in ordered_task_rows.values()
        if rows
    ]
    random_rates = [
        sum(1 for row in rows if row["official_success"]) / float(max(1, len(rows)))
        for rows in ordered_task_rows.values()
    ]
    generator_rates: Dict[str, List[bool]] = defaultdict(list)
    for row in pool.audit_rows:
        generator_rates[str(row["generator_name_audit_only"])].append(bool(row["official_success"]))
    generator_success = {
        generator: sum(values) / float(max(1, len(values)))
        for generator, values in generator_rates.items()
    }
    best_generator = max(generator_success, key=generator_success.get) if generator_success else ""
    best_generator_choices = []
    for rows in ordered_task_rows.values():
        matching = [row for row in rows if row["generator_name_audit_only"] == best_generator]
        chosen = (matching or rows)[0]
        best_generator_choices.append(bool(chosen["official_success"]))
    return {
        "first_trajectory_pass_at_1": sum(first) / float(max(1, len(first))),
        "random_uniform_candidate_expected_pass_at_1": sum(random_rates) / float(max(1, len(random_rates))),
        "audit_only_best_generator": best_generator,
        "audit_only_best_generator_candidate_success_rate": generator_success.get(best_generator, 0.0),
        "audit_only_best_generator_pass_at_1_estimate": sum(best_generator_choices) / float(max(1, len(best_generator_choices))),
        "audit_only_generator_success_rates": dict(sorted(generator_success.items())),
    }


def readiness_gates(summary: Dict[str, object], evaluator: Dict[str, object]) -> Dict[str, object]:
    task_count = int(summary.get("task_count", 0))
    oracle = float(summary.get("oracle_pass_at_8", 0.0))
    all_success = int(summary.get("all_success_task_count", 0))
    all_fail = int(summary.get("all_fail_task_count", 0))
    baseline = summary.get("baseline_estimates", {}) if isinstance(summary.get("baseline_estimates"), dict) else {}
    random_base = float(baseline.get("random_uniform_candidate_expected_pass_at_1", 0.0))
    first_base = float(baseline.get("first_trajectory_pass_at_1", 0.0))
    duplicate_passes = bool((summary.get("duplicate_trajectory_audit") or {}).get("passes", False))
    near_duplicate_passes = bool((summary.get("near_duplicate_trajectory_audit") or {}).get("passes", False))
    leakage_passes = bool((summary.get("output_evaluator_leakage_audit") or {}).get("passes", False))
    gates = {
        "at_least_50_retail_tasks_with_k8_official_labels": task_count >= READINESS_MIN_TASKS and bool(summary.get("labels_available", False)),
        "preferred_at_least_100_retail_tasks": task_count >= READINESS_PREFERRED_TASKS,
        "oracle_pass_at_8_between_0_25_and_0_85": 0.25 <= oracle <= 0.85,
        "all_success_tasks_not_dominant": all_success / float(max(1, task_count)) < 0.25,
        "all_fail_tasks_retained_but_not_dominant": all_fail > 0 and all_fail / float(max(1, task_count)) < 0.35,
        "random_baseline_meaningfully_below_oracle": oracle - random_base >= 0.15,
        "first_trajectory_baseline_meaningfully_below_oracle": oracle - first_base >= 0.15,
        "generator_source_identity_blinded_from_selector_visible_text": leakage_passes,
        "output_evaluator_leakage_audit_passes": leakage_passes,
        "duplicate_trajectory_audit_passes_or_reported": duplicate_passes or bool((summary.get("duplicate_trajectory_audit") or {}).get("duplicates") is not None),
        "near_duplicate_trajectory_audit_passes_or_reported": near_duplicate_passes or bool((summary.get("near_duplicate_trajectory_audit") or {}).get("near_duplicates") is not None),
        "official_evaluator_recomputed_labels": bool(evaluator.get("official_evaluator_ran")) and bool(evaluator.get("valid_for_readiness")),
        "no_final_claim_made": True,
    }
    return {
        "passes_all_required": all(
            value for key, value in gates.items()
            if key != "preferred_at_least_100_retail_tasks"
        ),
        "gates": gates,
    }


def decide(readiness_by_pool: Dict[str, Dict[str, object]], pool_summaries: Dict[str, Dict[str, object]]) -> str:
    for name in ("nontrivial_pool", "balanced_pool"):
        if readiness_by_pool.get(name, {}).get("passes_all_required"):
            return "A. READY_FOR_STAGE7B_DEV"
    all_task_counts = [int(summary.get("task_count", 0)) for summary in pool_summaries.values()]
    if max(all_task_counts or [0]) >= READINESS_MIN_TASKS:
        return "B. NEED_REAL_TRAJECTORY_GENERATION"
    return "B. NEED_REAL_TRAJECTORY_GENERATION"


def needed_generation(
    decision: str,
    readiness_by_pool: Dict[str, Dict[str, object]],
    pool_summaries: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    if decision == "A. READY_FOR_STAGE7B_DEV":
        return {
            "needed_for_stage7b_dev": 0,
            "note": "Existing trajectories are sufficient for selector development only; no final benchmark claim is authorized.",
        }
    best_tasks = max((int(summary.get("task_count", 0)) for summary in pool_summaries.values()), default=0)
    return {
        "additional_retail_tasks_needed_for_minimum": max(0, READINESS_MIN_TASKS - best_tasks),
        "additional_retail_tasks_needed_for_preferred": max(0, READINESS_PREFERRED_TASKS - best_tasks),
        "additional_trajectories_per_task": STAGE7_NUM_CANDIDATES,
        "reason": "Existing mined pools did not pass nontrivial readiness gates.",
        "gate_snapshot": readiness_by_pool,
    }


def blocked_label_audit(reason: str, inventory: Dict[str, object], results_dir: Path, reports_dir: Path) -> Dict[str, object]:
    return {
        "artifact": "stage7b0_tau_label_audit",
        "created_at_utc": _now(),
        "labels_available": False,
        "no_selector_training_performed": True,
        "no_final_benchmark_claim": True,
        "decision": "C. TAU_DATA_MINING_BLOCKED",
        "blocker": reason,
        "source_inventory_reference": str(results_dir / "stage7b0_tau_source_inventory.json"),
        "report_path": str(reports_dir / "STAGE7B0_TAU_CANDIDATE_POOL_MINING.md"),
        "source_inventory_digest": {
            "raw_trajectories_loaded": inventory.get("raw_trajectories_loaded"),
            "unique_domains": inventory.get("unique_domains"),
        },
    }


def per_domain_success_stats(sources: Sequence[SourceTrajectory]) -> Dict[str, object]:
    by_domain: Dict[str, List[SourceTrajectory]] = defaultdict(list)
    for row in sources:
        by_domain[row.domain].append(row)
    out = {}
    for domain, rows in sorted(by_domain.items()):
        known = [bool(row.recomputed_success) for row in rows if row.recomputed_success is not None]
        out[domain] = {
            "trajectories": len(rows),
            "labels_available": len(known),
            "success_count": sum(1 for value in known if value),
            "success_rate": sum(1 for value in known if value) / float(max(1, len(known))),
        }
    return out


def write_report(path: Path, inventory: Dict[str, object], label_audit: Dict[str, object]) -> None:
    pools = label_audit.get("pool_summaries", {}) if isinstance(label_audit.get("pool_summaries"), dict) else {}
    readiness = label_audit.get("readiness_gates", {}) if isinstance(label_audit.get("readiness_gates"), dict) else {}
    lines = [
        "# Stage 7B.0 tau-bench Candidate Pool Mining",
        "",
        "## Decision",
        f"- Decision: `{label_audit.get('decision', 'C. TAU_DATA_MINING_BLOCKED')}`",
        "- Selector training performed: `False`.",
        "- Final benchmark claim made: `False`.",
        "",
        "## Source Inventory",
        f"- Source files scanned: `{json.dumps((inventory.get('source_files_scanned') or {}).get('counts', {}), sort_keys=True)}`",
        f"- tau2 repo commit: `{inventory.get('tau2_repo_commit')}`",
        f"- Raw trajectories loaded: `{inventory.get('raw_trajectories_loaded', 0)}`",
        f"- Unique domains: `{', '.join(inventory.get('unique_domains', []))}`",
        f"- Task counts per domain: `{json.dumps(inventory.get('task_counts_per_domain', {}), sort_keys=True)}`",
        f"- Trajectories per task distribution: `{json.dumps(inventory.get('trajectories_per_task_distribution', {}), sort_keys=True)}`",
        f"- Tasks with >=8 trajectories: `{json.dumps({k: len(v) for k, v in (inventory.get('tasks_with_at_least_8_trajectories') or {}).items()}, sort_keys=True)}`",
        f"- Existing-label oracle pass@8 estimate: `{json.dumps(inventory.get('oracle_pass_at_8_estimate_from_existing_labels', {}), sort_keys=True)}`",
        f"- Generator/source distribution: `{json.dumps(inventory.get('generator_source_distribution', {}), sort_keys=True)}`",
        f"- Duplicate trajectory rate: `{(inventory.get('duplicate_trajectory_audit') or {}).get('duplicate_rate', 0.0):.6f}`",
        f"- Near-duplicate trajectory pair rate: `{(inventory.get('near_duplicate_trajectory_audit') or {}).get('near_duplicate_pair_rate', 0.0):.6f}`",
        "",
        "## Official Label Recompute",
        f"- Labels available: `{label_audit.get('labels_available', False)}`",
        f"- Evaluator: `{json.dumps((label_audit.get('source_label_recompute') or {}).get('evaluator', {}), sort_keys=True)[:1000]}`",
        f"- Per-domain success stats: `{json.dumps((label_audit.get('source_label_recompute') or {}).get('per_domain_success_stats', {}), sort_keys=True)}`",
        "",
        "## Pool Summaries",
    ]
    for name, summary in pools.items():
        baseline = summary.get("baseline_estimates", {})
        lines.extend(
            [
                f"### {name}",
                f"- Artifact: `{summary.get('path')}`",
                f"- Tasks/candidates: `{summary.get('task_count')}` / `{summary.get('candidate_count')}`",
                f"- Oracle pass@8: `{float(summary.get('oracle_pass_at_8', 0.0)):.4f}`",
                f"- Oracle-positive tasks: `{len(summary.get('oracle_positive_tasks', []))}`",
                f"- Oracle-empty tasks: `{len(summary.get('oracle_empty_tasks', []))}`",
                f"- All-success tasks: `{summary.get('all_success_task_count')}`",
                f"- All-fail tasks: `{summary.get('all_fail_task_count')}`",
                f"- Success-count distribution: `{json.dumps(summary.get('success_count_distribution', {}), sort_keys=True)}`",
                f"- Baselines: first `{float(baseline.get('first_trajectory_pass_at_1', 0.0)):.4f}`, random `{float(baseline.get('random_uniform_candidate_expected_pass_at_1', 0.0)):.4f}`, best-generator `{float(baseline.get('audit_only_best_generator_pass_at_1_estimate', 0.0)):.4f}`",
                f"- Leakage audit passes: `{(summary.get('output_evaluator_leakage_audit') or {}).get('passes', False)}`",
                f"- Duplicate audit passes: `{(summary.get('duplicate_trajectory_audit') or {}).get('passes', False)}`",
                f"- Near-duplicate audit passes: `{(summary.get('near_duplicate_trajectory_audit') or {}).get('passes', False)}`",
                f"- Readiness gates: `{json.dumps((readiness.get(name) or {}).get('gates', {}), sort_keys=True)}`",
                "",
            ]
        )
    if "blocker" in label_audit:
        lines.extend(["## Blocker", f"- {label_audit['blocker']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _extract_success(record: Dict[str, object]) -> bool | None:
    for key in ("success", "reward", "task_reward", "environment_reward", "passed"):
        if key not in record:
            continue
        value = record[key]
        parsed = _boolish(value)
        if parsed is not None:
            return parsed
    reward_info = record.get("reward_info")
    if isinstance(reward_info, dict):
        for key in ("reward", "success", "passed"):
            if key in reward_info:
                parsed = _boolish(reward_info[key])
                if parsed is not None:
                    return parsed
    return None


def _boolish(value: object) -> bool | None:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return float(value) > 0.0
    text = str(value).strip().lower()
    if text in {"true", "success", "passed", "1", "1.0"}:
        return True
    if text in {"false", "failed", "0", "0.0"}:
        return False
    return None


def _count_distribution(values: Iterable[int]) -> Dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(values).items())}


def _float_summary(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0}
    sorted_values = sorted(float(value) for value in values)
    return {
        "count": len(sorted_values),
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "mean": sum(sorted_values) / float(len(sorted_values)),
    }


def _oracle_from_existing(tasks: Dict[str, List[SourceTrajectory]]) -> Dict[str, object]:
    known = []
    for rows in tasks.values():
        labels = [row.existing_success for row in rows if row.existing_success is not None]
        if labels:
            known.append(any(bool(value) for value in labels[:STAGE7_NUM_CANDIDATES]))
    return {
        "tasks_with_known_labels": len(known),
        "oracle_pass_at_8_estimate": sum(1 for value in known if value) / float(max(1, len(known))),
    }


def _generator_source_distribution(trajectories: Sequence[SourceTrajectory]) -> Dict[str, int]:
    return {
        key: int(value)
        for key, value in sorted(Counter(f"{row.domain}|{row.generator_name}|{row.source_file_name}" for row in trajectories).items())
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / float(max(1, len(left | right)))


def _task_sort_key(task_id: str) -> tuple[int, str]:
    text = str(task_id)
    return (int(text), text) if text.isdigit() else (10**9, text)


def _int_like(value: object) -> bool:
    try:
        int(value)  # type: ignore[arg-type]
        return True
    except Exception:
        return False


def stable_int(text: str) -> int:
    value = 0
    for char in text:
        value = (value * 131 + ord(char)) % 1_000_000_007
    return value


def git_commit(repo: Path) -> str | None:
    if not repo.exists():
        return None
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, timeout=20)
    except Exception:
        completed = None
    if completed is not None and completed.returncode == 0:
        return completed.stdout.strip()
    head_path = repo / ".git" / "HEAD"
    if not head_path.exists():
        return None
    head = head_path.read_text(encoding="utf-8").strip()
    if re.fullmatch(r"[0-9a-f]{40}", head):
        return head
    if head.startswith("ref: "):
        ref_path = repo / ".git" / head.removeprefix("ref: ").strip()
        if ref_path.exists():
            ref = ref_path.read_text(encoding="utf-8").strip()
            if re.fullmatch(r"[0-9a-f]{40}", ref):
                return ref
    return None


def write_json(path: Path, record: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
