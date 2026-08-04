import json
from pathlib import Path

from src.experiments.run_stage6b1b_candidate_supply_expansion import (
    TASK_SKIPPED_STATUSES,
    run_stage6b1b_candidate_supply_expansion,
)


def _diff(instance_id: str, candidate_index: int, file_name: str | None = None) -> str:
    path = file_name or f"pkg/{instance_id}_{candidate_index}.py"
    return (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -1,2 +1,3 @@\n"
        " def value():\n"
        f"-    return 'old-{instance_id}'\n"
        f"+    marker = 'candidate-{candidate_index}'\n"
        "+    return marker\n"
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows), encoding="utf-8")


def test_stage6b1b_expands_distinct_trajectory_diff_blocks_and_reports_coverage(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.jsonl"
    source_root = tmp_path / "sources"
    source_path = source_root / "nested" / "predictions.jsonl"
    benchmark_rows = []
    source_rows = []
    for task_index in range(25):
        instance_id = f"repo__proj-{task_index}"
        benchmark_rows.append(
            {
                "instance_id": instance_id,
                "repo": "repo/proj",
                "base_commit": "abc123",
                "problem_statement": f"Fix task {task_index}",
                "hints_text": "",
                "FAIL_TO_PASS": "[]",
                "PASS_TO_PASS": "[]",
                "patch": _diff(instance_id, 99, file_name=f"reference/{instance_id}.py"),
            }
        )
        if task_index < 20:
            blocks = "\n".join(f"```diff\n{_diff(instance_id, candidate_index)}```" for candidate_index in range(8))
            source_rows.append(
                {
                    "instance_id": instance_id,
                    "model_name_or_path": "unit-model",
                    "generator_name": "unit-generator",
                    "trajectory": [{"role": "assistant", "content": blocks}],
                }
            )
    _write_jsonl(benchmark_path, benchmark_rows)
    _write_jsonl(source_path, source_rows)

    result = run_stage6b1b_candidate_supply_expansion(
        dataset_name="local/unit",
        split="test",
        benchmark_jsonl=benchmark_path,
        task_ids=(),
        task_ids_file=None,
        n_tasks=25,
        k=8,
        seed=123,
        output_path=tmp_path / "results" / "stage6b1b_generated_candidates.jsonl",
        attempts_path=tmp_path / "results" / "stage6b1b_generation_attempts.jsonl",
        audit_path=tmp_path / "results" / "stage6b1b_generation_audit.json",
        report_path=tmp_path / "reports" / "STAGE6B1B_CANDIDATE_SUPPLY_EXPANSION.md",
        trajectory_root=tmp_path / "results" / "trajectories",
        candidate_patch_root=tmp_path / "results" / "patches",
        prompt_root=tmp_path / "results" / "prompts",
        source_jsonl=(),
        source_json=(),
        source_roots=(str(source_root),),
        previous_stage6b1_jsonl=(),
        include_default_hf_public=False,
        enable_mini_swe_agent=False,
        mini_swe_agent_command_templates=(),
        mini_swe_agent_configs=(),
        api_generator_command_templates=(),
        api_generator_models=(),
        local_generator_command_templates=(),
        local_generator_models=(),
        generation_seeds=(),
        temperatures=(),
        prompt_variants=(),
        max_attempts_per_task=32,
        generator_timeout_seconds=30,
        apply_check=False,
        require_apply_check=False,
        repo_cache_root=None,
        apply_check_timeout_seconds=30,
        duplicate_rate_threshold=0.25,
        near_reference_threshold=1.01,
    )

    audit = result["audit"]
    statuses = {row["terminal_status"] for row in result["attempts"]}
    coverage = audit["source_coverage_report"]
    assert audit["quality_gates"]["candidate_supply_expansion_quality_gates_pass"] is True
    assert audit["tasks_attempted"] == 25
    assert audit["tasks_with_at_least_8_candidates"] == 20
    assert audit["accepted_candidate_rows"] == 160
    assert "task_skipped_no_matching_instance_id" in statuses
    assert "task_skipped" not in statuses
    assert set(audit["failure_taxonomy"]["task_skipped_statuses"]) == TASK_SKIPPED_STATUSES
    assert coverage["source_files_loaded"] == 1
    assert coverage["total_raw_records_loaded"] == 20
    assert coverage["unique_instance_ids_in_sources"] == 20
    assert coverage["overlap_with_selected_benchmark_tasks"]["count"] == 20
    assert coverage["candidates_per_overlapped_instance_before_filtering"]["repo__proj-0"] == 8
    assert coverage["candidates_per_overlapped_instance_after_filtering"]["repo__proj-0"] == 8
    assert coverage["top_source_files_by_accepted_candidates"][0]["accepted_candidates"] == 160

