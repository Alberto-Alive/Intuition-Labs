import json

from src.experiments.run_stage6b1_candidate_generation import BenchmarkTask
from src.experiments.run_stage6d3_candidate_source_expansion import (
    build_generation_plan,
    build_pre_harness_readiness,
    build_unlabeled_pool,
    group_accepted_attempts,
    scan_generated_sources,
)


def _diff(instance_id: str, slot: int, suffix: str = "") -> str:
    return (
        f"diff --git a/pkg/{instance_id}_{slot}.py b/pkg/{instance_id}_{slot}.py\n"
        f"--- a/pkg/{instance_id}_{slot}.py\n"
        f"+++ b/pkg/{instance_id}_{slot}.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        f"+new_{slot}{suffix}\n"
    )


def _task(instance_id: str) -> BenchmarkTask:
    return BenchmarkTask(
        instance_id=instance_id,
        record={
            "instance_id": instance_id,
            "repo": "repo/proj",
            "base_commit": "abc123",
            "version": "1.0",
            "problem_statement": "Fix generated behavior",
            "hints_text": "",
            "FAIL_TO_PASS": "[]",
            "PASS_TO_PASS": "[]",
            "patch": _diff(instance_id, 99, "reference"),
        },
    )


def test_stage6d3_scans_trajectory_blocks_and_filters_invalid_duplicate_exact_reference(tmp_path) -> None:
    instance_id = "repo__proj-1"
    path = tmp_path / "source.ndjson"
    blocks = "\n\n".join(f"```diff\n{_diff(instance_id, slot)}```" for slot in range(8))
    rows = [
        {"instance_id": instance_id, "messages": [{"role": "assistant", "content": blocks}], "model_name_or_path": "unit-model"},
        {"instance_id": instance_id, "model_patch": _diff(instance_id, 0), "model_name_or_path": "unit-model"},
        {"instance_id": instance_id, "model_patch": "not a diff", "model_name_or_path": "unit-model"},
        {"instance_id": instance_id, "model_patch": _task(instance_id).reference_patch, "model_name_or_path": "unit-model"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    attempts, source_rows = scan_generated_sources([path], {instance_id: _task(instance_id)})
    statuses = [attempt.terminal_status for attempt in attempts]
    grouped = group_accepted_attempts(attempts)
    pool_all, selection = build_unlabeled_pool(grouped, {instance_id: _task(instance_id)}, seed=123, strict=False)

    assert statuses.count("accepted") == 8
    assert "duplicate_patch" in statuses
    assert "invalid_diff" in statuses
    assert "exact_reference_hash" in statuses
    assert source_rows[0]["raw_records_loaded"] == 4
    assert source_rows[0]["accepted_candidates"] == 8
    assert len(pool_all) == 1
    assert len(pool_all[0].candidates) == 8
    assert pool_all[0].labels_pass_fail == ()
    assert {candidate.candidate_source_agent for candidate in pool_all[0].candidates} == {"source_blinded_generated"}
    assert selection["selected_task_count"] == 1


def test_stage6d3_readiness_blocks_harness_and_requests_self_generation_for_small_pool() -> None:
    instance_id = "repo__proj-1"
    task_records = {instance_id: _task(instance_id), "repo__proj-2": _task("repo__proj-2")}
    path_attempts = []
    for slot in range(8):
        path_attempts.append(
            {
                "instance_id": instance_id,
                "model_patch": _diff(instance_id, slot),
                "model_name_or_path": "unit-model",
            }
        )
    source = group_accepted_attempts(
        scan_generated_sources_from_records(path_attempts, task_records)
    )
    pool_all, _selection = build_unlabeled_pool(source, task_records, seed=123, strict=False)
    pool_strict, _strict_selection = build_unlabeled_pool(source, task_records, seed=123, strict=True)
    plan = build_generation_plan(task_records, source, pool_strict, minimum_k8_tasks=50, preferred_k8_tasks=100, max_generation_plan_tasks=5)
    readiness = build_pre_harness_readiness(pool_all, pool_strict, [], {"generator_source_distribution": {"unit": 8}}, plan, minimum_k8_tasks=50, seed=123)

    assert readiness["decision"] == "NEED_SELF_GENERATION"
    assert not readiness["official_harness_should_run_now"]
    assert plan["additional_k8_tasks_needed_for_minimum"] == 49


def scan_generated_sources_from_records(rows, task_records):
    path = None
    attempts = []
    import tempfile
    from pathlib import Path
    from src.experiments.run_stage6d3_candidate_source_expansion import scan_generated_sources

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rows.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        attempts, _source_rows = scan_generated_sources([path], task_records)
    return attempts
