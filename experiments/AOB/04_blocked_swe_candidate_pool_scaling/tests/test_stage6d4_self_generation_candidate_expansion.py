from pathlib import Path

from src.experiments.run_stage6b1_candidate_generation import BenchmarkTask
from src.experiments.run_stage6d3_candidate_source_expansion import build_unlabeled_pool, group_accepted_attempts
from src.experiments.run_stage6d4_self_generation_candidate_expansion import (
    PROMPT_VARIANTS,
    build_readiness_audit,
    generation_attempt_to_source_attempt,
    run_dry_run_backend,
    stable_patch_hash,
    validate_generated_patch,
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
            "repo": "django/django",
            "base_commit": "abc123",
            "version": "1.0",
            "problem_statement": "Fix generated behavior",
            "hints_text": "Use a small local change.",
            "FAIL_TO_PASS": "[]",
            "PASS_TO_PASS": "[]",
            "patch": _diff(instance_id, 99, "reference"),
        },
    )


def test_stage6d4_dry_run_backend_produces_blinded_k8_pool(tmp_path: Path) -> None:
    instance_id = "repo__proj-1"
    task_records = {instance_id: _task(instance_id)}
    rows = run_dry_run_backend(
        selected_tasks=[instance_id],
        task_records=task_records,
        existing_hashes={},
        trajectory_root=tmp_path / "trajectories",
        temperatures=(0.0,),
        generation_seeds=(123,),
        prompt_variants=PROMPT_VARIANTS,
        max_attempts_per_task=8,
    )
    source_attempts = [generation_attempt_to_source_attempt(row) for row in rows if row.accepted_for_pool_all]
    pool, _selection = build_unlabeled_pool(group_accepted_attempts(source_attempts), task_records, seed=123, strict=True)

    assert len(rows) == 8
    assert all(row.terminal_status == "accepted" for row in rows)
    assert all(Path(row.raw_trajectory_path).exists() for row in rows)
    assert len(pool) == 1
    assert len(pool[0].candidates) == 8
    assert {candidate.candidate_source_agent for candidate in pool[0].candidates} == {"source_blinded_generated"}
    assert pool[0].labels_pass_fail == ()


def test_stage6d4_validate_generated_patch_rejects_duplicate_and_exact_reference(tmp_path: Path) -> None:
    instance_id = "repo__proj-1"
    task = _task(instance_id)
    duplicate = _diff(instance_id, 0)
    duplicate_hash = stable_patch_hash(duplicate)

    dup_row = validate_generated_patch(
        instance_id,
        duplicate,
        task,
        existing_hashes={duplicate_hash},
        backend="unit",
        generator="unit",
        model_name="unit",
        prompt_variant="minimal_fix",
        seed=1,
        temperature=0.0,
        attempt_index=0,
        raw_trajectory_path=str(tmp_path / "dup.json"),
    )
    ref_row = validate_generated_patch(
        instance_id,
        task.reference_patch,
        task,
        existing_hashes=set(),
        backend="unit",
        generator="unit",
        model_name="unit",
        prompt_variant="minimal_fix",
        seed=1,
        temperature=0.0,
        attempt_index=1,
        raw_trajectory_path=str(tmp_path / "ref.json"),
    )
    bad_row = validate_generated_patch(
        instance_id,
        "not a diff",
        task,
        existing_hashes=set(),
        backend="unit",
        generator="unit",
        model_name="unit",
        prompt_variant="minimal_fix",
        seed=1,
        temperature=0.0,
        attempt_index=2,
        raw_trajectory_path=str(tmp_path / "bad.json"),
    )

    assert dup_row.terminal_status == "duplicate_patch"
    assert ref_row.terminal_status == "exact_reference_hash"
    assert bad_row.terminal_status == "invalid_diff"


def test_stage6d4_readiness_blocks_without_working_backend() -> None:
    audit = {"accepted_candidates_total": 0, "tasks_attempted": 0, "exact_reference_hits": 0}
    readiness = build_readiness_audit(audit, pool_all=[], pool_strict=[], generation_attempts=[], strict_target=50, preferred_target=100)

    assert readiness["decision"] == "SELF_GENERATION_BLOCKED"
    assert not readiness["official_harness_should_run_now"]
    assert readiness["remaining_tasks_needed_for_50_strict_k8"] == 50
