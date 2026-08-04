import json
import shutil
from pathlib import Path

import pytest

from src.experiments.run_stage6b1a_generator_reliability_ramp import (
    run_stage6b1a_generator_reliability_ramp,
    validate_unified_diff,
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


def _run_1a(tmp_path: Path, benchmark_path: Path, prediction_path: Path, **kwargs):
    defaults = {
        "dataset_name": "local/unit",
        "split": "test",
        "benchmark_jsonl": benchmark_path,
        "task_ids": (),
        "task_ids_file": None,
        "n_tasks": 25,
        "k": 8,
        "seed": 123,
        "output_path": tmp_path / "results" / "stage6b1a_generated_candidates.jsonl",
        "attempts_path": tmp_path / "results" / "stage6b1a_generation_attempts.jsonl",
        "audit_path": tmp_path / "results" / "stage6b1a_generation_audit.json",
        "report_path": tmp_path / "reports" / "STAGE6B1A_GENERATOR_RELIABILITY_RAMP.md",
        "trajectory_root": tmp_path / "results" / "trajectories",
        "candidate_patch_root": tmp_path / "results" / "patches",
        "prompt_root": tmp_path / "results" / "prompts",
        "source_jsonl": (str(prediction_path),),
        "source_json": (),
        "source_roots": (),
        "previous_stage6b1_jsonl": (),
        "include_default_hf_public": False,
        "enable_mini_swe_agent": False,
        "mini_swe_agent_command_templates": (),
        "mini_swe_agent_configs": (),
        "api_generator_command_templates": (),
        "api_generator_models": (),
        "local_generator_command_templates": (),
        "local_generator_models": (),
        "generation_seeds": (),
        "temperatures": (),
        "prompt_variants": (),
        "max_attempts_per_task": 32,
        "generator_timeout_seconds": 30,
        "apply_check": False,
        "require_apply_check": False,
        "repo_cache_root": None,
        "apply_check_timeout_seconds": 30,
        "duplicate_rate_threshold": 0.25,
        "near_reference_threshold": 1.01,
    }
    defaults.update(kwargs)
    return run_stage6b1a_generator_reliability_ramp(**defaults)


def test_stage6b1a_overgenerates_and_records_attempt_taxonomy(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    benchmark_rows = []
    prediction_rows = []
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
        prediction_rows.append(
            {
                "instance_id": instance_id,
                "model_name_or_path": "unit-model",
                "generator_name": "unit-generator",
                "seed": 1000,
                "temperature": 0.0,
                "config_name": "raw",
                "model_patch": _diff(instance_id, 0),
            }
        )
        prediction_rows.append(
            {
                "instance_id": instance_id,
                "model_name_or_path": "unit-model",
                "generator_name": "unit-generator",
                "seed": 1001,
                "temperature": 0.0,
                "config_name": "duplicate_fenced",
                "output_patch": f"Here is the patch:\n```diff\n{_diff(instance_id, 0)}```\nDone.",
            }
        )
        prediction_rows.append(
            {
                "instance_id": instance_id,
                "model_name_or_path": "unit-model",
                "generator_name": "unit-generator",
                "config_name": "invalid_prose",
                "model_patch": "I would update the parser but this is not a diff.",
            }
        )
        prediction_rows.append(
            {
                "instance_id": instance_id,
                "model_name_or_path": "unit-model",
                "generator_name": "unit-generator",
                "config_name": "messages_no_patch",
                "messages": [{"role": "assistant", "content": "No patch found."}],
            }
        )
        target_count = 8 if task_index < 20 else 4
        for candidate_index in range(1, target_count):
            if candidate_index == 1:
                prediction_rows.append(
                    {
                        "instance_id": instance_id,
                        "model_name_or_path": "unit-model",
                        "generator_name": "unit-generator",
                        "seed": 2000 + candidate_index,
                        "temperature": 0.2,
                        "config_name": "messages",
                        "messages": [{"role": "assistant", "content": f"```diff\n{_diff(instance_id, candidate_index)}```"}],
                    }
                )
            elif candidate_index == 2:
                prediction_rows.append(
                    {
                        "instance_id": instance_id,
                        "model_name_or_path": "unit-model",
                        "generator_name": "unit-generator",
                        "seed": 2000 + candidate_index,
                        "temperature": 0.7,
                        "config_name": "nested_output",
                        "prediction": {"output_patch": _diff(instance_id, candidate_index)},
                    }
                )
            else:
                prediction_rows.append(
                    {
                        "instance_id": instance_id,
                        "model_name_or_path": "unit-model",
                        "generator_name": f"unit-generator-{candidate_index % 3}",
                        "seed": 2000 + candidate_index,
                        "temperature": 1.0,
                        "config_name": "raw",
                        "model_patch": _diff(instance_id, candidate_index),
                    }
                )
    _write_jsonl(benchmark_path, benchmark_rows)
    _write_jsonl(prediction_path, prediction_rows)

    result = _run_1a(tmp_path, benchmark_path, prediction_path)

    audit = result["audit"]
    attempts = [
        json.loads(line)
        for line in (tmp_path / "results" / "stage6b1a_generation_attempts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output_rows = [
        json.loads(line)
        for line in (tmp_path / "results" / "stage6b1a_generated_candidates.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    statuses = {row["terminal_status"] for row in attempts}
    assert audit["quality_gates"]["generator_reliability_quality_gates_pass"] is True
    assert audit["tasks_attempted"] == 25
    assert audit["tasks_with_at_least_8_candidates"] == 20
    assert audit["tasks_with_at_least_4_candidates"] == 25
    assert audit["invalid_diff_audit"]["invalid_diff_rate"] < 0.50
    assert {"accepted", "duplicate_patch", "invalid_diff", "no_patch_found"}.issubset(statuses)
    assert len(output_rows) == 20 * 8 + 5 * 4
    assert all(row["source_visible_to_selector"] is False for row in output_rows)
    assert all(Path(row["trajectory_path"]).exists() for row in output_rows)
    assert all(row["generator_name"] not in row["model_patch"] for row in output_rows)


def test_stage6b1a_rejects_reference_hash_and_previous_stage6b1_ingest(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.jsonl"
    previous_path = tmp_path / "stage6b1_generated_candidates.jsonl"
    prediction_path = tmp_path / "empty.jsonl"
    instance_id = "repo__proj-reference"
    reference = _diff(instance_id, 99)
    _write_jsonl(
        benchmark_path,
        [
            {
                "instance_id": instance_id,
                "repo": "repo/proj",
                "base_commit": "abc123",
                "problem_statement": "Fix it",
                "hints_text": "",
                "FAIL_TO_PASS": "[]",
                "PASS_TO_PASS": "[]",
                "patch": reference,
            }
        ],
    )
    _write_jsonl(prediction_path, [])
    _write_jsonl(
        previous_path,
        [
            {
                "instance_id": instance_id,
                "candidate_id": "old-gold-like",
                "model_name_or_path": "unit-model",
                "model_patch": reference,
                "generator_name": "previous-stage",
                "seed": 1,
                "temperature": 0.0,
                "trajectory_path": "",
                "source_visible_to_selector": False,
            },
            {
                "instance_id": instance_id,
                "candidate_id": "old-generated",
                "model_name_or_path": "unit-model",
                "model_patch": _diff(instance_id, 0),
                "generator_name": "previous-stage",
                "seed": 2,
                "temperature": 0.0,
                "trajectory_path": "",
                "source_visible_to_selector": False,
            },
        ],
    )

    result = _run_1a(
        tmp_path,
        benchmark_path,
        prediction_path,
        n_tasks=1,
        previous_stage6b1_jsonl=(str(previous_path),),
        source_jsonl=(),
        duplicate_rate_threshold=1.0,
    )

    attempts = result["attempts"]
    assert any(row["terminal_status"] == "exact_reference_hash" for row in attempts)
    assert len(result["rows"]) == 1
    assert result["rows"][0].model_patch != reference


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required for apply-check diagnostics")
def test_stage6b1a_records_required_apply_check_failures(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    repo_cache = tmp_path / "repo_cache"
    instance_id = "repo__proj-apply"
    repo_path = repo_cache / instance_id
    (repo_path / "pkg").mkdir(parents=True)
    (repo_path / "pkg" / "app.py").write_text("def value():\n    return 'old'\n", encoding="utf-8")
    bad_patch = (
        "diff --git a/pkg/app.py b/pkg/app.py\n"
        "--- a/pkg/app.py\n"
        "+++ b/pkg/app.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def value():\n"
        "-    return 'missing'\n"
        "+    return 'new'\n"
    )
    _write_jsonl(
        benchmark_path,
        [
            {
                "instance_id": instance_id,
                "repo": "repo/proj",
                "base_commit": "abc123",
                "problem_statement": "Fix it",
                "hints_text": "",
                "FAIL_TO_PASS": "[]",
                "PASS_TO_PASS": "[]",
                "patch": _diff(instance_id, 99, file_name="reference/app.py"),
            }
        ],
    )
    _write_jsonl(
        prediction_path,
        [
            {
                "instance_id": instance_id,
                "model_name_or_path": "unit-model",
                "generator_name": "unit-generator",
                "model_patch": bad_patch,
            }
        ],
    )

    result = _run_1a(
        tmp_path,
        benchmark_path,
        prediction_path,
        n_tasks=1,
        apply_check=True,
        require_apply_check=True,
        repo_cache_root=repo_cache,
        duplicate_rate_threshold=1.0,
    )

    assert any(row["terminal_status"] == "patch_apply_check_failed" for row in result["attempts"])
    assert result["rows"] == []


def test_stage6b1a_diff_validator_rejects_malformed_hunks() -> None:
    assert validate_unified_diff(_diff("repo__proj", 1)).valid is True
    malformed = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ nope @@\n-old\n+new\n"
    result = validate_unified_diff(malformed)
    assert result.valid is False
    assert result.reason == "malformed_hunk_header"
