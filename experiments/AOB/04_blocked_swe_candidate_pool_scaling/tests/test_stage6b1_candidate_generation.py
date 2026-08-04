import json
from pathlib import Path

from src.experiments.run_stage6b0_candidate_pool_mining import (
    _load_prediction_file,
    build_candidate_pool,
)
from src.experiments.run_stage6b1_candidate_generation import run_stage6b1_candidate_generation


def _diff(instance_id: str, candidate_index: int) -> str:
    path = f"pkg/{instance_id}_{candidate_index}.py"
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


def test_stage6b1_generates_required_jsonl_and_quality_gate_from_local_predictions(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    task_ids_path = tmp_path / "task_ids.txt"
    benchmark_rows = []
    prediction_rows = []
    task_ids = []
    for task_index in range(25):
        instance_id = f"repo__proj-{task_index}"
        task_ids.append(instance_id)
        benchmark_rows.append(
            {
                "instance_id": instance_id,
                "repo": "repo/proj",
                "base_commit": "abc123",
                "problem_statement": f"Fix task {task_index}",
                "hints_text": "",
                "FAIL_TO_PASS": "[]",
                "PASS_TO_PASS": "[]",
                "patch": _diff(instance_id, 99),
            }
        )
        candidate_count = 8 if task_index < 20 else 3
        for candidate_index in range(candidate_count):
            prediction_rows.append(
                {
                    "instance_id": instance_id,
                    "candidate_id": f"{instance_id}-raw-{candidate_index}",
                    "model_name_or_path": "unit-model",
                    "generator_name": f"unit-generator-{candidate_index % 2}",
                    "seed": 1000 + candidate_index,
                    "temperature": 0.2,
                    "config_name": f"prompt-{candidate_index % 3}",
                    "model_patch": _diff(instance_id, candidate_index),
                    "source_visible_to_selector": False,
                }
            )
    _write_jsonl(benchmark_path, benchmark_rows)
    _write_jsonl(prediction_path, prediction_rows)
    task_ids_path.write_text("\n".join(task_ids), encoding="utf-8")

    result = run_stage6b1_candidate_generation(
        dataset_name="local/unit",
        split="test",
        benchmark_jsonl=benchmark_path,
        task_ids=(),
        task_ids_file=task_ids_path,
        n_tasks=25,
        k=8,
        seed=123,
        output_path=tmp_path / "results" / "stage6b1_generated_candidates.jsonl",
        audit_path=tmp_path / "results" / "stage6b1_generation_audit.json",
        report_path=tmp_path / "reports" / "STAGE6B1_CANDIDATE_GENERATION.md",
        trajectory_root=tmp_path / "results" / "trajectories",
        candidate_patch_root=tmp_path / "results" / "patches",
        prompt_root=tmp_path / "results" / "prompts",
        source_jsonl=(str(prediction_path),),
        source_json=(),
        source_roots=(),
        include_default_hf_public=False,
        enable_mini_swe_agent=False,
        mini_swe_agent_command_templates=(),
        mini_swe_agent_configs=(),
        api_generator_command_templates=(),
        api_generator_models=(),
        api_generator_temperatures=(),
        api_generator_seeds=(),
        duplicate_rate_threshold=0.20,
        near_reference_threshold=0.95,
    )

    audit = result["audit"]
    output_rows = [
        json.loads(line)
        for line in (tmp_path / "results" / "stage6b1_generated_candidates.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert audit["quality_gates"]["candidate_generation_quality_gates_pass"] is True
    assert audit["tasks_attempted"] == 25
    assert audit["tasks_with_at_least_k_unique_generated_candidates"] == 20
    assert len(output_rows) == 20 * 8 + 5 * 3
    required = {
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
    assert all(set(row) == required for row in output_rows)
    assert all(row["source_visible_to_selector"] is False for row in output_rows)
    assert all(Path(row["trajectory_path"]).exists() for row in output_rows)
    assert audit["raw_preserving_expansion_audit"]["passes"] is True


def test_stage6b1_excludes_reference_duplicates_and_stage6b0_can_mine_output(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "results" / "stage6b1_generated_candidates.jsonl"
    instance_id = "repo__proj-1"
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
    rows = [
        {
            "instance_id": instance_id,
            "model_name_or_path": "unit-model",
            "generator_name": "unit-generator",
            "seed": 1,
            "temperature": 0.0,
            "config_name": "gold-like",
            "model_patch": reference,
            "source_visible_to_selector": False,
        },
        {
            "instance_id": instance_id,
            "model_name_or_path": "unit-model",
            "generator_name": "unit-generator",
            "seed": 2,
            "temperature": 0.0,
            "config_name": "duplicate",
            "model_patch": _diff(instance_id, 0),
            "source_visible_to_selector": False,
        },
        {
            "instance_id": instance_id,
            "model_name_or_path": "unit-model",
            "generator_name": "unit-generator",
            "seed": 3,
            "temperature": 0.0,
            "config_name": "duplicate",
            "model_patch": _diff(instance_id, 0),
            "source_visible_to_selector": False,
        },
    ]
    for candidate_index in range(1, 8):
        rows.append(
            {
                "instance_id": instance_id,
                "model_name_or_path": "unit-model",
                "generator_name": f"unit-generator-{candidate_index}",
                "seed": 10 + candidate_index,
                "temperature": 0.1,
                "config_name": "unique",
                "model_patch": _diff(instance_id, candidate_index),
                "source_visible_to_selector": False,
            }
        )
    _write_jsonl(prediction_path, rows)

    result = run_stage6b1_candidate_generation(
        dataset_name="local/unit",
        split="test",
        benchmark_jsonl=benchmark_path,
        task_ids=(instance_id,),
        task_ids_file=None,
        n_tasks=1,
        k=8,
        seed=456,
        output_path=output_path,
        audit_path=tmp_path / "results" / "stage6b1_generation_audit.json",
        report_path=tmp_path / "reports" / "STAGE6B1_CANDIDATE_GENERATION.md",
        trajectory_root=tmp_path / "results" / "trajectories",
        candidate_patch_root=tmp_path / "results" / "patches",
        prompt_root=tmp_path / "results" / "prompts",
        source_jsonl=(str(prediction_path),),
        source_json=(),
        source_roots=(),
        include_default_hf_public=False,
        enable_mini_swe_agent=False,
        mini_swe_agent_command_templates=(),
        mini_swe_agent_configs=(),
        api_generator_command_templates=(),
        api_generator_models=(),
        api_generator_temperatures=(),
        api_generator_seeds=(),
        duplicate_rate_threshold=0.50,
        near_reference_threshold=0.95,
    )

    audit = result["audit"]
    assert audit["exact_reference_patch_hash_audit"]["source_hits_excluded_total"] == 1
    assert audit["exact_reference_patch_hash_audit"]["passes"] is True
    assert audit["duplicate_patch_hash_audit"]["duplicates_excluded_total"] == 1
    output_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(output_rows) == 8

    source_audit = {"sources": [], "errors": [], "redacted_metadata_fields": {}}
    benchmark_by_id = {instance_id: json.loads(benchmark_path.read_text(encoding="utf-8").splitlines()[0])}
    generated_records = _load_prediction_file(output_path, benchmark_by_id, source_audit, forced_kind="jsonl")
    examples, _, _, build_audit = build_candidate_pool(
        benchmark_by_id=benchmark_by_id,
        generated_records=generated_records,
        dataset_name="local/unit",
        split="test",
        n_tasks=1,
        seed=789,
    )
    assert len(examples) == 1
    assert len(examples[0].candidates) == 8
    assert build_audit["raw_preserving_expansion_used"] is False
    assert all(candidate.candidate_source_agent == "generated_candidate_source_blinded" for candidate in examples[0].candidates)
