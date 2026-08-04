import json
from pathlib import Path

from src.experiments.run_stage6b1_candidate_generation import load_benchmark_tasks
from src.experiments.run_stage6b2_official_harness_labeling import (
    HarnessCandidateResult,
    attach_official_labels,
    build_stage6b2_audit,
    build_unlabeled_stage6b2_examples,
    load_stage6b1b_candidate_source,
)


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


def _benchmark_and_candidates(tmp_path: Path) -> tuple[Path, Path, Path]:
    benchmark_path = tmp_path / "benchmark.jsonl"
    candidate_path = tmp_path / "stage6b1b_generated_candidates.jsonl"
    audit_path = tmp_path / "stage6b1b_generation_audit.json"
    benchmark_rows = []
    candidate_rows = []
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
                "patch": _diff(instance_id, 99),
            }
        )
        for candidate_index in range(8):
            candidate_rows.append(
                {
                    "candidate_id": f"{instance_id}-stage6b1b-candidate-{candidate_index:02d}",
                    "generator_name": "unit-generator",
                    "instance_id": instance_id,
                    "model_name_or_path": "unit-model",
                    "model_patch": _diff(instance_id, candidate_index),
                    "seed": None,
                    "source_visible_to_selector": False,
                    "temperature": None,
                    "trajectory_path": f"hidden/{instance_id}/{candidate_index}.json",
                }
            )
    _write_jsonl(benchmark_path, benchmark_rows)
    _write_jsonl(candidate_path, candidate_rows)
    audit_path.write_text(
        json.dumps(
            {
                "candidate_order_randomizable": True,
                "raw_preserving_and_fabrication_audit": {
                    "raw_preserving_expansion_used": False,
                    "missing_candidate_fabrication_used": False,
                },
            }
        ),
        encoding="utf-8",
    )
    return benchmark_path, candidate_path, audit_path


def _fake_results(examples, positive_task_count: int = 8) -> list[HarnessCandidateResult]:
    rows = []
    positive_ids = {example.issue_id for example in examples[:positive_task_count]}
    for example in examples:
        for candidate_index, candidate in enumerate(example.candidates):
            resolved = example.issue_id in positive_ids and candidate_index == 1
            rows.append(
                HarnessCandidateResult(
                    example_id=example.id,
                    issue_id=example.issue_id,
                    repo=example.repo,
                    candidate_index=candidate_index,
                    candidate_id=candidate.candidate_id,
                    candidate_source_agent=candidate.candidate_source_agent,
                    candidate_patch_hash=candidate.patch_hash,
                    run_id=f"stage6b2_seed_1_slot_{candidate_index}",
                    model_name_or_path=f"stage6b2_candidate_slot_{candidate_index}",
                    report_path=f"logs/run_evaluation/stage6b2_seed_1_slot_{candidate_index}/report.json",
                    test_output_path=f"logs/run_evaluation/stage6b2_seed_1_slot_{candidate_index}/test_output.txt",
                    log_dir=f"logs/run_evaluation/stage6b2_seed_1_slot_{candidate_index}/{example.issue_id}",
                    result_available=True,
                    resolved=resolved,
                    applied=True,
                    timed_out=False,
                    error=None,
                    official_label_source_field="report.json:instance.resolved",
                    cache_hit=False,
                )
            )
    return rows


def test_stage6b2_builds_official_labeled_pool_without_patch_or_log_leakage(tmp_path: Path) -> None:
    benchmark_path, candidate_path, audit_path = _benchmark_and_candidates(tmp_path)
    candidates_by_task, input_audit = load_stage6b1b_candidate_source(candidate_path, k=8)
    task_records, benchmark_audit = load_benchmark_tasks(
        dataset_name="local/unit",
        split="test",
        benchmark_jsonl=benchmark_path,
        requested_task_ids=tuple(candidates_by_task),
        task_ids_file=None,
        n_tasks=25,
        seed=1,
    )
    stage6b1b_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    examples = build_unlabeled_stage6b2_examples(
        candidates_by_task=candidates_by_task,
        task_records=task_records,
        dataset_name="local/unit",
        split="test",
        input_path=candidate_path,
        stage6b1b_audit=stage6b1b_audit,
        seed=1,
    )
    harness_results = _fake_results(examples, positive_task_count=8)
    labeled = attach_official_labels(examples, harness_results, k=8)
    for example in labeled:
        assert len(example.candidates) == 8
        assert len(example.labels_pass_fail) == 8
        assert "test_output" not in json.dumps(example.metadata)
        assert "report.json" not in json.dumps(example.metadata)
        for candidate in example.candidates:
            assert candidate.metadata["stage6b1b_input_patch_hash"] == candidate.patch_hash
            assert "trajectory_path" not in candidate.metadata

    harness_root = tmp_path / "stage6b2_official_harness"
    for candidate_index in range(8):
        rows = [
            {
                "instance_id": example.issue_id,
                "model_name_or_path": f"stage6b2_candidate_slot_{candidate_index}",
                "model_patch": example.candidates[candidate_index].candidate_diff,
            }
            for example in labeled
        ]
        _write_jsonl(harness_root / "predictions" / f"stage6b2_seed_1_slot_{candidate_index}.jsonl", rows)
    audit = build_stage6b2_audit(
        created_at="now",
        dataset_name="local/unit",
        split="test",
        input_path=candidate_path,
        stage6b1b_audit_path=audit_path,
        stage6b1b_audit=stage6b1b_audit,
        labeled_output_path=tmp_path / "labeled.jsonl",
        harness_results_path=tmp_path / "harness.jsonl",
        harness_root=harness_root,
        examples=labeled,
        candidates_by_task=candidates_by_task,
        task_records=task_records,
        input_audit=input_audit,
        benchmark_audit=benchmark_audit,
        harness_results=harness_results,
        environment={},
        k=8,
        seed=1,
        max_workers=1,
        timeout=30,
        cache_level="env",
        namespace="swebench",
        force_rerun=False,
        build_only=False,
    )
    assert audit["official_label_availability_audit"]["available_candidate_results"] == 200
    assert audit["oracle_pass_at_8_audit"]["oracle_positive_task_count"] == 8
    assert audit["exact_reference_patch_hash_audit"]["hits_total"] == 0
    assert audit["patch_integrity_audit"]["passes"] is True
    assert audit["quality_gates"]["stage6b2_official_label_quality_gates_pass"] is True


def test_stage6b2_does_not_fabricate_negative_labels_when_harness_result_missing(tmp_path: Path) -> None:
    benchmark_path, candidate_path, audit_path = _benchmark_and_candidates(tmp_path)
    candidates_by_task, _ = load_stage6b1b_candidate_source(candidate_path, k=8)
    task_records, _ = load_benchmark_tasks(
        dataset_name="local/unit",
        split="test",
        benchmark_jsonl=benchmark_path,
        requested_task_ids=tuple(candidates_by_task),
        task_ids_file=None,
        n_tasks=25,
        seed=1,
    )
    examples = build_unlabeled_stage6b2_examples(
        candidates_by_task=candidates_by_task,
        task_records=task_records,
        dataset_name="local/unit",
        split="test",
        input_path=candidate_path,
        stage6b1b_audit=json.loads(audit_path.read_text(encoding="utf-8")),
        seed=1,
    )
    labeled = attach_official_labels(examples, harness_results=[], k=8)
    assert all(len(example.candidates) == 8 for example in labeled)
    assert all(example.labels_pass_fail == () for example in labeled)
    assert all(example.metadata["official_labels_complete"] is False for example in labeled)
