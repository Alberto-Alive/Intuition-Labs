import json

from src.datasets.swe_patch_selection_dataset import PatchCandidate, PatchSelectionExample
from src.experiments.run_stage6d2_pool_scaling import (
    build_stage6d2_quality_gates,
    build_stage6d2_source_coverage_audit,
)


def _diff(instance_id: str, slot: int) -> str:
    return (
        f"diff --git a/pkg/{instance_id}_{slot}.py b/pkg/{instance_id}_{slot}.py\n"
        f"--- a/pkg/{instance_id}_{slot}.py\n"
        f"+++ b/pkg/{instance_id}_{slot}.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        f"+new_{slot}\n"
    )


def _example(instance_id: str, positive_slot: int | None = None) -> PatchSelectionExample:
    candidates = tuple(
        PatchCandidate(
            candidate_id=f"{instance_id}-candidate-{slot}",
            candidate_source_agent="unit-generator",
            candidate_diff=_diff(instance_id, slot),
        )
        for slot in range(8)
    )
    return PatchSelectionExample(
        id=f"stage6d2-{instance_id}",
        dataset_name="unit",
        repo="repo/proj",
        issue_id=instance_id,
        issue_text="Fix unit behavior",
        failing_test_summary="FAIL_TO_PASS: [test_unit]",
        retrieved_contexts=("context",),
        candidates=candidates,
        labels_pass_fail=tuple(1 if positive_slot == slot else 0 for slot in range(8)),
        split="test",
        metadata={
            "candidate_order_randomized": True,
            "candidate_order_source": "unit",
            "gold_patch_included": False,
            "raw_preserving_expansion_used": False,
            "missing_candidate_fabrication_used": False,
        },
    )


def test_stage6d2_source_coverage_counts_k8_unique_generated_tasks(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    rows = []
    for slot in range(8):
        rows.append({"instance_id": "repo__proj-1", "model_patch": _diff("repo__proj-1", slot)})
    for slot in range(7):
        rows.append({"instance_id": "repo__proj-2", "model_patch": _diff("repo__proj-2", slot)})
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    audit = build_stage6d2_source_coverage_audit(
        candidate_sources=[path],
        source_roots=[],
        pool_all=[_example("repo__proj-1")],
        pool_strict=[],
    )

    assert audit["summary"]["raw_records_loaded"] == 15
    assert audit["summary"]["patch_records_loaded"] == 15
    assert audit["summary"]["tasks_with_at_least_k8_unique_generated_candidates"] == 1
    assert audit["summary"]["overlap_k8_sources_with_pool_all"] == 1
    assert audit["per_instance_unique_generated_patch_counts"]["repo__proj-2"] == 7


def test_stage6d2_quality_gates_allow_compute_limit_but_block_low_positive_count() -> None:
    pool_bundle = {
        "summary": {"complete_tasks": 18, "oracle_positive_tasks": 5, "oracle_pass_at_8": 0.2778},
        "audits": {
            "exact_reference_hash_audit": {"passes": True},
            "near_reference_similarity_audit": {"passes": True},
            "output_leakage_audit": {"passes": True},
            "source_generator_leakage_audit": {"passes": True},
            "duplicate_patch_hash_audit": {"passes": True},
        },
    }
    stress = {"summary": {"pool_strict_passes": True}}
    expansion = {"compute_limit_documented": True}

    gates = build_stage6d2_quality_gates(
        pool_strict_bundle=pool_bundle,
        stress=stress,
        expansion=expansion,
        minimum_strict_tasks=50,
        minimum_strict_oracle_positive=15,
    )

    assert gates["pool_strict_has_at_least_50_complete_k8_tasks_or_compute_limit_documented"]
    assert not gates["pool_strict_has_at_least_15_oracle_positive_tasks"]
    assert gates["evidence_use_diagnostic_passes"]
