from src.datasets.swe_patch_selection_dataset import PatchCandidate, PatchSelectionExample
from src.experiments.run_stage6d_pool_scale_and_view_hardening import build_view_hardened_examples
from src.experiments.run_stage6d1_pool_expansion_and_evidence_stress import (
    build_near_reference_quarantine,
    run_evidence_stress_for_pool,
)


def _diff(instance_id: str, slot: int, terms: str) -> str:
    return (
        f"diff --git a/pkg/{instance_id}_{slot}.py b/pkg/{instance_id}_{slot}.py\n"
        f"--- a/pkg/{instance_id}_{slot}.py\n"
        f"+++ b/pkg/{instance_id}_{slot}.py\n"
        "@@ -1,3 +1,4 @@ def update_value(request):\n"
        " import os\n"
        "-old_value = request.value\n"
        f"+new_value = request.value  # {terms}\n"
        "+return new_value\n"
    )


def _example(index: int, positive_slot: int | None, issue_terms: str) -> PatchSelectionExample:
    instance_id = f"repo__proj-{index}"
    candidates = []
    for slot in range(8):
        terms = issue_terms if positive_slot == slot else f"unrelated fallback {slot}"
        candidates.append(
            PatchCandidate(
                candidate_id=f"{instance_id}-candidate-{slot}",
                candidate_source_agent="unit-generator",
                candidate_diff=_diff(instance_id, slot, terms),
            )
        )
    return PatchSelectionExample(
        id=f"stage6d1-{instance_id}",
        dataset_name="unit",
        repo="repo/proj",
        issue_id=instance_id,
        issue_text=f"Fix regression involving {issue_terms}",
        failing_test_summary=f"FAIL_TO_PASS: [test_{issue_terms.replace(' ', '_')}]",
        retrieved_contexts=("Repository: repo/proj",),
        candidates=tuple(candidates),
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


def test_stage6d1_quarantines_full_near_reference_task_to_preserve_k8() -> None:
    pool_all = [
        _example(0, positive_slot=3, issue_terms="timeout header"),
        _example(1, positive_slot=5, issue_terms="unicode label"),
        _example(2, positive_slot=None, issue_terms="decimal quantize"),
    ]
    near_hits = [
        {
            "instance_id": pool_all[1].issue_id,
            "candidate_index": 5,
            "candidate_id": pool_all[1].candidates[5].candidate_id,
            "similarity": 0.96,
        }
    ]
    pool_strict = [example for example in pool_all if example.issue_id != pool_all[1].issue_id]
    quarantine = build_near_reference_quarantine(pool_all, pool_strict, near_hits, seed=123)

    assert quarantine["passes"]
    assert quarantine["quarantined_task_ids"] == [pool_all[1].issue_id]
    assert quarantine["pool_all"]["complete_tasks"] == 3
    assert quarantine["pool_strict"]["complete_tasks"] == 2
    assert quarantine["delta"]["candidate_count"] == -8
    assert all(row["candidate_count"] == 8 for row in quarantine["quarantined_tasks"])


def test_stage6d1_evidence_stress_detects_context_or_mismatch_signal() -> None:
    pool = [
        _example(0, positive_slot=3, issue_terms="timeout header"),
        _example(1, positive_slot=5, issue_terms="unicode label"),
        _example(2, positive_slot=2, issue_terms="decimal quantize"),
    ]
    hardened = build_view_hardened_examples(pool, seed=456)
    stress = run_evidence_stress_for_pool("unit_pool", hardened, seed=456)

    assert stress["summary"]["passes"]
    assert (
        stress["summary"]["issue_context_patch_minus_patch_only"] >= 0.05
        or stress["summary"]["candidate_evidence_mismatch_degradation"] >= 0.05
        or stress["summary"]["cross_task_context_shuffle_degradation"] >= 0.05
        or stress["summary"]["edited_file_context_removed_degradation"] >= 0.05
    )
