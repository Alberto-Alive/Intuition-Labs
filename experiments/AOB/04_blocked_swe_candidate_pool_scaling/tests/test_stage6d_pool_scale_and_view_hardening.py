from src.datasets.swe_patch_selection_dataset import PatchCandidate, PatchSelectionExample, stable_patch_hash
from src.experiments.run_stage6d_pool_scale_and_view_hardening import (
    build_view_hardened_examples,
    run_view_diagnostics,
    source_generator_output_leakage_audit,
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


def _example(index: int, positive_slot: int, issue_terms: str) -> PatchSelectionExample:
    instance_id = f"repo__proj-{index}"
    candidates = []
    for slot in range(8):
        terms = issue_terms if slot == positive_slot else f"unrelated fallback {slot}"
        candidates.append(
            PatchCandidate(
                candidate_id=f"{instance_id}-candidate-{slot}",
                candidate_source_agent="unit-generator",
                candidate_diff=_diff(instance_id, slot, terms),
                metadata={"source_visible_to_selector": False},
            )
        )
    return PatchSelectionExample(
        id=f"stage6d-{instance_id}",
        dataset_name="unit",
        repo="repo/proj",
        issue_id=instance_id,
        issue_text=f"Fix regression involving {issue_terms}",
        failing_test_summary=f"FAIL_TO_PASS: [test_{issue_terms.replace(' ', '_')}]",
        retrieved_contexts=("Repository: repo/proj",),
        candidates=tuple(candidates),
        labels_pass_fail=tuple(1 if slot == positive_slot else 0 for slot in range(8)),
        split="test",
        metadata={
            "candidate_order_randomized": True,
            "candidate_order_source": "unit",
            "gold_patch_included": False,
            "raw_preserving_expansion_used": False,
            "missing_candidate_fabrication_used": False,
        },
    )


def test_stage6d_view_hardening_preserves_candidate_patches_and_labels() -> None:
    examples = [_example(0, positive_slot=3, issue_terms="timeout header")]
    before_hashes = [stable_patch_hash(candidate.candidate_diff) for candidate in examples[0].candidates]
    hardened = build_view_hardened_examples(examples, seed=123)
    after_hashes = [stable_patch_hash(candidate.candidate_diff) for candidate in hardened[0].candidates]

    assert before_hashes == after_hashes
    assert hardened[0].labels_pass_fail == examples[0].labels_pass_fail
    assert "Stage 6D retrieved source context" in "\n".join(hardened[0].retrieved_contexts)
    assert "Avenue local syntax/API compatibility" in hardened[0].metadata["dependency_callgraph_related_file_evidence"]
    assert source_generator_output_leakage_audit(hardened)["passes"]


def test_stage6d_view_diagnostics_detect_issue_context_signal() -> None:
    examples = [
        _example(0, positive_slot=3, issue_terms="timeout header"),
        _example(1, positive_slot=5, issue_terms="unicode label"),
        _example(2, positive_slot=2, issue_terms="decimal quantize"),
    ]
    hardened = build_view_hardened_examples(examples, seed=456)
    diagnostics = run_view_diagnostics(hardened, seed=456)
    summary = diagnostics["summary"]

    assert summary["issue_context_signal_detected"]
    assert (
        summary["issue_context_patch_pass_at_1"] > summary["patch_only_pass_at_1"]
        or summary["issue_context_patch_pass_at_1"] > summary["candidate_evidence_mismatch_pass_at_1"]
    )
