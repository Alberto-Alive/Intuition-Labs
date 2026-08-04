import json
from pathlib import Path

from src.datasets.swe_patch_selection_dataset import PatchCandidate, PatchSelectionExample
from src.experiments.run_stage6b3_label_completion import identify_missing_candidate_labels
from src.experiments.run_stage6c_mini_selector_validation import build_cv_splits, make_stratified_folds


def _example(index: int, positive_slot: int | None) -> PatchSelectionExample:
    instance_id = f"repo__proj-{index}"
    candidates = tuple(
        PatchCandidate(
            candidate_id=f"{instance_id}-candidate-{slot}",
            candidate_source_agent="generator",
            candidate_diff=(
                f"diff --git a/pkg/{instance_id}_{slot}.py b/pkg/{instance_id}_{slot}.py\n"
                f"--- a/pkg/{instance_id}_{slot}.py\n"
                f"+++ b/pkg/{instance_id}_{slot}.py\n"
                "@@ -1 +1 @@\n"
                f"-old\n+new-{slot}\n"
            ),
        )
        for slot in range(8)
    )
    labels = tuple(1 if positive_slot == slot else 0 for slot in range(8))
    return PatchSelectionExample(
        id=f"stage6b2-{instance_id}",
        dataset_name="unit",
        repo="repo/proj",
        issue_id=instance_id,
        issue_text="Fix it",
        failing_test_summary="FAIL_TO_PASS: []",
        retrieved_contexts=("context",),
        candidates=candidates,
        labels_pass_fail=labels,
        split="test",
        metadata={"gold_patch_included": False},
    )


def test_stage6b3_identifies_missing_candidate_without_fabricating_label() -> None:
    complete = _example(0, positive_slot=2)
    incomplete = PatchSelectionExample(
        **{
            **complete.__dict__,
            "id": "stage6b2-repo__proj-1",
            "issue_id": "repo__proj-1",
            "labels_pass_fail": (),
        }
    )
    harness_rows = [
        {
            "example_id": complete.id,
            "issue_id": complete.issue_id,
            "candidate_index": slot,
            "candidate_id": complete.candidates[slot].candidate_id,
            "result_available": True,
            "resolved": slot == 2,
        }
        for slot in range(8)
    ]
    harness_rows.extend(
        {
            "example_id": incomplete.id,
            "issue_id": incomplete.issue_id,
            "candidate_index": slot,
            "candidate_id": incomplete.candidates[slot].candidate_id,
            "result_available": slot != 7,
            "resolved": False if slot != 7 else None,
            "error": None if slot != 7 else "missing report.json",
        }
        for slot in range(8)
    )
    missing = identify_missing_candidate_labels([complete, incomplete], harness_rows)
    assert len(missing) == 1
    assert missing[0]["instance_id"] == "repo__proj-1"
    assert missing[0]["slot_index"] == 7
    assert missing[0]["reason_missing"] == "missing report.json"


def test_stage6c_mini_stratified_folds_keep_task_disjoint_and_positive_tests() -> None:
    examples = [_example(index, positive_slot=0 if index < 7 else None) for index in range(20)]
    folds = make_stratified_folds(examples, folds=5, seed=123)
    splits = build_cv_splits(examples, folds)
    assert len(splits["folds"]) == 5
    for fold in splits["folds"]:
        train = set(fold["train_indices"])
        dev = set(fold["dev_indices"])
        test = set(fold["test_indices"])
        assert train.isdisjoint(test)
        assert dev.isdisjoint(test)
        assert train.isdisjoint(dev)
        assert fold["train_oracle_positive"] > 0
        assert fold["dev_oracle_positive"] > 0
        assert fold["test_oracle_positive"] > 0
