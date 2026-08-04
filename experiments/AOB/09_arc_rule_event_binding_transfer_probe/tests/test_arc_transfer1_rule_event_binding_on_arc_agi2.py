from __future__ import annotations

import numpy as np

from src.datasets.arc_agi2_verification import ArcPair, ArcVerificationExample
from src.experiments.run_arc_transfer1_rule_event_binding_on_arc_agi2 import (
    _metric_block,
    _oracle_feature_audit,
    arc_event_order,
    arc_rule_order,
    arc_transfer_batch,
)


def _example() -> ArcVerificationExample:
    train_pairs = (
        ArcPair(input=[[0, 1], [0, 0]], output=[[0, 2], [0, 0]]),
        ArcPair(input=[[0, 1], [1, 0]], output=[[0, 2], [2, 0]]),
    )
    candidates = (
        [[0, 1], [0, 0]],
        [[0, 2], [0, 0]],
        [[2, 0], [0, 0]],
        [[0, 0], [0, 0]],
    )
    return ArcVerificationExample(
        id="unit__test0",
        task_id="unit",
        split="test",
        train_pairs=train_pairs,
        test_input=[[0, 1], [0, 0]],
        gold_output=[[0, 2], [0, 0]],
        candidates=candidates,
        label=1,
        negative_types=("candidate",) * 4,
        public_source_tags=("candidate_grid",) * 4,
        metadata={
            "offline_gold_present": True,
            "offline_correct_candidate_indices": [1],
            "candidate_train_execution_scores": [0.2, 0.9, 0.3, 0.1],
            "model_visible_candidate_sources": False,
            "model_visible_generator_rank": False,
            "model_visible_hidden_oracle_score": False,
            "test_output_visible_outside_candidate_objects": False,
            "transformation_type": "sparse_edit",
        },
    )


def test_arc_rule_and_event_orders_are_four_visible_slots() -> None:
    example = _example()
    assert len(arc_rule_order(example)) == 4
    assert len(arc_event_order(example, 1)) == 4
    assert set(arc_rule_order(example)) == {0, 1, 2, 3}
    assert set(arc_event_order(example, 1)) == {0, 1, 2, 3}


def test_arc_transfer_batch_uses_expected_shapes() -> None:
    batch = arc_transfer_batch([_example()], "cpu")
    assert tuple(batch["rule_order"].shape) == (1, 4)
    assert tuple(batch["event_order"].shape) == (1, 4, 4)
    assert tuple(batch["valid"].shape) == (1, 4)
    assert int(batch["labels"][0]) == 1


def test_metric_and_oracle_audit_for_gold_present_example() -> None:
    example = _example()
    logits = np.asarray([[0.0, 3.0, 1.0, -1.0]], dtype=np.float32)
    metric = _metric_block(logits, [example])
    assert metric["top1"] == 1.0
    assert metric["conditional_gold_present_top1"] == 1.0
    assert metric["overall_solve_rate"] == 1.0
    audit = _oracle_feature_audit([example])
    assert audit["pass"] is True
    assert audit["forbidden_test_outputs_as_inputs"] is False
