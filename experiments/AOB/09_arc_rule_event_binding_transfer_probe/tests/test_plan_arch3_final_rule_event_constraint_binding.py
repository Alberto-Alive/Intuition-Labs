from __future__ import annotations

import numpy as np

from src.experiments.run_plan_arch1_4_learned_constraint_planning import ConstraintDatasetConfig
from src.experiments.run_plan_arch3_final_rule_event_constraint_binding import (
    MAIN_VARIANT,
    _final_gate_summary,
    _final_metric_block,
    _stage_splits,
)


def test_final_stage_splits_use_repaired_candidate_counts() -> None:
    cfg = ConstraintDatasetConfig(train_examples=8, dev_examples=4, test_examples=8, num_candidates=8, plan_length=48)
    n8 = _stage_splits("N8_final", cfg, seed=70)
    n16 = _stage_splits("N16_final", cfg, seed=70)
    assert len(n8["test"][0].candidates) == 8
    assert len(n16["test"][0].candidates) == 16
    assert sum(bool(value) for value in n8["test"][0].satisfies_constraint) == 1
    assert sum(bool(value) for value in n16["test"][0].satisfies_constraint) == 1
    assert n16["test"][0].metadata["n16_repair"] is True


def test_final_metric_block_reports_topk_and_mrr() -> None:
    logits = np.asarray([[3.0, 1.0, 0.0], [0.0, 2.0, 3.0]], dtype=np.float32)
    labels = np.asarray([0, 1], dtype=np.int64)
    metric = _final_metric_block(logits, labels, [])
    assert metric["top1"] == 0.5
    assert metric["top2"] == 1.0
    assert metric["top3"] == 1.0
    assert metric["mrr"] == 0.75


def test_final_gate_summary_applies_n8_and_n16_thresholds() -> None:
    rows = []
    controls = []
    for seed in range(70, 80):
        delta = 0.30 if seed < 78 else -0.01
        rows.append(
            {
                "stage": "N8_final",
                "variant": MAIN_VARIANT,
                "seed": seed,
                "trainable": {"top1": 1.0},
                "frozen": {"top1": 1.0 - delta},
                "delta_trainable_minus_frozen": delta,
            }
        )
        controls.append(
            {
                "stage": "N8_final",
                "variant": MAIN_VARIANT,
                "seed": seed,
                "chance": 0.125,
                "control_pass": {
                    "overall": True,
                    "shortcut_baselines_do_not_explain": True,
                    "no_oracle_input_features_used": True,
                },
            }
        )
    gate = _final_gate_summary(rows, controls, MAIN_VARIANT, "N8_final")
    assert gate["seed_wins"] == 8
    assert gate["final_gates_pass"] is True
