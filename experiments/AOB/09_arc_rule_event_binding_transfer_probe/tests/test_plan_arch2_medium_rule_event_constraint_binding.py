from __future__ import annotations

from src.experiments.run_plan_arch1_5_learned_constraint_shortcut_repair import ConstraintDatasetConfig, build_repaired_constraint_splits
from src.experiments.run_plan_arch2_medium_rule_event_constraint_binding import (
    MAIN_VARIANT,
    _expand_splits_if_needed,
    _medium_gate_summary,
    _oracle_feature_audit,
)


def test_n16_expansion_preserves_one_valid_candidate_and_balanced_slots() -> None:
    splits = build_repaired_constraint_splits(
        ConstraintDatasetConfig(train_examples=16, dev_examples=8, test_examples=32, num_candidates=8, plan_length=48),
        seed=10,
    )
    expanded = _expand_splits_if_needed(splits, 16, seed=10)
    labels = []
    for example in expanded["test"]:
        labels.append(example.label)
        assert len(example.candidates) == 16
        assert sum(bool(value) for value in example.satisfies_constraint) == 1
        assert example.satisfies_constraint[example.label] is True
        assert example.metadata["all_candidates_same_endpoint"] is True
        assert example.metadata["all_candidates_reach_goal"] is True
    assert sorted(set(labels)) == list(range(16))


def test_oracle_feature_audit_marks_claimable_visible_inputs() -> None:
    splits = build_repaired_constraint_splits(
        ConstraintDatasetConfig(train_examples=8, dev_examples=4, test_examples=8, num_candidates=8, plan_length=48),
        seed=11,
    )
    audit = _oracle_feature_audit("N8_medium", 11, splits)
    assert audit["forbidden_inputs_absent"] is True
    assert audit["event_tokens_derived_from_visible_rollouts"] is True
    assert audit["rule_tokens_derived_from_visible_constraint_examples"] is True
    assert audit["teacher_labels_training_targets_only"] is True
    assert audit["claimable"] is True


def test_medium_gate_summary_requires_all_core_gates() -> None:
    rows = []
    controls = []
    for seed in [10, 11, 12, 13, 14]:
        rows.append(
            {
                "stage": "N8_medium",
                "variant": MAIN_VARIANT,
                "seed": seed,
                "trainable": {"top1": 1.0},
                "frozen": {"top1": 0.5},
                "delta_trainable_minus_frozen": 0.5,
            }
        )
        controls.append(
            {
                "stage": "N8_medium",
                "variant": MAIN_VARIANT,
                "seed": seed,
                "chance": 0.125,
                "control_pass": {
                    "overall": True,
                    "shortcut_baselines_do_not_explain": True,
                    "constraint_mismatch_collapses": True,
                    "rollout_mismatch_collapses": True,
                    "rule_event_mismatch_collapses": True,
                    "rule_token_shuffle_collapses": True,
                    "event_token_shuffle_collapses": True,
                    "ablate_constraint_rule_tokens_hurts": True,
                    "ablate_rollout_event_tokens_hurts": True,
                    "no_oracle_input_features_used": True,
                },
            }
        )
    summary = _medium_gate_summary(rows, controls, MAIN_VARIANT, "N8_medium")
    assert summary["medium_gates_pass"] is True
    assert summary["seed_wins"] == 5
