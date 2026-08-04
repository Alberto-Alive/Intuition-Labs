from dataclasses import replace

from src.datasets.latent_attention_capacity_dataset import Stage8DatasetConfig, build_stage8_examples
from src.experiments.run_stage8b4_top3_cleanqkv_validation import (
    _cleanqkv_challenger_config,
    _decision,
    _freeze_recommendation,
)
from src.experiments.stage8_controls import apply_stage8_control, control_expectation
from src.models.latent_attention_variants import build_stage8_selector, estimate_stage8_compute


def test_stage8b4_memory_controls_are_callable_and_labeled() -> None:
    examples = build_stage8_examples(Stage8DatasetConfig(n_examples=12, n_blocks=8), seed=8404)

    permuted = apply_stage8_control(examples, "memory_slot_permutation", seed=1)
    shuffled = apply_stage8_control(examples, "memory_shuffle_across_examples", seed=2)
    closed = apply_stage8_control(examples, "memory_gate_forced_closed", seed=3)

    assert control_expectation("memory_slot_permutation") == "invariant"
    assert control_expectation("memory_disabled") == "degrade"
    assert control_expectation("memory_gate_forced_open") == "diagnostic"
    assert all(before.evidence_blocks == after.evidence_blocks for before, after in zip(examples, permuted))
    assert all(example.metadata["memory_slot_permutation"] for example in permuted)
    assert any(before.evidence_blocks != after.evidence_blocks for before, after in zip(examples, shuffled))
    assert all(example.metadata["memory_shuffle_across_examples"] for example in shuffled)
    assert all(example.metadata["memory_gate_forced_closed"] for example in closed)


def test_stage8b4_cleanqkv_challenger_builds_predicts_and_reports_memory_diagnostics() -> None:
    config = replace(_cleanqkv_challenger_config(), epochs=1, max_train_examples=12)
    train = build_stage8_examples(Stage8DatasetConfig(n_examples=12, n_blocks=8), seed=8405)
    dev = build_stage8_examples(
        Stage8DatasetConfig(n_examples=8, n_blocks=8, split="dev", template_split="heldout_dev"),
        seed=8406,
    )

    selector = build_stage8_selector(config, seed=4)
    selector.fit(train, dev)
    predictions = selector.predict(dev)
    diagnostics = selector.diagnostics(dev)
    compute = estimate_stage8_compute(config, n_blocks=8, k_candidates=8)

    assert len(predictions) == len(dev)
    assert all(0 <= prediction < 8 for prediction in predictions)
    assert diagnostics["clean_qkv_current_self_attention"] == "candidate_query_only_no_evidence_tokens"
    assert diagnostics["memory_reader"] == "top_k_activation_memory_routing"
    assert diagnostics["memory_slot_entropy_mean"] > 0.0
    assert diagnostics["memory_gate_mean"] >= 0.0
    assert compute["latent_views"] == config.roles * config.avenues
    assert compute["parameter_count_estimate"] > 0


def test_stage8b4_decision_can_report_top3_win_when_cleanqkv_lags() -> None:
    top3 = {
        "architecture_name": "stage8b3_f_memory_slot_attention_02",
        "frozen_from_stage8b3": True,
        "ready_for_stage8c_freeze": True,
        "capacity_C": 64,
        "held_out_template_dev_accuracy": 0.90,
        "controls_pass": True,
    }
    clean = {
        "architecture_name": "stage8b4_clean_qkv_activation_memory_challenger",
        "cleanqkv_challenger": True,
        "controls_pass": False,
        "capacity_C": 32,
        "held_out_template_dev_accuracy": 0.70,
        "controls_summary": {},
    }

    decision = _decision([top3, clean], selected=top3)
    recommendation = _freeze_recommendation({"decision": decision}, top3)

    assert decision == "STAGE8B4_TOP3_WIN_CLEANQKV_NOT_COMPETITIVE"
    assert recommendation["recommend_freeze_for_stage8c"]
