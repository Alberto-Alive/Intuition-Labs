from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from src.experiments.e5_1_state_stability import (  # noqa: E402
    DECISION_CUDA_REQUIRED,
    K_CANDIDATES,
    STEP_TOKENS,
    FittedFullContextQKVTeacher,
    IntegratedRollingStateCleanQKVStudent,
    LossWeights,
    VARIANT_CONFIGS,
    build_e5_examples,
    compute_training_loss,
    cuda_preflight_status,
    examples_to_batch,
    parse_state_configs,
    state_effective_rank,
    validate_e5_examples,
)


def test_dataset_builds_and_labels_valid() -> None:
    examples = build_e5_examples(50, 32, seed=7, split="smoke")
    audit = validate_e5_examples(examples)

    assert audit["passes"], audit["failures"]
    assert all(0 <= ex.label < K_CANDIDATES for ex in examples)
    assert len(set(ex.family for ex in examples)) == 10
    assert all(len(ex.active_fact_targets) == len(ex.input_ids) for ex in examples)
    assert all(len(ex.stale_fact_targets) == len(ex.input_ids) for ex in examples)
    assert all(len(ex.support_fact_targets) == len(ex.input_ids) for ex in examples)


def test_teacher_can_attend_to_full_sequence() -> None:
    examples = build_e5_examples(4, 16, seed=8, split="smoke")
    batch = examples_to_batch(examples, "cpu")
    teacher = FittedFullContextQKVTeacher(d_model=32)
    out = teacher(batch)

    assert out["logits"].shape == (4, K_CANDIDATES)
    assert int(out["diagnostics"]["attended_token_count"].item()) == 16 * STEP_TOKENS
    assert int(out["diagnostics"]["current_only_token_count"].item()) == STEP_TOKENS


def test_student_forward_receives_only_current_step_and_state() -> None:
    torch.manual_seed(0)
    student = IntegratedRollingStateCleanQKVStudent(d_model=32, n_layers=1, n_heads=4)
    examples = build_e5_examples(3, 8, seed=9, split="smoke")
    batch = examples_to_batch(examples, "cpu")
    state = student.initial_state_for_batch(3, torch.device("cpu"))
    out = student.forward_step(batch.input_ids[:, 0, :], state)

    assert out["logits"].shape == (3, K_CANDIDATES)
    assert out["state"].shape == state.shape
    assert out["route_weights"].shape == (3, 1, 8)
    assert int(out["diagnostics"]["layer_0_guided_k_token_count"].item()) == STEP_TOKENS


def test_student_does_not_receive_old_tokens_or_old_kv() -> None:
    torch.manual_seed(1)
    student = IntegratedRollingStateCleanQKVStudent(d_model=32, n_layers=1, n_heads=4)
    examples = build_e5_examples(5, 12, seed=10, split="smoke")
    batch = examples_to_batch(examples, "cpu")
    out = student.forward_sequence(batch.input_ids)

    assert int(out["diagnostics"]["old_token_kv_received"].item()) == 0
    assert int(out["diagnostics"]["growing_activation_cache"].item()) == 0
    assert int(out["diagnostics"]["final_step_k_token_count"].item()) == STEP_TOKENS


def test_state_as_kv_variant_uses_fixed_state_memory_tokens() -> None:
    torch.manual_seed(101)
    student = IntegratedRollingStateCleanQKVStudent(
        d_model=32,
        n_layers=1,
        n_heads=4,
        num_state_slots=16,
        organized_writes=True,
        state_as_kv=True,
        disable_state_mean_paths=True,
        slot_cross_update=True,
        strict_routed_writes=True,
    )
    batch = examples_to_batch(build_e5_examples(3, 8, seed=101, split="smoke"), "cpu")
    out = student.forward_sequence(batch.input_ids)

    assert int(out["diagnostics"]["old_token_kv_received"].item()) == 0
    assert int(out["diagnostics"]["growing_activation_cache"].item()) == 0
    assert int(out["diagnostics"]["used_state_as_kv"].item()) == 1
    assert int(out["diagnostics"]["final_step_state_kv_token_count"].item()) == 16
    assert int(out["diagnostics"]["final_step_k_token_count"].item()) == STEP_TOKENS + 16


def test_learned_slot_geometry_adds_static_attention_parameters() -> None:
    torch.manual_seed(102)
    student = IntegratedRollingStateCleanQKVStudent(
        d_model=32,
        n_layers=1,
        n_heads=4,
        num_state_slots=16,
        organized_writes=True,
        state_as_kv=True,
        disable_state_mean_paths=True,
        learned_slot_geometry=True,
        static_key_scale=0.8,
        dynamic_key_scale=0.7,
    )
    batch = examples_to_batch(build_e5_examples(3, 8, seed=102, split="smoke"), "cpu")
    out = student.forward_sequence(batch.input_ids)

    assert hasattr(student, "static_read_key")
    assert hasattr(student.layers[0], "static_state_key")
    assert int(out["diagnostics"]["used_state_as_kv"].item()) == 1
    assert int(out["diagnostics"]["final_step_state_kv_token_count"].item()) == 16


def test_possibility_geometry_reports_dense_gate_usage() -> None:
    torch.manual_seed(103)
    student = IntegratedRollingStateCleanQKVStudent(
        d_model=32,
        n_layers=1,
        n_heads=4,
        num_state_slots=16,
        organized_writes=True,
        state_as_kv=True,
        disable_state_mean_paths=True,
        learned_slot_geometry=True,
        geometry_basis_count=16,
        geometry_gate_density_target=0.55,
        geometry_gate_std_floor=0.04,
    )
    batch = examples_to_batch(build_e5_examples(3, 8, seed=103, split="smoke"), "cpu")
    out = student.forward_sequence(batch.input_ids)
    diag = out["diagnostics"]

    assert hasattr(student, "read_geometry_basis_key")
    assert hasattr(student.layers[0], "geometry_basis_key")
    assert int(diag["used_state_as_kv"].item()) == 1
    assert diag["geometry_gate_density_mean"].item() > 0.0
    assert diag["geometry_gate_active_fraction"].item() > 0.0
    assert abs(diag["geometry_gate_density_target"].item() - 0.55) < 1e-6


def test_state_size_is_fixed_across_sequence_length() -> None:
    torch.manual_seed(2)
    student = IntegratedRollingStateCleanQKVStudent(d_model=64, n_layers=1, n_heads=4, num_state_slots=16)
    shapes = []
    for length in (8, 16, 32):
        batch = examples_to_batch(build_e5_examples(2, length, seed=length, split="smoke"), "cpu")
        out = student.forward_sequence(batch.input_ids)
        shapes.append(tuple(out["state"].shape[1:]))

    assert shapes == [(16, 64), (16, 64), (16, 64)]


def test_larger_state_configs_parse() -> None:
    configs = parse_state_configs("16x64,32x64,32x128")

    assert [config.label for config in configs] == ["16x64", "32x64", "32x128"]


def test_state_probes_decode_fact_targets_for_variable_slots() -> None:
    torch.manual_seed(22)
    student = IntegratedRollingStateCleanQKVStudent(d_model=64, n_layers=1, n_heads=4, num_state_slots=16)
    batch = examples_to_batch(build_e5_examples(3, 8, seed=22, split="smoke"), "cpu")
    out = student.forward_sequence(batch.input_ids)
    probes = student.probe_state_facts(out["states"])

    assert probes["value_logits"].shape == (3, 8, 8, K_CANDIDATES)
    assert probes["active_logits"].shape == batch.active_fact_targets.shape
    assert probes["stale_logits"].shape == batch.stale_fact_targets.shape


def test_combined_loss_records_probe_diversity_and_ablation_terms() -> None:
    torch.manual_seed(23)
    student = IntegratedRollingStateCleanQKVStudent(d_model=32, n_layers=1, n_heads=4, num_state_slots=16)
    teacher = FittedFullContextQKVTeacher(d_model=32)
    batch = examples_to_batch(build_e5_examples(4, 8, seed=23, split="smoke"), "cpu")
    loss, metrics, _ = compute_training_loss(
        student,
        batch,
        teacher,
        LossWeights(answer=1.0, state_usefulness=0.5, slot_diversity=0.2, state_ablation=0.1),
    )

    assert torch.isfinite(loss)
    assert metrics["state_usefulness_loss"] > 0.0
    assert metrics["slot_diversity_loss"] >= 0.0
    assert "state_ablation_loss" in metrics


def test_orthogonality_and_usage_losses_compute() -> None:
    torch.manual_seed(123)
    student = IntegratedRollingStateCleanQKVStudent(d_model=32, n_layers=1, n_heads=4, num_state_slots=16, organized_writes=True)
    teacher = FittedFullContextQKVTeacher(d_model=32)
    batch = examples_to_batch(build_e5_examples(4, 8, seed=123, split="smoke"), "cpu")
    loss, metrics, out = compute_training_loss(
        student,
        batch,
        teacher,
        LossWeights(
            answer=1.0,
            slot_cosine_decorrelation=0.2,
            slot_covariance=0.1,
            effective_rank=0.2,
            write_balance=0.1,
            read_balance=0.1,
            route_entropy_floor=0.1,
            load_balance=0.1,
        ),
    )

    assert torch.isfinite(loss)
    assert metrics["slot_cosine_decorrelation_loss"] >= 0.0
    assert metrics["slot_covariance_loss"] >= 0.0
    assert metrics["effective_rank_loss"] >= 0.0
    assert metrics["write_balance_loss"] >= 0.0
    assert metrics["read_balance_loss"] >= 0.0
    assert state_effective_rank(out["states"]) > 0.0


def test_organized_loss_records_slot_routing() -> None:
    torch.manual_seed(24)
    student = IntegratedRollingStateCleanQKVStudent(d_model=32, n_layers=1, n_heads=4, num_state_slots=16)
    teacher = FittedFullContextQKVTeacher(d_model=32)
    batch = examples_to_batch(build_e5_examples(4, 8, seed=24, split="smoke"), "cpu")
    loss, metrics, out = compute_training_loss(
        student,
        batch,
        teacher,
        LossWeights(answer=1.0, state_usefulness=0.2, slot_routing=0.5),
    )

    assert torch.isfinite(loss)
    assert metrics["slot_routing_loss"] > 0.0
    assert out["route_weights"].shape == (4, 8, 1, 16)


def test_route_shuffle_control_runs() -> None:
    torch.manual_seed(124)
    student = IntegratedRollingStateCleanQKVStudent(d_model=32, n_layers=1, n_heads=4, num_state_slots=16, organized_writes=True)
    batch = examples_to_batch(build_e5_examples(4, 12, seed=124, split="smoke"), "cpu")
    normal = student.forward_sequence(batch.input_ids)
    shuffled = student.forward_sequence(batch.input_ids, controls={"route_shuffle": True})

    assert normal["logits"].shape == shuffled["logits"].shape
    assert shuffled["route_weights"].shape == normal["route_weights"].shape


def test_variant_config_applies_dropout_and_temperature() -> None:
    cfg = VARIANT_CONFIGS["C1_route_temp_high_to_low"]
    student = IntegratedRollingStateCleanQKVStudent(
        d_model=32,
        n_layers=1,
        n_heads=4,
        num_state_slots=16,
        organized_writes=True,
        route_temperature_start=cfg.route_temperature_start,
        route_temperature_end=cfg.route_temperature_end,
    )
    student.set_training_progress(1.0)

    assert abs(student.layers[0].route_temperature - float(cfg.route_temperature_end)) < 1e-6


def test_state_update_changes_s_t() -> None:
    torch.manual_seed(3)
    student = IntegratedRollingStateCleanQKVStudent(d_model=32, n_layers=1, n_heads=4)
    batch = examples_to_batch(build_e5_examples(2, 8, seed=11, split="smoke"), "cpu")
    state0 = student.initial_state_for_batch(2, torch.device("cpu"))
    out = student.forward_step(batch.input_ids[:, 0, :], state0)

    assert not torch.allclose(state0, out["state"])
    assert float(out["diagnostics"]["state_update_norm"].item()) > 0.0


def test_state_reset_control_changes_outputs_on_memory_dependent_examples() -> None:
    torch.manual_seed(4)
    student = IntegratedRollingStateCleanQKVStudent(d_model=32, n_layers=1, n_heads=4)
    examples = build_e5_examples(8, 16, seed=12, split="smoke", task_families=("entity_binding",))
    batch = examples_to_batch(examples, "cpu")
    normal = student.forward_sequence(batch.input_ids)["logits"]
    reset = student.forward_sequence(batch.input_ids, controls={"state_reset_every_step": True})["logits"]

    assert not torch.allclose(normal, reset)


def test_randomized_labels_collapse_in_tiny_sanity() -> None:
    torch.manual_seed(5)
    examples = build_e5_examples(128, 8, seed=13, split="smoke")
    labels = torch.tensor([ex.label for ex in examples])
    random_labels = torch.randint(0, K_CANDIDATES, labels.shape)
    acc = (random_labels == labels).float().mean().item()

    assert acc < 0.30


def test_cuda_check_stops_correctly_if_cuda_unavailable() -> None:
    status = cuda_preflight_status(device="cuda", cuda_available=False)

    assert not status["passes"]
    assert status["decision"] == DECISION_CUDA_REQUIRED
