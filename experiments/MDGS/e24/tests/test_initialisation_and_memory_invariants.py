"""Pre-training invariant tests for the DIGIT Extrapolation E24 models."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.DIGIT.Extrapolation.e24.extrapolation.config import E24Config
from experiments.DIGIT.Extrapolation.e24.extrapolation.initialisation import initialise_e24_models


SEEDS = (0, 1, 2, 3, 4)


def _make_config(seed: int = 0) -> E24Config:
    return E24Config(seed=seed)


def _make_dummy_batch(batch_size: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    part_a = torch.tensor([0, 1, 2, 3][:batch_size], dtype=torch.long)
    part_b = torch.tensor([4, 5, 6, 7][:batch_size], dtype=torch.long)
    labels = torch.tensor([0, 1, 2, 3][:batch_size], dtype=torch.long)
    return part_a, part_b, labels


def test_models_2_and_3_have_identical_non_memory_trainable_tensors_at_step_0_for_every_seed() -> None:
    for seed in SEEDS:
        initialized = initialise_e24_models(_make_config(seed))
        e24_named_parameters = dict(initialized.e24.named_parameters())
        ablation_named_parameters = {
            name: parameter
            for name, parameter in initialized.ablation.named_parameters()
            if "raw_slots" not in name
        }

        assert set(e24_named_parameters) == set(ablation_named_parameters)
        for name, parameter in e24_named_parameters.items():
            assert torch.equal(parameter, ablation_named_parameters[name]), name


def test_models_2_and_3_have_identical_effective_memory_tensors_at_step_0_for_every_seed() -> None:
    for seed in SEEDS:
        initialized = initialise_e24_models(_make_config(seed))

        assert torch.equal(initialized.e24.layer1.prototypes, initialized.ablation.layer1.raw_slots)
        assert torch.equal(initialized.e24.layer2.prototypes, initialized.ablation.layer2.raw_slots)
        assert torch.allclose(
            initialized.e24.layer1.effective_memory(),
            initialized.ablation.layer1.effective_memory().detach(),
        )
        assert torch.allclose(
            initialized.e24.layer2.effective_memory(),
            initialized.ablation.layer2.effective_memory().detach(),
        )


def test_model_2_prototypes_are_buffers_and_excluded_from_the_optimizer() -> None:
    initialized = initialise_e24_models(_make_config(0))
    model = initialized.e24
    parameter_names = {name for name, _ in model.named_parameters()}
    buffer_names = {name for name, _ in model.named_buffers()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer_param_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}

    assert "layer1.prototypes" not in parameter_names
    assert "layer2.prototypes" not in parameter_names
    assert "layer1.valid_mask" not in parameter_names
    assert "layer2.valid_mask" not in parameter_names
    assert "layer1.prototypes" in buffer_names
    assert "layer2.prototypes" in buffer_names
    assert "layer1.valid_mask" in buffer_names
    assert "layer2.valid_mask" in buffer_names
    assert id(model.layer1.prototypes) not in optimizer_param_ids
    assert id(model.layer2.prototypes) not in optimizer_param_ids
    assert model.layer1.prototypes.requires_grad is False
    assert model.layer2.prototypes.requires_grad is False


def test_model_3_raw_slots_are_parameters_and_included_in_the_optimizer() -> None:
    initialized = initialise_e24_models(_make_config(0))
    model = initialized.ablation
    parameter_names = {name for name, _ in model.named_parameters()}
    buffer_names = {name for name, _ in model.named_buffers()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer_param_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}

    assert "layer1.raw_slots" in parameter_names
    assert "layer2.raw_slots" in parameter_names
    assert "layer1.raw_slots" not in buffer_names
    assert "layer2.raw_slots" not in buffer_names
    assert id(model.layer1.raw_slots) in optimizer_param_ids
    assert id(model.layer2.raw_slots) in optimizer_param_ids
    assert model.layer1.raw_slots.requires_grad
    assert model.layer2.raw_slots.requires_grad


def test_model_2_prototype_updates_use_detached_semantic_activations_only() -> None:
    initialized = initialise_e24_models(_make_config(0))
    model = initialized.e24
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    part_a, part_b, labels = _make_dummy_batch()

    optimizer.zero_grad(set_to_none=True)
    outputs = model(part_a, part_b)
    loss = F.cross_entropy(outputs.logits, labels)
    loss.backward()

    assert model.layer1.prototypes.grad is None
    assert model.layer2.prototypes.grad is None
    old_layer1 = model.layer1.prototypes.detach().clone()
    old_layer2 = model.layer2.prototypes.detach().clone()

    model.update_prototypes(outputs.layer1_semantic, outputs.layer2_semantic)

    assert model.layer1.last_update_inputs
    assert model.layer2.last_update_inputs
    assert all(tensor.requires_grad is False for tensor in model.layer1.last_update_inputs)
    assert all(tensor.requires_grad is False for tensor in model.layer2.last_update_inputs)
    assert all(tensor.grad_fn is None for tensor in model.layer1.last_update_inputs)
    assert all(tensor.grad_fn is None for tensor in model.layer2.last_update_inputs)
    assert not torch.equal(model.layer1.prototypes, old_layer1)
    assert not torch.equal(model.layer2.prototypes, old_layer2)


def test_model_3_slots_update_only_through_gradient_descent_without_manual_memory_rules() -> None:
    initialized = initialise_e24_models(_make_config(0))
    model = initialized.ablation
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    part_a, part_b, labels = _make_dummy_batch()

    assert not hasattr(model.layer1, "update_prototypes")
    assert not hasattr(model.layer2, "update_prototypes")
    assert not hasattr(model.layer1, "_apply_repulsion")
    assert not hasattr(model.layer2, "_apply_repulsion")
    assert not hasattr(model.layer1, "memory_lambda")
    assert not hasattr(model.layer2, "memory_lambda")

    old_layer1 = model.layer1.raw_slots.detach().clone()
    old_layer2 = model.layer2.raw_slots.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    outputs = model(part_a, part_b)
    loss = F.cross_entropy(outputs.logits, labels)
    loss.backward()

    assert model.layer1.raw_slots.grad is not None
    assert model.layer2.raw_slots.grad is not None
    assert torch.count_nonzero(model.layer1.raw_slots.grad) > 0
    assert torch.count_nonzero(model.layer2.raw_slots.grad) > 0

    optimizer.step()

    assert not torch.equal(model.layer1.raw_slots, old_layer1)
    assert not torch.equal(model.layer2.raw_slots, old_layer2)


def test_there_is_no_empty_slot_warmup_and_both_models_start_fully_valid_with_unit_gate() -> None:
    for seed in SEEDS:
        initialized = initialise_e24_models(_make_config(seed))
        part_a, part_b, _labels = _make_dummy_batch()

        e24_outputs = initialized.e24(part_a, part_b)
        ablation_outputs = initialized.ablation(part_a, part_b)

        assert bool(initialized.e24.layer1.valid_mask.all())
        assert bool(initialized.e24.layer2.valid_mask.all())
        assert bool(initialized.ablation.layer1.valid_mask.all())
        assert bool(initialized.ablation.layer2.valid_mask.all())
        assert initialized.e24.layer1.occupancy == 32
        assert initialized.e24.layer2.occupancy == 32
        assert initialized.ablation.layer1.occupancy == 32
        assert initialized.ablation.layer2.occupancy == 32
        assert torch.allclose(e24_outputs.layer1_occupancy_gate, torch.tensor(1.0))
        assert torch.allclose(e24_outputs.layer2_occupancy_gate, torch.tensor(1.0))
        assert torch.allclose(ablation_outputs.layer1_occupancy_gate, torch.tensor(1.0))
        assert torch.allclose(ablation_outputs.layer2_occupancy_gate, torch.tensor(1.0))


def test_gate_parameters_are_nn_parameters_and_included_in_the_optimizer_for_both_gated_models() -> None:
    initialized = initialise_e24_models(_make_config(0))

    for model in (initialized.e24, initialized.ablation):
        parameter_names = {name for name, _ in model.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer_param_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}

        assert "g_1" in parameter_names
        assert "g_2" in parameter_names
        assert isinstance(model.g_1, torch.nn.Parameter)
        assert isinstance(model.g_2, torch.nn.Parameter)
        assert model.g_1.requires_grad
        assert model.g_2.requires_grad
        assert id(model.g_1) in optimizer_param_ids
        assert id(model.g_2) in optimizer_param_ids


def test_initial_gate_sigmoids_are_below_point_zero_one_for_all_seeds_and_both_gated_models() -> None:
    for seed in SEEDS:
        initialized = initialise_e24_models(_make_config(seed))
        for model in (initialized.e24, initialized.ablation):
            assert torch.sigmoid(model.g_1).item() < 0.01
            assert torch.sigmoid(model.g_2).item() < 0.01
