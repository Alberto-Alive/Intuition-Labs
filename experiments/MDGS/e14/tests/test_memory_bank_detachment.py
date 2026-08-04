from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extrapolation.models import FrozenActivationBankMLP
from extrapolation.config import E14Config
from extrapolation.experiment import build_frozen_banks, make_model, train_single_seed_artifacts


def test_memory_bank_receives_zero_gradient_in_full_training_step() -> None:
    torch.manual_seed(7)

    model = FrozenActivationBankMLP(
        part_a_vocab_size=8,
        part_b_vocab_size=8,
        embedding_dim=4,
        hidden_dim=6,
        num_classes=4,
        bank_capacity=5,
    )

    model.seed_banks(
        layer1_bank=torch.randn(5, 6),
        layer2_bank=torch.randn(5, 6),
    )

    initial_layer1_bank = model.layer1_bank.detach().clone()
    initial_layer2_bank = model.layer2_bank.detach().clone()

    parameter_names = {name for name, _ in model.named_parameters()}
    buffer_names = {name for name, _ in model.named_buffers()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer_param_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}

    assert "layer1_bank" not in parameter_names
    assert "layer2_bank" not in parameter_names
    assert "layer1_bank" in buffer_names
    assert "layer2_bank" in buffer_names
    assert id(model.layer1_bank) not in optimizer_param_ids
    assert id(model.layer2_bank) not in optimizer_param_ids
    assert model.layer1_bank.requires_grad is False
    assert model.layer2_bank.requires_grad is False
    assert model.layer1_bank.grad is None
    assert model.layer2_bank.grad is None

    part_a = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    part_b = torch.tensor([3, 2, 1, 0], dtype=torch.long)
    targets = torch.tensor([1, 0, 3, 2], dtype=torch.long)

    optimizer.zero_grad(set_to_none=True)
    outputs = model(part_a, part_b, return_read_only_certainty=True)
    loss = F.cross_entropy(outputs.logits, targets)
    loss.backward()

    assert outputs.layer1_certainty is not None
    assert outputs.layer2_certainty is not None
    assert outputs.layer1_certainty.requires_grad is False
    assert outputs.layer2_certainty.requires_grad is False
    assert outputs.layer1_certainty.grad_fn is None
    assert outputs.layer2_certainty.grad_fn is None
    assert model.layer1_bank.grad is None
    assert model.layer2_bank.grad is None

    nonzero_parameter_gradients = []
    for parameter in model.parameters():
        if parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0:
            nonzero_parameter_gradients.append(parameter.grad)
    assert nonzero_parameter_gradients

    optimizer.step()

    assert model.layer1_bank.grad is None
    assert model.layer2_bank.grad is None
    assert torch.equal(model.layer1_bank, initial_layer1_bank)
    assert torch.equal(model.layer2_bank, initial_layer2_bank)


def test_built_frozen_bank_receives_zero_gradient_in_full_training_step() -> None:
    config = E14Config(
        seed=0,
        train_repeats_per_seen_combo=8,
        test_repeats_per_combo=4,
        batch_size=16,
        embedding_dim=8,
        hidden_dim=12,
        bank_capacity=64,
        max_epochs=60,
        loss_patience_epochs=10,
    )
    artifacts = train_single_seed_artifacts(config)
    bank = build_frozen_banks(artifacts.model, artifacts.dataset.train, config)

    model = make_model(config, bank_capacity=artifacts.dataset.train.size)
    trainable_state = {
        name: tensor
        for name, tensor in artifacts.model.state_dict().items()
        if not name.startswith("layer1_bank")
        and not name.startswith("layer2_bank")
    }
    model.load_state_dict(trainable_state, strict=False)
    model.seed_banks(bank.layer1_bank, bank.layer2_bank)

    initial_layer1_bank = model.layer1_bank.detach().clone()
    initial_layer2_bank = model.layer2_bank.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    assert tuple(model.layer1_bank.shape) == (artifacts.dataset.train.size, config.hidden_dim)
    assert tuple(model.layer2_bank.shape) == (artifacts.dataset.train.size, config.hidden_dim)
    assert model.layer1_bank.requires_grad is False
    assert model.layer2_bank.requires_grad is False
    assert model.layer1_bank.grad is None
    assert model.layer2_bank.grad is None

    part_a = artifacts.dataset.train.part_a[:16]
    part_b = artifacts.dataset.train.part_b[:16]
    targets = artifacts.dataset.train.labels[:16]

    optimizer.zero_grad(set_to_none=True)
    outputs = model(part_a, part_b, return_read_only_certainty=True)
    loss = F.cross_entropy(outputs.logits, targets)
    loss.backward()

    assert outputs.layer1_certainty is not None
    assert outputs.layer2_certainty is not None
    assert outputs.layer1_certainty.requires_grad is False
    assert outputs.layer2_certainty.requires_grad is False
    assert model.layer1_bank.grad is None
    assert model.layer2_bank.grad is None

    optimizer.step()

    assert model.layer1_bank.grad is None
    assert model.layer2_bank.grad is None
    assert torch.equal(model.layer1_bank, initial_layer1_bank)
    assert torch.equal(model.layer2_bank, initial_layer2_bank)
