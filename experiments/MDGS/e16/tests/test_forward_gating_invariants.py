"""Pre-implementation invariant tests for the DIGIT Extrapolation E16 gate."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class RecordingBank:
    """Minimal bank stub that records query/update inputs for invariant tests."""

    def __init__(self, certainty_value: float) -> None:
        self.certainty_value = certainty_value
        self.query_inputs: list[torch.Tensor] = []
        self.update_inputs: list[torch.Tensor] = []

    def query(self, activations: torch.Tensor) -> torch.Tensor:
        self.query_inputs.append(activations.detach().clone())
        if math.isnan(self.certainty_value):
            return torch.full((activations.shape[0],), float("nan"), dtype=activations.dtype, device=activations.device)
        return torch.full(
            (activations.shape[0],),
            self.certainty_value,
            dtype=activations.dtype,
            device=activations.device,
        )

    def update(self, activations: torch.Tensor) -> None:
        self.update_inputs.append(activations.clone())


class MinimalE15Baseline(nn.Module):
    """Ungated baseline used to verify initial-fill equivalence."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(3, 3, bias=False)
        self.out = nn.Linear(3, 1, bias=False)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.relu(self.hidden(inputs))
        logits = self.out(hidden)
        return logits, hidden


class MinimalE16GatedModel(nn.Module):
    """Minimal gated forward path used only for E16 invariant tests."""

    def __init__(self, bank: RecordingBank) -> None:
        super().__init__()
        self.hidden = nn.Linear(3, 3, bias=False)
        self.out = nn.Linear(3, 1, bias=False)
        self.bank = bank

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pre_gate = torch.relu(self.hidden(inputs))
        certainty = self.bank.query(pre_gate)
        gate = torch.where(torch.isnan(certainty), torch.ones_like(certainty), certainty.square())
        gated = pre_gate * gate.unsqueeze(-1)
        logits = self.out(gated)
        return logits, pre_gate, certainty, gate, gated

    def update_bank(self, pre_gate: torch.Tensor) -> None:
        self.bank.update(pre_gate.detach())


def _make_identity_weight_models(bank_value: float) -> tuple[MinimalE16GatedModel, torch.Tensor]:
    bank = RecordingBank(certainty_value=bank_value)
    model = MinimalE16GatedModel(bank)
    with torch.no_grad():
        model.hidden.weight.copy_(torch.eye(3))
        model.out.weight.copy_(torch.ones((1, 3)))
    inputs = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    return model, inputs


def test_certainty_is_computed_from_pre_gate_activations() -> None:
    """The bank query must receive the pre-gate hidden activation, not the gated one."""
    model, inputs = _make_identity_weight_models(bank_value=0.5)

    _, pre_gate, _, _, gated = model(inputs)

    assert len(model.bank.query_inputs) == 1
    assert torch.allclose(model.bank.query_inputs[0], pre_gate.detach())
    assert not torch.allclose(model.bank.query_inputs[0], gated.detach())


def test_gating_happens_after_certainty_and_uses_certainty_squared() -> None:
    """The gate must equal certainty squared and scale the hidden activation multiplicatively."""
    model, inputs = _make_identity_weight_models(bank_value=0.5)

    logits, pre_gate, certainty, gate, gated = model(inputs)

    expected_gate = torch.tensor([0.25], dtype=torch.float32)
    expected_gated = pre_gate.detach() * 0.25
    expected_logits = torch.tensor([[1.5]], dtype=torch.float32)

    assert torch.allclose(certainty.detach(), torch.tensor([0.5], dtype=torch.float32))
    assert torch.allclose(gate.detach(), expected_gate)
    assert torch.allclose(gated.detach(), expected_gated)
    assert torch.allclose(logits.detach(), expected_logits)


def test_bank_updates_use_pre_gate_detached_activations() -> None:
    """The bank update must receive the pre-gate detached activation after the optimizer step."""
    model, inputs = _make_identity_weight_models(bank_value=0.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    optimizer.zero_grad(set_to_none=True)
    logits, pre_gate, _, _, gated = model(inputs)
    loss = logits.sum()
    loss.backward()
    optimizer.step()
    model.update_bank(pre_gate)

    assert len(model.bank.update_inputs) == 1
    update_tensor = model.bank.update_inputs[0]
    assert update_tensor.requires_grad is False
    assert update_tensor.grad_fn is None
    assert torch.allclose(update_tensor, pre_gate.detach())
    assert not torch.allclose(update_tensor, gated.detach())


def test_invalid_certainty_produces_gate_one() -> None:
    """NaN certainty during the initial fill period must produce an identity gate."""
    model, inputs = _make_identity_weight_models(bank_value=float("nan"))

    _, pre_gate, certainty, gate, gated = model(inputs)

    assert torch.isnan(certainty).all()
    assert torch.allclose(gate.detach(), torch.ones_like(gate))
    assert torch.allclose(gated.detach(), pre_gate.detach())


def test_e16_matches_e15_during_initial_fill_period() -> None:
    """With invalid certainty, the gated model must behave identically to the e15 baseline."""
    baseline = MinimalE15Baseline()
    gated = MinimalE16GatedModel(RecordingBank(certainty_value=float("nan")))

    with torch.no_grad():
        baseline.hidden.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
        baseline.out.weight.copy_(torch.tensor([[0.5, -0.25, 1.5]]))
        gated.hidden.weight.copy_(baseline.hidden.weight)
        gated.out.weight.copy_(baseline.out.weight)

    inputs = torch.tensor([[2.0, 1.0, 3.0]], dtype=torch.float32)
    baseline_logits, baseline_hidden = baseline(inputs)
    gated_logits, pre_gate, certainty, gate, gated_hidden = gated(inputs)

    assert torch.isnan(certainty).all()
    assert torch.allclose(gate.detach(), torch.ones_like(gate))
    assert torch.allclose(pre_gate.detach(), baseline_hidden.detach())
    assert torch.allclose(gated_hidden.detach(), baseline_hidden.detach())
    assert torch.allclose(gated_logits.detach(), baseline_logits.detach())
