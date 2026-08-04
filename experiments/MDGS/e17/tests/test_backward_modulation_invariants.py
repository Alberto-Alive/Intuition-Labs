"""Pre-implementation invariant tests for DIGIT Extrapolation E17."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.DIGIT.Extrapolation.e16.extrapolation.models import ForwardGatedLiveBankMLP


ALPHA = 10.0
BETA = 0.1


class BackwardModulationProbe(ForwardGatedLiveBankMLP):
    """Minimal E17 probe that adds only the backward modulation hooks."""

    def __init__(self) -> None:
        super().__init__(
            part_a_vocab_size=4,
            part_b_vocab_size=4,
            embedding_dim=2,
            hidden_dim=3,
            num_classes=2,
            bank_capacity=8,
            min_bank_occupancy=1,
        )
        self.activation_hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self.parameter_hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self.activation_hook_target_ids: list[int] = []
        self.layer1_hook_saw_hidden1_grads_unfinalized: bool | None = None
        self.layer2_hook_saw_hidden2_grads_unfinalized: bool | None = None

    @staticmethod
    def build_modulation_weights(losses: torch.Tensor, certainty: torch.Tensor) -> torch.Tensor:
        """Build detached per-example modulation weights from unreduced losses."""
        detached_losses = losses.detach()
        detached_certainty = certainty.detach()
        if torch.isnan(detached_certainty).any():
            normalized_scale = torch.ones_like(detached_certainty)
        else:
            raw_scale = ALPHA * detached_certainty + BETA * (1.0 - detached_certainty)
            normalized_scale = raw_scale / raw_scale.mean()
        return detached_losses * normalized_scale

    def attach_gradient_modulation_hooks(
        self,
        *,
        losses: torch.Tensor,
        layer1_certainty: torch.Tensor,
        layer2_certainty: torch.Tensor,
        layer1_pre_gate_activation: torch.Tensor,
        layer2_pre_gate_activation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attach modulation hooks to the pre-gate activations only."""
        layer1_scale = self.build_modulation_weights(losses, layer1_certainty)
        layer2_scale = self.build_modulation_weights(losses, layer2_certainty)

        def layer1_hook(grad: torch.Tensor) -> torch.Tensor:
            self.layer1_hook_saw_hidden1_grads_unfinalized = (
                self.hidden1.weight.grad is None and self.hidden1.bias.grad is None
            )
            return grad * layer1_scale.unsqueeze(-1)

        def layer2_hook(grad: torch.Tensor) -> torch.Tensor:
            self.layer2_hook_saw_hidden2_grads_unfinalized = (
                self.hidden2.weight.grad is None and self.hidden2.bias.grad is None
            )
            return grad * layer2_scale.unsqueeze(-1)

        self.activation_hook_handles = [
            layer1_pre_gate_activation.register_hook(layer1_hook),
            layer2_pre_gate_activation.register_hook(layer2_hook),
        ]
        self.activation_hook_target_ids = [id(layer1_pre_gate_activation), id(layer2_pre_gate_activation)]
        return layer1_scale, layer2_scale


def _make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    part_a = torch.tensor([0, 1], dtype=torch.long)
    part_b = torch.tensor([1, 2], dtype=torch.long)
    part_a_noise = torch.zeros((2, 2), dtype=torch.float32)
    part_b_noise = torch.zeros((2, 2), dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    return part_a, part_b, part_a_noise, part_b_noise, labels


def _make_model() -> BackwardModulationProbe:
    torch.manual_seed(0)
    model = BackwardModulationProbe()
    with torch.no_grad():
        model.layer1_bank.update(
            torch.tensor(
                [
                    [0.5, 0.1, 0.2],
                    [0.1, 0.6, 0.3],
                ],
                dtype=torch.float32,
            )
        )
        model.layer2_bank.update(
            torch.tensor(
                [
                    [0.4, 0.2, 0.1],
                    [0.2, 0.5, 0.3],
                ],
                dtype=torch.float32,
            )
        )
    return model


def test_hooks_are_attached_to_pre_gate_activation_tensors_not_parameters() -> None:
    """E17 must attach tensor hooks to pre-gate activations, never to parameters."""
    model = _make_model()
    part_a, part_b, part_a_noise, part_b_noise, labels = _make_inputs()
    outputs = model(part_a, part_b, part_a_noise=part_a_noise, part_b_noise=part_b_noise)
    losses = F.cross_entropy(outputs.logits, labels, reduction="none")

    model.attach_gradient_modulation_hooks(
        losses=losses,
        layer1_certainty=outputs.layer1_certainty,
        layer2_certainty=outputs.layer2_certainty,
        layer1_pre_gate_activation=outputs.layer1_pre_gate_activation,
        layer2_pre_gate_activation=outputs.layer2_pre_gate_activation,
    )

    parameter_ids = {id(parameter) for parameter in model.parameters()}

    assert len(model.activation_hook_handles) == 2
    assert len(model.parameter_hook_handles) == 0
    assert id(outputs.layer1_pre_gate_activation) in model.activation_hook_target_ids
    assert id(outputs.layer2_pre_gate_activation) in model.activation_hook_target_ids
    assert all(target_id not in parameter_ids for target_id in model.activation_hook_target_ids)


def test_hooks_fire_before_parameter_gradients_are_finalized() -> None:
    """The activation hook must run before the corresponding layer parameter grads exist."""
    model = _make_model()
    part_a, part_b, part_a_noise, part_b_noise, labels = _make_inputs()

    model.zero_grad(set_to_none=True)
    outputs = model(part_a, part_b, part_a_noise=part_a_noise, part_b_noise=part_b_noise)
    losses = F.cross_entropy(outputs.logits, labels, reduction="none")
    model.attach_gradient_modulation_hooks(
        losses=losses,
        layer1_certainty=outputs.layer1_certainty,
        layer2_certainty=outputs.layer2_certainty,
        layer1_pre_gate_activation=outputs.layer1_pre_gate_activation,
        layer2_pre_gate_activation=outputs.layer2_pre_gate_activation,
    )

    losses.mean().backward()

    assert model.layer1_hook_saw_hidden1_grads_unfinalized is True
    assert model.layer2_hook_saw_hidden2_grads_unfinalized is True
    assert model.hidden1.weight.grad is not None
    assert model.hidden2.weight.grad is not None


def test_modulation_weights_use_unreduced_loss_and_detached_certainty() -> None:
    """The modulation scale must be per-example and detached from loss/certainty gradients."""
    losses = torch.tensor([1.25, 2.5], dtype=torch.float32, requires_grad=True)
    certainty = torch.tensor([0.2, 0.8], dtype=torch.float32, requires_grad=True)

    scale = BackwardModulationProbe.build_modulation_weights(losses, certainty)
    raw_scale = torch.tensor(
        [
            ALPHA * 0.2 + BETA * 0.8,
            ALPHA * 0.8 + BETA * 0.2,
        ],
        dtype=torch.float32,
    )
    normalized_scale = raw_scale / raw_scale.mean()
    expected = torch.tensor(
        [
            1.25 * float(normalized_scale[0].item()),
            2.5 * float(normalized_scale[1].item()),
        ],
        dtype=torch.float32,
    )

    activation = torch.ones((2, 3), dtype=torch.float32, requires_grad=True)
    activation.register_hook(lambda grad: grad * scale.unsqueeze(-1))
    activation.sum().backward()

    assert scale.shape == losses.shape
    assert torch.allclose(scale, expected)
    assert scale.requires_grad is False
    assert scale.grad_fn is None
    assert losses.grad is None
    assert certainty.grad is None


def test_invalid_certainty_produces_neutral_normalized_scale_one() -> None:
    """NaN certainty must map to a neutral normalized scale of exactly 1.0."""
    losses = torch.tensor([1.0, 2.5], dtype=torch.float64)
    certainty = torch.tensor([float("nan"), float("nan")], dtype=torch.float64)

    scale = BackwardModulationProbe.build_modulation_weights(losses, certainty)
    expected = torch.tensor([1.0, 2.5], dtype=torch.float64)

    assert torch.allclose(scale, expected, atol=0.0, rtol=0.0)


def test_forward_path_and_live_bank_behavior_remain_identical_to_e16() -> None:
    """With no backward modulation attached, E17 must match E16 exactly."""
    torch.manual_seed(0)
    e16_model = ForwardGatedLiveBankMLP(
        part_a_vocab_size=4,
        part_b_vocab_size=4,
        embedding_dim=2,
        hidden_dim=3,
        num_classes=2,
        bank_capacity=8,
        min_bank_occupancy=1,
    )
    with torch.no_grad():
        e16_model.layer1_bank.update(
            torch.tensor(
                [
                    [0.5, 0.1, 0.2],
                    [0.1, 0.6, 0.3],
                ],
                dtype=torch.float32,
            )
        )
        e16_model.layer2_bank.update(
            torch.tensor(
                [
                    [0.4, 0.2, 0.1],
                    [0.2, 0.5, 0.3],
                ],
                dtype=torch.float32,
            )
        )

    e17_model = BackwardModulationProbe()
    e17_model.load_state_dict(e16_model.state_dict())

    part_a, part_b, part_a_noise, part_b_noise, _ = _make_inputs()
    e16_outputs = e16_model(part_a, part_b, part_a_noise=part_a_noise, part_b_noise=part_b_noise)
    e17_outputs = e17_model(part_a, part_b, part_a_noise=part_a_noise, part_b_noise=part_b_noise)

    assert torch.allclose(e16_outputs.logits, e17_outputs.logits)
    assert torch.allclose(e16_outputs.input_embedding, e17_outputs.input_embedding)
    assert torch.allclose(e16_outputs.layer1_pre_gate_activation, e17_outputs.layer1_pre_gate_activation)
    assert torch.allclose(e16_outputs.layer2_pre_gate_activation, e17_outputs.layer2_pre_gate_activation)
    assert torch.allclose(e16_outputs.layer1_certainty, e17_outputs.layer1_certainty)
    assert torch.allclose(e16_outputs.layer2_certainty, e17_outputs.layer2_certainty)
    assert torch.allclose(e16_outputs.gate_layer1, e17_outputs.gate_layer1)
    assert torch.allclose(e16_outputs.gate_layer2, e17_outputs.gate_layer2)

    e16_model.update_live_banks(
        e16_outputs.layer1_pre_gate_activation,
        e16_outputs.layer2_pre_gate_activation,
    )
    e17_model.update_live_banks(
        e17_outputs.layer1_pre_gate_activation,
        e17_outputs.layer2_pre_gate_activation,
    )

    assert torch.allclose(e16_model.layer1_bank.entries, e17_model.layer1_bank.entries)
    assert torch.equal(e16_model.layer1_bank.valid_mask, e17_model.layer1_bank.valid_mask)
    assert torch.equal(e16_model.layer1_bank.write_pointer, e17_model.layer1_bank.write_pointer)
    assert torch.equal(e16_model.layer1_bank.valid_count, e17_model.layer1_bank.valid_count)
    assert torch.allclose(e16_model.layer2_bank.entries, e17_model.layer2_bank.entries)
    assert torch.equal(e16_model.layer2_bank.valid_mask, e17_model.layer2_bank.valid_mask)
    assert torch.equal(e16_model.layer2_bank.write_pointer, e17_model.layer2_bank.write_pointer)
    assert torch.equal(e16_model.layer2_bank.valid_count, e17_model.layer2_bank.valid_count)
