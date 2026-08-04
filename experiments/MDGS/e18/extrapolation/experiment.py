"""Seed-0 gradient norm diagnostic helpers for DIGIT Extrapolation E18."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from experiments.DIGIT.Extrapolation.e14.extrapolation.experiment import build_dataset, make_loader
from experiments.DIGIT.Extrapolation.e16.extrapolation.config import E16Config
from experiments.DIGIT.Extrapolation.e16.extrapolation.models import ForwardGatedLiveBankMLP

from .config import E18Config
from .models import TargetedBackwardModulatedLiveBankMLP


@dataclass(frozen=True)
class GradientNormStepSummary:
    """Layer-1 gradient norm comparison for one training step."""

    step: int
    e16_gradient_norm: float
    e18_gradient_norm: float
    ratio_e18_over_e16: float
    layer1_modulated_fraction: float
    layer2_modulated_fraction: float


def set_global_determinism(seed: int) -> None:
    """Set the approved seed across Python and PyTorch."""
    random.seed(seed)
    torch.manual_seed(seed)


def make_e16_model(config: E16Config) -> ForwardGatedLiveBankMLP:
    """Instantiate the matched E16 baseline model."""
    return ForwardGatedLiveBankMLP(
        part_a_vocab_size=config.num_part_a_values,
        part_b_vocab_size=config.num_part_b_values,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        bank_capacity=config.bank_capacity,
        min_bank_occupancy=config.min_bank_occupancy,
    )


def make_e18_model(config: E18Config) -> TargetedBackwardModulatedLiveBankMLP:
    """Instantiate the targeted backward-modulated E18 model."""
    return TargetedBackwardModulatedLiveBankMLP(
        part_a_vocab_size=config.num_part_a_values,
        part_b_vocab_size=config.num_part_b_values,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        bank_capacity=config.bank_capacity,
        min_bank_occupancy=config.min_bank_occupancy,
        alpha=config.alpha,
        beta=config.beta,
        certainty_threshold=config.certainty_threshold,
    )


def _mean_layer1_gradient_norm(model: ForwardGatedLiveBankMLP) -> float:
    gradients = [model.hidden1.weight.grad, model.hidden1.bias.grad]
    norms = [gradient.norm().item() for gradient in gradients if gradient is not None]
    return float(sum(norms) / len(norms))


def run_seed0_gradient_norm_diagnostic(num_steps: int = 10) -> list[GradientNormStepSummary]:
    """Compare the first training steps of matched E16 and E18 runs under seed 0."""
    e16_config = E16Config(seed=0)
    e18_config = E18Config(seed=0)
    set_global_determinism(0)

    dataset = build_dataset(e18_config)
    loader = make_loader(dataset.train, e18_config, shuffle=True)

    e16_model = make_e16_model(e16_config)
    e18_model = make_e18_model(e18_config)
    e18_model.load_state_dict(e16_model.state_dict())

    e16_optimizer = torch.optim.Adam(e16_model.parameters(), lr=e16_config.learning_rate)
    e18_optimizer = torch.optim.Adam(e18_model.parameters(), lr=e18_config.learning_rate)

    step_summaries: list[GradientNormStepSummary] = []
    for step_index, (part_a, part_b, part_a_noise, part_b_noise, labels, _) in enumerate(loader, start=1):
        if step_index > num_steps:
            break

        e16_optimizer.zero_grad(set_to_none=True)
        e16_outputs = e16_model(
            part_a,
            part_b,
            part_a_noise=part_a_noise,
            part_b_noise=part_b_noise,
        )
        e16_loss = F.cross_entropy(e16_outputs.logits, labels)
        e16_loss.backward()
        e16_gradient_norm = _mean_layer1_gradient_norm(e16_model)
        e16_optimizer.step()
        e16_model.update_live_banks(
            e16_outputs.layer1_pre_gate_activation,
            e16_outputs.layer2_pre_gate_activation,
        )

        e18_optimizer.zero_grad(set_to_none=True)
        e18_outputs = e18_model(
            part_a,
            part_b,
            part_a_noise=part_a_noise,
            part_b_noise=part_b_noise,
        )
        e18_losses = F.cross_entropy(e18_outputs.logits, labels, reduction="none")
        e18_modulation_weights = e18_model.build_modulation_weights(e18_losses, e18_outputs)
        e18_model.attach_gradient_modulation_hooks(e18_outputs, e18_modulation_weights)
        e18_loss = e18_losses.mean()
        e18_loss.backward()
        e18_gradient_norm = _mean_layer1_gradient_norm(e18_model)
        e18_model.clear_gradient_modulation_hooks()
        e18_optimizer.step()
        e18_model.update_live_banks(
            e18_outputs.layer1_pre_gate_activation,
            e18_outputs.layer2_pre_gate_activation,
        )

        step_summaries.append(
            GradientNormStepSummary(
                step=step_index,
                e16_gradient_norm=e16_gradient_norm,
                e18_gradient_norm=e18_gradient_norm,
                ratio_e18_over_e16=e18_gradient_norm / e16_gradient_norm,
                layer1_modulated_fraction=e18_modulation_weights.layer1_modulated_fraction,
                layer2_modulated_fraction=e18_modulation_weights.layer2_modulated_fraction,
            )
        )

    return step_summaries

