"""Shared initialisation helpers for DIGIT Extrapolation E24."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from .config import E24Config
from .models import ActiveLearnedMemoryMLP, BaselineMLP, DualParameterAttentionMLP, normalize_rows


@dataclass
class E24InitialisedModels:
    """The three E24 models after applying the frozen shared initialisation rule."""

    baseline: BaselineMLP
    e24: DualParameterAttentionMLP
    ablation: ActiveLearnedMemoryMLP
    initial_layer1_slots: torch.Tensor
    initial_layer2_slots: torch.Tensor


def set_global_determinism(seed: int) -> None:
    """Set Python and PyTorch RNGs for deterministic model construction."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_e24_model(config: E24Config) -> DualParameterAttentionMLP:
    return DualParameterAttentionMLP(
        part_a_vocab_size=config.num_part_a_values,
        part_b_vocab_size=config.num_part_b_values,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        key_dim=config.key_dim,
        num_classes=config.num_classes,
        num_memory_slots=config.num_memory_slots,
        memory_lambda=config.memory_lambda,
        eps=config.memory_eps,
    )


def _make_ablation_model(config: E24Config) -> ActiveLearnedMemoryMLP:
    return ActiveLearnedMemoryMLP(
        part_a_vocab_size=config.num_part_a_values,
        part_b_vocab_size=config.num_part_b_values,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        key_dim=config.key_dim,
        num_classes=config.num_classes,
        num_memory_slots=config.num_memory_slots,
        eps=config.memory_eps,
    )


def _make_baseline_model(config: E24Config) -> BaselineMLP:
    return BaselineMLP(
        part_a_vocab_size=config.num_part_a_values,
        part_b_vocab_size=config.num_part_b_values,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
    )


@torch.no_grad()
def _copy_shared_baseline_weights(source: DualParameterAttentionMLP, target: BaselineMLP) -> None:
    target.part_a_embedding.weight.copy_(source.part_a_embedding.weight)
    target.part_b_embedding.weight.copy_(source.part_b_embedding.weight)
    target.hidden1.weight.copy_(source.layer1.semantic.weight)
    target.hidden1.bias.copy_(source.layer1.semantic.bias)
    target.hidden2.weight.copy_(source.layer2.semantic.weight)
    target.hidden2.bias.copy_(source.layer2.semantic.bias)
    target.classifier.weight.copy_(source.classifier.weight)
    target.classifier.bias.copy_(source.classifier.bias)


@torch.no_grad()
def _copy_shared_ablation_weights(source: DualParameterAttentionMLP, target: ActiveLearnedMemoryMLP) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    for name, tensor in target_state.items():
        if "raw_slots" in name or "valid_mask" in name:
            continue
        if name in source_state:
            tensor.copy_(source_state[name])


def build_initial_slot_tensors(config: E24Config) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate the shared per-seed initial slot tensors and normalize each row."""
    generator = torch.Generator()
    generator.manual_seed(config.seed + 30_000)
    layer1_raw = torch.randn(
        config.num_memory_slots,
        config.hidden_dim,
        generator=generator,
        dtype=torch.float32,
    ) * config.memory_init_std
    layer2_raw = torch.randn(
        config.num_memory_slots,
        config.hidden_dim,
        generator=generator,
        dtype=torch.float32,
    ) * config.memory_init_std
    return (
        normalize_rows(layer1_raw, eps=config.memory_eps),
        normalize_rows(layer2_raw, eps=config.memory_eps),
    )


def initialise_e24_models(config: E24Config) -> E24InitialisedModels:
    """Build the three models and apply the frozen shared initialisation procedure."""
    set_global_determinism(config.seed)

    e24 = _make_e24_model(config)
    baseline = _make_baseline_model(config)
    ablation = _make_ablation_model(config)

    _copy_shared_baseline_weights(e24, baseline)
    _copy_shared_ablation_weights(e24, ablation)

    e24.set_gates(config.gate_init)
    ablation.set_gates(config.gate_init)

    layer1_slots, layer2_slots = build_initial_slot_tensors(config)
    e24.set_memory(layer1_slots, layer2_slots)
    ablation.set_memory(layer1_slots, layer2_slots)

    return E24InitialisedModels(
        baseline=baseline,
        e24=e24,
        ablation=ablation,
        initial_layer1_slots=layer1_slots,
        initial_layer2_slots=layer2_slots,
    )

