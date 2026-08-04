"""Minimal Regime 2 transformer and ablation variants for E12."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import E12Config
from .support_components import (
    BackwardPlasticityModulator,
    ForwardSupportInteraction,
    JointSupportSketch,
    LocalSupportState,
    SupportPropagation,
)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    use_local: bool
    use_joint: bool
    use_propagation: bool
    forward_mode: str
    backward_mode: str


VARIANT_SPECS: dict[str, VariantSpec] = {
    "baseline": VariantSpec("baseline", False, False, False, "neutral", "none"),
    "full": VariantSpec("full", True, True, True, "full", "support"),
    "a1": VariantSpec("a1", False, True, True, "full", "support"),
    "a2": VariantSpec("a2", True, False, True, "full", "support"),
    "a3": VariantSpec("a3", True, True, False, "full", "support"),
    "a4": VariantSpec("a4", True, True, True, "full", "none"),
    "a5": VariantSpec("a5", True, True, True, "neutral", "none"),
    "a6": VariantSpec("a6", False, False, False, "neutral", "difficulty"),
    "a7": VariantSpec("a7", True, True, True, "a7", "support"),
}


@dataclass
class LayerSupportMetrics:
    raw_local_support: torch.Tensor
    raw_joint_support: torch.Tensor
    raw_joint_distance: torch.Tensor
    propagated_support: torch.Tensor
    final_support: torch.Tensor


@dataclass
class ModelOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    confidence: torch.Tensor
    loss_per_example: torch.Tensor | None
    local_support: torch.Tensor
    joint_support: torch.Tensor
    propagated_support: torch.Tensor
    final_support: torch.Tensor
    layer_metrics: list[LayerSupportMetrics]


class SupportAwareEncoderLayer(nn.Module):
    """A transformer encoder layer with the frozen E12 support primitives."""

    def __init__(self, config: E12Config, layer_index: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.nhead,
            dropout=config.dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.activation = nn.ReLU()
        self.local_state = LocalSupportState(config.d_ff, decay=config.local_support_decay)
        self.joint_sketch = JointSupportSketch(
            input_dim=config.sequence_length * config.d_ff,
            sketch_dim=config.joint_sketch_dim,
            mean_decay=config.joint_mean_decay,
            variance_decay=config.joint_variance_decay,
            eps=config.support_eps,
            seed=config.seed + layer_index,
        )
        self.propagation = SupportPropagation(eps=config.support_eps)
        self.forward_interaction = ForwardSupportInteraction(interaction_mode="full")

    def reset_support_state(self) -> None:
        self.local_state.support.zero_()
        self.local_state.num_updates.zero_()
        self.joint_sketch.mean.zero_()
        self.joint_sketch.variance.fill_(1.0)
        self.joint_sketch.num_updates.zero_()

    def _distance_to_support(self, distance: torch.Tensor) -> torch.Tensor:
        return torch.exp(-distance / float(self.joint_sketch.sketch_dim)).clamp(0.0, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        previous_context_logit: torch.Tensor,
        variant: VariantSpec,
        support_active: bool,
        update_support_states: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, LayerSupportMetrics]:
        attn_out, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)

        hidden = self.activation(self.linear1(x))
        flat_hidden = hidden.reshape(hidden.size(0), -1)

        if update_support_states:
            self.local_state.update(hidden)
            self.joint_sketch.update(flat_hidden)

        raw_local_support = self.local_state.support.unsqueeze(0).expand(hidden.size(0), -1)
        raw_joint_distance = self.joint_sketch.novelty_distance(flat_hidden)
        raw_joint_support = self._distance_to_support(raw_joint_distance)

        effective_local_support = (
            raw_local_support if variant.use_local else torch.full_like(raw_local_support, 0.5)
        )
        effective_joint_support = (
            raw_joint_support if variant.use_joint else torch.full_like(raw_joint_support, 0.5)
        )
        propagation_input = previous_context_logit if variant.use_propagation else torch.zeros_like(previous_context_logit)
        propagated = self.propagation(
            previous_context_logit=propagation_input,
            local_support=effective_local_support,
            joint_support=effective_joint_support,
        )

        support_broadcast = propagated.per_neuron_support.unsqueeze(1).expand_as(hidden)
        if not support_active or variant.forward_mode == "neutral":
            interacted = hidden
        elif variant.forward_mode == "a7":
            interacted = hidden * support_broadcast
        else:
            interacted = self.forward_interaction(
                activations=hidden,
                support=support_broadcast,
                context_support=propagated.next_context_support,
            )

        x = self.norm2(x + self.linear2(interacted))
        metrics = LayerSupportMetrics(
            raw_local_support=raw_local_support.mean(dim=-1),
            raw_joint_support=raw_joint_support,
            raw_joint_distance=raw_joint_distance,
            propagated_support=propagated.next_context_support,
            final_support=propagated.per_neuron_support.mean(dim=-1),
        )
        return x, propagated.next_context_logit, metrics


class SupportAwareTransformerClassifier(nn.Module):
    """Small transformer classifier with frozen E12 support variants."""

    def __init__(self, config: E12Config) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.sequence_length, config.d_model)
        self.layers = nn.ModuleList(
            [SupportAwareEncoderLayer(config, layer_index=i) for i in range(config.num_layers)]
        )
        self.classifier = nn.Linear(config.d_model, config.num_classes)
        self.backward_modulator = BackwardPlasticityModulator(
            alpha=config.backward_alpha,
            beta=config.backward_beta,
        )

    def reset_support_state(self) -> None:
        for layer in self.layers:
            layer.reset_support_state()

    def support_snapshot(self) -> torch.Tensor:
        return torch.cat([layer.local_state.support.detach().cpu() for layer in self.layers], dim=0)

    def variant(self, variant_name: str) -> VariantSpec:
        try:
            return VARIANT_SPECS[variant_name]
        except KeyError as exc:
            raise ValueError(f"Unknown variant {variant_name!r}") from exc

    def forward(
        self,
        tokens: torch.Tensor,
        variant_name: str,
        support_active: bool,
        update_support_states: bool,
        labels: torch.Tensor | None = None,
    ) -> ModelOutput:
        variant = self.variant(variant_name)
        positions = torch.arange(self.config.sequence_length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions).unsqueeze(0)
        context_logit = torch.zeros(tokens.size(0), device=tokens.device)
        layer_metrics = []

        for layer in self.layers:
            x, context_logit, metrics = layer(
                x=x,
                previous_context_logit=context_logit,
                variant=variant,
                support_active=support_active,
                update_support_states=update_support_states,
            )
            layer_metrics.append(metrics)

        pooled = x[:, 0]
        logits = self.classifier(pooled)
        probabilities = torch.softmax(logits, dim=-1)
        confidence = probabilities.max(dim=-1).values
        loss_per_example = None
        if labels is not None:
            loss_per_example = F.cross_entropy(logits, labels, reduction="none")

        last_metrics = layer_metrics[-1]
        return ModelOutput(
            logits=logits,
            probabilities=probabilities,
            confidence=confidence,
            loss_per_example=loss_per_example,
            local_support=last_metrics.raw_local_support,
            joint_support=last_metrics.raw_joint_support,
            propagated_support=last_metrics.propagated_support,
            final_support=last_metrics.final_support,
            layer_metrics=layer_metrics,
        )
