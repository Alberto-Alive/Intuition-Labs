"""Support-state components for the frozen E12 support-aware neuron plan."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class LocalSupportState(nn.Module):
    """Persistent per-neuron support updated only from activation experience."""

    def __init__(self, num_neurons: int, decay: float = 0.95) -> None:
        super().__init__()
        if num_neurons <= 0:
            raise ValueError("num_neurons must be positive")
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must lie strictly between 0 and 1")

        self.num_neurons = num_neurons
        self.decay = decay
        self.register_buffer("support", torch.zeros(num_neurons))
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    def activation_mass(self, activations: torch.Tensor) -> torch.Tensor:
        """Soft meaningful-activation mass ρ(a) used by the support EMA."""
        if activations.shape[-1] != self.num_neurons:
            raise ValueError(
                f"Expected trailing dimension {self.num_neurons}, got {activations.shape[-1]}"
            )
        return torch.sigmoid(activations)

    def batch_support(self, activations: torch.Tensor) -> torch.Tensor:
        """Average soft activation mass across every axis except the neuron axis."""
        mass = self.activation_mass(activations.detach())
        if mass.ndim == 1:
            return mass
        reduce_dims = tuple(range(mass.ndim - 1))
        return mass.mean(dim=reduce_dims)

    @torch.no_grad()
    def update(self, activations: torch.Tensor) -> torch.Tensor:
        """Apply the experience-only EMA update to the persistent support state."""
        observed_support = self.batch_support(activations)
        self.support.mul_(self.decay).add_(observed_support, alpha=1.0 - self.decay)
        self.num_updates.add_(1)
        return self.support

    def broadcast(self, activations: torch.Tensor) -> torch.Tensor:
        """Broadcast support values across the leading activation axes."""
        leading_shape = activations.shape[:-1]
        view_shape = (1,) * len(leading_shape) + (self.num_neurons,)
        return self.support.view(view_shape).expand(*leading_shape, self.num_neurons)


class JointSupportSketch(nn.Module):
    """Fixed sketch memory for combination-level support at the layer level."""

    def __init__(
        self,
        input_dim: int,
        sketch_dim: int,
        mean_decay: float = 0.95,
        variance_decay: float = 0.95,
        eps: float = 1e-6,
        projection: torch.Tensor | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or sketch_dim <= 0:
            raise ValueError("input_dim and sketch_dim must be positive")
        if not 0.0 < mean_decay < 1.0:
            raise ValueError("mean_decay must lie strictly between 0 and 1")
        if not 0.0 < variance_decay < 1.0:
            raise ValueError("variance_decay must lie strictly between 0 and 1")

        self.input_dim = input_dim
        self.sketch_dim = sketch_dim
        self.mean_decay = mean_decay
        self.variance_decay = variance_decay
        self.eps = eps

        if projection is None:
            generator = torch.Generator()
            generator.manual_seed(seed)
            projection = torch.randn(sketch_dim, input_dim, generator=generator)
            projection = torch.nn.functional.normalize(projection, dim=-1)
        elif projection.shape != (sketch_dim, input_dim):
            raise ValueError(
                f"Expected projection of shape {(sketch_dim, input_dim)}, got {tuple(projection.shape)}"
            )

        self.register_buffer("projection", projection.detach().clone())
        self.register_buffer("mean", torch.zeros(sketch_dim))
        self.register_buffer("variance", torch.ones(sketch_dim))
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    def project(self, activation_vectors: torch.Tensor) -> torch.Tensor:
        """Project layer activations into the fixed sketch space."""
        if activation_vectors.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected trailing dimension {self.input_dim}, got {activation_vectors.shape[-1]}"
            )
        return activation_vectors @ self.projection.t()

    def novelty_distance(self, activation_vectors: torch.Tensor) -> torch.Tensor:
        """Normalized sketch distance from the running joint-support statistics."""
        sketches = self.project(activation_vectors)
        centered = sketches - self.mean.view(*((1,) * (sketches.ndim - 1)), self.sketch_dim)
        variance = self.variance.view(*((1,) * (sketches.ndim - 1)), self.sketch_dim)
        return (centered.square() / (variance + self.eps)).sum(dim=-1)

    @torch.no_grad()
    def update(self, activation_vectors: torch.Tensor) -> torch.Tensor:
        """Update running mean and variance in sketch space from detached experience."""
        sketches = self.project(activation_vectors.detach())
        if sketches.ndim == 1:
            batch_mean = sketches
            centered = sketches - self.mean
            batch_variance = centered.square()
        else:
            reduce_dims = tuple(range(sketches.ndim - 1))
            batch_mean = sketches.mean(dim=reduce_dims)
            centered = sketches - self.mean.view(*((1,) * (sketches.ndim - 1)), self.sketch_dim)
            batch_variance = centered.square().mean(dim=reduce_dims)

        self.mean.mul_(self.mean_decay).add_(batch_mean, alpha=1.0 - self.mean_decay)
        self.variance.mul_(self.variance_decay).add_(
            batch_variance,
            alpha=1.0 - self.variance_decay,
        )
        self.num_updates.add_(1)
        return sketches


@dataclass
class PropagatedSupportState:
    """Explicit certainty state carried from one layer into the next."""

    per_neuron_support: torch.Tensor
    next_context_logit: torch.Tensor
    next_context_support: torch.Tensor
    layer_evidence: torch.Tensor


class SupportPropagation(nn.Module):
    """Logit-space certainty propagation across layers."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def _clamp(self, values: torch.Tensor) -> torch.Tensor:
        return values.clamp(self.eps, 1.0 - self.eps)

    def forward(
        self,
        previous_context_logit: torch.Tensor,
        local_support: torch.Tensor,
        joint_support: torch.Tensor,
    ) -> PropagatedSupportState:
        if local_support.ndim != 2:
            raise ValueError("local_support must have shape [batch, neurons]")
        if previous_context_logit.ndim != 1:
            raise ValueError("previous_context_logit must have shape [batch]")
        if joint_support.ndim != 1:
            raise ValueError("joint_support must have shape [batch]")
        if local_support.shape[0] != previous_context_logit.shape[0]:
            raise ValueError("batch size mismatch between previous_context_logit and local_support")
        if joint_support.shape[0] != previous_context_logit.shape[0]:
            raise ValueError("batch size mismatch between previous_context_logit and joint_support")

        local_logit = torch.logit(self._clamp(local_support))
        joint_logit = torch.logit(self._clamp(joint_support)).unsqueeze(-1)
        propagated_logit = previous_context_logit.unsqueeze(-1) + local_logit + joint_logit
        per_neuron_support = torch.sigmoid(propagated_logit)

        layer_evidence = 0.5 * (local_support.mean(dim=-1) + joint_support)
        next_context_logit = previous_context_logit + torch.logit(self._clamp(layer_evidence))
        next_context_support = torch.sigmoid(next_context_logit)

        return PropagatedSupportState(
            per_neuron_support=per_neuron_support,
            next_context_logit=next_context_logit,
            next_context_support=next_context_support,
            layer_evidence=layer_evidence,
        )


class ForwardSupportInteraction(nn.Module):
    """Exact forward interaction operator Ω plus the A7 scalar-gate control."""

    def __init__(self, interaction_mode: str = "full") -> None:
        super().__init__()
        if interaction_mode not in {"full", "a7", "neutral"}:
            raise ValueError("interaction_mode must be one of {'full', 'a7', 'neutral'}")
        self.interaction_mode = interaction_mode
        self.vector = nn.Parameter(torch.zeros(3))

    def forward(
        self,
        activations: torch.Tensor,
        support: torch.Tensor,
        context_support: torch.Tensor,
    ) -> torch.Tensor:
        if self.interaction_mode == "neutral":
            return activations
        if activations.shape != support.shape:
            raise ValueError("activations and support must share the same shape")

        if self.interaction_mode == "a7":
            return activations * support

        context = context_support
        while context.ndim < activations.ndim:
            context = context.unsqueeze(-1)
        context = context.expand_as(activations)
        stacked = torch.stack((activations, support, context), dim=-1)
        gate = torch.sigmoid(stacked @ self.vector)
        return activations * gate


class _GradientAuthorityFunction(torch.autograd.Function):
    """Identity in the forward pass, authority scaling in the backward pass."""

    @staticmethod
    def forward(ctx, activations: torch.Tensor, authority: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(authority)
        return activations

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (authority,) = ctx.saved_tensors
        while authority.ndim < grad_output.ndim:
            authority = authority.unsqueeze(-1)
        return grad_output * authority, None


class BackwardPlasticityModulator(nn.Module):
    """Fixed analytical support-gated plasticity rule h(e, c)."""

    def __init__(self, alpha: float = 10.0, beta: float = 0.1) -> None:
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha))
        self.register_buffer("beta", torch.tensor(beta))

    def authority(self, support: torch.Tensor) -> torch.Tensor:
        detached_support = support.detach()
        return self.alpha * detached_support + self.beta * (1.0 - detached_support)

    def modulate_error(self, error: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        return error * self.authority(support)

    def modulate_activation_gradient(
        self,
        activations: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        authority = self.authority(support)
        return _GradientAuthorityFunction.apply(activations, authority)
