"""Pre-implementation invariant tests for the DIGIT Extrapolation E22 layer."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LayerForwardResult:
    """Outputs exposed by the minimal E22 layer used in invariant tests."""

    h_semantic: torch.Tensor
    query: torch.Tensor
    attention_weights: torch.Tensor
    attention_output: torch.Tensor
    occupancy_gate: torch.Tensor
    h_out: torch.Tensor
    certainty: torch.Tensor


class MinimalDualParameterAttentionLayer(nn.Module):
    """Minimal E22 layer with prototype-key attention and detached prototype updates."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        key_dim: int,
        num_prototypes: int,
        lambda_decay: float = 0.99,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if key_dim % 2 != 0:
            raise ValueError("key_dim must be even")

        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.num_prototypes = num_prototypes
        self.lambda_decay = lambda_decay
        self.eps = eps
        self.repulsion_threshold = sqrt(2.0 / num_prototypes)

        self.semantic = nn.Linear(input_dim, hidden_dim, bias=False)
        self.q_proj = nn.Linear(hidden_dim, key_dim, bias=False)
        self.k_semantic_proj = nn.Linear(hidden_dim, key_dim // 2, bias=False)
        self.k_familiarity_proj = nn.Linear(hidden_dim, key_dim // 2, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.register_buffer("prototypes", torch.zeros(num_prototypes, hidden_dim))
        self.register_buffer("valid_mask", torch.zeros(num_prototypes, dtype=torch.bool))

        self.last_update_inputs: list[torch.Tensor] = []

    def forward(self, inputs: torch.Tensor) -> LayerForwardResult:
        h_semantic = torch.relu(self.semantic(inputs))
        query = self.q_proj(h_semantic)
        attention_output, attention_weights = self._prototype_attention(query)
        occupancy_gate = self.valid_mask.to(dtype=h_semantic.dtype).mean()
        h_out = h_semantic + occupancy_gate * attention_output
        certainty = self.read_only_certainty(h_semantic)
        return LayerForwardResult(
            h_semantic=h_semantic,
            query=query,
            attention_weights=attention_weights,
            attention_output=attention_output,
            occupancy_gate=occupancy_gate,
            h_out=h_out,
            certainty=certainty,
        )

    @torch.no_grad()
    def seed_prototypes(self, prototypes: torch.Tensor, valid_mask: torch.Tensor) -> None:
        """Write prototype buffers directly for deterministic tests."""
        if prototypes.shape != self.prototypes.shape:
            raise ValueError(f"Expected prototype shape {tuple(self.prototypes.shape)}, got {tuple(prototypes.shape)}")
        if valid_mask.shape != self.valid_mask.shape:
            raise ValueError(f"Expected valid_mask shape {tuple(self.valid_mask.shape)}, got {tuple(valid_mask.shape)}")
        self.prototypes.copy_(prototypes)
        self.valid_mask.copy_(valid_mask)

    @torch.no_grad()
    def update_prototypes(self, semantic_activations: torch.Tensor) -> None:
        """Apply the approved seed-then-EMA prototype update rule."""
        self.last_update_inputs = [row.detach().clone() for row in semantic_activations]
        for row in semantic_activations.detach():
            normalized_row = self._normalize_rows(row.unsqueeze(0))[0]
            invalid_indices = torch.nonzero(~self.valid_mask, as_tuple=False).flatten()
            if invalid_indices.numel() > 0:
                write_index = int(invalid_indices[0].item())
                self.prototypes[write_index].copy_(normalized_row)
                self.valid_mask[write_index] = True
            else:
                valid_indices = torch.nonzero(self.valid_mask, as_tuple=False).flatten()
                valid_prototypes = self.prototypes[valid_indices]
                distances = torch.norm(valid_prototypes - normalized_row.unsqueeze(0), dim=-1)
                nearest_offset = int(distances.argmin().item())
                nearest_index = int(valid_indices[nearest_offset].item())
                updated = self.lambda_decay * self.prototypes[nearest_index] + (1.0 - self.lambda_decay) * normalized_row
                self.prototypes[nearest_index].copy_(self._normalize_rows(updated.unsqueeze(0))[0])

            self._apply_repulsion()

    @torch.no_grad()
    def read_only_certainty(self, semantic_activations: torch.Tensor) -> torch.Tensor:
        """Return detached nearest-prototype max cosine certainty over valid prototypes only."""
        if not bool(self.valid_mask.any()):
            return torch.zeros(semantic_activations.shape[0], dtype=semantic_activations.dtype, device=semantic_activations.device)

        normalized_queries = self._normalize_rows(semantic_activations.detach())
        normalized_prototypes = self._normalize_rows(self.prototypes[self.valid_mask].detach())
        return (normalized_queries @ normalized_prototypes.t()).max(dim=-1).values

    def _prototype_attention(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.shape[0]
        if not bool(self.valid_mask.any()):
            return (
                torch.zeros(batch_size, self.hidden_dim, dtype=query.dtype, device=query.device),
                torch.zeros(batch_size, 0, dtype=query.dtype, device=query.device),
            )

        valid_prototypes = self.prototypes[self.valid_mask].to(device=query.device, dtype=query.dtype)
        keys = torch.cat(
            (
                self.k_semantic_proj(valid_prototypes),
                self.k_familiarity_proj(valid_prototypes),
            ),
            dim=-1,
        )
        values = self.v_proj(valid_prototypes)
        logits = query @ keys.t() / sqrt(self.key_dim)
        weights = logits.softmax(dim=-1)
        return weights @ values, weights

    @torch.no_grad()
    def _apply_repulsion(self) -> None:
        valid_indices = torch.nonzero(self.valid_mask, as_tuple=False).flatten()
        for left_offset, left_index_tensor in enumerate(valid_indices[:-1]):
            left_index = int(left_index_tensor.item())
            for right_index_tensor in valid_indices[left_offset + 1 :]:
                right_index = int(right_index_tensor.item())
                diff = self.prototypes[left_index] - self.prototypes[right_index]
                distance = float(torch.norm(diff, p=2).item())
                if distance >= self.repulsion_threshold:
                    continue
                direction = diff / max(distance, self.eps)
                delta = 0.5 * (self.repulsion_threshold - distance) * direction
                self.prototypes[left_index].add_(delta)
                self.prototypes[right_index].sub_(delta)

        if valid_indices.numel() > 0:
            self.prototypes[valid_indices] = self._normalize_rows(self.prototypes[valid_indices])

    def _normalize_rows(self, rows: torch.Tensor) -> torch.Tensor:
        return F.normalize(rows, dim=-1, eps=self.eps)


class MinimalDualParameterAttentionClassifier(nn.Module):
    """Single-layer classifier wrapper for optimizer and gradient invariants."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = MinimalDualParameterAttentionLayer(
            input_dim=3,
            hidden_dim=4,
            key_dim=4,
            num_prototypes=4,
        )
        self.classifier = nn.Linear(4, 3, bias=False)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, LayerForwardResult]:
        layer_outputs = self.layer(inputs)
        logits = self.classifier(layer_outputs.h_out)
        return logits, layer_outputs


def _make_identity_attention_layer() -> MinimalDualParameterAttentionLayer:
    layer = MinimalDualParameterAttentionLayer(
        input_dim=2,
        hidden_dim=2,
        key_dim=2,
        num_prototypes=2,
    )
    with torch.no_grad():
        layer.semantic.weight.copy_(torch.eye(2))
        layer.q_proj.weight.copy_(torch.eye(2))
        layer.k_semantic_proj.weight.copy_(torch.tensor([[1.0, 0.0]], dtype=torch.float32))
        layer.k_familiarity_proj.weight.copy_(torch.tensor([[0.0, 1.0]], dtype=torch.float32))
        layer.v_proj.weight.copy_(torch.eye(2))
    return layer


def test_prototypes_are_buffers_and_excluded_from_optimizer_parameters() -> None:
    model = MinimalDualParameterAttentionClassifier()
    parameter_names = {name for name, _ in model.named_parameters()}
    buffer_names = {name for name, _ in model.named_buffers()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer_param_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}

    assert "layer.prototypes" not in parameter_names
    assert "layer.valid_mask" not in parameter_names
    assert "layer.prototypes" in buffer_names
    assert "layer.valid_mask" in buffer_names
    assert id(model.layer.prototypes) not in optimizer_param_ids
    assert id(model.layer.valid_mask) not in optimizer_param_ids
    assert model.layer.prototypes.requires_grad is False
    assert model.layer.valid_mask.requires_grad is False


def test_prototypes_receive_no_gradient_and_are_unchanged_by_optimizer_step() -> None:
    torch.manual_seed(3)

    model = MinimalDualParameterAttentionClassifier()
    model.layer.seed_prototypes(
        prototypes=torch.randn(4, 4),
        valid_mask=torch.tensor([True, True, True, True]),
    )
    initial_prototypes = model.layer.prototypes.detach().clone()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    inputs = torch.tensor(
        [
            [1.0, 0.5, -0.5],
            [0.5, 1.0, 0.25],
            [0.25, -0.5, 1.0],
            [1.25, 0.75, 0.5],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 2, 1], dtype=torch.long)

    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(inputs)
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    assert model.layer.prototypes.grad is None
    nonzero_parameter_grads = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
    ]
    assert nonzero_parameter_grads

    optimizer.step()

    assert model.layer.prototypes.grad is None
    assert torch.equal(model.layer.prototypes, initial_prototypes)


def test_prototype_updates_use_detached_semantic_activations_only() -> None:
    torch.manual_seed(5)

    model = MinimalDualParameterAttentionClassifier()
    with torch.no_grad():
        model.layer.semantic.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=torch.float32,
            )
        )
    inputs = torch.tensor([[1.0, -0.5, 0.75]], dtype=torch.float32)
    logits, layer_outputs = model(inputs)
    loss = logits.sum()
    loss.backward()

    model.layer.update_prototypes(layer_outputs.h_semantic)

    assert len(model.layer.last_update_inputs) == 1
    recorded = model.layer.last_update_inputs[0]
    assert recorded.requires_grad is False
    assert recorded.grad_fn is None
    expected_prototype = F.normalize(layer_outputs.h_semantic.detach(), dim=-1, eps=model.layer.eps)[0]
    assert torch.allclose(model.layer.prototypes[0], expected_prototype)
    assert bool(model.layer.valid_mask[0])


def test_warmup_with_no_valid_prototypes_returns_zero_attention_and_semantic_only_output() -> None:
    layer = _make_identity_attention_layer()
    inputs = torch.tensor([[1.5, 0.5], [0.25, 2.0]], dtype=torch.float32)

    outputs = layer(inputs)

    assert torch.allclose(outputs.occupancy_gate, torch.tensor(0.0))
    assert outputs.attention_weights.shape == (2, 0)
    assert torch.allclose(outputs.attention_output, torch.zeros_like(outputs.attention_output))
    assert torch.allclose(outputs.h_out, outputs.h_semantic)


def test_invalid_prototypes_are_masked_from_attention_and_certainty() -> None:
    layer = _make_identity_attention_layer()
    layer.seed_prototypes(
        prototypes=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        valid_mask=torch.tensor([True, False]),
    )
    inputs = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

    outputs = layer(inputs)

    assert outputs.attention_weights.shape == (1, 1)
    assert torch.allclose(outputs.attention_weights, torch.ones_like(outputs.attention_weights))
    assert torch.allclose(outputs.attention_output, torch.tensor([[1.0, 0.0]], dtype=torch.float32))
    assert torch.allclose(outputs.certainty, torch.tensor([0.0], dtype=torch.float32), atol=1e-6)


def test_attention_over_multiple_prototype_keys_produces_nontrivial_example_specific_weights() -> None:
    layer = _make_identity_attention_layer()
    layer.seed_prototypes(
        prototypes=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        valid_mask=torch.tensor([True, True]),
    )
    inputs = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    outputs = layer(inputs)

    assert outputs.attention_weights.shape == (2, 2)
    assert torch.allclose(
        outputs.attention_weights.sum(dim=-1),
        torch.ones(2, dtype=torch.float32),
    )
    assert outputs.attention_weights[0, 0] > outputs.attention_weights[0, 1]
    assert outputs.attention_weights[1, 1] > outputs.attention_weights[1, 0]
    assert not torch.allclose(outputs.attention_weights[0], outputs.attention_weights[1])
    assert not torch.allclose(
        outputs.attention_weights,
        torch.full_like(outputs.attention_weights, 0.5),
    )
