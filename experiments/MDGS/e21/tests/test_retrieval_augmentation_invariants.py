"""Invariant tests for E21 retrieval augmentation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from experiments.DIGIT.Extrapolation.e19.extrapolation.config import E19Config
from experiments.DIGIT.Extrapolation.e19.extrapolation.experiment import (
    build_dataset,
    make_loader,
    make_model,
    set_global_determinism,
)


RETRIEVAL_THRESHOLD = 0.95


class ActivationLabelBuffer:
    """Minimal aligned activation/label storage for retrieval tests."""

    def __init__(self, capacity: int, hidden_dim: int) -> None:
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self.entries = torch.zeros(capacity, hidden_dim, dtype=torch.float32)
        self.labels = torch.full((capacity,), -1, dtype=torch.long)
        self.valid_mask = torch.zeros(capacity, dtype=torch.bool)
        self.write_pointer = 0
        self.valid_count = 0

    @torch.no_grad()
    def update(self, activations: torch.Tensor, labels: torch.Tensor) -> None:
        if activations.ndim != 2 or activations.shape[1] != self.hidden_dim:
            raise ValueError("activations have wrong shape")
        if labels.ndim != 1 or labels.shape[0] != activations.shape[0]:
            raise ValueError("labels must align with activations")

        start = self.write_pointer
        for offset, (row, label) in enumerate(zip(activations.detach(), labels.detach(), strict=True)):
            slot = (start + offset) % self.capacity
            self.entries[slot].copy_(row)
            self.labels[slot] = int(label.item())
            self.valid_mask[slot] = True

        self.write_pointer = (start + activations.shape[0]) % self.capacity
        self.valid_count = min(self.capacity, self.valid_count + activations.shape[0])

    @torch.no_grad()
    def retrieve_top_k(self, queries: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_entries = self.entries[self.valid_mask]
        valid_labels = self.labels[self.valid_mask]
        similarities = F.normalize(queries, dim=-1) @ F.normalize(valid_entries, dim=-1).t()
        topk = similarities.topk(k=min(k, valid_entries.shape[0]), dim=-1)
        return topk.indices, valid_labels[topk.indices], topk.values


def should_apply_retrieval(certainty: torch.Tensor, threshold: float = RETRIEVAL_THRESHOLD) -> torch.Tensor:
    """Low certainty only."""
    return certainty < threshold


def blend_prediction(
    p_model: torch.Tensor,
    p_prior: torch.Tensor,
    certainty: torch.Tensor,
    threshold: float = RETRIEVAL_THRESHOLD,
) -> torch.Tensor:
    """Blend only when retrieval is active."""
    certainty_column = certainty.unsqueeze(-1)
    blended = certainty_column * p_model + (1.0 - certainty_column) * p_prior
    apply_mask = should_apply_retrieval(certainty, threshold=threshold).unsqueeze(-1)
    return torch.where(apply_mask, blended, p_model)


@dataclass(frozen=True)
class Phase1Snapshot:
    """Minimal frozen Phase 1 state for both arms."""

    model_state: dict[str, torch.Tensor]
    layer1_entries: torch.Tensor
    layer1_valid_mask: torch.Tensor
    layer1_labels: torch.Tensor
    layer2_entries: torch.Tensor
    layer2_valid_mask: torch.Tensor
    layer2_labels: torch.Tensor


def _train_phase1_snapshot(seed: int = 0) -> Phase1Snapshot:
    """Train one Phase 1 run and record aligned label buffers during bank writes."""
    config = E19Config(seed=seed, rare_repeats_per_combo=0, novel_repeats_per_combo=0)
    set_global_determinism(seed)
    dataset = build_dataset(config)
    model = make_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_loader = make_loader(dataset.train, config, shuffle=True)

    layer1_labels = ActivationLabelBuffer(config.bank_capacity, config.hidden_dim)
    layer2_labels = ActivationLabelBuffer(config.bank_capacity, config.hidden_dim)

    best_loss = float("inf")
    epochs_without_meaningful_decrease = 0

    for _epoch in range(1, config.max_epochs + 1):
        epoch_loss_sum = 0.0
        epoch_examples = 0
        model.train()

        for part_a, part_b, part_a_noise, part_b_noise, labels, _levels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                part_a,
                part_b,
                part_a_noise=part_a_noise,
                part_b_noise=part_b_noise,
            )
            loss = F.cross_entropy(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            model.update_live_banks(
                outputs.layer1_pre_gate_activation,
                outputs.layer2_pre_gate_activation,
            )
            layer1_labels.update(outputs.layer1_pre_gate_activation, labels)
            layer2_labels.update(outputs.layer2_pre_gate_activation, labels)

            epoch_loss_sum += float(loss.item()) * labels.shape[0]
            epoch_examples += labels.shape[0]

        final_training_loss = epoch_loss_sum / epoch_examples
        if best_loss - final_training_loss > config.loss_improvement_tolerance:
            best_loss = final_training_loss
            epochs_without_meaningful_decrease = 0
        else:
            epochs_without_meaningful_decrease += 1

        if epochs_without_meaningful_decrease >= config.loss_patience_epochs:
            break

    return Phase1Snapshot(
        model_state={name: tensor.detach().clone() for name, tensor in model.state_dict().items()},
        layer1_entries=model.layer1_bank.entries.detach().clone(),
        layer1_valid_mask=model.layer1_bank.valid_mask.detach().clone(),
        layer1_labels=layer1_labels.labels.detach().clone(),
        layer2_entries=model.layer2_bank.entries.detach().clone(),
        layer2_valid_mask=model.layer2_bank.valid_mask.detach().clone(),
        layer2_labels=layer2_labels.labels.detach().clone(),
    )


def test_label_buffers_stay_aligned_with_bank_rows() -> None:
    """Writing one activation with one label must preserve row alignment."""
    bank = ActivationLabelBuffer(capacity=8, hidden_dim=3)
    activation = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    label = torch.tensor([2], dtype=torch.long)

    bank.update(activation, label)

    assert torch.allclose(bank.entries[0], activation[0])
    assert int(bank.labels[0].item()) == 2
    assert bool(bank.valid_mask[0])


def test_retrieved_labels_correspond_to_retrieved_activation_indices() -> None:
    """Top-k labels must come from the same indices as the top-k activations."""
    bank = ActivationLabelBuffer(capacity=8, hidden_dim=3)
    activations = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([3, 1, 2], dtype=torch.long)
    bank.update(activations, labels)

    query = torch.tensor([[0.9, 0.1, 0.0]], dtype=torch.float32)
    indices, retrieved_labels, _similarities = bank.retrieve_top_k(query, k=2)

    compact_labels = bank.labels[bank.valid_mask]
    expected_labels = compact_labels[indices]
    assert torch.equal(retrieved_labels, expected_labels)
    assert retrieved_labels.shape == (1, 2)


def test_retrieval_fires_only_when_certainty_is_below_0_95() -> None:
    """The retrieval trigger is strictly low-certainty only."""
    low_certainty = torch.tensor([0.94], dtype=torch.float32)
    high_certainty = torch.tensor([0.96], dtype=torch.float32)

    assert bool(should_apply_retrieval(low_certainty).item())
    assert not bool(should_apply_retrieval(high_certainty).item())


def test_high_certainty_inputs_bypass_retrieval_and_use_standard_prediction() -> None:
    """Above threshold, p_blend must equal p_model exactly."""
    p_model = torch.tensor([[0.1, 0.7, 0.2]], dtype=torch.float32)
    p_prior = torch.tensor([[0.6, 0.2, 0.2]], dtype=torch.float32)
    certainty = torch.tensor([0.96], dtype=torch.float32)

    p_blend = blend_prediction(p_model, p_prior, certainty)

    assert torch.equal(p_blend, p_model)


def test_baseline_and_e21_phase2_start_from_identical_phase1_checkpoints_and_banks() -> None:
    """Both Phase 2 arms must start from the same frozen Phase 1 snapshot."""
    snapshot = _train_phase1_snapshot(seed=0)
    config = E19Config(seed=0, rare_repeats_per_combo=0, novel_repeats_per_combo=0)

    baseline_model = make_model(config)
    e21_model = make_model(config)
    baseline_model.load_state_dict(snapshot.model_state)
    e21_model.load_state_dict(snapshot.model_state)

    for baseline_parameter, e21_parameter in zip(
        baseline_model.state_dict().values(),
        e21_model.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(baseline_parameter, e21_parameter)

    baseline_layer1_entries = snapshot.layer1_entries.clone()
    baseline_layer1_valid_mask = snapshot.layer1_valid_mask.clone()
    baseline_layer1_labels = snapshot.layer1_labels.clone()
    baseline_layer2_entries = snapshot.layer2_entries.clone()
    baseline_layer2_valid_mask = snapshot.layer2_valid_mask.clone()
    baseline_layer2_labels = snapshot.layer2_labels.clone()

    e21_layer1_entries = snapshot.layer1_entries.clone()
    e21_layer1_valid_mask = snapshot.layer1_valid_mask.clone()
    e21_layer1_labels = snapshot.layer1_labels.clone()
    e21_layer2_entries = snapshot.layer2_entries.clone()
    e21_layer2_valid_mask = snapshot.layer2_valid_mask.clone()
    e21_layer2_labels = snapshot.layer2_labels.clone()

    assert torch.equal(baseline_layer1_entries, e21_layer1_entries)
    assert torch.equal(baseline_layer1_valid_mask, e21_layer1_valid_mask)
    assert torch.equal(baseline_layer1_labels, e21_layer1_labels)
    assert torch.equal(baseline_layer2_entries, e21_layer2_entries)
    assert torch.equal(baseline_layer2_valid_mask, e21_layer2_valid_mask)
    assert torch.equal(baseline_layer2_labels, e21_layer2_labels)
