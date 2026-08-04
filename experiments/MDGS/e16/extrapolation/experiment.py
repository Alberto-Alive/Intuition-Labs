"""Training loop and reporting helpers for DIGIT Extrapolation E16."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from experiments.DIGIT.Extrapolation.e14.extrapolation.experiment import (
    E14Dataset,
    SplitTensors,
    build_dataset,
    make_loader,
)

from .config import E16Config
from .models import ForwardGatedLiveBankMLP


@dataclass(frozen=True)
class GateSeenUnseenSummary:
    """Mean gate values on seen and unseen test combinations."""

    seen_mean: float
    unseen_mean: float


@dataclass(frozen=True)
class SeedTrainingResult:
    """Requested Deliverable 1 outputs for one E16 seed."""

    seed: int
    stopping_epoch: int
    final_training_loss: float
    final_training_accuracy: float
    seen_test_accuracy: float
    unseen_test_accuracy: float
    layer1_gate_summary: GateSeenUnseenSummary
    layer2_gate_summary: GateSeenUnseenSummary


def set_global_determinism(seed: int) -> None:
    """Set the approved seed across Python and PyTorch."""
    random.seed(seed)
    torch.manual_seed(seed)


def make_model(config: E16Config) -> ForwardGatedLiveBankMLP:
    """Instantiate the approved 2-layer MLP with forward certainty gating."""
    return ForwardGatedLiveBankMLP(
        part_a_vocab_size=config.num_part_a_values,
        part_b_vocab_size=config.num_part_b_values,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        bank_capacity=config.bank_capacity,
        min_bank_occupancy=config.min_bank_occupancy,
    )


def train_single_seed(config: E16Config) -> SeedTrainingResult:
    """Train one E16 seed and return Deliverable 1 metrics."""
    return train_single_seed_artifacts(config).result


@dataclass
class TrainedSeedArtifacts:
    """Trained model plus dataset and stopping metrics for one seed."""

    config: E16Config
    model: ForwardGatedLiveBankMLP
    dataset: E14Dataset
    result: SeedTrainingResult


def train_single_seed_artifacts(config: E16Config) -> TrainedSeedArtifacts:
    """Train one E16 seed and return the trained model plus materialized dataset."""
    set_global_determinism(config.seed)
    dataset = build_dataset(config)

    device = torch.device(config.device)
    model = make_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_loader = make_loader(dataset.train, config, shuffle=True)

    best_loss = float("inf")
    epochs_without_meaningful_decrease = 0
    final_training_loss = float("inf")
    final_training_accuracy = 0.0
    stopping_epoch = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_examples = 0

        for part_a, part_b, part_a_noise, part_b_noise, labels, _ in train_loader:
            part_a = part_a.to(device)
            part_b = part_b.to(device)
            part_a_noise = part_a_noise.to(device)
            part_b_noise = part_b_noise.to(device)
            labels = labels.to(device)

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

            batch_size = labels.size(0)
            epoch_loss_sum += float(loss.item()) * batch_size
            predictions = outputs.logits.argmax(dim=-1)
            epoch_correct += int((predictions == labels).sum().item())
            epoch_examples += batch_size

        final_training_loss = epoch_loss_sum / epoch_examples
        final_training_accuracy = epoch_correct / epoch_examples
        stopping_epoch = epoch

        if best_loss - final_training_loss > config.loss_improvement_tolerance:
            best_loss = final_training_loss
            epochs_without_meaningful_decrease = 0
        else:
            epochs_without_meaningful_decrease += 1

        if epochs_without_meaningful_decrease >= config.loss_patience_epochs:
            break

    seen_test_accuracy, unseen_test_accuracy, layer1_gate_summary, layer2_gate_summary = evaluate_test_metrics(
        model,
        dataset.test,
        config,
    )
    result = SeedTrainingResult(
        seed=config.seed,
        stopping_epoch=stopping_epoch,
        final_training_loss=final_training_loss,
        final_training_accuracy=final_training_accuracy,
        seen_test_accuracy=seen_test_accuracy,
        unseen_test_accuracy=unseen_test_accuracy,
        layer1_gate_summary=layer1_gate_summary,
        layer2_gate_summary=layer2_gate_summary,
    )
    return TrainedSeedArtifacts(config=config, model=model, dataset=dataset, result=result)


def evaluate_test_metrics(
    model: ForwardGatedLiveBankMLP,
    split: SplitTensors,
    config: E16Config,
) -> tuple[float, float, GateSeenUnseenSummary, GateSeenUnseenSummary]:
    """Evaluate seen/unseen accuracy and end-of-training gate means on the test set."""
    device = torch.device(config.device)
    loader = make_loader(split, config, shuffle=False)
    model.eval()

    seen_correct = 0
    seen_total = 0
    unseen_correct = 0
    unseen_total = 0
    seen_l1_gates: list[torch.Tensor] = []
    unseen_l1_gates: list[torch.Tensor] = []
    seen_l2_gates: list[torch.Tensor] = []
    unseen_l2_gates: list[torch.Tensor] = []

    with torch.no_grad():
        for part_a, part_b, part_a_noise, part_b_noise, labels, seen_unseen in loader:
            part_a = part_a.to(device)
            part_b = part_b.to(device)
            part_a_noise = part_a_noise.to(device)
            part_b_noise = part_b_noise.to(device)
            labels = labels.to(device)

            outputs = model(
                part_a,
                part_b,
                part_a_noise=part_a_noise,
                part_b_noise=part_b_noise,
            )
            predictions = outputs.logits.argmax(dim=-1).cpu()
            labels_cpu = labels.cpu()
            seen_mask = seen_unseen == 0
            unseen_mask = seen_unseen == 1

            if int(seen_mask.sum().item()) > 0:
                seen_correct += int((predictions[seen_mask] == labels_cpu[seen_mask]).sum().item())
                seen_total += int(seen_mask.sum().item())
                seen_l1_gates.append(outputs.gate_layer1.detach().cpu()[seen_mask])
                seen_l2_gates.append(outputs.gate_layer2.detach().cpu()[seen_mask])

            if int(unseen_mask.sum().item()) > 0:
                unseen_correct += int((predictions[unseen_mask] == labels_cpu[unseen_mask]).sum().item())
                unseen_total += int(unseen_mask.sum().item())
                unseen_l1_gates.append(outputs.gate_layer1.detach().cpu()[unseen_mask])
                unseen_l2_gates.append(outputs.gate_layer2.detach().cpu()[unseen_mask])

    seen_accuracy = seen_correct / seen_total if seen_total else 0.0
    unseen_accuracy = unseen_correct / unseen_total if unseen_total else 0.0
    layer1_gate_summary = GateSeenUnseenSummary(
        seen_mean=float(torch.cat(seen_l1_gates).mean().item()),
        unseen_mean=float(torch.cat(unseen_l1_gates).mean().item()),
    )
    layer2_gate_summary = GateSeenUnseenSummary(
        seen_mean=float(torch.cat(seen_l2_gates).mean().item()),
        unseen_mean=float(torch.cat(unseen_l2_gates).mean().item()),
    )
    return seen_accuracy, unseen_accuracy, layer1_gate_summary, layer2_gate_summary


def run_training_suite(seeds: list[int] | tuple[int, ...]) -> list[SeedTrainingResult]:
    """Run Deliverable 1 training over the requested seeds."""
    return [train_single_seed(E16Config(seed=seed)) for seed in seeds]

