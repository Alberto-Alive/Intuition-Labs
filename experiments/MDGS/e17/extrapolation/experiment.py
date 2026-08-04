"""Training and diagnostic helpers for DIGIT Extrapolation E17."""

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
from experiments.DIGIT.Extrapolation.e16.extrapolation.config import E16Config
from experiments.DIGIT.Extrapolation.e16.extrapolation.models import ForwardGatedLiveBankMLP

from .config import E17Config
from .models import BackwardModulatedLiveBankMLP


@dataclass(frozen=True)
class GradientNormStepSummary:
    """Layer-1 gradient norm comparison for one training step."""

    step: int
    e16_gradient_norm: float
    e17_gradient_norm: float
    ratio_e17_over_e16: float


@dataclass(frozen=True)
class MatchedRunSummary:
    """Training summary for one matched E16 or E17 run."""

    final_training_loss: float
    final_training_accuracy: float
    seen_test_accuracy: float
    unseen_test_accuracy: float
    stopping_epoch: int
    normalized_training_accuracy_area: float
    training_accuracy_history: tuple[float, ...]


@dataclass(frozen=True)
class SeedAreaComparison:
    """Requested Deliverable 1 outputs for one E17 seed."""

    seed: int
    final_training_loss: float
    final_training_accuracy: float
    seen_test_accuracy: float
    unseen_test_accuracy: float
    stopping_epoch: int
    e17_normalized_area: float
    e16_normalized_area: float
    delta_area: float


@dataclass(frozen=True)
class SeedTrajectoryPoint:
    """Training accuracy snapshot for one epoch in the seed-0 trajectory report."""

    label: str
    e16_epoch: int
    e16_training_accuracy: float | None
    e17_epoch: int
    e17_training_accuracy: float | None


@dataclass(frozen=True)
class MatchedSeedTrainingArtifacts:
    """Matched E16 baseline and E17 run artifacts for one seed."""

    seed: int
    e16_summary: MatchedRunSummary
    e17_summary: MatchedRunSummary
    comparison: SeedAreaComparison


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


def make_e17_model(config: E17Config) -> BackwardModulatedLiveBankMLP:
    """Instantiate the backward-modulated E17 model."""
    return BackwardModulatedLiveBankMLP(
        part_a_vocab_size=config.num_part_a_values,
        part_b_vocab_size=config.num_part_b_values,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        bank_capacity=config.bank_capacity,
        min_bank_occupancy=config.min_bank_occupancy,
        alpha=config.alpha,
        beta=config.beta,
        neutral_certainty=config.neutral_certainty,
    )


def _mean_layer1_gradient_norm(model: ForwardGatedLiveBankMLP) -> float:
    gradients = [model.hidden1.weight.grad, model.hidden1.bias.grad]
    norms = [gradient.norm().item() for gradient in gradients if gradient is not None]
    return float(sum(norms) / len(norms))


def _evaluate_seen_unseen_accuracy(
    model: ForwardGatedLiveBankMLP,
    split: SplitTensors,
    config: E16Config | E17Config,
) -> tuple[float, float]:
    loader = make_loader(split, config, shuffle=False)
    model.eval()
    seen_correct = 0
    seen_total = 0
    unseen_correct = 0
    unseen_total = 0

    with torch.no_grad():
        for part_a, part_b, part_a_noise, part_b_noise, labels, seen_unseen in loader:
            outputs = model(
                part_a,
                part_b,
                part_a_noise=part_a_noise,
                part_b_noise=part_b_noise,
            )
            predictions = outputs.logits.argmax(dim=-1)
            seen_mask = seen_unseen == 0
            unseen_mask = seen_unseen == 1

            if int(seen_mask.sum().item()) > 0:
                seen_correct += int((predictions[seen_mask] == labels[seen_mask]).sum().item())
                seen_total += int(seen_mask.sum().item())

            if int(unseen_mask.sum().item()) > 0:
                unseen_correct += int((predictions[unseen_mask] == labels[unseen_mask]).sum().item())
                unseen_total += int(unseen_mask.sum().item())

    return seen_correct / seen_total, unseen_correct / unseen_total


def _train_matched_run(
    model: ForwardGatedLiveBankMLP | BackwardModulatedLiveBankMLP,
    dataset: E14Dataset,
    config: E16Config | E17Config,
    *,
    use_gradient_modulation: bool,
) -> MatchedRunSummary:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_loader = make_loader(dataset.train, config, shuffle=True)

    best_loss = float("inf")
    epochs_without_meaningful_decrease = 0
    final_training_loss = float("inf")
    final_training_accuracy = 0.0
    stopping_epoch = 0
    training_accuracy_history: list[float] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_examples = 0

        for part_a, part_b, part_a_noise, part_b_noise, labels, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                part_a,
                part_b,
                part_a_noise=part_a_noise,
                part_b_noise=part_b_noise,
            )

            if use_gradient_modulation:
                assert isinstance(model, BackwardModulatedLiveBankMLP)
                losses = F.cross_entropy(outputs.logits, labels, reduction="none")
                modulation_weights = model.build_modulation_weights(losses, outputs)
                model.attach_gradient_modulation_hooks(outputs, modulation_weights)
                loss = losses.mean()
            else:
                loss = F.cross_entropy(outputs.logits, labels)

            loss.backward()
            if use_gradient_modulation:
                assert isinstance(model, BackwardModulatedLiveBankMLP)
                model.clear_gradient_modulation_hooks()
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
        training_accuracy_history.append(final_training_accuracy)

        if best_loss - final_training_loss > config.loss_improvement_tolerance:
            best_loss = final_training_loss
            epochs_without_meaningful_decrease = 0
        else:
            epochs_without_meaningful_decrease += 1

        if epochs_without_meaningful_decrease >= config.loss_patience_epochs:
            break

    seen_test_accuracy, unseen_test_accuracy = _evaluate_seen_unseen_accuracy(model, dataset.test, config)
    normalized_area = float(sum(training_accuracy_history) / len(training_accuracy_history))
    return MatchedRunSummary(
        final_training_loss=final_training_loss,
        final_training_accuracy=final_training_accuracy,
        seen_test_accuracy=seen_test_accuracy,
        unseen_test_accuracy=unseen_test_accuracy,
        stopping_epoch=stopping_epoch,
        normalized_training_accuracy_area=normalized_area,
        training_accuracy_history=tuple(training_accuracy_history),
    )


def train_matched_seed(seed: int) -> MatchedSeedTrainingArtifacts:
    """Train matched E16 and E17 runs for one seed and compare normalized areas."""
    e16_config = E16Config(seed=seed)
    e17_config = E17Config(seed=seed)
    set_global_determinism(seed)
    dataset = build_dataset(e17_config)

    e16_model = make_e16_model(e16_config)
    e17_model = make_e17_model(e17_config)
    e17_model.load_state_dict(e16_model.state_dict())

    e16_summary = _train_matched_run(
        e16_model,
        dataset,
        e16_config,
        use_gradient_modulation=False,
    )
    e17_summary = _train_matched_run(
        e17_model,
        dataset,
        e17_config,
        use_gradient_modulation=True,
    )
    comparison = SeedAreaComparison(
        seed=seed,
        final_training_loss=e17_summary.final_training_loss,
        final_training_accuracy=e17_summary.final_training_accuracy,
        seen_test_accuracy=e17_summary.seen_test_accuracy,
        unseen_test_accuracy=e17_summary.unseen_test_accuracy,
        stopping_epoch=e17_summary.stopping_epoch,
        e17_normalized_area=e17_summary.normalized_training_accuracy_area,
        e16_normalized_area=e16_summary.normalized_training_accuracy_area,
        delta_area=e17_summary.normalized_training_accuracy_area - e16_summary.normalized_training_accuracy_area,
    )
    return MatchedSeedTrainingArtifacts(
        seed=seed,
        e16_summary=e16_summary,
        e17_summary=e17_summary,
        comparison=comparison,
    )


def run_matched_training_suite(seeds: list[int] | tuple[int, ...]) -> list[MatchedSeedTrainingArtifacts]:
    """Run the full matched E16-vs-E17 training suite over the requested seeds."""
    return [train_matched_seed(seed) for seed in seeds]


def build_seed0_trajectory_report(
    artifacts: MatchedSeedTrainingArtifacts,
) -> list[SeedTrajectoryPoint]:
    """Return the requested seed-0 training-accuracy trajectory checkpoints."""
    report: list[SeedTrajectoryPoint] = []
    for epoch in (1, 10, 20):
        e16_accuracy = (
            artifacts.e16_summary.training_accuracy_history[epoch - 1]
            if len(artifacts.e16_summary.training_accuracy_history) >= epoch
            else None
        )
        e17_accuracy = (
            artifacts.e17_summary.training_accuracy_history[epoch - 1]
            if len(artifacts.e17_summary.training_accuracy_history) >= epoch
            else None
        )
        report.append(
            SeedTrajectoryPoint(
                label=f"epoch_{epoch}",
                e16_epoch=epoch,
                e16_training_accuracy=e16_accuracy,
                e17_epoch=epoch,
                e17_training_accuracy=e17_accuracy,
            )
        )

    report.append(
        SeedTrajectoryPoint(
            label="final",
            e16_epoch=artifacts.e16_summary.stopping_epoch,
            e16_training_accuracy=artifacts.e16_summary.training_accuracy_history[-1],
            e17_epoch=artifacts.e17_summary.stopping_epoch,
            e17_training_accuracy=artifacts.e17_summary.training_accuracy_history[-1],
        )
    )
    return report


def run_seed0_gradient_norm_diagnostic(num_steps: int = 10) -> list[GradientNormStepSummary]:
    """Compare the first training steps of matched E16 and E17 runs under seed 0."""
    e16_config = E16Config(seed=0)
    e17_config = E17Config(seed=0)
    set_global_determinism(0)

    dataset = build_dataset(e16_config)
    loader = make_loader(dataset.train, e16_config, shuffle=True)

    e16_model = make_e16_model(e16_config)
    e17_model = make_e17_model(e17_config)
    e17_model.load_state_dict(e16_model.state_dict())

    e16_optimizer = torch.optim.Adam(e16_model.parameters(), lr=e16_config.learning_rate)
    e17_optimizer = torch.optim.Adam(e17_model.parameters(), lr=e17_config.learning_rate)

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

        e17_optimizer.zero_grad(set_to_none=True)
        e17_outputs = e17_model(
            part_a,
            part_b,
            part_a_noise=part_a_noise,
            part_b_noise=part_b_noise,
        )
        e17_losses = F.cross_entropy(e17_outputs.logits, labels, reduction="none")
        e17_modulation_weights = e17_model.build_modulation_weights(e17_losses, e17_outputs)
        e17_model.attach_gradient_modulation_hooks(e17_outputs, e17_modulation_weights)
        e17_loss = e17_losses.mean()
        e17_loss.backward()
        e17_gradient_norm = _mean_layer1_gradient_norm(e17_model)
        e17_model.clear_gradient_modulation_hooks()
        e17_optimizer.step()
        e17_model.update_live_banks(
            e17_outputs.layer1_pre_gate_activation,
            e17_outputs.layer2_pre_gate_activation,
        )

        step_summaries.append(
            GradientNormStepSummary(
                step=step_index,
                e16_gradient_norm=e16_gradient_norm,
                e17_gradient_norm=e17_gradient_norm,
                ratio_e17_over_e16=e17_gradient_norm / e16_gradient_norm,
            )
        )

    return step_summaries
