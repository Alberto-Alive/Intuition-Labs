"""Training scaffold for DIGIT Extrapolation E26."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from math import ceil
from statistics import mean
from typing import Literal

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.DIGIT.Extrapolation.e26.extrapolation.config import E26Config
from experiments.DIGIT.Extrapolation.e26.extrapolation.data import E26Dataset, SplitTensors, build_dataset, make_loader
from experiments.DIGIT.Extrapolation.e26.extrapolation.models import (
    E26ForwardResult,
    NoveltyDirectedGatedMLP,
    ScalarFamiliarityGatedMLP,
    StandardBaselineMLP,
)


ArmName = Literal["baseline", "novelty_directed", "scalar"]


@dataclass(frozen=True)
class PhaseResult:
    """Training metrics for one phase."""

    final_loss: float
    final_accuracy: float
    stopping_epoch: int


@dataclass(frozen=True)
class SeedRunResult:
    """Results for one seed and one arm."""

    seed: int
    arm: ArmName
    task_a_phase: PhaseResult
    task_a_probe_before: float
    task_a_probe_after: float
    forgetting: float
    retained_accuracy: float
    task_b_phase: PhaseResult
    task_b_accuracy: float


def set_global_determinism(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def make_model(config: E26Config, arm: ArmName):
    if arm == "baseline":
        return StandardBaselineMLP(
            part_a_vocab_size=config.num_part_a_values,
            part_b_vocab_size=config.num_part_b_values,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            num_classes=config.num_classes,
        )
    if arm == "novelty_directed":
        return NoveltyDirectedGatedMLP(
            part_a_vocab_size=config.num_part_a_values,
            part_b_vocab_size=config.num_part_b_values,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            key_dim=config.key_dim,
            num_classes=config.num_classes,
            num_prototypes=config.num_prototypes,
            prototype_lambda=config.prototype_lambda,
            eps=config.prototype_eps,
        )
    if arm == "scalar":
        return ScalarFamiliarityGatedMLP(
            part_a_vocab_size=config.num_part_a_values,
            part_b_vocab_size=config.num_part_b_values,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            key_dim=config.key_dim,
            num_classes=config.num_classes,
            num_prototypes=config.num_prototypes,
            prototype_lambda=config.prototype_lambda,
            eps=config.prototype_eps,
        )
    raise ValueError(f"Unknown arm: {arm}")


def _forward_model(
    model: torch.nn.Module,
    part_a: torch.Tensor,
    part_b: torch.Tensor,
    part_a_noise: torch.Tensor,
    part_b_noise: torch.Tensor,
) -> E26ForwardResult | object:
    return model(
        part_a,
        part_b,
        part_a_noise=part_a_noise,
        part_b_noise=part_b_noise,
    )


def _loss_for_arm(
    arm: ArmName,
    outputs: E26ForwardResult | object,
    labels: torch.Tensor,
) -> torch.Tensor:
    if arm == "scalar":
        per_example_loss = F.cross_entropy(outputs.logits, labels, reduction="none")  # type: ignore[attr-defined]
        familiarity = outputs.layer2_familiarity.detach().clamp(0.0, 1.0)  # type: ignore[attr-defined]
        return (familiarity * per_example_loss).mean()
    return F.cross_entropy(outputs.logits, labels)  # type: ignore[attr-defined]


def _evaluate_accuracy(
    model: torch.nn.Module,
    split: SplitTensors,
    config: E26Config,
) -> float:
    loader = make_loader(split, config, shuffle=False)
    device = torch.device(config.device)
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for part_a, part_b, part_a_noise, part_b_noise, labels, _levels in loader:
            part_a = part_a.to(device)
            part_b = part_b.to(device)
            part_a_noise = part_a_noise.to(device)
            part_b_noise = part_b_noise.to(device)
            labels = labels.to(device)

            outputs = _forward_model(model, part_a, part_b, part_a_noise, part_b_noise)
            predictions = outputs.logits.argmax(dim=-1)  # type: ignore[attr-defined]
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())

    return correct / total if total > 0 else 0.0


def _train_phase(
    model: torch.nn.Module,
    split: SplitTensors,
    config: E26Config,
    arm: ArmName,
    *,
    max_epochs: int,
    early_stopping: bool,
) -> PhaseResult:
    loader = make_loader(split, config, shuffle=True)
    device = torch.device(config.device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_loss = float("inf")
    epochs_without_meaningful_decrease = 0
    final_training_loss = float("inf")
    final_training_accuracy = 0.0
    stopping_epoch = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_examples = 0

        for part_a, part_b, part_a_noise, part_b_noise, labels, _levels in loader:
            part_a = part_a.to(device)
            part_b = part_b.to(device)
            part_a_noise = part_a_noise.to(device)
            part_b_noise = part_b_noise.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = _forward_model(model, part_a, part_b, part_a_noise, part_b_noise)
            loss = _loss_for_arm(arm, outputs, labels)
            loss.backward()
            optimizer.step()

            if arm in {"novelty_directed", "scalar"}:
                model.update_prototypes(  # type: ignore[attr-defined]
                    outputs.layer1_semantic,
                    outputs.layer2_semantic,
                )

            batch_size = labels.size(0)
            epoch_loss_sum += float(loss.item()) * batch_size
            predictions = outputs.logits.argmax(dim=-1)  # type: ignore[attr-defined]
            epoch_correct += int((predictions == labels).sum().item())
            epoch_examples += batch_size

        final_training_loss = epoch_loss_sum / epoch_examples
        final_training_accuracy = epoch_correct / epoch_examples
        stopping_epoch = epoch

        if early_stopping:
            if best_loss - final_training_loss > config.loss_improvement_tolerance:
                best_loss = final_training_loss
                epochs_without_meaningful_decrease = 0
            else:
                epochs_without_meaningful_decrease += 1

            if epochs_without_meaningful_decrease >= config.loss_patience_epochs:
                break

    return PhaseResult(
        final_loss=final_training_loss,
        final_accuracy=final_training_accuracy,
        stopping_epoch=stopping_epoch,
    )


def _train_with_task_b_budget(
    model: torch.nn.Module,
    dataset: E26Dataset,
    config: E26Config,
    arm: ArmName,
    *,
    task_b_epochs: int,
) -> SeedRunResult:
    task_a_phase = _train_phase(
        model,
        dataset.task_a_train,
        config,
        arm,
        max_epochs=config.max_epochs,
        early_stopping=True,
    )
    task_a_probe_before = _evaluate_accuracy(model, dataset.task_a_probe, config)

    task_b_phase = _train_phase(
        model,
        dataset.task_b_train,
        config,
        arm,
        max_epochs=task_b_epochs,
        early_stopping=False,
    )
    task_a_probe_after = _evaluate_accuracy(model, dataset.task_a_probe, config)
    task_b_accuracy = _evaluate_accuracy(model, dataset.task_b_test, config)

    forgetting = task_a_probe_before - task_a_probe_after
    retained_accuracy = task_a_probe_after / task_a_probe_before if task_a_probe_before > 0 else 0.0

    return SeedRunResult(
        seed=config.seed,
        arm=arm,
        task_a_phase=task_a_phase,
        task_a_probe_before=task_a_probe_before,
        task_a_probe_after=task_a_probe_after,
        forgetting=forgetting,
        retained_accuracy=retained_accuracy,
        task_b_phase=task_b_phase,
        task_b_accuracy=task_b_accuracy,
    )


def _continue_from_task_a_artifact(
    model: torch.nn.Module,
    dataset: E26Dataset,
    config: E26Config,
    arm: ArmName,
    *,
    task_a_phase: PhaseResult,
    task_b_epochs: int,
) -> SeedRunResult:
    task_a_probe_before = _evaluate_accuracy(model, dataset.task_a_probe, config)
    task_b_phase = _train_phase(
        model,
        dataset.task_b_train,
        config,
        arm,
        max_epochs=task_b_epochs,
        early_stopping=False,
    )
    task_a_probe_after = _evaluate_accuracy(model, dataset.task_a_probe, config)
    task_b_accuracy = _evaluate_accuracy(model, dataset.task_b_test, config)

    forgetting = task_a_probe_before - task_a_probe_after
    retained_accuracy = task_a_probe_after / task_a_probe_before if task_a_probe_before > 0 else 0.0

    return SeedRunResult(
        seed=config.seed,
        arm=arm,
        task_a_phase=task_a_phase,
        task_a_probe_before=task_a_probe_before,
        task_a_probe_after=task_a_probe_after,
        forgetting=forgetting,
        retained_accuracy=retained_accuracy,
        task_b_phase=task_b_phase,
        task_b_accuracy=task_b_accuracy,
    )


def run_training_suite(seeds: tuple[int, ...] | list[int] | None = None) -> dict[str, list[SeedRunResult]]:
    """Train all three arms across seeds using the baseline-derived Task B budget."""
    seed_list = list(seeds if seeds is not None else E26Config().seeds)

    baseline_task_a_stopping_epochs: list[int] = []
    baseline_task_a_artifacts: list[tuple[E26Config, E26Dataset, torch.nn.Module, PhaseResult]] = []

    for seed in seed_list:
        config = E26Config(seed=seed)
        set_global_determinism(seed)
        dataset = build_dataset(config)
        model = make_model(config, "baseline")
        task_a_phase = _train_phase(
            model,
            dataset.task_a_train,
            config,
            "baseline",
            max_epochs=config.max_epochs,
            early_stopping=True,
        )
        baseline_task_a_stopping_epochs.append(task_a_phase.stopping_epoch)
        baseline_task_a_artifacts.append((config, dataset, model, task_a_phase))

    task_b_budget = int(round(mean(baseline_task_a_stopping_epochs)))

    results: dict[str, list[SeedRunResult]] = {
        "baseline": [],
        "novelty_directed": [],
        "scalar": [],
    }

    for config, dataset, model, task_a_phase in baseline_task_a_artifacts:
        results["baseline"].append(
            _continue_from_task_a_artifact(
                model,
                dataset,
                config,
                "baseline",
                task_a_phase=task_a_phase,
                task_b_epochs=task_b_budget,
            )
        )

    for seed in seed_list:
        config = E26Config(seed=seed)
        set_global_determinism(seed)
        dataset = build_dataset(config)
        for arm in ("novelty_directed", "scalar"):
            model = make_model(config, arm)
            results[arm].append(
                _train_with_task_b_budget(
                    model,
                    dataset,
                    config,
                    arm,
                    task_b_epochs=task_b_budget,
                )
            )

    return results


def build_task_b_budget_from_baseline(seeds: tuple[int, ...] | list[int] | None = None) -> int:
    """Helper for scaffold review when only the Task B budget is needed."""
    seed_list = list(seeds if seeds is not None else E26Config().seeds)
    stopping_epochs: list[int] = []
    for seed in seed_list:
        config = E26Config(seed=seed)
        set_global_determinism(seed)
        dataset = build_dataset(config)
        model = make_model(config, "baseline")
        task_a_phase = _train_phase(
            model,
            dataset.task_a_train,
            config,
            "baseline",
            max_epochs=config.max_epochs,
            early_stopping=True,
        )
        stopping_epochs.append(task_a_phase.stopping_epoch)
    return int(round(mean(stopping_epochs)))
