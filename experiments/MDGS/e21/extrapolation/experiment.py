"""Phase 1 training and frozen-bank diagnostics for DIGIT Extrapolation E21."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from math import ceil

import torch
import torch.nn.functional as F

from experiments.DIGIT.Extrapolation.e16.extrapolation.models import ForwardGatedLiveBankMLP
from experiments.DIGIT.Extrapolation.e19.extrapolation.data import (
    LEVEL_CORE,
    LEVEL_FAMILIAR,
    LEVEL_NAMES,
    LEVEL_NOVEL,
    LEVEL_ORDER,
    LEVEL_RARE,
)
from experiments.DIGIT.Extrapolation.e19.extrapolation.experiment import (
    E19Dataset,
    SplitTensors,
    build_dataset as build_e19_dataset,
    make_loader as make_e19_loader,
    set_global_determinism,
)

from .config import E21Config
from .label_bank import ActivationMetadataBuffer, FrozenBankMetadata


@dataclass(frozen=True)
class LevelCertaintySummary:
    """Mean retrieval certainty for one level against the frozen Phase 1 bank."""

    level: int
    level_name: str
    mean_certainty: float


@dataclass(frozen=True)
class Phase1Result:
    """Requested Deliverable 1 outputs for one E21 seed."""

    seed: int
    final_training_loss: float
    final_training_accuracy: float
    stopping_epoch: int
    core_bank_fraction: float
    familiar_bank_fraction: float
    certainty_by_level: tuple[LevelCertaintySummary, ...]


@dataclass
class FrozenPhase1Artifacts:
    """Frozen Phase 1 state for later Phase 2 use."""

    config: E21Config
    model: ForwardGatedLiveBankMLP
    dataset: E19Dataset
    layer1_metadata: FrozenBankMetadata
    layer2_metadata: FrozenBankMetadata
    result: Phase1Result


@dataclass(frozen=True)
class Phase2EpochResult:
    """One Phase 2 epoch summary for one arm and seed."""

    epoch: int
    training_accuracy: float
    rare_test_accuracy: float
    novel_test_accuracy: float
    rare_union_novel_test_accuracy: float
    retrieval_fraction: float | None
    mean_novel_train_certainty: float | None


@dataclass(frozen=True)
class Phase2ArmRun:
    """Full Phase 2 history for one arm and one seed."""

    arm_name: str
    seed: int
    epoch_results: tuple[Phase2EpochResult, ...]


@dataclass(frozen=True)
class Phase2SeedDiagnostics:
    """Deliverable 2 diagnostics for one seed."""

    seed: int
    baseline_first_epoch_train_accuracy_ge_0_99: int
    e21_first_epoch_train_accuracy_ge_0_99: int
    seed0_e21_retrieval_fraction_by_epoch: tuple[float, ...] | None
    seed0_e21_epoch1_mean_novel_certainty: float | None


def build_phase1_dataset(config: E21Config) -> E19Dataset:
    """Phase 1 reuses the E19 dataset machinery with rare/novel removed from training."""
    return build_e19_dataset(
        config=type("Phase1E19Config", (), {
            "seed": config.seed,
            "device": config.device,
            "num_part_a_values": config.num_part_a_values,
            "num_part_b_values": config.num_part_b_values,
            "num_classes": config.num_classes,
            "core_repeats_per_combo": config.phase1_core_repeats_per_combo,
            "familiar_repeats_per_combo": config.phase1_familiar_repeats_per_combo,
            "rare_repeats_per_combo": config.phase1_rare_repeats_per_combo,
            "novel_repeats_per_combo": config.phase1_novel_repeats_per_combo,
            "test_repeats_per_combo": config.test_repeats_per_combo,
            "batch_size": config.batch_size,
            "embedding_dim": config.embedding_dim,
            "hidden_dim": config.hidden_dim,
            "bank_capacity": config.bank_capacity,
            "min_bank_occupancy": config.min_bank_occupancy,
            "input_jitter_std": config.input_jitter_std,
            "train_noise_seed_offset": config.train_noise_seed_offset,
            "test_noise_seed_offset": config.test_noise_seed_offset,
            "learning_rate": config.learning_rate,
            "max_epochs": config.max_epochs,
            "loss_improvement_tolerance": config.loss_improvement_tolerance,
            "loss_patience_epochs": config.loss_patience_epochs,
        })()
    )


def make_loader(split: SplitTensors, config: E21Config, *, shuffle: bool):
    """Reuse the E19 DataLoader construction with the Phase 1 config."""
    return make_e19_loader(
        split,
        config=type("Phase1E19Config", (), {
            "seed": config.seed,
            "batch_size": config.batch_size,
        })(),
        shuffle=shuffle,
    )


def build_phase2_dataset(config: E21Config) -> E19Dataset:
    """Phase 2 uses all four levels with equal repeats per combination."""
    return build_e19_dataset(
        config=type("Phase2E19Config", (), {
            "seed": config.seed,
            "device": config.device,
            "num_part_a_values": config.num_part_a_values,
            "num_part_b_values": config.num_part_b_values,
            "num_classes": config.num_classes,
            "core_repeats_per_combo": config.phase2_repeats_per_combo,
            "familiar_repeats_per_combo": config.phase2_repeats_per_combo,
            "rare_repeats_per_combo": config.phase2_repeats_per_combo,
            "novel_repeats_per_combo": config.phase2_repeats_per_combo,
            "test_repeats_per_combo": config.test_repeats_per_combo,
            "batch_size": config.batch_size,
            "embedding_dim": config.embedding_dim,
            "hidden_dim": config.hidden_dim,
            "bank_capacity": config.bank_capacity,
            "min_bank_occupancy": config.min_bank_occupancy,
            "input_jitter_std": config.input_jitter_std,
            "train_noise_seed_offset": config.train_noise_seed_offset,
            "test_noise_seed_offset": config.test_noise_seed_offset,
            "learning_rate": config.learning_rate,
            "max_epochs": config.max_epochs,
            "loss_improvement_tolerance": config.loss_improvement_tolerance,
            "loss_patience_epochs": config.loss_patience_epochs,
        })()
    )


def make_model(config: E21Config) -> ForwardGatedLiveBankMLP:
    """Instantiate the approved two-layer model with the unchanged E16 gate and bank."""
    return ForwardGatedLiveBankMLP(
        part_a_vocab_size=config.num_part_a_values,
        part_b_vocab_size=config.num_part_b_values,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        bank_capacity=config.bank_capacity,
        min_bank_occupancy=config.min_bank_occupancy,
    )


def _build_phase2_epoch_batches(split: SplitTensors, config: E21Config, epoch: int) -> tuple[tuple[torch.Tensor, ...], ...]:
    """Create a deterministic per-epoch shuffled batch order shared by both arms."""
    generator = torch.Generator()
    generator.manual_seed(config.seed + 1_000 + epoch)
    permutation = torch.randperm(split.size, generator=generator)
    batches: list[tuple[torch.Tensor, ...]] = []
    for start in range(0, split.size, config.batch_size):
        indices = permutation[start : start + config.batch_size]
        batches.append(
            (
                split.part_a[indices],
                split.part_b[indices],
                split.part_a_noise[indices],
                split.part_b_noise[indices],
                split.labels[indices],
                split.levels[indices],
            )
        )
    return tuple(batches)


def _retrieve_layer_prior(
    queries: torch.Tensor,
    bank_entries: torch.Tensor,
    bank_valid_mask: torch.Tensor,
    bank_labels: torch.Tensor,
    *,
    num_classes: int,
    k: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Retrieve a similarity-weighted soft label prior from one frozen bank."""
    valid_entries = bank_entries[bank_valid_mask].to(queries.device)
    valid_labels = bank_labels[bank_valid_mask].to(queries.device)
    normalized_queries = F.normalize(queries.detach(), dim=-1, eps=eps)
    normalized_entries = F.normalize(valid_entries.detach(), dim=-1, eps=eps)
    similarities = normalized_queries @ normalized_entries.t()
    topk = similarities.topk(k=min(k, valid_entries.shape[0]), dim=-1)
    topk_labels = valid_labels[topk.indices]
    one_hot = F.one_hot(topk_labels, num_classes=num_classes).to(dtype=queries.dtype)

    similarity_sums = topk.values.sum(dim=-1, keepdim=True)
    uniform_weights = torch.full_like(topk.values, 1.0 / topk.values.shape[1])
    normalized_weights = torch.where(
        similarity_sums.abs() > eps,
        topk.values / similarity_sums,
        uniform_weights,
    )
    return (normalized_weights.unsqueeze(-1) * one_hot).sum(dim=1)


def _bottom_fraction_retrieval_mask(
    certainty: torch.Tensor,
    *,
    fraction: float,
) -> torch.Tensor:
    """Select the least-certain fraction of a batch for retrieval augmentation."""
    if certainty.ndim != 1:
        raise ValueError("certainty must be one-dimensional")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")

    num_selected = max(1, ceil(certainty.shape[0] * fraction))
    _, indices = torch.topk(certainty, k=num_selected, largest=False)
    mask = torch.zeros_like(certainty, dtype=torch.bool)
    mask[indices] = True
    return mask


def _evaluate_phase2_epoch(
    model: ForwardGatedLiveBankMLP,
    split: SplitTensors,
    config: E21Config,
) -> tuple[float, float, float]:
    """Evaluate rare, novel, and rare∪novel test accuracy with retrieval disabled."""
    loader = make_loader(split, config, shuffle=False)
    device = torch.device(config.device)
    rare_correct = 0
    rare_total = 0
    novel_correct = 0
    novel_total = 0
    union_correct = 0
    union_total = 0
    model.eval()

    with torch.no_grad():
        for part_a, part_b, part_a_noise, part_b_noise, labels, levels in loader:
            outputs = model(
                part_a.to(device),
                part_b.to(device),
                part_a_noise=part_a_noise.to(device),
                part_b_noise=part_b_noise.to(device),
            )
            predictions = outputs.logits.argmax(dim=-1).cpu()
            labels_cpu = labels.cpu()

            rare_mask = levels == LEVEL_RARE
            novel_mask = levels == LEVEL_NOVEL
            union_mask = rare_mask | novel_mask

            rare_correct += int((predictions[rare_mask] == labels_cpu[rare_mask]).sum().item())
            rare_total += int(rare_mask.sum().item())
            novel_correct += int((predictions[novel_mask] == labels_cpu[novel_mask]).sum().item())
            novel_total += int(novel_mask.sum().item())
            union_correct += int((predictions[union_mask] == labels_cpu[union_mask]).sum().item())
            union_total += int(union_mask.sum().item())

    return (
        rare_correct / rare_total,
        novel_correct / novel_total,
        union_correct / union_total,
    )


def _run_phase2_arm(
    phase1_artifacts: FrozenPhase1Artifacts,
    *,
    arm_name: str,
) -> Phase2ArmRun:
    """Run one Phase 2 arm from the frozen Phase 1 checkpoint."""
    config = phase1_artifacts.config
    device = torch.device(config.device)
    model = copy.deepcopy(phase1_artifacts.model).to(device)
    phase2_dataset = build_phase2_dataset(config)
    epoch_results: list[Phase2EpochResult] = []

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    layer1_labels = phase1_artifacts.layer1_metadata.labels
    layer2_labels = phase1_artifacts.layer2_metadata.labels
    layer1_valid_mask = phase1_artifacts.layer1_metadata.valid_mask
    layer2_valid_mask = phase1_artifacts.layer2_metadata.valid_mask

    for epoch in range(1, config.phase2_epochs + 1):
        model.train()
        correct = 0
        total = 0
        retrieval_hits = 0
        novel_certainty_sum = 0.0
        novel_certainty_count = 0

        for part_a, part_b, part_a_noise, part_b_noise, labels, levels in _build_phase2_epoch_batches(
            phase2_dataset.train,
            config,
            epoch,
        ):
            part_a = part_a.to(device)
            part_b = part_b.to(device)
            part_a_noise = part_a_noise.to(device)
            part_b_noise = part_b_noise.to(device)
            labels_device = labels.to(device)
            levels_device = levels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                part_a,
                part_b,
                part_a_noise=part_a_noise,
                part_b_noise=part_b_noise,
            )

            p_model = torch.softmax(outputs.logits, dim=-1)
            retrieval_certainty = torch.minimum(outputs.layer1_certainty, outputs.layer2_certainty)
            retrieval_mask = _bottom_fraction_retrieval_mask(
                retrieval_certainty,
                fraction=config.retrieval_bottom_fraction,
            )

            if arm_name == "e21" and bool(retrieval_mask.any().item()):
                layer1_prior = _retrieve_layer_prior(
                    outputs.layer1_pre_gate_activation[retrieval_mask],
                    model.layer1_bank.entries,
                    model.layer1_bank.valid_mask,
                    layer1_labels,
                    num_classes=config.num_classes,
                    k=config.retrieved_neighbor_count,
                )
                layer2_prior = _retrieve_layer_prior(
                    outputs.layer2_pre_gate_activation[retrieval_mask],
                    model.layer2_bank.entries,
                    model.layer2_bank.valid_mask,
                    layer2_labels,
                    num_classes=config.num_classes,
                    k=config.retrieved_neighbor_count,
                )
                p_prior = 0.5 * (layer1_prior + layer2_prior)
                blended_for_low = (
                    retrieval_certainty[retrieval_mask].unsqueeze(-1) * p_model[retrieval_mask]
                    + (1.0 - retrieval_certainty[retrieval_mask]).unsqueeze(-1) * p_prior
                )
                final_probabilities = p_model.clone()
                final_probabilities[retrieval_mask] = blended_for_low
            else:
                final_probabilities = p_model

            loss = F.nll_loss(
                torch.log(final_probabilities.clamp_min(1e-12)),
                labels_device,
            )
            loss.backward()
            optimizer.step()

            predictions = final_probabilities.argmax(dim=-1)
            correct += int((predictions == labels_device).sum().item())
            total += labels_device.shape[0]

            if arm_name == "e21":
                retrieval_hits += int(retrieval_mask.sum().item())
                novel_mask = levels_device == LEVEL_NOVEL
                if int(novel_mask.sum().item()) > 0:
                    novel_certainty_sum += float(retrieval_certainty[novel_mask].sum().item())
                    novel_certainty_count += int(novel_mask.sum().item())

        rare_acc, novel_acc, union_acc = _evaluate_phase2_epoch(model, phase2_dataset.test, config)
        epoch_results.append(
            Phase2EpochResult(
                epoch=epoch,
                training_accuracy=correct / total,
                rare_test_accuracy=rare_acc,
                novel_test_accuracy=novel_acc,
                rare_union_novel_test_accuracy=union_acc,
                retrieval_fraction=(retrieval_hits / total) if arm_name == "e21" else None,
                mean_novel_train_certainty=(
                    novel_certainty_sum / novel_certainty_count if arm_name == "e21" and novel_certainty_count > 0 else None
                ),
            )
        )

    return Phase2ArmRun(
        arm_name=arm_name,
        seed=config.seed,
        epoch_results=tuple(epoch_results),
    )


def _first_epoch_at_or_above_threshold(epoch_results: tuple[Phase2EpochResult, ...], threshold: float) -> int:
    """Return the first epoch whose training accuracy reaches the requested threshold."""
    for result in epoch_results:
        if result.training_accuracy >= threshold:
            return result.epoch
    return len(epoch_results) + 1


def run_phase2_diagnostics(
    phase1_artifacts: list[FrozenPhase1Artifacts] | None = None,
) -> tuple[list[Phase2SeedDiagnostics], list[Phase2ArmRun], list[Phase2ArmRun]]:
    """Run both Phase 2 arms and return only the diagnostics needed before primary analysis."""
    frozen_artifacts = run_phase1_suite() if phase1_artifacts is None else phase1_artifacts
    baseline_runs: list[Phase2ArmRun] = []
    e21_runs: list[Phase2ArmRun] = []
    diagnostics: list[Phase2SeedDiagnostics] = []

    for item in frozen_artifacts:
        baseline_run = _run_phase2_arm(item, arm_name="baseline")
        e21_run = _run_phase2_arm(item, arm_name="e21")
        baseline_runs.append(baseline_run)
        e21_runs.append(e21_run)

        diagnostics.append(
            Phase2SeedDiagnostics(
                seed=item.config.seed,
                baseline_first_epoch_train_accuracy_ge_0_99=_first_epoch_at_or_above_threshold(
                    baseline_run.epoch_results,
                    0.99,
                ),
                e21_first_epoch_train_accuracy_ge_0_99=_first_epoch_at_or_above_threshold(
                    e21_run.epoch_results,
                    0.99,
                ),
                seed0_e21_retrieval_fraction_by_epoch=(
                    tuple(result.retrieval_fraction for result in e21_run.epoch_results)
                    if item.config.seed == 0
                    else None
                ),
                seed0_e21_epoch1_mean_novel_certainty=(
                    e21_run.epoch_results[0].mean_novel_train_certainty if item.config.seed == 0 else None
                ),
            )
        )

    return diagnostics, baseline_runs, e21_runs


def _evaluate_retrieval_certainty_by_level(
    model: ForwardGatedLiveBankMLP,
    split: SplitTensors,
    config: E21Config,
) -> tuple[LevelCertaintySummary, ...]:
    """Compute mean retrieval certainty `min(layer1, layer2)` per level on test examples."""
    loader = make_loader(split, config, shuffle=False)
    device = torch.device(config.device)
    certainties_by_level: dict[int, list[torch.Tensor]] = {level: [] for level in LEVEL_ORDER}
    model.eval()

    with torch.no_grad():
        for part_a, part_b, part_a_noise, part_b_noise, _labels, levels in loader:
            outputs = model(
                part_a.to(device),
                part_b.to(device),
                part_a_noise=part_a_noise.to(device),
                part_b_noise=part_b_noise.to(device),
            )
            retrieval_certainty = torch.minimum(
                outputs.layer1_certainty.detach().cpu(),
                outputs.layer2_certainty.detach().cpu(),
            )
            for level in LEVEL_ORDER:
                level_mask = levels == level
                if int(level_mask.sum().item()) > 0:
                    certainties_by_level[level].append(retrieval_certainty[level_mask])

    return tuple(
        LevelCertaintySummary(
            level=level,
            level_name=LEVEL_NAMES[level],
            mean_certainty=float(torch.cat(certainties_by_level[level]).mean().item()),
        )
        for level in LEVEL_ORDER
    )


def train_phase1_single_seed(config: E21Config) -> FrozenPhase1Artifacts:
    """Train the frozen Phase 1 curriculum and return frozen model and bank metadata."""
    set_global_determinism(config.seed)
    dataset = build_phase1_dataset(config)
    model = make_model(config).to(torch.device(config.device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_loader = make_loader(dataset.train, config, shuffle=True)

    layer1_metadata = ActivationMetadataBuffer(config.bank_capacity)
    layer2_metadata = ActivationMetadataBuffer(config.bank_capacity)

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

        for part_a, part_b, part_a_noise, part_b_noise, labels, levels in train_loader:
            part_a = part_a.to(config.device)
            part_b = part_b.to(config.device)
            part_a_noise = part_a_noise.to(config.device)
            part_b_noise = part_b_noise.to(config.device)
            labels = labels.to(config.device)

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
            layer1_metadata.update(labels.detach().cpu(), levels.detach().cpu())
            layer2_metadata.update(labels.detach().cpu(), levels.detach().cpu())

            batch_size = labels.shape[0]
            epoch_loss_sum += float(loss.item()) * batch_size
            epoch_correct += int((outputs.logits.argmax(dim=-1) == labels).sum().item())
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

    frozen_layer1_metadata = layer1_metadata.freeze()
    frozen_layer2_metadata = layer2_metadata.freeze()

    layer1_level_values = frozen_layer1_metadata.levels[frozen_layer1_metadata.valid_mask]
    core_bank_fraction = float((layer1_level_values == LEVEL_CORE).to(dtype=torch.float32).mean().item())
    familiar_bank_fraction = float((layer1_level_values == LEVEL_FAMILIAR).to(dtype=torch.float32).mean().item())
    certainty_by_level = _evaluate_retrieval_certainty_by_level(model, dataset.test, config)

    return FrozenPhase1Artifacts(
        config=config,
        model=model,
        dataset=dataset,
        layer1_metadata=frozen_layer1_metadata,
        layer2_metadata=frozen_layer2_metadata,
        result=Phase1Result(
            seed=config.seed,
            final_training_loss=final_training_loss,
            final_training_accuracy=final_training_accuracy,
            stopping_epoch=stopping_epoch,
            core_bank_fraction=core_bank_fraction,
            familiar_bank_fraction=familiar_bank_fraction,
            certainty_by_level=certainty_by_level,
        ),
    )


def run_phase1_suite(seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> list[FrozenPhase1Artifacts]:
    """Run Phase 1 across the approved five seeds."""
    return [train_phase1_single_seed(E21Config(seed=seed)) for seed in seeds]
