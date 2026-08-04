"""Training loop and reporting helpers for DIGIT Extrapolation E19."""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm, spearmanr
from torch.utils.data import DataLoader, TensorDataset

from experiments.DIGIT.Extrapolation.e16.extrapolation.models import ForwardGatedLiveBankMLP

from .config import E19Config
from .data import (
    LEVEL_CORE,
    LEVEL_FAMILIAR,
    LEVEL_NAMES,
    LEVEL_NOVEL,
    LEVEL_ORDER,
    LEVEL_RARE,
    FrequencyLevelRecord,
    build_frequency_level_records,
)


@dataclass(frozen=True)
class SplitTensors:
    """Tensorized dataset split for E19."""

    part_a: torch.Tensor
    part_b: torch.Tensor
    part_a_noise: torch.Tensor
    part_b_noise: torch.Tensor
    labels: torch.Tensor
    levels: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.labels.shape[0])


@dataclass(frozen=True)
class E19Dataset:
    """Train/test tensors for one seeded E19 run."""

    train: SplitTensors
    test: SplitTensors


@dataclass(frozen=True)
class LevelAccuracySummary:
    """Test accuracy broken down by one frozen familiarity level."""

    level: int
    level_name: str
    accuracy: float


@dataclass(frozen=True)
class SeedTrainingResult:
    """Requested Deliverable 1 outputs for one E19 seed."""

    seed: int
    stopping_epoch: int
    final_training_loss: float
    final_training_accuracy: float
    level_accuracies: tuple[LevelAccuracySummary, ...]


@dataclass(frozen=True)
class Seed0CertaintyLevelSummary:
    """Layer-1 mean certainty by level for the seed-0 report."""

    level: int
    level_name: str
    mean_certainty: float


@dataclass
class TrainedSeedArtifacts:
    """Trained model plus dataset and stopping metrics for one seed."""

    config: E19Config
    model: ForwardGatedLiveBankMLP
    dataset: E19Dataset
    result: SeedTrainingResult


@dataclass(frozen=True)
class LayerPrimaryAnalysis:
    """Primary ordered-trend analysis outputs for one seed and layer."""

    seed: int
    layer_name: str
    jonckheere_terpstra_statistic: float
    jonckheere_terpstra_p_value: float
    strict_mean_ordering: bool
    spearman_rho: float
    level_means: tuple[float, float, float, float]


@dataclass(frozen=True)
class AggregatedLayerPrimaryAnalysis:
    """Cross-seed summary for one certainty layer."""

    layer_name: str
    significant_seed_count: int
    strict_mean_ordering_seed_count: int
    mean_spearman_rho: float
    std_spearman_rho: float


def set_global_determinism(seed: int) -> None:
    """Set the approved seed across Python and PyTorch."""
    random.seed(seed)
    torch.manual_seed(seed)


def build_dataset(config: E19Config) -> E19Dataset:
    """Materialize the frozen four-level familiarity dataset."""
    records = build_frequency_level_records()
    train_noise_generator = torch.Generator()
    train_noise_generator.manual_seed(config.seed + config.train_noise_seed_offset)
    test_noise_generator = torch.Generator()
    test_noise_generator.manual_seed(config.seed + config.test_noise_seed_offset)

    repeats_by_level = {
        LEVEL_CORE: config.core_repeats_per_combo,
        LEVEL_FAMILIAR: config.familiar_repeats_per_combo,
        LEVEL_RARE: config.rare_repeats_per_combo,
        LEVEL_NOVEL: config.novel_repeats_per_combo,
    }

    train_part_a: list[int] = []
    train_part_b: list[int] = []
    train_part_a_noise: list[torch.Tensor] = []
    train_part_b_noise: list[torch.Tensor] = []
    train_labels: list[int] = []
    train_levels: list[int] = []

    test_part_a: list[int] = []
    test_part_b: list[int] = []
    test_part_a_noise: list[torch.Tensor] = []
    test_part_b_noise: list[torch.Tensor] = []
    test_labels: list[int] = []
    test_levels: list[int] = []

    for record in records:
        for _ in range(repeats_by_level[record.level]):
            train_part_a.append(record.part_a)
            train_part_b.append(record.part_b)
            train_part_a_noise.append(
                torch.randn(config.embedding_dim, generator=train_noise_generator) * config.input_jitter_std
            )
            train_part_b_noise.append(
                torch.randn(config.embedding_dim, generator=train_noise_generator) * config.input_jitter_std
            )
            train_labels.append(record.class_id)
            train_levels.append(record.level)

        for _ in range(config.test_repeats_per_combo):
            test_part_a.append(record.part_a)
            test_part_b.append(record.part_b)
            test_part_a_noise.append(
                torch.randn(config.embedding_dim, generator=test_noise_generator) * config.input_jitter_std
            )
            test_part_b_noise.append(
                torch.randn(config.embedding_dim, generator=test_noise_generator) * config.input_jitter_std
            )
            test_labels.append(record.class_id)
            test_levels.append(record.level)

    return E19Dataset(
        train=SplitTensors(
            part_a=torch.tensor(train_part_a, dtype=torch.long),
            part_b=torch.tensor(train_part_b, dtype=torch.long),
            part_a_noise=torch.stack(train_part_a_noise).to(dtype=torch.float32),
            part_b_noise=torch.stack(train_part_b_noise).to(dtype=torch.float32),
            labels=torch.tensor(train_labels, dtype=torch.long),
            levels=torch.tensor(train_levels, dtype=torch.long),
        ),
        test=SplitTensors(
            part_a=torch.tensor(test_part_a, dtype=torch.long),
            part_b=torch.tensor(test_part_b, dtype=torch.long),
            part_a_noise=torch.stack(test_part_a_noise).to(dtype=torch.float32),
            part_b_noise=torch.stack(test_part_b_noise).to(dtype=torch.float32),
            labels=torch.tensor(test_labels, dtype=torch.long),
            levels=torch.tensor(test_levels, dtype=torch.long),
        ),
    )


def make_loader(split: SplitTensors, config: E19Config, *, shuffle: bool) -> DataLoader[tuple[torch.Tensor, ...]]:
    """Create a deterministic DataLoader for one split."""
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    dataset = TensorDataset(
        split.part_a,
        split.part_b,
        split.part_a_noise,
        split.part_b_noise,
        split.labels,
        split.levels,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def make_model(config: E19Config) -> ForwardGatedLiveBankMLP:
    """Instantiate the approved 2-layer MLP with the unchanged E16 live bank and gate."""
    return ForwardGatedLiveBankMLP(
        part_a_vocab_size=config.num_part_a_values,
        part_b_vocab_size=config.num_part_b_values,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        bank_capacity=config.bank_capacity,
        min_bank_occupancy=config.min_bank_occupancy,
    )


def train_single_seed(config: E19Config) -> SeedTrainingResult:
    """Train one E19 seed and return Deliverable 1 metrics."""
    return train_single_seed_artifacts(config).result


def train_single_seed_artifacts(config: E19Config) -> TrainedSeedArtifacts:
    """Train one E19 seed and return the trained model plus materialized dataset."""
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

        for part_a, part_b, part_a_noise, part_b_noise, labels, _levels in train_loader:
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

    level_accuracies = evaluate_level_accuracies(model, dataset.test, config)
    result = SeedTrainingResult(
        seed=config.seed,
        stopping_epoch=stopping_epoch,
        final_training_loss=final_training_loss,
        final_training_accuracy=final_training_accuracy,
        level_accuracies=level_accuracies,
    )
    return TrainedSeedArtifacts(config=config, model=model, dataset=dataset, result=result)


def evaluate_level_accuracies(
    model: ForwardGatedLiveBankMLP,
    split: SplitTensors,
    config: E19Config,
) -> tuple[LevelAccuracySummary, ...]:
    """Evaluate test accuracy broken down by the four frozen familiarity levels."""
    device = torch.device(config.device)
    loader = make_loader(split, config, shuffle=False)
    model.eval()

    correct_by_level = {level: 0 for level in LEVEL_ORDER}
    total_by_level = {level: 0 for level in LEVEL_ORDER}

    with torch.no_grad():
        for part_a, part_b, part_a_noise, part_b_noise, labels, levels in loader:
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

            for level in LEVEL_ORDER:
                level_mask = levels == level
                if int(level_mask.sum().item()) == 0:
                    continue
                correct_by_level[level] += int((predictions[level_mask] == labels_cpu[level_mask]).sum().item())
                total_by_level[level] += int(level_mask.sum().item())

    return tuple(
        LevelAccuracySummary(
            level=level,
            level_name=LEVEL_NAMES[level],
            accuracy=correct_by_level[level] / total_by_level[level],
        )
        for level in LEVEL_ORDER
    )


def evaluate_seed0_layer1_certainty_by_level(
    model: ForwardGatedLiveBankMLP,
    split: SplitTensors,
    config: E19Config,
) -> tuple[Seed0CertaintyLevelSummary, ...]:
    """Evaluate mean layer-1 certainty per level for the seed-0 report."""
    device = torch.device(config.device)
    loader = make_loader(split, config, shuffle=False)
    model.eval()

    certainties_by_level: dict[int, list[torch.Tensor]] = {level: [] for level in LEVEL_ORDER}

    with torch.no_grad():
        for part_a, part_b, part_a_noise, part_b_noise, _labels, levels in loader:
            part_a = part_a.to(device)
            part_b = part_b.to(device)
            part_a_noise = part_a_noise.to(device)
            part_b_noise = part_b_noise.to(device)

            outputs = model(
                part_a,
                part_b,
                part_a_noise=part_a_noise,
                part_b_noise=part_b_noise,
            )
            layer1_certainty = outputs.layer1_certainty.detach().cpu()

            for level in LEVEL_ORDER:
                level_mask = levels == level
                if int(level_mask.sum().item()) > 0:
                    certainties_by_level[level].append(layer1_certainty[level_mask])

    return tuple(
        Seed0CertaintyLevelSummary(
            level=level,
            level_name=LEVEL_NAMES[level],
            mean_certainty=float(torch.cat(certainties_by_level[level]).mean().item()),
        )
        for level in LEVEL_ORDER
    )


def _collect_test_certainties_by_level(
    model: ForwardGatedLiveBankMLP,
    split: SplitTensors,
    config: E19Config,
) -> dict[str, dict[int, np.ndarray]]:
    """Collect per-example layer certainties grouped by frozen familiarity level."""
    device = torch.device(config.device)
    loader = make_loader(split, config, shuffle=False)
    model.eval()

    layer1_by_level: dict[int, list[torch.Tensor]] = {level: [] for level in LEVEL_ORDER}
    layer2_by_level: dict[int, list[torch.Tensor]] = {level: [] for level in LEVEL_ORDER}

    with torch.no_grad():
        for part_a, part_b, part_a_noise, part_b_noise, _labels, levels in loader:
            part_a = part_a.to(device)
            part_b = part_b.to(device)
            part_a_noise = part_a_noise.to(device)
            part_b_noise = part_b_noise.to(device)

            outputs = model(
                part_a,
                part_b,
                part_a_noise=part_a_noise,
                part_b_noise=part_b_noise,
            )
            layer1_certainty = outputs.layer1_certainty.detach().cpu()
            layer2_certainty = outputs.layer2_certainty.detach().cpu()

            for level in LEVEL_ORDER:
                level_mask = levels == level
                if int(level_mask.sum().item()) == 0:
                    continue
                layer1_by_level[level].append(layer1_certainty[level_mask])
                layer2_by_level[level].append(layer2_certainty[level_mask])

    return {
        "layer1": {
            level: torch.cat(layer1_by_level[level]).numpy()
            for level in LEVEL_ORDER
        },
        "layer2": {
            level: torch.cat(layer2_by_level[level]).numpy()
            for level in LEVEL_ORDER
        },
    }


def _jonckheere_terpstra_greater(level_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> tuple[float, float]:
    """Compute a one-sided Jonckheere-Terpstra test for ordered means decreasing by level rank."""
    statistic = 0.0
    for left_index, left_values in enumerate(level_arrays[:-1]):
        left_column = left_values[:, None]
        for right_values in level_arrays[left_index + 1 :]:
            right_row = right_values[None, :]
            statistic += float(np.count_nonzero(left_column > right_row))
            statistic += 0.5 * float(np.count_nonzero(left_column == right_row))

    group_sizes = np.array([values.size for values in level_arrays], dtype=np.float64)
    total_size = float(group_sizes.sum())
    mean_under_null = (total_size**2 - float(np.square(group_sizes).sum())) / 4.0
    variance_under_null = (
        total_size**2 * (2.0 * total_size + 3.0)
        - float((np.square(group_sizes) * (2.0 * group_sizes + 3.0)).sum())
    ) / 72.0
    if variance_under_null <= 0.0:
        return statistic, 1.0

    z_score = (statistic - mean_under_null) / sqrt(variance_under_null)
    return statistic, float(norm.sf(z_score))


def analyze_primary_seed(
    artifacts: TrainedSeedArtifacts,
) -> tuple[LayerPrimaryAnalysis, LayerPrimaryAnalysis]:
    """Run the frozen ordered-certainty analysis for one trained seed."""
    certainties_by_layer = _collect_test_certainties_by_level(
        artifacts.model,
        artifacts.dataset.test,
        artifacts.config,
    )
    analyses: list[LayerPrimaryAnalysis] = []

    for layer_name in ("layer1", "layer2"):
        level_arrays = tuple(
            certainties_by_layer[layer_name][level] for level in LEVEL_ORDER
        )
        level_means = tuple(float(values.mean()) for values in level_arrays)
        jt_statistic, jt_p_value = _jonckheere_terpstra_greater(level_arrays)
        strict_mean_ordering = bool(
            level_means[0] > level_means[1] > level_means[2] > level_means[3]
        )
        rank_vector = np.concatenate(
            [
                np.full(certainties_by_layer[layer_name][level].shape[0], level, dtype=np.int64)
                for level in LEVEL_ORDER
            ]
        )
        certainty_vector = np.concatenate(
            [certainties_by_layer[layer_name][level] for level in LEVEL_ORDER]
        )
        spearman_result = spearmanr(rank_vector, certainty_vector)
        analyses.append(
            LayerPrimaryAnalysis(
                seed=artifacts.result.seed,
                layer_name=layer_name,
                jonckheere_terpstra_statistic=jt_statistic,
                jonckheere_terpstra_p_value=jt_p_value,
                strict_mean_ordering=strict_mean_ordering,
                spearman_rho=float(spearman_result.statistic),
                level_means=level_means,
            )
        )

    return analyses[0], analyses[1]


def aggregate_primary_analysis(
    per_seed_layer_analyses: tuple[LayerPrimaryAnalysis, ...],
) -> AggregatedLayerPrimaryAnalysis:
    """Aggregate the ordered-certainty analysis across seeds for one layer."""
    spearman_rhos = np.array([item.spearman_rho for item in per_seed_layer_analyses], dtype=np.float64)
    return AggregatedLayerPrimaryAnalysis(
        layer_name=per_seed_layer_analyses[0].layer_name,
        significant_seed_count=sum(item.jonckheere_terpstra_p_value < 0.05 for item in per_seed_layer_analyses),
        strict_mean_ordering_seed_count=sum(item.strict_mean_ordering for item in per_seed_layer_analyses),
        mean_spearman_rho=float(spearman_rhos.mean()),
        std_spearman_rho=float(spearman_rhos.std(ddof=0)),
    )


def run_training_suite(seeds: list[int] | tuple[int, ...]) -> list[TrainedSeedArtifacts]:
    """Run Deliverable 1 training over the requested seeds."""
    return [train_single_seed_artifacts(E19Config(seed=seed)) for seed in seeds]
