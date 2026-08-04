"""Dataset helpers for DIGIT Extrapolation E26."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.DIGIT.Extrapolation.e19.extrapolation.data import (
    LEVEL_NAMES,
    LEVEL_ORDER,
    FrequencyLevelRecord,
    build_frequency_level_records,
)


@dataclass(frozen=True)
class SplitTensors:
    """Tensorized dataset split for E26."""

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
class E26Dataset:
    """Sequential Task A / Task B data with frozen probe and test splits."""

    task_a_train: SplitTensors
    task_b_train: SplitTensors
    task_a_probe: SplitTensors
    task_b_test: SplitTensors


def _materialize_split(
    records: tuple[FrequencyLevelRecord, ...],
    *,
    allowed_part_b_values: tuple[int, ...],
    repeats_per_combo: int,
    noise_generator: torch.Generator,
    embedding_dim: int,
    input_jitter_std: float,
) -> SplitTensors:
    allowed_part_b = set(allowed_part_b_values)

    part_a_values: list[int] = []
    part_b_values: list[int] = []
    part_a_noise: list[torch.Tensor] = []
    part_b_noise: list[torch.Tensor] = []
    labels: list[int] = []
    levels: list[int] = []

    for record in records:
        if record.part_b not in allowed_part_b:
            continue
        for _ in range(repeats_per_combo):
            part_a_values.append(record.part_a)
            part_b_values.append(record.part_b)
            part_a_noise.append(
                torch.randn(embedding_dim, generator=noise_generator) * input_jitter_std
            )
            part_b_noise.append(
                torch.randn(embedding_dim, generator=noise_generator) * input_jitter_std
            )
            labels.append(record.class_id)
            levels.append(record.level)

    return SplitTensors(
        part_a=torch.tensor(part_a_values, dtype=torch.long),
        part_b=torch.tensor(part_b_values, dtype=torch.long),
        part_a_noise=torch.stack(part_a_noise).to(dtype=torch.float32),
        part_b_noise=torch.stack(part_b_noise).to(dtype=torch.float32),
        labels=torch.tensor(labels, dtype=torch.long),
        levels=torch.tensor(levels, dtype=torch.long),
    )


def build_dataset(config: "E26Config") -> E26Dataset:
    """Materialize the sequential E26 dataset once per seed."""
    records = build_frequency_level_records()

    task_a_train_generator = torch.Generator().manual_seed(config.seed + config.train_noise_seed_offset)
    task_b_train_generator = torch.Generator().manual_seed(config.seed + config.train_noise_seed_offset + 1)
    probe_generator = torch.Generator().manual_seed(config.seed + config.probe_noise_seed_offset)
    test_generator = torch.Generator().manual_seed(config.seed + config.test_noise_seed_offset)

    task_a_train = _materialize_split(
        records,
        allowed_part_b_values=config.task_a_part_b_values,
        repeats_per_combo=config.task_a_train_repeats_per_combo,
        noise_generator=task_a_train_generator,
        embedding_dim=config.embedding_dim,
        input_jitter_std=config.input_jitter_std,
    )
    task_b_train = _materialize_split(
        records,
        allowed_part_b_values=config.task_b_part_b_values,
        repeats_per_combo=config.task_b_train_repeats_per_combo,
        noise_generator=task_b_train_generator,
        embedding_dim=config.embedding_dim,
        input_jitter_std=config.input_jitter_std,
    )
    task_a_probe = _materialize_split(
        records,
        allowed_part_b_values=config.task_a_part_b_values,
        repeats_per_combo=config.task_a_probe_repeats_per_combo,
        noise_generator=probe_generator,
        embedding_dim=config.embedding_dim,
        input_jitter_std=config.input_jitter_std,
    )
    task_b_test = _materialize_split(
        records,
        allowed_part_b_values=config.task_b_part_b_values,
        repeats_per_combo=config.task_b_test_repeats_per_combo,
        noise_generator=test_generator,
        embedding_dim=config.embedding_dim,
        input_jitter_std=config.input_jitter_std,
    )

    return E26Dataset(
        task_a_train=task_a_train,
        task_b_train=task_b_train,
        task_a_probe=task_a_probe,
        task_b_test=task_b_test,
    )


def make_loader(split: SplitTensors, config: "E26Config", *, shuffle: bool) -> DataLoader[tuple[torch.Tensor, ...]]:
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
