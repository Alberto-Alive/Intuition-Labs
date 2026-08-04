"""Controlled compositional Regime 2 dataset for E12."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import E12Config


@dataclass(frozen=True)
class Regime2Data:
    train: TensorDataset
    val: TensorDataset
    test: TensorDataset
    probe_seen_tokens: torch.Tensor
    probe_perturbed_tokens: torch.Tensor
    color_frequencies: torch.Tensor
    shape_frequencies: torch.Tensor


def is_withheld_combination(color_id: int, shape_id: int, config: E12Config) -> bool:
    return ((color_id + shape_id) % config.withheld_modulus) == config.withheld_remainder


def label_for_combination(color_id: int, shape_id: int) -> int:
    return ((color_id % 2) << 1) | (shape_id % 2)


def tokens_for_combination(color_id: int, shape_id: int, config: E12Config) -> list[int]:
    return [config.cls_token_id, 1 + color_id, config.shape_token_offset + shape_id]


def _build_examples(
    combos: list[tuple[int, int]],
    repeats: int,
    config: E12Config,
    seen_label: int | None = None,
) -> tuple[torch.Tensor, ...]:
    tokens = []
    labels = []
    seen_unseen = []
    color_ids = []
    shape_ids = []

    for _ in range(repeats):
        for color_id, shape_id in combos:
            tokens.append(tokens_for_combination(color_id, shape_id, config))
            labels.append(label_for_combination(color_id, shape_id))
            seen_unseen.append(
                0 if seen_label is None and not is_withheld_combination(color_id, shape_id, config) else 1
                if seen_label is None
                else seen_label
            )
            color_ids.append(color_id)
            shape_ids.append(shape_id)

    return (
        torch.tensor(tokens, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(seen_unseen, dtype=torch.long),
        torch.tensor(color_ids, dtype=torch.long),
        torch.tensor(shape_ids, dtype=torch.long),
    )


def _find_mild_perturbation(color_id: int, shape_id: int, config: E12Config) -> tuple[int, int]:
    for new_color in range(config.num_colors):
        if new_color == color_id:
            continue
        if is_withheld_combination(new_color, shape_id, config):
            return new_color, shape_id

    for new_shape in range(config.num_shapes):
        if new_shape == shape_id:
            continue
        if is_withheld_combination(color_id, new_shape, config):
            return color_id, new_shape

    for new_color in range(config.num_colors):
        if new_color != color_id:
            return new_color, shape_id

    for new_shape in range(config.num_shapes):
        if new_shape != shape_id:
            return color_id, new_shape

    raise RuntimeError("Unable to construct a mild perturbation")


def build_regime2_data(config: E12Config) -> Regime2Data:
    seen_combos = []
    withheld_combos = []
    for color_id in range(config.num_colors):
        for shape_id in range(config.num_shapes):
            if is_withheld_combination(color_id, shape_id, config):
                withheld_combos.append((color_id, shape_id))
            else:
                seen_combos.append((color_id, shape_id))

    train_tensors = _build_examples(
        combos=seen_combos,
        repeats=config.train_repeats_per_seen_combo,
        config=config,
        seen_label=0,
    )
    val_tensors = _build_examples(
        combos=seen_combos,
        repeats=config.val_repeats_per_seen_combo,
        config=config,
        seen_label=0,
    )
    all_test_combos = seen_combos + withheld_combos
    test_tensors = _build_examples(
        combos=all_test_combos,
        repeats=config.test_repeats_per_combo,
        config=config,
        seen_label=None,
    )

    color_frequencies = torch.zeros(config.num_colors, dtype=torch.float32)
    shape_frequencies = torch.zeros(config.num_shapes, dtype=torch.float32)
    for color_id, shape_id in seen_combos:
        color_frequencies[color_id] += config.train_repeats_per_seen_combo
        shape_frequencies[shape_id] += config.train_repeats_per_seen_combo

    probe_color_ids = val_tensors[3][: config.warmup_probe_size]
    probe_shape_ids = val_tensors[4][: config.warmup_probe_size]
    probe_seen_tokens = val_tensors[0][: config.warmup_probe_size]
    perturbed_tokens = []
    for color_id, shape_id in zip(probe_color_ids.tolist(), probe_shape_ids.tolist(), strict=True):
        perturbed_color, perturbed_shape = _find_mild_perturbation(color_id, shape_id, config)
        perturbed_tokens.append(tokens_for_combination(perturbed_color, perturbed_shape, config))

    return Regime2Data(
        train=TensorDataset(*train_tensors),
        val=TensorDataset(*val_tensors),
        test=TensorDataset(*test_tensors),
        probe_seen_tokens=probe_seen_tokens,
        probe_perturbed_tokens=torch.tensor(perturbed_tokens, dtype=torch.long),
        color_frequencies=color_frequencies,
        shape_frequencies=shape_frequencies,
    )


def make_loader(dataset: TensorDataset, config: E12Config, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=generator,
    )
