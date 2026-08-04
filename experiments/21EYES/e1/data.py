from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class SyntheticTaskConfig:
    n_train: int = 1200
    n_val: int = 400
    n_test: int = 400
    seq_len: int = 18
    vocab_size: int = 64
    num_classes: int = 4
    batch_size: int = 128
    mask_context_token: bool = True


def _make_split(
    n_examples: int,
    *,
    seq_len: int,
    vocab_size: int,
    num_classes: int,
    seed: int,
) -> TensorDataset:
    """Build a synthetic context-conditioned sequence classification split.

    Position 0 is a context token identifying the rule family. Position 2
    contains a class-bearing base token. The context selects a label remapping,
    so the target is `(base_class + context_group) % num_classes`. WDA models may
    use the context token only through the weight controller when configured to
    mask that token in the transformer stream.
    """

    if seq_len < 15:
        raise ValueError("seq_len must be at least 15 for the default task")
    if vocab_size < 16:
        raise ValueError("vocab_size must be at least 16")

    g = torch.Generator().manual_seed(seed)
    groups = torch.randint(0, num_classes, (n_examples,), generator=g)
    base_classes = torch.randint(0, num_classes, (n_examples,), generator=g)
    labels = (base_classes + groups) % num_classes
    tokens = torch.randint(8, vocab_size, (n_examples, seq_len), generator=g)
    tokens[:, 0] = groups

    rule_positions = torch.tensor([2, 6, 10, 14], dtype=torch.long)
    decoys = torch.randint(0, num_classes, (n_examples, num_classes), generator=g)
    decoys[:, 0] = base_classes
    tokens[:, rule_positions] = decoys + 4

    return TensorDataset(tokens.long(), labels.long(), groups.long())


def make_dataloaders(config: SyntheticTaskConfig, seed: int) -> dict[str, DataLoader]:
    train = _make_split(
        config.n_train,
        seq_len=config.seq_len,
        vocab_size=config.vocab_size,
        num_classes=config.num_classes,
        seed=seed * 1009 + 11,
    )
    val = _make_split(
        config.n_val,
        seq_len=config.seq_len,
        vocab_size=config.vocab_size,
        num_classes=config.num_classes,
        seed=seed * 1009 + 23,
    )
    test = _make_split(
        config.n_test,
        seq_len=config.seq_len,
        vocab_size=config.vocab_size,
        num_classes=config.num_classes,
        seed=seed * 1009 + 37,
    )
    return {
        "train": DataLoader(train, batch_size=config.batch_size, shuffle=True),
        "val": DataLoader(val, batch_size=config.batch_size, shuffle=False),
        "test": DataLoader(test, batch_size=config.batch_size, shuffle=False),
    }


CapacityBenchmark = Literal[
    "rule_switching_sequence",
    "compositional_rule_switching",
    "low_capacity_generalization",
    "longer_seq_rule_switching",
]


@dataclass(frozen=True)
class CapacityTaskConfig:
    n_train: int = 5000
    n_val: int = 1000
    n_test: int = 1000
    seq_len: int = 24
    vocab_size: int = 64
    num_classes: int = 16
    batch_size: int = 128
    mask_context_token: bool = True


def _values_from_tokens(tokens: torch.Tensor, num_classes: int) -> torch.Tensor:
    return (tokens % num_classes).long()


def _rule_switch_labels(values: torch.Tensor, rules: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Vectorized labels for context-selected sequence rules."""

    labels_by_rule = torch.stack(
        [
            values[:, 0],
            values[:, -1],
            (values[:, 0] + values[:, 1]) % num_classes,
            (values[:, 0] + 2 * values[:, 3] + 3) % num_classes,
            values[:, : min(8, values.shape[1])].sum(dim=1) % num_classes,
            (values[:, 2].bitwise_xor(values[:, 5])) % num_classes,
            (values[:, : min(12, values.shape[1])] < (num_classes // 2)).sum(dim=1)
            % num_classes,
            (values[:, ::2].sum(dim=1) - values[:, 1::2].sum(dim=1)) % num_classes,
        ],
        dim=1,
    )
    return labels_by_rule.gather(1, rules.view(-1, 1)).squeeze(1)


def _apply_composition_rule(
    value: torch.Tensor,
    rule: torch.Tensor,
    values: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    candidates = torch.stack(
        [
            value,
            (value + values[:, 1] + 1) % num_classes,
            value.bitwise_xor(values[:, 2]),
            (3 * value + values[:, 3]) % num_classes,
            (value + 7 * (values[:, 4:10].sum(dim=1) % 2)) % num_classes,
            (value + values[:, 5] - values[:, 6]) % num_classes,
        ],
        dim=1,
    )
    return candidates.gather(1, rule.view(-1, 1)).squeeze(1)


def _sample_composition_pairs(
    n_examples: int,
    *,
    generator: torch.Generator,
    split: str,
    heldout_only: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_pairs = torch.tensor(
        [(first, second) for first in range(6) for second in range(6)],
        dtype=torch.long,
    )
    heldout_mask = (all_pairs[:, 0] + 2 * all_pairs[:, 1]) % 5 == 0
    if heldout_only:
        pool = all_pairs[heldout_mask]
    elif split == "train":
        pool = all_pairs[~heldout_mask]
    else:
        pool = all_pairs
    indices = torch.randint(0, pool.shape[0], (n_examples,), generator=generator)
    pairs = pool[indices]
    return pairs[:, 0], pairs[:, 1]


def _make_capacity_split(
    n_examples: int,
    *,
    benchmark: CapacityBenchmark,
    split: str,
    seq_len: int,
    vocab_size: int,
    num_classes: int,
    seed: int,
) -> TensorDataset:
    if seq_len < 12:
        raise ValueError("seq_len must be at least 12 for capacity benchmarks")
    if vocab_size < max(32, num_classes + 16):
        raise ValueError("vocab_size must leave room for context and payload tokens")

    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(16, vocab_size, (n_examples, seq_len), generator=g)
    payload_start = 2
    values = _values_from_tokens(tokens[:, payload_start:], num_classes)

    if benchmark in {"rule_switching_sequence", "longer_seq_rule_switching"}:
        rules = torch.randint(0, 8, (n_examples,), generator=g)
        tokens[:, 0] = rules
        tokens[:, 1] = 8 + torch.randint(0, 8, (n_examples,), generator=g)
        labels = _rule_switch_labels(values, rules, num_classes)
        groups = rules
    elif benchmark == "compositional_rule_switching":
        first, second = _sample_composition_pairs(
            n_examples,
            generator=g,
            split=split,
            heldout_only=False,
        )
        tokens[:, 0] = 8 + first
        tokens[:, 1] = 16 + second
        mid = _apply_composition_rule(values[:, 0], first, values, num_classes)
        labels = _apply_composition_rule(mid, second, values, num_classes)
        groups = first * 6 + second
    elif benchmark == "low_capacity_generalization":
        first, second = _sample_composition_pairs(
            n_examples,
            generator=g,
            split=split,
            heldout_only=split in {"val", "test"},
        )
        tokens[:, 0] = 8 + first
        tokens[:, 1] = 16 + second
        mid = _apply_composition_rule(values[:, 0], first, values, num_classes)
        labels = _apply_composition_rule(mid, second, values, num_classes)
        groups = first * 6 + second
    else:  # pragma: no cover - Literal keeps this closed for normal callers.
        raise ValueError(f"unknown benchmark={benchmark!r}")

    return TensorDataset(tokens.long(), labels.long(), groups.long())


def make_capacity_dataloaders(
    config: CapacityTaskConfig,
    seed: int,
    benchmark: CapacityBenchmark,
) -> dict[str, DataLoader]:
    seq_len = config.seq_len
    if benchmark == "longer_seq_rule_switching":
        train_seq_len = max(12, seq_len - 8)
    else:
        train_seq_len = seq_len

    train = _make_capacity_split(
        config.n_train,
        benchmark=benchmark,
        split="train",
        seq_len=train_seq_len,
        vocab_size=config.vocab_size,
        num_classes=config.num_classes,
        seed=seed * 1009 + 101,
    )
    val = _make_capacity_split(
        config.n_val,
        benchmark=benchmark,
        split="val",
        seq_len=seq_len,
        vocab_size=config.vocab_size,
        num_classes=config.num_classes,
        seed=seed * 1009 + 211,
    )
    test = _make_capacity_split(
        config.n_test,
        benchmark=benchmark,
        split="test",
        seq_len=seq_len,
        vocab_size=config.vocab_size,
        num_classes=config.num_classes,
        seed=seed * 1009 + 307,
    )
    return {
        "train": DataLoader(train, batch_size=config.batch_size, shuffle=True),
        "val": DataLoader(val, batch_size=config.batch_size, shuffle=False),
        "test": DataLoader(test, batch_size=config.batch_size, shuffle=False),
    }
