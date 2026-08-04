from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class StrictCoordinationConfig:
    n_train: int = 1600
    n_dev: int = 500
    n_test: int = 800
    num_classes: int = 4
    num_tasks: int = 1
    n_evidence_bits: int = 4


@dataclass(frozen=True)
class StrictCoordinationExample:
    example_id: str
    global_index: int
    split: str
    task_id: int
    evidence_bits: Tuple[int, ...]
    label: int
    prompt: str


def strict_label(evidence_bits: Tuple[int, ...]) -> int:
    if len(evidence_bits) < 4:
        raise ValueError("strict coordination benchmark requires at least four evidence bits")
    high = evidence_bits[0] ^ evidence_bits[1]
    low = evidence_bits[2] ^ evidence_bits[3]
    return int((high << 1) | low)


def build_strict_coordination_splits(
    config: StrictCoordinationConfig,
    seed: int,
) -> Dict[str, List[StrictCoordinationExample]]:
    seen: set[Tuple[int, ...]] = set()
    train = _build_split("train", config.n_train, 0, seed, config, seen)
    dev = _build_split("dev", config.n_dev, 1_000_000, seed, config, seen)
    test = _build_split("test", config.n_test, 2_000_000, seed, config, seen)
    return {"train": train, "dev": dev, "test": test}


def _build_split(
    split: str,
    n_examples: int,
    start_index: int,
    seed: int,
    config: StrictCoordinationConfig,
    seen: set[Tuple[int, ...]],
) -> List[StrictCoordinationExample]:
    examples: List[StrictCoordinationExample] = []
    global_index = start_index
    while len(examples) < n_examples:
        bits = _evidence_from_index(global_index, seed, config.n_evidence_bits)
        # Include the full deterministic bit tuple and index phase in the key;
        # many bit patterns repeat by design, but prompts remain unique.
        key = bits + (global_index % 9973,)
        if key not in seen:
            seen.add(key)
            label = strict_label(bits)
            prompt = (
                "strict_partial_evidence; "
                f"agents observe one private bit each; evidence_id={global_index}"
            )
            examples.append(
                StrictCoordinationExample(
                    example_id=f"{split}-{len(examples):05d}",
                    global_index=global_index,
                    split=split,
                    task_id=0,
                    evidence_bits=bits,
                    label=label,
                    prompt=prompt,
                )
            )
        global_index += 1
    return examples


def _evidence_from_index(global_index: int, seed: int, n_bits: int) -> Tuple[int, ...]:
    values = []
    for bit_id in range(n_bits):
        raw = (
            (global_index + 1) * (1_103_515_245 + 97_531 * bit_id)
            + (seed + 3) * (12_345 + 43_721 * bit_id)
            + 2_654_435_761 * (bit_id + 1)
        ) & 0xFFFFFFFF
        raw ^= (raw >> (3 + bit_id)) & 0xFFFFFFFF
        values.append(int(raw & 1))
    return tuple(values)


def labels_for(examples: Iterable[StrictCoordinationExample]) -> np.ndarray:
    return np.asarray([example.label for example in examples], dtype=np.int64)
