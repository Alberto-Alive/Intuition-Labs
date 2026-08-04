from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class SyntheticDatasetConfig:
    n_train: int = 1600
    n_dev: int = 500
    n_test: int = 800
    num_tasks: int = 4
    num_classes: int = 4
    feature_mod: int = 997


@dataclass(frozen=True)
class SyntheticExample:
    example_id: str
    global_index: int
    split: str
    task_id: int
    x: Tuple[int, int, int]
    label: int
    prompt: str


def deterministic_label(
    task_id: int,
    x: Tuple[int, int, int],
    num_classes: int,
) -> int:
    """Deterministic arithmetic labels for an inspectable synthetic benchmark."""
    a, b, c = x
    if task_id == 0:
        value = a + 2 * b + c
    elif task_id == 1:
        value = a * b + 3 * c
    elif task_id == 2:
        value = (a ^ b ^ c) + a
    elif task_id == 3:
        value = 3 * a - b + 2 * c
    else:
        value = (task_id + 1) * a + (task_id + 3) * b - c
    return int(value % num_classes)


def _features_from_index(global_index: int, seed: int, feature_mod: int) -> Tuple[int, int, int]:
    # Large co-prime-ish multipliers make accidental duplicate prompts rare while
    # keeping generation deterministic and reproducible.
    a = (37 * global_index + 101 * seed + 17) % feature_mod
    b = (83 * global_index + 53 * seed + 29) % feature_mod
    c = (131 * global_index + 19 * seed + 43) % feature_mod
    return int(a), int(b), int(c)


def _task_from_index(global_index: int, seed: int, num_tasks: int) -> int:
    return int((17 * global_index + 31 * seed + 7) % num_tasks)


def _build_split(
    split: str,
    n_examples: int,
    start_index: int,
    seed: int,
    config: SyntheticDatasetConfig,
    seen_prompts: set[Tuple[int, Tuple[int, int, int]]],
) -> List[SyntheticExample]:
    examples: List[SyntheticExample] = []
    global_index = start_index
    while len(examples) < n_examples:
        x = _features_from_index(global_index, seed, config.feature_mod)
        task_id = _task_from_index(global_index, seed, config.num_tasks)
        prompt_key = (task_id, x)
        if prompt_key not in seen_prompts:
            label = deterministic_label(task_id, x, config.num_classes)
            prompt = (
                f"task={task_id}; a={x[0]}; b={x[1]}; c={x[2]}; "
                f"return class in [0,{config.num_classes - 1}]"
            )
            examples.append(
                SyntheticExample(
                    example_id=f"{split}-{len(examples):05d}",
                    global_index=global_index,
                    split=split,
                    task_id=task_id,
                    x=x,
                    label=label,
                    prompt=prompt,
                )
            )
            seen_prompts.add(prompt_key)
        global_index += 1
    return examples


def build_synthetic_splits(
    config: SyntheticDatasetConfig,
    seed: int,
) -> Dict[str, List[SyntheticExample]]:
    """Create train/dev/test splits without duplicate prompts across splits."""
    seen_prompts: set[Tuple[int, Tuple[int, int, int]]] = set()
    train = _build_split("train", config.n_train, 0, seed, config, seen_prompts)
    dev = _build_split("dev", config.n_dev, 1_000_000, seed, config, seen_prompts)
    test = _build_split("test", config.n_test, 2_000_000, seed, config, seen_prompts)
    return {"train": train, "dev": dev, "test": test}


def labels_for(examples: Iterable[SyntheticExample]) -> np.ndarray:
    return np.asarray([example.label for example in examples], dtype=np.int64)

