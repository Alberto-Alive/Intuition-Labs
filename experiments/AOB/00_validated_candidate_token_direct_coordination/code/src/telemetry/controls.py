from __future__ import annotations

from typing import Dict

import numpy as np

from src.agents.types import AttemptBatch


CONTROL_NAMES = {
    "none",
    "shuffled_activations",
    "random_activations",
    "wrong_task_activations",
    "wrong_agent_activations",
}


def apply_activation_control(
    batches: Dict[str, AttemptBatch],
    control: str,
    seed: int,
) -> Dict[str, AttemptBatch]:
    if control not in CONTROL_NAMES:
        raise ValueError(f"unknown activation control: {control}")
    if control == "none":
        return {split: batch.copy_with_hidden(batch.hidden_states) for split, batch in batches.items()}
    if control == "random_activations":
        train_hidden = batches["train"].hidden_states
        mean = float(train_hidden.mean())
        std = float(train_hidden.std() + 1e-6)
        return {
            split: batch.copy_with_hidden(
                np.random.default_rng(seed + _split_offset(split)).normal(
                    mean,
                    std,
                    size=batch.hidden_states.shape,
                )
            )
            for split, batch in batches.items()
        }
    if control == "shuffled_activations":
        return {
            split: batch.copy_with_hidden(_shuffle_examples(batch.hidden_states, seed + _split_offset(split)))
            for split, batch in batches.items()
        }
    if control == "wrong_task_activations":
        return {
            split: batch.copy_with_hidden(_wrong_task_hidden(batch, seed + _split_offset(split)))
            for split, batch in batches.items()
        }
    if control == "wrong_agent_activations":
        return {
            split: batch.copy_with_hidden(_wrong_agent_hidden(batch, seed + _split_offset(split)))
            for split, batch in batches.items()
        }
    raise AssertionError("unreachable")


def randomized_train_labels(batch: AttemptBatch, seed: int, num_classes: int) -> AttemptBatch:
    rng = np.random.default_rng(seed + 87_277)
    labels = rng.integers(0, num_classes, size=batch.labels.shape, dtype=np.int64)
    return batch.copy_with_labels(labels)


def _split_offset(split: str) -> int:
    return {"train": 0, "dev": 10_000, "test": 20_000}.get(split, 30_000)


def _shuffle_examples(hidden: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(hidden.shape[0])
    if hidden.shape[0] > 1 and np.all(perm == np.arange(hidden.shape[0])):
        perm = np.roll(perm, 1)
    return hidden[perm].astype(np.float32, copy=True)


def _wrong_task_hidden(batch: AttemptBatch, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = batch.n_examples
    source = np.empty(n, dtype=np.int64)
    all_indices = np.arange(n)
    for i in range(n):
        candidates = all_indices[batch.task_ids != batch.task_ids[i]]
        if len(candidates) == 0:
            source[i] = (i + 1) % n
        else:
            source[i] = int(rng.choice(candidates))
    return batch.hidden_states[source].astype(np.float32, copy=True)


def _wrong_agent_hidden(batch: AttemptBatch, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hidden = batch.hidden_states.copy()
    for i in range(batch.n_examples):
        perm = rng.permutation(batch.n_agents)
        if batch.n_agents > 1 and np.all(perm == np.arange(batch.n_agents)):
            perm = np.roll(perm, 1)
        hidden[i] = hidden[i, perm]
    return hidden.astype(np.float32, copy=True)

