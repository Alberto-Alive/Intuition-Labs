from __future__ import annotations

import re
from typing import Iterable, Optional

import numpy as np

from src.agents.types import AttemptBatch


def one_hot(values: np.ndarray, depth: int) -> np.ndarray:
    values = values.astype(np.int64)
    output = np.zeros(values.shape + (depth,), dtype=np.float32)
    np.put_along_axis(output, values[..., None], 1.0, axis=-1)
    return output


def visible_features(
    batch: AttemptBatch,
    num_classes: int,
    num_tasks: int,
) -> np.ndarray:
    """Features available from visible agent outputs only."""
    answers_oh = one_hot(batch.answers, num_classes).reshape(batch.n_examples, -1)
    conf = batch.confidences.astype(np.float32)
    conf_weighted_answers = (
        one_hot(batch.answers, num_classes) * conf[..., None]
    ).reshape(batch.n_examples, -1)
    vote_counts = np.zeros((batch.n_examples, num_classes), dtype=np.float32)
    for class_id in range(num_classes):
        vote_counts[:, class_id] = np.mean(batch.answers == class_id, axis=1)
    max_conf_by_answer = np.zeros((batch.n_examples, num_classes), dtype=np.float32)
    for class_id in range(num_classes):
        mask = batch.answers == class_id
        masked_conf = np.where(mask, conf, -1.0)
        max_conf_by_answer[:, class_id] = np.maximum(masked_conf.max(axis=1), 0.0)
    task_oh = one_hot(batch.task_ids, num_tasks)
    return np.concatenate(
        [
            answers_oh,
            conf,
            conf_weighted_answers,
            vote_counts,
            max_conf_by_answer,
            task_oh,
        ],
        axis=1,
    ).astype(np.float32)


def trace_features(batch: AttemptBatch, rationale_buckets: int = 7) -> np.ndarray:
    """Small structured trace features from visible synthetic rationales."""
    features = np.zeros((batch.n_examples, batch.n_agents, rationale_buckets + 2), dtype=np.float32)
    for row_id, texts in enumerate(batch.visible_texts):
        for agent_id, text in enumerate(texts):
            code = _extract_int_field(text, "rationale_code")
            if code is not None:
                features[row_id, agent_id, code % rationale_buckets] = 1.0
            evidence_bit = _extract_int_field(text, "private_evidence_bit")
            if evidence_bit is not None:
                features[row_id, agent_id, rationale_buckets] = 1.0
                features[row_id, agent_id, rationale_buckets + 1] = float(int(evidence_bit) == 1)
    return features.reshape(batch.n_examples, -1)


def output_oracle_features(
    batch: AttemptBatch,
    num_classes: int,
    num_tasks: int,
) -> np.ndarray:
    """All visible outputs plus parseable visible trace fields, no activations."""
    return np.concatenate(
        [
            visible_features(batch, num_classes=num_classes, num_tasks=num_tasks),
            trace_features(batch),
        ],
        axis=1,
    ).astype(np.float32)


def pooled_hidden_states(
    batch: AttemptBatch,
    layers: Optional[Iterable[int]] = None,
    pooling: str = "mean",
) -> np.ndarray:
    """Pool hidden states over selected layers and keep the agent axis."""
    hidden = batch.hidden_states
    if layers is not None:
        layer_ids = list(layers)
        hidden = hidden[:, :, layer_ids, :]
    if pooling == "mean":
        pooled = hidden.mean(axis=2)
    elif pooling == "last":
        pooled = hidden[:, :, -1, :]
    else:
        raise ValueError(f"unknown pooling mode: {pooling}")
    return pooled.astype(np.float32)


def activation_features(
    batch: AttemptBatch,
    layers: Optional[Iterable[int]] = None,
    pooling: str = "mean",
) -> np.ndarray:
    pooled = pooled_hidden_states(batch, layers=layers, pooling=pooling)
    return pooled.reshape(batch.n_examples, -1).astype(np.float32)


def text_plus_activation_features(
    batch: AttemptBatch,
    num_classes: int,
    num_tasks: int,
    layers: Optional[Iterable[int]] = None,
    pooling: str = "mean",
) -> np.ndarray:
    return np.concatenate(
        [
            visible_features(batch, num_classes=num_classes, num_tasks=num_tasks),
            activation_features(batch, layers=layers, pooling=pooling),
        ],
        axis=1,
    ).astype(np.float32)


def agent_text_features(
    batch: AttemptBatch,
    agent_id: int,
    num_classes: int,
    num_tasks: int,
) -> np.ndarray:
    answer = one_hot(batch.answers[:, agent_id], num_classes)
    confidence = batch.confidences[:, agent_id : agent_id + 1].astype(np.float32)
    task = one_hot(batch.task_ids, num_tasks)
    trace = trace_features(batch).reshape(batch.n_examples, batch.n_agents, -1)[:, agent_id, :]
    return np.concatenate([answer, confidence, task, trace], axis=1).astype(np.float32)


def agent_activation_features(
    batch: AttemptBatch,
    agent_id: int,
    pooling: str = "mean",
) -> np.ndarray:
    pooled = pooled_hidden_states(batch, pooling=pooling)
    return pooled[:, agent_id, :].astype(np.float32)


def agent_text_activation_features(
    batch: AttemptBatch,
    agent_id: int,
    num_classes: int,
    num_tasks: int,
    pooling: str = "mean",
) -> np.ndarray:
    return np.concatenate(
        [
            agent_text_features(batch, agent_id, num_classes, num_tasks),
            agent_activation_features(batch, agent_id, pooling=pooling),
        ],
        axis=1,
    ).astype(np.float32)


def expand_text_features(
    x: np.ndarray,
    target_dim: int,
    seed: int,
) -> np.ndarray:
    """Capacity-matched deterministic text expansion with no activation input."""
    if x.shape[1] >= target_dim:
        return x.astype(np.float32)
    rng = np.random.default_rng(seed + 44_411)
    projection = rng.normal(0.0, 1.0 / max(1, x.shape[1]) ** 0.5, size=(x.shape[1], target_dim - x.shape[1]))
    projected = np.tanh(x @ projection).astype(np.float32)
    return np.concatenate([x, projected], axis=1).astype(np.float32)


def _extract_int_token(text: str, key: str) -> int | None:
    prefix = f"{key}="
    for token in text.split():
        if token.startswith(prefix):
            try:
                return int(token[len(prefix) :])
            except ValueError:
                return None
    return None


def _extract_int_field(text: str, key: str) -> int | None:
    value = _extract_int_token(text, key)
    if value is not None:
        return value
    match = re.search(rf'"{re.escape(key)}"\s*:\s*(-?\d+)', text)
    if match:
        return int(match.group(1))
    return None
