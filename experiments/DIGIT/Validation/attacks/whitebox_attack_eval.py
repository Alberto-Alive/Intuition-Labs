"""White-box query optimization attack."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..comparative.querying import QueryAccounting, primitive_vector_from_system
from ..data.base import PrivateDataset
from .privacy_common import run_binary_attack


def _score_difference(response_in: Dict[str, Any], response_out: Dict[str, Any]) -> float:
    return float(np.abs(response_in["primitive_vector"].astype(float) - response_out["primitive_vector"].astype(float)).sum())


def _neighbor_dataset(private_data: PrivateDataset, idx: int) -> PrivateDataset:
    from ..data.base import DatasetInfo, TabularPrivateDataset

    mask = torch.ones(private_data.num_records, dtype=torch.bool)
    mask[idx] = False
    feat = private_data.features[mask]
    lbl = private_data.labels[mask]
    info = DatasetInfo(
        dataset_name=private_data.info.dataset_name,
        split_name=f"{private_data.info.split_name}_neighbor",
        n_records=int(feat.shape[0]),
        n_features=private_data.info.n_features,
        label_name=private_data.info.label_name,
        positive_rate=float(lbl.float().mean()) if len(lbl) else 0.0,
        query_fields=private_data.info.query_fields,
        field_vocab_sizes=private_data.info.field_vocab_sizes,
    )
    return TabularPrivateDataset(feat, lbl, private_data.query_fields, info)


def _sample_query_from_logits(logits: np.ndarray, vocab_sizes: List[int], rng: np.random.RandomState) -> torch.Tensor:
    q = torch.zeros(len(vocab_sizes), dtype=torch.long)
    for i, vs in enumerate(vocab_sizes[:-1]):
        probs = np.exp(logits[i, :vs])
        probs = probs / probs.sum()
        q[i] = int(rng.choice(np.arange(1, vs + 1), p=probs))
    q[-1] = int((q[:-1] > 0).sum().item())
    return q


def _optimize_query(
    system_factory,
    private_data: PrivateDataset,
    target_idx: int,
    seed: int,
    steps: int = 40,
    samples_per_step: int = 8,
    lr: float = 0.25,
) -> torch.Tensor:
    neighbor = _neighbor_dataset(private_data, target_idx)
    system_in = system_factory(private_data)
    system_out = system_factory(neighbor)
    vocab_sizes = private_data.info.field_vocab_sizes
    max_vocab = max(vocab_sizes[:-1])
    logits = np.zeros((len(vocab_sizes), max_vocab), dtype=float)
    rng = np.random.RandomState(seed)
    best_query = torch.zeros(len(vocab_sizes), dtype=torch.long)
    best_score = -1.0

    for _ in range(steps):
        sampled: List[torch.Tensor] = []
        rewards: List[float] = []
        for _ in range(samples_per_step):
            q = _sample_query_from_logits(logits, vocab_sizes, rng)
            r_in = primitive_vector_from_system(system_in, q, private_data)
            r_out = primitive_vector_from_system(system_out, q, neighbor)
            reward = _score_difference(r_in, r_out)
            sampled.append(q)
            rewards.append(reward)
            if reward > best_score:
                best_score = reward
                best_query = q.clone()
        baseline = float(np.mean(rewards))
        for q, reward in zip(sampled, rewards):
            advantage = reward - baseline
            for i, value in enumerate(q[:-1].tolist()):
                logits[i, value - 1] += lr * advantage
    return best_query


def run_gradient_query_attack(
    system_factory,
    private_data: PrivateDataset,
    seed: int,
    num_targets: int = 100,
) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    n = min(num_targets, private_data.num_records)
    idxs = rng.choice(private_data.num_records, size=n, replace=False)
    accounting = QueryAccounting()
    X_opt: List[np.ndarray] = []
    y_opt: List[int] = []
    X_rand: List[np.ndarray] = []
    y_rand: List[int] = []

    for offset, idx in enumerate(idxs):
        q_opt = _optimize_query(system_factory, private_data, int(idx), seed * 1000 + offset)
        q_rand = _sample_query_from_logits(
            np.zeros((len(private_data.info.field_vocab_sizes), max(private_data.info.field_vocab_sizes[:-1])), dtype=float),
            private_data.info.field_vocab_sizes,
            np.random.RandomState(seed * 2000 + offset),
        )
        neighbor = _neighbor_dataset(private_data, int(idx))
        system_in = system_factory(private_data)
        system_out = system_factory(neighbor)

        r_in_opt = primitive_vector_from_system(system_in, q_opt, private_data)
        r_out_opt = primitive_vector_from_system(system_out, q_opt, neighbor)
        r_in_rand = primitive_vector_from_system(system_in, q_rand, private_data)
        r_out_rand = primitive_vector_from_system(system_out, q_rand, neighbor)
        accounting.record(4, float(getattr(system_in, "epsilon", 0.0)) if hasattr(system_in, "epsilon") else None)

        X_opt.append(np.concatenate([r_in_opt["primitive_vector"], r_out_opt["primitive_vector"]]).astype(float))
        X_opt.append(np.concatenate([r_out_opt["primitive_vector"], r_in_opt["primitive_vector"]]).astype(float))
        y_opt.extend([1, 0])

        X_rand.append(np.concatenate([r_in_rand["primitive_vector"], r_out_rand["primitive_vector"]]).astype(float))
        X_rand.append(np.concatenate([r_out_rand["primitive_vector"], r_in_rand["primitive_vector"]]).astype(float))
        y_rand.extend([1, 0])

    opt_metrics, _, _, _ = run_binary_attack(np.asarray(X_opt, dtype=float), np.asarray(y_opt, dtype=int), seed)
    rand_metrics, _, _, _ = run_binary_attack(np.asarray(X_rand, dtype=float), np.asarray(y_rand, dtype=int), seed)
    return {
        "optimized_auroc": float(opt_metrics["auroc"]),
        "random_auroc": float(rand_metrics["auroc"]),
        "auroc_gain": float(opt_metrics["auroc"] - rand_metrics["auroc"]),
        "query_count": accounting.query_count,
        "cumulative_epsilon": accounting.cumulative_epsilon,
    }
