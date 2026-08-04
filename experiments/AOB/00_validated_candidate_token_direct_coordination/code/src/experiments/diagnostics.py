from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import torch
from torch.nn import functional as F

from src.agents.types import AttemptBatch
from src.coordinators.activation import _assign, _kmeans
from src.coordinators.mlp import MLPClassifier, MLPTrainingConfig
from src.evaluation.metrics import accuracy, binary_auc
from src.telemetry.features import (
    agent_activation_features,
    agent_text_activation_features,
    agent_text_features,
    pooled_hidden_states,
)


ArrayFeatureBuilder = Callable[[AttemptBatch], np.ndarray]


@dataclass
class FittedAgentCorrectnessProbe:
    benchmark: str
    seed: int
    agent_id: int
    feature_mode: str
    feature_builder: ArrayFeatureBuilder
    model: MLPClassifier
    device: torch.device
    param_count: int

    def evaluate(self, batch: AttemptBatch) -> Dict[str, object]:
        x = torch.as_tensor(self.feature_builder(batch), dtype=torch.float32, device=self.device)
        labels = (batch.answers[:, self.agent_id] == batch.labels).astype(np.int64)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        return {
            "benchmark": self.benchmark,
            "seed": self.seed,
            "split": batch.split,
            "probe": "agent_correctness_probe",
            "agent_id": self.agent_id,
            "feature_mode": self.feature_mode,
            "accuracy": accuracy(preds, labels),
            "auc": binary_auc(labels, probs),
            "n_examples": batch.n_examples,
            "param_count": self.param_count,
        }


@dataclass
class FittedRedundancyProbe:
    benchmark: str
    seed: int
    centers: np.ndarray
    clusters: int
    pooling: str

    def evaluate(self, batch: AttemptBatch) -> List[Dict[str, object]]:
        hidden = pooled_hidden_states(batch, pooling=self.pooling)
        flat_hidden = hidden.reshape(batch.n_examples * batch.n_agents, batch.hidden_dim)
        assignments = _assign(flat_hidden, self.centers)
        answer_labels = batch.answers.reshape(-1)
        correctness_labels = (batch.answers == batch.labels[:, None]).reshape(-1).astype(np.int64)
        return [
            {
                "benchmark": self.benchmark,
                "seed": self.seed,
                "split": batch.split,
                "probe": "redundancy_probe",
                "metric": "cluster_purity_by_answer_label",
                "value": _cluster_purity(assignments, answer_labels, self.clusters),
                "clusters": self.clusters,
            },
            {
                "benchmark": self.benchmark,
                "seed": self.seed,
                "split": batch.split,
                "probe": "redundancy_probe",
                "metric": "cluster_purity_by_correctness",
                "value": _cluster_purity(assignments, correctness_labels, self.clusters),
                "clusters": self.clusters,
            },
        ]


def fit_agent_correctness_probes(
    benchmark: str,
    seed: int,
    train_batch: AttemptBatch,
    dev_batch: AttemptBatch,
    num_classes: int,
    num_tasks: int,
    training: MLPTrainingConfig,
    device: str,
) -> tuple[List[Dict[str, object]], List[FittedAgentCorrectnessProbe]]:
    fitted: List[FittedAgentCorrectnessProbe] = []
    dev_metrics: List[Dict[str, object]] = []
    for agent_id in range(train_batch.n_agents):
        builders: Dict[str, ArrayFeatureBuilder] = {
            "text_output_only": lambda batch, a=agent_id: agent_text_features(batch, a, num_classes, num_tasks),
            "activation_only": lambda batch, a=agent_id: agent_activation_features(batch, a),
            "text_plus_activation": lambda batch, a=agent_id: agent_text_activation_features(
                batch, a, num_classes, num_tasks
            ),
        }
        for feature_mode, builder in builders.items():
            model, param_count = _fit_binary_array_mlp(
                x_train=builder(train_batch),
                y_train=(train_batch.answers[:, agent_id] == train_batch.labels).astype(np.int64),
                x_dev=builder(dev_batch),
                y_dev=(dev_batch.answers[:, agent_id] == dev_batch.labels).astype(np.int64),
                training=training,
                seed=seed + 10_000 + agent_id * 97 + len(feature_mode),
                device=device,
            )
            probe = FittedAgentCorrectnessProbe(
                benchmark=benchmark,
                seed=seed,
                agent_id=agent_id,
                feature_mode=feature_mode,
                feature_builder=builder,
                model=model,
                device=torch.device(device),
                param_count=param_count,
            )
            dev_metrics.append(probe.evaluate(dev_batch))
            fitted.append(probe)
    return dev_metrics, fitted


def fit_redundancy_probe(
    benchmark: str,
    seed: int,
    train_batch: AttemptBatch,
    dev_batch: AttemptBatch,
    clusters: int,
    pooling: str = "mean",
) -> tuple[List[Dict[str, object]], FittedRedundancyProbe]:
    hidden = pooled_hidden_states(train_batch, pooling=pooling)
    flat_hidden = hidden.reshape(train_batch.n_examples * train_batch.n_agents, train_batch.hidden_dim)
    centers, _ = _kmeans(flat_hidden, clusters, seed + 61_001)
    probe = FittedRedundancyProbe(
        benchmark=benchmark,
        seed=seed,
        centers=centers,
        clusters=min(clusters, flat_hidden.shape[0]),
        pooling=pooling,
    )
    return probe.evaluate(dev_batch), probe


def _fit_binary_array_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    training: MLPTrainingConfig,
    seed: int,
    device: str,
) -> tuple[MLPClassifier, int]:
    torch_device = torch.device(device)
    torch.manual_seed(seed + 2_003)
    model = MLPClassifier(x_train.shape[1], training.hidden_dims, 2).to(torch_device)
    param_count = int(sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.lr, weight_decay=training.weight_decay)
    tx = torch.as_tensor(x_train, dtype=torch.float32, device=torch_device)
    ty = torch.as_tensor(y_train, dtype=torch.long, device=torch_device)
    dx = torch.as_tensor(x_dev, dtype=torch.float32, device=torch_device)
    dy = torch.as_tensor(y_dev, dtype=torch.long, device=torch_device)
    rng = np.random.default_rng(seed + 7_919)
    best_state = None
    best_dev = -1.0
    stale_epochs = 0
    for _epoch in range(training.epochs):
        model.train()
        permutation = rng.permutation(len(y_train))
        for start in range(0, len(y_train), training.batch_size):
            batch_idx = permutation[start : start + training.batch_size]
            idx = torch.as_tensor(batch_idx, dtype=torch.long, device=torch_device)
            logits = model(tx.index_select(0, idx))
            loss = F.cross_entropy(logits, ty.index_select(0, idx))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        dev_acc = _torch_accuracy(model, dx, dy)
        if dev_acc > best_dev:
            best_dev = dev_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= training.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(torch_device)
    model.eval()
    return model, param_count


def _torch_accuracy(model: MLPClassifier, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(x), dim=1)
        return float((preds == y).float().mean().item())


def _cluster_purity(assignments: np.ndarray, labels: np.ndarray, clusters: int) -> float:
    total = 0
    correct = 0
    for cluster_id in range(clusters):
        mask = assignments == cluster_id
        count = int(mask.sum())
        if count == 0:
            continue
        values, counts = np.unique(labels[mask], return_counts=True)
        del values
        correct += int(counts.max())
        total += count
    return float(correct / max(total, 1))

