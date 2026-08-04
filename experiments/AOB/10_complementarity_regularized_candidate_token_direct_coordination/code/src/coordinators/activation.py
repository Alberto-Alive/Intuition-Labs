from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from src.agents.types import AttemptBatch
from src.coordinators.mlp import FeatureMLPCoordinator, MLPTrainingConfig
from src.telemetry.features import (
    activation_features,
    expand_text_features,
    pooled_hidden_states,
    text_plus_activation_features,
    visible_features,
)


class TextOnlyMLPCoordinator(FeatureMLPCoordinator):
    def __init__(
        self,
        num_classes: int,
        num_tasks: int,
        training: MLPTrainingConfig,
        seed: int,
        device: str,
    ) -> None:
        super().__init__(
            name="text_only_coordinator",
            feature_builder=lambda batch: visible_features(batch, num_classes, num_tasks),
            num_classes=num_classes,
            training=training,
            seed=seed,
            device=device,
        )


class CapacityMatchedTextOnlyCoordinator(FeatureMLPCoordinator):
    def __init__(
        self,
        num_classes: int,
        num_tasks: int,
        target_dim: int,
        training: MLPTrainingConfig,
        seed: int,
        device: str,
    ) -> None:
        self.target_dim = target_dim
        super().__init__(
            name="capacity_matched_text_only",
            feature_builder=lambda batch: expand_text_features(
                visible_features(batch, num_classes, num_tasks),
                target_dim=target_dim,
                seed=seed,
            ),
            num_classes=num_classes,
            training=training,
            seed=seed,
            device=device,
        )

    def metadata(self) -> Dict[str, object]:
        meta = super().metadata()
        meta["target_feature_dim"] = self.target_dim
        return meta


class ActivationPoolingMLPCoordinator(FeatureMLPCoordinator):
    def __init__(
        self,
        num_classes: int,
        num_tasks: int,
        training: MLPTrainingConfig,
        seed: int,
        device: str,
        pooling: str = "mean",
        name: str = "activation_pca_mlp",
    ) -> None:
        self.pooling = pooling
        super().__init__(
            name="activation_pool_mlp",
            feature_builder=lambda batch: text_plus_activation_features(
                batch,
                num_classes=num_classes,
                num_tasks=num_tasks,
                pooling=pooling,
            ),
            num_classes=num_classes,
            training=training,
            seed=seed,
            device=device,
        )

    def metadata(self) -> Dict[str, object]:
        meta = super().metadata()
        meta["pooling"] = self.pooling
        return meta


class ActivationPCAMLPCoordinator(FeatureMLPCoordinator):
    def __init__(
        self,
        num_classes: int,
        num_tasks: int,
        components: int,
        training: MLPTrainingConfig,
        seed: int,
        device: str,
        pooling: str = "mean",
        name: str = "activation_pca_mlp",
    ) -> None:
        self.num_tasks = num_tasks
        self.components = components
        self.pooling = pooling
        self._mean: Optional[np.ndarray] = None
        self._basis: Optional[np.ndarray] = None
        self._explained_variance_ratio: Optional[np.ndarray] = None
        super().__init__(
            name=name,
            feature_builder=self._features,
            num_classes=num_classes,
            training=training,
            seed=seed,
            device=device,
        )
        self.num_classes_for_features = num_classes

    def fit(self, train_batch: AttemptBatch, dev_batch: AttemptBatch | None = None) -> None:
        train_activation = activation_features(train_batch, pooling=self.pooling)
        self._fit_pca(train_activation)
        super().fit(train_batch, dev_batch)

    def _fit_pca(self, x: np.ndarray) -> None:
        self._mean = x.mean(axis=0, keepdims=True)
        centered = x - self._mean
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        keep = min(self.components, vt.shape[0])
        self._basis = vt[:keep].astype(np.float32)
        variances = (singular_values.astype(np.float64) ** 2) / max(1, x.shape[0] - 1)
        total = float(variances.sum())
        if total > 0.0:
            self._explained_variance_ratio = (variances[:keep] / total).astype(np.float32)
        else:
            self._explained_variance_ratio = np.zeros(keep, dtype=np.float32)

    def _features(self, batch: AttemptBatch) -> np.ndarray:
        if self._mean is None or self._basis is None:
            # During shape probes before fit, use raw activations truncated in a
            # deterministic way. The real fit path installs train-only PCA.
            raw = activation_features(batch, pooling=self.pooling)
            compressed = raw[:, : self.components]
        else:
            raw = activation_features(batch, pooling=self.pooling)
            compressed = (raw - self._mean) @ self._basis.T
        return np.concatenate(
            [
                visible_features(batch, self.num_classes_for_features, self.num_tasks),
                compressed.astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)

    def metadata(self) -> Dict[str, object]:
        meta = super().metadata()
        meta["pca_components"] = self.components
        meta["pooling"] = self.pooling
        if self._explained_variance_ratio is not None:
            meta["pca_explained_variance_ratio_sum"] = float(self._explained_variance_ratio.sum())
            meta["pca_first_component_variance_ratio"] = float(self._explained_variance_ratio[0]) if len(self._explained_variance_ratio) else 0.0
        return meta


class ActivationClusterRouter:
    name = "activation_cluster_router"

    def __init__(
        self,
        clusters: int,
        seed: int,
        pooling: str = "mean",
        confidence_weight: float = 0.05,
    ) -> None:
        self.clusters = clusters
        self.seed = seed
        self.pooling = pooling
        self.confidence_weight = confidence_weight
        self.centers: Optional[np.ndarray] = None
        self.cluster_correct_rate: Optional[np.ndarray] = None
        self.param_count = 0

    def fit(self, train_batch: AttemptBatch, dev_batch: AttemptBatch | None = None) -> None:
        hidden = pooled_hidden_states(train_batch, pooling=self.pooling)
        flat_hidden = hidden.reshape(train_batch.n_examples * train_batch.n_agents, train_batch.hidden_dim)
        correct = (train_batch.answers == train_batch.labels[:, None]).reshape(-1).astype(np.float32)
        self.centers, assignments = _kmeans(flat_hidden, self.clusters, self.seed)
        rates = np.zeros(self.clusters, dtype=np.float32)
        for cluster_id in range(self.clusters):
            mask = assignments == cluster_id
            # Beta(1,1) smoothing avoids overconfident empty or tiny clusters.
            rates[cluster_id] = (float(correct[mask].sum()) + 1.0) / (float(mask.sum()) + 2.0)
        self.cluster_correct_rate = rates

    def predict(self, batch: AttemptBatch) -> np.ndarray:
        if self.centers is None or self.cluster_correct_rate is None:
            raise RuntimeError("activation_cluster_router has not been fit")
        hidden = pooled_hidden_states(batch, pooling=self.pooling)
        flat_hidden = hidden.reshape(batch.n_examples * batch.n_agents, batch.hidden_dim)
        assignments = _assign(flat_hidden, self.centers).reshape(batch.n_examples, batch.n_agents)
        scores = self.cluster_correct_rate[assignments] + self.confidence_weight * batch.confidences
        best_agent = np.argmax(scores, axis=1)
        return batch.answers[np.arange(batch.n_examples), best_agent].astype(np.int64)

    def metadata(self) -> Dict[str, object]:
        return {
            "clusters": self.clusters,
            "pooling": self.pooling,
            "confidence_weight": self.confidence_weight,
            "param_count": self.param_count,
        }


def _kmeans(x: np.ndarray, k: int, seed: int, iterations: int = 30) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 55_003)
    n = x.shape[0]
    k = min(k, n)
    centers = x[rng.choice(n, size=k, replace=False)].astype(np.float32, copy=True)
    assignments = np.zeros(n, dtype=np.int64)
    for _ in range(iterations):
        assignments = _assign(x, centers)
        new_centers = centers.copy()
        for cluster_id in range(k):
            mask = assignments == cluster_id
            if np.any(mask):
                new_centers[cluster_id] = x[mask].mean(axis=0)
            else:
                new_centers[cluster_id] = x[int(rng.integers(0, n))]
        shift = float(np.linalg.norm(new_centers - centers))
        centers = new_centers
        if shift < 1e-5:
            break
    return centers.astype(np.float32), assignments


def _assign(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(distances, axis=1).astype(np.int64)
