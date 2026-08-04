from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from src.agents.types import AttemptBatch
from src.coordinators.mlp import FeatureMLPCoordinator, MLPTrainingConfig
from src.telemetry.features import activation_features, output_oracle_features


class TelemetryOnlyLabelProbe(FeatureMLPCoordinator):
    """Predict final labels from activation telemetry only."""

    def __init__(
        self,
        num_classes: int,
        components: int,
        training: MLPTrainingConfig,
        seed: int,
        device: str,
        pooling: str = "mean",
        name: str = "telemetry_only_label_probe",
    ) -> None:
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

    def fit(self, train_batch: AttemptBatch, dev_batch: AttemptBatch | None = None) -> None:
        self._fit_pca(activation_features(train_batch, pooling=self.pooling))
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
        raw = activation_features(batch, pooling=self.pooling)
        if self._mean is None or self._basis is None:
            return raw[:, : self.components].astype(np.float32)
        return ((raw - self._mean) @ self._basis.T).astype(np.float32)

    def metadata(self) -> Dict[str, object]:
        meta = super().metadata()
        meta["pca_components"] = self.components
        meta["pooling"] = self.pooling
        meta["visible_outputs"] = False
        if self._explained_variance_ratio is not None:
            meta["pca_explained_variance_ratio_sum"] = float(self._explained_variance_ratio.sum())
            meta["pca_first_component_variance_ratio"] = float(self._explained_variance_ratio[0]) if len(self._explained_variance_ratio) else 0.0
        return meta


class HiddenStateOnlyLabelProbe(TelemetryOnlyLabelProbe):
    """Predict final labels from captured model hidden states only."""

    def __init__(
        self,
        num_classes: int,
        components: int,
        training: MLPTrainingConfig,
        seed: int,
        device: str,
        pooling: str = "mean",
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            components=components,
            training=training,
            seed=seed,
            device=device,
            pooling=pooling,
            name="hidden_state_only_probe",
        )


class OutputOnlyOracleProbe(FeatureMLPCoordinator):
    """Budget-matched final-label probe over every visible field and trace."""

    def __init__(
        self,
        num_classes: int,
        num_tasks: int,
        target_param_count: int,
        training: MLPTrainingConfig,
        seed: int,
        device: str,
    ) -> None:
        self.num_tasks = num_tasks
        self.target_param_count = target_param_count
        super().__init__(
            name="output_only_oracle_probe",
            feature_builder=lambda batch: output_oracle_features(batch, num_classes, num_tasks),
            num_classes=num_classes,
            training=training,
            seed=seed,
            device=device,
        )

    def metadata(self) -> Dict[str, object]:
        meta = super().metadata()
        meta["target_param_count"] = self.target_param_count
        meta["param_delta_from_target"] = int(meta["param_count"]) - self.target_param_count
        meta["hidden_activations"] = False
        meta["uses_visible_traces"] = True
        return meta
