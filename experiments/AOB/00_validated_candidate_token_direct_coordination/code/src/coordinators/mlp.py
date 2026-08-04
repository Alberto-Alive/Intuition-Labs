from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.agents.types import AttemptBatch


FeatureBuilder = Callable[[AttemptBatch], np.ndarray]


@dataclass(frozen=True)
class MLPTrainingConfig:
    epochs: int = 35
    batch_size: int = 256
    lr: float = 0.003
    weight_decay: float = 0.0001
    patience: int = 7
    hidden_dims: tuple[int, ...] = (96, 48)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Iterable[int], num_classes: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        current = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current, hidden_dim))
            layers.append(nn.ReLU())
            current = hidden_dim
        layers.append(nn.Linear(current, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_mlp_params(input_dim: int, hidden_dims: Iterable[int], num_classes: int) -> int:
    current = input_dim
    total = 0
    for hidden_dim in hidden_dims:
        total += (current + 1) * int(hidden_dim)
        current = int(hidden_dim)
    total += (current + 1) * num_classes
    return int(total)


def match_hidden_dims_to_budget(
    input_dim: int,
    num_classes: int,
    target_params: int,
    max_hidden_dim: int = 256,
) -> tuple[int, int]:
    """Find a two-layer MLP size closest to a target parameter budget."""
    best_dims = (16, 8)
    best_delta = float("inf")
    for h1 in range(8, max_hidden_dim + 1):
        for h2 in range(4, max_hidden_dim + 1):
            params = count_mlp_params(input_dim, (h1, h2), num_classes)
            delta = abs(params - target_params)
            if delta < best_delta:
                best_delta = delta
                best_dims = (h1, h2)
            if delta == 0:
                return best_dims
    return best_dims


class FeatureMLPCoordinator:
    def __init__(
        self,
        name: str,
        feature_builder: FeatureBuilder,
        num_classes: int,
        training: MLPTrainingConfig,
        seed: int,
        device: str,
    ) -> None:
        self.name = name
        self.feature_builder = feature_builder
        self.num_classes = num_classes
        self.training = training
        self.seed = seed
        self.device = torch.device(device)
        self.model: Optional[MLPClassifier] = None
        self.param_count = 0
        self.history: List[Dict[str, float]] = []

    def fit(self, train_batch: AttemptBatch, dev_batch: AttemptBatch | None = None) -> None:
        torch.manual_seed(self.seed + 12_019)
        x_train = self.feature_builder(train_batch)
        y_train = train_batch.labels.astype(np.int64)
        x_dev = self.feature_builder(dev_batch) if dev_batch is not None else None
        y_dev = dev_batch.labels.astype(np.int64) if dev_batch is not None else None

        self.model = MLPClassifier(
            input_dim=x_train.shape[1],
            hidden_dims=self.training.hidden_dims,
            num_classes=self.num_classes,
        ).to(self.device)
        self.param_count = int(sum(p.numel() for p in self.model.parameters()))
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training.lr,
            weight_decay=self.training.weight_decay,
        )

        tx = torch.as_tensor(x_train, dtype=torch.float32, device=self.device)
        ty = torch.as_tensor(y_train, dtype=torch.long, device=self.device)
        dx = torch.as_tensor(x_dev, dtype=torch.float32, device=self.device) if x_dev is not None else None
        dy = torch.as_tensor(y_dev, dtype=torch.long, device=self.device) if y_dev is not None else None

        best_state = None
        best_dev = -1.0
        epochs_without_improvement = 0
        rng = np.random.default_rng(self.seed + 91_531)

        for epoch in range(self.training.epochs):
            self.model.train()
            permutation = rng.permutation(len(y_train))
            for start in range(0, len(y_train), self.training.batch_size):
                batch_idx = permutation[start : start + self.training.batch_size]
                idx = torch.as_tensor(batch_idx, dtype=torch.long, device=self.device)
                logits = self.model(tx.index_select(0, idx))
                loss = F.cross_entropy(logits, ty.index_select(0, idx))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            train_acc = self._accuracy_tensor(tx, ty)
            if dx is not None and dy is not None:
                dev_acc = self._accuracy_tensor(dx, dy)
            else:
                dev_acc = train_acc
            self.history.append({"epoch": float(epoch), "train_acc": train_acc, "dev_acc": dev_acc})
            if dev_acc > best_dev:
                best_dev = dev_acc
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= self.training.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

    def predict(self, batch: AttemptBatch) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"{self.name} has not been fit")
        self.model.eval()
        x = torch.as_tensor(self.feature_builder(batch), dtype=torch.float32, device=self.device)
        preds: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, x.shape[0], 2048):
                logits = self.model(x[start : start + 2048])
                preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
        return np.concatenate(preds).astype(np.int64)

    def predict_proba(self, batch: AttemptBatch) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"{self.name} has not been fit")
        self.model.eval()
        x = torch.as_tensor(self.feature_builder(batch), dtype=torch.float32, device=self.device)
        probs: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, x.shape[0], 2048):
                logits = self.model(x[start : start + 2048])
                probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
        return np.concatenate(probs).astype(np.float32)

    def _accuracy_tensor(self, x: torch.Tensor, y: torch.Tensor) -> float:
        if self.model is None:
            raise RuntimeError("model not initialized")
        self.model.eval()
        with torch.no_grad():
            preds = torch.argmax(self.model(x), dim=1)
            return float((preds == y).float().mean().item())

    def metadata(self) -> Dict[str, object]:
        return {
            "training_seed": self.seed,
            "param_count": self.param_count,
            "epochs_run": len(self.history),
            "best_dev_during_fit": max((row["dev_acc"] for row in self.history), default=None),
            "hidden_dims": list(self.training.hidden_dims),
        }
