from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.agents.types import AttemptBatch
from src.coordinators.mlp import MLPTrainingConfig


@dataclass(frozen=True)
class HiddenSetConfig:
    slot_index: int
    agent_dropout: float = 0.25
    min_agents: int = 1


class HiddenSetCoordinator:
    """Permutation-invariant coordinator over frozen per-agent hidden states."""

    def __init__(
        self,
        name: str,
        num_classes: int,
        training: MLPTrainingConfig,
        seed: int,
        device: str,
        config: HiddenSetConfig,
        pooling: str,
    ) -> None:
        self.name = name
        self.num_classes = num_classes
        self.training = training
        self.seed = seed
        self.device = torch.device(device)
        self.config = config
        self.pooling = pooling
        self.model: Optional[_HiddenSetClassifier] = None
        self.param_count = 0
        self.history: List[Dict[str, float]] = []

    def fit(self, train_batch: AttemptBatch, dev_batch: AttemptBatch | None = None) -> None:
        torch.manual_seed(self.seed + 71_003)
        x_train = self._features(train_batch)
        y_train = train_batch.labels.astype(np.int64)
        x_dev = self._features(dev_batch) if dev_batch is not None else None
        y_dev = dev_batch.labels.astype(np.int64) if dev_batch is not None else None
        input_dim = int(x_train.shape[-1])
        encoder_dims = tuple(int(value) for value in self.training.hidden_dims)
        self.model = _HiddenSetClassifier(
            input_dim=input_dim,
            encoder_dims=encoder_dims,
            num_classes=self.num_classes,
            pooling=self.pooling,
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
        dev_mask = (
            torch.ones((dx.shape[0], dx.shape[1]), dtype=torch.bool, device=self.device)
            if dx is not None
            else None
        )
        best_state = None
        best_dev = -1.0
        stale_epochs = 0
        rng = np.random.default_rng(self.seed + 73_919)
        for epoch in range(self.training.epochs):
            self.model.train()
            permutation = rng.permutation(len(y_train))
            for start in range(0, len(y_train), self.training.batch_size):
                batch_idx = permutation[start : start + self.training.batch_size]
                bx = tx.index_select(0, torch.as_tensor(batch_idx, dtype=torch.long, device=self.device))
                by = ty.index_select(0, torch.as_tensor(batch_idx, dtype=torch.long, device=self.device))
                aug_x, aug_mask = self._augment_batch(bx, rng)
                logits = self.model(aug_x, aug_mask)
                loss = F.cross_entropy(logits, by)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            train_acc = self._accuracy_tensor(tx, ty, torch.ones((tx.shape[0], tx.shape[1]), dtype=torch.bool, device=self.device))
            if dx is not None and dy is not None and dev_mask is not None:
                dev_acc = self._accuracy_tensor(dx, dy, dev_mask)
            else:
                dev_acc = train_acc
            self.history.append({"epoch": float(epoch), "train_acc": train_acc, "dev_acc": dev_acc})
            if dev_acc > best_dev:
                best_dev = dev_acc
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= self.training.patience:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        self.model.eval()

    def predict(self, batch: AttemptBatch) -> np.ndarray:
        mask = np.ones((batch.n_examples, batch.n_agents), dtype=bool)
        return self.predict_with_mask(batch, mask)

    def predict_with_mask(self, batch: AttemptBatch, agent_mask: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"{self.name} has not been fit")
        x = torch.as_tensor(self._features(batch), dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(agent_mask.astype(bool), dtype=torch.bool, device=self.device)
        preds: List[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, x.shape[0], 2048):
                logits = self.model(x[start : start + 2048], mask[start : start + 2048])
                preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
        return np.concatenate(preds).astype(np.int64)

    def predict_with_permutation(self, batch: AttemptBatch, seed: int) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"{self.name} has not been fit")
        rng = np.random.default_rng(seed)
        x = self._features(batch)
        mask = np.ones((batch.n_examples, batch.n_agents), dtype=bool)
        for row_id in range(batch.n_examples):
            perm = rng.permutation(batch.n_agents)
            x[row_id] = x[row_id, perm]
            mask[row_id] = mask[row_id, perm]
        tensor_x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        tensor_mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        preds: List[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, tensor_x.shape[0], 2048):
                logits = self.model(tensor_x[start : start + 2048], tensor_mask[start : start + 2048])
                preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
        return np.concatenate(preds).astype(np.int64)

    def metadata(self) -> Dict[str, object]:
        return {
            "training_seed": self.seed,
            "param_count": self.param_count,
            "epochs_run": len(self.history),
            "best_dev_during_fit": max((row["dev_acc"] for row in self.history), default=None),
            "hidden_dims": list(self.training.hidden_dims),
            "slot_index": self.config.slot_index,
            "agent_dropout": self.config.agent_dropout,
            "min_agents": self.config.min_agents,
            "pooling": self.pooling,
            "uses_visible_outputs": False,
        }

    def _features(self, batch: AttemptBatch | None) -> np.ndarray:
        if batch is None:
            raise ValueError("batch is required")
        return batch.hidden_states[:, :, self.config.slot_index, :].astype(np.float32, copy=True)

    def _augment_batch(self, x: torch.Tensor, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_agents, _hidden = x.shape
        order = np.zeros((batch_size, n_agents), dtype=np.int64)
        mask = np.zeros((batch_size, n_agents), dtype=bool)
        for row_id in range(batch_size):
            perm = rng.permutation(n_agents)
            keep = rng.random(n_agents) >= self.config.agent_dropout
            if int(keep.sum()) < self.config.min_agents:
                chosen = rng.choice(n_agents, size=min(self.config.min_agents, n_agents), replace=False)
                keep[chosen] = True
            order[row_id] = perm
            mask[row_id] = keep[perm]
        order_tensor = torch.as_tensor(order, dtype=torch.long, device=x.device)
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=x.device)
        gathered = x.gather(1, order_tensor.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        return gathered, mask_tensor

    def _accuracy_tensor(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
        if self.model is None:
            raise RuntimeError("model not initialized")
        self.model.eval()
        with torch.no_grad():
            preds = torch.argmax(self.model(x, mask), dim=1)
            return float((preds == y).float().mean().item())


class DeepSetsHiddenCoordinator(HiddenSetCoordinator):
    def __init__(
        self,
        num_classes: int,
        training: MLPTrainingConfig,
        seed: int,
        device: str,
        config: HiddenSetConfig,
    ) -> None:
        super().__init__(
            name="deepsets_hidden_coordinator",
            num_classes=num_classes,
            training=training,
            seed=seed,
            device=device,
            config=config,
            pooling="mean",
        )


class AttentionHiddenCoordinator(HiddenSetCoordinator):
    def __init__(
        self,
        num_classes: int,
        training: MLPTrainingConfig,
        seed: int,
        device: str,
        config: HiddenSetConfig,
    ) -> None:
        super().__init__(
            name="attention_hidden_coordinator",
            num_classes=num_classes,
            training=training,
            seed=seed,
            device=device,
            config=config,
            pooling="attention",
        )


class _HiddenSetClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoder_dims: Iterable[int],
        num_classes: int,
        pooling: str,
    ) -> None:
        super().__init__()
        dims = [int(value) for value in encoder_dims]
        layers: List[nn.Module] = []
        current = input_dim
        for hidden_dim in dims:
            layers.append(nn.Linear(current, hidden_dim))
            layers.append(nn.ReLU())
            current = hidden_dim
        if not layers:
            layers.append(nn.Identity())
        self.encoder = nn.Sequential(*layers)
        self.pooling = pooling
        self.query = nn.Parameter(torch.zeros(current)) if pooling == "attention" else None
        if self.query is not None:
            nn.init.normal_(self.query, mean=0.0, std=1.0 / max(1, current) ** 0.5)
        self.classifier = nn.Linear(current, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        mask = mask.bool()
        if self.pooling == "attention":
            if self.query is None:
                raise RuntimeError("attention query not initialized")
            scores = encoded @ self.query
            scores = scores.masked_fill(~mask, -1.0e9)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            pooled = (encoded * weights).sum(dim=1)
        else:
            weights = mask.to(dtype=encoded.dtype).unsqueeze(-1)
            pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return self.classifier(pooled)
