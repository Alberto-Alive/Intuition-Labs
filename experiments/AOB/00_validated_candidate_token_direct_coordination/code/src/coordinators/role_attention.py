from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.agents.types import AttemptBatch
from src.coordinators.mlp import MLPTrainingConfig


@dataclass(frozen=True)
class RoleAttentionConfig:
    slot_indices: tuple[int, ...]
    slot_labels: tuple[str, ...]
    family: str
    variant_name: str
    num_attention_layers: int = 1
    num_heads: int = 1
    pooling: str = "cls"
    feature_mode: str = "late_final"
    use_aux_private_bit_loss: bool = False
    aux_loss_weight: float = 0.2
    agent_dropout: float = 0.25
    min_agents: int = 1


class RoleAwareAttentionCoordinator:
    """Role-aware attention coordinator over frozen per-agent hidden states."""

    def __init__(
        self,
        name: str,
        num_classes: int,
        training: MLPTrainingConfig,
        seed: int,
        device: str,
        config: RoleAttentionConfig,
    ) -> None:
        self.name = name
        self.num_classes = num_classes
        self.training = training
        self.seed = seed
        self.device = torch.device(device)
        self.config = config
        self.model: Optional[_RoleAttentionModel] = None
        self.param_count = 0
        self.history: List[Dict[str, float]] = []
        self.aux_targets_seen = False

    def fit(self, train_batch: AttemptBatch, dev_batch: AttemptBatch | None = None) -> None:
        torch.manual_seed(self.seed + 811_337)
        x_train = self._features(train_batch)
        roles_train = self._role_ids(train_batch)
        bits_train = self._private_bit_targets(train_batch)
        y_train = train_batch.labels.astype(np.int64)
        x_dev = self._features(dev_batch) if dev_batch is not None else None
        roles_dev = self._role_ids(dev_batch) if dev_batch is not None else None
        y_dev = dev_batch.labels.astype(np.int64) if dev_batch is not None else None
        input_dim = int(x_train.shape[-1])
        model_dim = _compatible_model_dim(self.training.hidden_dims, self.config.num_heads)
        self.model = _RoleAttentionModel(
            input_dim=input_dim,
            model_dim=model_dim,
            num_classes=self.num_classes,
            num_roles=x_train.shape[1],
            num_slices=x_train.shape[2],
            config=self.config,
        ).to(self.device)
        self.param_count = int(sum(p.numel() for p in self.model.parameters()))
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training.lr,
            weight_decay=self.training.weight_decay,
        )
        tx = torch.as_tensor(x_train, dtype=torch.float32, device=self.device)
        tr = torch.as_tensor(roles_train, dtype=torch.long, device=self.device)
        ty = torch.as_tensor(y_train, dtype=torch.long, device=self.device)
        tb = torch.as_tensor(bits_train, dtype=torch.long, device=self.device)
        self.aux_targets_seen = bool((bits_train >= 0).all())
        dx = torch.as_tensor(x_dev, dtype=torch.float32, device=self.device) if x_dev is not None else None
        dr = torch.as_tensor(roles_dev, dtype=torch.long, device=self.device) if roles_dev is not None else None
        dy = torch.as_tensor(y_dev, dtype=torch.long, device=self.device) if y_dev is not None else None
        dev_mask = (
            torch.ones((dx.shape[0], dx.shape[1]), dtype=torch.bool, device=self.device)
            if dx is not None
            else None
        )
        best_state = None
        best_dev = -1.0
        stale_epochs = 0
        rng = np.random.default_rng(self.seed + 829_421)
        for epoch in range(self.training.epochs):
            self.model.train()
            permutation = rng.permutation(len(y_train))
            for start in range(0, len(y_train), self.training.batch_size):
                batch_idx = permutation[start : start + self.training.batch_size]
                idx = torch.as_tensor(batch_idx, dtype=torch.long, device=self.device)
                bx = tx.index_select(0, idx)
                br = tr.index_select(0, idx)
                by = ty.index_select(0, idx)
                bb = tb.index_select(0, idx)
                aug_x, aug_roles, aug_mask, aug_bits = self._augment_batch(bx, br, bb, rng)
                label_logits, bit_logits = self.model(aug_x, aug_roles, aug_mask)
                loss = F.cross_entropy(label_logits, by)
                if self.config.use_aux_private_bit_loss and bit_logits is not None:
                    valid = aug_mask & (aug_bits >= 0)
                    if bool(valid.any().item()):
                        aux_loss = F.cross_entropy(bit_logits[valid], aug_bits[valid])
                        loss = loss + self.config.aux_loss_weight * aux_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            train_mask = torch.ones((tx.shape[0], tx.shape[1]), dtype=torch.bool, device=self.device)
            train_acc = self._accuracy_tensor(tx, tr, ty, train_mask)
            if dx is not None and dr is not None and dy is not None and dev_mask is not None:
                dev_acc = self._accuracy_tensor(dx, dr, dy, dev_mask)
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
        roles = torch.as_tensor(self._role_ids(batch), dtype=torch.long, device=self.device)
        mask = torch.as_tensor(agent_mask.astype(bool), dtype=torch.bool, device=self.device)
        return self._predict_tensors(x, roles, mask)

    def predict_with_permutation(self, batch: AttemptBatch, seed: int) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"{self.name} has not been fit")
        rng = np.random.default_rng(seed)
        x = self._features(batch)
        roles = self._role_ids(batch)
        mask = np.ones((batch.n_examples, batch.n_agents), dtype=bool)
        for row_id in range(batch.n_examples):
            perm = rng.permutation(batch.n_agents)
            x[row_id] = x[row_id, perm]
            roles[row_id] = roles[row_id, perm]
            mask[row_id] = mask[row_id, perm]
        return self._predict_numpy(x, roles, mask)

    def predict_with_slot_label_shuffle(self, batch: AttemptBatch, seed: int) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"{self.name} has not been fit")
        rng = np.random.default_rng(seed)
        x = self._features(batch)
        roles = self._role_ids(batch)
        mask = np.ones((batch.n_examples, batch.n_agents), dtype=bool)
        for row_id in range(batch.n_examples):
            perm = rng.permutation(batch.n_agents)
            if batch.n_agents > 1 and np.all(perm == np.arange(batch.n_agents)):
                perm = np.roll(perm, 1)
            roles[row_id] = roles[row_id, perm]
        return self._predict_numpy(x, roles, mask)

    def metadata(self) -> Dict[str, object]:
        return {
            "training_seed": self.seed,
            "param_count": self.param_count,
            "epochs_run": len(self.history),
            "best_dev_during_fit": max((row["dev_acc"] for row in self.history), default=None),
            "hidden_dims": list(self.training.hidden_dims),
            "family": self.config.family,
            "variant_name": self.config.variant_name,
            "num_attention_layers": self.config.num_attention_layers,
            "num_heads": self.config.num_heads,
            "pooling": self.config.pooling,
            "feature_mode": self.config.feature_mode,
            "slot_indices": list(self.config.slot_indices),
            "slot_labels": list(self.config.slot_labels),
            "agent_dropout": self.config.agent_dropout,
            "use_aux_private_bit_loss": self.config.use_aux_private_bit_loss,
            "aux_loss_weight": self.config.aux_loss_weight,
            "aux_targets_seen": self.aux_targets_seen,
            "uses_visible_outputs": False,
            "base_transformer_trainable": False,
        }

    def _features(self, batch: AttemptBatch | None) -> np.ndarray:
        if batch is None:
            raise ValueError("batch is required")
        return batch.hidden_states[:, :, list(self.config.slot_indices), :].astype(np.float32, copy=True)

    def _role_ids(self, batch: AttemptBatch | None) -> np.ndarray:
        if batch is None:
            raise ValueError("batch is required")
        return np.broadcast_to(np.arange(batch.n_agents, dtype=np.int64), (batch.n_examples, batch.n_agents)).copy()

    def _private_bit_targets(self, batch: AttemptBatch) -> np.ndarray:
        bits = np.full((batch.n_examples, batch.n_agents), -1, dtype=np.int64)
        for row_id, row in enumerate(batch.private_agent_views):
            for agent_id, text in enumerate(row[: batch.n_agents]):
                bits[row_id, agent_id] = _parse_private_bit(text)
        return bits

    def _augment_batch(
        self,
        x: torch.Tensor,
        roles: torch.Tensor,
        bits: torch.Tensor,
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, n_agents, _n_slices, _hidden = x.shape
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
        gathered_x = x.gather(
            1,
            order_tensor.view(batch_size, n_agents, 1, 1).expand(-1, -1, x.shape[2], x.shape[3]),
        )
        gathered_roles = roles.gather(1, order_tensor)
        gathered_bits = bits.gather(1, order_tensor)
        return gathered_x, gathered_roles, mask_tensor, gathered_bits

    def _predict_numpy(self, x: np.ndarray, roles: np.ndarray, mask: np.ndarray) -> np.ndarray:
        tx = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        tr = torch.as_tensor(roles, dtype=torch.long, device=self.device)
        tm = torch.as_tensor(mask.astype(bool), dtype=torch.bool, device=self.device)
        return self._predict_tensors(tx, tr, tm)

    def _predict_tensors(self, x: torch.Tensor, roles: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"{self.name} has not been fit")
        preds: List[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, x.shape[0], 2048):
                logits, _bit_logits = self.model(
                    x[start : start + 2048],
                    roles[start : start + 2048],
                    mask[start : start + 2048],
                )
                preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
        return np.concatenate(preds).astype(np.int64)

    def _accuracy_tensor(self, x: torch.Tensor, roles: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
        if self.model is None:
            raise RuntimeError("model not initialized")
        self.model.eval()
        with torch.no_grad():
            logits, _bit_logits = self.model(x, roles, mask)
            preds = torch.argmax(logits, dim=1)
            return float((preds == y).float().mean().item())


class RoleAwareSelfAttentionCoordinator(RoleAwareAttentionCoordinator):
    def __init__(
        self,
        num_classes: int,
        training: MLPTrainingConfig,
        seed: int,
        device: str,
        config: RoleAttentionConfig,
    ) -> None:
        super().__init__(
            name=f"role_self_attention:{config.variant_name}",
            num_classes=num_classes,
            training=training,
            seed=seed,
            device=device,
            config=config,
        )


class CoordinatorTokenCrossAttentionCoordinator(RoleAwareAttentionCoordinator):
    def __init__(
        self,
        num_classes: int,
        training: MLPTrainingConfig,
        seed: int,
        device: str,
        config: RoleAttentionConfig,
    ) -> None:
        super().__init__(
            name=f"coordinator_cross_attention:{config.variant_name}",
            num_classes=num_classes,
            training=training,
            seed=seed,
            device=device,
            config=config,
        )


class _RoleAttentionModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_classes: int,
        num_roles: int,
        num_slices: int,
        config: RoleAttentionConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.role_embedding = nn.Embedding(num_roles, model_dim)
        self.slice_embedding = nn.Embedding(num_slices, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        if config.family == "self_attention":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=config.num_heads,
                dim_feedforward=max(model_dim * 2, 64),
                dropout=0.0,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.self_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_attention_layers)
            self.cross_layers = None
        elif config.family == "cross_attention":
            self.self_encoder = None
            self.cross_layers = nn.ModuleList(
                [_CrossAttentionBlock(model_dim, config.num_heads) for _ in range(config.num_attention_layers)]
            )
        else:
            raise ValueError(f"unknown role attention family: {config.family}")
        self.label_head = nn.Linear(model_dim, num_classes)
        self.bit_head = nn.Linear(model_dim, 2) if config.use_aux_private_bit_loss else None
        nn.init.normal_(self.cls_token, mean=0.0, std=1.0 / max(1, model_dim) ** 0.5)

    def forward(
        self,
        x: torch.Tensor,
        role_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        agent_tokens = self._agent_tokens(x, role_ids)
        if self.config.family == "self_attention":
            pooled, agent_context = self._self_attention(agent_tokens, mask)
        else:
            pooled, agent_context = self._cross_attention(agent_tokens, mask)
        bit_logits = self.bit_head(agent_context) if self.bit_head is not None else None
        return self.label_head(pooled), bit_logits

    def _agent_tokens(self, x: torch.Tensor, role_ids: torch.Tensor) -> torch.Tensor:
        batch_size, n_agents, n_slices, _hidden = x.shape
        projected = self.input_projection(x)
        slice_ids = torch.arange(n_slices, dtype=torch.long, device=x.device)
        projected = projected + self.slice_embedding(slice_ids).view(1, 1, n_slices, -1)
        agent_tokens = projected.mean(dim=2)
        agent_tokens = agent_tokens + self.role_embedding(role_ids.clamp(min=0, max=self.role_embedding.num_embeddings - 1))
        return agent_tokens.view(batch_size, n_agents, -1)

    def _self_attention(self, agent_tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.self_encoder is None:
            raise RuntimeError("self encoder is not initialized")
        if self.config.pooling == "cls":
            cls = self.cls_token.expand(agent_tokens.shape[0], -1, -1)
            tokens = torch.cat([cls, agent_tokens], dim=1)
            key_padding_mask = torch.cat(
                [
                    torch.zeros((mask.shape[0], 1), dtype=torch.bool, device=mask.device),
                    ~mask.bool(),
                ],
                dim=1,
            )
            encoded = self.self_encoder(tokens, src_key_padding_mask=key_padding_mask)
            return encoded[:, 0, :], encoded[:, 1:, :]
        encoded = self.self_encoder(agent_tokens, src_key_padding_mask=~mask.bool())
        weights = mask.to(dtype=encoded.dtype).unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return pooled, encoded

    def _cross_attention(self, agent_tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cross_layers is None:
            raise RuntimeError("cross-attention layers are not initialized")
        coordinator = self.cls_token.expand(agent_tokens.shape[0], -1, -1)
        for layer in self.cross_layers:
            coordinator = layer(coordinator, agent_tokens, mask)
        return coordinator[:, 0, :], agent_tokens


class _CrossAttentionBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(model_dim, num_heads, dropout=0.0, batch_first=True)
        self.norm1 = nn.LayerNorm(model_dim)
        self.ff = nn.Sequential(
            nn.Linear(model_dim, max(model_dim * 2, 64)),
            nn.GELU(),
            nn.Linear(max(model_dim * 2, 64), model_dim),
        )
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, coordinator: torch.Tensor, agent_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attended, _weights = self.attn(
            query=self.norm1(coordinator),
            key=agent_tokens,
            value=agent_tokens,
            key_padding_mask=~mask.bool(),
            need_weights=False,
        )
        coordinator = coordinator + attended
        coordinator = coordinator + self.ff(self.norm2(coordinator))
        return coordinator


def _compatible_model_dim(hidden_dims: Iterable[int], num_heads: int) -> int:
    dims = [int(value) for value in hidden_dims]
    model_dim = dims[0] if dims else 96
    if model_dim < num_heads:
        model_dim = num_heads
    remainder = model_dim % num_heads
    if remainder:
        model_dim += num_heads - remainder
    return int(model_dim)


def _parse_private_bit(text: str) -> int:
    upper = text.upper()
    if "BIT_ONE" in upper or "EV_POS" in upper:
        return 1
    if "BIT_ZERO" in upper or "EV_NEG" in upper:
        return 0
    match = re.search(r"PRIVATE_EVIDENCE_BIT[=:]\s*([01])", upper)
    if match:
        return int(match.group(1))
    return -1

