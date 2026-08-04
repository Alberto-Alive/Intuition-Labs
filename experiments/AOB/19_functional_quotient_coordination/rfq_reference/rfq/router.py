from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyRouter(nn.Module):
    def __init__(self, input_dim: int, n_strategies: int, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_strategies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


def build_router_targets(
    strategy_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    strategy_logits: [N, K, C]
    The target is the strategy assigning highest probability to the true label.
    """
    log_probs = F.log_softmax(strategy_logits, dim=-1)
    true_lp = log_probs.gather(
        2,
        labels[:, None, None].expand(-1, strategy_logits.shape[1], 1),
    ).squeeze(-1)
    return true_lp.argmax(dim=1)


def train_router(
    *,
    features: torch.Tensor,
    strategy_logits: torch.Tensor,
    labels: torch.Tensor,
    hidden_dim: int = 32,
    epochs: int = 250,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
) -> TinyRouter:
    router = TinyRouter(
        input_dim=features.shape[1],
        n_strategies=strategy_logits.shape[1],
        hidden_dim=hidden_dim,
    )
    targets = build_router_targets(strategy_logits, labels)

    opt = torch.optim.Adam(
        router.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    for _ in range(epochs):
        logits = router(features)
        loss = F.cross_entropy(logits, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return router


@torch.no_grad()
def evaluate_router(
    *,
    router: nn.Module,
    features: torch.Tensor,
    strategy_logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    route = router(features).argmax(dim=1)
    routed_logits = strategy_logits[
        torch.arange(strategy_logits.shape[0]),
        route,
    ]
    routed_accuracy = float(
        (routed_logits.argmax(dim=1) == labels).float().mean().item()
    )

    ensemble_accuracy = float(
        (strategy_logits.mean(dim=1).argmax(dim=1) == labels)
        .float()
        .mean()
        .item()
    )

    oracle = (
        strategy_logits.argmax(dim=2) == labels[:, None]
    ).any(dim=1)

    return {
        "routed_accuracy": routed_accuracy,
        "ensemble_accuracy": ensemble_accuracy,
        "oracle_coverage": float(oracle.float().mean().item()),
        "route_distribution": [
            float((route == i).float().mean().item())
            for i in range(strategy_logits.shape[1])
        ],
    }
