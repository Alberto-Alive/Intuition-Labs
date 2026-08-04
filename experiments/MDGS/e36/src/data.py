from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class MoonConfig:
    n_train: int = 4000
    n_val: int = 1000
    n_test: int = 1000
    noise: float = 0.10
    seed: int = 123
    ood_ratio_train: float = 0.25


def make_two_moons(n: int, noise: float = 0.10, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a two-moons dataset without sklearn.

    Returns:
        x: [n, 2] float32
        y: [n] int64, class 0 or 1
    """
    rng = np.random.default_rng(seed)
    n0 = n // 2
    n1 = n - n0

    t0 = rng.uniform(0.0, np.pi, size=n0)
    t1 = rng.uniform(0.0, np.pi, size=n1)

    x0 = np.stack([np.cos(t0), np.sin(t0)], axis=1)
    x1 = np.stack([1.0 - np.cos(t1), -np.sin(t1) - 0.5], axis=1)

    x = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([np.zeros(n0), np.ones(n1)], axis=0).astype(np.int64)

    x += rng.normal(scale=noise, size=x.shape)

    # Normalize to a friendly range.
    x = (x - np.array([0.5, 0.0])) / np.array([1.5, 1.5])

    perm = rng.permutation(n)
    return x[perm].astype(np.float32), y[perm]


def make_ood(n: int, seed: int = 0) -> np.ndarray:
    """Generate synthetic OOD points around/outside the moons.

    These are unlabeled for prediction and are used only to train support/uncertainty.
    """
    rng = np.random.default_rng(seed)
    # Mixture: outer ring + far box corners.
    n_ring = int(n * 0.7)
    n_box = n - n_ring

    angles = rng.uniform(0, 2 * np.pi, size=n_ring)
    radii = rng.uniform(1.25, 1.75, size=n_ring)
    ring = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)

    box = rng.uniform(-2.0, 2.0, size=(n_box, 2))
    # Keep mostly far points.
    mask = np.linalg.norm(box, axis=1) < 1.15
    while mask.any():
        box[mask] = rng.uniform(-2.0, 2.0, size=(mask.sum(), 2))
        mask = np.linalg.norm(box, axis=1) < 1.15

    x = np.concatenate([ring, box], axis=0)
    rng.shuffle(x)
    return x.astype(np.float32)


class MoonDataset(Dataset):
    """In-distribution two-moons optionally mixed with OOD samples.

    y == -1 means OOD and should be excluded from prediction cross-entropy.
    is_ood == 1 means synthetic unsupported point.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, ood_x: np.ndarray | None = None):
        if ood_x is not None and len(ood_x) > 0:
            ood_y = -np.ones(len(ood_x), dtype=np.int64)
            x = np.concatenate([x, ood_x], axis=0)
            y = np.concatenate([y, ood_y], axis=0)
            is_ood = np.concatenate([
                np.zeros(len(x) - len(ood_x), dtype=np.float32),
                np.ones(len(ood_x), dtype=np.float32),
            ])
        else:
            is_ood = np.zeros(len(x), dtype=np.float32)

        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.is_ood = torch.from_numpy(is_ood.astype(np.float32))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.is_ood[idx]


def build_datasets(cfg: MoonConfig):
    x_train, y_train = make_two_moons(cfg.n_train, cfg.noise, cfg.seed)
    x_val, y_val = make_two_moons(cfg.n_val, cfg.noise, cfg.seed + 1)
    x_test, y_test = make_two_moons(cfg.n_test, cfg.noise, cfg.seed + 2)

    n_ood_train = int(cfg.n_train * cfg.ood_ratio_train)
    ood_train = make_ood(n_ood_train, cfg.seed + 10)
    ood_val = make_ood(cfg.n_val, cfg.seed + 11)
    ood_test = make_ood(cfg.n_test, cfg.seed + 12)

    train = MoonDataset(x_train, y_train, ood_train)
    val = MoonDataset(x_val, y_val, None)
    test = MoonDataset(x_test, y_test, None)
    ood_val_ds = MoonDataset(ood_val, -np.ones(len(ood_val), dtype=np.int64), None)
    ood_test_ds = MoonDataset(ood_test, -np.ones(len(ood_test), dtype=np.int64), None)
    return train, val, test, ood_val_ds, ood_test_ds
