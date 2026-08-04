from __future__ import annotations

import numpy as np


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)), axis=-1)
    kl_qm = np.sum(q * (np.log(q) - np.log(m)), axis=-1)
    return float(np.mean(0.5 * (kl_pm + kl_qm)))


def cosine_flat(a: np.ndarray, b: np.ndarray) -> float:
    av = a.reshape(-1)
    bv = b.reshape(-1)
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    if denom <= 0:
        return 0.0
    return float(np.dot(av, bv) / denom)


def prediction_agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a == b))


def jaccard_binary(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(bool)
    bb = b.astype(bool)
    union = np.logical_or(aa, bb).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(aa, bb).sum() / union)
