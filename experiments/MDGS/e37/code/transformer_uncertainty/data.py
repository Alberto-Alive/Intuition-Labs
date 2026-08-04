from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SeqTaskConfig:
    """Config for a small rule-conditioned sequence task.

    Token 0 in each sequence is a rule token:
    - 1 means class evidence in channel A matters.
    - 2 means class evidence in channel B matters.

    The inactive channel can contain strong distractor evidence, so the model
    needs attention to route by the rule token rather than simple counting.
    """

    seq_len: int = 18
    vocab_size: int = 64
    n_train: int = 1200
    n_val: int = 400
    n_test: int = 400
    ood_ratio_train: float = 0.25
    seed: int = 7
    ood_mode: str = "easy"


RULE_A = 1
RULE_B = 2
COMMON_LOW = 26
COMMON_HIGH = 55
OOD_LOW = 56
OOD_HIGH = 63

CHANNELS = {
    0: {
        0: np.arange(10, 14, dtype=np.int64),
        1: np.arange(14, 18, dtype=np.int64),
    },
    1: {
        0: np.arange(18, 22, dtype=np.int64),
        1: np.arange(22, 26, dtype=np.int64),
    },
}


class SequenceReliabilityDataset(Dataset):
    def __init__(self, items: Dict[str, np.ndarray]):
        self.x = torch.from_numpy(items["x"].astype(np.int64))
        self.y = torch.from_numpy(items["y"].astype(np.int64))
        self.is_ood = torch.from_numpy(items["is_ood"].astype(np.float32))
        self.m_target = torch.from_numpy(items["m_target"].astype(np.float32))
        self.d_target = torch.from_numpy(items["d_target"].astype(np.float32))
        self.g_target = torch.from_numpy(items["g_target"].astype(np.float32))
        self.s_target = torch.from_numpy(items["s_target"].astype(np.float32))
        self.unc_target = torch.from_numpy(items["unc_target"].astype(np.float32))
        self.commit_target = torch.from_numpy(items["commit_target"].astype(np.float32))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return {
            "x": self.x[idx],
            "y": self.y[idx],
            "is_ood": self.is_ood[idx],
            "m_target": self.m_target[idx],
            "d_target": self.d_target[idx],
            "g_target": self.g_target[idx],
            "s_target": self.s_target[idx],
            "unc_target": self.unc_target[idx],
            "commit_target": self.commit_target[idx],
        }


def _place_tokens(rng: np.random.Generator, seq: np.ndarray, tokens: np.ndarray) -> None:
    if len(tokens) == 0:
        return
    positions = rng.choice(np.arange(1, len(seq)), size=len(tokens), replace=False)
    seq[positions] = tokens


def _sample_from(rng: np.random.Generator, values: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0,), dtype=np.int64)
    return rng.choice(values, size=count, replace=True).astype(np.int64)


def _targets_from_mdgs(m: float, d: float, g: float, s: float) -> Tuple[float, float]:
    commitment = float(np.clip(s * m * (1.0 - d) * (1.0 - g), 0.0, 1.0))
    uncertainty = float(np.clip(1.0 - commitment, 0.0, 1.0))
    return uncertainty, commitment


def _make_id_sample(rng: np.random.Generator, cfg: SeqTaskConfig):
    seq = rng.integers(COMMON_LOW, COMMON_HIGH + 1, size=cfg.seq_len, dtype=np.int64)
    active_channel = int(rng.integers(0, 2))
    inactive_channel = 1 - active_channel
    rule_token = RULE_A if active_channel == 0 else RULE_B
    seq[0] = rule_token

    y = int(rng.integers(0, 2))
    difficulty = rng.choice(["easy", "medium", "boundary"], p=[0.55, 0.30, 0.15])

    if difficulty == "easy":
        active_y, active_opp = 4, 0
        inactive_count = int(rng.integers(0, 3))
        m, d, g = 0.98, 0.02, 0.05
    elif difficulty == "medium":
        active_y, active_opp = 3, 1
        inactive_count = int(rng.integers(2, 4))
        m, d, g = 0.68, 0.25, 0.25
    else:
        active_y, active_opp = 2, 1
        inactive_count = int(rng.integers(4, 6))
        m, d, g = 0.38, 0.48, 0.65

    active_tokens = np.concatenate(
        [
            _sample_from(rng, CHANNELS[active_channel][y], active_y),
            _sample_from(rng, CHANNELS[active_channel][1 - y], active_opp),
        ]
    )
    rng.shuffle(active_tokens)

    # Inactive-channel evidence is a deliberate distractor. Boundary cases get
    # more opposite evidence to punish shallow bag-of-token solutions.
    inactive_majority = 1 - y if difficulty == "boundary" else int(rng.integers(0, 2))
    inactive_tokens = _sample_from(rng, CHANNELS[inactive_channel][inactive_majority], inactive_count)
    tokens = np.concatenate([active_tokens, inactive_tokens])
    if len(tokens) > cfg.seq_len - 1:
        tokens = tokens[: cfg.seq_len - 1]
    _place_tokens(rng, seq, tokens)

    support = 1.0
    unc, commit = _targets_from_mdgs(m, d, g, support)
    return seq, y, 0.0, m, d, g, support, unc, commit


def _make_easy_ood_sample(rng: np.random.Generator, cfg: SeqTaskConfig):
    seq = rng.integers(OOD_LOW, OOD_HIGH + 1, size=cfg.seq_len, dtype=np.int64)
    seq[0] = RULE_A if int(rng.integers(0, 2)) == 0 else RULE_B

    # A few normal-looking evidence tokens make OOD detection non-trivial.
    normal_noise_count = int(rng.integers(1, 4))
    channel = int(rng.integers(0, 2))
    cls = int(rng.integers(0, 2))
    noise = _sample_from(rng, CHANNELS[channel][cls], normal_noise_count)
    _place_tokens(rng, seq, noise)

    m, d, g, support = 0.0, 1.0, 1.0, 0.0
    unc, commit = _targets_from_mdgs(m, d, g, support)
    return seq, -1, 1.0, m, d, g, support, unc, commit


def _make_medium_ood_sample(rng: np.random.Generator, cfg: SeqTaskConfig):
    """Mostly valid sequence with shifted background/noisy support.

    The rule token and some evidence tokens look normal, but the background has
    unfamiliar tokens and the active channel is intentionally ambiguous.
    """

    seq = rng.integers(COMMON_LOW, COMMON_HIGH + 1, size=cfg.seq_len, dtype=np.int64)
    active_channel = int(rng.integers(0, 2))
    inactive_channel = 1 - active_channel
    seq[0] = RULE_A if active_channel == 0 else RULE_B

    pseudo_y = int(rng.integers(0, 2))
    active_tokens = np.concatenate(
        [
            _sample_from(rng, CHANNELS[active_channel][pseudo_y], 2),
            _sample_from(rng, CHANNELS[active_channel][1 - pseudo_y], 2),
        ]
    )
    inactive_tokens = _sample_from(rng, CHANNELS[inactive_channel][1 - pseudo_y], int(rng.integers(2, 5)))
    shifted_tokens = rng.integers(OOD_LOW, OOD_HIGH + 1, size=int(rng.integers(3, 6)), dtype=np.int64)
    tokens = np.concatenate([active_tokens, inactive_tokens, shifted_tokens])
    rng.shuffle(tokens)
    if len(tokens) > cfg.seq_len - 1:
        tokens = tokens[: cfg.seq_len - 1]
    _place_tokens(rng, seq, tokens)

    m, d, g, support = 0.15, 0.78, 0.72, 0.20
    unc, commit = _targets_from_mdgs(m, d, g, support)
    return seq, -1, 1.0, m, d, g, support, unc, commit


def _make_hard_ood_sample(rng: np.random.Generator, cfg: SeqTaskConfig):
    """Normal-vocabulary boundary cases with no decisive supported label."""

    seq = rng.integers(COMMON_LOW, COMMON_HIGH + 1, size=cfg.seq_len, dtype=np.int64)
    active_channel = int(rng.integers(0, 2))
    inactive_channel = 1 - active_channel
    seq[0] = RULE_A if active_channel == 0 else RULE_B

    active_tokens = np.concatenate(
        [
            _sample_from(rng, CHANNELS[active_channel][0], 2),
            _sample_from(rng, CHANNELS[active_channel][1], 2),
        ]
    )
    inactive_majority = int(rng.integers(0, 2))
    inactive_tokens = _sample_from(rng, CHANNELS[inactive_channel][inactive_majority], int(rng.integers(4, 7)))
    tokens = np.concatenate([active_tokens, inactive_tokens])
    rng.shuffle(tokens)
    if len(tokens) > cfg.seq_len - 1:
        tokens = tokens[: cfg.seq_len - 1]
    _place_tokens(rng, seq, tokens)

    m, d, g, support = 0.02, 0.92, 0.88, 0.25
    unc, commit = _targets_from_mdgs(m, d, g, support)
    return seq, -1, 1.0, m, d, g, support, unc, commit


def _make_support_mismatch_ood_sample(rng: np.random.Generator, cfg: SeqTaskConfig):
    """Valid-looking evidence under an invalid rule token."""

    seq = rng.integers(COMMON_LOW, COMMON_HIGH + 1, size=cfg.seq_len, dtype=np.int64)
    active_channel = int(rng.integers(0, 2))
    inactive_channel = 1 - active_channel
    seq[0] = int(rng.choice(np.array([3, 4, 5, 6], dtype=np.int64)))

    pseudo_y = int(rng.integers(0, 2))
    active_tokens = _sample_from(rng, CHANNELS[active_channel][pseudo_y], 4)
    inactive_tokens = _sample_from(rng, CHANNELS[inactive_channel][1 - pseudo_y], int(rng.integers(2, 5)))
    tokens = np.concatenate([active_tokens, inactive_tokens])
    rng.shuffle(tokens)
    _place_tokens(rng, seq, tokens[: cfg.seq_len - 1])

    m, d, g, support = 0.0, 0.88, 0.95, 0.0
    unc, commit = _targets_from_mdgs(m, d, g, support)
    return seq, -1, 1.0, m, d, g, support, unc, commit


OOD_FACTORIES: Dict[str, Callable[[np.random.Generator, SeqTaskConfig], Tuple[np.ndarray, int, float, float, float, float, float, float, float]]] = {
    "easy": _make_easy_ood_sample,
    "medium": _make_medium_ood_sample,
    "hard": _make_hard_ood_sample,
    "support_mismatch": _make_support_mismatch_ood_sample,
}


def _make_ood_sample(rng: np.random.Generator, cfg: SeqTaskConfig, mode: str):
    try:
        return OOD_FACTORIES[mode](rng, cfg)
    except KeyError as exc:
        valid = ", ".join(sorted(OOD_FACTORIES))
        raise ValueError(f"unknown ood_mode={mode!r}; expected one of: {valid}") from exc


def make_items(cfg: SeqTaskConfig, n_id: int, n_ood: int, seed: int, ood_mode: str | None = None) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    mode = cfg.ood_mode if ood_mode is None else ood_mode
    rows = []
    for _ in range(n_id):
        rows.append(_make_id_sample(rng, cfg))
    for _ in range(n_ood):
        rows.append(_make_ood_sample(rng, cfg, mode))
    rng.shuffle(rows)

    x = np.stack([r[0] for r in rows], axis=0)
    y = np.array([r[1] for r in rows], dtype=np.int64)
    is_ood = np.array([r[2] for r in rows], dtype=np.float32)
    m_target = np.array([r[3] for r in rows], dtype=np.float32)
    d_target = np.array([r[4] for r in rows], dtype=np.float32)
    g_target = np.array([r[5] for r in rows], dtype=np.float32)
    s_target = np.array([r[6] for r in rows], dtype=np.float32)
    unc_target = np.array([r[7] for r in rows], dtype=np.float32)
    commit_target = np.array([r[8] for r in rows], dtype=np.float32)

    return {
        "x": x,
        "y": y,
        "is_ood": is_ood,
        "m_target": m_target,
        "d_target": d_target,
        "g_target": g_target,
        "s_target": s_target,
        "unc_target": unc_target,
        "commit_target": commit_target,
    }


def build_datasets(cfg: SeqTaskConfig):
    n_ood_train = int(cfg.n_train * cfg.ood_ratio_train)
    train = SequenceReliabilityDataset(make_items(cfg, cfg.n_train, n_ood_train, cfg.seed, cfg.ood_mode))
    val = SequenceReliabilityDataset(make_items(cfg, cfg.n_val, 0, cfg.seed + 1))
    test = SequenceReliabilityDataset(make_items(cfg, cfg.n_test, 0, cfg.seed + 2))
    ood_val = SequenceReliabilityDataset(make_items(cfg, 0, cfg.n_val, cfg.seed + 3, cfg.ood_mode))
    ood_test = SequenceReliabilityDataset(make_items(cfg, 0, cfg.n_test, cfg.seed + 4, cfg.ood_mode))
    return train, val, test, ood_val, ood_test
