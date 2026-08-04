"""Generic training loop for ValidationDIGITModel (bottleneck-only).

Trains encoder + bottleneck against cross-entropy on the four primitives.
No text decoder.  Supports early stopping and multi-seed runs.

Produces a checkpoint at:
    checkpoints/{dataset}/seed_{seed}/best_model.pt
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ..data.base import PrivateDataset
from ..data.dataset import ValidationDataset, collate_fn
from ..data.ground_truth import GTConfig
from ..models.digit import ValidationConfig, ValidationDIGITModel

logger = logging.getLogger(__name__)

SEEDS = [0, 1, 2, 3, 4]


def _class_counts_from_cfg(cfg: ValidationConfig) -> Dict[str, int]:
    return {
        "answer": int(cfg.num_answer_classes),
        "support": int(cfg.num_support_classes),
        "confidence": int(cfg.num_confidence_classes),
        "risk": int(cfg.num_risk_classes),
    }


# ── Gumbel annealing ──────────────────────────────────────────────────────────
def _tau(step: int, total_steps: int, tau_start: float = 2.0, tau_end: float = 0.5) -> float:
    r = min(1.0, step / total_steps)
    return tau_start + (tau_end - tau_start) * r


# ── Loss ──────────────────────────────────────────────────────────────────────
def _primitive_loss(
    primitives,
    targets: torch.Tensor,
    class_weights: Optional[Dict] = None,
) -> torch.Tensor:
    """Cross-entropy on all four primitives, optionally weighted."""
    names = ["answer", "support", "confidence", "risk"]
    logit_attrs = ["answer_logits", "support_logits", "confidence_logits", "risk_logits"]
    total = torch.tensor(0.0, device=targets.device)
    for i, (name, attr) in enumerate(zip(names, logit_attrs)):
        logits = getattr(primitives, attr)
        target_i = targets[:, i]
        w = class_weights[name].to(targets.device) if class_weights else None
        total = total + nn.functional.cross_entropy(logits, target_i, weight=w)
    return total / len(names)


# ── Single-seed training ──────────────────────────────────────────────────────
def train_single_seed(
    model: ValidationDIGITModel,
    train_data: PrivateDataset,
    val_data: PrivateDataset,
    seed: int,
    out_dir: Path,
    n_train_queries: int = 8000,
    n_val_queries: int = 2000,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    num_epochs: int = 60,
    patience: int = 10,
    device: torch.device = torch.device("cpu"),
    gt_cfg: Optional[GTConfig] = None,
) -> Dict:
    """Train for one seed and save best checkpoint.

    Returns dict with training history and best_val_loss.
    """
    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)

    class_counts = _class_counts_from_cfg(model.cfg)
    train_ds = ValidationDataset(
        train_data,
        n_train_queries,
        seed=seed,
        gt_cfg=gt_cfg,
        class_counts=class_counts,
    )
    val_ds = ValidationDataset(
        val_data,
        n_val_queries,
        seed=seed + 1000,
        gt_cfg=gt_cfg,
        class_counts=class_counts,
    )
    class_weights = train_ds.get_class_weights()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = num_epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val = float("inf")
    patience_counter = 0
    history = []
    global_step = 0

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = {k: 0 for k in ["answer", "support", "confidence", "risk"]}
        epoch_total = 0
        t0 = time.time()

        for queries, targets in train_loader:
            queries = queries.to(device)
            targets = targets.to(device)

            tau = _tau(global_step, total_steps)
            out = model(queries, train_data, mode="gumbel", tau=tau)
            loss = _primitive_loss(out["primitives"], targets, class_weights)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            B = queries.size(0)
            epoch_loss  += loss.item() * B
            epoch_total += B
            p = out["primitives"]
            epoch_correct["answer"]     += (p.answer_logits.argmax(-1)     == targets[:, 0]).sum().item()
            epoch_correct["support"]    += (p.support_logits.argmax(-1)    == targets[:, 1]).sum().item()
            epoch_correct["confidence"] += (p.confidence_logits.argmax(-1) == targets[:, 2]).sum().item()
            epoch_correct["risk"]       += (p.risk_logits.argmax(-1)       == targets[:, 3]).sum().item()
            global_step += 1

        scheduler.step()
        train_acc = {k: v / max(epoch_total, 1) for k, v in epoch_correct.items()}

        # ── Validation ──────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_correct = {k: 0 for k in epoch_correct}
        val_total = 0
        with torch.no_grad():
            for queries, targets in val_loader:
                queries = queries.to(device)
                targets = targets.to(device)
                out  = model(queries, val_data, mode="hard", tau=1.0)
                loss = _primitive_loss(out["primitives"], targets)
                B    = queries.size(0)
                val_loss  += loss.item() * B
                val_total += B
                p = out["primitives"]
                val_correct["answer"]     += (p.answer_logits.argmax(-1)     == targets[:, 0]).sum().item()
                val_correct["support"]    += (p.support_logits.argmax(-1)    == targets[:, 1]).sum().item()
                val_correct["confidence"] += (p.confidence_logits.argmax(-1) == targets[:, 2]).sum().item()
                val_correct["risk"]       += (p.risk_logits.argmax(-1)       == targets[:, 3]).sum().item()

        val_loss /= max(val_total, 1)
        val_acc = {k: v / max(val_total, 1) for k, v in val_correct.items()}
        elapsed = time.time() - t0

        logger.info(
            f"[Seed {seed}] Ep {epoch+1}/{num_epochs} ({elapsed:.1f}s) | "
            f"train={epoch_loss/max(epoch_total,1):.4f} | "
            f"val={val_loss:.4f} | "
            f"ans={val_acc['answer']:.3f} τ={_tau(global_step, total_steps):.2f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), seed_dir / "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  → Early stopping at epoch {epoch+1}")
                break

        history.append({
            "epoch": epoch + 1,
            "train_loss": epoch_loss / max(epoch_total, 1),
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        })

    return {"seed": seed, "best_val_loss": best_val, "history": history}


# ── Multi-seed training ───────────────────────────────────────────────────────
def train_all_seeds(
    cfg: ValidationConfig,
    train_data: PrivateDataset,
    val_data: PrivateDataset,
    out_dir: Path,
    seeds: List[int] = SEEDS,
    device: Optional[torch.device] = None,
    **train_kwargs,
) -> Dict[int, Path]:
    """Train one model per seed; return {seed: checkpoint_path}."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoints = {}
    for s in seeds:
        logger.info(f"\n{'='*50}\nTraining seed {s}\n{'='*50}")
        model = ValidationDIGITModel(cfg).to(device)
        train_single_seed(model, train_data, val_data, s, out_dir, device=device, **train_kwargs)
        checkpoints[s] = out_dir / f"seed_{s}" / "best_model.pt"

    return checkpoints


def load_checkpoint(cfg: ValidationConfig, path: Path, device: torch.device) -> ValidationDIGITModel:
    model = ValidationDIGITModel(cfg).to(device)
    model.load_state_dict(torch.load(str(path), map_location=device, weights_only=True))
    model.eval()
    return model
