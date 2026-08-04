from __future__ import annotations

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def _grid_eval(
    model,
    device: str,
    disable_fusion: bool = False,
    disable_primitive_attn: bool = False,
    support_only: bool = False,
    n: int = 250,
):
    xs = np.linspace(-2.0, 2.0, n)
    ys = np.linspace(-2.0, 2.0, n)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    x = torch.from_numpy(grid).to(device)
    outs = []
    bs = 4096
    for i in range(0, len(x), bs):
        out = model(
            x[i:i+bs],
            disable_fusion=disable_fusion,
            disable_primitive_attn=disable_primitive_attn,
            support_only=support_only,
        )
        probs = F.softmax(out["logits"], dim=-1)
        item = {
            "p1": probs[:, 1].detach().cpu(),
            "uncertainty": out["uncertainty"].detach().cpu(),
            "support": out["support"].detach().cpu(),
            "commitment": out["commitment"].detach().cpu(),
            "gate": out["gate"].detach().cpu(),
        }
        if "m_scalar" in out:
            item["m_scalar"] = out["m_scalar"].detach().cpu()
        if "order" in out:
            item["order"] = out["order"].detach().cpu()
        if "boundary" in out:
            item["boundary"] = out["boundary"].detach().cpu()
        outs.append(item)
    merged: Dict[str, torch.Tensor] = {}
    for k in outs[0].keys():
        merged[k] = torch.cat([o[k] for o in outs]).numpy().reshape(n, n)
    return xx, yy, merged


def _scatter_train(ax, dataset):
    x = dataset.x.numpy()
    y = dataset.y.numpy()
    mask = y >= 0
    ax.scatter(x[mask & (y == 0), 0], x[mask & (y == 0), 1], s=5, alpha=0.45, label="class 0")
    ax.scatter(x[mask & (y == 1), 0], x[mask & (y == 1), 1], s=5, alpha=0.45, label="class 1")
    if (~mask).any():
        ax.scatter(x[~mask, 0], x[~mask, 1], s=4, alpha=0.15, label="train OOD")


def save_surfaces(
    model,
    dataset,
    device: str,
    out_dir: str,
    disable_fusion: bool = False,
    disable_primitive_attn: bool = False,
    support_only: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    xx, yy, vals = _grid_eval(
        model,
        device,
        disable_fusion=disable_fusion,
        disable_primitive_attn=disable_primitive_attn,
        support_only=support_only,
    )

    specs = [
        ("decision_surface.png", "p1", "P(class 1)"),
    ]
    if "m_scalar" in vals:
        specs.append(("m_surface.png", "m_scalar", "M primitive"))
    if "order" in vals:
        specs.extend([
            ("order_surface.png", "order", "Order O"),
            ("boundary_surface.png", "boundary", "Boundary B"),
        ])
    specs.extend([
        ("uncertainty_surface.png", "uncertainty", "Uncertainty U"),
        ("support_surface.png", "support", "Support S"),
        ("commitment_surface.png", "commitment", "Commitment C"),
        (
            "gate_surface.png",
            "gate",
            "Evidence confidence proxy" if "order" in vals else "UGLY fusion gate",
        ),
    ])

    for fname, key, title in specs:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
        im = ax.contourf(xx, yy, vals[key], levels=40)
        _scatter_train(ax, dataset)
        ax.set_title(title)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.legend(loc="upper right", fontsize=7)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname))
        plt.close(fig)


def save_training_curves(history, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
    ax.plot(history["epoch"], history["train_loss"], label="train loss")
    ax.plot(history["epoch"], history["val_nll"], label="val nll")
    ax.plot(history["epoch"], history["val_acc"], label="val acc")
    ax.set_xlabel("epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"))
    plt.close(fig)
