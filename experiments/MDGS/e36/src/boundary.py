from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F


BOUNDARY_X_MIN = -2.5
BOUNDARY_X_MAX = 3.5
BOUNDARY_Y_MIN = -2.0
BOUNDARY_Y_MAX = 2.5


def make_boundary_grid(
    grid_size: int,
    device: str,
    dtype: torch.dtype = torch.float32,
    x_min: float = BOUNDARY_X_MIN,
    x_max: float = BOUNDARY_X_MAX,
    y_min: float = BOUNDARY_Y_MIN,
    y_max: float = BOUNDARY_Y_MAX,
) -> torch.Tensor:
    xs = np.linspace(x_min, x_max, grid_size, dtype=np.float32)
    ys = np.linspace(y_min, y_max, grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    return torch.from_numpy(grid).to(device=device, dtype=dtype)


@torch.no_grad()
def _grid_predictions(
    model,
    x_grid: torch.Tensor,
    gate_scale: float,
    disable_fusion: bool,
    disable_primitive_attn: bool,
    support_only: bool,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    try:
        preds = []
        batch_size = 4096
        for i in range(0, len(x_grid), batch_size):
            out = model(
                x_grid[i : i + batch_size],
                gate_scale=gate_scale,
                disable_fusion=disable_fusion,
                disable_primitive_attn=disable_primitive_attn,
                support_only=support_only,
            )
            probs = F.softmax(out["logits"], dim=-1)
            preds.append(probs.argmax(dim=-1).detach())
    finally:
        model.train(was_training)
    return torch.cat(preds, dim=0)


def _boundary_mask_from_predictions(pred_grid: torch.Tensor) -> torch.Tensor:
    boundary_mask = torch.zeros_like(pred_grid, dtype=torch.bool)
    boundary_mask[:-1, :] |= pred_grid[:-1, :] != pred_grid[1:, :]
    boundary_mask[1:, :] |= pred_grid[1:, :] != pred_grid[:-1, :]
    boundary_mask[:, :-1] |= pred_grid[:, :-1] != pred_grid[:, 1:]
    boundary_mask[:, 1:] |= pred_grid[:, 1:] != pred_grid[:, :-1]
    return boundary_mask


@torch.no_grad()
def estimate_boundary_grid_points(
    model,
    x_grid: torch.Tensor,
    grid_size: int,
    gate_scale: float,
    disable_fusion: bool,
    disable_primitive_attn: bool,
    support_only: bool,
):
    if grid_size < 2:
        raise ValueError("boundary grid size must be at least 2")
    if x_grid.shape[0] != grid_size * grid_size:
        raise ValueError(
            f"expected a square grid with {grid_size * grid_size} points, got {x_grid.shape[0]}"
        )

    pred = _grid_predictions(
        model,
        x_grid,
        gate_scale=gate_scale,
        disable_fusion=disable_fusion,
        disable_primitive_attn=disable_primitive_attn,
        support_only=support_only,
    )
    pred_grid = pred.view(grid_size, grid_size)
    boundary_mask = _boundary_mask_from_predictions(pred_grid)
    boundary_points = x_grid[boundary_mask.reshape(-1)]
    return boundary_points, boundary_mask


@torch.no_grad()
def boundary_distance_target(
    x: torch.Tensor,
    boundary_points: torch.Tensor,
    boundary_distance_max: float,
    boundary_distance_eps: float,
):
    if boundary_points is None or boundary_points.numel() == 0:
        return None, None
    dist = torch.cdist(x.detach(), boundary_points).min(dim=1).values
    scale = max(float(boundary_distance_max), float(boundary_distance_eps))
    m_target = torch.clamp(dist / scale, 0.0, 1.0)
    return dist, m_target.detach()


def _grid_probabilities(
    model,
    x_grid: torch.Tensor,
    gate_scale: float,
    disable_fusion: bool,
    disable_primitive_attn: bool,
    support_only: bool,
    return_m_scalar: bool = False,
    return_boundary: bool = False,
):
    probs_1 = []
    m_scalars = [] if return_m_scalar else None
    boundary_scalars = [] if return_boundary else None
    batch_size = 4096

    for i in range(0, len(x_grid), batch_size):
        out = model(
            x_grid[i : i + batch_size],
            gate_scale=gate_scale,
            disable_fusion=disable_fusion,
            disable_primitive_attn=disable_primitive_attn,
            support_only=support_only,
        )
        probs = F.softmax(out["logits"], dim=-1)
        probs_1.append(probs[:, 1].detach())
        if return_m_scalar:
            if "order" in out:
                m_scalars.append(out["order"].abs().detach())
            elif "m_scalar" in out:
                m_scalars.append(out["m_scalar"].detach())
            else:
                return None, None, None
        if return_boundary:
            if "boundary" not in out:
                return None, None, None
            boundary_scalars.append(out["boundary"].detach())

    p1 = torch.cat(probs_1, dim=0)
    m_scalar = torch.cat(m_scalars, dim=0) if return_m_scalar else None
    boundary_scalar = torch.cat(boundary_scalars, dim=0) if return_boundary else None
    return p1, m_scalar, boundary_scalar


def _pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> float:
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom) <= eps:
        return 0.0
    return float((x * y).sum() / denom)


@torch.no_grad()
def sample_boundary_batch(
    model,
    x_grid: torch.Tensor,
    gate_scale: float,
    disable_fusion: bool,
    disable_primitive_attn: bool,
    support_only: bool,
    near_low: float,
    near_high: float,
    far_low: float,
    far_high: float,
    boundary_batch_size: int,
):
    was_training = model.training
    model.eval()
    try:
        p1, _, _ = _grid_probabilities(
            model,
            x_grid,
            gate_scale=gate_scale,
            disable_fusion=disable_fusion,
            disable_primitive_attn=disable_primitive_attn,
            support_only=support_only,
            return_m_scalar=False,
            return_boundary=False,
        )
    finally:
        model.train(was_training)

    stats = {
        "num_boundary_points": torch.tensor(0.0, device=x_grid.device),
        "num_far_points": torch.tensor(0.0, device=x_grid.device),
        "boundary_batches_used": torch.tensor(0.0, device=x_grid.device),
        "mean_boundary_m_target": torch.tensor(0.0, device=x_grid.device),
    }

    if p1 is None:
        return None, None, stats

    near_mask = (p1 >= near_low) & (p1 <= near_high)
    far_mask = (p1 <= far_low) | (p1 >= far_high)

    near_idx = torch.nonzero(near_mask, as_tuple=False).flatten()
    far_idx = torch.nonzero(far_mask, as_tuple=False).flatten()

    half = boundary_batch_size // 2
    if half <= 0 or near_idx.numel() < half or far_idx.numel() < half:
        return None, None, stats

    near_idx = near_idx[torch.randperm(near_idx.numel(), device=x_grid.device)[:half]]
    far_idx = far_idx[torch.randperm(far_idx.numel(), device=x_grid.device)[:half]]

    x_m = torch.cat([x_grid[near_idx], x_grid[far_idx]], dim=0)
    m_target = torch.cat(
        [
            torch.zeros(half, device=x_grid.device, dtype=x_grid.dtype),
            torch.ones(half, device=x_grid.device, dtype=x_grid.dtype),
        ],
        dim=0,
    )
    perm = torch.randperm(x_m.shape[0], device=x_grid.device)
    x_m = x_m[perm]
    m_target = m_target[perm]

    stats.update(
        {
            "num_boundary_points": torch.tensor(float(half), device=x_grid.device),
            "num_far_points": torch.tensor(float(half), device=x_grid.device),
            "boundary_batches_used": torch.tensor(1.0, device=x_grid.device),
            "mean_boundary_m_target": m_target.mean().detach(),
        }
    )
    return x_m, m_target, stats


@torch.no_grad()
def evaluate_boundary_distance_m_diagnostics(
    model,
    loader,
    x_grid: torch.Tensor,
    grid_size: int,
    gate_scale: float,
    disable_fusion: bool,
    disable_primitive_attn: bool,
    support_only: bool,
    boundary_distance_max: float,
    boundary_distance_eps: float,
):
    was_training = model.training
    model.eval()
    try:
        boundary_points, boundary_mask = estimate_boundary_grid_points(
            model,
            x_grid,
            grid_size=grid_size,
            gate_scale=gate_scale,
            disable_fusion=disable_fusion,
            disable_primitive_attn=disable_primitive_attn,
            support_only=support_only,
        )
    finally:
        model.train(was_training)

    num_boundary_grid_points = float(boundary_mask.sum().item())
    if boundary_points is None or boundary_points.numel() == 0:
        return {
            "m_distance_corr": 0.0,
            "m_distance_mae": 0.0,
            "m_near_mean": 0.0,
            "m_far_mean": 0.0,
            "m_distance_separation": 0.0,
            "num_boundary_grid_points": num_boundary_grid_points,
            "mean_m_distance_target": 0.0,
            "mean_m_distance_pred": 0.0,
        }

    device = x_grid.device
    all_dist = []
    all_m_target = []
    all_m_pred = []
    for x, y, is_ood in loader:
        mask = y >= 0
        if not mask.any():
            continue
        x_batch = x[mask].to(device)
        out = model(
            x_batch,
            gate_scale=gate_scale,
            disable_fusion=disable_fusion,
            disable_primitive_attn=disable_primitive_attn,
            support_only=support_only,
        )
        if "m_scalar" not in out:
            return {}
        dist, m_target = boundary_distance_target(
            x_batch,
            boundary_points,
            boundary_distance_max=boundary_distance_max,
            boundary_distance_eps=boundary_distance_eps,
        )
        if dist is None or m_target is None:
            continue
        all_dist.append(dist.detach().cpu())
        all_m_target.append(m_target.detach().cpu())
        all_m_pred.append(out["m_scalar"].detach().cpu())

    if not all_m_pred:
        return {
            "m_distance_corr": 0.0,
            "m_distance_mae": 0.0,
            "m_near_mean": 0.0,
            "m_far_mean": 0.0,
            "m_distance_separation": 0.0,
            "num_boundary_grid_points": num_boundary_grid_points,
            "mean_m_distance_target": 0.0,
            "mean_m_distance_pred": 0.0,
        }

    dist = torch.cat(all_dist)
    m_target = torch.cat(all_m_target)
    m_pred = torch.cat(all_m_pred)

    order = torch.argsort(dist)
    k = max(1, math.ceil(0.2 * len(dist)))
    near_idx = order[:k]
    far_idx = order[-k:]

    m_near_mean = float(m_pred[near_idx].mean())
    m_far_mean = float(m_pred[far_idx].mean())

    return {
        "m_distance_corr": _pearson_corr(m_pred, m_target, eps=boundary_distance_eps),
        "m_distance_mae": float((m_pred - m_target).abs().mean()),
        "m_near_mean": m_near_mean,
        "m_far_mean": m_far_mean,
        "m_distance_separation": m_far_mean - m_near_mean,
        "num_boundary_grid_points": num_boundary_grid_points,
        "mean_m_distance_target": float(m_target.mean()),
        "mean_m_distance_pred": float(m_pred.mean()),
    }


@torch.no_grad()
def evaluate_boundary_m_diagnostics(
    model,
    x_grid: torch.Tensor,
    gate_scale: float,
    disable_fusion: bool,
    disable_primitive_attn: bool,
    support_only: bool,
    near_low: float,
    near_high: float,
    far_low: float,
    far_high: float,
):
    was_training = model.training
    model.eval()
    evidence_mode = bool(getattr(getattr(model, "cfg", None), "evidence_mode", False))
    try:
        p1, m_scalar, boundary_scalar = _grid_probabilities(
            model,
            x_grid,
            gate_scale=gate_scale,
            disable_fusion=disable_fusion,
            disable_primitive_attn=disable_primitive_attn,
            support_only=support_only,
            return_m_scalar=True,
            return_boundary=evidence_mode,
        )
    finally:
        model.train(was_training)

    if p1 is None or m_scalar is None:
        return {}

    near_mask = (p1 >= near_low) & (p1 <= near_high)
    far_mask = (p1 <= far_low) | (p1 >= far_high)

    m_near_mean = float("nan")
    m_far_mean = float("nan")
    if near_mask.any():
        m_near_mean = float(m_scalar[near_mask].mean().detach().cpu())
    if far_mask.any():
        m_far_mean = float(m_scalar[far_mask].mean().detach().cpu())

    separation = float("nan")
    if not np.isnan(m_near_mean) and not np.isnan(m_far_mean):
        separation = m_far_mean - m_near_mean

    metrics = {
        "m_near_mean": m_near_mean,
        "m_far_mean": m_far_mean,
        "m_boundary_separation": separation,
    }
    if boundary_scalar is not None:
        boundary_near_mean = float("nan")
        boundary_far_mean = float("nan")
        if near_mask.any():
            boundary_near_mean = float(boundary_scalar[near_mask].mean().detach().cpu())
        if far_mask.any():
            boundary_far_mean = float(boundary_scalar[far_mask].mean().detach().cpu())

        boundary_separation = float("nan")
        if not np.isnan(boundary_near_mean) and not np.isnan(boundary_far_mean):
            boundary_separation = boundary_near_mean - boundary_far_mean

        metrics.update({
            "boundary_near_mean": boundary_near_mean,
            "boundary_far_mean": boundary_far_mean,
            "boundary_separation": boundary_separation,
        })
    return metrics
