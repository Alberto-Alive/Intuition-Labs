from __future__ import annotations

import argparse
import math
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import MoonConfig, build_datasets
from .boundary import (
    boundary_distance_target,
    estimate_boundary_grid_points,
    evaluate_boundary_distance_m_diagnostics,
    make_boundary_grid,
    sample_boundary_batch,
)
from .metrics import evaluate_model, evaluate_ood
from .model import BaselineMLP, ModelConfig, UGLYNet
from .plot import save_surfaces, save_training_curves


def choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def symmetric_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    kl_pq = (p * (p.log() - q.log())).sum(dim=-1)
    kl_qp = (q * (q.log() - p.log())).sum(dim=-1)
    return 0.5 * (kl_pq + kl_qp)


def build_witness_targets(
    out_clean: Dict[str, torch.Tensor],
    out_w: Dict[str, torch.Tensor],
    num_witnesses: int,
) -> Dict[str, torch.Tensor]:
    logits_clean = out_clean["logits"].detach()
    batch_size = logits_clean.shape[0]
    num_classes = logits_clean.shape[-1]

    logits_w = out_w["logits"].reshape(batch_size, num_witnesses, num_classes)
    h_w = out_w["h"].reshape(batch_size, num_witnesses, -1)

    probs_w = F.softmax(logits_w, dim=-1)
    pred_clean = logits_clean.argmax(dim=-1)
    pred_w = probs_w.argmax(dim=-1)

    flip_rate = (pred_w != pred_clean[:, None]).float().mean(dim=1)

    vote_counts = F.one_hot(pred_w, num_classes=num_classes).float().sum(dim=1)
    vote_probs = vote_counts / float(num_witnesses)
    vote_entropy = -(vote_probs * (vote_probs + 1e-8).log()).sum(dim=-1)
    vote_entropy_norm = math.log(float(num_classes)) if num_classes > 1 else 1.0
    vote_entropy = (vote_entropy / vote_entropy_norm).clamp(0.0, 1.0)

    h_w_norm = F.normalize(h_w, dim=-1)
    h_center = h_w_norm.mean(dim=1, keepdim=True)
    latent_spread = torch.norm(h_w_norm - h_center, dim=-1).mean(dim=1)
    spread_norm = latent_spread / (latent_spread.detach().mean() + 1e-6)
    g_threshold = spread_norm.detach().median()

    m_target = ((flip_rate <= 0.125) & (vote_entropy <= 0.20)).float().detach()
    d_target = ((flip_rate >= 0.25) | (vote_entropy >= 0.35)).float().detach()
    g_target = (spread_norm >= g_threshold).float().detach()

    return {
        "m_target": m_target,
        "d_target": d_target,
        "g_target": g_target,
        "mean_m_target": m_target.mean(),
        "mean_d_target": d_target.mean(),
        "mean_g_target": g_target.mean(),
    }


def loss_for_batch(
    model,
    batch,
    device: str,
    cfg,
    gate_scale: float,
    disable_fusion: bool,
    disable_uncertainty_loss: bool,
    disable_support_loss: bool,
    disable_primitive_attn: bool,
    support_only: bool,
    is_ugly: bool,
    boundary_grid: torch.Tensor | None = None,
    boundary_distance_grid: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    x, y, is_ood = batch
    x = x.to(device)
    y = y.to(device)
    is_ood = is_ood.to(device)

    x_tgt = None
    if is_ugly:
        # JEPA-style latent target uses a slightly perturbed view.
        x_tgt = x + torch.randn_like(x) * cfg.jepa_noise

    out_clean = model(
        x,
        jepa_target_x=x_tgt,
        gate_scale=gate_scale,
        disable_fusion=disable_fusion,
        disable_primitive_attn=disable_primitive_attn,
        support_only=support_only,
    )

    in_mask = y >= 0
    per_ce = torch.zeros(len(x), device=device)
    if in_mask.any():
        per_ce[in_mask] = F.cross_entropy(out_clean["logits"][in_mask], y[in_mask], reduction="none")
        pred_loss = per_ce[in_mask].mean()
    else:
        pred_loss = torch.tensor(0.0, device=device)

    total = pred_loss
    losses = {"loss": total, "pred": pred_loss}
    m_loss = torch.tensor(0.0, device=device)
    d_loss = torch.tensor(0.0, device=device)
    g_loss = torch.tensor(0.0, device=device)
    m_witness_loss = torch.tensor(0.0, device=device)
    d_witness_loss = torch.tensor(0.0, device=device)
    g_witness_loss = torch.tensor(0.0, device=device)
    boundary_m_loss = torch.tensor(0.0, device=device)
    boundary_distance_m_loss = torch.tensor(0.0, device=device)
    boundary_evidence_loss = torch.tensor(0.0, device=device)
    evidence_conf_loss = torch.tensor(0.0, device=device)
    mean_m_target = torch.tensor(0.0, device=device)
    mean_d_target = torch.tensor(0.0, device=device)
    mean_g_target = torch.tensor(0.0, device=device)
    mean_boundary_m_target = torch.tensor(0.0, device=device)
    mean_boundary_m_pred = torch.tensor(0.0, device=device)
    mean_m_distance_target = torch.tensor(0.0, device=device)
    mean_m_distance_pred = torch.tensor(0.0, device=device)
    num_boundary_points = torch.tensor(0.0, device=device)
    num_boundary_grid_points = torch.tensor(0.0, device=device)
    num_far_points = torch.tensor(0.0, device=device)
    boundary_batches_used = torch.tensor(0.0, device=device)
    boundary_distance_batches_used = torch.tensor(0.0, device=device)
    evidence_mode = bool(getattr(cfg, "evidence_mode", False))

    if is_ugly:
        losses.update({
            "gate_mean": out_clean["gate"].mean().detach(),
            "unc_mean": out_clean["uncertainty"].mean().detach(),
            "support_mean": out_clean["support"].mean().detach(),
        })
        if evidence_mode:
            losses.update({
                "mean_order": out_clean["order"].mean().detach(),
                "mean_abs_order": out_clean["m_scalar"].mean().detach(),
                "mean_boundary": out_clean["boundary"].mean().detach(),
                "mean_support": out_clean["support"].mean().detach(),
                "mean_uncertainty": out_clean["uncertainty"].mean().detach(),
                "mean_commitment": out_clean["commitment"].mean().detach(),
                "mean_gate": out_clean["gate"].mean().detach(),
            })

        if not disable_uncertainty_loss:
            # U target means estimated distance-to-truth.
            # In-distribution: normalized CE. OOD: high uncertainty.
            unc_target = torch.where(
                in_mask,
                1.0 - torch.exp(-per_ce.detach()).clamp(0, 1),
                torch.ones_like(per_ce),
            )
            unc_loss = F.mse_loss(out_clean["uncertainty"], unc_target)

            # Support target: in-distribution high, synthetic OOD low.
            support_target = 1.0 - is_ood
            support_loss = F.binary_cross_entropy(out_clean["support"].clamp(1e-5, 1 - 1e-5), support_target)

            # Commitment target: commit if in-distribution and current prediction is correct.
            # Small weight; this is a behavioral regularizer, not the main objective.
            with torch.no_grad():
                pred = out_clean["logits"].argmax(dim=-1)
                correct = (in_mask & (pred == y)).float()
            commit_loss = F.binary_cross_entropy(out_clean["commitment"].clamp(1e-5, 1 - 1e-5), correct)

            jepa_loss = torch.tensor(0.0, device=device)
            if "jepa_pred" in out_clean:
                jepa_loss = F.mse_loss(out_clean["jepa_pred"], out_clean["jepa_target"])

            total = pred_loss
            total = total + cfg.beta_unc * unc_loss
            if not disable_support_loss:
                total = total + cfg.beta_support * support_loss
            total = total + cfg.beta_commit * commit_loss
            total = total + cfg.beta_jepa * jepa_loss
            losses.update({
                "loss": total,
                "unc": unc_loss,
                "support": support_loss,
                "commit": commit_loss,
                "jepa": jepa_loss,
            })

        if evidence_mode and getattr(cfg, "lambda_evidence_coop", 0.0) > 0.0:
            pred_conf = out_clean["logits"].softmax(dim=-1).max(dim=-1).values.detach()
            evidence_conf = out_clean["support"] * (1.0 - out_clean["boundary"])
            evidence_conf_loss = F.mse_loss(evidence_conf, pred_conf)
            total = total + cfg.lambda_evidence_coop * evidence_conf_loss
            losses["loss"] = total
            losses["l_evidence_conf"] = evidence_conf_loss

        if cfg.use_perturbation_loss and not support_only:
            with torch.no_grad():
                x_pert = x + cfg.perturb_std * torch.randn_like(x)
                out_pert = model(
                    x_pert,
                    gate_scale=gate_scale,
                    disable_fusion=disable_fusion,
                    disable_primitive_attn=disable_primitive_attn,
                    support_only=support_only,
                )

                logits_clean = out_clean["logits"].detach()
                logits_pert = out_pert["logits"].detach()
                h_clean = out_clean["h"].detach()
                h_pert = out_pert["h"].detach()

                probs_clean = F.softmax(logits_clean, dim=-1)
                probs_pert = F.softmax(logits_pert, dim=-1)
                pred_clean = probs_clean.argmax(dim=-1)
                pred_pert = probs_pert.argmax(dim=-1)
                flip = (pred_clean != pred_pert).float()

                top2 = torch.topk(logits_clean, k=min(2, logits_clean.shape[-1]), dim=-1).values
                if top2.shape[-1] > 1:
                    margin = top2[:, 0] - top2[:, 1]
                else:
                    margin = top2[:, 0]
                m_target = torch.sigmoid(margin) * (1.0 - flip)

                d_target = symmetric_kl(probs_clean, probs_pert)
                d_target = d_target / (d_target.detach().mean() + 1e-6)
                d_target = d_target.clamp(0.0, 1.0)
                d_target = torch.maximum(d_target, flip)

                latent_shift = torch.norm(
                    F.normalize(h_clean, dim=-1) - F.normalize(h_pert, dim=-1),
                    dim=-1,
                )
                g_target = latent_shift / (latent_shift.detach().mean() + 1e-6)
                g_target = g_target.clamp(0.0, 1.0)

            m_loss = F.mse_loss(out_clean["m_scalar"], m_target)
            d_loss = F.mse_loss(out_clean["d_scalar"], d_target)
            g_loss = F.mse_loss(out_clean["g_scalar"], g_target)
            total = total + cfg.lambda_m * m_loss + cfg.lambda_d * d_loss + cfg.lambda_g * g_loss
            losses["loss"] = total

        if getattr(cfg, "use_witness_mdg_loss", False) and not support_only and not disable_uncertainty_loss:
            with torch.no_grad():
                batch_size, input_dim = x.shape
                num_witnesses = getattr(cfg, "num_witnesses", 8)
                witness_std = getattr(cfg, "witness_std", 0.20)
                witness_noise = torch.randn(
                    batch_size,
                    num_witnesses,
                    input_dim,
                    device=device,
                    dtype=x.dtype,
                )
                x_w = x[:, None, :].expand(-1, num_witnesses, -1) + witness_std * witness_noise
                x_w = x_w.reshape(batch_size * num_witnesses, input_dim)
                out_w = model(
                    x_w,
                    gate_scale=gate_scale,
                    disable_fusion=disable_fusion,
                    disable_primitive_attn=disable_primitive_attn,
                    support_only=support_only,
                )
                witness_targets = build_witness_targets(out_clean, out_w, num_witnesses)

            m_witness_loss = F.binary_cross_entropy(
                out_clean["m_scalar"].clamp(1e-5, 1 - 1e-5),
                witness_targets["m_target"],
            )
            d_witness_loss = F.binary_cross_entropy(
                out_clean["d_scalar"].clamp(1e-5, 1 - 1e-5),
                witness_targets["d_target"],
            )
            g_witness_loss = F.binary_cross_entropy(
                out_clean["g_scalar"].clamp(1e-5, 1 - 1e-5),
                witness_targets["g_target"],
            )
            total = total + cfg.lambda_m * m_witness_loss + cfg.lambda_d * d_witness_loss + cfg.lambda_g * g_witness_loss
            losses["loss"] = total
            mean_m_target = witness_targets["mean_m_target"]
            mean_d_target = witness_targets["mean_d_target"]
            mean_g_target = witness_targets["mean_g_target"]

        if (
            getattr(cfg, "use_boundary_m_loss", False)
            and not support_only
            and boundary_grid is not None
            and (evidence_mode or not disable_uncertainty_loss)
        ):
            x_m, m_target, boundary_stats = sample_boundary_batch(
                model,
                boundary_grid,
                gate_scale=gate_scale,
                disable_fusion=disable_fusion,
                disable_primitive_attn=disable_primitive_attn,
                support_only=False,
                near_low=getattr(cfg, "boundary_near_low", 0.45),
                near_high=getattr(cfg, "boundary_near_high", 0.55),
                far_low=getattr(cfg, "boundary_far_low", 0.05),
                far_high=getattr(cfg, "boundary_far_high", 0.95),
                boundary_batch_size=getattr(cfg, "boundary_batch_size", 256),
            )
            mean_boundary_m_target = boundary_stats["mean_boundary_m_target"]
            num_boundary_points = boundary_stats["num_boundary_points"]
            num_far_points = boundary_stats["num_far_points"]
            boundary_batches_used = boundary_stats["boundary_batches_used"]
            if x_m is not None and m_target is not None:
                out_m = model(
                    x_m,
                    gate_scale=gate_scale,
                    disable_fusion=disable_fusion,
                    disable_primitive_attn=disable_primitive_attn,
                    support_only=False,
                )
                if evidence_mode:
                    boundary_target = 1.0 - m_target
                    boundary_pred = out_m["boundary"]
                    m_pred = out_m["m_scalar"]
                    boundary_loss = F.binary_cross_entropy(boundary_pred.clamp(1e-5, 1 - 1e-5), boundary_target)
                    order_margin_loss = F.binary_cross_entropy(m_pred.clamp(1e-5, 1 - 1e-5), m_target)
                    boundary_evidence_loss = boundary_loss + order_margin_loss
                    mean_boundary_m_pred = m_pred.mean().detach()
                    total = total + cfg.lambda_boundary_m * boundary_evidence_loss
                    losses["l_boundary_evidence"] = boundary_evidence_loss
                else:
                    m_pred = out_m["m_scalar"]
                    boundary_m_loss = F.binary_cross_entropy(m_pred.clamp(1e-5, 1 - 1e-5), m_target)
                    mean_boundary_m_pred = m_pred.mean().detach()
                    total = total + cfg.lambda_boundary_m * boundary_m_loss
                losses["loss"] = total

        if (
            getattr(cfg, "use_boundary_distance_m_loss", False)
            and not support_only
            and not disable_uncertainty_loss
            and boundary_distance_grid is not None
        ):
            boundary_points, boundary_mask = estimate_boundary_grid_points(
                model,
                boundary_distance_grid,
                grid_size=getattr(cfg, "boundary_distance_grid_size", 200),
                gate_scale=gate_scale,
                disable_fusion=disable_fusion,
                disable_primitive_attn=disable_primitive_attn,
                support_only=False,
            )
            num_boundary_grid_points = torch.tensor(float(boundary_mask.sum().item()), device=device)
            if boundary_points.numel() > 0:
                _, m_target = boundary_distance_target(
                    x,
                    boundary_points,
                    boundary_distance_max=getattr(cfg, "boundary_distance_max", 2.0),
                    boundary_distance_eps=getattr(cfg, "boundary_distance_eps", 1e-6),
                )
                if m_target is not None:
                    m_pred = out_clean["m_scalar"]
                    boundary_distance_m_loss = F.smooth_l1_loss(m_pred, m_target)
                    mean_m_distance_target = m_target.mean().detach()
                    mean_m_distance_pred = m_pred.mean().detach()
                    boundary_distance_batches_used = torch.tensor(1.0, device=device)
                    total = total + cfg.lambda_boundary_distance_m * boundary_distance_m_loss
                    losses["loss"] = total

    losses.update({
        "l_m": m_loss,
        "l_d": d_loss,
        "l_g": g_loss,
        "l_m_witness": m_witness_loss,
        "l_d_witness": d_witness_loss,
        "l_g_witness": g_witness_loss,
        "l_boundary_m": boundary_m_loss,
        "l_boundary_distance_m": boundary_distance_m_loss,
        "mean_m_target": mean_m_target,
        "mean_d_target": mean_d_target,
        "mean_g_target": mean_g_target,
        "mean_boundary_m_target": mean_boundary_m_target,
        "mean_boundary_m_pred": mean_boundary_m_pred,
        "mean_m_distance_target": mean_m_distance_target,
        "mean_m_distance_pred": mean_m_distance_pred,
        "num_boundary_points": num_boundary_points,
        "num_boundary_grid_points": num_boundary_grid_points,
        "num_far_points": num_far_points,
        "boundary_batches_used": boundary_batches_used,
        "boundary_distance_batches_used": boundary_distance_batches_used,
    })
    if evidence_mode:
        losses.setdefault("l_evidence_conf", evidence_conf_loss)
        losses.setdefault("l_boundary_evidence", boundary_evidence_loss)
    return losses


def train(args):
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    device = choose_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.use_boundary_m_loss and args.use_boundary_distance_m_loss:
        raise ValueError("cannot use --use-boundary-m-loss and --use-boundary-distance-m-loss together")

    data_cfg = MoonConfig(
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        noise=args.noise,
        seed=args.seed,
        ood_ratio_train=args.ood_ratio_train,
    )
    train_ds, val_ds, test_ds, ood_val_ds, ood_test_ds = build_datasets(data_cfg)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    ood_val_loader = DataLoader(ood_val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    ood_test_loader = DataLoader(ood_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model_cfg = ModelConfig(
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        token_dim=args.token_dim,
        num_views=args.num_views,
        num_prototypes=args.num_prototypes,
        dropout=args.dropout,
        use_jepa=not args.no_jepa,
        ema_decay=args.ema_decay,
        evidence_mode=args.evidence_mode,
        order_temp=args.order_temp,
    )
    is_ugly = args.model == "ugly"
    model = UGLYNet(model_cfg) if is_ugly else BaselineMLP(model_cfg)
    model.to(device)
    boundary_grid = make_boundary_grid(args.boundary_grid_size, device=device, dtype=train_ds.x.dtype) if is_ugly else None
    boundary_distance_grid = (
        make_boundary_grid(args.boundary_distance_grid_size, device=device, dtype=train_ds.x.dtype)
        if is_ugly
        else None
    )
    effective_disable_fusion = args.disable_fusion or args.evidence_mode
    effective_disable_primitive_attn = args.disable_primitive_attn or args.evidence_mode
    print({
        "evidence_mode": args.evidence_mode,
        "order_temp": args.order_temp,
        "lambda_evidence_coop": args.lambda_evidence_coop,
        "use_boundary_m_loss": args.use_boundary_m_loss,
        "use_boundary_distance_m_loss": args.use_boundary_distance_m_loss,
        "disable_fusion": effective_disable_fusion,
        "disable_primitive_attn": effective_disable_primitive_attn,
        "requested_disable_fusion": args.disable_fusion,
        "requested_disable_primitive_attn": args.disable_primitive_attn,
        "disable_uncertainty_loss": args.disable_uncertainty_loss,
        "disable_support_loss": args.disable_support_loss,
        "use_witness_mdg_loss": args.use_witness_mdg_loss,
        "num_witnesses": args.num_witnesses,
        "witness_std": args.witness_std,
        "lambda_m": args.lambda_m,
        "lambda_d": args.lambda_d,
        "lambda_g": args.lambda_g,
        "lambda_boundary_m": args.lambda_boundary_m,
        "lambda_boundary_distance_m": args.lambda_boundary_distance_m,
        "boundary_grid_size": args.boundary_grid_size,
        "boundary_distance_grid_size": args.boundary_distance_grid_size,
        "boundary_batch_size": args.boundary_batch_size,
        "boundary_near_low": args.boundary_near_low,
        "boundary_near_high": args.boundary_near_high,
        "boundary_far_low": args.boundary_far_low,
        "boundary_far_high": args.boundary_far_high,
        "boundary_distance_max": args.boundary_distance_max,
        "boundary_distance_eps": args.boundary_distance_eps,
        "support_only": args.support_only,
        "use_perturbation_loss": args.use_perturbation_loss,
        "perturb_std": args.perturb_std,
    })

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_cfg = vars(args).copy()
    run_cfg["device_resolved"] = device
    run_cfg["data_cfg"] = asdict(data_cfg)
    run_cfg["model_cfg"] = asdict(model_cfg)
    with open(out_dir / "config.json", "w") as f:
        json.dump(run_cfg, f, indent=2)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_l_m": [],
        "train_l_d": [],
        "train_l_g": [],
        "train_l_m_witness": [],
        "train_l_d_witness": [],
        "train_l_g_witness": [],
        "train_l_boundary_m": [],
        "train_l_boundary_distance_m": [],
        "mean_m_target": [],
        "mean_d_target": [],
        "mean_g_target": [],
        "mean_boundary_m_target": [],
        "mean_boundary_m_pred": [],
        "mean_m_distance_target": [],
        "mean_m_distance_pred": [],
        "num_boundary_points": [],
        "num_boundary_grid_points": [],
        "num_far_points": [],
        "boundary_batches_used": [],
        "val_nll": [],
        "val_acc": [],
    }
    if args.evidence_mode:
        history.update({
            "mean_order": [],
            "mean_abs_order": [],
            "mean_boundary": [],
            "mean_support": [],
            "mean_uncertainty": [],
            "mean_commitment": [],
            "train_l_evidence_conf": [],
            "train_l_boundary_evidence": [],
        })
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        # System B earns influence gradually.
        gate_scale = min(1.0, epoch / max(1, args.gate_warmup_epochs)) if is_ugly else 0.0

        loss_sums: Dict[str, float] = {}
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            losses = loss_for_batch(
                model,
                batch,
                device,
                args,
                gate_scale,
                args.disable_fusion,
                args.disable_uncertainty_loss,
                args.disable_support_loss,
                args.disable_primitive_attn,
                args.support_only,
                is_ugly,
                boundary_grid=boundary_grid,
                boundary_distance_grid=boundary_distance_grid,
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if is_ugly:
                model.update_target_encoder()

            for k, v in losses.items():
                loss_sums[k] = loss_sums.get(k, 0.0) + float(v.detach().cpu())
            n_batches += 1
            pbar.set_postfix({"loss": loss_sums["loss"] / n_batches})

        val_metrics = evaluate_model(
            model,
            val_loader,
            device,
            disable_fusion=args.disable_fusion,
            disable_primitive_attn=args.disable_primitive_attn,
            support_only=args.support_only,
        )
        if is_ugly:
            val_metrics.update(
                evaluate_ood(
                    model,
                    val_loader,
                    ood_val_loader,
                    device,
                    disable_fusion=args.disable_fusion,
                    disable_primitive_attn=args.disable_primitive_attn,
                    support_only=args.support_only,
                )
            )

        train_loss = loss_sums["loss"] / max(1, n_batches)
        train_l_m = loss_sums.get("l_m", 0.0) / max(1, n_batches)
        train_l_d = loss_sums.get("l_d", 0.0) / max(1, n_batches)
        train_l_g = loss_sums.get("l_g", 0.0) / max(1, n_batches)
        train_l_m_witness = loss_sums.get("l_m_witness", 0.0) / max(1, n_batches)
        train_l_d_witness = loss_sums.get("l_d_witness", 0.0) / max(1, n_batches)
        train_l_g_witness = loss_sums.get("l_g_witness", 0.0) / max(1, n_batches)
        train_l_boundary_m = loss_sums.get("l_boundary_m", 0.0) / max(1, n_batches)
        train_l_boundary_distance_m = loss_sums.get("l_boundary_distance_m", 0.0) / max(1, n_batches)
        train_l_evidence_conf = loss_sums.get("l_evidence_conf", 0.0) / max(1, n_batches)
        train_l_boundary_evidence = loss_sums.get("l_boundary_evidence", 0.0) / max(1, n_batches)
        mean_m_target = loss_sums.get("mean_m_target", 0.0) / max(1, n_batches)
        mean_d_target = loss_sums.get("mean_d_target", 0.0) / max(1, n_batches)
        mean_g_target = loss_sums.get("mean_g_target", 0.0) / max(1, n_batches)
        mean_order = loss_sums.get("mean_order", 0.0) / max(1, n_batches)
        mean_abs_order = loss_sums.get("mean_abs_order", 0.0) / max(1, n_batches)
        mean_boundary = loss_sums.get("mean_boundary", 0.0) / max(1, n_batches)
        mean_support = loss_sums.get("mean_support", 0.0) / max(1, n_batches)
        mean_uncertainty = loss_sums.get("mean_uncertainty", 0.0) / max(1, n_batches)
        mean_commitment = loss_sums.get("mean_commitment", 0.0) / max(1, n_batches)
        boundary_batches_used = loss_sums.get("boundary_batches_used", 0.0)
        if boundary_batches_used > 0:
            mean_boundary_m_target = loss_sums.get("mean_boundary_m_target", 0.0) / boundary_batches_used
            mean_boundary_m_pred = loss_sums.get("mean_boundary_m_pred", 0.0) / boundary_batches_used
            num_boundary_points = loss_sums.get("num_boundary_points", 0.0) / boundary_batches_used
            num_far_points = loss_sums.get("num_far_points", 0.0) / boundary_batches_used
        else:
            mean_boundary_m_target = 0.0
            mean_boundary_m_pred = 0.0
            num_boundary_points = 0.0
            num_far_points = 0.0
        boundary_distance_batches_used = loss_sums.get("boundary_distance_batches_used", 0.0)
        if boundary_distance_batches_used > 0:
            mean_m_distance_target = loss_sums.get("mean_m_distance_target", 0.0) / boundary_distance_batches_used
            mean_m_distance_pred = loss_sums.get("mean_m_distance_pred", 0.0) / boundary_distance_batches_used
            num_boundary_grid_points = loss_sums.get("num_boundary_grid_points", 0.0) / boundary_distance_batches_used
        else:
            mean_m_distance_target = 0.0
            mean_m_distance_pred = 0.0
            num_boundary_grid_points = 0.0
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_l_m"].append(train_l_m)
        history["train_l_d"].append(train_l_d)
        history["train_l_g"].append(train_l_g)
        history["train_l_m_witness"].append(train_l_m_witness)
        history["train_l_d_witness"].append(train_l_d_witness)
        history["train_l_g_witness"].append(train_l_g_witness)
        history["train_l_boundary_m"].append(train_l_boundary_m)
        history["train_l_boundary_distance_m"].append(train_l_boundary_distance_m)
        history["mean_m_target"].append(mean_m_target)
        history["mean_d_target"].append(mean_d_target)
        history["mean_g_target"].append(mean_g_target)
        history["mean_boundary_m_target"].append(mean_boundary_m_target)
        history["mean_boundary_m_pred"].append(mean_boundary_m_pred)
        history["mean_m_distance_target"].append(mean_m_distance_target)
        history["mean_m_distance_pred"].append(mean_m_distance_pred)
        history["num_boundary_points"].append(num_boundary_points)
        history["num_boundary_grid_points"].append(num_boundary_grid_points)
        history["num_far_points"].append(num_far_points)
        history["boundary_batches_used"].append(boundary_batches_used)
        history["val_nll"].append(val_metrics["nll"])
        history["val_acc"].append(val_metrics["accuracy"])
        if args.evidence_mode:
            history["mean_order"].append(mean_order)
            history["mean_abs_order"].append(mean_abs_order)
            history["mean_boundary"].append(mean_boundary)
            history["mean_support"].append(mean_support)
            history["mean_uncertainty"].append(mean_uncertainty)
            history["mean_commitment"].append(mean_commitment)
            history["train_l_evidence_conf"].append(train_l_evidence_conf)
            history["train_l_boundary_evidence"].append(train_l_boundary_evidence)

        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            line = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_l_m": train_l_m,
                "train_l_d": train_l_d,
                "train_l_g": train_l_g,
                "train_l_m_witness": train_l_m_witness,
                "train_l_d_witness": train_l_d_witness,
                "train_l_g_witness": train_l_g_witness,
                "train_l_boundary_m": train_l_boundary_m,
                "train_l_boundary_distance_m": train_l_boundary_distance_m,
                "mean_m_target": mean_m_target,
                "mean_d_target": mean_d_target,
                "mean_g_target": mean_g_target,
                "mean_boundary_m_target": mean_boundary_m_target,
                "mean_boundary_m_pred": mean_boundary_m_pred,
                "mean_m_distance_target": mean_m_distance_target,
                "mean_m_distance_pred": mean_m_distance_pred,
                "num_boundary_points": num_boundary_points,
                "num_boundary_grid_points": num_boundary_grid_points,
                "num_far_points": num_far_points,
                "boundary_batches_used": boundary_batches_used,
                "val_acc": val_metrics["accuracy"],
                "val_nll": val_metrics["nll"],
                "val_ece": val_metrics["ece"],
                "val_risk_auc": val_metrics["risk_coverage_auc"],
            }
            if args.evidence_mode:
                line.update({
                    "train_l_evidence_conf": train_l_evidence_conf,
                    "train_l_boundary_evidence": train_l_boundary_evidence,
                    "mean_order": mean_order,
                    "mean_abs_order": mean_abs_order,
                    "mean_boundary": mean_boundary,
                    "mean_support": mean_support,
                    "mean_uncertainty": mean_uncertainty,
                    "mean_commitment": mean_commitment,
                })
            if is_ugly:
                line.update({
                    "ood_auc_unc": val_metrics.get("ood_auroc_by_uncertainty"),
                    "ood_auc_support": val_metrics.get("ood_auroc_by_negative_support"),
                    "gate_scale": gate_scale,
                })
            print(json.dumps(line, indent=None))

        if val_metrics["nll"] < best_val:
            best_val = val_metrics["nll"]
            torch.save(model.state_dict(), out_dir / "model.pt")

    # Final metrics from best checkpoint.
    model.load_state_dict(torch.load(out_dir / "model.pt", map_location=device))
    test_metrics = evaluate_model(
        model,
        test_loader,
        device,
        disable_fusion=args.disable_fusion,
        disable_primitive_attn=args.disable_primitive_attn,
        support_only=args.support_only,
    )
    if is_ugly:
        test_metrics.update(
            evaluate_ood(
                model,
                test_loader,
                ood_test_loader,
                device,
                disable_fusion=args.disable_fusion,
                disable_primitive_attn=args.disable_primitive_attn,
                support_only=args.support_only,
            )
        )
        test_metrics.update(
            evaluate_boundary_distance_m_diagnostics(
                model,
                test_loader,
                boundary_distance_grid,
                grid_size=args.boundary_distance_grid_size,
                gate_scale=1.0,
                disable_fusion=args.disable_fusion,
                disable_primitive_attn=args.disable_primitive_attn,
                support_only=args.support_only,
                boundary_distance_max=args.boundary_distance_max,
                boundary_distance_eps=args.boundary_distance_eps,
            )
        )

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"test": test_metrics, "history": history}, f, indent=2)

    save_surfaces(
        model,
        train_ds,
        device,
        str(out_dir),
        disable_fusion=args.disable_fusion,
        disable_primitive_attn=args.disable_primitive_attn,
        support_only=args.support_only,
    )
    save_training_curves(history, str(out_dir))

    print("Final test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print(f"Saved outputs to: {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Train UGLY on a 2D uncertainty sanity test.")
    p.add_argument("--model", choices=["ugly", "baseline"], default="ugly")
    p.add_argument("--out", type=str, default="runs/ugly")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num-threads", type=int, default=1, help="CPU torch threads; use 1 for small toy runs to avoid overhead.")

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--eval-every", type=int, default=10)

    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--n-val", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--noise", type=float, default=0.10)
    p.add_argument("--ood-ratio-train", type=float, default=0.25)
    p.add_argument("--disable-fusion", action="store_true")
    p.add_argument("--disable-uncertainty-loss", action="store_true")
    p.add_argument("--disable-support-loss", action="store_true")
    p.add_argument("--disable-primitive-attn", action="store_true")
    p.add_argument("--evidence-mode", action="store_true")
    p.add_argument("--order-temp", type=float, default=6.0)
    p.add_argument("--lambda-evidence-coop", type=float, default=0.1)
    p.add_argument("--support-only", action="store_true")
    p.add_argument("--use-perturbation-loss", action="store_true")
    p.add_argument("--perturb-std", type=float, default=0.15)
    p.add_argument("--use-witness-mdg-loss", action="store_true", help="Train M/D/G with binary witness targets.")
    p.add_argument("--num-witnesses", type=int, default=8, help="Number of noisy witnesses per input.")
    p.add_argument("--witness-std", type=float, default=0.20, help="Stddev used to sample witness perturbations.")
    p.add_argument("--lambda-m", type=float, default=0.1)
    p.add_argument("--lambda-d", type=float, default=0.1)
    p.add_argument("--lambda-g", type=float, default=0.1)
    p.add_argument("--use-boundary-m-loss", action="store_true")
    p.add_argument("--lambda-boundary-m", type=float, default=0.1)
    p.add_argument("--boundary-grid-size", type=int, default=160)
    p.add_argument("--boundary-batch-size", type=int, default=256)
    p.add_argument("--boundary-near-low", type=float, default=0.45)
    p.add_argument("--boundary-near-high", type=float, default=0.55)
    p.add_argument("--boundary-far-low", type=float, default=0.05)
    p.add_argument("--boundary-far-high", type=float, default=0.95)
    p.add_argument("--use-boundary-distance-m-loss", action="store_true")
    p.add_argument("--lambda-boundary-distance-m", type=float, default=0.1)
    p.add_argument("--boundary-distance-grid-size", type=int, default=200)
    p.add_argument("--boundary-distance-max", type=float, default=2.0)
    p.add_argument("--boundary-distance-eps", type=float, default=1e-6)

    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--token-dim", type=int, default=64)
    p.add_argument("--num-views", type=int, default=6)
    p.add_argument("--num-prototypes", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.05)

    p.add_argument("--beta-unc", type=float, default=0.15)
    p.add_argument("--beta-support", type=float, default=0.15)
    p.add_argument("--beta-commit", type=float, default=0.03)
    p.add_argument("--beta-jepa", type=float, default=0.05)
    p.add_argument("--jepa-noise", type=float, default=0.05)
    p.add_argument("--ema-decay", type=float, default=0.99)
    p.add_argument("--no-jepa", action="store_true")
    p.add_argument("--gate-warmup-epochs", type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
