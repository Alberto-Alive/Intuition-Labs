from __future__ import annotations

import argparse
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
from .metrics import evaluate_model, evaluate_ood
from .model import BaselineMLP, ModelConfig, UGLYNet
from .plot import save_surfaces, save_training_curves


def choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


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
) -> Dict[str, torch.Tensor]:
    x, y, is_ood = batch
    x = x.to(device)
    y = y.to(device)
    is_ood = is_ood.to(device)

    x_tgt = None
    if is_ugly:
        # JEPA-style latent target uses a slightly perturbed view.
        x_tgt = x + torch.randn_like(x) * cfg.jepa_noise

    out = model(
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
        per_ce[in_mask] = F.cross_entropy(out["logits"][in_mask], y[in_mask], reduction="none")
        pred_loss = per_ce[in_mask].mean()
    else:
        pred_loss = torch.tensor(0.0, device=device)

    total = pred_loss
    losses = {"loss": total, "pred": pred_loss}

    if is_ugly:
        losses.update({
            "gate_mean": out["gate"].mean().detach(),
            "unc_mean": out["uncertainty"].mean().detach(),
            "support_mean": out["support"].mean().detach(),
        })

        if not disable_uncertainty_loss:
            # U target means estimated distance-to-truth.
            # In-distribution: normalized CE. OOD: high uncertainty.
            unc_target = torch.where(
                in_mask,
                1.0 - torch.exp(-per_ce.detach()).clamp(0, 1),
                torch.ones_like(per_ce),
            )
            unc_loss = F.mse_loss(out["uncertainty"], unc_target)

            # Support target: in-distribution high, synthetic OOD low.
            support_target = 1.0 - is_ood
            support_loss = F.binary_cross_entropy(out["support"].clamp(1e-5, 1 - 1e-5), support_target)

            # Commitment target: commit if in-distribution and current prediction is correct.
            # Small weight; this is a behavioral regularizer, not the main objective.
            with torch.no_grad():
                pred = out["logits"].argmax(dim=-1)
                correct = (in_mask & (pred == y)).float()
            commit_loss = F.binary_cross_entropy(out["commitment"].clamp(1e-5, 1 - 1e-5), correct)

            jepa_loss = torch.tensor(0.0, device=device)
            if "jepa_pred" in out:
                jepa_loss = F.mse_loss(out["jepa_pred"], out["jepa_target"])

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

    return losses


def train(args):
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    device = choose_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    )
    is_ugly = args.model == "ugly"
    model = UGLYNet(model_cfg) if is_ugly else BaselineMLP(model_cfg)
    model.to(device)
    print({
        "disable_fusion": args.disable_fusion,
        "disable_primitive_attn": args.disable_primitive_attn,
        "disable_uncertainty_loss": args.disable_uncertainty_loss,
        "disable_support_loss": args.disable_support_loss,
        "support_only": args.support_only,
    })

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_cfg = vars(args).copy()
    run_cfg["device_resolved"] = device
    run_cfg["data_cfg"] = asdict(data_cfg)
    run_cfg["model_cfg"] = asdict(model_cfg)
    with open(out_dir / "config.json", "w") as f:
        json.dump(run_cfg, f, indent=2)

    history = {"epoch": [], "train_loss": [], "val_nll": [], "val_acc": []}
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
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_nll"].append(val_metrics["nll"])
        history["val_acc"].append(val_metrics["accuracy"])

        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            line = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_acc": val_metrics["accuracy"],
                "val_nll": val_metrics["nll"],
                "val_ece": val_metrics["ece"],
                "val_risk_auc": val_metrics["risk_coverage_auc"],
            }
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
    p.add_argument("--support-only", action="store_true")

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
