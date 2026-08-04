"""
Minimal training script for disturbance-profile uncertainty on vector/tabular data.

Expected CSV format:
    feature_1,feature_2,...,feature_N,label

Example:
    python train_disturbance_profile.py \
      --csv train.csv \
      --label-col label \
      --num-classes 3 \
      --latent-dim 64 \
      --epochs 20 \
      --stage train_all

Stages:
    train_base      Train only the base classifier. No profile risk loss.
    train_risk      Freeze base classifier, train profile aggregator.
    train_all       Train base + probes + aggregator jointly.

This is intentionally simple. For images/text/graph data, replace CSVDataset and
MLPLatentClassifier with your existing dataset/model, while keeping the probe
wrapper and loss.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, random_split

from uncertainty_profile import (
    DisturbanceProfileConfig,
    DisturbanceProfileLoss,
    DisturbanceProfileModel,
    LossConfig,
    MLPLatentClassifier,
    ProfileAggregator,
    build_default_probes,
    freeze_base_model,
)


class CSVDataset(Dataset):
    def __init__(self, path: str, label_col: str, feature_cols: Optional[List[str]] = None) -> None:
        df = pd.read_csv(path)
        if label_col not in df.columns:
            raise ValueError(f"label_col={label_col!r} not in CSV columns: {list(df.columns)}")
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != label_col]
        self.feature_cols = feature_cols
        self.label_col = label_col
        x = df[feature_cols].to_numpy(dtype=np.float32)
        y = df[label_col].to_numpy(dtype=np.int64)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.x[idx], self.y[idx]


@torch.no_grad()
def evaluate(model: DisturbanceProfileModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    n = 0
    correct = 0
    committed = 0
    committed_correct = 0
    risk_sum = 0.0
    profile_sum: Optional[Tensor] = None

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        out = model.predict_with_abstention(x)
        logits = out["clean_logits"]
        pred = (logits.squeeze(-1) > 0).long() if logits.shape[-1] == 1 else logits.argmax(dim=-1)
        commit = out["commit"]
        batch = y.shape[0]
        n += batch
        correct += (pred == y).sum().item()
        committed += commit.sum().item()
        committed_correct += ((pred == y) & commit).sum().item()
        risk_sum += out["risk_prob"].sum().item()
        p = out["profile"].detach().sum(dim=0).cpu()
        profile_sum = p if profile_sum is None else profile_sum + p

    accuracy = correct / max(n, 1)
    commit_rate = committed / max(n, 1)
    committed_accuracy = committed_correct / max(committed, 1)
    mean_risk = risk_sum / max(n, 1)
    metrics = {
        "accuracy": accuracy,
        "commit_rate": commit_rate,
        "committed_accuracy": committed_accuracy,
        "mean_risk": mean_risk,
    }
    if profile_sum is not None:
        profile_mean = profile_sum / max(n, 1)
        for name, val in zip(model.profile_names, profile_mean.tolist()):
            metrics[f"profile_mean/{name}"] = float(val)
    return metrics


def train_one_epoch(
    model: DisturbanceProfileModel,
    criterion: DisturbanceProfileLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    amp: bool = False,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    sums: Dict[str, float] = {}
    n_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type in {"cuda", "cpu"}):
            out = model(x)
            loss, metrics = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        n_batches += 1
        for k, v in metrics.items():
            sums[k] = sums.get(k, 0.0) + float(v)

    return {k: v / max(n_batches, 1) for k, v in sums.items()}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--label-col", default="label")
    p.add_argument("--feature-cols", nargs="*", default=None)
    p.add_argument("--num-classes", type=int, required=True)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--stage", choices=["train_base", "train_risk", "train_all"], default="train_all")
    p.add_argument("--include-learned-probe", action="store_true")
    p.add_argument("--uncertain-label", type=int, default=None)
    p.add_argument("--ignore-uncertain-for-task", action="store_true")
    p.add_argument("--risk-target-mode", choices=["incorrect", "uncertain_label", "incorrect_or_uncertain"], default="incorrect_or_uncertain")
    p.add_argument("--commit-risk-threshold", type=float, default=0.5)
    p.add_argument("--min-clean-margin", type=float, default=0.0)
    p.add_argument("--detach-profile-from-base", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--out-dir", default="runs/disturbance_profile")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = CSVDataset(args.csv, args.label_col, args.feature_cols)
    n_val = max(1, int(len(dataset) * args.val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = len(dataset.feature_cols)

    base = MLPLatentClassifier(
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
    )
    probes = build_default_probes(args.latent_dim, include_learned=args.include_learned_probe)
    model = DisturbanceProfileModel(
        base_model=base,
        probes=probes,
        aggregator=ProfileAggregator(hidden_dim=128, dropout=args.dropout),
        config=DisturbanceProfileConfig(
            detach_profile_from_base=args.detach_profile_from_base,
            commit_risk_threshold=args.commit_risk_threshold,
            min_clean_margin=args.min_clean_margin,
        ),
    ).to(device)

    if args.stage == "train_base":
        loss_cfg = LossConfig(task_weight=1.0, risk_weight=0.0)
    elif args.stage == "train_risk":
        freeze_base_model(model, freeze=True)
        loss_cfg = LossConfig(
            task_weight=0.0,
            risk_weight=1.0,
            uncertain_label=args.uncertain_label,
            ignore_uncertain_for_task=args.ignore_uncertain_for_task,
            risk_target_mode=args.risk_target_mode,
        )
    else:
        loss_cfg = LossConfig(
            task_weight=1.0,
            risk_weight=1.0,
            uncertain_label=args.uncertain_label,
            ignore_uncertain_for_task=args.ignore_uncertain_for_task,
            risk_target_mode=args.risk_target_mode,
        )

    criterion = DisturbanceProfileLoss(loss_cfg)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history: List[Dict[str, float]] = []
    best_committed_accuracy = -1.0
    best_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, criterion, train_loader, optimizer, device, amp=args.amp)
        val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, **{f"train/{k}": v for k, v in train_metrics.items()}, **{f"val/{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(json.dumps(row, indent=None, sort_keys=True))

        score = val_metrics["committed_accuracy"]
        if score > best_committed_accuracy:
            best_committed_accuracy = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "profile_names": model.profile_names,
                    "metrics": val_metrics,
                },
                best_path,
            )

    with (out_dir / "history.jsonl").open("w") as f:
        for row in history:
            f.write(json.dumps(row) + "\n")
    with (out_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved best checkpoint to {best_path}")


if __name__ == "__main__":
    main()
