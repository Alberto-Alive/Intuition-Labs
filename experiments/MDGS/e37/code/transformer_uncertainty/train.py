from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import SeqTaskConfig, build_datasets
from .metrics import evaluate_id, evaluate_ood, reliability_score
from .models import TransformerUncertaintyConfig, build_model


VARIANTS = [
    "baseline",
    "aux",
    "aux_no_mdgs",
    "aux_random_mdgs",
    "aux_shuffled_mdgs",
    "aux_detached_mdgs",
    "aux_frozen_mdgs",
    "gated",
    "shared",
    "gated_anchor",
    "shared_anchor",
    "shared_anchor_no_mdgs",
    "shared_anchor_random_mdgs",
    "shared_anchor_shuffled_mdgs",
    "shared_anchor_detached_mdgs",
    "shared_anchor_frozen_mdgs",
]

ANCHOR_VARIANTS = {
    "gated_anchor",
    "shared_anchor",
    "shared_anchor_no_mdgs",
    "shared_anchor_random_mdgs",
    "shared_anchor_shuffled_mdgs",
    "shared_anchor_detached_mdgs",
    "shared_anchor_frozen_mdgs",
}

NO_AUX_LOSS_VARIANTS = {"aux_no_mdgs"}
RANDOM_AUX_TARGET_VARIANTS = {"aux_random_mdgs"}
SHUFFLED_AUX_TARGET_VARIANTS = {"aux_shuffled_mdgs"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def _to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _aux_target(
    batch: Dict[str, torch.Tensor],
    key: str,
    variant: str,
    shuffle_order: torch.Tensor | None = None,
) -> torch.Tensor:
    target = batch[key]
    if variant in RANDOM_AUX_TARGET_VARIANTS:
        return torch.rand_like(target)
    if variant in SHUFFLED_AUX_TARGET_VARIANTS and shuffle_order is not None:
        return target[shuffle_order]
    return target


def _aux_view_labels(y: torch.Tensor, variant: str, num_classes: int) -> torch.Tensor:
    if variant in RANDOM_AUX_TARGET_VARIANTS:
        return torch.randint(0, num_classes, y.shape, device=y.device)
    if variant in SHUFFLED_AUX_TARGET_VARIANTS and y.shape[0] > 1:
        order = torch.randperm(y.shape[0], device=y.device)
        return y[order]
    return y


def loss_for_batch(model, batch: Dict[str, torch.Tensor], variant: str, aux_weight_scale: float) -> Dict[str, torch.Tensor]:
    x = batch["x"]
    y = batch["y"]
    is_ood = batch["is_ood"]
    out = model(x)

    in_mask = y >= 0
    pred_loss = torch.tensor(0.0, device=x.device)
    per_ce = torch.zeros(x.shape[0], device=x.device)
    if in_mask.any():
        per_ce[in_mask] = F.cross_entropy(out["logits"][in_mask], y[in_mask], reduction="none")
        pred_loss = per_ce[in_mask].mean()

    total = pred_loss
    losses: Dict[str, torch.Tensor] = {"loss": total, "pred": pred_loss}

    if variant != "baseline" and variant not in NO_AUX_LOSS_VARIANTS:
        anchor_loss = torch.tensor(0.0, device=x.device)
        if variant in ANCHOR_VARIANTS and in_mask.any() and "base_logits" in out:
            anchor_loss = F.cross_entropy(out["base_logits"][in_mask], y[in_mask])

        shuffle_order = None
        if variant in SHUFFLED_AUX_TARGET_VARIANTS and x.shape[0] > 1:
            shuffle_order = torch.randperm(x.shape[0], device=x.device)

        m_loss = F.mse_loss(out["m_scalar"], _aux_target(batch, "m_target", variant, shuffle_order))
        d_loss = F.mse_loss(out["d_scalar"], _aux_target(batch, "d_target", variant, shuffle_order))
        g_loss = F.mse_loss(out["g_scalar"], _aux_target(batch, "g_target", variant, shuffle_order))
        unc_loss = F.mse_loss(out["uncertainty"], _aux_target(batch, "unc_target", variant, shuffle_order))
        support_loss = F.binary_cross_entropy(
            out["support"].clamp(1e-5, 1 - 1e-5),
            _aux_target(batch, "s_target", variant, shuffle_order),
        )
        commit_loss = F.binary_cross_entropy(
            out["commitment"].clamp(1e-5, 1 - 1e-5),
            _aux_target(batch, "commit_target", variant, shuffle_order),
        )
        view_loss = torch.tensor(0.0, device=x.device)
        if in_mask.any() and "view_logits" in out:
            view_logits = out["view_logits"][in_mask]
            view_y = _aux_view_labels(y[in_mask], variant, view_logits.shape[-1])
            view_y = view_y.unsqueeze(1).expand(-1, view_logits.shape[1]).reshape(-1)
            view_loss = F.cross_entropy(view_logits.reshape(-1, view_logits.shape[-1]), view_y)

        aux_loss = (
            0.12 * m_loss
            + 0.12 * d_loss
            + 0.12 * g_loss
            + 0.18 * unc_loss
            + 0.18 * support_loss
            + 0.05 * commit_loss
            + 0.10 * view_loss
        )
        total = pred_loss + 0.25 * anchor_loss + aux_weight_scale * aux_loss
        losses.update(
            {
                "loss": total,
                "anchor": anchor_loss.detach(),
                "m": m_loss.detach(),
                "d": d_loss.detach(),
                "g": g_loss.detach(),
                "unc": unc_loss.detach(),
                "support": support_loss.detach(),
                "commit": commit_loss.detach(),
                "view": view_loss.detach(),
            }
        )

    with torch.no_grad():
        if in_mask.any():
            pred = out["logits"][in_mask].argmax(dim=-1)
            losses["train_acc"] = (pred == y[in_mask]).float().mean()
        else:
            losses["train_acc"] = torch.tensor(0.0, device=x.device)
        losses["mean_uncertainty"] = out["uncertainty"].mean()
        losses["mean_support"] = out["support"].mean()

    return losses


def train_one(
    variant: str,
    args,
    data_cfg: SeqTaskConfig,
    model_cfg: TransformerUncertaintyConfig,
    out_dir: Path,
) -> Dict[str, float]:
    set_seed(args.seed)
    device = choose_device(args.device)
    train_ds, val_ds, test_ds, ood_val_ds, ood_test_ds = build_datasets(data_cfg)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    ood_val_loader = DataLoader(ood_val_ds, batch_size=args.batch_size)
    ood_test_loader = DataLoader(ood_test_ds, batch_size=args.batch_size)

    model = build_model(variant, model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -1e9
    best_path = out_dir / f"{variant}.pt"
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums: Dict[str, float] = {}
        n_batches = 0
        aux_scale = min(1.0, epoch / max(1, args.aux_warmup_epochs))
        for batch in tqdm(train_loader, desc=f"{variant} epoch {epoch}", leave=False):
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            losses = loss_for_batch(model, batch, variant, aux_scale)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            n_batches += 1
            for key, value in losses.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach().cpu())

        val_metrics = evaluate_id(model, val_loader, device)
        val_metrics.update(evaluate_ood(model, val_loader, ood_val_loader, device))
        score = reliability_score(val_metrics)
        row = {
            "epoch": float(epoch),
            "score": float(score),
            **{f"train_{k}": v / max(1, n_batches) for k, v in sums.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            json.dumps(
                {
                    "model": variant,
                    "epoch": epoch,
                    "train_loss": row["train_loss"],
                    "val_acc": val_metrics["accuracy"],
                    "val_nll": val_metrics["nll"],
                    "val_ood_auc_unc": val_metrics["ood_auroc_by_uncertainty"],
                    "val_ood_auc_support": val_metrics["ood_auroc_by_negative_support"],
                    "score": score,
                }
            )
        )
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate_id(model, test_loader, device)
    test_metrics.update(evaluate_ood(model, test_loader, ood_test_loader, device))
    test_metrics["score"] = reliability_score(test_metrics)

    with open(out_dir / f"{variant}_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"variant": variant, "test": test_metrics, "history": history}, f, indent=2)

    return test_metrics


def write_architecture_docs(out_dir: Path) -> None:
    arch = """# Transformer MDGS Architecture Iterations

This folder tests whether MDGS uncertainty can improve transformer training and prediction.

## Task

The synthetic task is sequence classification. The first token chooses which evidence channel matters. The inactive channel deliberately contains distractor evidence, so the transformer must use attention rather than simple token counting.

The dataset also supplies fast reliability targets:

- `M`: decisive margin of the active evidence.
- `D`: conflict/disagreement in the active evidence.
- `G`: fragility/instability of the example.
- `S`: support/knownness. Synthetic OOD sequences have low support.

## Iteration 0: `baseline`

Normal transformer encoder:

```text
tokens -> transformer -> CLS -> class logits
```

Uncertainty is just `1 - softmax confidence`. This is the control.

## Iteration 1: `aux`

MDGS is extracted from the same encoded memory using witness queries:

```text
tokens -> transformer memory
CLS -> prediction
witness queries attend to memory -> MDGS -> auxiliary losses
```

Prediction does not directly consume MDGS. This tests MDGS as a training teacher.

## Iteration 2: `gated`

MDGS context is fused into the prediction state:

```text
CLS + gate(CLS, MDGS) * adapter(MDGS) -> final logits
```

This tests direct reliability-conditioned prediction.

## Iteration 3: `shared`

Prediction and reliability tokens share a final transformer attention block:

```text
CLS + MDGS tokens + content tokens -> shared refiner attention -> final CLS -> logits
```

This is the closest implementation of uncertainty-aided attention: the final answer is made after content and reliability tokens attend together.

## Iteration 4: `shared_anchor`

The same shared-attention model, but the base CLS head also receives a small prediction loss:

```text
base CLS -> auxiliary class loss
CLS + MDGS tokens + content tokens -> shared refiner -> final class loss
```

This keeps the stable "MDGS as training teacher" behavior while still making the final prediction use the reliability-aware shared attention path.

## Selection Rule

The quick sweep ranks models by a reliability-oriented score that rewards:

- high ID accuracy
- lower NLL
- lower risk-coverage AUC
- fewer confident wrong predictions
- high OOD AUROC from uncertainty/support

This score is only for fast iteration. The detailed metrics table is the real comparison.
"""
    (out_dir / "ARCHITECTURES.md").write_text(arch, encoding="utf-8")


def write_summary(out_dir: Path, results: Dict[str, Dict[str, float]], args, data_cfg, model_cfg) -> None:
    ordered = sorted(results.items(), key=lambda kv: kv[1]["score"], reverse=True)
    best_name, best_metrics = ordered[0]

    headers = [
        "model",
        "score",
        "accuracy",
        "nll",
        "ece",
        "conf_wrong90",
        "risk_auc",
        "ood_auc_unc",
        "ood_auc_support",
        "id_unc",
        "ood_unc",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for name, metrics in ordered:
        vals = [
            name,
            f"{metrics.get('score', float('nan')):.4f}",
            f"{metrics.get('accuracy', float('nan')):.4f}",
            f"{metrics.get('nll', float('nan')):.4f}",
            f"{metrics.get('ece', float('nan')):.4f}",
            f"{metrics.get('confident_wrong_90', float('nan')):.4f}",
            f"{metrics.get('risk_coverage_auc', float('nan')):.4f}",
            f"{metrics.get('ood_auroc_by_uncertainty', float('nan')):.4f}",
            f"{metrics.get('ood_auroc_by_negative_support', float('nan')):.4f}",
            f"{metrics.get('id_mean_uncertainty', float('nan')):.4f}",
            f"{metrics.get('ood_mean_uncertainty', float('nan')):.4f}",
        ]
        lines.append("| " + " | ".join(vals) + " |")

    text = f"""# MDGS Transformer Quick Results

Short training comparison generated by `python -m transformer_uncertainty.train`.

## Run Config

```json
{json.dumps({"args": vars(args), "data": asdict(data_cfg), "model": asdict(model_cfg)}, indent=2)}
```

## Results

{chr(10).join(lines)}

## Current Best

`{best_name}` is the best quick-sweep architecture by the reliability score.

Interpretation:

- `baseline` shows what plain transformer attention does when uncertainty is just output confidence.
- `aux` tests whether MDGS improves the shared representation as a training teacher.
- `gated` tests whether MDGS should directly modify the prediction state.
- `shared` tests the intended idea most directly: reliability tokens and content tokens use the same final attention path before prediction.
- `shared_anchor` tests the stabilized version: shared final attention plus a small base prediction anchor.

Best model metrics:

```json
{json.dumps(best_metrics, indent=2)}
```

## Recommendation

Use the best-scoring architecture as the next base, then rerun with more seeds and longer training before treating the result as stable. This file is a fast architecture selection pass, not a final benchmark.
"""
    (out_dir / "RESULTS_SUMMARY.md").write_text(text, encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="Quick MDGS transformer architecture sweep.")
    p.add_argument("--model", choices=VARIANTS + ["all"], default="all")
    p.add_argument("--out", type=str, default="transformer_uncertainty/runs/quick")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--aux-warmup-epochs", type=int, default=2)

    p.add_argument("--n-train", type=int, default=1200)
    p.add_argument("--n-val", type=int, default=400)
    p.add_argument("--n-test", type=int, default=400)
    p.add_argument("--ood-ratio-train", type=float, default=0.25)
    p.add_argument("--ood-mode", choices=["easy", "medium", "hard", "support_mismatch"], default="easy")
    p.add_argument("--seq-len", type=int, default=18)
    p.add_argument("--vocab-size", type=int, default=64)

    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-refine-layers", type=int, default=1)
    p.add_argument("--dim-feedforward", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--num-views", type=int, default=6)
    p.add_argument("--num-prototypes", type=int, default=16)
    return p.parse_args()


def main(args=None):
    args = parse_args() if args is None else args
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_architecture_docs(out_dir)

    data_cfg = SeqTaskConfig(
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        ood_ratio_train=args.ood_ratio_train,
        seed=args.seed,
        ood_mode=args.ood_mode,
    )
    model_cfg = TransformerUncertaintyConfig(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        num_refine_layers=args.num_refine_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_views=args.num_views,
        num_prototypes=args.num_prototypes,
    )

    variants = VARIANTS if args.model == "all" else [args.model]
    results: Dict[str, Dict[str, float]] = {}
    for variant in variants:
        print(f"Running {variant}")
        results[variant] = train_one(variant, args, data_cfg, model_cfg, out_dir)

    with open(out_dir / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    write_summary(out_dir, results, args, data_cfg, model_cfg)
    print(f"Saved comparison to {out_dir / 'RESULTS_SUMMARY.md'}")


if __name__ == "__main__":
    main()
