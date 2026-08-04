from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def binary_auroc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """AUROC where higher score means positive.

    Uses rank statistic; no sklearn required.
    """
    scores = np.concatenate([scores_pos, scores_neg])
    labels = np.concatenate([np.ones_like(scores_pos), np.zeros_like(scores_neg)])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = ranks[labels == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


@torch.no_grad()
def ece(probs: torch.Tensor, y: torch.Tensor, n_bins: int = 15) -> float:
    conf, pred = probs.max(dim=-1)
    correct = (pred == y).float()
    bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    total = torch.tensor(0.0, device=probs.device)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            total += mask.float().mean() * (conf[mask].mean() - correct[mask].mean()).abs()
    return float(total.cpu())


@torch.no_grad()
def risk_coverage_auc(commitment: torch.Tensor, correct: torch.Tensor) -> Tuple[float, Dict[str, float]]:
    """Area under risk-coverage curve. Lower is better.

    Sort by commitment descending. At each coverage, risk = error among accepted examples.
    """
    order = torch.argsort(commitment, descending=True)
    correct_sorted = correct.float()[order]
    n = len(correct_sorted)
    coverages = torch.arange(1, n + 1, device=commitment.device).float() / n
    risks = 1.0 - torch.cumsum(correct_sorted, dim=0) / torch.arange(1, n + 1, device=commitment.device).float()
    auc = torch.trapz(risks, coverages).item()
    summary = {}
    for cov in [0.5, 0.8, 0.9, 1.0]:
        idx = max(0, min(n - 1, int(cov * n) - 1))
        summary[f"risk_at_{int(cov*100)}cov"] = float(risks[idx].cpu())
    return float(auc), summary


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device: str,
    disable_fusion: bool = False,
    disable_primitive_attn: bool = False,
    support_only: bool = False,
) -> Dict[str, float]:
    model.eval()
    all_logits = []
    all_y = []
    all_unc = []
    all_sup = []
    all_com = []
    all_gate = []
    for x, y, is_ood in loader:
        x, y = x.to(device), y.to(device)
        mask = y >= 0
        if not mask.any():
            continue
        out = model(
            x,
            disable_fusion=disable_fusion,
            disable_primitive_attn=disable_primitive_attn,
            support_only=support_only,
        )
        all_logits.append(out["logits"][mask].detach().cpu())
        all_y.append(y[mask].detach().cpu())
        all_unc.append(out["uncertainty"][mask].detach().cpu())
        all_sup.append(out["support"][mask].detach().cpu())
        all_com.append(out["commitment"][mask].detach().cpu())
        all_gate.append(out["gate"][mask].detach().cpu())

    logits = torch.cat(all_logits)
    y = torch.cat(all_y)
    unc = torch.cat(all_unc)
    sup = torch.cat(all_sup)
    com = torch.cat(all_com)
    gate = torch.cat(all_gate)
    probs = F.softmax(logits, dim=-1)
    pred = probs.argmax(dim=-1)
    correct = pred == y
    conf = probs.max(dim=-1).values

    rauc, rsummary = risk_coverage_auc(com, correct)
    metrics = {
        "accuracy": float(correct.float().mean()),
        "nll": float(F.cross_entropy(logits, y)),
        "ece": ece(probs, y),
        "mean_confidence": float(conf.mean()),
        "mean_uncertainty": float(unc.mean()),
        "mean_support": float(sup.mean()),
        "mean_commitment": float(com.mean()),
        "mean_gate": float(gate.mean()),
        "confident_wrong_90": float(((conf > 0.90) & (~correct)).float().mean()),
        "risk_coverage_auc": rauc,
        **rsummary,
    }
    return metrics


@torch.no_grad()
def evaluate_ood(
    model,
    id_loader,
    ood_loader,
    device: str,
    disable_fusion: bool = False,
    disable_primitive_attn: bool = False,
    support_only: bool = False,
) -> Dict[str, float]:
    model.eval()
    id_unc, ood_unc = [], []
    id_sup, ood_sup = [], []
    id_com, ood_com = [], []

    for loader, unc_list, sup_list, com_list in [
        (id_loader, id_unc, id_sup, id_com),
        (ood_loader, ood_unc, ood_sup, ood_com),
    ]:
        for x, y, is_ood in loader:
            x = x.to(device)
            out = model(
                x,
                disable_fusion=disable_fusion,
                disable_primitive_attn=disable_primitive_attn,
                support_only=support_only,
            )
            unc_list.append(out["uncertainty"].detach().cpu())
            sup_list.append(out["support"].detach().cpu())
            com_list.append(out["commitment"].detach().cpu())

    id_unc = torch.cat(id_unc).numpy()
    ood_unc = torch.cat(ood_unc).numpy()
    id_sup = torch.cat(id_sup).numpy()
    ood_sup = torch.cat(ood_sup).numpy()
    id_com = torch.cat(id_com).numpy()
    ood_com = torch.cat(ood_com).numpy()

    return {
        "ood_auroc_by_uncertainty": binary_auroc(ood_unc, id_unc),
        "ood_auroc_by_negative_support": binary_auroc(-ood_sup, -id_sup),
        "ood_mean_uncertainty": float(np.mean(ood_unc)),
        "id_mean_uncertainty": float(np.mean(id_unc)),
        "ood_mean_support": float(np.mean(ood_sup)),
        "id_mean_support": float(np.mean(id_sup)),
        "ood_mean_commitment": float(np.mean(ood_com)),
        "id_mean_commitment": float(np.mean(id_com)),
    }
