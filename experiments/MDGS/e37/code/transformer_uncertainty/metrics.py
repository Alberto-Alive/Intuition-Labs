from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def binary_auroc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    scores = np.concatenate([scores_pos, scores_neg])
    labels = np.concatenate([np.ones_like(scores_pos), np.zeros_like(scores_neg)])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = ranks[labels == 1].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ece(probs: torch.Tensor, y: torch.Tensor, n_bins: int = 15) -> float:
    conf, pred = probs.max(dim=-1)
    correct = (pred == y).float()
    bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    total = torch.tensor(0.0, device=probs.device)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        if mask.any():
            total = total + mask.float().mean() * (conf[mask].mean() - correct[mask].mean()).abs()
    return float(total.cpu())


def brier_score(probs: torch.Tensor, y: torch.Tensor) -> float:
    target = F.one_hot(y, num_classes=probs.shape[-1]).float()
    return float((probs - target).pow(2).sum(dim=-1).mean().cpu())


def error_auroc_by_uncertainty(uncertainty: torch.Tensor, correct: torch.Tensor) -> float:
    wrong_scores = uncertainty[~correct].detach().cpu().numpy()
    correct_scores = uncertainty[correct].detach().cpu().numpy()
    if len(wrong_scores) == 0 or len(correct_scores) == 0:
        return float("nan")
    return binary_auroc(wrong_scores, correct_scores)


def selective_accuracy_by_uncertainty(
    uncertainty: torch.Tensor,
    correct: torch.Tensor,
    keep_fraction: float,
) -> float:
    n = len(correct)
    if n == 0:
        return float("nan")
    keep = max(1, min(n, int(round(keep_fraction * n))))
    order = torch.argsort(uncertainty, descending=False)
    kept = correct.float()[order[:keep]]
    return float(kept.mean().cpu())


def risk_coverage_auc(commitment: torch.Tensor, correct: torch.Tensor) -> Tuple[float, Dict[str, float]]:
    order = torch.argsort(commitment, descending=True)
    correct_sorted = correct.float()[order]
    n = len(correct_sorted)
    coverages = torch.arange(1, n + 1, device=commitment.device).float() / n
    risks = 1.0 - torch.cumsum(correct_sorted, dim=0) / torch.arange(1, n + 1, device=commitment.device).float()
    auc = float(torch.trapz(risks, coverages).cpu())
    summary: Dict[str, float] = {}
    for cov in [0.5, 0.8, 0.9, 1.0]:
        idx = max(0, min(n - 1, int(cov * n) - 1))
        summary[f"risk_at_{int(cov * 100)}cov"] = float(risks[idx].cpu())
    return auc, summary


@torch.no_grad()
def evaluate_id(model, loader, device: str) -> Dict[str, float]:
    model.eval()
    logits_all = []
    y_all = []
    unc_all = []
    sup_all = []
    com_all = []
    m_all = []
    d_all = []
    g_all = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        mask = y >= 0
        if not mask.any():
            continue
        out = model(x)
        logits_all.append(out["logits"][mask].cpu())
        y_all.append(y[mask].cpu())
        unc_all.append(out["uncertainty"][mask].cpu())
        sup_all.append(out["support"][mask].cpu())
        com_all.append(out["commitment"][mask].cpu())
        if "m_scalar" in out:
            m_all.append(out["m_scalar"][mask].cpu())
            d_all.append(out["d_scalar"][mask].cpu())
            g_all.append(out["g_scalar"][mask].cpu())

    logits = torch.cat(logits_all)
    y = torch.cat(y_all)
    uncertainty = torch.cat(unc_all)
    support = torch.cat(sup_all)
    commitment = torch.cat(com_all)
    probs = F.softmax(logits, dim=-1)
    pred = probs.argmax(dim=-1)
    conf = probs.max(dim=-1).values
    correct = pred == y
    rauc, risk_summary = risk_coverage_auc(commitment, correct)

    metrics: Dict[str, float] = {
        "accuracy": float(correct.float().mean()),
        "nll": float(F.cross_entropy(logits, y)),
        "ece": ece(probs, y),
        "brier": brier_score(probs, y),
        "mean_confidence": float(conf.mean()),
        "mean_uncertainty": float(uncertainty.mean()),
        "mean_support": float(support.mean()),
        "mean_commitment": float(commitment.mean()),
        "confident_wrong_90": float(((conf > 0.90) & (~correct)).float().mean()),
        "risk_coverage_auc": rauc,
        "error_auroc_by_uncertainty": error_auroc_by_uncertainty(uncertainty, correct),
        "selective_acc_reject10_uncertainty": selective_accuracy_by_uncertainty(uncertainty, correct, 0.90),
        "selective_acc_reject20_uncertainty": selective_accuracy_by_uncertainty(uncertainty, correct, 0.80),
        "selective_acc_reject50_uncertainty": selective_accuracy_by_uncertainty(uncertainty, correct, 0.50),
        **risk_summary,
    }
    if m_all:
        metrics.update(
            {
                "mean_m_scalar": float(torch.cat(m_all).mean()),
                "mean_d_scalar": float(torch.cat(d_all).mean()),
                "mean_g_scalar": float(torch.cat(g_all).mean()),
            }
        )
    return metrics


def _finite_mean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(values.float().mean().cpu())


@torch.no_grad()
def evaluate_attention_behavior(model, id_loader, ood_loader, device: str) -> Dict[str, float]:
    """Summarize final shared-refiner attention to MDGS token slots.

    Shared-anchor variants expose `refiner_attn` with shape [batch, heads, query, key].
    Token indices are: 0=CLS, 1:5=MDGS primitive tokens, 5:=content tokens.
    """

    model.eval()

    def collect(loader, expect_labels: bool):
        mdgs_vals = []
        content_vals = []
        cls_vals = []
        uncertainty_vals = []
        support_vals = []
        correct_vals = []
        for batch in loader:
            x = batch["x"].to(device)
            try:
                out = model(x, return_attention=True)
            except TypeError:
                return None
            attn = out.get("refiner_attn")
            if attn is None:
                return None
            if attn.dim() == 3:
                cls_attn = attn[:, 0, :]
            else:
                cls_attn = attn.mean(dim=1)[:, 0, :]
            mdgs_vals.append(cls_attn[:, 1:5].sum(dim=-1).detach().cpu())
            content_vals.append(cls_attn[:, 5:].sum(dim=-1).detach().cpu())
            cls_vals.append(cls_attn[:, 0].detach().cpu())
            uncertainty_vals.append(out["uncertainty"].detach().cpu())
            support_vals.append(out["support"].detach().cpu())
            if expect_labels:
                y = batch["y"].to(device)
                logits = out["logits"]
                mask = y >= 0
                if mask.any():
                    correct_vals.append((logits[mask].argmax(dim=-1) == y[mask]).detach().cpu())

        if not mdgs_vals:
            return None
        result = {
            "mdgs": torch.cat(mdgs_vals),
            "content": torch.cat(content_vals),
            "cls": torch.cat(cls_vals),
            "uncertainty": torch.cat(uncertainty_vals),
            "support": torch.cat(support_vals),
        }
        if correct_vals:
            result["correct"] = torch.cat(correct_vals)
        return result

    id_stats = collect(id_loader, True)
    if id_stats is None:
        return {}
    ood_stats = collect(ood_loader, False)
    if ood_stats is None:
        return {}

    metrics: Dict[str, float] = {
        "id_cls_attn_to_mdgs": _finite_mean(id_stats["mdgs"]),
        "id_cls_attn_to_content": _finite_mean(id_stats["content"]),
        "id_cls_attn_to_cls": _finite_mean(id_stats["cls"]),
        "ood_cls_attn_to_mdgs": _finite_mean(ood_stats["mdgs"]),
        "ood_cls_attn_to_content": _finite_mean(ood_stats["content"]),
        "ood_cls_attn_to_cls": _finite_mean(ood_stats["cls"]),
    }
    metrics["ood_minus_id_cls_attn_to_mdgs"] = metrics["ood_cls_attn_to_mdgs"] - metrics["id_cls_attn_to_mdgs"]

    correct = id_stats.get("correct")
    if correct is not None and len(correct) == len(id_stats["mdgs"]):
        metrics["id_correct_cls_attn_to_mdgs"] = _finite_mean(id_stats["mdgs"][correct])
        metrics["id_wrong_cls_attn_to_mdgs"] = _finite_mean(id_stats["mdgs"][~correct])
        metrics["id_wrong_minus_correct_cls_attn_to_mdgs"] = (
            metrics["id_wrong_cls_attn_to_mdgs"] - metrics["id_correct_cls_attn_to_mdgs"]
        )

    for prefix, stats in [("id", id_stats), ("ood", ood_stats)]:
        unc = stats["uncertainty"]
        sup = stats["support"]
        high_unc = unc >= torch.quantile(unc, 0.80)
        low_unc = unc <= torch.quantile(unc, 0.20)
        low_support = sup <= torch.quantile(sup, 0.20)
        high_support = sup >= torch.quantile(sup, 0.80)
        metrics[f"{prefix}_high_unc_cls_attn_to_mdgs"] = _finite_mean(stats["mdgs"][high_unc])
        metrics[f"{prefix}_low_unc_cls_attn_to_mdgs"] = _finite_mean(stats["mdgs"][low_unc])
        metrics[f"{prefix}_high_minus_low_unc_cls_attn_to_mdgs"] = (
            metrics[f"{prefix}_high_unc_cls_attn_to_mdgs"] - metrics[f"{prefix}_low_unc_cls_attn_to_mdgs"]
        )
        metrics[f"{prefix}_low_support_cls_attn_to_mdgs"] = _finite_mean(stats["mdgs"][low_support])
        metrics[f"{prefix}_high_support_cls_attn_to_mdgs"] = _finite_mean(stats["mdgs"][high_support])
        metrics[f"{prefix}_low_minus_high_support_cls_attn_to_mdgs"] = (
            metrics[f"{prefix}_low_support_cls_attn_to_mdgs"] - metrics[f"{prefix}_high_support_cls_attn_to_mdgs"]
        )

    return metrics


@torch.no_grad()
def evaluate_ood(model, id_loader, ood_loader, device: str) -> Dict[str, float]:
    model.eval()
    id_unc, ood_unc = [], []
    id_sup, ood_sup = [], []
    for loader, unc_list, sup_list in [
        (id_loader, id_unc, id_sup),
        (ood_loader, ood_unc, ood_sup),
    ]:
        for batch in loader:
            out = model(batch["x"].to(device))
            unc_list.append(out["uncertainty"].detach().cpu())
            sup_list.append(out["support"].detach().cpu())

    id_unc_np = torch.cat(id_unc).numpy()
    ood_unc_np = torch.cat(ood_unc).numpy()
    id_sup_np = torch.cat(id_sup).numpy()
    ood_sup_np = torch.cat(ood_sup).numpy()
    return {
        "ood_auroc_by_uncertainty": binary_auroc(ood_unc_np, id_unc_np),
        "ood_auroc_by_negative_support": binary_auroc(-ood_sup_np, -id_sup_np),
        "id_mean_uncertainty": float(np.mean(id_unc_np)),
        "ood_mean_uncertainty": float(np.mean(ood_unc_np)),
        "id_mean_support": float(np.mean(id_sup_np)),
        "ood_mean_support": float(np.mean(ood_sup_np)),
    }


def reliability_score(metrics: Dict[str, float]) -> float:
    """Single rough score for quick sweeps.

    This intentionally rewards the behavior this experiment cares about:
    decent prediction, lower confident-wrong/risk, and useful OOD uncertainty.
    """

    acc = metrics.get("accuracy", 0.0)
    nll = metrics.get("nll", 1.0)
    risk = metrics.get("risk_coverage_auc", 1.0)
    cw = metrics.get("confident_wrong_90", 1.0)
    auc_u = metrics.get("ood_auroc_by_uncertainty", 0.5)
    auc_s = metrics.get("ood_auroc_by_negative_support", 0.5)
    return float(acc - 0.05 * nll - 0.5 * risk - 2.0 * cw + 0.1 * auc_u + 0.1 * auc_s)
