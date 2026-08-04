"""Format results as LaTeX tables for paper inclusion."""

from __future__ import annotations

from typing import Dict, List


def format_main_results_table(
    results: Dict[str, Dict[str, Dict[str, float]]],
) -> str:
    """Generate LaTeX table for main results.

    Args:
        results: {system_name: {metric: {mean, std}}}
    """
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Main results on UCI Adult Income (mean $\pm$ std over 5 seeds).}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"System & Ans. Acc & Sup. Acc & Conf. Acc & Risk Acc & Joint Acc & Leakage \\",
        r"\midrule",
    ]

    for name, metrics in results.items():
        row_parts = [name]
        for key in ["answer_accuracy", "support_accuracy", "confidence_accuracy",
                     "risk_accuracy", "joint_accuracy", "leakage"]:
            if key in metrics:
                m = metrics[key]
                if isinstance(m, dict):
                    row_parts.append(f"${m['mean']:.3f} \\pm {m['std']:.3f}$")
                else:
                    row_parts.append(f"${m:.3f}$")
            else:
                row_parts.append("--")
        lines.append(" & ".join(row_parts) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def format_privacy_table(
    results: Dict[str, Dict[str, float]],
) -> str:
    """Generate LaTeX table for privacy attack results."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Privacy attack results. MIA AUC near 0.5 indicates strong privacy.}",
        r"\label{tab:privacy}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"System & MIA AUC (LR) & MIA AUC (MLP) & AIA Baseline & AIA Informed & AIA Adv. \\",
        r"\midrule",
    ]

    for name, metrics in results.items():
        row = [
            name,
            _fmt(metrics, "mia_lr_auc_mean", "mia_lr_auc_std"),
            _fmt(metrics, "mia_mlp_auc_mean", "mia_mlp_auc_std"),
            _fmt(metrics, "aia_baseline_auc_mean", "aia_baseline_auc_std"),
            _fmt(metrics, "aia_informed_auc_mean", "aia_informed_auc_std"),
            _fmt(metrics, "aia_auc_advantage_mean", "aia_auc_advantage_std"),
        ]
        lines.append(" & ".join(row) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def format_dp_comparison_table(
    dp_results: List[Dict[str, float]],
    digit_results: Dict[str, float],
) -> str:
    """Generate LaTeX table comparing DIGIT with DP at various epsilon."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{DIGIT vs. Differential Privacy (Laplace) at various $\epsilon$ values.}",
        r"\label{tab:dp_comparison}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"System & Ans. Accuracy & MIA AUC & Bits/Query \\",
        r"\midrule",
    ]

    # DIGIT row
    lines.append(
        f"DIGIT & ${digit_results.get('answer_accuracy', 0):.3f}$ "
        f"& ${digit_results.get('mia_auc', 0.5):.3f}$ "
        f"& ${digit_results.get('max_bits', 7.75):.2f}$ \\\\"
    )
    lines.append(r"\midrule")

    for r in dp_results:
        eps = r.get("epsilon", 0)
        lines.append(
            f"DP ($\\epsilon={eps}$) & ${r.get('answer_accuracy', 0):.3f}$ "
            f"& ${r.get('mia_auc', 0.5):.3f}$ "
            f"& $\\infty$ \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def _fmt(d: Dict, mean_key: str, std_key: str) -> str:
    """Format mean +/- std for LaTeX."""
    mean = d.get(mean_key, 0)
    std = d.get(std_key, 0)
    if std > 0:
        return f"${mean:.3f} \\pm {std:.3f}$"
    return f"${mean:.3f}$"
