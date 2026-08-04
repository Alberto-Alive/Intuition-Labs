"""Publication-quality figure generation."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# Publication style
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_training_curves(
    histories: List[List[Dict]],
    seeds: List[int],
    output_dir: str,
):
    """Plot training curves with error bands across seeds.

    Creates a 2x3 grid: total loss, primitive loss, generation loss,
    policy loss, leakage loss, primitive accuracy.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    metrics = [
        ("val_losses", "total", "Total Loss"),
        ("val_losses", "primitive", "Primitive Loss"),
        ("val_losses", "generation", "Generation Loss"),
        ("val_losses", "leakage", "Leakage Loss"),
        ("val_acc", "answer", "Answer Accuracy"),
        ("val_acc", "support", "Support Accuracy"),
    ]

    for ax, (group, key, title) in zip(axes.flat, metrics):
        all_curves = []
        for history in histories:
            curve = [epoch[group][key] for epoch in history]
            all_curves.append(curve)

        # Align to shortest
        min_len = min(len(c) for c in all_curves)
        all_curves = [c[:min_len] for c in all_curves]
        arr = np.array(all_curves)

        epochs = np.arange(1, min_len + 1)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)

        ax.plot(epochs, mean, linewidth=1.5)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.pdf"))
    plt.savefig(os.path.join(output_dir, "training_curves.png"))
    plt.close()


def plot_system_comparison(
    results: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: str,
):
    """Bar chart comparing all systems on key metrics."""
    systems = list(results.keys())
    metrics = ["answer_accuracy", "support_accuracy"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    colors = sns.color_palette("Set2", len(systems))

    for ax, metric in zip(axes, metrics):
        means = []
        stds = []
        for sys_name in systems:
            r = results[sys_name]
            if metric in r:
                means.append(r[metric].get("mean", r[metric]) if isinstance(r[metric], dict) else r[metric])
                stds.append(r[metric].get("std", 0) if isinstance(r[metric], dict) else 0)
            else:
                means.append(0)
                stds.append(0)

        x = np.arange(len(systems))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors[:len(systems)])
        ax.set_xticks(x)
        ax.set_xticklabels(systems, rotation=25, ha="right")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "system_comparison.pdf"))
    plt.savefig(os.path.join(output_dir, "system_comparison.png"))
    plt.close()


def plot_privacy_utility_tradeoff(
    digit_results: Dict[str, float],
    dp_results: List[Dict[str, float]],
    output_dir: str,
):
    """Privacy-utility Pareto plot.

    X-axis: MIA AUC (lower = more private, 0.5 = ideal)
    Y-axis: answer accuracy (higher = more useful)
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    # DIGIT point
    ax.scatter(
        digit_results.get("mia_best_auc", 0.5),
        digit_results.get("answer_accuracy", 0),
        s=150, marker="*", color="red", zorder=5, label="DIGIT",
    )

    # DP curve
    if dp_results:
        dp_mia = [r.get("mia_auc", 0.5) for r in dp_results]
        dp_acc = [r.get("answer_accuracy", 0) for r in dp_results]
        dp_eps = [r.get("epsilon", 0) for r in dp_results]

        ax.plot(dp_mia, dp_acc, "o-", color="blue", label="DP Laplace")
        for i, eps in enumerate(dp_eps):
            ax.annotate(f"ε={eps}", (dp_mia[i], dp_acc[i]),
                        fontsize=8, ha="left", va="bottom")

    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="Random guess (ideal)")
    ax.set_xlabel("Membership Inference AUC (lower = more private)")
    ax.set_ylabel("Answer Accuracy (higher = better)")
    ax.set_title("Privacy–Utility Tradeoff")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "privacy_utility_tradeoff.pdf"))
    plt.savefig(os.path.join(output_dir, "privacy_utility_tradeoff.png"))
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_dir: str,
    filename: str = "confusion_matrix",
):
    """Plot a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Answer Primitive Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}.pdf"))
    plt.savefig(os.path.join(output_dir, f"{filename}.png"))
    plt.close()


def plot_ablation(
    ablation_results: Dict[str, Dict[str, float]],
    output_dir: str,
):
    """Plot ablation results: utility and leakage vs bottleneck size."""
    configs = list(ablation_results.keys())
    bits = [ablation_results[c].get("max_bits", 0) for c in configs]
    acc = [ablation_results[c].get("answer_accuracy_mean", 0) for c in configs]
    acc_std = [ablation_results[c].get("answer_accuracy_std", 0) for c in configs]
    leak = [ablation_results[c].get("leakage_mean", 0) for c in configs]
    leak_std = [ablation_results[c].get("leakage_std", 0) for c in configs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.errorbar(bits, acc, yerr=acc_std, fmt="o-", capsize=5)
    ax1.set_xlabel("Max Bits per Query")
    ax1.set_ylabel("Answer Accuracy")
    ax1.set_title("Utility vs. Bottleneck Size")
    ax1.grid(True, alpha=0.3)

    ax2.errorbar(bits, leak, yerr=leak_std, fmt="s-", color="red", capsize=5)
    ax2.set_xlabel("Max Bits per Query")
    ax2.set_ylabel("Leakage Score")
    ax2.set_title("Leakage vs. Bottleneck Size")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ablation_bottleneck.pdf"))
    plt.savefig(os.path.join(output_dir, "ablation_bottleneck.png"))
    plt.close()
