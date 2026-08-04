"""Generate publication plots from a results directory."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SYSTEMS = ["grep", "vector", "hybrid", "greprag", "digit"]
COLORS = {"grep": "#888888", "vector": "#4C72B0", "hybrid": "#55A868",
          "greprag": "#C44E52", "digit": "#8172B3", "digit_raw": "#CCB974"}
LABELS = {"grep": "Grep", "vector": "Vector", "hybrid": "Hybrid",
          "greprag": "GrepRAG", "digit": "DIGIT", "digit_raw": "DIGIT (no gate)"}


def load(results_dir):
    m = json.load(open(os.path.join(results_dir, "metrics.json"), encoding="utf-8"))["metrics"]
    rows = [json.loads(l) for l in open(os.path.join(results_dir, "per_question.jsonl"), encoding="utf-8")]
    return m, rows


def bar_main(m, out):
    syss = [s for s in SYSTEMS if s in m]
    metrics = [("final_answer_acc", "Final-answer accuracy"),
               ("exact_value_acc", "Exact-value accuracy"),
               ("source_span_acc", "Source-span correctness")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (key, title) in zip(axes, metrics):
        vals = [m[s][key] or 0 for s in syss]
        ax.bar([LABELS[s] for s in syss], vals, color=[COLORS[s] for s in syss])
        ax.set_title(title); ax.set_ylim(0, 1.0)
        ax.tick_params(axis="x", rotation=30)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.suptitle("DIGRAG: answer quality by system", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "main_accuracy.png"), dpi=140)
    plt.close(fig)


def heatmap_case(m, out):
    syss = [s for s in SYSTEMS if s in m]
    cts = list(m["digit"]["by_case_type"].keys())
    M = np.array([[m[s]["by_case_type"].get(ct) or 0 for s in syss] for ct in cts])
    fig, ax = plt.subplots(figsize=(8, 9))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(syss))); ax.set_xticklabels([LABELS[s] for s in syss], rotation=30)
    ax.set_yticks(range(len(cts))); ax.set_yticklabels(cts)
    for i in range(len(cts)):
        for j in range(len(syss)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Final-answer accuracy by case type", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "case_type_heatmap.png"), dpi=140)
    plt.close(fig)


def scatter_efficiency(m, out):
    syss = [s for s in SYSTEMS if s in m]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for s in syss:
        ax.scatter(m[s]["mean_context_tokens"], m[s]["final_answer_acc"],
                   s=160, color=COLORS[s], label=LABELS[s], edgecolor="k", zorder=3)
        ax.annotate(LABELS[s], (m[s]["mean_context_tokens"], m[s]["final_answer_acc"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_xlabel("Mean context tokens fed to decoder")
    ax.set_ylabel("Final-answer accuracy")
    ax.set_title("Accuracy vs. context size (efficiency frontier)", fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "efficiency_frontier.png"), dpi=140)
    plt.close(fig)


def bar_hallucination(m, out):
    syss = [s for s in SYSTEMS if s in m]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(syss)); w = 0.38
    unsup = [m[s]["unsupported_claim_rate"] or 0 for s in syss]
    ans = [1 - (m[s]["final_answer_acc"] or 0) for s in syss]
    ax.bar(x - w/2, unsup, w, label="Unsupported-claim rate", color="#C44E52")
    ax.bar(x + w/2, ans, w, label="Answer error rate", color="#4C72B0")
    ax.set_xticks(x); ax.set_xticklabels([LABELS[s] for s in syss], rotation=30)
    ax.set_title("Hallucination vs. error", fontweight="bold"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "hallucination.png"), dpi=140)
    plt.close(fig)


def recall_curves(m, out):
    syss = [s for s in SYSTEMS if s in m]
    ks = sorted(int(k.split("@")[1]) for k in m["grep"] if k.startswith("recall@"))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for s in syss:
        ys = [m[s].get(f"recall@{k}") or 0 for k in ks]
        ax.plot(ks, ys, "-o", color=COLORS[s], label=LABELS[s])
    ax.set_xlabel("k"); ax.set_ylabel("Recall@k (gold evidence docs)")
    ax.set_title("Retrieval recall@k", fontweight="bold")
    ax.set_xticks(ks); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "recall_at_k.png"), dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    out = os.path.join(a.results, "plots")
    os.makedirs(out, exist_ok=True)
    m, rows = load(a.results)
    bar_main(m, out); heatmap_case(m, out); scatter_efficiency(m, out)
    bar_hallucination(m, out); recall_curves(m, out)
    print(f"[plots] wrote 5 figures to {out}")


if __name__ == "__main__":
    main()
