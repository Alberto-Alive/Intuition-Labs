"""Generate tables, plots, and failure analysis from the result blocks."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(RES, "results")
TABLES = os.path.join(RESULTS, "tables")
PLOTS = os.path.join(RESULTS, "plots")

MAIN_ORDER = ["lexical", "vector", "hybrid", "greprag", "digrag_raw", "digrag_gated"]
MAIN_LABEL = {"lexical": "Lexical (BM25)", "vector": "Vector", "hybrid": "Hybrid",
              "greprag": "GrepRAG", "digrag_raw": "DIGRAG-raw", "digrag_gated": "DIGRAG-gated"}
ABL_ORDER = ["A_full", "B_no_gate", "C_no_buckets", "D_buckets_only", "E_no_temporal",
             "F_no_conflict_flags", "G_unconstrained_packet", "H_stuff_passages"]
ABL_LABEL = {"A_full": "Full DIGRAG (gated)", "B_no_gate": "− gate (raw packet)",
             "C_no_buckets": "− typed buckets (context only)", "D_buckets_only": "− context (buckets only)",
             "E_no_temporal": "− temporal/freshness", "F_no_conflict_flags": "− conflict flags",
             "G_unconstrained_packet": "− constrained decoding", "H_stuff_passages": "chunk-stuffing (no packet)"}
COLORS = {"lexical": "#888", "vector": "#4C72B0", "hybrid": "#55A868", "greprag": "#C44E52",
          "digrag_raw": "#CCB974", "digrag_gated": "#8172B3"}


def load_block(name):
    p = os.path.join(RESULTS, name, "metrics.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def fmt(v, d=3):
    return f"{v:.{d}f}" if isinstance(v, (int, float)) else "—"


METRIC_ROWS = [
    ("final_answer_acc", "Final-answer accuracy", "↑"),
    ("answerable_acc", "  · answerable subset", "↑"),
    ("abstention_acc", "  · abstention (unanswerable)", "↑"),
    ("temporal_update_acc", "Temporal/update accuracy", "↑"),
    ("answerability_acc", "Answerability accuracy", "↑"),
    ("support_recall", "Supporting-evidence recall", "↑"),
    ("recall@5", "Retrieval recall@5", "↑"),
    ("source_span_acc", "Source-span correctness", "↑"),
    ("unsupported_claim_rate", "Unsupported-claim rate", "↓"),
    ("mean_context_tokens", "Mean context tokens", "↓"),
    ("mean_latency_ms", "Mean latency (ms)", "↓"),
]


def table(metrics, order, label):
    syss = [s for s in order if s in metrics]
    lines = ["| Metric | " + " | ".join(label[s] for s in syss) + " |",
             "|" + "---|" * (len(syss) + 1)]
    for key, name, arrow in METRIC_ROWS:
        cells = []
        for s in syss:
            v = metrics[s].get(key)
            cell = fmt(v if not (isinstance(v, float) and abs(v) > 5) else round(v, 1))
            if key == "final_answer_acc" and metrics[s].get("final_answer_acc_std") is not None:
                cell += f" ±{metrics[s]['final_answer_acc_std']:.3f}"
            cells.append(cell)
        lines.append(f"| {name} ({arrow}) | " + " | ".join(cells) + " |")
    return "\n".join(syss and lines or ["(no data)"])


# ---------------- plots -------------------------------------------------
def plot_main_accuracy(lme, mh):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    syss = [s for s in MAIN_ORDER if (lme and s in lme["metrics"])]
    x = np.arange(len(syss)); w = 0.38
    for off, blk, lab in [(-w/2, lme, "LongMemEval"), (w/2, mh, "MultiHop-RAG")]:
        if not blk:
            continue
        ys = [blk["metrics"][s]["final_answer_acc"] or 0 for s in syss]
        es = [blk["metrics"][s].get("final_answer_acc_std", 0) for s in syss]
        ax.bar(x + off, ys, w, yerr=es, capsize=3, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([MAIN_LABEL[s] for s in syss], rotation=20)
    ax.set_ylabel("Final-answer accuracy"); ax.set_title("Final-answer accuracy by system", fontweight="bold")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "main_accuracy.png"), dpi=140); plt.close(fig)


def plot_recall_vs_answer(lme, mh):
    fig, ax = plt.subplots(figsize=(6.8, 5))
    for blk, mk in [(lme, "o"), (mh, "s")]:
        if not blk:
            continue
        for s in MAIN_ORDER:
            if s not in blk["metrics"]:
                continue
            m = blk["metrics"][s]
            ax.scatter(m.get("support_recall") or 0, m.get("final_answer_acc") or 0,
                       marker=mk, s=130, color=COLORS.get(s, "#333"), edgecolor="k", zorder=3)
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
    ax.set_xlabel("Supporting-evidence recall (retrieval finds it)")
    ax.set_ylabel("Final-answer accuracy (answer is right)")
    ax.set_title("Retrieval recall ≠ answer correctness\n(○ LongMemEval  □ MultiHop)", fontweight="bold")
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[s], label=MAIN_LABEL[s],
               markeredgecolor="k") for s in MAIN_ORDER if (lme and s in lme["metrics"])]
    ax.legend(handles=handles, fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "recall_vs_answer.png"), dpi=140); plt.close(fig)


def plot_unsupported(lme, mh):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    syss = [s for s in MAIN_ORDER if (lme and s in lme["metrics"])]
    x = np.arange(len(syss)); w = 0.38
    for off, blk, lab in [(-w/2, lme, "LongMemEval"), (w/2, mh, "MultiHop")]:
        if not blk:
            continue
        ys = [blk["metrics"][s].get("unsupported_claim_rate") or 0 for s in syss]
        ax.bar(x + off, ys, w, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([MAIN_LABEL[s] for s in syss], rotation=20)
    ax.set_ylabel("Unsupported-claim rate (↓)"); ax.set_title("Hallucination / unsupported claims", fontweight="bold")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "unsupported.png"), dpi=140); plt.close(fig)


def plot_token_efficiency(lme):
    if not lme:
        return
    fig, ax = plt.subplots(figsize=(6.6, 5))
    for s in MAIN_ORDER:
        if s not in lme["metrics"]:
            continue
        m = lme["metrics"][s]
        ax.scatter(m["mean_context_tokens"], m["final_answer_acc"], s=150,
                   color=COLORS.get(s, "#333"), edgecolor="k", zorder=3)
        ax.annotate(MAIN_LABEL[s], (m["mean_context_tokens"], m["final_answer_acc"]),
                    textcoords="offset points", xytext=(7, 3), fontsize=8)
    ax.set_xlabel("Mean context tokens"); ax.set_ylabel("Final-answer accuracy")
    ax.set_title("Token efficiency (LongMemEval)", fontweight="bold"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "token_efficiency.png"), dpi=140); plt.close(fig)


def plot_ablation(abl):
    if not abl:
        return
    syss = [s for s in ABL_ORDER if s in abl["metrics"]]
    full = abl["metrics"].get("A_full", {}).get("final_answer_acc") or 0
    ys = [abl["metrics"][s]["final_answer_acc"] or 0 for s in syss]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#8172B3"] + ["#C44E52" if y < full else "#55A868" for y in ys[1:]]
    ax.bar(range(len(syss)), ys, color=colors)
    ax.axhline(full, ls="--", color="gray", label=f"full = {full:.3f}")
    ax.set_xticks(range(len(syss))); ax.set_xticklabels([ABL_LABEL[s] for s in syss], rotation=30, ha="right")
    ax.set_ylabel("Final-answer accuracy"); ax.set_title("Ablations (LongMemEval): removing each component", fontweight="bold")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "ablation.png"), dpi=140); plt.close(fig)


# ---------------- failure analysis -------------------------------------
def failure_analysis(lme, mh):
    out = ["## Failure analysis\n"]
    for blk, name in [(lme, "LongMemEval"), (mh, "MultiHop-RAG")]:
        if not blk:
            continue
        out.append(f"### {name}: final-answer accuracy by question type\n")
        syss = [s for s in MAIN_ORDER if s in blk["metrics"]]
        qts = sorted(blk["metrics"]["digrag_gated"]["by_question_type"].keys())
        out.append("| Question type | " + " | ".join(MAIN_LABEL[s] for s in syss) + " |")
        out.append("|" + "---|" * (len(syss) + 1))
        for qt in qts:
            cells = [fmt(blk["metrics"][s]["by_question_type"].get(qt)) for s in syss]
            out.append(f"| {qt} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def main():
    os.makedirs(TABLES, exist_ok=True); os.makedirs(PLOTS, exist_ok=True)
    lme = load_block("longmemeval_main")
    abl = load_block("longmemeval_ablation")
    mh = load_block("multihop_main")
    cf = load_block("conflict_stress")

    if lme:
        open(os.path.join(TABLES, "main_results.md"), "w", encoding="utf-8").write(
            f"# LongMemEval — main results\n\nDecoder `{lme['decoder']}`, "
            f"{lme['n_per_seed']}/seed × {len(lme['seeds'])} seeds, top-k={lme['top_k']}, "
            f"ctx budget {lme.get('ctx_token_budget')} tok, dev-calibrated abstention "
            f"τ={lme.get('abstention_tau_dev_calibrated')}.\n\n" + table(lme["metrics"], MAIN_ORDER, MAIN_LABEL))
    if abl:
        full = abl["metrics"].get("A_full", {}).get("final_answer_acc")
        deltas = []
        for s in ABL_ORDER:
            if s in abl["metrics"] and s != "A_full":
                d = (abl["metrics"][s]["final_answer_acc"] or 0) - (full or 0)
                deltas.append((ABL_LABEL[s], round(d, 4)))
        deltas.sort(key=lambda x: x[1])
        body = "# LongMemEval — ablation study\n\n" + table(abl["metrics"], ABL_ORDER, ABL_LABEL)
        body += "\n\n**Δ final-answer accuracy vs full DIGRAG** (most harmful removal first):\n\n"
        body += "\n".join(f"- {n}: {d:+.3f}" for n, d in deltas)
        open(os.path.join(TABLES, "ablation_results.md"), "w", encoding="utf-8").write(body)
    if mh:
        open(os.path.join(TABLES, "generalization_results.md"), "w", encoding="utf-8").write(
            f"# MultiHop-RAG — generalization\n\nDecoder `{mh['decoder']}`, {mh['n_per_seed']}/seed × "
            f"{len(mh['seeds'])} seeds.\n\n" + table(mh["metrics"], MAIN_ORDER, MAIN_LABEL))
    if cf:
        open(os.path.join(TABLES, "conflict_stress.md"), "w", encoding="utf-8").write(
            f"# Conflict / freshness stress-test (semi-real, separate)\n\n"
            f"Injected stale contradictory variants; correct = current value. "
            f"{cf['n_per_seed']}/seed × {len(cf['seeds'])} seeds.\n\n" + table(cf["metrics"], MAIN_ORDER, MAIN_LABEL))

    open(os.path.join(TABLES, "failure_analysis.md"), "w", encoding="utf-8").write(failure_analysis(lme, mh))

    plot_main_accuracy(lme, mh); plot_recall_vs_answer(lme, mh); plot_unsupported(lme, mh)
    plot_token_efficiency(lme); plot_ablation(abl)
    print("[report] tables + plots written to", RESULTS)


if __name__ == "__main__":
    main()
