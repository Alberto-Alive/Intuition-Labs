"""Build result tables and a failure analysis from a results directory."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

SYS_ORDER = ["grep", "vector", "hybrid", "greprag", "digit", "digit_raw"]
SYS_LABEL = {"grep": "Grep", "vector": "Vector RAG", "hybrid": "Hybrid RAG",
             "greprag": "GrepRAG", "digit": "DIGIT", "digit_raw": "DIGIT (no gate)"}

METRIC_ROWS = [
    ("final_answer_acc", "Final-answer accuracy", "↑"),
    ("exact_value_acc", "Exact-value accuracy", "↑"),
    ("source_span_acc", "Source-span correctness", "↑"),
    ("unsupported_claim_rate", "Unsupported-claim rate", "↓"),
    ("conflict_detect_acc", "Conflict-detection accuracy", "↑"),
    ("missing_evidence_recall", "Missing-evidence recall", "↑"),
    ("answerability_acc", "Answerability accuracy", "↑"),
    ("recall@5", "Retrieval recall@5", "↑"),
    ("mean_context_tokens", "Mean context tokens", "↓"),
    ("mean_output_tokens", "Mean output tokens", "↓"),
    ("mean_latency_ms", "Mean latency (ms)", "↓"),
    ("usd_per_q", "Cost ($/question)", "↓"),
]


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"
    return str(v)


def main_table(m, syss):
    lines = ["| Metric | " + " | ".join(SYS_LABEL[s] for s in syss) + " |",
             "|" + "---|" * (len(syss) + 1)]
    for key, label, arrow in METRIC_ROWS:
        cells = [fmt(m[s].get(key)) for s in syss]
        lines.append(f"| {label} ({arrow}) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def case_table(m, syss):
    cts = list(m["digit"]["by_case_type"].keys())
    lines = ["| Case type | " + " | ".join(SYS_LABEL[s] for s in syss) + " |",
             "|" + "---|" * (len(syss) + 1)]
    for ct in cts:
        cells = [fmt(m[s]["by_case_type"].get(ct)) for s in syss]
        lines.append(f"| {ct} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def failure_analysis(preds, qs, rows):
    by = defaultdict(dict)            # (system,qid) -> row
    for r in rows:
        by[(r["system"], r["qid"])] = r
    pred_by = defaultdict(dict)
    for p in preds:
        pred_by[(p["system"], p["qid"])] = p

    out = ["## Failure analysis\n",
           "### Failure counts by case type (final-answer errors)\n"]
    cts = sorted({q["case_type"] for q in qs.values()})
    header = "| Case type | " + " | ".join(SYS_LABEL[s] for s in SYS_ORDER if (s, list(qs)[0]) in by or True) + " |"
    syss = [s for s in SYS_ORDER if any(k[0] == s for k in by)]
    out.append("| Case type | " + " | ".join(SYS_LABEL[s] for s in syss) + " |")
    out.append("|" + "---|" * (len(syss) + 1))
    for ct in cts:
        qids = [qid for qid, q in qs.items() if q["case_type"] == ct]
        cells = []
        for s in syss:
            errs = sum(1 for qid in qids if (s, qid) in by and not by[(s, qid)]["answer_correct"])
            cells.append(f"{errs}/{len(qids)}")
        out.append(f"| {ct} | " + " | ".join(cells) + " |")

    out.append("\n### Representative traps (grep finds the literal, but it does not apply)\n")
    shown = 0
    for qid, q in qs.items():
        if shown >= 8:
            break
        gp = by.get(("grep", qid)); dp = by.get(("digit", qid))
        if not gp or not dp:
            continue
        # interesting = grep wrong, digit right
        if gp["answer_correct"] or not dp["answer_correct"]:
            continue
        gpred = pred_by[("grep", qid)]; dpred = pred_by[("digit", qid)]
        out.append(f"**[{q['case_type']}] {q['question']}**")
        out.append(f"- Gold: `{q['gold_label']}` / value `{q['gold_value'] or '—'}`  ")
        out.append(f"- Distractor literal grep is tempted by: `{q['distractor_value'] or '—'}`  ")
        out.append(f"- **Grep** → `{gpred['pred_label']}` value `{gpred['used_value'] or '—'}`: "
                   f"{gpred['answer_text'][:160]}  ")
        out.append(f"- **DIGIT** → `{dpred['pred_label']}` value `{dpred['used_value'] or '—'}`: "
                   f"{dpred['answer_text'][:160]}\n")
        shown += 1
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    tdir = os.path.join(a.results, "tables")
    os.makedirs(tdir, exist_ok=True)
    summary = json.load(open(os.path.join(a.results, "metrics.json"), encoding="utf-8"))
    m = summary["metrics"]
    syss = [s for s in SYS_ORDER if s in m]
    preds = [json.loads(l) for l in open(os.path.join(a.results, "predictions.jsonl"), encoding="utf-8")]
    rows = [json.loads(l) for l in open(os.path.join(a.results, "per_question.jsonl"), encoding="utf-8")]
    qs = {json.loads(l)["qid"]: json.loads(l)
          for l in open(os.path.join(a.data, "questions.jsonl"), encoding="utf-8")}

    with open(os.path.join(tdir, "main_results.md"), "w", encoding="utf-8") as f:
        f.write("# Main results\n\n")
        f.write(f"Dataset: {summary['dataset']['n_docs']} docs, {summary['n_questions']} questions. "
                f"Decoder: shared `{summary['config']['models']['decoder']}`.\n\n")
        f.write(main_table(m, syss))
    with open(os.path.join(tdir, "by_case_type.md"), "w", encoding="utf-8") as f:
        f.write("# Final-answer accuracy by case type\n\n")
        f.write(case_table(m, [s for s in syss if s != "digit_raw"]))
    with open(os.path.join(tdir, "failure_analysis.md"), "w", encoding="utf-8") as f:
        f.write(failure_analysis(preds, qs, rows))
    print(f"[analyze] wrote tables to {tdir}")


if __name__ == "__main__":
    main()
