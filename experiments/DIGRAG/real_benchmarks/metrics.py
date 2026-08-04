"""Deterministic scoring for real benchmarks.

Answer correctness uses normalized exact-match OR containment (for short gold
answers) OR token-F1 >= 0.5 — the standard surface judge for MultiHop-RAG /
LongMemEval short answers (no LLM judge, fully reproducible). Gold is used ONLY
here, never by the systems.
"""
from __future__ import annotations

import re
import statistics
from typing import Dict, List, Optional

from .schema import Example, SystemOutput

ARTICLES = {"a", "an", "the"}
FACT_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\$?\d[\d,\.]*\s?(?:%|percent|million|billion|k)?"
    r"|\b[A-Z][a-zA-Z0-9&.]+(?:\s+[A-Z][a-zA-Z0-9&.]+){0,3}\b", )


def norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    toks = [t for t in s.split() if t not in ARTICLES]
    return " ".join(toks)


def tokens(s: str) -> List[str]:
    return norm(s).split()


def f1(pred: str, gold: str) -> float:
    p, g = tokens(pred), tokens(gold)
    if not p or not g:
        return 0.0
    common = 0
    gp = list(g)
    for t in p:
        if t in gp:
            common += 1; gp.remove(t)
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def answer_correct(pred: str, gold: str) -> bool:
    np_, ng = norm(pred), norm(gold)
    if not ng:
        return False
    if np_ == ng:
        return True
    if len(ng.split()) <= 6 and ng in np_:      # short gold contained in answer
        return True
    return f1(pred, gold) >= 0.5


def _support_set(pid: str) -> str:
    return pid.split("::")[0]


def extract_facts(text: str) -> List[str]:
    return [m.group(0).strip() for m in FACT_RE.finditer(text or "") if len(m.group(0).strip()) > 1]


def score_one(out: SystemOutput, ex: Example, recall_ks: List[int]) -> Dict:
    gold_supp = set(ex.gold_passage_ids)

    # --- final answer accuracy / abstention ---
    if ex.answerable:
        correct = (not out.abstained) and answer_correct(out.pred_answer, ex.gold_answer)
    else:
        correct = out.abstained                       # should abstain
    abstain_correct = (out.abstained == (not ex.answerable))

    # --- supporting-evidence recall (session/doc granularity) ---
    def covered(pids):
        hit = {p for p in pids} | {_support_set(p) for p in pids}
        return len(gold_supp & hit) / len(gold_supp) if gold_supp else None
    support_recall = covered(out.retrieved_pids)
    recall_at = {k: covered(out.retrieved_pids[:k]) for k in recall_ks} if gold_supp else {}

    # --- source-span / citation correctness ---
    if gold_supp and not out.abstained:
        cited = {c for c in out.cited_pids} | {_support_set(c) for c in out.cited_pids}
        source_correct = len(cited & gold_supp) > 0
    else:
        source_correct = None

    # --- unsupported-claim rate ---
    if out.abstained:
        unsupported_rate, n_vals = 0.0, 0
    else:
        allowed = norm(out.context_text + " " + ex.question)
        facts = extract_facts(out.pred_answer)
        unsup = [f for f in facts if norm(f) not in allowed]
        n_vals = len(facts)
        unsupported_rate = (len(unsup) / n_vals) if n_vals else 0.0

    return {
        "qid": ex.qid, "system": out.system, "seed": out.seed, "benchmark": out.benchmark,
        "question_type": ex.question_type, "answerable": ex.answerable,
        "is_temporal": ex.is_temporal, "answer_correct": correct,
        "abstain_correct": abstain_correct, "abstained": out.abstained,
        "support_recall": support_recall, "recall": recall_at,
        "source_correct": source_correct, "flagged_conflict": out.flagged_conflict,
        "unsupported_rate": unsupported_rate, "n_values": n_vals,
        "context_tokens": out.context_tokens, "output_tokens": out.output_tokens,
        "latency_ms": out.retrieval_ms + out.generation_ms,
    }


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(statistics.mean(xs)), 4) if xs else None


def aggregate(rows: List[Dict], recall_ks: List[int],
              usd_in: float = 0.0, usd_out: float = 0.0) -> Dict:
    out = {}
    for sysname in sorted({r["system"] for r in rows}):
        rs = [r for r in rows if r["system"] == sysname]
        ans = [r for r in rs if r["answerable"]]
        unans = [r for r in rs if not r["answerable"]]
        temp = [r for r in rs if r["is_temporal"]]
        agg = {
            "n": len(rs),
            "final_answer_acc": _mean([r["answer_correct"] for r in rs]),
            "answerable_acc": _mean([r["answer_correct"] for r in ans]),
            "abstention_acc": _mean([r["abstained"] for r in unans]) if unans else None,
            "answerability_acc": _mean([r["abstain_correct"] for r in rs]),
            "temporal_update_acc": _mean([r["answer_correct"] for r in temp]) if temp else None,
            "support_recall": _mean([r["support_recall"] for r in rs]),
            "source_span_acc": _mean([r["source_correct"] for r in rs]),
            "unsupported_claim_rate": _mean([r["unsupported_rate"] for r in rs]),
            "mean_context_tokens": _mean([r["context_tokens"] for r in rs]),
            "mean_output_tokens": _mean([r["output_tokens"] for r in rs]),
            "mean_latency_ms": _mean([r["latency_ms"] for r in rs]),
        }
        for k in recall_ks:
            agg[f"recall@{k}"] = _mean([r["recall"].get(k) for r in rs if r["recall"]])
        ctx, otk = agg["mean_context_tokens"] or 0, agg["mean_output_tokens"] or 0
        agg["usd_per_q"] = round(ctx / 1000 * usd_in + otk / 1000 * usd_out, 6)
        agg["by_question_type"] = {}
        for qt in sorted({r["question_type"] for r in rs}):
            qr = [r for r in rs if r["question_type"] == qt]
            agg["by_question_type"][qt] = _mean([r["answer_correct"] for r in qr])
        out[sysname] = agg
    return out


def aggregate_with_seeds(rows: List[Dict], recall_ks: List[int], **kw) -> Dict:
    """Mean over seeds with std on the headline metric."""
    seeds = sorted({r["seed"] for r in rows})
    per_seed = {s: aggregate([r for r in rows if r["seed"] == s], recall_ks, **kw) for s in seeds}
    base = aggregate(rows, recall_ks, **kw)     # pooled
    for sysname, agg in base.items():
        accs = [per_seed[s][sysname]["final_answer_acc"] for s in seeds
                if sysname in per_seed[s] and per_seed[s][sysname]["final_answer_acc"] is not None]
        agg["final_answer_acc_std"] = round(float(statistics.pstdev(accs)), 4) if len(accs) > 1 else 0.0
        agg["n_seeds"] = len(seeds)
    return base
