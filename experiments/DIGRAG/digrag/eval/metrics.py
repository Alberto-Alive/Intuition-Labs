"""Deterministic evaluation metrics.

Scoring is structured rather than LLM-judged: the LABEL/VALUE answer contract
plus normalized value-matching gives reproducible numbers with no extra model
calls.  All metrics requested by the brief are computed here.
"""
from __future__ import annotations

import re
import statistics
from typing import Dict, List, Optional

from ..schema import Prediction, Question

VALUE_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}"                      # ISO date
    r"|\$\d[\d,\.]*k?"                        # money ($120k, $499)
    r"|\d[\d,\.]*\s?(?:days?|months?|seconds?|%|TB|GB)"  # value+unit
    r"|\b[A-Z]{2,}-[A-Za-z0-9\-]+\b"         # ids (P-1234, ATL-7)
    r"|\b\d+\.\d+%?\b"                        # decimals / 99.9%
    r"|\b\d{2,}\b",                           # bare integers (>=2 digits)
    re.I)


def norm(s: str) -> str:
    s = (s or "").lower().replace("$", "").replace(",", "").rstrip(".")
    return re.sub(r"\s+", "", s)   # whitespace-insensitive ("99.95 %" == "99.95%")


def value_match(gold: str, *candidates: str) -> bool:
    g = norm(gold)
    if not g:
        return False
    for c in candidates:
        cn = norm(c)
        if g and (g in cn or cn in g and len(cn) >= 2):
            return True
    return False


def extract_values(text: str) -> List[str]:
    return [m.group(0) for m in VALUE_RE.finditer(text or "")]


def score_one(p: Prediction, q: Question, recall_ks: List[int]) -> Dict:
    gold_docs = set(q.gold_doc_ids())
    # --- answer correctness ---
    vmatch = value_match(q.gold_value, p.used_value, p.answer_text) if q.gold_value else None
    if q.gold_label == "VALUE":
        answer_correct = bool(vmatch)
    elif q.gold_label == "INSUFFICIENT":
        answer_correct = (p.pred_label == "INSUFFICIENT")
    else:  # YES / NO / CONDITIONAL
        answer_correct = (p.pred_label == q.gold_label)

    # --- exact-value accuracy (value-lookup questions only) ---
    value_acc = bool(vmatch) if q.gold_label == "VALUE" else None

    # --- source-span / citation correctness ---
    if gold_docs:
        cited = set(p.cited_doc_ids)
        source_correct = len(cited & gold_docs) > 0
    else:
        source_correct = None

    # --- unsupported-claim rate (hallucination) ---
    allowed = norm(p.context_text + " " + q.question)
    ans_vals = extract_values(p.answer_text + " " + p.used_value)
    unsupported = [v for v in ans_vals if norm(v) not in allowed]
    n_vals = len(ans_vals)
    unsupported_rate = (len(unsupported) / n_vals) if n_vals else 0.0

    # --- conflict detection ---
    should_conflict = (q.case_type == "conflict")
    conflict_correct = (bool(p.flagged_conflict) == should_conflict)

    # --- missing-evidence detection ---
    missing_target = (q.case_type == "unanswerable")
    missing_detected = (p.pred_label == "INSUFFICIENT")

    # --- answerability ---
    answerable_pred = (p.pred_label != "INSUFFICIENT")
    answerable_correct = (answerable_pred == q.answerable)

    # --- retrieval recall@k ---
    recall = {}
    if gold_docs:
        ranked = p.ranked_doc_ids or p.retrieved_doc_ids
        for k in recall_ks:
            recall[k] = len(set(ranked[:k]) & gold_docs) / len(gold_docs)

    latency = p.retrieval_ms + p.generation_ms
    return {
        "qid": q.qid, "system": p.system, "case_type": q.case_type,
        "answer_correct": answer_correct, "value_acc": value_acc,
        "source_correct": source_correct,
        "unsupported_rate": unsupported_rate, "n_values": n_vals,
        "unsupported_count": len(unsupported),
        "should_conflict": should_conflict, "pred_conflict": bool(p.flagged_conflict),
        "conflict_correct": conflict_correct,
        "missing_target": missing_target, "missing_detected": missing_detected,
        "answerable_gold": q.answerable, "answerable_pred": answerable_pred,
        "answerable_correct": answerable_correct,
        "recall": recall, "context_tokens": p.context_tokens,
        "output_tokens": p.output_tokens, "retrieval_ms": p.retrieval_ms,
        "generation_ms": p.generation_ms, "latency_ms": latency,
    }


def _mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return round(float(statistics.mean(xs)), 4) if xs else None


def aggregate(rows: List[Dict], recall_ks: List[int],
              usd_in: float = 0.0, usd_out: float = 0.0) -> Dict:
    systems = sorted({r["system"] for r in rows})
    out: Dict[str, Dict] = {}
    for sysname in systems:
        rs = [r for r in rows if r["system"] == sysname]
        conflict_set = [r for r in rs if r["should_conflict"]]
        nonconf = [r for r in rs if not r["should_conflict"] and r["answerable_gold"]]
        miss_set = [r for r in rs if r["missing_target"]]
        agg = {
            "n": len(rs),
            "final_answer_acc": _mean([r["answer_correct"] for r in rs]),
            "exact_value_acc": _mean([r["value_acc"] for r in rs]),
            "source_span_acc": _mean([r["source_correct"] for r in rs]),
            "unsupported_claim_rate": _mean([r["unsupported_rate"] for r in rs]),
            "total_unsupported_claims": sum(r["unsupported_count"] for r in rs),
            "conflict_detect_acc": _mean([r["conflict_correct"] for r in rs]),
            "conflict_recall": _mean([r["pred_conflict"] for r in conflict_set]),
            "conflict_false_pos": _mean([r["pred_conflict"] for r in nonconf]),
            "missing_evidence_recall": _mean([r["missing_detected"] for r in miss_set]),
            "answerability_acc": _mean([r["answerable_correct"] for r in rs]),
            "mean_context_tokens": _mean([r["context_tokens"] for r in rs]),
            "mean_output_tokens": _mean([r["output_tokens"] for r in rs]),
            "mean_retrieval_ms": _mean([r["retrieval_ms"] for r in rs]),
            "mean_generation_ms": _mean([r["generation_ms"] for r in rs]),
            "mean_latency_ms": _mean([r["latency_ms"] for r in rs]),
        }
        for k in recall_ks:
            agg[f"recall@{k}"] = _mean([r["recall"].get(k) for r in rs if r["recall"]])
        # cost ($) per question using configured rates
        ctx = agg["mean_context_tokens"] or 0
        otk = agg["mean_output_tokens"] or 0
        agg["usd_per_q"] = round(ctx / 1000 * usd_in + otk / 1000 * usd_out, 6)
        # per-case-type answer accuracy
        agg["by_case_type"] = {}
        for ct in sorted({r["case_type"] for r in rs}):
            cr = [r for r in rs if r["case_type"] == ct]
            agg["by_case_type"][ct] = _mean([r["answer_correct"] for r in cr])
        out[sysname] = agg
    return out
