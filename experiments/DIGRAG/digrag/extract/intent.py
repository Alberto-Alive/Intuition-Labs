"""Query intent + facet parsing (derived from the question text only).

This is the query-side half of the evidence compiler.  It decides what *type*
of value the answer needs and which *facets* (region, plan, scope, entity,
date-role, ...) constrain applicability, so the compiler can keep only the
values that actually apply.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..dataset.generator import REGIONS, PLANS

ID_RE = re.compile(r"\b(?:[A-Z]{2,}-\d+|C-\d+|P-\d+|[A-Z][a-z]+-\d+)\b")
SYMBOL_RE = re.compile(r"\b([A-Z][A-Z0-9_]{3,})\b")
AMOUNT_RE = re.compile(r"\$(\d[\d,\.]*)")
# The unique, question-addressable subject token, e.g. Atlas-014, C-101, ATL-14.
SUBJECT_RE = re.compile(r"[A-Z][a-z]+-\d{2,}|[A-Z]{2,}-\d+|C-\d+|P-\d+")


@dataclass
class Intent:
    raw: str
    target_field: str                       # number|date|id|code_symbol|table_cell|policy_field|clause
    decision: bool                          # yes/no/conditional question?
    facets: Dict[str, str] = field(default_factory=dict)
    keywords: List[str] = field(default_factory=list)
    alias: Optional[str] = None             # an id/codename to resolve
    subjects: List[str] = field(default_factory=list)   # unique handles to scope evidence
    summary: str = ""


def parse_intent(question: str) -> Intent:
    q = question
    low = q.lower()
    facets: Dict[str, str] = {}

    # --- facets ---
    for r in REGIONS:
        if re.search(rf"\b{r}\b", q):
            facets["region"] = r
    for p in PLANS:
        if re.search(rf"\b{p}\b", low):
            facets["plan"] = p
    if "renewal" in low:
        facets["purchase"] = "renewal"
    if "non-refundable" in low:
        facets["purchase"] = "non-refundable"
    if "production" in low or "prod" in low:
        facets["scope"] = "prod"
    if "test" in low:
        facets.setdefault("scope", "test")
    ma = AMOUNT_RE.search(q)
    if ma:
        facets["amount"] = ma.group(1).replace(",", "")
    subjects = list(dict.fromkeys(SUBJECT_RE.findall(q)))
    if subjects:
        facets["subject"] = subjects[0]
    # an all-caps code like ATL-14 is an alias to resolve (vs a readable subject)
    alias = next((s for s in subjects if s.split("-")[0].isupper()), None)

    # --- target field ---
    decision = bool(re.search(r"\b(can|is|are|does|do|will|should|entitled|allowed|without|may)\b", low)) \
        and "what" not in low and "which" not in low and "where" not in low and "when" not in low
    if "when" in low or "date" in low or "renew" in low or "expire" in low:
        target = "date"
    elif re.search(r"\bvalue of\b|production value|default value", low) or (
            SYMBOL_RE.search(q) and "=" not in q and any(
                s in q for s in ["TIMEOUT", "MAX_", "BATCH", "CACHE", "POOL"])):
        target = "code_symbol"
    elif "price" in low and ("plan" in low or "table" in low or "matrix" in low):
        target = "table_cell"
    elif "sla" in low or "uptime" in low:
        target = "number"
    elif "budget" in low or "amount" in low or "window" in low or "period" in low or \
            "retention" in low or "timeout" in low or "how many" in low or "how much" in low:
        target = "number"
    elif "where" in low or "stored" in low or "region" in low:
        target = "policy_field"
    elif decision:
        target = "policy_field"
    else:
        target = "number"

    # --- keywords (reuse grep extractor) ---
    from ..retrieval.grep import extract_keywords
    kws = extract_keywords(question)

    intent = Intent(raw=q, target_field=target, decision=decision, facets=facets,
                    keywords=kws, alias=alias, subjects=subjects)
    intent.summary = (f"intent={'decision' if decision else 'lookup'}; "
                      f"target={target}; facets={facets}")
    return intent
