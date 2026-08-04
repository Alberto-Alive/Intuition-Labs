"""Typed value extraction — the per-document half of the evidence compiler.

Given a Document, emit typed `EvidenceItem`s: numbers, dates, names, ids,
clauses, table cells, code symbols and policy fields.  These are deliberately
schema-aware heuristics (this is a synthetic-domain PoC); the contribution is
the *architecture* of compiling typed evidence before generation, not a
state-of-the-art extractor.
"""
from __future__ import annotations

import re
from typing import List

from ..schema import Document, EvidenceItem

DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
DATE_ROLE_RE = re.compile(r"([A-Za-z][A-Za-z\-]*)\s+(\d{4}-\d{2}-\d{2})")
NUM_UNIT_RE = re.compile(r"(\$?\d[\d,\.]*[kKmM]?)\s?(days?|%|TB|GB|months?|seconds?|/yr|/year)?", re.I)
CODE_RE = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\s*=\s*([^\s#]+)")
STATUS_RE = re.compile(r"Status:\s*([A-Za-z]+)", re.I)
ID_RE = re.compile(r"\b(?:[A-Z]{2,}-[A-Za-z0-9\-]+|[A-Z][a-z]+-\d+|RP-[A-Z]+-\d+|C-\d+|P-\d+)\b")
CLAUSE_MARKERS = ("override", "except", "statutory", "not available", "require",
                  "auto-approved", "manager approval", "or more", "under $",
                  "supersede", "cooling-off")


def _ctx(doc: Document) -> str:
    """Applicability context = title + facet hints carried by the document."""
    bits = [doc.title]
    if doc.meta:
        bits += [f"{k}={v}" for k, v in doc.meta.items()]
    bits.append(f"status={doc.status}")
    return "; ".join(str(b) for b in bits)


def extract_items(doc: Document) -> List[EvidenceItem]:
    items: List[EvidenceItem] = []
    ctx = _ctx(doc)

    def add(field_type, entity, value, span):
        items.append(EvidenceItem(
            field_type=field_type, entity=str(entity), exact_value=str(value).strip(),
            source_span=span.strip(), document_id=doc.doc_id, timestamp=doc.timestamp,
            applicability_context=ctx, applies=True, status=doc.status))

    # ----- table cells (parse first so numbers in tables are typed as cells)
    table_lines = [ln for ln in doc.lines() if ln.strip().startswith("|")]
    consumed_lines = set()
    if len(table_lines) >= 2:
        header = [c.strip() for c in table_lines[0].strip().strip("|").split("|")]
        cols = header[1:]  # first header cell is the row-label column
        for ln in table_lines[2:]:  # skip header + separator
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if len(cells) < 2:
                continue
            row_label = cells[0]
            for ci, val in enumerate(cells[1:]):
                col = cols[ci] if ci < len(cols) else f"col{ci}"
                add("table_cell", f"{row_label}|{col}", val, ln)
            consumed_lines.add(ln)

    for line in doc.lines():
        if line in consumed_lines:
            continue
        # ----- code symbols
        for sym, val in CODE_RE.findall(line):
            add("code_symbol", sym, val, line)
        # ----- dates with role
        roled = DATE_ROLE_RE.findall(line)
        roled_dates = {d for _, d in roled}
        for role, d in roled:
            add("date", role, d, line)
        for d in DATE_RE.findall(line):
            if d not in roled_dates:
                add("date", "date", d, line)
        # ----- status / policy fields
        for st in STATUS_RE.findall(line):
            add("policy_field", "status", st.upper(), line)
        # ----- ids
        for _id in ID_RE.findall(line):
            add("id", _id, _id, line)
        # ----- numbers with units (skip lines already handled as code/table)
        if not CODE_RE.search(line):
            for val, unit in NUM_UNIT_RE.findall(line):
                if not val or val in {"$"}:
                    continue
                value = (val + ((" " + unit) if unit else "")).strip()
                add("number", "value", value, line)
        # ----- clauses / policy conditions
        low = line.lower()
        if any(m in low for m in CLAUSE_MARKERS):
            add("clause", "clause", line.strip(), line)
        # ----- capitalized entity names
        for ent in re.findall(r"\bProject\s+([A-Z][a-z]+)\b", line):
            add("name", ent, ent, line)

    # dedup identical (field_type, entity, exact_value, span)
    seen, uniq = set(), []
    for it in items:
        key = (it.field_type, it.entity, it.exact_value, it.source_span)
        if key not in seen:
            seen.add(key)
            uniq.append(it)
    return uniq
