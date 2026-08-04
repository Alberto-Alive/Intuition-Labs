"""Prompt templates shared by all systems.

Every system uses the SAME answer contract and the SAME base instruction, so
the comparison isolates *what context reaches the decoder*, not how it is
asked to answer.
"""
from __future__ import annotations

import json
from typing import List

from ..schema import RetrievedSpan, EvidencePacket

ANSWER_CONTRACT = """Respond in EXACTLY this format, one field per line:
LABEL: <YES|NO|CONDITIONAL|INSUFFICIENT|VALUE>
VALUE: <the single exact value your answer depends on, copied verbatim from the context; or NONE>
CITES: <comma-separated source ids (like D0123) you used; or NONE>
CONFLICT: <YES if the context gives conflicting values for what is asked, else NO>
ANSWER: <one or two sentence answer>

Label guidance:
- Use VALUE when the question asks for a specific value (number, date, price, id, setting).
- Use YES / NO for yes-no questions; CONDITIONAL if the answer depends on a stated condition.
- Use INSUFFICIENT if the context lacks the needed fact or sources conflict unresolvably."""

BASE_SYSTEM = (
    "You are a meticulous question-answering assistant for an enterprise knowledge base. "
    "Answer the QUESTION using ONLY the information in the provided CONTEXT. "
    "Do not use prior knowledge and do not guess. If the needed fact is absent, say so.\n\n"
    + ANSWER_CONTRACT
)

DIGIT_SYSTEM = (
    "You are a meticulous question-answering assistant. You are given a typed "
    "EVIDENCE_PACKET, not raw documents. Each item has an exact_value, a field_type, "
    "a source document_id, and an \"applies\" flag set by upstream applicability checks.\n"
    "Strict rules:\n"
    "- Use ONLY items with \"applies\": true. Items with \"applies\": false were retrieved "
    "but do NOT apply; ignore their values.\n"
    "- You may NOT output any number, date, name, price, or id that is not present in an "
    "exact_value field of the packet.\n"
    "- If \"conflicts\" is non-empty, answer LABEL: INSUFFICIENT and CONFLICT: YES.\n"
    "- If \"missing_evidence\" is non-empty or there are no applicable items, answer "
    "LABEL: INSUFFICIENT and VALUE: NONE.\n"
    "- Otherwise answer from the applicable item(s) only.\n\n"
    + ANSWER_CONTRACT
)


def format_spans(spans: List[RetrievedSpan]) -> str:
    lines = []
    for s in spans:
        lines.append(f"[{s.doc_id}] {s.text.strip()}")
    return "\n".join(lines) if lines else "(no matches found)"


def context_prompt(question: str, spans: List[RetrievedSpan]) -> str:
    return f"CONTEXT:\n{format_spans(spans)}\n\nQUESTION: {question}"


def _slim_packet(packet: EvidencePacket) -> dict:
    """Compact view shown to the decoder — the typed bottleneck, not raw docs."""
    items = []
    for it in packet.items:
        d = {"type": it.field_type, "value": it.exact_value,
             "doc": it.document_id or "-", "applies": it.applies}
        if it.entity and it.entity not in ("value", "clause", "date"):
            d["entity"] = it.entity
        note = it.applicability_context.split(";", 1)[1].strip() if ";" in it.applicability_context else ""
        if not it.applies and note:
            d["why_excluded"] = note[:60]
        items.append(d)
    return {"intent": packet.intent, "target_field": packet.target_field,
            "facets": packet.facets, "items": items,
            "conflicts": packet.conflicts, "missing_evidence": packet.missing_evidence}


def packet_prompt(question: str, packet: EvidencePacket) -> str:
    pj = json.dumps(_slim_packet(packet), ensure_ascii=False)
    return f"EVIDENCE_PACKET:\n{pj}\n\nQUESTION: {question}"
