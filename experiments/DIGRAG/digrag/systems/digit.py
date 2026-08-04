"""DIGIT-inspired Evidence-Bucket RAG.

Pipeline:
  query -> intent + facets
  retrieve candidate docs (lexical UNION dense)
  compile typed EVIDENCE_PACKET (alias resolution, applicability alignment,
       conflict + missing-evidence detection)
  decode using ONLY the packet (no raw chunks reach the generator)

The evidence packet is the *bottleneck*: the decoder cannot introduce a value
that is not a typed exact_value with applies=true.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Dict, List

from ..schema import Document, Question
from ..retrieval.grep import RipgrepIndex
from ..retrieval.vector import VectorIndex
from ..retrieval.chunk import Chunk
from ..extract.intent import parse_intent
from ..extract.evidence import EvidenceCompiler
from .base import System, Context
from .prompts import DIGIT_SYSTEM, packet_prompt


DECISION_ENTITIES = {"approval_decision", "proposal_status", "exception_decision"}


def _normv(s: str) -> str:
    import re
    return re.sub(r"\s+", "", (s or "").lower().replace("$", "").replace(",", ""))


def apply_evidence_gate(packet: dict, parsed) -> dict:
    """Deterministically derive the DIGIT decision from the typed packet.

    The bottleneck principle: the decoder may verbalize but may NOT override the
    typed evidence.  Conflicts / missing-evidence force abstention; synthesized
    decision items fix YES/NO; value questions take the packet's applicable value
    (the decoder cannot substitute a value absent from the packet).
    Returns dict(label, value, conflict, abstained).
    """
    applies = [it for it in packet["items"] if it["applies"]]
    cites = sorted({it["document_id"] for it in applies if it["document_id"]})
    if packet["conflicts"]:
        cdocs = sorted({d for c in packet["conflicts"] for d in (c["doc_a"], c["doc_b"]) if d})
        return {"label": "INSUFFICIENT", "value": "", "conflict": True, "abstained": True, "cites": cdocs}
    if packet["missing_evidence"] and not applies:
        return {"label": "INSUFFICIENT", "value": "", "conflict": False, "abstained": True, "cites": []}
    for it in applies:                       # decision items map to YES/NO
        if it.get("entity") in DECISION_ENTITIES:
            v = it["exact_value"].lower()
            if "not in effect" in v or "approval required" in v:
                return {"label": "NO", "value": it["exact_value"], "conflict": False, "abstained": False, "cites": cites}
            if "in effect" in v or "auto-approved" in v or "refund allowed" in v:
                return {"label": "YES", "value": it["exact_value"], "conflict": False, "abstained": False, "cites": cites}
    if applies:                              # value questions: take packet's value
        pv = [it["exact_value"] for it in applies]
        value = parsed.value if any(_normv(parsed.value) == _normv(x) for x in pv) else pv[0]
        label = parsed.label if parsed.label in ("VALUE", "YES", "NO", "CONDITIONAL") else "VALUE"
        return {"label": label, "value": value, "conflict": False, "abstained": False, "cites": cites}
    return {"label": parsed.label, "value": parsed.value, "conflict": parsed.conflict,
            "abstained": parsed.label == "INSUFFICIENT", "cites": parsed.cites}


class DigitSystem(System):
    name = "digit"

    def __init__(self, ripgrep: RipgrepIndex, vector: VectorIndex, chunks: List[Chunk],
                 docs_by_id: Dict[str, Document], lex_k: int = 8, dense_k: int = 8,
                 max_candidates: int = 12):
        self.rg = ripgrep
        self.vector = vector
        self.chunks = chunks
        self.docs_by_id = docs_by_id
        self.compiler = EvidenceCompiler(docs_by_id)
        self.lex_k = lex_k
        self.dense_k = dense_k
        self.max_candidates = max_candidates

    def _candidate_docs(self, q: Question, intent) -> List[Document]:
        ids: List[str] = []
        # lexical candidates
        for s in self.rg.retrieve(q.question, top_k=self.lex_k):
            if s.doc_id not in ids:
                ids.append(s.doc_id)
        # dense candidates
        for idx, _ in self.vector.search(q.question, self.dense_k):
            did = self.chunks[idx].doc_id
            if did not in ids:
                ids.append(did)
        docs = [self.docs_by_id[i] for i in ids[:self.max_candidates] if i in self.docs_by_id]
        return docs

    def build_context(self, q: Question, **_) -> Context:
        t0 = time.perf_counter()
        intent = parse_intent(q.question)
        cands = self._candidate_docs(q, intent)
        packet = self.compiler.compile(intent, cands)
        ms = (time.perf_counter() - t0) * 1000
        doc_ids = sorted({it.document_id for it in packet.items if it.document_id})
        spans = [it.source_span for it in packet.items]
        ranked = [d.doc_id for d in cands][:10]
        return Context(system_prompt=DIGIT_SYSTEM,
                       user_prompt=packet_prompt(q.question, packet),
                       retrieved_doc_ids=doc_ids, ranked_doc_ids=ranked,
                       retrieved_spans=spans, retrieval_ms=ms, packet=asdict(packet))
