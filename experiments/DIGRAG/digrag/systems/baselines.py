"""The four retrieval baselines: Grep, Vector RAG, Hybrid RAG, GrepRAG."""
from __future__ import annotations

import re
import time
from typing import Dict, List

from ..schema import Question, RetrievedSpan
from ..retrieval.grep import RipgrepIndex, extract_keywords
from ..retrieval.chunk import Chunk
from ..retrieval.vector import VectorIndex
from ..retrieval.hybrid import HybridIndex
from .base import System, Context
from .prompts import BASE_SYSTEM, context_prompt


def _spans_from_chunks(hits, chunks: List[Chunk]) -> List[RetrievedSpan]:
    spans = []
    for idx, score in hits:
        c = chunks[idx]
        spans.append(RetrievedSpan(doc_id=c.doc_id, text=c.text, score=float(score)))
    return spans


def _doc_ids(spans: List[RetrievedSpan]) -> List[str]:
    seen, out = set(), []
    for s in spans:
        if s.doc_id not in seen:
            seen.add(s.doc_id); out.append(s.doc_id)
    return out


# --------------------------- 1. Grep ------------------------------------
class GrepSystem(System):
    name = "grep"

    def __init__(self, ripgrep: RipgrepIndex, top_k: int = 5):
        self.rg = ripgrep
        self.top_k = top_k

    def build_context(self, q: Question, **_) -> Context:
        t0 = time.perf_counter()
        ranked = self.rg.retrieve(q.question, top_k=10)
        spans = ranked[:self.top_k]
        ms = (time.perf_counter() - t0) * 1000
        return Context(system_prompt=BASE_SYSTEM,
                       user_prompt=context_prompt(q.question, spans),
                       retrieved_doc_ids=_doc_ids(spans), ranked_doc_ids=_doc_ids(ranked),
                       retrieved_spans=[s.text for s in spans], retrieval_ms=ms)


# --------------------------- 2. Vector RAG ------------------------------
class VectorSystem(System):
    name = "vector"

    def __init__(self, vector: VectorIndex, chunks: List[Chunk], top_k: int = 5):
        self.vector = vector
        self.chunks = chunks
        self.top_k = top_k

    def build_context(self, q: Question, **_) -> Context:
        t0 = time.perf_counter()
        hits = self.vector.search(q.question, 10)
        ranked = _spans_from_chunks(hits, self.chunks)
        spans = ranked[:self.top_k]
        ms = (time.perf_counter() - t0) * 1000
        return Context(system_prompt=BASE_SYSTEM,
                       user_prompt=context_prompt(q.question, spans),
                       retrieved_doc_ids=_doc_ids(spans), ranked_doc_ids=_doc_ids(ranked),
                       retrieved_spans=[s.text for s in spans], retrieval_ms=ms)


# --------------------------- 3. Hybrid RAG ------------------------------
class HybridSystem(System):
    name = "hybrid"

    def __init__(self, hybrid: HybridIndex, chunks: List[Chunk], top_k: int = 5):
        self.hybrid = hybrid
        self.chunks = chunks
        self.top_k = top_k

    def build_context(self, q: Question, **_) -> Context:
        t0 = time.perf_counter()
        hits = self.hybrid.search(q.question, 10, rerank=True)
        ranked = _spans_from_chunks(hits, self.chunks)
        spans = ranked[:self.top_k]
        ms = (time.perf_counter() - t0) * 1000
        return Context(system_prompt=BASE_SYSTEM,
                       user_prompt=context_prompt(q.question, spans),
                       retrieved_doc_ids=_doc_ids(spans), ranked_doc_ids=_doc_ids(ranked),
                       retrieved_spans=[s.text for s in spans], retrieval_ms=ms)


# --------------------------- 4. GrepRAG ---------------------------------
QUERYGEN_SYSTEM = (
    "You write search queries for ripgrep over an enterprise corpus. "
    "Given a QUESTION, output up to 3 short literal search terms (single words or short "
    "phrases, no regex, no explanation) that would locate the answer. "
    "Output ONLY the terms, comma-separated, on a single line."
)


class GrepRAGSystem(System):
    name = "greprag"

    def __init__(self, ripgrep: RipgrepIndex, top_k: int = 5):
        self.rg = ripgrep
        self.top_k = top_k
        self.patterns_by_qid: Dict[str, List[str]] = {}

    def needs_query_gen(self) -> bool:
        return True

    def query_gen_prompt(self, q: Question):
        return QUERYGEN_SYSTEM, f"QUESTION: {q.question}"

    def set_patterns(self, qid: str, raw: str):
        terms = [t.strip().strip('"').strip("'") for t in re.split(r"[,\n]", raw or "")]
        terms = [t for t in terms if 1 <= len(t) <= 40][:4]
        self.patterns_by_qid[qid] = terms

    def build_context(self, q: Question, **_) -> Context:
        t0 = time.perf_counter()
        patterns = self.patterns_by_qid.get(q.qid) or extract_keywords(q.question)[:4]
        spans = self.rg.retrieve(q.question, top_k=10, patterns=patterns)
        # dedup spans by (doc_id, text)
        seen, uniq = set(), []
        for s in spans:
            k = (s.doc_id, s.text)
            if k not in seen:
                seen.add(k); uniq.append(s)
        ms = (time.perf_counter() - t0) * 1000
        shown = uniq[:self.top_k]
        ctx = context_prompt(q.question, shown)
        ctx = f"(ripgrep queries: {', '.join(patterns)})\n" + ctx
        return Context(system_prompt=BASE_SYSTEM, user_prompt=ctx,
                       retrieved_doc_ids=_doc_ids(shown), ranked_doc_ids=_doc_ids(uniq),
                       retrieved_spans=[s.text for s in shown], retrieval_ms=ms)
