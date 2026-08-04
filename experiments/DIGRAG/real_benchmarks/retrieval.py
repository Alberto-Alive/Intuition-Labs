"""Unified retrieval over a passage list (shared or per-question).

Provides the four retrieval modes used by the baselines and DIGRAG:
lexical (BM25), dense, hybrid (RRF + optional cross-encoder rerank), and a
keyword-driven lexical mode for GrepRAG.  Every system draws top-k passages
from the SAME retriever family under the SAME budget, so comparisons are fair.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np

from digrag.retrieval.bm25 import BM25
from digrag.services.embedder import Embedder
from .schema import Passage, Retrieved

STOP = set("the a an of to in on at by is are was were be been for and or with from this that "
           "what which who when where why how did do does i my me you your we our it its as".split())


def keywords(text: str, n: int = 8) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']+", text)
    out, seen = [], set()
    for t in toks:
        lt = t.lower()
        if lt in STOP or len(t) < 3 or lt in seen:
            continue
        seen.add(lt); out.append(t)
    return out[:n]


class RetrievalBundle:
    def __init__(self, passages: List[Passage], embedder: Embedder,
                 rrf_k: int = 60, build_dense: bool = True):
        self.passages = passages
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.texts = [f"{p.title} {p.text}".strip() for p in passages]
        self.bm25 = BM25(self.texts)
        self.matrix = embedder.encode(self.texts) if (build_dense and passages) else None

    def _wrap(self, idxs_scores) -> List[Retrieved]:
        out = []
        for i, s in idxs_scores:
            p = self.passages[i]
            out.append(Retrieved(pid=p.pid, text=p.text, score=float(s),
                                  timestamp=p.timestamp, title=p.title, meta=p.meta))
        return out

    def lexical(self, query: str, k: int) -> List[Retrieved]:
        return self._wrap(self.bm25.topk(query, k))

    def greprag(self, query: str, k: int, terms: List[str]) -> List[Retrieved]:
        q = " ".join(terms) if terms else query
        return self._wrap(self.bm25.topk(q, k))

    def dense(self, query: str, k: int) -> List[Retrieved]:
        if self.matrix is None:
            return []
        qv = self.embedder.encode([query])[0]
        sims = self.matrix @ qv
        idx = np.argsort(-sims)[:k]
        return self._wrap([(int(i), float(sims[i])) for i in idx])

    def hybrid(self, query: str, k: int, pool: int = 30, rerank: bool = False) -> List[Retrieved]:
        bm = [i for i, _ in self.bm25.topk(query, pool)]
        dv = []
        if self.matrix is not None:
            qv = self.embedder.encode([query])[0]
            sims = self.matrix @ qv
            dv = list(np.argsort(-sims)[:pool])
        scores: Dict[int, float] = {}
        for lst in (bm, dv):
            for rank, i in enumerate(lst):
                i = int(i)
                scores[i] = scores.get(i, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        cand = sorted(scores, key=lambda i: -scores[i])[:pool]
        if rerank and cand:
            rr = self.embedder.rerank(query, [self.texts[i] for i in cand])
            order = sorted(range(len(cand)), key=lambda j: -rr[j])
            return self._wrap([(cand[j], rr[j]) for j in order[:k]])
        return self._wrap([(i, scores[i]) for i in cand[:k]])


def select_within_budget(retrieved: List[Retrieved], top_k: int,
                         per_passage_chars: int = 480, total_char_budget: int = 6000) -> List[Retrieved]:
    """Token-budget-equalizing selection shared by all systems."""
    out, used = [], 0
    for r in retrieved[:top_k]:
        snippet = r.text[:per_passage_chars]
        if used + len(snippet) > total_char_budget and out:
            break
        out.append(Retrieved(pid=r.pid, text=snippet, score=r.score,
                             timestamp=r.timestamp, title=r.title, meta=r.meta))
        used += len(snippet)
    return out
