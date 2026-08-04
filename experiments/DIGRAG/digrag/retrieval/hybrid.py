"""Hybrid retrieval: BM25 + dense fused with Reciprocal Rank Fusion, then an
optional cross-encoder rerank of the fused candidate pool."""
from __future__ import annotations

from typing import List, Tuple

from .bm25 import BM25
from .chunk import Chunk
from .vector import VectorIndex
from ..services.embedder import Embedder


class HybridIndex:
    def __init__(self, chunks: List[Chunk], embedder: Embedder, rrf_k: int = 60):
        self.chunks = chunks
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.bm25 = BM25([c.text for c in chunks])
        self.vector = VectorIndex(chunks, embedder)

    def _rrf(self, ranked_lists: List[List[int]]) -> dict:
        scores: dict = {}
        for lst in ranked_lists:
            for rank, idx in enumerate(lst):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        return scores

    def search(self, query: str, k: int, pool: int = 20, rerank: bool = True
               ) -> List[Tuple[int, float]]:
        bm = [i for i, _ in self.bm25.topk(query, pool)]
        dv = [i for i, _ in self.vector.search(query, pool)]
        fused = self._rrf([bm, dv])
        cand = sorted(fused, key=lambda i: -fused[i])[:pool]
        if rerank and cand:
            passages = [self.chunks[i].text for i in cand]
            rr = self.embedder.rerank(query, passages)              # higher = better
            order = sorted(range(len(cand)), key=lambda j: -rr[j])
            return [(cand[j], float(rr[j])) for j in order[:k]]
        return [(i, fused[i]) for i in cand[:k]]
