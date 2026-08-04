"""Dense vector index over chunks (cosine sim via normalized dot product)."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .chunk import Chunk
from ..services.embedder import Embedder


class VectorIndex:
    def __init__(self, chunks: List[Chunk], embedder: Embedder):
        self.chunks = chunks
        self.embedder = embedder
        self.matrix = embedder.encode([c.text for c in chunks])  # (N, d) normalized

    def search(self, query: str, k: int) -> List[Tuple[int, float]]:
        q = self.embedder.encode([query])[0]            # (d,)
        sims = self.matrix @ q                            # cosine (both normalized)
        idx = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in idx]

    def search_many(self, queries: List[str], k: int) -> List[List[Tuple[int, float]]]:
        Q = self.embedder.encode(queries)                # (M, d)
        sims = Q @ self.matrix.T                          # (M, N)
        out = []
        for row in sims:
            idx = np.argsort(-row)[:k]
            out.append([(int(i), float(row[i])) for i in idx])
        return out
