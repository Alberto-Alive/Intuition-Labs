"""Embedding + reranking service (shared by vector & hybrid systems).

Dense embeddings come from a SentenceTransformer on the GPU.  An optional
cross-encoder reranker is lazy-loaded for the hybrid system.  Offline mode
falls back to a deterministic hashing embedder so the pipeline runs with no
downloads.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class Embedder:
    def __init__(self, model_name: str, device: str = "cuda", offline: bool = False,
                 reranker_name: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.offline = offline
        self.reranker_name = reranker_name
        self._model = None
        self._reranker = None
        self.dim = 384
        if not offline:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name, device=device)
            self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.offline:
            return self._hash_encode(texts)
        emb = self._model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False)
        return emb.astype(np.float32)

    # deterministic bag-of-hashed-tokens fallback (cosine-comparable)
    def _hash_encode(self, texts: List[str]) -> np.ndarray:
        import re
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                out[i, hash(tok) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    # cross-encoder rerank; falls back to identity if unavailable
    def rerank(self, query: str, passages: List[str]) -> List[float]:
        if not passages:
            return []
        if self.offline or not self.reranker_name:
            return list(range(len(passages), 0, -1))  # keep input order
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(self.reranker_name, device=self.device)
            except Exception:
                self._reranker = False
        if self._reranker is False:
            return list(range(len(passages), 0, -1))
        scores = self._reranker.predict([(query, p) for p in passages])
        return [float(s) for s in scores]
