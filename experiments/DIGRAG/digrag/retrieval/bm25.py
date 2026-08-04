"""Minimal BM25 (Okapi) over a list of texts. Pure python, no dependency."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Tuple

_TOK = re.compile(r"[A-Za-z0-9$%\.\-]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOK.findall(text)]


class BM25:
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [tokenize(t) for t in corpus]
        self.k1, self.b = k1, b
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(1, self.N)
        self.df: Counter = Counter()
        self.tf: List[Counter] = []
        for d in self.docs:
            c = Counter(d)
            self.tf.append(c)
            for term in c:
                self.df[term] += 1
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
                    for t, n in self.df.items()}

    def scores(self, query: str) -> List[float]:
        q = tokenize(query)
        out = [0.0] * self.N
        for i, c in enumerate(self.tf):
            dl = len(self.docs[i])
            s = 0.0
            for term in q:
                if term not in c:
                    continue
                idf = self.idf.get(term, 0.0)
                f = c[term]
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out[i] = s
        return out

    def topk(self, query: str, k: int) -> List[Tuple[int, float]]:
        sc = self.scores(query)
        idx = sorted(range(self.N), key=lambda i: -sc[i])[:k]
        return [(i, sc[i]) for i in idx]
