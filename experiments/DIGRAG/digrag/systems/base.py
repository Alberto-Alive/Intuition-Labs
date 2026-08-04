"""System interface.

A System turns a Question into (a) a retrieval/evidence context and (b) a
(system_prompt, user_prompt) pair for the shared decoder.  The driver collects
prompts from every system and decodes them in batches on the GPU.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..schema import Question, RetrievedSpan


@dataclass
class Context:
    system_prompt: str
    user_prompt: str
    retrieved_doc_ids: List[str] = field(default_factory=list)   # docs shown to decoder
    ranked_doc_ids: List[str] = field(default_factory=list)       # ranked top-10 for recall@k
    retrieved_spans: List[str] = field(default_factory=list)
    retrieval_ms: float = 0.0
    packet: Optional[dict] = None


class System:
    name = "base"

    def build_context(self, q: Question) -> Context:        # pragma: no cover
        raise NotImplementedError

    # GrepRAG overrides this to request a query-generation pre-pass.
    def needs_query_gen(self) -> bool:
        return False


def timed(fn):
    def wrap(*a, **k):
        t0 = time.perf_counter()
        out = fn(*a, **k)
        return out, (time.perf_counter() - t0) * 1000.0
    return wrap
