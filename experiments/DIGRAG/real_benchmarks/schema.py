"""Shared schema for real-benchmark evaluation.

Every benchmark (LongMemEval, MultiHop-RAG, conflict stress-test) is normalized
into `Passage` + `Example`.  Gold fields (`gold_answer`, `gold_passage_ids`,
`gold_facts`) are **eval-only**: the retrieval and the DIGRAG compiler must
never read them (enforced by convention — systems receive only `question`,
`question_date`, and the retrievable `Passage` text/timestamp).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Passage:
    pid: str
    text: str
    timestamp: Optional[str] = None      # ISO-ish date string if known
    title: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)   # e.g. {"session_id": ...}


@dataclass
class Example:
    qid: str
    question: str
    question_type: str
    # --- eval-only gold (NEVER read by systems) ---
    gold_answer: str
    gold_passage_ids: List[str] = field(default_factory=list)   # supporting docs/sessions
    gold_facts: List[str] = field(default_factory=list)
    # --- question-side context the systems MAY use ---
    question_date: Optional[str] = None
    answerable: bool = True
    is_temporal: bool = False            # temporal/update question (freshness matters)
    # --- corpus ---
    passages: List[Passage] = field(default_factory=list)   # per-question (LongMemEval)
    use_shared_corpus: bool = False                          # True -> retrieve over shared corpus (MultiHop)


@dataclass
class Retrieved:
    pid: str
    text: str
    score: float
    timestamp: Optional[str] = None
    title: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemOutput:
    qid: str
    system: str
    seed: int
    benchmark: str
    pred_answer: str
    pred_label: str                      # ANSWER | ABSTAIN
    used_value: str = ""
    cited_pids: List[str] = field(default_factory=list)
    abstained: bool = False
    flagged_conflict: bool = False
    retrieved_pids: List[str] = field(default_factory=list)   # ranked, for recall@k
    context_text: str = ""               # exactly what the decoder saw (for unsupported-claim)
    context_tokens: int = 0
    output_tokens: int = 0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    packet: Optional[dict] = None
