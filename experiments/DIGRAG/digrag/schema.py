"""Shared data structures for the DIGRAG benchmark and systems.

Everything that flows between the dataset, the retrievers, the evidence
compiler and the evaluator is one of these typed records, so the pipeline is
easy to serialize and audit.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ----------------------------- Corpus ------------------------------------
@dataclass
class Document:
    doc_id: str
    title: str
    text: str                       # full document text (may be multi-line)
    doc_type: str = "policy"        # policy | contract | table | code | memo | faq
    timestamp: Optional[str] = None  # ISO date; used for stale-vs-current
    status: str = "active"          # active | superseded | draft | rejected
    meta: Dict[str, Any] = field(default_factory=dict)

    def lines(self) -> List[str]:
        return self.text.splitlines()


# ----------------------------- Gold --------------------------------------
@dataclass
class GoldEvidence:
    """One piece of evidence the correct answer must rest on."""
    doc_id: str
    field_type: str                 # number | date | name | id | clause | table_cell | code_symbol | policy_field
    value: str                      # the canonical exact value (e.g. "30 days")
    span: str                       # the literal substring that contains it


@dataclass
class Question:
    qid: str
    case_type: str                  # stale_current | approved_rejected | exception | alias | conflict |
                                    # multi_doc | table | date | policy_condition | code_symbol | ambiguous_keyword
    question: str
    # Categorical answer used for scoring. CONDITIONAL = depends on a qualifier.
    gold_label: str                 # YES | NO | CONDITIONAL | INSUFFICIENT
    gold_answer: str                # canonical short natural-language answer
    gold_value: str                 # the exact value the answer hinges on ("" if none)
    answerable: bool                # False -> the corpus lacks the needed evidence
    gold_evidence: List[GoldEvidence] = field(default_factory=list)
    # The tempting literal grep finds that does NOT apply (the trap). "" if none.
    distractor_value: str = ""
    distractor_doc_id: str = ""
    # Pairs of doc_ids that genuinely conflict (for conflict-detection scoring).
    gold_conflicts: List[List[str]] = field(default_factory=list)
    facets: Dict[str, str] = field(default_factory=dict)   # region=EU, plan=enterprise, ...

    def gold_doc_ids(self) -> List[str]:
        return sorted({e.doc_id for e in self.gold_evidence})


# ----------------------------- System I/O --------------------------------
@dataclass
class RetrievedSpan:
    doc_id: str
    text: str                       # the span/chunk text actually shown to the model
    score: float = 0.0
    line: int = -1                  # line index within the source document (-1 if chunk)


@dataclass
class EvidenceItem:
    """One typed cell of the DIGIT evidence packet."""
    field_type: str
    entity: str
    exact_value: str
    source_span: str
    document_id: str
    timestamp: Optional[str] = None
    applicability_context: str = ""
    applies: bool = True            # did facet-alignment keep this value?
    status: str = "active"


@dataclass
class EvidencePacket:
    intent: str
    target_field: str
    facets: Dict[str, str]
    items: List[EvidenceItem]
    conflicts: List[Dict[str, str]] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)

    def to_prompt_json(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Prediction:
    qid: str
    system: str
    answer_text: str
    pred_label: str                 # YES | NO | CONDITIONAL | INSUFFICIENT
    used_value: str                 # the exact value the system's answer relied on
    cited_doc_ids: List[str] = field(default_factory=list)
    flagged_conflict: bool = False
    abstained: bool = False
    # Bookkeeping for metrics.
    retrieved_doc_ids: List[str] = field(default_factory=list)
    ranked_doc_ids: List[str] = field(default_factory=list)
    retrieved_spans: List[str] = field(default_factory=list)
    context_text: str = ""          # exactly what was given to the decoder
    context_tokens: int = 0
    output_tokens: int = 0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    packet: Optional[Dict[str, Any]] = None   # serialized EvidencePacket for DIGIT
