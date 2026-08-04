"""Generic DIGRAG evidence compiler for real benchmarks.

Strictly benchmark-agnostic: it reads ONLY the question text, an optional
question date, and the *same* retrieved passages the baselines receive.  It
never sees gold answers, gold evidence, or benchmark question-type labels.

It compiles retrieved prose into a typed, provenance- and recency-annotated
evidence packet:
  * sentence-level evidence UNITS, each tagged with source pid + timestamp;
  * generic typed spans per unit (dates, numbers, named entities, quotes, ids);
  * freshness logic (parse timestamps; mark the most recent unit) gated on
    generic temporal markers in the question;
  * conflict markers (contradiction cues + disagreeing values);
  * an answerability estimate from question<->unit lexical overlap.

The decoder is then constrained to answer ONLY from the packet.  This is the
"evidence bottleneck"; nothing benchmark-specific is hand-coded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from .schema import Retrieved

# ---- generic lexical resources -----------------------------------------
TEMPORAL_MARKERS = ("current", "currently", "now", "latest", "most recent", "recent",
                    "last", "still", "today", "as of", "update", "updated", "before",
                    "after", "first", "anymore", "no longer", "these days")
CONFLICT_CUES = ("however", "but ", "previously", "no longer", "instead", "changed",
                 "used to", "originally", "later", "updated", "revised", "contrary",
                 "denied", "rejected", "actually", "correction")
NEGATION = ("not ", "n't", "never", "no ", "without", "denied", "rejected", "failed")
COMPARISON = ("compare", "both", "more than", "less than", "higher", "lower", "than",
              "which ... and", "versus", " vs ", "same", "different")

DATE_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b"
    r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
    r"|\b(?:19|20)\d{2}\b", re.I)
NUM_RE = re.compile(r"\$?\d[\d,\.]*\s?(?:%|percent|million|billion|trillion|k|years?|days?|months?|hours?)?", re.I)
QUOTE_RE = re.compile(r"[\"“]([^\"”]{2,80})[\"”]")
ID_RE = re.compile(r"\b[A-Z]{2,}[\-/]?\d+[A-Za-z0-9\-]*\b")
ENT_RE = re.compile(r"\b(?:[A-Z][a-zA-Z0-9&.]+(?:\s+[A-Z][a-zA-Z0-9&.]+){0,4})\b")
STOP_ENT = {"The", "A", "An", "This", "That", "What", "When", "Where", "Who", "Why",
            "How", "Which", "I", "It", "He", "She", "They", "We", "You"}


def parse_date_key(s: Optional[str]) -> Tuple:
    """Sortable key from heterogeneous timestamp strings (later = larger)."""
    if not s:
        return (0, 0, 0, 0, 0)
    m = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", s)
    if m:
        y, mo, d = map(int, m.groups())
    else:
        m2 = re.search(r"(19|20)\d{2}", s)
        y, mo, d = (int(m2.group(0)), 1, 1) if m2 else (0, 0, 0)
    tm = re.search(r"(\d{1,2}):(\d{2})", s)
    h, mi = (int(tm.group(1)), int(tm.group(2))) if tm else (0, 0)
    return (y, mo, d, h, mi)


def _entities(text: str) -> List[str]:
    out = []
    for m in ENT_RE.finditer(text):
        e = m.group(0).strip()
        first = e.split()[0]
        if first in STOP_ENT and len(e.split()) == 1:
            continue
        if len(e) >= 3:
            out.append(e)
    return out


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 8]


@dataclass
class Unit:
    pid: str
    timestamp: Optional[str]
    text: str
    dates: List[str] = field(default_factory=list)
    numbers: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    quotes: List[str] = field(default_factory=list)
    ids: List[str] = field(default_factory=list)
    date_key: Tuple = (0, 0, 0, 0, 0)
    overlap: float = 0.0


@dataclass
class EvidencePacket:
    question_markers: Dict[str, bool]
    units: List[dict]
    most_recent_pid: Optional[str]
    conflicts: List[dict]
    answerability: float
    candidate_entities: List[str]
    # toggles recording which components are active (for ablations)
    components: Dict[str, bool] = field(default_factory=dict)


def _qkeywords(question: str) -> set:
    return {w.lower() for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']+", question)
            if len(w) >= 3}


def compile_packet(question: str, question_date: Optional[str],
                   retrieved: List[Retrieved],
                   use_buckets: bool = True, use_context: bool = True,
                   use_temporal: bool = True, use_conflict: bool = True,
                   max_units: int = 12) -> EvidencePacket:
    qlow = question.lower()
    markers = {
        "temporal": any(t in qlow for t in TEMPORAL_MARKERS),
        "comparison": any(c in qlow for c in COMPARISON),
        "negation": any(n in qlow for n in NEGATION),
    }
    qk = _qkeywords(question)

    units: List[Unit] = []
    seen = set()
    for r in retrieved:
        sents = _split_sentences(r.text) if use_context else [r.text]
        for s in sents:
            key = re.sub(r"\s+", " ", s.lower())[:120]
            if key in seen:
                continue
            seen.add(key)
            u = Unit(pid=r.pid, timestamp=r.timestamp, text=s[:180],
                     date_key=parse_date_key(r.timestamp))
            if use_buckets:
                u.dates = DATE_RE.findall(s)[:4]     # DATE_RE has no capturing groups
                u.numbers = [m.group(0).strip() for m in NUM_RE.finditer(s)][:4]
                u.entities = _entities(s)[:6]
                u.quotes = QUOTE_RE.findall(s)[:3]
                u.ids = ID_RE.findall(s)[:3]
            sk = {w.lower() for w in re.findall(r"[A-Za-z0-9\-']+", s)}
            u.overlap = len(qk & sk) / (len(qk) + 1e-6)
            units.append(u)

    # rank units by question overlap (keeps the packet within budget)
    units.sort(key=lambda u: -u.overlap)
    units = units[:max_units]

    # freshness: most-recent unit among the evidence (only if temporal-ish)
    most_recent = None
    if use_temporal and units:
        timed = [u for u in units if u.date_key != (0, 0, 0, 0, 0)]
        if timed and (markers["temporal"] or len({u.pid for u in timed}) > 1):
            most_recent = max(timed, key=lambda u: u.date_key).pid

    # conflict detection: contradiction cue OR disagreeing numbers/dates among
    # the most question-relevant units
    conflicts = []
    if use_conflict:
        top = units[:6]
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                a, b = top[i], top[j]
                cue = any(c in (a.text + " " + b.text).lower() for c in CONFLICT_CUES)
                num_disagree = (a.numbers and b.numbers and set(a.numbers) != set(b.numbers)
                                and a.overlap > 0.15 and b.overlap > 0.15)
                if (cue and a.pid != b.pid and a.overlap > 0.2 and b.overlap > 0.2) or num_disagree:
                    conflicts.append({"pid_a": a.pid, "pid_b": b.pid,
                                      "a": a.text[:80], "b": b.text[:80]})
                    break
            if conflicts:
                break

    answerability = max((u.overlap for u in units), default=0.0)

    # candidate entities by cross-unit frequency (helps multi-hop synthesis)
    from collections import Counter
    cnt = Counter(e for u in units for e in u.entities)
    candidate_entities = [e for e, _ in cnt.most_common(8)]

    return EvidencePacket(
        question_markers=markers,
        units=[asdict(u) for u in units],
        most_recent_pid=most_recent, conflicts=conflicts,
        answerability=round(answerability, 3), candidate_entities=candidate_entities,
        components={"buckets": use_buckets, "context": use_context,
                    "temporal": use_temporal, "conflict": use_conflict})
