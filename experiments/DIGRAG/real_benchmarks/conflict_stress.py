"""Semi-real conflict stress-test (SEPARATE from main scores).

Built from MultiHop-RAG: for each answerable query we inject one *stale,
contradictory* variant of a gold evidence document — same topic/keywords, but
the gold answer is replaced by a plausible decoy (another query's answer) and
stamped with an OLDER date.  The genuine document keeps its real (newer) date.

A freshness/conflict-aware reader should answer the CURRENT (newer) value; a
reader that ignores recency is tempted by the stale contradictory value.  This
isolates temporal/conflict handling.  It is clearly semi-synthetic and is never
mixed into the LongMemEval / MultiHop main results.

Benchmark construction uses gold answers (to build the contradiction); the
*systems* still never see gold — only the injected corpus + question.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from .schema import Example, Passage
from .adapters.multihop import load_corpus, _load


def _older(ts: str | None) -> str:
    if ts and re.match(r"\d{4}", ts):
        y = int(ts[:4])
        return str(y - 2) + ts[4:]
    return "2019-01-01T00:00:00+00:00"


def load(limit: int = 60, seed: int = 0) -> Tuple[List[Passage], List[Example]]:
    queries, corpus = _load()
    title_to_idx = {d.get("title", ""): i for i, d in enumerate(corpus)}
    passages = load_corpus()

    # answerable queries with a short, gold evidence doc we can contradict
    cand = [q for q in queries if q.get("question_type") != "null_query"
            and q.get("evidence_list") and len(str(q.get("answer", "")).split()) <= 5]
    import random
    rng = random.Random(seed)
    rng.shuffle(cand)
    cand = cand[:limit]
    decoys = [str(q["answer"]) for q in cand]

    examples: List[Example] = []
    for i, q in enumerate(cand):
        answer = str(q["answer"])
        decoy = next((d for d in (decoys[(i + 7) % len(decoys)], decoys[(i + 3) % len(decoys)])
                      if d and d.lower() != answer.lower()), "an unnamed party")
        ev = q["evidence_list"][0]
        gold_title = ev.get("title", "")
        gidx = title_to_idx.get(gold_title)
        if gidx is None:
            continue
        gold_pid = f"doc{gidx}"
        fact = ev.get("fact", "") or corpus[gidx].get("body", "")[:300]
        stale_text = fact.replace(answer, decoy) if answer in fact else f"{decoy}. {fact}"
        stale = Passage(pid=f"{gold_pid}__stale", text=stale_text,
                        timestamp=_older(corpus[gidx].get("published_at")),
                        title=gold_title + " (earlier report)",
                        meta={"stale_variant": True, "of": gold_pid})
        passages.append(stale)
        examples.append(Example(
            qid=f"cf_{i}", question=q["query"], question_type="conflict_freshness",
            gold_answer=answer, gold_passage_ids=[gold_pid], gold_facts=[fact],
            answerable=True, is_temporal=True, passages=[], use_shared_corpus=True))
    return passages, examples
