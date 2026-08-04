"""MultiHop-RAG adapter -> shared schema.

A single shared news corpus (609 docs); each query needs evidence from several
documents.  `null_query` questions are unanswerable (gold answer
"Insufficient information.") and drive the abstention/answerability metric.
Gold supporting docs are recovered by matching `evidence_list` titles to corpus
titles (eval-only).
"""
from __future__ import annotations

import json
import os
from typing import List, Tuple

from huggingface_hub import hf_hub_download

from ..schema import Example, Passage

NULL_ANSWER = "Insufficient information."


def _load():
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    q = hf_hub_download("yixuantt/MultiHopRAG", "MultiHopRAG.json", repo_type="dataset")
    c = hf_hub_download("yixuantt/MultiHopRAG", "corpus.json", repo_type="dataset")
    return json.load(open(q, encoding="utf-8")), json.load(open(c, encoding="utf-8"))


def load_corpus() -> List[Passage]:
    _, corpus = _load()
    passages = []
    for i, d in enumerate(corpus):
        passages.append(Passage(
            pid=f"doc{i}", text=d.get("body", ""), timestamp=d.get("published_at"),
            title=d.get("title", ""),
            meta={"source": d.get("source"), "category": d.get("category")}))
    return passages


def load(limit: int | None = None, seed: int = 0,
         skip_null: bool = False) -> Tuple[List[Passage], List[Example]]:
    queries, corpus = _load()
    title_to_pid = {d.get("title", ""): f"doc{i}" for i, d in enumerate(corpus)}
    passages = load_corpus()

    import random
    rng = random.Random(seed)
    if skip_null:
        queries = [q for q in queries if q.get("question_type") != "null_query"]
    if limit and limit < len(queries):
        queries = rng.sample(queries, limit)

    examples: List[Example] = []
    for i, q in enumerate(queries):
        qtype = q.get("question_type", "")
        answerable = qtype != "null_query"
        ev = q.get("evidence_list", []) or []
        gold_pids = sorted({title_to_pid[e["title"]] for e in ev
                            if e.get("title") in title_to_pid})
        gold_facts = [e.get("fact", "") for e in ev]
        examples.append(Example(
            qid=f"mh_{i}", question=q["query"], question_type=qtype,
            gold_answer=str(q.get("answer", "")),
            gold_passage_ids=gold_pids, gold_facts=gold_facts,
            answerable=answerable, is_temporal=(qtype == "temporal_query"),
            passages=[], use_shared_corpus=True))
    return passages, examples
