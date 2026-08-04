"""LongMemEval adapter -> shared schema.

Each question carries its own chat haystack.  We treat every *turn* as a
retrievable passage, tagged with its session id and the session date (so
freshness/temporal logic is available to every system, not just DIGRAG).
Gold supporting evidence is the set of `answer_session_ids` (session
granularity).  Abstention questions are flagged via the `_abs` suffix.
"""
from __future__ import annotations

import json
import os
from typing import List, Tuple

from huggingface_hub import hf_hub_download

from ..schema import Example, Passage

TEMPORAL_TYPES = {"temporal-reasoning", "knowledge-update"}


def _load(variant: str = "longmemeval_s"):
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    path = hf_hub_download("xiaowu0162/longmemeval", variant, repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load(variant: str = "longmemeval_s", limit: int | None = None,
         seed: int = 0) -> Tuple[None, List[Example]]:
    raw = _load(variant)
    import random
    rng = random.Random(seed)
    if limit and limit < len(raw):
        raw = rng.sample(raw, limit)
    examples: List[Example] = []
    for item in raw:
        qid = item["question_id"]
        answerable = not qid.endswith("_abs")
        gold_sessions = item.get("answer_session_ids", [])
        passages: List[Passage] = []
        for sess, sess_id, sdate in zip(item["haystack_sessions"],
                                        item["haystack_session_ids"],
                                        item["haystack_dates"]):
            for ti, turn in enumerate(sess):
                content = (turn.get("content") or "").strip()
                if not content:
                    continue
                passages.append(Passage(
                    pid=f"{sess_id}::{ti}", text=f"{turn.get('role','user')}: {content}",
                    timestamp=sdate, title=sess_id,
                    meta={"session_id": sess_id, "role": turn.get("role")}))
        examples.append(Example(
            qid=qid, question=item["question"], question_type=item["question_type"],
            gold_answer=str(item.get("answer", "")),
            gold_passage_ids=list(gold_sessions),
            question_date=item.get("question_date"),
            answerable=answerable,
            is_temporal=item["question_type"] in TEMPORAL_TYPES,
            passages=passages, use_shared_corpus=False))
    return None, examples
