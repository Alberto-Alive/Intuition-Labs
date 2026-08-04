"""Lexical retrieval via ripgrep (with a pure-python fallback).

Used by the Grep baseline and the GrepRAG baseline.  We materialize the corpus
to one text file per document so we can drive the real `rg` binary, exactly as
an agent harness would grep a repository.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Dict, List, Tuple

from ..schema import Document, RetrievedSpan

STOPWORDS = set("""a an the of for to in on at by is are was were be been being do does did
what which who whom when where why how can could will would should may might must shall this
that these those it its their our your his her can't cannot get got has have had with without
within now current currently does do not no yes about into out over under than then so as if
they them he she we you i me my mine value values does company wide company-wide""".split())

ID_RE = re.compile(r"\b(?:[A-Z]{2,}-\d+|[A-Z][a-z]+-\d+|D\d{3,}|[A-Z][A-Z0-9_]{3,})\b")
NUM_RE = re.compile(r"\$?\d[\d,\.]*%?")


def extract_keywords(question: str) -> List[str]:
    """Salient literals an agent would grep for: IDs, numbers, capitalized
    entities, and content words."""
    kws: List[str] = []
    for m in ID_RE.findall(question):
        kws.append(m)
    # capitalized entity words (e.g., Atlas, EU, Pro) — keep originals
    for w in re.findall(r"\b[A-Z][A-Za-z0-9]+\b", question):
        if w.lower() not in STOPWORDS:
            kws.append(w)
    # content words
    for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", question.lower()):
        if len(w) >= 3 and w not in STOPWORDS:
            kws.append(w)
    # dedup preserve order
    seen, out = set(), []
    for k in kws:
        lk = k.lower()
        if lk not in seen:
            seen.add(lk)
            out.append(k)
    return out


class RipgrepIndex:
    def __init__(self, docs: List[Document], work_dir: str):
        self.docs = {d.doc_id: d for d in docs}
        self.work_dir = work_dir
        self.files_dir = os.path.join(work_dir, "corpus_files")
        self.has_rg = shutil.which("rg") is not None
        self._materialize()

    def _materialize(self):
        os.makedirs(self.files_dir, exist_ok=True)
        for d in self.docs.values():
            p = os.path.join(self.files_dir, f"{d.doc_id}.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(d.text)

    # -- core grep: return (doc_id, line_no, line_text) for a fixed string --
    def _rg(self, pattern: str, fixed: bool = True) -> List[Tuple[str, int, str]]:
        if not self.has_rg:
            return self._py_grep(pattern, fixed)
        args = ["rg", "--no-heading", "--line-number", "--color", "never"]
        if fixed:
            args.append("--fixed-strings")
        args += ["--ignore-case", pattern, self.files_dir]
        try:
            out = subprocess.run(args, capture_output=True, text=True, timeout=20)
        except Exception:
            return self._py_grep(pattern, fixed)
        res = []
        for line in out.stdout.splitlines():
            # path:lineno:content
            parts = line.split(":", 2)
            if len(parts) < 3:
                continue
            path, lineno, content = parts
            doc_id = os.path.splitext(os.path.basename(path))[0]
            if doc_id in self.docs:
                res.append((doc_id, int(lineno), content))
        return res

    def _py_grep(self, pattern: str, fixed: bool) -> List[Tuple[str, int, str]]:
        res = []
        pl = pattern.lower()
        for d in self.docs.values():
            for i, line in enumerate(d.lines(), start=1):
                if (pl in line.lower()) if fixed else re.search(pattern, line, re.I):
                    res.append((d.doc_id, i, line))
        return res

    # -- retrieve top-k spans for an NL question --------------------------
    def retrieve(self, question: str, top_k: int = 5,
                 patterns: List[str] | None = None) -> List[RetrievedSpan]:
        kws = patterns if patterns is not None else extract_keywords(question)
        # score lines by how many distinct query keywords they contain
        line_hits: Dict[Tuple[str, int], Dict] = {}
        for kw in kws:
            for doc_id, lineno, content in self._rg(kw, fixed=True):
                key = (doc_id, lineno)
                e = line_hits.setdefault(key, {"content": content, "hits": set()})
                e["hits"].add(kw.lower())
        scored: List[RetrievedSpan] = []
        for (doc_id, lineno), e in line_hits.items():
            score = len(e["hits"])
            # exact phrase bonus if the whole question (minus stopwords) appears
            scored.append(RetrievedSpan(doc_id=doc_id, text=e["content"],
                                        score=float(score), line=lineno))
        scored.sort(key=lambda s: (-s.score, s.doc_id, s.line))
        return scored[:top_k]
