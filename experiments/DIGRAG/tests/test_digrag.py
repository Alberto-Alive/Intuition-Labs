"""Offline tests for the DIGRAG pipeline (no GPU, no downloads).

Validates the parts that must be correct independent of the decoder:
the dataset, the evidence compiler, and the metric functions.

    python -m pytest experiments/DIGRAG/tests -q
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from digrag.dataset.generator import Generator               # noqa: E402
from digrag.retrieval.grep import RipgrepIndex                # noqa: E402
from digrag.retrieval.chunk import chunk_documents            # noqa: E402
from digrag.retrieval.vector import VectorIndex               # noqa: E402
from digrag.services.embedder import Embedder                 # noqa: E402
from digrag.systems.digit import DigitSystem, apply_evidence_gate  # noqa: E402
from digrag.services.llm import parse_structured              # noqa: E402
from digrag.eval.metrics import value_match, norm, extract_values  # noqa: E402


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    g = Generator(seed=0)
    docs, qs = g.build(n_scenarios=120, distractor_docs=80)
    wd = str(tmp_path_factory.mktemp("data"))
    rg = RipgrepIndex(docs, wd)
    chunks = chunk_documents(docs)
    emb = Embedder("offline", device="cpu", offline=True)     # hashing embedder
    vec = VectorIndex(chunks, emb)
    dbi = {d.doc_id: d for d in docs}
    dig = DigitSystem(rg, vec, chunks, dbi)
    return docs, qs, dig


def test_dataset_balanced(world):
    _, qs, _ = world
    from collections import Counter
    c = Counter(q.case_type for q in qs)
    assert len(c) == 12
    assert all(v >= 8 for v in c.values())


def test_gold_spans_are_substrings(world):
    docs, qs, _ = world
    dbi = {d.doc_id: d for d in docs}
    for q in qs:
        for ev in q.gold_evidence:
            assert ev.span in dbi[ev.doc_id].text, f"{q.qid}: gold span not in doc"


def test_compiler_packets_correct(world):
    """The evidence packet must carry the applicable answer / abstention signal
    for every case type (this is the part the decoder relies on)."""
    _, qs, dig = world
    from collections import defaultdict
    buckets = defaultdict(list)
    for q in qs:
        buckets[q.case_type].append(q)
    failures = []
    for ct, group in buckets.items():
        for q in group[:6]:
            pk = dig.build_context(q).packet
            applies = [it["exact_value"] for it in pk["items"] if it["applies"]]
            miss, conf = bool(pk["missing_evidence"]), bool(pk["conflicts"])
            gv = q.gold_value.lower().replace("$", "").replace(" ", "")
            if q.gold_label == "VALUE":
                ok = any(gv in a.lower().replace("$", "").replace(" ", "") for a in applies)
            elif q.gold_label == "INSUFFICIENT":
                ok = miss or conf
            else:                                   # YES / NO
                ok = bool(applies) and not miss and not conf
            if not ok:
                failures.append((ct, q.qid))
    assert not failures, f"packet failures: {failures}"


def test_gate_forces_abstention_on_conflict(world):
    _, qs, dig = world
    q = next(q for q in qs if q.case_type == "conflict")
    pk = dig.build_context(q).packet
    parsed = parse_structured("LABEL: VALUE\nVALUE: 30 days\nCITES: NONE\nCONFLICT: NO\nANSWER: x")
    g = apply_evidence_gate(pk, parsed)
    assert g["label"] == "INSUFFICIENT" and g["conflict"] is True


def test_metric_value_match_whitespace_insensitive():
    assert value_match("99.95%", "99.95 %")
    assert value_match("30 days", "the window is 30 days")
    assert not value_match("30 days", "45 days")


def test_extract_values():
    vals = extract_values("The budget is $120k and renews 2025-01-01 after 30 days.")
    assert any("120k" in norm(v) for v in vals)
    assert any("2025-01-01" in v for v in vals)
