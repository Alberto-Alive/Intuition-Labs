"""Offline tests for the real-benchmark pipeline (no GPU, no network).

Focus: the GENERIC compiler is benchmark-agnostic and gold-free, the ablation
toggles actually change the packet, and the deterministic judge behaves.

    python -m pytest experiments/DIGRAG/tests/test_real_benchmarks.py -q
"""
import inspect
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from real_benchmarks.schema import Retrieved                       # noqa: E402
from real_benchmarks.compiler import compile_packet                # noqa: E402
from real_benchmarks.systems import (build_user_prompt, MAIN_SYSTEMS, ABLATIONS,  # noqa: E402
                                      apply_gate, parse_output)
from real_benchmarks.metrics import answer_correct, f1             # noqa: E402

PASSAGES = [
    Retrieved(pid="d1", text="Acme acquired Beta Corp for $2 billion in 2021.",
              score=1.0, timestamp="2021-03-01", title="old"),
    Retrieved(pid="d2", text="In 2024 Acme sold Beta Corp; it is no longer owned by Acme.",
              score=0.9, timestamp="2024-06-01", title="new"),
    Retrieved(pid="d3", text="Beta Corp makes payment software in Berlin.",
              score=0.8, timestamp="2023-01-01", title="bg"),
]


def _count(t):
    return len(t.split())


def test_compiler_is_gold_free():
    # the compiler signature must not accept any gold/answer/label argument
    params = set(inspect.signature(compile_packet).parameters)
    assert not (params & {"gold", "answer", "gold_answer", "label", "gold_passage_ids"})


def test_compiler_extracts_typed_and_recency():
    pkt = compile_packet("Who currently owns Beta Corp?", None, PASSAGES)
    assert pkt.units, "should produce evidence units"
    # freshness: question is temporal ('currently') -> most_recent set to latest ts
    assert pkt.most_recent_pid == "d2"
    # typed buckets present somewhere
    assert any(u["numbers"] or u["entities"] or u["dates"] for u in pkt.units)
    # provenance on every unit
    assert all(u["pid"] for u in pkt.units)


def test_ablation_toggles_change_packet():
    full = compile_packet("Who currently owns Beta Corp?", None, PASSAGES)
    no_buckets = compile_packet("Who currently owns Beta Corp?", None, PASSAGES, use_buckets=False)
    no_temporal = compile_packet("Who currently owns Beta Corp?", None, PASSAGES, use_temporal=False)
    assert any(u["entities"] for u in full.units)
    assert all(not u["entities"] for u in no_buckets.units)      # buckets removed
    assert no_temporal.most_recent_pid is None                   # freshness removed


def test_budget_parity_packet_vs_chunks():
    spec_pkt = next(s for s in MAIN_SYSTEMS if s.name == "digrag_raw")
    spec_chunk = next(s for s in MAIN_SYSTEMS if s.name == "hybrid")
    _, up_pkt, _ = build_user_prompt(spec_pkt, "Who owns Beta Corp?", None, PASSAGES, _count)
    _, up_chunk, _ = build_user_prompt(spec_chunk, "Who owns Beta Corp?", None, PASSAGES, _count)
    # packet must not blow the budget relative to chunks (within 1.5x here)
    assert _count(up_pkt) <= _count(up_chunk) * 1.6 + 50


def test_gate_abstains_on_weak_evidence():
    spec = next(s for s in MAIN_SYSTEMS if s.name == "digrag_gated")
    weak = compile_packet("What is the capital of an unmentioned country?", None,
                          [Retrieved(pid="x", text="Unrelated text about cooking.", score=0.1, timestamp=None)])
    ans, cites, ab, conf = apply_gate(spec, weak, "Paris", [], False, tau=0.4)
    assert ab and ans == "NOT_FOUND"


def test_judge():
    assert answer_correct("Sam Bankman-Fried", "Sam Bankman-Fried")
    assert answer_correct("The answer is Sam Bankman-Fried.", "Sam Bankman-Fried")  # containment
    assert not answer_correct("Elon Musk", "Sam Bankman-Fried")
    assert f1("a b c", "a b c") == 1.0


def test_eight_ablations_registered():
    assert len(ABLATIONS) == 8 and ABLATIONS[0].name == "A_full"
