"""System registry: 6 main systems + ablations, over the shared retriever.

All systems share one decoder, one embedder, the same top-k, and the same
context-token budget.  They differ only in (a) which retriever family produces
the top-k passages and (b) whether those passages are stuffed as chunks or
compiled into a typed evidence packet (optionally gated).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .schema import Retrieved
from .compiler import EvidencePacket, compile_packet

# ---------------- answer contract (shared) ------------------------------
CONTRACT = (
    "Respond in EXACTLY this format:\n"
    "ANSWER: <a concise answer; or NOT_FOUND if the context does not contain it>\n"
    "CITES: <comma-separated source ids you used (e.g. doc12), or NONE>")

BASE_SYS = (
    "You are a precise question-answering assistant. Answer the QUESTION using ONLY the "
    "provided CONTEXT. Do not use outside knowledge. If the context does not contain the "
    "answer, respond NOT_FOUND.\n\n" + CONTRACT)

PACKET_SYS = (
    "You are a precise question-answering assistant. You are given a typed EVIDENCE_PACKET "
    "compiled from retrieved sources — not raw documents. Answer the QUESTION using ONLY the "
    "evidence units in the packet; cite the source ids (pid) you used. "
    "If the question asks for the current/latest/most-recent state and the packet marks a "
    "most_recent unit, prefer it. If no unit supports an answer, respond NOT_FOUND. "
    "Do not introduce facts that are not in the packet.\n\n" + CONTRACT)


@dataclass
class SystemSpec:
    name: str
    retrieval: str = "hybrid"        # lexical | dense | hybrid | greprag
    fmt: str = "chunks"              # chunks | packet | buckets_only
    constrained: bool = False        # use PACKET_SYS (vs BASE_SYS)
    gate: bool = False               # deterministic abstention gate
    compiler: Dict = field(default_factory=dict)   # toggles for compile_packet


MAIN_SYSTEMS: List[SystemSpec] = [
    SystemSpec("lexical", retrieval="lexical", fmt="chunks"),
    SystemSpec("vector", retrieval="dense", fmt="chunks"),
    SystemSpec("hybrid", retrieval="hybrid", fmt="chunks"),
    SystemSpec("greprag", retrieval="greprag", fmt="chunks"),
    SystemSpec("digrag_raw", retrieval="hybrid", fmt="packet", constrained=True, gate=False),
    SystemSpec("digrag_gated", retrieval="hybrid", fmt="packet", constrained=True, gate=True),
]

# Ablations (run on LongMemEval). All use hybrid retrieval; vary the packet.
ABLATIONS: List[SystemSpec] = [
    SystemSpec("A_full", retrieval="hybrid", fmt="packet", constrained=True, gate=True),
    SystemSpec("B_no_gate", retrieval="hybrid", fmt="packet", constrained=True, gate=False),
    SystemSpec("C_no_buckets", retrieval="hybrid", fmt="packet", constrained=True, gate=True,
               compiler={"use_buckets": False}),
    SystemSpec("D_buckets_only", retrieval="hybrid", fmt="buckets_only", constrained=True, gate=True,
               compiler={"use_context": False}),
    SystemSpec("E_no_temporal", retrieval="hybrid", fmt="packet", constrained=True, gate=True,
               compiler={"use_temporal": False}),
    SystemSpec("F_no_conflict_flags", retrieval="hybrid", fmt="packet", constrained=True, gate=True,
               compiler={"use_conflict": False}),
    SystemSpec("G_unconstrained_packet", retrieval="hybrid", fmt="packet", constrained=False, gate=False),
    SystemSpec("H_stuff_passages", retrieval="hybrid", fmt="chunks"),   # #8 ≡ #9: same passages, no packet
]

GATE_TAU = 0.12          # fixed a-priori abstention threshold (NOT tuned on test)


# ---------------- rendering (readable, token-budgeted) ------------------
CTX_TOKEN_BUDGET = 600          # context payload budget shared by ALL systems


def render_chunks(question: str, retrieved: List[Retrieved], count_tokens, budget) -> str:
    lines, used = [], 0
    for r in retrieved:
        ts = f" | {r.timestamp}" if r.timestamp else ""
        ln = f"[{r.pid}{ts}] {r.text.strip()}"
        t = count_tokens(ln)
        if used + t > budget and lines:
            break
        lines.append(ln); used += t
    ctx = "\n".join(lines) if lines else "(no passages retrieved)"
    return f"CONTEXT:\n{ctx}\n\nQUESTION: {question}"


def _unit_line(i: int, u: dict, buckets_only: bool) -> str:
    ts = u["timestamp"] or "n/a"
    head = f"{i}. [{u['pid']} | {ts}]"
    typed_bits = []
    for k, lab in (("dates", "dates"), ("numbers", "nums"), ("entities", "ents"),
                   ("quotes", "quotes"), ("ids", "ids")):
        if u.get(k):
            typed_bits.append(f"{lab}: {', '.join(u[k][:4])}")
    typed = (" «" + "; ".join(typed_bits) + "»") if typed_bits else ""
    if buckets_only:
        return f"{head}{typed}"
    return f"{head} {u['text']}{typed}"


def render_packet(question: str, pkt: EvidencePacket, buckets_only: bool,
                  count_tokens, budget) -> str:
    flags = [f"answerability={pkt.answerability}"]
    if pkt.most_recent_pid:
        flags.append(f"most_recent={pkt.most_recent_pid}")
    if pkt.question_markers.get("temporal"):
        flags.append("temporal_question=yes")
    header = "EVIDENCE_PACKET (" + "; ".join(flags) + "):"
    conf = ""
    if pkt.conflicts:
        c = pkt.conflicts[0]
        conf = f"\nCONFLICT: {c['pid_a']} vs {c['pid_b']} (values disagree)"
    lines, used = [], count_tokens(header + conf)
    for i, u in enumerate(pkt.units, 1):
        ln = _unit_line(i, u, buckets_only)
        t = count_tokens(ln)
        if used + t > budget and lines:
            break
        lines.append(ln); used += t
    body = header + "\n" + "\n".join(lines) + conf
    return f"{body}\n\nQUESTION: {question}"


def build_user_prompt(spec: SystemSpec, question: str, question_date: Optional[str],
                      retrieved: List[Retrieved], count_tokens):
    """Returns (system_prompt, user_prompt, packet_or_None). All systems share
    the same context-token budget."""
    if spec.fmt == "chunks":
        return BASE_SYS, render_chunks(question, retrieved, count_tokens, CTX_TOKEN_BUDGET), None
    pkt = compile_packet(question, question_date, retrieved, **spec.compiler)
    sysp = PACKET_SYS if spec.constrained else BASE_SYS
    user = render_packet(question, pkt, (spec.fmt == "buckets_only"), count_tokens, CTX_TOKEN_BUDGET)
    return sysp, user, pkt


# ---------------- parsing + gate ---------------------------------------
ABSTAIN_RE = re.compile(r"not[_ ]found|insufficient|don'?t know|cannot (?:find|determine)|"
                        r"no (?:information|answer|mention)|unable to", re.I)


def parse_output(raw: str):
    m = re.search(r"ANSWER:\s*(.*)", raw, re.I)
    ans = (m.group(1).strip() if m else raw.strip().splitlines()[0] if raw.strip() else "").strip()
    cm = re.search(r"CITES:\s*(.*)", raw, re.I)
    cites = []
    if cm:
        cites = [c.strip() for c in re.split(r"[,\s]+", cm.group(1)) if re.match(r"\w+\d", c.strip())]
    abstained = bool(ABSTAIN_RE.search(ans)) or ans.upper().startswith("NOT_FOUND") or ans == ""
    return ans, cites, abstained


def apply_gate(spec: SystemSpec, pkt: Optional[EvidencePacket], ans: str,
               cites: List[str], abstained: bool, tau: float = GATE_TAU):
    """Deterministic abstention/conflict gate (gated DIGRAG only).

    `tau` is the answerability abstention threshold, CALIBRATED ON A DEV SPLIT
    (never the test set) per benchmark.
    """
    conflict = bool(pkt.conflicts) if pkt else False
    if not spec.gate or pkt is None:
        return ans, cites, abstained, conflict
    if not pkt.units or pkt.answerability < tau:      # evidence too weak -> abstain
        return "NOT_FOUND", [], True, conflict
    return ans, cites, abstained, conflict
