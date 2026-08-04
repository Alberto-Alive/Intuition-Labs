"""Real-benchmark driver.

Pipeline per (benchmark, system-set, seeds):
  1. load + subsample examples (seeded);
  2. [if GrepRAG] one batched LLM pass to write keyword queries;
  3. per example: build ONE retrieval bundle, retrieve each mode once, build
     every system's prompt (chunk-stuffed or compiled packet);
  4. ONE big GPU-batched decode of all prompts;
  5. parse + gate + score + aggregate (mean over seeds).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Dict, List

from digrag.services.llm import DecoderLLM
from digrag.services.embedder import Embedder
from .schema import SystemOutput
from .retrieval import RetrievalBundle, select_within_budget, keywords
from .systems import (MAIN_SYSTEMS, ABLATIONS, SystemSpec, build_user_prompt,
                      parse_output, apply_gate, compile_packet)
from .metrics import score_one, aggregate_with_seeds

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECODER = "Qwen/Qwen2.5-1.5B-Instruct"
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_benchmark(name: str, limit: int, seed: int):
    if name == "longmemeval":
        from .adapters.longmemeval import load
        return load(limit=limit, seed=seed)
    if name == "multihop":
        from .adapters.multihop import load
        return load(limit=limit, seed=seed)
    if name == "conflict":
        from .conflict_stress import load
        return load(limit=limit, seed=seed)
    raise ValueError(name)


def calibrate_tau(benchmark: str, embedder, top_k: int = 6, dev_n: int = 40,
                  dev_seed: int = 999) -> float:
    """Pick the abstention threshold on a DEV split (never the test seeds) by
    maximizing balanced accuracy of (answerability >= tau) vs gold answerable.
    Uses only answerability scores — no decoding."""
    shared, dev = load_benchmark(benchmark, dev_n, dev_seed)
    shared_bundle = RetrievalBundle(shared, embedder) if shared is not None else None
    pts = []
    for ex in dev:
        bundle = shared_bundle or RetrievalBundle(ex.passages, embedder)
        ranked = bundle.hybrid(ex.question, max(30, top_k * 3), rerank=False)
        shown = select_within_budget(ranked, top_k)
        pkt = compile_packet(ex.question, ex.question_date, shown)
        pts.append((pkt.answerability, ex.answerable))
    pos = [s for s, a in pts if a]; neg = [s for s, a in pts if not a]
    if not neg or not pos:                      # no unanswerable in dev -> be conservative
        return 0.06
    best_tau, best_acc = 0.12, -1.0
    for tau in sorted({round(s, 3) for s, _ in pts}):
        tpr = sum(1 for s in pos if s >= tau) / len(pos)
        tnr = sum(1 for s in neg if s < tau) / len(neg)
        bal = (tpr + tnr) / 2
        if bal > best_acc:
            best_acc, best_tau = bal, tau
    return float(min(0.4, max(0.04, best_tau)))


def retrieve_modes(bundle: RetrievalBundle, query: str, top_k: int, pool_k: int,
                   terms, need):
    """Return {mode: ranked Retrieved list (length pool_k)} for needed modes."""
    out = {}
    if "lexical" in need:
        out["lexical"] = bundle.lexical(query, pool_k)
    if "dense" in need:
        out["dense"] = bundle.dense(query, pool_k)
    if "hybrid" in need:
        out["hybrid"] = bundle.hybrid(query, pool_k, rerank=True)
    if "greprag" in need:
        out["greprag"] = bundle.greprag(query, pool_k, terms)
    return out


def run(benchmark: str, systems: List[SystemSpec], seeds: List[int], limit: int,
        top_k: int, out_dir: str, batch_size: int, max_new: int, embedder, llm,
        usd_in=0.0, usd_out=0.0, recall_ks=(1, 3, 5, 10)):
    needed_modes = {s.retrieval for s in systems}
    want_greprag = "greprag" in needed_modes
    pool_k = max(30, top_k * 3)
    gated = any(s.gate for s in systems)
    tau = calibrate_tau(benchmark, embedder, top_k) if gated else 0.12
    if gated:
        print(f"[{benchmark}] calibrated abstention tau={tau:.3f} (dev split, seed 999)", flush=True)

    records = []          # dicts: spec, ex, sys_prompt, user_prompt, retrieved_pids, packet, retrieval_ms
    examples_by_qid = {}

    for seed in seeds:
        shared_corpus, examples = load_benchmark(benchmark, limit, seed)
        shared_bundle = (RetrievalBundle(shared_corpus, embedder)
                         if shared_corpus is not None else None)
        if shared_bundle is not None:
            print(f"[{benchmark} seed{seed}] shared corpus {len(shared_corpus)} passages indexed", flush=True)

        # GrepRAG keyword pre-pass (batched)
        terms_by_qid = {}
        if want_greprag:
            sp = ["You write search keywords. Given a question, output 3-6 keywords "
                  "(space-separated, no punctuation) that would retrieve the answer. Output only keywords."] * len(examples)
            up = [f"Question: {e.question}" for e in examples]
            outs = llm.generate(sp, up, max_new_tokens=24, batch_size=batch_size)
            for e, o in zip(examples, outs):
                terms_by_qid[e.qid] = keywords(o["raw"], 6) or keywords(e.question, 6)

        t0 = time.time()
        for ei, ex in enumerate(examples):
            examples_by_qid[(seed, ex.qid)] = ex
            bundle = shared_bundle or RetrievalBundle(ex.passages, embedder)
            tr0 = time.time()
            modes = retrieve_modes(bundle, ex.question, top_k, pool_k,
                                   terms_by_qid.get(ex.qid), needed_modes)
            ret_ms = (time.time() - tr0) * 1000
            for spec in systems:
                ranked = modes[spec.retrieval]
                shown = select_within_budget(ranked, top_k)
                sysp, userp, pkt = build_user_prompt(spec, ex.question, ex.question_date,
                                                     shown, llm.count_tokens)
                records.append({
                    "spec": spec, "seed": seed, "qid": ex.qid,
                    "sys": sysp, "user": userp,
                    "ctx_tokens": llm.count_tokens(userp),
                    "retrieved_pids": [r.pid for r in ranked[:max(recall_ks)]],
                    "packet": pkt, "retrieval_ms": ret_ms / len(systems)})
            if (ei + 1) % 50 == 0:
                print(f"[{benchmark} seed{seed}] retrieved {ei+1}/{len(examples)} "
                      f"({(time.time()-t0):.0f}s)", flush=True)

    # ---- one big batched decode ----
    print(f"[{benchmark}] decoding {len(records)} prompts (bs={batch_size}) ...", flush=True)
    t0 = time.time()
    outs = llm.generate([r["sys"] for r in records], [r["user"] for r in records],
                        max_new_tokens=max_new, batch_size=batch_size)
    gen_wall = (time.time() - t0) * 1000
    tot = sum(o["input_tokens"] + o["output_tokens"] for o in outs) or 1
    print(f"[{benchmark}] decode done {gen_wall/1000:.0f}s "
          f"({len(records)/(gen_wall/1000):.1f}/s)", flush=True)

    # ---- parse + gate + score ----
    sys_outputs: List[SystemOutput] = []
    rows = []
    for r, o in zip(records, outs):
        spec = r["spec"]; pkt = r["packet"]
        ans, cites, abstained = parse_output(o["raw"])
        ans, cites, abstained, conflict = apply_gate(spec, pkt, ans, cites, abstained, tau)
        so = SystemOutput(
            qid=r["qid"], system=spec.name, seed=r["seed"], benchmark=benchmark,
            pred_answer=ans, pred_label="ABSTAIN" if abstained else "ANSWER",
            used_value=ans, cited_pids=cites, abstained=abstained, flagged_conflict=conflict,
            retrieved_pids=r["retrieved_pids"], context_text=r["user"],
            context_tokens=r["ctx_tokens"], output_tokens=o["output_tokens"],
            retrieval_ms=r["retrieval_ms"], generation_ms=gen_wall * (o["input_tokens"]+o["output_tokens"]) / tot,
            packet=(asdict(pkt) if pkt is not None else None))
        sys_outputs.append(so)
        rows.append(score_one(so, examples_by_qid[(r["seed"], r["qid"])], list(recall_ks)))

    agg = aggregate_with_seeds(rows, list(recall_ks), usd_in=usd_in, usd_out=usd_out)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for so in sys_outputs:
            d = asdict(so); d.pop("packet", None)        # keep predictions light
            f.write(json.dumps(d) + "\n")
    with open(os.path.join(out_dir, "retrieved_evidence.jsonl"), "w", encoding="utf-8") as f:
        for so in sys_outputs:
            if so.system.startswith("digrag") or so.system in ("A_full",):
                f.write(json.dumps({"qid": so.qid, "seed": so.seed, "system": so.system,
                                    "packet": so.packet}) + "\n")
    with open(os.path.join(out_dir, "per_question.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = {"benchmark": benchmark, "n_per_seed": limit, "seeds": seeds,
               "top_k": top_k, "decoder": DECODER, "gen_wall_s": round(gen_wall/1000, 1),
               "abstention_tau_dev_calibrated": tau, "ctx_token_budget": 600,
               "metrics": agg}
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["longmemeval", "multihop", "conflict"])
    ap.add_argument("--systems", default="main", choices=["main", "ablation"])
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    systems = MAIN_SYSTEMS if a.systems == "main" else ABLATIONS
    seeds = [int(s) for s in a.seeds.split(",")]
    out_dir = a.out or os.path.join(ROOT, "real_benchmarks", "results", f"{a.benchmark}_{a.systems}")

    print(f"[load] decoder + embedder ...", flush=True)
    llm = DecoderLLM(DECODER, device="cuda", dtype="float16")
    embedder = Embedder(EMBEDDER, device="cuda", reranker_name=RERANKER)
    summary = run(a.benchmark, systems, seeds, a.limit, a.top_k, out_dir,
                  a.batch_size, a.max_new, embedder, llm)
    print_summary(summary)
    print(f"[done] {out_dir}", flush=True)


def print_summary(summary):
    m = summary["metrics"]
    keys = [("final_answer_acc", "Ans"), ("answerable_acc", "AnsAble"),
            ("abstention_acc", "Abst"), ("temporal_update_acc", "Temp"),
            ("support_recall", "SupRec"), ("unsupported_claim_rate", "Unsup"),
            ("recall@5", "Rec@5"), ("mean_context_tokens", "CtxTok")]
    print("\n" + "=" * 92)
    print(f"{summary['benchmark']}  (n/seed={summary['n_per_seed']}, seeds={summary['seeds']})")
    print(f"{'system':22}" + "".join(f"{k[1]:>9}" for k in keys))
    for s, a in m.items():
        line = f"{s:22}"
        for key, _ in keys:
            v = a.get(key)
            line += f"{v:>9.3f}" if isinstance(v, (int, float)) else f"{'-':>9}"
        print(line)
    print("=" * 92)


if __name__ == "__main__":
    main()
