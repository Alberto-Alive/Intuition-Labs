"""Main experiment driver.

Runs all five systems over the benchmark.  Retrieval/evidence-compilation is
done first (CPU), then EVERY (system x question) answer prompt is decoded in
large batches on the GPU — one model load, full GPU saturation.  This batched
decode is the "run everything in parallel" path that makes the PoC finish fast.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Dict, List

from .config import load_config
from .schema import Document, Question, Prediction, GoldEvidence
from .dataset.generator import generate
from .services.llm import DecoderLLM, parse_structured
from .services.embedder import Embedder
from .retrieval.grep import RipgrepIndex
from .retrieval.chunk import chunk_documents
from .retrieval.vector import VectorIndex
from .retrieval.hybrid import HybridIndex
from .systems.baselines import GrepSystem, VectorSystem, HybridSystem, GrepRAGSystem
from .systems.digit import DigitSystem, apply_evidence_gate
from .eval.metrics import score_one, aggregate

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)


def load_dataset(data_dir: str):
    docs, questions = [], []
    with open(os.path.join(data_dir, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            docs.append(Document(**json.loads(line)))
    with open(os.path.join(data_dir, "questions.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            d["gold_evidence"] = [GoldEvidence(**g) for g in d.get("gold_evidence", [])]
            questions.append(Question(**d))
    return docs, questions


def build(cfg, docs, work_dir):
    print(f"[build] {len(docs)} docs — materializing ripgrep corpus + indexes ...", flush=True)
    rg = RipgrepIndex(docs, work_dir)
    chunks = chunk_documents(docs, cfg.get("retrieval.chunk_size", 240),
                             cfg.get("retrieval.chunk_overlap", 40))
    embedder = Embedder(cfg.get("models.embedder"), device=cfg.get("models.device", "cuda"),
                        offline=cfg.get("models.offline", False),
                        reranker_name=cfg.get("models.reranker"))
    t0 = time.time()
    vector = VectorIndex(chunks, embedder)
    hybrid = HybridIndex(chunks, embedder, rrf_k=cfg.get("retrieval.rrf_k", 60))
    print(f"[build] {len(chunks)} chunks embedded in {time.time()-t0:.1f}s "
          f"(emb dim {embedder.dim}, rg={'yes' if rg.has_rg else 'py-fallback'})", flush=True)
    docs_by_id = {d.doc_id: d for d in docs}
    k = cfg.get("retrieval.top_k", 5)
    systems = [
        GrepSystem(rg, top_k=k),
        VectorSystem(vector, chunks, top_k=k),
        HybridSystem(hybrid, chunks, top_k=k),
        GrepRAGSystem(rg, top_k=k),
        DigitSystem(rg, vector, chunks, docs_by_id, lex_k=8, dense_k=8),
    ]
    return systems, embedder


def run(cfg, limit=None, out_dir=None):
    out_dir = out_dir or os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    data_dir = os.path.join(ROOT, cfg.get("dataset.out_dir", "data"))

    # 1) dataset
    stats = generate(data_dir, cfg.get("seed", 0), cfg.get("dataset.n_scenarios", 220),
                     cfg.get("dataset.distractor_docs", 400))
    docs, questions = load_dataset(data_dir)
    if limit:
        questions = questions[:limit]
    print(f"[data] {stats['n_docs']} docs, {len(questions)} questions", flush=True)

    # 2) indexes + systems
    systems, _ = build(cfg, docs, data_dir)

    # 3) decoder LLM
    print(f"[llm] loading {cfg.get('models.decoder')} (offline={cfg.get('models.offline')}) ...", flush=True)
    llm = DecoderLLM(cfg.get("models.decoder"), device=cfg.get("models.device", "cuda"),
                     dtype=cfg.get("models.dtype", "float16"), offline=cfg.get("models.offline", False))
    bs = cfg.get("generation.batch_size", 24)
    max_new = cfg.get("generation.max_new_tokens", 200)

    # 4) GrepRAG query-generation pre-pass (batched)
    greprag = next(s for s in systems if s.name == "greprag")
    qg_sys, qg_usr, qg_qids = [], [], []
    for q in questions:
        s, u = greprag.query_gen_prompt(q)
        qg_sys.append(s); qg_usr.append(u); qg_qids.append(q.qid)
    print(f"[greprag] generating {len(qg_qids)} ripgrep queries ...", flush=True)
    t0 = time.time()
    qg_out = llm.generate(qg_sys, qg_usr, max_new_tokens=32, batch_size=bs)
    for qid, o in zip(qg_qids, qg_out):
        greprag.set_patterns(qid, o["raw"])
    print(f"[greprag] query-gen done in {time.time()-t0:.1f}s", flush=True)

    # 5) retrieval / evidence compilation for every (system, question)
    print("[retrieve] building contexts for all systems ...", flush=True)
    records = []   # dicts carrying everything needed to build Prediction
    t0 = time.time()
    for s in systems:
        for q in questions:
            ctx = s.build_context(q)
            records.append({"system": s.name, "q": q, "ctx": ctx})
    print(f"[retrieve] {len(records)} contexts in {time.time()-t0:.1f}s", flush=True)

    # 6) ONE big batched decode on the GPU
    sys_prompts = [r["ctx"].system_prompt for r in records]
    usr_prompts = [r["ctx"].user_prompt for r in records]
    print(f"[decode] {len(usr_prompts)} prompts, batch_size={bs}, max_new={max_new} ...", flush=True)
    t0 = time.time()
    outs = llm.generate(sys_prompts, usr_prompts, max_new_tokens=max_new, batch_size=bs)
    gen_wall_ms = (time.time() - t0) * 1000
    print(f"[decode] done in {gen_wall_ms/1000:.1f}s "
          f"({len(usr_prompts)/(gen_wall_ms/1000):.1f} prompts/s)", flush=True)

    # attribute generation time proportional to tokens processed
    tot_tok = sum(o["input_tokens"] + o["output_tokens"] for o in outs) or 1

    # 7) parse -> Prediction
    preds: List[Prediction] = []
    for r, o in zip(records, outs):
        parsed = parse_structured(o["raw"])
        ctx = r["ctx"]
        gen_ms = gen_wall_ms * (o["input_tokens"] + o["output_tokens"]) / tot_tok

        def mk(system, label, value, conflict, abstained, cites=None):
            return Prediction(
                qid=r["q"].qid, system=system, answer_text=parsed.answer,
                pred_label=label, used_value=value,
                cited_doc_ids=parsed.cites if cites is None else cites,
                flagged_conflict=conflict, abstained=abstained,
                retrieved_doc_ids=ctx.retrieved_doc_ids, ranked_doc_ids=ctx.ranked_doc_ids,
                retrieved_spans=ctx.retrieved_spans, context_text=ctx.user_prompt,
                context_tokens=o["input_tokens"], output_tokens=o["output_tokens"],
                retrieval_ms=ctx.retrieval_ms, generation_ms=gen_ms, packet=ctx.packet)

        if r["system"] == "digit":
            g = apply_evidence_gate(ctx.packet, parsed)
            preds.append(mk("digit", g["label"], g["value"], g["conflict"], g["abstained"], cites=g["cites"]))
            # ablation: same packet + generation, but WITHOUT the evidence gate
            preds.append(mk("digit_raw", parsed.label, parsed.value, parsed.conflict,
                            parsed.label == "INSUFFICIENT"))
        else:
            preds.append(mk(r["system"], parsed.label, parsed.value, parsed.conflict,
                            parsed.label == "INSUFFICIENT"))

    # 8) score + aggregate
    qby = {q.qid: q for q in questions}
    recall_ks = cfg.get("retrieval.recall_k", [1, 3, 5, 10])
    rows = [score_one(p, qby[p.qid], recall_ks) for p in preds]
    agg = aggregate(rows, recall_ks,
                    usd_in=cfg.get("cost.usd_per_1k_input_tokens", 0.0),
                    usd_out=cfg.get("cost.usd_per_1k_output_tokens", 0.0))

    # 9) persist
    with open(os.path.join(out_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(asdict(p)) + "\n")
    with open(os.path.join(out_dir, "per_question.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = {"dataset": stats, "config": dict(cfg), "n_questions": len(questions),
               "gen_wall_s": round(gen_wall_ms / 1000, 1), "metrics": agg}
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print_summary(agg, recall_ks)
    print(f"\n[done] results in {out_dir}", flush=True)
    return summary


def print_summary(agg: Dict, recall_ks):
    order = ["grep", "vector", "hybrid", "greprag", "digit"]
    cols = [("final_answer_acc", "Ans"), ("exact_value_acc", "ExactVal"),
            ("source_span_acc", "Src"), ("unsupported_claim_rate", "Unsup"),
            ("conflict_detect_acc", "Conf"), ("missing_evidence_recall", "Miss"),
            ("answerability_acc", "Answbl"), ("mean_context_tokens", "CtxTok"),
            ("mean_latency_ms", "Lat_ms")]
    header = f"{'system':<9}" + "".join(f"{c[1]:>9}" for c in cols)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for s in [*order, "digit_raw"]:
        if s not in agg:
            continue
        a = agg[s]
        line = f"{s:<9}"
        for key, _ in cols:
            v = a.get(key)
            line += f"{v:>9.3f}" if isinstance(v, (int, float)) else f"{'-':>9}"
        print(line)
    print("=" * len(header))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None, help="subsample N questions (smoke test)")
    ap.add_argument("--offline", action="store_true", help="no-download deterministic mode")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.offline:
        cfg["models"]["offline"] = True
    if a.batch_size:
        cfg["generation"]["batch_size"] = a.batch_size
    run(cfg, limit=a.limit, out_dir=a.out)


if __name__ == "__main__":
    main()
