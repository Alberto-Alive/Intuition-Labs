"""Run every experiment block in one process (load models once).

Blocks:
  1. LongMemEval  — 6 main systems, 3 seeds
  2. LongMemEval  — 8 ablations, 1 seed
  3. MultiHop-RAG — 6 main systems, 3 seeds
  4. Conflict stress-test (semi-real) — 6 main systems, 3 seeds  [reported separately]

Each block does retrieval on CPU then one big GPU-batched decode.
"""
from __future__ import annotations

import argparse
import os

from digrag.services.llm import DecoderLLM
from digrag.services.embedder import Embedder
from .run import run, print_summary, DECODER, EMBEDDER, RERANKER
from .systems import MAIN_SYSTEMS, ABLATIONS

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "real_benchmarks", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lme-n", type=int, default=90)
    ap.add_argument("--abl-n", type=int, default=110)
    ap.add_argument("--mh-n", type=int, default=150)
    ap.add_argument("--cf-n", type=int, default=70)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--only", default="", help="comma list: lme,abl,mh,cf (default all)")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    only = set(a.only.split(",")) if a.only else {"lme", "abl", "mh", "cf"}

    print("[load] decoder + embedder (shared by all systems) ...", flush=True)
    llm = DecoderLLM(DECODER, device="cuda", dtype="float16")
    embedder = Embedder(EMBEDDER, device="cuda", reranker_name=RERANKER)
    common = dict(top_k=6, batch_size=a.batch_size, max_new=80, embedder=embedder, llm=llm)

    if "lme" in only:
        s = run("longmemeval", MAIN_SYSTEMS, seeds, a.lme_n,
                out_dir=os.path.join(RES, "longmemeval_main"), **common)
        print_summary(s)
    if "abl" in only:
        s = run("longmemeval", ABLATIONS, [0], a.abl_n,
                out_dir=os.path.join(RES, "longmemeval_ablation"), **common)
        print_summary(s)
    if "mh" in only:
        s = run("multihop", MAIN_SYSTEMS, seeds, a.mh_n,
                out_dir=os.path.join(RES, "multihop_main"), **common)
        print_summary(s)
    if "cf" in only:
        s = run("conflict", MAIN_SYSTEMS, seeds, a.cf_n,
                out_dir=os.path.join(RES, "conflict_stress"), **common)
        print_summary(s)
    print("\n[all done]", flush=True)


if __name__ == "__main__":
    main()
