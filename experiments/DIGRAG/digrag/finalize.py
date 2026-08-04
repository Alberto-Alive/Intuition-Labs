"""Fill README placeholders + result tables from a finished run.

Runs plotting and analysis, then substitutes headline numbers and tables into
README.md so the paper is self-contained.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys


def run_module(mod, *args):
    subprocess.run([sys.executable, "-m", mod, *args], check=True, cwd=ROOT)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    run_module("digrag.plots", "--results", a.results)
    run_module("digrag.analyze", "--results", a.results, "--data", a.data)

    summary = json.load(open(os.path.join(ROOT, a.results, "metrics.json"), encoding="utf-8"))
    m = summary["metrics"]
    main_tbl = open(os.path.join(ROOT, a.results, "tables", "main_results.md"), encoding="utf-8").read()
    main_tbl = main_tbl.split("\n\n", 2)[-1]   # drop the analyze.py heading
    case_tbl = open(os.path.join(ROOT, a.results, "tables", "by_case_type.md"), encoding="utf-8").read()
    case_tbl = case_tbl.split("\n\n", 1)[-1]

    def pct(s, k):
        v = m[s].get(k)
        return f"{v:.3f}" if isinstance(v, (int, float)) else "—"

    repl = {
        "{N_DOCS}": str(summary["dataset"]["n_docs"]),
        "{N_Q}": str(summary["n_questions"]),
        "{GREP_ANS}": pct("grep", "final_answer_acc"),
        "{HYBRID_ANS}": pct("hybrid", "final_answer_acc"),
        "{DIGIT_ANS}": pct("digit", "final_answer_acc"),
        "{DIGIT_UNSUP}": pct("digit", "unsupported_claim_rate"),
    }
    readme_path = os.path.join(ROOT, "README.md")
    text = open(readme_path, encoding="utf-8").read()
    for k, v in repl.items():
        text = text.replace(k, v)
    text = text.replace("<!-- RESULTS_TABLE -->", main_tbl.strip())
    text = text.replace("<!-- CASE_TABLE -->", case_tbl.strip())
    open(readme_path, "w", encoding="utf-8").write(text)
    print("[finalize] README.md patched with headline numbers + tables")


if __name__ == "__main__":
    main()
